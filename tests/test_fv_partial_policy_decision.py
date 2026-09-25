"""Archive-bound refusal accounting without another FV solver run."""

import hashlib
import json
import shutil

import pytest

from examples.weather_scenarios import fv_partial_policy_decision as decision


def _copy_archive(tmp_path, monkeypatch):
    plan = tmp_path / decision.PLAN.name
    manifest_path = tmp_path / decision.MANIFEST.name
    raw = tmp_path / decision.RAW.name
    shutil.copyfile(decision.PLAN, plan)
    shutil.copyfile(decision.MANIFEST, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["output_sha256"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(decision.EVIDENCE / name, target)
    monkeypatch.setattr(decision, "EVIDENCE", tmp_path)
    monkeypatch.setattr(decision, "PLAN", plan)
    monkeypatch.setattr(decision, "MANIFEST", manifest_path)
    monkeypatch.setattr(decision, "RAW", raw)
    return manifest_path, raw


def _rewrite_output(manifest_path, path, edit):
    report = json.loads(path.read_text())
    edit(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = json.loads(manifest_path.read_text())
    name = f"partial_branch_gate_detail_attempt3/{path.name}"
    manifest["output_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_archived_refusal_is_a_policy_outcome_not_root_absence():
    result = decision.classify()
    assert result["execution_status"] == "nonzero_exit_without_resource_termination"
    assert result["numerical_status"] == "refused_by_declared_nominal_policy"
    assert result["support_status"] == "unsupported_by_declared_nominal_policy"
    assert result["first_failing_branch_reasons"] == {
        "signature_changed": 3, "face_flux_margin": 13,
    }
    assert result["finite_armijo_evaluations"] == 0
    assert result["response_computed"] is False
    assert result["response_validation"] == "not_established"
    assert result["proof_of_stationary_root_absence"] is False
    assert "not signature-tested" in result["reason_count_scope"]
    assert "after-run preflight recheck pending" in result["archive_source_scope"]


def test_output_cannot_overwrite_plan_manifest_or_archived_raw(tmp_path):
    archived = decision.RAW / "fv_partial_reanalysis.json"
    protected = (decision.PLAN, decision.MANIFEST, archived)
    originals = [path.read_bytes() for path in protected]
    alias = tmp_path / "archived-alias.json"
    alias.symlink_to(archived)
    for path in (*protected, alias, decision.RAW / "new-output.json"):
        with pytest.raises(ValueError, match="cannot overwrite archived evidence"):
            decision.write_result(path)
    assert [path.read_bytes() for path in protected] == originals
    assert not (decision.RAW / "new-output.json").exists()


@pytest.mark.parametrize("mutation", [
    "issued_response", "zero_exit", "wrong_reason_counts", "resource_killed",
    "path_escape",
])
def test_reclassification_rejects_tampered_archived_semantics(tmp_path, monkeypatch, mutation):
    manifest, raw = _copy_archive(tmp_path, monkeypatch)
    child = raw / "fv_partial_reanalysis.json"
    resource = raw / "fv_partial_reanalysis.resource.json"
    if mutation == "issued_response":
        _rewrite_output(manifest, child, lambda value: value.update(
            response_validation="passed", endpoints=[{"status": "eligible"}]))
    elif mutation == "zero_exit":
        _rewrite_output(manifest, resource, lambda value: value.update(exit_code=0))
    elif mutation == "resource_killed":
        _rewrite_output(manifest, resource, lambda value: value.update(resource_termination="wall"))
    elif mutation == "path_escape":
        archived = json.loads(manifest.read_text())
        key = next(iter(archived["output_sha256"]))
        archived["output_sha256"]["partial_branch_gate_detail_attempt3/../../AGENTS.md"] = archived["output_sha256"].pop(key)
        manifest.write_text(json.dumps(archived, indent=2, sort_keys=True) + "\n")
    else:
        _rewrite_output(manifest, child, lambda value: value.update(
            error=value["error"].replace("'partial branch signature changed': 3",
                                         "'partial branch signature changed': 4")))
    with pytest.raises(ValueError):
        decision.classify()
