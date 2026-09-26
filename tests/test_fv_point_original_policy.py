"""Archived original-point refusals stay distinct from constructed success."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from examples.weather_scenarios import fv_point_original_policy as policy


def _copied_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copied = tmp_path / "evidence"
    copied.mkdir()
    for name in ("point_sector_root_attempt1", "point_merit_root_attempt1"):
        shutil.copytree(policy.EVIDENCE / name, copied / name)
    relative = Path("point_centered_response_attempt1/point_centered_response.json")
    (copied / relative.parent).mkdir()
    shutil.copyfile(policy.EVIDENCE / relative, copied / relative)
    plan = copied / policy.PLAN.name
    shutil.copyfile(policy.PLAN, plan)
    monkeypatch.setattr(policy, "EVIDENCE", copied)
    monkeypatch.setattr(policy, "PLAN", plan)
    return copied


def _change_archived_report(copied: Path, name: str, filename: str, edit) -> None:
    directory = copied / name
    report = directory / filename
    document = json.loads(report.read_text())
    edit(document)
    report.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    manifest = directory / "manifest.json"
    pinned = json.loads(manifest.read_text())
    pinned["sha256"][report.name] = hashlib.sha256(report.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")


def _change_archived_child(copied: Path, name: str, edit) -> None:
    stem = "point_sector_root" if "sector" in name else "point_merit_root"
    _change_archived_report(copied, name, f"{stem}.json", edit)


def test_exact_original_input_is_unsupported_by_current_search_policies():
    result = policy.classify()
    sector = result["attempts"]["triple_decrease"]
    merit = result["attempts"]["gradient_merit"]
    assert result["numerical_solver_runs"] == 0
    assert result["support_status"] == "unsupported_by_current_declared_nominal_search_policies"
    assert result["response_computed"] is False
    assert result["stationary_root_absence_proved"] is False
    assert result["original_problem_sha256"] != result["constructed_problem_sha256"]
    assert [sector["accepted_signature_switches"], merit["accepted_signature_switches"]] == [2, 1]
    assert sector["final_policy_rejections"] == merit["final_policy_rejections"] == 16
    assert sector["last_accepted_gradient_max"] > 1e-10
    assert merit["last_accepted_gradient_max"] > 1e-10
    assert sector["child_exit_code"] == merit["child_exit_code"] == 2


@pytest.mark.parametrize("mutation", [
    "published_response", "false_gradient", "accepted_final_trial",
    "wrong_merit_seed", "constructed_identity_reused", "changed_raw_hash",
    "child_status", "parent_status", "resource_limited",
])
def test_archive_semantic_mutations_cannot_be_reclassified_as_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
):
    copied = _copied_evidence(tmp_path, monkeypatch)
    sector = copied / "point_sector_root_attempt1"
    merit = copied / "point_merit_root_attempt1"
    if mutation == "published_response":
        _change_archived_child(copied, merit.name,
                               lambda report: report.update(response={"total": 1.0}))
    elif mutation == "false_gradient":
        def change(report):
            [trial for trial in report["trial_records"] if trial.get("accepted")][-1]["gradient_max"] = 1e-12
        _change_archived_child(copied, merit.name, change)
    elif mutation == "accepted_final_trial":
        def change(report):
            [trial for trial in report["trial_records"] if trial.get("iteration") == 7][0]["accepted"] = True
        _change_archived_child(copied, sector.name, change)
    elif mutation == "wrong_merit_seed":
        _change_archived_child(copied, merit.name,
                               lambda report: report.update(seed_control_sha256="f" * 64))
    elif mutation == "child_status":
        _change_archived_child(copied, merit.name,
                               lambda report: report.update(numerical_status="merit_root_margin_qualified"))
    elif mutation == "parent_status":
        _change_archived_report(copied, merit.name, "point_merit_root.run.json",
                                lambda report: report.update(execution_status="failed"))
    elif mutation == "resource_limited":
        _change_archived_report(copied, sector.name, "point_sector_root.run.json",
                                lambda report: report["resource"].update(resource_termination="wall_time_limit"))
    elif mutation == "constructed_identity_reused":
        constructed = copied / "point_centered_response_attempt1/point_centered_response.json"
        report = json.loads(constructed.read_text())
        original = json.loads((merit / "point_merit_root.json").read_text())
        report["input_before"]["problem"] = original["input_before"]["problem_identity"]
        constructed.write_text(json.dumps(report))
    else:
        with (sector / "point_sector_root.json").open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError):
        policy.classify()


def test_raw_manifest_cannot_escape_its_attempt_directory(tmp_path: Path, monkeypatch):
    copied = _copied_evidence(tmp_path, monkeypatch)
    manifest = copied / "point_merit_root_attempt1/manifest.json"
    pinned = json.loads(manifest.read_text())
    key = next(iter(pinned["sha256"]))
    pinned["sha256"]["../point_sector_root_attempt1/point_sector_root.json"] = pinned["sha256"].pop(key)
    manifest.write_text(json.dumps(pinned))
    monkeypatch.setattr(policy, "EXPECTED_RAW_MANIFEST_SHA256", {
        **policy.EXPECTED_RAW_MANIFEST_SHA256,
        "point_merit_root_attempt1": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    with pytest.raises(ValueError, match="archived artifact changed"):
        policy.classify()


def test_coordinated_resealed_attempts_cannot_change_the_original_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    copied = _copied_evidence(tmp_path, monkeypatch)
    sector = "point_sector_root_attempt1"
    merit = "point_merit_root_attempt1"
    for name in (sector, merit):
        def wrong_identity(report):
            for key in ("input_before", "input_after"):
                report[key]["problem_identity"]["fixed_problem_sha256"] = "a" * 64
                report[key]["tensor_sha256"]["parameters"] = "b" * 64
        _change_archived_child(copied, name, wrong_identity)
    sector_sha = hashlib.sha256((copied / sector / "point_sector_root.json").read_bytes()).hexdigest()
    _change_archived_child(
        copied, merit,
        lambda report: report.update(prior_report_sha256=sector_sha),
    )
    # Even a coordinated update of both raw manifests cannot repin the
    # scientific input contract after the fact.
    monkeypatch.setattr(policy, "EXPECTED_RAW_MANIFEST_SHA256", {
        name: hashlib.sha256((copied / name / "manifest.json").read_bytes()).hexdigest()
        for name in (sector, merit)
    })
    with pytest.raises(ValueError, match="archived refusal changed meaning"):
        policy.classify()
