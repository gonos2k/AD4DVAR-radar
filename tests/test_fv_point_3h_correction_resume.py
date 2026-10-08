from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.weather_scenarios import fv_point_3h_correction_resume as resume
from examples.weather_scenarios import fv_point_3h_model_guided_continuation as guided


def _plan(tmp_path: Path) -> tuple[Path, str]:
    inherited = json.loads(resume.PRODUCER_PLAN.read_text())
    manifest = json.loads(resume.SOURCE_MANIFEST.read_text())
    sources = dict(inherited["source_files"])
    sources[resume.GUIDED] = resume._sha(resume.ROOT / resume.GUIDED)
    sources[resume.SELF] = resume._sha(resume.ROOT / resume.SELF)
    sources[resume.TEST] = resume._sha(resume.ROOT / resume.TEST)
    archives = dict(inherited["archive_files"])
    for path in (resume.PRODUCER_PLAN, resume.BASE_STEP, resume.BASE_RUN,
                 resume.BASE_RESOURCE, resume.SOURCE_MANIFEST):
        archives[path.relative_to(resume.ROOT).as_posix()] = resume._sha(path)
    for item in manifest["snapshots"].values():
        path = resume.ROOT / item["archive_path"]
        archives[item["archive_path"]] = resume._sha(path)
    plan = {"experiment_kind": "model_guided_correction_resume",
        "policy": resume.policy(), "producing_plan": resume.PRODUCER_PLAN.relative_to(resume.ROOT).as_posix(),
        "producing_plan_sha256": resume.PRODUCER_PLAN_SHA,
        "base_step": resume.BASE_STEP.relative_to(resume.ROOT).as_posix(),
        "base_run": resume.BASE_RUN.relative_to(resume.ROOT).as_posix(),
        "base_resource": resume.BASE_RESOURCE.relative_to(resume.ROOT).as_posix(),
        "base_step_sha256": resume.BASE_STEP_SHA, "base_run_sha256": resume.BASE_RUN_SHA,
        "base_resource_sha256": resume.BASE_RESOURCE_SHA,
        "base_control_sha256": resume.BASE_CONTROL_SHA,
        "producer_source_snapshots": manifest["snapshots"],
        "source_files": sources, "archive_files": archives}
    path = resume.ROOT / f".test-correction-resume-{tmp_path.name}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def valid_plan(monkeypatch, tmp_path, request):
    path, digest = _plan(tmp_path)
    monkeypatch.setattr(resume, "PLAN", path)
    request.addfinalizer(lambda: path.unlink(missing_ok=True))
    return path, digest


def test_real_pr263_receipts_normalize_as_fresh_resume_base(valid_plan):
    path, digest = valid_plan
    plan = resume._load_plan(path, digest)
    base = resume._load_base(plan)
    assert base["accepted_control_sha256"] == resume.BASE_CONTROL_SHA
    assert base["current_state"]["control_sha256"] == resume.BASE_CONTROL_SHA
    assert base["resume_plan_sha256"] == resume.PRODUCER_PLAN_SHA
    assert "cycle_reference" not in base


def test_plan_digest_is_required_even_when_receipts_are_real(valid_plan):
    path, _ = valid_plan
    with pytest.raises(ValueError, match="plan identity"):
        resume._load_plan(path, "0" * 64)


@pytest.mark.parametrize("damage", ["uncommitted", "lineage", "last_gradient"])
def test_loader_rejects_open_or_mutated_pr263_endpoint(monkeypatch, valid_plan, damage):
    path, digest = valid_plan
    resume._load_plan(path, digest)
    original_loads = json.loads

    def altered_loads(value, *args, **kwargs):
        parsed = original_loads(value, *args, **kwargs)
        if isinstance(parsed, dict) and parsed.get("plan_sha256") == resume.PRODUCER_PLAN_SHA:
            if damage == "uncommitted":
                parsed["iterations"][-1]["committed"] = False
            elif damage == "lineage":
                parsed["iterations"][-1]["base_control_sha256"] = "f" * 64
            else:
                parsed["current_state"]["gradient"][0] += 1.0
        return parsed

    monkeypatch.setattr(resume.json, "loads", altered_loads)
    with pytest.raises(ValueError, match="PR263 accepted endpoint"):
        resume._load_base({"archive_files": json.loads(path.read_text())["archive_files"]})


def test_guided_dispatch_and_wrapper_reuse_existing_runner(monkeypatch, tmp_path):
    plan_path = tmp_path / "CORRECTION_RESUME_PLAN_20261008.json"
    plan_path.write_text("{}")
    monkeypatch.setattr(guided, "EVIDENCE", tmp_path)
    marker = {"resume": True}
    monkeypatch.setattr(resume, "_load_plan", lambda *_args: marker)
    assert guided._load_plan(plan_path, "ignored") is marker
    monkeypatch.setattr(resume, "_load_base", lambda _plan: marker)
    assert guided._load_base({"experiment_kind": "model_guided_correction_resume"}) is marker
    called = {}

    def existing_runner(*args):
        called["args"] = args
        return marker

    monkeypatch.setattr(guided, "run", existing_runner)
    args = (plan_path, "digest", tmp_path / "step.json", tmp_path / "resource.json",
            tmp_path / "step.log")
    assert resume.run(*args) is marker
    assert called["args"] == args
