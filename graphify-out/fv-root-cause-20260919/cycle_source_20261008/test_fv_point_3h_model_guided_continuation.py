from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_model_guided_continuation as guided
from examples.weather_scenarios import fv_point_3h_qy32_tangent_step as tangent


def test_model_alpha_is_full_space_radius_and_phi_model_bound():
    g = torch.zeros(26, dtype=torch.float64)
    g[0] = 0.2
    hd = -g
    alpha = guided.model_alpha(g, hd)
    assert alpha == pytest.approx(min(1.0, guided.RADIUS / 0.2, 1.0))
    assert alpha * float(torch.linalg.vector_norm(g)) <= guided.RADIUS
    with pytest.raises(guided.guard_policy.StepRefusal):
        guided.model_alpha(torch.zeros_like(g), hd)


def test_linearized_phi_is_exact_norm_of_linearized_gradient():
    g = torch.arange(26, dtype=torch.float64) / 20
    d, hd, alpha = -g, -0.5 * g, 0.1
    model = guided.first_order_model(1.0, 0.5, g, d, hd, alpha)
    expected = torch.dot(g + alpha * hd, g + alpha * hd) / 2
    assert model["predicted_Phi_from_linear_gradient"] == pytest.approx(float(expected))
    assert torch.allclose(torch.tensor(model["predicted_gradient"], dtype=torch.float64), g + alpha * hd)


def test_hvp_started_count_is_written_only_after_deadline_admission():
    control = torch.zeros(26, dtype=torch.float64)
    params = torch.zeros(13, dtype=torch.float64)
    direction = torch.ones_like(control)
    starts = 0
    completions = 0

    def on_start():
        nonlocal starts
        starts += 1

    def on_complete():
        nonlocal completions
        completions += 1

    with pytest.raises(TimeoutError):
        guided._current_hvp(lambda value, _p: value, control, params, direction,
                            time.monotonic() - 1, on_start, on_complete)
    assert starts == completions == 0
    product = guided._current_hvp(lambda value, _p: value, control, params, direction,
                                  time.monotonic() + 10, on_start, on_complete)
    assert torch.equal(product, direction)
    assert (starts, completions) == (1, 1)


def test_actual_phi_armijo_refuses_nonlinear_phi_increase_without_j_floor():
    base_j = 1.0e12
    result = guided.merit_acceptance(base_j, 0.5, base_j, 0.5001,
                                     0.01, -0.1, -0.2)
    assert result["J_armijo_passed"] is True  # floating-point J threshold rounds to its offset
    assert result["Phi_armijo_passed"] is False
    assert result["phi_decrease_resolved"] is False
    assert result["accepted"] is False


def test_iteration_commit_updates_state_only_after_closure():
    control0 = torch.zeros(26, dtype=torch.float64)
    control1 = control0.clone()
    control1[0] = 0.1
    measure0 = {"objective": 1.0, "phi": 1.0, "gradient": [1.0] * 26,
        "gradient_inf": 1.0, "gradient_norm": 5.0,
        "branch": {"signature_sha256": "a"}, "branch_partition": {}}
    measure1 = {**measure0, "objective": 0.9, "gradient": [0.5] * 26,
                "gradient_inf": 0.5, "gradient_norm": 2.5,
                "branch": {"signature_sha256": "b"}}
    record: dict[str, Any] = {"iterations": []}
    first = {"accepted_control": control1.tolist(), "accepted_control_sha256": "one"}
    assert guided._commit_iteration(record, control1, measure1, first, closure_ok=True)
    committed_state = dict(record["current_state"])
    second = {"accepted_control": control0.tolist(), "accepted_control_sha256": "two"}
    assert not guided._commit_iteration(record, control0, measure0, second, closure_ok=False)
    assert len(record["iterations"]) == 1
    assert record["current_state"] == committed_state


class _Quadratic:
    def __init__(self, scale: float = 1.0, offset: float = 1.0e12) -> None:
        self.scale = scale
        self.offset = offset

    def objective(self, control: torch.Tensor, _parameters: torch.Tensor) -> torch.Tensor:
        return control.new_tensor(self.offset) + self.scale * control.square().sum() / 2


def _blackbox_case(monkeypatch, *, fail_second_hvp: bool = False,
                   scale: float = 1.0, offset: float = 1.0e12,
                   initial: float = 0.2):
    control = torch.zeros(26, dtype=torch.float64)
    control[0] = initial
    parameters = torch.zeros(13, dtype=torch.float64)
    truth = torch.zeros(1, dtype=torch.float64)
    original = torch.zeros(1, dtype=torch.float64)
    problem = _Quadratic(scale, offset)
    parameter_sha = tangent._tensor_sha(parameters)
    truth_sha = "truth-fixed"
    base_identity = {"control_sha256": tangent._tensor_sha(control),
        "parameters_sha256": parameter_sha, "terminal_truth_sha256": truth_sha,
        "archived_input": {"profile": "same"}}

    def input_identity(_problem, _original, value, _params, _truth):
        return {**base_identity, "control_sha256": tangent._tensor_sha(value)}

    branch = {"status": "passed_strict_branch", "signature_sha256": "fixed-branch",
              "euler_stages": 3600, "choice_stage_count": 3600, "face_sign_stage_count": 3600}
    def measure(_problem, _params, value, _eta, gradient_fn, *, with_gradient, deadline):
        objective = problem.objective(value, _params)
        gradient = gradient_fn(value, _params)
        phi = torch.dot(gradient, gradient) / 2
        return {"objective": float(objective), "phi": float(phi), "gradient": gradient.tolist(),
            "gradient_inf": float(gradient.abs().max()), "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "branch": branch, "branch_partition": {"full_sha256": "fixed-branch"},
            "geometry": {}, "branch_margins": {"complete": True}, "branch_scope": "test",
            "branch_signature": {"choices": [], "face_signs": []}, "static_flux_signs": {}}

    fresh_gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
    state = {"objective": float(problem.objective(control, parameters)),
             "phi": float(torch.dot(fresh_gradient, fresh_gradient) / 2),
             "gradient": fresh_gradient.tolist(), "gradient_inf": float(fresh_gradient.abs().max()),
             "branch": branch}
    trial = {"status": "accepted", "control_sha256": tangent._tensor_sha(control),
             "objective": state["objective"], "phi": state["phi"],
             "gradient": state["gradient"], "gradient_inf": state["gradient_inf"],
             "J_armijo_passed": True, "Phi_armijo_passed": True,
             "branch_signature_sha256": "fixed-branch", "branch": branch}
    raw = {"accepted_control": control.tolist(), "accepted_control_sha256": tangent._tensor_sha(control),
           "parameters_sha256": parameter_sha, "input_after": base_identity,
           "runtime": {"runtime": "fixed"}, "current_state": state,
           "accepted_objective": state["objective"], "accepted_phi": state["phi"],
           "accepted_gradient": state["gradient"], "trials": [trial]}
    monkeypatch.setattr(guided, "CONTROL_SHA", tangent._tensor_sha(control))
    monkeypatch.setattr(guided, "PARAMETERS_SHA", parameter_sha)
    monkeypatch.setattr(guided, "_load_plan", lambda *_args: {"source_files": {}, "archive_files": {}})
    monkeypatch.setattr(guided, "_load_base", lambda *_args: raw)
    monkeypatch.setattr(guided.seed, "_prepare_fixed_seed", lambda: (
        problem, original, control, parameters, truth, base_identity))
    monkeypatch.setattr(guided.seed, "_input_identity", input_identity)
    monkeypatch.setattr(guided.qy, "production_qy32", lambda *_args: 0.0)
    monkeypatch.setattr(guided.qy, "_measure", measure)
    monkeypatch.setattr(guided, "_static_zero_faces", lambda *_args: [])
    monkeypatch.setattr(guided.guard_policy.blocks, "runtime_identity", lambda: {"runtime": "fixed"})
    plan_sha = "p" * 64
    monkeypatch.setattr(guided, "_sha", lambda path: plan_sha if Path(path) == guided.PLAN
                        else hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if fail_second_hvp:
        live = guided._current_hvp
        count = 0
        def fail_after_one(gradient_fn, c, p, d, deadline, on_start, on_complete):
            nonlocal count
            count += 1
            if count == 2:
                raise TimeoutError("synthetic second-iteration budget")
            return live(gradient_fn, c, p, d, deadline, on_start, on_complete)
        monkeypatch.setattr(guided, "_current_hvp", fail_after_one)
    return control, raw, plan_sha


def test_continuation_uses_each_new_point_gradient_and_hvp(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch)
    output = tmp_path / "step.json"
    result = guided._run_child(guided.PLAN, plan_sha, output)
    assert result["optimizer_steps_applied"] == 3
    assert result["hvp_calls"] == result["hvp_calls_completed"] == 3
    assert len(result["iterations"]) == 3
    for index, iteration in enumerate(result["iterations"]):
        expected_gradient = torch.tensor(
            _blackbox_initial_gradient(control) if index == 0
            else result["iterations"][index - 1]["accepted_gradient"], dtype=torch.float64)
        assert torch.allclose(torch.tensor(iteration["direction"], dtype=torch.float64), -expected_gradient)
        assert torch.allclose(torch.tensor(iteration["H_direction"], dtype=torch.float64), -expected_gradient)
    assert result["candidate_committed"] is True


def _blackbox_initial_gradient(control: torch.Tensor) -> list[float]:
    return control.tolist()  # Identity Hessian for the mock quadratic.


def test_second_iteration_budget_preserves_first_committed_point_and_hvp_count(
        monkeypatch, tmp_path):
    _, _, plan_sha = _blackbox_case(monkeypatch, fail_second_hvp=True)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "partial.json")
    assert result["numerical_status"] == "budget_refusal"
    assert result["optimizer_steps_applied"] == 1
    assert result["hvp_calls"] == result["hvp_calls_completed"] == 1
    assert result["candidate_committed"] is True
    assert result["current_control_sha256"] == result["iterations"][0]["accepted_control_sha256"]
    assert result["current_state"]["gradient"] == result["iterations"][0]["accepted_gradient"]
    assert result["active_iteration"]["hvp_calls_started"] == 0


def test_large_phi_increase_backtracks_to_smaller_candidate(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch)
    actual_measure = guided.qy._measure
    monkeypatch.setattr(guided, "model_alpha", lambda *_args: 0.2)

    def nonlinear_measure(problem, params, candidate, eta, gradient_fn, *, with_gradient, deadline):
        result = actual_measure(problem, params, candidate, eta, gradient_fn,
                                with_gradient=with_gradient, deadline=deadline)
        if float(torch.linalg.vector_norm(candidate - control)) > 0.03:
            result["phi"] = 0.125  # The larger candidate increases actual Phi.
        return result

    monkeypatch.setattr(guided.qy, "_measure", nonlinear_measure)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "backtrack.json")
    trials = result["iterations"][0]["trials"]
    assert trials[0]["phi"] > result["iterations"][0]["base_phi"]
    assert trials[0]["Phi_armijo_passed"] is False
    assert trials[1]["status"] == "accepted"
    assert trials[1]["alpha"] == pytest.approx(trials[0]["alpha"] / 2)


def test_near_root_uses_gradient_threshold_without_an_objective_floor(monkeypatch, tmp_path):
    control, _, plan_sha = _blackbox_case(monkeypatch, scale=1e-8, offset=1.0, initial=1e-4)
    result = guided._run_child(guided.PLAN, plan_sha, tmp_path / "root-pending.json")
    assert result["numerical_status"] == "root_pending_audit"
    assert result["root_pending_audit"] is True
    assert result["full_root_claim"] is False
    assert result["hvp_calls"] == 0
    assert result["optimizer_steps_applied"] == 0
    assert result["current_control"] == control.tolist()
    assert result["current_state"]["objective"] == 1.0


def _resume_artifacts(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(guided, "ROOT", root)
    plan_path = root / "MODEL_GUIDED_FULL_SPACE_PLAN_20261008.json"
    legacy_plan_path = root / "MODEL_GUIDED_FULL_SPACE_PLAN_20261007.json"
    manifest_path = root / "model_guided_source_20261007/manifest.json"
    step_path = root / "model_guided_attempt1/step.json"
    parent_path = step_path.with_suffix(".run.json")
    resource_path = step_path.with_suffix(".resource.json")
    monkeypatch.setattr(guided, "PLAN", plan_path)
    monkeypatch.setattr(guided, "LEGACY_PLAN", legacy_plan_path)
    monkeypatch.setattr(guided, "SOURCE_SNAPSHOT_MANIFEST", manifest_path)
    monkeypatch.setattr(guided, "LEGACY_STEP", step_path)
    monkeypatch.setattr(guided, "LEGACY_STEP_PARENT", parent_path)
    monkeypatch.setattr(guided, "LEGACY_STEP_RESOURCE", resource_path)

    snapshot_values = {
        guided.SELF: ("model_guided_source_20261007/producer.py", "old producer bytes"),
        guided.TEST: ("model_guided_source_20261007/tests.py", "old test bytes"),
    }
    source_hashes: dict[str, str] = {}
    snapshots: dict[str, dict[str, str]] = {}
    for source, (archive, contents) in snapshot_values.items():
        old_bytes = contents.encode()
        source_hashes[source] = hashlib.sha256(old_bytes).hexdigest()
        snapshots[source] = {"archive_path": archive, "sha256": source_hashes[source]}
        archived = root / archive
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(old_bytes)
        current = root / source
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text(f"new live {source}")
    op_name = "src/operator.py"
    op_path = root / op_name
    op_path.parent.mkdir(parents=True, exist_ok=True)
    op_path.write_text("immutable operator")
    source_hashes[op_name] = hashlib.sha256(op_path.read_bytes()).hexdigest()
    old_archive_path = root / "evidence/old.json"
    old_archive_path.parent.mkdir(parents=True, exist_ok=True)
    old_archive_path.write_text("old producer archive")
    old_archive_hash = hashlib.sha256(old_archive_path.read_bytes()).hexdigest()
    legacy_plan = {"source_files": source_hashes.copy(),
                   "archive_files": {"evidence/old.json": old_archive_hash}}
    legacy_plan_path.write_text(json.dumps(legacy_plan))
    legacy_plan_sha = hashlib.sha256(legacy_plan_path.read_bytes()).hexdigest()
    manifest = {"snapshots": snapshots}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(guided, "LEGACY_PLAN_SHA", legacy_plan_sha)
    monkeypatch.setattr(guided, "SOURCE_SNAPSHOT_MANIFEST_SHA", manifest_sha)
    monkeypatch.setattr(guided, "RESUME_CONTROL_SHA", "new-control-sha")
    monkeypatch.setattr(guided.tangent, "_load_plan", lambda *_args: legacy_plan)
    monkeypatch.setattr(guided.tangent, "_pinned_path", lambda name, digest: _pin(root, name, digest))

    control = [0.0] * 26
    control[0] = 0.1
    branch = {"status": "passed_strict_branch", "signature_sha256": "branch-final"}
    params_sha = guided.PARAMETERS_SHA
    input_before = {"control_sha256": "old-control", "parameters_sha256": params_sha,
                    "terminal_truth_sha256": "truth", "archived_input": {"fixed": 1}}
    input_after = {**input_before, "control_sha256": "new-control-sha"}
    gradient = [0.0] * 26
    current_state = {"objective": 0.9, "phi": 0.01, "gradient": gradient,
                     "gradient_inf": 0.0, "branch": branch,
                     "branch_partition": {"full_sha256": "branch-final"}}
    accepted_trial = {"status": "accepted", "control_sha256": "new-control-sha",
        "objective": 0.9, "phi": 0.01, "gradient": gradient,
        "J_armijo_passed": True, "Phi_armijo_passed": True,
        "strict_point_passed": True, "branch": branch}
    step: dict[str, Any] = {"phase": "finished", "execution_status": "completed",
        "numerical_status": "direction_refusal", "plan_sha256": legacy_plan_sha,
        "optimizer_steps_applied": 1, "hvp_calls": 2, "hvp_calls_completed": 2,
        "candidate_committed": True, "active_candidate_committed": False,
        "base_control_sha256": "old-control", "current_control": control,
        "current_control_sha256": "new-control-sha", "parameters_sha256": params_sha,
        "current_state": current_state, "iterations": [{"status": "accepted",
            "committed": True, "base_control_sha256": "old-control",
            "accepted_control": control, "accepted_control_sha256": "new-control-sha",
            "accepted_objective": 0.9, "accepted_phi": 0.01,
            "accepted_gradient": gradient, "accepted_gradient_inf": 0.0,
            "trials": [accepted_trial]}],
        "source_before": source_hashes, "source_after": source_hashes,
        "source_unchanged": True, "fixed_input_unchanged": True,
        "input_before": input_before, "input_after": input_after,
        "runtime": {"runtime": "fixed"}, "runtime_after": {"runtime": "fixed"}}
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text(json.dumps(step))
    step_sha = hashlib.sha256(step_path.read_bytes()).hexdigest()
    monkeypatch.setattr(guided, "LEGACY_STEP_SHA", step_sha)
    resource = {"exit_code": 0, "resource_termination": None, "monitor_error": None,
        "received_sigterm": False, "wall_limit_seconds": guided.WALL_SECONDS,
        "rss_limit_bytes": guided.RSS_BYTES, "elapsed_seconds": 1.0,
        "sampled_peak_rss_bytes": 1000, "child_process_group_cleanup_sent": False,
        "child_process_group_cleanup_error": None}
    resource_path.write_text(json.dumps(resource))
    resource_sha = hashlib.sha256(resource_path.read_bytes()).hexdigest()
    monkeypatch.setattr(guided, "LEGACY_STEP_RESOURCE_SHA", resource_sha)
    parent = {"execution_status": "completed", "child_read_error": None,
              "child_sha256": step_sha, "numerical_status": step["numerical_status"]}
    parent_path.write_text(json.dumps(parent))
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    monkeypatch.setattr(guided, "LEGACY_STEP_PARENT_SHA", parent_sha)
    archive_files = {**legacy_plan["archive_files"],
        legacy_plan_path.relative_to(root).as_posix(): legacy_plan_sha,
        step_path.relative_to(root).as_posix(): step_sha,
        parent_path.relative_to(root).as_posix(): parent_sha,
        resource_path.relative_to(root).as_posix(): resource_sha,
        manifest_path.relative_to(root).as_posix(): manifest_sha}
    archive_files.update({entry["archive_path"]: entry["sha256"]
                          for entry in snapshots.values()})
    live_sources = dict(source_hashes)
    for source in snapshots:
        live_sources[source] = hashlib.sha256((root / source).read_bytes()).hexdigest()
    plan: dict[str, Any] = {"experiment_kind": "model_guided_resume",
        "producing_plan": legacy_plan_path.relative_to(root).as_posix(),
        "producing_plan_sha256": legacy_plan_sha,
        "base_step": step_path.relative_to(root).as_posix(), "base_step_sha256": step_sha,
        "base_run": parent_path.relative_to(root).as_posix(), "base_run_sha256": parent_sha,
        "base_resource": resource_path.relative_to(root).as_posix(),
        "base_resource_sha256": resource_sha, "base_control_sha256": "new-control-sha",
        "producer_source_snapshots": snapshots, "policy": guided.policy(),
        "source_files": live_sources, "archive_files": archive_files}
    plan_path.write_text(json.dumps(plan))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    return plan, plan_sha, step, parent, resource


def _pin(root: Path, name: str, digest: str) -> Path:
    path = root / name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return path


def test_resume_loader_checks_archived_source_lineage_and_last_state(tmp_path, monkeypatch):
    plan, plan_sha, step, _, _ = _resume_artifacts(tmp_path, monkeypatch)
    loaded = guided._load_plan(guided.PLAN, plan_sha)
    normalized = guided._load_resume_base(loaded)
    assert normalized["accepted_control_sha256"] == "new-control-sha"
    assert normalized["accepted_objective"] == normalized["current_state"]["objective"]

    bad_plan: dict[str, Any] = dict(plan)
    bad_plan["producer_source_snapshots"] = dict(plan["producer_source_snapshots"])
    bad_plan["producer_source_snapshots"][guided.SELF] = {
        **bad_plan["producer_source_snapshots"][guided.SELF], "sha256": "bad"}
    guided.PLAN.write_text(json.dumps(bad_plan))
    bad_sha = hashlib.sha256(guided.PLAN.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="snapshot"):
        guided._load_plan(guided.PLAN, bad_sha)

    plan["producer_source_snapshots"] = json.loads(
        (tmp_path / "model_guided_source_20261007/manifest.json").read_text())["snapshots"]
    step["current_state"]["phi"] = 0.5
    step_path = tmp_path / plan["base_step"]
    step_path.write_text(json.dumps(step))
    new_step_sha = hashlib.sha256(step_path.read_bytes()).hexdigest()
    plan["base_step_sha256"] = new_step_sha
    plan["archive_files"][plan["base_step"]] = new_step_sha
    parent_path = tmp_path / plan["base_run"]
    parent = json.loads(parent_path.read_text())
    parent["child_sha256"] = new_step_sha
    parent_path.write_text(json.dumps(parent))
    new_parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    plan["base_run_sha256"] = new_parent_sha
    plan["archive_files"][plan["base_run"]] = new_parent_sha
    with pytest.raises(ValueError, match="resume point"):
        guided._load_resume_base(plan)
