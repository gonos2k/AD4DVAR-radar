"""Contract checks for the fixed-dynamics field correction experiment."""
from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest
import torch
from torch import Tensor

from advar.local_refinement import RefinementTrial
from examples.weather_scenarios import fv_partial_field_correction_probe as probe


def _trial(old: float, new: float) -> RefinementTrial:
    return RefinementTrial(
        iteration=1, backtrack=0,
        current_objective=old, candidate_objective=new,
        current_gradient_norm=2.0, candidate_gradient_norm=1.0,
        current_gradient_max=1.0, candidate_gradient_max=0.5,
        current_branch={"choices": [0]}, candidate_branch={"choices": [0]},
        step_scale=1.0, normalized_slope=-0.25, armijo_ratio=0.5,
    )


def test_reduced_gradient_hessian_and_fixed_dynamics_match_full_quadratic() -> None:
    torch.manual_seed(31)
    matrix = torch.randn(26, 26, dtype=torch.float64)
    hessian = matrix.T @ matrix + torch.eye(26, dtype=torch.float64)
    linear = torch.randn(26, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    dynamics = torch.randn(6, dtype=torch.float64, requires_grad=True)
    dynamics_bytes = dynamics.detach().numpy().tobytes()
    def objective(control: Tensor, _parameters: Tensor) -> Tensor:
        return 0.5 * control @ hessian @ control - linear @ control

    reduced = probe._field_objective(objective, dynamics)
    field = torch.randn(20, dtype=torch.float64)
    field_gradient = torch.func.grad(reduced, argnums=0)(field, parameters)
    reduced_hessian = torch.func.jacrev(
        torch.func.grad(reduced, argnums=0), argnums=0,
    )(field, parameters)

    full = probe._full_control(field, dynamics.detach())
    full_gradient = torch.func.grad(objective, argnums=0)(full, parameters)
    expected_gradient = hessian[:20, :20] @ field + hessian[:20, 20:] @ dynamics.detach() - linear[:20]
    assert torch.allclose(field_gradient, expected_gradient, rtol=1e-13, atol=1e-13)
    assert torch.allclose(reduced_hessian, hessian[:20, :20], rtol=1e-13, atol=1e-13)
    assert torch.allclose(field_gradient, full_gradient[:20], rtol=1e-13, atol=1e-13)
    assert full_gradient[20:].shape == (6,)
    assert dynamics.grad is None
    assert dynamics.detach().numpy().tobytes() == dynamics_bytes
    assert full[20:].detach().numpy().tobytes() == dynamics_bytes


def test_objective_gate_preserves_armijo_when_j_does_not_increase() -> None:
    assert probe.objective_gate(_trial(3.0, 2.0)) is None
    tiny = 128.0 * torch.finfo(torch.float64).eps * 3.0
    assert probe.objective_gate(_trial(3.0, 3.0 + tiny)) is None
    assert probe.objective_gate(_trial(3.0, 3.0 + 1e-10)) == (
        False, "actual_objective_increase",
    )
    assert probe.objective_gate(_trial(math.inf, 2.0)) == (
        False, "actual_objective_nonfinite",
    )


def _fake_child(source: dict[str, str], inputs: dict[str, Any]) -> dict[str, Any]:
    seed, control = probe.seed_probe._seed_evidence()
    dynamics_sha = probe._tensor_sha(control[20:])
    problem, _, parameters = probe._problem()
    branch_raw, _, face_margin = probe.preflight.branch_with_face_margin(
        problem, control, parameters,
    )
    branch = probe._branch_summary(branch_raw, face_margin)
    full_gradient = probe._gradient_audit(problem, control, parameters)
    reduced = probe._field_objective(problem.objective, control[20:])
    reduced_gradient = torch.func.grad(reduced, argnums=0)(control[:20], parameters)
    reduced_gradient_max = float(reduced_gradient.abs().max())
    parameter_sha = inputs["fixed_input_fields"]["tensor_sha256"]["parameters"]
    return {
        "phase": "finished", "numerical_status": "field_correction_refused",
        "pid": 812, "plan_sha256": probe.PLAN_SHA256,
        "reviewed_probe_sha256": "a" * 64,
        "shared_resource_runner_sha256": probe.RUNNER_SHA256,
        "source_before": source, "source_after": source,
        "input_before": inputs, "input_after": inputs,
        "parameters_sha256": parameter_sha,
        "parameters_sha256_after": parameter_sha,
        "seed_control": control.tolist(),
        "final_control": control.tolist(),
        "seed_control_sha256": probe._tensor_sha(control),
        "final_control_sha256": probe._tensor_sha(control),
        "fixed_dynamics_sha256": dynamics_sha,
        "final_fixed_dynamics_sha256": dynamics_sha,
        "fixed_dynamics_byte_identical": True,
        "seed_branch": branch,
        "final_branch": branch,
        "final_objective": float(problem.objective(control, parameters)),
        "final_reduced_gradient_max": reduced_gradient_max,
        "final_full_gradient": full_gradient,
        "reduced_stationarity": reduced_gradient_max < probe.STATIONARITY_TOLERANCE,
        "full_stationarity": full_gradient["full_inf"] < probe.STATIONARITY_TOLERANCE,
        "response_eligibility": "not_eligible",
        "iterations": 1,
        "linear_solves": [{
            "iteration": 1, "rtol": probe.TRUE_RESIDUAL_TOLERANCE,
            "max_iterations": 80, "converged": True,
            "true_relative_residual": 2e-10,
            "error": "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance",
        }],
        "refusal": "RefinementNumericalRefusal: matrix-free Newton PCG true residual exceeds tolerance",
        "raw_unchanged": True,
        "seed_raw_hashes_before": probe.seed_probe._seed_record_hashes(),
        "seed_raw_hashes_after": probe.seed_probe._seed_record_hashes(),
        "response_validation": "not_performed",
        "physical_validation": "not_performed",
        "execution_completion": "child_completed",
    }


@pytest.fixture
def archived_seed_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Supply a test vector, without certifying an old execution on this host.
    # Production seed hashes/source/runtime validation stays in the loader tests.
    control, _, _ = probe.seed_probe._archived_seed_control()
    seed = json.loads((probe.seed_probe.SEED_ARCHIVE / "alternate_seed.json").read_text())
    problem, initial_control, parameters = probe._problem()
    mode = problem.frozen.observation_whitener.mode
    assert mode is not None
    host_input_identity = {
        "current_problem_identity": problem.identity,
        "fixture_scope": "host-local synthetic fake-child/audit input; not archived certification",
        "fixed_input_fields": {
            "tensor_sha256": {
                "parameters": probe._tensor_sha(parameters),
                "verification": probe._tensor_sha(problem.verification),
                "observation_dbz": probe._tensor_sha(problem.observations.dbz),
                "valid_mask": probe._tensor_sha(problem.observations.valid_mask),
                "missing_mask": probe._tensor_sha(problem.observations.missing_mask),
                "whitener_mode": probe._tensor_sha(mode),
                "initial_control": probe._tensor_sha(initial_control),
            },
        },
    }
    monkeypatch.setattr(
        probe.seed_probe, "_seed_evidence",
        lambda: (copy.deepcopy(seed), control.clone()),
    )
    monkeypatch.setattr(probe, "_problem", lambda: (problem, initial_control, parameters))
    monkeypatch.setattr(
        probe.prior, "_preflight_identity", lambda: copy.deepcopy(host_input_identity),
    )


def _fake_resource(command: list[str]) -> dict[str, Any]:
    return {
        "command": command, "child_pid": 812, "exit_code": 2,
        "resource_termination": None, "monitor_error": None,
        "wall_limit_seconds": 300, "rss_limit_bytes": 1024**3,
        "rss_samples": 2, "sampled_peak_rss_bytes": 1024,
        "elapsed_seconds": 0.5,
    }


def test_fake_guard_requires_reviewed_child_identity_and_resource_completion(
    archived_seed_vectors: None,
) -> None:
    command = ["python", "probe.py", "--child"]
    source = {probe.SELF_PATH: "d" * 64}
    inputs = probe.prior._preflight_identity()
    raw = probe.seed_probe._seed_record_hashes()
    resource = _fake_resource(command)
    child = _fake_child(source, inputs)
    assert probe._execution_status(
        resource, child, command, source, inputs, raw, "a" * 64,
    ) == "completed"

    nonconverged_residual = {**child, "linear_solves": [{
        "iteration": 1, "rtol": probe.TRUE_RESIDUAL_TOLERANCE,
        "max_iterations": 80, "converged": False,
        "true_relative_residual": 2e-10,
        "error": child["refusal"],
    }]}
    assert probe._execution_status(
        resource, nonconverged_residual, command, source, inputs, raw, "a" * 64,
    ) == "completed"

    refiner_residual = {**child, "linear_solves": [{
        **child["linear_solves"][0], "error": None,
    }]}
    assert probe._execution_status(
        resource, refiner_residual, command, source, inputs, raw, "a" * 64,
    ) == "completed"
    impossible_refiner_refusal = {**refiner_residual, "linear_solves": [{
        **refiner_residual["linear_solves"][0], "converged": False,
    }]}
    assert probe._execution_status(
        resource, impossible_refiner_refusal, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    wrong_pid = {**resource, "child_pid": 813}
    assert probe._execution_status(
        wrong_pid, child, command, source, inputs, raw, "a" * 64,
    ) == "failed"
    limited = {**resource, "resource_termination": "wall_time_limit"}
    assert probe._execution_status(
        limited, None, command, source, inputs, raw, "a" * 64,
    ) == "resource_limited"

    changed_source = {**child, "source_after": {probe.SELF_PATH: "e" * 64}}
    assert probe._execution_status(
        resource, changed_source, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    false_candidate = {**child, "reduced_stationarity": True}
    assert probe._execution_status(
        resource, false_candidate, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    missing_metrics = {key: value for key, value in child.items()
                       if key != "final_reduced_gradient_max"}
    assert probe._execution_status(
        resource, missing_metrics, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    wrong_parameters = {**child, "parameters_sha256": "b" * 64,
                        "parameters_sha256_after": "b" * 64}
    assert probe._execution_status(
        resource, wrong_parameters, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    bad_residual = {**resource, "exit_code": False}
    assert probe._execution_status(
        bad_residual, child, command, source, inputs, raw, "a" * 64,
    ) == "failed"

    pcg_error = {**child, "linear_solves": [{
        "iteration": 1, "rtol": probe.TRUE_RESIDUAL_TOLERANCE,
        "max_iterations": 80,
        "error": "RuntimeError: operator must be symmetric positive definite",
    }], "refusal": (
        "RefinementNumericalRefusal: operator must be symmetric positive definite"
    )}
    assert probe._execution_status(
        resource, pcg_error, command, source, inputs, raw, "a" * 64,
    ) == "completed"


@pytest.mark.parametrize("field,value", [
    ("rss_samples", 0),
    ("sampled_peak_rss_bytes", 1024**3 + 1),
    ("elapsed_seconds", 301.0),
])
def test_fake_guard_rejects_invalid_resource_evidence(
    field: str, value: object, archived_seed_vectors: None,
) -> None:
    command = ["python", "probe.py", "--child"]
    source = {probe.SELF_PATH: "d" * 64}
    inputs = probe.prior._preflight_identity()
    raw = probe.seed_probe._seed_record_hashes()
    child = _fake_child(source, inputs)
    resource = {**_fake_resource(command), field: value}
    assert probe._execution_status(
        resource, child, command, source, inputs, raw, "a" * 64,
    ) == "failed"


def test_parent_final_audit_recomputes_metrics_at_reported_control(
    archived_seed_vectors: None,
) -> None:
    source = {probe.SELF_PATH: "d" * 64}
    inputs = probe.prior._preflight_identity()
    child = _fake_child(source, inputs)
    assert probe._audit_final_child(child, inputs)["passed"] is True

    forged = {**child, "final_reduced_gradient_max": 0.0}
    audit = probe._audit_final_child(forged, inputs)
    assert audit["passed"] is False
    assert audit["checks"]["reduced_gradient_matches_full_field_block"] is False
