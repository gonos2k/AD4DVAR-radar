from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/perturbation_response.py"
SPEC = importlib.util.spec_from_file_location("native_perturbation_response", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RESPONSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESPONSE)


def test_unqualified_base_writes_gate_without_calling_qualifier_or_solver(tmp_path: Path) -> None:
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    result = RESPONSE.run_reoptimized_response(
        object(), control, parameters, {"status": "pending"},
        qualify=lambda *_: (_ for _ in ()).throw(AssertionError("must remain gated")),
        budget={}, output=tmp_path / "response.json")

    assert result == {
        "status": "gate_pending_base",
        "reason": "supplied base has not passed the regular local-minimum qualification",
        "physical_calls": 0,
        "objective_evaluations": 0,
    }
    assert (tmp_path / "response.json").is_file()


def test_qualification_requires_hashes_gradient_pd_and_regular_branch() -> None:
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    record = {
        "status": RESPONSE.QUALIFIED_STATUS,
        "minimum_claim": True,
        "control_sha256": RESPONSE._tensor_sha(control),
        "parameters_sha256": RESPONSE._tensor_sha(parameters),
        "native_gradient_norm": 9.0e-11,
        "positive_definite": True,
        "regular_branch": True,
        "branch_signature": "smooth-single-branch",
    }

    assert RESPONSE._qualified(record, control, parameters)
    assert not RESPONSE._qualified({**record, "native_gradient_norm": 1.0e-10}, control, parameters)
    assert not RESPONSE._qualified({**record, "positive_definite": False}, control, parameters)
    assert not RESPONSE._qualified({**record, "parameters_sha256": "wrong"}, control, parameters)


def test_qualified_callback_control_is_the_point_that_must_be_scored() -> None:
    control = torch.zeros(26, dtype=torch.float64)
    corrected = control.clone()
    corrected[0] = 1.0e-12
    parameters = torch.zeros(13, dtype=torch.float64)
    record = {
        "status": RESPONSE.QUALIFIED_STATUS,
        "minimum_claim": True,
        "candidate_control_sha256": RESPONSE._tensor_sha(control),
        "qualified_control": corrected,
        "control_sha256": RESPONSE._tensor_sha(corrected),
        "parameters_sha256": RESPONSE._tensor_sha(parameters),
        "native_gradient_norm": 1.0e-11,
        "positive_definite": True,
        "regular_branch": True,
        "branch_signature": "branch-sha",
    }

    admitted = RESPONSE._qualified_control(record, control)
    assert torch.equal(admitted, corrected)
    assert RESPONSE._qualified(record, admitted, parameters, candidate_control=control)
    assert not RESPONSE._qualified(record, control, parameters, candidate_control=control)


def test_raw_red_qualification_schema_is_admitted_only_for_smooth_minimum() -> None:
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    record = {
        "status": "smooth_local_minimum_qualified",
        "minimum_claim": True,
        "candidate_control_sha256": RESPONSE._tensor_sha(control),
        "control": control.tolist(),
        "input_after": {"parameters_sha256": RESPONSE._tensor_sha(parameters)},
        "gradient_inf": 9.0e-11,
        "full_hessian_positive_definite": True,
        "branch_stable": True,
        "final_branch_before_hessian": {
            "classification": "smooth_regular",
            "actual_selector_signature_sha256": "branch-sha",
        },
    }

    assert RESPONSE._qualified(record, control, parameters)
    assert RESPONSE._branch_signature(record) == "branch-sha"
    assert not RESPONSE._qualified({**record, "branch_stable": False}, control, parameters)


def test_smooth_runner_does_not_accept_an_active_face_receipt(tmp_path: Path) -> None:
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    result = RESPONSE.run_reoptimized_response(
        object(), control, parameters,
        {"status": "singleface_local_minimum_qualified", "minimum_claim": True},
        qualify=lambda *_: (_ for _ in ()).throw(AssertionError("active phase must remain gated")),
        budget={}, output=tmp_path / "active-gate.json")

    assert result["status"] == "gate_pending_base"
    assert result["physical_calls"] == 0
    assert result["objective_evaluations"] == 0


def test_singleface_gate_requires_event_chart_strict_normal_and_two_sided_p2() -> None:
    from types import SimpleNamespace

    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    chart = SimpleNamespace(identity={"chart_sha256": "chart-digest"}, pivot_index=4)
    record = {
        "status": "singleface_local_minimum_qualified",
        "minimum_claim": True,
        "P2_passed": True,
        "tangent_stationary": True,
        "one_sided_normal_pass": True,
        "input_parameters_sha256": RESPONSE._tensor_sha(parameters),
        "event_control_sha256": RESPONSE._tensor_sha(control),
        "minimum_mixture_theta": 0.5,
        "event_branch": {"actual_selector_signature_sha256": "branch-digest",
            "classification": "singleface", "stage_count": 360,
            "unique_nearzero_face_entities": ["y:4:3"],
            "unique_near_tie_slope_entities": []},
        "event_face": {"axis": "y", "row": 4, "column": 3,
            "q_value": 0.0, "chart_residual_ratio": 1.0e-15,
            "chart": {"chart_sha256": "chart-digest", "pivot_flow_index": 4}},
        "one_sided_normal_slopes": {"-1": -2.0, "1": 3.0},
        "one_sided_normal_roundoff_budgets": {"-1": 0.1, "1": 0.1},
        "tangent_gradient_inf_by_side": {"-1": 1e-11, "1": 2e-11},
        "full_side_gradient_inf_by_side": {"-1": 2.0, "1": 3.0},
        "reduced_curvature_by_side": {
            "-1": {"positive_definite": True}, "1": {"positive_definite": True}},
    }

    assert RESPONSE._qualified_singleface_base(record, control, parameters, chart)
    assert not RESPONSE._qualified_singleface_base(
        {**record, "one_sided_normal_pass": False}, control, parameters, chart)
    assert not RESPONSE._qualified_singleface_base(
        {**record, "minimum_mixture_theta": None}, control, parameters, chart)


def test_closed_shared_objective_cap_stops_before_qualification_callback(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace
    import time

    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    qualification = {
        "status": RESPONSE.QUALIFIED_STATUS,
        "minimum_claim": True,
        "control_sha256": RESPONSE._tensor_sha(control),
        "parameters_sha256": RESPONSE._tensor_sha(parameters),
        "native_gradient_norm": 0.0,
        "positive_definite": True,
        "regular_branch": True,
        "branch_signature": "smooth",
    }
    monkeypatch.setattr(RESPONSE, "_validate_parent_plan", lambda _problem: ({}, {
        "runtime_identity": {"python": "3.10.11", "torch": "2.13.0", "device": "CPU FP64"}}))
    result = RESPONSE.run_reoptimized_response(
        SimpleNamespace(), control, parameters, qualification,
        qualify=lambda *_: (_ for _ in ()).throw(AssertionError("closed cap must stop first")),
        budget={"global_started_at": time.monotonic(), "total_seconds": 3600.0,
            "objective_evaluations": 2003, "max_objective_evaluations": 2000},
        output=tmp_path / "budget-gate.json")

    assert result["status"] == "budget_exhausted_no_response"
    assert result["objective_evaluations_cumulative"] == 2003


def test_perturbation_steps_use_noise_scale_and_beta_coefficient_units() -> None:
    class Problem:
        observation_std_dbz = torch.full((3, 4), 0.1, dtype=torch.float64)

    problem = Problem()
    assert RESPONSE._parameter_step(problem, torch.zeros(13, dtype=torch.float64), 0)[0] == 0.001
    assert RESPONSE._parameter_step(problem, torch.zeros(13, dtype=torch.float64), 12)[0] == 0.001
