"""Synthetic response-format and coordinate-curvature regressions; no FV execution."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from advar.local_response import compute_local_response
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart
from examples.weather_scenarios.fv_point_active_face_binding import FVPointActiveFaceBinding
from examples.weather_scenarios.fv_response_contract import (
    compose_full_control, local_response_payload, response_envelope,
)


def _hash(char: str) -> str:
    return format(ord(char) % 16, "x") * 64


def _toy_binding(pivot: int, full_objective: Any, full_score: Any):
    chart = FVFaceFluxCoordinateChart(
        field_count=0, growth_count=0, pivot_index=pivot,
        limits=torch.ones(2, dtype=torch.float64),
        weights=torch.tensor([1.0, 2.0], dtype=torch.float64), face=("x", 3, 4),
    )
    binding: Any = object.__new__(FVPointActiveFaceBinding)
    binding.field_count = 0
    binding.pivot_index = pivot
    binding.control_count = 1
    binding.chart = chart

    def lift(tangent: Tensor, _parameters: Tensor) -> Tensor:
        return FVPointActiveFaceBinding.lift(binding, tangent)

    def reduced_objective(tangent: Tensor, parameters: Tensor) -> Tensor:
        control = lift(tangent, parameters)
        return full_objective(control, parameters) - 0.5 * control[pivot].square()

    def reduced_score(tangent: Tensor, parameters: Tensor) -> Tensor:
        return full_score(lift(tangent, parameters), parameters)

    binding.reduced_problem = SimpleNamespace(objective=reduced_objective, score=reduced_score)
    return binding, lift


def test_nonlinear_chart_curvature_and_stationary_response_match_closed_form():
    def full_j(control: Tensor, parameters: Tensor) -> Tensor:
        return 0.5 * control.square().sum() + 0.75 * control[1] - parameters[0] * control[0]

    def score(control: Tensor, _parameters: Tensor) -> Tensor:
        return control[0]

    def gamma(tangent: Tensor, _parameters: Tensor) -> Tensor:
        return torch.stack((tangent[0], tangent[0].square()))

    objective = compose_full_control(full_j, gamma)
    composed_score = compose_full_control(score, gamma)
    t = torch.zeros(1, dtype=torch.float64)
    p = torch.zeros(1, dtype=torch.float64)
    hessian = torch.func.hessian(lambda value: objective(value, p))(t)
    assert float(hessian[0, 0]) == pytest.approx(2.5)
    ambient_hessian = torch.eye(2, dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert float(tangent @ ambient_hessian @ tangent) == pytest.approx(1.0)

    response = compute_local_response(
        objective, composed_score, t, p, {"unit": torch.ones_like(p)},
        branch_check=lambda _c, _p: ({"branch": "toy"}, "fixed synthetic branch"),
        input_identity={"control_sha256": _hash("a")},
    )
    assert float(response.total["unit"]) == pytest.approx(0.4)
    payload = local_response_payload(response)
    assert payload["total"] == {"unit": pytest.approx(0.4)}
    assert payload["true_adjoint_relative_residual"] <= 1e-10
    assert len(payload["adjoint"]) == 1


def test_parameter_dependent_chart_preserves_direct_indirect_and_mixed_terms():
    def full_j(control: Tensor, parameters: Tensor) -> Tensor:
        return 0.5 * control.square().sum() + 0.75 * control[1] - parameters[0] * control[0]

    def score(control: Tensor, _parameters: Tensor) -> Tensor:
        return control[0]

    def gamma(tangent: Tensor, parameters: Tensor) -> Tensor:
        value = tangent[0] + parameters[0]
        return torch.stack((value, value.square()))

    objective = compose_full_control(full_j, gamma)
    composed_score = compose_full_control(score, gamma)
    t = torch.zeros(1, dtype=torch.float64)
    p = torch.zeros(1, dtype=torch.float64)
    response = compute_local_response(
        objective, composed_score, t, p, {"unit": torch.ones_like(p)},
        branch_check=lambda _c, _p: ({"branch": "toy"}, "fixed synthetic branch"),
        input_identity={"control_sha256": _hash("b")},
    )
    assert float(response.mixed_gradients["unit"][0]) == pytest.approx(1.5)
    assert float(response.direct["unit"]) == pytest.approx(1.0)
    assert float(response.indirect["unit"]) == pytest.approx(-0.6)
    assert float(response.total["unit"]) == pytest.approx(0.4)


def test_actual_active_face_objective_restores_pivot_prior_and_chart_curvature():
    parameters = torch.zeros(1, dtype=torch.float64)
    probe_binding, lift = _toy_binding(0, lambda c, _p: 0.5 * c.square().sum(), lambda _c, _p: torch.zeros((), dtype=torch.float64))
    tangent = torch.tensor([0.1], dtype=torch.float64)
    full = lift(tangent, parameters)
    assert abs(float(full[0])) > 1e-4
    actual = FVPointActiveFaceBinding.objective(probe_binding, tangent, parameters)
    assert float(actual) == pytest.approx(float(0.5 * full.square().sum()))

    tangent_direction = torch.ones_like(tangent)
    chart_tangent = torch.func.jvp(lambda z: lift(z, parameters), (tangent,), (tangent_direction,))[1]
    normal_gradient = torch.stack((torch.ones_like(chart_tangent[0]), -chart_tangent[0]))
    assert abs(float(normal_gradient @ chart_tangent)) < 1e-12
    ambient_gradient = normal_gradient - full

    def original_j(control: Tensor, _p: Tensor) -> Tensor:
        return 0.5 * control.square().sum() + torch.dot(ambient_gradient, control)

    def reduced_j(tangent_value: Tensor, candidate_parameters: Tensor) -> Tensor:
        control = lift(tangent_value, candidate_parameters)
        return original_j(control, candidate_parameters) - 0.5 * control[0].square()

    probe_binding.reduced_problem.objective = reduced_j
    restored = lambda value: FVPointActiveFaceBinding.objective(probe_binding, value, parameters)
    uncorrected = lambda value: reduced_j(value, parameters)
    restored_hessian = torch.func.hessian(restored)(tangent)
    uncorrected_hessian = torch.func.hessian(uncorrected)(tangent)
    jacobian_term = chart_tangent @ chart_tangent
    assert float(restored_hessian[0, 0]) == pytest.approx(3.967967381155587)
    assert float(uncorrected_hessian[0, 0]) == pytest.approx(-0.5436120625838408)
    assert float(restored_hessian[0, 0]) == pytest.approx(float(jacobian_term + (normal_gradient @ torch.func.jvp(
        lambda z: torch.func.jvp(lambda q: lift(q, parameters), (z,), (torch.ones_like(z),))[1],
        (tangent,), (torch.ones_like(tangent),))[1])))
    assert abs(float(restored_hessian[0, 0] - uncorrected_hessian[0, 0])) > 1e-5
    assert abs(float(torch.func.grad(restored)(tangent)[0])) < 1e-10
    assert float(ambient_gradient.norm()) > 1e-4


def test_alternate_pivots_on_same_face_restore_prior_and_agree_in_response():
    def full_j(control: Tensor, parameters: Tensor) -> Tensor:
        return 0.5 * control.square().sum() + parameters[0] * control[0]

    def full_score(control: Tensor, parameters: Tensor) -> Tensor:
        return control[0] + 0.3 * parameters[0]

    results = []
    full_points = []
    for pivot in (0, 1):
        binding, lift = _toy_binding(pivot, full_j, full_score)
        tangent = torch.zeros(1, dtype=torch.float64)
        parameters = torch.zeros(1, dtype=torch.float64)
        full = lift(tangent, parameters)
        full_points.append(full)
        assert float(binding.chart.to_face_coordinates(full)[pivot]) == 0.0
        bound_objective = lambda t, p: FVPointActiveFaceBinding.objective(binding, t, p)
        bound_score = lambda t, p: FVPointActiveFaceBinding.score(binding, t, p)
        assert float(bound_objective(tangent, parameters)) == pytest.approx(float(full_j(full, parameters)))
        response = compute_local_response(
            bound_objective, bound_score, tangent, parameters, {"unit": torch.ones_like(parameters)},
            branch_check=lambda _c, _p: ({"face": "x[3,4]=0"}, "same selected face"),
            input_identity={"control_sha256": _hash("c")},
        )
        results.append(float(response.total["unit"]))
    assert torch.equal(full_points[0], full_points[1])
    assert results[0] == pytest.approx(results[1], abs=1e-12)
    assert results[0] == pytest.approx(-0.5)


def _envelope_kwargs(kind: str = "selected_face_conditional") -> dict[str, Any]:
    identities = {name: {"sha256": _hash(char)} for name, char in zip(
        ("problem", "control", "parameters", "code", "branch", "source", "runtime", "plan"),
        "abcdefgh", strict=True)}
    point = identities["control"]["sha256"]
    parameters = identities["parameters"]["sha256"]
    return {
        "kind": kind,
        "execution_status": "completed",
        "numerical_status": "not_evaluated",
        "physical_status": "not_validated",
        "identities": identities,
        "objective_identity": {"sha256": _hash("i"), "name": "full original J"},
        "prior_identity": {"sha256": _hash("j"), "contract": "original zero-centered full-control prior",
                           "restored_full_objective": True, "control_dimension": 26},
        "coordinate_map": {"ambient_dimension": 26, "tangent_dimension": 25, "lift_sha256": _hash("k")},
        "stationarity": {"space": "tangent", "tolerance": 1e-10, "value": 2e-11, "status": "eligible"},
        "ambient_diagnostics": {"status": "not_evaluated"},
        "curvature": {"space": "tangent", "status": "qualified"},
        "validations": [{"control_sha256": point, "direction_sha256": _hash("l"),
                          "scope": "same face at tested endpoints", "status": "not_run"}],
        "response": {"direct": 0.0, "indirect": 0.4, "total": 0.4},
        "resource": {"budget_seconds": 600, "elapsed_seconds": None, "peak_rss_bytes": None},
        "claims": {"full_root_supported": False, "physical_validation_supported": False},
        "selected_face": {"axis": "x", "row": 3, "column": 4},
        "chart": {"sha256": _hash("k"), "pivot": 0},
        "normal_evidence": {"status": "measured", "control_sha256": point, "parameters_sha256": parameters,
                            "scope": "finite-point two-sided interval normal",
                            "references": [{"sha256": _hash("o"), "control_sha256": point,
                                            "parameters_sha256": parameters,
                                            "scope": "two-sided interval normal at base point"}]},
    }


def test_common_envelope_keeps_response_kinds_and_assessments_separate():
    conditional = response_envelope(**_envelope_kwargs())
    assert conditional["kind"] == "selected_face_conditional"
    assert conditional["assessment"] == {"execution": "completed", "numerical": "not_evaluated",
                                          "physical": "not_validated"}
    assert conditional["resource"] == _envelope_kwargs()["resource"]
    smooth_kwargs = _envelope_kwargs("smooth_stationary")
    smooth_kwargs.update(selected_face=None, chart=None, normal_evidence=None)
    smooth_kwargs["coordinate_map"]["tangent_dimension"] = 26
    smooth_kwargs["stationarity"]["space"] = "full_control"
    smooth_kwargs["curvature"]["space"] = "full_control"
    smooth = response_envelope(**smooth_kwargs)
    assert smooth["geometry"] == {"selected_face": None, "chart": None, "normal_evidence": None}
    assert smooth["claims"]["full_root_supported"] is False


def test_conditional_response_can_record_unrun_normal_without_claiming_eligibility():
    kwargs = _envelope_kwargs()
    kwargs["normal_evidence"] = {"status": "not_run", "control_sha256": kwargs["identities"]["control"]["sha256"],
                                 "parameters_sha256": kwargs["identities"]["parameters"]["sha256"],
                                 "scope": "one-sided finite-point check pending", "references": []}
    kwargs["numerical_status"] = "not_evaluated"
    envelope = response_envelope(**kwargs)
    assert envelope["geometry"]["normal_evidence"] == kwargs["normal_evidence"]
    assert envelope["claims"]["full_root_supported"] is False


@pytest.mark.parametrize("status", ["not_run", "refused"])
def test_unmeasured_normal_evidence_rejects_inline_interval_values(status: str):
    kwargs = _envelope_kwargs()
    kwargs["normal_evidence"] = {"status": status, "control_sha256": kwargs["identities"]["control"]["sha256"],
                                 "scope": "one-sided interval normal pending", "references": [], "slopes": [1.0]}
    with pytest.raises(ValueError, match="cannot carry inline measured"):
        response_envelope(**kwargs)


def test_not_run_validation_needs_no_fabricated_steps_or_errors():
    kwargs = _envelope_kwargs()
    point = kwargs["identities"]["control"]["sha256"]
    kwargs["validations"] = [{"control_sha256": point, "direction_sha256": _hash("l"),
                              "scope": "planned finite-difference check", "status": "not_run"}]
    envelope = response_envelope(**kwargs)
    validation = envelope["independent_validation"][0]
    assert validation["step_sizes"] == [] and validation["endpoint_control_sha256"] == []
    assert validation["criterion"] == {} and validation["error"] == {}


def test_not_run_validation_rejects_fabricated_observed_evidence():
    kwargs = _envelope_kwargs()
    row = kwargs["validations"][0]
    row.update(step_sizes=[0.01], endpoint_control_sha256=[_hash("m")], criterion={"tol": 1e-4}, error={"value": 0.0})
    with pytest.raises(ValueError, match="cannot carry observed"):
        response_envelope(**kwargs)


def test_not_run_validation_rejects_parameter_endpoint_evidence():
    kwargs = _envelope_kwargs()
    kwargs["validations"][0]["endpoint_parameters_sha256"] = [_hash("q"), _hash("r")]
    with pytest.raises(ValueError, match="cannot carry observed"):
        response_envelope(**kwargs)


def test_selected_face_kind_cannot_claim_full_root_support():
    kwargs = _envelope_kwargs()
    kwargs["claims"]["full_root_supported"] = True
    with pytest.raises(ValueError, match="cannot claim full-root"):
        response_envelope(**kwargs)


def test_smooth_kind_rejects_reduced_tangent_record():
    kwargs = _envelope_kwargs("smooth_stationary")
    kwargs.update(selected_face=None, chart=None, normal_evidence=None)
    with pytest.raises(ValueError, match="full-control stationarity"):
        response_envelope(**kwargs)


def test_smooth_full_root_claim_requires_explicit_numerical_eligibility_but_is_not_inferred():
    kwargs = _envelope_kwargs("smooth_stationary")
    kwargs.update(selected_face=None, chart=None, normal_evidence=None)
    kwargs["coordinate_map"]["tangent_dimension"] = 26
    kwargs["stationarity"].update(space="full_control", status="passed")
    kwargs["curvature"].update(space="full_control", status="qualified")
    # Even a recorded eligible smooth result does not gain a root claim by default.
    kwargs["numerical_status"] = "eligible"
    envelope = response_envelope(**kwargs)
    assert envelope["claims"]["full_root_supported"] is False
    kwargs["claims"]["full_root_supported"] = True
    kwargs["numerical_status"] = "ineligible"
    with pytest.raises(ValueError, match="requires eligible full-control"):
        response_envelope(**kwargs)


def test_full_root_claim_requires_stationarity_value_below_reported_gate():
    kwargs = _envelope_kwargs("smooth_stationary")
    kwargs.update(selected_face=None, chart=None, normal_evidence=None)
    kwargs["coordinate_map"]["tangent_dimension"] = 26
    kwargs["stationarity"].update(space="full_control", status="passed", value=1.0, tolerance=1e-10)
    kwargs["curvature"].update(space="full_control", status="qualified")
    kwargs["numerical_status"] = "eligible"
    kwargs["claims"]["full_root_supported"] = True
    with pytest.raises(ValueError, match="below its explicit tolerance"):
        response_envelope(**kwargs)


def test_passed_central_validation_pins_both_endpoints_for_each_h():
    kwargs = _envelope_kwargs()
    point = kwargs["identities"]["control"]["sha256"]
    row = {"control_sha256": point, "parameters_sha256": kwargs["identities"]["parameters"]["sha256"],
           "direction_sha256": _hash("l"), "scope": "same selected face",
           "status": "passed", "method": "central_finite_difference", "step_sizes": [0.01, 0.005],
           "endpoint_control_sha256": [_hash(c) for c in "mnop"],
           "endpoint_parameters_sha256": [_hash(c) for c in "qrst"],
           "criterion": {"relative_error_tolerance": 1e-4}, "error": {"relative": 2e-5}}
    kwargs["validations"] = [row]
    assert response_envelope(**kwargs)["independent_validation"][0]["status"] == "passed"
    row.pop("endpoint_parameters_sha256")
    with pytest.raises(ValueError, match="two control and parameter endpoints"):
        response_envelope(**kwargs)


def _central_row(kwargs: dict[str, Any], *, relative: Any = 2e-5) -> dict[str, Any]:
    return {"control_sha256": kwargs["identities"]["control"]["sha256"],
            "parameters_sha256": kwargs["identities"]["parameters"]["sha256"],
            "direction_sha256": _hash("l"), "scope": "same point central score difference",
            "status": "passed", "method": "central_finite_difference", "step_sizes": [0.01],
            "endpoint_control_sha256": [_hash("m"), _hash("m")],
            "endpoint_parameters_sha256": [_hash("n"), _hash("o")],
            "criterion": {"relative_error_tolerance": 1e-4}, "error": {"relative": relative}}


@pytest.mark.parametrize("error", [1.0, float("inf"), None])
def test_passed_known_central_metric_must_meet_explicit_tolerance(error: Any):
    kwargs = _envelope_kwargs()
    kwargs["validations"] = [_central_row(kwargs, relative=error)]
    with pytest.raises(ValueError, match="relative error violates"):
        response_envelope(**kwargs)


def test_central_metric_checks_are_narrow_and_allow_failed_or_generic_criteria():
    kwargs = _envelope_kwargs()
    row = _central_row(kwargs, relative=1.0)
    row["criterion"] = {"absolute_error_limit": 1e-8}
    kwargs["validations"] = [row]
    assert response_envelope(**kwargs)["independent_validation"][0]["status"] == "passed"
    row["status"] = "failed"
    row["criterion"] = {"relative_error_tolerance": 1e-4}
    assert response_envelope(**kwargs)["independent_validation"][0]["status"] == "failed"


def test_passed_central_validation_checks_both_relative_error_aliases():
    kwargs = _envelope_kwargs()
    row = _central_row(kwargs, relative=2e-5)
    row["error"]["relative_error"] = 1.0
    kwargs["validations"] = [row]
    with pytest.raises(ValueError, match="relative error violates"):
        response_envelope(**kwargs)


def test_relative_error_aliases_must_agree_when_both_are_recorded():
    kwargs = _envelope_kwargs()
    row = _central_row(kwargs, relative=2e-5)
    row["error"]["relative_error"] = 2e-5
    kwargs["validations"] = [row]
    assert response_envelope(**kwargs)["independent_validation"][0]["status"] == "passed"
    row["error"]["relative_error"] = 3e-5
    with pytest.raises(ValueError, match="aliases disagree"):
        response_envelope(**kwargs)


def test_legitimate_duplicate_control_endpoints_with_direct_response_are_allowed():
    kwargs = _envelope_kwargs()
    kwargs["validations"] = [_central_row(kwargs, relative=2e-5)]
    kwargs["response"] = {"direct": 0.3, "indirect": -0.1, "total": 0.2}
    validation = response_envelope(**kwargs)["independent_validation"][0]
    assert validation["endpoint_control_sha256"][0] == validation["endpoint_control_sha256"][1]
    assert validation["endpoint_parameters_sha256"][0] != validation["endpoint_parameters_sha256"][1]


@pytest.mark.parametrize(("execution", "numerical"), [
    ("not_recorded", "not_evaluated"),
    ("failed", "validation_failed"),
    ("resource_limited", "ineligible"),
])
def test_execution_and_numerical_statuses_remain_independent(execution: str, numerical: str):
    kwargs = _envelope_kwargs()
    kwargs["execution_status"] = execution
    kwargs["numerical_status"] = numerical
    envelope = response_envelope(**kwargs)
    assert envelope["assessment"]["execution"] == execution
    assert envelope["assessment"]["numerical"] == numerical


def test_common_envelope_rejects_normal_evidence_from_a_different_point():
    kwargs = _envelope_kwargs()
    kwargs["normal_evidence"]["references"][0]["control_sha256"] = _hash("p")
    with pytest.raises(ValueError, match="response control"):
        response_envelope(**kwargs)


@pytest.mark.parametrize("location", ["top", "reference"])
@pytest.mark.parametrize("mode", ["missing", "mismatched"])
def test_measured_normal_must_bind_the_same_base_parameters(location: str, mode: str):
    kwargs = _envelope_kwargs()
    record = kwargs["normal_evidence"] if location == "top" else kwargs["normal_evidence"]["references"][0]
    if mode == "missing":
        record.pop("parameters_sha256")
    else:
        record["parameters_sha256"] = _hash("f")
    with pytest.raises(ValueError, match="parameters|SHA-256"):
        response_envelope(**kwargs)


@pytest.mark.parametrize("status", ["passed", "failed", "inconclusive"])
@pytest.mark.parametrize("mode", ["missing", "mismatched"])
def test_performed_validation_must_bind_the_same_base_parameters(status: str, mode: str):
    kwargs = _envelope_kwargs()
    row = {"control_sha256": kwargs["identities"]["control"]["sha256"],
           "direction_sha256": _hash("l"), "scope": "performed response validation", "status": status,
           "method": "directional_check", "step_sizes": [0.01], "endpoint_control_sha256": [_hash("m")],
           "criterion": {"tolerance": 1e-4}, "error": {"absolute": 0.0}}
    if mode == "mismatched":
        row["parameters_sha256"] = _hash("f")
    kwargs["validations"] = [row]
    with pytest.raises(ValueError, match="base parameters"):
        response_envelope(**kwargs)


@pytest.mark.parametrize("record", ["normal", "validation"])
def test_unrun_optional_parameter_identity_must_match_base(record: str):
    kwargs = _envelope_kwargs()
    if record == "normal":
        kwargs["normal_evidence"] = {"status": "not_run", "control_sha256": kwargs["identities"]["control"]["sha256"],
                                     "parameters_sha256": _hash("f"), "scope": "pending", "references": []}
    else:
        kwargs["validations"][0]["parameters_sha256"] = _hash("f")
    with pytest.raises(ValueError, match="parameter identity differs"):
        response_envelope(**kwargs)
