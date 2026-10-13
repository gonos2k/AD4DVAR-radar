from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/active_face_response.py"
SPEC = importlib.util.spec_from_file_location("native_active_face_response", SOURCE)
assert SPEC is not None and SPEC.loader is not None
ACTIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTIVE)
TWO_FACE_SOURCE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/active_two_face_response.py"
TWO_FACE_SPEC = importlib.util.spec_from_file_location("native_active_two_face_response_test", TWO_FACE_SOURCE)
assert TWO_FACE_SPEC is not None and TWO_FACE_SPEC.loader is not None
TWO_FACE = importlib.util.module_from_spec(TWO_FACE_SPEC)
TWO_FACE_SPEC.loader.exec_module(TWO_FACE)


def test_unqualified_active_base_writes_gate_without_callback_or_solver(tmp_path: Path) -> None:
    parameters = torch.zeros(13, dtype=torch.float64)
    called = False

    def qualify(*_args):
        nonlocal called
        called = True
        raise AssertionError("qualification must not run while base gate is closed")

    result = ACTIVE.run_active_face_response(
        object(), parameters,
        {"status": "singleface_first_order_not_qualified", "minimum_claim": False},
        phase_plan_path=tmp_path / "phase-plan.json", phase_plan_sha256="",
        qualify_face=qualify, output=tmp_path / "gate.json")

    assert result["status"] == "gate_pending_base"
    assert result["native_J_evaluations_started"] == 0
    assert result["forecast_score_evaluations_started"] == 0
    assert called is False


def test_active_adjoint_uses_explicit_parameter_argument_in_phi() -> None:
    class Chart:
        layout = {"face_flux_coordinate": 25}

        @staticmethod
        def from_face_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates

    class QuadraticProblem:
        @staticmethod
        def objective(control: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
            z = control[:25]
            return (z.square().sum() + parameters[0] * z[0]
                + parameters[12] * z[1])

        @staticmethod
        def score(control: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
            z = control[:25]
            return 3.0 * z[0] + 4.0 * z[1] + 2.0 * parameters[0] - 5.0 * parameters[12]

    tangent = torch.zeros(25, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    hessian = 2.0 * torch.eye(25, dtype=torch.float64)
    no_charge = lambda: None

    first = ACTIVE._exact_base_response(QuadraticProblem(), Chart(), tangent,
        parameters, torch.tensor(0.4, dtype=torch.float64), hessian, 0,
        no_charge, no_charge)
    last = ACTIVE._exact_base_response(QuadraticProblem(), Chart(), tangent,
        parameters, torch.tensor(0.4, dtype=torch.float64), hessian, 12,
        no_charge, no_charge)

    assert first["predicted_dE_dp"] == 0.5
    assert last["predicted_dE_dp"] == -7.0
    assert first["linear_system_residual_norm"] < 1e-14
    assert last["linear_system_residual_norm"] < 1e-14


def test_active_point_callback_rejects_undeclared_parameter_changes_before_model_work() -> None:
    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    budget = {"declared_parameter_index": 0, "declared_parameter_delta": 1e-3,
        "base_parameters_sha256": ACTIVE._tensor_sha(parameters), "face_pivot_index": 4,
        "phase_deadline_monotonic": 1e30, "native_J_successes_completed": 0,
        "max_phase_native_J_successes": 10}
    chart = SimpleNamespace(pivot_index=4)

    try:
        ACTIVE.qualify_active_point(object(), control, parameters, budget,
            chart=chart, base_parameters=parameters,
            base_branch_signature="base-branch")
    except ValueError as error:
        assert "declared transition" in str(error)
    else:
        raise AssertionError("active callback accepted a parameter vector without its declared delta")

    assert budget.get("native_J_successes_started", 0) == 0


def test_existing_face_chart_has_25d_jacobian_tangent_to_q_zero() -> None:
    case_path = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
    case = torch.load(case_path, map_location="cpu", weights_only=False)
    spec = case["problem"].frozen.fv_transport
    assert spec is not None
    from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart

    chart = FVFaceFluxCoordinateChart.from_basis(field_count=20, growth_count=1,
        coefficient_limits=spec.coefficient_limits.clone(), psi_basis=spec.psi_basis,
        axis="y", face=(4, 3), pivot_index=4)
    tangent = torch.zeros(25, dtype=torch.float64)
    control = ACTIVE._full_control(chart, tangent)
    jacobian = torch.func.jacrev(lambda z: ACTIVE._full_control(chart, z))(tangent)
    face_gradient = torch.func.grad(lambda c: ACTIVE.face_flux(c, spec))(control)

    assert jacobian.shape == (26, 25)
    assert torch.isfinite(jacobian).all()
    assert torch.linalg.vector_norm(face_gradient @ jacobian) < 1e-13


def test_two_face_intersection_chart_has_24d_jacobian_in_both_face_tangents() -> None:
    diagnostic_path = ROOT / (
        "graphify-out/fv-root-cause-20260919/native_reference_20261012/"
        "two_face_intersection_20261013/two_face_diagnostic.json")
    diagnostic = json.loads(diagnostic_path.read_text())
    control = torch.as_tensor(diagnostic["projected_control"], dtype=torch.float64)
    case = torch.load(ROOT / (
        "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"),
        map_location="cpu", weights_only=False)
    lift, coordinates, _metadata = TWO_FACE._two_face_chart(control, case["problem"])
    tangent_jacobian = torch.func.jacrev(lift)(coordinates)
    face_jacobian = torch.func.jacrev(
        lambda candidate: TWO_FACE._two_face_flux_vector(candidate, case["problem"]))(control)

    assert tangent_jacobian.shape == (26, 24)
    assert face_jacobian.shape == (2, 26)
    assert torch.linalg.matrix_rank(face_jacobian) == 2
    assert torch.linalg.matrix_norm(face_jacobian @ tangent_jacobian) < 1e-12


def test_two_face_normal_cone_uses_gram_coordinates_for_oblique_normals() -> None:
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]], dtype=torch.float64)
    gradients = []
    expected = []
    for side_one, side_two in TWO_FACE.SECTORS:
        eta = torch.tensor([2.0 * side_one, 3.0 * side_two], dtype=torch.float64)
        gradients.append(normals.T @ eta + torch.tensor([0.0, 0.0, 0.25], dtype=torch.float64))
        expected.append(eta)

    certificates = TWO_FACE._sector_normal_coefficients(normals, gradients)

    for row, eta in zip(certificates, expected):
        assert torch.allclose(torch.tensor(row["eta"], dtype=torch.float64), eta, atol=1e-14, rtol=0.0)
        assert row["strict_sign_match"] == [True, True]


def test_two_face_p2_base_accepts_zero_reference_weights_but_binds_hessians_to_point() -> None:
    parameters = torch.zeros(13, dtype=torch.float64)
    control = torch.linspace(-0.2, 0.3, 26, dtype=torch.float64)
    control_sha = TWO_FACE._tensor_sha(control)
    sides = [list(value) for value in TWO_FACE.SECTORS]
    branch = {"classification": "multiplefaces", "stage_count": 360,
        "unique_nearzero_face_entities": ["y:2:1", "y:4:3"],
        "unique_near_tie_slope_entities": [],
        "actual_selector_signature_sha256": "branch-sha"}
    point = {"control": control.tolist(), "control_sha256": control_sha, "branch": branch,
        "branch_supported_exactly_two_faces": True, "first_order_pass": True,
        "full_clarke_hull": {"vector": [0.0] * 26, "weights": [1.0, 0.0, 0.0, 0.0]},
        "tangent_hull": {"vector": [0.0] * 24},
        "normal_coefficients": [{"sides": pair, "strict_sign_match": [True, True]}
            for pair in sides]}
    record = {"status": "qualified_original_native_J_two_face_minimum", "minimum_claim": True,
        "input_parameters_sha256": TWO_FACE._tensor_sha(parameters), "plan_sha256": "plan-sha",
        "source_transition": {"path": TWO_FACE.TRANSPORT_PATH,
            "before": TWO_FACE.SUPPORT_BEFORE_SHA, "after": TWO_FACE.SUPPORT_AFTER_SHA},
        "final_point": point,
        "final_hessians": {f"{pair[0]:+d},{pair[1]:+d}": {
            "hessian": torch.eye(24, dtype=torch.float64).tolist(),
            "hessian_sha256": TWO_FACE._tensor_sha(torch.eye(24, dtype=torch.float64)),
            "control_sha256": control_sha, "positive_definite": True}
            for pair in TWO_FACE.SECTORS}}

    actual_control, weights, signature, plan_sha, _ = TWO_FACE._qualified_twoface_base(record, parameters)

    assert torch.equal(actual_control, control)
    assert weights == [1.0, 0.0, 0.0, 0.0]
    assert signature == "branch-sha"
    assert plan_sha == "plan-sha"

    record["final_hessians"]["-1,-1"]["control_sha256"] = "other-point"
    try:
        TWO_FACE._qualified_twoface_base(record, parameters)
    except ValueError as error:
        assert "not from the final control" in str(error)
    else:
        raise AssertionError("P2 Hessian from another point was accepted")


def test_twoface_perturbation_callback_requires_actual_four_sector_contract() -> None:
    control = torch.linspace(-0.1, 0.2, 26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    chart = {"pivots": [2, 4], "free": [0, 1, 3]}
    branch = {"classification": "multiplefaces", "stage_count": 360,
        "unique_nearzero_face_entities": ["y:2:1", "y:4:3"],
        "unique_near_tie_slope_entities": [], "actual_selector_signature_sha256": "same-branch"}
    sectors = [{"sides": list(sides)} for sides in TWO_FACE.SECTORS]
    normals = [{"sides": list(sides), "strict_sign_match": [True, True]}
        for sides in TWO_FACE.SECTORS]
    qualified = {"status": "twoface_first_order_qualified",
        "input_parameters_sha256": TWO_FACE._tensor_sha(parameters),
        "event_control_sha256": TWO_FACE._tensor_sha(control), "branch": branch,
        "branch_supported_exactly_two_faces": True, "first_order_pass": True,
        "same_actual_branch": True, "chart": chart,
        "full_clarke_hull": {"vector": [0.0] * 26, "weights": [1.0, 0.0, 0.0, 0.0]},
        "tangent_hull": {"vector": [0.0] * 24}, "sectors": sectors,
        "normal_coefficients": normals}

    assert TWO_FACE._qualified_twoface_perturbation(
        qualified, control, parameters, chart, "same-branch")
    qualified["normal_coefficients"][2]["strict_sign_match"] = [True, False]
    assert not TWO_FACE._qualified_twoface_perturbation(
        qualified, control, parameters, chart, "same-branch")


def test_twoface_qualification_rejects_undeclared_parameter_before_problem_work() -> None:
    class NoEvaluation:
        def contract(self, *_args):
            raise AssertionError("parameter gate must precede model work")

    control = torch.zeros(26, dtype=torch.float64)
    parameters = torch.zeros(13, dtype=torch.float64)
    budget = {"declared_parameter_index": 12, "declared_parameter_delta": 1e-3,
        "base_parameters_sha256": TWO_FACE._tensor_sha(parameters),
        "phase_deadline_monotonic": 1e30, "native_J_successes_completed": 0,
        "max_phase_native_J_successes": 10, "chart_coordinates": []}
    try:
        TWO_FACE.qualify_two_face_point(NoEvaluation(), control, parameters, budget,
            lift=lambda z: torch.cat((z[:2], z.new_zeros(1), z[2:], z.new_zeros(1))),
            chart_metadata={}, base_parameters=parameters, base_branch_signature="base")
    except ValueError as error:
        assert "undeclared parameter transition" in str(error)
    else:
        raise AssertionError("two-face callback accepted an undeclared parameter vector")

    assert budget.get("native_J_successes_started", 0) == 0


def test_two_face_first_order_diagnostic_stays_gated_before_p2(tmp_path: Path) -> None:
    parameters = torch.zeros(13, dtype=torch.float64)
    called = False

    def callback(*_args):
        nonlocal called
        called = True
        raise AssertionError("two-face diagnostic cannot launch response before P2")

    result = ACTIVE.run_active_face_response(
        object(), parameters,
        {"status": "two_face_first_order_diagnostic_complete", "minimum_claim": False},
        phase_plan_path=tmp_path / "not-frozen.json", phase_plan_sha256="",
        qualify_face=callback, output=tmp_path / "two-face-gate.json")

    assert result["status"] == "gate_pending_base"
    assert result["response_claim"] is False
    assert result["native_J_evaluations_started"] == 0
    assert called is False


def test_twoface_qualification_plan_pins_current_tuple_extension_source(monkeypatch) -> None:
    plan_path = ROOT / (
        "graphify-out/fv-root-cause-20260919/native_reference_20261012/"
        "two_face_intersection_20261013/NATIVE24_PLAN_v6.json")
    digest = plan_path.with_name("NATIVE24_PLAN_v6_SHA256.txt").read_text().strip()
    monkeypatch.setattr(TWO_FACE.sys, "version", "3.10.11")
    monkeypatch.setattr(TWO_FACE.torch, "__version__", "2.13.0")

    plan = TWO_FACE._validate_twoface_qualification_plan(plan_path, digest)

    assert plan["transport_extension_transition"] == {
        "path": "src/advar/transport.py",
        "before": TWO_FACE.SUPPORT_BEFORE_SHA,
        "after": TWO_FACE.SUPPORT_AFTER_SHA,
        "scope": plan["transport_extension_transition"]["scope"],
    }
