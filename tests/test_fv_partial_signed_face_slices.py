"""Synthetic-only tests for signed-face slice wiring and false-success gates."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from advar.local_refinement import RefinementNumericalRefusal
from examples.weather_scenarios.fv_face_flux_coordinates import FVFaceFluxCoordinateChart
from examples.weather_scenarios import fv_partial_signed_face_slices as slices


def _chart():
    return FVFaceFluxCoordinateChart(
        field_count=20,
        growth_count=1,
        pivot_index=1,
        limits=torch.tensor([0.11, 0.08, 0.07, 0.04, 0.03], dtype=torch.float64),
        weights=torch.tensor([0.0, -1.0, -2.0, -0.5, -2.0], dtype=torch.float64),
        face=("y", 2, 0),
    )


def test_slice_map_preserves_all_nonpivot_slots_and_objective():
    chart = _chart()
    original = torch.linspace(-0.02, 0.03, chart.control_count, dtype=torch.float64)
    coordinates = chart.to_face_coordinates(original)
    tangent = torch.cat((coordinates[:slices.PIVOT_CONTROL_INDEX],
                         coordinates[slices.PIVOT_CONTROL_INDEX + 1:]))
    eta = coordinates[slices.PIVOT_CONTROL_INDEX]
    recovered = slices._slice_control(chart, tangent, eta)
    torch.testing.assert_close(recovered, original, rtol=2e-13, atol=2e-13)

    target = torch.linspace(0.01, -0.01, original.numel(), dtype=original.dtype)
    parameters = torch.tensor([0.4, -0.2], dtype=original.dtype)

    def objective(control, p):
        return 0.5 * (control - target).square().sum() + p[0] * control.sin().sum() + p[1]

    transformed = slices._slice_objective(objective, chart, eta)
    torch.testing.assert_close(transformed(tangent, parameters), objective(original, parameters),
                               rtol=2e-13, atol=2e-13)
    gradient = torch.func.grad(transformed)(tangent, parameters)
    assert gradient.shape == tangent.shape
    assert bool(torch.isfinite(gradient).all())


def test_out_of_domain_slice_refuses_without_clipping():
    chart = _chart()
    tangent = torch.zeros(chart.control_count - 1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="strict representable chart domain"):
        slices._slice_control(chart, tangent, torch.tensor(4.0, dtype=torch.float64))


def test_archive_runtime_mismatch_is_a_refusal():
    with pytest.raises(slices.SliceIdentityRefusal, match="runtime differs"):
        slices._validate_runtime({"python": "other", "torch": "other", "device": "CPU FP64"},
                                 {"python": "current", "torch": "current", "device": "CPU FP64"})


def test_probe_pin_rejects_modified_source_even_when_caller_hashes_it():
    slices._verify_probe_file(slices._sha(Path(slices.__file__)))
    template = b'PROBE_SOURCE_PIN = "<PIN>"\nvalue = 1\n'
    canonical_pin = slices._canonical_probe_sha256(template)
    pinned_source = template.replace(b"<PIN>", canonical_pin.encode())
    raw_pin = slices.hashlib.sha256(pinned_source).hexdigest()
    slices._verify_probe_bytes(pinned_source, raw_pin, canonical_pin)

    edited_source = pinned_source.replace(b"value = 1", b"value = 2")
    edited_caller_hash = slices.hashlib.sha256(edited_source).hexdigest()
    with pytest.raises(slices.SliceIdentityRefusal, match="reviewed canonical pins"):
        slices._verify_probe_bytes(edited_source, edited_caller_hash, canonical_pin)


def test_parent_audit_and_within_slice_signature_reject_false_success():
    with pytest.raises(slices.SliceAuditRefusal, match="objective differs"):
        slices._assert_scalar_match("objective", 0.25, 0.5)
    with pytest.raises(slices.SliceBranchRefusal, match="signature differs"):
        slices._require_slice_signature("slice-start-signature", "changed-trial-signature")


def test_parent_completion_cannot_mask_guard_or_audit_failure():
    child = {"execution_status": "child_completed"}
    good_resource = {"exit_code": 0, "resource_termination": None}
    assert slices._wrapper_completion_status(child, good_resource, False) == "parent_audit_refused"
    assert slices._wrapper_completion_status(
        child, {"exit_code": 0, "resource_termination": "wall_time_limit"}, True,
    ) == "guard_refused"
    assert slices._wrapper_completion_status(child, good_resource, True) == "completed"
    assert slices.ETA_FACTORS == (-2.0, -1.0, 1.0, 2.0)


def test_independent_linear_residual_rejects_reported_false_convergence():
    records = []
    rhs = torch.tensor([1.0, -2.0], dtype=torch.float64)

    def false_solver(_operator, right_hand_side, **_kwargs):
        return SimpleNamespace(
            solution=torch.zeros_like(right_hand_side),
            converged=True,
            iterations=1,
            relative_residual=0.0,
        )

    state = {"tangent": torch.zeros(25, dtype=torch.float64), "signature": "synthetic"}
    wrapped = slices._true_residual_monitor(records, eta=1e-4, state=state)(false_solver)
    with pytest.raises(RefinementNumericalRefusal, match="independently recomputed tangent PCG residual"):
        wrapped(lambda direction: direction, rhs, rtol=1e-10, max_iterations=4)
    assert len(records) == 1
    assert records[0]["reported_relative_residual"] == 0.0
    assert records[0]["independently_recomputed_relative_residual"] == 1.0
    assert records[0]["converged"] is True


def test_parent_recomputes_exact_tangent_newton_rhs_and_residual():
    chart = _chart()
    control = torch.linspace(-0.02, 0.03, chart.control_count, dtype=torch.float64)
    coordinates = chart.to_face_coordinates(control)
    eta = coordinates[slices.PIVOT_CONTROL_INDEX]
    tangent = torch.cat((coordinates[:slices.PIVOT_CONTROL_INDEX],
                         coordinates[slices.PIVOT_CONTROL_INDEX + 1:]))
    target = tangent + 0.1
    parameters = torch.tensor([0.2], dtype=torch.float64)

    def objective(full_control, _parameters):
        free = torch.cat((full_control[:slices.PIVOT_CONTROL_INDEX],
                           full_control[slices.PIVOT_CONTROL_INDEX + 1:]))
        return 0.5 * (free - target).square().sum()

    transformed = slices._slice_objective(objective, chart, eta)
    gradient = torch.func.grad(transformed)(tangent, parameters)
    solution = -gradient
    hessian_solution = torch.func.jvp(
        lambda value: torch.func.grad(transformed)(value, parameters),
        (tangent,), (solution,),
    )[1]
    residual = hessian_solution + gradient
    row = {
        "eta": float(eta),
        "input_tangent": tangent.tolist(),
        "input_signature": "slice-signature",
        "rtol": 1e-10,
        "max_iterations": slices.MAX_PCG,
        "converged": True,
        "iterations": 1,
        "rhs": (-gradient).tolist(),
        "solution": solution.tolist(),
        "true_residual": residual.tolist(),
        "independently_recomputed_relative_residual": float(
            torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)
        ),
    }
    slice_record = {
        "eta": float(eta), "status": "tangent_stationary_candidate",
        "start_branch": {"signature_sha256": "slice-signature"},
        "linear_solves": [row],
    }
    assert slices._audit_linear_solves(slice_record, SimpleNamespace(objective=objective),
                                       chart, parameters) == 1

    false_convergence = dict(row, converged=False)
    with pytest.raises(slices.SliceAuditRefusal, match="failed PCG solve"):
        slices._audit_linear_solves({**slice_record, "linear_solves": [false_convergence]},
                                    SimpleNamespace(objective=objective), chart, parameters)

    small_target = torch.zeros_like(tangent)
    small_target[0] = 1e-8

    def small_objective(full_control, _parameters):
        free = torch.cat((full_control[:slices.PIVOT_CONTROL_INDEX],
                           full_control[slices.PIVOT_CONTROL_INDEX + 1:]))
        return 0.5 * (free - small_target).square().sum()

    small_transformed = slices._slice_objective(small_objective, chart, eta)
    small_gradient = torch.func.grad(small_transformed)(torch.zeros_like(tangent), parameters)
    forged_rhs = -small_gradient
    forged_rhs[0] += 1e-14
    forged_solution = forged_rhs.clone()
    saved_zero_residual = torch.zeros_like(forged_rhs)
    forged_row = {
        "eta": float(eta), "input_tangent": torch.zeros_like(tangent).tolist(),
        "input_signature": "slice-signature", "rtol": 1e-10,
        "max_iterations": slices.MAX_PCG, "converged": True, "iterations": 1,
        "rhs": forged_rhs.tolist(), "solution": forged_solution.tolist(),
        "true_residual": saved_zero_residual.tolist(),
        "independently_recomputed_relative_residual": 0.0,
    }
    small_slice_record = {
        "eta": float(eta), "status": "tangent_stationary_candidate",
        "start_branch": {"signature_sha256": "slice-signature"},
        "linear_solves": [forged_row],
    }
    with pytest.raises(slices.SliceAuditRefusal, match="RHS differs"):
        slices._audit_linear_solves(small_slice_record,
                                    SimpleNamespace(objective=small_objective),
                                    chart, parameters)


def test_candidate_audit_requires_one_solve_and_accepted_step_per_iteration():
    candidate = {
        "status": "tangent_stationary_candidate",
        "refiner_iterations": 1,
        "refiner_hvp_count": 2,
        "linear_solves": [],
        "accepted_steps": [{}],
    }
    with pytest.raises(slices.SliceAuditRefusal, match="counts differ"):
        slices._audit_candidate_counts(candidate)


def test_nonfinite_start_refusal_is_recomputed_by_parent():
    chart = _chart()
    control = torch.zeros(chart.control_count, dtype=torch.float64)
    parameters = torch.zeros(1, dtype=torch.float64)
    finite_problem = SimpleNamespace(objective=lambda value, _p: value.square().sum())
    with pytest.raises(slices.SliceAuditRefusal, match="finite objective and gradient"):
        slices._audit_nonfinite_start(control, finite_problem, chart, parameters, 0.0)

    nonfinite_problem = SimpleNamespace(
        objective=lambda value, _p: value.sum() * torch.tensor(float("nan")),
    )
    slices._audit_nonfinite_start(control, nonfinite_problem, chart, parameters, 0.0)
