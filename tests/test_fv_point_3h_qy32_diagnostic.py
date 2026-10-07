from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as diagnostic


def _profile() -> SimpleNamespace:
    y, x = torch.meshgrid(torch.arange(5, dtype=torch.float64),
                          torch.arange(6, dtype=torch.float64), indexing="ij")
    basis = torch.stack((y, x, x * y, (x.square() - y.square()) / 2, x.square() * y))
    spec = SimpleNamespace(psi_basis=basis,
        coefficient_limits=torch.tensor([0.11, 0.08, 0.07, 0.04, 0.03], dtype=torch.float64),
        substeps_per_interval=900, spacing_yx=(10.0, 10.0), reconstruction="minmod",
        max_courant=0.5)
    return SimpleNamespace(frozen=SimpleNamespace(fv_transport=spec,
        nowcast_config=SimpleNamespace(interval_minutes=10)))


def _control() -> torch.Tensor:
    return torch.linspace(-0.08, 0.09, 26, dtype=torch.float64)


def test_phase_profile_geometry_weights_match_declared_face():
    weights = diagnostic.face_weights(_profile())
    assert torch.allclose(weights, torch.tensor(diagnostic.EXPECTED_FACE_WEIGHTS,
                                                dtype=torch.float64), atol=2e-16, rtol=0.0)


@pytest.mark.parametrize("eta", diagnostic.ETA_VALUES)
def test_chart_eta_matches_actual_bounded_fv_face_flux(eta: float):
    problem = _profile()
    point = diagnostic.chart(_control(), eta)
    actual = diagnostic.production_qy32(problem, point)
    formula = diagnostic.qy32(point, diagnostic.face_weights(problem))
    assert float(actual) == pytest.approx(eta, abs=2e-16)
    assert float(actual) == pytest.approx(float(formula), abs=2e-16)


@pytest.mark.parametrize("eta", diagnostic.ETA_VALUES + (0.0,))
def test_chart_reconstructs_eta_and_jacobian_spans_face_tangent(eta: float):
    control = _control()
    point = diagnostic.chart(control, eta)
    assert float(diagnostic.qy32(point)) == pytest.approx(eta, abs=2e-16)

    z, gamma_eta = diagnostic.chart_jacobian(control, eta)
    normal = torch.zeros_like(control)
    normal[21:25] = torch.tensor(diagnostic.EXPECTED_FACE_WEIGHTS, dtype=torch.float64) * (
        1 - torch.tanh(point[21:25]).square())
    assert torch.max(torch.abs(normal @ z)) < 2e-16
    assert float(normal @ gamma_eta) == pytest.approx(1.0, abs=2e-15)

    retained = [i for i in range(26) if i != diagnostic.PIVOT]
    def gamma(values: torch.Tensor) -> torch.Tensor:
        base = control.clone()
        base[retained] = values
        return diagnostic.chart(base, eta)
    jacobian = torch.func.jacrev(gamma)(control[retained])
    assert torch.allclose(jacobian, z, atol=2e-15, rtol=2e-15)


def test_chart_rejects_values_outside_real_pivot_domain():
    control = torch.zeros(26, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside the pivot chart domain"):
        diagnostic.chart(control, eta=0.451)


def test_tangent_projection_matches_normal_projection_and_is_chart_rescaling_invariant():
    control = diagnostic.chart(_control(), eta=1e-6)
    gradient = torch.sin(torch.arange(26, dtype=torch.float64) * 0.37)
    values = diagnostic.geometry(control, 1e-6, gradient)
    weights = torch.tensor(diagnostic.EXPECTED_FACE_WEIGHTS, dtype=torch.float64)
    normal = torch.zeros(26, dtype=torch.float64)
    normal[21:25] = weights * (1 - torch.tanh(control[21:25]).square())
    direct = gradient - normal * (normal @ gradient) / (normal @ normal)
    z, _ = diagnostic.chart_jacobian(control, 1e-6)
    q, _ = torch.linalg.qr(z, mode="reduced")
    rescaled = z @ torch.diag(torch.linspace(0.5, 2.0, 25, dtype=torch.float64))
    q_scaled, _ = torch.linalg.qr(rescaled, mode="reduced")
    assert values["tangent_gradient_norm"] == pytest.approx(float(torch.linalg.vector_norm(direct)), abs=2e-14)
    assert values["tangent_gradient_norm_projection"] == pytest.approx(
        values["tangent_gradient_norm"], abs=2e-14)
    assert values["j_eta_fixed_t"] != pytest.approx(
        values["intrinsic_normal_flux_slope"], abs=1e-6)
    assert torch.allclose(q @ (q.T @ gradient), q_scaled @ (q_scaled.T @ gradient),
                          atol=2e-14, rtol=2e-14)
    assert values["j_t_fixed_eta"] == values["tangent_covector"]


def test_chart_preserves_full_control_objective_including_pivot_term():
    control = _control()
    point = diagnostic.chart(control, eta=1e-6)
    parameters = torch.linspace(-0.3, 0.2, 26, dtype=torch.float64)
    observed: list[torch.Tensor] = []

    class FullObjective:
        frozen: object

        def __init__(self, frozen: object) -> None:
            self.frozen = frozen

        def objective(self, value: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            observed.append(value.detach().clone())
            # Each component represents one retained full-control prior term.
            return torch.sum((value - params).square())

    problem = FullObjective(_profile().frozen)
    gradient_calls = 0

    def must_not_evaluate_gradient(*_args: object) -> torch.Tensor:
        nonlocal gradient_calls
        gradient_calls += 1
        raise AssertionError("eta=0 cost-only path must not evaluate a gradient")

    result = diagnostic._measure(problem, parameters, point, 1e-6,
                                 must_not_evaluate_gradient,
                                 with_gradient=False, deadline=time.monotonic() + 10.0)
    gradient = torch.func.grad(problem.objective, argnums=0)(point, parameters)
    expected = torch.sum((point - parameters).square())
    assert result["objective"] == pytest.approx(float(expected), abs=1e-14)
    assert result["gradient_computed"] is False
    assert gradient_calls == 0
    assert observed and all(torch.equal(value, point) for value in observed)
    assert point[diagnostic.PIVOT] != control[diagnostic.PIVOT]
    assert float(gradient[diagnostic.PIVOT]) == pytest.approx(
        float(2 * (point[diagnostic.PIVOT] - parameters[diagnostic.PIVOT])))


def test_diagnostic_policy_has_one_bounded_nonoptimization_launch():
    policy = diagnostic.diagnostic_policy()
    assert policy["outer_seconds"] == 300.0
    assert policy["internal_seconds"] == 240.0
    assert policy["rss_bytes"] == 1024**3
    assert policy["guarded_launches"] == 1
    assert policy["optimizer_steps"] == policy["hvp_calls"] == policy["pcg_solves"] == 0


def _resource(*, limited: bool = False) -> dict[str, object]:
    return {"exit_code": -15 if limited else 0,
            "resource_termination": "wall_time_limit" if limited else None,
            "monitor_error": None, "received_sigterm": False,
            "wall_limit_seconds": 300.0, "rss_limit_bytes": 1024**3,
            "elapsed_seconds": 301.0 if limited else 1.0,
            "sampled_peak_rss_bytes": 1024,
            "child_process_group_cleanup_sent": False,
            "child_process_group_cleanup_error": None}


def _valid_child(plan_sha: str) -> dict[str, object]:
    return {"phase": "finished", "execution_status": "completed",
            "numerical_status": "finite_chart_samples", "plan_sha256": plan_sha,
            "base_control_sha256": diagnostic.ENDPOINT_SHA,
            "source_unchanged": True, "fixed_input_unchanged": True,
            "runtime": {"device": "CPU FP64"}, "runtime_after": {"device": "CPU FP64"},
            "eta_zero_cost_only": {"gradient_computed": False},
            "samples": [{"requested_eta": eta} for eta in diagnostic.ETA_VALUES]}


def test_parent_receipt_is_durable_before_child_parse(tmp_path, monkeypatch):
    plan_sha = "a" * 64
    plan_path = diagnostic.PLAN
    output, resource, log = (tmp_path / name for name in
                             ("diagnostic.json", "diagnostic.resource.json", "diagnostic.log"))
    parent = output.with_suffix(".run.json")
    monkeypatch.setattr(diagnostic, "_sha", lambda path: plan_sha if Path(path) == plan_path else
                        hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(diagnostic, "load_base", lambda *_args: ({}, {}, None))

    def child_runner(_command, **kwargs):
        output.write_text(json.dumps(_valid_child(plan_sha)))
        return _resource()

    monkeypatch.setattr(diagnostic.guard, "run_guarded_diagnostic", child_runner)
    read_text = Path.read_text

    def checked_read(path: Path, *args, **kwargs):
        if path == output:
            assert parent.exists()
            saved = json.loads(read_text(parent))
            assert saved["execution_status"] == "completed"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    result = diagnostic.run(plan_path, plan_sha, output, resource, log)
    assert result["parent"]["execution_status"] == "completed"
    assert result["child"]["numerical_status"] == "finite_chart_samples"


def test_parent_retains_outer_resource_limited_partial_child(tmp_path, monkeypatch):
    plan_sha = "b" * 64
    plan_path = diagnostic.PLAN
    output, resource, log = (tmp_path / name for name in
                             ("partial.json", "partial.resource.json", "partial.log"))
    monkeypatch.setattr(diagnostic, "_sha", lambda path: plan_sha if Path(path) == plan_path else
                        hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(diagnostic, "load_base", lambda *_args: ({}, {}, None))

    def child_runner(_command, **kwargs):
        output.write_text(json.dumps({"phase": "running", "samples": [{"eta": -2e-6}]}))
        return _resource(limited=True)

    monkeypatch.setattr(diagnostic.guard, "run_guarded_diagnostic", child_runner)
    result = diagnostic.run(plan_path, plan_sha, output, resource, log)
    assert result["parent"]["execution_status"] == "resource_limited"
    assert result["child"]["samples"] == [{"eta": -2e-6}]


def test_parent_fails_closed_on_nonobject_child_and_keeps_receipt(tmp_path, monkeypatch):
    plan_sha = "c" * 64
    plan_path = diagnostic.PLAN
    output, resource, log = (tmp_path / name for name in
                             ("bad.json", "bad.resource.json", "bad.log"))
    parent = output.with_suffix(".run.json")
    monkeypatch.setattr(diagnostic, "_sha", lambda path: plan_sha if Path(path) == plan_path else
                        hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(diagnostic, "load_base", lambda *_args: ({}, {}, None))

    def child_runner(_command, **kwargs):
        output.write_text("[]")
        return _resource()

    monkeypatch.setattr(diagnostic.guard, "run_guarded_diagnostic", child_runner)
    with pytest.raises(RuntimeError, match="could not be verified"):
        diagnostic.run(plan_path, plan_sha, output, resource, log)
    saved = json.loads(parent.read_text())
    assert saved["execution_status"] == "failed"
    assert "JSON object" in saved["child_read_error"]


def test_internal_timeout_keeps_cost_only_and_completed_samples(tmp_path, monkeypatch):
    plan_sha = "d" * 64
    parameters = torch.zeros(13, dtype=torch.float64)
    base = torch.zeros(26, dtype=torch.float64)
    base[24] = torch.atanh(torch.tensor(5.91061e-7 / 0.45, dtype=torch.float64))
    identity: dict[str, object] = {"fixed": "identity"}
    raw: dict[str, object] = {"accepted_control": base.tolist(), "input_after": identity,
                              "runtime": {"device": "CPU FP64"}}

    class Objective:
        def __init__(self) -> None:
            self.frozen = _profile().frozen

        def objective(self, value: torch.Tensor, _parameters: torch.Tensor) -> torch.Tensor:
            return value.square().sum()

    problem = Objective()
    monkeypatch.setattr(diagnostic, "load_base", lambda *_args: ({"source_files": {},
        "archive_files": {}}, raw, None))
    monkeypatch.setattr(diagnostic.seed, "_prepare_fixed_seed", lambda: (
        problem, torch.zeros(1, dtype=torch.float64), torch.zeros(26, dtype=torch.float64),
        parameters, torch.zeros(1, dtype=torch.float64), {}))
    monkeypatch.setattr(diagnostic.seed, "_input_identity", lambda *_args: identity)
    monkeypatch.setattr(diagnostic, "_tensor_sha", lambda value: diagnostic.PARAMETERS_SHA
                        if value.shape == (13,) else diagnostic.ENDPOINT_SHA)
    runtime = {"device": "CPU FP64"}
    monkeypatch.setattr(diagnostic.guarded_policy.blocks, "runtime_identity", lambda: runtime)
    monkeypatch.setattr(diagnostic, "_sha", lambda path: plan_sha if Path(path) == diagnostic.PLAN
                        else hashlib.sha256(Path(path).read_bytes()).hexdigest())
    measure = diagnostic._measure
    completed_count = 0

    def time_out_after_zero(problem_arg, params_arg, control_arg, eta, gradient_fn, *,
                            with_gradient, deadline):
        nonlocal completed_count
        if with_gradient:
            if completed_count == 0:
                completed_count += 1
                return {"requested_eta": eta, "gradient_computed": True,
                        "branch_signature": {"choices": [0, 1], "face_signs": [1, -1]},
                        "static_flux_signs": {"qx": [], "qy": []}}
            raise TimeoutError("test internal budget stop")
        return measure(problem_arg, params_arg, control_arg, eta, gradient_fn,
                       with_gradient=False, deadline=deadline)

    monkeypatch.setattr(diagnostic, "_measure", time_out_after_zero)
    output = tmp_path / "partial.json"
    result = diagnostic._run_child(diagnostic.PLAN, plan_sha, output)
    saved = json.loads(output.read_text())
    assert result["numerical_status"] == "budget_refusal"
    assert saved["eta_zero_cost_only"]["gradient_computed"] is False
    assert len(saved["samples"]) == 1
    assert saved["samples"][0]["requested_eta"] == diagnostic.ETA_VALUES[0]
    assert saved["samples"][0]["branch_signature"]["choices"] == [0, 1]
    assert saved["source_unchanged"] is True
    assert saved["fixed_input_unchanged"] is True
    assert saved["runtime_after"] == runtime
