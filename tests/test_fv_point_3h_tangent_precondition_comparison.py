from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from advar import variational as variational
from advar.fv_point_research_problem import FVPointResearchProblem
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_precondition_comparison as precondition


def _synthetic_point_problem(monkeypatch):
    problem = object.__new__(FVPointResearchProblem)
    control = torch.linspace(-0.15, 0.2, 26, dtype=torch.float64)
    observations = torch.linspace(0.1, 1.2, 12, dtype=torch.float64).reshape(3, 4)
    quality = torch.linspace(0.5, 1.0, 12, dtype=torch.float64).reshape(3, 4)
    std = torch.linspace(0.08, 0.14, 12, dtype=torch.float64).reshape(3, 4)
    whitener = torch.eye(4, dtype=torch.float64)
    whitener[0, 1] = whitener[1, 0] = 0.04
    problem_fields = {
        "observation_dbz": observations, "quality_weight": quality,
        "observation_std_dbz": std,
        "observation_coordinates": torch.zeros((4, 2), dtype=torch.float64),
        "_detected_mask": torch.ones((3, 4), dtype=torch.bool),
        "_correlation_whitener": whitener,
        "_masked_whiteners": (None, None, None), "observation_status": None,
    }
    for name, value in problem_fields.items():
        object.__setattr__(problem, name, value)
    contract = SimpleNamespace(
        nowcast_config=SimpleNamespace(min_dbz=-10.0),
        analysis_config=SimpleNamespace(pseudo_huber_delta=2.0))
    object.__setattr__(problem, "contract", lambda _parameters: contract)
    monkeypatch.setattr(variational, "analysis_trajectory",
        lambda value, _contract: SimpleNamespace(frames_linear=value[:12].reshape(3, 4)))
    monkeypatch.setattr("advar.fv_point_research_problem.echo_to_dbz", lambda value, **_kwargs: value)
    monkeypatch.setattr("advar.fv_point_research_problem.point_dbz_bilinear",
        lambda field, _coordinates: field)
    monkeypatch.setattr(variational, "_control_prior_residual", lambda value, _contract: value)
    monkeypatch.setattr(variational, "_field_smoothness_prior_cost",
        lambda value, _contract: value.new_zeros(()))
    parameters = torch.cat((observations.flatten(), torch.zeros(1, dtype=torch.float64)))
    return problem, control, parameters, observations, quality, std, whitener


def test_point_objective_and_row_accessor_share_the_exact_standardized_whitened_residual(monkeypatch):
    problem, control, parameters, observations, quality, std, whitener = _synthetic_point_problem(monkeypatch)
    predicted = torch.linspace(0.2, 1.4, 12, dtype=torch.float64).reshape(3, 4)
    monkeypatch.setattr(variational, "analysis_trajectory",
        lambda _control, _contract: SimpleNamespace(frames_linear=predicted))
    monkeypatch.setattr("advar.fv_point_research_problem.echo_to_dbz", lambda value, **_kwargs: value)
    monkeypatch.setattr("advar.fv_point_research_problem.point_dbz_bilinear",
        lambda field, _coordinates: field)

    expected = (quality.sqrt() * (predicted - observations) / std) @ whitener.T
    residual = problem.observation_residual(control, parameters)
    torch.testing.assert_close(residual, expected)
    cost = problem.objective(control, parameters)
    expected_cost = variational._pseudo_huber_cost(expected, 2.0).sum() + 0.5 * torch.dot(control, control)
    torch.testing.assert_close(cost, expected_cost)
    assert residual.shape == (3, 4)


def test_point_objective_preserves_masked_per_time_residual_groups(monkeypatch):
    problem, control, parameters, observations, quality, std, _whitener = _synthetic_point_problem(monkeypatch)
    detected = torch.tensor([[True, True, True, True], [True, False, True, False],
        [False, False, False, False]])
    object.__setattr__(problem, "_detected_mask", detected)
    masked_whitener = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64)
    object.__setattr__(problem, "_masked_whiteners", (None, masked_whitener, None))
    object.__setattr__(problem, "_correlation_whitener", None)
    predicted = torch.linspace(0.2, 1.4, 12, dtype=torch.float64).reshape(3, 4)
    monkeypatch.setattr(variational, "analysis_trajectory",
        lambda _control, _contract: SimpleNamespace(frames_linear=predicted))
    groups = problem._observation_residual_groups(control, parameters, problem.contract(parameters))
    expected0 = quality[0].sqrt() * (predicted[0] - observations[0]) / std[0]
    index = torch.tensor([0, 2])
    raw1 = quality[1, index].sqrt() * (predicted[1, index] - observations[1, index]) / std[1, index]
    expected1 = masked_whitener @ raw1
    assert len(groups) == 2
    torch.testing.assert_close(groups[0], expected0)
    torch.testing.assert_close(groups[1], expected1)
    expected_cost = sum(variational._pseudo_huber_cost(group, 2.0).sum() for group in groups)
    expected_cost = expected_cost + 0.5 * torch.dot(control, control)
    torch.testing.assert_close(problem.objective(control, parameters), expected_cost)


def test_new_bounded_base_loader_reads_the_pinned_33cb_endpoint_only():
    base = precondition._load_current_base({})
    assert tangent._tensor_sha(base["control"]) == precondition.BASE_CONTROL_SHA
    assert base["theta"] == precondition.BASE_THETA
    assert base["objective"] == precondition.BASE_OBJECTIVE
    assert base["accepted"]["F_squared"] == precondition.BASE_F_SQUARED
    assert base["child_sha256"] == "15c780de8e6b55216407d399d7ce1984280024cfb7f53cbc0ce9e6e4a644db1d"


def test_residual_rows_use_scalar_vjps_without_a_batched_vmap():
    control = torch.linspace(-0.2, 0.3, 26, dtype=torch.float64)
    matrix = torch.arange(12 * 26, dtype=torch.float64).reshape(12, 26) / 300.0
    offset = torch.linspace(-0.1, 0.1, 12, dtype=torch.float64)

    values, rows = precondition.residual_jacobian_rows(lambda x: matrix @ x + offset, control)

    torch.testing.assert_close(values, matrix @ control + offset)
    torch.testing.assert_close(rows, matrix)


def test_pseudo_huber_second_derivative_has_exact_center_and_finite_tail():
    residual = torch.tensor([0.0, 2.0, 1e300], dtype=torch.float64)
    curvature = precondition.pseudo_huber_second_derivative(residual, 2.0)
    torch.testing.assert_close(curvature[:2], torch.tensor([1.0, 1 / (2 ** 1.5)], dtype=torch.float64))
    assert torch.isfinite(curvature).all()
    assert curvature[2] == 0.0


def test_whitened_residual_pair_budget_uses_observation_construction_scale():
    quality = torch.full((3, 4), 0.25, dtype=torch.float64)
    std = torch.full((3, 4), 0.5, dtype=torch.float64)
    problem = SimpleNamespace(quality_weight=quality, observation_std_dbz=std,
        observation_dbz=torch.zeros((3, 4), dtype=torch.float64),
        observation_correlation=None, _correlation_whitener=None)
    parameters = torch.zeros(13, dtype=torch.float64)
    near_zero = torch.zeros((3, 4), dtype=torch.float64)
    mismatched = torch.full((3, 4), 1e-12, dtype=torch.float64)
    strict_budget, strict_scale = precondition._residual_pair_roundoff_budget(
        problem, parameters, near_zero, mismatched)
    assert bool((mismatched > strict_budget).all())
    assert bool((strict_scale == 2e-12).all())

    observed = torch.full((3, 4), 100.0, dtype=torch.float64)
    object.__setattr__(problem, "observation_dbz", observed)
    parameters[:-1] = observed.flatten()
    loose_budget, loose_scale = precondition._residual_pair_roundoff_budget(
        problem, parameters, near_zero, mismatched)
    assert bool((mismatched <= loose_budget).all())
    assert bool((loose_scale > strict_scale).all())


def test_projected_row_parity_budget_includes_ambient_scale_near_face_cancellation():
    dtype = torch.float64
    normal = torch.tensor([1.0, 0.0], dtype=dtype)
    projector = torch.eye(2, dtype=dtype) - torch.outer(normal, normal)
    minus = torch.tensor([[1e8, 0.0], [1e8, 1.0]], dtype=dtype)
    plus = torch.tensor([[1e8, 1e-6], [1e8, 1.0 + 1e-6]], dtype=dtype)
    error, budget, ambient_minus, ambient_plus, projected_norms = (
        precondition._projected_row_parity(minus, plus, projector))
    assert error[0] > 0 and error[0] <= budget[0]
    assert ambient_minus[0] == ambient_plus[0] == 1e8
    assert projected_norms[0, 0] == 0.0


def test_projected_diagonal_matches_small_dense_formula_and_uses_prior_floor():
    dtype = torch.float64
    control = torch.linspace(-0.12, 0.18, 26, dtype=dtype)
    normal = torch.zeros(26, dtype=dtype)
    normal[20:25] = torch.tensor([0.3, -0.6, 0.5, 0.4, -0.2], dtype=dtype)
    tangent = torch.linspace(-0.2, 0.3, 26, dtype=dtype)
    unit = normal / torch.linalg.vector_norm(normal)
    tangent = tangent - unit * torch.dot(unit, tangent)
    gminus, gplus = tangent - 0.2 * normal, tangent + 0.8 * normal
    generator = torch.Generator().manual_seed(19)
    a_minus = torch.randn((12, 26), dtype=dtype, generator=generator) * 0.1
    a_plus = torch.randn((12, 26), dtype=dtype, generator=generator) * 0.1
    residual = torch.linspace(-2.5, 2.0, 12, dtype=dtype)
    weights = torch.tensor([0.4, -0.7, 0.2, 0.5, -0.3], dtype=dtype)
    theta = 0.35

    direction, audit = precondition.projected_gn_jacobi_diagonal(
        normal, gminus, gplus, theta, residual, a_minus, a_plus,
        control, weights, 2.0, floor=1.0)
    projector = torch.eye(26, dtype=dtype) - torch.outer(unit, unit)
    robust = precondition.pseudo_huber_second_derivative(residual, 2.0)
    hgn = (1 - theta) * a_minus.T @ (robust[:, None] * a_minus) \
        + theta * a_plus.T @ (robust[:, None] * a_plus)
    qdiag = torch.zeros(26, dtype=dtype)
    flow = torch.tanh(control[20:25])
    qdiag[20:25] = -2 * weights * flow * (1 - flow.square())
    mix = (1 - theta) * gminus + theta * gplus
    mu = torch.dot(normal, mix) / torch.dot(normal, normal)
    projected = projector @ (torch.eye(26, dtype=dtype) + hgn) @ projector \
        - mu * projector @ torch.diag(qdiag) @ projector
    expected_raw = projected.diagonal()
    expected_w = torch.reciprocal(torch.maximum(expected_raw, torch.ones_like(expected_raw)))
    expected_direction = -projector @ (expected_w * (projector @ mix))
    torch.testing.assert_close(torch.tensor(audit["raw_diagonal"], dtype=dtype), expected_raw)
    torch.testing.assert_close(torch.tensor(audit["inverse_diagonal"], dtype=dtype), expected_w)
    torch.testing.assert_close(direction, expected_direction)
    assert torch.isfinite(direction).all() and (torch.tensor(audit["inverse_diagonal"]) > 0).all()
    assert abs(float(torch.dot(normal, direction))) < 2e-15


def test_identity_jacobi_weight_matches_the_existing_projected_baseline():
    dtype = torch.float64
    control = torch.linspace(-0.1, 0.15, 26, dtype=dtype)
    weights = torch.tensor([0.4, -0.7, 0.2, 0.5, -0.3], dtype=dtype)
    pivot = 20 + int(torch.argmax(weights.abs()))
    control, chart, _ = geometry._chart_jacobian(control, weights, pivot, 0.0)
    normal = geometry._face_normal(control, weights)
    gminus = torch.linspace(-1.0, 1.0, 26, dtype=dtype)
    unit = normal / torch.linalg.vector_norm(normal)
    gminus = gminus - unit * torch.dot(unit, gminus) - 0.1 * normal
    gplus = gminus + 0.5 * normal
    _, _, baseline, _ = tangent.tangent_direction(gminus, gplus, normal, chart, 0.3,
        pivot=pivot)
    direction, audit = precondition.projected_gn_jacobi_diagonal(
        normal, gminus, gplus, 0.3, torch.zeros(12, dtype=dtype),
        torch.zeros((12, 26), dtype=dtype), torch.zeros((12, 26), dtype=dtype),
        control, weights, 2.0, floor=1.0)
    torch.testing.assert_close(torch.tensor(audit["inverse_diagonal"], dtype=dtype),
        torch.ones(26, dtype=dtype))
    torch.testing.assert_close(direction, baseline)


def test_bounded_comparison_runs_both_arms_then_commits_only_lower_actual_f_squared():
    dtype = torch.float64
    control = torch.zeros(2, dtype=dtype)
    gminus, gplus = torch.tensor([-1.0, -1.0], dtype=dtype), torch.tensor([1.0, -1.0], dtype=dtype)
    normal, chart = torch.tensor([1.0, 0.0], dtype=dtype), torch.tensor([[0.0], [1.0]], dtype=dtype)
    directions = {"baseline_tangent": torch.tensor([0.0, 1.0], dtype=dtype),
        "robust_gn_jacobi": torch.tensor([0.0, 0.5], dtype=dtype)}
    hvps = []

    def products(point, theta):
        return gminus, gplus, normal, chart, 0.0

    def direction_models(point, theta, gm, gp, n, z, pivot):
        return [{"name": name, "direction": value, "direction_override": value,
            "diagnostics": {"arm": name}} for name, value in directions.items()]

    def hvp(point, direction, side):
        hvps.append((side, direction.clone()))
        diagonal = torch.tensor([0.2, 0.4] if side < 0 else [0.6, 0.8], dtype=dtype)
        return diagonal * direction

    def evaluate(model, candidate, theta, alpha):
        actual_f_squared = (0.30 if model.direction[1] > 0.75 else 0.20)
        return {key: True for key in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"objective": 1.0, "F_squared": actual_f_squared,
                "theta": theta, "control": candidate.tolist(),
                "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}}

    def final(proposal):
        return {key: True for key in ("side_gradients_finite", "native_objective_matches_proposal",
            "side_objectives_match_native", "merit_matches_proposal", "face_audit_passed",
            "branch_pair_passed", "trace_matches_proposal", "source_unchanged",
            "fixed_input_unchanged", "runtime_unchanged", "deadline_passed")} | {
                "theta": proposal["theta"], "control": proposal["control"],
                "side_gradients": proposal["side_gradients"]}

    result = tangent.bounded_continuation(control, 0.25, products, hvp,
        lambda base, direction, alpha: base + alpha * direction,
        lambda _candidate, _theta, _alpha: {}, final,
        direction_models=direction_models,
        model_side_hvp=lambda point, direction, side, _name: hvp(point, direction, side),
        model_candidate_eval=evaluate, max_accepted=1)
    assert result["accepted_iterations"] == 1
    assert len(hvps) == 4
    assert len(result["iterations"][0]["model_comparisons"]) == 2
    assert result["iterations"][0]["selected_direction_model"] == "robust_gn_jacobi"
    assert result["iterations"][0]["accepted"] is True


def test_incomplete_second_arm_never_commits_baseline_candidate():
    dtype = torch.float64
    control = torch.zeros(2, dtype=dtype)
    gminus, gplus = torch.tensor([-1.0, -1.0], dtype=dtype), torch.tensor([1.0, -1.0], dtype=dtype)
    normal, chart = torch.tensor([1.0, 0.0], dtype=dtype), torch.tensor([[0.0], [1.0]], dtype=dtype)
    directions = [torch.tensor([0.0, 1.0], dtype=dtype), torch.tensor([0.0, 0.5], dtype=dtype)]
    hvp_calls = 0

    def products(point, theta):
        return gminus, gplus, normal, chart, 0.0

    def models(point, theta, gm, gp, n, z, pivot):
        return [{"name": name, "direction": direction, "direction_override": direction}
            for name, direction in zip(("baseline_tangent", "robust_gn_jacobi"), directions, strict=True)]

    def hvp(point, direction, side):
        nonlocal hvp_calls
        hvp_calls += 1
        if hvp_calls == 4:
            raise TimeoutError("synthetic paired-arm interruption")
        return torch.tensor([0.0, 0.4], dtype=dtype)

    def evaluator(model, candidate, theta, alpha):
        return {key: True for key in ("J_armijo_passed", "F_squared_armijo_passed",
            "face_audit_passed", "branch_pair_passed", "side_objectives_match_native",
            "side_gradients_finite")} | {"objective": 1.0, "F_squared": 0.2,
                "theta": theta, "control": candidate.tolist(),
                "side_gradients": {"-1": [-1.0, -1.0], "1": [1.0, -1.0]}}

    result = tangent.bounded_continuation(control, 0.25, products, hvp,
        lambda base, direction, alpha: base + alpha * direction,
        lambda _candidate, _theta, _alpha: {},
        lambda _proposal: {}, direction_models=models,
        model_side_hvp=lambda point, direction, side, _name: hvp(point, direction, side),
        model_candidate_eval=evaluator, max_accepted=1)
    torch.testing.assert_close(result["control"], control)
    assert result["theta"] == 0.25
    assert result["accepted_iterations"] == 0
    assert "TimeoutError" in result["iterations"][0]["refusal"]


def test_direction_spec_mismatch_is_refused_before_its_hvp():
    dtype = torch.float64
    control = torch.zeros(2, dtype=dtype)
    gminus, gplus = torch.tensor([-1.0, -1.0], dtype=dtype), torch.tensor([1.0, -1.0], dtype=dtype)
    normal, chart = torch.tensor([1.0, 0.0], dtype=dtype), torch.tensor([[0.0], [1.0]], dtype=dtype)
    hvp_calls = 0

    def products(point, theta):
        return gminus, gplus, normal, chart, 0.0

    def models(point, theta, gm, gp, n, z, pivot):
        return [{"name": "baseline_tangent", "direction": torch.tensor([0.0, 0.5], dtype=dtype),
            "direction_override": None}]

    def hvp(point, direction, side):
        nonlocal hvp_calls
        hvp_calls += 1
        return torch.zeros_like(direction)

    result = tangent.bounded_continuation(control, 0.25, products, hvp,
        lambda base, direction, alpha: base + alpha * direction,
        lambda _candidate, _theta, _alpha: {}, lambda _proposal: {},
        direction_models=models, model_side_hvp=lambda p, d, s, _n: hvp(p, d, s),
        model_candidate_eval=lambda *_args: {}, max_accepted=1)
    assert hvp_calls == 0
    assert result["accepted_iterations"] == 0
    assert "differs from the baseline HVP direction" in result["iterations"][0]["refusal"]


def test_candidate_selector_applies_actual_f_squared_then_j_then_baseline_ties():
    dtype = torch.float64
    eps = torch.finfo(dtype).eps
    baseline = {"direction_model": "baseline_tangent", "F_squared": 1.0,
        "objective": 2.0, "predicted_reduction": -50.0}
    metric = {"direction_model": "robust_gn_jacobi", "F_squared": 1.0 + 64 * eps,
        "objective": 1.0, "predicted_reduction": 100.0}
    assert tangent._select_direction_candidate([baseline, metric], dtype) is metric

    tied_metric_j = {**metric, "objective": 2.0}
    assert tangent._select_direction_candidate([tied_metric_j, baseline], dtype) is baseline

    actual_better = {**metric, "F_squared": 0.9, "objective": 3.0}
    assert tangent._select_direction_candidate([baseline, actual_better], dtype) is actual_better


def test_fake_full_child_obeys_row_hvp_and_per_arm_grid_budgets(tmp_path, monkeypatch):
    base = precondition._load_current_base({})
    control = base["control"]
    theta = base["theta"]
    accepted = base["accepted"]
    gminus = torch.as_tensor(accepted["side_gradients"]["-1"], dtype=torch.float64)
    gplus = torch.as_tensor(accepted["side_gradients"]["1"], dtype=torch.float64)
    jump = gplus - gminus
    flow_derivative = 1.0 - torch.tanh(control[20:25]).square()
    face_weights = jump[20:25] / flow_derivative
    face_weights = 0.84 * face_weights / face_weights.abs().max()
    basis = torch.zeros((5, 5, 5), dtype=torch.float64)
    basis[:, 4, 4] = -face_weights
    transport_spec = SimpleNamespace(psi_basis=basis,
        coefficient_limits=torch.ones(5, dtype=torch.float64))
    frozen = SimpleNamespace(fv_transport=transport_spec,
        active_field_index=torch.arange(20),
        neural_prior_std_dbz=None, neural_prior_valid_mask=None,
        neural_prior_dependency=None,
        analysis_config=SimpleNamespace(field_smoothness_weight=0.0,
            pseudo_huber_delta=2.0))
    hessian_diagonal = torch.linspace(0.01, 0.04, 26, dtype=torch.float64)
    mixed = (1.0 - theta) * gminus + theta * gplus
    base_objective = base["objective"]

    class ToyPointProblem:
        layout = {"controls": 26, "parameters": 13}
        observation_dbz = torch.zeros((3, 4), dtype=torch.float64)
        observation_std_dbz = torch.ones((3, 4), dtype=torch.float64)
        quality_weight = torch.ones((3, 4), dtype=torch.float64)
        observation_correlation = None
        _correlation_whitener = None
        observation_status = None
        _detected_mask = torch.ones((3, 4), dtype=torch.bool)

        def __init__(self):
            self.frozen = frozen

        def contract(self, _parameters):
            return frozen

        def objective(self, value, _parameters):
            displacement = value - control
            return (value.new_tensor(base_objective) + torch.dot(mixed, displacement)
                + 0.5 * torch.dot(hessian_diagonal * displacement, displacement))

        def observation_residual(self, value, _parameters):
            residual_base = torch.linspace(-0.4, 0.5, 12, dtype=torch.float64)
            row_map = torch.arange(12 * 26, dtype=torch.float64).reshape(12, 26) / 10000.0
            return (residual_base + row_map @ (value - control)).reshape(3, 4)

    problem = ToyPointProblem()
    parameters = torch.zeros(13, dtype=torch.float64)
    original = torch.zeros((4, 5), dtype=torch.float64)
    truth = torch.zeros((4, 5), dtype=torch.float64)
    runtime = base["raw"]["runtime_after"]
    source = {"fixed.py": "saved"}
    plan = {"experiment_kind": "current_tangent_precondition_comparison",
        "policy": precondition.POLICY, "source_files": {}, "archive_files": {},
        "base_control_sha256": precondition.BASE_CONTROL_SHA,
        "initial_theta": precondition.BASE_THETA, "face": precondition.FACE}
    output = tmp_path / "synthetic-child.json"

    def input_identity(_problem, _original, _control, _parameters, _truth):
        return base["raw"]["input_after"]

    traces = accepted["branch_trace"]
    pair = accepted["current_branch_pair_gate"]

    def observe(_probe, _problem, value, _parameters, weights):
        displacement = value - control
        objective = problem.objective(value, parameters)
        minus = gminus + hessian_diagonal * displacement
        plus = gplus + hessian_diagonal * displacement
        q = geometry._face_value(value, weights)
        return {"native_j": objective,
            "side": {-1: (objective, minus), 1: (objective, plus)},
            "traces": {-1: traces["-1"], 1: traces["1"]},
            "pair": pair, "q": q, "production_q": q,
            "face_bound": 1e-12, "face_ok": bool(abs(float(q)) <= 1e-12),
            "side_objectives_match_native": True, "side_gradients_finite": True}

    # The row residual is synthetic; keep its declared data-cost sum consistent
    # with this toy J while exercising the same saved-input runner contract.
    prior_cost = 0.5 * torch.dot(control, control)
    data_cost_per_row = (base_objective - float(prior_cost)) / 12.0
    monkeypatch.setattr(precondition, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(precondition, "_load_current_base", lambda _plan: base)
    monkeypatch.setattr(tangent.seed, "_prepare_fixed_seed", lambda: (
        problem, original, None, parameters, truth, {}))
    monkeypatch.setattr(shared, "_input_identity", input_identity)
    monkeypatch.setattr(shared, "_runtime", lambda: runtime)
    monkeypatch.setattr(shared, "_source_hashes", lambda *_: source)
    monkeypatch.setattr(tangent, "_observe", observe)
    monkeypatch.setattr(geometry.model, "_fixed_input", lambda *_args: True)
    monkeypatch.setattr(variational, "_control_prior_residual", lambda value, _frozen: value)
    monkeypatch.setattr(variational, "_field_smoothness_prior_cost",
        lambda value, _frozen: value.new_zeros(()))
    monkeypatch.setattr(variational, "_pseudo_huber_cost",
        lambda residual, _delta: residual.new_full(residual.shape, data_cost_per_row))

    child = precondition._run_child_impl(tmp_path / "plan.json", "synthetic-plan", output)

    assert child["execution_status"] == "completed"
    assert child["base_control_sha256"] == precondition.BASE_CONTROL_SHA
    assert child["jacobian_rows_started"] == child["jacobian_rows_completed"] == 24
    assert len(child["jacobian_row_history"]) == 24
    assert child["hvp_calls_started"] == child["hvp_calls_completed"] == 4
    assert len(child["iterations"]) == 1
    arms = child["iterations"][0]["model_comparisons"]
    assert {arm["name"] for arm in arms} == {"baseline_tangent", "robust_gn_jacobi"}
    assert all(len(arm["trials"]) <= 16 for arm in arms)
    assert {row["direction_model"] for row in child["hvp_history"]} == {
        "baseline_tangent", "robust_gn_jacobi"}
    assert child["candidate_committed"] is (child["accepted_iterations"] == 1)
