"""Bounded gates for the research FV observation response."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from advar.fv_sensitivity import (
    compute_fv_observation_response,
    refine_fv_stationarity,
)
from advar.variational import (
    forecast_fv_analysis,
    initial_control,
    robust_objective,
    solve_analysis,
)
from advar.physics import echo_to_dbz

_PROBE_SPEC = spec_from_file_location(
    "fv_sensitivity_probe",
    Path(__file__).parents[1] / "examples/weather_scenarios/fv_sensitivity_probe.py",
)
if _PROBE_SPEC is None or _PROBE_SPEC.loader is None:
    raise RuntimeError("FV sensitivity probe module is unavailable")
_PROBE = module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(_PROBE)


def _refined_case():
    observations, frozen, boundary_echo, boundary_support = _PROBE.make_case()
    configured = replace(
        frozen,
        analysis_config=replace(
            frozen.analysis_config,
            maximum_outer_iterations=16,
            maximum_pcg_iterations=96,
            gradient_tolerance=1.0e-10,
            step_tolerance=1.0e-12,
            pcg_relative_tolerance=1.0e-10,
        ),
    )
    result = solve_analysis(observations, configured)
    assert not result.stationarity_verified

    def contract(y):
        return replace(frozen, initial_background_dbz=y[0])

    def objective(control, y):
        return robust_objective(
            control,
            replace(observations, dbz=y),
            contract(y),
        )

    gradient = torch.func.grad(objective, argnums=0)(result.control, observations.dbz)
    assert torch.max(torch.abs(gradient)) < 1.0e-8
    return observations, configured, result.control, boundary_echo, boundary_support


@pytest.fixture(scope="module")
def refined_case():
    return _refined_case()


@pytest.fixture(scope="module")
def partial_refined_case():
    observations, frozen, boundary_echo, boundary_support = _PROBE.make_case()
    detected = observations.detected_mask.clone()
    valid = observations.valid_mask.clone()
    missing = observations.missing_mask.clone()
    detected[1, 0, 0] = False
    valid[1, 0, 0] = False
    missing[1, 0, 0] = True
    observations = replace(
        observations,
        detected_mask=detected,
        valid_mask=valid,
        missing_mask=missing,
    )
    configured = replace(
        frozen,
        analysis_config=replace(
            frozen.analysis_config,
            maximum_outer_iterations=16,
            maximum_pcg_iterations=96,
            gradient_tolerance=1.0e-10,
            step_tolerance=1.0e-12,
            pcg_relative_tolerance=1.0e-10,
        ),
    )
    result = solve_analysis(observations, configured)
    gradient = torch.func.grad(
        lambda control: robust_objective(control, observations, configured)
    )(result.control)
    assert torch.max(torch.abs(gradient)) < 1.0e-8
    return observations, configured, result.control, boundary_echo, boundary_support


def _replace_first_edge(schedule, edge):
    return _replace_stage_edge(schedule, 0, edge)


def _replace_stage_edge(schedule, edge_index, edge):
    stages = list(schedule[0])
    edges = list(stages[0])
    edges[edge_index] = edge
    stages[0] = tuple(edges)
    updated = list(schedule)
    updated[0] = tuple(stages)
    return tuple(updated)


def _response_kwargs(observations, boundary_echo, boundary_support):
    shape = (2, *observations.dbz.shape[-2:])
    return dict(
        verification_dbz=torch.zeros(shape, dtype=torch.float64),
        metric_weight=torch.ones(shape, dtype=torch.float64),
        leads=2,
        boundary_start_interval=2,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
        background_dependency="first_observation",
        maximum_normal_products=128,
    )


def _mask_unknown_support_weights(
    control, frozen, boundary_echo, boundary_support, weights
):
    trajectory = forecast_fv_analysis(
        control,
        frozen,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
    )
    assert trajectory.support_frames is not None
    known = trajectory.support_frames[1:] >= (
        1 - 128 * torch.finfo(control.dtype).eps
    )
    result = weights.clone()
    result[~known] = 0.0
    assert torch.any(result > 0)
    return result


def _inputs():
    observations, frozen, boundary_echo, boundary_support = _PROBE.make_case()
    control = initial_control(frozen)
    size = frozen.active_field_index.numel()
    control[size:-1] = control.new_tensor([0.2, -0.1, 0.3])
    leads = 2
    shape = (leads, *observations.dbz.shape[-2:])
    verification = torch.zeros(shape, dtype=control.dtype)
    weights = torch.ones(shape, dtype=control.dtype)
    kwargs = dict(
        verification_dbz=verification,
        metric_weight=weights,
        leads=leads,
        boundary_start_interval=2,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
        background_dependency="frozen",
        maximum_normal_products=64,
    )
    return observations, frozen, control, kwargs


def test_response_rejects_non_donorcell_before_stationarity():
    observations, frozen, control, kwargs = _inputs()
    fv = frozen.fv_transport
    assert fv is not None
    minmod_frozen = replace(frozen, fv_transport=replace(fv, reconstruction="minmod"))
    with pytest.raises(ValueError, match="donorcell"):
        compute_fv_observation_response(control, observations, minmod_frozen, **kwargs)


def test_response_is_cpu_fp64_only():
    observations, frozen, control, kwargs = _inputs()
    with pytest.raises(ValueError, match="CPU FP64"):
        compute_fv_observation_response(
            control.to(torch.float32), observations, frozen, **kwargs,
        )


def test_response_rejects_nonstationary_nonzero_dynamics():
    observations, frozen, control, kwargs = _inputs()
    with pytest.raises(ValueError, match="refined robust stationary"):
        compute_fv_observation_response(control, observations, frozen, **kwargs)


def test_response_requires_the_declared_first_observation_background():
    observations, frozen, control, kwargs = _inputs()
    mismatched = replace(
        frozen,
        initial_background_dbz=frozen.initial_background_dbz + 0.1,
    )
    with pytest.raises(ValueError, match=r"B=y\[0\]"):
        compute_fv_observation_response(
            control,
            observations,
            mismatched,
            **{**kwargs, "background_dependency": "first_observation"},
        )


def test_response_rejects_missing_first_observation_background():
    observations, frozen, control, kwargs = _inputs()
    detected = observations.detected_mask.clone()
    valid = observations.valid_mask.clone()
    missing = observations.missing_mask.clone()
    detected[0, 0, 0] = False
    valid[0, 0, 0] = False
    missing[0, 0, 0] = True
    missing_observation = replace(
        observations,
        detected_mask=detected,
        valid_mask=valid,
        missing_mask=missing,
    )
    with pytest.raises(ValueError, match="fully detected initial background"):
        compute_fv_observation_response(
            control,
            missing_observation,
            frozen,
            **{**kwargs, "background_dependency": "first_observation"},
        )


@pytest.mark.parametrize(
    "metric_weight, message",
    [
        (torch.zeros((2, 4, 5), dtype=torch.float64), "positive total"),
        (-torch.ones((2, 4, 5), dtype=torch.float64), "nonnegative"),
    ],
)
def test_response_rejects_invalid_metric_weights(metric_weight, message):
    observations, frozen, control, kwargs = _inputs()
    with pytest.raises(ValueError, match=message):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, "metric_weight": metric_weight},
        )


def test_response_rejects_an_exhausted_normal_product_budget():
    observations, frozen, control, kwargs = _inputs()
    with pytest.raises(ValueError, match="positive integer"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, "maximum_normal_products": 0},
        )


def test_zero_flow_control_is_rejected_as_a_sensitive_face_tie():
    observations, frozen, _, kwargs = _inputs()
    zero = initial_control(frozen)
    with pytest.raises(ValueError, match="coefficient-sensitive face sign"):
        compute_fv_observation_response(zero, observations, frozen, **kwargs)


def test_response_accepts_partial_future_boundary_support(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    support = kwargs["boundary_support"]
    bad_edge = support[0][0][0].clone()
    bad_edge[0] = 0.5
    weights = _mask_unknown_support_weights(
        control,
        frozen,
        boundary_echo,
        _replace_first_edge(support, bad_edge),
        kwargs["metric_weight"],
    )
    response = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{
            **kwargs,
            "metric_weight": weights,
            "boundary_support": _replace_first_edge(support, bad_edge),
        },
    )
    assert torch.isfinite(response.sensitivity_dbz).all()


def test_response_rejects_support_outside_fixed_unit_interval(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    bad_edge = boundary_support[0][0][0].clone()
    bad_edge[0] = 1.0 + 1.0e-12
    with pytest.raises(ValueError, match="fixed support range"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, "boundary_support": _replace_first_edge(boundary_support, bad_edge)},
        )


def test_response_rejects_near_one_fractional_support(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    near_one = boundary_support[0][0][0].clone()
    near_one[0] = torch.nextafter(
        near_one.new_tensor(1.0), near_one.new_tensor(0.0)
    )
    with pytest.raises(ValueError, match="without fixed known support"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{
                **kwargs,
                "boundary_support": _replace_first_edge(boundary_support, near_one),
            },
        )


def test_response_accepts_partial_outflow_analysis_support_with_missing_data(
    partial_refined_case,
):
    observations, frozen, control, boundary_echo, boundary_support = (
        partial_refined_case
    )
    fv = frozen.fv_transport
    assert fv is not None
    partial_edge = fv.boundary_support[0][0][1].clone()
    partial_edge[0] = 0.5
    partial_fv = replace(
        fv,
        boundary_support=_replace_stage_edge(fv.boundary_support, 1, partial_edge),
    )
    partial_frozen = replace(frozen, fv_transport=partial_fv)
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    response = compute_fv_observation_response(
        control,
        observations,
        partial_frozen,
        **{**kwargs, "background_dependency": "frozen"},
    )
    assert torch.isfinite(response.sensitivity_dbz).all()


def test_response_rejects_active_observations_outside_known_support(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    fv = frozen.fv_transport
    assert fv is not None
    near_one = fv.boundary_support[0][0][0].clone()
    near_one[0] = 1.0 - 1.0e-12
    partial_fv = replace(
        fv,
        boundary_support=_replace_first_edge(fv.boundary_support, near_one),
    )
    partial_frozen = replace(frozen, fv_transport=partial_fv)
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    with pytest.raises(ValueError, match="active observations"):
        compute_fv_observation_response(
            control,
            observations,
            partial_frozen,
            **{**kwargs, "background_dependency": "frozen"},
        )


@pytest.mark.parametrize("input_name", ["verification_dbz", "metric_weight"])
def test_response_requires_fixed_verification_and_weight_tensors(input_name):
    observations, frozen, control, kwargs = _inputs()
    attached = kwargs[input_name].clone().requires_grad_()
    with pytest.raises(ValueError, match="fixed tensors"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, input_name: attached},
        )


def test_response_requires_fixed_future_boundary_tensors():
    observations, frozen, control, kwargs = _inputs()
    edge = kwargs["boundary_echo"][0][0][0].clone().requires_grad_()
    with pytest.raises(ValueError, match="fixed tensors"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{
                **kwargs,
                "boundary_echo": _replace_first_edge(
                    kwargs["boundary_echo"], edge,
                ),
            },
        )


def test_refined_response_ignores_finite_huge_verification_at_zero_weight(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    zero_weights = kwargs["metric_weight"].clone()
    zero_weights[0, 0, 0] = 0.0
    finite_verification = kwargs["verification_dbz"].clone()
    huge_verification = finite_verification.clone()
    huge_verification[0, 0, 0] = torch.finfo(control.dtype).max / 4.0
    finite = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{
            **kwargs,
            "verification_dbz": finite_verification,
            "metric_weight": zero_weights,
        },
    )
    huge = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{
            **kwargs,
            "verification_dbz": huge_verification,
            "metric_weight": zero_weights,
        },
    )
    assert torch.isfinite(huge.sensitivity_dbz).all()
    assert huge.score == pytest.approx(finite.score, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(
        huge.sensitivity_dbz,
        finite.sensitivity_dbz,
        rtol=1e-10,
        atol=1e-12,
    )


def test_refined_response_is_invariant_to_uniform_weight_scaling(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    baseline = compute_fv_observation_response(control, observations, frozen, **kwargs)
    scaled = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{**kwargs, "metric_weight": kwargs["metric_weight"] * 1.0e308},
    )
    assert scaled.score == pytest.approx(baseline.score, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(
        scaled.sensitivity_dbz,
        baseline.sensitivity_dbz,
        rtol=1e-10,
        atol=1e-12,
    )


def test_exact_robust_hessian_matches_dense_small_oracle(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    response = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{**kwargs, "curvature": "exact_robust_hessian"},
    )
    assert response.curvature == "exact_robust_hessian"

    def contract(y):
        return replace(frozen, initial_background_dbz=y[0])

    def objective(value, y):
        return robust_objective(value, replace(observations, dbz=y), contract(y))

    def score(value, y):
        trajectory = forecast_fv_analysis(
            value,
            contract(y),
            leads=2,
            boundary_start_interval=2,
            boundary_echo=boundary_echo,
            boundary_support=boundary_support,
        )
        prediction = echo_to_dbz(
            trajectory.frames_linear[1:], min_dbz=frozen.nowcast_config.min_dbz
        )
        return (prediction.square()).mean()

    gradient = torch.func.grad(objective, argnums=0)
    hessian = _PROBE.dense_hessian(gradient, control, observations.dbz)
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(control, observations.dbz)
    _, pullback = torch.func.vjp(
        lambda y: gradient(control, y), observations.dbz
    )
    expected = direct - pullback(torch.linalg.solve(hessian.T, rhs))[0]
    torch.testing.assert_close(
        response.sensitivity_dbz,
        expected,
        rtol=2e-8,
        atol=2e-10,
    )


def test_exact_response_matches_reanalysis_fd_with_missing_and_partial_support(
    partial_refined_case,
):
    observations, frozen, control, boundary_echo, boundary_support = (
        partial_refined_case
    )
    partial_edge = boundary_support[0][0][0].clone()
    partial_edge[0] = 0.5
    boundary_support = _replace_first_edge(boundary_support, partial_edge)
    metric_weight = torch.arange(1, 41, dtype=torch.float64).reshape(2, 4, 5)
    metric_weight = _mask_unknown_support_weights(
        control,
        frozen,
        boundary_echo,
        boundary_support,
        metric_weight,
    )
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    kwargs.update(metric_weight=metric_weight, curvature="exact_robust_hessian")
    response = compute_fv_observation_response(
        control, observations, frozen, **kwargs
    )

    direction = torch.zeros_like(observations.dbz)
    detected = observations.detected_mask
    direction[detected] = torch.linspace(
        -1.0, 1.0, int(detected.sum()), dtype=direction.dtype
    )
    direction = direction / torch.linalg.vector_norm(direction)
    step = 1.0e-4

    def reanalyzed_score(value):
        shifted = replace(observations, dbz=value)
        shifted_frozen = replace(frozen, initial_background_dbz=value[0])
        analyzed = solve_analysis(shifted, shifted_frozen)
        analyzed_control, _ = refine_fv_stationarity(
            analyzed.control,
            shifted,
            shifted_frozen,
            gradient_tolerance=1.0e-10,
            maximum_iterations=4,
            maximum_normal_products=96,
        )
        gradient = torch.func.grad(
            lambda candidate: robust_objective(candidate, shifted, shifted_frozen)
        )(analyzed_control)
        assert torch.max(torch.abs(gradient)) < 1.0e-8
        trajectory = forecast_fv_analysis(
            analyzed_control,
            shifted_frozen,
            leads=2,
            boundary_start_interval=2,
            boundary_echo=boundary_echo,
            boundary_support=boundary_support,
        )
        prediction = echo_to_dbz(
            trajectory.frames_linear[1:],
            min_dbz=frozen.nowcast_config.min_dbz,
        )
        normalized = metric_weight / metric_weight.max()
        normalized = normalized / normalized.sum()
        return (normalized * (prediction - kwargs["verification_dbz"]).square()).sum()

    finite_difference = (
        reanalyzed_score(observations.dbz + step * direction)
        - reanalyzed_score(observations.dbz - step * direction)
    ) / (2.0 * step)
    directional_response = (response.sensitivity_dbz * direction).sum()
    torch.testing.assert_close(
        directional_response,
        finite_difference,
        # Polish endpoints: differencing amplifies their solve error by 1/h.
        rtol=1.0e-9,
        atol=2.0e-9,
    )


def test_exact_robust_hessian_allows_zero_score_response(refined_case):
    observations, frozen, control, boundary_echo, boundary_support = refined_case
    trajectory = forecast_fv_analysis(
        control,
        frozen,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=boundary_echo,
        boundary_support=boundary_support,
    )
    verification = echo_to_dbz(
        trajectory.frames_linear[1:], min_dbz=frozen.nowcast_config.min_dbz
    )
    kwargs = _response_kwargs(observations, boundary_echo, boundary_support)
    response = compute_fv_observation_response(
        control,
        observations,
        frozen,
        **{
            **kwargs,
            "verification_dbz": verification,
            "curvature": "exact_robust_hessian",
        },
    )
    assert response.curvature == "exact_robust_hessian"
    assert response.normal_products == 0
    assert response.score == pytest.approx(0.0, abs=1e-24)
    torch.testing.assert_close(
        response.sensitivity_dbz,
        torch.zeros_like(response.sensitivity_dbz),
        rtol=0.0,
        atol=1e-12,
    )


def test_exact_robust_hessian_rejects_negative_curvature(monkeypatch):
    observations, frozen, control, kwargs = _inputs()
    center = control.clone()

    def concave_objective(value, _observations, _frozen):
        delta = value - center
        return -0.5 * torch.dot(delta, delta)

    monkeypatch.setattr("advar.fv_sensitivity.robust_objective", concave_objective)
    with pytest.raises(ValueError, match="non-positive curvature"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, "curvature": "exact_robust_hessian"},
        )


def test_response_rejects_weight_range_that_underflows_normalization():
    observations, frozen, control, kwargs = _inputs()
    weights = torch.full((2, 4, 5), 1.0e308, dtype=torch.float64)
    weights[0, 0, 0] = 1.0e-308
    with pytest.raises(ValueError, match="dynamic range|underflows"):
        compute_fv_observation_response(
            control,
            observations,
            frozen,
            **{**kwargs, "metric_weight": weights},
        )


def test_response_rejects_observation_dtype_mismatch():
    observations, frozen, control, kwargs = _inputs()
    fp32 = replace(
        observations,
        dbz=observations.dbz.float(),
        std_dbz=observations.std_dbz.float(),
        quality_weight=observations.quality_weight.float(),
    )
    with pytest.raises(ValueError, match="frozen CPU FP64 analysis grid"):
        compute_fv_observation_response(control, fp32, frozen, **kwargs)


def test_response_rejects_narrow_observation_shape_without_broadcasting():
    observations, frozen, control, kwargs = _inputs()
    spatial = (..., slice(None, -1), slice(None))
    narrowed = replace(
        observations,
        dbz=observations.dbz[spatial],
        std_dbz=observations.std_dbz[spatial],
        quality_weight=observations.quality_weight[spatial],
        valid_mask=observations.valid_mask[spatial],
        detected_mask=observations.detected_mask[spatial],
        censored_mask=observations.censored_mask[spatial],
        missing_mask=observations.missing_mask[spatial],
        qc_rejected_mask=observations.qc_rejected_mask[spatial],
    )
    with pytest.raises(ValueError, match="frozen CPU FP64 analysis grid"):
        compute_fv_observation_response(control, narrowed, frozen, **kwargs)
