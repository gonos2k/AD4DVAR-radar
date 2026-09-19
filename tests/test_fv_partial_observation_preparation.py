"""End-to-end research FV checks for fixed-support partial observations.

These tests deliberately keep the initial active support and prescribed FV
boundary support complete.  They cover the preparation boundary only; they do
not broaden the stationarity certificate or typed-prior contracts.
"""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from advar.fv_sensitivity import (
    compute_fv_observation_response,
    refine_fv_stationarity,
)
from advar.physics import echo_to_dbz
from advar.variational import (
    forecast_fv_analysis,
    prepare_analysis,
    solve_analysis,
)


_PROBE_SPEC = spec_from_file_location(
    "fv_sensitivity_probe_partial_preparation",
    Path(__file__).parents[1] / "examples/weather_scenarios/fv_sensitivity_probe.py",
)
if _PROBE_SPEC is None or _PROBE_SPEC.loader is None:
    raise RuntimeError("FV sensitivity probe module is unavailable")
_PROBE = module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(_PROBE)


def _prepare(frames, specification, *, qc_mask=None, quality_weight=None):
    observations, frozen = prepare_analysis(
        frames,
        nowcast_config=_PROBE.v.NowcastConfig(
            interval_minutes=1,
            horizon_minutes=1,
            min_dbz=-10.0,
        ),
        analysis_config=_PROBE.v.AnalysisConfig(field_smoothness_weight=0.0),
        observation_std_dbz=0.1,
        qc_mask=qc_mask,
        quality_weight=quality_weight,
        fv_transport=specification,
    )
    return observations, frozen


@pytest.mark.parametrize("masked_kind", ["missing", "qc", "quality"])
def test_partial_prepare_solve_refine_forecast_and_exact_response(masked_kind):
    original, baseline, future_echo, future_support = _PROBE.make_case()
    frames = original.dbz.clone()
    qc_mask = torch.ones_like(frames, dtype=torch.bool)
    quality_weight = torch.ones_like(frames)
    index = (1, 0, 0)
    if masked_kind == "missing":
        frames[index] = torch.nan
    elif masked_kind == "qc":
        qc_mask[index] = False
    else:
        quality_weight[index] = 0.0

    observations, frozen = _prepare(
        frames,
        baseline.fv_transport,
        qc_mask=qc_mask,
        quality_weight=quality_weight,
    )
    assert not bool(observations.valid_mask[index])
    assert bool(observations.missing_mask[index]) is (masked_kind == "missing")
    # prepare_input records finite samples excluded by the combined accepted
    # mask (including zero quality) as rejected; missing remains distinct.
    assert bool(observations.qc_rejected_mask[index]) is (
        masked_kind != "missing"
    )
    assert bool(observations.valid_mask.any())
    assert torch.equal(frozen.initial_support_mask, torch.ones_like(
        frozen.initial_support_mask
    ))

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
    control, records = refine_fv_stationarity(
        result.control,
        observations,
        configured,
        gradient_tolerance=1.0e-10,
        maximum_iterations=4,
        maximum_normal_products=96,
    )
    assert isinstance(records, list)
    forecast = forecast_fv_analysis(
        control,
        configured,
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future_echo,
        boundary_support=future_support,
    )
    prediction = echo_to_dbz(
        forecast.frames_linear[1:],
        min_dbz=configured.nowcast_config.min_dbz,
    )
    assert torch.isfinite(prediction).all()

    response = compute_fv_observation_response(
        control,
        observations,
        configured,
        verification_dbz=torch.zeros_like(prediction),
        metric_weight=torch.ones_like(prediction),
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future_echo,
        boundary_support=future_support,
        background_dependency="frozen",
        curvature="exact_robust_hessian",
        maximum_normal_products=96,
    )
    assert response.curvature == "exact_robust_hessian"
    assert torch.isfinite(response.sensitivity_dbz).all()
    assert torch.equal(response.sensitivity_dbz[index], response.sensitivity_dbz.new_zeros(()))


def test_finite_masked_values_do_not_change_prepared_fv_state():
    original, baseline, _, _ = _PROBE.make_case()
    qc_mask = torch.ones_like(original.dbz, dtype=torch.bool)
    qc_mask[1, 0, 0] = False
    first = original.dbz.clone()
    second = first.clone()
    second[1, 0, 0] = 65.0
    obs_first, frozen_first = _prepare(
        first,
        baseline.fv_transport,
        qc_mask=qc_mask,
    )
    obs_second, frozen_second = _prepare(
        second,
        baseline.fv_transport,
        qc_mask=qc_mask,
    )
    assert not bool(obs_first.valid_mask[1, 0, 0])
    assert bool(obs_first.qc_rejected_mask[1, 0, 0])
    torch.testing.assert_close(obs_first.dbz, obs_second.dbz, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        frozen_first.baseline_frames_dbz,
        frozen_second.baseline_frames_dbz,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        frozen_first.initial_support_mask,
        frozen_second.initial_support_mask,
    )


def test_parameterized_background_masks_invalid_observation_dependency():
    original, baseline, future_echo, future_support = _PROBE.make_case()
    qc_mask = torch.ones_like(original.dbz, dtype=torch.bool)
    index = (1, 0, 0)
    qc_mask[index] = False
    observations, frozen = _prepare(
        original.dbz,
        baseline.fv_transport,
        qc_mask=qc_mask,
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
    control, _ = refine_fv_stationarity(
        result.control,
        observations,
        configured,
        gradient_tolerance=1.0e-10,
        maximum_iterations=4,
        maximum_normal_products=96,
    )
    theta0 = observations.dbz.new_tensor(0.02)
    minimum = observations.dbz.new_full((), configured.nowcast_config.min_dbz)
    canonical_y0 = torch.where(observations.valid_mask, observations.dbz, minimum)
    height, width = observations.dbz.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=observations.dbz.dtype),
        torch.arange(width, dtype=observations.dbz.dtype),
        indexing="ij",
    )
    direction = 0.01 * (yy - yy.mean()) + 0.02 * (xx - xx.mean())
    baseline_background = configured.initial_background_dbz

    def builder(canonical_y, theta):
        return (
            baseline_background
            + (theta - theta0) * direction
            + 0.2 * (canonical_y[1] - canonical_y0[1])
        )

    kwargs = dict(
        verification_dbz=torch.zeros((2, height, width), dtype=torch.float64),
        metric_weight=torch.ones((2, height, width), dtype=torch.float64),
        leads=2,
        boundary_start_interval=2,
        boundary_echo=future_echo,
        boundary_support=future_support,
        background_dependency="parameterized",
        background_parameter=theta0,
        background_builder=builder,
        curvature="exact_robust_hessian",
        maximum_normal_products=96,
    )
    response = compute_fv_observation_response(
        control, observations, configured, **kwargs
    )
    assert response.sensitivity_theta is not None
    assert torch.isfinite(response.sensitivity_dbz).all()
    assert torch.isfinite(response.sensitivity_theta).all()
    assert bool((response.sensitivity_dbz[observations.valid_mask].abs() > 0).any())
    assert bool(response.sensitivity_theta.abs().max() > 0)
    assert torch.equal(
        response.sensitivity_dbz[index], response.sensitivity_dbz.new_zeros(())
    )

    changed = replace(observations, dbz=observations.dbz.clone())
    changed.dbz[index] = 65.0
    changed_response = compute_fv_observation_response(
        control, changed, configured, **kwargs
    )
    torch.testing.assert_close(
        response.sensitivity_dbz,
        changed_response.sensitivity_dbz,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        response.sensitivity_theta,
        changed_response.sensitivity_theta,
        rtol=0.0,
        atol=0.0,
    )


def test_fv_prepare_rejects_no_valid_observations():
    original, baseline, _, _ = _PROBE.make_case()
    with pytest.raises(ValueError, match="at least one valid observation"):
        _prepare(
            original.dbz,
            baseline.fv_transport,
            quality_weight=torch.zeros_like(original.dbz),
        )


def test_fv_prepare_rejects_missing_initial_support_without_background():
    original, baseline, _, _ = _PROBE.make_case()
    frames = original.dbz.clone()
    frames[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="fully known initial active support"):
        _prepare(frames, baseline.fv_transport)
