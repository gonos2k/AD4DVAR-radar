"""Small, synthetic Phase 1 oracle for the local forecast equations.

The reference below is deliberately written as destination scatter in NumPy.
It does not call the production remapper, so a wrong source/destination
orientation, fractional-cell branch, growth accumulation, or boundary policy
can fail independently of the implementation helper.

Scope of this demonstration: CPU binary64, a known lattice echo, 18 forecast
steps, constant displacement and exponentially damped log-growth.  It is an
equation/invariant check and does not claim universal P0/P1 skill or real-radar
performance.
"""

import math

import numpy as np
import torch

import advar.sensitivity as sensitivity
from advar.nowcast import (
    NowcastConfig,
    RadarState,
    forecast_linear_from_state,
    nowcast,
)
from advar.physics import echo_to_dbz


GRID_SHAPE = (7, 9)
FORECAST_STEPS = 18
INTERVAL_MINUTES = 10
GROWTH_DECAY_MINUTES = 50.0
# Torch and NumPy can differ by a few binary64 ulps in exp; this is a
# linear-echo absolute tolerance, not a model tolerance.
ORACLE_ABSOLUTE_TOLERANCE = 1.0e-12


def _known_lattice() -> np.ndarray:
    """Return a nonuniform, positive lattice whose values are easy to audit."""

    return np.arange(
        1,
        1 + GRID_SHAPE[0] * GRID_SHAPE[1],
        dtype=np.float64,
    ).reshape(GRID_SHAPE) / 10.0


def _destination_scatter(
    source: np.ndarray,
    displacement_yx: tuple[float, float],
) -> np.ndarray:
    """Scatter each source cell to its four destination bilinear cells."""

    height, width = source.shape
    dy, dx = displacement_yx
    integer_y, integer_x = math.floor(dy), math.floor(dx)
    fraction_y, fraction_x = dy - integer_y, dx - integer_x
    moved = np.zeros_like(source)
    for source_y in range(height):
        for source_x in range(width):
            value = source[source_y, source_x]
            for offset_y, weight_y in ((0, 1.0 - fraction_y), (1, fraction_y)):
                for offset_x, weight_x in (
                    (0, 1.0 - fraction_x),
                    (1, fraction_x),
                ):
                    destination_y = source_y + integer_y + offset_y
                    destination_x = source_x + integer_x + offset_x
                    if (
                        0 <= destination_y < height
                        and 0 <= destination_x < width
                    ):
                        moved[destination_y, destination_x] += (
                            value * weight_y * weight_x
                        )
    return moved


def _reference_forecast(
    source: np.ndarray,
    displacement_yx: tuple[float, float],
    log_growth_per_step: float,
    *,
    config: NowcastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each lead directly from the initial lattice and return masses."""

    retention = math.exp(
        -config.interval_minutes / config.growth_decay_minutes
    )
    forecasts = []
    moved_fields = []
    for step in range(1, config.forecast_steps + 1):
        moved = _destination_scatter(
            source,
            (step * displacement_yx[0], step * displacement_yx[1]),
        )
        growth_sum = sum(retention**power for power in range(step))
        forecasts.append(
            moved * math.exp(log_growth_per_step * growth_sum)
        )
        moved_fields.append(moved)
    return np.stack(forecasts), np.stack(moved_fields)


def test_full_horizon_latent_echo_matches_independent_destination_scatter() -> None:
    """Integer and subpixel motion obey direct warp, damping and outflow."""

    config = NowcastConfig(
        horizon_minutes=FORECAST_STEPS * INTERVAL_MINUTES,
        interval_minutes=INTERVAL_MINUTES,
        growth_decay_minutes=GROWTH_DECAY_MINUTES,
    )
    source = _known_lattice()
    cases = (
        ("integer", (1.0, -1.0), math.log(1.08)),
        ("fractional", (0.375, -0.625), -math.log(1.06)),
        ("stationary_growth", (0.0, 0.0), math.log(1.07)),
    )

    for label, displacement, log_growth in cases:
        state = RadarState(
            echo_linear=torch.from_numpy(source.copy()),
            displacement_yx=torch.tensor(displacement, dtype=torch.float64),
            log_growth_per_step=torch.tensor(log_growth, dtype=torch.float64),
        )
        actual = forecast_linear_from_state(state, config).detach().numpy()
        expected, moved = _reference_forecast(
            source,
            displacement,
            log_growth,
            config=config,
        )
        maximum_error = float(np.max(np.abs(actual - expected)))
        assert maximum_error <= ORACLE_ABSOLUTE_TOLERANCE, (
            f"{label} oracle error {maximum_error:.3e} exceeds "
            f"{ORACLE_ABSOLUTE_TOLERANCE:.3e}"
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=ORACLE_ABSOLUTE_TOLERANCE,
        )

        assert bool(np.isfinite(actual).all())
        assert float(actual.min()) >= 0.0
        if label == "stationary_growth":
            # Keep one case supported at the final lead so the geometric
            # growth sum is tested beyond the boundary-outflow cases.
            assert float(actual[-1].max()) > 0.0
        source_mass = float(source.sum())
        outflow = source_mass - moved.sum(axis=(-2, -1))
        assert bool(np.all(outflow >= -ORACLE_ABSOLUTE_TOLERANCE))
        if label == "stationary_growth":
            assert float(outflow[-1]) <= ORACLE_ABSOLUTE_TOLERANCE
        else:
            assert float(outflow[-1]) > float(outflow[0])
            # The final lead has left the small lattice in both cases;
            # discarded mass proves that transport is outflow-only and does
            # not wrap around.
            assert (
                abs(float(outflow[-1]) - source_mass)
                <= ORACLE_ABSOLUTE_TOLERANCE
            )


def test_constructed_state_uses_detached_frozen_metric_weights() -> None:
    """A fixed issued-domain weight is masked once and remains independent."""

    config = NowcastConfig(
        horizon_minutes=FORECAST_STEPS * INTERVAL_MINUTES,
        interval_minutes=INTERVAL_MINUTES,
    )
    source = torch.from_numpy(_known_lattice())
    latest_dbz = echo_to_dbz(
        source,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )
    result = nowcast(
        torch.stack((latest_dbz, latest_dbz, latest_dbz)),
        config,
    )
    lead = config.forecast_steps - 1
    verification_finite = torch.ones_like(result.valid_mask[lead])
    verification_finite[0, 0] = False
    verification_finite[-1, -1] = False
    supplied_weight = torch.linspace(
        0.2,
        1.0,
        source.numel(),
        dtype=source.dtype,
    ).reshape(source.shape).requires_grad_()
    frozen = sensitivity._metric_domain_weight(
        result,
        verification_finite,
        lead,
        "issued",
        verification_metric_weight=supplied_weight,
    )
    expected = torch.where(
        result.valid_mask[lead] & verification_finite,
        supplied_weight,
        torch.zeros_like(supplied_weight),
    )
    torch.testing.assert_close(frozen, expected, rtol=0.0, atol=0.0)
    positive = frozen[frozen > 0.0]
    assert int(positive.numel()) >= source.numel() - 2
    assert float(positive.max()) > float(positive.min())
    with torch.no_grad():
        supplied_weight.zero_()
    torch.testing.assert_close(frozen, expected, rtol=0.0, atol=0.0)
    assert not frozen.requires_grad
