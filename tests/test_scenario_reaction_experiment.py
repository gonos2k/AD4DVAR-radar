"""Independent tests for the standalone C2 reaction experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "weather_scenarios"))

from reaction_experiment import (  # noqa: E402
    estimate_kappa_from_observations,
    forecast_from_observations,
    reaction_dbz,
)


MIN_DBZ = -10.0
KAPPA = 0.045
LOG10_OVER_10 = math.log(10.0) / 10.0


def _oracle_reaction(
    dbz: np.ndarray,
    kappa: float,
    horizon_steps: float,
) -> np.ndarray:
    """Independent NumPy evaluation of the dBZ closed form."""

    scale = 10.0 ** (MIN_DBZ / 10.0)
    q0 = scale * np.expm1(LOG10_OVER_10 * (dbz - MIN_DBZ))
    qh = scale * np.expm1(
        np.exp(-kappa * horizon_steps) * np.log1p(q0 / scale)
    )
    return MIN_DBZ + np.log1p(qh / scale) / LOG10_OVER_10


@pytest.mark.parametrize("q0", (0.0, 1.0e-14, 1.0e-8, 0.2))
def test_closed_form_small_q_and_zero_kappa_boundaries(q0: float) -> None:
    scale = 10.0 ** (MIN_DBZ / 10.0)
    dbz_value = MIN_DBZ + math.log1p(q0 / scale) / LOG10_OVER_10
    dbz = torch.tensor([dbz_value], dtype=torch.float64)

    actual = reaction_dbz(dbz, KAPPA, 7.0, min_dbz=MIN_DBZ).numpy()
    expected = _oracle_reaction(np.array([dbz_value]), KAPPA, 7.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
    torch.testing.assert_close(
        reaction_dbz(dbz, 0.0, 7.0, min_dbz=MIN_DBZ),
        dbz,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert bool(torch.isfinite(reaction_dbz(dbz, KAPPA, 7.0, min_dbz=MIN_DBZ)).all())


def test_large_finite_dbz_stays_finite() -> None:
    dbz = torch.tensor([1000.0], dtype=torch.float64)
    actual = reaction_dbz(dbz, KAPPA, 7.0, min_dbz=MIN_DBZ)
    expected = MIN_DBZ + math.exp(-KAPPA * 7.0) * (dbz - MIN_DBZ)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)
    assert bool(torch.isfinite(actual).all())


def test_reaction_has_finite_origin_jvp_vjp_and_gradgrad() -> None:
    horizon = 5.0
    point = torch.tensor(MIN_DBZ, dtype=torch.float64)
    tangent = torch.tensor(1.7, dtype=torch.float64)
    expected_slope = math.exp(-KAPPA * horizon)
    function = lambda value: reaction_dbz(
        value,
        KAPPA,
        horizon,
        min_dbz=MIN_DBZ,
    )

    output, jvp = torch.func.jvp(function, (point,), (tangent,))
    _, pullback = torch.func.vjp(function, point)
    vjp = pullback(torch.tensor(2.3, dtype=torch.float64))[0]
    torch.testing.assert_close(output, point, rtol=0.0, atol=0.0)
    torch.testing.assert_close(jvp, tangent * expected_slope, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        vjp,
        torch.tensor(2.3 * expected_slope, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    interior = torch.tensor(MIN_DBZ + 1.0e-4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(function, (interior,), eps=1.0e-6)
    first = torch.autograd.grad(function(interior), interior, create_graph=True)[0]
    if first.requires_grad:
        second = torch.autograd.grad(first, interior)[0]
        assert bool(torch.isfinite(second))
    else:
        # The equivalent dBZ form is affine in the input, so its exact
        # second derivative is zero and PyTorch may omit a grad function.
        assert first.item() == pytest.approx(expected_slope)
    assert bool(torch.isfinite(first))


def test_fit_actual_past_decay_and_eighteen_lead_closed_form_oracle() -> None:
    initial_excess = np.array([1.0, 5.0, 16.0, 32.0], dtype=np.float64)
    past_steps = np.array([-2.0, -1.0, 0.0])[:, None]
    observations_np = MIN_DBZ + initial_excess[None, :] * np.exp(-KAPPA * past_steps)
    observations = torch.from_numpy(observations_np)

    fitted, forecasts = forecast_from_observations(
        observations,
        min_dbz=MIN_DBZ,
        lead_count=18,
    )
    assert abs(float(fitted) - KAPPA) < 1.0e-12
    truth_np = np.stack([
        MIN_DBZ + initial_excess * np.exp(-KAPPA * lead)
        for lead in range(1, 19)
    ])
    expected_np = np.stack([
        _oracle_reaction(observations_np[-1], KAPPA, lead)
        for lead in range(1, 19)
    ])
    np.testing.assert_allclose(forecasts.detach().numpy(), expected_np, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(forecasts.detach().numpy(), truth_np, rtol=0.0, atol=3.0e-14)
    assert float(np.max(np.abs(forecasts.detach().numpy() - truth_np))) < 3.0e-14


def test_kappa_zero_is_identifiable_from_constant_positive_history() -> None:
    observations = torch.full((3, 2, 2), MIN_DBZ + 3.0, dtype=torch.float64)
    fitted = estimate_kappa_from_observations(observations, min_dbz=MIN_DBZ)
    torch.testing.assert_close(fitted, torch.tensor(0.0, dtype=torch.float64), atol=0.0, rtol=0.0)


def test_float32_fit_ignores_unresolved_floor_excess() -> None:
    steps = np.array([-2.0, -1.0, 0.0])[:, None]
    excess = np.array([1.0e-6, 3.0], dtype=np.float64)
    observations = torch.tensor(
        MIN_DBZ + excess[None, :] * np.exp(-KAPPA * steps),
        dtype=torch.float32,
    )
    fitted = estimate_kappa_from_observations(observations, min_dbz=MIN_DBZ)
    torch.testing.assert_close(
        fitted,
        torch.tensor(KAPPA, dtype=torch.float32),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_unsupported_zero_missing_positive_growth_and_nonconstant_regimes_reject() -> None:
    zero = torch.full((3, 2), MIN_DBZ, dtype=torch.float64)
    with pytest.raises(ValueError, match="zero reaction signal"):
        estimate_kappa_from_observations(zero, min_dbz=MIN_DBZ)

    missing = torch.full((3, 2), MIN_DBZ + 2.0, dtype=torch.float64)
    missing[1, 0] = torch.nan
    with pytest.raises(ValueError, match="missing values"):
        estimate_kappa_from_observations(missing, min_dbz=MIN_DBZ)

    positive_growth = MIN_DBZ + torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    with pytest.raises(ValueError, match="positive growth"):
        estimate_kappa_from_observations(positive_growth, min_dbz=MIN_DBZ)
    with pytest.raises(ValueError, match="positive growth"):
        reaction_dbz(torch.tensor([MIN_DBZ + 1.0]), -0.01, 1.0, min_dbz=MIN_DBZ)

    nonconstant = MIN_DBZ + torch.tensor(
        [[1.0, 1.0], [math.exp(-0.04), math.exp(-0.08)], [math.exp(-0.08), math.exp(-0.16)]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="constant-kappa"):
        estimate_kappa_from_observations(nonconstant, min_dbz=MIN_DBZ)
