"""Fixed QC-rejected point rows remain distinct from genuinely missing rows."""

from dataclasses import replace

import pytest
import torch

from examples.weather_scenarios.fv_point_research_case import make_case


@pytest.fixture(scope="module")
def case():
    return make_case()


def _prepared(problem):
    status = torch.zeros((3, 4), dtype=torch.uint8)
    status[0, 1] = 2
    status[1, 2] = 2
    status[2, 3] = 2
    values = problem.observation_dbz.clone()
    values[status == 2] = problem.frozen.nowcast_config.min_dbz
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 2] = correlation[2, 0] = 0.3
    correlation[1, 3] = correlation[3, 1] = -0.2
    qc = replace(problem, observation_dbz=values,
                 observation_status=status, observation_correlation=correlation)
    missing = replace(qc, observation_status=torch.where(
        status == 2, torch.ones_like(status), status
    ))
    return qc, missing, status


def test_qc_rows_are_inactive_but_have_a_distinct_fixed_identity(case):
    problem, control, parameters = case
    qc, missing, status = _prepared(problem)
    assert qc.identity["fixed_problem_sha256"] != missing.identity["fixed_problem_sha256"]
    assert "QC-rejected" in qc.support["observation_masks"]

    for evaluate in ("objective", "forecast", "score"):
        actual = getattr(qc, evaluate)(control, parameters)
        expected = getattr(missing, evaluate)(control, parameters)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    changed = parameters.clone()
    changed[:-1].reshape_as(status)[status == 2] += 100.0
    torch.testing.assert_close(qc.objective(control, changed),
                               qc.objective(control, parameters), rtol=0, atol=0)
    torch.testing.assert_close(qc.forecast(control, changed),
                               qc.forecast(control, parameters), rtol=0, atol=0)
    gradient = torch.func.grad(qc.objective, argnums=1)(control, parameters)
    assert torch.count_nonzero(gradient[:-1].reshape_as(status)[status == 2]) == 0
    direction = torch.zeros_like(parameters)
    direction[:-1].reshape_as(status)[status == 2] = 1.0
    mixed = torch.func.jvp(
        lambda p: torch.func.grad(qc.objective, argnums=0)(control, p),
        (parameters,), (direction,),
    )[1]
    assert torch.count_nonzero(mixed) == 0


def test_qc_requires_canonical_prepared_fill_and_explicit_status(case):
    problem, _control, _parameters = case
    qc, _missing, status = _prepared(problem)
    raw_values = qc.observation_dbz.clone()
    raw_values[0, 1] = problem.observation_dbz[0, 1]
    with pytest.raises(ValueError, match="canonical"):
        replace(qc, observation_dbz=raw_values)

    zero_quality = qc.quality_weight.clone()
    zero_quality[0, 1] = 0
    with pytest.raises(ValueError, match="std/quality"):
        replace(qc, quality_weight=zero_quality)

    censored = status.clone()
    censored[0, 1] = 3
    with pytest.raises(ValueError, match="status"):
        replace(qc, observation_status=censored)

    no_detection = status.clone()
    no_detection[1] = 2
    with pytest.raises(ValueError, match="at least one detected"):
        replace(qc, observation_status=no_detection)

    status[0, 0] = 2
    with pytest.raises(ValueError, match="status changed"):
        qc.objective(_control, _parameters)
