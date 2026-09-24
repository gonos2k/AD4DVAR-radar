"""Fixed-mask partial-observation contract for the small FV research binding."""
from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import pytest
import torch

from advar import variational as v
from advar.fv_research_problem import FVResearchProblem
from advar.physics import echo_to_dbz
from advar.transport import BoundarySchedule
from examples.weather_scenarios import fv_minmod_inverse_probe as probe
from examples.weather_scenarios import fv_multilead_research_case as multilead


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
MISSING = ((1, 1, 2), (2, 2, 3))


def _problem(*, missing=MISSING, common_bias_std=0.25, qc_reject=None,
             censor=None, zero_quality=None):
    observations, frozen, boundary, boundary_support = probe.make_spatial_case()
    boundary_support = cast(BoundarySchedule, boundary_support)
    frames = observations.dbz.clone()
    for index in missing:
        frames[index] = torch.nan
    qc_mask = torch.ones_like(frames, dtype=torch.bool)
    if qc_reject is not None:
        qc_mask[qc_reject] = False
    if censor is not None:
        frames[censor] = frozen.analysis_config.detection_limit_dbz - 1.0
    quality = torch.ones_like(frames)
    if zero_quality is not None:
        quality[zero_quality] = 0.0

    analysis_config = replace(
        frozen.analysis_config,
        observation_common_bias_std_dbz=common_bias_std,
    )
    prepared, partial_frozen = v.prepare_analysis(
        frames,
        nowcast_config=frozen.nowcast_config,
        analysis_config=analysis_config,
        observation_std_dbz=0.1,
        quality_weight=quality,
        qc_mask=qc_mask,
        fv_transport=frozen.fv_transport,
    )

    report = json.loads(
        (EVIDENCE / "minmod_middle_time_bias_final.json").read_text()
    )
    control = torch.tensor(report["nominal_control"], dtype=torch.float64)
    pattern = torch.linspace(-0.2, 0.3, 20, dtype=torch.float64).reshape(4, 5)
    parameters = torch.cat((prepared.dbz.flatten(), prepared.dbz.new_tensor([0.02])))
    initial = replace(
        partial_frozen,
        initial_background_dbz=prepared.dbz[0] + parameters[-1] * pattern,
    )
    assert initial.fv_transport is not None
    forecast = v.forecast_fv_analysis(
        control,
        initial,
        leads=1,
        boundary_start_interval=2,
        boundary_echo=boundary,
        boundary_support=boundary_support,
    ).frames_linear[-1]
    verification = (
        echo_to_dbz(forecast, min_dbz=frozen.nowcast_config.min_dbz)
        + 0.1 * pattern
    ).detach()
    return (
        FVResearchProblem(
            prepared,
            partial_frozen,
            boundary,
            boundary_support,
            pattern,
            verification,
            probe.inspect_branches,
        ),
        control,
        parameters,
    )


def _flat_index(index, shape):
    return index[0] * shape[1] * shape[2] + index[1] * shape[2] + index[2]


def test_missing_middle_and_last_observations_are_masked_in_objective_and_whitener():
    problem, control, parameters = _problem()
    observations = problem.observations
    assert problem.layout["controls"] == 26
    assert observations.valid_mask[0].all()
    assert observations.detected_mask[0].all()
    for index in MISSING:
        assert observations.missing_mask[index]
        assert not observations.valid_mask[index]
        assert not observations.detected_mask[index]
    mode = problem.frozen.observation_whitener.mode
    assert mode is not None
    for index in MISSING:
        assert mode[index] == 0

    baseline_objective = problem.objective(control, parameters)
    baseline_score = problem.score(control, parameters)
    baseline_forecast = problem.forecast(control, parameters)
    changed = parameters.clone()
    for index in MISSING:
        changed[_flat_index(index, observations.dbz.shape)] = 100.0

    torch.testing.assert_close(problem.objective(control, changed), baseline_objective,
                               rtol=0, atol=0)
    torch.testing.assert_close(problem.score(control, changed), baseline_score,
                               rtol=0, atol=0)
    torch.testing.assert_close(problem.forecast(control, changed), baseline_forecast,
                               rtol=0, atol=0)
    problem.branch_check(control, changed)

    objective_gradient = torch.func.grad(problem.objective, argnums=1)(control, changed)
    score_gradient = torch.func.grad(problem.score, argnums=1)(control, changed)
    missing_direction = torch.zeros_like(changed)
    for index in MISSING:
        flat = _flat_index(index, observations.dbz.shape)
        assert objective_gradient[flat] == 0
        assert score_gradient[flat] == 0
        missing_direction[flat] = 1
    mixed = torch.func.jvp(
        lambda values: torch.func.grad(problem.objective, argnums=0)(control, values),
        (changed,),
        (missing_direction,),
    )[1]
    assert torch.equal(mixed, torch.zeros_like(mixed))
    valid_index = _flat_index((1, 0, 0), observations.dbz.shape)
    assert objective_gradient[valid_index] != 0
    assert objective_gradient[valid_index].abs() > 1e-8
    torch.testing.assert_close(
        objective_gradient[valid_index],
        torch.func.grad(problem.objective, argnums=1)(control, parameters)[valid_index],
        rtol=0,
        atol=0,
    )


def test_partial_observations_keep_two_lead_time_and_branch_contract():
    _, two, control, _ = multilead.make_case()
    frames = two.observations.dbz.clone()
    frames[MISSING[0]] = torch.nan
    prepared, frozen = v.prepare_analysis(
        frames,
        nowcast_config=two.frozen.nowcast_config,
        analysis_config=two.frozen.analysis_config,
        observation_std_dbz=0.1,
        fv_transport=two.frozen.fv_transport,
    )
    problem = replace(two, observations=prepared, frozen=frozen)
    parameters = torch.cat((prepared.dbz.flatten(), prepared.dbz.new_tensor([0.02])))
    assert problem.forecast(control, parameters).shape == (2, 4, 5)
    assert problem.branch_check(control, parameters)[0]["euler_stages"] == 72
    assert torch.isfinite(problem.objective(control, parameters))
    assert torch.isfinite(problem.score(control, parameters))


@pytest.mark.parametrize(
    ("invalid_kind", "error"),
    [
        ("first_frame_missing", "fully known initial active support"),
        ("qc_rejected", "research minmod"),
        ("censored", "research minmod"),
        ("zero_quality", "research minmod"),
    ],
)
def test_research_binding_rejects_invalid_data_outside_genuine_later_missing(
    invalid_kind, error,
):
    kwargs = {}
    if invalid_kind == "first_frame_missing":
        kwargs["missing"] = ((0, 1, 2),)
    elif invalid_kind == "qc_rejected":
        kwargs["missing"] = ()
        kwargs["qc_reject"] = (1, 1, 3)
    elif invalid_kind == "censored":
        kwargs["missing"] = ()
        kwargs["censor"] = (1, 1, 3)
    else:
        kwargs["missing"] = ()
        kwargs["zero_quality"] = (1, 1, 3)
    with pytest.raises(ValueError, match=error):
        _problem(**kwargs)
