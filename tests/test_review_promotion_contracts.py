from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from itertools import pairwise

import pytest
import torch

import advar.promotion as promotion_module
from advar._digest import json_digest

DIGEST = "1" * 64
TIMES = (
    "2026-08-09T00:00:00Z",
    "2026-08-09T00:10:00Z",
    "2026-08-09T00:20:00Z",
)
SUBSECOND_TRACK_TIMES = (
    "2026-08-09T00:00:00Z",
    "2026-08-09T00:00:00.100Z",
    "2026-08-09T00:00:00.200Z",
)


def _input_plan(**overrides: object) -> promotion_module.NeuralPriorInputPlan:
    values: dict[str, object] = {
        "valid_times": TIMES,
        "grid_contract_digest": DIGEST,
        "radar_product_digest": "2" * 64,
        "qc_pipeline_digest": "3" * 64,
        "background_cycle_rule_digest": "4" * 64,
        "mask_policy_digest": "5" * 64,
        "observation_valid_time": TIMES[-1],
        "input_available_time": "2026-08-09T00:25:00Z",
        "decision_deadline": "2026-08-09T00:30:00Z",
        "publication_time": "2026-08-09T00:40:00Z",
    }
    values.update(overrides)
    return promotion_module.NeuralPriorInputPlan(**values)


def test_current_input_plan_contract_rejects_v1_and_arbitrary_tags() -> None:
    plan = _input_plan()
    for contract in ("neural-prior-input-plan-v1", "arbitrary-input-plan"):
        with pytest.raises(ValueError, match="unsupported neural-prior input plan"):
            replace(plan, contract=contract)


@pytest.mark.parametrize(
    "valid_times",
    [
        TIMES[:2],
        (*TIMES, "2026-08-09T00:30:00Z"),
        (TIMES[0], TIMES[0], TIMES[2]),
        (TIMES[1], TIMES[0], TIMES[2]),
    ],
)
def test_current_input_plan_requires_three_strictly_ordered_times(
    valid_times: tuple[str, ...],
) -> None:
    plan = _input_plan()
    with pytest.raises(ValueError, match="three increasing times"):
        replace(plan, valid_times=valid_times)


def _issuance_plan(
    *, radar_source_kind: str = "single_site"
) -> promotion_module.OperationalIssuanceDomainPlan:
    return promotion_module.OperationalIssuanceDomainPlan(
        case_id="case-1",
        grid_contract_digest=DIGEST,
        radar_source_contract_digest="2" * 64,
        lead_minutes=(10,),
        publication_policy_digest="3" * 64,
        source_coverage_policy_digest="4" * 64,
        permanent_exclusion_policy_digest="5" * 64,
        publication_eligible_mask_digest="6" * 64,
        source_coverage_mask_digest="7" * 64,
        permanent_exclusion_mask_digest="8" * 64,
        radar_source_kind=radar_source_kind,
    )


def test_operational_issuance_domain_rejects_bogus_source_kind() -> None:
    with pytest.raises(ValueError, match="operational issuance-domain plan"):
        _issuance_plan(radar_source_kind="bogus")


def _track(
    *,
    centroids: tuple[tuple[float, float], ...],
    timestamps: tuple[str, ...] = (
        "2026-08-09T00:00:00Z",
        "2026-08-09T00:00:01Z",
        "2026-08-09T00:00:02Z",
    ),
) -> promotion_module.PhysicalEventTrackArtifact:
    return promotion_module.PhysicalEventTrackArtifact(
        timestamps=timestamps,
        centroid_xy_m=centroids,
        object_mask_digests=("a" * 64,) * 3,
        source_radar_ids=("radar-1",) * 3,
        association_edge_digests=("b" * 64,) * 2,
        spatial_reference_digest="c" * 64,
    )


def test_physical_event_track_allows_valid_constant_velocity() -> None:
    track = _track(centroids=((0.0, 0.0), (0.1, 0.0), (0.2, 0.0)))
    promotion_module.validate_physical_event_track_artifact(track)


@pytest.mark.parametrize(
    "centroids",
    [
        ((0.0, 0.0), (0.01, 0.0), (0.02, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (0.001, 0.0)),
    ],
    ids=("constant-velocity", "acceleration-below-limit"),
)
def test_physical_event_track_preserves_subsecond_intervals(
    centroids: tuple[tuple[float, float], ...],
) -> None:
    track = _track(centroids=centroids, timestamps=SUBSECOND_TRACK_TIMES)
    assert track.timestamps == (
        "2026-08-09T00:00:00Z",
        "2026-08-09T00:00:00.100000Z",
        "2026-08-09T00:00:00.200000Z",
    )
    parsed = tuple(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in track.timestamps
    )
    assert tuple(
        (second - first).total_seconds()
        for first, second in pairwise(parsed)
    ) == pytest.approx((0.1, 0.1))
    promotion_module.validate_physical_event_track_artifact(track)


def test_physical_event_track_rejects_subsecond_acceleration_in_constructor() -> None:
    # Velocities 0 and 0.1 m/s are 0.1 s apart: a=1, not the old a=0.1.
    with pytest.raises(ValueError, match="acceleration"):
        _track(
            centroids=((0.0, 0.0), (0.0, 0.0), (0.01, 0.0)),
            timestamps=SUBSECOND_TRACK_TIMES,
        )


def test_physical_event_track_validator_rejects_rehashed_acceleration_tamper() -> None:
    track = _track(
        centroids=((0.0, 0.0), (0.01, 0.0), (0.02, 0.0)),
        timestamps=SUBSECOND_TRACK_TIMES,
    )
    object.__setattr__(
        track,
        "centroid_xy_m",
        ((0.0, 0.0), (0.0, 0.0), (0.01, 0.0)),
    )
    object.__setattr__(track, "artifact_digest", json_digest(track.payload))
    with pytest.raises(ValueError, match="physical event track artifact is invalid"):
        promotion_module.validate_physical_event_track_artifact(track)


def _range_band_evaluation(
    *,
    domain_count: int = 1,
    parent_count: int = 1,
    candidate_count: int = 1,
    withdrawn_count: int = 0,
    newly_issued_count: int = 0,
    parent_confidence_area: float | None = None,
    candidate_confidence_area: float | None = None,
    metric_area: float = 1.0,
) -> promotion_module.RangeBandEvaluation:
    domain_area = float(domain_count)
    values: dict[str, object] = {
        "range_regime": "low",
        "range_band_mask_digest": "a" * 64,
        "range_geometry_contract_digest": "b" * 64,
        "metric_change": torch.tensor([[0.0]]),
        "end_to_end_metric_change": torch.tensor([[0.0]]),
        "metric_available": torch.tensor([[True]]),
        "candidate_uncertainty_component_scores": (("support", 0.1),),
        "parent_uncertainty_component_scores": (("support", 0.2),),
        "uncertainty_component_differences": (("support", -0.1),),
        "uncertainty_component_sample_counts": (("support", 2),),
        "evaluated_area_km2": domain_area,
        "metric_valid_area_km2_by_lead": (domain_area,),
        "metric_valid_area_km2": torch.tensor([[metric_area]]),
        "issuance_domain_digest": "c" * 64,
        "issuance_domain_cell_count_by_lead": (domain_count,),
        "issuance_domain_area_km2_by_lead": (domain_area,),
        "parent_issued_count_by_lead": (parent_count,),
        "candidate_issued_count_by_lead": (candidate_count,),
        "withdrawn_count_by_lead": (withdrawn_count,),
        "newly_issued_count_by_lead": (newly_issued_count,),
        "parent_fallback_count_by_lead": (0,),
        "candidate_fallback_count_by_lead": (0,),
        "parent_confidence_weighted_issued_area_by_lead": (
            domain_area if parent_confidence_area is None else parent_confidence_area,
        ),
        "candidate_confidence_weighted_issued_area_by_lead": (
            domain_area
            if candidate_confidence_area is None
            else candidate_confidence_area,
        ),
        "withdrawn_fraction_by_lead": torch.tensor(
            [withdrawn_count / max(1, parent_count)]
        ),
        "newly_issued_fraction_by_lead": torch.tensor(
            [newly_issued_count / domain_count]
        ),
        "background_fallback_increase_by_lead": torch.tensor([0.0]),
        "confidence_weighted_coverage_change_by_lead": torch.tensor([0.0]),
        "probability_valid_area_km2": domain_area,
        "state_valid_area_km2": domain_area,
        "echo_pixel_count": 0,
        "clear_pixel_count": 2,
        "echo_object_count": 0,
        "state_echo_pixel_count": 0,
        "state_clear_pixel_count": 2,
        "state_echo_object_count": 0,
    }
    return promotion_module.RangeBandEvaluation(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain_count": 1, "parent_count": 2, "candidate_count": 2},
        {"domain_count": 1, "parent_count": 1, "candidate_count": 0, "withdrawn_count": 0},
        {"domain_count": 1, "parent_count": 0, "candidate_count": 1, "newly_issued_count": 0},
        {"domain_count": 1, "parent_count": 1, "candidate_count": 1, "parent_confidence_area": 2.0},
        {"domain_count": 1, "parent_count": 1, "candidate_count": 1, "candidate_confidence_area": 2.0},
        {"domain_count": 1, "parent_count": 1, "candidate_count": 1, "metric_area": 2.0},
    ],
)
def test_range_band_evaluation_rejects_impossible_cross_field_evidence(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _range_band_evaluation(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain_count": 1, "parent_count": 1, "candidate_count": 1},
        {"domain_count": 1, "parent_count": 1, "candidate_count": 0, "withdrawn_count": 1},
        {"domain_count": 1, "parent_count": 0, "candidate_count": 1, "newly_issued_count": 1},
        {
            "domain_count": 2,
            "parent_count": 1,
            "candidate_count": 1,
            "withdrawn_count": 1,
            "newly_issued_count": 1,
        },
    ],
)
def test_range_band_evaluation_preserves_valid_set_difference_transitions(
    kwargs: dict[str, object],
) -> None:
    evaluation = _range_band_evaluation(**kwargs)
    assert evaluation.evaluation_digest == json_digest(evaluation.payload)
