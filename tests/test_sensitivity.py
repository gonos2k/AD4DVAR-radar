from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import stat
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar._digest import tensor_digest  # noqa: E402
import advar.sensitivity as sensitivity_module  # noqa: E402
from advar.linearization_artifact import (  # noqa: E402
    load_p1_linearization,
    save_p1_linearization,
)
from advar.ledger import EpisodeLedger  # noqa: E402
from advar.matrix_free import PCGResult  # noqa: E402
from advar.nowcast import (  # noqa: E402
    _estimate_source_tendencies,
    _forecast_linear_at_step_core,
    _forecast_run_identity_digest,
    DataStatus,
    ForecastMetadata,
    ForecastResult,
    ForecastRunContract,
    NowcastConfig,
    RadarGridTimeContract,
    RadarState,
    StatePathProvenance,
    TendencySource,
    TendencyPairSelection,
    forecast_from_state,
    forecast_linear_at_step,
    forecast_linear_from_state,
    nowcast,
    state_metadata_digest,
)
from advar.physics import (  # noqa: E402
    dbz_to_echo,
    echo_to_dbz,
    freeze_remap_cell,
)
from advar.sensitivity import (  # noqa: E402
    AutomatedLearningPolicy,
    MetricTaylorThreshold,
    _baseline_branch_is_stable,
    _gauss_newton_curvature_diagnostics,
    _apply_output_cap,
    _metric_evidence_ratios,
    _metric_domain_weight,
    _NormalProductBudget,
    _p0_tendency_branch_signature,
    _remap_fraction_margin,
    SensitivityConfig,
    VerificationBundle,
    VariationalAdjointConfig,
    VariationalObservationPerturbation,
    compute_sensitivity_snapshot,
    compute_variational_fso,
    compute_variational_fsoi,
    compute_variational_fsoi_for_learning,
    score_candidate_perturbations,
    validate_top_k_learning_impacts,
    extract_context_features,
    forecast_metric,
    validate_variational_fso,
    validate_variational_fsoi,
    validate_variational_learning_impact,
    variational_fso_digest,
)
from advar.variational import (  # noqa: E402
    _analysis_trajectory,
    AnalysisConfig,
    residual_vector,
    variational_nowcast,
)


def dbz_to_linear(dbz: torch.Tensor, config: NowcastConfig) -> torch.Tensor:
    return dbz_to_echo(
        dbz,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )


def linear_to_dbz(echo: torch.Tensor, config: NowcastConfig) -> torch.Tensor:
    return echo_to_dbz(
        echo,
        min_dbz=config.min_dbz,
        max_dbz=config.max_dbz,
    )


def metadata_for(
    state: RadarState,
    *,
    pair_motion: torch.Tensor | None = None,
    pair_growth: torch.Tensor | None = None,
) -> ForecastMetadata:
    pair_motion = (
        torch.stack((state.displacement_yx, state.displacement_yx))
        if pair_motion is None
        else pair_motion
    )
    pair_growth = (
        torch.stack(
            (state.log_growth_per_step, state.log_growth_per_step)
        )
        if pair_growth is None
        else pair_growth
    )
    return ForecastMetadata(
        data_status=DataStatus.OBSERVED,
        coverage_by_frame=torch.ones(
            3,
            dtype=state.echo_linear.dtype,
            device=state.echo_linear.device,
        ),
        background_used=False,
        background_contribution_fraction=0.0,
        background_age_minutes=None,
        source_support=torch.ones_like(state.echo_linear),
        observation_source_support=torch.ones_like(state.echo_linear),
        background_source_support=torch.zeros_like(state.echo_linear),
        path_verified_source_support=torch.ones_like(state.echo_linear),
        verified_source_support=torch.ones_like(state.echo_linear),
        local_motion_verified_support=torch.ones_like(state.echo_linear),
        local_growth_verified_support=torch.ones_like(state.echo_linear),
        local_dynamics_verified_support=torch.ones_like(state.echo_linear),
        observation_verified_source_support=torch.ones_like(
            state.echo_linear
        ),
        background_verified_source_support=torch.zeros_like(
            state.echo_linear
        ),
        motion_disagreement_px=torch.linalg.vector_norm(
            pair_motion[1] - pair_motion[0]
        ),
        motion_disagreement_mps=state.echo_linear.new_full((), torch.nan),
        growth_disagreement=torch.abs(pair_growth[1] - pair_growth[0]),
        maximum_growth_saturation_excess=state.echo_linear.new_zeros(()),
        posterior_velocity_uncertainty_mps=state.echo_linear.new_full(
            (), torch.nan
        ),
        posterior_log_growth_uncertainty_per_step=(
            state.echo_linear.new_full((), torch.nan)
        ),
        p1_velocity_saturation_uncertainty_mps=(
            state.echo_linear.new_full((), torch.nan)
        ),
        p1_log_growth_saturation_uncertainty_per_step=(
            state.echo_linear.new_full((), torch.nan)
        ),
        minimum_phase_correlation_psr=state.echo_linear.new_tensor(10.0),
        tendency_pair_count=2,
        tendency_source=TendencySource.OBSERVATION,
        state_path_source=TendencySource.OBSERVATION,
        state_path_age_minutes=0.0,
        observation_path=StatePathProvenance(age_minutes=0.0),
        minimum_growth_overlap_support=float(state.echo_linear.numel()),
    )


def result_for(
    state: RadarState,
    config: NowcastConfig,
    *,
    frames: torch.Tensor | None = None,
    accepted_mask: torch.Tensor | None = None,
    background: torch.Tensor | None = None,
    pair_motion: torch.Tensor | None = None,
    pair_growth: torch.Tensor | None = None,
    grid_time_contract: RadarGridTimeContract | None = None,
) -> ForecastResult:
    if frames is None:
        latest = linear_to_dbz(state.echo_linear, config)
        frames = torch.stack((latest, latest, latest))
    if accepted_mask is None:
        accepted_mask = torch.isfinite(frames)
    metadata = metadata_for(
        state,
        pair_motion=pair_motion,
        pair_growth=pair_growth,
    )
    if grid_time_contract is not None:
        motions = (
            torch.stack((state.displacement_yx, state.displacement_yx))
            if pair_motion is None
            else pair_motion
        )
        metadata = replace(
            metadata,
            motion_disagreement_mps=torch.linalg.vector_norm(
                grid_time_contract.projected_displacement_xy(
                    motions[1] - motions[0]
                )
            )
            / (config.interval_minutes * 60.0),
            minimum_growth_overlap_area_km2=(
                metadata.minimum_growth_overlap_support
                * grid_time_contract.cell_area_m2
                / 1.0e6
            ),
        )
    return forecast_from_state(
        state,
        metadata,
        config,
        run=ForecastRunContract.from_inputs(
            config,
            frames,
            accepted_mask,
            background,
            0.0 if background is not None else None,
            grid_time_contract=grid_time_contract,
        ),
    )


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nowcast_config = NowcastConfig()
        cls.sensitivity_config = SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(10,),
            tile_size=6,
        )
        cls.height, cls.width = 13, 17
        y, x = torch.meshgrid(
            torch.arange(cls.height, dtype=torch.float64),
            torch.arange(cls.width, dtype=torch.float64),
            indexing="ij",
        )
        latest = 18.0 + 26.0 * torch.exp(
            -((y - 6.0) ** 2 + (x - 8.0) ** 2) / 20.0
        )
        cls.frames = torch.stack((latest - 2.0, latest - 1.0, latest))
        cls.frames[2, 0, 0] = float("nan")
        cls.frames[2, 0, 1] = cls.nowcast_config.min_dbz
        cls.frames[2, 0, 2] = cls.nowcast_config.max_dbz
        cls.qc_mask = torch.ones_like(cls.frames, dtype=torch.bool)
        cls.qc_mask[2, 0, 3] = False

        cls.state = RadarState(
            echo_linear=dbz_to_linear(cls.frames[2], cls.nowcast_config),
            displacement_yx=torch.tensor([0.35, -0.25], dtype=torch.float64),
            log_growth_per_step=torch.tensor(0.015, dtype=torch.float64),
        )
        clean_frames = torch.nan_to_num(
            cls.frames,
            nan=cls.nowcast_config.min_dbz,
            posinf=cls.nowcast_config.max_dbz,
            neginf=cls.nowcast_config.min_dbz,
        ).clamp(
            cls.nowcast_config.min_dbz,
            cls.nowcast_config.max_dbz,
        )
        cls.background = (
            clean_frames - 0.3 * torch.cos(x / 4.0)[None]
        ).clamp(
            cls.nowcast_config.min_dbz,
            cls.nowcast_config.max_dbz,
        )
        cls.result = result_for(
            cls.state,
            cls.nowcast_config,
            frames=cls.frames,
            accepted_mask=cls.qc_mask & torch.isfinite(cls.frames),
            background=cls.background,
            pair_motion=torch.tensor(
                [[0.32, -0.20], [0.37, -0.28]],
                dtype=torch.float64,
            ),
            pair_growth=torch.tensor(
                [0.01, 0.02],
                dtype=torch.float64,
            ),
        )
        cls.verification = (
            cls.result.forecast_dbz + 0.4 * torch.sin(y / 3.0)[None]
        )
        common = {
            "sensitivity_config": cls.sensitivity_config,
            "observation_std_dbz": 2.0,
        }
        cls.snapshot = compute_sensitivity_snapshot(
            cls.frames[2],
            cls.result,
            cls.verification,
            latest_background_dbz=cls.background[2],
            **common,
        )
        result_without_background = result_for(
            cls.state,
            cls.nowcast_config,
            frames=cls.frames,
            accepted_mask=cls.qc_mask & torch.isfinite(cls.frames),
            pair_motion=torch.tensor(
                [[0.32, -0.20], [0.37, -0.28]],
                dtype=torch.float64,
            ),
            pair_growth=torch.tensor(
                [0.01, 0.02],
                dtype=torch.float64,
            ),
        )
        cls.snapshot_without_background = compute_sensitivity_snapshot(
            cls.frames[2],
            result_without_background,
            cls.verification,
            **common,
        )

    def test_forecast_linear_core_matches_step_and_dbz_paths(self) -> None:
        config = NowcastConfig(horizon_minutes=30)
        y, x = torch.meshgrid(
            torch.arange(12, dtype=torch.float64),
            torch.arange(14, dtype=torch.float64),
            indexing="ij",
        )
        latest_dbz = 15.0 + 15.0 * torch.exp(
            -((y - 5.0) ** 2 + (x - 7.0) ** 2) / 18.0
        )
        state = RadarState(
            echo_linear=dbz_to_linear(latest_dbz, config),
            displacement_yx=torch.tensor([0.2, -0.15], dtype=torch.float64),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        )

        linear = forecast_linear_from_state(state, config)
        by_step = torch.stack(
            [
                forecast_linear_at_step(state, step, config)
                for step in range(1, config.forecast_steps + 1)
            ]
        )

        torch.testing.assert_close(linear, by_step)
        frames = torch.stack((latest_dbz, latest_dbz, latest_dbz))
        result = result_for(state, config, frames=frames)
        expected_dbz = linear_to_dbz(linear, config)
        self.assertTrue(
            torch.equal(
                torch.isfinite(result.forecast_dbz),
                result.valid_mask,
            )
        )
        torch.testing.assert_close(
            result.forecast_dbz[result.valid_mask],
            expected_dbz[result.valid_mask],
        )
        torch.testing.assert_close(
            dbz_to_linear(result.forecast_dbz, config)[result.valid_mask],
            linear[result.valid_mask],
        )

    def test_metric_domain_policies_freeze_distinct_weights(self) -> None:
        finite = torch.ones(
            (self.height, self.width),
            dtype=torch.bool,
        )
        finite[0, 0] = False
        issued = _metric_domain_weight(
            self.result,
            finite,
            0,
            "issued",
        )
        anchored = _metric_domain_weight(
            self.result,
            finite,
            0,
            "radar_dynamics_anchored",
        )
        confidence = _metric_domain_weight(
            self.result,
            finite,
            0,
            "confidence_weighted",
        )
        torch.testing.assert_close(
            issued,
            (finite & self.result.valid_mask[0]).to(issued),
        )
        torch.testing.assert_close(
            anchored,
            (
                finite
                & self.result.radar_dynamics_anchored_valid_mask[0]
            ).to(anchored),
        )
        torch.testing.assert_close(
            confidence,
            torch.where(
                finite & self.result.valid_mask[0],
                self.result.forecast_confidence[0],
                torch.zeros_like(confidence),
            ),
        )
        self.assertTrue(bool(torch.all(anchored <= issued)))
        self.assertTrue(bool(torch.all(confidence <= issued)))

        forecast = torch.tensor([[1.0, 4.0]], dtype=torch.float64)
        truth = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        weight = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        weighted_score = forecast_metric(
            "log_echo_mse",
            forecast,
            truth,
            weight,
            self.nowcast_config,
            replace(
                self.sensitivity_config,
                metric_domain="confidence_weighted",
            ),
        )
        torch.testing.assert_close(
            weighted_score,
            torch.zeros_like(weighted_score),
        )

    def test_confidence_weighted_fss_downweights_low_confidence_windows(
        self,
    ) -> None:
        forecast = dbz_to_linear(
            torch.tensor([[40.0, 40.0]], dtype=torch.float64),
            self.nowcast_config,
        )
        truth = dbz_to_linear(
            torch.tensor([[40.0, 0.0]], dtype=torch.float64),
            self.nowcast_config,
        )
        config = replace(
            self.sensitivity_config,
            metric_domain="confidence_weighted",
            soft_fss_window=1,
        )
        equal_weight = forecast_metric(
            "soft_fss_error_35",
            forecast,
            truth,
            torch.ones_like(forecast),
            self.nowcast_config,
            config,
        )
        low_bad_region_weight = forecast_metric(
            "soft_fss_error_35",
            forecast,
            truth,
            torch.tensor([[1.0, 1.0e-6]], dtype=torch.float64),
            self.nowcast_config,
            config,
        )

        self.assertLess(low_bad_region_weight, 1.0e-4 * equal_weight)

    def test_snapshot_shapes_and_m0_scope(self) -> None:
        snapshot = self.snapshot
        tile_rows = math.ceil(self.height / self.sensitivity_config.tile_size)
        tile_columns = math.ceil(
            self.width / self.sensitivity_config.tile_size
        )

        self.assertEqual(snapshot.lead_minutes, tuple(range(10, 181, 10)))
        self.assertIsNone(snapshot.grid_time_contract_digest)
        self.assertEqual(snapshot.full_map_lead_minutes, (10,))
        self.assertEqual(
            snapshot.context_features.shape,
            (len(snapshot.context_feature_names),),
        )
        self.assertIn("motion_pair_conflict", snapshot.context_feature_names)
        self.assertIn("growth_pair_conflict", snapshot.context_feature_names)
        self.assertIn("log_integrated_echo", snapshot.context_feature_names)
        self.assertNotIn("log_echo_mass", snapshot.context_feature_names)
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        for component in ("motion", "growth"):
            selection_values = tuple(
                float(
                    context[
                        f"{component}_pair_selection_{selection.value.lower()}"
                    ]
                )
                for selection in TendencyPairSelection
            )
            self.assertEqual(sum(selection_values), 1.0)
            self.assertEqual(
                context[f"{component}_pair_selection_blended"],
                1.0,
            )
        self.assertEqual(context["phase_correlation_psr_available"], 1.0)
        torch.testing.assert_close(
            context["log1p_minimum_phase_correlation_psr"],
            torch.log1p(self.result.metadata.minimum_phase_correlation_psr),
        )
        self.assertEqual(context["projected_velocity_available"], 0.0)
        for name in (
            "projected_velocity_x_mps",
            "projected_velocity_y_mps",
            "projected_speed_mps",
        ):
            self.assertEqual(context[name], 0.0)
        self.assertEqual(context["motion_disagreement_mps_available"], 0.0)
        self.assertEqual(context["motion_disagreement_mps"], 0.0)
        self.assertEqual(context["area_weighted_echo_available"], 0.0)
        self.assertEqual(
            context["log1p_linear_reflectivity_integral_km2"], 0.0
        )
        self.assertEqual(context["grid_spacing_available"], 0.0)
        self.assertEqual(context["grid_column_spacing_m"], 0.0)
        self.assertEqual(context["grid_row_spacing_m"], 0.0)
        self.assertEqual(snapshot.analysis_control.shape, (3,))
        self.assertEqual(snapshot.forecast_scores.shape, (18, 1))
        self.assertEqual(snapshot.metric_available.shape, (18, 1))
        self.assertEqual(snapshot.control_sensitivity.shape, (18, 1, 3))
        self.assertEqual(
            snapshot.forecast_sensitivity.shape,
            (1, 1, self.height, self.width),
        )
        self.assertEqual(
            snapshot.forecast_cap_active_mask.shape,
            (1, self.height, self.width),
        )
        self.assertEqual(
            snapshot.forecast_confidence.shape,
            (18, self.height, self.width),
        )
        self.assertEqual(snapshot.path_evidence_by_metric.shape, (18, 1))
        self.assertEqual(
            snapshot.observation_source_fraction_by_metric.shape,
            (18, 1),
        )
        self.assertEqual(
            snapshot.observation_verified_evidence_by_metric.shape,
            (18, 1),
        )
        self.assertEqual(
            snapshot.background_verified_evidence_by_metric.shape,
            (18, 1),
        )
        self.assertEqual(
            snapshot.direct.maps.shape,
            (1, 1, self.height, self.width),
        )
        self.assertEqual(
            snapshot.direct.norm.shape,
            (18, 1),
        )
        self.assertEqual(
            snapshot.direct.tile_norm.shape,
            (18, 1, tile_rows, tile_columns),
        )
        self.assertIsNotNone(snapshot.direct.impact)
        self.assertEqual(
            snapshot.direct.impact.shape,
            (18, 1),
        )
        self.assertEqual(snapshot.latest_sensitivity_mask.shape, (13, 17))
        self.assertEqual(
            snapshot.observation_innovation_mask.shape,
            (13, 17),
        )
        self.assertTrue(snapshot.whitened_tile_norm_available)
        self.assertEqual(
            snapshot.nowcast_config_digest,
            self.nowcast_config.digest,
        )
        self.assertEqual(
            snapshot.sensitivity_config_digest,
            self.sensitivity_config.digest,
        )
        self.assertEqual(
            snapshot.forecast_run_digest,
            self.result.forecast_run_digest,
        )

    def test_sensitivity_uses_the_forecast_run_config(self) -> None:
        config = NowcastConfig(
            horizon_minutes=20,
            growth_decay_minutes=120.0,
            max_dbz=60.0,
            min_publish_support=0.7,
        )
        frames = torch.full((3, 5, 6), 20.0, dtype=torch.float64)
        result = nowcast(frames, config)
        verification = result.forecast_dbz.clone()

        snapshot = compute_sensitivity_snapshot(
            frames[-1],
            result,
            verification,
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
            ),
        )

        self.assertEqual(snapshot.nowcast_config_digest, config.digest)
        torch.testing.assert_close(
            snapshot.forecast_scores,
            torch.zeros_like(snapshot.forecast_scores),
        )

    def test_sensitivity_rejects_mismatched_run_inputs(self) -> None:
        different_frame = self.frames[-1].clone()
        different_frame[6, 8] += 0.1
        with self.assertRaisesRegex(ValueError, "latest frame disagrees"):
            compute_sensitivity_snapshot(
                different_frame,
                self.result,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        different_background = self.background[-1].clone()
        different_background[6, 8] += 0.1
        with self.assertRaisesRegex(ValueError, "background disagrees"):
            compute_sensitivity_snapshot(
                self.frames[-1],
                self.result,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                latest_background_dbz=different_background,
            )

    def test_sensitivity_rejects_a_changed_acceptance_mask(self) -> None:
        changed_mask = self.result.run.latest_observation_mask
        changed_mask[0, 3] = True
        changed_run = replace(
            self.result.run,
            _latest_observation_mask=changed_mask,
        )

        with self.assertRaisesRegex(
            ValueError,
            "observation mask disagrees",
        ):
            compute_sensitivity_snapshot(
                self.frames[-1],
                replace(self.result, run=changed_run),
                self.verification,
                sensitivity_config=self.sensitivity_config,
                latest_background_dbz=self.background[-1],
            )

    def test_returned_acceptance_mask_is_an_independent_copy(self) -> None:
        changed_mask = self.result.run.latest_observation_mask
        changed_mask[:] = True

        snapshot = compute_sensitivity_snapshot(
            self.frames[-1],
            self.result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[-1],
        )

        self.assertFalse(bool(snapshot.latest_sensitivity_mask[0, 3]))

    def test_sensitivity_rejects_a_forecast_that_does_not_close(self) -> None:
        changed = self.result.forecast_dbz.clone()
        changed[0, 6, 8] += 0.1
        with self.assertRaisesRegex(
            ValueError,
            "disagrees with the issued forecast",
        ):
            compute_sensitivity_snapshot(
                self.frames[-1],
                replace(self.result, forecast_dbz=changed),
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

    def test_sensitivity_rejects_a_shrunken_issued_domain(self) -> None:
        changed = self.result.forecast_dbz.clone()
        changed[0, 6, 8] = float("nan")

        with self.assertRaisesRegex(
            ValueError,
            "disagrees with the issued forecast",
        ):
            compute_sensitivity_snapshot(
                self.frames[-1],
                replace(self.result, forecast_dbz=changed),
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

    def test_control_gradient_matches_centered_finite_difference(self) -> None:
        truth = dbz_to_linear(self.verification[0], self.nowcast_config)
        valid = torch.isfinite(self.verification[0])

        def score(control: torch.Tensor) -> torch.Tensor:
            candidate_state = RadarState(
                echo_linear=self.state.echo_linear,
                displacement_yx=control[:2],
                log_growth_per_step=control[2],
            )
            return forecast_metric(
                "log_echo_mse",
                forecast_linear_at_step(
                    candidate_state,
                    1,
                    self.nowcast_config,
                ),
                truth,
                valid,
                self.nowcast_config,
                self.sensitivity_config,
            )

        control = self.snapshot.analysis_control
        epsilon = 1.0e-5
        finite_difference = []
        for index in range(3):
            delta = torch.zeros_like(control)
            delta[index] = epsilon
            finite_difference.append(
                (score(control + delta) - score(control - delta))
                / (2.0 * epsilon)
            )

        torch.testing.assert_close(
            self.snapshot.control_sensitivity[0, 0],
            torch.stack(finite_difference),
            atol=1.0e-6,
            rtol=1.0e-5,
        )

    def test_latest_direct_dbz_gradient_obeys_frozen_active_set(self) -> None:
        mask = self.snapshot.latest_sensitivity_mask
        gradient = self.snapshot.direct.maps[0, 0]

        self.assertTrue(bool(torch.all(torch.isfinite(gradient))))
        self.assertFalse(bool(mask[0, 0]))  # non-finite
        self.assertFalse(bool(mask[0, 1]))  # dBZ floor
        self.assertFalse(bool(mask[0, 2]))  # dBZ cap
        self.assertFalse(bool(mask[0, 3]))  # rejected by QC
        self.assertTrue(bool(mask[1, 1]))
        self.assertTrue(
            torch.equal(
                gradient[~mask],
                torch.zeros_like(gradient[~mask]),
            )
        )
        self.assertGreater(
            int(torch.count_nonzero(gradient[mask])),
            0,
        )

        row, column = 6, 8
        epsilon = 1.0e-4
        clean_latest = torch.nan_to_num(
            self.frames[2],
            nan=self.nowcast_config.min_dbz,
        )

        def score(latest_dbz: torch.Tensor) -> torch.Tensor:
            candidate_echo = dbz_to_linear(
                latest_dbz,
                self.nowcast_config,
            )
            candidate_state = RadarState(
                echo_linear=torch.where(
                    mask,
                    candidate_echo,
                    self.state.echo_linear,
                ),
                displacement_yx=self.state.displacement_yx,
                log_growth_per_step=self.state.log_growth_per_step,
            )
            return forecast_metric(
                "log_echo_mse",
                forecast_linear_at_step(
                    candidate_state,
                    1,
                    self.nowcast_config,
                ),
                dbz_to_linear(
                    self.verification[0],
                    self.nowcast_config,
                ),
                torch.isfinite(self.verification[0]),
                self.nowcast_config,
                self.sensitivity_config,
            )

        perturbation = torch.zeros_like(clean_latest)
        perturbation[row, column] = epsilon
        finite_difference = (
            score(clean_latest + perturbation)
            - score(clean_latest - perturbation)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            gradient[row, column],
            finite_difference,
            atol=1.0e-8,
            rtol=1.0e-5,
        )

    def test_soft_fss_is_zero_for_identity_and_symmetric_for_a_miss(
        self,
    ) -> None:
        config = SensitivityConfig(
            metric_names=("soft_fss_error_35",),
            full_map_lead_minutes=(10,),
            soft_fss_window=5,
        )
        truth_dbz = torch.full((15, 15), 10.0, dtype=torch.float64)
        truth_dbz[5:10, 5:10] = 45.0
        miss_dbz = torch.full_like(truth_dbz, 10.0)
        miss_dbz[1:5, 1:5] = 45.0
        truth = dbz_to_linear(truth_dbz, self.nowcast_config)
        miss = dbz_to_linear(miss_dbz, self.nowcast_config)
        valid = torch.ones_like(truth, dtype=torch.bool)

        identical_score = forecast_metric(
            "soft_fss_error_35",
            truth,
            truth,
            valid,
            self.nowcast_config,
            config,
        )
        miss_score = forecast_metric(
            "soft_fss_error_35",
            miss,
            truth,
            valid,
            self.nowcast_config,
            config,
        )
        reverse_score = forecast_metric(
            "soft_fss_error_35",
            truth,
            miss,
            valid,
            self.nowcast_config,
            config,
        )

        torch.testing.assert_close(
            identical_score,
            torch.zeros_like(identical_score),
        )
        self.assertTrue(bool(torch.isfinite(miss_score)))
        self.assertGreater(float(miss_score), 0.0)
        self.assertLessEqual(float(miss_score), 1.0)
        torch.testing.assert_close(miss_score, reverse_score)

    def test_odd_grid_tile_impacts_close_to_total(self) -> None:
        self.assertNotEqual(
            self.height % self.sensitivity_config.tile_size,
            0,
        )
        self.assertNotEqual(
            self.width % self.sensitivity_config.tile_size,
            0,
        )
        tile_sum = self.snapshot.direct.tile_impact.sum(
            dim=(-1, -2)
        )

        self.assertTrue(bool(torch.all(torch.isfinite(tile_sum))))
        torch.testing.assert_close(
            tile_sum,
            self.snapshot.direct.impact,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    def test_missing_background_marks_impacts_unavailable(self) -> None:
        snapshot = self.snapshot_without_background

        self.assertFalse(snapshot.impact_available)
        self.assertFalse(snapshot.reward_available)
        self.assertIsNone(snapshot.observation_innovation_dbz)
        self.assertIsNone(snapshot.observation_innovation_mask)
        self.assertIsNone(snapshot.direct.impact)
        self.assertIsNone(snapshot.direct.tile_impact)
        self.assertIsNone(snapshot.direct.reward)

    def test_unlineaged_baseline_scores_are_rejected(self) -> None:
        baseline_scores = torch.ones_like(self.snapshot.forecast_scores)

        with self.assertRaisesRegex(
            ValueError,
            "verified lineage contract",
        ):
            compute_sensitivity_snapshot(
                self.frames[2],
                self.result,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                latest_background_dbz=self.background[2],
                observation_std_dbz=2.0,
                baseline_scores=baseline_scores,
            )

    def test_sensitivity_scores_the_issued_capped_forecast(self) -> None:
        config = self.nowcast_config
        frames = torch.full(
            (3, 8, 9),
            config.max_dbz - 0.5,
            dtype=torch.float64,
        )
        state = RadarState(
            echo_linear=dbz_to_linear(frames[2], config),
            displacement_yx=torch.zeros(2, dtype=torch.float64),
            log_growth_per_step=torch.tensor(
                config.max_log_growth_per_step,
                dtype=torch.float64,
            ),
        )
        result = result_for(state, config, frames=frames)
        verification = torch.full(
            (config.forecast_steps, 8, 9),
            config.max_dbz,
            dtype=torch.float64,
        )
        snapshot = compute_sensitivity_snapshot(
            frames[2],
            result,
            verification,
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
            ),
        )

        torch.testing.assert_close(
            snapshot.forecast_scores,
            torch.zeros_like(snapshot.forecast_scores),
        )
        torch.testing.assert_close(
            snapshot.control_sensitivity,
            torch.zeros_like(snapshot.control_sensitivity),
        )
        self.assertFalse(bool(torch.any(snapshot.forecast_cap_active_mask)))

    def test_missing_verification_is_not_recorded_as_zero_error(self) -> None:
        verification = torch.full_like(self.verification, float("nan"))
        snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            self.result,
            verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[2],
        )

        self.assertFalse(bool(torch.any(snapshot.metric_available)))
        self.assertTrue(bool(torch.all(torch.isnan(snapshot.forecast_scores))))
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.control_sensitivity)))
        )
        self.assertFalse(snapshot.impact_available)

    def test_scores_and_gradients_use_only_the_issued_domain(self) -> None:
        issued = torch.zeros_like(self.result.forecast_dbz, dtype=torch.bool)
        issued[:, 3:10, 4:13] = True
        issued &= self.result.valid_mask
        issued_dbz = torch.where(
            issued,
            self.result.forecast_dbz,
            torch.full_like(self.result.forecast_dbz, float("nan")),
        )
        forecast_digest = tensor_digest(issued_dbz)
        valid_mask_digest = tensor_digest(issued)
        result = replace(
            self.result,
            forecast_dbz=issued_dbz,
            valid_mask=issued,
            forecast_dbz_digest=forecast_digest,
            valid_mask_digest=valid_mask_digest,
            forecast_run_digest=_forecast_run_identity_digest(
                self.result.run,
                self.result.state_metadata_digest,
                forecast_digest,
                valid_mask_digest,
            ),
        )
        snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[2],
        )
        truth = dbz_to_linear(self.verification[0], self.nowcast_config)
        expected = forecast_metric(
            "log_echo_mse",
            forecast_linear_at_step(
                self.result.state,
                1,
                self.nowcast_config,
            ),
            truth,
            issued[0],
            self.nowcast_config,
            self.sensitivity_config,
        )

        torch.testing.assert_close(snapshot.forecast_scores[0, 0], expected)
        self.assertTrue(
            torch.equal(
                snapshot.forecast_sensitivity[0, 0][~issued[0]],
                torch.zeros_like(
                    snapshot.forecast_sensitivity[0, 0][~issued[0]]
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "valid mask disagrees"):
            compute_sensitivity_snapshot(
                self.frames[2],
                replace(self.result, valid_mask=issued),
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

    def test_unissued_forecast_has_no_sensitivity(self) -> None:
        metadata = replace(
            self.result.metadata,
            data_status=DataStatus.UNAVAILABLE,
        )
        metadata_digest = state_metadata_digest(
            self.result.state,
            metadata,
        )
        valid_mask = torch.zeros_like(self.result.valid_mask)
        forecast_dbz = torch.full_like(self.result.forecast_dbz, torch.nan)
        forecast_digest = tensor_digest(forecast_dbz)
        valid_mask_digest = tensor_digest(valid_mask)
        result = replace(
            self.result,
            forecast_dbz=forecast_dbz,
            valid_mask=valid_mask,
            metadata=metadata,
            state_metadata_digest=metadata_digest,
            forecast_dbz_digest=forecast_digest,
            valid_mask_digest=valid_mask_digest,
            forecast_run_digest=_forecast_run_identity_digest(
                self.result.run,
                metadata_digest,
                forecast_digest,
                valid_mask_digest,
            ),
        )

        with self.assertRaisesRegex(ValueError, "unissued forecast"):
            compute_sensitivity_snapshot(
                self.frames[2],
                result,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                latest_background_dbz=self.background[2],
            )

    def test_background_only_state_has_no_direct_sensitivity(self) -> None:
        frames = torch.full_like(self.frames, float("nan"))
        background = torch.full_like(self.frames, 20.0)
        result = nowcast(
            frames,
            self.nowcast_config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        with self.assertRaisesRegex(ValueError, "valid latest observation"):
            compute_sensitivity_snapshot(
                frames[2],
                result,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                latest_background_dbz=background[2],
            )

    def test_context_separates_observed_weather_from_missing_area(self) -> None:
        frames = torch.full((3, 4, 5), float("nan"), dtype=torch.float64)
        frames[2, 1, 2] = 40.0
        metadata = replace(
            self.result.metadata,
            coverage_by_frame=torch.tensor(
                [0.0, 0.0, 0.05],
                dtype=torch.float64,
            ),
            tendency_pair_count=1,
            tendency_source=TendencySource.BACKGROUND,
            background_contribution_fraction=0.25,
            source_support=torch.full((4, 5), 0.5, dtype=torch.float64),
        )
        state = replace(
            self.state,
            echo_linear=torch.zeros((4, 5), dtype=torch.float64),
        )
        features = extract_context_features(
            frames[2],
            state,
            metadata,
            self.nowcast_config,
            latest_observation_mask=torch.isfinite(frames[2]),
        )
        values = dict(zip(self.snapshot.context_feature_names, features))

        torch.testing.assert_close(values["latest_mean_dbz"], features.new_tensor(40.0))
        for name, expected in (
            ("echo_fraction_5dbz", 1.0),
            ("latest_observation_coverage", 0.05),
            ("current_state_support_fraction", 0.5),
            ("tendency_source_background", 1.0),
        ):
            torch.testing.assert_close(
                values[name],
                features.new_tensor(expected),
            )

    def test_context_preserves_independent_pair_conflicts(self) -> None:
        metadata = replace(
            self.result.metadata,
            motion_pair_conflict=True,
            growth_pair_conflict=False,
        )
        features = extract_context_features(
            self.frames[2],
            self.state,
            metadata,
            self.nowcast_config,
            latest_observation_mask=(
                self.qc_mask[2] & torch.isfinite(self.frames[2])
            ),
        )
        values = dict(zip(self.snapshot.context_feature_names, features))

        self.assertEqual(float(values["motion_pair_conflict"]), 1.0)
        self.assertEqual(float(values["growth_pair_conflict"]), 0.0)

    def test_context_records_reconstruction_and_growth_evidence(self) -> None:
        metadata = replace(
            self.result.metadata,
            state_path_source=TendencySource.OBSERVATION,
            state_path_mode=TendencyPairSelection.RECENT,
            state_path_pair_count=1,
            state_path_minimum_psr=12.0,
            state_path_conflict=True,
            state_path_extrapolated=False,
            state_path_age_minutes=10.0,
            minimum_growth_overlap_support=5.0,
            minimum_growth_overlap_area_km2=2.0,
            observation_path=StatePathProvenance(
                mode=TendencyPairSelection.RECENT,
                pair_count=1,
                minimum_psr=13.0,
                conflict=False,
                extrapolated=False,
                age_minutes=10.0,
            ),
            background_path=StatePathProvenance(
                mode=TendencyPairSelection.LONG,
                pair_count=1,
                minimum_psr=7.0,
                conflict=True,
                extrapolated=True,
                age_minutes=20.0,
            ),
        )
        features = extract_context_features(
            self.frames[2],
            self.state,
            metadata,
            self.nowcast_config,
            latest_observation_mask=(
                self.qc_mask[2] & torch.isfinite(self.frames[2])
            ),
        )
        values = dict(zip(self.snapshot.context_feature_names, features))

        expected = {
            "state_path_pair_count": 1.0,
            "state_path_source_observation": 1.0,
            "state_path_source_background": 0.0,
            "state_path_conflict": 1.0,
            "state_path_extrapolated": 0.0,
            "state_path_age_available": 1.0,
            "state_path_age_minutes": 10.0,
            "state_path_psr_available": 1.0,
            "log1p_state_path_minimum_psr": math.log1p(12.0),
            "growth_overlap_support_available": 1.0,
            "log1p_minimum_growth_overlap_support": math.log1p(5.0),
            "growth_overlap_area_available": 1.0,
            "log1p_minimum_growth_overlap_area_km2": math.log1p(2.0),
            "state_path_mode_recent": 1.0,
            "observation_path_pair_count": 1.0,
            "observation_path_conflict": 0.0,
            "observation_path_extrapolated": 0.0,
            "observation_path_age_available": 1.0,
            "observation_path_age_minutes": 10.0,
            "observation_path_psr_available": 1.0,
            "log1p_observation_path_minimum_psr": math.log1p(13.0),
            "background_path_pair_count": 1.0,
            "background_path_conflict": 1.0,
            "background_path_extrapolated": 1.0,
            "background_path_age_available": 1.0,
            "background_path_age_minutes": 20.0,
            "background_path_psr_available": 1.0,
            "log1p_background_path_minimum_psr": math.log1p(7.0),
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                torch.testing.assert_close(
                    values[name],
                    features.new_tensor(value),
                )

    def test_context_distinguishes_missing_psr_from_numeric_zero(self) -> None:
        metadata = replace(
            self.result.metadata,
            tendency_pair_count=0,
            motion_pair_count=0,
            growth_pair_count=0,
            motion_pair_selection=TendencyPairSelection.NONE,
            growth_pair_selection=TendencyPairSelection.NONE,
            minimum_phase_correlation_psr=torch.tensor(float("nan")),
        )
        features = extract_context_features(
            self.frames[2],
            self.state,
            metadata,
            self.nowcast_config,
            latest_observation_mask=(
                self.qc_mask[2] & torch.isfinite(self.frames[2])
            ),
        )
        values = dict(zip(self.snapshot.context_feature_names, features))

        self.assertEqual(float(values["phase_correlation_psr_available"]), 0.0)
        self.assertEqual(
            float(values["log1p_minimum_phase_correlation_psr"]),
            0.0,
        )

    def test_context_uses_projected_velocity_from_grid_contract(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-08-02T00:00:00Z",
                "2026-08-02T00:10:00Z",
                "2026-08-02T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="1" * 64,
            pixel_to_projected_matrix_m=(
                (800.0, -600.0),
                (600.0, 800.0),
            ),
        )
        result = result_for(
            self.state,
            self.nowcast_config,
            frames=self.frames,
            accepted_mask=self.qc_mask & torch.isfinite(self.frames),
            pair_motion=torch.stack(
                (
                    torch.zeros_like(self.state.displacement_yx),
                    self.state.displacement_yx,
                )
            ),
            grid_time_contract=contract,
        )
        snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
        )
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        self.assertEqual(snapshot.grid_time_contract_digest, contract.digest)
        expected = self.state.displacement_yx.new_tensor(
            (-410.0 / 600.0, 130.0 / 600.0)
        )

        self.assertEqual(context["projected_velocity_available"], 1.0)
        torch.testing.assert_close(
            context["projected_velocity_x_mps"],
            expected[0],
        )
        torch.testing.assert_close(
            context["projected_velocity_y_mps"],
            expected[1],
        )
        torch.testing.assert_close(
            context["projected_speed_mps"],
            torch.linalg.vector_norm(expected),
        )
        self.assertEqual(
            context["motion_disagreement_mps_available"], 1.0
        )
        torch.testing.assert_close(
            context["motion_disagreement_mps"],
            torch.linalg.vector_norm(expected),
        )
        self.assertEqual(context["area_weighted_echo_available"], 1.0)
        torch.testing.assert_close(
            context["log1p_linear_reflectivity_integral_km2"],
            context["log_integrated_echo"],
        )
        self.assertEqual(context["grid_spacing_available"], 1.0)
        self.assertEqual(context["grid_column_spacing_m"], contract.dx_m)
        self.assertEqual(context["grid_row_spacing_m"], contract.dy_m)

        issue_time = datetime(2026, 8, 2, 0, 20, tzinfo=timezone.utc)
        verification = VerificationBundle(
            frames_dbz=self.verification,
            valid_mask=torch.isfinite(self.verification),
            valid_times=tuple(
                (issue_time + timedelta(minutes=lead))
                .isoformat()
                .replace("+00:00", "Z")
                for lead in range(10, 181, 10)
            ),
            grid_contract_digest=contract.digest,
            radar_product_digest="2" * 64,
            qc_pipeline_digest="3" * 64,
        )
        lineage_snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            result,
            verification,
            sensitivity_config=replace(
                self.sensitivity_config,
                require_verification_lineage=True,
            ),
        )
        self.assertTrue(lineage_snapshot.verification_lineage_complete)
        self.assertEqual(
            lineage_snapshot.verification_bundle_digest,
            verification.content_digest,
        )
        self.assertEqual(
            lineage_snapshot.verification_valid_times,
            verification.valid_times,
        )

        with self.assertRaisesRegex(ValueError, "VerificationBundle"):
            compute_sensitivity_snapshot(
                self.frames[2],
                result,
                self.verification,
                sensitivity_config=replace(
                    self.sensitivity_config,
                    require_verification_lineage=True,
                ),
            )

    def test_area_weighted_echo_is_resolution_invariant(self) -> None:
        physical_integrals = []
        pixel_integrals = []
        for size, spacing_m in ((2, 1000.0), (4, 500.0)):
            latest = torch.full(
                (size, size), 20.0, dtype=torch.float64
            )
            state = RadarState(
                echo_linear=dbz_to_linear(latest, self.nowcast_config),
                displacement_yx=torch.zeros(2, dtype=torch.float64),
                log_growth_per_step=torch.zeros((), dtype=torch.float64),
            )
            contract = RadarGridTimeContract(
                valid_times=(
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:10:00Z",
                    "2026-08-02T00:20:00Z",
                ),
                dx_m=spacing_m,
                dy_m=spacing_m,
                projection="EPSG:5179",
                grid_hash=str(size) * 64,
            )
            features = extract_context_features(
                latest,
                state,
                metadata_for(state),
                self.nowcast_config,
                latest_observation_mask=torch.ones_like(
                    latest, dtype=torch.bool
                ),
                grid_time_contract=contract,
            )
            context = dict(zip(self.snapshot.context_feature_names, features))
            physical_integrals.append(
                context["log1p_linear_reflectivity_integral_km2"]
            )
            pixel_integrals.append(context["log_integrated_echo"])

        torch.testing.assert_close(
            physical_integrals[0], physical_integrals[1]
        )
        self.assertNotEqual(pixel_integrals[0], pixel_integrals[1])

    def test_pair_conflict_reduces_trust_without_changing_other_components(
        self,
    ) -> None:
        penalty = 0.4
        sensitivity_config = replace(
            self.sensitivity_config,
            pair_conflict_trust_penalty=penalty,
        )
        conflict_metadata = replace(
            self.result.metadata,
            motion_pair_count=0,
            motion_pair_selection=TendencyPairSelection.PERSISTENCE,
            motion_pair_conflict=True,
        )
        conflict_result = forecast_from_state(
            self.state,
            conflict_metadata,
            self.nowcast_config,
            run=self.result.run,
        )
        baseline = compute_sensitivity_snapshot(
            self.frames[2],
            self.result,
            self.verification,
            sensitivity_config=sensitivity_config,
            latest_background_dbz=self.background[2],
        )
        conflict = compute_sensitivity_snapshot(
            self.frames[2],
            conflict_result,
            self.verification,
            sensitivity_config=sensitivity_config,
            latest_background_dbz=self.background[2],
        )

        self.assertEqual(baseline.trust_components["pair_consistency"], 1.0)
        self.assertEqual(
            conflict.trust_components["pair_consistency"],
            penalty,
        )
        for name in (
            "linearity",
            "verification",
            "metric_support",
        ):
            self.assertAlmostEqual(
                conflict.trust_components[name],
                baseline.trust_components[name],
            )
        self.assertLess(
            conflict.trust_components["observation_verified_evidence"],
            baseline.trust_components["observation_verified_evidence"],
        )
        self.assertLess(
            conflict.trust_score,
            baseline.trust_score * penalty,
        )

    def test_unverified_path_support_reduces_trust(self) -> None:
        verified = self.result.metadata.verified_source_support.clone()
        verified[:, verified.shape[1] // 2 :] = 0.0
        metadata = replace(
            self.result.metadata,
            verified_source_support=verified,
            local_motion_verified_support=verified,
            local_growth_verified_support=verified,
            local_dynamics_verified_support=verified,
            observation_verified_source_support=verified,
        )
        unverified_result = forecast_from_state(
            self.state,
            metadata,
            self.nowcast_config,
            run=self.result.run,
        )

        baseline = compute_sensitivity_snapshot(
            self.frames[2],
            self.result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[2],
        )
        unverified = compute_sensitivity_snapshot(
            self.frames[2],
            unverified_result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[2],
        )

        baseline_available = (
            baseline.metric_available
            & torch.isfinite(baseline.path_evidence_by_metric)
            & torch.isfinite(
                baseline.observation_verified_evidence_by_metric
            )
        )
        unverified_available = (
            unverified.metric_available
            & torch.isfinite(unverified.path_evidence_by_metric)
            & torch.isfinite(
                unverified.observation_verified_evidence_by_metric
            )
        )
        baseline_observation_verified_evidence = float(
            baseline.observation_verified_evidence_by_metric[
                baseline_available
            ].mean()
        )
        unverified_observation_verified_evidence = float(
            unverified.observation_verified_evidence_by_metric[
                unverified_available
            ].mean()
        )
        self.assertAlmostEqual(
            baseline.trust_components["observation_verified_evidence"],
            baseline_observation_verified_evidence,
        )
        self.assertAlmostEqual(
            unverified.trust_components["observation_verified_evidence"],
            unverified_observation_verified_evidence,
        )
        self.assertLess(
            unverified_observation_verified_evidence,
            baseline_observation_verified_evidence,
        )
        for name in (
            "linearity",
            "verification",
            "metric_support",
            "pair_consistency",
        ):
            self.assertAlmostEqual(
                unverified.trust_components[name],
                baseline.trust_components[name],
            )
        self.assertAlmostEqual(
            unverified.trust_score,
            baseline.trust_score
            * unverified_observation_verified_evidence
            / baseline_observation_verified_evidence,
        )

    def test_joint_evidence_rejects_spatially_disjoint_marginals(
        self,
    ) -> None:
        weight = torch.ones((2, 2), dtype=torch.float64)
        observation_source = torch.tensor(
            [[1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        background_verified = 1.0 - observation_source
        evidence = _metric_evidence_ratios(
            weight,
            torch.ones_like(weight),
            background_verified,
            observation_source,
            torch.zeros_like(weight),
            background_verified,
            self.sensitivity_config.epsilon,
        )

        self.assertIsNotNone(evidence)
        assert evidence is not None
        path, observation_source_fraction, observation_verified, background = (
            evidence
        )
        self.assertEqual(float(path), 0.5)
        self.assertEqual(float(observation_source_fraction), 0.5)
        self.assertEqual(float(observation_verified), 0.0)
        self.assertEqual(float(background), 0.5)

    def test_m0_uses_the_issued_forecast_confidence(self) -> None:
        snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            self.result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=self.background[2],
        )
        torch.testing.assert_close(
            snapshot.forecast_confidence,
            self.result.forecast_confidence,
        )

    def test_nonfinite_background_does_not_create_fake_innovation(self) -> None:
        background = torch.full_like(self.frames, float("nan"))
        result = result_for(
            self.state,
            self.nowcast_config,
            frames=self.frames,
            accepted_mask=self.qc_mask & torch.isfinite(self.frames),
            background=background,
        )
        snapshot = compute_sensitivity_snapshot(
            self.frames[2],
            result,
            self.verification,
            sensitivity_config=self.sensitivity_config,
            latest_background_dbz=background[2],
        )

        self.assertFalse(snapshot.impact_available)
        self.assertIsNone(snapshot.observation_innovation_mask)
        self.assertIsNone(snapshot.observation_innovation_dbz)
        self.assertIsNone(snapshot.direct.impact)

    def test_soft_fss_handles_a_grid_smaller_than_its_window(self) -> None:
        config = SensitivityConfig(
            metric_names=("soft_fss_error_35",),
            full_map_lead_minutes=(10,),
            soft_fss_window=9,
        )
        truth_dbz = torch.full((5, 5), 10.0, dtype=torch.float64)
        truth_dbz[2, 2] = 45.0
        miss_dbz = torch.full_like(truth_dbz, 10.0)
        miss_dbz[0, 0] = 45.0
        valid = torch.ones_like(truth_dbz, dtype=torch.bool)
        score = forecast_metric(
            "soft_fss_error_35",
            dbz_to_linear(miss_dbz, self.nowcast_config),
            dbz_to_linear(truth_dbz, self.nowcast_config),
            valid,
            self.nowcast_config,
            config,
        )

        self.assertTrue(bool(torch.isfinite(score)))
        self.assertGreater(float(score), 0.0)

    def test_empty_centroid_and_invalid_configs_are_explicit(self) -> None:
        empty = torch.zeros((5, 5), dtype=torch.float64)
        valid = torch.ones_like(empty, dtype=torch.bool)
        score = forecast_metric(
            "centroid_error",
            empty,
            empty,
            valid,
            self.nowcast_config,
            self.sensitivity_config,
        )
        self.assertTrue(bool(torch.isnan(score)))

        invalid_configs = (
            {"metric_domain": "unknown"},
            {"tile_size": 2.5},
            {"tile_size_m": 0.0},
            {"soft_fss_window": 4.5},
            {"soft_fss_window_m": 0.0},
            {"soft_fss_temperature_dbz": float("nan")},
            {"active_margin_dbz": float("nan")},
            {"linearity_delta": (0.1, 0.2)},
            {"pair_conflict_trust_penalty": 0.0},
            {"pair_conflict_trust_penalty": 1.1},
            {"pair_conflict_trust_penalty": float("nan")},
            {"require_verification_lineage": 1},
            {"required_verification_radar_product_digest": "1" * 64},
            {
                "required_verification_radar_product_digest": "1" * 64,
                "required_verification_qc_pipeline_digest": "2" * 64,
            },
            {
                "require_verification_lineage": True,
                "required_verification_radar_product_digest": "bad",
                "required_verification_qc_pipeline_digest": "2" * 64,
            },
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    SensitivityConfig(**values)

    def test_physical_metric_settings_resolve_from_the_grid(self) -> None:
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-05T00:00:00Z",
                "2026-08-05T00:10:00Z",
                "2026-08-05T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="4" * 64,
        )
        forecast = torch.zeros((5, 5), dtype=torch.float64)
        truth = torch.zeros_like(forecast)
        forecast[2, 1] = 10.0
        truth[2, 3] = 10.0
        valid = torch.ones_like(forecast, dtype=torch.bool)
        centroid = forecast_metric(
            "centroid_error_m2",
            forecast,
            truth,
            valid,
            self.nowcast_config,
            SensitivityConfig(metric_names=("centroid_error_m2",)),
            grid,
        )
        self.assertAlmostEqual(float(centroid), 4_000_000.0)

        physical = forecast_metric(
            "soft_fss_error_35",
            forecast,
            truth,
            valid,
            self.nowcast_config,
            SensitivityConfig(
                metric_names=("soft_fss_error_35",),
                soft_fss_window_m=5_000.0,
            ),
            grid,
        )
        pixels = forecast_metric(
            "soft_fss_error_35",
            forecast,
            truth,
            valid,
            self.nowcast_config,
            SensitivityConfig(
                metric_names=("soft_fss_error_35",),
                soft_fss_window=5,
            ),
        )
        torch.testing.assert_close(physical, pixels)


class VariationalFSOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        cls.frames = torch.stack(
            tuple(
                -10.0
                + 40.0
                * torch.exp(
                    -(
                        (y - center).square()
                        + (x - center).square()
                    )
                    / 4.0
                )
                for center in (3.0, 3.5, 4.0)
            )
        )
        cls.nowcast_config = NowcastConfig(horizon_minutes=10)
        cls.analysis_config = AnalysisConfig(
            censored_background_policy="detection_limit",
            maximum_outer_iterations=12,
            maximum_pcg_iterations=100,
            pcg_relative_tolerance=1.0e-8,
        )
        cls.forecast, cls.analysis = variational_nowcast(
            cls.frames,
            nowcast_config=cls.nowcast_config,
            analysis_config=cls.analysis_config,
        )
        if (
            not cls.analysis.converged
            or cls.analysis.degraded
            or cls.analysis.linearization is None
        ):
            raise RuntimeError("variational FSO fixture did not converge")
        cls.verification = torch.where(
            torch.isfinite(cls.forecast.forecast_dbz),
            cls.forecast.forecast_dbz - 0.5,
            cls.forecast.forecast_dbz,
        )
        cls.sensitivity_config = SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(10,),
            tile_size=4,
        )
        cls.fso = compute_variational_fso(
            cls.forecast,
            cls.analysis,
            cls.verification,
            sensitivity_config=cls.sensitivity_config,
        )

    def test_variational_fso_covers_all_observation_times(self) -> None:
        fso = self.fso
        self.assertEqual(fso.contract, "p1-variational-fso-v13")
        self.assertEqual(
            fso.sensitivity_scope,
            "residual_plus_observation_derived_baseline_with_frozen_selection",
        )
        self.assertFalse(fso.baseline_dynamics_frozen)
        self.assertTrue(fso.baseline_pair_selection_frozen)
        self.assertEqual(fso.baseline_dynamics_branch_status, "unknown")
        self.assertIsNone(
            fso.observation.trusted_frozen_structure_input_dbz
        )
        self.assertEqual(
            fso.verification_contract,
            "legacy-verification-tensor-v1",
        )
        self.assertFalse(fso.verification_lineage_complete)
        self.assertIsNone(fso.verification_valid_times)
        self.assertEqual(fso.metric_names, ("log_echo_mse",))
        self.assertEqual(fso.metric_domain, "issued")
        self.assertNotEqual(fso.metric_domain_digest, "")
        self.assertEqual(fso.lead_minutes, (10,))
        self.assertEqual(fso.full_map_lead_minutes, (10,))
        self.assertEqual(
            fso.linearization_contract,
            "p1-final-frozen-irls-gn-v13",
        )
        self.assertEqual(
            fso.forecast_run_digest,
            self.forecast.forecast_run_digest,
        )
        linearization = self.analysis.linearization
        assert linearization is not None
        self.assertEqual(
            linearization.forecast_run_digest,
            self.forecast.forecast_run_digest,
        )
        self.assertEqual(fso.linearization_digest, linearization.linearization_digest)
        self.assertEqual(
            fso.algorithm_bundle_digest,
            linearization.algorithm_bundle_digest,
        )
        self.assertEqual(
            fso.numerical_runtime_digest,
            linearization.numerical_runtime_digest,
        )
        self.assertEqual(
            fso.feasibility_margins.reachability_support,
            linearization.feasibility_margins.reachability_support,
        )
        self.assertEqual(
            fso.feasibility_margins.unresolved_amplitude_fraction,
            linearization.feasibility_margins.unresolved_amplitude_fraction,
        )
        self.assertEqual(
            fso.feasibility_margins.motion_saturation_fraction,
            linearization.feasibility_margins.motion_saturation_fraction,
        )
        self.assertEqual(
            fso.feasibility_margins.growth_saturation_per_step,
            linearization.feasibility_margins.growth_saturation_per_step,
        )
        self.assertEqual(
            fso.variational_fso_digest,
            variational_fso_digest(fso),
        )
        validate_variational_fso(fso)
        self.assertLessEqual(
            linearization.relative_stationarity,
            self.analysis_config
            .final_linearization_relative_stationarity_tolerance,
        )
        self.assertEqual(
            self.analysis.linearization_relative_stationarity,
            linearization.relative_stationarity,
        )
        self.assertEqual(
            self.analysis.linearization_polish_iterations,
            linearization.polish_iterations,
        )
        self.assertEqual(
            fso.analysis_input_digest,
            self.forecast.run.analysis_input_digest,
        )
        self.assertEqual(fso.forecast_scores.shape, (1, 1))
        self.assertEqual(fso.metric_available.shape, (1, 1))
        self.assertEqual(fso.forecast_cap_active_mask.shape, (1, 8, 8))
        self.assertEqual(
            fso.observation.detected_dbz.maps.shape,
            (1, 1, 3, 8, 8),
        )
        self.assertEqual(
            fso.observation.detected_dbz.norm_by_time.shape,
            (1, 1, 3),
        )
        self.assertEqual(
            fso.observation.initial_background_dbz.maps.shape,
            (1, 1, 3, 8, 8),
        )
        self.assertEqual(
            fso.observation.baseline_dynamics_dbz.maps.shape,
            (1, 1, 3, 8, 8),
        )
        torch.testing.assert_close(
            fso.observation.frozen_structure_input_dbz.maps,
            fso.observation.detected_dbz.maps
            + fso.observation.initial_background_dbz.maps
            + fso.observation.baseline_dynamics_dbz.maps,
        )
        self.assertEqual(
            int(
                torch.count_nonzero(
                    fso.observation.initial_background_dbz.maps[
                        ..., 1:, :, :
                    ]
                )
            ),
            0,
        )
        self.assertGreater(fso.total_normal_products, 0)
        self.assertEqual(
            fso.total_normal_products,
            int(fso.adjoint_normal_products.sum())
            + fso.gauss_newton_diagnostics.normal_products,
        )
        self.assertGreater(fso.materialized_output_bytes, 0)
        self.assertGreaterEqual(
            float(fso.adjoint_true_residual_norm[0, 0]),
            0.0,
        )
        self.assertEqual(fso.adjoint_config_digest, VariationalAdjointConfig().digest)
        self.assertFalse(bool(fso.adjoint_warm_started[0, 0]))
        self.assertEqual(
            fso.gauss_newton_diagnostics.relative_curvature_defect.shape,
            (VariationalAdjointConfig().gauss_newton_probe_count,),
        )
        self.assertEqual(
            fso.gauss_newton_diagnostics.exact_hessian_products,
            VariationalAdjointConfig().gauss_newton_probe_count,
        )
        self.assertTrue(
            bool(
                torch.all(
                    torch.isfinite(
                        fso.gauss_newton_diagnostics
                        .relative_curvature_defect
                    )
                )
            )
        )

    def test_verification_bundle_binds_time_grid_qc_and_content(self) -> None:
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-05T00:00:00Z",
                "2026-08-05T00:10:00Z",
                "2026-08-05T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="4" * 64,
        )
        forecast, analysis = variational_nowcast(
            self.frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            grid_time_contract=grid,
        )
        self.assertTrue(analysis.converged)
        self.assertFalse(analysis.degraded)
        verification_frames = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        bundle = VerificationBundle(
            frames_dbz=verification_frames,
            valid_mask=torch.isfinite(verification_frames),
            valid_times=("2026-08-05T00:30:00Z",),
            grid_contract_digest=grid.digest,
            radar_product_digest="5" * 64,
            qc_pipeline_digest="6" * 64,
        )
        strict_config = replace(
            self.sensitivity_config,
            require_verification_lineage=True,
        )
        fso = compute_variational_fso(
            forecast,
            analysis,
            bundle,
            sensitivity_config=strict_config,
        )
        self.assertTrue(fso.verification_lineage_complete)
        self.assertEqual(
            fso.verification_contract,
            "radar-verification-bundle-v1",
        )
        self.assertEqual(fso.verification_bundle_digest, bundle.content_digest)
        self.assertEqual(fso.verification_valid_times, bundle.valid_times)
        self.assertEqual(fso.verification_grid_contract_digest, grid.digest)
        validate_variational_fso(fso)

        approved_config = replace(
            strict_config,
            required_verification_radar_product_digest="5" * 64,
            required_verification_qc_pipeline_digest="6" * 64,
        )
        compute_variational_fso(
            forecast,
            analysis,
            bundle,
            sensitivity_config=approved_config,
        )
        with self.assertRaisesRegex(ValueError, "not approved"):
            compute_variational_fso(
                forecast,
                analysis,
                bundle,
                sensitivity_config=replace(
                    approved_config,
                    required_verification_qc_pipeline_digest="8" * 64,
                ),
            )

        with self.assertRaisesRegex(ValueError, "VerificationBundle"):
            compute_variational_fso(
                forecast,
                analysis,
                verification_frames,
                sensitivity_config=strict_config,
            )
        wrong_time = VerificationBundle(
            frames_dbz=verification_frames,
            valid_mask=torch.isfinite(verification_frames),
            valid_times=("2026-08-05T00:40:00Z",),
            grid_contract_digest=grid.digest,
            radar_product_digest="5" * 64,
            qc_pipeline_digest="6" * 64,
        )
        with self.assertRaisesRegex(ValueError, "valid times"):
            compute_variational_fso(
                forecast,
                analysis,
                wrong_time,
                sensitivity_config=strict_config,
            )
        wrong_grid = VerificationBundle(
            frames_dbz=verification_frames,
            valid_mask=torch.isfinite(verification_frames),
            valid_times=bundle.valid_times,
            grid_contract_digest="7" * 64,
            radar_product_digest="5" * 64,
            qc_pipeline_digest="6" * 64,
        )
        with self.assertRaisesRegex(ValueError, "grid contracts"):
            compute_variational_fso(
                forecast,
                analysis,
                wrong_grid,
                sensitivity_config=strict_config,
            )

        valid_index = tuple(
            int(value)
            for value in torch.nonzero(bundle.valid_mask, as_tuple=False)[0]
        )
        bundle.frames_dbz[valid_index] += 1.0
        with self.assertRaisesRegex(ValueError, "content digest"):
            bundle.validate_integrity()

    def test_correlated_observation_fso_matches_perturb_and_resolve(
        self,
    ) -> None:
        analysis_config = replace(
            self.analysis_config,
            maximum_outer_iterations=12,
            observation_common_bias_std_dbz=0.2,
        )
        x_fraction = torch.linspace(
            0.0,
            1.0,
            self.frames.shape[-1],
            dtype=self.frames.dtype,
        ).expand(self.frames.shape[-2], -1)
        mode_weights = torch.stack(
            (torch.sqrt(1.0 - x_fraction), torch.sqrt(x_fraction))
        )
        forecast, analysis = variational_nowcast(
            self.frames,
            nowcast_config=self.nowcast_config,
            analysis_config=analysis_config,
            observation_common_bias_mode_weights=mode_weights,
        )
        self.assertTrue(analysis.converged, analysis.reason)
        self.assertFalse(analysis.degraded, analysis.reason)
        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        fso = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=self.sensitivity_config,
        )
        validate_variational_fso(fso)
        linearization = analysis.linearization
        assert linearization is not None
        original_mode_weights = (
            linearization.observations.common_bias_mode_weights
        )
        assert original_mode_weights is not None
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overlapping-linearization.npz"
            save_p1_linearization(analysis, path)
            restarted = load_p1_linearization(path)
            assert restarted.linearization is not None
            restarted_mode_weights = (
                restarted.linearization.observations.common_bias_mode_weights
            )
            assert restarted_mode_weights is not None
            torch.testing.assert_close(
                restarted_mode_weights,
                original_mode_weights,
            )
        observations = linearization.observations
        frozen = linearization.frozen
        truth = dbz_to_echo(
            verification[0],
            min_dbz=self.nowcast_config.min_dbz,
            max_dbz=self.nowcast_config.max_dbz,
        )
        valid = torch.isfinite(verification[0]) & forecast.valid_mask[0]
        lead_cell = freeze_remap_cell(analysis.state.displacement_yx)
        cap_active = fso.forecast_cap_active_mask[0]

        def score(candidate_control: torch.Tensor) -> torch.Tensor:
            trajectory = _analysis_trajectory(candidate_control, frozen)
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                self.nowcast_config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(
                    latent,
                    cap_active,
                    self.nowcast_config,
                ),
                truth,
                valid,
                self.nowcast_config,
                self.sensitivity_config,
            )

        def solve_perturbed(changed_dbz: torch.Tensor) -> torch.Tensor:
            changed_observations = replace(observations, dbz=changed_dbz)
            candidate = torch.nn.Parameter(analysis.control.clone())
            optimizer = torch.optim.LBFGS(
                (candidate,),
                lr=0.25,
                max_iter=100,
                tolerance_grad=1.0e-12,
                tolerance_change=1.0e-14,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                residual = residual_vector(
                    candidate,
                    changed_observations,
                    frozen,
                )
                objective = 0.5 * torch.dot(residual, residual)
                objective.backward()
                return objective

            optimizer.step(closure)
            return candidate.detach()

        sensitivity = fso.observation.detected_dbz.maps[0, 0]
        flat_index = int(torch.argmax(sensitivity.abs()))
        index = torch.unravel_index(
            torch.tensor(flat_index),
            observations.dbz.shape,
        )
        self.assertTrue(bool(observations.detected_mask[index]))
        delta = 1.0e-3
        perturbation = torch.zeros_like(observations.dbz)
        perturbation[index] = delta
        plus = solve_perturbed(observations.dbz + perturbation)
        minus = solve_perturbed(observations.dbz - perturbation)
        finite_difference = (score(plus) - score(minus)) / (2.0 * delta)
        torch.testing.assert_close(
            finite_difference,
            sensitivity[index],
            rtol=2.0e-1,
            atol=1.0e-8,
        )

    def test_adjoint_config_validation_and_active_set_margin(self) -> None:
        invalid = (
            {"lead_minutes": ()},
            {"lead_minutes": (20, 10)},
            {"lead_minutes": (10, 10)},
            {"pcg_relative_tolerance": 0.0},
            {"maximum_pcg_iterations": 0},
            {"maximum_normal_products": 0},
            {"maximum_materialized_output_bytes": 0},
            {"preconditioner": "unknown"},
            {"minimum_remap_fraction_margin": -1.0},
            {"minimum_reachability_margin": -1.0},
            {"minimum_unresolved_amplitude_fraction_margin": -1.0},
            {"minimum_amplitude_confidence_margin": -1.0},
            {"minimum_motion_saturation_margin_fraction": -1.0},
            {"minimum_motion_speed_saturation_margin_mps": -1.0},
            {"minimum_growth_saturation_margin_per_step": -1.0},
            {"gauss_newton_probe_count": 0},
            {"gauss_newton_probe_seed": -1},
            {"maximum_gauss_newton_relative_curvature_defect": -1.0},
            {"maximum_detected_delta_dbz": 0.0},
            {"maximum_censor_delta_dbz": 0.0},
            {"maximum_observation_weight_delta": 0.0},
            {"maximum_background_delta_dbz": 0.0},
            {"maximum_perturbed_pixel_count": 0},
            {"maximum_perturbed_fraction": 0.0},
            {"maximum_perturbed_fraction": 1.1},
            {"maximum_perturbed_area_km2": 0.0},
            {"maximum_whitened_perturbation_l2": 0.0},
            {"perturbation_tile_size": 0},
            {"perturbation_tile_size_m": 0.0},
            {"maximum_per_tile_whitened_norm": 0.0},
            {"maximum_observation_weight_l2": 0.0},
            {"minimum_observation_multiplier": 0.0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    VariationalAdjointConfig(**values)

        metric_policy = SensitivityConfig.for_automated_learning(
            radar_product_digest="1" * 64,
            qc_pipeline_digest="2" * 64,
        )
        self.assertEqual(
            metric_policy.metric_domain,
            "radar_dynamics_anchored",
        )
        self.assertTrue(metric_policy.require_verification_lineage)
        self.assertEqual(
            metric_policy.required_verification_radar_product_digest,
            "1" * 64,
        )
        self.assertEqual(
            metric_policy.required_verification_qc_pipeline_digest,
            "2" * 64,
        )
        adjoint_policy = VariationalAdjointConfig.for_automated_learning()
        self.assertTrue(adjoint_policy.require_active_set_margin)
        self.assertTrue(adjoint_policy.require_feasibility_margin)
        self.assertTrue(adjoint_policy.require_gauss_newton_reliability)
        self.assertTrue(
            adjoint_policy.require_baseline_dynamics_branch_validity
        )
        linearization = self.analysis.linearization
        assert linearization is not None
        learning_policy = AutomatedLearningPolicy(
            sensitivity_config=metric_policy,
            adjoint_config=adjoint_policy,
            algorithm_bundle_digest=linearization.algorithm_bundle_digest,
            numerical_runtime_digest=linearization.numerical_runtime_digest,
        )
        self.assertNotEqual(learning_policy.digest, "")
        for values in (
            {"maximum_candidate_count": 0},
            {"maximum_learning_resolves": 0},
            {
                "maximum_candidate_count": 1,
                "maximum_learning_resolves": 2,
            },
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    replace(learning_policy, **values)
        with self.assertRaisesRegex(ValueError, "verification lineage"):
            AutomatedLearningPolicy(
                sensitivity_config=SensitivityConfig(),
                adjoint_config=adjoint_policy,
                algorithm_bundle_digest=linearization.algorithm_bundle_digest,
                numerical_runtime_digest=(
                    linearization.numerical_runtime_digest
                ),
            )
        with self.assertRaisesRegex(ValueError, "local-validity gate"):
            AutomatedLearningPolicy(
                sensitivity_config=metric_policy,
                adjoint_config=VariationalAdjointConfig(),
                algorithm_bundle_digest=linearization.algorithm_bundle_digest,
                numerical_runtime_digest=(
                    linearization.numerical_runtime_digest
                ),
            )

        displacement = torch.tensor((0.25, 0.40), dtype=torch.float64)
        self.assertAlmostEqual(
            _remap_fraction_margin(
                displacement,
                freeze_remap_cell(displacement),
            ),
            0.25,
        )
        near_boundary = torch.tensor(
            (0.99999, 0.25),
            dtype=torch.float64,
        )
        self.assertLess(
            _remap_fraction_margin(
                near_boundary,
                freeze_remap_cell(near_boundary),
            ),
            2.0e-5,
        )

        self.assertFalse(self.fso.active_set_margins.low_local_validity)
        with self.assertRaisesRegex(ValueError, "active-set margin"):
            compute_variational_fso(
                self.forecast,
                self.analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    minimum_remap_fraction_margin=(
                        self.fso.active_set_margins.analysis_remap_fraction
                        + 1.0
                    ),
                    require_active_set_margin=True,
                ),
            )
        feasibility = self.fso.feasibility_margins
        with self.assertRaisesRegex(ValueError, "feasibility margin"):
            compute_variational_fso(
                self.forecast,
                self.analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    minimum_reachability_margin=(
                        feasibility.reachability_support + 1.0
                    ),
                    require_feasibility_margin=True,
                ),
            )
        self.assertTrue(self.fso.gauss_newton_diagnostics.reliable)
        with self.assertRaisesRegex(ValueError, "curvature approximation"):
            compute_variational_fso(
                self.forecast,
                self.analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    maximum_gauss_newton_relative_curvature_defect=1.0e-3,
                    require_gauss_newton_reliability=True,
                ),
            )
        with self.assertRaisesRegex(ValueError, "branch margins"):
            compute_variational_fso(
                self.forecast,
                self.analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    require_baseline_dynamics_branch_validity=True,
                ),
            )

    def test_feasibility_margins_are_content_addressed(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        changed_linearization = replace(
            linearization,
            feasibility_margins=replace(
                linearization.feasibility_margins,
                reachability_support=(
                    linearization.feasibility_margins.reachability_support
                    + 0.01
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "content digest"):
            compute_variational_fso(
                self.forecast,
                replace(self.analysis, linearization=changed_linearization),
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        changed_fso = replace(
            self.fso,
            feasibility_margins=replace(
                self.fso.feasibility_margins,
                growth_saturation_per_step=(
                    self.fso.feasibility_margins
                    .growth_saturation_per_step
                    + 0.01
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "result digest"):
            validate_variational_fso(changed_fso)

    def test_variational_fso_binds_metric_domain_policy(self) -> None:
        confidence = compute_variational_fso(
            self.forecast,
            self.analysis,
            self.verification,
            sensitivity_config=replace(
                self.sensitivity_config,
                metric_domain="confidence_weighted",
            ),
        )
        anchored = compute_variational_fso(
            self.forecast,
            self.analysis,
            self.verification,
            sensitivity_config=replace(
                self.sensitivity_config,
                metric_domain="radar_dynamics_anchored",
            ),
        )
        self.assertEqual(confidence.metric_domain, "confidence_weighted")
        self.assertEqual(
            anchored.metric_domain,
            "radar_dynamics_anchored",
        )
        self.assertNotEqual(
            confidence.metric_domain_digest,
            self.fso.metric_domain_digest,
        )
        self.assertNotEqual(
            anchored.metric_domain_digest,
            self.fso.metric_domain_digest,
        )
        self.assertLessEqual(
            float(confidence.metric_domain_weight_sum[0]),
            float(self.fso.metric_domain_weight_sum[0]),
        )
        self.assertLessEqual(
            float(anchored.metric_domain_weight_sum[0]),
            float(self.fso.metric_domain_weight_sum[0]),
        )

    def test_gauss_newton_curvature_defect_has_analytic_reference(self) -> None:
        control = torch.ones(1, dtype=torch.float64)

        def residual(value: torch.Tensor) -> torch.Tensor:
            return torch.cat((value, value.square()))

        def gauss_newton_product(value: torch.Tensor) -> torch.Tensor:
            return 5.0 * value

        config = VariationalAdjointConfig(
            gauss_newton_probe_count=1,
            maximum_gauss_newton_relative_curvature_defect=0.5,
        )
        diagnostics = _gauss_newton_curvature_diagnostics(
            control,
            residual,
            gauss_newton_product,
            _NormalProductBudget(maximum=2),
            config,
        )
        torch.testing.assert_close(
            diagnostics.relative_curvature_defect,
            torch.tensor((0.4,), dtype=torch.float64),
        )
        self.assertTrue(diagnostics.reliable)

        with self.assertRaisesRegex(ValueError, "curvature approximation"):
            _gauss_newton_curvature_diagnostics(
                control,
                residual,
                gauss_newton_product,
                _NormalProductBudget(maximum=2),
                replace(
                    config,
                    maximum_gauss_newton_relative_curvature_defect=0.3,
                    require_gauss_newton_reliability=True,
                ),
            )

    def test_selected_leads_warm_start_and_resource_budgets(self) -> None:
        nowcast_config = NowcastConfig(horizon_minutes=20)
        forecast, analysis = variational_nowcast(
            self.frames,
            nowcast_config=nowcast_config,
            analysis_config=self.analysis_config,
        )
        self.assertTrue(analysis.converged)
        self.assertFalse(analysis.degraded)
        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        all_maps = SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(10, 20),
            tile_size=4,
        )
        selected_maps = replace(all_maps, full_map_lead_minutes=(20,))
        cold = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=all_maps,
            adjoint_config=VariationalAdjointConfig(
                warm_start_by_metric=False,
            ),
        )
        warm = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=all_maps,
            adjoint_config=VariationalAdjointConfig(),
        )
        selected = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=selected_maps,
            adjoint_config=VariationalAdjointConfig(
                lead_minutes=(20,),
                warm_start_by_metric=False,
            ),
        )
        unpreconditioned = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=selected_maps,
            adjoint_config=VariationalAdjointConfig(
                lead_minutes=(20,),
                warm_start_by_metric=False,
                preconditioner="none",
            ),
        )

        self.assertEqual(selected.lead_minutes, (20,))
        self.assertEqual(selected.forecast_scores.shape, (1, 1))
        torch.testing.assert_close(
            selected.forecast_scores[0],
            cold.forecast_scores[1],
        )
        torch.testing.assert_close(
            selected.observation.detected_dbz.maps[0],
            cold.observation.detected_dbz.maps[1],
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        torch.testing.assert_close(
            selected.observation.detected_dbz.maps,
            unpreconditioned.observation.detected_dbz.maps,
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        self.assertFalse(bool(warm.adjoint_warm_started[0, 0]))
        self.assertTrue(bool(warm.adjoint_warm_started[1, 0]))
        torch.testing.assert_close(
            warm.observation.detected_dbz.maps,
            cold.observation.detected_dbz.maps,
            rtol=5.0e-4,
            atol=1.0e-9,
        )

        with self.assertRaisesRegex(ValueError, "byte budget"):
            compute_variational_fso(
                forecast,
                analysis,
                verification,
                sensitivity_config=selected_maps,
                adjoint_config=VariationalAdjointConfig(
                    lead_minutes=(20,),
                    maximum_materialized_output_bytes=1,
                ),
            )
        with self.assertRaisesRegex(ValueError, "normal-product budget"):
            compute_variational_fso(
                forecast,
                analysis,
                verification,
                sensitivity_config=selected_maps,
                adjoint_config=VariationalAdjointConfig(
                    lead_minutes=(20,),
                    maximum_normal_products=1,
                ),
            )

    def test_scalar_observation_sensitivity_sign_is_independent(self) -> None:
        sigma = 2.0
        truth = -1.5
        observation = torch.tensor(3.0, dtype=torch.float64)

        def analyzed(value: torch.Tensor) -> torch.Tensor:
            return value / (1.0 + sigma**2)

        def metric(value: torch.Tensor) -> torch.Tensor:
            return 0.5 * (analyzed(value) - truth).square()

        expected = (float(analyzed(observation)) - truth) / (1.0 + sigma**2)
        delta = 1.0e-5
        finite_difference = (
            metric(observation + delta) - metric(observation - delta)
        ) / (2.0 * delta)
        self.assertGreater(expected, 0.0)
        torch.testing.assert_close(
            finite_difference,
            torch.tensor(expected, dtype=torch.float64),
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_variational_fso_binds_verification_and_output_contents(
        self,
    ) -> None:
        fso = self.fso
        linearization = self.analysis.linearization
        assert linearization is not None
        changed_verification = self.verification.clone()
        finite_index = tuple(
            int(value)
            for value in torch.nonzero(
                torch.isfinite(changed_verification),
                as_tuple=False,
            )[0]
        )
        changed_verification[finite_index] += 0.25
        changed = compute_variational_fso(
            self.forecast,
            self.analysis,
            changed_verification,
            sensitivity_config=self.sensitivity_config,
        )
        self.assertNotEqual(
            self.fso.verification_bundle_digest,
            changed.verification_bundle_digest,
        )
        self.assertNotEqual(
            self.fso.variational_fso_digest,
            changed.variational_fso_digest,
        )

        changed_scores = self.fso.forecast_scores.clone()
        changed_scores[0, 0] += 1.0
        tampered = replace(self.fso, forecast_scores=changed_scores)
        with self.assertRaisesRegex(ValueError, "result digest mismatch"):
            validate_variational_fso(tampered)
        self.assertEqual(
            fso.observation.detected_dbz.tile_norm_by_time.shape,
            (1, 1, 3, 2, 2),
        )
        self.assertTrue(bool(fso.metric_available[0, 0]))
        self.assertGreater(int(fso.adjoint_iterations[0, 0]), 0)
        self.assertLess(float(fso.adjoint_relative_residual[0, 0]), 1.0e-8)
        self.assertTrue(
            bool(
                torch.all(
                    fso.observation.detected_dbz.norm_by_time[0, 0] > 0.0
                )
            )
        )
        censored = ~linearization.observations.detected_mask
        torch.testing.assert_close(
            fso.observation.detected_dbz.maps[0, 0][censored],
            torch.zeros_like(
                fso.observation.detected_dbz.maps[0, 0][censored]
            ),
        )
        self.assertGreater(
            float(fso.observation.censor_threshold_dbz.maps.abs().max()),
            0.0,
        )
        self.assertGreater(
            float(fso.observation.observation_weight.maps.abs().max()),
            0.0,
        )
        torch.testing.assert_close(
            fso.observation.censor_threshold_dbz.maps[0, 0][
                linearization.observations.detected_mask
            ],
            torch.zeros_like(
                fso.observation.censor_threshold_dbz.maps[0, 0][
                    linearization.observations.detected_mask
                ]
            ),
        )

    def test_p1_linearization_artifact_restarts_identical_fso(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1-linearization.npz"
            save_p1_linearization(self.analysis, path)
            loaded = load_p1_linearization(path)
            restarted = compute_variational_fso(
                self.forecast,
                loaded,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        self.assertEqual(
            loaded.linearization.linearization_digest,
            linearization.linearization_digest,
        )
        self.assertEqual(
            restarted.variational_fso_digest,
            self.fso.variational_fso_digest,
        )
        torch.testing.assert_close(
            restarted.observation.detected_dbz.maps,
            self.fso.observation.detected_dbz.maps,
            rtol=0.0,
            atol=0.0,
        )

    def test_p1_linearization_artifact_syncs_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1-linearization.npz"
            with patch(
                "advar.linearization_artifact.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                save_p1_linearization(self.analysis, path)

        self.assertEqual(fsync.call_count, 2)

    def test_p1_linearization_artifact_rejects_tamper_and_runtime_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1-linearization.npz"
            save_p1_linearization(self.analysis, path)
            with patch(
                "advar.linearization_artifact.algorithm_bundle_digest",
                return_value="0" * 64,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "algorithm bundle mismatch",
                ):
                    load_p1_linearization(path)
            with patch(
                "advar.linearization_artifact.numerical_runtime_identity_digest",
                return_value="0" * 64,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "numerical runtime mismatch",
                ):
                    load_p1_linearization(path)

            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            tensor_name = next(
                name
                for name, value in arrays.items()
                if name.startswith("tensor_") and value.dtype.kind == "f"
            )
            arrays[tensor_name].reshape(-1)[0] += 1.0
            with path.open("wb") as stream:
                np.savez(stream, **arrays)
            with self.assertRaisesRegex(
                ValueError,
                "artifact digest mismatch",
            ):
                load_p1_linearization(path)

    def test_variational_fso_matches_dense_and_finite_difference(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        frozen = linearization.frozen
        control = self.analysis.control
        config = self.nowcast_config
        sensitivity_config = self.sensitivity_config

        residual_fn = lambda value: residual_vector(
            value,
            observations,
            frozen,
        )
        jacobian = torch.func.jacrev(residual_fn)(control)
        normal_matrix = jacobian.mT @ jacobian
        truth = dbz_to_echo(
            self.verification[0],
            min_dbz=config.min_dbz,
            max_dbz=config.max_dbz,
        )
        valid = torch.isfinite(self.verification[0]) & self.forecast.valid_mask[0]
        lead_cell = freeze_remap_cell(self.analysis.state.displacement_yx)
        cap_active = self.fso.forecast_cap_active_mask[0]

        def score(candidate_control: torch.Tensor) -> torch.Tensor:
            trajectory = _analysis_trajectory(candidate_control, frozen)
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(latent, cap_active, config),
                truth,
                valid,
                config,
                sensitivity_config,
            )

        metric_rhs = torch.func.grad(score)(control)
        dense_adjoint = torch.linalg.solve(normal_matrix, metric_rhs)
        observation_count = observations.dbz.numel()
        observation_scale = torch.where(
            observations.detected_mask,
            torch.sqrt(observations.quality_weight)
            * frozen.irls_sqrt_weight
            / observations.std_dbz,
            torch.zeros_like(observations.dbz),
        )
        dense_sensitivity = observation_scale * (
            jacobian[:observation_count] @ dense_adjoint
        ).reshape_as(observations.dbz)
        torch.testing.assert_close(
            self.fso.observation.detected_dbz.maps[0, 0],
            dense_sensitivity,
            rtol=1.0e-5,
            atol=2.0e-10,
        )
        prediction_response = (
            jacobian[:observation_count] @ dense_adjoint
        ).reshape_as(observations.dbz)
        base_scale = (
            torch.sqrt(observations.quality_weight)
            * frozen.irls_sqrt_weight
            / observations.std_dbz
        )
        analyzed_dbz = echo_to_dbz(
            _analysis_trajectory(control, frozen).frames_linear,
            min_dbz=config.min_dbz,
        )
        censor_response = torch.sigmoid(
            (
                analyzed_dbz
                - frozen.analysis_config.detection_limit_dbz
            )
            / frozen.analysis_config.censor_temperature_dbz
        )
        censor_error = (
            frozen.analysis_config.censor_temperature_dbz
            * torch.nn.functional.softplus(
                (
                    analyzed_dbz
                    - frozen.analysis_config.detection_limit_dbz
                )
                / frozen.analysis_config.censor_temperature_dbz
            )
        )
        censor_scale = torch.where(
            observations.censored_mask,
            base_scale
            * (
                censor_response
                + (
                    censor_error
                    / frozen.analysis_config.censor_temperature_dbz
                )
                * (1.0 - censor_response)
            ),
            torch.zeros_like(observations.dbz),
        )
        dense_censor_sensitivity = censor_scale * prediction_response
        torch.testing.assert_close(
            self.fso.observation.censor_threshold_dbz.maps[0, 0],
            dense_censor_sensitivity,
            rtol=1.0e-5,
            atol=2.0e-10,
        )
        weighted_residual = residual_fn(control)[:observation_count].reshape_as(
            observations.dbz
        )
        dense_weight_sensitivity = -weighted_residual * prediction_response
        torch.testing.assert_close(
            self.fso.observation.observation_weight.maps[0, 0],
            dense_weight_sensitivity,
            rtol=1.0e-5,
            atol=2.0e-10,
        )

        def score_from_background(
            candidate_background: torch.Tensor,
        ) -> torch.Tensor:
            candidate_frozen = replace(
                frozen,
                initial_background_dbz=candidate_background,
            )
            trajectory = _analysis_trajectory(control, candidate_frozen)
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(latent, cap_active, config),
                truth,
                valid,
                config,
                sensitivity_config,
            )

        def stationarity_from_background(
            candidate_background: torch.Tensor,
        ) -> torch.Tensor:
            def objective(candidate_control: torch.Tensor) -> torch.Tensor:
                residual = residual_vector(
                    candidate_control,
                    observations,
                    replace(
                        frozen,
                        initial_background_dbz=candidate_background,
                    ),
                )
                return 0.5 * torch.dot(residual, residual)

            return torch.func.grad(objective)(control)

        background_mixed_gradient = torch.func.jacrev(
            stationarity_from_background
        )(frozen.initial_background_dbz)
        direct_background = torch.func.grad(score_from_background)(
            frozen.initial_background_dbz
        )
        dense_background = direct_background - torch.einsum(
            "chw,c->hw",
            background_mixed_gradient,
            dense_adjoint,
        )
        expected_background = torch.zeros_like(observations.dbz)
        expected_background[0] = torch.where(
            observations.valid_mask[0] & frozen.observed_mask[0],
            dense_background,
            torch.zeros_like(dense_background),
        )
        torch.testing.assert_close(
            self.fso.observation.initial_background_dbz.maps[0, 0],
            expected_background,
            rtol=1.0e-5,
            atol=2.0e-10,
        )

        def dynamics_from_observation(
            candidate_dbz: torch.Tensor,
        ) -> torch.Tensor:
            candidate_linear = dbz_to_echo(
                candidate_dbz,
                min_dbz=config.min_dbz,
                max_dbz=config.max_dbz,
            )
            estimate = _estimate_source_tendencies(
                candidate_dbz,
                frozen.observed_mask,
                candidate_linear,
                config,
                frozen.grid_time_contract,
            )
            return torch.cat(
                (
                    estimate.displacement_yx,
                    estimate.log_growth_per_step.reshape(1),
                )
            )

        nominal_dynamics = torch.cat(
            (
                frozen.baseline_state.displacement_yx,
                frozen.baseline_state.log_growth_per_step.reshape(1),
            )
        )

        def frozen_from_dynamics(candidate: torch.Tensor):
            return replace(
                frozen,
                baseline_state=RadarState(
                    echo_linear=frozen.baseline_state.echo_linear,
                    displacement_yx=candidate[:2],
                    log_growth_per_step=candidate[2],
                ),
            )

        def score_from_dynamics(candidate: torch.Tensor) -> torch.Tensor:
            candidate_frozen = frozen_from_dynamics(candidate)
            trajectory = _analysis_trajectory(control, candidate_frozen)
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(latent, cap_active, config),
                truth,
                valid,
                config,
                sensitivity_config,
            )

        dynamics_jacobian = torch.func.jacrev(
            dynamics_from_observation
        )(observations.dbz)
        def stationarity_from_dynamics(
            candidate: torch.Tensor,
        ) -> torch.Tensor:
            def objective(candidate_control: torch.Tensor) -> torch.Tensor:
                residual = residual_vector(
                    candidate_control,
                    observations,
                    frozen_from_dynamics(candidate),
                )
                return 0.5 * torch.dot(residual, residual)

            return torch.func.grad(objective)(control)

        dynamics_mixed_gradient = torch.func.jacrev(
            stationarity_from_dynamics
        )(nominal_dynamics)
        dynamics_metric_gradient = torch.func.grad(score_from_dynamics)(
            nominal_dynamics
        )
        dynamics_parameter_sensitivity = dynamics_metric_gradient - (
            dynamics_mixed_gradient.mT @ dense_adjoint
        )
        expected_dynamics = torch.einsum(
            "dthw,d->thw",
            dynamics_jacobian,
            dynamics_parameter_sensitivity,
        )
        expected_dynamics = torch.where(
            observations.valid_mask & frozen.observed_mask,
            expected_dynamics,
            torch.zeros_like(expected_dynamics),
        )
        torch.testing.assert_close(
            self.fso.observation.baseline_dynamics_dbz.maps[0, 0],
            expected_dynamics,
            rtol=1.0e-5,
            atol=2.0e-10,
        )
        torch.testing.assert_close(
            self.fso.observation.frozen_structure_input_dbz.maps[0, 0],
            dense_sensitivity + expected_background + expected_dynamics,
            rtol=1.0e-5,
            atol=2.0e-10,
        )

        time_index, row, column = 2, 4, 4
        observation_direction = torch.zeros_like(observations.dbz)
        observation_direction[time_index, row, column] = 1.0
        control_direction = torch.linalg.solve(
            normal_matrix,
            jacobian[:observation_count].mT
            @ (observation_scale * observation_direction).reshape(-1),
        )
        delta = 1.0e-3
        finite_difference = (
            score(control + delta * control_direction)
            - score(control - delta * control_direction)
        ) / (2.0 * delta)
        torch.testing.assert_close(
            finite_difference,
            self.fso.observation.detected_dbz.maps[
                0,
                0,
                time_index,
                row,
                column,
            ],
            rtol=1.0e-5,
            atol=1.0e-10,
        )

    def test_variational_fso_matches_observation_perturb_and_resolve(
        self,
    ) -> None:
        coordinates = torch.arange(8, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        moving_frames = torch.stack(
            tuple(
                -10.0
                + 35.0
                * torch.exp(
                    -(
                        (y - (3.0 + 0.18 * index)).square()
                        + (x - (3.1 + 0.21 * index)).square()
                    )
                    / 3.0
                )
                + 8.0
                * torch.exp(
                    -(
                        (y - (5.2 + 0.18 * index)).square()
                        + (x - (5.7 + 0.21 * index)).square()
                    )
                    / 0.8
                )
                for index in range(3)
            )
        )
        forecast, analysis = variational_nowcast(
            moving_frames,
            nowcast_config=self.nowcast_config,
            analysis_config=replace(
                self.analysis_config,
                maximum_outer_iterations=12,
            ),
        )
        self.assertTrue(analysis.converged, analysis.reason)
        self.assertFalse(analysis.degraded, analysis.reason)
        verification = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        fso = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=self.sensitivity_config,
        )
        self.assertFalse(fso.active_set_margins.low_local_validity)

        linearization = analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        frozen = linearization.frozen
        config = self.nowcast_config
        truth = dbz_to_echo(
            verification[0],
            min_dbz=config.min_dbz,
            max_dbz=config.max_dbz,
        )
        valid = torch.isfinite(verification[0]) & forecast.valid_mask[0]
        lead_cell = freeze_remap_cell(analysis.state.displacement_yx)
        cap_active = fso.forecast_cap_active_mask[0]

        def score(
            candidate_control: torch.Tensor,
            candidate_frozen=frozen,
        ) -> torch.Tensor:
            trajectory = _analysis_trajectory(
                candidate_control,
                candidate_frozen,
            )
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(latent, cap_active, config),
                truth,
                valid,
                config,
                self.sensitivity_config,
            )

        def stationarity_at_control(
            candidate_observations,
            candidate_frozen,
        ) -> torch.Tensor:
            def objective(candidate_control: torch.Tensor) -> torch.Tensor:
                residual = residual_vector(
                    candidate_control,
                    candidate_observations,
                    candidate_frozen,
                )
                return 0.5 * torch.dot(residual, residual)

            return torch.func.grad(objective)(analysis.control)

        nominal_residual_function = lambda candidate_control: residual_vector(
            candidate_control,
            observations,
            frozen,
        )
        nominal_jacobian = torch.func.jacrev(
            nominal_residual_function
        )(analysis.control)
        nominal_normal = nominal_jacobian.mT @ nominal_jacobian
        nominal_stationarity = stationarity_at_control(observations, frozen)

        def frozen_gn_response(candidate_observations, candidate_frozen):
            stationarity_change = (
                stationarity_at_control(
                    candidate_observations,
                    candidate_frozen,
                )
                - nominal_stationarity
            )
            return analysis.control - torch.linalg.solve(
                nominal_normal,
                stationarity_change,
            )

        time_index, row, column = 2, 4, 4
        delta = 1.0e-3
        perturbation = torch.zeros_like(observations.dbz)
        perturbation[time_index, row, column] = delta
        plus = frozen_gn_response(
            replace(observations, dbz=observations.dbz + perturbation),
            frozen,
        )
        minus = frozen_gn_response(
            replace(observations, dbz=observations.dbz - perturbation),
            frozen,
        )
        finite_difference = (score(plus) - score(minus)) / (2.0 * delta)

        torch.testing.assert_close(
            finite_difference,
            fso.observation.detected_dbz.maps[
                0,
                0,
                time_index,
                row,
                column,
            ],
            rtol=5.0e-3,
            atol=1.0e-8,
        )

        first_frame_detected = torch.nonzero(
            observations.detected_mask[0],
            as_tuple=False,
        )
        total_magnitude = torch.abs(
            fso.observation.frozen_structure_input_dbz.maps[
                0,
                0,
                0,
            ]
        )
        detected_magnitude = total_magnitude[
            first_frame_detected[:, 0],
            first_frame_detected[:, 1],
        ]
        first_frame_detected = first_frame_detected[
            int(torch.argmax(detected_magnitude))
        ]
        first_row, first_column = (
            int(first_frame_detected[0]),
            int(first_frame_detected[1]),
        )
        input_delta = 1.0e-2
        input_perturbation = torch.zeros_like(observations.dbz)
        input_perturbation[0, first_row, first_column] = input_delta
        background_perturbation = torch.zeros_like(
            frozen.initial_background_dbz
        )
        background_perturbation[first_row, first_column] = input_delta
        plus_observations = replace(
            observations,
            dbz=observations.dbz + input_perturbation,
        )
        minus_observations = replace(
            observations,
            dbz=observations.dbz - input_perturbation,
        )

        def frozen_for_input(
            candidate_observations,
            candidate_background: torch.Tensor,
        ):
            candidate_linear = dbz_to_echo(
                candidate_observations.dbz,
                min_dbz=config.min_dbz,
                max_dbz=config.max_dbz,
            )
            estimate = _estimate_source_tendencies(
                candidate_observations.dbz,
                frozen.observed_mask,
                candidate_linear,
                config,
                frozen.grid_time_contract,
            )
            return replace(
                frozen,
                initial_background_dbz=candidate_background,
                baseline_state=RadarState(
                    echo_linear=frozen.baseline_state.echo_linear,
                    displacement_yx=estimate.displacement_yx,
                    log_growth_per_step=estimate.log_growth_per_step,
                ),
            )

        plus_frozen = frozen_for_input(
            plus_observations,
            frozen.initial_background_dbz + background_perturbation,
        )
        minus_frozen = frozen_for_input(
            minus_observations,
            frozen.initial_background_dbz - background_perturbation,
        )

        plus = frozen_gn_response(plus_observations, plus_frozen)
        minus = frozen_gn_response(minus_observations, minus_frozen)
        frozen_structure_difference = (
            score(plus, plus_frozen) - score(minus, minus_frozen)
        ) / (2.0 * input_delta)
        torch.testing.assert_close(
            frozen_structure_difference,
            fso.observation.frozen_structure_input_dbz.maps[
                0,
                0,
                0,
                first_row,
                first_column,
            ],
            # Re-solve the retained frozen-GN stationarity equation from an
            # actual input perturbation, including both baseline pathways.
            rtol=2.0e-2,
            atol=1.0e-8,
        )

    def test_censored_and_weight_sensitivity_match_perturb_and_resolve(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        frozen = linearization.frozen
        control = self.analysis.control
        config = self.nowcast_config
        observation_count = observations.dbz.numel()
        base_scale = (
            torch.sqrt(observations.quality_weight)
            * frozen.irls_sqrt_weight
            / observations.std_dbz
        )
        truth = dbz_to_echo(
            self.verification[0],
            min_dbz=config.min_dbz,
            max_dbz=config.max_dbz,
        )
        valid = torch.isfinite(self.verification[0]) & self.forecast.valid_mask[0]
        lead_cell = freeze_remap_cell(self.analysis.state.displacement_yx)
        cap_active = self.fso.forecast_cap_active_mask[0]

        def score(candidate_control: torch.Tensor) -> torch.Tensor:
            trajectory = _analysis_trajectory(candidate_control, frozen)
            state = RadarState(
                echo_linear=trajectory.frames_linear[-1],
                displacement_yx=trajectory.displacement_yx,
                log_growth_per_step=trajectory.log_growth_per_step,
            )
            latent = _forecast_linear_at_step_core(
                state,
                1,
                config,
                lead_cell,
            )
            return forecast_metric(
                "log_echo_mse",
                _apply_output_cap(latent, cap_active, config),
                truth,
                valid,
                config,
                self.sensitivity_config,
            )

        def solve(
            residual_function,
            delta: float,
        ) -> torch.Tensor:
            candidate = torch.nn.Parameter(control.clone())
            optimizer = torch.optim.LBFGS(
                (candidate,),
                lr=0.25,
                max_iter=150,
                tolerance_grad=1.0e-12,
                tolerance_change=1.0e-14,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                residual = residual_function(candidate, delta)
                objective = 0.5 * torch.dot(residual, residual)
                objective.backward()
                return objective

            optimizer.step(closure)
            return candidate.detach()

        censor_flat_index = int(
            torch.argmax(
                self.fso.observation.censor_threshold_dbz.maps[0, 0].abs()
            )
        )
        censor_index = torch.unravel_index(
            torch.tensor(censor_flat_index),
            observations.dbz.shape,
        )
        censor_mask = torch.zeros_like(observations.dbz, dtype=torch.bool)
        censor_mask[censor_index] = True

        def censor_residual(
            candidate_control: torch.Tensor,
            delta: float,
        ) -> torch.Tensor:
            base = residual_vector(candidate_control, observations, frozen)
            trajectory = _analysis_trajectory(candidate_control, frozen)
            prediction = echo_to_dbz(
                trajectory.frames_linear,
                min_dbz=config.min_dbz,
            )
            censor_error = (
                frozen.analysis_config.censor_temperature_dbz
                * torch.nn.functional.softplus(
                    (
                        prediction
                        - frozen.analysis_config.detection_limit_dbz
                        - delta * censor_mask
                    )
                    / frozen.analysis_config.censor_temperature_dbz
                )
            )
            observation_block = torch.where(
                censor_mask,
                base_scale * censor_error,
                base[:observation_count].reshape_as(observations.dbz),
            )
            return torch.cat(
                (observation_block.reshape(-1), base[observation_count:])
            )

        weight_flat_index = int(
            torch.argmax(
                self.fso.observation.observation_weight.maps[0, 0].abs()
            )
        )
        weight_index = torch.unravel_index(
            torch.tensor(weight_flat_index),
            observations.dbz.shape,
        )
        weight_mask = torch.zeros_like(observations.dbz)
        weight_mask[weight_index] = 1.0

        def weight_residual(
            candidate_control: torch.Tensor,
            delta: float,
        ) -> torch.Tensor:
            base = residual_vector(candidate_control, observations, frozen)
            observation_block = base[:observation_count].reshape_as(
                observations.dbz
            ) * torch.sqrt(1.0 + delta * weight_mask)
            return torch.cat(
                (observation_block.reshape(-1), base[observation_count:])
            )

        delta = 1.0e-3
        censor_difference = (
            score(solve(censor_residual, delta))
            - score(solve(censor_residual, -delta))
        ) / (2.0 * delta)
        weight_difference = (
            score(solve(weight_residual, delta))
            - score(solve(weight_residual, -delta))
        ) / (2.0 * delta)
        torch.testing.assert_close(
            censor_difference,
            self.fso.observation.censor_threshold_dbz.maps[0, 0][
                censor_index
            ],
            # The production inverse is Gauss--Newton while this independent
            # re-solve uses the exact nonlinear least-squares curvature.
            rtol=1.2e-1,
            atol=1.0e-8,
        )
        torch.testing.assert_close(
            weight_difference,
            self.fso.observation.observation_weight.maps[0, 0][weight_index],
            rtol=2.5e-1,
            atol=1.0e-8,
        )

    def test_variational_fsoi_requires_and_applies_explicit_perturbation(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        detected = torch.zeros_like(observations.dbz)
        censor_threshold = torch.zeros_like(observations.dbz)
        observation_weight = torch.zeros_like(observations.dbz)
        initial_background = torch.zeros_like(observations.dbz)
        baseline_dynamics = torch.zeros_like(observations.dbz)
        detected_index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.detected_mask,
                as_tuple=False,
            )[0]
        )
        censored_index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.censored_mask,
                as_tuple=False,
            )[0]
        )
        detected[detected_index] = 0.25
        censor_threshold[censored_index] = -0.5
        observation_weight[censored_index] = -0.1
        first_frame_index = (
            0,
            *tuple(
                int(value)
                for value in torch.nonzero(
                    observations.valid_mask[0],
                    as_tuple=False,
                )[0]
            ),
        )
        initial_background[first_frame_index] = 0.125
        baseline_dynamics[detected_index] = -0.125
        perturbation = VariationalObservationPerturbation(
            detected_dbz=detected,
            censor_threshold_dbz=censor_threshold,
            observation_weight=observation_weight,
            initial_background_dbz=initial_background,
            baseline_dynamics_dbz=baseline_dynamics,
        )

        fsoi = compute_variational_fsoi(
            self.forecast,
            self.analysis,
            self.verification,
            perturbation,
            sensitivity_config=self.sensitivity_config,
        )

        self.assertEqual(
            fsoi.contract,
            "p1-linearized-observation-impact-v11",
        )
        self.assertEqual(
            fsoi.perturbation_contract,
            "p1-observation-perturbation-v7",
        )
        self.assertEqual(fsoi.perturbation_digest, perturbation.digest)
        self.assertEqual(
            fsoi.perturbation_diagnostics.perturbed_pixel_count,
            2,
        )
        self.assertGreater(fsoi.perturbation_diagnostics.whitened_l2, 0.0)
        torch.testing.assert_close(
            fsoi.fso.observation.detected_dbz.maps,
            self.fso.observation.detected_dbz.maps,
        )
        expected_components = (
            self.fso.observation.detected_dbz.maps[0, 0] * detected,
            self.fso.observation.censor_threshold_dbz.maps[0, 0]
            * censor_threshold,
            self.fso.observation.observation_weight.maps[0, 0]
            * observation_weight,
            self.fso.observation.initial_background_dbz.maps[0, 0]
            * initial_background,
            self.fso.observation.baseline_dynamics_dbz.maps[0, 0]
            * baseline_dynamics,
        )
        actual_components = (
            fsoi.observation.detected_dbz.maps[0, 0],
            fsoi.observation.censor_threshold_dbz.maps[0, 0],
            fsoi.observation.observation_weight.maps[0, 0],
            fsoi.observation.initial_background_dbz.maps[0, 0],
            fsoi.observation.baseline_dynamics_dbz.maps[0, 0],
        )
        for actual, expected in zip(
            actual_components,
            expected_components,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        expected_total = sum(
            expected_components,
            torch.zeros_like(expected_components[0]),
        )
        torch.testing.assert_close(
            fsoi.observation.total.maps[0, 0],
            expected_total,
        )
        torch.testing.assert_close(
            fsoi.observation.total.sum_by_time[0, 0],
            expected_total.reshape(3, -1).sum(dim=1),
        )
        torch.testing.assert_close(
            fsoi.observation.total.tile_sum_by_time,
            fsoi.observation.detected_dbz.tile_sum_by_time
            + fsoi.observation.censor_threshold_dbz.tile_sum_by_time
            + fsoi.observation.observation_weight.tile_sum_by_time
            + fsoi.observation.initial_background_dbz.tile_sum_by_time
            + fsoi.observation.baseline_dynamics_dbz.tile_sum_by_time,
        )
        self.assertIn(
            fsoi.baseline_dynamics_branch_status,
            ("certified", "invalid"),
        )
        self.assertEqual(
            fsoi.observation.trusted_total is not None,
            fsoi.baseline_dynamics_branch_status == "certified",
        )
        validate_variational_fsoi(fsoi)
        self.assertNotEqual(fsoi.variational_fsoi_digest, "")

        changed = replace(
            perturbation,
            detected_dbz=2.0 * perturbation.detected_dbz,
        )
        self.assertNotEqual(perturbation.digest, changed.digest)

    def test_variational_fsoi_rejects_invalid_perturbation_contracts(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        zeros = torch.zeros_like(observations.dbz)
        invalid_detected = zeros.clone()
        censored_index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.censored_mask,
                as_tuple=False,
            )[0]
        )
        invalid_detected[censored_index] = 1.0
        invalid = VariationalObservationPerturbation(
            detected_dbz=invalid_detected,
            censor_threshold_dbz=zeros,
            observation_weight=zeros,
        )
        with self.assertRaisesRegex(ValueError, "outside its active mask"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                invalid,
                sensitivity_config=self.sensitivity_config,
            )

        invalid_weight = zeros.clone()
        invalid_weight[censored_index] = -1.0
        invalid = replace(
            invalid,
            detected_dbz=zeros,
            observation_weight=invalid_weight,
        )
        with self.assertRaisesRegex(ValueError, "local first-order limit"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                invalid,
                sensitivity_config=self.sensitivity_config,
            )

        detected_index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.detected_mask,
                as_tuple=False,
            )[0]
        )
        large_detected = zeros.clone()
        large_detected[detected_index] = 0.51
        invalid = replace(
            invalid,
            detected_dbz=large_detected,
            observation_weight=zeros,
        )
        with self.assertRaisesRegex(ValueError, "local first-order limit"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                invalid,
                sensitivity_config=self.sensitivity_config,
            )

        invalid_background = zeros.clone()
        later_valid_index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.valid_mask[1],
                as_tuple=False,
            )[0]
        )
        invalid_background[(1, *later_valid_index)] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "accepted first-frame observations",
        ):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                replace(
                    invalid,
                    detected_dbz=zeros,
                    observation_weight=zeros,
                    initial_background_dbz=invalid_background,
                ),
                sensitivity_config=self.sensitivity_config,
            )
        with self.assertRaisesRegex(
            ValueError,
            "baseline_dynamics_dbz perturbation shape mismatch",
        ):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                replace(
                    invalid,
                    detected_dbz=zeros,
                    observation_weight=zeros,
                    initial_background_dbz=None,
                    baseline_dynamics_dbz=zeros[:, :-1],
                ),
                sensitivity_config=self.sensitivity_config,
            )

    def test_variational_fsoi_enforces_global_trust_radius(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        zeros = torch.zeros_like(observations.dbz)
        dense = torch.where(
            observations.detected_mask,
            torch.full_like(zeros, 0.1),
            zeros,
        )
        perturbation = VariationalObservationPerturbation(
            detected_dbz=dense,
            censor_threshold_dbz=zeros,
            observation_weight=zeros,
        )
        with self.assertRaisesRegex(ValueError, "area fraction"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                perturbation,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    maximum_perturbed_fraction=0.01,
                ),
            )

        sparse = torch.zeros_like(zeros)
        index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.detected_mask,
                as_tuple=False,
            )[0]
        )
        sparse[index] = 0.25
        perturbation = replace(perturbation, detected_dbz=sparse)
        with self.assertRaisesRegex(ValueError, "whitened trust radius"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                perturbation,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    maximum_whitened_perturbation_l2=0.01,
                ),
            )
        with self.assertRaisesRegex(ValueError, "requires a grid contract"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                perturbation,
                sensitivity_config=self.sensitivity_config,
                adjoint_config=VariationalAdjointConfig(
                    maximum_perturbed_area_km2=1.0,
                ),
            )

    def test_variational_fsoi_rejects_directional_classification_change(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        zeros = torch.zeros_like(observations.dbz)
        censored_values = torch.where(
            observations.censored_mask,
            observations.dbz,
            observations.dbz.new_full((), -math.inf),
        )
        flat_index = int(torch.argmax(censored_values))
        index = torch.unravel_index(
            torch.tensor(flat_index),
            observations.dbz.shape,
        )
        threshold_delta = zeros.clone()
        threshold_delta[index] = -0.5
        perturbation = VariationalObservationPerturbation(
            detected_dbz=zeros,
            censor_threshold_dbz=threshold_delta,
            observation_weight=zeros,
        )
        with self.assertRaisesRegex(ValueError, "detected/censored branch"):
            compute_variational_fsoi(
                self.forecast,
                self.analysis,
                self.verification,
                perturbation,
                sensitivity_config=self.sensitivity_config,
            )

    def test_baseline_branch_signature_covers_pair_and_peak_identity(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        signature = _p0_tendency_branch_signature(
            linearization.observations.dbz,
            linearization.frozen,
        )

        self.assertEqual(signature.pair_spans, ((0, 1), (1, 2), (0, 2)))
        self.assertEqual(len(signature.pair_available_by_span), 3)
        self.assertEqual(len(signature.growth_evidence_available_by_span), 3)
        self.assertEqual(len(signature.integer_peak_yx_by_pair), 3)
        self.assertEqual(len(signature.peak_is_search_interior_by_pair), 3)
        self.assertIsInstance(signature.motion_pair_spans, tuple)
        self.assertIsInstance(signature.growth_pair_spans, tuple)
        self.assertEqual(len(signature.motion_remap_cells), 2)

    def test_baseline_branch_validation_checks_half_perturbation(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        unchanged = _p0_tendency_branch_signature(
            linearization.observations.dbz,
            linearization.frozen,
        )
        changed = replace(
            unchanged,
            motion_conflict=not unchanged.motion_conflict,
        )
        with patch(
            "advar.sensitivity._p0_tendency_branch_signature",
            side_effect=(unchanged, changed),
        ) as signature:
            delta = torch.where(
                linearization.observations.detected_mask,
                torch.full_like(linearization.observations.dbz, 0.1),
                torch.zeros_like(linearization.observations.dbz),
            )
            stable = _baseline_branch_is_stable(
                linearization.observations,
                linearization.frozen,
                delta,
            )

        self.assertFalse(stable)
        self.assertEqual(signature.call_count, 2)

    def test_physical_radar_perturbation_factory_connects_input_paths(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        delta = torch.zeros_like(observations.dbz)
        index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.detected_mask[0],
                as_tuple=False,
            )[0]
        )
        full_index = (0, *index)
        delta[full_index] = 0.1
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            linearization,
        )
        self.assertEqual(
            perturbation.perturbation_semantics,
            "physical_radar_value",
        )
        self.assertEqual(float(perturbation.detected_dbz[full_index]), 0.1)
        assert perturbation.initial_background_dbz is not None
        self.assertEqual(
            float(perturbation.initial_background_dbz[full_index]),
            0.1,
        )
        fsoi = compute_variational_fsoi(
            self.forecast,
            self.analysis,
            self.verification,
            perturbation,
            sensitivity_config=self.sensitivity_config,
        )
        self.assertEqual(
            fsoi.perturbation_diagnostics.perturbed_pixel_count,
            1,
        )
        self.assertEqual(
            fsoi.baseline_dynamics_branch_status,
            "certified",
        )
        self.assertIsNotNone(
            fsoi.perturbation_diagnostics
            .baseline_dynamics_branch_signature_digest
        )
        self.assertIsNotNone(
            fsoi.observation.baseline_branch_trusted_total
        )
        crossing = torch.zeros_like(observations.dbz)
        crossing[full_index] = (
            linearization.frozen.nowcast_config.max_dbz
            - observations.dbz[full_index]
            + 0.1
        )
        with self.assertRaisesRegex(ValueError, "crosses input clamp"):
            VariationalObservationPerturbation.from_radar_dbz_delta(
                crossing,
                linearization,
            )

    def test_learning_policy_is_external_and_requires_physical_input(
        self,
    ) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        zeros = torch.zeros_like(observations.dbz)
        policy = AutomatedLearningPolicy(
            sensitivity_config=SensitivityConfig.for_automated_learning(
                radar_product_digest="1" * 64,
                qc_pipeline_digest="2" * 64,
            ),
            adjoint_config=(
                VariationalAdjointConfig.for_automated_learning()
            ),
            algorithm_bundle_digest=linearization.algorithm_bundle_digest,
            numerical_runtime_digest=linearization.numerical_runtime_digest,
        )
        for invalid_ratio in (0.0, 1.0, float("nan"), True):
            with self.subTest(linearity_relative_error=invalid_ratio):
                with self.assertRaisesRegex(ValueError, "must be in"):
                    replace(
                        policy,
                        maximum_linearity_relative_error=invalid_ratio,
                    )
        for invalid_value in (-1.0, float("nan"), True):
            with self.subTest(linearity_absolute_error=invalid_value):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    replace(
                        policy,
                        metric_taylor_thresholds=(
                            MetricTaylorThreshold(
                                "log_echo_mse",
                                invalid_value,
                                1.0e-6,
                            ),
                        ),
                    )
        for invalid_value in (0.0, float("nan"), True):
            with self.subTest(material_impact_threshold=invalid_value):
                with self.assertRaisesRegex(ValueError, "positive"):
                    replace(
                        policy,
                        metric_taylor_thresholds=(
                            MetricTaylorThreshold(
                                "log_echo_mse",
                                1.0e-6,
                                invalid_value,
                            ),
                        ),
                    )
        available = torch.ones((1, 1), dtype=torch.bool)
        prediction = torch.tensor(((1.0e-3,),), dtype=torch.float64)
        self.assertTrue(
            sensitivity_module._taylor_step_is_valid(
                prediction,
                prediction,
                available,
                policy,
            )
        )
        self.assertFalse(
            sensitivity_module._taylor_step_is_valid(
                0.5 * prediction,
                0.2 * prediction,
                available,
                policy,
            )
        )
        self.assertFalse(
            sensitivity_module._material_impact_signs_are_consistent(
                ((prediction, -prediction),),
                available,
                ("log_echo_mse",),
                policy,
            )
        )
        self.assertEqual(
            policy.threshold_for("centroid_error_m2").maximum_absolute_error,
            1.0,
        )
        self.assertEqual(
            sensitivity_module._material_impact_summary(
                ((torch.zeros_like(prediction), torch.zeros_like(prediction)),),
                available,
                ("log_echo_mse",),
                policy,
            ),
            (0, 0.0, 0.0),
        )
        physical_delta = zeros.clone()
        index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.detected_mask,
                as_tuple=False,
            )[0]
        )
        physical_delta[index] = 0.1
        physical = VariationalObservationPerturbation.from_radar_dbz_delta(
            physical_delta,
            linearization,
        )
        with self.assertRaisesRegex(TypeError, "approved_policy_digests"):
            compute_variational_fsoi_for_learning(
                self.forecast,
                self.analysis,
                self.verification,
                physical,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
                approved_policy_digests=frozenset((policy.digest,)),  # type: ignore[call-arg]
            )
        with patch.object(
            sensitivity_module,
            "_load_learning_policy_trust_store",
            return_value=sensitivity_module._LearningPolicyTrustStore(
                approved_policy_digests=frozenset(),
                content_digest="7" * 64,
            ),
        ):
            rejected = compute_variational_fsoi_for_learning(
                self.forecast,
                self.analysis,
                self.verification,
                physical,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertFalse(rejected.eligibility.eligible)
        self.assertEqual(
            rejected.eligibility.reasons,
            ("unapproved_learning_policy",),
        )

        augmented = VariationalObservationPerturbation(
            detected_dbz=zeros,
            censor_threshold_dbz=zeros,
            observation_weight=zeros,
        )
        with patch.object(
            sensitivity_module,
            "_load_learning_policy_trust_store",
            return_value=sensitivity_module._LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="7" * 64,
            ),
        ):
            rejected = compute_variational_fsoi_for_learning(
                self.forecast,
                self.analysis,
                self.verification,
                augmented,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertFalse(rejected.eligibility.eligible)
        self.assertEqual(
            rejected.eligibility.reasons,
            ("physical_radar_perturbation_required",),
        )

        wrong_algorithm = replace(
            policy,
            algorithm_bundle_digest="3" * 64,
        )
        with patch.object(
            sensitivity_module,
            "_load_learning_policy_trust_store",
            return_value=sensitivity_module._LearningPolicyTrustStore(
                approved_policy_digests=frozenset((wrong_algorithm.digest,)),
                content_digest="7" * 64,
            ),
        ):
            rejected = compute_variational_fsoi_for_learning(
                self.forecast,
                self.analysis,
                self.verification,
                physical,
                policy=wrong_algorithm,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertEqual(
            rejected.eligibility.reasons,
            ("algorithm_bundle_not_approved",),
        )

    def test_approved_learning_policy_certifies_physical_branch(self) -> None:
        grid = RadarGridTimeContract(
            valid_times=(
                "2026-08-05T00:00:00Z",
                "2026-08-05T00:10:00Z",
                "2026-08-05T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="4" * 64,
        )
        forecast, analysis = variational_nowcast(
            self.frames,
            nowcast_config=self.nowcast_config,
            analysis_config=self.analysis_config,
            grid_time_contract=grid,
        )
        linearization = analysis.linearization
        assert linearization is not None
        verification_frames = torch.where(
            torch.isfinite(forecast.forecast_dbz),
            forecast.forecast_dbz - 0.5,
            forecast.forecast_dbz,
        )
        verification = VerificationBundle(
            frames_dbz=verification_frames,
            valid_mask=torch.isfinite(verification_frames),
            valid_times=("2026-08-05T00:30:00Z",),
            grid_contract_digest=grid.digest,
            radar_product_digest="5" * 64,
            qc_pipeline_digest="6" * 64,
        )
        policy = AutomatedLearningPolicy(
            sensitivity_config=replace(
                SensitivityConfig.for_automated_learning(
                    radar_product_digest="5" * 64,
                    qc_pipeline_digest="6" * 64,
                ),
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=4,
            ),
            adjoint_config=(
                replace(
                    VariationalAdjointConfig.for_automated_learning(),
                    lead_minutes=(10,),
                )
            ),
            algorithm_bundle_digest=linearization.algorithm_bundle_digest,
            numerical_runtime_digest=linearization.numerical_runtime_digest,
            metric_taylor_thresholds=(
                MetricTaylorThreshold(
                    "log_echo_mse",
                    1.0e-6,
                    1.0e-8,
                ),
            ),
        )
        delta = torch.zeros_like(linearization.observations.dbz)
        index = tuple(
            int(value)
            for value in torch.nonzero(
                linearization.observations.detected_mask,
                as_tuple=False,
            )[0]
        )
        nonlinear_delta = torch.zeros_like(linearization.observations.dbz)
        nonlinear_delta[index] = 0.01
        with patch.object(
            sensitivity_module,
            "_load_learning_policy_trust_store",
            return_value=sensitivity_module._LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="7" * 64,
            ),
        ):
            nonlinear = compute_variational_fsoi_for_learning(
                forecast,
                analysis,
                verification,
                VariationalObservationPerturbation.from_radar_dbz_delta(
                    nonlinear_delta,
                    linearization,
                ),
                policy=policy,
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
            )
        self.assertFalse(nonlinear.eligibility.eligible)
        self.assertEqual(
            nonlinear.eligibility.reasons,
            ("first_order_validation_failed",),
        )
        assert nonlinear.first_order_validation is not None
        self.assertFalse(nonlinear.first_order_validation.first_order_valid)

        delta[index] = 5.0e-6
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            linearization,
        )
        resolved_change = (
            nonlinear.first_order_validation
            .full_step_resolved_metric_change
        )
        certified_validation = replace(
            nonlinear.first_order_validation,
            full_step_prediction=resolved_change,
            full_step_absolute_error=torch.zeros_like(resolved_change),
            half_step_prediction=0.5 * resolved_change,
            half_step_resolved_metric_change=0.5 * resolved_change,
            half_step_absolute_error=torch.zeros_like(resolved_change),
            full_step_valid=True,
            half_step_valid=True,
            sign_consistent_for_material_impacts=True,
            material_metric_count=1,
            maximum_material_impact=float(torch.amax(torch.abs(resolved_change))),
            aggregate_material_impact_norm=float(
                torch.linalg.vector_norm(resolved_change)
            ),
            first_order_valid=True,
        )

        with (
            patch.object(
                sensitivity_module,
                "_load_learning_policy_trust_store",
                return_value=sensitivity_module._LearningPolicyTrustStore(
                    approved_policy_digests=frozenset((policy.digest,)),
                    content_digest="7" * 64,
                ),
            ),
            patch.object(
                sensitivity_module,
                "_validate_first_order_learning_impact",
                return_value=certified_validation,
            ),
        ):
            learning = compute_variational_fsoi_for_learning(
                forecast,
                analysis,
                verification,
                perturbation,
                policy=policy,
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
            )

        self.assertTrue(
            learning.eligibility.eligible,
            (
                learning.eligibility.reasons,
                learning.first_order_validation,
            ),
        )
        self.assertEqual(learning.eligibility.reasons, ())
        assert learning.fsoi is not None
        assert learning.first_order_validation is not None
        self.assertTrue(learning.first_order_validation.first_order_valid)
        self.assertTrue(
            learning.first_order_validation
            .full_step_resolved_analysis_converged
        )
        self.assertTrue(
            learning.first_order_validation
            .half_step_resolved_analysis_converged
        )
        self.assertTrue(learning.first_order_validation.full_step_valid)
        self.assertTrue(learning.first_order_validation.half_step_valid)
        self.assertTrue(
            learning.first_order_validation
            .sign_consistent_for_material_impacts
        )
        self.assertEqual(
            learning.first_order_validation.metric_domain_contract,
            "frozen_metric_domain",
        )
        self.assertTrue(learning.first_order_validation.active_branch_valid)
        self.assertEqual(
            learning.fsoi.baseline_dynamics_branch_status,
            "certified",
        )
        self.assertEqual(
            learning.fsoi.perturbation_diagnostics.perturbed_area_km2,
            1.0,
        )
        self.assertEqual(learning.fsoi.fso.tile_size, 16)
        self.assertIsNotNone(learning.frozen_domain_learning_impact)
        assert learning.approval_evidence is not None
        self.assertEqual(
            learning.approval_evidence.trust_store_digest,
            "7" * 64,
        )
        validate_variational_learning_impact(
            learning,
            expected_trust_store_digest="7" * 64,
        )
        assert learning.fsoi is not None
        weaker_delta = 0.5 * delta
        candidates = (
            ("stronger", perturbation),
            (
                "weaker",
                VariationalObservationPerturbation.from_radar_dbz_delta(
                    weaker_delta,
                    linearization,
                ),
            ),
        )
        ranking_fso = compute_variational_fso(
            forecast,
            analysis,
            verification,
            sensitivity_config=policy.sensitivity_config,
            adjoint_config=policy.ranking_adjoint_config,
        )
        ranking = score_candidate_perturbations(
            ranking_fso,
            candidates,
            policy=policy,
        )
        self.assertEqual(
            tuple(score.candidate_id for score in ranking.scores),
            ("stronger", "weaker"),
        )
        with self.assertRaisesRegex(ValueError, "policy budget"):
            validate_top_k_learning_impacts(
                forecast,
                analysis,
                verification,
                ranking,
                policy=replace(policy, maximum_learning_resolves=1),
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
                maximum_resolves=2,
            )
        with (
            patch.object(
                sensitivity_module,
                "_load_learning_policy_trust_store",
                return_value=sensitivity_module._LearningPolicyTrustStore(
                    approved_policy_digests=frozenset((policy.digest,)),
                    content_digest="7" * 64,
                ),
            ),
            patch.object(
                sensitivity_module,
                "_learning_impact_from_fsoi",
                return_value=sensitivity_module._rejected_learning_impact(
                    policy,
                    "test_validation_stub",
                ),
            ) as validate_mock,
            patch.object(
                sensitivity_module,
                "pcg",
                side_effect=AssertionError("adjoint must not be recomputed"),
            ),
        ):
            top = validate_top_k_learning_impacts(
                forecast,
                analysis,
                verification,
                ranking,
                policy=policy,
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
                maximum_resolves=1,
            )
        self.assertEqual(len(top), 1)
        self.assertEqual(validate_mock.call_count, 1)
        validated_fsoi = validate_mock.call_args.args[3]
        torch.testing.assert_close(
            validated_fsoi.observation.total.sum_by_time.sum(dim=-1),
            ranking.scores[0].predicted_metric_change,
        )
        ranking.scores[0].predicted_metric_change.add_(1.0)
        with self.assertRaisesRegex(ValueError, "ranking digest"):
            validate_top_k_learning_impacts(
                forecast,
                analysis,
                verification,
                ranking,
                policy=policy,
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
                maximum_resolves=1,
            )
        with tempfile.TemporaryDirectory() as temporary:
            ledger = EpisodeLedger(temporary)
            stored_digest = ledger.append_variational_learning_approval(
                learning
            )
            stored = ledger.load_variational_learning_approval(stored_digest)
        self.assertEqual(stored, learning.approval_evidence)
        no_material = replace(
            certified_validation,
            material_metric_count=0,
            maximum_material_impact=0.0,
            aggregate_material_impact_norm=0.0,
            first_order_valid=False,
        )
        with (
            patch.object(
                sensitivity_module,
                "_load_learning_policy_trust_store",
                return_value=sensitivity_module._LearningPolicyTrustStore(
                    approved_policy_digests=frozenset((policy.digest,)),
                    content_digest="7" * 64,
                ),
            ),
            patch.object(
                sensitivity_module,
                "_validate_first_order_learning_impact",
                return_value=no_material,
            ),
        ):
            rejected = compute_variational_fsoi_for_learning(
                forecast,
                analysis,
                verification,
                perturbation,
                policy=policy,
                policy_trust_store_path=(
                    "/etc/advar/learning-policies.json"
                ),
            )
        self.assertEqual(
            rejected.eligibility.reasons,
            ("no_material_learning_signal",),
        )
        learning.first_order_validation.full_step_absolute_error.add_(1.0)
        with self.assertRaisesRegex(ValueError, "validation digest"):
            validate_variational_learning_impact(learning)

    def test_learning_policy_trust_store_requires_root_ownership(self) -> None:
        digest = "a" * 64
        document = {
            "contract": "advar-learning-policy-trust-store-v1",
            "approved_policy_digests": [digest],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "learning-policies.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            real_fstat = os.fstat

            with patch.object(
                sensitivity_module.os,
                "fstat",
                side_effect=lambda descriptor: SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o644,
                    st_uid=1000,
                    st_size=real_fstat(descriptor).st_size,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "root-owned"):
                    sensitivity_module._load_learning_policy_trust_store(
                        path
                    )
            with patch.object(
                sensitivity_module.os,
                "fstat",
                side_effect=lambda descriptor: SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o644,
                    st_uid=0,
                    st_size=real_fstat(descriptor).st_size,
                ),
            ):
                store = (
                    sensitivity_module
                    ._load_learning_policy_trust_store(path)
                )
                self.assertEqual(
                    store.approved_policy_digests,
                    frozenset((digest,)),
                )
                self.assertEqual(len(store.content_digest), 64)

    def test_censored_perturbations_use_event_specific_factories(self) -> None:
        linearization = self.analysis.linearization
        assert linearization is not None
        observations = linearization.observations
        delta = torch.zeros_like(observations.dbz)
        index = tuple(
            int(value)
            for value in torch.nonzero(
                observations.censored_mask,
                as_tuple=False,
            )[0]
        )
        delta[index] = 0.1

        with self.assertRaisesRegex(ValueError, "detected observations"):
            VariationalObservationPerturbation.from_radar_dbz_delta(
                delta,
                linearization,
            )

        threshold = (
            VariationalObservationPerturbation.from_censor_threshold_delta(
                delta,
                linearization,
            )
        )
        weight = (
            VariationalObservationPerturbation.from_censored_event_weight_delta(
                delta,
                linearization,
            )
        )
        self.assertEqual(float(threshold.censor_threshold_dbz[index]), 0.1)
        self.assertEqual(float(weight.observation_weight[index]), 0.1)
        self.assertEqual(float(threshold.detected_dbz[index]), 0.0)
        self.assertEqual(float(weight.detected_dbz[index]), 0.0)

    def test_variational_fso_fails_closed(self) -> None:
        p0_forecast = nowcast(self.frames, self.nowcast_config)
        with self.assertRaisesRegex(ValueError, "accepted P1"):
            compute_variational_fso(
                p0_forecast,
                self.analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        with self.assertRaisesRegex(ValueError, "converged P1"):
            compute_variational_fso(
                self.forecast,
                replace(self.analysis, degraded=True),
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        linearization = self.analysis.linearization
        assert linearization is not None
        changed_std = linearization.observations.std_dbz.clone()
        changed_std[0, 0, 0] += 0.1
        changed_observations = replace(
            linearization.observations,
            std_dbz=changed_std,
        )
        changed_analysis = replace(
            self.analysis,
            linearization=replace(
                linearization,
                observations=changed_observations,
            ),
        )
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            compute_variational_fso(
                self.forecast,
                changed_analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        changed_run_analysis = replace(
            self.analysis,
            linearization=replace(
                linearization,
                forecast_run_digest="0" * 64,
            ),
        )
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            compute_variational_fso(
                self.forecast,
                changed_run_analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        changed_stationarity_analysis = replace(
            self.analysis,
            linearization=replace(
                linearization,
                gradient_norm=linearization.gradient_norm + 1.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            compute_variational_fso(
                self.forecast,
                changed_stationarity_analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        changed_analysis_diagnostics = replace(
            self.analysis,
            linearization_gradient_norm=(
                self.analysis.linearization_gradient_norm or 0.0
            )
            + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "diagnostics mismatch"):
            compute_variational_fso(
                self.forecast,
                changed_analysis_diagnostics,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )

        failed_pcg = PCGResult(
            solution=torch.zeros_like(self.analysis.control),
            converged=False,
            iterations=1,
            relative_residual=1.0,
        )
        with patch("advar.sensitivity.pcg", return_value=failed_pcg):
            with self.assertRaisesRegex(ValueError, "did not converge"):
                compute_variational_fso(
                    self.forecast,
                    self.analysis,
                    self.verification,
                    sensitivity_config=self.sensitivity_config,
                )

        unpolished_forecast, unpolished_analysis = variational_nowcast(
            self.frames,
            nowcast_config=self.nowcast_config,
            analysis_config=replace(
                self.analysis_config,
                maximum_final_linearization_polish_iterations=0,
            ),
        )
        self.assertFalse(unpolished_analysis.final_linearization_stationary)
        self.assertFalse(unpolished_analysis.fso_eligible)
        self.assertTrue(unpolished_analysis.degraded)
        self.assertIsNone(unpolished_analysis.linearization)
        self.assertGreater(
            unpolished_analysis.linearization_relative_stationarity or 0.0,
            self.analysis_config
            .final_linearization_relative_stationarity_tolerance,
        )
        with self.assertRaisesRegex(ValueError, "converged P1 analysis"):
            compute_variational_fso(
                unpolished_forecast,
                unpolished_analysis,
                self.verification,
                sensitivity_config=self.sensitivity_config,
            )


if __name__ == "__main__":
    unittest.main()
