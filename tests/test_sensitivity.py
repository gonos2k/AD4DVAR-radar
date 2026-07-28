from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.nowcast import (  # noqa: E402
    DataStatus,
    ForecastMetadata,
    ForecastResult,
    NowcastConfig,
    RadarState,
    TendencySource,
    forecast_from_state,
    forecast_linear_at_step,
    forecast_linear_from_state,
)
from advar.physics import dbz_to_echo, echo_to_dbz  # noqa: E402
from advar.sensitivity import (  # noqa: E402
    SensitivityConfig,
    compute_sensitivity_snapshot,
    forecast_metric,
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
        source_support=None,
        motion_disagreement_px=torch.linalg.vector_norm(
            pair_motion[1] - pair_motion[0]
        ),
        growth_disagreement=torch.abs(pair_growth[1] - pair_growth[0]),
        tendency_pair_count=2,
        tendency_source=TendencySource.OBSERVATION,
    )


def result_for(
    state: RadarState,
    config: NowcastConfig,
    *,
    pair_motion: torch.Tensor | None = None,
    pair_growth: torch.Tensor | None = None,
) -> ForecastResult:
    return forecast_from_state(
        state,
        metadata_for(
            state,
            pair_motion=pair_motion,
            pair_growth=pair_growth,
        ),
        config,
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
        cls.result = result_for(
            cls.state,
            cls.nowcast_config,
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
        baseline_scores = torch.ones(
            cls.nowcast_config.forecast_steps,
            1,
            dtype=torch.float64,
        )
        common = {
            "nowcast_config": cls.nowcast_config,
            "sensitivity_config": cls.sensitivity_config,
            "observation_std_dbz": 2.0,
            "baseline_scores": baseline_scores,
            "qc_mask": cls.qc_mask,
        }
        cls.snapshot = compute_sensitivity_snapshot(
            cls.frames,
            cls.result,
            cls.verification,
            background_frames_dbz=cls.background,
            **common,
        )
        cls.snapshot_without_background = compute_sensitivity_snapshot(
            cls.frames,
            cls.result,
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
        result = result_for(state, config)
        torch.testing.assert_close(
            result.forecast_dbz,
            linear_to_dbz(linear, config),
        )
        torch.testing.assert_close(
            dbz_to_linear(result.forecast_dbz, config),
            linear,
        )

    def test_snapshot_shapes_and_m0_scope(self) -> None:
        snapshot = self.snapshot
        tile_rows = math.ceil(self.height / self.sensitivity_config.tile_size)
        tile_columns = math.ceil(
            self.width / self.sensitivity_config.tile_size
        )

        self.assertEqual(snapshot.lead_minutes, tuple(range(10, 181, 10)))
        self.assertEqual(snapshot.full_map_lead_minutes, (10,))
        self.assertEqual(snapshot.context_features.shape, (15,))
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
            snapshot.direct_observation_sensitivity.shape,
            (1, 1, 3, self.height, self.width),
        )
        self.assertEqual(
            snapshot.direct_observation_sensitivity_norm.shape,
            (18, 1, 3),
        )
        self.assertEqual(
            snapshot.tile_direct_sensitivity_norm.shape,
            (18, 1, 3, tile_rows, tile_columns),
        )
        self.assertEqual(
            snapshot.direct_observation_impact.shape,
            (18, 1, 3),
        )
        self.assertEqual(snapshot.latest_sensitivity_mask.shape, (13, 17))
        self.assertEqual(
            snapshot.observation_innovation_mask.shape,
            (3, 13, 17),
        )
        self.assertTrue(snapshot.whitened_tile_norm_available)
        self.assertFalse(snapshot.indirect_observation_sensitivity_available)
        self.assertFalse(snapshot.promotion_eligible)
        torch.testing.assert_close(
            snapshot.direct_observation_sensitivity[:, :, :2],
            torch.zeros_like(
                snapshot.direct_observation_sensitivity[:, :, :2]
            ),
        )
        torch.testing.assert_close(
            snapshot.direct_observation_sensitivity_norm[:, :, :2],
            torch.zeros_like(
                snapshot.direct_observation_sensitivity_norm[:, :, :2]
            ),
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
        gradient = self.snapshot.direct_observation_sensitivity[0, 0, 2]

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
        tile_sum = self.snapshot.tile_direct_observation_impact.sum(
            dim=(-1, -2)
        )

        self.assertTrue(bool(torch.all(torch.isfinite(tile_sum))))
        torch.testing.assert_close(
            tile_sum,
            self.snapshot.direct_observation_impact,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    def test_missing_background_marks_impacts_unavailable(self) -> None:
        snapshot = self.snapshot_without_background

        self.assertFalse(snapshot.impact_available)
        self.assertFalse(snapshot.reward_available)
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.observation_innovation_dbz)))
        )
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.direct_observation_impact)))
        )
        self.assertTrue(
            bool(
                torch.all(
                    torch.isnan(snapshot.tile_direct_observation_impact)
                )
            )
        )
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.direct_normalized_reward)))
        )

    def test_sensitivity_scores_the_issued_capped_forecast(self) -> None:
        config = self.nowcast_config
        frames = torch.full((3, 8, 9), config.max_dbz, dtype=torch.float64)
        state = RadarState(
            echo_linear=dbz_to_linear(frames[2], config),
            displacement_yx=torch.zeros(2, dtype=torch.float64),
            log_growth_per_step=torch.tensor(
                config.max_log_growth_per_step,
                dtype=torch.float64,
            ),
        )
        result = result_for(state, config)
        verification = torch.full(
            (config.forecast_steps, 8, 9),
            config.max_dbz,
            dtype=torch.float64,
        )
        snapshot = compute_sensitivity_snapshot(
            frames,
            result,
            verification,
            nowcast_config=config,
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
            self.frames,
            self.result,
            verification,
            nowcast_config=self.nowcast_config,
            sensitivity_config=self.sensitivity_config,
            background_frames_dbz=self.background,
            qc_mask=self.qc_mask,
        )

        self.assertFalse(bool(torch.any(snapshot.metric_available)))
        self.assertTrue(bool(torch.all(torch.isnan(snapshot.forecast_scores))))
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.control_sensitivity)))
        )
        self.assertFalse(snapshot.impact_available)

    def test_nonfinite_background_does_not_create_fake_innovation(self) -> None:
        background = torch.full_like(self.frames, float("nan"))
        snapshot = compute_sensitivity_snapshot(
            self.frames,
            self.result,
            self.verification,
            nowcast_config=self.nowcast_config,
            sensitivity_config=self.sensitivity_config,
            background_frames_dbz=background,
            qc_mask=self.qc_mask,
        )

        self.assertFalse(snapshot.impact_available)
        self.assertFalse(
            bool(torch.any(snapshot.observation_innovation_mask))
        )
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.observation_innovation_dbz)))
        )
        self.assertTrue(
            bool(torch.all(torch.isnan(snapshot.direct_observation_impact)))
        )

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
            {"tile_size": 2.5},
            {"soft_fss_window": 4.5},
            {"soft_fss_temperature_dbz": float("nan")},
            {"active_margin_dbz": float("nan")},
            {"linearity_delta": (0.1, 0.2)},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    SensitivityConfig(**values)


if __name__ == "__main__":
    unittest.main()
