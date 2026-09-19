"""Small independent scenario matrix for the phase-one nowcast contract."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.nowcast import (  # noqa: E402
    DataStatus,
    NowcastConfig,
    RadarState,
    TendencyPairSelection,
    TendencySource,
    forecast_linear_from_state,
    nowcast,
    prepare_input,
)
from advar.physics import echo_to_dbz, remap  # noqa: E402


class Phase1NowcastScenarioTests(unittest.TestCase):
    config = NowcastConfig()

    @staticmethod
    def _frames() -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(24, dtype=torch.float64),
            torch.arange(24, dtype=torch.float64),
            indexing="ij",
        )
        base = 2.0e4 * torch.exp(
            -((y - 12.0).square() + (x - 12.0).square()) / 12.0
        )
        echoes = [
            remap(base, torch.tensor((step * 0.9, -step * 0.4)))
            for step in range(3)
        ]
        return echo_to_dbz(
            torch.stack(echoes),
            min_dbz=-10.0,
            max_dbz=70.0,
        )

    def _assert_finite_matches_valid(self, result) -> None:
        self.assertTrue(
            bool(torch.equal(torch.isfinite(result.forecast_dbz), result.valid_mask))
        )
        result.validate_issuance()

    @staticmethod
    def _rotating_echo_frames() -> torch.Tensor:
        """An anisotropic, asymmetric echo whose orientation changes by lead."""
        y, x = torch.meshgrid(
            torch.arange(24, dtype=torch.float64),
            torch.arange(24, dtype=torch.float64),
            indexing="ij",
        )

        def frame(angle: float) -> torch.Tensor:
            dy, dx = y - 12.0, x - 12.0
            cosine, sine = torch.cos(torch.tensor(angle)), torch.sin(
                torch.tensor(angle)
            )
            rotated_x = cosine * dx + sine * dy
            rotated_y = -sine * dx + cosine * dy
            major = 1.8e4 * torch.exp(
                -((rotated_x / 4.0).square() + (rotated_y / 1.4).square())
                / 2.0
            )
            asymmetric_lobe = 8.0e3 * torch.exp(
                -(((rotated_x + 2.7) / 2.0).square()
                  + ((rotated_y - 1.0) / 1.0).square())
                / 2.0
            )
            return major + asymmetric_lobe

        return echo_to_dbz(
            torch.stack(tuple(frame(angle) for angle in (0.0, 0.3, 0.6))),
            min_dbz=-10.0,
            max_dbz=70.0,
        )

    def _newcell_frames(self) -> torch.Tensor:
        base = self._to_echo(self._frames()[0])
        translation = torch.tensor((0.8, -0.3), dtype=torch.float64)
        return echo_to_dbz(
            torch.stack(
                (
                    base,
                    remap(base, translation),
                    remap(remap(base, translation), translation)
                    + 1.0e4
                    * torch.exp(
                        -(
                            (torch.arange(24, dtype=torch.float64)[:, None] - 4.0)
                            .square()
                            + (torch.arange(24, dtype=torch.float64)[None, :] - 4.0)
                            .square()
                        )
                        / 2.0
                    ),
                )
            ),
            min_dbz=-10.0,
            max_dbz=70.0,
        )

    def test_nonfinite_values_are_missing_and_qc_rejection_is_distinct(self) -> None:
        frames = self._frames()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                candidate = frames.clone()
                candidate[1, 5, 6] = value
                prepared = prepare_input(candidate, self.config)
                self.assertFalse(bool(prepared.observed_mask[1, 5, 6]))
                self.assertTrue(bool(prepared.missing_mask[1, 5, 6]))
                self.assertFalse(bool(prepared.qc_rejected_mask[1, 5, 6]))
                self.assertEqual(prepared.data_status, DataStatus.PARTIAL)
                self._assert_finite_matches_valid(nowcast(candidate))

        accepted = torch.ones_like(frames, dtype=torch.bool)
        accepted[1, 5, 6] = False
        prepared = prepare_input(frames, self.config, accepted_mask=accepted)
        self.assertFalse(bool(prepared.observed_mask[1, 5, 6]))
        self.assertFalse(bool(prepared.missing_mask[1, 5, 6]))
        self.assertTrue(bool(prepared.qc_rejected_mask[1, 5, 6]))
        self.assertEqual(prepared.data_status, DataStatus.PARTIAL)
        self._assert_finite_matches_valid(nowcast(frames, qc_mask=accepted))

    def test_mixed_missing_qc_and_background_preserve_source_semantics(self) -> None:
        frames = self._frames()
        frames[:, 10:13, 10:13] = float("nan")
        accepted = torch.ones_like(frames, dtype=torch.bool)
        accepted[1, 7, 7] = False
        background = torch.full_like(frames, 8.0)

        result = nowcast(
            frames,
            qc_mask=accepted,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        metadata = result.metadata
        self.assertEqual(metadata.data_status, DataStatus.PARTIAL)
        self.assertTrue(metadata.background_used)
        self.assertGreater(float(metadata.background_source_support[11, 11]), 0.0)
        self.assertGreater(float(metadata.observation_source_support[7, 7]), 0.0)
        torch.testing.assert_close(
            metadata.source_support,
            (metadata.observation_source_support + metadata.background_source_support)
            .clamp(0.0, 1.0),
        )
        expected_fraction = float(
            metadata.background_source_support.sum()
            / metadata.source_support.sum()
        )
        self.assertAlmostEqual(
            metadata.background_contribution_fraction,
            expected_fraction,
            places=6,
        )
        self._assert_finite_matches_valid(result)

    def test_all_qc_rejected_input_is_unavailable_without_support(self) -> None:
        frames = torch.full((3, 8, 8), 20.0, dtype=torch.float64)
        rejected = torch.zeros_like(frames, dtype=torch.bool)
        result = nowcast(frames, qc_mask=rejected)

        self.assertEqual(result.metadata.data_status, DataStatus.UNAVAILABLE)
        self.assertEqual(result.metadata.tendency_source, TendencySource.NONE)
        self.assertFalse(bool(torch.any(result.valid_mask)))
        self.assertTrue(bool(torch.all(torch.isnan(result.forecast_dbz))))
        self._assert_finite_matches_valid(result)

    def test_border_pulse_exits_without_wrap_for_all_eighteen_leads(self) -> None:
        pulse = torch.zeros(8, 8, dtype=torch.float64)
        pulse[0, 4] = 1.0e5
        state = RadarState(
            echo_linear=pulse,
            displacement_yx=torch.tensor((-0.5, 0.0), dtype=torch.float64),
            log_growth_per_step=torch.zeros((), dtype=torch.float64),
        )
        forecast = forecast_linear_from_state(state, self.config)

        self.assertEqual(forecast.shape, (18, 8, 8))
        self.assertTrue(bool(torch.all(torch.isfinite(forecast))))
        self.assertTrue(bool(torch.all(forecast >= 0.0)))
        self.assertAlmostEqual(float(forecast[0].sum()), 5.0e4, places=8)
        self.assertTrue(bool(torch.all(forecast[1:].sum(dim=(1, 2)) == 0.0)))
        self.assertEqual(float(forecast[:, -1, :].max()), 0.0)

    def test_coherent_translation_and_conflicting_motion_are_bounded(self) -> None:
        frames = self._frames()
        coherent = nowcast(frames)
        self.assertEqual(coherent.metadata.tendency_source, TendencySource.OBSERVATION)
        self.assertEqual(coherent.metadata.motion_pair_count, 2)
        self.assertFalse(coherent.metadata.motion_pair_conflict)
        torch.testing.assert_close(
            coherent.state.displacement_yx,
            torch.tensor((0.9, -0.4), dtype=torch.float64),
            atol=0.2,
            rtol=0.0,
        )
        self._assert_finite_matches_valid(coherent)

        base = self._to_echo(frames[0])
        echo = torch.stack(
            (
                base,
                remap(
                    base,
                    torch.tensor((2.0, 0.0), dtype=torch.float64),
                ),
                remap(
                    base,
                    torch.tensor((-2.0, 0.0), dtype=torch.float64),
                ),
            )
        )
        conflicting = nowcast(
            echo_to_dbz(echo, min_dbz=-10.0, max_dbz=70.0)
        )
        self.assertTrue(conflicting.metadata.motion_pair_conflict)
        self.assertEqual(
            conflicting.metadata.motion_pair_selection,
            TendencyPairSelection.EARLIER,
        )
        self.assertEqual(conflicting.metadata.motion_pair_count, 1)
        self.assertGreater(
            float(conflicting.forecast_velocity_uncertainty_mps),
            float(coherent.forecast_velocity_uncertainty_mps),
        )
        self._assert_finite_matches_valid(conflicting)

    def test_clear_sky_is_observed_but_all_missing_is_unavailable(self) -> None:
        clear = torch.full((3, 8, 8), self.config.min_dbz, dtype=torch.float64)
        clear_result = nowcast(clear, self.config)
        self.assertEqual(clear_result.metadata.data_status, DataStatus.OBSERVED)
        self.assertEqual(clear_result.metadata.tendency_source, TendencySource.NONE)
        self.assertTrue(bool(torch.all(clear_result.valid_mask)))
        self._assert_finite_matches_valid(clear_result)

        missing = torch.full_like(clear, float("nan"))
        missing_result = nowcast(missing, self.config)
        self.assertEqual(
            missing_result.metadata.data_status,
            DataStatus.UNAVAILABLE,
        )
        self.assertFalse(bool(torch.any(missing_result.valid_mask)))
        self.assertTrue(bool(torch.all(torch.isnan(missing_result.forecast_dbz))))
        self._assert_finite_matches_valid(missing_result)

    def test_rotation_and_newcell_scenarios_are_bounded_model_limitations(self) -> None:
        for name, frames in (
            ("anisotropic_rotation", self._rotating_echo_frames()),
            ("new_cell", self._newcell_frames()),
        ):
            with self.subTest(name=name):
                result = nowcast(frames, self.config)
                self.assertEqual(result.forecast_dbz.shape[0], 18)
                self.assertTrue(bool(torch.all(torch.isfinite(result.state.displacement_yx))))
                self.assertTrue(bool(torch.isfinite(result.state.log_growth_per_step)))
                self._assert_finite_matches_valid(result)

    @staticmethod
    def _to_echo(dbz: torch.Tensor) -> torch.Tensor:
        from advar.physics import dbz_to_echo

        return dbz_to_echo(dbz, min_dbz=-10.0, max_dbz=70.0)


if __name__ == "__main__":
    unittest.main()
