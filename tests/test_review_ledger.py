"""Focused regression tests for the bounded ledger review findings."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from advar.ledger import EpisodeLedger, SensitivityEpisode
from advar.nowcast import NowcastConfig, nowcast
from advar.sensitivity import SensitivityConfig, compute_sensitivity_snapshot
from test_ledger import _computed_snapshot, _contract


class LedgerReviewRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()

    def _episode(self, snapshot, episode_id: str) -> SensitivityEpisode:
        return SensitivityEpisode(
            episode_id=episode_id,
            issue_time="2026-07-26T05:00:00+00:00",
            radar_id="KTLX",
            contract=_contract(snapshot),
            snapshot=snapshot,
        )

    def _append(self, snapshot, episode_id: str) -> tuple[EpisodeLedger, object]:
        temporary = tempfile.TemporaryDirectory()
        ledger = EpisodeLedger(Path(temporary.name))
        target = ledger.append(self._episode(snapshot, episode_id))
        self.addCleanup(temporary.cleanup)
        return ledger, target

    def test_unavailable_nan_evidence_roundtrips(self) -> None:
        source = self.snapshot
        unavailable = torch.zeros_like(source.metric_available)
        nan_scores = torch.full_like(source.forecast_scores, float("nan"))
        nan_control = torch.full_like(source.control_sensitivity, float("nan"))
        nan_evidence = torch.full_like(
            source.path_evidence_by_metric,
            float("nan"),
        )
        nan_forecast_maps = torch.full_like(
            source.forecast_sensitivity,
            float("nan"),
        )
        direct = replace(
            source.direct,
            maps=torch.full_like(source.direct.maps, float("nan")),
            norm=torch.full_like(source.direct.norm, float("nan")),
            tile_norm=torch.full_like(source.direct.tile_norm, float("nan")),
            whitened_tile_norm=(
                None
                if source.direct.whitened_tile_norm is None
                else torch.full_like(
                    source.direct.whitened_tile_norm,
                    float("nan"),
                )
            ),
            impact=None,
            tile_impact=None,
            reward=None,
        )
        snapshot = replace(
            source,
            metric_available=unavailable,
            forecast_scores=nan_scores,
            control_sensitivity=nan_control,
            forecast_sensitivity=nan_forecast_maps,
            path_evidence_by_metric=nan_evidence,
            observation_source_fraction_by_metric=nan_evidence.clone(),
            observation_verified_evidence_by_metric=nan_evidence.clone(),
            background_verified_evidence_by_metric=nan_evidence.clone(),
            direct=direct,
        )

        ledger, _ = self._append(snapshot, "review-unavailable-nan")
        ledger.verify("review-unavailable-nan")
        loaded = ledger.load("review-unavailable-nan")

        self.assertTrue(
            bool(torch.isnan(torch.from_numpy(loaded.arrays["path_evidence_by_metric"])).all())
        )
        self.assertFalse(bool(loaded.manifest["impact_available"]))

    def test_infinite_evidence_is_rejected_for_each_channel(self) -> None:
        channels = (
            "path_evidence_by_metric",
            "observation_source_fraction_by_metric",
            "observation_verified_evidence_by_metric",
            "background_verified_evidence_by_metric",
        )
        for index, channel in enumerate(channels):
            with self.subTest(channel=channel):
                value = getattr(self.snapshot, channel).clone()
                value[0, 0] = float("inf")
                snapshot = replace(self.snapshot, **{channel: value})
                with tempfile.TemporaryDirectory() as directory:
                    ledger = EpisodeLedger(Path(directory))
                    with self.assertRaisesRegex(ValueError, "metric evidence"):
                        ledger.append(
                            self._episode(snapshot, f"review-inf-{index}")
                        )

    def test_producer_evidence_channels_reject_mixed_nan(self) -> None:
        config = NowcastConfig()
        frames = torch.full((3, 2, 2), 20.0, dtype=torch.float64)
        background = frames - 0.5
        result = nowcast(
            frames,
            config,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        snapshot = compute_sensitivity_snapshot(
            frames[-1],
            result,
            frames.new_full((config.forecast_steps, 2, 2), 20.5),
            latest_background_dbz=background[-1],
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
            ),
        )
        channels = (
            "path_evidence_by_metric",
            "observation_source_fraction_by_metric",
            "observation_verified_evidence_by_metric",
            "background_verified_evidence_by_metric",
        )
        for name in channels:
            self.assertTrue(bool(torch.isfinite(getattr(snapshot, name)).all()))

        ledger, _ = self._append(snapshot, "review-finite-evidence")
        ledger.verify("review-finite-evidence")
        loaded = ledger.load("review-finite-evidence")
        for name in channels:
            self.assertTrue(bool(torch.isfinite(torch.from_numpy(loaded.arrays[name])).all()))

        for index, name in enumerate(channels):
            with self.subTest(channel=name):
                value = getattr(snapshot, name).clone()
                value[0, 0] = float("nan")
                mixed = replace(snapshot, **{name: value})
                with tempfile.TemporaryDirectory() as directory:
                    ledger = EpisodeLedger(Path(directory))
                    with self.assertRaisesRegex(ValueError, "evidence"):
                        ledger.append(
                            self._episode(mixed, f"review-mixed-evidence-{index}")
                        )

    def test_available_all_nan_evidence_roundtrips_zero_gradient_snapshot(self) -> None:
        snapshot = self.snapshot
        channels = (
            "path_evidence_by_metric",
            "observation_source_fraction_by_metric",
            "observation_verified_evidence_by_metric",
            "background_verified_evidence_by_metric",
        )
        self.assertTrue(bool(torch.any(snapshot.metric_available)))
        for name in channels:
            self.assertTrue(bool(torch.isnan(getattr(snapshot, name)).all()))

        ledger, _ = self._append(snapshot, "review-available-all-nan")
        ledger.verify("review-available-all-nan")
        loaded = ledger.load("review-available-all-nan")
        self.assertTrue(bool(loaded.manifest["impact_available"]))
        for name in channels:
            self.assertTrue(bool(torch.isnan(torch.from_numpy(loaded.arrays[name])).all()))

    def test_trust_components_outside_unit_interval_are_rejected(self) -> None:
        for index, invalid in enumerate((2.0, -1.0, True)):
            with self.subTest(invalid=invalid):
                components = {name: 1.0 for name in self.snapshot.trust_components}
                components["linearity"] = invalid
                snapshot = replace(
                    self.snapshot,
                    trust_components=components,
                    trust_score=math.prod(components.values()),
                )
                with tempfile.TemporaryDirectory() as directory:
                    ledger = EpisodeLedger(Path(directory))
                    with self.assertRaises(ValueError):
                        ledger.append(
                            self._episode(snapshot, f"review-trust-{index}")
                        )

    def test_trust_score_tiny_out_of_range_is_rejected(self) -> None:
        components = {name: 1.0 for name in self.snapshot.trust_components}
        for invalid in (1.0 + 5.0e-13, True):
            with self.subTest(invalid=invalid):
                snapshot = replace(
                    self.snapshot,
                    trust_components=components,
                    trust_score=invalid,
                )
                with tempfile.TemporaryDirectory() as directory:
                    ledger = EpisodeLedger(Path(directory))
                    with self.assertRaisesRegex(ValueError, "trust_score"):
                        ledger.append(self._episode(snapshot, "review-trust-score"))


if __name__ == "__main__":
    unittest.main()
