from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from advar.ledger import EpisodeLedger, SensitivityEpisode
from test_ledger import _computed_snapshot, _contract


class LedgerSnapshotA2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _computed_snapshot()

    def _episode(
        self,
        snapshot,
        episode_id: str,
    ) -> SensitivityEpisode:
        return SensitivityEpisode(
            episode_id=episode_id,
            issue_time="2026-07-26T05:00:00+00:00",
            radar_id="KTLX",
            contract=_contract(snapshot),
            snapshot=snapshot,
        )

    def test_computed_snapshot_public_round_trip_preserves_nan_evidence(self) -> None:
        episode = self._episode(self.snapshot, "a2-valid-computed")
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            target = ledger.append(episode)

            self.assertTrue(target.is_dir())
            ledger.verify(episode.episode_id)
            loaded = ledger.load(episode.episode_id)

            self.assertEqual(
                loaded.manifest["context_feature_names"],
                list(self.snapshot.context_feature_names),
            )
            self.assertTrue(
                torch.isnan(
                    torch.from_numpy(loaded.arrays["path_evidence_by_metric"])
                ).all()
            )

    def test_noncanonical_context_names_are_rejected_before_publication(self) -> None:
        bad_names = tuple(
            f"wrong_{index}"
            for index in range(len(self.snapshot.context_feature_names))
        )
        snapshot = replace(self.snapshot, context_feature_names=bad_names)
        episode = self._episode(snapshot, "a2-bad-context-names")

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with self.assertRaisesRegex(ValueError, "current schema"):
                ledger.append(episode)
            self.assertFalse(
                (Path(directory) / "episodes" / episode.episode_id).exists()
            )

    def test_unavailable_tile_impact_must_be_nan(self) -> None:
        source = self.snapshot
        unavailable = torch.zeros_like(source.metric_available)
        unavailable[0, 0] = True
        available = ~unavailable

        def mask_metric(value: torch.Tensor) -> torch.Tensor:
            masked = value.clone()
            masked[unavailable] = float("nan")
            return masked

        direct = replace(
            source.direct,
            norm=mask_metric(source.direct.norm),
            tile_norm=mask_metric(source.direct.tile_norm),
            impact=mask_metric(source.direct.impact),
            tile_impact=source.direct.tile_impact.clone(),
        )
        assert direct.tile_impact is not None
        direct.tile_impact[unavailable] = 17.0
        snapshot = replace(
            source,
            metric_available=available,
            forecast_scores=mask_metric(source.forecast_scores),
            control_sensitivity=mask_metric(source.control_sensitivity),
            path_evidence_by_metric=mask_metric(
                source.path_evidence_by_metric
            ),
            observation_source_fraction_by_metric=mask_metric(
                source.observation_source_fraction_by_metric
            ),
            observation_verified_evidence_by_metric=mask_metric(
                source.observation_verified_evidence_by_metric
            ),
            background_verified_evidence_by_metric=mask_metric(
                source.background_verified_evidence_by_metric
            ),
            direct=direct,
        )
        episode = self._episode(snapshot, "a2-finite-unavailable-tile-impact")

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with self.assertRaisesRegex(ValueError, "unavailable tile impacts"):
                ledger.append(episode)

    def test_whitened_tile_norm_checks_only_retained_maps(self) -> None:
        source = self.snapshot
        maps = source.direct.maps.clone()
        maps[0, 0, 0, 0] = 3.0
        norm = source.direct.norm.clone()
        norm[2, 0] = 3.0
        tile_norm = source.direct.tile_norm.clone()
        tile_norm[2, 0, 0, 0] = 3.0

        std = torch.ones_like(source.latest_sensitivity_mask, dtype=maps.dtype)
        whitened = torch.zeros_like(tile_norm)
        for position, lead in enumerate(source.full_map_lead_minutes):
            lead_index = source.lead_minutes.index(lead)
            whitened[lead_index, 0, 0, 0] = torch.linalg.vector_norm(
                maps[position, 0] * std
            )
        valid_snapshot = replace(
            source,
            direct=replace(
                source.direct,
                maps=maps,
                norm=norm,
                tile_norm=tile_norm,
                whitened_tile_norm=whitened,
            ),
            observation_std_dbz=std,
        )
        bad_whitened = whitened.clone()
        bad_whitened[2, 0, 0, 0] = 999.0
        bad_snapshot = replace(
            valid_snapshot,
            direct=replace(
                valid_snapshot.direct,
                whitened_tile_norm=bad_whitened,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            valid_episode = self._episode(
                valid_snapshot,
                "a2-valid-whitened-tile-norm",
            )
            ledger.append(valid_episode)
            ledger.verify(valid_episode.episode_id)
            self.assertIsNotNone(ledger.load(valid_episode.episode_id))
            episode = self._episode(
                bad_snapshot,
                "a2-bad-whitened-tile-norm",
            )
            with self.assertRaisesRegex(
                ValueError,
                "whitened tile norms and retained direct maps",
            ):
                ledger.append(episode)

    def test_zero_spatial_snapshot_is_rejected(self) -> None:
        source = self.snapshot
        lead_count = len(source.lead_minutes)
        metric_count = len(source.metric_names)
        selected_count = len(source.full_map_lead_minutes)
        dtype = source.forecast_scores.dtype

        def nan_tensor(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.full(shape, float("nan"), dtype=dtype)

        zero_direct = replace(
            source.direct,
            maps=nan_tensor((selected_count, metric_count, 0, 0)),
            norm=nan_tensor((lead_count, metric_count)),
            tile_norm=nan_tensor((lead_count, metric_count, 0, 0)),
            whitened_tile_norm=None,
            impact=None,
            tile_impact=None,
            reward=None,
        )
        snapshot = replace(
            source,
            metric_available=torch.zeros_like(source.metric_available),
            forecast_scores=nan_tensor(source.forecast_scores.shape),
            control_sensitivity=nan_tensor(source.control_sensitivity.shape),
            forecast_sensitivity=nan_tensor(
                (selected_count, metric_count, 0, 0)
            ),
            forecast_cap_active_mask=torch.zeros(
                (selected_count, 0, 0), dtype=torch.bool
            ),
            forecast_confidence=torch.empty(
                (lead_count, 0, 0), dtype=dtype
            ),
            path_evidence_by_metric=nan_tensor(
                (lead_count, metric_count)
            ),
            observation_source_fraction_by_metric=nan_tensor(
                (lead_count, metric_count)
            ),
            observation_verified_evidence_by_metric=nan_tensor(
                (lead_count, metric_count)
            ),
            background_verified_evidence_by_metric=nan_tensor(
                (lead_count, metric_count)
            ),
            direct=zero_direct,
            latest_sensitivity_mask=torch.empty((0, 0), dtype=torch.bool),
            observation_std_dbz=None,
            observation_innovation_dbz=None,
            observation_innovation_mask=None,
        )
        episode = self._episode(snapshot, "a2-zero-spatial")

        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with self.assertRaisesRegex(
                ValueError,
                "positive spatial dimensions",
            ):
                ledger.append(episode)


if __name__ == "__main__":
    unittest.main()
