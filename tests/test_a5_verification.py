from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import advar.sensitivity as sensitivity_module
from advar.sensitivity import (
    VerificationCellState,
    VerificationObservationErrorContract,
)

from tests.test_sensitivity import _current_verification_bundle


class A5VerificationTests(unittest.TestCase):
    @staticmethod
    def _grid(*, shape: tuple[int, int] = (2, 2)) -> object:
        from advar.nowcast import (
            CURRENT_RADAR_METRIC_DOMAIN,
            CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE,
            RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
            RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
            RadarGridTimeContract,
            radar_projected_crs_semantic_digest,
        )

        return RadarGridTimeContract(
            valid_times=(
                "2026-08-05T00:00:00Z",
                "2026-08-05T00:10:00Z",
                "2026-08-05T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="1" * 64,
            spatial_grid_contract="radar-spatial-grid-identity-v6",
            grid_shape_yx=shape,
            projected_crs_digest=radar_projected_crs_semantic_digest(
                "EPSG:5179"
            ),
            metric_domain_digest=CURRENT_RADAR_METRIC_DOMAIN.digest,
            metric_domain_evidence_digest=(
                CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE.digest
            ),
            cell_center_origin_xy_m=(1_000_000.0, 2_000_000.0),
            grid_coordinate_dtype=RADAR_PROJECTED_GRID_COORDINATE_DTYPE,
            cell_center_convention=RADAR_PROJECTED_GRID_CELL_CENTER_CONVENTION,
        )

    def test_current_v22_uses_independent_spatial_temporal_and_weight_oracle(
        self,
    ) -> None:
        bundle = _current_verification_bundle(
            torch.full((1, 2, 2), 10.0, dtype=torch.float64),
            valid_times=("2026-08-05T00:30:00Z",),
            grid_time_contract=self._grid(),
            acquisition_time_offset_seconds=-60.0,
        )
        derivation = bundle.observation_error_derivation
        self.assertIsNotNone(derivation)
        assert derivation is not None
        raw = derivation.raw_inputs
        mask = raw.mask_derivation
        self.assertIsNotNone(mask)
        assert mask is not None
        evidence = mask.raw_evidence
        (
            _,
            selected_indices,
            _,
            _,
            _,
            _,
            selected_range_km,
            selected_elevation_deg,
            selected_blockage,
            selected_attenuation,
        ) = sensitivity_module._selected_verification_spatial_evidence(evidence)
        ground_upper = evidence.ground_range_upper_km_by_source
        self.assertIsNotNone(ground_upper)
        assert ground_upper is not None
        effective_range_km = torch.gather(
            ground_upper,
            0,
            selected_indices.unsqueeze(0),
        ).squeeze(0)
        self.assertFalse(torch.equal(effective_range_km, selected_range_km))

        plan = derivation.plan
        range_fraction = (
            effective_range_km / float(plan.maximum_range_km)
        ).clamp(0.0, 1.0)
        elevation_scale = max(
            abs(float(plan.minimum_elevation_deg)),
            abs(float(plan.maximum_elevation_deg)),
            torch.finfo(raw.frames_dbz.dtype).eps,
        )
        elevation_fraction = (
            selected_elevation_deg.abs() / elevation_scale
        ).clamp(max=1.0)
        spatial_quality = (
            selected_attenuation
            * (1.0 - selected_blockage)
            * (1.0 - 0.5 * range_fraction).clamp(0.5, 1.0)
        ).clamp(0.0, 1.0)
        spatial_std_multiplier = (
            1.0
            + range_fraction
            + elevation_fraction
            + selected_blockage
            + (1.0 - selected_attenuation)
        )
        age = raw.acquisition_age_seconds
        self.assertIsNotNone(age)
        assert age is not None
        temporal_decay = torch.exp(
            -torch.pow(
                age / float(plan.temporal_quality_decay_scale_seconds),
                float(plan.temporal_quality_decay_power),
            )
        )
        temporal_error = float(plan.temporal_error_growth_dbz_per_second) * age
        baseline_std_dbz = (
            derivation.source_registry.ordered_sources[0].observation_std_dbz
        )
        expected_quality = spatial_quality * temporal_decay
        expected_std = torch.sqrt(
            (baseline_std_dbz * spatial_std_multiplier).square()
            + temporal_error.square()
        )

        torch.testing.assert_close(
            derivation.quality_weight,
            expected_quality,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            derivation.observation_std_dbz,
            expected_std,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertLess(float(derivation.quality_weight.max()), 1.0)
        self.assertGreater(
            float(derivation.observation_std_dbz.min()),
            2.0,
        )

        expected_weight = (
            expected_quality
            * (
                float(plan.observation_error_reference_std_dbz)
                / expected_std
            ).square()
            * (
                derivation.observation_state_code
                != VerificationCellState.BELOW_DETECTION_CENSORED
            )
        )
        expected_weight = torch.where(
            derivation.valid_mask,
            expected_weight,
            torch.zeros_like(expected_weight),
        )
        torch.testing.assert_close(
            bundle.metric_weight,
            expected_weight,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertGreater(
            float(expected_weight.max() - expected_weight.min()),
            0.0,
        )

        # Error composition changes weights while the Boolean observation
        # coverage and state classification remain the mask derivation's
        # declared values.
        self.assertTrue(bool(torch.all(bundle.valid_mask)))
        self.assertTrue(
            torch.equal(bundle.valid_mask, derivation.valid_mask)
        )
        self.assertTrue(
            torch.equal(
                bundle.observation_state_code,
                derivation.observation_state_code,
            )
        )

    def test_current_producer_dtypes_remain_native(self) -> None:
        bundle = _current_verification_bundle(
            torch.full((1, 1, 2), 10.0, dtype=torch.float32),
            valid_times=("2026-08-05T00:20:00Z",),
            grid_time_contract=self._grid(shape=(1, 2)),
        )
        derivation = bundle.observation_error_derivation
        self.assertIsNotNone(derivation)
        assert derivation is not None
        self.assertIs(bundle.frames_dbz.dtype, torch.float32)
        self.assertIs(derivation.quality_weight.dtype, torch.float32)
        self.assertIs(derivation.observation_std_dbz.dtype, torch.float32)
        self.assertIs(bundle.valid_mask.dtype, torch.bool)
        self.assertIs(derivation.observation_state_code.dtype, torch.uint8)

    def test_replay_and_contract_boundaries_require_exact_dtypes(self) -> None:
        bundle = _current_verification_bundle(
            torch.full((1, 2, 2), 10.0, dtype=torch.float64),
            valid_times=("2026-08-05T00:30:00Z",),
            grid_time_contract=self._grid(),
        )
        derivation = bundle.observation_error_derivation
        self.assertIsNotNone(derivation)
        assert derivation is not None
        raw = derivation.raw_inputs
        mask = raw.mask_derivation
        self.assertIsNotNone(mask)
        assert mask is not None

        with self.assertRaisesRegex(ValueError, "mask replay mismatch"):
            replace(
                mask,
                _source_present_mask=mask.source_present_mask.to(torch.uint8),
            )
        with self.assertRaisesRegex(
            ValueError,
            "observation-error derivation replay mismatch",
        ):
            replace(
                derivation,
                _valid_mask=derivation.valid_mask.to(torch.uint8),
            )
        with self.assertRaisesRegex(
            ValueError,
            "observation-error derivation replay mismatch",
        ):
            replace(
                derivation,
                _observation_state_code=derivation.observation_state_code.to(
                    torch.int64
                ),
            )

        registry = derivation.source_registry
        source_epochs = tuple(
            (source.radar_site_digest, source.calibration_epoch_digest)
            for source in registry.ordered_sources
        )
        evidence = mask.raw_evidence
        with self.assertRaisesRegex(ValueError, "verification valid-mask"):
            VerificationObservationErrorContract.from_tensors(
                plan=derivation.plan,
                valid_mask=derivation.valid_mask.to(torch.uint8),
                quality_weight=derivation.quality_weight,
                observation_std_dbz=derivation.observation_std_dbz,
                frames_dbz=raw.frames_dbz,
                observation_state_code=derivation.observation_state_code,
                source_radar_index_map=derivation.source_radar_index_map,
                source_calibration_epochs=source_epochs,
                range_elevation_validity_domain_digest=(
                    evidence.range_elevation_validity_domain_digest
                ),
                beam_blockage_visibility_mask_digest=(
                    evidence.beam_blockage_visibility_mask_digest
                ),
                spatial_correlation_block_digest=(
                    evidence.spatial_correlation_block_digest
                ),
            )
        with self.assertRaisesRegex(
            ValueError,
            "verification observation-error tensor",
        ):
            VerificationObservationErrorContract.from_tensors(
                plan=derivation.plan,
                valid_mask=derivation.valid_mask,
                quality_weight=derivation.quality_weight.to(torch.float32),
                observation_std_dbz=derivation.observation_std_dbz,
                frames_dbz=raw.frames_dbz,
                observation_state_code=derivation.observation_state_code,
                source_radar_index_map=derivation.source_radar_index_map,
                source_calibration_epochs=source_epochs,
                range_elevation_validity_domain_digest=(
                    evidence.range_elevation_validity_domain_digest
                ),
                beam_blockage_visibility_mask_digest=(
                    evidence.beam_blockage_visibility_mask_digest
                ),
                spatial_correlation_block_digest=(
                    evidence.spatial_correlation_block_digest
                ),
            )

    def test_selection_certification_keeps_transcendentals_non_authoritative(
        self,
    ) -> None:
        bundle = _current_verification_bundle(
            torch.full((1, 1, 2), 10.0, dtype=torch.float64),
            valid_times=("2026-08-05T00:30:00Z",),
            grid_time_contract=self._grid(shape=(1, 2)),
            acquisition_time_offset_seconds=-60.0,
        )
        derivation = bundle.observation_error_derivation
        self.assertIsNotNone(derivation)
        assert derivation is not None
        evidence = derivation.raw_inputs.mask_derivation.raw_evidence
        lower = evidence.ground_range_lower_km_by_source
        upper = evidence.ground_range_upper_km_by_source
        certain = evidence.detection_classification_uncertain_by_source
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(certain)
        assert lower is not None and upper is not None and certain is not None
        with (
            patch.object(torch, "pow", side_effect=AssertionError("pow called")),
            patch.object(torch, "exp", side_effect=AssertionError("exp called")),
        ):
            replayed = sensitivity_module._product_owned_source_assignment_scores(
                plan=derivation.plan,
                valid_times=evidence.source_identity.valid_times,
                acquisition_valid_times_by_source=(
                    evidence.source_identity.acquisition_valid_times_by_source
                ),
                acquisition_time_offset_seconds_by_source=(
                    evidence.acquisition_time_offset_seconds_by_source
                ),
                source_availability_by_time=evidence.source_availability_by_time,
                range_km_by_source=evidence.range_km_by_source,
                elevation_deg_by_source=evidence.elevation_deg_by_source,
                beam_blockage_fraction_by_source=(
                    evidence.beam_blockage_fraction_by_source
                ),
                attenuation_qc_score_by_source=(
                    evidence.attenuation_qc_score_by_source
                ),
                detection_classification_certain_by_source=~certain,
                ground_range_lower_km_by_source=lower,
                ground_range_upper_km_by_source=upper,
            )
        self.assertTrue(torch.equal(replayed, evidence.source_assignment_scores))


if __name__ == "__main__":
    unittest.main()
