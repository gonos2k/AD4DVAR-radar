from collections import Counter
from dataclasses import replace
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any
import zipfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar import (  # noqa: E402
    DynamicsSource,
    ForecastRunContract,
    NeuralPriorApplication,
    NeuralPriorInferenceRunner,
    NowcastConfig,
    RadarGridTimeContract,
    SensitivityConfig,
    TendencyPairSelection,
    compute_sensitivity_snapshot,
    compute_sensitivity_snapshot_from_run,
    load_forecast_run,
    nowcast,
    save_forecast_run,
    variational_nowcast,
)
from advar.variational import prepare_analysis  # noqa: E402
from advar._digest import tensor_digest  # noqa: E402
from advar.physics import FORECAST_INTEGRATOR_VERSION  # noqa: E402
import advar.run_artifact as run_artifact  # noqa: E402
from advar.run_artifact import seal_forecast_run_arrays  # noqa: E402


class ForecastRunArtifactTests(unittest.TestCase):
    class _Prior(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.offset

    def _save_arrays(self, path: Path, arrays: dict[str, Any]) -> None:
        np.savez_compressed(path, **seal_forecast_run_arrays(arrays))

    def test_tensor_digest_accepts_scalar_tensors(self) -> None:
        value = torch.tensor(0.125, dtype=torch.float64)

        self.assertEqual(tensor_digest(value), tensor_digest(value.clone()))
        self.assertNotEqual(
            tensor_digest(value),
            tensor_digest(value.to(dtype=torch.float32)),
        )

    def test_artifact_digest_is_independent_of_numpy_memory_layout(self) -> None:
        contiguous = np.arange(24, dtype=np.float64).reshape(4, 6)
        fortran = np.asfortranarray(contiguous)

        self.assertEqual(
            run_artifact._forecast_run_artifact_digest(
                {"value": contiguous}
            ),
            run_artifact._forecast_run_artifact_digest({"value": fortran}),
        )

    def frames(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        coordinates = torch.arange(8, dtype=dtype)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        echo = -10.0 + 40.0 * torch.exp(
            -((y - 3.5).square() + (x - 3.5).square()) / 4.0
        )
        return torch.stack((echo, echo, echo))

    def test_restart_round_trip_preserves_m0_snapshot(self) -> None:
        frames = self.frames()
        config = NowcastConfig(
            growth_decay_minutes=120.0,
            max_dbz=60.0,
            min_publish_support=0.7,
        )
        result = nowcast(frames, config)
        sensitivity_config = SensitivityConfig(
            full_map_lead_minutes=(30,),
            tile_size=4,
            soft_fss_window=3,
        )
        in_memory = compute_sensitivity_snapshot(
            frames[-1],
            result,
            result.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        loaded.validate_issuance()
        self.assertEqual(loaded.run.config, config)
        self.assertEqual(
            loaded.state_metadata_digest,
            result.state_metadata_digest,
        )
        self.assertEqual(
            loaded.run.input_bundle_digest,
            result.run.input_bundle_digest,
        )
        self.assertEqual(
            loaded.forecast_run_digest,
            result.forecast_run_digest,
        )
        self.assertIsNone(loaded.run.analysis_config_json)
        self.assertEqual(loaded.forecast_dbz_digest, result.forecast_dbz_digest)
        self.assertEqual(loaded.valid_mask_digest, result.valid_mask_digest)
        self.assertIsNone(loaded.audit)
        torch.testing.assert_close(
            loaded.forecast_dbz, result.forecast_dbz, equal_nan=True
        )
        torch.testing.assert_close(
            loaded.state.echo_linear,
            result.state.echo_linear,
        )
        torch.testing.assert_close(
            loaded.metadata.source_support,
            result.metadata.source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.path_verified_source_support,
            result.metadata.path_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.verified_source_support,
            result.metadata.verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_motion_verified_support,
            result.metadata.local_motion_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_growth_verified_support,
            result.metadata.local_growth_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.local_dynamics_verified_support,
            result.metadata.local_dynamics_verified_support,
        )
        torch.testing.assert_close(
            loaded.metadata.observation_verified_source_support,
            result.metadata.observation_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.metadata.background_verified_source_support,
            result.metadata.background_verified_source_support,
        )
        torch.testing.assert_close(
            loaded.forecast_path_verified_support,
            result.forecast_path_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_verified_support,
            result.forecast_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_motion_verified_support,
            result.forecast_local_motion_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_growth_verified_support,
            result.forecast_local_growth_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_local_dynamics_verified_support,
            result.forecast_local_dynamics_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_observation_verified_support,
            result.forecast_observation_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_background_verified_support,
            result.forecast_background_verified_support,
        )
        torch.testing.assert_close(
            loaded.forecast_velocity_uncertainty_mps,
            result.forecast_velocity_uncertainty_mps,
        )
        torch.testing.assert_close(
            loaded.forecast_position_uncertainty_m,
            result.forecast_position_uncertainty_m,
        )
        torch.testing.assert_close(
            loaded.forecast_log_growth_uncertainty,
            result.forecast_log_growth_uncertainty,
        )
        torch.testing.assert_close(
            loaded.metadata.maximum_growth_saturation_excess,
            result.metadata.maximum_growth_saturation_excess,
        )
        torch.testing.assert_close(
            loaded.metadata.posterior_velocity_uncertainty_mps,
            result.metadata.posterior_velocity_uncertainty_mps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.posterior_log_growth_uncertainty_per_step,
            result.metadata.posterior_log_growth_uncertainty_per_step,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.p1_velocity_saturation_uncertainty_mps,
            result.metadata.p1_velocity_saturation_uncertainty_mps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.metadata.p1_log_growth_saturation_uncertainty_per_step,
            result.metadata.p1_log_growth_saturation_uncertainty_per_step,
            equal_nan=True,
        )
        torch.testing.assert_close(
            loaded.forecast_confidence,
            result.forecast_confidence,
        )
        torch.testing.assert_close(
            loaded.radar_anchored_valid_mask,
            result.radar_anchored_valid_mask,
        )
        torch.testing.assert_close(
            loaded.background_fallback_mask,
            result.background_fallback_mask,
        )
        for loaded_path, expected_path in (
            (
                loaded.metadata.observation_path,
                result.metadata.observation_path,
            ),
            (
                loaded.metadata.background_path,
                result.metadata.background_path,
            ),
        ):
            self.assertEqual(loaded_path.mode, expected_path.mode)
            self.assertEqual(loaded_path.pair_count, expected_path.pair_count)
            self.assertTrue(math.isnan(loaded_path.minimum_psr))
            self.assertEqual(loaded_path.conflict, expected_path.conflict)
            self.assertEqual(
                loaded_path.extrapolated,
                expected_path.extrapolated,
            )
            self.assertEqual(
                loaded_path.age_minutes,
                expected_path.age_minutes,
            )
        self.assertEqual(
            loaded.metadata.background_tendency_used,
            result.metadata.background_tendency_used,
        )
        self.assertEqual(
            loaded.metadata.background_state_support_fraction,
            result.metadata.background_state_support_fraction,
        )
        self.assertEqual(
            loaded.metadata.dynamics_source,
            DynamicsSource.P0_RECONSTRUCTION,
        )
        torch.testing.assert_close(
            loaded.metadata.minimum_phase_correlation_psr,
            result.metadata.minimum_phase_correlation_psr,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_count,
            result.metadata.motion_pair_count,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_count,
            result.metadata.growth_pair_count,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            result.metadata.motion_pair_selection,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            result.metadata.growth_pair_selection,
        )
        self.assertEqual(
            loaded.metadata.motion_pair_conflict,
            result.metadata.motion_pair_conflict,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_conflict,
            result.metadata.growth_pair_conflict,
        )
        for name in (
            "dynamics_source",
            "state_path_source",
            "state_path_mode",
            "state_path_pair_count",
            "state_path_minimum_psr",
            "state_path_conflict",
            "state_path_extrapolated",
            "state_path_age_minutes",
            "minimum_growth_overlap_support",
            "minimum_growth_overlap_area_km2",
        ):
            with self.subTest(name=name):
                loaded_value = getattr(loaded.metadata, name)
                expected = getattr(result.metadata, name)
                if isinstance(expected, float) and math.isnan(expected):
                    self.assertTrue(math.isnan(loaded_value))
                else:
                    self.assertEqual(loaded_value, expected)
        torch.testing.assert_close(
            loaded.run.latest_observation_mask,
            result.run.latest_observation_mask,
        )
        self.assertEqual(
            loaded.run.forecast_integrator_version,
            FORECAST_INTEGRATOR_VERSION,
        )

        reloaded = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=sensitivity_config,
        )
        torch.testing.assert_close(
            reloaded.context_features,
            in_memory.context_features,
        )
        torch.testing.assert_close(
            reloaded.forecast_scores,
            in_memory.forecast_scores,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.control_sensitivity,
            in_memory.control_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.forecast_sensitivity,
            in_memory.forecast_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.maps,
            in_memory.direct.maps,
            equal_nan=True,
        )
        torch.testing.assert_close(
            reloaded.direct.tile_norm,
            in_memory.direct.tile_norm,
            equal_nan=True,
        )
        self.assertEqual(reloaded.trust_score, in_memory.trust_score)

    def test_round_trip_preserves_long_pair_provenance(self) -> None:
        frames = self.frames()
        frames[1] = torch.nan
        result = nowcast(frames)

        self.assertEqual(
            result.metadata.motion_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(
            result.metadata.growth_pair_selection,
            TendencyPairSelection.LONG,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "long-pair-run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            TendencyPairSelection.LONG,
        )
        self.assertEqual(loaded.metadata.tendency_pair_count, 1)
        self.assertFalse(loaded.metadata.motion_pair_conflict)
        self.assertFalse(loaded.metadata.growth_pair_conflict)

    def test_round_trip_preserves_motion_conditioned_growth_conflict(
        self,
    ) -> None:
        coordinates = torch.arange(64, dtype=torch.float64)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        first = -10.0 + 40.0 * torch.exp(
            -((y - 31.5).square() + (x - 31.5).square()) / 32.0
        )
        result = nowcast(
            torch.stack(
                (first, torch.roll(first, shifts=10, dims=1), first)
            )
        )

        self.assertTrue(result.metadata.motion_pair_conflict)
        self.assertTrue(result.metadata.growth_pair_conflict)
        self.assertEqual(
            result.metadata.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(
            result.metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pair-conflict-run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertTrue(loaded.metadata.motion_pair_conflict)
        self.assertTrue(loaded.metadata.growth_pair_conflict)
        self.assertEqual(
            loaded.metadata.motion_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        self.assertEqual(
            loaded.metadata.growth_pair_selection,
            TendencyPairSelection.PERSISTENCE,
        )
        loaded.validate_issuance()
        snapshot = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=16,
            ),
        )
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        self.assertEqual(float(context["motion_pair_conflict"]), 1.0)
        self.assertEqual(float(context["growth_pair_conflict"]), 1.0)
        self.assertEqual(
            float(context["motion_pair_selection_persistence"]),
            1.0,
        )
        self.assertEqual(
            float(context["growth_pair_selection_persistence"]),
            1.0,
        )
        self.assertEqual(float(context["phase_correlation_psr_available"]), 0.0)
        self.assertEqual(
            float(context["log1p_minimum_phase_correlation_psr"]),
            0.0,
        )

    def test_loader_materializes_each_archive_member_once(self) -> None:
        result = nowcast(self.frames())
        reads: Counter[str] = Counter()
        original_getitem = np.lib.npyio.NpzFile.__getitem__

        def counted_getitem(
            archive: np.lib.npyio.NpzFile,
            name: str,
        ) -> np.ndarray:
            reads[name] += 1
            return original_getitem(archive, name)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "single-materialization-run.npz"
            save_forecast_run(result, path)
            with patch.object(
                np.lib.npyio.NpzFile,
                "__getitem__",
                counted_getitem,
            ):
                loaded = load_forecast_run(path)

        loaded.validate_issuance()
        self.assertEqual(set(reads), run_artifact._CORE_ARRAY_NAMES)
        self.assertEqual(set(reads.values()), {1})

    def test_restart_m0_uses_embedded_latest_background(self) -> None:
        frames = self.frames()
        background = frames - 2.0
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )
        expected = compute_sensitivity_snapshot(
            frames[-1],
            result,
            result.forecast_dbz.clone(),
            latest_background_dbz=background[-1],
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        torch.testing.assert_close(loaded.run.latest_frame_dbz, frames[-1])
        loaded_background = loaded.run.latest_background_dbz
        self.assertIsNotNone(loaded_background)
        assert loaded_background is not None
        torch.testing.assert_close(
            loaded_background,
            background[-1],
        )
        actual = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
        )
        torch.testing.assert_close(
            actual.control_sensitivity,
            expected.control_sensitivity,
            equal_nan=True,
        )
        torch.testing.assert_close(
            actual.direct.maps,
            expected.direct.maps,
            equal_nan=True,
        )

    def test_round_trip_preserves_grid_time_contract(self) -> None:
        frames = self.frames()
        background = frames - 1.0
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=500.0,
            dy_m=750.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
            background_valid_times=(
                "2026-07-30T23:50:00Z",
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
            ),
        )
        result = nowcast(
            frames,
            background_frames_dbz=background,
            background_age_minutes=10.0,
            grid_time_contract=contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.grid_time_contract, contract)
        self.assertEqual(
            loaded.run.grid_time_contract_digest,
            contract.digest,
        )
        self.assertEqual(loaded.run.background_age_minutes, 10.0)
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)
        torch.testing.assert_close(
            loaded.displacement_mps_yx,
            result.displacement_mps_yx,
        )
        torch.testing.assert_close(
            loaded.projected_velocity_mps_xy,
            result.projected_velocity_mps_xy,
        )
        torch.testing.assert_close(
            loaded.metadata.motion_disagreement_mps,
            result.metadata.motion_disagreement_mps,
            equal_nan=True,
        )
        snapshot = compute_sensitivity_snapshot_from_run(
            loaded,
            loaded.forecast_dbz.clone(),
            sensitivity_config=SensitivityConfig(
                metric_names=("log_echo_mse",),
                full_map_lead_minutes=(10,),
                tile_size=4,
            ),
        )
        context = dict(
            zip(snapshot.context_feature_names, snapshot.context_features)
        )
        self.assertEqual(snapshot.grid_time_contract_digest, contract.digest)
        self.assertEqual(context["projected_velocity_available"], 1.0)
        self.assertEqual(
            context["motion_disagreement_mps_available"], 1.0
        )
        torch.testing.assert_close(
            context["motion_disagreement_mps"],
            loaded.metadata.motion_disagreement_mps,
        )
        self.assertEqual(context["area_weighted_echo_available"], 1.0)
        self.assertGreater(
            context["log1p_linear_reflectivity_integral_km2"], 0.0
        )
        torch.testing.assert_close(
            torch.stack(
                (
                    context["projected_velocity_x_mps"],
                    context["projected_velocity_y_mps"],
                )
            ),
            loaded.projected_velocity_mps_xy,
        )
        torch.testing.assert_close(
            context["projected_speed_mps"],
            torch.linalg.vector_norm(loaded.projected_velocity_mps_xy),
        )

    def test_load_rejects_tampered_grid_time_contract(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=1000.0,
            projection="EPSG:5179",
            grid_hash="c" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            contract_value = arrays["grid_time_contract_json"].item()
            arrays["grid_time_contract_json"] = np.asarray(
                contract_value.replace(
                    '"projection":"EPSG:5179"',
                    '"projection":"EPSG:3857"',
                )
            )
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "grid/time contract digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_physical_displacement(self) -> None:
        contract = RadarGridTimeContract(
            valid_times=(
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:10:00Z",
                "2026-07-31T00:20:00Z",
            ),
            dx_m=1000.0,
            dy_m=500.0,
            projection="EPSG:5179",
            grid_hash="d" * 64,
        )
        result = nowcast(self.frames(), grid_time_contract=contract)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["displacement_mps_yx"] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "displacement_mps_yx disagrees",
            ):
                load_forecast_run(path)

    def test_input_bundle_digest_covers_all_three_input_times(self) -> None:
        frames = self.frames()
        accepted = torch.ones_like(frames, dtype=torch.bool)
        background = frames - 1.0
        base = nowcast(
            frames,
            qc_mask=accepted,
            background_frames_dbz=background,
            background_age_minutes=10.0,
        )

        earlier_frame = frames.clone()
        earlier_frame[0, 2, 2] += 0.25
        earlier_mask = accepted.clone()
        earlier_mask[0, 2, 2] = False
        earlier_background = background.clone()
        earlier_background[0, 2, 2] += 0.25
        variants = (
            nowcast(
                earlier_frame,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=earlier_mask,
                background_frames_dbz=background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=earlier_background,
                background_age_minutes=10.0,
            ),
            nowcast(
                frames,
                qc_mask=accepted,
                background_frames_dbz=background,
                background_age_minutes=20.0,
            ),
        )

        for variant in variants:
            self.assertNotEqual(
                variant.run.input_bundle_digest,
                base.run.input_bundle_digest,
            )
            self.assertNotEqual(
                variant.forecast_run_digest,
                base.forecast_run_digest,
            )

    def test_neural_prior_lineage_round_trips_with_the_run(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        prior = runner.infer(
            frames,
            input_run=input_run,
            role="candidate",
        )
        result, _ = variational_nowcast(frames, neural_prior=prior)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.neural_prior_digest, prior.neural_prior_digest)
        self.assertEqual(loaded.run.prior_application_digest, prior.application_digest)
        self.assertEqual(loaded.run.prior_model_contract_digest, "2" * 64)
        self.assertEqual(loaded.run.prior_feature_schema_digest, "3" * 64)
        self.assertEqual(loaded.run.prior_training_manifest_digest, "4" * 64)
        self.assertEqual(
            loaded.run.prior_inference_evidence_digest,
            prior.inference_evidence.evidence_digest,
        )
        self.assertEqual(loaded.run.prior_dependency, "radar_dependent")
        self.assertEqual(loaded.run.prior_role, "candidate")
        self.assertEqual(loaded.forecast_run_digest, result.forecast_run_digest)
        _, frozen = prepare_analysis(frames, neural_prior=prior)
        torch.testing.assert_close(
            frozen.initial_background_dbz,
            torch.where(
                frozen.initial_support_mask,
                prior.initial_background_dbz,
                torch.full_like(prior.initial_background_dbz, -10.0),
            ),
        )

    def test_neural_prior_lineage_is_all_or_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "neural-prior run lineage"):
            ForecastRunContract.from_inputs(
                NowcastConfig(),
                self.frames(),
                torch.ones_like(self.frames(), dtype=torch.bool),
                None,
                neural_prior_digest="1" * 64,
            )

    def test_v42_artifact_migrates_without_prior_lineage(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v42.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name not in {
                        "neural_prior_digest",
                        "prior_application_digest",
                        "prior_model_contract_digest",
                        "prior_feature_schema_digest",
                        "prior_training_manifest_digest",
                        "prior_role",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v42"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertIsNone(loaded.run.neural_prior_digest)
        torch.testing.assert_close(loaded.forecast_dbz, result.forecast_dbz)

    def test_v44_artifact_preserves_existing_prior_lineage(self) -> None:
        frames = self.frames()
        input_run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        runner = NeuralPriorInferenceRunner(
            self._Prior().eval(),
            lambda value: value[0],
            example_frames=frames,
            model_contract_digest="2" * 64,
            feature_schema_digest="3" * 64,
            training_manifest_digest="4" * 64,
            allow_constant_uncertainty=True,
            dependency="radar_dependent",
        )
        prior = runner.infer(frames, input_run=input_run, role="candidate")
        result, _ = variational_nowcast(frames, neural_prior=prior)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-v44.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name not in {
                        "prior_inference_evidence_digest",
                        "prior_inference_algorithm_digest",
                        "prior_numerical_runtime_digest",
                        "prior_dependency",
                    }
                }
            arrays["forecast_run_artifact_version"] = np.asarray(
                "forecast-run-v44"
            )
            self._save_arrays(path, arrays)

            loaded = load_forecast_run(path)

        self.assertEqual(loaded.run.neural_prior_digest, prior.neural_prior_digest)
        self.assertEqual(
            loaded.run.prior_lineage_contract,
            "neural-prior-run-lineage-v1-audit",
        )
        self.assertIsNone(loaded.run.prior_inference_evidence_digest)

    def test_load_rejects_tampered_state(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["state_echo_linear"][0, 0] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast does not close against the issued state",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_forecast_verified_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_local_dynamics_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_local_dynamics_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast local dynamics verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_local_motion_support(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_local_motion_verified_support"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast local motion verified support mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_resigned_forecast_confidence(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "forecast_run_artifact_digest"
                }
            arrays["forecast_confidence"][0, 0, 0] = 0.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "forecast confidence mismatch",
            ):
                load_forecast_run(path)

    def test_save_rejects_mutated_metadata(self) -> None:
        result = nowcast(self.frames())
        result.metadata.source_support[0, 0] = 0.0
        result.metadata.path_verified_source_support[0, 0] = 0.0
        result.metadata.verified_source_support[0, 0] = 0.0
        result.metadata.observation_verified_source_support[0, 0] = 0.0

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "source support and actual contributions",
            ):
                save_forecast_run(result, path)
            self.assertFalse(path.exists())

    def test_load_rejects_tampered_phase_correlation_psr(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["minimum_phase_correlation_psr"] += 1.0
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "state or metadata",
            ):
                load_forecast_run(path)

    def test_load_rejects_pair_selection_count_mismatch(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["motion_pair_selection"] = np.asarray("PERSISTENCE")
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "motion pair count and selection disagree",
            ):
                load_forecast_run(path)

    def test_load_rejects_incorrect_pair_union_count(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["motion_pair_count"] = np.asarray(1)
            arrays["growth_pair_count"] = np.asarray(1)
            arrays["motion_pair_selection"] = np.asarray("EARLIER")
            arrays["growth_pair_selection"] = np.asarray("RECENT")
            arrays["tendency_pair_count"] = np.asarray(1)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "tendency_pair_count is inconsistent",
            ):
                load_forecast_run(path)

    def test_load_rejects_malformed_phase_correlation_psr_schema(self) -> None:
        result = nowcast(self.frames())
        malformed_values = (
            np.asarray(1, dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray(np.inf, dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                original: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }

            for malformed in malformed_values:
                with self.subTest(value=malformed):
                    arrays = dict(original)
                    arrays["minimum_phase_correlation_psr"] = malformed
                    self._save_arrays(path, arrays)

                    with self.assertRaisesRegex(
                        ValueError,
                        "minimum_phase_correlation_psr must be",
                    ):
                        load_forecast_run(path)

    def test_load_rejects_tampered_background_tendency_provenance(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["background_tendency_used"] = np.asarray(True)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "background tendency provenance mismatch",
            ):
                load_forecast_run(path)

    def test_load_uses_central_metadata_semantics(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                original: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }

            for name, value, message in (
                (
                    "background_used",
                    np.asarray(True),
                    "background usage provenance",
                ),
                (
                    "minimum_growth_overlap_support",
                    np.asarray(np.nan),
                    "used growth pairs",
                ),
            ):
                with self.subTest(name=name):
                    arrays = dict(original)
                    arrays[name] = value
                    self._save_arrays(path, arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        load_forecast_run(path)

    def test_save_rejects_mutated_analysis_lineage(self) -> None:
        result = nowcast(self.frames())
        changed_run = replace(
            result.run,
            analysis_config_json="{}",
            analysis_config_digest="0" * 64,
            analysis_input_digest="1" * 64,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "analysis config digest mismatch",
            ):
                save_forecast_run(
                    replace(result, run=changed_run),
                    path,
                )
            self.assertFalse(path.exists())

    def test_save_rejects_incompatible_forecast_integrator(self) -> None:
        result = nowcast(self.frames())
        changed_run = replace(
            result.run,
            forecast_integrator_version="incompatible-integrator",
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            with self.assertRaisesRegex(
                ValueError,
                "forecast integrator version",
            ):
                save_forecast_run(replace(result, run=changed_run), path)
            self.assertFalse(path.exists())

    def test_load_rejects_config_json_that_disagrees_with_digest(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["nowcast_config_json"] = np.asarray(
                '{"interval_minutes":5,"horizon_minutes":180,'
                '"min_dbz":-10.0,"max_dbz":70.0,'
                '"echo_threshold_dbz":5.0,"recent_weight":0.6666666666666666,'
                '"pair_echo_dilation_px":3,"max_displacement_px":20.0,'
                '"max_log_growth_per_step":0.30010459245033816,'
                '"growth_decay_minutes":60.0,"min_publish_support":0.95,'
                '"epsilon":1e-06}'
            )
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(
                ValueError,
                "config digest mismatch",
            ):
                load_forecast_run(path)

    def test_load_rejects_tampered_input_bundle_identity(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["input_bundle_digest"] = np.asarray("0" * 64)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(ValueError, "run identity"):
                load_forecast_run(path)

    def test_load_rejects_unknown_archive_member(self) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                arrays: dict[str, Any] = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["unexpected_payload"] = np.asarray(1)
            self._save_arrays(path, arrays)

            with self.assertRaisesRegex(ValueError, "unknown members"):
                load_forecast_run(path)

    def test_load_applies_archive_resource_limits_before_materialization(
        self,
    ) -> None:
        result = nowcast(self.frames())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.npz"
            save_forecast_run(result, path)
            with np.load(path, allow_pickle=False) as archive:
                member_count = len(archive.files)

            cases = (
                ({"maximum_member_count": member_count - 1}, "too many"),
                ({"maximum_member_bytes": 1}, "member is too large"),
                (
                    {"maximum_total_expanded_bytes": 1},
                    "expands beyond",
                ),
            )
            for limits, message in cases:
                with self.subTest(limits=limits):
                    with self.assertRaisesRegex(ValueError, message):
                        load_forecast_run(path, **limits)

    def test_load_rejects_declared_array_size_before_materialization(
        self,
    ) -> None:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.dtype(np.float64).str,
                "fortran_order": False,
                "shape": (1_000_000_000,),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hostile.npz"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("forecast_dbz.npy", header.getvalue())

            with self.assertRaisesRegex(
                ValueError,
                "declares too much array data",
            ):
                load_forecast_run(path, maximum_member_bytes=1024)

    def test_load_rejects_object_array_header_before_materialization(
        self,
    ) -> None:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.dtype(object).str,
                "fortran_order": False,
                "shape": (1,),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object.npz"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("forecast_dbz.npy", header.getvalue())

            with self.assertRaisesRegex(ValueError, "non-object dtypes"):
                load_forecast_run(path)

    def test_load_uses_the_same_file_after_resource_preflight(self) -> None:
        original_result = nowcast(self.frames())
        replacement_frames = self.frames() + 1.0
        replacement_result = nowcast(replacement_frames)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "run.npz"
            replacement = directory / "replacement.npz"
            save_forecast_run(original_result, path)
            save_forecast_run(replacement_result, replacement)
            real_preflight = run_artifact._preflight_archive

            def preflight_then_replace(
                source: Any,
                **limits: Any,
            ) -> None:
                real_preflight(source, **limits)
                replacement.replace(path)

            with patch(
                "advar.run_artifact._preflight_archive",
                side_effect=preflight_then_replace,
            ):
                loaded = load_forecast_run(path)

        self.assertEqual(
            loaded.forecast_run_digest,
            original_result.forecast_run_digest,
        )


if __name__ == "__main__":
    unittest.main()
