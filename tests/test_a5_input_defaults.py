"""Public input defaults agree with source-aware observation semantics."""
from dataclasses import replace
import unittest

import torch

from advar.nowcast import ForecastRunContract, NowcastConfig, forecast_from_state, nowcast
from advar.sensitivity import SensitivityConfig
from advar.variational import AnalysisConfig, variational_nowcast


class SourceAvailabilityDefaultsTests(unittest.TestCase):
    def test_default_quality_matches_explicit_effective_mask(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                frames = torch.zeros((3, 4, 4), dtype=dtype)
                observed = torch.ones_like(frames, dtype=torch.bool)
                observed[:, 1, 1] = False
                source = torch.ones_like(observed)
                source[:, 0, 0] = False
                quality = (observed & source).to(frames)
                default = ForecastRunContract.from_inputs(
                    NowcastConfig(), frames, observed, None,
                    source_available_mask=source,
                )
                explicit = ForecastRunContract.from_inputs(
                    NowcastConfig(), frames, observed, None,
                    source_available_mask=source,
                    observation_quality_weight=quality,
                )
                self.assertEqual(default.input_bundle_digest, explicit.input_bundle_digest)
                self.assertEqual(
                    default.full_analysis_input_digest, explicit.full_analysis_input_digest,
                )
                self.assertEqual(
                    default.learned_model_input_features_digest,
                    explicit.learned_model_input_features_digest,
                )

    def test_public_variational_forecast_uses_same_defaults(self):
        frames = torch.zeros((3, 4, 4), dtype=torch.float64)
        source = torch.ones_like(frames, dtype=torch.bool)
        source[:, 0, 0] = False
        options = dict(
            source_available_mask=source,
            analysis_config=AnalysisConfig(
                maximum_outer_iterations=1, maximum_pcg_iterations=1,
            ),
        )
        default, _ = variational_nowcast(frames, **options)
        explicit, _ = variational_nowcast(
            frames, quality_weight=source.to(frames), **options,
        )
        self.assertEqual(default.forecast_run_digest, explicit.forecast_run_digest)
        torch.testing.assert_close(
            default.forecast_dbz, explicit.forecast_dbz, equal_nan=True,
        )

    def test_explicit_quality_on_an_unavailable_source_remains_invalid(self):
        frames = torch.zeros((3, 4, 4), dtype=torch.float64)
        observed = torch.ones_like(frames, dtype=torch.bool)
        source = observed.clone()
        source[:, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "quality weights"):
            ForecastRunContract.from_inputs(
                NowcastConfig(), frames, observed, None,
                source_available_mask=source,
                observation_quality_weight=torch.ones_like(frames),
            )

    def test_zero_taylor_probe_is_not_a_linearity_check(self):
        with self.assertRaisesRegex(ValueError, "nonzero probe"):
            SensitivityConfig(linearity_delta=(0.0, 0.0, 0.0))
        self.assertEqual(
            SensitivityConfig(linearity_delta=(0.0, 0.0, 0.001)).linearity_delta,
            (0.0, 0.0, 0.001),
        )

    def test_rehashed_psr_metadata_must_be_floating_scalar(self):
        y, x = torch.meshgrid(
            torch.arange(24, dtype=torch.float64),
            torch.arange(24, dtype=torch.float64), indexing="ij",
        )
        frames = torch.stack([
            -10 + 40 * torch.exp(-((y - 12) ** 2 + (x - center) ** 2) / 8)
            for center in (9, 10, 11)
        ])
        original = nowcast(frames, NowcastConfig(horizon_minutes=10))
        self.assertGreater(original.metadata.tendency_pair_count, 0)
        original.validate_issuance()
        for value in (torch.tensor(30), torch.tensor(True), torch.tensor([30.])):
            with self.subTest(value=value):
                changed = forecast_from_state(
                    original.state,
                    replace(original.metadata, minimum_phase_correlation_psr=value),
                    original.run.config, run=original.run,
                )
                with self.assertRaisesRegex(ValueError, "float32 or float64|scalar"):
                    changed.validate_issuance()
