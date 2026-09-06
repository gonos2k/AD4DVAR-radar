"""Small, independent regression oracles for the A5 scoring contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import advar.promotion as promotion_module  # noqa: E402
import advar.sensitivity as sensitivity_module  # noqa: E402
from advar.nowcast import NowcastConfig, RadarState  # noqa: E402
from advar.promotion import (  # noqa: E402
    NeuralPriorStateCalibrationPlan,
    NeuralPriorStateCalibrationTarget,
    PriorUncertaintyTarget,
    PriorUncertaintyTargetPlan,
)


class A5ScoringTests(unittest.TestCase):
    @staticmethod
    def _target_inputs(
        *, support_threshold_dbz: float
    ) -> tuple[object, object, object]:
        from test_promotion import (  # noqa: PLC0415
            NeuralPriorPromotionTests,
            _verification_bundle_v4,
            _verification_observation_error_plan,
        )

        grid = NeuralPriorPromotionTests.input_grid(1)
        valid_time = grid.valid_times[0]
        censor_policy_digest = "4" * 64
        source_digest = "6" * 64
        error_plan = _verification_observation_error_plan(
            radar_product_digest=source_digest,
            qc_pipeline_digest="9" * 64,
            censor_policy_digest=censor_policy_digest,
            grid_time_contract=grid,
        )
        frames = torch.tensor(
            [[[-11.0, 1.0], [10.0, 2.0]]],
            dtype=torch.float64,
        )
        verification = _verification_bundle_v4(
            frames_dbz=frames,
            valid_mask=torch.ones_like(frames, dtype=torch.bool),
            valid_times=(valid_time,),
            grid_time_contract=grid,
            radar_product_digest=source_digest,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=censor_policy_digest,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            threshold_bin_convention="nearest_rounding_threshold_censor",
            floor_representation_contract_digest="e" * 64,
        )
        common = dict(
            source_identity_digest=source_digest,
            qc_pipeline_digest="9" * 64,
            mask_policy_digest="3" * 64,
            censor_policy_digest=censor_policy_digest,
            floor_representation_contract_digest="e" * 64,
            grid_contract_digest=grid.digest,
            feature_exclusion_contract_digest="5" * 64,
            independence_evidence_digest="8" * 64,
            verification_observation_error_plan_digest=error_plan.plan_digest,
            target_valid_time=valid_time,
            support_threshold_dbz=support_threshold_dbz,
        )
        uncertainty_plan = PriorUncertaintyTargetPlan(
            plan_id="scoring-uncertainty",
            target_kind="independent_sensor",
            prior_probability_contract_digest="f" * 64,
            **common,
        )
        state_plan = NeuralPriorStateCalibrationPlan(
            plan_id="scoring-state",
            target_kind="withheld_target_mask",
            state_contract_digest="f" * 64,
            **common,
        )
        return verification, uncertainty_plan, state_plan

    def test_public_targets_use_detection_limit_for_censored_event_domain(self) -> None:
        verification, uncertainty_plan, state_plan = self._target_inputs(
            support_threshold_dbz=5.0
        )
        uncertainty = PriorUncertaintyTarget.from_verification_bundle(
            plan=uncertainty_plan,
            verification=verification,
        )
        state = NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=state_plan,
            verification=verification,
        )
        censored = verification.observation_state_code[0] == (
            promotion_module.VerificationCellState.BELOW_DETECTION_CENSORED
        )
        self.assertTrue(bool(torch.any(censored)))
        self.assertTrue(
            bool(torch.all(uncertainty._quality_weight[censored] > 0.0))
        )
        self.assertTrue(bool(torch.all(state._quality_weight[censored] > 0.0)))
        self.assertTrue(bool(torch.all(~uncertainty._echo_support[censored])))
        self.assertTrue(bool(torch.all(~state._echo_support[censored])))

        verification_at_limit, uncertainty_at_limit, state_at_limit = self._target_inputs(
            support_threshold_dbz=-10.0
        )
        uncertainty_unknown = PriorUncertaintyTarget.from_verification_bundle(
            plan=uncertainty_at_limit,
            verification=verification_at_limit,
        )
        state_unknown = NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=state_at_limit,
            verification=verification_at_limit,
        )
        self.assertTrue(
            bool(torch.all(uncertainty_unknown._quality_weight[censored] == 0.0))
        )
        self.assertTrue(
            bool(torch.all(state_unknown._quality_weight[censored] == 0.0))
        )

    def test_public_targets_keep_confirmed_clear_out_of_echo_support(self) -> None:
        from unittest.mock import patch  # noqa: PLC0415

        import test_promotion as fixture_module  # noqa: PLC0415
        from test_promotion import (  # noqa: PLC0415
            VerificationObservationMaskEvidence,
            VerificationObservationReportKind,
        )

        original_registry = fixture_module._observation_source_registry
        original_issue = VerificationObservationMaskEvidence.issue

        def registry_with_five_dbz_detection_limit(**kwargs: object) -> object:
            registry = original_registry(**kwargs)
            return replace(
                registry,
                ordered_sources=tuple(
                    replace(source, detection_limit_dbz=5.0)
                    for source in registry.ordered_sources
                ),
            )

        def issue_with_confirmed_clear(*args: object, **kwargs: object) -> object:
            report_kind = kwargs[
                "observation_report_kind_by_source"
            ].clone()
            report_kind[..., 0, 1] = int(
                VerificationObservationReportKind.CONFIRMED_CLEAR
            )
            kwargs["observation_report_kind_by_source"] = report_kind
            return original_issue(*args, **kwargs)

        with patch.object(
            fixture_module,
            "_observation_source_registry",
            side_effect=registry_with_five_dbz_detection_limit,
        ), patch.object(
            VerificationObservationMaskEvidence,
            "issue",
            side_effect=issue_with_confirmed_clear,
        ):
            verification, uncertainty_plan, state_plan = self._target_inputs(
                support_threshold_dbz=0.0
            )
        uncertainty = PriorUncertaintyTarget.from_verification_bundle(
            plan=uncertainty_plan,
            verification=verification,
        )
        state = NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=state_plan,
            verification=verification,
        )
        observation_state = verification.observation_state_code[0]
        clear = observation_state == promotion_module.VerificationCellState.OBSERVED_CLEAR
        echo = observation_state == promotion_module.VerificationCellState.OBSERVED_ECHO
        censored = (
            observation_state
            == promotion_module.VerificationCellState.BELOW_DETECTION_CENSORED
        )
        self.assertTrue(bool(clear[0, 1]))  # dBZ=1, below L=5
        self.assertTrue(bool(echo[1, 0]))  # dBZ=10, detected echo
        self.assertTrue(bool(censored[0, 0]))  # dBZ=-11, ambiguous at T=0
        for target in (uncertainty, state):
            self.assertGreater(float(target._quality_weight[0, 1]), 0.0)
            self.assertFalse(bool(target._echo_support[0, 1]))
            self.assertTrue(bool(target._echo_support[1, 0]))
            self.assertEqual(float(target._quality_weight[0, 0]), 0.0)

    def test_issuance_change_fractions_use_operational_and_parent_denominators(
        self,
    ) -> None:
        eligible = torch.ones((1, 2, 2), dtype=torch.bool)
        parent = torch.tensor(
            [[[True, True], [False, False]]],
            dtype=torch.bool,
        )
        candidate = torch.tensor(
            [[[True, False], [True, False]]],
            dtype=torch.bool,
        )
        newly, withdrawn = promotion_module._issuance_change_fractions(
            candidate,
            parent,
            eligible,
        )
        torch.testing.assert_close(newly, torch.tensor([0.25], dtype=torch.float64))
        torch.testing.assert_close(
            withdrawn,
            torch.tensor([0.5], dtype=torch.float64),
        )

        zero_parent = torch.zeros_like(parent)
        newly_zero_parent, withdrawn_zero_parent = (
            promotion_module._issuance_change_fractions(
                candidate,
                zero_parent,
                eligible,
            )
        )
        torch.testing.assert_close(
            newly_zero_parent,
            torch.tensor([0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            withdrawn_zero_parent,
            torch.zeros(1, dtype=torch.float64),
        )

        newly_empty, withdrawn_empty = promotion_module._issuance_change_fractions(
            candidate,
            parent,
            torch.zeros_like(eligible),
        )
        torch.testing.assert_close(
            newly_empty,
            torch.zeros(1, dtype=torch.float64),
        )
        torch.testing.assert_close(
            withdrawn_empty,
            torch.zeros(1, dtype=torch.float64),
        )

    def test_support_and_intensity_oracles_use_their_declared_weights(self) -> None:
        application = SimpleNamespace(
            event_probability=torch.tensor(
                [[0.2, 0.1], [0.8, 0.4]], dtype=torch.float64
            ),
            truncated_location_dbz=torch.tensor(
                [[9.0, 1.0], [11.0, 2.0]], dtype=torch.float64
            ),
            truncated_scale_dbz=torch.ones((2, 2), dtype=torch.float64),
        )
        reference = torch.tensor(
            [[10.0, 1.0], [12.0, 2.0]], dtype=torch.float64
        )
        support_target = torch.tensor(
            [[True, False], [True, False]], dtype=torch.bool
        )
        evaluation_mask = torch.ones((2, 2), dtype=torch.bool)
        evaluation_weight = torch.tensor(
            [[1.0, 0.5], [2.0, 0.25]], dtype=torch.float64
        )
        scores = promotion_module._prior_uncertainty_scores(
            application,
            reference,
            support_target,
            evaluation_mask,
            evaluation_weight,
            support_threshold_dbz=5.0,
        )
        expected_support = (0.64 + 0.005 + 0.08 + 0.04) / 3.75
        expected_clear = (0.5 * 0.1**2 + 0.25 * 0.4**2) / 0.75
        expected_echo_miss = (1.0 * 0.8**2 + 2.0 * 0.2**2) / 3.0
        self.assertAlmostEqual(scores.support_brier_score, expected_support)
        self.assertAlmostEqual(
            scores.clear_sky_false_echo_score or 0.0,
            expected_clear,
        )
        self.assertAlmostEqual(
            scores.echo_support_miss_score or 0.0,
            expected_echo_miss,
        )

    def test_full_map_scoring_rejects_off_grid_leads(self) -> None:
        from test_promotion import NeuralPriorPromotionTests  # noqa: PLC0415
        from test_sensitivity import (  # noqa: PLC0415
            _current_verification_bundle,
            result_for,
        )

        config = NowcastConfig(interval_minutes=30, horizon_minutes=60)
        state = RadarState(
            echo_linear=torch.full((2, 2), 1.0, dtype=torch.float64),
            displacement_yx=torch.tensor([0.1, -0.2], dtype=torch.float64),
            log_growth_per_step=torch.tensor(0.01, dtype=torch.float64),
        )
        frames = torch.full((3, 2, 2), 10.0, dtype=torch.float64)
        grid = replace(
            NeuralPriorPromotionTests.input_grid(1),
            valid_times=(
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:30:00Z",
                "2026-08-09T01:00:00Z",
            ),
        )
        result = result_for(
            state,
            config,
            frames=frames,
            accepted_mask=torch.ones_like(frames, dtype=torch.bool),
            grid_time_contract=grid,
        )
        verification = _current_verification_bundle(
            torch.full((2, 2, 2), 10.0, dtype=torch.float64),
            valid_times=("2026-08-09T01:30:00Z", "2026-08-09T02:00:00Z"),
            grid_time_contract=grid,
        )
        sensitivity_config = sensitivity_module.SensitivityConfig(
            metric_names=("log_echo_mse",),
            full_map_lead_minutes=(60,),
        )
        resolved = sensitivity_module._resolve_verification(
            verification,
            result,
            sensitivity_config,
        )
        scores, available = sensitivity_module._resolved_forecast_scores(
            result,
            state,
            resolved,
            (60,),
            sensitivity_config,
        )
        self.assertEqual(tuple(scores.shape), (1, 1))
        self.assertTrue(bool(available[0, 0]))
        with self.assertRaisesRegex(ValueError, "outside forecast horizon"):
            sensitivity_module._resolved_forecast_domain_weights(
                result,
                resolved,
                (45,),
                sensitivity_config,
            )


if __name__ == "__main__":
    unittest.main()
