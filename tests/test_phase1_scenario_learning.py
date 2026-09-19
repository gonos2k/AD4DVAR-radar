"""Real P1/FSOI eligibility checks using a test policy trust store.

These scenarios do not update neural parameters.
"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import test_phase1_learning_demonstration as demonstration  # noqa: E402
import advar.sensitivity as sensitivity  # noqa: E402
from advar.nowcast import ForecastRunContract  # noqa: E402
from advar.physics import dbz_to_echo  # noqa: E402
from advar.sensitivity import (  # noqa: E402
    VariationalObservationPerturbation,
    compute_variational_fsoi_for_learning,
    forecast_metric,
)
from advar.variational import variational_nowcast  # noqa: E402


TRUST_STORE_PATH = Path("/tmp/advar-scenarios-20260908/learning/learning-policies.json")
PRIOR_CELL = (0, 6, 6)


class Phase1ScenarioLearningTests(unittest.TestCase):
    """Exercise independent signs, branches, support, and rejection gates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = demonstration._build_case()
        cls.case["location"] = PRIOR_CELL

        prior_valid = cls.case["linearization"].frozen.neural_prior_valid_mask
        assert prior_valid is not None
        if not bool(prior_valid[PRIOR_CELL[1:]]):
            raise AssertionError("scenario cell is not on the retained prior branch")

    def _run_learning(
        self,
        delta_value: float,
        *,
        policy=None,
        locations: tuple[tuple[int, int, int], ...] = (PRIOR_CELL,),
    ):
        case = self.case
        delta = torch.zeros_like(case["frames"])
        for location in locations:
            delta[location] = delta_value
        perturbation = VariationalObservationPerturbation.from_radar_dbz_delta(
            delta,
            case["linearization"],
            neural_prior_runner=case["prior"],
            neural_prior_application=case["application"],
        )
        selected_policy = case["policy"] if policy is None else policy
        trust_store = sensitivity._LearningPolicyTrustStore(
            approved_policy_digests=frozenset((selected_policy.digest,)),
            content_digest="7" * 64,
        )
        with patch.object(
            sensitivity,
            "_load_learning_policy_trust_store",
            return_value=trust_store,
        ):
            result = compute_variational_fsoi_for_learning(
                case["forecast"],
                case["analysis"],
                case["verification"],
                perturbation,
                policy=selected_policy,
                policy_trust_store_path=TRUST_STORE_PATH,
                neural_prior_runner=case["prior"],
                neural_prior_application=case["application"],
            )
        return result, delta

    def _independent_score_change(self, delta: torch.Tensor) -> float:
        """Re-solve P1 and score the changed forecast independently."""
        case = self.case
        changed_frames = case["frames"] + delta
        changed_run = ForecastRunContract.from_inputs(
            case["nowcast_config"],
            changed_frames,
            case["observation_mask"],
            case["background"],
            0.0,
            grid_time_contract=case["grid"],
        )
        changed_prior = case["prior"].infer(
            changed_frames, input_run=changed_run, role="candidate"
        )
        changed_forecast, _ = variational_nowcast(
            changed_frames,
            nowcast_config=case["nowcast_config"],
            analysis_config=case["analysis_config"],
            qc_mask=case["observation_mask"],
            background_frames_dbz=case["background"],
            background_age_minutes=0.0,
            grid_time_contract=case["grid"],
            neural_prior=changed_prior,
        )
        verification = case["verification"]
        weight = sensitivity._metric_domain_weight(
            case["forecast"],
            verification.valid_mask[0],
            0,
            "radar_dynamics_anchored",
            verification_metric_weight=verification.fso_metric_weight[0],
        )
        truth = dbz_to_echo(
            verification.frames_dbz,
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        nominal = dbz_to_echo(
            case["forecast"].forecast_dbz[0],
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        changed = dbz_to_echo(
            changed_forecast.forecast_dbz[0],
            min_dbz=case["nowcast_config"].min_dbz,
            max_dbz=case["nowcast_config"].max_dbz,
        )
        config = case["nowcast_config"]
        policy = case["policy"]
        nominal_score = forecast_metric(
            "log_echo_mse", nominal, truth[0], weight, config,
            policy.sensitivity_config, case["grid"],
        )
        changed_score = forecast_metric(
            "log_echo_mse", changed, truth[0], weight, config,
            policy.sensitivity_config, case["grid"],
        )
        return float(changed_score - nominal_score)

    def test_positive_and_negative_prior_cell_use_real_full_half_validation(self) -> None:
        for delta_value, expected_sign in ((-0.1, 1), (0.1, -1)):
            learning, delta = self._run_learning(delta_value)
            self.assertTrue(learning.eligibility.eligible, learning.eligibility.reasons)
            validation = learning.first_order_validation
            self.assertIsNotNone(validation)
            assert validation is not None
            self.assertTrue(validation.full_step_resolved_analysis_converged)
            self.assertTrue(validation.half_step_resolved_analysis_converged)
            self.assertTrue(validation.full_step_valid)
            self.assertTrue(validation.half_step_valid)
            self.assertTrue(validation.first_order_valid)
            self.assertGreater(validation.material_metric_count, 0)
            actual_change = self._independent_score_change(delta)
            resolved_change = float(validation.full_step_resolved_metric_change[0, 0])
            self.assertEqual(actual_change > 0.0, expected_sign > 0)
            self.assertEqual(resolved_change > 0.0, expected_sign > 0)
            self.assertEqual(
                learning.fsoi.perturbation_diagnostics.baseline_dynamics_branch_status,
                "not_applicable",
            )
            self.assertEqual(
                float(learning.fsoi.observation.total.sum_by_time.sum(dim=-1)[0, 0]),
                float(validation.full_step_prediction[0, 0]),
            )
            self.assertEqual(
                float(validation.full_step_prediction[0, 0]) > 0.0,
                expected_sign > 0,
            )
            self.assertLessEqual(
                abs(actual_change - float(validation.full_step_resolved_metric_change[0, 0])),
                self.case["nowcast_config"].contract_absolute_tolerance,
            )

    def test_metric_weight_is_nonuniform_and_partially_supported(self) -> None:
        weight = self.case["verification"].fso_metric_weight[0]
        positive = weight[weight > 0.0]
        self.assertGreater(positive.numel(), 0)
        self.assertLess(positive.numel(), weight.numel())
        self.assertGreater(torch.unique(positive).numel(), 1)

    def test_no_material_signal_is_rejected_after_real_validation(self) -> None:
        learning, _ = self._run_learning(1.0e-6)
        self.assertFalse(learning.eligibility.eligible)
        self.assertEqual(learning.eligibility.reasons, ("no_material_learning_signal",))
        validation = learning.first_order_validation
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertEqual(validation.material_metric_count, 0)
        self.assertTrue(validation.full_step_resolved_analysis_converged)
        self.assertTrue(validation.half_step_resolved_analysis_converged)

    def test_pixel_budget_is_rejected_before_learning(self) -> None:
        policy = replace(
            self.case["policy"],
            adjoint_config=replace(
                self.case["policy"].adjoint_config,
                maximum_perturbed_pixel_count=1,
            ),
        )
        learning, _ = self._run_learning(
            0.1, policy=policy, locations=(PRIOR_CELL, (0, 6, 7))
        )
        self.assertFalse(learning.eligibility.eligible)
        self.assertEqual(
            learning.eligibility.reasons,
            ("observation perturbation exceeds its pixel budget",),
        )
        self.assertIsNone(learning.fsoi)

    def test_classification_margin_gate_is_rejected_before_learning(self) -> None:
        policy = replace(
            self.case["policy"],
            adjoint_config=replace(
                self.case["policy"].adjoint_config,
                minimum_detection_margin_dbz=100.0,
            ),
        )
        learning, _ = self._run_learning(0.1, policy=policy)
        self.assertFalse(learning.eligibility.eligible)
        self.assertEqual(
            learning.eligibility.reasons,
            ("observation perturbation crosses the detected/censored branch",),
        )
        self.assertIsNone(learning.fsoi)


if __name__ == "__main__":
    unittest.main()
