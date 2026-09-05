from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


LAB_SERVER = Path(__file__).parents[1] / "examples" / "initial_field_lab" / "server.py"
SPEC = importlib.util.spec_from_file_location("initial_field_lab_server", LAB_SERVER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {LAB_SERVER}")
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)


class InitialFieldLabTest(unittest.TestCase):
    def test_default_experiment_runs_the_p0_nowcast(self) -> None:
        result = LAB.run_experiment(LAB.ExperimentSettings())

        self.assertEqual(len(result["fields"]["forecast"]), LAB.CASE_HEIGHT)
        self.assertEqual(len(result["fields"]["forecast"][0]), LAB.CASE_WIDTH)
        self.assertTrue(result["state"]["background_used"])
        self.assertGreater(result["state"]["motion_pair_count"], 0)
        self.assertGreater(result["metrics"]["background_contribution_fraction"], 0)
        self.assertGreater(result["metrics"]["skill_dbz"], 0)

    def test_background_can_be_removed_or_visibly_changed(self) -> None:
        without_background = LAB.run_experiment(
            LAB.ExperimentSettings(use_background=False)
        )
        changed_background = LAB.run_experiment(
            LAB.ExperimentSettings(
                shift_y=4,
                shift_x=-3,
                intensity_bias_dbz=6.0,
                coverage_percent=50,
            )
        )

        self.assertFalse(without_background["state"]["background_used"])
        self.assertEqual(
            without_background["metrics"]["background_contribution_fraction"],
            0.0,
        )
        self.assertTrue(
            all(
                value is None
                for row in without_background["fields"]["background"]
                for value in row
            )
        )
        self.assertLess(changed_background["case"]["background_coverage"], 0.5)
        self.assertNotEqual(
            changed_background["fields"]["background"],
            without_background["fields"]["background"],
        )

    def test_request_rejects_an_unsupported_lead_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "lead_minutes"):
            LAB.ExperimentSettings.from_payload({"lead_minutes": 25})

    def test_age_changes_metadata_without_aging_the_background(self) -> None:
        with patch.object(LAB, "nowcast", wraps=LAB.nowcast) as nowcast_call:
            young = LAB.run_experiment(LAB.ExperimentSettings(background_age_minutes=0))
            old = LAB.run_experiment(LAB.ExperimentSettings(background_age_minutes=60))
        self.assertEqual(young["fields"]["background"], old["fields"]["background"])
        self.assertEqual(
            [call.kwargs["background_age_minutes"] for call in nowcast_call.call_args_list],
            [0, 60],
        )

    def test_scored_pixels_exclude_missing_persistence_on_valid_forecast(self) -> None:
        settings = LAB.ExperimentSettings()
        result, observation, background, target = LAB._run_case(settings)
        lead = settings.lead_minutes // LAB.FORECAST_STEP_MINUTES - 1
        # Force persistence to be missing on otherwise valid forecast pixels.
        valid = result.valid_mask[lead]
        observation[valid] = torch.nan
        background[valid] = torch.nan
        with patch.object(LAB, "_run_case", return_value=(result, observation, background, target)):
            output = LAB.run_experiment(settings)
        metrics = output["metrics"]
        self.assertGreater(metrics["valid_fraction"], 0)
        self.assertEqual(metrics["scored_pixels"], 0)
        self.assertEqual(metrics["scored_fraction"], 0)
        self.assertIsNone(metrics["forecast_mae_dbz"])

    def test_comparison_uses_fixed_reference_and_domain(self) -> None:
        target = torch.zeros(2, 2)
        reference = torch.tensor([[2.0, 2.0], [2.0, torch.nan]])
        persistence = torch.full((2, 2), 3.0)
        better = LAB._compare_forecasts(reference, torch.ones(2, 2), persistence, target)
        worse = LAB._compare_forecasts(reference, torch.full((2, 2), 4.0), persistence, target)
        self.assertEqual(better["domain_pixels"], 3)
        self.assertEqual(better["domain_fraction"], 0.75)
        self.assertEqual(worse["reference"], better["reference"])
        self.assertEqual(better["improvement_dbz"], 1.0)
        self.assertEqual(worse["improvement_dbz"], -2.0)
        self.assertEqual(better["candidate"]["persistence_mae_dbz"], 3.0)
        self.assertEqual(worse["candidate"]["persistence_mae_dbz"], 3.0)

        missing = torch.ones(2, 2)
        missing[0, 0] = torch.nan
        incomplete = LAB._compare_forecasts(reference, missing, persistence, target)
        self.assertEqual(incomplete["domain_pixels"], 3)
        self.assertEqual(incomplete["candidate"]["missing_pixels"], 1)
        self.assertEqual(incomplete["candidate"]["scored_pixels"], 0)
        self.assertIsNone(incomplete["candidate"]["forecast_mae_dbz"])
        self.assertIsNone(incomplete["improvement_dbz"])

    def test_comparison_without_reference_domain_has_no_score(self) -> None:
        empty = torch.full((2, 2), torch.nan)
        finite = torch.zeros(2, 2)
        comparison = LAB._compare_forecasts(empty, finite, finite, finite)
        self.assertEqual(comparison["domain_pixels"], 0)
        self.assertIsNone(comparison["improvement_dbz"])

    def test_run_comparison_keeps_a_when_background_b_changes(self) -> None:
        reference = LAB.ExperimentSettings()
        same = LAB.run_experiment(reference, reference)["comparison"]
        changed = LAB.run_experiment(
            LAB.ExperimentSettings(use_background=False), reference,
        )["comparison"]
        self.assertEqual(same["improvement_dbz"], 0)
        self.assertEqual(same["reference"], changed["reference"])
        self.assertEqual(same["domain_pixels"], changed["domain_pixels"])
        self.assertGreater(changed["candidate"]["missing_pixels"], 0)
        self.assertIsNone(changed["improvement_dbz"])

    def test_comparison_rejects_different_valid_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "same lead_minutes"):
            LAB.run_experiment(LAB.ExperimentSettings(lead_minutes=60), LAB.ExperimentSettings())


if __name__ == "__main__":
    unittest.main()
