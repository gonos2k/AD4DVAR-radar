from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
