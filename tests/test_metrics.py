from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advar.metrics import critical_success_index, mae, rmse  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_continuous_scores_ignore_non_finite_pairs(self) -> None:
        forecast = torch.tensor([10.0, 30.0, float("nan")])
        truth = torch.tensor([14.0, 22.0, 50.0])

        self.assertAlmostEqual(float(mae(forecast, truth)), 6.0)
        self.assertAlmostEqual(float(rmse(forecast, truth)), 6.324555, places=5)

    def test_csi_counts_hits_misses_and_false_alarms(self) -> None:
        forecast = torch.tensor([40.0, 40.0, 10.0])
        truth = torch.tensor([40.0, 10.0, 40.0])
        self.assertAlmostEqual(
            float(critical_success_index(forecast, truth, 35.0)),
            1.0 / 3.0,
        )

    def test_integer_inputs_are_rejected(self) -> None:
        values = torch.tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            mae(values, values)


if __name__ == "__main__":
    unittest.main()
