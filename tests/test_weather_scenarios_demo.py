"""Contract checks for the standalone synthetic weather-scenarios demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "weather_scenarios"))

from build_demo import render_template  # noqa: E402
from scenarios import (  # noqa: E402
    HEIGHT,
    INTERVAL_MINUTES,
    SCENARIO_SPECS,
    TRUTH_STEPS,
    _metric,
    analytic_field,
    build_dataset,
    dumps_dataset,
)


class WeatherScenarioDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_dataset(source_commit="test-commit")
        cls.by_id = {item["id"]: item for item in cls.payload["scenarios"]}

    def test_schema_and_grid_contract_are_complete(self) -> None:
        meta = self.payload["meta"]
        self.assertEqual(meta["interval_minutes"], INTERVAL_MINUTES)
        self.assertEqual(meta["grid_shape"], [48, 48])
        self.assertEqual(len(self.payload["scenarios"]), len(SCENARIO_SPECS))
        for scenario in self.payload["scenarios"]:
            with self.subTest(scenario=scenario["id"]):
                self.assertEqual(len(scenario["observations"]), 3)
                self.assertEqual(len(scenario["truth"]), len(TRUTH_STEPS))
                self.assertEqual(len(scenario["forecast"]), len(TRUTH_STEPS))
                self.assertEqual(len(scenario["metrics"]), len(TRUTH_STEPS))
                self.assertEqual(len(scenario["persistence"]), HEIGHT)
                self.assertIn("tendency_source", scenario["state"])
                self.assertIn("reason", scenario["state"])
                for metric, lead in zip(scenario["metrics"], TRUTH_STEPS):
                    self.assertEqual(metric["lead_minutes"], lead * INTERVAL_MINUTES)
                    self.assertGreaterEqual(metric["scored_pixels"], 0)
                    self.assertLessEqual(metric["scored_fraction"], 1.0)
                    self.assertLessEqual(metric["valid_fraction"], 1.0)
                    self.assertEqual(set(metric["detection"]), {"10", "20"})
                    self.assertGreaterEqual(metric["issued_pixels"], metric["scored_pixels"])
                    self.assertEqual(
                        metric["issued_pixels"] + metric["issued_missing_pixels"],
                        48 * 48,
                    )
                    for threshold in ("10", "20"):
                        detection = metric["detection"][threshold]
                        self.assertIn("hits", detection)
                        self.assertIn("misses", detection)
                        self.assertIn("false_alarms", detection)
                        self.assertIn("truth_echo_pixels", detection)
                        self.assertIn("excluded_truth_echo_pixels", detection)
                        self.assertEqual(
                            detection["truth_echo_pixels"],
                            detection["scored_truth_echo_pixels"]
                            + detection["excluded_truth_echo_pixels"],
                        )

    def test_translation_geometry_and_signed_score_are_independent(self) -> None:
        # The analytic source moves toward increasing rows and decreasing
        # columns in array coordinates by design;
        # this does not rely on the production remapper or its estimated state.
        y, x = torch.meshgrid(
            torch.arange(48, dtype=torch.float64),
            torch.arange(48, dtype=torch.float64),
            indexing="ij",
        )
        centroids = []
        for step in (-2, -1, 0):
            field = analytic_field("translation", step) + 10.0
            weight = field.clamp_min(0.0)
            centroids.append(
                (float((weight * y).sum() / weight.sum()), float((weight * x).sum() / weight.sum()))
            )
        self.assertGreater(centroids[-1][0] - centroids[0][0], 0.5)
        self.assertLess(centroids[-1][1] - centroids[0][1], -0.2)

        translation = self.by_id["translation"]
        first = translation["metrics"][0]
        self.assertIsNotNone(first["mae"])
        self.assertIsNotNone(first["persistence_mae"])
        self.assertGreaterEqual(first["mae"], 0.0)
        self.assertGreaterEqual(first["persistence_mae"], 0.0)
        self.assertLess(first["mae"], first["persistence_mae"])

    def test_missing_coverage_keeps_null_observations_and_reduces_score_domain(self) -> None:
        scenario = self.by_id["missing_coverage"]
        self.assertEqual(scenario["state"]["data_status"], "PARTIAL")
        self.assertEqual(scenario["state"]["tendency_source"], "NONE")
        self.assertEqual(scenario["state"]["reason"], "no_usable_pair_support")
        self.assertEqual(
            scenario["persistence_source"],
            "latest_observation",
        )
        self.assertTrue(
            any(value is None for row in scenario["observations"][2] for value in row)
        )
        self.assertTrue(
            any(value is None for row in scenario["persistence"] for value in row)
        )
        self.assertLess(scenario["metrics"][0]["scored_pixels"], 48 * 48)
        self.assertLess(scenario["metrics"][0]["valid_fraction"], 1.0)
        self.assertTrue(
            all(
                metric["scored_pixels"] <= 48 * 48
                for metric in scenario["metrics"]
            )
        )
        self.assertGreater(
            scenario["metrics"][0]["detection"]["10"][
                "excluded_truth_echo_pixels"
            ],
            0,
        )

    def test_detection_counts_use_common_raw_domain_and_empty_denominators(self) -> None:
        forecast = torch.tensor(
            [[15.0, 25.0, float("nan")], [float("nan"), 19.0, 5.0]],
            dtype=torch.float64,
        )
        persistence = torch.ones((2, 3), dtype=torch.float64)
        truth = torch.tensor(
            [[15.0, 25.0, 25.0], [float("nan"), 21.0, 5.0]],
            dtype=torch.float64,
        )
        valid = torch.tensor(
            [[True, True, True], [False, True, True]], dtype=torch.bool
        )
        confidence = torch.ones((2, 3), dtype=torch.float64)
        common = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        metric = _metric(forecast, persistence, truth, valid, confidence, common)
        self.assertEqual(metric["scored_pixels"], 4)
        self.assertEqual(metric["issued_pixels"], 4)
        self.assertEqual(metric["issued_missing_pixels"], 2)
        self.assertEqual(metric["detection"]["10"]["hits"], 3)
        self.assertEqual(metric["detection"]["10"]["misses"], 0)
        self.assertEqual(metric["detection"]["10"]["false_alarms"], 0)
        self.assertEqual(metric["detection"]["10"]["csi"], 1.0)
        self.assertEqual(metric["detection"]["10"]["pod"], 1.0)
        self.assertEqual(metric["detection"]["10"]["truth_echo_pixels"], 4)
        self.assertEqual(metric["detection"]["10"]["scored_truth_echo_pixels"], 3)
        self.assertEqual(
            metric["detection"]["10"]["truth_positive_outside_common"], 1
        )
        self.assertEqual(
            metric["detection"]["10"]["excluded_truth_echo_pixels"], 1
        )
        self.assertEqual(
            metric["detection"]["10"]["excluded_truth_echo_pixels"], 1
        )
        self.assertEqual(metric["detection"]["20"]["hits"], 1)
        self.assertEqual(metric["detection"]["20"]["misses"], 1)
        self.assertEqual(metric["detection"]["20"]["false_alarms"], 0)
        self.assertEqual(metric["detection"]["20"]["csi"], 0.5)
        self.assertEqual(metric["detection"]["20"]["pod"], 0.5)
        self.assertEqual(metric["detection"]["20"]["truth_echo_pixels"], 3)
        self.assertEqual(metric["detection"]["20"]["scored_truth_echo_pixels"], 2)
        self.assertEqual(
            metric["detection"]["20"]["truth_positive_outside_common"], 1
        )

        empty = _metric(
            torch.full((2, 2), float("nan"), dtype=torch.float64),
            torch.full((2, 2), float("nan"), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.bool),
            torch.ones((2, 2), dtype=torch.float64),
        )
        for detection in empty["detection"].values():
            self.assertIsNone(detection["csi"])
            self.assertIsNone(detection["pod"])

    def test_marker_renderer_embeds_json_without_unescaped_script_boundary(self) -> None:
        probe = {"html": "</script><script>alert(1)</script>", "value": 3}
        rendered = render_template("<script>const DATA = __DEMO_DATA__;</script>", probe)
        encoded = rendered.split("const DATA = ", 1)[1].split(";</script>", 1)[0]
        decoded = json.loads(encoded)
        self.assertEqual(decoded, probe)
        self.assertNotIn("</script>", encoded.lower())
        self.assertEqual(
            json.loads(dumps_dataset(self.payload))["meta"]["grid_shape"],
            [48, 48],
        )
        with self.assertRaises(ValueError):
            render_template("no marker", probe)
        with self.assertRaises(ValueError):
            render_template("__DEMO_DATA__ __DEMO_DATA__", probe)

    def test_empty_score_domain_is_null_and_display_rounding_is_bounded(self) -> None:
        shape = (2, 2)
        finite = torch.ones(shape, dtype=torch.float64)
        absent = torch.full(shape, float("nan"), dtype=torch.float64)
        empty = _metric(absent, absent, finite, torch.zeros(shape, dtype=torch.bool), finite)
        self.assertEqual(empty["scored_pixels"], 0)
        self.assertIsNone(empty["mae"])
        self.assertIsNone(empty["persistence_mae"])
        self.assertIsNone(empty["confidence"])

        scenario = self.by_id["translation"]
        forecast = torch.tensor(
            [[float("nan") if value is None else value for value in row] for row in scenario["forecast"][0]],
            dtype=torch.float64,
        )
        persistence = torch.tensor(scenario["persistence"], dtype=torch.float64)
        truth = torch.tensor(scenario["truth"][0], dtype=torch.float64)
        domain = torch.isfinite(forecast) & torch.isfinite(persistence) & torch.isfinite(truth)
        rounded_mae = float((forecast[domain] - truth[domain]).abs().mean())
        self.assertLessEqual(abs(rounded_mae - scenario["metrics"][0]["mae"]), 0.0101)


if __name__ == "__main__":
    unittest.main()
