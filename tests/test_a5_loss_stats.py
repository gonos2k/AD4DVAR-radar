"""Focused A5 numerical and promotion-contract regression tests."""

from __future__ import annotations

import math
from statistics import NormalDist
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from advar import promotion as promotion_module


def _preflight_fixture() -> tuple[object, object, tuple[tuple[str, int], ...]]:
    metric = SimpleNamespace(
        weather_regime="convective",
        range_regime="near_range",
        metric_name="log_echo_mse",
        lead_minutes=60,
        minimum_physical_events=1,
        maximum_harmful_fraction_upper_bound=1.0,
        maximum_end_to_end_harmful_fraction_upper_bound=1.0,
    )
    issuance = SimpleNamespace(
        weather_regime="convective",
        range_regime="near_range",
        lead_minutes=60,
        minimum_physical_events=1,
        maximum_withdrawn_fraction=1.0,
        maximum_newly_issued_fraction=1.0,
        maximum_background_fallback_increase=1.0,
        maximum_confidence_weighted_coverage_loss=1.0,
    )
    policy = SimpleNamespace(
        confidence_level=0.95,
        required_range_metrics=(metric,),
        required_range_issuance=(issuance,),
        minimum_regime_classifier_accuracy_lower_bound=0.0,
        minimum_regime_classifier_recall_lower_bound=0.0,
        minimum_range_set_precision_lower_bound=0.0,
        minimum_range_set_recall_lower_bound=0.0,
        maximum_regime_classifier_false_routing_upper_bound=1.0,
        maximum_false_active_band_upper_bound=1.0,
        maximum_harmful_fraction=1.0,
        minimum_regime_classifier_ood_abstention_lower_bound=0.0,
        minimum_range_classifier_ood_abstention_lower_bound=0.0,
        minimum_regime_classifier_ood_cases=0,
        minimum_range_classifier_ood_cases=0,
        minimum_range_exact_set_accuracy_lower_bound=0.0,
        maximum_regime_classifier_brier_score_upper_bound=1.0,
        maximum_weather_multiclass_brier_score_upper_bound=1.0,
        maximum_range_multilabel_brier_score_upper_bound=1.0,
        maximum_weather_ood_brier_score_upper_bound=1.0,
        maximum_range_ood_brier_score_upper_bound=1.0,
        minimum_material_clusters=1,
        minimum_regime_classifier_clusters=1,
        minimum_range_band_clusters=1,
        minimum_deployment_metric_cell_events=1,
        minimum_continuous_metric_cell_events=1,
        allow_shadow_small_sample_bootstrap=False,
    )
    plan = SimpleNamespace(
        promotion_experiment_family=SimpleNamespace(total_family_size=1),
        range_band_contracts=(
            SimpleNamespace(
                registered_active_range_regime_sets=(
                    ("near_range",),
                )
            ),
        ),
    )
    classifier_counts = (
        ("known_weather", 10_000),
        ("known_range", 10_000),
        ("weather_ood", 10_000),
        ("range_ood", 10_000),
        ("brier_valid", 10_000),
        ("known_weather:convective", 10_000),
        ("known_range:near_range", 10_000),
    )
    return plan, policy, classifier_counts


def _truncated_interval_oracle(
    location: float,
    scale: float,
    lower: float,
    upper: float,
    threshold: float,
) -> tuple[float, float]:
    """Independent NormalDist oracle for one lower-truncated interval."""

    normal = NormalDist()
    truncation = (threshold - location) / scale
    lower_z = (lower - location) / scale
    upper_z = (upper - location) / scale
    truncation_cdf = normal.cdf(truncation)
    normalizer = 1.0 - truncation_cdf
    lower_cdf = (normal.cdf(lower_z) - truncation_cdf) / normalizer
    upper_cdf = (normal.cdf(upper_z) - truncation_cdf) / normalizer
    midpoint = 0.5 * (lower_cdf + upper_cdf)
    nll = -math.log(
        (normal.cdf(upper_z) - normal.cdf(lower_z)) / normalizer
    )
    return nll, normal.inv_cdf(midpoint)


def _bounded_mean_oracle(
    values: tuple[float, ...],
    clusters: tuple[str, ...],
    *,
    family_size: int,
    absolute_bound: float,
    confidence_level: float,
) -> float:
    """Independent arithmetic oracle for the analytic event UCB."""

    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    means = tuple(
        sum(items) / len(items)
        for _, items in sorted(grouped.items())
    )
    event_count = len(means)
    observed = sum(means) / event_count
    variance = sum((value - observed) ** 2 for value in means) / (
        event_count - 1
    )
    alpha = (1.0 - confidence_level) / (2.0 * family_size)
    log_term = math.log(3.0 / alpha)
    radius = math.sqrt(2.0 * variance * log_term / event_count) + (
        6.0 * absolute_bound * log_term / event_count
    )
    return min(absolute_bound, observed + radius)


class A5LossAndStatisticsTests(unittest.TestCase):
    def test_weighted_loss_promotes_accumulators_and_preserves_ad(self) -> None:
        prediction = torch.ones(
            (256, 256), dtype=torch.float16, requires_grad=True
        )
        target = torch.zeros_like(prediction)
        quality = torch.ones_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)

        loss = promotion_module.weighted_training_target_loss(
            prediction, target, valid, quality
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(float(loss.detach()), 1.0, places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertAlmostEqual(float(prediction.grad[0, 0]), 2.0 / 65536.0, places=7)

    def test_weighted_loss_handles_representable_weighted_value(self) -> None:
        prediction = torch.tensor(
            [1.0e20, 0.0], dtype=torch.float32, requires_grad=True
        )
        target = torch.zeros_like(prediction)
        quality = torch.tensor([1.0e-10, 1.0], dtype=torch.float32)
        valid = torch.ones_like(prediction, dtype=torch.bool)

        loss = promotion_module.weighted_training_target_loss(
            prediction, target, valid, quality
        )
        q0 = float(quality[0].item())
        q1 = float(quality[1].item())
        p0 = float(prediction[0].item())
        expected = q0 * p0**2 / (q0 + q1)
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertAlmostEqual(
            float(loss.detach()),
            expected,
            delta=expected * 2.0e-6,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_weighted_loss_handles_fp64_weighted_extreme(self) -> None:
        prediction = torch.tensor([1.0e200, 0.0], dtype=torch.float64)
        target = torch.zeros_like(prediction)
        quality = torch.tensor([1.0e-200, 1.0], dtype=torch.float64)
        valid = torch.ones_like(prediction, dtype=torch.bool)

        loss = promotion_module.weighted_training_target_loss(
            prediction, target, valid, quality
        )
        expected = 1.0e200 / (1.0 + 1.0e-200)
        self.assertEqual(loss.dtype, torch.float64)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertAlmostEqual(float(loss), expected, delta=expected * 1.0e-12)

    def test_weighted_loss_masks_before_overflow_and_scalar_weight_cancels(
        self,
    ) -> None:
        prediction = torch.tensor(
            [1.0e20, 2.0], dtype=torch.float32, requires_grad=True
        )
        target = torch.zeros_like(prediction)
        valid = torch.tensor([False, True])
        quality = torch.tensor([0.0, 1.0], dtype=torch.float32)

        loss = promotion_module.weighted_training_target_loss(
            prediction,
            target,
            valid,
            quality,
            sample_weight=1.0e300,
        )
        self.assertEqual(float(loss.detach()), 4.0)
        loss.backward()
        torch.testing.assert_close(
            prediction.grad,
            torch.tensor([0.0, 4.0]),
        )

    def test_weighted_loss_zero_quality_excludes_prediction_gradient(self) -> None:
        prediction = torch.tensor([2.0, 3.0], requires_grad=True)
        target = torch.zeros_like(prediction)
        valid = torch.ones(2, dtype=torch.bool)
        quality = torch.tensor([0.0, 1.0])

        loss = promotion_module.weighted_training_target_loss(
            prediction, target, valid, quality
        )
        loss.backward()
        self.assertEqual(float(loss.detach()), 9.0)
        torch.testing.assert_close(prediction.grad, torch.tensor([0.0, 6.0]))

    def test_weighted_loss_rejects_unrepresentable_mathematical_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "weighted loss is not finite"):
            promotion_module.weighted_training_target_loss(
                torch.tensor([1.0e308], dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, dtype=torch.bool),
                torch.ones(1, dtype=torch.float64),
            )

    def test_truncated_pit_matches_independent_noncentral_oracle(self) -> None:
        location = 0.75
        scale = 1.4
        threshold = 1.5
        width = 0.5
        reference_values = (1.5, 2.0, 2.5)
        nll, pit = promotion_module._truncated_gaussian_diagnostics(
            torch.tensor(reference_values, dtype=torch.float64).new_full(
                (3,), location
            ),
            torch.tensor(reference_values, dtype=torch.float64).new_full(
                (3,), scale
            ),
            torch.tensor(reference_values, dtype=torch.float64),
            support_threshold_dbz=threshold,
            reflectivity_resolution_dbz=width,
            quantization_origin_dbz=-10.0,
        )
        expected = tuple(
            _truncated_interval_oracle(
                location,
                scale,
                max(value - 0.5 * width, threshold),
                value + 0.5 * width,
                threshold,
            )
            for value in reference_values
        )
        torch.testing.assert_close(
            nll,
            torch.tensor([item[0] for item in expected], dtype=torch.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            pit,
            torch.tensor([item[1] for item in expected], dtype=torch.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertLess(float(pit[0]), float(pit[1]))

    def test_simultaneous_family_bounds_match_independent_oracle(self) -> None:
        policy = SimpleNamespace(
            confidence_level=0.95,
            allow_shadow_small_sample_bootstrap=False,
        )
        clusters = tuple(f"event-{index}" for index in range(100))
        values_a = tuple(0.1 + 0.001 * (index % 5) for index in range(100))
        values_b = tuple(0.2 - 0.001 * (index % 5) for index in range(100))
        comparisons = (
            promotion_module._UncertaintyComparison(
                component="support",
                group=None,
                values=values_a,
                clusters=clusters,
            ),
            promotion_module._UncertaintyComparison(
                component="support",
                group=("convective", "near_range"),
                values=values_b,
                clusters=clusters,
            ),
        )

        result = promotion_module._simultaneous_uncertainty_upper_bounds(
            comparisons,
            policy,
            candidate_family_size=2,
        )
        expected_a = _bounded_mean_oracle(
            values_a,
            clusters,
            family_size=4,
            absolute_bound=1.0,
            confidence_level=0.95,
        )
        expected_b = _bounded_mean_oracle(
            values_b,
            clusters,
            family_size=4,
            absolute_bound=1.0,
            confidence_level=0.95,
        )
        self.assertEqual(result.test_count, 4)
        self.assertEqual(result.method, "support_bounded_hybrid")
        self.assertAlmostEqual(
            result.comparison_bounds[("support", None)], expected_a, places=14
        )
        self.assertAlmostEqual(
            result.comparison_bounds[("support", ("convective", "near_range"))],
            expected_b,
            places=14,
        )
        self.assertAlmostEqual(
            result.bounds["support"],
            max(expected_a, expected_b),
            places=14,
        )
        self.assertGreater(
            expected_a,
            _bounded_mean_oracle(
                values_a,
                clusters,
                family_size=2,
                absolute_bound=1.0,
                confidence_level=0.95,
            ),
        )

    def test_classifier_joint_min_is_diagnostic_only(self) -> None:
        self.assertNotIn(
            "legacy_routing_brier",
            promotion_module._CLASSIFIER_SIMULTANEOUS_ENDPOINTS,
        )

    def test_preflight_brier_requirement_uses_only_proper_scores(self) -> None:
        plan, policy, classifier_counts = _preflight_fixture()
        policy.maximum_regime_classifier_brier_score_upper_bound = 1.0e-12
        with patch.object(
            promotion_module,
            "validate_neural_prior_holdout_plan",
        ):
            result = promotion_module.promotion_sample_size_preflight(
                plan,
                policy,
                available_physical_events=10_000,
                classifier_subset_event_counts=classifier_counts,
            )
        classifier_family_size = promotion_module._classifier_simultaneous_family_size(
            plan, policy
        )
        expected = promotion_module._required_bounded_mean_events(
            threshold=1.0,
            absolute_bound=1.0,
            confidence_level=policy.confidence_level,
            family_size=classifier_family_size,
        )
        brier = next(
            item
            for item in result.classifier_subset_event_counts
            if item[0] == "brier_valid"
        )
        self.assertEqual(brier[2], expected)

    def test_explicit_preflight_maps_cover_exact_policy_keys(self) -> None:
        plan, policy, classifier_counts = _preflight_fixture()
        metric = ("convective", "near_range", "log_echo_mse", 60, 10_000, 1)
        issuance = ("convective", "near_range", 60, 10_000, 1)
        with patch.object(
            promotion_module,
            "validate_neural_prior_holdout_plan",
        ):
            result = promotion_module.promotion_sample_size_preflight(
                plan,
                policy,
                available_physical_events=10_000,
                metric_cell_event_counts=(metric,),
                issuance_cell_event_counts=(issuance,),
                classifier_subset_event_counts=classifier_counts,
            )
        self.assertTrue(result.cell_feasible)
        for bad_metric in (
            (),
            (metric, metric),
            (("convective", "near_range", "other", 60, 10_000, 1),),
            (("convective", "near_range", "log_echo_mse", 60, 10_000.0, 1),),
        ):
            with self.subTest(bad_metric=bad_metric), patch.object(
                promotion_module,
                "validate_neural_prior_holdout_plan",
            ), self.assertRaisesRegex(ValueError, "metric cell preflight counts"):
                promotion_module.promotion_sample_size_preflight(
                    plan,
                    policy,
                    available_physical_events=10_000,
                    metric_cell_event_counts=bad_metric,
                    issuance_cell_event_counts=(issuance,),
                    classifier_subset_event_counts=classifier_counts,
                )
        for bad_issuance in (
            (),
            (issuance, issuance),
            (("convective", "other", 60, 10_000, 1),),
            (("convective", "near_range", 60, 10_000, 1.0),),
        ):
            with self.subTest(bad_issuance=bad_issuance), patch.object(
                promotion_module,
                "validate_neural_prior_holdout_plan",
            ), self.assertRaisesRegex(ValueError, "issuance cell preflight counts"):
                promotion_module.promotion_sample_size_preflight(
                    plan,
                    policy,
                    available_physical_events=10_000,
                    metric_cell_event_counts=(metric,),
                    issuance_cell_event_counts=bad_issuance,
                    classifier_subset_event_counts=classifier_counts,
                )


if __name__ == "__main__":
    unittest.main()
