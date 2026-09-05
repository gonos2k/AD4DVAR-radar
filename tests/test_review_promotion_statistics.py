from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


# Reuse the narrow, deterministic promotion fixture without changing the
# existing test module.  The test still calls the production compute entrypoint.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_promotion as promotion_test_fixtures  # noqa: E402


class PromotionStatisticsReviewTests(unittest.TestCase):
    def test_automatic_metric_cells_ignore_bootstrap_tail_resolution(self) -> None:
        fixture = promotion_test_fixtures.NeuralPriorPromotionTests("runTest")
        evaluations = (
            fixture.evaluation(1, -0.2),
            fixture.evaluation(2, -0.3),
        )
        template = fixture.policy()

        def automatic_policy(bootstrap_samples: int):
            return replace(
                template,
                allow_shadow_small_sample_bootstrap=False,
                bootstrap_samples=bootstrap_samples,
                minimum_bootstrap_tail_replicates=20,
            )

        low_resolution = fixture.compute_with_policy(
            evaluations,
            automatic_policy(1024),
        )
        high_resolution = fixture.compute_with_policy(
            evaluations,
            automatic_policy(16_384),
        )

        # The fixed evidence reaches the actual metric-cell branch for both
        # policies.  Bootstrap resolution is retained as diagnostic evidence,
        # while the automatic bounds themselves are analytic bounded UCBs.
        self.assertTrue(low_resolution.range_metric_cell_bounds)
        self.assertEqual(
            low_resolution.range_metric_cell_bounds,
            high_resolution.range_metric_cell_bounds,
        )
        self.assertEqual(
            low_resolution.range_metric_end_to_end_cell_bounds,
            high_resolution.range_metric_end_to_end_cell_bounds,
        )
        self.assertLess(
            low_resolution.metric_cell_tail_replicates,
            20.0,
        )
        self.assertGreaterEqual(
            high_resolution.metric_cell_tail_replicates,
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
