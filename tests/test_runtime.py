from __future__ import annotations

import unittest
from unittest.mock import patch

from advar._runtime import (
    MPSBackendCertificationEvidence,
    numerical_runtime_manifest,
)


class NumericalRuntimeManifestTest(unittest.TestCase):
    def test_runtime_digest_changes_with_device_model(self) -> None:
        with patch(
            "advar._runtime._execution_device_identity",
            return_value={"type": "mps", "index": 0, "model": "Apple M2"},
        ):
            first = numerical_runtime_manifest("mps")
        with patch(
            "advar._runtime._execution_device_identity",
            return_value={"type": "mps", "index": 0, "model": "Apple M4"},
        ):
            second = numerical_runtime_manifest("mps")

        self.assertEqual(first.compatibility_digest, second.compatibility_digest)
        self.assertNotEqual(first.exact_digest, second.exact_digest)

    def test_runtime_digest_changes_with_matmul_precision(self) -> None:
        with patch(
            "advar._runtime.torch.get_float32_matmul_precision",
            return_value="highest",
        ):
            first = numerical_runtime_manifest("cpu")
        with patch(
            "advar._runtime.torch.get_float32_matmul_precision",
            return_value="medium",
        ):
            second = numerical_runtime_manifest("cpu")

        self.assertNotEqual(first.compatibility_digest, second.compatibility_digest)
        self.assertNotEqual(first.exact_digest, second.exact_digest)

    def test_cpu_mps_decision_is_invariant_outside_numerical_margin(self) -> None:
        evidence = MPSBackendCertificationEvidence(
            cpu_runtime_exact_digest="1" * 64,
            mps_runtime_exact_digest="2" * 64,
            algorithm_source_manifest_digest="3" * 64,
            certification_runner_digest="4" * 64,
            pcg_extreme_scale_relative_error=1.0e-6,
            pcg_relative_error_tolerance=1.0e-5,
            frozen_stationarity_max_abs_difference=1.0e-5,
            robust_stationarity_max_abs_difference=1.0e-5,
            analysis_max_abs_difference_dbz=2.0e-4,
            score_max_abs_difference=1.0e-4,
            numerical_tolerance=1.0e-3,
            decision_statistic_max_abs_difference=2.0e-3,
            minimum_decision_margin=1.0e-2,
            decision_invariant=True,
            nonfinite_fallback_verified=True,
            deterministic_policy_verified=True,
        )

        self.assertTrue(evidence.eligible)
        self.assertFalse(
            MPSBackendCertificationEvidence(
                **{
                    key: value
                    for key, value in evidence.__dict__.items()
                    if key != "evidence_digest"
                }
                | {"minimum_decision_margin": 2.5e-3}
            ).eligible
        )


if __name__ == "__main__":
    unittest.main()
