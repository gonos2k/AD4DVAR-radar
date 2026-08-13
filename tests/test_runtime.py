from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from advar._digest import json_digest
from advar._runtime import (
    MPSBackendCertificationEvidence,
    MPSBackendCertificationPolicy,
    numerical_runtime_manifest,
    validate_mps_backend_certification,
)
from advar.mps_certification import (
    _run_pcg_fixture,
    create_mps_backend_certification_policy,
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

    @staticmethod
    def certification() -> tuple[
        MPSBackendCertificationPolicy,
        MPSBackendCertificationEvidence,
    ]:
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        cpu_runtime = numerical_runtime_manifest("cpu")
        mps_runtime = numerical_runtime_manifest("mps")
        policy = MPSBackendCertificationPolicy(
            fixture_set_digest="1" * 64,
            algorithm_source_manifest_digest="2" * 64,
            approved_certification_runner_digest="3" * 64,
            approved_runner_id="self-hosted-mac-1",
            approved_runner_public_key_hex=public_key,
            cpu_runtime_compatibility_digest=(
                cpu_runtime.compatibility_digest
            ),
            mps_runtime_compatibility_digest=(
                mps_runtime.compatibility_digest
            ),
        )
        unsigned = MPSBackendCertificationEvidence(
            policy_digest=policy.policy_digest,
            fixture_set_digest=policy.fixture_set_digest,
            cpu_runtime_compatibility_digest=(
                cpu_runtime.compatibility_digest
            ),
            cpu_runtime_exact_digest=cpu_runtime.exact_digest,
            mps_runtime_compatibility_digest=(
                mps_runtime.compatibility_digest
            ),
            mps_runtime_exact_digest=mps_runtime.exact_digest,
            algorithm_source_manifest_digest=(
                policy.algorithm_source_manifest_digest
            ),
            certification_runner_digest=(
                policy.approved_certification_runner_digest
            ),
            runner_id=policy.approved_runner_id,
            runner_public_key_hex=public_key,
            cpu_raw_result_digest="4" * 64,
            mps_raw_result_digest="5" * 64,
            mps_repeat_raw_result_digest="6" * 64,
            cpu_pcg_solution_relative_error=1.0e-6,
            mps_pcg_solution_relative_error=2.0e-6,
            pcg_cross_backend_relative_error=1.0e-6,
            cpu_pcg_true_relative_residual=1.0e-6,
            mps_pcg_true_relative_residual=2.0e-6,
            cpu_pcg_iterations=20,
            mps_pcg_iterations=21,
            frozen_stationarity_max_abs_difference=1.0e-5,
            robust_stationarity_max_abs_difference=1.0e-5,
            analysis_max_abs_difference_dbz=2.0e-4,
            forecast_max_abs_difference_dbz=3.0e-4,
            metric_score_max_abs_difference=1.0e-4,
            cpu_promotion_decision_statistic=-1.0,
            mps_promotion_decision_statistic=-0.999,
            cpu_nonfinite_fallback_reason="no_valid_observations",
            mps_nonfinite_fallback_reason="no_valid_observations",
            mps_repeat_analysis_max_abs_difference_dbz=0.0,
            mps_repeat_forecast_max_abs_difference_dbz=0.0,
            mps_repeat_decision_statistic_max_abs_difference=0.0,
            runner_signature_hex="0" * 128,
        )
        return policy, MPSBackendCertificationEvidence.sign(
            unsigned,
            runner_private_key=private_key,
        )

    def test_policy_not_evidence_controls_cpu_mps_acceptance(self) -> None:
        policy, evidence = self.certification()

        self.assertFalse(
            any("tolerance" in key for key in evidence.payload)
        )
        self.assertFalse(
            any(isinstance(value, bool) for value in evidence.payload.values())
        )

        with patch("advar._runtime.torch.backends.mps.is_available", return_value=True):
            validate_mps_backend_certification(
                evidence,
                policy,
                execution_device="mps",
                active_algorithm_source_manifest_digest="2" * 64,
                active_certification_runner_digest="3" * 64,
            )

        stricter = replace(
            policy,
            decision_statistic_tolerance=5.0e-4,
        )
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
        stricter_evidence = MPSBackendCertificationEvidence.sign(
            replace(
                evidence,
                policy_digest=stricter.policy_digest,
                runner_signature_hex="0" * 128,
            ),
            runner_private_key=private_key,
        )
        with (
            patch(
                "advar._runtime.torch.backends.mps.is_available",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "not deployment-certified"),
        ):
            validate_mps_backend_certification(
                stricter_evidence,
                stricter,
                execution_device="mps",
                active_algorithm_source_manifest_digest="2" * 64,
                active_certification_runner_digest="3" * 64,
            )

    def test_certification_rejects_tampered_signature_and_active_source(self) -> None:
        policy, evidence = self.certification()
        with (
            patch(
                "advar._runtime.torch.backends.mps.is_available",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "not deployment-certified"),
        ):
            validate_mps_backend_certification(
                evidence,
                policy,
                execution_device="mps",
                active_algorithm_source_manifest_digest="9" * 64,
                active_certification_runner_digest="3" * 64,
            )

        object.__setattr__(evidence, "runner_signature_hex", "0" * 128)
        object.__setattr__(
            evidence,
            "evidence_digest",
            json_digest(evidence.payload),
        )
        with (
            patch(
                "advar._runtime.torch.backends.mps.is_available",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "signature"),
        ):
            validate_mps_backend_certification(
                evidence,
                policy,
                execution_device="mps",
                active_algorithm_source_manifest_digest="2" * 64,
                active_certification_runner_digest="3" * 64,
            )

    def test_nontrivial_pcg_fixture_uses_multiple_krylov_iterations(self) -> None:
        solution, relative_error, true_residual, iterations = _run_pcg_fixture(
            torch.device("cpu")
        )

        self.assertGreaterEqual(iterations, 10)
        self.assertLess(relative_error, 5.0e-4)
        self.assertLess(true_residual, 5.0e-5)
        self.assertTrue(bool(torch.all(torch.isfinite(solution))))

    def test_certification_policy_requires_strict_deterministic_runtime(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(b"\x08" * 32)

        with (
            patch(
                "advar.mps_certification.torch.are_deterministic_algorithms_enabled",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "deterministic"),
        ):
            create_mps_backend_certification_policy(
                runner_id="self-hosted-mac-1",
                runner_private_key=private_key,
            )


if __name__ == "__main__":
    unittest.main()
