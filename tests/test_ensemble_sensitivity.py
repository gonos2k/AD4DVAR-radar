from __future__ import annotations

import unittest
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from advar.ensemble_sensitivity import (
    EnsembleFSOStatistics,
    PrecisionOperatorArtifact,
    PrecisionWeightedInnovationEvidence,
    compute_ensemble_fso,
    validate_ensemble_fso,
    validate_precision_operator_artifact,
)


@contextmanager
def _precision_trust_store(operator_digest: str):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "precision.json"
        path.write_text(
            json.dumps(
                {
                    "contract": "advar-precision-operator-trust-store-v2",
                    "approved_operator_digests": [operator_digest],
                }
            ),
            encoding="utf-8",
        )
        metadata = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o400,
            st_uid=0,
            st_size=path.stat().st_size,
        )
        with patch("advar.ensemble_sensitivity.os.fstat", return_value=metadata):
            yield path


class EnsembleFSOTests(unittest.TestCase):
    def statistics(self) -> EnsembleFSOStatistics:
        return EnsembleFSOStatistics.from_diagonal_r(
            innovation=torch.tensor([2.0, -1.0], dtype=torch.float64),
            analysis_observation_perturbations=torch.tensor(
                [[1.0, 0.0], [-0.5, 1.0], [-0.5, -1.0]],
                dtype=torch.float64,
            ),
            forecast_error_projection_by_member=torch.tensor(
                [[[-2.0]], [[1.0]], [[1.0]]],
                dtype=torch.float64,
            ),
            inverse_observation_variance=torch.tensor(
                [0.25, 1.0],
                dtype=torch.float64,
            ),
            lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            verification_reference_digest="3" * 64,
            observation_error_model_digest="4" * 64,
            observation_ids=("o0", "o1"),
            ensemble_member_ids=("m0", "m1", "m2"),
        )

    def test_computes_direct_ensemble_observation_impact(self) -> None:
        result = compute_ensemble_fso(self.statistics())
        validate_ensemble_fso(result)

        torch.testing.assert_close(
            result.observation_impact,
            torch.tensor([[[-0.75, 0.0]]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            result.total_impact,
            torch.tensor([[-0.75]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            result.beneficial_fraction,
            torch.tensor([[0.5]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_localization_and_inputs_are_content_addressed(self) -> None:
        innovation = torch.tensor([2.0, -1.0], dtype=torch.float64)
        statistics = self.statistics()
        localized = replace(
            statistics,
            localization=torch.tensor(
                [[[0.5, 1.0]]],
                dtype=torch.float64,
            ),
        )
        innovation.add_(10.0)

        result = compute_ensemble_fso(localized)
        self.assertEqual(
            float(localized.precision_evidence.precision_weighted_innovation[0]),
            0.5,
        )
        self.assertEqual(float(result.total_impact[0, 0]), -0.375)
        self.assertEqual(len(result.ensemble_fso_digest), 64)

    def test_rejects_uncentered_ensemble(self) -> None:
        statistics = self.statistics()
        with self.assertRaisesRegex(ValueError, "digest|centered"):
            replace(
                statistics,
                analysis_observation_perturbations=(
                    statistics.analysis_observation_perturbations + 1.0
                ),
            )

    def test_beneficial_fraction_excludes_zero_localization_support(self) -> None:
        statistics = self.statistics()
        localized = replace(
            statistics,
            localization=torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        )
        result = compute_ensemble_fso(localized)
        self.assertEqual(float(result.beneficial_fraction[0, 0]), 1.0)

    def test_full_r_factory_verifies_the_precision_solve(self) -> None:
        precision = torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64))
        operator = PrecisionOperatorArtifact(
            precision=precision,
            covariance=torch.linalg.inv(precision),
            observation_ids=("o0", "o1"),
            forecast_run_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            calibration_manifest_digest="3" * 64,
        )
        with _precision_trust_store(operator.operator_digest) as trust_store:
            evidence = PrecisionWeightedInnovationEvidence.from_operator_artifact(
                innovation=torch.tensor([2.0, -4.0], dtype=torch.float64),
                operator=operator,
                trust_store_path=trust_store,
            )
        torch.testing.assert_close(
            evidence.precision_weighted_innovation,
            torch.tensor([1.0, -1.0], dtype=torch.float64),
        )
        self.assertEqual(evidence.relative_solve_residual, 0.0)

    def test_full_r_factory_requires_root_approved_operator(self) -> None:
        operator = PrecisionOperatorArtifact(
            precision=torch.eye(2, dtype=torch.float64),
            covariance=torch.eye(2, dtype=torch.float64),
            observation_ids=("o0", "o1"),
            forecast_run_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            calibration_manifest_digest="3" * 64,
        )
        with _precision_trust_store("5" * 64) as trust_store:
            with self.assertRaisesRegex(ValueError, "not trusted"):
                PrecisionWeightedInnovationEvidence.from_operator_artifact(
                    innovation=torch.ones(2, dtype=torch.float64),
                    operator=operator,
                    trust_store_path=trust_store,
                )

    def test_full_r_approval_cannot_be_reused_for_another_spd_operator(self) -> None:
        approved = PrecisionOperatorArtifact(
            precision=torch.eye(2, dtype=torch.float64),
            covariance=torch.eye(2, dtype=torch.float64),
            observation_ids=("o0", "o1"),
            forecast_run_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            calibration_manifest_digest="3" * 64,
        )
        other = PrecisionOperatorArtifact(
            precision=torch.eye(2, dtype=torch.float64) * 2.0,
            covariance=torch.eye(2, dtype=torch.float64) * 0.5,
            observation_ids=("o0", "o1"),
            forecast_run_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            calibration_manifest_digest="3" * 64,
        )
        with _precision_trust_store(approved.operator_digest) as trust_store:
            with self.assertRaisesRegex(ValueError, "not trusted"):
                PrecisionWeightedInnovationEvidence.from_operator_artifact(
                    innovation=torch.ones(2, dtype=torch.float64),
                    operator=other,
                    trust_store_path=trust_store,
                )

    def test_full_r_factory_rejects_an_unverified_precision_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            PrecisionOperatorArtifact(
                precision=torch.eye(2, dtype=torch.float64) * 0.5,
                covariance=torch.eye(2, dtype=torch.float64),
                observation_ids=("o0", "o1"),
                forecast_run_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                calibration_manifest_digest="3" * 64,
            )

    def test_full_r_rejects_an_indefinite_covariance(self) -> None:
        indefinite = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            PrecisionOperatorArtifact(
                precision=indefinite,
                covariance=indefinite,
                observation_ids=("o0", "o1"),
                forecast_run_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                calibration_manifest_digest="3" * 64,
            )

    def test_full_r_rejects_dense_resource_and_condition_budgets(self) -> None:
        identity = torch.eye(3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "observation budget"):
            PrecisionOperatorArtifact(
                precision=identity,
                covariance=identity,
                observation_ids=("o0", "o1", "o2"),
                forecast_run_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                calibration_manifest_digest="3" * 64,
                maximum_observation_count=2,
            )
        with self.assertRaisesRegex(ValueError, "condition number"):
            PrecisionOperatorArtifact(
                precision=torch.diag(torch.tensor([1.0, 1.0e-4], dtype=torch.float64)),
                covariance=torch.diag(torch.tensor([1.0, 1.0e4], dtype=torch.float64)),
                observation_ids=("o0", "o1"),
                forecast_run_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                calibration_manifest_digest="3" * 64,
                maximum_condition_number=100.0,
            )
        with self.assertRaisesRegex(ValueError, "factorization budget"):
            PrecisionOperatorArtifact(
                precision=identity,
                covariance=identity,
                observation_ids=("o0", "o1", "o2"),
                forecast_run_digest="1" * 64,
                observation_error_model_digest="2" * 64,
                calibration_manifest_digest="3" * 64,
                maximum_factorization_flops=100,
            )

    def test_full_r_operator_mutation_is_detected_before_use(self) -> None:
        operator = PrecisionOperatorArtifact(
            precision=torch.eye(2, dtype=torch.float64),
            covariance=torch.eye(2, dtype=torch.float64),
            observation_ids=("o0", "o1"),
            forecast_run_digest="1" * 64,
            observation_error_model_digest="2" * 64,
            calibration_manifest_digest="3" * 64,
        )
        operator.precision[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "operator digest"):
            validate_precision_operator_artifact(operator)
        with _precision_trust_store(operator.operator_digest) as trust_store:
            with self.assertRaisesRegex(ValueError, "operator digest"):
                PrecisionWeightedInnovationEvidence.from_operator_artifact(
                    innovation=torch.ones(2, dtype=torch.float64),
                    operator=operator,
                    trust_store_path=trust_store,
                )

    def test_nested_precision_mutation_is_rejected(self) -> None:
        statistics = self.statistics()
        statistics.precision_evidence.precision_weighted_innovation[0] = 99.0
        with self.assertRaisesRegex(ValueError, "precision evidence digest"):
            compute_ensemble_fso(statistics)

    def test_localization_mutation_is_rejected(self) -> None:
        statistics = replace(
            self.statistics(),
            localization=torch.ones((1, 1, 2), dtype=torch.float64),
        )
        assert statistics.localization is not None
        statistics.localization[0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "statistics digest"):
            compute_ensemble_fso(statistics)

    def test_member_and_observation_ordering_must_match(self) -> None:
        statistics = self.statistics()
        with self.assertRaisesRegex(ValueError, "observation ordering"):
            replace(
                statistics,
                analysis_observation_index_digest="9" * 64,
            )
        with self.assertRaisesRegex(ValueError, "member ordering"):
            replace(
                statistics,
                forecast_ensemble_member_index_digest="9" * 64,
            )

    def test_member_jackknife_uncertainty_is_reported(self) -> None:
        statistics = self.statistics()
        result = compute_ensemble_fso(statistics)
        leave_one = []
        members = statistics.analysis_observation_perturbations.shape[0]
        for omitted in range(members):
            retained = torch.arange(members) != omitted
            observation = statistics.analysis_observation_perturbations[retained]
            projection = statistics.forecast_error_projection_by_member[retained]
            observation = observation - observation.mean(dim=0)
            projection = projection - projection.mean(dim=0)
            leave_one.append(
                torch.einsum(
                    "ko,klm,o->lm",
                    observation,
                    projection,
                    statistics.precision_evidence.precision_weighted_innovation,
                )
                / (members - 2)
            )
        leave_one_tensor = torch.stack(leave_one)
        expected = torch.sqrt(
            (members - 1)
            / members
            * torch.sum(
                (leave_one_tensor - leave_one_tensor.mean(dim=0)) ** 2,
                dim=0,
            )
        )
        torch.testing.assert_close(result.total_impact_jackknife_std, expected)


if __name__ == "__main__":
    unittest.main()
