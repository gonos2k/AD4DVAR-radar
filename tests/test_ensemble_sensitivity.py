from __future__ import annotations

import unittest

import torch

from advar import (
    EnsembleFSOStatistics,
    compute_ensemble_fso,
    validate_ensemble_fso,
)


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
            analysis_ensemble_digest="1" * 64,
            forecast_ensemble_digest="2" * 64,
            verification_reference_digest="3" * 64,
            observation_error_model_digest="4" * 64,
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
        localized = EnsembleFSOStatistics(
            precision_weighted_innovation=(
                statistics.precision_weighted_innovation
            ),
            analysis_observation_perturbations=(
                statistics.analysis_observation_perturbations
            ),
            forecast_error_projection_by_member=(
                statistics.forecast_error_projection_by_member
            ),
            lead_minutes=statistics.lead_minutes,
            metric_names=statistics.metric_names,
            analysis_ensemble_digest=statistics.analysis_ensemble_digest,
            forecast_ensemble_digest=statistics.forecast_ensemble_digest,
            verification_reference_digest=(
                statistics.verification_reference_digest
            ),
            observation_error_model_digest=(
                statistics.observation_error_model_digest
            ),
            localization=torch.tensor(
                [[[0.5, 1.0]]],
                dtype=torch.float64,
            ),
        )
        innovation.add_(10.0)

        result = compute_ensemble_fso(localized)
        self.assertEqual(float(localized.precision_weighted_innovation[0]), 0.5)
        self.assertEqual(float(result.total_impact[0, 0]), -0.375)
        self.assertEqual(len(result.ensemble_fso_digest), 64)

    def test_rejects_uncentered_ensemble(self) -> None:
        statistics = self.statistics()
        with self.assertRaisesRegex(ValueError, "centered"):
            EnsembleFSOStatistics(
                precision_weighted_innovation=(
                    statistics.precision_weighted_innovation
                ),
                analysis_observation_perturbations=(
                    statistics.analysis_observation_perturbations + 1.0
                ),
                forecast_error_projection_by_member=(
                    statistics.forecast_error_projection_by_member
                ),
                lead_minutes=statistics.lead_minutes,
                metric_names=statistics.metric_names,
                analysis_ensemble_digest="1" * 64,
                forecast_ensemble_digest="2" * 64,
                verification_reference_digest="3" * 64,
                observation_error_model_digest="4" * 64,
            )

    def test_beneficial_fraction_excludes_zero_localization_support(self) -> None:
        statistics = self.statistics()
        localized = EnsembleFSOStatistics(
            precision_weighted_innovation=(
                statistics.precision_weighted_innovation
            ),
            analysis_observation_perturbations=(
                statistics.analysis_observation_perturbations
            ),
            forecast_error_projection_by_member=(
                statistics.forecast_error_projection_by_member
            ),
            lead_minutes=statistics.lead_minutes,
            metric_names=statistics.metric_names,
            analysis_ensemble_digest=statistics.analysis_ensemble_digest,
            forecast_ensemble_digest=statistics.forecast_ensemble_digest,
            verification_reference_digest=(
                statistics.verification_reference_digest
            ),
            observation_error_model_digest=(
                statistics.observation_error_model_digest
            ),
            localization=torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        )
        result = compute_ensemble_fso(localized)
        self.assertEqual(float(result.beneficial_fraction[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
