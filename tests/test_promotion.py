from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import advar.promotion as promotion_module
import advar.ledger as ledger_module
from advar.nowcast import _validate_input_plan_resolution
from advar import (
    EpisodeLedger,
    DeployedNeuralPriorPolicy,
    ForecastRunContract,
    NowcastConfig,
    NeuralPriorCandidateManifest,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorPromotionPolicy,
    NeuralPriorProbabilityContract,
    NeuralPriorRegimeClassifier,
    NeuralPriorStateContract,
    NeuralPriorStateCalibrationPlan,
    NeuralPriorStateCalibrationTarget,
    PriorUncertaintyTarget,
    PriorUncertaintyTargetPlan,
    PromotionMetricScale,
    ProspectiveInterventionDecision,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    VerificationBundle,
    compute_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_promotion,
    validate_neural_prior_promotion_applicability,
    verification_plan_digest,
    neural_prior_state_censor_policy_digest,
)
from advar.sensitivity import _LearningPolicyTrustStore


class _FixedRegimeClassifier(nn.Module):
    def __init__(self, regime_logits: tuple[float, ...], range_logits: tuple[float, ...]):
        super().__init__()
        self.register_buffer("regime_logits", torch.tensor(regime_logits))
        self.register_buffer("range_logits", torch.tensor(range_logits))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retained = frames.sum() * 0.0
        return self.regime_logits + retained, self.range_logits + retained


class NeuralPriorPromotionTests(unittest.TestCase):
    def state_contract(self) -> NeuralPriorStateContract:
        return NeuralPriorStateContract(
            state_product_digest="a" * 64,
            state_qc_pipeline_digest="9" * 64,
            state_mask_policy_digest="3" * 64,
            state_censor_policy_digest=neural_prior_state_censor_policy_digest(
                detection_limit_dbz=5.0,
                censor_temperature_dbz=1.0,
                censored_background_policy="floor",
                minimum_dbz=-10.0,
                maximum_dbz=70.0,
            ),
            support_threshold_dbz=5.0,
            minimum_state_dbz=-10.0,
            maximum_state_dbz=70.0,
            minimum_state_std_dbz=0.1,
            maximum_state_std_dbz=20.0,
        )

    def probability_contract(self) -> NeuralPriorProbabilityContract:
        return NeuralPriorProbabilityContract(
            support_threshold_dbz=5.0,
            support_product_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )

    def verification_plan(self, valid_time: str) -> str:
        return verification_plan_digest(
            valid_times=(valid_time,),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
        )

    def plan(self) -> NeuralPriorHoldoutPlan:
        input_plans = tuple(
            promotion_module.NeuralPriorInputPlan(
                valid_times=(issue,),
                grid_contract_digest="2" * 64,
                radar_product_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
                background_cycle_rule_digest=("1" if index == 1 else "2") * 64,
                mask_policy_digest="3" * 64,
                observation_valid_time=issue,
                input_available_time=issue,
                decision_deadline=(
                    datetime.fromisoformat(issue.replace("Z", "+00:00"))
                    + timedelta(minutes=2)
                ).isoformat(),
                publication_time=(
                    datetime.fromisoformat(issue.replace("Z", "+00:00"))
                    + timedelta(minutes=5)
                ).isoformat(),
            )
            for index, issue in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        target_plans = tuple(
            PriorUncertaintyTargetPlan(
                plan_id=f"uncertainty-{index}",
                target_kind="independent_sensor",
                source_identity_digest="6" * 64,
                qc_pipeline_digest="9" * 64,
                mask_policy_digest="3" * 64,
                censor_policy_digest=self.state_contract().state_censor_policy_digest,
                floor_representation_contract_digest="e" * 64,
                grid_contract_digest="2" * 64,
                feature_exclusion_contract_digest="5" * 64,
                independence_evidence_digest="8" * 64,
                target_valid_time=valid_time,
                prior_probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
            )
            for index, valid_time in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        state_target_plans = tuple(
            NeuralPriorStateCalibrationPlan(
                plan_id=f"state-calibration-{index}",
                target_kind="withheld_target_mask",
                source_identity_digest="a" * 64,
                qc_pipeline_digest="9" * 64,
                mask_policy_digest="3" * 64,
                censor_policy_digest=self.state_contract().state_censor_policy_digest,
                floor_representation_contract_digest="e" * 64,
                grid_contract_digest="2" * 64,
                feature_exclusion_contract_digest="5" * 64,
                independence_evidence_digest="8" * 64,
                target_valid_time=valid_time,
                state_contract_digest=self.state_contract().contract_digest,
                support_threshold_dbz=5.0,
            )
            for index, valid_time in enumerate(
                ("2026-08-09T00:00:00Z", "2026-08-10T00:00:00Z"),
                start=1,
            )
        )
        range_labels = ("near_range", "far_range")
        range_masks = (
            torch.ones((2, 2), dtype=torch.bool),
            torch.zeros((2, 2), dtype=torch.bool),
        )
        range_contracts = tuple(
            promotion_module.RangeBandContract(
                case_id=f"case-{index}",
                range_regime_labels=range_labels,
                range_band_mask_digests=tuple(
                    promotion_module.tensor_digest(mask)
                    for mask in (
                        range_masks
                        if index == 1
                        else tuple(reversed(range_masks))
                    )
                ),
                reference_active_range_regimes=(
                    "near_range" if index == 1 else "far_range",
                ),
                grid_contract_digest="2" * 64,
            )
            for index in (1, 2)
        )
        classifier_manifest = promotion_module.RegimeClassifierManifest(
            classifier_digest="e" * 64,
            training_dataset_digest="4" * 64,
            training_case_ids=("classifier-training-case",),
            training_storm_ids=("classifier-training-storm",),
            training_days=("2026-06-01",),
            training_time_windows=((
                "2026-06-01T00:00:00Z",
                "2026-06-01T01:00:00Z",
            ),),
            training_algorithm_digest="5" * 64,
            numerical_runtime_digest=(
                promotion_module.numerical_runtime_identity_digest("cpu")
            ),
            reference_label_contract_digest="7" * 64,
        )
        return NeuralPriorHoldoutPlan(
            plan_id="holdout-plan",
            parent_prior_digest="d" * 64,
            candidate_family_digests=("c" * 64,),
            cases=(
                NeuralPriorHoldoutPlanCase(
                    case_id="case-1",
                    storm_id="storm-1",
                    day="2026-08-08",
                    radar_id="radar-1",
                    regime="convective",
                    range_regime="near_range",
                    input_plan_digest=input_plans[0].plan_digest,
                    verification_plan_digest=self.verification_plan(
                        "2026-08-09T01:00:00Z"
                    ),
                    metric_contract_digest="b" * 64,
                    uncertainty_target_plan_digest=target_plans[0].plan_digest,
                    state_calibration_target_plan_digest=(
                        state_target_plans[0].plan_digest
                    ),
                    range_band_contract_digest=range_contracts[0].contract_digest,
                    reference_active_range_regimes=("near_range",),
                    issue_time="2026-08-09T00:00:00Z",
                ),
                NeuralPriorHoldoutPlanCase(
                    case_id="case-2",
                    storm_id="storm-2",
                    day="2026-08-09",
                    radar_id="radar-1",
                    regime="stratiform",
                    range_regime="far_range",
                    input_plan_digest=input_plans[1].plan_digest,
                    verification_plan_digest=self.verification_plan(
                        "2026-08-10T01:00:00Z"
                    ),
                    metric_contract_digest="b" * 64,
                    uncertainty_target_plan_digest=target_plans[1].plan_digest,
                    state_calibration_target_plan_digest=(
                        state_target_plans[1].plan_digest
                    ),
                    range_band_contract_digest=range_contracts[1].contract_digest,
                    reference_active_range_regimes=("far_range",),
                    issue_time="2026-08-10T00:00:00Z",
                ),
            ),
            input_plans=input_plans,
            uncertainty_target_plans=target_plans,
            state_calibration_target_plans=state_target_plans,
            range_band_contracts=range_contracts,
            regime_classifier_manifests=(classifier_manifest,),
            reference_label_contract_digest="7" * 64,
            registered_at="2026-08-07T00:00:00Z",
        )

    def completed_case(self, index: int) -> NeuralPriorHoldoutCase:
        planned = self.plan().cases[index - 1]
        uncertainty_target = self.uncertainty_target(index)
        state_target = self.state_target(index)
        full_digest = ("1" if index == 1 else "2") * 64
        return NeuralPriorHoldoutCase(
            case_id=planned.case_id,
            storm_id=planned.storm_id,
            day=planned.day,
            radar_id=planned.radar_id,
            regime=planned.regime,
            range_regime=planned.range_regime,
            input_plan_digest=planned.input_plan_digest,
            input_plan_resolution_digest=(
                promotion_module._forecast_input_plan_resolution_digest(
                    input_plan_digest=planned.input_plan_digest,
                    full_analysis_input_digest=full_digest,
                )
            ),
            input_bundle_digest=("e" if index == 1 else "f") * 64,
            full_analysis_input_digest=full_digest,
            fixed_input_context_digest=("a" if index == 1 else "b") * 64,
            observation_quality_weight_digest=(
                "c" if index == 1 else "d"
            ) * 64,
            observation_std_dbz_digest=("4" if index == 1 else "5") * 64,
            verification_plan_digest=planned.verification_plan_digest,
            verification_bundle_digest="a" * 64,
            metric_contract_digest=planned.metric_contract_digest,
            uncertainty_target_plan_digest=(
                planned.uncertainty_target_plan_digest
            ),
            uncertainty_target_digest=uncertainty_target.target_digest,
            state_calibration_target_plan_digest=(
                planned.state_calibration_target_plan_digest
            ),
            state_calibration_target_digest=state_target.target_digest,
            prior_state_contract_digest=self.state_contract().contract_digest,
            issue_time=planned.issue_time,
            candidate_forecast_digest=("6" if index == 1 else "8") * 64,
            parent_forecast_digest=("7" if index == 1 else "9") * 64,
            candidate_prior_application_digest=("3" if index == 1 else "4") * 64,
            parent_prior_application_digest=("5" if index == 1 else "6") * 64,
            candidate_inference_evidence_digest=("7" if index == 1 else "8") * 64,
            parent_inference_evidence_digest=("9" if index == 1 else "0") * 64,
            prior_probability_contract_digest=(
                self.probability_contract().contract_digest
            ),
            range_band_contract_digest=planned.range_band_contract_digest,
            reference_active_range_regimes=(
                planned.reference_active_range_regimes
            ),
        )

    def state_target(self, index: int) -> NeuralPriorStateCalibrationTarget:
        plan = self.plan()
        target_plan = plan.state_calibration_target_plans[index - 1]
        verification = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v2",
        )
        return NeuralPriorStateCalibrationTarget.from_verification_bundle(
            plan=target_plan,
            verification=verification,
        )

    def uncertainty_target(self, index: int) -> PriorUncertaintyTarget:
        plan = self.plan()
        target_plan = plan.uncertainty_target_plans[index - 1]
        verification = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=(
                target_plan.reflectivity_resolution_dbz
            ),
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v2",
        )
        return PriorUncertaintyTarget.from_verification_bundle(
            plan=target_plan,
            verification=verification,
        )

    def test_uncertainty_target_requires_its_planned_verification_source(self) -> None:
        target_plan = self.plan().uncertainty_target_plans[0]
        wrong_source = VerificationBundle(
            frames_dbz=torch.ones((1, 2, 2)),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest="f" * 64,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            PriorUncertaintyTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong_source,
            )
        legacy_measurement_source = VerificationBundle(
            frames_dbz=torch.ones((1, 2, 2)),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            PriorUncertaintyTarget.from_verification_bundle(
                plan=target_plan,
                verification=legacy_measurement_source,
            )
        self.assertFalse(hasattr(PriorUncertaintyTarget, "from_tensors"))

    def test_input_plan_must_match_actual_operational_identity(self) -> None:
        plan = self.plan().input_plans[0]
        identity = promotion_module.OperationalDataIdentity(
            radar_class="test",
            qc_pipeline_digest=plan.qc_pipeline_digest,
            observation_error_model_digest="4" * 64,
            background_model_digest="5" * 64,
            radar_product_digest=plan.radar_product_digest,
            background_cycle_rule_digest=plan.background_cycle_rule_digest,
            mask_policy_digest=plan.mask_policy_digest,
        )
        grid = SimpleNamespace(
            digest=plan.grid_contract_digest,
            valid_times=plan.valid_times,
        )
        _validate_input_plan_resolution(plan.json, identity.json, grid)
        for field in (
            "radar_product_digest",
            "background_cycle_rule_digest",
            "mask_policy_digest",
        ):
            changed = replace(identity, **{field: "f" * 64})
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "operational data identity"):
                    _validate_input_plan_resolution(
                        plan.json,
                        changed.json,
                        grid,
                    )

    def manifest(self) -> NeuralPriorCandidateManifest:
        plan = self.plan()
        return NeuralPriorCandidateManifest(
            candidate_prior_digest="c" * 64,
            parent_prior_digest="d" * 64,
            training_learning_approval_digests=("a" * 64,),
            training_intervention_digests=("f" * 64,),
            training_dataset_digest="1" * 64,
            candidate_training_manifest_digest="2" * 64,
            parent_training_manifest_digest="3" * 64,
            model_contract_digest="2" * 64,
            feature_schema_digest="4" * 64,
            algorithm_bundle_digest="3" * 64,
            numerical_runtime_digest="4" * 64,
            holdout_dataset_digest=plan.holdout_dataset_digest,
            holdout_plan_digest=plan.plan_digest,
            training_case_ids=("training-case",),
            training_input_bundle_digests=("0" * 64,),
            training_storm_ids=("training-storm",),
            training_days=("2026-07-01",),
            training_radars=("radar-1",),
            training_regimes=("convective",),
            training_time_windows=(
                (
                    "2026-07-01T00:00:00Z",
                    "2026-07-01T01:00:00Z",
                ),
            ),
            holdout_cases=(self.completed_case(1), self.completed_case(2)),
        )

    def evaluation(
        self,
        index: int,
        change: float,
        *,
        end_to_end: float | None = None,
        candidate_issuance: float = 0.0,
        prior_residual_mean_abs: float = 0.5,
        prior_underdispersion_fraction: float = 0.0,
        prior_sample_count: int = 16,
        prior_candidate_valid_fraction: float = 1.0,
        prior_parent_valid_fraction: float = 1.0,
        prior_candidate_valid_area_km2: float = 4.0,
        prior_echo_intensity_nll: float = 0.5,
        parent_prior_echo_intensity_nll: float = 0.5,
        prior_support_brier_score: float = 0.05,
        parent_prior_support_brier_score: float = 0.05,
        prior_echo_support_miss_score: float = 0.05,
        parent_prior_echo_support_miss_score: float = 0.05,
        prior_echo_object_miss_score: float = 0.05,
        parent_prior_echo_object_miss_score: float = 0.05,
        prior_clear_sky_false_echo_score: float = 0.05,
        parent_prior_clear_sky_false_echo_score: float = 0.05,
        parent_prior_underdispersion_fraction: float | None = None,
        state_candidate_gaussian_nll: float = 0.5,
        state_parent_gaussian_nll: float = 0.5,
        state_candidate_support_brier_score: float = 0.05,
        state_parent_support_brier_score: float = 0.05,
        state_candidate_false_support_score: float = 0.05,
        state_parent_false_support_score: float = 0.05,
        echo_available: bool = True,
        clear_available: bool = True,
        regime_classifier_digest: str = "e" * 64,
        classified_regime: str | None = None,
        classified_range_regimes: tuple[str, ...] | None = None,
        classifier_regime_confidence: float = 1.0,
        classifier_range_confidence: float = 1.0,
        classifier_regime_entropy: float = 0.0,
        classifier_is_ood: bool = False,
        classifier_reference_agreement: bool = True,
        range_change: float | None = None,
    ) -> promotion_module.PriorHoldoutEvaluation:
        manifest = self.manifest()
        case = manifest.holdout_cases[index - 1]
        plan = self.plan()
        classifier_manifest = plan.regime_classifier_manifests[0]
        reference_ranges = case.reference_active_range_regimes
        predicted_ranges = (
            (case.range_regime,)
            if classified_range_regimes is None
            else classified_range_regimes
        )
        reference_set = set(reference_ranges)
        predicted_set = set(predicted_ranges)
        intersection = len(reference_set & predicted_set)
        range_precision = (
            1.0 if not predicted_set and not reference_set else
            intersection / len(predicted_set) if predicted_set else 0.0
        )
        range_recall = (
            1.0 if not reference_set else intersection / len(reference_set)
        )
        false_active_fraction = (
            len(predicted_set - reference_set) / len(predicted_set)
            if predicted_set
            else 0.0
        )
        echo_count = (
            prior_sample_count
            if echo_available and not clear_available
            else prior_sample_count // 2
            if echo_available
            else 0
        )
        clear_count = (
            prior_sample_count
            if clear_available and not echo_available
            else prior_sample_count - prior_sample_count // 2
            if clear_available
            else 0
        )
        range_contract = next(
            item
            for item in plan.range_band_contracts
            if item.contract_digest == case.range_band_contract_digest
        )
        range_component_names = tuple(
            name
            for name in promotion_module._UNCERTAINTY_COMPONENT_NAMES
            if (
                echo_available
                or name not in (
                    "intensity",
                    "echo_miss",
                    "object_miss",
                    "underdispersion",
                )
            )
            and (clear_available or name != "clear")
        )
        range_components = tuple((name, 0.0) for name in range_component_names)
        range_component_counts = tuple(
            (name, 8)
            for name in range_component_names
        )
        band_change = change if range_change is None else range_change
        range_evaluations = tuple(
            promotion_module.RangeBandEvaluation(
                range_regime=range_regime,
                range_band_mask_digest=range_contract.mask_digest(range_regime),
                metric_change=torch.tensor([[band_change]], dtype=torch.float64),
                end_to_end_metric_change=torch.tensor(
                    [[band_change if end_to_end is None else end_to_end]],
                    dtype=torch.float64,
                ),
                metric_available=torch.tensor([[True]]),
                uncertainty_component_differences=range_components,
                uncertainty_component_sample_counts=range_component_counts,
                evaluated_area_km2=1.0,
            )
            for range_regime in reference_ranges
        )
        return promotion_module._new_prior_holdout_evaluation(
            holdout_plan_digest=manifest.holdout_plan_digest,
            candidate_manifest_digest=manifest.manifest_digest,
            candidate_prior_digest=manifest.candidate_prior_digest,
            parent_prior_digest=manifest.parent_prior_digest,
            case_id=case.case_id,
            storm_id=case.storm_id,
            day=case.day,
            radar_id=case.radar_id,
            regime=case.regime,
            range_regime=case.range_regime,
            reference_active_range_regimes=reference_ranges,
            range_band_contract_digest=case.range_band_contract_digest,
            range_band_evaluations=range_evaluations,
            regime_classifier_digest=regime_classifier_digest,
            regime_classifier_manifest_digest=classifier_manifest.manifest_digest,
            regime_classification_evidence_digest=(
                ("e" if index == 1 else "f") * 64
            ),
            classified_regime=(
                case.regime if classified_regime is None else classified_regime
            ),
            classified_range_regimes=(
                predicted_ranges
            ),
            classifier_regime_confidence=classifier_regime_confidence,
            classifier_range_confidence=classifier_range_confidence,
            classifier_regime_entropy=classifier_regime_entropy,
            classifier_is_ood=classifier_is_ood,
            classifier_reference_agreement=classifier_reference_agreement,
            classifier_weather_reference_agreement=(
                (case.regime if classified_regime is None else classified_regime)
                == case.regime
            ),
            classifier_range_set_precision=range_precision,
            classifier_range_set_recall=range_recall,
            classifier_range_exact_set_match=(predicted_set == reference_set),
            classifier_false_active_band_fraction=false_active_fraction,
            classifier_reference_range_is_ood=not reference_ranges,
            classifier_numerical_runtime_digest=(
                classifier_manifest.numerical_runtime_digest
            ),
            classifier_input_dtype=str(torch.float32),
            classifier_input_device="cpu",
            classifier_weather_top1_top2_gap=1.0,
            classifier_minimum_range_presence_margin=0.5,
            candidate_forecast_digest=case.candidate_forecast_digest,
            parent_forecast_digest=case.parent_forecast_digest,
            candidate_prior_application_digest=(
                case.candidate_prior_application_digest
            ),
            parent_prior_application_digest=case.parent_prior_application_digest,
            candidate_inference_evidence_digest=(
                case.candidate_inference_evidence_digest
            ),
            parent_inference_evidence_digest=(case.parent_inference_evidence_digest),
            metric_change=torch.tensor([[change]], dtype=torch.float64),
            candidate_issuance_effect=torch.tensor(
                [[candidate_issuance]], dtype=torch.float64
            ),
            parent_issuance_effect=torch.zeros((1, 1), dtype=torch.float64),
            end_to_end_metric_change=torch.tensor(
                [[change if end_to_end is None else end_to_end]],
                dtype=torch.float64,
            ),
            metric_available=torch.tensor([[True]]),
            lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            verification_digest="a" * 64,
            metric_contract_digest="b" * 64,
            coverage_candidate=torch.tensor([1.0], dtype=torch.float64),
            coverage_parent=torch.tensor([1.0], dtype=torch.float64),
            coverage_common=torch.tensor([1.0], dtype=torch.float64),
            newly_issued_fraction=torch.tensor([0.0], dtype=torch.float64),
            withdrawn_fraction=torch.tensor([0.0], dtype=torch.float64),
            prior_conditional_pit_residual_mean_abs=(
                prior_residual_mean_abs if echo_available else None
            ),
            prior_conditional_underdispersion_fraction=(
                prior_underdispersion_fraction if echo_available else None
            ),
            prior_echo_intensity_nll=(
                prior_echo_intensity_nll if echo_available else None
            ),
            prior_support_brier_score=prior_support_brier_score,
            prior_echo_support_miss_score=(
                prior_echo_support_miss_score if echo_available else None
            ),
            prior_echo_object_miss_score=(
                prior_echo_object_miss_score if echo_available else None
            ),
            prior_clear_sky_false_echo_score=(
                prior_clear_sky_false_echo_score if clear_available else None
            ),
            parent_prior_conditional_underdispersion_fraction=(
                None
                if not echo_available
                else prior_underdispersion_fraction
                if parent_prior_underdispersion_fraction is None
                else parent_prior_underdispersion_fraction
            ),
            parent_prior_echo_intensity_nll=(
                parent_prior_echo_intensity_nll if echo_available else None
            ),
            parent_prior_support_brier_score=parent_prior_support_brier_score,
            parent_prior_echo_support_miss_score=(
                parent_prior_echo_support_miss_score if echo_available else None
            ),
            parent_prior_echo_object_miss_score=(
                parent_prior_echo_object_miss_score if echo_available else None
            ),
            parent_prior_clear_sky_false_echo_score=(
                parent_prior_clear_sky_false_echo_score
                if clear_available
                else None
            ),
            prior_echo_intensity_status=(
                "available" if echo_available else "not_applicable"
            ),
            prior_clear_sky_status=(
                "available" if clear_available else "not_applicable"
            ),
            prior_candidate_valid_fraction=prior_candidate_valid_fraction,
            prior_parent_valid_fraction=prior_parent_valid_fraction,
            prior_candidate_valid_area_km2=prior_candidate_valid_area_km2,
            prior_abstention_increase_vs_parent=(
                prior_parent_valid_fraction - prior_candidate_valid_fraction
            ),
            prior_uncertainty_target_digest=case.uncertainty_target_digest,
            prior_uncertainty_sample_count=prior_sample_count,
            prior_echo_intensity_sample_count=echo_count,
            prior_clear_sky_sample_count=clear_count,
            prior_echo_area_km2=echo_count * 0.25,
            prior_clear_sky_area_km2=clear_count * 0.25,
            prior_echo_object_count=1 if echo_available else 0,
            state_candidate_gaussian_nll=state_candidate_gaussian_nll,
            state_parent_gaussian_nll=state_parent_gaussian_nll,
            state_candidate_pit_residual_mean_abs=0.5,
            state_parent_pit_residual_mean_abs=0.5,
            state_candidate_underdispersion_fraction=0.0,
            state_parent_underdispersion_fraction=0.0,
            state_candidate_support_brier_score=(
                state_candidate_support_brier_score
            ),
            state_parent_support_brier_score=state_parent_support_brier_score,
            state_candidate_echo_support_miss_score=0.05,
            state_parent_echo_support_miss_score=0.05,
            state_candidate_echo_object_miss_score=0.05,
            state_parent_echo_object_miss_score=0.05,
            state_candidate_false_support_score=(
                state_candidate_false_support_score
            ),
            state_parent_false_support_score=state_parent_false_support_score,
            state_candidate_valid_brier_score=0.05,
            state_parent_valid_brier_score=0.05,
            state_calibration_target_digest=case.state_calibration_target_digest,
            state_calibration_sample_count=16,
            state_calibration_echo_sample_count=8,
            state_calibration_clear_sample_count=8,
            state_calibration_echo_object_count=1,
            issue_time=case.issue_time,
            verification_valid_times=(f"2026-08-{8 + index:02d}T01:00:00Z",),
        )

    def policy(self) -> NeuralPriorPromotionPolicy:
        return NeuralPriorPromotionPolicy(
            metric_scales=(PromotionMetricScale("log_echo_mse", 1.0, 0.01),),
            approved_candidate_manifest_digests=(self.manifest().manifest_digest,),
            approved_holdout_plan_digests=(self.plan().plan_digest,),
            approved_metric_contract_digests=("b" * 64,),
            deployment_regime_classifier_digest="e" * 64,
            deployment_regime_classifier_manifest_digest=(
                self.plan().regime_classifier_manifests[0].manifest_digest
            ),
            minimum_holdout_cases=2,
            minimum_material_cases=2,
            minimum_material_case_fraction=1.0,
            minimum_independent_cases=2,
            minimum_distinct_storms=2,
            minimum_distinct_days=2,
            minimum_distinct_radars=1,
            minimum_distinct_regimes=2,
            minimum_distinct_range_regimes=2,
            minimum_material_clusters=2,
            minimum_prior_echo_cases=2,
            minimum_prior_clear_cases=2,
            minimum_prior_echo_clusters=2,
            minimum_prior_clear_clusters=2,
            minimum_uncertainty_cases_per_regime=1,
            minimum_echo_cases_per_regime=1,
            minimum_clear_cases_per_regime=1,
            minimum_uncertainty_clusters_per_regime=1,
            minimum_echo_clusters_per_regime=1,
            minimum_clear_clusters_per_regime=1,
            minimum_prior_echo_pixels_per_case=1,
            minimum_prior_clear_pixels_per_case=1,
            minimum_prior_echo_area_km2_per_case=0.0,
            minimum_prior_clear_area_km2_per_case=0.0,
            minimum_prior_echo_objects_per_case=1,
            minimum_bootstrap_tail_replicates=1,
            minimum_state_calibration_samples_per_case=1,
            minimum_state_calibration_cases_per_regime=1,
            minimum_state_calibration_clusters_per_regime=1,
            minimum_regime_classifier_ood_cases=0,
            minimum_range_classifier_ood_cases=0,
            minimum_range_band_cases=1,
            minimum_range_band_clusters=1,
            minimum_range_band_area_km2=0.0,
            minimum_weather_top1_top2_gap=0.0,
            minimum_range_presence_margin=0.0,
            minimum_beneficial_fraction=1.0,
            maximum_harmful_fraction=0.0,
            minimum_mean_normalized_improvement=0.1,
            bootstrap_samples=64,
        )

    def compute(self, evaluations):
        policy = self.policy()
        return self.compute_with_policy(evaluations, policy)

    def compute_with_policy(self, evaluations, policy):
        plan = self.plan()
        manifest = self.manifest()
        default_classifier = plan.regime_classifier_manifests[0]
        if (
            policy.deployment_regime_classifier_digest
            != default_classifier.classifier_digest
        ):
            classifier_manifest = replace(
                default_classifier,
                classifier_digest=policy.deployment_regime_classifier_digest,
            )
            plan = replace(
                plan,
                regime_classifier_manifests=(classifier_manifest,),
            )
            manifest = replace(
                manifest,
                holdout_plan_digest=plan.plan_digest,
                holdout_dataset_digest=plan.holdout_dataset_digest,
            )
            policy = replace(
                policy,
                approved_candidate_manifest_digests=(manifest.manifest_digest,),
                approved_holdout_plan_digests=(plan.plan_digest,),
                deployment_regime_classifier_manifest_digest=(
                    classifier_manifest.manifest_digest
                ),
            )
            evaluations = tuple(
                promotion_module._new_prior_holdout_evaluation(
                    **{
                        key: value
                        for key, value in evaluation.__dict__.items()
                        if key not in ("contract", "evaluation_digest")
                    }
                    | {
                        "holdout_plan_digest": plan.plan_digest,
                        "candidate_manifest_digest": manifest.manifest_digest,
                        "regime_classifier_manifest_digest": (
                            classifier_manifest.manifest_digest
                        ),
                    }
                )
                for evaluation in evaluations
            )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=_LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            ),
        ):
            return compute_neural_prior_promotion(
                manifest,
                plan,
                evaluations,
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_promotes_only_independent_material_holdout_cases(self) -> None:
        result = self.compute((self.evaluation(1, -0.2), self.evaluation(2, -0.3)))
        self.assertTrue(result.eligible)
        validate_neural_prior_promotion(result)

    def test_current_promotion_round_trips_durable_v10_evidence(self) -> None:
        evaluations = (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        evidence = self.compute(evaluations)
        manifest = self.manifest()
        plan = self.plan()
        policy = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            ledger = EpisodeLedger(Path(directory))
            with sqlite3.connect(ledger.index_path) as connection:
                connection.execute(
                    "INSERT INTO neural_prior_holdout_plans "
                    "(plan_digest, plan_id, plan_json, policy_digest, "
                    "trust_store_digest, registered_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_digest,
                        plan.plan_id,
                        json.dumps(asdict(plan), sort_keys=True),
                        "6" * 64,
                        "7" * 64,
                        plan.registered_at,
                        "2026-08-07T00:00:00+00:00",
                    ),
                )
                approval_schema = connection.execute(
                    "PRAGMA table_info(variational_learning_approvals)"
                ).fetchall()
                approval_columns = [str(row[1]) for row in approval_schema]
                approval_overrides: dict[str, object] = {
                    "learning_result_digest": "8" * 64,
                    "approval_evidence_digest": "a" * 64,
                    "created_at": "2026-07-01T00:00:00+00:00",
                }
                approval_values = [
                    approval_overrides.get(
                        str(row[1]),
                        0 if str(row[2]).upper() == "INTEGER" else 0.0
                        if str(row[2]).upper() == "REAL"
                        else "",
                    )
                    for row in approval_schema
                ]
                connection.execute(
                    f"INSERT INTO variational_learning_approvals "
                    f"({','.join(approval_columns)}) VALUES "
                    f"({','.join('?' for _ in approval_columns)})",
                    approval_values,
                )
                connection.execute(
                    "INSERT INTO prospective_intervention_decisions "
                    "(decision_digest, decision_id, decision_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        "e" * 64,
                        "training-decision",
                        json.dumps(
                            {"decision_basis_digest": "a" * 64},
                            sort_keys=True,
                        ),
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "INSERT INTO realized_intervention_receipts "
                    "(receipt_digest, decision_digest, receipt_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        "f" * 64,
                        "e" * 64,
                        json.dumps(
                            {
                                "actual_input_bundle_digest": "0" * 64,
                                "decision_digest": "e" * 64,
                            },
                            sort_keys=True,
                        ),
                        "2026-07-01T00:00:00+00:00",
                    ),
                )
            trust = _LearningPolicyTrustStore(
                approved_policy_digests=frozenset((policy.digest,)),
                content_digest="b" * 64,
            )
            with patch.object(
                promotion_module,
                "_load_learning_policy_trust_store",
                return_value=trust,
            ):
                stored = ledger.append_neural_prior_promotion(
                    evidence,
                    manifest,
                    plan,
                    evaluations,
                    policy=policy,
                    policy_trust_store_path="/etc/advar/learning-policies.json",
                )
            loaded = ledger.load_neural_prior_promotion(stored)
            self.assertEqual(loaded.promotion_evidence_digest, stored)
            self.assertEqual(loaded.contract, "neural-prior-promotion-evidence-v10")

    def test_promotion_requires_every_preregistered_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "every planned case"):
            self.compute((self.evaluation(1, -0.2),))

    def test_prior_holdout_evidence_has_no_intervention_selection_fields(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        self.assertFalse(hasattr(evaluation, "intervention_digest"))
        self.assertFalse(hasattr(evaluation, "population_contract"))

    def test_end_to_end_harm_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2, end_to_end=2.0),
                self.evaluation(2, -0.3),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("excessive_end_to_end_degradation", result.rejection_reasons)

    def test_nonfinite_issuance_effect_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite metrics"):
            self.evaluation(1, -0.2, candidate_issuance=float("nan"))

    def test_unreliable_prior_uncertainty_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_underdispersion_fraction=0.5,
                ),
                self.evaluation(2, -0.3),
            )
        )
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)

    def test_prior_uncertainty_must_not_regress_against_parent(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_intensity_nll=3.9,
                    parent_prior_echo_intensity_nll=1.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_intensity_nll=3.9,
                    parent_prior_echo_intensity_nll=1.0,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_probability_skill_cannot_hide_unreliable_state_uncertainty(
        self,
    ) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    state_candidate_gaussian_nll=100.0,
                ),
                self.evaluation(2, -0.3),
            )
        )

        self.assertFalse(result.eligible)
        self.assertFalse(result.state_calibration_eligible)
        self.assertIn("unreliable_state_head", result.rejection_reasons)

    def test_state_uncertainty_regression_against_parent_blocks_promotion(
        self,
    ) -> None:
        result = self.compute(
            tuple(
                self.evaluation(
                    index,
                    -0.1 * index,
                    state_candidate_gaussian_nll=2.0,
                    state_parent_gaussian_nll=0.5,
                )
                for index in (1, 2)
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("inferior_state_head", result.rejection_reasons)

    def test_state_false_support_is_a_direct_promotion_guard(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    state_candidate_false_support_score=1.0,
                    state_parent_false_support_score=0.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    state_candidate_false_support_score=1.0,
                    state_parent_false_support_score=0.0,
                ),
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("unreliable_state_head", result.rejection_reasons)
        self.assertIn("inferior_state_head", result.rejection_reasons)

    def test_clear_sky_gain_cannot_hide_echo_intensity_regression(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_intensity_nll=1.5,
                    parent_prior_echo_intensity_nll=0.5,
                    prior_clear_sky_false_echo_score=0.0,
                    parent_prior_clear_sky_false_echo_score=0.2,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_intensity_nll=1.5,
                    parent_prior_echo_intensity_nll=0.5,
                    prior_clear_sky_false_echo_score=0.0,
                    parent_prior_clear_sky_false_echo_score=0.2,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_clear_sky_hurdle_score_is_floor_representation_invariant(self) -> None:
        application = SimpleNamespace(
            truncated_location_dbz=torch.zeros((1, 3)),
            truncated_scale_dbz=torch.ones((1, 3)),
            event_probability=torch.full((1, 3), 0.2),
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        support = torch.zeros((1, 3), dtype=torch.bool)
        scores = [
            promotion_module._prior_uncertainty_scores(
                application,
                torch.full((1, 3), floor),
                support,
                mask,
                support_threshold_dbz=5.0,
            )
            for floor in (-10.0, 0.0, 4.9)
        ]
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[1], scores[2])
        self.assertIsNone(scores[0].echo_intensity_nll)
        self.assertEqual(scores[0].echo_sample_count, 0)
        self.assertEqual(scores[0].clear_sample_count, 3)

    def test_pure_clear_and_pure_echo_cases_use_component_applicability(self) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_cases=1,
            minimum_prior_clear_cases=1,
            minimum_prior_echo_clusters=1,
            minimum_prior_clear_clusters=1,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    echo_available=False,
                    clear_available=True,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    echo_available=True,
                    clear_available=False,
                ),
            ),
            policy,
        )

        self.assertTrue(result.eligible)
        self.assertEqual(result.prior_echo_case_count, 1)
        self.assertEqual(result.prior_clear_sky_case_count, 1)

    def test_one_cluster_regime_cannot_support_promotion(self) -> None:
        policy = replace(
            self.policy(),
            minimum_uncertainty_clusters_per_regime=2,
        )
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )

        self.assertTrue(result.eligible)
        self.assertFalse(result.deployment_eligible)
        self.assertEqual(result.certified_applicability_regime_groups, ())

    def test_simultaneous_uncertainty_count_covers_family_components_groups(
        self,
    ) -> None:
        components = (
            "intensity",
            "support",
            "echo_miss",
            "object_miss",
            "clear",
            "underdispersion",
        )
        groups = (None,) + tuple((f"regime-{i}", "range") for i in range(6))
        comparisons = tuple(
            promotion_module._UncertaintyComparison(
                component=component,
                group=group,
                values=(0.0, 0.0),
                clusters=(("storm-1", "day-1", "radar-1"),
                          ("storm-2", "day-2", "radar-1")),
            )
            for component in components
            for group in groups
        )

        result = promotion_module._simultaneous_uncertainty_upper_bounds(
            comparisons,
            self.policy(),
            candidate_family_size=10,
        )

        self.assertEqual(result.test_count, 10 * 6 * 7)
        self.assertEqual(result.method, "exact_sign_enumeration")
        self.assertEqual(result.effective_replicates, 4)

    def test_truncated_tail_score_is_stable_and_differentiable(self) -> None:
        for lower in (0.0, 5.0, 8.0, 12.0, 20.0):
            location = torch.tensor([5.0 - lower], dtype=torch.float32, requires_grad=True)
            scale = torch.ones(1, dtype=torch.float32, requires_grad=True)
            reference = torch.tensor([5.0], dtype=torch.float32)
            nll, _ = promotion_module._truncated_gaussian_diagnostics(
                location,
                scale,
                reference,
                support_threshold_dbz=5.0,
            )
            log_lower = torch.special.log_ndtr(
                torch.tensor(-lower, dtype=torch.float64)
            )
            log_upper = torch.special.log_ndtr(
                torch.tensor(-(lower + 0.5), dtype=torch.float64)
            )
            expected = -torch.log(-torch.expm1(log_upper - log_lower))
            torch.testing.assert_close(nll[0], expected)
            nll.sum().backward()
            self.assertTrue(torch.isfinite(location.grad).all())
            self.assertTrue(torch.isfinite(scale.grad).all())

    def test_echo_miss_is_not_hidden_by_clear_sky_prevalence(self) -> None:
        application = SimpleNamespace(
            truncated_location_dbz=torch.full((1000,), 5.0),
            truncated_scale_dbz=torch.ones(1000),
            event_probability=torch.zeros(1000),
        )
        support = torch.zeros(1000, dtype=torch.bool)
        support[0] = True
        scores = promotion_module._prior_uncertainty_scores(
            application,
            torch.where(support, 5.0, -10.0),
            support,
            torch.ones(1000, dtype=torch.bool),
            support_threshold_dbz=5.0,
        )
        self.assertAlmostEqual(scores.support_brier_score, 0.001)
        self.assertEqual(scores.echo_support_miss_score, 1.0)
        self.assertEqual(scores.clear_sky_false_echo_score, 0.0)

    def test_conditional_pit_replaces_raw_truncated_z_score(self) -> None:
        lower = 8.0
        log_lower_survival = torch.special.log_ndtr(
            torch.tensor(-lower, dtype=torch.float64)
        )
        left, right = lower, lower + 4.0
        for _ in range(80):
            midpoint = (left + right) / 2.0
            log_ratio = float(
                torch.special.log_ndtr(
                    torch.tensor(-midpoint, dtype=torch.float64)
                )
                - log_lower_survival
            )
            conditional_cdf = -torch.expm1(
                torch.tensor(log_ratio, dtype=torch.float64)
            ).item()
            if conditional_cdf < 0.5:
                left = midpoint
            else:
                right = midpoint
        reference = torch.tensor([(left + right) / 2.0], dtype=torch.float64)
        _, pit = promotion_module._truncated_gaussian_diagnostics(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            reference,
            support_threshold_dbz=8.0,
            reflectivity_resolution_dbz=1.0e-6,
            quantization_origin_dbz=float(reference[0]),
        )
        self.assertGreater(float(reference[0]), 8.0)
        self.assertAlmostEqual(float(pit[0]), 0.0, places=6)

    def test_quantized_threshold_uses_interval_midpoint_pit(self) -> None:
        threshold = 5.0
        width = 0.5
        location = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        reference = torch.tensor(
            [threshold, threshold + width, threshold + 2.0 * width],
            dtype=torch.float32,
        )

        nll, pit = promotion_module._truncated_gaussian_diagnostics(
            location,
            scale,
            reference,
            support_threshold_dbz=threshold,
            reflectivity_resolution_dbz=width,
            quantization_origin_dbz=-10.0,
        )

        self.assertTrue(torch.all(torch.isfinite(nll)))
        self.assertTrue(torch.all(torch.isfinite(pit)))
        self.assertGreater(float(pit[0]), -2.0)
        self.assertNotAlmostEqual(float(pit[0]), -8.126, places=2)
        with self.assertRaisesRegex(ValueError, "off its declared lattice"):
            promotion_module._truncated_gaussian_diagnostics(
                torch.tensor([5.0]),
                torch.tensor([1.0]),
                torch.tensor([5.3]),
                support_threshold_dbz=threshold,
                reflectivity_resolution_dbz=width,
                quantization_origin_dbz=-10.0,
            )
        with self.assertRaisesRegex(ValueError, "threshold.*lattice"):
            promotion_module._truncated_gaussian_diagnostics(
                torch.tensor([5.0]),
                torch.tensor([1.0]),
                torch.tensor([5.5]),
                support_threshold_dbz=5.1,
                reflectivity_resolution_dbz=width,
                quantization_origin_dbz=-10.0,
            )

    def test_quantized_gaussian_upper_tail_has_finite_large_nll(self) -> None:
        nll, pit = promotion_module._quantized_gaussian_diagnostics(
            torch.tensor([-10.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32),
            torch.tensor([5.5], dtype=torch.float32),
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
            support_threshold_dbz=5.0,
            threshold_bin_convention="threshold_edge_centered_bins",
        )
        self.assertTrue(torch.all(torch.isfinite(nll)))
        self.assertTrue(torch.all(torch.isfinite(pit)))
        self.assertGreater(float(nll[0]), 1_000.0)

    def test_state_target_requires_v2_measurement_attestation(self) -> None:
        target_plan = self.plan().state_calibration_target_plans[0]
        wrong = VerificationBundle(
            frames_dbz=torch.tensor([[[10.0, 1.0], [10.0, 1.0]]]),
            valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            valid_times=(target_plan.target_valid_time,),
            grid_contract_digest=target_plan.grid_contract_digest,
            radar_product_digest=target_plan.source_identity_digest,
            qc_pipeline_digest=target_plan.qc_pipeline_digest,
            mask_policy_digest="f" * 64,
            censor_policy_digest=target_plan.censor_policy_digest,
            reflectivity_resolution_dbz=target_plan.reflectivity_resolution_dbz,
            quantization_origin_dbz=target_plan.quantization_origin_dbz,
            threshold_bin_convention=target_plan.threshold_bin_convention,
            floor_representation_contract_digest=(
                target_plan.floor_representation_contract_digest
            ),
            contract="radar-verification-bundle-v2",
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            NeuralPriorStateCalibrationTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong,
            )
        wrong_censor = replace(
            wrong,
            mask_policy_digest=target_plan.mask_policy_digest,
            censor_policy_digest="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "source disagrees"):
            NeuralPriorStateCalibrationTarget.from_verification_bundle(
                plan=target_plan,
                verification=wrong_censor,
            )

    def test_component_geometry_rejects_single_pixel_evidence(self) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_pixels_per_case=2,
            minimum_prior_echo_area_km2_per_case=1.0,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(1, -0.2, prior_sample_count=2),
                self.evaluation(2, -0.3, prior_sample_count=2),
            ),
            policy,
        )
        self.assertIn("insufficient_component_samples", result.rejection_reasons)
        self.assertIn("insufficient_component_area", result.rejection_reasons)

    def test_missed_echo_objects_fail_the_probability_guard(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_echo_object_miss_score=1.0,
                    parent_prior_echo_object_miss_score=0.0,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_echo_object_miss_score=1.0,
                    parent_prior_echo_object_miss_score=0.0,
                ),
            )
        )
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_bootstrap_tail_resolution_fails_closed(self) -> None:
        policy = replace(
            self.policy(),
            bootstrap_samples=10,
            minimum_bootstrap_tail_replicates=1,
        )
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )
        self.assertIn(
            "insufficient_bootstrap_tail_resolution",
            result.rejection_reasons,
        )

    def test_family_ten_with_one_thousand_bootstraps_is_rejected(self) -> None:
        plan = replace(
            self.plan(),
            candidate_family_digests=("c" * 64,)
            + tuple(character * 64 for character in "012345678"),
        )
        manifest = replace(
            self.manifest(),
            holdout_plan_digest=plan.plan_digest,
        )
        rebound: list[promotion_module.PriorHoldoutEvaluation] = []
        for evaluation in (
            self.evaluation(1, -0.2),
            self.evaluation(2, -0.3),
        ):
            values = {
                name: value
                for name, value in evaluation.__dict__.items()
                if name not in {"contract", "evaluation_digest"}
            }
            values["holdout_plan_digest"] = plan.plan_digest
            values["candidate_manifest_digest"] = manifest.manifest_digest
            rebound.append(
                promotion_module._new_prior_holdout_evaluation(**values)
            )
        policy = replace(
            self.policy(),
            approved_candidate_manifest_digests=(manifest.manifest_digest,),
            approved_holdout_plan_digests=(plan.plan_digest,),
            bootstrap_samples=1000,
            minimum_bootstrap_tail_replicates=20,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            result = compute_neural_prior_promotion(
                manifest,
                plan,
                tuple(rebound),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )
        self.assertAlmostEqual(result.cluster_bootstrap_tail_replicates, 2.5)
        self.assertIn(
            "insufficient_bootstrap_tail_resolution",
            result.rejection_reasons,
        )

    def test_uncertified_regime_requires_parent_fallback(self) -> None:
        evidence = self.compute(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3))
        )
        validate_neural_prior_promotion_applicability(
            evidence,
            regime="convective",
            range_regime="near_range",
        )
        with self.assertRaisesRegex(ValueError, "parent prior"):
            validate_neural_prior_promotion_applicability(
                evidence,
                regime="unseen",
                range_regime="far_range",
            )

    def test_classifier_attested_uncertified_regime_selects_parent(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((0.0, 12.0, -12.0), (0.0, 12.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unseen", "unknown"),
            range_regime_labels=("near_range", "unseen_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
        )

        classified = classifier.classify(frames, input_run=run)
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                deployment_policy,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, parent)
        self.assertEqual(selection.selected_role, "parent")
        self.assertEqual(selection.fallback_reason, "uncertified_regime")
        self.assertEqual(
            selection.full_analysis_input_digest,
            run.full_analysis_input_digest,
        )

    def test_regime_classifier_rehashes_mutable_execution_contract(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0), (12.0,)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unknown"),
            range_regime_labels=("near_range",),
            classifier_algorithm_digest="1" * 64,
        )
        classifier.classify(frames, input_run=run)
        classifier.regime_labels = ("stratiform", "unknown")

        with self.assertRaisesRegex(ValueError, "artifact changed"):
            classifier.classify(frames, input_run=run)

    def test_classifier_attested_certified_regime_selects_candidate(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 0.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unseen", "unknown"),
            range_regime_labels=("near_range", "unseen_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
        )

        classified = classifier.classify(frames, input_run=run)
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                deployment_policy,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, candidate)
        self.assertEqual(selection.selected_role, "candidate")
        self.assertEqual(selection.fallback_reason, "certified_candidate")

        unapproved = replace(deployment_policy, minimum_regime_confidence=0.01)
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "unapproved"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                unapproved,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        changed_trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((unapproved.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [unapproved.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=changed_trust,
        ):
            _, changed_selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                unapproved,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        self.assertNotEqual(
            changed_selection.selection_digest,
            selection.selection_digest,
        )
        self.assertEqual(
            changed_selection.deployment_policy_digest,
            unapproved.policy_digest,
        )
        tampered = replace(deployment_policy, minimum_regime_confidence=0.01)
        object.__setattr__(
            tampered,
            "policy_digest",
            deployment_policy.policy_digest,
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "policy digest mismatch"):
            promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classified,
                tampered,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

    def test_all_active_range_bands_must_be_certified(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 12.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )
        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classifier.classify(frames, input_run=run),
                deployment_policy,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )
        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "uncertified_range_band")

    def test_regime_classifier_evidence_is_bound_to_current_input(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0, -12.0), (12.0, 0.0)).eval(),
            example_frames=frames,
            regime_labels=("convective", "stratiform", "unknown"),
            range_regime_labels=("near_range", "far_range"),
            classifier_algorithm_digest="1" * 64,
        )

        with self.assertRaisesRegex(ValueError, "input or artifact changed"):
            classifier.classify(frames + 1.0, input_run=run)

    def test_classifier_holdout_rejects_constant_false_routing(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2),
                self.evaluation(
                    2,
                    -0.3,
                    classified_regime="convective",
                    classified_range_regimes=("near_range",),
                    classifier_reference_agreement=False,
                ),
            )
        )
        self.assertFalse(result.regime_classifier_validated)
        self.assertIn(
            "unreliable_regime_classifier",
            result.rejection_reasons,
        )

    def test_all_active_range_classifier_fails_set_precision(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    classified_range_regimes=("near_range", "far_range"),
                ),
                self.evaluation(
                    2,
                    -0.3,
                    classified_range_regimes=("near_range", "far_range"),
                ),
            )
        )

        self.assertFalse(result.regime_classifier_validated)
        self.assertAlmostEqual(result.range_set_precision, 0.5)
        self.assertEqual(result.range_exact_set_accuracy, 0.0)
        self.assertIn("unreliable_range_classifier", result.rejection_reasons)

    def test_range_band_harm_is_not_hidden_by_whole_domain_skill(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.2, range_change=-0.2),
                self.evaluation(2, -0.3, range_change=1.1),
            )
        )

        self.assertTrue(result.eligible)
        self.assertIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_classifier_training_storm_cannot_overlap_holdout(self) -> None:
        plan = self.plan()
        overlapping = replace(
            plan.regime_classifier_manifests[0],
            training_storm_ids=(plan.cases[0].storm_id,),
        )

        with self.assertRaisesRegex(ValueError, "classifier training overlaps"):
            replace(plan, regime_classifier_manifests=(overlapping,))

    def test_promotion_rejects_classifier_outside_preregistered_family(self) -> None:
        plan = self.plan()
        manifest = self.manifest()
        policy = replace(
            self.policy(),
            deployment_regime_classifier_digest="f" * 64,
            deployment_regime_classifier_manifest_digest="0" * 64,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((policy.digest,)),
            content_digest="b" * 64,
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ), self.assertRaisesRegex(ValueError, "not preregistered"):
            compute_neural_prior_promotion(
                manifest,
                plan,
                (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
                policy=policy,
                policy_trust_store_path="/etc/advar/learning-policies.json",
            )

    def test_ambiguous_current_range_branch_falls_back_to_parent(self) -> None:
        frames = torch.zeros((3, 2, 2))
        run = ForecastRunContract.from_inputs(
            NowcastConfig(),
            frames,
            torch.ones_like(frames, dtype=torch.bool),
            None,
        )
        classifier = NeuralPriorRegimeClassifier(
            _FixedRegimeClassifier((12.0, 0.0), (1.4500102,)).eval(),
            example_frames=frames,
            regime_labels=("convective", "unknown"),
            range_regime_labels=("near_range",),
            classifier_algorithm_digest="1" * 64,
        )
        promotion_policy = replace(
            self.policy(),
            deployment_regime_classifier_digest=classifier.classifier_digest,
        )
        evidence = self.compute_with_policy(
            (
                self.evaluation(
                    1,
                    -0.2,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    regime_classifier_digest=classifier.classifier_digest,
                ),
            ),
            promotion_policy,
        )
        candidate = SimpleNamespace(neural_prior_digest="c" * 64)
        parent = SimpleNamespace(neural_prior_digest="d" * 64)
        deployment_policy = DeployedNeuralPriorPolicy(
            candidate_prior_digest=candidate.neural_prior_digest,
            parent_prior_digest=parent.neural_prior_digest,
            promotion_evidence_digest=evidence.promotion_evidence_digest,
            regime_classifier_digest=classifier.classifier_digest,
            regime_classifier_manifest_digest=(
                evidence.deployment_regime_classifier_manifest_digest
            ),
            minimum_range_presence_margin=0.05,
        )
        trust = _LearningPolicyTrustStore(
            approved_policy_digests=frozenset((deployment_policy.policy_digest,)),
            content_digest=promotion_module.json_digest(
                {
                    "contract": "advar-learning-policy-trust-store-v1",
                    "approved_policy_digests": [deployment_policy.policy_digest],
                }
            ),
        )

        with patch.object(
            promotion_module,
            "_load_learning_policy_trust_store",
            return_value=trust,
        ):
            selected, selection = promotion_module._select_deployed_prior(
                candidate,
                parent,
                evidence,
                classifier.classify(frames, input_run=run),
                deployment_policy,
                policy_trust_store_path="/etc/advar/deployment-policies.json",
            )

        self.assertIs(selected, parent)
        self.assertEqual(selection.fallback_reason, "ambiguous_classifier_branch")

    def test_deployment_requires_preregistered_ood_validation(self) -> None:
        policy = replace(self.policy(), minimum_regime_classifier_ood_cases=1)
        result = self.compute_with_policy(
            (self.evaluation(1, -0.2), self.evaluation(2, -0.3)),
            policy,
        )
        self.assertFalse(result.deployment_eligible)
        self.assertIn("unreliable_regime_classifier", result.rejection_reasons)

    def test_partial_regime_certification_only_falls_back_for_missing_group(
        self,
    ) -> None:
        policy = replace(
            self.policy(),
            minimum_prior_echo_cases=1,
            minimum_prior_echo_clusters=1,
        )
        result = self.compute_with_policy(
            (
                self.evaluation(1, -0.2),
                self.evaluation(2, -0.3, echo_available=False),
            ),
            policy,
        )
        self.assertTrue(result.eligible)
        self.assertIn(
            ("convective", "near_range"),
            result.certified_applicability_regime_groups,
        )
        self.assertNotIn(
            ("stratiform", "far_range"),
            result.certified_applicability_regime_groups,
        )

    def test_regime_specific_uncertainty_regression_is_not_averaged_away(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_support_brier_score=0.2,
                    parent_prior_support_brier_score=0.1,
                ),
                self.evaluation(
                    2,
                    -0.3,
                    prior_support_brier_score=0.0,
                    parent_prior_support_brier_score=0.1,
                ),
            )
        )
        self.assertFalse(result.eligible)
        self.assertIn("inferior_prior_uncertainty", result.rejection_reasons)

    def test_self_selected_one_percent_validity_blocks_promotion(self) -> None:
        result = self.compute(
            (
                self.evaluation(
                    1,
                    -0.2,
                    prior_candidate_valid_fraction=0.01,
                    prior_candidate_valid_area_km2=0.04,
                ),
                self.evaluation(2, -0.3),
            )
        )

        self.assertFalse(result.eligible)
        self.assertIn("unreliable_prior_uncertainty", result.rejection_reasons)

    def test_end_to_end_harm_is_checked_when_common_skill_is_immaterial(self) -> None:
        result = self.compute(
            (
                self.evaluation(1, -0.001, end_to_end=2.0),
                self.evaluation(2, -0.3),
            )
        )
        self.assertIn("excessive_end_to_end_degradation", result.rejection_reasons)

    def test_training_and_holdout_storms_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "storms must be disjoint"):
            replace(self.manifest(), training_storm_ids=("storm-1",))

    def test_training_and_holdout_inputs_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "inputs must be disjoint"):
            replace(
                self.manifest(),
                training_input_bundle_digests=(
                    self.manifest().holdout_cases[0].input_bundle_digest,
                ),
            )

    def test_mutated_metric_is_detected(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        evaluation.metric_change[0, 0] = -10.0
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.compute((evaluation, self.evaluation(2, -0.3)))

    def test_promotion_audit_payload_recomputes_tensor_digests(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        payload = ledger_module._evaluation_audit_payload(evaluation)
        decoded = ledger_module._decode_evaluation_audit_payloads([payload])
        self.assertEqual(decoded[0].evaluation_digest, evaluation.evaluation_digest)

        tensor_payload = payload["metric_change"]
        assert isinstance(tensor_payload, dict)
        values = tensor_payload["values"]
        assert isinstance(values, list)
        values[0][0] = 99.0
        with self.assertRaisesRegex(ValueError, "tensor digest mismatch"):
            ledger_module._decode_evaluation_audit_payloads([payload])

    def test_legacy_promotion_tensor_lists_load_as_audit_only(self) -> None:
        evaluation = self.evaluation(1, -0.2)
        payload = ledger_module._evaluation_audit_payload(evaluation)
        for name, value in tuple(payload.items()):
            if isinstance(value, dict) and value.get("kind") == "tensor":
                payload[name] = value["values"]

        decoded = ledger_module._decode_evaluation_audit_payloads([payload])

        self.assertIsInstance(
            decoded[0],
            ledger_module.LegacyPromotionEvaluationAudit,
        )
        self.assertEqual(decoded[0].evaluation_digest, evaluation.evaluation_digest)
        self.assertFalse(decoded[0].content_digest_verified)
        self.assertFalse(decoded[0].statistical_reuse_permitted)
        with self.assertRaisesRegex(ValueError, "audit-only"):
            self.compute(decoded)

    def test_v6_hurdle_evaluation_remains_content_verified_audit_only(
        self,
    ) -> None:
        payload = ledger_module._evaluation_audit_payload(
            self.evaluation(1, -0.2)
        )
        payload["contract"] = "prior-holdout-evaluation-v6"
        payload.pop("prior_echo_intensity_status")
        payload.pop("prior_clear_sky_status")
        normalized = dict(payload)
        normalized.pop("evaluation_digest")
        for name, value in tuple(normalized.items()):
            if isinstance(value, dict) and value.get("kind") == "tensor":
                normalized[name] = value["digest"]
        payload["evaluation_digest"] = promotion_module.json_digest(normalized)

        decoded = ledger_module._decode_evaluation_audit_payloads([payload])

        self.assertIsInstance(
            decoded[0],
            ledger_module.LegacyPromotionEvaluationAudit,
        )
        self.assertTrue(decoded[0].content_digest_verified)
        self.assertFalse(decoded[0].statistical_reuse_permitted)

    def test_direct_evaluation_construction_is_disabled(self) -> None:
        with self.assertRaisesRegex(TypeError, "from_forecasts"):
            promotion_module.PriorHoldoutEvaluation()

    def test_legacy_intervention_is_not_a_prospective_receipt(self) -> None:
        legacy = RealizedObservationIntervention(
            intervention_id="legacy",
            intervention_type="realized_qc_intervention",
            action_digest="a" * 64,
            applied_time="2026-08-08T00:00:00Z",
            actual_input_before_digest="b" * 64,
            actual_input_after_digest="c" * 64,
            outcome_resolution_contract_digest="d" * 64,
            execution_policy_digest="e" * 64,
            execution_trust_store_digest="f" * 64,
            predicted_normalized_benefit=0.0,
            resolved_normalized_benefit=0.0,
            learning_result_digest="1" * 64,
            learning_approval_evidence_digest="2" * 64,
            counterfactual_perturbation_digest="3" * 64,
            linearization_digest="4" * 64,
        )
        self.assertNotIsInstance(legacy, RealizedInterventionReceipt)

    def test_prospective_decision_direct_construction_is_disabled(self) -> None:
        with self.assertRaisesRegex(TypeError, "from_policy"):
            ProspectiveInterventionDecision()
        with self.assertRaisesRegex(TypeError, "from_decision"):
            RealizedInterventionReceipt()

    def test_candidate_manifest_digest_detects_lineage_mutation(self) -> None:
        manifest = self.manifest()
        object.__setattr__(manifest, "training_storm_ids", ("changed",))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_neural_prior_candidate_manifest(manifest)

    def test_holdout_factory_uses_common_domain_and_inference_evidence(self) -> None:
        manifest = self.manifest()
        case = manifest.holdout_cases[0]
        plan = self.plan()
        planned_input = next(
            item for item in plan.input_plans
            if item.plan_digest == case.input_plan_digest
        )
        grid = SimpleNamespace(
            valid_times=planned_input.valid_times,
            cell_area_m2=1_000_000.0,
        )
        data_identity = promotion_module.OperationalDataIdentity(
            radar_class="test",
            qc_pipeline_digest=planned_input.qc_pipeline_digest,
            observation_error_model_digest="4" * 64,
            background_model_digest="5" * 64,
            radar_product_digest=planned_input.radar_product_digest,
            background_cycle_rule_digest=(
                planned_input.background_cycle_rule_digest
            ),
            mask_policy_digest=planned_input.mask_policy_digest,
        )
        common = dict(
            grid_time_contract_digest="2" * 64,
            grid_time_contract=grid,
            input_bundle_digest=case.input_bundle_digest,
            full_analysis_input_digest=case.full_analysis_input_digest,
            fixed_input_context_digest=case.fixed_input_context_digest,
            observation_quality_weight_digest=(
                case.observation_quality_weight_digest
            ),
            observation_std_dbz_digest=case.observation_std_dbz_digest,
            input_plan_digest=case.input_plan_digest,
            config=SimpleNamespace(digest="3" * 64, interval_minutes=10),
            analysis_config_digest="4" * 64,
            operational_calibration_manifest_digest="5" * 64,
            operational_data_identity_json=data_identity.json,
            operational_data_identity_digest=data_identity.digest,
            prior_model_contract_digest=manifest.model_contract_digest,
            prior_feature_schema_digest=manifest.feature_schema_digest,
            prior_inference_algorithm_digest="8" * 64,
            prior_numerical_runtime_digest="4" * 64,
            prior_dependency="radar_dependent",
            input_plan_json=planned_input.json,
            input_plan_resolution_digest=case.input_plan_resolution_digest,
        )
        candidate_app = SimpleNamespace(
            application_digest=case.candidate_prior_application_digest,
            initial_background_dbz=torch.zeros((2, 2)),
            state_background_dbz=torch.zeros((2, 2)),
            std_dbz=torch.ones((2, 2)),
            state_std_dbz=torch.ones((2, 2)),
            valid_mask=torch.tensor(
                [[True, False], [False, False]], dtype=torch.bool
            ),
            support_probability=torch.zeros((2, 2)),
            state_support_probability=torch.zeros((2, 2)),
            state_valid_probability=torch.tensor(
                [[1.0, 0.0], [0.0, 0.0]]
            ),
            truncated_location_dbz=torch.zeros((2, 2)),
            truncated_scale_dbz=torch.ones((2, 2)),
            event_probability=torch.zeros((2, 2)),
            inference_evidence=SimpleNamespace(
                evidence_digest=case.candidate_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="4" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                full_analysis_input_digest=case.full_analysis_input_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.candidate_prior_digest,
                neural_prior_digest=manifest.candidate_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=(manifest.candidate_training_manifest_digest),
                uncertainty_contract="model_spatial",
                probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
                support_event_digest=(
                    self.probability_contract().support_event_digest
                ),
                state_contract_digest=self.state_contract().contract_digest,
                prior_output_valid_time="2026-08-09T00:00:00Z",
                feature_source_valid_times=planned_input.valid_times,
                feature_source_identity_digests=("a" * 64,),
                feature_exclusion_contract_digest="5" * 64,
                feature_exclusion_mask_digest=promotion_module.tensor_digest(
                    torch.ones((3, 2, 2), dtype=torch.bool)
                ),
            ),
        )
        parent_app = SimpleNamespace(
            application_digest=case.parent_prior_application_digest,
            initial_background_dbz=torch.zeros((2, 2)),
            state_background_dbz=torch.zeros((2, 2)),
            std_dbz=torch.ones((2, 2)),
            state_std_dbz=torch.ones((2, 2)),
            valid_mask=torch.ones((2, 2), dtype=torch.bool),
            support_probability=torch.zeros((2, 2)),
            state_support_probability=torch.zeros((2, 2)),
            state_valid_probability=torch.ones((2, 2)),
            truncated_location_dbz=torch.zeros((2, 2)),
            truncated_scale_dbz=torch.ones((2, 2)),
            event_probability=torch.zeros((2, 2)),
            inference_evidence=SimpleNamespace(
                evidence_digest=case.parent_inference_evidence_digest,
                inference_algorithm_digest="8" * 64,
                numerical_runtime_digest="4" * 64,
                dependency="radar_dependent",
                input_bundle_digest=case.input_bundle_digest,
                full_analysis_input_digest=case.full_analysis_input_digest,
                input_frames_digest=promotion_module.tensor_digest(torch.zeros(3, 2, 2)),
                execution_contract_digest=manifest.parent_prior_digest,
                neural_prior_digest=manifest.parent_prior_digest,
                model_contract_digest=manifest.model_contract_digest,
                feature_schema_digest=manifest.feature_schema_digest,
                training_manifest_digest=manifest.parent_training_manifest_digest,
                uncertainty_contract="model_spatial",
                probability_contract_digest=(
                    self.probability_contract().contract_digest
                ),
                support_event_digest=(
                    self.probability_contract().support_event_digest
                ),
                state_contract_digest=self.state_contract().contract_digest,
                prior_output_valid_time="2026-08-09T00:00:00Z",
                feature_source_valid_times=planned_input.valid_times,
                feature_source_identity_digests=("a" * 64,),
                feature_exclusion_contract_digest="5" * 64,
                feature_exclusion_mask_digest=promotion_module.tensor_digest(
                    torch.ones((3, 2, 2), dtype=torch.bool)
                ),
            ),
        )
        candidate_run = SimpleNamespace(
            **common,
            neural_prior_digest=manifest.candidate_prior_digest,
            prior_application_digest=candidate_app.application_digest,
            prior_inference_evidence_digest=(
                candidate_app.inference_evidence.evidence_digest
            ),
            prior_role="candidate",
            prior_training_manifest_digest=(
                manifest.candidate_training_manifest_digest
            ),
        )
        parent_run = SimpleNamespace(
            **common,
            neural_prior_digest=manifest.parent_prior_digest,
            prior_application_digest=parent_app.application_digest,
            prior_inference_evidence_digest=(
                parent_app.inference_evidence.evidence_digest
            ),
            prior_role="parent",
            prior_training_manifest_digest=manifest.parent_training_manifest_digest,
        )
        candidate = SimpleNamespace(
            run=candidate_run,
            state=object(),
            validate_issuance=Mock(),
        )
        parent = SimpleNamespace(
            run=parent_run,
            state=object(),
            validate_issuance=Mock(),
        )
        resolved = SimpleNamespace(
            content_digest=case.verification_bundle_digest,
            valid_times=("2026-08-09T01:00:00Z",),
            grid_contract_digest="2" * 64,
            radar_product_digest="a" * 64,
            qc_pipeline_digest="9" * 64,
            valid_mask=torch.ones((6, 2, 2), dtype=torch.bool),
            frames_dbz=torch.zeros((6, 2, 2)),
        )
        verification = SimpleNamespace(valid_times=resolved.valid_times)
        config = SimpleNamespace(
            digest=case.metric_contract_digest,
            full_map_lead_minutes=(60,),
            metric_names=("log_echo_mse",),
            metric_domain="issued",
        )
        candidate_runner = SimpleNamespace(
            reproduce=Mock(),
            inference_algorithm_digest="8" * 64,
            numerical_runtime_digest="4" * 64,
            feature_exclusion_mask=torch.ones((3, 2, 2), dtype=torch.bool),
            state_contract=self.state_contract(),
            probability_contract=self.probability_contract(),
        )
        parent_runner = SimpleNamespace(
            reproduce=Mock(),
            inference_algorithm_digest="8" * 64,
            numerical_runtime_digest="4" * 64,
            feature_exclusion_mask=torch.ones((3, 2, 2), dtype=torch.bool),
            state_contract=self.state_contract(),
            probability_contract=self.probability_contract(),
        )
        regime_evidence = SimpleNamespace(
            validate_integrity=Mock(),
            full_analysis_input_digest=case.full_analysis_input_digest,
            input_frames_digest=promotion_module.tensor_digest(
                torch.zeros((3, 2, 2))
            ),
            classifier_digest="e" * 64,
            evidence_digest="f" * 64,
            numerical_runtime_digest=(
                plan.regime_classifier_manifests[0].numerical_runtime_digest
            ),
            input_dtype=str(torch.float32),
            input_device="cpu",
            regime=case.regime,
            active_range_regimes=(case.range_regime,),
            regime_confidence=1.0,
            range_regime_confidence=1.0,
            regime_entropy=0.0,
            is_ood=False,
            weather_top1_top2_gap=1.0,
            minimum_range_presence_margin=0.5,
        )
        regime_classifier = SimpleNamespace(
            classifier_digest=regime_evidence.classifier_digest,
            numerical_runtime_digest=regime_evidence.numerical_runtime_digest,
            classify=Mock(return_value=regime_evidence),
        )
        candidate_weights = torch.full((1, 2, 2), 0.5)
        parent_weights = torch.ones((1, 2, 2))
        with (
            patch.object(
                promotion_module,
                "_forecast_result_content_digest",
                side_effect=(
                    case.candidate_forecast_digest,
                    case.parent_forecast_digest,
                ),
            ),
            patch.object(
                promotion_module, "_resolve_verification", return_value=resolved
            ),
            patch.object(
                promotion_module,
                "_resolved_forecast_domain_weights",
                side_effect=(candidate_weights, parent_weights),
            ),
            patch.object(
                promotion_module,
                "_resolved_forecast_scores",
                side_effect=(
                    (torch.tensor([[0.8]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                    (torch.tensor([[0.75]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                    (torch.tensor([[0.8]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                    (torch.tensor([[0.75]]), torch.tensor([[True]])),
                    (torch.tensor([[1.0]]), torch.tensor([[True]])),
                ),
            ),
            patch.object(
                promotion_module,
                "_forecast_coverage",
                side_effect=(torch.tensor([0.9]), torch.tensor([1.0])),
            ),
        ):
            evaluation = promotion_module.PriorHoldoutEvaluation.from_forecasts(
                manifest,
                plan,
                case_id=case.case_id,
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
                candidate_prior_application=candidate_app,
                parent_prior_application=parent_app,
                candidate_prior_runner=candidate_runner,
                parent_prior_runner=parent_runner,
                input_frames_dbz=torch.zeros((3, 2, 2)),
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=(
                    plan.regime_classifier_manifests[0]
                ),
                range_band_masks={
                    "near_range": torch.ones((2, 2), dtype=torch.bool),
                    "far_range": torch.zeros((2, 2), dtype=torch.bool),
                },
            )
        self.assertAlmostEqual(float(evaluation.metric_change[0, 0]), -0.2)
        self.assertAlmostEqual(float(evaluation.end_to_end_metric_change[0, 0]), -0.25)
        self.assertEqual(evaluation.prior_uncertainty_sample_count, 4)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_fraction, 0.25)
        self.assertAlmostEqual(evaluation.prior_candidate_valid_area_km2, 1.0)
        self.assertAlmostEqual(evaluation.prior_abstention_increase_vs_parent, 0.75)
        candidate_runner.reproduce.assert_called_once()
        parent_runner.reproduce.assert_called_once()

        candidate_runner.probability_contract = NeuralPriorProbabilityContract(
            support_threshold_dbz=35.0,
            support_product_digest="6" * 64,
            qc_pipeline_digest="9" * 64,
            reflectivity_resolution_dbz=0.5,
            quantization_origin_dbz=-10.0,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "probability event"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
                manifest,
                plan,
                case_id=case.case_id,
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
                candidate_prior_application=candidate_app,
                parent_prior_application=parent_app,
                candidate_prior_runner=candidate_runner,
                parent_prior_runner=parent_runner,
                input_frames_dbz=torch.zeros((3, 2, 2)),
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=(
                    plan.regime_classifier_manifests[0]
                ),
                range_band_masks={
                    "near_range": torch.ones((2, 2), dtype=torch.bool),
                    "far_range": torch.zeros((2, 2), dtype=torch.bool),
                },
            )
        candidate_runner.probability_contract = self.probability_contract()

        parent_run.observation_quality_weight_digest = "0" * 64
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "holdout inputs disagree"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
                manifest,
                plan,
                case_id=case.case_id,
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
                candidate_prior_application=candidate_app,
                parent_prior_application=parent_app,
                candidate_prior_runner=candidate_runner,
                parent_prior_runner=parent_runner,
                input_frames_dbz=torch.zeros((3, 2, 2)),
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=plan.regime_classifier_manifests[0],
                range_band_masks={
                    "near_range": torch.ones((2, 2), dtype=torch.bool),
                    "far_range": torch.zeros((2, 2), dtype=torch.bool),
                },
            )
        parent_run.observation_quality_weight_digest = (
            case.observation_quality_weight_digest
        )
        parent_run.observation_std_dbz_digest = "0" * 64
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ), self.assertRaisesRegex(ValueError, "holdout inputs disagree"):
            promotion_module.PriorHoldoutEvaluation.from_forecasts(
                manifest,
                plan,
                case_id=case.case_id,
                candidate_forecast=candidate,
                parent_forecast=parent,
                verification=verification,
                metric_config=config,
                candidate_prior_application=candidate_app,
                parent_prior_application=parent_app,
                candidate_prior_runner=candidate_runner,
                parent_prior_runner=parent_runner,
                input_frames_dbz=torch.zeros((3, 2, 2)),
                uncertainty_target=self.uncertainty_target(1),
                state_calibration_target=self.state_target(1),
                regime_classifier=regime_classifier,
                regime_classifier_manifest=plan.regime_classifier_manifests[0],
                range_band_masks={
                    "near_range": torch.ones((2, 2), dtype=torch.bool),
                    "far_range": torch.zeros((2, 2), dtype=torch.bool),
                },
            )
        parent_run.observation_std_dbz_digest = case.observation_std_dbz_digest

        candidate_app.inference_evidence.prior_output_valid_time = (
            "2026-08-09T01:00:00Z"
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "target time"):
                promotion_module.PriorHoldoutEvaluation.from_forecasts(
                    manifest,
                    plan,
                    case_id=case.case_id,
                    candidate_forecast=candidate,
                    parent_forecast=parent,
                    verification=verification,
                    metric_config=config,
                    candidate_prior_application=candidate_app,
                    parent_prior_application=parent_app,
                    candidate_prior_runner=candidate_runner,
                    parent_prior_runner=parent_runner,
                    input_frames_dbz=torch.zeros((3, 2, 2)),
                    uncertainty_target=self.uncertainty_target(1),
                    state_calibration_target=self.state_target(1),
                    regime_classifier=regime_classifier,
                    regime_classifier_manifest=(
                        plan.regime_classifier_manifests[0]
                    ),
                    range_band_masks={
                        "near_range": torch.ones((2, 2), dtype=torch.bool),
                        "far_range": torch.zeros((2, 2), dtype=torch.bool),
                    },
                )

        candidate_app.inference_evidence.prior_output_valid_time = (
            "2026-08-09T00:00:00Z"
        )
        candidate_app.inference_evidence.feature_source_identity_digests = (
            "6" * 64,
        )
        parent_app.inference_evidence.feature_source_identity_digests = (
            "6" * 64,
        )
        with patch.object(
            promotion_module,
            "_forecast_result_content_digest",
            side_effect=(
                case.candidate_forecast_digest,
                case.parent_forecast_digest,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "visible to the features"):
                promotion_module.PriorHoldoutEvaluation.from_forecasts(
                    manifest,
                    plan,
                    case_id=case.case_id,
                    candidate_forecast=candidate,
                    parent_forecast=parent,
                    verification=verification,
                    metric_config=config,
                    candidate_prior_application=candidate_app,
                    parent_prior_application=parent_app,
                    candidate_prior_runner=candidate_runner,
                    parent_prior_runner=parent_runner,
                    input_frames_dbz=torch.zeros((3, 2, 2)),
                    uncertainty_target=self.uncertainty_target(1),
                    state_calibration_target=self.state_target(1),
                    regime_classifier=regime_classifier,
                    regime_classifier_manifest=(
                        plan.regime_classifier_manifests[0]
                    ),
                    range_band_masks={
                        "near_range": torch.ones((2, 2), dtype=torch.bool),
                        "far_range": torch.zeros((2, 2), dtype=torch.bool),
                    },
                )


if __name__ == "__main__":
    unittest.main()
