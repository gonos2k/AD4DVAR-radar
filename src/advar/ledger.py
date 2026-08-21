"""Append-only, checksum-verified storage for sensitivity episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ._digest import tensor_digest
from ._runtime import (
    MPSBackendCertificationEvidence,
    MPSBackendCertificationPolicy,
    numerical_runtime_manifest,
)
from ._learned_input import learned_radar_input_features
from .calibration import algorithm_bundle_digest, OperationalDataIdentity
from .action_artifacts import (
    expanded_tensor_bytes,
    preflight_npz_archive,
    validate_artifact_directory,
)
from .action_contracts import canonicalize_action_frames
from .nowcast import (
    ForecastRunContract,
    _forecast_fixed_input_context_digest,
    _forecast_full_analysis_input_digest,
    _forecast_input_bundle_digest,
    _forecast_input_plan_resolution_digest,
)
from .range_geometry import (
    resolve_mosaic_range_geometry,
    resolve_range_geometry,
    restrict_range_partition_domain,
)

from .sensitivity import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES_V13,
    LearningApprovalEvidence,
    SensitivitySnapshot,
    VariationalLearningImpact,
    MosaicObservationSourceRegistry,
    ObservationRadarSource,
    ObservationErrorDerivationArtifact,
    VerificationBundle,
    VerificationObservationDerivationInputs,
    VerificationObservationErrorPlan,
    VerificationObservationMaskDerivationArtifact,
    VerificationObservationMaskEvidence,
    VerificationObservationSourceIdentity,
    derive_verification_observation_error,
    derive_verification_observation_masks,
    _load_learning_policy_trust_store,
    validate_variational_learning_impact,
)
from .intervention import (
    InterventionInputContext,
    InterventionActionGenerator,
    OperatorActionApproval,
    ProspectiveInterventionDecision,
    ReusableInterventionPolicyEvidence,
    RealizedInterventionReceipt,
    RealizedObservationIntervention,
    RetrospectiveCounterfactualReplay,
    validate_prospective_intervention,
    validate_intervention_action_transition,
    validate_action_tensor_replay,
    validate_retrospective_counterfactual_replay,
    verify_intervention_receipt_signature,
    verify_operator_action_approval_signature,
    _compute_action_safety,
    _new_prospective_decision,
    _new_operator_action_approval,
    _new_realized_intervention_receipt,
)
from .promotion import (
    LegacyNeuralPriorCandidateManifestAuditV2,
    LegacyNeuralPriorCandidateManifestAuditV3,
    LegacyNeuralPriorCandidateManifestAuditV4,
    LegacyNeuralPriorCandidateManifestAuditV5,
    LegacyNeuralPriorCandidateManifestAuditV6,
    LegacyNeuralPriorCandidateManifestAuditV7,
    LegacyNeuralPriorCandidateManifestAuditV8,
    LegacyNeuralPriorCandidateManifestAuditV9,
    LegacyNeuralPriorCandidateManifestAuditV10,
    LegacyNeuralPriorCandidateManifestAuditV11,
    LegacyNeuralPriorCandidateManifestAuditV12,
    LegacyNeuralPriorCandidateManifestAuditV13,
    LegacyNeuralPriorCandidateManifestAuditV14,
    LegacyNeuralPriorCandidateManifestAuditV15,
    LegacyNeuralPriorCandidateManifestAuditV16,
    LegacyNeuralPriorCandidateManifestAuditV17,
    LegacyNeuralPriorCandidateManifestAuditV18,
    NeuralPriorCandidateManifest,
    LegacyNeuralPriorHoldoutPlanAudit,
    LegacyNeuralPriorHoldoutPlanCase,
    LegacyNeuralPriorHoldoutPlanV2Audit,
    LegacyNeuralPriorHoldoutPlanV2Case,
    LegacyNeuralPriorHoldoutPlanV3Case,
    LegacyNeuralPriorHoldoutPlanV3Audit,
    LegacyNeuralPriorHoldoutPlanV4Audit,
    LegacyNeuralPriorHoldoutPlanV5Audit,
    LegacyNeuralPriorHoldoutPlanV6Audit,
    LegacyNeuralPriorHoldoutPlanV7Audit,
    LegacyNeuralPriorHoldoutPlanV8Audit,
    LegacyNeuralPriorHoldoutPlanV9Audit,
    LegacyNeuralPriorHoldoutPlanV10Audit,
    LegacyNeuralPriorHoldoutPlanV11Audit,
    LegacyNeuralPriorHoldoutPlanV12Audit,
    LegacyNeuralPriorHoldoutPlanV13Audit,
    LegacyNeuralPriorHoldoutPlanV14Audit,
    LegacyNeuralPriorHoldoutPlanV15Audit,
    LegacyNeuralPriorHoldoutPlanV16Audit,
    LegacyNeuralPriorHoldoutPlanV17Audit,
    LegacyNeuralPriorHoldoutPlanV18Audit,
    LegacyNeuralPriorHoldoutPlanV19Audit,
    LegacyNeuralPriorHoldoutPlanV20Audit,
    LegacyNeuralPriorHoldoutPlanV21Audit,
    LegacyNeuralPriorHoldoutPlanV22Audit,
    LegacyNeuralPriorHoldoutPlanV23Audit,
    LegacyNeuralPriorHoldoutPlanV24Audit,
    LegacyNeuralPriorHoldoutPlanV25Audit,
    NeuralPriorHoldoutCase,
    NeuralPriorHoldoutPlan,
    NeuralPriorHoldoutPlanCase,
    NeuralPriorInputPlan,
    RawObservationSlotPlan,
    CanonicalRawGridVolumeArtifact,
    CanonicalRawVolumeIdentity,
    RawVolumeAttestation,
    RawIngestorTrustStore,
    TrainingTargetSourceTrustStore,
    TrainingDatasetDerivationArtifact,
    ResolvedRawObservationReceipt,
    MissingRawObservationReceipt,
    RawObservationResolutionReceipt,
    GlobalRawVolumeResolutionReceipt,
    OperationalAnalysisInputProvenancePlan,
    OperationalRawResolutionHistoryEntry,
    OperationalRawVolumeResolutionReceipt,
    OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST,
    AnalysisInputDerivationArtifact,
    _analysis_input_derivation_from_json,
    _operational_raw_resolution_history_entry_from_json,
    _operational_raw_volume_resolution_receipt_from_json,
    _training_raw_registry_receipt_from_json,
    MeteorologicalSamplingUnit,
    GlobalSamplingReservationReceipt,
    GLOBAL_SAMPLING_REGISTRY_GENESIS_DIGEST,
    PriorUncertaintyTargetPlan,
    NeuralPriorStateCalibrationPlan,
    RangeBandContract,
    OperationalIssuanceDomainPlan,
    OperationalIssuanceDomainArtifact,
    ResolvedSourceCoverageArtifact,
    validate_resolved_source_coverage_artifact,
    _derive_analysis_inputs_from_raw_products,
    _background_input_identity_digests,
    RangeGeometryContract,
    MosaicRangeGeometryContract,
    RangeBandEvaluation,
    RegimeClassifierManifest,
    RegimeReferencePlan,
    RegimeReferenceEvidence,
    PhysicalEventCaseSpatialEvidence,
    PhysicalEventCatalogEvidence,
    PhysicalEventTrackArtifact,
    PhysicalEventCatalogPlan,
    PhysicalEventCatalogResult,
    TrustedProcessStartReceipt,
    TrustedProcessCompletionReceipt,
    ProcessLogArtifact,
    HoldoutScoringInputArtifact,
    HoldoutScoringArtifact,
    PromotionDecisionRule,
    PromotionExperimentTrial,
    PromotionExperimentFamily,
    NeuralPriorHoldoutPlanPolicy,
    NeuralPriorPromotionEvidence,
    NeuralPriorRegimeClassifier,
    RegimeClassificationEvidence,
    RangePartitionEvidence,
    LedgeredPromotionDeploymentCertificate,
    DeployedNeuralPriorPolicy,
    DeploymentBundleReleaseApproval,
    DeploymentRuntimeActivationReceipt,
    OperationalDeploymentDecisionCertificate,
    OperationalDecisionPublicationReceipt,
    OperationalDecisionActivationReceipt,
    OperationalDecisionCommitAuthorizationReceipt,
    _EpisodeLedgerOperationalDecisionClient,
    DeploymentAuthoritySigner,
    PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
    OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST,
    DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST,
    _PromotionDeploymentAuthorityTrustStore,
    _issue_ledger_issuance_receipt,
    _issue_ledgered_promotion_deployment_certificate,
    _ledgered_promotion_deployment_certificate_from_payload,
    _load_promotion_deployment_authority_trust_store,
    _trusted_authority_key,
    _validate_deployment_bundle_release_approval,
    _deployment_bundle_release_approval_from_payload,
    _validate_deployment_runtime_activation_receipt,
    _deployment_runtime_activation_receipt_from_payload,
    _validate_ledgered_promotion_deployment_certificate,
    _issue_operational_decision_ledger_receipt,
    _operational_decision_ledger_receipt_from_payload,
    _issue_operational_deployment_decision_certificate,
    _replay_operational_deployment_selection,
    _operational_deployment_decision_certificate_from_payload,
    _validate_operational_deployment_decision_certificate,
    _issue_operational_decision_publication_receipt,
    _operational_decision_publication_receipt_from_payload,
    _validate_operational_decision_publication_receipt,
    _issue_operational_decision_activation_receipt,
    _operational_decision_activation_receipt_from_payload,
    _validate_operational_decision_activation_receipt,
    _issue_operational_decision_commit_authorization_receipt,
    _operational_decision_commit_authorization_receipt_from_payload,
    _validate_operational_decision_commit_authorization_receipt,
    _operational_decision_commit_digests,
    LegacyNeuralPriorPromotionEvidenceAuditV3,
    LegacyNeuralPriorPromotionEvidenceAuditV4,
    LegacyNeuralPriorPromotionEvidenceAuditV5,
    LegacyNeuralPriorPromotionEvidenceAuditV6,
    LegacyNeuralPriorPromotionEvidenceAuditV7,
    LegacyNeuralPriorPromotionEvidenceAuditV8,
    LegacyNeuralPriorPromotionEvidenceAuditV9,
    LegacyNeuralPriorPromotionEvidenceAuditV10,
    LegacyNeuralPriorPromotionEvidenceAuditV11,
    LegacyNeuralPriorPromotionEvidenceAuditV12,
    LegacyNeuralPriorPromotionEvidenceAuditV13,
    LegacyNeuralPriorPromotionEvidenceAuditV14,
    LegacyNeuralPriorPromotionEvidenceAuditV15,
    LegacyNeuralPriorPromotionEvidenceAuditV16,
    LegacyNeuralPriorPromotionEvidenceAuditV17,
    LegacyNeuralPriorPromotionEvidenceAuditV18,
    LegacyNeuralPriorPromotionEvidenceAuditV19,
    LegacyNeuralPriorPromotionEvidenceAuditV20,
    LegacyNeuralPriorPromotionEvidenceAuditV21,
    LegacyNeuralPriorPromotionEvidenceAuditV22,
    LegacyNeuralPriorPromotionEvidenceAuditV23,
    LegacyNeuralPriorPromotionEvidenceAuditV24,
    LegacyNeuralPriorPromotionEvidenceAuditV25,
    LegacyNeuralPriorPromotionEvidenceAuditV26,
    LegacyNeuralPriorPromotionEvidenceAuditV27,
    LegacyNeuralPriorPromotionEvidenceAuditV28,
    LegacyNeuralPriorPromotionEvidenceAuditV29,
    LegacyNeuralPriorPromotionEvidenceAuditV30,
    LegacyHoldoutScoringArtifactAuditV10,
    LegacyHoldoutScoringArtifactAuditV11,
    LegacyHoldoutScoringArtifactAuditV12,
    LegacyHoldoutScoringArtifactAuditV13,
    NeuralPriorPromotionPolicy,
    PriorHoldoutEvaluation,
    ScoringReplayCaseArtifact,
    SEMANTIC_SCORING_REPLAY_CONTRACT,
    SEMANTIC_SCORING_REPLAY_METHOD,
    SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST,
    recompute_prior_holdout_evaluation_from_bundle,
    compute_neural_prior_promotion,
    validate_neural_prior_promotion,
    validate_neural_prior_candidate_manifest,
    validate_neural_prior_holdout_plan,
    validate_physical_event_catalog_result,
    validate_trusted_process_start_receipt,
    validate_trusted_process_completion_receipt,
    validate_process_log_artifact,
    validate_resolved_source_coverage_artifact,
    _new_prior_holdout_evaluation,
    _new_regime_reference_evidence,
    _new_physical_event_catalog_evidence,
    _new_physical_event_catalog_result,
    _new_trusted_process_start_receipt,
    _new_trusted_process_completion_receipt,
    _new_holdout_scoring_artifact,
    _training_target_source_trust_store_from_json,
    validate_holdout_scoring_artifact,
    validate_promotion_decision_rule,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_EXECUTOR_TRUST_STORE_CONTRACT = "advar-executor-trust-store-v2"
_OPERATOR_TRUST_STORE_CONTRACT = "advar-operator-trust-store-v1"
_SCHEDULER_TRUST_STORE_CONTRACT = "advar-trusted-scheduler-store-v1"
_EPISODE_FILES = {"manifest.json", "sensitivity_arrays.npz"}
_INDEX_SCHEMA_VERSION = 42
_EPISODE_SCHEMA_VERSION = 18
_MODEL_CONTRACT_SCHEMA_VERSION = 11
_RAW_TRUST_ACTIVATION_KINDS = frozenset(
    {
        "scoring_replay_bundle",
        "scoring_completion",
        "promotion_evidence",
        "promotion_deployment_certificate",
    }
)
_EPISODE_LEDGER_INITIALIZATION_TOKEN = object()
_NATIVE_PATH_TYPE = type(Path())
_PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST = (
    PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
)
_MAXIMUM_ACTION_ARTIFACT_MEMBERS = 12
_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES = 2 * 1024**3
_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES = 8 * 1024**3
_MAXIMUM_SCORING_REPLAY_SHARDS = 4096
_MAXIMUM_SCORING_REPLAY_TOTAL_EXPANDED_BYTES = 64 * 1024**3
_MAXIMUM_ACTION_GENERATOR_BYTES = 512 * 1024**2
_MAXIMUM_RAW_INGESTOR_TRUST_STORE_BYTES = 1024 * 1024
_MAXIMUM_TRAINING_TARGET_SOURCE_TRUST_STORE_BYTES = 1024 * 1024
_MAXIMUM_ANALYSIS_PROVENANCE_FILE_BYTES = 2 * 1024**3
_MAXIMUM_ANALYSIS_PROVENANCE_EXPANDED_BYTES = 8 * 1024**3


def _semantic_replay_execution_device(
    cases: tuple[ScoringReplayCaseArtifact, ...],
    case_tensors: Mapping[str, Mapping[str, Tensor]],
) -> torch.device:
    """Require one exact numerical runtime across every replay component."""

    devices = {
        str(tensor.device)
        for tensors in case_tensors.values()
        for tensor in tensors.values()
    }
    if len(devices) != 1:
        raise ValueError("semantic scoring used multiple tensor devices")
    device = torch.device(next(iter(devices)))
    active_runtime = numerical_runtime_manifest(device)
    for case in cases:
        runtime_digests = (
            case.candidate_prior_runner.numerical_runtime_digest,
            case.parent_prior_runner.numerical_runtime_digest,
            case.candidate_prior_application.inference_evidence.numerical_runtime_digest,
            case.parent_prior_application.inference_evidence.numerical_runtime_digest,
            case.regime_classifier.numerical_runtime_digest,
            case.candidate_forecast.run.prior_numerical_runtime_digest,
            case.parent_forecast.run.prior_numerical_runtime_digest,
        )
        if any(value != active_runtime.exact_digest for value in runtime_digests):
            raise ValueError("semantic scoring runtime lineage is inconsistent")
    return device


def _validate_scoring_backend_certification(
    execution_device: torch.device,
    policy: MPSBackendCertificationPolicy | None,
    evidence: MPSBackendCertificationEvidence | None,
) -> tuple[str | None, str | None]:
    """Keep automatic promotion scoring on the certified CPU path.

    The generic MPS certificate covers the numerical backend, but it does not
    yet cover the exported prior/classifier graphs and the complete holdout
    metric engine.  Accepting it here would therefore overstate the scope of
    the certificate.  The policy/evidence parameters remain in the API so a
    future model-scoring certificate can be introduced without silently
    changing the archive format.
    """

    if execution_device.type != "cpu":
        raise ValueError(
            "automatic promotion scoring requires the certified CPU backend"
        )
    if policy is not None or evidence is not None:
        raise ValueError("CPU scoring cannot claim MPS certification")
    return None, None


@dataclass(frozen=True)
class _ExecutorTrustStore:
    keys: dict[str, Ed25519PublicKey]
    content_digest: str


@dataclass(frozen=True)
class _OperatorTrustStore:
    keys: dict[str, Ed25519PublicKey]
    roles: dict[str, frozenset[str]]
    content_digest: str


@dataclass(frozen=True)
class _SchedulerTrustStore:
    keys: dict[str, Ed25519PublicKey]
    content_digest: str


@dataclass(frozen=True)
class LegacyPromotionEvaluationAudit:
    """Read-only payload retained before typed Tensor audit evidence existed."""

    evaluation_digest: str
    payload_json: str
    contract: str = "legacy-promotion-evaluation-audit-v1"
    content_digest_verified: bool = field(init=False)
    statistical_reuse_permitted: bool = field(init=False, default=False)
    audit_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.evaluation_digest) is None:
            raise ValueError("invalid legacy promotion evaluation digest")
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("invalid legacy promotion evaluation payload")
        if payload.get("evaluation_digest") != self.evaluation_digest:
            raise ValueError("legacy promotion evaluation digest mismatch")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if canonical != self.payload_json:
            raise ValueError("legacy promotion evaluation payload is not canonical")
        normalized = dict(payload)
        normalized.pop("evaluation_digest")
        tensor_payloads = [
            value
            for value in normalized.values()
            if isinstance(value, dict) and value.get("kind") == "tensor"
        ]
        verified = bool(tensor_payloads)
        if verified:
            for name, value in tuple(normalized.items()):
                if not isinstance(value, dict) or value.get("kind") != "tensor":
                    continue
                tensor = _decode_audit_tensor(name, value)
                normalized[name] = tensor_digest(tensor)
            if _json_digest(normalized) != self.evaluation_digest:
                raise ValueError("legacy promotion evaluation digest mismatch")
        object.__setattr__(self, "content_digest_verified", verified)
        object.__setattr__(self, "statistical_reuse_permitted", False)
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": self.contract,
                    "evaluation_digest": self.evaluation_digest,
                    "payload": payload,
                    "content_digest_verified": verified,
                    "statistical_reuse_permitted": False,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyProspectiveInterventionDecisionAudit:
    """Pre-operator-signature decision retained without execution reuse."""

    decision_digest: str
    payload_json: str
    audit_digest: str = ""

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict) or (
            payload.get("contract") not in {
                "prospective-intervention-decision-v2",
                "prospective-intervention-decision-v3",
                "prospective-intervention-decision-v4",
                "prospective-intervention-decision-v5",
            }
            or payload.get("decision_digest") != self.decision_digest
        ):
            raise ValueError("invalid legacy prospective decision")
        retained = dict(payload)
        retained.pop("decision_digest")
        if _json_digest(retained) != self.decision_digest:
            raise ValueError("legacy prospective decision digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": "legacy-prospective-decision-audit-v1",
                    "payload": payload,
                }
            ),
        )


@dataclass(frozen=True)
class LegacyRealizedInterventionReceiptAudit:
    """Earlier signed receipt retained without current approval semantics."""

    receipt_digest: str
    decision_digest: str
    payload_json: str
    audit_digest: str = ""

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        if not isinstance(payload, dict) or (
            payload.get("contract") not in {
                "realized-intervention-receipt-v2",
                "realized-intervention-receipt-v3",
                "realized-intervention-receipt-v4",
                "realized-intervention-receipt-v5",
            }
            or payload.get("receipt_digest") != self.receipt_digest
            or payload.get("decision_digest") != self.decision_digest
        ):
            raise ValueError("invalid legacy realized receipt")
        retained = dict(payload)
        retained.pop("receipt_digest")
        if _json_digest(retained) != self.receipt_digest:
            raise ValueError("legacy realized receipt digest mismatch")
        object.__setattr__(
            self,
            "audit_digest",
            _json_digest(
                {
                    "contract": "legacy-realized-receipt-audit-v1",
                    "payload": payload,
                }
            ),
        )


def _load_executor_trust_store(path: str | Path) -> _ExecutorTrustStore:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError("executor trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("executor trust store must be root-owned and non-writable")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "public_keys",
    }:
        raise ValueError("invalid executor trust store")
    if document["contract"] != _EXECUTOR_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported executor trust store")
    raw = document["public_keys"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("executor trust store requires keys")
    keys: dict[str, Ed25519PublicKey] = {}
    for key_id, public_hex in raw.items():
        if not isinstance(key_id, str) or not key_id or not isinstance(public_hex, str):
            raise ValueError("invalid executor trust-store key")
        try:
            public = bytes.fromhex(public_hex)
            key = Ed25519PublicKey.from_public_bytes(public)
        except ValueError as error:
            raise ValueError("invalid executor trust-store public key") from error
        if len(public) != 32:
            raise ValueError("executor public keys must contain 32 bytes")
        keys[key_id] = key
    return _ExecutorTrustStore(keys=keys, content_digest=_json_digest(document))


def _load_operator_trust_store(path: str | Path) -> _OperatorTrustStore:
    """Load a root-owned public-key store used only for action review."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError("operator trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("operator trust store must be root-owned and non-writable")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {"contract", "operators"}:
        raise ValueError("invalid operator trust store")
    if document["contract"] != _OPERATOR_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported operator trust store")
    raw = document["operators"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("operator trust store requires operators")
    keys: dict[str, Ed25519PublicKey] = {}
    roles: dict[str, frozenset[str]] = {}
    for key_id, record in raw.items():
        if (
            not isinstance(key_id, str)
            or not key_id
            or not isinstance(record, dict)
            or set(record) != {"public_key", "roles"}
            or not isinstance(record["public_key"], str)
            or not isinstance(record["roles"], list)
            or not record["roles"]
            or any(
                not isinstance(role, str) or not role or role.strip() != role
                for role in record["roles"]
            )
        ):
            raise ValueError("invalid operator trust-store entry")
        try:
            public = bytes.fromhex(record["public_key"])
            key = Ed25519PublicKey.from_public_bytes(public)
        except ValueError as error:
            raise ValueError("invalid operator trust-store public key") from error
        if len(public) != 32:
            raise ValueError("operator public keys must contain 32 bytes")
        keys[key_id] = key
        roles[key_id] = frozenset(record["roles"])
    return _OperatorTrustStore(
        keys=keys,
        roles=roles,
        content_digest=_json_digest(document),
    )


def _load_scheduler_trust_store(path: str | Path) -> _SchedulerTrustStore:
    """Load the root-owned authority that may launch promotion jobs."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError("scheduler trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError("scheduler trust store must be root-owned and non-writable")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "schedulers",
    }:
        raise ValueError("invalid scheduler trust store")
    if document["contract"] != _SCHEDULER_TRUST_STORE_CONTRACT:
        raise ValueError("unsupported scheduler trust store")
    raw = document["schedulers"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("scheduler trust store requires keys")
    keys: dict[str, Ed25519PublicKey] = {}
    for key_id, public_hex in raw.items():
        if (
            not isinstance(key_id, str)
            or not key_id
            or key_id.strip() != key_id
            or not isinstance(public_hex, str)
        ):
            raise ValueError("invalid scheduler trust-store key")
        try:
            public = bytes.fromhex(public_hex)
            key = Ed25519PublicKey.from_public_bytes(public)
        except ValueError as error:
            raise ValueError("invalid scheduler trust-store public key") from error
        if len(public) != 32:
            raise ValueError("scheduler public keys must contain 32 bytes")
        keys[key_id] = key
    return _SchedulerTrustStore(keys=keys, content_digest=_json_digest(document))


def _load_raw_ingestor_trust_store(path: str | Path) -> RawIngestorTrustStore:
    """Load the current root-owned raw-ingestor revocation view."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError("raw-ingestor trust store path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
            or metadata.st_size > _MAXIMUM_RAW_INGESTOR_TRUST_STORE_BYTES
        ):
            raise ValueError(
                "raw-ingestor trust store must be root-owned, immutable, and bounded"
            )
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or set(document) != {
        "contract",
        "authorities",
    }:
        raise ValueError("invalid raw-ingestor trust store")
    authorities = document["authorities"]
    if not isinstance(authorities, list):
        raise ValueError("invalid raw-ingestor trust store")
    try:
        return RawIngestorTrustStore(
            authorities=tuple(tuple(item) for item in authorities),
            contract=cast(str, document["contract"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid raw-ingestor trust store") from error


def _load_training_target_source_trust_store(
    path: str | Path,
) -> TrainingTargetSourceTrustStore:
    """Load the sole root-owned current target-source revocation view."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError("training target-source trust store path must be absolute")
    for parent in (source.parent, *source.parents):
        metadata = parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
        ):
            raise ValueError(
                "training target-source trust ancestry must be root-owned"
            )
        if parent == parent.parent:
            break
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
            or metadata.st_size
            > _MAXIMUM_TRAINING_TARGET_SOURCE_TRUST_STORE_BYTES
        ):
            raise ValueError(
                "training target-source trust store must be root-owned, immutable, and bounded"
            )
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            document = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict):
        raise ValueError("invalid training target-source trust store")
    retained_digest = document.get("content_digest")
    if not isinstance(retained_digest, str):
        raise ValueError("invalid training target-source trust store")
    return _training_target_source_trust_store_from_json(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        expected_digest=retained_digest,
    )


def _validate_current_raw_ingestor_receipt(
    receipt: RawObservationResolutionReceipt,
    slot: RawObservationSlotPlan,
    *,
    pinned_trust_store: RawIngestorTrustStore,
    current_trust_store: RawIngestorTrustStore,
) -> None:
    """Require the plan-pinned key and the current revocation view."""

    pinned_keys = {
        (authority_id, public_key_hex)
        for authority_id, public_key_hex, *_ in pinned_trust_store.authorities
    }
    current_keys = {
        (authority_id, public_key_hex)
        for authority_id, public_key_hex, *_ in current_trust_store.authorities
    }
    if pinned_keys != current_keys:
        raise ValueError(
            "current raw-ingestor trust store changed pinned identities"
        )
    receipt.validate_against(slot, pinned_trust_store)
    receipt.validate_against(slot, current_trust_store)


def _raw_resolution_encoded_arrays(
    receipts: tuple[RawObservationResolutionReceipt, ...],
    *,
    derived_frames: Tensor,
) -> dict[str, Tensor]:
    """Encode only present raw volumes; missing receipts live in metadata."""

    retained = tuple(
        item for item in receipts if type(item) is ResolvedRawObservationReceipt
    )
    height, width = derived_frames.shape[-2:]

    def stack_or_empty(attribute: str, dtype: torch.dtype) -> Tensor:
        if not retained:
            return torch.empty(
                (0, height, width),
                dtype=dtype,
                device=derived_frames.device,
            )
        return torch.stack(
            tuple(getattr(item.raw_grid_volume, attribute) for item in retained)
        )

    return {
        "raw_source_reflectivity_bits": stack_or_empty(
            "_raw_reflectivity_bits", torch.int32
        ),
        "raw_source_qc_flags": stack_or_empty("_raw_qc_flags", torch.bool),
        "raw_source_quality_bits": stack_or_empty(
            "_raw_quality_bits", torch.int32
        ),
        "raw_source_observation_std_bits": stack_or_empty(
            "_raw_observation_std_bits", torch.int32
        ),
    }


def _validate_current_scoring_raw_ingestor_receipts(
    cases: tuple[ScoringReplayCaseArtifact, ...],
    *,
    raw_ingestor_trust_store_path: str | Path,
) -> RawIngestorTrustStore:
    """Recheck every replay receipt against the current root revocation view."""

    if not cases or any(
        type(item) is not ScoringReplayCaseArtifact for item in cases
    ):
        raise TypeError("raw-ingestor revalidation requires typed replay cases")
    current = _load_raw_ingestor_trust_store(
        raw_ingestor_trust_store_path
    )
    for case in cases:
        planned_case = case.plan.case(case.case_id)
        sampling_unit = next(
            item
            for item in case.plan.meteorological_sampling_units
            if item.sampling_unit_digest
            == planned_case.meteorological_sampling_unit_digest
        )
        slots = {
            item.slot_digest: item
            for item in case.plan.raw_observation_slot_plans
            if item.slot_digest in sampling_unit.raw_observation_slot_digests
        }
        if set(slots) != {
            item.slot_plan_digest for item in case.resolved_raw_observations
        }:
            raise ValueError("scoring replay raw-ingestor lineage is incomplete")
        for receipt in case.resolved_raw_observations:
            _validate_current_raw_ingestor_receipt(
                receipt,
                slots[receipt.slot_plan_digest],
                pinned_trust_store=case.plan.raw_ingestor_trust_store,
                current_trust_store=current,
            )
    return current


def _require_current_raw_ingestor_trust_store_digest(
    raw_ingestor_trust_store_path: str | Path,
    expected_digest: str,
) -> RawIngestorTrustStore:
    """Reload the root-owned revocation view and require one exact snapshot."""

    current = _load_raw_ingestor_trust_store(raw_ingestor_trust_store_path)
    if current.content_digest != expected_digest:
        raise ValueError("raw-ingestor trust store changed during issuance")
    return current


def _validate_scheduler_authority(
    plan: PhysicalEventCatalogPlan,
    trust: _SchedulerTrustStore,
) -> None:
    key = trust.keys.get(plan.scheduler_id)
    if (
        key is None
        or trust.content_digest != plan.scheduler_trust_store_digest
        or key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        != plan.scheduler_public_key_hex
    ):
        raise ValueError("event-catalog scheduler is not root-approved")


def _torch_dtype(value: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError("unsupported durable action dtype") from error


def _intervention_context_tensor(
    frames: Tensor,
    masks: Tensor,
    quality: Tensor,
    std: Tensor,
    background: Tensor | None,
    applicability: Tensor,
    *,
    minimum_dbz: float,
    maximum_dbz: float,
    missing_fill_dbz: float,
) -> Tensor:
    finite_mask = torch.isfinite(frames)
    canonical_frames = canonicalize_action_frames(
        frames,
        masks,
        minimum_dbz=minimum_dbz,
        maximum_dbz=maximum_dbz,
        missing_fill_dbz=missing_fill_dbz,
    )
    finite = finite_mask.to(canonical_frames)
    canonical_background = (
        torch.zeros_like(canonical_frames)
        if background is None
        else canonicalize_action_frames(
            background,
            torch.ones_like(background, dtype=torch.bool),
            minimum_dbz=minimum_dbz,
            maximum_dbz=maximum_dbz,
            missing_fill_dbz=missing_fill_dbz,
        )
    )
    background_present = torch.full_like(
        canonical_frames,
        float(background is not None),
    )
    return torch.stack(
        (
            canonical_frames,
            masks.to(canonical_frames),
            finite,
            quality,
            std,
            canonical_background,
            background_present,
            applicability.to(canonical_frames),
        )
    )


def _artifact_intervention_context(
    manifest: dict[str, object],
    tensors: dict[str, Tensor],
    background: Tensor | None,
    *,
    prefix: str = "before",
) -> InterventionInputContext:
    """Rebuild the immutable action context from its durable tensor members."""

    result = object.__new__(InterventionInputContext)
    values: tuple[tuple[str, object], ...] = (
        ("_frames_dbz", tensors[f"{prefix}_frames"]),
        ("_observation_masks", tensors[f"{prefix}_masks"]),
        ("_quality_weight", tensors[f"{prefix}_quality"]),
        ("_observation_std_dbz", tensors[f"{prefix}_std"]),
        ("_background_frames_dbz", background),
        ("_applicability_mask", tensors[f"{prefix}_applicability"]),
        ("radar_id", manifest[f"{prefix}_radar_id"]),
        ("input_bundle_digest", manifest[f"{prefix}_input_bundle_digest"]),
        ("input_plan_digest", manifest["input_plan_digest"]),
        (
            "input_plan_resolution_digest",
            manifest[f"{prefix}_input_plan_resolution_digest"],
        ),
        (
            "analysis_input_identity_digest",
            manifest.get(f"{prefix}_analysis_input_identity_digest"),
        ),
        ("context_schema_digest", manifest[f"{prefix}_context_schema_digest"]),
        (
            "applicability_region_digest",
            manifest[f"{prefix}_applicability_region_digest"],
        ),
        (
            "applicability_mask_digest",
            manifest[f"{prefix}_applicability_mask_digest"],
        ),
        (
            "canonicalization_contract_digest",
            manifest[f"{prefix}_canonicalization_contract_digest"],
        ),
        ("min_dbz", manifest["minimum_dbz"]),
        ("max_dbz", manifest["maximum_dbz"]),
        ("missing_fill_dbz", manifest["missing_fill_dbz"]),
        ("context_digest", manifest[f"{prefix}_context_digest"]),
    )
    for name, value in values:
        object.__setattr__(result, name, value)
    return result
_TRUST_COMPONENTS_V13 = {
    "linearity",
    "verification",
    "metric_support",
    "pair_consistency",
    "path_evidence",
    "observation_evidence",
}
_TRUST_COMPONENTS_V16 = {
    "linearity",
    "verification",
    "metric_support",
    "pair_consistency",
    "observation_verified_evidence",
}
_MODEL_CONTRACT_SCHEMA_BY_EPISODE_SCHEMA = {
    3: 3,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
}
_LEGACY_MODEL_CONTRACT_FIELDS_V1_V2 = {
    "model_commit",
    "residual_contract_version",
    "forecast_metric_version",
    "observation_contract_version",
    "forecast_integrator_version",
    "grid_geometry_version",
    "radar_qc_version",
}
_SCHEMA_ONE_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_echo_mass",
)
_SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
)
_SCHEMA_FIVE_CONTEXT_FEATURE_NAMES = (
    "motion_dy",
    "motion_dx",
    "motion_speed",
    "log_growth",
    "motion_disagreement",
    "growth_disagreement",
    "motion_pair_conflict",
    "growth_pair_conflict",
    "tendency_pair_count",
    "tendency_source_observation",
    "tendency_source_background",
    "current_state_support_fraction",
    "background_contribution_fraction",
    "latest_observation_coverage",
    "latest_mean_dbz",
    "latest_max_dbz",
    "latest_q90_dbz",
    "echo_fraction_5dbz",
    "echo_fraction_35dbz",
    "boundary_echo_fraction",
    "centroid_y",
    "centroid_x",
    "log_integrated_echo",
)
_SCHEMA_SIX_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_FIVE_CONTEXT_FEATURE_NAMES,
    "motion_pair_selection_none",
    "motion_pair_selection_single",
    "motion_pair_selection_long",
    "motion_pair_selection_blended",
    "motion_pair_selection_earlier",
    "motion_pair_selection_recent",
    "motion_pair_selection_persistence",
    "growth_pair_selection_none",
    "growth_pair_selection_single",
    "growth_pair_selection_long",
    "growth_pair_selection_blended",
    "growth_pair_selection_earlier",
    "growth_pair_selection_recent",
    "growth_pair_selection_persistence",
)
_SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_SIX_CONTEXT_FEATURE_NAMES,
    "phase_correlation_psr_available",
    "log1p_minimum_phase_correlation_psr",
)
_SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES,
    "projected_velocity_available",
    "projected_velocity_x_mps",
    "projected_velocity_y_mps",
    "projected_speed_mps",
)
_SCHEMA_NINE_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES,
    "motion_disagreement_mps_available",
    "motion_disagreement_mps",
)
_SCHEMA_TEN_CONTEXT_FEATURE_NAMES = (
    *_SCHEMA_NINE_CONTEXT_FEATURE_NAMES,
    "area_weighted_echo_available",
    "log1p_linear_reflectivity_integral_km2",
)


@dataclass(frozen=True)
class ModelContract:
    """Versions that determine whether stored gradients are compatible."""

    model_commit: str
    residual_contract_version: str
    forecast_metric_version: str
    observation_contract_version: str
    forecast_integrator_version: str
    grid_geometry_version: str
    radar_qc_version: str
    nowcast_config_digest: str
    sensitivity_config_digest: str
    grid_time_contract_digest: str | None = None

    def __post_init__(self) -> None:
        required = asdict(self)
        grid_digest = required.pop("grid_time_contract_digest")
        if not all(required.values()):
            raise ValueError("all model contract fields must be non-empty")
        if grid_digest is not None and re.fullmatch(
            r"[0-9a-f]{64}", grid_digest
        ) is None:
            raise ValueError(
                "grid_time_contract_digest must be a SHA-256 digest"
            )

    @property
    def digest(self) -> str:
        return _model_contract_digest(
            self,
            schema_version=_MODEL_CONTRACT_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class SensitivityEpisode:
    """One finalized forecast/verification/sensitivity experience."""

    episode_id: str
    issue_time: str
    radar_id: str
    contract: ModelContract
    snapshot: SensitivitySnapshot
    action_features: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _validate_episode_id(self.episode_id)
        issue_time = datetime.fromisoformat(self.issue_time)
        if issue_time.tzinfo is None:
            raise ValueError("issue_time must include a timezone")
        if not self.radar_id:
            raise ValueError("radar_id must be non-empty")
        if not all(math.isfinite(value) for value in self.action_features):
            raise ValueError("action_features must be finite")
        forecast_run_digest = self.snapshot.forecast_run_digest
        if (
            not isinstance(forecast_run_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", forecast_run_digest) is None
        ):
            raise ValueError("forecast_run_digest must be a SHA-256 digest")
        if (
            self.contract.nowcast_config_digest
            != self.snapshot.nowcast_config_digest
            or self.contract.sensitivity_config_digest
            != self.snapshot.sensitivity_config_digest
        ):
            raise ValueError(
                "model contract config digests must match the sensitivity snapshot"
            )
        if (
            self.contract.grid_time_contract_digest
            != self.snapshot.grid_time_contract_digest
        ):
            raise ValueError(
                "model contract grid digest must match the sensitivity snapshot"
            )
        if self.snapshot.verification_lineage_complete:
            if (
                self.snapshot.verification_grid_contract_digest
                != self.snapshot.grid_time_contract_digest
            ):
                raise ValueError(
                    "verification grid digest must match the forecast grid"
                )
            expected_times = _expected_verification_times(
                self.issue_time,
                self.snapshot.lead_minutes,
            )
            if self.snapshot.verification_valid_times != expected_times:
                raise ValueError(
                    "verification valid times must match issue time and leads"
                )

    @property
    def forecast_run_digest(self) -> str:
        return self.snapshot.forecast_run_digest


@dataclass(frozen=True)
class LoadedEpisode:
    """Verified data loaded without Python object deserialization."""

    manifest: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6 = frozenset(
    {
        "input_radar_frames",
        "input_qc_valid_mask",
        "input_quality_weight",
        "candidate_forecast_dbz",
        "candidate_publication_mask",
        "candidate_background_fallback_mask",
        "candidate_forecast_confidence",
        "candidate_state_echo_linear",
        "candidate_state_displacement_yx",
        "candidate_state_log_growth",
        "parent_forecast_dbz",
        "parent_publication_mask",
        "parent_background_fallback_mask",
        "parent_forecast_confidence",
        "parent_state_echo_linear",
        "parent_state_displacement_yx",
        "parent_state_log_growth",
        "verification_frames_dbz",
        "verification_valid_mask",
        "candidate_state_background_dbz",
        "candidate_state_std_dbz",
        "candidate_state_valid_mask",
        "candidate_state_valid_probability",
        "candidate_state_support_probability",
        "candidate_event_probability",
        "candidate_truncated_location_dbz",
        "candidate_truncated_scale_dbz",
        "parent_state_background_dbz",
        "parent_state_std_dbz",
        "parent_state_valid_mask",
        "parent_state_valid_probability",
        "parent_state_support_probability",
        "parent_event_probability",
        "parent_truncated_location_dbz",
        "parent_truncated_scale_dbz",
        "uncertainty_target_dbz",
        "uncertainty_target_valid_mask",
        "uncertainty_target_echo_support",
        "state_target_dbz",
        "state_target_valid_mask",
        "state_target_echo_support",
        "classifier_regime_logits",
        "classifier_range_logits",
        "range_grid_x_m",
        "range_grid_y_m",
        "operational_issuance_mask",
    }
)
SCORING_REPLAY_RAW_PRODUCT_TENSOR_ROLES = frozenset(
    {
        "raw_source_reflectivity_bits",
        "raw_source_qc_flags",
        "raw_source_quality_bits",
        "raw_source_observation_std_bits",
    }
)
LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V8 = (
    LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
    | SCORING_REPLAY_RAW_PRODUCT_TENSOR_ROLES
)
LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13 = (
    LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V8
    | frozenset(
        {
            "input_observation_std_dbz",
            "input_source_available_mask",
        }
    )
)
SCORING_REPLAY_REQUIRED_TENSOR_ROLES = (
    LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13
    | frozenset(
        {
            "verification_quality_weight",
            "verification_observation_std_dbz",
            "verification_observation_state_code",
            "verification_source_radar_index_map",
            "verification_detection_limit_dbz",
            "verification_acquisition_time_offset_seconds",
            "verification_source_reflectivity_dbz",
            "verification_source_detection_limit_dbz",
            "verification_source_acquisition_time_offset_seconds",
            "verification_source_below_detection_reported",
            "verification_source_assignment_scores",
            "verification_source_availability_by_time",
            "verification_source_range_km",
            "verification_source_elevation_deg",
            "verification_source_beam_blockage_fraction",
            "verification_source_attenuation_qc_score",
            "verification_source_present_mask",
            "verification_range_elevation_valid_mask",
            "verification_beam_blocked_mask",
            "verification_attenuation_qc_valid_mask",
            "verification_below_detection_censored_mask",
        }
    )
)
SCORING_REPLAY_BACKGROUND_TENSOR_ROLES = frozenset(
    {"background_frames_dbz"}
)
LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V1 = frozenset(
    {
        "input_radar_frames",
        "input_qc_valid_mask",
        "input_quality_weight",
        "background_echo_linear",
        "source_radar_index_map",
        "outage_mask",
        "candidate_prior_mean",
        "candidate_prior_scale",
        "parent_prior_mean",
        "parent_prior_scale",
        "candidate_forecast_dbz",
        "candidate_publication_mask",
        "parent_forecast_dbz",
        "parent_publication_mask",
        "verification_frames_dbz",
        "verification_valid_mask",
        "operational_issuance_mask",
        "range_band_index",
        "event_object_labels",
    }
)
LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES = frozenset(
    {"source_radar_index_map", "outage_mask", "dynamic_qc_valid_mask"}
)
SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES = (
    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
    | frozenset(
        {
            "input_history_source_radar_index_map",
            "nominal_source_coverage_mask",
            "resolved_source_coverage_mask",
            "effective_horizontal_range_m",
        }
    )
)
_SCORING_REPLAY_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class ScoringReplayTensorRecord:
    case_id: str
    role: str
    archive_member: str
    dtype: str
    shape: tuple[int, ...]
    tensor_digest: str
    archive_sha256: str | None = None

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "case_id": self.case_id,
            "role": self.role,
            "archive_member": self.archive_member,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "tensor_digest": self.tensor_digest,
        }
        if self.archive_sha256 is not None:
            payload["archive_sha256"] = self.archive_sha256
        return payload


@dataclass(frozen=True)
class ScoringReplayBundleManifest:
    scoring_input_artifact_digest: str
    ordered_case_ids: tuple[str, ...]
    ordered_evaluation_digests: tuple[str, ...]
    semantic_case_digests: tuple[str, ...]
    dynamic_source_case_ids: tuple[str, ...]
    background_case_ids: tuple[str, ...]
    algorithm_source_manifest_digest: str
    runtime_compatibility_digest: str
    runtime_exact_digest: str
    scoring_backend_certification_policy_digest: str | None
    scoring_backend_certification_evidence_digest: str | None
    tensor_records: tuple[ScoringReplayTensorRecord, ...]
    tensor_archive_sha256: str
    evaluation_payload_sha256: str
    raw_provenance_payload_sha256: str
    verification_provenance_payload_sha256: str | None = None
    raw_ingestor_trust_store_digest: str | None = None
    tensor_shard_sha256s: tuple[str, ...] = ()
    replay_method: str = SEMANTIC_SCORING_REPLAY_METHOD
    contract: str = SEMANTIC_SCORING_REPLAY_CONTRACT
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != SEMANTIC_SCORING_REPLAY_CONTRACT
            or self.replay_method != SEMANTIC_SCORING_REPLAY_METHOD
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in self.dynamic_source_case_ids
            )
            or len(set(self.dynamic_source_case_ids))
            != len(self.dynamic_source_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in self.background_case_ids
            )
            or len(set(self.background_case_ids))
            != len(self.background_case_ids)
        ):
            raise ValueError("scoring replay bundle manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("scoring replay bundle digest is invalid")
        certification_digests = (
            self.scoring_backend_certification_policy_digest,
            self.scoring_backend_certification_evidence_digest,
        )
        if (certification_digests[0] is None) != (
            certification_digests[1] is None
        ):
            raise ValueError("scoring replay backend certification is incomplete")
        if any(value is not None for value in certification_digests):
            raise ValueError("current scoring replay is CPU-only")
        if (
            self.verification_provenance_payload_sha256 is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.verification_provenance_payload_sha256,
            )
            is None
            or self.raw_ingestor_trust_store_digest is None
            or re.fullmatch(
                r"[0-9a-f]{64}", self.raw_ingestor_trust_store_digest
            )
            is None
        ):
            raise ValueError(
                "current scoring replay requires raw-ingestor trust lineage"
            )
        if (
            not self.tensor_shard_sha256s
            or len(self.tensor_shard_sha256s) > _MAXIMUM_SCORING_REPLAY_SHARDS
            or tuple(sorted(set(self.tensor_shard_sha256s)))
            != self.tensor_shard_sha256s
            or any(
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in self.tensor_shard_sha256s
            )
            or self.tensor_archive_sha256
            != _json_digest(
                {
                    "contract": "neural-prior-scoring-replay-shard-set-v1",
                    "ordered_shard_sha256s": list(self.tensor_shard_sha256s),
                }
            )
        ):
            raise ValueError("current scoring replay tensor shard set is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                SCORING_REPLAY_REQUIRED_TENSOR_ROLES
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("scoring replay bundle tensor set is incomplete")
        for record in self.tensor_records:
            if (
                not _SAFE_ID.fullmatch(record.case_id)
                or record.role
                not in (
                    SCORING_REPLAY_REQUIRED_TENSOR_ROLES
                    | SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    | SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                )
                or not _SAFE_ID.fullmatch(record.archive_member)
                or (
                    not record.shape
                    and record.role not in _REPLAY_SCALAR_ROLES
                )
                or any(type(value) is not int or value <= 0 for value in record.shape)
                or re.fullmatch(r"[0-9a-f]{64}", record.tensor_digest) is None
                or record.archive_sha256 not in self.tensor_shard_sha256s
                or record.archive_member != "tensor"
                or record.dtype not in _SCORING_REPLAY_DTYPE_BYTES
            ):
                raise ValueError("scoring replay tensor record is invalid")
        shard_layouts: dict[str, tuple[str, tuple[int, ...], str]] = {}
        total_expanded_bytes = 0
        for record in self.tensor_records:
            assert record.archive_sha256 is not None
            layout = (record.dtype, record.shape, record.tensor_digest)
            retained_layout = shard_layouts.setdefault(
                record.archive_sha256,
                layout,
            )
            if retained_layout != layout:
                raise ValueError("scoring replay tensor shard layout equivocated")
            if retained_layout is layout:
                total_expanded_bytes += (
                    math.prod(record.shape or (1,))
                    * _SCORING_REPLAY_DTYPE_BYTES[record.dtype]
                )
        if total_expanded_bytes > _MAXIMUM_SCORING_REPLAY_TOTAL_EXPANDED_BYTES:
            raise ValueError("scoring replay tensor shard set exceeds budget")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract": self.contract,
            "scoring_input_artifact_digest": self.scoring_input_artifact_digest,
            "ordered_case_ids": list(self.ordered_case_ids),
            "ordered_evaluation_digests": list(
                self.ordered_evaluation_digests
            ),
            "semantic_case_digests": list(self.semantic_case_digests),
            "dynamic_source_case_ids": list(self.dynamic_source_case_ids),
            "background_case_ids": list(self.background_case_ids),
            "algorithm_source_manifest_digest": (
                self.algorithm_source_manifest_digest
            ),
            "runtime_compatibility_digest": self.runtime_compatibility_digest,
            "runtime_exact_digest": self.runtime_exact_digest,
            "scoring_backend_certification_policy_digest": (
                self.scoring_backend_certification_policy_digest
            ),
            "scoring_backend_certification_evidence_digest": (
                self.scoring_backend_certification_evidence_digest
            ),
            "tensor_records": [item.payload for item in self.tensor_records],
            "tensor_archive_sha256": self.tensor_archive_sha256,
            "evaluation_payload_sha256": self.evaluation_payload_sha256,
            "raw_provenance_payload_sha256": (
                self.raw_provenance_payload_sha256
            ),
            "replay_method": self.replay_method,
        }
        if self.verification_provenance_payload_sha256 is not None:
            payload["verification_provenance_payload_sha256"] = (
                self.verification_provenance_payload_sha256
            )
        if self.raw_ingestor_trust_store_digest is not None:
            payload["raw_ingestor_trust_store_digest"] = (
                self.raw_ingestor_trust_store_digest
            )
        if self.tensor_shard_sha256s:
            payload["tensor_shard_sha256s"] = list(
                self.tensor_shard_sha256s
            )
        return payload


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV1:
    """Byte-verified PR #110 snapshot retained for audit, never promotion."""

    scoring_input_artifact_digest: str
    ordered_case_ids: tuple[str, ...]
    ordered_evaluation_digests: tuple[str, ...]
    algorithm_source_manifest_digest: str
    runtime_compatibility_digest: str
    runtime_exact_digest: str
    tensor_records: tuple[ScoringReplayTensorRecord, ...]
    tensor_archive_sha256: str
    evaluation_payload_sha256: str
    replay_method: str = "typed-evaluation-reconstruction-v1"
    contract: str = "neural-prior-scoring-replay-bundle-v1"
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v1"
            or self.replay_method != "typed-evaluation-reconstruction-v1"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
        ):
            raise ValueError("legacy scoring replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy scoring replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V1
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy scoring replay tensor set is incomplete")
        for record in self.tensor_records:
            if (
                not _SAFE_ID.fullmatch(record.case_id)
                or record.role
                not in LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V1
                or not _SAFE_ID.fullmatch(record.archive_member)
                or not record.shape
                or any(
                    type(value) is not int or value <= 0
                    for value in record.shape
                )
                or re.fullmatch(r"[0-9a-f]{64}", record.tensor_digest) is None
            ):
                raise ValueError("legacy scoring replay tensor record is invalid")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "scoring_input_artifact_digest": self.scoring_input_artifact_digest,
            "ordered_case_ids": list(self.ordered_case_ids),
            "ordered_evaluation_digests": list(
                self.ordered_evaluation_digests
            ),
            "algorithm_source_manifest_digest": (
                self.algorithm_source_manifest_digest
            ),
            "runtime_compatibility_digest": self.runtime_compatibility_digest,
            "runtime_exact_digest": self.runtime_exact_digest,
            "tensor_records": [item.payload for item in self.tensor_records],
            "tensor_archive_sha256": self.tensor_archive_sha256,
            "evaluation_payload_sha256": self.evaluation_payload_sha256,
            "replay_method": self.replay_method,
        }


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV2:
    """PR #111 semantic bundle retained for byte audit, never promotion."""

    scoring_input_artifact_digest: str
    ordered_case_ids: tuple[str, ...]
    ordered_evaluation_digests: tuple[str, ...]
    semantic_case_digests: tuple[str, ...]
    dynamic_source_case_ids: tuple[str, ...]
    background_case_ids: tuple[str, ...]
    algorithm_source_manifest_digest: str
    runtime_compatibility_digest: str
    runtime_exact_digest: str
    tensor_records: tuple[ScoringReplayTensorRecord, ...]
    tensor_archive_sha256: str
    evaluation_payload_sha256: str
    replay_method: str = "builtin-semantic-scoring-recomputation-v2"
    contract: str = "neural-prior-scoring-replay-bundle-v2"
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v2"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v2"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or len(set(self.dynamic_source_case_ids))
            != len(self.dynamic_source_case_ids)
            or len(set(self.background_case_ids))
            != len(self.background_case_ids)
        ):
            raise ValueError("legacy semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
                | (
                    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy semantic replay tensor set is incomplete")
        allowed_roles = (
            LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
            | LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
            | SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
        )
        for record in self.tensor_records:
            if (
                not _SAFE_ID.fullmatch(record.case_id)
                or record.role not in allowed_roles
                or not _SAFE_ID.fullmatch(record.archive_member)
                or (
                    not record.shape
                    and record.role not in _REPLAY_SCALAR_ROLES
                )
                or any(
                    type(value) is not int or value <= 0
                    for value in record.shape
                )
                or re.fullmatch(r"[0-9a-f]{64}", record.tensor_digest) is None
            ):
                raise ValueError("legacy semantic replay tensor record is invalid")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "scoring_input_artifact_digest": self.scoring_input_artifact_digest,
            "ordered_case_ids": list(self.ordered_case_ids),
            "ordered_evaluation_digests": list(
                self.ordered_evaluation_digests
            ),
            "semantic_case_digests": list(self.semantic_case_digests),
            "dynamic_source_case_ids": list(self.dynamic_source_case_ids),
            "background_case_ids": list(self.background_case_ids),
            "algorithm_source_manifest_digest": (
                self.algorithm_source_manifest_digest
            ),
            "runtime_compatibility_digest": self.runtime_compatibility_digest,
            "runtime_exact_digest": self.runtime_exact_digest,
            "tensor_records": [item.payload for item in self.tensor_records],
            "tensor_archive_sha256": self.tensor_archive_sha256,
            "evaluation_payload_sha256": self.evaluation_payload_sha256,
            "replay_method": self.replay_method,
        }


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV3(
    LegacyScoringReplayBundleManifestAuditV2
):
    """PR #114 semantic replay retained for audit, never promotion."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v3"
    contract: str = "neural-prior-scoring-replay-bundle-v3"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v3"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v3"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
        ):
            raise ValueError("legacy v3 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v3 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
                | (
                    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v3 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV4(
    LegacyScoringReplayBundleManifestAuditV3
):
    """Certified-MPS-capable v4 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v4"
    contract: str = "neural-prior-scoring-replay-bundle-v4"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v4"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v4"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
        ):
            raise ValueError("legacy v4 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v4 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
                | (
                    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v4 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV5(
    LegacyScoringReplayBundleManifestAuditV4
):
    """CPU-only generation-v3 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v5"
    contract: str = "neural-prior-scoring-replay-bundle-v5"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v5"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v5"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
        ):
            raise ValueError("legacy v5 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v5 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
                | (
                    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v5 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV6(
    LegacyScoringReplayBundleManifestAuditV5
):
    """Pre-provenance generation-v4 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v6"
    contract: str = "neural-prior-scoring-replay-bundle-v6"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v6"
            or self.replay_method != "builtin-semantic-scoring-recomputation-v6"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids) != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
        ):
            raise ValueError("legacy v6 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v6 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V6
                | (
                    LEGACY_SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v6 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV7(ScoringReplayBundleManifest):
    """Pre-canonical-masked-input v7 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v7"
    contract: str = "neural-prior-scoring-replay-bundle-v7"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v7"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v7"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
        ):
            raise ValueError("legacy v7 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v7 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V8
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v7 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV8(ScoringReplayBundleManifest):
    """Pre-five-channel v8 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v8"
    contract: str = "neural-prior-scoring-replay-bundle-v8"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v8"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v8"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
        ):
            raise ValueError("legacy v8 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v8 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V8
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v8 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV9(ScoringReplayBundleManifest):
    """Pre-bounded-quality v9 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v9"
    contract: str = "neural-prior-scoring-replay-bundle-v9"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v9"
            or self.replay_method != "builtin-semantic-scoring-recomputation-v9"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids) != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
            or self.raw_ingestor_trust_store_digest is None
            or re.fullmatch(
                r"[0-9a-f]{64}", self.raw_ingestor_trust_store_digest
            )
            is None
        ):
            raise ValueError("legacy v9 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v9 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v9 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV11(ScoringReplayBundleManifest):
    """Pre-observation-error v11 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v11"
    contract: str = "neural-prior-scoring-replay-bundle-v11"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v11"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v11"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
            or self.raw_ingestor_trust_store_digest is None
        ):
            raise ValueError("legacy v11 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
            self.raw_ingestor_trust_store_digest,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v11 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v11 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV12(ScoringReplayBundleManifest):
    """Pre-source-specific v12 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v12"
    contract: str = "neural-prior-scoring-replay-bundle-v12"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v12"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v12"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
            or self.raw_ingestor_trust_store_digest is None
        ):
            raise ValueError("legacy v12 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
            self.raw_ingestor_trust_store_digest,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v12 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v12 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV13(ScoringReplayBundleManifest):
    """Pre-source-composition v13 replay retained for byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v13"
    contract: str = "neural-prior-scoring-replay-bundle-v13"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v13"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v13"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
            or self.raw_ingestor_trust_store_digest is None
        ):
            raise ValueError("legacy v13 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
            self.raw_ingestor_trust_store_digest,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v13 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                LEGACY_SCORING_REPLAY_REQUIRED_TENSOR_ROLES_V13
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if actual != expected or len(actual) != len(self.tensor_records):
            raise ValueError("legacy v13 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LegacyScoringReplayBundleManifestAuditV14(ScoringReplayBundleManifest):
    """Pre-sharded v14 replay retained for durable byte audit only."""

    replay_method: str = "builtin-semantic-scoring-recomputation-v14"
    contract: str = "neural-prior-scoring-replay-bundle-v14"

    def __post_init__(self) -> None:
        if (
            self.contract != "neural-prior-scoring-replay-bundle-v14"
            or self.replay_method
            != "builtin-semantic-scoring-recomputation-v14"
            or not self.ordered_case_ids
            or len(set(self.ordered_case_ids)) != len(self.ordered_case_ids)
            or len(self.ordered_case_ids)
            != len(self.ordered_evaluation_digests)
            or len(self.semantic_case_digests) != len(self.ordered_case_ids)
            or any(
                case_id not in self.ordered_case_ids
                for case_id in (
                    *self.dynamic_source_case_ids,
                    *self.background_case_ids,
                )
            )
            or any(
                value is not None
                for value in (
                    self.scoring_backend_certification_policy_digest,
                    self.scoring_backend_certification_evidence_digest,
                )
            )
            or self.raw_ingestor_trust_store_digest is None
            or self.verification_provenance_payload_sha256 is None
            or self.tensor_shard_sha256s
        ):
            raise ValueError("legacy v14 semantic replay manifest is invalid")
        for value in (
            self.scoring_input_artifact_digest,
            *self.ordered_evaluation_digests,
            *self.semantic_case_digests,
            self.algorithm_source_manifest_digest,
            self.runtime_compatibility_digest,
            self.runtime_exact_digest,
            self.tensor_archive_sha256,
            self.evaluation_payload_sha256,
            self.raw_provenance_payload_sha256,
            self.verification_provenance_payload_sha256,
            self.raw_ingestor_trust_store_digest,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("legacy v14 semantic replay digest is invalid")
        expected = {
            (case_id, role)
            for case_id in self.ordered_case_ids
            for role in (
                SCORING_REPLAY_REQUIRED_TENSOR_ROLES
                | (
                    SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
                    if case_id in self.dynamic_source_case_ids
                    else frozenset()
                )
                | (
                    SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
                    if case_id in self.background_case_ids
                    else frozenset()
                )
            )
        }
        actual = {(item.case_id, item.role) for item in self.tensor_records}
        if (
            actual != expected
            or len(actual) != len(self.tensor_records)
            or any(item.archive_sha256 is not None for item in self.tensor_records)
        ):
            raise ValueError("legacy v14 semantic replay tensor set is incomplete")
        object.__setattr__(self, "bundle_digest", _json_digest(self.payload))


@dataclass(frozen=True)
class LoadedScoringReplayBundle:
    manifest: (
        ScoringReplayBundleManifest
        | LegacyScoringReplayBundleManifestAuditV1
        | LegacyScoringReplayBundleManifestAuditV2
        | LegacyScoringReplayBundleManifestAuditV3
        | LegacyScoringReplayBundleManifestAuditV4
        | LegacyScoringReplayBundleManifestAuditV5
        | LegacyScoringReplayBundleManifestAuditV6
        | LegacyScoringReplayBundleManifestAuditV7
        | LegacyScoringReplayBundleManifestAuditV8
        | LegacyScoringReplayBundleManifestAuditV9
        | LegacyScoringReplayBundleManifestAuditV11
        | LegacyScoringReplayBundleManifestAuditV12
        | LegacyScoringReplayBundleManifestAuditV13
        | LegacyScoringReplayBundleManifestAuditV14
    )
    evaluations: tuple[PriorHoldoutEvaluation, ...]
    tensors: Mapping[tuple[str, str], Tensor]
    verification_bytes_verified: bool
    verification_reconstructed: bool
    verification_semantic_replay_verified: bool
    semantic_replay_verified: bool


class _ScoringReplayTensorShardStore(
    Mapping[tuple[str, str], Tensor]
):
    """Open and verify one content-addressed replay tensor at a time."""

    def __init__(
        self,
        root: Path,
        records: tuple[ScoringReplayTensorRecord, ...],
    ) -> None:
        self._root = root
        self._records = {
            (record.case_id, record.role): record for record in records
        }
        if len(self._records) != len(records):
            raise ValueError("scoring replay tensor records are duplicated")

    def __getitem__(self, key: tuple[str, str]) -> Tensor:
        record = self._records[key]
        if record.archive_sha256 is None:
            raise ValueError("scoring replay tensor shard identity is missing")
        archive_path = self._root / f"tensor_{record.archive_sha256}.npz"
        if _file_digest(archive_path) != record.archive_sha256:
            raise ValueError("scoring replay tensor shard checksum mismatch")
        preflight_npz_archive(
            archive_path,
            expected_members=frozenset({record.archive_member}),
            maximum_members=1,
            maximum_expanded_bytes=_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES,
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            tensor = torch.from_numpy(archive[record.archive_member].copy())
        if (
            str(tensor.dtype).removeprefix("torch.") != record.dtype
            or tuple(tensor.shape) != record.shape
            or tensor_digest(tensor) != record.tensor_digest
        ):
            raise ValueError("scoring replay tensor digest mismatch")
        return tensor

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


_REPLAY_BOOLEAN_ROLES = frozenset(
    {
        "input_qc_valid_mask",
        "input_source_available_mask",
        "candidate_publication_mask",
        "candidate_background_fallback_mask",
        "parent_publication_mask",
        "parent_background_fallback_mask",
        "verification_valid_mask",
        "candidate_state_valid_mask",
        "parent_state_valid_mask",
        "uncertainty_target_valid_mask",
        "uncertainty_target_echo_support",
        "state_target_valid_mask",
        "state_target_echo_support",
        "operational_issuance_mask",
        "outage_mask",
        "dynamic_qc_valid_mask",
        "nominal_source_coverage_mask",
        "resolved_source_coverage_mask",
        "raw_source_qc_flags",
        "verification_source_below_detection_reported",
        "verification_source_availability_by_time",
        "verification_source_present_mask",
        "verification_range_elevation_valid_mask",
        "verification_beam_blocked_mask",
        "verification_attenuation_qc_valid_mask",
        "verification_below_detection_censored_mask",
    }
)
_REPLAY_INTEGER_ROLES = frozenset(
    {
        "source_radar_index_map",
        "input_history_source_radar_index_map",
        "raw_source_reflectivity_bits",
        "raw_source_quality_bits",
        "raw_source_observation_std_bits",
        "verification_observation_state_code",
        "verification_source_radar_index_map",
    }
)
_REPLAY_VECTOR_ROLES = frozenset(
    {
        "candidate_state_displacement_yx",
        "parent_state_displacement_yx",
        "classifier_regime_logits",
        "classifier_range_logits",
    }
)
_REPLAY_SCALAR_ROLES = frozenset(
    {"candidate_state_log_growth", "parent_state_log_growth"}
)


def _validate_scoring_replay_case_tensors(
    tensors: Mapping[str, Tensor],
    *,
    dynamic_source: bool,
    background_present: bool,
) -> None:
    expected = SCORING_REPLAY_REQUIRED_TENSOR_ROLES | (
        SCORING_REPLAY_DYNAMIC_SOURCE_TENSOR_ROLES
        if dynamic_source
        else frozenset()
    ) | (
        SCORING_REPLAY_BACKGROUND_TENSOR_ROLES
        if background_present
        else frozenset()
    )
    if set(tensors) != expected:
        raise ValueError("scoring replay tensor roles are incomplete")
    input_frames = tensors["input_radar_frames"]
    forecast = tensors["candidate_forecast_dbz"]
    verification = tensors["verification_frames_dbz"]
    raw_reflectivity = tensors["raw_source_reflectivity_bits"]
    spatial_shape = input_frames.shape[-2:]
    if (
        input_frames.ndim != 3
        or not input_frames.is_floating_point()
        or forecast.ndim != 3
        or not forecast.is_floating_point()
        or verification.ndim != 3
        or not verification.is_floating_point()
        or forecast.shape != tensors["parent_forecast_dbz"].shape
        or verification.shape != tensors["verification_valid_mask"].shape
        or forecast.shape[-2:] != spatial_shape
        or verification.shape[-2:] != spatial_shape
        or raw_reflectivity.ndim != 3
        or raw_reflectivity.shape[-2:] != spatial_shape
        or tensors["raw_source_qc_flags"].shape != raw_reflectivity.shape
        or tensors["raw_source_quality_bits"].shape != raw_reflectivity.shape
        or tensors["raw_source_observation_std_bits"].shape
        != raw_reflectivity.shape
    ):
        raise ValueError("scoring replay forecast shapes are invalid")
    for role, tensor in tensors.items():
        if tensor.layout is not torch.strided or tensor.numel() == 0:
            raise ValueError("scoring replay tensor is invalid")
        if role in _REPLAY_BOOLEAN_ROLES and tensor.dtype is not torch.bool:
            raise ValueError(f"scoring replay {role} must be boolean")
        if role in _REPLAY_INTEGER_ROLES and tensor.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"scoring replay {role} must be integral")
        if (
            role not in _REPLAY_BOOLEAN_ROLES | _REPLAY_INTEGER_ROLES
            and not tensor.is_floating_point()
        ):
            raise ValueError(f"scoring replay {role} must be floating point")
        if role in _REPLAY_SCALAR_ROLES and tensor.ndim != 0:
            raise ValueError(f"scoring replay {role} must be scalar")
        if role in _REPLAY_VECTOR_ROLES and tensor.ndim != 1:
            raise ValueError(f"scoring replay {role} must be one-dimensional")
    two_dimensional = {
        role
        for role in expected
        if role
        not in _REPLAY_SCALAR_ROLES
        | _REPLAY_VECTOR_ROLES
        | {
            "input_radar_frames",
            "input_qc_valid_mask",
            "input_quality_weight",
            "input_observation_std_dbz",
            "input_source_available_mask",
            "background_frames_dbz",
            "candidate_forecast_dbz",
            "candidate_publication_mask",
            "candidate_background_fallback_mask",
            "candidate_forecast_confidence",
            "parent_forecast_dbz",
            "parent_publication_mask",
            "parent_background_fallback_mask",
            "parent_forecast_confidence",
            "verification_frames_dbz",
            "verification_valid_mask",
            "verification_quality_weight",
            "verification_observation_std_dbz",
            "verification_observation_state_code",
            "verification_source_radar_index_map",
            "verification_detection_limit_dbz",
            "verification_acquisition_time_offset_seconds",
            "verification_source_reflectivity_dbz",
            "verification_source_detection_limit_dbz",
            "verification_source_acquisition_time_offset_seconds",
            "verification_source_below_detection_reported",
            "verification_source_assignment_scores",
            "verification_source_availability_by_time",
            "verification_source_range_km",
            "verification_source_elevation_deg",
            "verification_source_beam_blockage_fraction",
            "verification_source_attenuation_qc_score",
            "verification_source_present_mask",
            "verification_range_elevation_valid_mask",
            "verification_beam_blocked_mask",
            "verification_attenuation_qc_valid_mask",
            "verification_below_detection_censored_mask",
            "operational_issuance_mask",
            "input_history_source_radar_index_map",
            "raw_source_reflectivity_bits",
            "raw_source_qc_flags",
            "raw_source_quality_bits",
            "raw_source_observation_std_bits",
            "nominal_source_coverage_mask",
            "resolved_source_coverage_mask",
            "effective_horizontal_range_m",
        }
    }
    if any(tensors[role].shape != spatial_shape for role in two_dimensional):
        raise ValueError("scoring replay spatial tensor shape is invalid")
    if dynamic_source and (
        tensors["input_history_source_radar_index_map"].shape
        != input_frames.shape
    ):
        raise ValueError("scoring replay source history shape is invalid")
    for role in (
        "input_qc_valid_mask",
        "input_quality_weight",
        "input_observation_std_dbz",
        "input_source_available_mask",
    ):
        if tensors[role].shape != input_frames.shape:
            raise ValueError("scoring replay input tensor shape is invalid")
    if background_present and (
        tensors["background_frames_dbz"].shape != input_frames.shape
    ):
        raise ValueError("scoring replay background tensor shape is invalid")
    for prefix in ("candidate", "parent"):
        for suffix in (
            "publication_mask",
            "background_fallback_mask",
            "forecast_confidence",
        ):
            if tensors[f"{prefix}_{suffix}"].shape != forecast.shape:
                raise ValueError("scoring replay issuance tensor shape is invalid")
    if tensors["operational_issuance_mask"].ndim != 3:
        raise ValueError("scoring replay issuance domain must be [lead,H,W]")
    verification_shape = verification.shape
    selected_roles = (
        "verification_valid_mask",
        "verification_quality_weight",
        "verification_observation_std_dbz",
        "verification_observation_state_code",
        "verification_source_radar_index_map",
        "verification_detection_limit_dbz",
        "verification_acquisition_time_offset_seconds",
        "verification_source_present_mask",
        "verification_range_elevation_valid_mask",
        "verification_beam_blocked_mask",
        "verification_attenuation_qc_valid_mask",
        "verification_below_detection_censored_mask",
    )
    source_cube_roles = (
        "verification_source_reflectivity_dbz",
        "verification_source_detection_limit_dbz",
        "verification_source_acquisition_time_offset_seconds",
        "verification_source_below_detection_reported",
        "verification_source_assignment_scores",
        "verification_source_range_km",
        "verification_source_elevation_deg",
        "verification_source_beam_blockage_fraction",
        "verification_source_attenuation_qc_score",
    )
    source_shape = tensors[source_cube_roles[0]].shape
    if (
        any(tensors[role].shape != verification_shape for role in selected_roles)
        or len(source_shape) != 4
        or source_shape[1:] != verification_shape
        or any(tensors[role].shape != source_shape for role in source_cube_roles)
        or tensors["verification_source_availability_by_time"].shape
        != source_shape[:2]
    ):
        raise ValueError("scoring replay verification provenance shape is invalid")
    if dynamic_source and (
        tensors["nominal_source_coverage_mask"].ndim != 3
        or tensors["resolved_source_coverage_mask"].shape
        != tensors["nominal_source_coverage_mask"].shape
        or tensors["nominal_source_coverage_mask"].shape[-2:] != spatial_shape
        or tensors["effective_horizontal_range_m"].shape != spatial_shape
    ):
        raise ValueError("scoring replay source coverage must be [lead,H,W]")


def _scoring_replay_range_geometry(
    case: ScoringReplayCaseArtifact,
) -> RangeGeometryContract | MosaicRangeGeometryContract:
    planned_case = case.plan.case(case.case_id)
    range_band = next(
        item
        for item in case.plan.range_band_contracts
        if item.contract_digest == planned_case.range_band_contract_digest
    )
    return next(
        item
        for item in case.plan.range_geometry_contracts
        if item.contract_digest == range_band.range_geometry_contract_digest
    )


def _current_verification_provenance_payload(
    case: ScoringReplayCaseArtifact,
) -> dict[str, object]:
    """Serialize the non-tensor half of one current verification chain."""

    verification = case.verification
    if (
        verification.contract != "radar-verification-bundle-v10"
        or type(verification.observation_error_derivation)
        is not ObservationErrorDerivationArtifact
    ):
        raise ValueError("current replay requires source-composed verification")
    derivation = cast(
        ObservationErrorDerivationArtifact,
        verification.observation_error_derivation,
    )
    raw_inputs = derivation.raw_inputs
    if (
        raw_inputs.contract != "verification-observation-derivation-inputs-v4"
        or type(raw_inputs.mask_derivation)
        is not VerificationObservationMaskDerivationArtifact
        or type(raw_inputs.source_identity)
        is not VerificationObservationSourceIdentity
    ):
        raise ValueError("current verification provenance is incomplete")
    mask_derivation = cast(
        VerificationObservationMaskDerivationArtifact,
        raw_inputs.mask_derivation,
    )
    evidence = mask_derivation.raw_evidence
    registry = derivation.source_registry
    plan = derivation.plan
    error_contract = derivation.observation_error_contract
    return {
        "case_id": case.case_id,
        "plan": plan.payload | {"plan_digest": plan.plan_digest},
        "source_registry": registry.payload
        | {
            "registry_digest": registry.registry_digest,
            "ordered_sources": [
                source.payload | {"source_digest": source.source_digest}
                for source in registry.ordered_sources
            ],
        },
        "source_identity": cast(
            VerificationObservationSourceIdentity,
            raw_inputs.source_identity,
        ).payload
        | {
            "identity_digest": cast(
                VerificationObservationSourceIdentity,
                raw_inputs.source_identity,
            ).identity_digest
        },
        "mask_evidence": evidence.payload
        | {"evidence_digest": evidence.evidence_digest},
        "mask_derivation": mask_derivation.payload
        | {"artifact_digest": mask_derivation.artifact_digest},
        "derivation_inputs": raw_inputs.payload
        | {"content_digest": raw_inputs.content_digest},
        "error_derivation": derivation.payload
        | {"artifact_digest": derivation.artifact_digest},
        "error_contract": error_contract.payload
        | {"contract_digest": error_contract.contract_digest},
        "verification_bundle": {
            "contract": verification.contract,
            "valid_times": list(verification.valid_times),
            "grid_contract_digest": verification.grid_contract_digest,
            "radar_product_digest": verification.radar_product_digest,
            "qc_pipeline_digest": verification.qc_pipeline_digest,
            "mask_policy_digest": verification.mask_policy_digest,
            "censor_policy_digest": verification.censor_policy_digest,
            "reflectivity_resolution_dbz": (
                verification.reflectivity_resolution_dbz
            ),
            "quantization_origin_dbz": verification.quantization_origin_dbz,
            "threshold_bin_convention": verification.threshold_bin_convention,
            "floor_representation_contract_digest": (
                verification.floor_representation_contract_digest
            ),
            "content_digest": verification.content_digest,
        },
    }


def _validate_current_verification_provenance_payload(
    payload_text: str,
    *,
    manifest: ScoringReplayBundleManifest,
    tensors: Mapping[tuple[str, str], Tensor],
) -> None:
    """Cold-start the current source-composed verification chain from bytes."""

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("scoring replay verification provenance is invalid") from error
    if (
        not isinstance(payload, list)
        or payload_text
        != json.dumps(payload, sort_keys=True, separators=(",", ":"))
        or tuple(item.get("case_id") for item in payload if isinstance(item, dict))
        != manifest.ordered_case_ids
    ):
        raise ValueError("scoring replay verification provenance is invalid")

    for retained in payload:
        if not isinstance(retained, dict):
            raise ValueError("scoring replay verification provenance is invalid")
        case_id = cast(str, retained["case_id"])
        case_tensors = {
            record.role: tensors[(record.case_id, record.role)]
            for record in manifest.tensor_records
            if record.case_id == case_id
        }

        plan_payload = dict(cast(dict[str, object], retained["plan"]))
        plan_digest = plan_payload.pop("plan_digest", None)
        plan = VerificationObservationErrorPlan(**cast(Any, plan_payload))
        if plan.plan_digest != plan_digest:
            raise ValueError("verification replay plan digest mismatch")

        registry_payload = dict(
            cast(dict[str, object], retained["source_registry"])
        )
        source_payloads = registry_payload.pop("ordered_sources", None)
        registry_digest = registry_payload.pop("registry_digest", None)
        if not isinstance(source_payloads, list):
            raise ValueError("verification replay source registry is invalid")
        sources: list[ObservationRadarSource] = []
        for source_payload_value in source_payloads:
            source_payload = dict(cast(dict[str, object], source_payload_value))
            source_digest = source_payload.pop("source_digest", None)
            source = ObservationRadarSource(**cast(Any, source_payload))
            if source.source_digest != source_digest:
                raise ValueError("verification replay source digest mismatch")
            sources.append(source)
        registry = MosaicObservationSourceRegistry(
            radar_source_kind=cast(Any, registry_payload["radar_source_kind"]),
            ordered_sources=tuple(sources),
            contract=cast(str, registry_payload["contract"]),
        )
        if (
            registry.registry_digest != registry_digest
            or registry.payload != registry_payload
        ):
            raise ValueError("verification replay registry digest mismatch")
        registry.validate_against_plan(plan)

        identity_payload = dict(
            cast(dict[str, object], retained["source_identity"])
        )
        identity_digest = identity_payload.pop("identity_digest", None)
        identity_payload.pop("source_acquisition_time_identity_digest", None)
        valid_times_value = identity_payload["valid_times"]
        acquisition_times_value = identity_payload["acquisition_valid_times"]
        if not isinstance(valid_times_value, list) or not all(
            isinstance(value, str) for value in valid_times_value
        ):
            raise ValueError("verification replay valid times are invalid")
        if not isinstance(acquisition_times_value, list) or not all(
            isinstance(value, str) for value in acquisition_times_value
        ):
            raise ValueError(
                "verification replay acquisition valid times are invalid"
            )
        identity_payload["valid_times"] = tuple(valid_times_value)
        identity_payload["acquisition_valid_times"] = tuple(
            acquisition_times_value
        )
        source_identity = VerificationObservationSourceIdentity(
            **cast(Any, identity_payload)
        )
        if source_identity.identity_digest != identity_digest:
            raise ValueError("verification replay source identity mismatch")

        evidence_payload = dict(
            cast(dict[str, object], retained["mask_evidence"])
        )
        evidence_digest = evidence_payload.pop("evidence_digest", None)
        evidence = VerificationObservationMaskEvidence(
            source_identity=source_identity,
            source_registry_artifact_digest=cast(
                str, evidence_payload["source_registry_artifact_digest"]
            ),
            ordered_source_digests=tuple(
                cast(list[str], evidence_payload["ordered_source_digests"])
            ),
            reflectivity_dbz_by_source=case_tensors[
                "verification_source_reflectivity_dbz"
            ],
            detection_limit_dbz_by_source=case_tensors[
                "verification_source_detection_limit_dbz"
            ],
            acquisition_time_offset_seconds_by_source=case_tensors[
                "verification_source_acquisition_time_offset_seconds"
            ],
            below_detection_reported_by_source=case_tensors[
                "verification_source_below_detection_reported"
            ],
            source_assignment_scores=case_tensors[
                "verification_source_assignment_scores"
            ],
            source_availability_by_time=case_tensors[
                "verification_source_availability_by_time"
            ],
            range_km_by_source=case_tensors["verification_source_range_km"],
            elevation_deg_by_source=case_tensors[
                "verification_source_elevation_deg"
            ],
            beam_blockage_fraction_by_source=case_tensors[
                "verification_source_beam_blockage_fraction"
            ],
            attenuation_qc_score_by_source=case_tensors[
                "verification_source_attenuation_qc_score"
            ],
            range_elevation_validity_domain_digest=cast(
                str,
                evidence_payload["range_elevation_validity_domain_digest"],
            ),
            beam_blockage_visibility_mask_digest=cast(
                str,
                evidence_payload["beam_blockage_visibility_mask_digest"],
            ),
            spatial_correlation_block_digest=cast(
                str, evidence_payload["spatial_correlation_block_digest"]
            ),
            contract=cast(str, evidence_payload["contract"]),
        )
        if evidence.evidence_digest != evidence_digest:
            raise ValueError("verification replay evidence digest mismatch")

        mask_derivation = derive_verification_observation_masks(
            plan=plan,
            raw_evidence=evidence,
            source_registry=registry,
        )
        retained_mask = cast(dict[str, object], retained["mask_derivation"])
        if (
            mask_derivation.artifact_digest
            != retained_mask.get("artifact_digest")
            or mask_derivation.payload
            != {key: value for key, value in retained_mask.items() if key != "artifact_digest"}
        ):
            raise ValueError("verification replay mask derivation mismatch")
        expected_mask_tensors = {
            "verification_frames_dbz": mask_derivation.selected_frames_dbz,
            "verification_detection_limit_dbz": (
                mask_derivation.selected_detection_limit_dbz
            ),
            "verification_acquisition_time_offset_seconds": (
                mask_derivation.selected_acquisition_time_offset_seconds
            ),
            "verification_source_present_mask": mask_derivation.source_present_mask,
            "verification_range_elevation_valid_mask": (
                mask_derivation.range_elevation_valid_mask
            ),
            "verification_beam_blocked_mask": mask_derivation.beam_blocked_mask,
            "verification_attenuation_qc_valid_mask": (
                mask_derivation.attenuation_qc_valid_mask
            ),
            "verification_below_detection_censored_mask": (
                mask_derivation.below_detection_censored_mask
            ),
        }
        if any(
            not bool(torch.equal(case_tensors[role], value))
            for role, value in expected_mask_tensors.items()
        ):
            raise ValueError("verification replay mask tensor mismatch")

        raw_inputs = VerificationObservationDerivationInputs.from_mask_derivation(
            mask_derivation
        )
        retained_inputs = cast(dict[str, object], retained["derivation_inputs"])
        if (
            raw_inputs.content_digest != retained_inputs.get("content_digest")
            or raw_inputs.payload
            != {key: value for key, value in retained_inputs.items() if key != "content_digest"}
        ):
            raise ValueError("verification replay input digest mismatch")
        derivation = derive_verification_observation_error(
            plan=plan,
            raw_verification_source=raw_inputs,
            source_registry=registry,
        )
        retained_derivation = cast(
            dict[str, object], retained["error_derivation"]
        )
        if (
            derivation.artifact_digest
            != retained_derivation.get("artifact_digest")
            or derivation.payload
            != {
                key: value
                for key, value in retained_derivation.items()
                if key != "artifact_digest"
            }
        ):
            raise ValueError("verification replay error derivation mismatch")
        error_contract = derivation.observation_error_contract
        retained_contract = cast(dict[str, object], retained["error_contract"])
        if (
            error_contract.contract_digest
            != retained_contract.get("contract_digest")
            or error_contract.payload
            != {
                key: value
                for key, value in retained_contract.items()
                if key != "contract_digest"
            }
        ):
            raise ValueError("verification replay error contract mismatch")
        source_map = (
            None
            if registry.radar_source_kind == "single_site"
            else case_tensors["verification_source_radar_index_map"]
        )
        if registry.radar_source_kind == "single_site" and bool(
            torch.any(case_tensors["verification_source_radar_index_map"] != 0)
        ):
            raise ValueError(
                "single-site verification replay source map is not canonical"
            )
        bundle_payload = dict(
            cast(dict[str, object], retained["verification_bundle"])
        )
        content_digest = bundle_payload.pop("content_digest", None)
        bundle_valid_times = bundle_payload["valid_times"]
        if not isinstance(bundle_valid_times, list) or not all(
            isinstance(value, str) for value in bundle_valid_times
        ):
            raise ValueError("verification replay bundle valid times are invalid")
        bundle_payload["valid_times"] = tuple(bundle_valid_times)
        verification = VerificationBundle(
            **cast(Any, bundle_payload),
            frames_dbz=case_tensors["verification_frames_dbz"],
            valid_mask=case_tensors["verification_valid_mask"],
            quality_weight=case_tensors["verification_quality_weight"],
            observation_std_dbz=case_tensors[
                "verification_observation_std_dbz"
            ],
            observation_state_code=case_tensors[
                "verification_observation_state_code"
            ],
            source_radar_index_map=source_map,
            detection_limit_dbz=case_tensors[
                "verification_detection_limit_dbz"
            ],
            acquisition_time_offset_seconds=case_tensors[
                "verification_acquisition_time_offset_seconds"
            ],
            observation_error_contract=error_contract,
            observation_error_derivation=derivation,
        )
        if verification.content_digest != content_digest:
            raise ValueError("verification replay bundle digest mismatch")


def _validate_current_raw_provenance_payload(
    payload_text: str,
    *,
    manifest: ScoringReplayBundleManifest,
    tensors: Mapping[tuple[str, str], Tensor],
) -> None:
    """Rebuild raw products and replay their deterministic analysis transform."""

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("scoring replay raw provenance is invalid") from error
    if (
        not isinstance(payload, list)
        or payload_text
        != json.dumps(payload, sort_keys=True, separators=(",", ":"))
        or tuple(item.get("case_id") for item in payload if isinstance(item, dict))
        != manifest.ordered_case_ids
    ):
        raise ValueError("scoring replay raw provenance is invalid")

    for case_payload in payload:
        if not isinstance(case_payload, dict):
            raise ValueError("scoring replay raw provenance is invalid")
        case_id = cast(str, case_payload["case_id"])
        case_tensors = {
            role: value
            for (retained_case, role), value in tensors.items()
            if retained_case == case_id
        }
        input_plan_payload = dict(cast(dict[str, object], case_payload["input_plan"]))
        retained_plan_digest = input_plan_payload.pop("plan_digest", None)
        valid_times = input_plan_payload.get("valid_times")
        if not isinstance(valid_times, list):
            raise ValueError("scoring replay input plan is invalid")
        input_plan_payload["valid_times"] = tuple(valid_times)
        input_plan = NeuralPriorInputPlan(**cast(Any, input_plan_payload))
        if input_plan.plan_digest != retained_plan_digest:
            raise ValueError("scoring replay input-plan digest mismatch")

        raw_payloads = case_payload.get("resolved_raw_observations")
        if not isinstance(raw_payloads, list) or not raw_payloads:
            raise ValueError("scoring replay raw products are incomplete")
        ordered_payloads = sorted(
            raw_payloads,
            key=lambda item: (
                (
                    item["raw_grid_volume"]["acquisition_valid_time"]
                    if item.get("contract")
                    == "resolved-raw-observation-receipt-v1"
                    else item["acquisition_valid_time"]
                ),
                (
                    item["raw_grid_volume"]["radar_site_digest"]
                    if item.get("contract")
                    == "resolved-raw-observation-receipt-v1"
                    else item["radar_site_digest"]
                ),
            ),
        )
        raw_roles = (
            "raw_source_reflectivity_bits",
            "raw_source_qc_flags",
            "raw_source_quality_bits",
            "raw_source_observation_std_bits",
        )
        present_count = sum(
            item.get("contract") == "resolved-raw-observation-receipt-v1"
            for item in ordered_payloads
        )
        if any(
            case_tensors[role].shape[0] != present_count for role in raw_roles
        ):
            raise ValueError("scoring replay raw product count is invalid")
        receipts: list[RawObservationResolutionReceipt] = []
        raw_index = 0
        for retained_receipt in ordered_payloads:
            if not isinstance(retained_receipt, dict):
                raise ValueError("scoring replay raw receipt is invalid")
            if (
                retained_receipt.get("contract")
                == "missing-raw-observation-receipt-v1"
            ):
                missing_values = dict(retained_receipt)
                retained_receipt_digest = missing_values.pop(
                    "receipt_digest", None
                )
                retained_identity_digest = missing_values.pop(
                    "resolution_identity_digest", None
                )
                missing = MissingRawObservationReceipt(
                    **cast(Any, missing_values)
                )
                if (
                    missing.receipt_digest != retained_receipt_digest
                    or missing.resolution_identity_digest
                    != retained_identity_digest
                ):
                    raise ValueError("scoring replay missing receipt changed")
                receipts.append(missing)
                continue
            retained_volume = dict(retained_receipt["raw_grid_volume"])
            rebuilt_volume = CanonicalRawGridVolumeArtifact.from_encoded_tensors(
                raw_reflectivity_bits=(
                    case_tensors["raw_source_reflectivity_bits"][raw_index]
                ),
                raw_qc_flags=case_tensors["raw_source_qc_flags"][raw_index],
                raw_quality_bits=case_tensors["raw_source_quality_bits"][
                    raw_index
                ],
                raw_observation_std_bits=(
                    case_tensors["raw_source_observation_std_bits"][raw_index]
                ),
                radar_site_digest=cast(str, retained_volume["radar_site_digest"]),
                acquisition_valid_time=cast(
                    str, retained_volume["acquisition_valid_time"]
                ),
                canonical_scan_identity_digest=cast(
                    str, retained_volume["canonical_scan_identity_digest"]
                ),
                radar_product_digest=cast(
                    str, retained_volume["radar_product_digest"]
                ),
                grid_contract_digest=cast(
                    str, retained_volume["grid_contract_digest"]
                ),
            )
            if retained_volume != rebuilt_volume.payload | {
                "artifact_digest": rebuilt_volume.artifact_digest
            }:
                raise ValueError("scoring replay raw-grid product digest mismatch")
            identity_values = dict(retained_receipt["raw_volume_identity"])
            retained_identity_digest = identity_values.pop("identity_digest", None)
            identity = CanonicalRawVolumeIdentity(**cast(Any, identity_values))
            attestation_values = dict(retained_receipt["raw_volume_attestation"])
            retained_attestation_digest = attestation_values.pop(
                "attestation_digest", None
            )
            attestation = RawVolumeAttestation(**cast(Any, attestation_values))
            receipt = ResolvedRawObservationReceipt(
                slot_plan_digest=cast(str, retained_receipt["slot_plan_digest"]),
                raw_grid_volume=rebuilt_volume,
                raw_volume_identity=identity,
                raw_volume_attestation=attestation,
                contract=cast(str, retained_receipt["contract"]),
            )
            if (
                identity.identity_digest != retained_identity_digest
                or attestation.attestation_digest != retained_attestation_digest
                or receipt.receipt_digest != retained_receipt.get("receipt_digest")
            ):
                raise ValueError("scoring replay raw receipt digest mismatch")
            receipts.append(receipt)
            raw_index += 1

        resolution_values = dict(case_payload["global_raw_resolution_receipt"])
        retained_resolution_digest = resolution_values.pop("receipt_digest", None)
        resolution_values["slot_identity_bindings"] = tuple(
            tuple(item) for item in resolution_values["slot_identity_bindings"]
        )
        resolution = GlobalRawVolumeResolutionReceipt(
            **cast(Any, resolution_values)
        )
        derivation_values = dict(case_payload["analysis_input_derivation"])
        retained_derivation_digest = derivation_values.pop("artifact_digest", None)
        for name in (
            "resolved_raw_observation_receipt_digests",
            "canonical_raw_volume_identity_digests",
            "background_valid_times",
            "background_input_identity_digests",
        ):
            derivation_values[name] = tuple(derivation_values[name])
        derivation = AnalysisInputDerivationArtifact(
            **cast(Any, derivation_values)
        )
        if (
            resolution.receipt_digest != retained_resolution_digest
            or derivation.artifact_digest != retained_derivation_digest
            or derivation.input_plan_digest != input_plan.plan_digest
            or derivation.global_raw_resolution_receipt_digest
            != resolution.receipt_digest
            or derivation.resolved_raw_observation_receipt_digests
            != tuple(sorted(item.receipt_digest for item in receipts))
            or resolution.slot_identity_bindings
            != tuple(
                sorted(
                    (
                        item.slot_plan_digest,
                        item.resolution_identity_digest,
                    )
                    for item in receipts
                )
            )
        ):
            raise ValueError("scoring replay raw derivation lineage mismatch")

        retained_background = case_payload.get("background_run_lineage")
        if derivation.background_frames_digest is None:
            if retained_background is not None or case_id in manifest.background_case_ids:
                raise ValueError("scoring replay background lineage is unexpected")
        else:
            if (
                not isinstance(retained_background, dict)
                or set(retained_background)
                != {
                    "background_valid_times",
                    "operational_data_identity_json",
                    "operational_data_identity_digest",
                }
                or case_id not in manifest.background_case_ids
            ):
                raise ValueError("scoring replay background lineage is incomplete")
            retained_identity_json = retained_background[
                "operational_data_identity_json"
            ]
            retained_identity_digest = retained_background[
                "operational_data_identity_digest"
            ]
            if not isinstance(retained_identity_json, str):
                raise ValueError("scoring replay background identity is invalid")
            background_identity = OperationalDataIdentity.from_json(
                retained_identity_json
            )
            background_times = tuple(
                cast(list[str], retained_background["background_valid_times"])
            )
            background_tensor = case_tensors["background_frames_dbz"]
            source_digest = _json_digest(
                {
                    "contract": "background-source-cycle-identity-v1",
                    "background_model_digest": (
                        background_identity.background_model_digest
                    ),
                    "background_cycle_rule_digest": (
                        background_identity.background_cycle_rule_digest
                    ),
                    "background_valid_times": list(background_times),
                    "operational_data_identity_digest": background_identity.digest,
                }
            )
            frame_digests = tuple(
                _json_digest(
                    {
                        "contract": "background-input-identity-v2",
                        "background_source_identity_digest": source_digest,
                        "background_valid_time": valid_time,
                        "background_frame_digest": tensor_digest(
                            background_tensor[index]
                        ),
                    }
                )
                for index, valid_time in enumerate(background_times)
            )
            if (
                retained_identity_digest != background_identity.digest
                or background_times != derivation.background_valid_times
                or source_digest != derivation.background_source_identity_digest
                or frame_digests != derivation.background_input_identity_digests
                or tensor_digest(background_tensor)
                != derivation.background_frames_digest
            ):
                raise ValueError("scoring replay background model lineage changed")

        retained_coverage = case_payload.get("resolved_source_coverage")
        coverage: ResolvedSourceCoverageArtifact | None = None
        if retained_coverage is not None:
            if not isinstance(retained_coverage, dict):
                raise ValueError("scoring replay source coverage is invalid")
            coverage_values = dict(retained_coverage)
            retained_coverage_digest = coverage_values.pop("artifact_digest", None)
            coverage = object.__new__(ResolvedSourceCoverageArtifact)
            for name, value in coverage_values.items():
                if name in {"source_radar_site_digests", "resolved_cell_counts"}:
                    value = tuple(cast(list[object], value))
                object.__setattr__(coverage, name, value)
            for name, value in (
                ("_source_radar_index_map", case_tensors["source_radar_index_map"]),
                (
                    "_input_history_source_radar_index_map",
                    case_tensors["input_history_source_radar_index_map"],
                ),
                ("_outage_mask", case_tensors["outage_mask"]),
                ("_dynamic_qc_valid_mask", case_tensors["dynamic_qc_valid_mask"]),
                (
                    "_nominal_source_coverage_mask",
                    case_tensors["nominal_source_coverage_mask"],
                ),
                (
                    "_resolved_mask",
                    case_tensors["resolved_source_coverage_mask"],
                ),
            ):
                object.__setattr__(coverage, name, value)
            object.__setattr__(coverage, "artifact_digest", retained_coverage_digest)
            validate_resolved_source_coverage_artifact(coverage)
            if derivation.source_selection_evidence_digest != coverage.artifact_digest:
                raise ValueError("scoring replay source-selection lineage mismatch")
        elif case_id in manifest.dynamic_source_case_ids:
            raise ValueError("scoring replay mosaic source coverage is missing")

        retained_geometry = case_payload.get("range_geometry_contract")
        if not isinstance(retained_geometry, dict):
            raise ValueError("scoring replay range geometry is incomplete")
        geometry_values = dict(retained_geometry)
        retained_geometry_digest = geometry_values.pop(
            "contract_digest",
            None,
        )
        for name in (
            "range_regime_labels",
            "radial_distance_edges_m",
            "radar_site_digests",
            "radar_site_location_digests",
        ):
            if name in geometry_values:
                value = geometry_values[name]
                if not isinstance(value, list):
                    raise ValueError("scoring replay range geometry is invalid")
                geometry_values[name] = tuple(value)
        if "radar_projected_xy_m" in geometry_values:
            coordinates = geometry_values["radar_projected_xy_m"]
            if not isinstance(coordinates, list):
                raise ValueError("scoring replay range geometry is invalid")
            geometry_values["radar_projected_xy_m"] = tuple(
                tuple(point) for point in coordinates
            )
        try:
            geometry = (
                MosaicRangeGeometryContract(**cast(Any, geometry_values))
                if geometry_values.get("contract")
                == "mosaic-horizontal-range-geometry-contract-v3"
                else RangeGeometryContract(**cast(Any, geometry_values))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("scoring replay range geometry is invalid") from error
        if geometry.contract_digest != retained_geometry_digest:
            raise ValueError("scoring replay range geometry digest mismatch")
        if coverage is not None:
            if (
                type(geometry) is not MosaicRangeGeometryContract
                or geometry.source_radar_registry_digest
                != coverage.source_radar_registry_digest
            ):
                raise ValueError("scoring replay mosaic geometry is invalid")
            _, effective_range = resolve_mosaic_range_geometry(
                geometry,
                grid_x_m=case_tensors["range_grid_x_m"],
                grid_y_m=case_tensors["range_grid_y_m"],
                source_radar_index_map=case_tensors[
                    "source_radar_index_map"
                ],
            )
            if tensor_digest(effective_range) != tensor_digest(
                case_tensors["effective_horizontal_range_m"]
            ):
                raise ValueError("scoring replay effective mosaic range changed")

        derived_frames, derived_masks, derived_quality, derived_std = (
            _derive_analysis_inputs_from_raw_products(
                input_plan=input_plan,
                resolved_raw_observations=tuple(receipts),
                resolved_source_coverage=coverage,
            )
        )
        derived_source_available = (
            torch.ones_like(derived_masks, dtype=torch.bool)
            if coverage is None
            else coverage.input_history_source_radar_index_map >= 0
        )
        if (
            tensor_digest(derived_frames)
            != tensor_digest(case_tensors["input_radar_frames"])
            or tensor_digest(derived_masks)
            != tensor_digest(case_tensors["input_qc_valid_mask"])
            or tensor_digest(derived_quality)
            != tensor_digest(case_tensors["input_quality_weight"])
            or tensor_digest(derived_std)
            != tensor_digest(case_tensors["input_observation_std_dbz"])
            or tensor_digest(derived_source_available)
            != tensor_digest(case_tensors["input_source_available_mask"])
            or tensor_digest(derived_std) != derivation.observation_std_dbz_digest
            or tensor_digest(derived_source_available)
            != derivation.source_available_mask_digest
            or tensor_digest(
                learned_radar_input_features(
                    derived_frames,
                    derived_masks,
                    derived_quality,
                    derived_std,
                    derived_source_available,
                )
            )
            != derivation.learned_model_input_features_digest
            or derivation.input_frames_digest != tensor_digest(derived_frames)
            or derivation.observation_masks_digest != tensor_digest(derived_masks)
            or derivation.observation_quality_weight_digest
            != tensor_digest(derived_quality)
        ):
            raise ValueError("scoring replay raw products do not reproduce input")


@dataclass(frozen=True)
class _EpisodeLedgerInitializationState:
    """Private proof that an EpisodeLedger completed normal initialization."""

    token: object
    root: Path
    index_path: Path


class EpisodeLedger:
    """Immutable M0 storage backed by SQLite, JSON, and NPZ."""

    __slots__ = (
        "root",
        "episodes_dir",
        "interventions_dir",
        "scoring_replays_dir",
        "analysis_input_provenance_dir",
        "index_path",
        "_initialization_state",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.interventions_dir = self.root / "interventions"
        self.scoring_replays_dir = self.root / "scoring_replays"
        self.analysis_input_provenance_dir = (
            self.root / "analysis_input_provenance"
        )
        self.index_path = self.root / "index.sqlite"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.interventions_dir.mkdir(parents=True, exist_ok=True)
        self.scoring_replays_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_input_provenance_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_index()
        self._initialization_state = _EpisodeLedgerInitializationState(
            token=_EPISODE_LEDGER_INITIALIZATION_TOKEN,
            root=self.root,
            index_path=self.index_path,
        )

    @classmethod
    def _open_existing_index(cls, index_path: Path) -> EpisodeLedger:
        """Bind a verifier to an existing approved ledger without creating files."""

        if type(index_path) is not _NATIVE_PATH_TYPE or not index_path.is_absolute():
            raise ValueError("automatic deployment ledger index is invalid")
        root = index_path.parent
        retained = object.__new__(cls)
        retained.root = root
        retained.episodes_dir = root / "episodes"
        retained.interventions_dir = root / "interventions"
        retained.scoring_replays_dir = root / "scoring_replays"
        retained.analysis_input_provenance_dir = (
            root / "analysis_input_provenance"
        )
        retained.index_path = index_path
        if (
            not index_path.is_file()
            or index_path.is_symlink()
            or any(
                not directory.is_dir() or directory.is_symlink()
                for directory in (
                    retained.episodes_dir,
                    retained.interventions_dir,
                    retained.scoring_replays_dir,
                    retained.analysis_input_provenance_dir,
                )
            )
        ):
            raise ValueError("automatic deployment ledger layout is invalid")
        retained._initialization_state = _EpisodeLedgerInitializationState(
            token=_EPISODE_LEDGER_INITIALIZATION_TOKEN,
            root=root,
            index_path=index_path,
        )
        retained._require_initialized()
        return retained

    def _require_initialized(self) -> None:
        """Reject object.__new__ instances and mutable path rebinding."""

        try:
            state = object.__getattribute__(self, "_initialization_state")
            root = object.__getattribute__(self, "root")
            index_path = object.__getattribute__(self, "index_path")
            episodes_dir = object.__getattribute__(self, "episodes_dir")
            interventions_dir = object.__getattribute__(
                self,
                "interventions_dir",
            )
            scoring_replays_dir = object.__getattribute__(
                self,
                "scoring_replays_dir",
            )
            analysis_input_provenance_dir = object.__getattribute__(
                self,
                "analysis_input_provenance_dir",
            )
        except AttributeError as error:
            raise TypeError("EpisodeLedger was not normally initialized") from error
        if type(state) is not _EpisodeLedgerInitializationState:
            raise TypeError("EpisodeLedger initialization state is invalid")
        path_values = (
            state.root,
            state.index_path,
            root,
            index_path,
            episodes_dir,
            interventions_dir,
            scoring_replays_dir,
            analysis_input_provenance_dir,
        )
        if (
            any(type(value) is not _NATIVE_PATH_TYPE for value in path_values)
            or state.token is not _EPISODE_LEDGER_INITIALIZATION_TOKEN
            or state.root != root
            or state.index_path != index_path
            or root != root.expanduser().resolve()
            or episodes_dir != root / "episodes"
            or interventions_dir != root / "interventions"
            or scoring_replays_dir != root / "scoring_replays"
            or analysis_input_provenance_dir
            != root / "analysis_input_provenance"
            or index_path != root / "index.sqlite"
            or not index_path.is_file()
        ):
            raise TypeError("EpisodeLedger initialization state is invalid")

    def append(self, episode: SensitivityEpisode) -> Path:
        """Append one episode; the SQLite commit makes it visible."""

        episode = _owned_episode_copy(episode)
        _validate_m0_snapshot(episode.snapshot)
        target = self.episodes_dir / episode.episode_id
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{episode.episode_id}.",
                dir=self.episodes_dir,
            )
        )
        published = False
        connection: sqlite3.Connection | None = None
        try:
            arrays = _snapshot_arrays(episode)
            arrays_path = temporary / "sensitivity_arrays.npz"
            np.savez_compressed(
                arrays_path,
                **cast(dict[str, Any], arrays),
            )

            manifest = _episode_manifest(episode, arrays)
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(_json_text(manifest), encoding="utf-8")

            checksums = {
                "manifest.json": _file_digest(manifest_path),
                "sensitivity_arrays.npz": _file_digest(arrays_path),
            }
            (temporary / "checksums.json").write_text(
                _json_text(checksums),
                encoding="utf-8",
            )
            for name in (*_EPISODE_FILES, "checksums.json"):
                _fsync_file(temporary / name)
            _fsync_directory(temporary)

            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            if _row_exists(connection, episode.episode_id):
                raise FileExistsError(
                    f"episode already exists: {episode.episode_id}"
                )
            if target.exists():
                raise FileExistsError(
                    f"uncommitted episode directory exists: {episode.episode_id}"
                )

            os.rename(temporary, target)
            published = True
            _fsync_directory(self.episodes_dir)
            self._insert_scalar_impacts(connection, episode)
            self._insert_episode(connection, episode, target, checksums)
            connection.commit()
        except Exception:
            if connection is not None:
                connection.rollback()
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and target.exists() and not self._is_indexed(
                episode.episode_id
            ):
                shutil.rmtree(target)
            raise
        finally:
            if connection is not None:
                connection.close()
        return target

    def load(self, episode_id: str, *, verify: bool = True) -> LoadedEpisode:
        """Load a committed episode and optionally verify its integrity."""

        target, _ = self._indexed_target(episode_id)
        if verify:
            self.verify(episode_id)

        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        with np.load(target / "sensitivity_arrays.npz", allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        return LoadedEpisode(manifest=manifest, arrays=arrays)

    def verify(self, episode_id: str) -> None:
        """Raise ``ValueError`` if a committed episode was modified."""

        target, row = self._indexed_target(episode_id)
        required = _EPISODE_FILES | {"checksums.json"}
        if {path.name for path in target.iterdir()} != required:
            raise ValueError(f"episode file set is invalid: {episode_id}")
        if not all(
            (target / name).is_file()
            and not (target / name).is_symlink()
            for name in required
        ):
            raise ValueError(f"episode files are incomplete: {episode_id}")

        checksums = json.loads((target / "checksums.json").read_text("utf-8"))
        if set(checksums) != _EPISODE_FILES:
            raise ValueError("checksums.json has an invalid file set")

        manifest_hash = _file_digest(target / "manifest.json")
        arrays_hash = _file_digest(target / "sensitivity_arrays.npz")
        if manifest_hash != checksums["manifest.json"]:
            raise ValueError(f"checksum mismatch: {episode_id}/manifest.json")
        if arrays_hash != checksums["sensitivity_arrays.npz"]:
            raise ValueError(
                f"checksum mismatch: {episode_id}/sensitivity_arrays.npz"
            )
        if manifest_hash != row["manifest_sha256"]:
            raise ValueError("manifest checksum disagrees with the index")
        if arrays_hash != row["arrays_sha256"]:
            raise ValueError("array checksum disagrees with the index")

        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        _verify_manifest(manifest, episode_id, row)
        _verify_array_schema(
            target / "sensitivity_arrays.npz",
            manifest["arrays"],
        )

    def list_episodes(
        self,
        *,
        contract_hash: str | None = None,
        minimum_trust: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return searchable committed metadata, newest first."""

        query = """
            SELECT episode_id, issue_time, radar_id, contract_hash,
                   trust_score, promotion_eligible, impact_available, path
            FROM episodes
            WHERE trust_score >= ?
        """
        parameters: list[Any] = [minimum_trust]
        if contract_hash is not None:
            query += " AND contract_hash = ?"
            parameters.append(contract_hash)
        query += " ORDER BY issue_time DESC"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_impacts(self, episode_id: str) -> list[dict[str, Any]]:
        """Return scalar lead/metric/input summaries for one episode."""

        _validate_episode_id(episode_id)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            if not _row_exists(connection, episode_id):
                raise KeyError(f"unknown episode: {episode_id}")
            rows = connection.execute(
                """
                SELECT lead_minutes, metric_name, input_offset_minutes,
                       direct_path_status, forecast_score,
                       direct_sensitivity_norm, direct_impact,
                       direct_normalized_reward
                FROM episode_impacts
                WHERE episode_id = ?
                ORDER BY lead_minutes, metric_name, input_offset_minutes
                """,
                (episode_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_variational_learning_approval(
        self,
        learning: VariationalLearningImpact,
    ) -> str:
        """Append the digest evidence for one eligible P1 learning result."""

        validate_variational_learning_impact(learning)
        evidence = learning.approval_evidence
        if not learning.eligibility.eligible or evidence is None:
            raise ValueError("only eligible learning impacts can be recorded")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO variational_learning_approvals (
                    learning_result_digest, approval_evidence_digest,
                    evidence_contract, policy_digest, trust_store_digest,
                    fsoi_digest, full_step_analysis_digest,
                    half_step_analysis_digest, full_step_forecast_digest,
                    half_step_forecast_digest,
                    first_order_validation_digest,
                    learning_impact_digest, approved_action_digest,
                    nominal_input_bundle_digest,
                    nominal_full_analysis_input_digest,
                    selection_mode, candidate_id,
                    candidate_rank, candidate_score,
                    candidate_perturbation_digest, ranking_digest,
                    ranking_policy_digest, ranking_objective,
                    whitener_operations_per_apply,
                    observed_whitener_apply_count,
                    observed_whitener_total_operations, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learning.learning_result_digest,
                    evidence.digest,
                    evidence.contract,
                    evidence.policy_digest,
                    evidence.trust_store_digest,
                    evidence.fsoi_digest,
                    evidence.full_step_analysis_digest,
                    evidence.half_step_analysis_digest,
                    evidence.full_step_forecast_digest,
                    evidence.half_step_forecast_digest,
                    evidence.first_order_validation_digest,
                    evidence.learning_impact_digest,
                    evidence.approved_action_digest,
                    evidence.nominal_input_bundle_digest,
                    evidence.nominal_full_analysis_input_digest,
                    evidence.selection_mode,
                    evidence.candidate_id,
                    evidence.candidate_rank,
                    evidence.candidate_score,
                    evidence.candidate_perturbation_digest,
                    evidence.ranking_digest,
                    evidence.ranking_policy_digest,
                    evidence.ranking_objective,
                    evidence.whitener_operations_per_apply,
                    evidence.observed_whitener_apply_count,
                    evidence.observed_whitener_total_operations,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return learning.learning_result_digest

    def load_variational_learning_approval(
        self,
        learning_result_digest: str,
    ) -> LearningApprovalEvidence:
        """Load and verify one immutable P1 learning approval record."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM variational_learning_approvals
                WHERE learning_result_digest = ?
                """,
                (learning_result_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown learning approval: {learning_result_digest}")
        evidence = LearningApprovalEvidence(
            policy_digest=row["policy_digest"],
            trust_store_digest=row["trust_store_digest"],
            fsoi_digest=row["fsoi_digest"],
            full_step_analysis_digest=row["full_step_analysis_digest"],
            half_step_analysis_digest=row["half_step_analysis_digest"],
            full_step_forecast_digest=row["full_step_forecast_digest"],
            half_step_forecast_digest=row["half_step_forecast_digest"],
            first_order_validation_digest=(
                row["first_order_validation_digest"]
            ),
            learning_impact_digest=row["learning_impact_digest"],
            approved_action_digest=row["approved_action_digest"],
            nominal_input_bundle_digest=row["nominal_input_bundle_digest"],
            nominal_full_analysis_input_digest=(
                row["nominal_full_analysis_input_digest"]
            ),
            selection_mode=row["selection_mode"],
            candidate_id=row["candidate_id"],
            candidate_rank=row["candidate_rank"],
            candidate_score=row["candidate_score"],
            candidate_perturbation_digest=(
                row["candidate_perturbation_digest"]
            ),
            ranking_digest=row["ranking_digest"],
            ranking_policy_digest=row["ranking_policy_digest"],
            ranking_objective=row["ranking_objective"],
            whitener_operations_per_apply=(
                row["whitener_operations_per_apply"]
            ),
            observed_whitener_apply_count=(
                row["observed_whitener_apply_count"]
            ),
            observed_whitener_total_operations=(
                row["observed_whitener_total_operations"]
            ),
            contract=row["evidence_contract"],
        )
        if evidence.digest != row["approval_evidence_digest"]:
            raise ValueError("learning approval evidence digest mismatch")
        return evidence

    def append_realized_observation_intervention(
        self,
        intervention: RealizedObservationIntervention,
    ) -> str:
        """Append one realized action linked to approved learning evidence."""

        if not isinstance(intervention, RealizedObservationIntervention):
            raise TypeError("intervention must be realized evidence")
        with self._connect() as connection:
            approved = connection.execute(
                """
                SELECT approval_evidence_digest
                FROM variational_learning_approvals
                WHERE learning_result_digest = ?
                """,
                (intervention.learning_result_digest,),
            ).fetchone()
            if approved is None:
                raise ValueError("intervention learning result is not recorded")
            if approved[0] != intervention.learning_approval_evidence_digest:
                raise ValueError("intervention learning approval mismatch")
            connection.execute(
                """
                INSERT INTO realized_observation_interventions (
                    intervention_digest, intervention_id, intervention_type,
                    action_digest, applied_time,
                    actual_input_before_digest, actual_input_after_digest,
                    observed_outcome_digest, execution_policy_digest,
                    execution_trust_store_digest,
                    predicted_normalized_benefit,
                    resolved_normalized_benefit, learning_result_digest,
                    learning_approval_evidence_digest,
                    counterfactual_perturbation_digest,
                    linearization_digest, case_id, radar_id, issue_time,
                    input_bundle_before_digest, input_bundle_after_digest,
                    resolved_issuance_validation_digest,
                    evidence_contract, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intervention.intervention_digest,
                    intervention.intervention_id,
                    intervention.intervention_type,
                    intervention.action_digest,
                    intervention.applied_time,
                    intervention.actual_input_before_digest,
                    intervention.actual_input_after_digest,
                    intervention.outcome_resolution_contract_digest,
                    intervention.execution_policy_digest,
                    intervention.execution_trust_store_digest,
                    intervention.predicted_normalized_benefit,
                    intervention.resolved_normalized_benefit,
                    intervention.learning_result_digest,
                    intervention.learning_approval_evidence_digest,
                    intervention.counterfactual_perturbation_digest,
                    intervention.linearization_digest,
                    intervention.case_id,
                    intervention.radar_id,
                    intervention.issue_time,
                    intervention.input_bundle_before_digest,
                    intervention.input_bundle_after_digest,
                    intervention.resolved_issuance_validation_digest,
                    intervention.contract,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return intervention.intervention_digest

    def append_retrospective_counterfactual_replay(
        self,
        replay: RetrospectiveCounterfactualReplay,
    ) -> str:
        """Retain a replay for audit without making it promotion evidence."""

        validate_retrospective_counterfactual_replay(replay)
        payload = asdict(replay)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retrospective_counterfactual_replays "
                "(replay_digest, replay_json, created_at) VALUES (?, ?, ?)",
                (
                    replay.replay_digest,
                    json.dumps(payload, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return replay.replay_digest

    def load_retrospective_counterfactual_replay(
        self,
        replay_digest: str,
    ) -> RetrospectiveCounterfactualReplay:
        """Load and revalidate one historical replay audit."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT replay_json FROM retrospective_counterfactual_replays "
                "WHERE replay_digest = ?",
                (replay_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown retrospective replay: {replay_digest}")
        values = json.loads(row[0])
        if not isinstance(values, dict):
            raise ValueError("invalid retrospective replay payload")
        values.pop("replay_digest", None)
        replay = RetrospectiveCounterfactualReplay(**values)
        if replay.replay_digest != replay_digest:
            raise ValueError("retrospective replay ledger digest mismatch")
        validate_retrospective_counterfactual_replay(replay)
        return replay

    def _write_intervention_action_artifact(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        action_policy: ReusableInterventionPolicyEvidence,
        generator: InterventionActionGenerator,
        before: InterventionInputContext,
        before_run: ForecastRunContract,
        after: InterventionInputContext,
        after_run: ForecastRunContract,
    ) -> None:
        target = self.interventions_dir / receipt.receipt_digest
        if target.exists():
            self._replay_intervention_action_artifact(
                decision, receipt, action_policy
            )
            return
        generator_bytes = generator.artifact_bytes
        if len(generator_bytes) > _MAXIMUM_ACTION_GENERATOR_BYTES:
            raise ValueError("action generator artifact exceeds its byte budget")
        expanded_bytes = expanded_tensor_bytes((
            before.frames_dbz,
            before.observation_masks,
            before.quality_weight,
            before.observation_std_dbz,
            before.applicability_mask,
            after.frames_dbz,
            after.observation_masks,
            after.quality_weight,
            after.observation_std_dbz,
            after.applicability_mask,
            before.background_frames_dbz,
            after.background_frames_dbz,
        ))
        if expanded_bytes > _MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES:
            raise ValueError("action transition artifact exceeds its expanded-byte budget")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{receipt.receipt_digest}.",
                dir=self.interventions_dir,
            )
        )
        try:
            before_background = before.background_frames_dbz
            after_background = after.background_frames_dbz
            np.savez_compressed(
                temporary / "transition.npz",
                before_frames=before.frames_dbz.detach().cpu().numpy(),
                before_masks=before.observation_masks.detach().cpu().numpy(),
                before_quality=before.quality_weight.detach().cpu().numpy(),
                before_std=before.observation_std_dbz.detach().cpu().numpy(),
                before_background=(
                    np.empty((0,), dtype=np.float32)
                    if before_background is None
                    else before_background.detach().cpu().numpy()
                ),
                before_applicability=before.applicability_mask.detach().cpu().numpy(),
                after_frames=after.frames_dbz.detach().cpu().numpy(),
                after_masks=after.observation_masks.detach().cpu().numpy(),
                after_quality=after.quality_weight.detach().cpu().numpy(),
                after_std=after.observation_std_dbz.detach().cpu().numpy(),
                after_background=(
                    np.empty((0,), dtype=np.float32)
                    if after_background is None
                    else after_background.detach().cpu().numpy()
                ),
                after_applicability=after.applicability_mask.detach().cpu().numpy(),
            )
            (temporary / "generator.pt2").write_bytes(generator_bytes)
            manifest: dict[str, object] = {
                "contract": "durable-intervention-action-artifact-v5",
                "receipt_digest": receipt.receipt_digest,
                "action_artifact_digest": receipt.action_artifact_digest,
                "action_payload_digest": receipt.action_payload_digest,
                "generator_digest": generator.generator_digest,
                "generator_shape": list(generator._shape),
                "generator_dtype": str(generator._dtype),
                "intervention_type": generator.intervention_type,
                "action_reason": generator.action_reason,
                "action_policy_digest": action_policy.policy_digest,
                "action_policy_json": asdict(action_policy),
                "action_safety_diagnostics_digest": (
                    receipt.action_safety_diagnostics_digest
                ),
                "before_context_digest": before.context_digest,
                "after_context_digest": after.context_digest,
                "before_radar_id": before.radar_id,
                "after_radar_id": after.radar_id,
                "before_context_schema_digest": before.context_schema_digest,
                "after_context_schema_digest": after.context_schema_digest,
                "before_applicability_region_digest": (
                    before.applicability_region_digest
                ),
                "after_applicability_region_digest": (
                    after.applicability_region_digest
                ),
                "before_applicability_mask_digest": tensor_digest(
                    before.applicability_mask
                ),
                "after_applicability_mask_digest": tensor_digest(
                    after.applicability_mask
                ),
                "before_canonicalization_contract_digest": (
                    before.canonicalization_contract_digest
                ),
                "after_canonicalization_contract_digest": (
                    after.canonicalization_contract_digest
                ),
                "minimum_dbz": before.min_dbz,
                "maximum_dbz": before.max_dbz,
                "missing_fill_dbz": before.missing_fill_dbz,
                "cell_area_m2": before_run.grid_time_contract.cell_area_m2
                if before_run.grid_time_contract is not None
                else None,
                "analysis_config_json": before_run.analysis_config_json,
                "analysis_config_digest": before_run.analysis_config_digest,
                "before_input_bundle_digest": before.input_bundle_digest,
                "after_input_bundle_digest": after.input_bundle_digest,
                "before_fixed_input_context_digest": (
                    before_run.fixed_input_context_digest
                ),
                "after_fixed_input_context_digest": (
                    after_run.fixed_input_context_digest
                ),
                "before_full_analysis_input_digest": (
                    before_run.full_analysis_input_digest
                ),
                "after_full_analysis_input_digest": (
                    after_run.full_analysis_input_digest
                ),
                "before_analysis_input_identity_digest": (
                    before.analysis_input_identity_digest
                ),
                "after_analysis_input_identity_digest": (
                    after.analysis_input_identity_digest
                ),
                "input_plan_digest": before.input_plan_digest,
                "before_input_plan_resolution_digest": (
                    before.input_plan_resolution_digest
                ),
                "after_input_plan_resolution_digest": (
                    after.input_plan_resolution_digest
                ),
                "before_background_present": before_background is not None,
                "after_background_present": after_background is not None,
            }
            for prefix, run in (("before", before_run), ("after", after_run)):
                manifest[f"{prefix}_background_age_minutes"] = (
                    run.background_age_minutes
                )
                manifest[f"{prefix}_grid_time_contract_digest"] = (
                    run.grid_time_contract_digest
                )
                manifest[f"{prefix}_calibration_manifest_digest"] = (
                    run.operational_calibration_manifest_digest
                )
                manifest[f"{prefix}_calibration_approval_digest"] = (
                    run.operational_calibration_approval_digest
                )
                manifest[f"{prefix}_data_identity_digest"] = (
                    run.operational_data_identity_digest
                )
                manifest[f"{prefix}_source_available_mask_digest"] = (
                    run.source_available_mask_digest
                )
                manifest[f"{prefix}_learned_model_input_features_digest"] = (
                    run.learned_model_input_features_digest
                )
            (temporary / "manifest.json").write_text(
                _json_text(manifest),
                encoding="utf-8",
            )
            checksums = {
                name: _file_digest(temporary / name)
                for name in ("manifest.json", "generator.pt2", "transition.npz")
            }
            (temporary / "checksums.json").write_text(
                _json_text(checksums),
                encoding="utf-8",
            )
            if any(
                (temporary / name).stat().st_size
                > _MAXIMUM_ACTION_ARTIFACT_FILE_BYTES
                for name in (*checksums, "checksums.json")
            ):
                raise ValueError("action artifact member exceeds its file-byte budget")
            for name in (*checksums, "checksums.json"):
                _fsync_file(temporary / name)
            _fsync_directory(temporary)
            os.rename(temporary, target)
            _fsync_directory(self.interventions_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _replay_intervention_action_artifact(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        action_policy: ReusableInterventionPolicyEvidence,
    ) -> None:
        source = self.interventions_dir / receipt.receipt_digest
        if not source.is_dir() or source.is_symlink():
            raise ValueError("durable intervention artifact is missing")
        validate_artifact_directory(
            source,
            expected_members=frozenset(
                {
                    "checksums.json",
                    "manifest.json",
                    "generator.pt2",
                    "transition.npz",
                }
            ),
            maximum_members=_MAXIMUM_ACTION_ARTIFACT_MEMBERS,
            maximum_file_bytes=_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES,
        )
        checksum_path = source / "checksums.json"
        checksums = json.loads(checksum_path.read_text("utf-8"))
        expected_members = {"manifest.json", "generator.pt2", "transition.npz"}
        if not isinstance(checksums, dict) or set(checksums) != expected_members:
            raise ValueError("durable intervention artifact members are invalid")
        if any(
            not (source / name).is_file()
            or (source / name).is_symlink()
            or (source / name).stat().st_size
            > _MAXIMUM_ACTION_ARTIFACT_FILE_BYTES
            or _file_digest(source / name) != digest
            for name, digest in checksums.items()
        ):
            raise ValueError("durable intervention artifact checksum mismatch")
        manifest = json.loads((source / "manifest.json").read_text("utf-8"))
        if not isinstance(manifest, dict) or (
            manifest.get("contract")
            not in {
                "durable-intervention-action-artifact-v3",
                "durable-intervention-action-artifact-v4",
                "durable-intervention-action-artifact-v5",
            }
            or manifest.get("receipt_digest") != receipt.receipt_digest
            or manifest.get("action_artifact_digest")
            != receipt.action_artifact_digest
            or manifest.get("before_input_bundle_digest")
            != receipt.actual_input_before_bundle_digest
            or manifest.get("after_input_bundle_digest")
            != receipt.actual_input_bundle_digest
            or manifest.get("action_policy_digest") != action_policy.policy_digest
            or manifest.get("action_safety_diagnostics_digest")
            != receipt.action_safety_diagnostics_digest
            or manifest.get("before_context_digest")
            != decision.intervention_input_context_digest
            or manifest.get("before_fixed_input_context_digest")
            != decision.actual_input_before_fixed_context_digest
            or manifest.get("before_full_analysis_input_digest")
            != decision.actual_input_before_full_analysis_input_digest
            or manifest.get("before_applicability_mask_digest")
            != decision.applicability_mask_digest
        ):
            raise ValueError("durable intervention artifact manifest mismatch")
        retained_policy_json = manifest.get("action_policy_json")
        if not isinstance(retained_policy_json, dict):
            raise ValueError("durable intervention policy is invalid")
        policy_payload = dict(cast(dict[str, object], retained_policy_json))
        policy_payload.pop("policy_digest", None)
        for name in ("allowed_intervention_types", "validation_evidence_digests"):
            policy_payload[name] = tuple(cast(list[object], policy_payload[name]))
        retained_policy = ReusableInterventionPolicyEvidence(
            **cast(Any, policy_payload)
        )
        if retained_policy.policy_digest != action_policy.policy_digest:
            raise ValueError("durable intervention policy changed")
        transition_path = source / "transition.npz"
        expected_arrays = frozenset(
            {
                "before_frames",
                "before_masks",
                "before_quality",
                "before_std",
                "before_background",
                "before_applicability",
                "after_frames",
                "after_masks",
                "after_quality",
                "after_std",
                "after_background",
                "after_applicability",
            }
        )
        preflight_npz_archive(
            transition_path,
            expected_members=expected_arrays,
            maximum_members=_MAXIMUM_ACTION_ARTIFACT_MEMBERS,
            maximum_expanded_bytes=_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES,
        )
        with np.load(transition_path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_arrays:
                raise ValueError("durable intervention tensor members are invalid")
            tensors = {
                name: torch.from_numpy(arrays[name].copy())
                for name in arrays.files
            }
        before_background = (
            tensors["before_background"]
            if bool(manifest["before_background_present"])
            else None
        )
        after_background = (
            tensors["after_background"]
            if bool(manifest["after_background_present"])
            else None
        )
        before_applicability_digest = tensor_digest(
            tensors["before_applicability"]
        )
        after_applicability_digest = tensor_digest(
            tensors["after_applicability"]
        )
        if (
            before_applicability_digest
            != manifest.get("before_applicability_mask_digest")
            or after_applicability_digest
            != manifest.get("after_applicability_mask_digest")
            or before_applicability_digest != after_applicability_digest
        ):
            raise ValueError("durable intervention applicability changed")
        for prefix, background in (
            ("before", before_background),
            ("after", after_background),
        ):
            mask_digest = tensor_digest(tensors[f"{prefix}_masks"])
            quality_digest = tensor_digest(tensors[f"{prefix}_quality"])
            std_digest = tensor_digest(tensors[f"{prefix}_std"])
            background_digest = (
                None if background is None else tensor_digest(background)
            )
            expected_bundle = _forecast_input_bundle_digest(
                tensors[f"{prefix}_frames"],
                tensors[f"{prefix}_masks"],
                background,
                cast(float | None, manifest[f"{prefix}_background_age_minutes"]),
                None,
                cast(
                    str | None,
                    manifest[f"{prefix}_calibration_manifest_digest"],
                ),
                cast(
                    str | None,
                    manifest[f"{prefix}_calibration_approval_digest"],
                ),
                cast(str | None, manifest[f"{prefix}_data_identity_digest"]),
                grid_time_contract_digest=cast(
                    str | None,
                    manifest[f"{prefix}_grid_time_contract_digest"],
                ),
            )
            expected_fixed = (
                _forecast_fixed_input_context_digest(
                    observation_masks_digest=mask_digest,
                    observation_quality_weight_digest=quality_digest,
                    observation_std_dbz_digest=std_digest,
                    source_available_mask_digest=cast(
                        str,
                        manifest[f"{prefix}_source_available_mask_digest"],
                    ),
                    learned_model_input_features_digest=cast(
                        str,
                        manifest[
                            f"{prefix}_learned_model_input_features_digest"
                        ],
                    ),
                    background_frames_digest=background_digest,
                    background_age_minutes=cast(
                        float | None,
                        manifest[f"{prefix}_background_age_minutes"],
                    ),
                    grid_time_contract_digest=cast(
                        str | None,
                        manifest[f"{prefix}_grid_time_contract_digest"],
                    ),
                    operational_calibration_manifest_digest=cast(
                        str | None,
                        manifest[f"{prefix}_calibration_manifest_digest"],
                    ),
                    operational_calibration_approval_digest=cast(
                        str | None,
                        manifest[f"{prefix}_calibration_approval_digest"],
                    ),
                    operational_data_identity_digest=cast(
                        str | None,
                        manifest[f"{prefix}_data_identity_digest"],
                    ),
                    input_plan_digest=cast(str, manifest["input_plan_digest"]),
                )
                if manifest["contract"]
                == "durable-intervention-action-artifact-v5"
                else _json_digest(
                    {
                        "contract": "forecast-fixed-input-context-v1",
                        "observation_masks_digest": mask_digest,
                        "observation_quality_weight_digest": quality_digest,
                        "observation_std_dbz_digest": std_digest,
                        "background_frames_digest": background_digest,
                        "background_age_minutes": manifest[
                            f"{prefix}_background_age_minutes"
                        ],
                        "grid_time_contract_digest": manifest[
                            f"{prefix}_grid_time_contract_digest"
                        ],
                        "operational_calibration_manifest_digest": manifest[
                            f"{prefix}_calibration_manifest_digest"
                        ],
                        "operational_calibration_approval_digest": manifest[
                            f"{prefix}_calibration_approval_digest"
                        ],
                        "operational_data_identity_digest": manifest[
                            f"{prefix}_data_identity_digest"
                        ],
                        "input_plan_digest": manifest["input_plan_digest"],
                    }
                )
            )
            expected_full = _forecast_full_analysis_input_digest(
                input_frames_digest=tensor_digest(tensors[f"{prefix}_frames"]),
                fixed_input_context_digest=expected_fixed,
            )
            if manifest["contract"] in {
                "durable-intervention-action-artifact-v4",
                "durable-intervention-action-artifact-v5",
            }:
                expected_resolution = _forecast_input_plan_resolution_digest(
                    input_plan_digest=cast(str, manifest["input_plan_digest"]),
                    full_analysis_input_digest=expected_full,
                )
            else:
                expected_resolution = _json_digest(
                    {
                        "contract": "forecast-input-plan-resolution-v1",
                        "input_plan_digest": manifest["input_plan_digest"],
                        "input_bundle_digest": expected_bundle,
                    }
                )
            if (
                expected_bundle != manifest[f"{prefix}_input_bundle_digest"]
                or expected_resolution
                != manifest[f"{prefix}_input_plan_resolution_digest"]
            ):
                raise ValueError("durable intervention input bundle changed")
            expected_context = _json_digest(
                {
                    "contract": (
                        "intervention-input-context-v4"
                        if manifest["contract"]
                        in {
                            "durable-intervention-action-artifact-v4",
                            "durable-intervention-action-artifact-v5",
                        }
                        else "intervention-input-context-v3"
                    ),
                    "input_bundle_digest": manifest[
                        f"{prefix}_input_bundle_digest"
                    ],
                    "input_plan_digest": manifest["input_plan_digest"],
                    "input_plan_resolution_digest": manifest[
                        f"{prefix}_input_plan_resolution_digest"
                    ],
                    **(
                        {
                            "analysis_input_identity_digest": manifest[
                                f"{prefix}_analysis_input_identity_digest"
                            ]
                        }
                        if manifest["contract"]
                        in {
                            "durable-intervention-action-artifact-v4",
                            "durable-intervention-action-artifact-v5",
                        }
                        else {}
                    ),
                    "frames_digest": tensor_digest(tensors[f"{prefix}_frames"]),
                    "observation_masks_digest": mask_digest,
                    "quality_weight_digest": quality_digest,
                    "observation_std_dbz_digest": std_digest,
                    "background_frames_digest": background_digest,
                    "radar_id": manifest[f"{prefix}_radar_id"],
                    "context_schema_digest": manifest[
                        f"{prefix}_context_schema_digest"
                    ],
                    "applicability_region_digest": manifest[
                        f"{prefix}_applicability_region_digest"
                    ],
                    "applicability_mask_digest": manifest[
                        f"{prefix}_applicability_mask_digest"
                    ],
                    "canonicalization_contract_digest": manifest[
                        f"{prefix}_canonicalization_contract_digest"
                    ],
                    "minimum_dbz": manifest["minimum_dbz"],
                    "maximum_dbz": manifest["maximum_dbz"],
                    "missing_fill_dbz": manifest["missing_fill_dbz"],
                }
            )
            if expected_context != manifest[f"{prefix}_context_digest"]:
                raise ValueError("durable intervention context digest mismatch")
            retained_fixed = getattr(
                receipt,
                f"fixed_input_context_{prefix}_digest",
            )
            if (
                expected_fixed != manifest[f"{prefix}_fixed_input_context_digest"]
                or expected_fixed != retained_fixed
            ):
                raise ValueError("durable fixed input context changed")
            retained_full = getattr(
                receipt,
                f"full_analysis_input_{prefix}_digest",
            )
            if (
                expected_full != manifest[f"{prefix}_full_analysis_input_digest"]
                or expected_full != retained_full
            ):
                raise ValueError("durable full analysis input changed")
            if manifest["contract"] in {
                "durable-intervention-action-artifact-v4",
                "durable-intervention-action-artifact-v5",
            }:
                expected_identity = _json_digest(
                    {
                        "frames_digest": tensor_digest(
                            tensors[f"{prefix}_frames"]
                        ),
                        "fixed_context_digest": expected_fixed,
                        "full_data_digest": expected_full,
                        "input_plan_digest": manifest["input_plan_digest"],
                        "plan_resolution_digest": expected_resolution,
                        "contract": "analysis-input-identity-v1",
                    }
                )
                if expected_identity != manifest[
                    f"{prefix}_analysis_input_identity_digest"
                ]:
                    raise ValueError("durable analysis input identity changed")
        context_tensor = _intervention_context_tensor(
            tensors["before_frames"],
            tensors["before_masks"],
            tensors["before_quality"],
            tensors["before_std"],
            before_background,
            tensors["before_applicability"],
            minimum_dbz=float(manifest["minimum_dbz"]),
            maximum_dbz=float(manifest["maximum_dbz"]),
            missing_fill_dbz=float(manifest["missing_fill_dbz"]),
        )
        generator = InterventionActionGenerator.from_artifact(
            artifact=(source / "generator.pt2").read_bytes(),
            shape=tuple(int(value) for value in manifest["generator_shape"]),
            dtype=_torch_dtype(str(manifest["generator_dtype"])),
            intervention_type=cast(Any, manifest["intervention_type"]),
            action_reason=cast(str | None, manifest["action_reason"]),
            generator_digest=str(manifest["generator_digest"]),
        )
        action = generator.replay(context_tensor)
        if generator.intervention_type != receipt.intervention_type:
            raise ValueError("durable action type disagrees with its receipt")
        if action.payload_digest != receipt.action_payload_digest:
            raise ValueError("durable action payload changed")
        analysis_config_json = manifest.get("analysis_config_json")
        analysis_config_digest = manifest.get("analysis_config_digest")
        if (analysis_config_json is None) != (analysis_config_digest is None):
            raise ValueError("durable action analysis config is incomplete")
        if analysis_config_json is not None:
            if not isinstance(analysis_config_json, str) or not isinstance(
                analysis_config_digest, str
            ):
                raise ValueError("durable action analysis config is invalid")
            try:
                analysis_config = json.loads(analysis_config_json)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "durable action analysis config is invalid"
                ) from error
            if (
                not isinstance(analysis_config, dict)
                or _json_digest(analysis_config) != analysis_config_digest
                or float(
                    analysis_config.get("observation_common_bias_std_dbz", 0.0)
                )
                > 0.0
            ):
                raise ValueError(
                    "durable prospective action used correlated observation error"
                )
        diagnostics = _compute_action_safety(
            action,
            _artifact_intervention_context(manifest, tensors, before_background),
            retained_policy,
            minimum_dbz=float(manifest["minimum_dbz"]),
            maximum_dbz=float(manifest["maximum_dbz"]),
            cell_area_m2=float(manifest["cell_area_m2"]),
        )
        if diagnostics.diagnostics_digest != receipt.action_safety_diagnostics_digest:
            raise ValueError("durable action safety diagnostics changed")
        validate_action_tensor_replay(
            action,
            before_frames=tensors["before_frames"],
            before_masks=tensors["before_masks"],
            before_quality_weight=tensors["before_quality"],
            after_frames=tensors["after_frames"],
            after_masks=tensors["after_masks"],
            after_quality_weight=tensors["after_quality"],
        )
        retained_tensor_digests = (
            (tensor_digest(tensors["before_frames"]), receipt.actual_input_before_frames_digest),
            (tensor_digest(tensors["after_frames"]), receipt.actual_input_after_frames_digest),
            (tensor_digest(tensors["before_masks"]), receipt.actual_input_before_masks_digest),
            (tensor_digest(tensors["after_masks"]), receipt.actual_input_after_masks_digest),
            (
                tensor_digest(tensors["before_quality"]),
                receipt.actual_quality_weight_before_digest,
            ),
            (
                tensor_digest(tensors["after_quality"]),
                receipt.actual_quality_weight_after_digest,
            ),
        )
        if any(actual != expected for actual, expected in retained_tensor_digests):
            raise ValueError("durable intervention tensors disagree with receipt")
        replay_digest = _json_digest(
            {
                "contract": "intervention-action-artifact-v1",
                "generator_digest": generator.generator_digest,
                "before_context_digest": manifest["before_context_digest"],
                "after_context_digest": manifest["after_context_digest"],
                "action_payload_digest": action.payload_digest,
                "before_frames_digest": tensor_digest(tensors["before_frames"]),
                "after_frames_digest": tensor_digest(tensors["after_frames"]),
                "before_masks_digest": tensor_digest(tensors["before_masks"]),
                "after_masks_digest": tensor_digest(tensors["after_masks"]),
                "before_quality_weight_digest": tensor_digest(
                    tensors["before_quality"]
                ),
                "after_quality_weight_digest": tensor_digest(
                    tensors["after_quality"]
                ),
            }
        )
        if replay_digest != receipt.action_artifact_digest:
            raise ValueError("durable action transition digest mismatch")

    def append_prospective_intervention_decision(
        self,
        decision: ProspectiveInterventionDecision,
        *,
        operator_approval: OperatorActionApproval,
        action_policy: ReusableInterventionPolicyEvidence,
        action_generator: InterventionActionGenerator,
        actual_input_before_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        trust_store_path: str | Path,
        operator_trust_store_path: str | Path,
    ) -> str:
        """Commit a policy-generated current-input action before its deadline."""

        action_policy.validate_integrity()
        reproduced = ProspectiveInterventionDecision.from_policy(
            action_policy,
            action_generator=action_generator,
            decision_id=decision.decision_id,
            case_id=decision.case_id,
            radar_id=decision.radar_id,
            intervention_type=decision.intervention_type,
            actual_input_context=actual_input_before_context,
            actual_input_before_run=actual_input_before_run,
            input_plan_digest=decision.input_plan_digest,
            decision_basis_digest=decision.decision_basis_digest,
            decision_policy_digest=decision.decision_policy_digest,
            decision_trust_store_digest=decision.decision_trust_store_digest,
            decided_at=decision.decided_at,
            observation_valid_time=decision.observation_valid_time,
            input_available_time=decision.input_available_time,
            decision_deadline=decision.decision_deadline,
            publication_time=decision.publication_time,
        )
        if reproduced.decision_digest != decision.decision_digest:
            raise ValueError("prospective decision is not the policy output")
        trust = _load_learning_policy_trust_store(trust_store_path)
        if trust.content_digest != decision.decision_trust_store_digest:
            raise ValueError("prospective decision trust-store mismatch")
        if decision.decision_policy_digest not in trust.approved_policy_digests:
            raise ValueError("prospective decision policy is not approved")
        if action_policy.policy_digest not in trust.approved_policy_digests:
            raise ValueError("reusable intervention policy is not approved")
        if (
            decision.action_policy_digest != action_policy.policy_digest
            or decision.action_generator_digest
            != action_policy.action_generator_digest
            or decision.decision_policy_digest
            != action_policy.execution_policy_digest
            or decision.decision_basis_digest
            not in action_policy.validation_evidence_digests
        ):
            raise ValueError("prospective decision disagrees with its action policy")
        decided = datetime.fromisoformat(decision.decided_at.replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(
            decision.decision_deadline.replace("Z", "+00:00")
        )
        publication = datetime.fromisoformat(
            decision.publication_time.replace("Z", "+00:00")
        )
        expected = _json_digest(
            {
                key: value
                for key, value in decision.__dict__.items()
                if key != "decision_digest"
            }
        )
        if expected != decision.decision_digest:
            raise ValueError("prospective decision digest mismatch")
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        if operator_approval.operator_trust_store_digest != (
            operator_trust.content_digest
        ):
            raise ValueError("operator approval trust-store mismatch")
        public_key = operator_trust.keys.get(operator_approval.operator_key_id)
        approved_roles = operator_trust.roles.get(
            operator_approval.operator_key_id,
            frozenset(),
        )
        if public_key is None or operator_approval.operator_role not in approved_roles:
            raise ValueError("operator is not approved for the declared role")
        if (
            operator_approval.decision_digest != decision.decision_digest
            or operator_approval.action_digest != decision.action_digest
            or operator_approval.full_analysis_input_digest
            != decision.actual_input_before_full_analysis_input_digest
            or operator_approval.safety_diagnostics_digest
            != decision.action_safety_diagnostics_digest
        ):
            raise ValueError("operator approval is bound to another decision")
        expected_approval_digest = _json_digest(
            {
                key: value
                for key, value in operator_approval.__dict__.items()
                if key != "approval_digest"
            }
        )
        if operator_approval.approval_digest != expected_approval_digest:
            raise ValueError("operator action approval digest mismatch")
        verify_operator_action_approval_signature(operator_approval, public_key)
        reviewed = datetime.fromisoformat(
            operator_approval.reviewed_at.replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            operator_approval.expires_at.replace("Z", "+00:00")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            if (
                decided > reviewed
                or reviewed > now
                or now >= expires
                or expires > deadline
                or deadline >= publication
            ):
                raise ValueError("prospective decision missed its decision deadline")
            retained = connection.execute(
                "SELECT policy_json FROM reusable_intervention_policies "
                "WHERE policy_digest = ?",
                (action_policy.policy_digest,),
            ).fetchone()
            policy_json = json.dumps(asdict(action_policy), sort_keys=True)
            if retained is not None and retained[0] != policy_json:
                raise ValueError("recorded reusable intervention policy changed")
            if retained is None:
                connection.execute(
                    "INSERT INTO reusable_intervention_policies "
                    "(policy_digest, policy_json, created_at) VALUES (?, ?, ?)",
                    (action_policy.policy_digest, policy_json, now.isoformat()),
                )
            connection.execute(
                "INSERT INTO prospective_intervention_decisions "
                "(decision_digest, decision_id, decision_json, "
                "operator_approval_digest, operator_approval_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_digest,
                    decision.decision_id,
                    json.dumps(asdict(decision), sort_keys=True),
                    operator_approval.approval_digest,
                    json.dumps(asdict(operator_approval), sort_keys=True),
                    now.isoformat(),
                ),
            )
            if datetime.now(timezone.utc) > deadline:
                raise ValueError("prospective decision crossed its deadline")
        return decision.decision_digest

    def append_realized_intervention_receipt(
        self,
        decision: ProspectiveInterventionDecision,
        receipt: RealizedInterventionReceipt,
        *,
        action_policy: ReusableInterventionPolicyEvidence,
        action_generator: InterventionActionGenerator,
        actual_input_before_context: InterventionInputContext,
        actual_input_before_run: ForecastRunContract,
        actual_input_after_context: InterventionInputContext,
        actual_input_after_run: ForecastRunContract,
        trust_store_path: str | Path,
        executor_trust_store_path: str | Path,
        operator_trust_store_path: str | Path,
    ) -> str:
        """Record the executor receipt while the issue is still prospective."""

        validate_prospective_intervention(decision, receipt)
        transition = validate_intervention_action_transition(
            decision,
            action_policy=action_policy,
            action_generator=action_generator,
            actual_input_before_context=actual_input_before_context,
            actual_input_before_run=actual_input_before_run,
            actual_input_after_context=actual_input_after_context,
            actual_input_after_run=actual_input_after_run,
        )
        if (
            receipt.actual_input_before_frames_digest
            != transition.before_frames_digest
            or receipt.actual_input_after_frames_digest
            != transition.after_frames_digest
            or receipt.actual_input_before_bundle_digest
            != actual_input_before_run.input_bundle_digest
            or receipt.actual_input_bundle_digest
            != actual_input_after_run.input_bundle_digest
            or receipt.full_analysis_input_before_digest
            != actual_input_before_run.full_analysis_input_digest
            or receipt.full_analysis_input_after_digest
            != actual_input_after_run.full_analysis_input_digest
        ):
            raise ValueError("realized receipt transition evidence disagrees")
        trust = _load_learning_policy_trust_store(trust_store_path)
        if trust.content_digest != decision.decision_trust_store_digest:
            raise ValueError("realized receipt trust-store mismatch")
        executor_trust = _load_executor_trust_store(executor_trust_store_path)
        if receipt.executor_trust_store_digest != executor_trust.content_digest:
            raise ValueError("realized receipt executor trust-store mismatch")
        public_key = executor_trust.keys.get(receipt.executor_key_id)
        if public_key is None:
            raise ValueError("realized receipt executor is not approved")
        verify_intervention_receipt_signature(receipt, public_key)
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        publication = datetime.fromisoformat(
            receipt.publication_time.replace("Z", "+00:00")
        )
        received = datetime.fromisoformat(receipt.receipt_time.replace("Z", "+00:00"))
        applied = datetime.fromisoformat(receipt.applied_time.replace("Z", "+00:00"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            recorded = connection.execute(
                "SELECT decision_json, operator_approval_digest, "
                "operator_approval_json, created_at "
                "FROM prospective_intervention_decisions "
                "WHERE decision_digest = ?",
                (decision.decision_digest,),
            ).fetchone()
            if recorded is None:
                raise ValueError("realized receipt decision is not recorded")
            if json.loads(recorded[0]) != asdict(decision):
                raise ValueError("recorded prospective decision changed")
            if not recorded[1] or not recorded[2]:
                raise ValueError("realized receipt requires operator approval")
            approval_values = json.loads(recorded[2])
            retained_approval_digest = approval_values.pop(
                "approval_digest",
                None,
            )
            approval = _new_operator_action_approval(**approval_values)
            if (
                approval.approval_digest != retained_approval_digest
                or approval.approval_digest != recorded[1]
                or approval.decision_digest != decision.decision_digest
                or approval.operator_trust_store_digest
                != operator_trust.content_digest
            ):
                raise ValueError("recorded operator approval changed")
            operator_key = operator_trust.keys.get(approval.operator_key_id)
            if operator_key is None or approval.operator_role not in (
                operator_trust.roles.get(approval.operator_key_id, frozenset())
            ):
                raise ValueError("recorded operator is not approved")
            verify_operator_action_approval_signature(approval, operator_key)
            decision_created = datetime.fromisoformat(recorded[3])
            approval_expiry = datetime.fromisoformat(
                approval.expires_at.replace("Z", "+00:00")
            )
            if not (
                decision_created
                <= applied
                < approval_expiry
                and applied <= received <= now < publication
            ):
                raise ValueError("realized receipt violates trusted clock order")
            previous_sequence = connection.execute(
                "SELECT MAX(executor_sequence_number) "
                "FROM realized_intervention_receipts WHERE executor_key_id = ?",
                (receipt.executor_key_id,),
            ).fetchone()[0]
            if previous_sequence is not None and (
                receipt.executor_sequence_number <= previous_sequence
            ):
                raise ValueError("executor receipt sequence must increase")
            self._write_intervention_action_artifact(
                decision,
                receipt,
                action_policy,
                action_generator,
                actual_input_before_context,
                actual_input_before_run,
                actual_input_after_context,
                actual_input_after_run,
            )
            connection.execute(
                "INSERT INTO realized_intervention_receipts "
                "(receipt_digest, decision_digest, executor_key_id, "
                "executor_sequence_number, receipt_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_digest,
                    receipt.decision_digest,
                    receipt.executor_key_id,
                    receipt.executor_sequence_number,
                    json.dumps(asdict(receipt), sort_keys=True),
                    now.isoformat(),
                ),
            )
            if datetime.now(timezone.utc) >= publication:
                raise ValueError("realized receipt crossed publication time")
        return receipt.receipt_digest

    def load_prospective_intervention(
        self,
        receipt_digest: str,
        *,
        executor_trust_store_path: str | Path,
        operator_trust_store_path: str | Path | None = None,
    ) -> (
        tuple[
            ProspectiveInterventionDecision,
            RealizedInterventionReceipt,
            OperatorActionApproval,
        ]
        | tuple[
            LegacyProspectiveInterventionDecisionAudit,
            LegacyRealizedInterventionReceiptAudit,
        ]
    ):
        """Load and verify one causally ordered action and receipt."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT r.receipt_json, d.decision_json, "
                "d.operator_approval_digest, d.operator_approval_json "
                "FROM realized_intervention_receipts r "
                "JOIN prospective_intervention_decisions d "
                "ON d.decision_digest = r.decision_digest "
                "WHERE r.receipt_digest = ?",
                (receipt_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown realized receipt: {receipt_digest}")
        decision_values = json.loads(row["decision_json"])
        receipt_values = json.loads(row["receipt_json"])
        legacy_contracts = {
            (
                "prospective-intervention-decision-v2",
                "realized-intervention-receipt-v2",
            ),
            (
                "prospective-intervention-decision-v3",
                "realized-intervention-receipt-v3",
            ),
            (
                "prospective-intervention-decision-v4",
                "realized-intervention-receipt-v4",
            ),
        }
        approval_missing = not row["operator_approval_digest"] or not row[
            "operator_approval_json"
        ]
        if approval_missing or (
            decision_values.get("contract"),
            receipt_values.get("contract"),
        ) in legacy_contracts:
            decision_digest = decision_values.get("decision_digest")
            retained_receipt_digest = receipt_values.get("receipt_digest")
            if not isinstance(decision_digest, str) or (
                retained_receipt_digest != receipt_digest
                or receipt_values.get("decision_digest") != decision_digest
            ):
                raise ValueError("legacy prospective storage linkage is invalid")
            decision_audit = LegacyProspectiveInterventionDecisionAudit(
                decision_digest=decision_digest,
                payload_json=json.dumps(
                    decision_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            receipt_audit = LegacyRealizedInterventionReceiptAudit(
                receipt_digest=receipt_digest,
                decision_digest=decision_digest,
                payload_json=json.dumps(
                    receipt_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            executor_trust = _load_executor_trust_store(
                executor_trust_store_path
            )
            if receipt_values.get("executor_trust_store_digest") != (
                executor_trust.content_digest
            ):
                raise ValueError("legacy receipt executor trust-store mismatch")
            key_id = receipt_values.get("executor_key_id")
            public_key = executor_trust.keys.get(str(key_id))
            signature = receipt_values.get("executor_signature")
            if public_key is None or not isinstance(signature, str):
                raise ValueError("legacy receipt executor is not approved")
            signed_values = dict(receipt_values)
            signed_values.pop("receipt_digest")
            signed_values["executor_signature"] = ""
            try:
                public_key.verify(
                    bytes.fromhex(signature),
                    _json_digest(signed_values).encode("ascii"),
                )
            except (InvalidSignature, ValueError) as error:
                raise ValueError("legacy receipt executor signature mismatch") from error
            with self._connect() as connection:
                sequences = connection.execute(
                    "SELECT executor_sequence_number FROM "
                    "realized_intervention_receipts WHERE executor_key_id = ? "
                    "ORDER BY created_at, receipt_digest",
                    (key_id,),
                ).fetchall()
            sequence_values = [int(item[0]) for item in sequences]
            if sequence_values != sorted(set(sequence_values)):
                raise ValueError("legacy executor receipt sequence is invalid")
            return decision_audit, receipt_audit
        retained_decision_digest = decision_values.pop("decision_digest", None)
        retained_receipt_digest = receipt_values.pop("receipt_digest", None)
        decision = _new_prospective_decision(**decision_values)
        receipt = _new_realized_intervention_receipt(**receipt_values)
        if (
            decision.decision_digest != retained_decision_digest
            or receipt.receipt_digest != retained_receipt_digest
        ):
            raise ValueError("prospective intervention storage digest mismatch")
        validate_prospective_intervention(decision, receipt)
        executor_trust = _load_executor_trust_store(executor_trust_store_path)
        if receipt.executor_trust_store_digest != executor_trust.content_digest:
            raise ValueError("realized receipt executor trust-store mismatch")
        public_key = executor_trust.keys.get(receipt.executor_key_id)
        if public_key is None:
            raise ValueError("realized receipt executor is not approved")
        verify_intervention_receipt_signature(receipt, public_key)
        if operator_trust_store_path is None:
            raise ValueError("current prospective action requires operator trust store")
        try:
            approval_values = json.loads(row["operator_approval_json"])
        except json.JSONDecodeError as error:
            raise ValueError("stored operator approval is invalid") from error
        if not isinstance(approval_values, dict):
            raise ValueError("stored operator approval is invalid")
        retained_approval_digest = approval_values.pop("approval_digest", None)
        approval = _new_operator_action_approval(**approval_values)
        if (
            approval.approval_digest != retained_approval_digest
            or approval.approval_digest != row["operator_approval_digest"]
            or approval.decision_digest != decision.decision_digest
            or approval.action_digest != decision.action_digest
            or approval.full_analysis_input_digest
            != decision.actual_input_before_full_analysis_input_digest
            or approval.safety_diagnostics_digest
            != decision.action_safety_diagnostics_digest
        ):
            raise ValueError("stored operator approval changed")
        decided_at = datetime.fromisoformat(
            decision.decided_at.replace("Z", "+00:00")
        )
        reviewed_at = datetime.fromisoformat(
            approval.reviewed_at.replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            approval.expires_at.replace("Z", "+00:00")
        )
        deadline = datetime.fromisoformat(
            decision.decision_deadline.replace("Z", "+00:00")
        )
        applied_at = datetime.fromisoformat(
            receipt.applied_time.replace("Z", "+00:00")
        )
        if not (
            decided_at <= reviewed_at <= applied_at < expires_at <= deadline
        ):
            raise ValueError("stored operator approval time order is invalid")
        operator_trust = _load_operator_trust_store(operator_trust_store_path)
        if approval.operator_trust_store_digest != operator_trust.content_digest:
            raise ValueError("operator approval trust-store mismatch")
        operator_key = operator_trust.keys.get(approval.operator_key_id)
        if operator_key is None or approval.operator_role not in (
            operator_trust.roles.get(approval.operator_key_id, frozenset())
        ):
            raise ValueError("stored operator is not approved")
        verify_operator_action_approval_signature(approval, operator_key)
        with self._connect() as connection:
            sequences = connection.execute(
                "SELECT executor_sequence_number FROM "
                "realized_intervention_receipts WHERE executor_key_id = ? "
                "ORDER BY created_at, receipt_digest",
                (receipt.executor_key_id,),
            ).fetchall()
        values = [int(row[0]) for row in sequences]
        if values != sorted(set(values)):
            raise ValueError("executor receipt sequence ledger is invalid")
        with self._connect() as connection:
            policy_row = connection.execute(
                "SELECT policy_json FROM reusable_intervention_policies "
                "WHERE policy_digest = ?",
                (decision.action_policy_digest,),
            ).fetchone()
        if policy_row is None or not isinstance(policy_row[0], str):
            raise ValueError("prospective intervention policy is not retained")
        policy_values = json.loads(policy_row[0])
        if not isinstance(policy_values, dict):
            raise ValueError("prospective intervention policy is invalid")
        retained_policy_digest = policy_values.pop("policy_digest", None)
        policy_values["allowed_intervention_types"] = tuple(
            policy_values["allowed_intervention_types"]
        )
        policy_values["validation_evidence_digests"] = tuple(
            policy_values["validation_evidence_digests"]
        )
        action_policy = ReusableInterventionPolicyEvidence(
            **cast(Any, policy_values)
        )
        if (
            action_policy.policy_digest != retained_policy_digest
            or action_policy.policy_digest != decision.action_policy_digest
        ):
            raise ValueError("prospective intervention policy digest mismatch")
        self._replay_intervention_action_artifact(
            decision, receipt, action_policy
        )
        return decision, receipt, approval

    def load_realized_observation_intervention(
        self,
        intervention_digest: str,
    ) -> RealizedObservationIntervention:
        """Load and verify one immutable realized intervention."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM realized_observation_interventions
                WHERE intervention_digest = ?
                """,
                (intervention_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown realized intervention: {intervention_digest}")
        intervention = RealizedObservationIntervention(
            intervention_id=row["intervention_id"],
            intervention_type=row["intervention_type"],
            action_digest=row["action_digest"],
            applied_time=row["applied_time"],
            actual_input_before_digest=row["actual_input_before_digest"],
            actual_input_after_digest=row["actual_input_after_digest"],
            outcome_resolution_contract_digest=row["observed_outcome_digest"],
            execution_policy_digest=row["execution_policy_digest"],
            execution_trust_store_digest=row["execution_trust_store_digest"],
            predicted_normalized_benefit=row["predicted_normalized_benefit"],
            resolved_normalized_benefit=row["resolved_normalized_benefit"],
            learning_result_digest=row["learning_result_digest"],
            learning_approval_evidence_digest=(
                row["learning_approval_evidence_digest"]
            ),
            counterfactual_perturbation_digest=(
                row["counterfactual_perturbation_digest"]
            ),
            linearization_digest=row["linearization_digest"],
            case_id=row["case_id"],
            radar_id=row["radar_id"],
            issue_time=row["issue_time"],
            input_bundle_before_digest=row["input_bundle_before_digest"],
            input_bundle_after_digest=row["input_bundle_after_digest"],
            resolved_issuance_validation_digest=(
                row["resolved_issuance_validation_digest"]
            ),
            contract=row["evidence_contract"],
        )
        if intervention.intervention_digest != intervention_digest:
            raise ValueError("realized intervention digest mismatch")
        return intervention

    def append_neural_prior_holdout_plan(
        self,
        plan: NeuralPriorHoldoutPlan,
        *,
        promotion_decision_rule: PromotionDecisionRule,
        policy: NeuralPriorHoldoutPlanPolicy,
        policy_trust_store_path: str | Path,
        scheduler_trust_store_path: str | Path,
    ) -> str:
        """Pre-register one approved plan before its first forecast issue."""

        trust = _load_learning_policy_trust_store(policy_trust_store_path)
        scheduler_trust = _load_scheduler_trust_store(
            scheduler_trust_store_path
        )
        validate_neural_prior_holdout_plan(plan)
        validate_promotion_decision_rule(promotion_decision_rule)
        _validate_scheduler_authority(
            plan.physical_event_catalog_plan,
            scheduler_trust,
        )
        if policy.digest not in trust.approved_policy_digests:
            raise ValueError("holdout plan policy is not approved")
        if promotion_decision_rule.rule_digest not in trust.approved_policy_digests:
            raise ValueError("promotion decision rule is not root-approved")
        if (
            plan.promotion_experiment_family.family_digest
            not in trust.approved_policy_digests
        ):
            raise ValueError("promotion experiment family is not root-approved")
        if (
            plan.promotion_decision_rule_digest
            != promotion_decision_rule.rule_digest
        ):
            raise ValueError("holdout plan decision rule is inconsistent")
        if plan.plan_digest not in policy.approved_plan_digests:
            raise ValueError("holdout plan is not approved")
        if len(plan.candidate_family_digests) > policy.maximum_candidate_family_size:
            raise ValueError("holdout candidate family exceeds policy")
        sampling_reservation = (
            plan.promotion_experiment_family.global_sampling_reservation
        )
        classifier_training_receipts = tuple(
            _training_raw_registry_receipt_from_json(
                item.training_raw_registry_receipt_payload_json,
                expected_digest=item.training_raw_registry_receipt_digest,
            )
            for item in plan.regime_classifier_manifests
        )
        unique_training_receipts = {
            item.receipt_digest: item for item in classifier_training_receipts
        }
        if len(unique_training_receipts) != 1:
            raise ValueError(
                "one family-wide training raw registry receipt is required"
            )
        family_training_receipt = next(iter(unique_training_receipts.values()))
        if (
            sampling_reservation.registry_id != policy.sampling_registry_id
            or sampling_reservation.committed_registry_root_digest
            not in policy.approved_sampling_registry_root_digests
            or sampling_reservation.authority_id
            != policy.sampling_registry_authority_id
            or sampling_reservation.authority_public_key_hex
            != policy.sampling_registry_authority_public_key_hex
            or plan.raw_ingestor_trust_store.content_digest
            != policy.raw_ingestor_trust_store_digest
            or plan.training_target_source_trust_store.content_digest
            != policy.training_target_source_trust_store_digest
            or plan.analysis_processor_id != policy.analysis_processor_id
            or plan.analysis_processor_public_key_hex
            != policy.analysis_processor_public_key_hex
            or plan.training_target_source_authority_id
            != policy.training_target_source_authority_id
            or plan.training_target_source_authority_public_key_hex
            != policy.training_target_source_authority_public_key_hex
            or family_training_receipt.registry_id
            != sampling_reservation.registry_id
            or family_training_receipt.authority_id
            != sampling_reservation.authority_id
            or family_training_receipt.authority_public_key_hex
            != sampling_reservation.authority_public_key_hex
            or family_training_receipt.registry_sequence_number + 1
            != sampling_reservation.registry_sequence_number
            or family_training_receipt.committed_registry_root_digest
            != sampling_reservation.previous_registry_root_digest
        ):
            raise ValueError("global sampling reservation authority is not approved")
        approved_metrics = set(policy.approved_metric_contract_digests)
        if any(item.metric_contract_digest not in approved_metrics for item in plan.cases):
            raise ValueError("holdout metric contract is not approved")
        registered = datetime.fromisoformat(plan.registered_at.replace("Z", "+00:00"))
        issue_times = tuple(
            datetime.fromisoformat(item.issue_time.replace("Z", "+00:00"))
            for item in plan.cases
        )
        raw_slot_times = tuple(
            datetime.fromisoformat(
                item.acquisition_valid_time.replace("Z", "+00:00")
            )
            for item in plan.raw_observation_slot_plans
        )
        started = None
        if plan.mode == "sealed_historical":
            assert plan.candidate_training_started_at is not None
            started = datetime.fromisoformat(
                plan.candidate_training_started_at.replace("Z", "+00:00")
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            if registered > now:
                raise ValueError("holdout plan registration cannot be in the future")
            if plan.mode == "prospective" and any(
                now >= issue for issue in issue_times
            ):
                raise ValueError("holdout plan must be recorded before forecast issue")
            if plan.mode == "prospective" and (
                not raw_slot_times or now >= min(raw_slot_times)
            ):
                raise ValueError(
                    "holdout plan must be durably recorded before raw observation slots"
                )
            if started is not None and now >= started:
                raise ValueError(
                    "historical holdout must be recorded before candidate training"
                )
            connection.execute(
                """
                INSERT INTO neural_prior_holdout_plans (
                    plan_digest, plan_id, plan_json, policy_digest,
                    trust_store_digest, registered_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_digest,
                    plan.plan_id,
                    json.dumps(asdict(plan), sort_keys=True),
                    policy.digest,
                    trust.content_digest,
                    plan.registered_at,
                    now.isoformat(),
                ),
            )
            rule_json = json.dumps(
                promotion_decision_rule.payload
                | {"rule_digest": promotion_decision_rule.rule_digest},
                sort_keys=True,
            )
            connection.execute(
                "INSERT OR IGNORE INTO neural_prior_promotion_rule_definitions "
                "(rule_digest, payload_json, trust_store_digest, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    promotion_decision_rule.rule_digest,
                    rule_json,
                    trust.content_digest,
                    now.isoformat(),
                ),
            )
            retained_rule = connection.execute(
                "SELECT payload_json FROM neural_prior_promotion_rule_definitions "
                "WHERE rule_digest = ?",
                (promotion_decision_rule.rule_digest,),
            ).fetchone()
            if retained_rule is None or retained_rule[0] != rule_json:
                raise ValueError("registered promotion decision rule is inconsistent")
            connection.execute(
                "INSERT INTO neural_prior_holdout_plan_rule_bindings "
                "(holdout_plan_digest, rule_digest, bound_at) VALUES (?, ?, ?)",
                (
                    plan.plan_digest,
                    promotion_decision_rule.rule_digest,
                    now.isoformat(),
                ),
            )
            family = plan.promotion_experiment_family
            family_json = json.dumps(
                family.payload | {"family_digest": family.family_digest},
                sort_keys=True,
            )
            connection.execute(
                "INSERT OR IGNORE INTO neural_prior_promotion_experiment_families "
                "(family_digest, holdout_cohort_digest, payload_json, "
                "trust_store_digest, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    family.family_digest,
                    family.holdout_cohort_digest,
                    family_json,
                    trust.content_digest,
                    now.isoformat(),
                ),
            )
            retained_family = connection.execute(
                "SELECT family_digest, payload_json FROM "
                "neural_prior_promotion_experiment_families "
                "WHERE holdout_cohort_digest = ?",
                (family.holdout_cohort_digest,),
            ).fetchone()
            if retained_family is None or tuple(retained_family) != (
                family.family_digest,
                family_json,
            ):
                raise ValueError(
                    "holdout cohort is already bound to another experiment family"
                )
            for sampling_unit_digest in (
                family.meteorological_sampling_unit_digests
            ):
                connection.execute(
                    "INSERT OR IGNORE INTO promotion_sampling_unit_reservations "
                    "(sampling_unit_digest,family_digest,reserved_at) "
                    "VALUES (?,?,?)",
                    (
                        sampling_unit_digest,
                        family.family_digest,
                        now.isoformat(),
                    ),
                )
                retained_sampling_unit = connection.execute(
                    "SELECT family_digest FROM "
                    "promotion_sampling_unit_reservations "
                    "WHERE sampling_unit_digest = ?",
                    (sampling_unit_digest,),
                ).fetchone()
                if retained_sampling_unit != (family.family_digest,):
                    raise ValueError(
                        "meteorological sampling unit is reserved by another family"
                    )
            for raw_observation_slot_digest in (
                family.raw_observation_slot_digests
            ):
                connection.execute(
                    "INSERT OR IGNORE INTO promotion_raw_observation_slot_reservations "
                    "(raw_observation_slot_digest,family_digest,global_receipt_digest,"
                    "reserved_at) VALUES (?,?,?,?)",
                    (
                        raw_observation_slot_digest,
                        family.family_digest,
                        family.global_sampling_reservation.receipt_digest,
                        now.isoformat(),
                    ),
                )
                retained_raw_observation = connection.execute(
                    "SELECT family_digest,global_receipt_digest FROM "
                    "promotion_raw_observation_slot_reservations "
                    "WHERE raw_observation_slot_digest = ?",
                    (raw_observation_slot_digest,),
                ).fetchone()
                if retained_raw_observation != (
                    family.family_digest,
                    family.global_sampling_reservation.receipt_digest,
                ):
                    raise ValueError(
                        "raw observation slot is reserved by another family"
                    )
            for training_receipt in unique_training_receipts.values():
                receipt_json = json.dumps(
                    training_receipt.payload
                    | {"receipt_digest": training_receipt.receipt_digest},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO training_raw_registry_entries "
                    "(receipt_digest,registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,committed_registry_root_digest,"
                    "payload_json,created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        training_receipt.receipt_digest,
                        training_receipt.registry_id,
                        training_receipt.registry_sequence_number,
                        training_receipt.previous_registry_root_digest,
                        training_receipt.committed_registry_root_digest,
                        receipt_json,
                        now.isoformat(),
                    ),
                )
                retained_training = connection.execute(
                    "SELECT payload_json FROM training_raw_registry_entries "
                    "WHERE receipt_digest = ?",
                    (training_receipt.receipt_digest,),
                ).fetchone()
                if retained_training != (receipt_json,):
                    raise ValueError("training raw registry entry equivocated")
                EpisodeLedger._record_global_sampling_registry_entry(
                    connection,
                    registry_id=training_receipt.registry_id,
                    registry_sequence_number=(
                        training_receipt.registry_sequence_number
                    ),
                    previous_registry_root_digest=(
                        training_receipt.previous_registry_root_digest
                    ),
                    committed_registry_root_digest=(
                        training_receipt.committed_registry_root_digest
                    ),
                    receipt_digest=training_receipt.receipt_digest,
                    entry_kind="training_raw",
                    family_digest=family.family_digest,
                    created_at=now.isoformat(),
                )
            reservation_entry = connection.execute(
                "SELECT registry_sequence_number,previous_registry_root_digest,"
                "committed_registry_root_digest,receipt_digest FROM "
                "global_sampling_registry_entries WHERE registry_id = ? "
                "ORDER BY registry_sequence_number DESC LIMIT 1",
                (sampling_reservation.registry_id,),
            ).fetchone()
            if reservation_entry is None:
                raise ValueError("training raw registry entry was not committed")
            expected_sequence = int(reservation_entry[0]) + 1
            expected_previous = str(reservation_entry[2])
            if (
                sampling_reservation.registry_sequence_number
                != expected_sequence
                or sampling_reservation.previous_registry_root_digest
                != expected_previous
            ):
                raise ValueError(
                    "global sampling registry reservation is not contiguous"
                )
            EpisodeLedger._record_global_sampling_registry_entry(
                connection,
                registry_id=sampling_reservation.registry_id,
                registry_sequence_number=(
                    sampling_reservation.registry_sequence_number
                ),
                previous_registry_root_digest=(
                    sampling_reservation.previous_registry_root_digest
                ),
                committed_registry_root_digest=(
                    sampling_reservation.committed_registry_root_digest
                ),
                receipt_digest=sampling_reservation.receipt_digest,
                entry_kind="slot_reservation",
                family_digest=family.family_digest,
                created_at=now.isoformat(),
            )
            matching_trial = next(
                item
                for item in family.trials
                if item.candidate_prior_digest == plan.candidate_family_digests[0]
                and item.promotion_decision_rule_digest
                == plan.promotion_decision_rule_digest
                and item.classifier_manifest_digests
                == tuple(
                    manifest.manifest_digest
                    for manifest in plan.regime_classifier_manifests
                )
            )
            connection.execute(
                "INSERT INTO neural_prior_holdout_plan_experiment_bindings "
                "(holdout_plan_digest, family_digest, trial_digest, bound_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    plan.plan_digest,
                    family.family_digest,
                    matching_trial.trial_digest,
                    now.isoformat(),
                ),
            )
            final = datetime.now(timezone.utc)
            if plan.mode == "prospective" and any(
                final >= issue for issue in issue_times
            ):
                raise ValueError("holdout plan crossed its forecast issue time")
            if plan.mode == "prospective" and final >= min(raw_slot_times):
                raise ValueError(
                    "holdout plan crossed its earliest raw observation slot"
                )
            if started is not None and final >= started:
                raise ValueError("holdout plan crossed candidate training time")
        return plan.plan_digest

    def load_neural_prior_holdout_plan(
        self,
        plan_digest: str,
    ) -> (
        NeuralPriorHoldoutPlan
        | LegacyNeuralPriorHoldoutPlanAudit
        | LegacyNeuralPriorHoldoutPlanV2Audit
        | LegacyNeuralPriorHoldoutPlanV3Audit
        | LegacyNeuralPriorHoldoutPlanV4Audit
        | LegacyNeuralPriorHoldoutPlanV5Audit
        | LegacyNeuralPriorHoldoutPlanV6Audit
        | LegacyNeuralPriorHoldoutPlanV7Audit
        | LegacyNeuralPriorHoldoutPlanV8Audit
        | LegacyNeuralPriorHoldoutPlanV9Audit
        | LegacyNeuralPriorHoldoutPlanV10Audit
        | LegacyNeuralPriorHoldoutPlanV11Audit
        | LegacyNeuralPriorHoldoutPlanV12Audit
        | LegacyNeuralPriorHoldoutPlanV13Audit
        | LegacyNeuralPriorHoldoutPlanV14Audit
        | LegacyNeuralPriorHoldoutPlanV15Audit
        | LegacyNeuralPriorHoldoutPlanV16Audit
        | LegacyNeuralPriorHoldoutPlanV17Audit
        | LegacyNeuralPriorHoldoutPlanV18Audit
        | LegacyNeuralPriorHoldoutPlanV19Audit
        | LegacyNeuralPriorHoldoutPlanV20Audit
        | LegacyNeuralPriorHoldoutPlanV21Audit
        | LegacyNeuralPriorHoldoutPlanV22Audit
        | LegacyNeuralPriorHoldoutPlanV23Audit
        | LegacyNeuralPriorHoldoutPlanV24Audit
        | LegacyNeuralPriorHoldoutPlanV25Audit
    ):
        """Load and verify one immutable pre-registered holdout plan."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM neural_prior_holdout_plans "
                "WHERE plan_digest = ?",
                (plan_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown neural-prior holdout plan: {plan_digest}")
        value = json.loads(row[0])
        if value.get("contract") == "neural-prior-holdout-plan-v5":
            return LegacyNeuralPriorHoldoutPlanV5Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v6":
            return LegacyNeuralPriorHoldoutPlanV6Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v7":
            return LegacyNeuralPriorHoldoutPlanV7Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v8":
            return LegacyNeuralPriorHoldoutPlanV8Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v9":
            return LegacyNeuralPriorHoldoutPlanV9Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v10":
            return LegacyNeuralPriorHoldoutPlanV10Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v11":
            return LegacyNeuralPriorHoldoutPlanV11Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v12":
            return LegacyNeuralPriorHoldoutPlanV12Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v13":
            return LegacyNeuralPriorHoldoutPlanV13Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v14":
            return LegacyNeuralPriorHoldoutPlanV14Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v15":
            return LegacyNeuralPriorHoldoutPlanV15Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v16":
            return LegacyNeuralPriorHoldoutPlanV16Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v17":
            return LegacyNeuralPriorHoldoutPlanV17Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v18":
            return LegacyNeuralPriorHoldoutPlanV18Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v19":
            return LegacyNeuralPriorHoldoutPlanV19Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v20":
            return LegacyNeuralPriorHoldoutPlanV20Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v21":
            return LegacyNeuralPriorHoldoutPlanV21Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v22":
            return LegacyNeuralPriorHoldoutPlanV22Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v23":
            return LegacyNeuralPriorHoldoutPlanV23Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v24":
            return LegacyNeuralPriorHoldoutPlanV24Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        if value.get("contract") == "neural-prior-holdout-plan-v25":
            return LegacyNeuralPriorHoldoutPlanV25Audit(
                plan_digest=plan_digest,
                payload_json=json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                ),
            )
        value.pop("plan_digest", None)
        value["candidate_family_digests"] = tuple(
            value["candidate_family_digests"]
        )
        if value.get("contract") == "neural-prior-holdout-plan-v1":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanCase(**item) for item in value["cases"]
            )
            plan = LegacyNeuralPriorHoldoutPlanAudit(**value)
            if plan.plan_digest != plan_digest:
                raise ValueError("legacy neural-prior holdout digest mismatch")
            return plan
        if value.get("contract") == "neural-prior-holdout-plan-v2":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanV2Case(**item)
                for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            plan_v2 = LegacyNeuralPriorHoldoutPlanV2Audit(**value)
            if plan_v2.plan_digest != plan_digest:
                raise ValueError("legacy v2 neural-prior holdout digest mismatch")
            return plan_v2
        if value.get("contract") == "neural-prior-holdout-plan-v3":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanV3Case(**item)
                for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            value["uncertainty_target_plans"] = tuple(
                {
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
                for item in value["uncertainty_target_plans"]
            )
            plan_v3 = LegacyNeuralPriorHoldoutPlanV3Audit(**value)
            if plan_v3.plan_digest != plan_digest:
                raise ValueError("legacy v3 neural-prior holdout digest mismatch")
            return plan_v3
        if value.get("contract") == "neural-prior-holdout-plan-v4":
            value["cases"] = tuple(
                LegacyNeuralPriorHoldoutPlanV3Case(**item)
                for item in value["cases"]
            )
            value["input_plans"] = tuple(
                NeuralPriorInputPlan(
                    **{
                        key: entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    }
                )
                for item in value["input_plans"]
            )
            value["uncertainty_target_plans"] = tuple(
                {
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
                for item in value["uncertainty_target_plans"]
            )
            plan_v4 = LegacyNeuralPriorHoldoutPlanV4Audit(**value)
            if plan_v4.plan_digest != plan_digest:
                raise ValueError("legacy v4 neural-prior holdout digest mismatch")
            return plan_v4
        value["cases"] = tuple(
            NeuralPriorHoldoutPlanCase(
                **cast(
                    Any,
                    {
                        **item,
                        "reference_active_range_regimes": tuple(
                            item["reference_active_range_regimes"]
                        ),
                    },
                )
            )
            for item in value["cases"]
        )
        value["input_plans"] = tuple(
            NeuralPriorInputPlan(
                **cast(
                    Any,
                    {
                        key: tuple(entry) if key == "valid_times" else entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    },
                )
            )
            for item in value["input_plans"]
        )
        value["raw_observation_slot_plans"] = tuple(
            RawObservationSlotPlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key != "slot_digest"
                }
            )
            for item in value["raw_observation_slot_plans"]
        )
        value["meteorological_sampling_units"] = tuple(
            MeteorologicalSamplingUnit(
                **cast(
                    Any,
                    {
                        key: (
                            tuple(entry)
                            if key == "raw_observation_slot_digests"
                            else entry
                        )
                        for key, entry in item.items()
                        if key != "sampling_unit_digest"
                    },
                )
            )
            for item in value["meteorological_sampling_units"]
        )
        raw_trust_values = dict(value["raw_ingestor_trust_store"])
        raw_trust_values.pop("content_digest", None)
        raw_trust_values["authorities"] = tuple(
            tuple(item) for item in raw_trust_values["authorities"]
        )
        value["raw_ingestor_trust_store"] = RawIngestorTrustStore(
            **cast(Any, raw_trust_values)
        )
        target_trust_values = dict(
            value["training_target_source_trust_store"]
        )
        target_trust_values.pop("content_digest", None)
        target_trust_values["authorities"] = tuple(
            (
                str(item[0]),
                str(item[1]),
                int(item[2]),
                str(item[3]),
                str(item[4]),
                None if item[5] is None else str(item[5]),
                tuple(item[6]),
                tuple(item[7]),
            )
            for item in target_trust_values["authorities"]
        )
        value["training_target_source_trust_store"] = (
            TrainingTargetSourceTrustStore(
                **cast(Any, target_trust_values)
            )
        )
        value["uncertainty_target_plans"] = tuple(
            PriorUncertaintyTargetPlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key not in {"plan_digest", "support_event_digest"}
                }
            )
            for item in value["uncertainty_target_plans"]
        )
        value["state_calibration_target_plans"] = tuple(
            NeuralPriorStateCalibrationPlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
            )
            for item in value["state_calibration_target_plans"]
        )
        value["verification_observation_error_plans"] = tuple(
            VerificationObservationErrorPlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
            )
            for item in value["verification_observation_error_plans"]
        )
        value["range_band_contracts"] = tuple(
            RangeBandContract(
                **cast(
                    Any,
                    {
                        key: (
                            tuple(tuple(values) for values in entry)
                            if key == "registered_active_range_regime_sets"
                            else tuple(entry)
                            if key
                            in {
                                "range_regime_labels",
                                "range_band_mask_digests",
                                "reference_active_range_regimes",
                            }
                            else entry
                        )
                        for key, entry in item.items()
                        if key != "contract_digest"
                    },
                )
            )
            for item in value["range_band_contracts"]
        )
        decoded_geometries: list[
            RangeGeometryContract | MosaicRangeGeometryContract
        ] = []
        for item in value["range_geometry_contracts"]:
            geometry_values = {
                key: entry
                for key, entry in item.items()
                if key != "contract_digest"
            }
            for name in (
                "range_regime_labels",
                "radial_distance_edges_m",
                "radar_site_digests",
                "radar_site_location_digests",
            ):
                if name in geometry_values:
                    geometry_values[name] = tuple(geometry_values[name])
            if "radar_projected_xy_m" in geometry_values:
                geometry_values["radar_projected_xy_m"] = tuple(
                    tuple(point)
                    for point in geometry_values["radar_projected_xy_m"]
                )
            if (
                geometry_values.get("contract")
                == "mosaic-horizontal-range-geometry-contract-v3"
            ):
                decoded_geometries.append(
                    MosaicRangeGeometryContract(**cast(Any, geometry_values))
                )
            else:
                decoded_geometries.append(
                    RangeGeometryContract(**cast(Any, geometry_values))
                )
        value["range_geometry_contracts"] = tuple(decoded_geometries)
        value["operational_issuance_domain_plans"] = tuple(
            OperationalIssuanceDomainPlan(
                **cast(
                    Any,
                    {
                        key: tuple(entry) if key == "lead_minutes" else entry
                        for key, entry in item.items()
                        if key != "plan_digest"
                    },
                )
            )
            for item in value["operational_issuance_domain_plans"]
        )
        value["regime_reference_plans"] = tuple(
            RegimeReferencePlan(
                **{
                    key: entry
                    for key, entry in item.items()
                    if key != "plan_digest"
                }
            )
            for item in value["regime_reference_plans"]
        )
        value["regime_classifier_manifests"] = tuple(
            RegimeClassifierManifest(
                **cast(
                    Any,
                    {
                        key: (
                            tuple(tuple(value) for value in entry)
                            if key == "training_time_windows"
                            else tuple(entry)
                            if key
                            in {
                                "training_case_ids",
                                "training_input_bundle_digests",
                                "training_full_analysis_input_digests",
                                "training_physical_event_digests",
                                "training_storm_ids",
                                "training_days",
                                "training_radar_ids",
                                "training_grid_contract_digests",
                                "training_raw_volume_identity_digests",
                                "training_sampling_unit_digests",
                            }
                            else entry
                        )
                        for key, entry in item.items()
                        if key != "manifest_digest"
                    },
                )
            )
            for item in value["regime_classifier_manifests"]
        )
        family_values = dict(value["promotion_experiment_family"])
        family_values.pop("family_digest", None)
        family_values.pop("total_family_size", None)
        family_values["meteorological_sampling_unit_digests"] = tuple(
            family_values["meteorological_sampling_unit_digests"]
        )
        family_values["raw_observation_slot_digests"] = tuple(
            family_values["raw_observation_slot_digests"]
        )
        reservation_values = dict(
            family_values["global_sampling_reservation"]
        )
        reservation_values.pop("receipt_digest", None)
        reservation_values["raw_observation_slot_digests"] = tuple(
            reservation_values["raw_observation_slot_digests"]
        )
        family_values["global_sampling_reservation"] = (
            GlobalSamplingReservationReceipt(**reservation_values)
        )
        family_values["trials"] = tuple(
            PromotionExperimentTrial(
                **cast(
                    Any,
                    {
                        key: entry
                        for key, entry in item.items()
                        if key != "trial_digest"
                    }
                    | {
                        "classifier_manifest_digests": tuple(
                            item["classifier_manifest_digests"]
                        )
                    },
                )
            )
            for item in family_values["trials"]
        )
        value["promotion_experiment_family"] = PromotionExperimentFamily(
            **cast(Any, family_values)
        )
        event_plan_values = {
            key: entry
            for key, entry in value["physical_event_catalog_plan"].items()
            if key != "plan_digest"
        }
        event_plan_values["holdout_case_ids"] = tuple(
            event_plan_values["holdout_case_ids"]
        )
        value["physical_event_catalog_plan"] = PhysicalEventCatalogPlan(
            **cast(Any, event_plan_values)
        )
        plan = NeuralPriorHoldoutPlan(**value)
        if plan.plan_digest != plan_digest:
            raise ValueError("neural-prior holdout plan digest mismatch")
        return plan

    def append_physical_event_catalog_result(
        self,
        plan: NeuralPriorHoldoutPlan | PhysicalEventCatalogPlan,
        result: PhysicalEventCatalogResult,
    ) -> str:
        """Append the sole candidate-neutral holdout or training catalog."""

        catalog_plan = (
            plan.physical_event_catalog_plan
            if isinstance(plan, NeuralPriorHoldoutPlan)
            else plan
        )
        if isinstance(plan, NeuralPriorHoldoutPlan):
            validate_neural_prior_holdout_plan(plan)
        validate_physical_event_catalog_result(
            result,
            catalog_plan,
        )
        result_json = json.dumps(
            result.payload | {"result_digest": result.result_digest},
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            cataloged_at = datetime.fromisoformat(
                result.cataloged_at.replace("Z", "+00:00")
            )
            latest_input_available = max(
                datetime.fromisoformat(
                    item.input_available_time.replace("Z", "+00:00")
                )
                for item in result.case_spatial_membership_evidences
            )
            if cataloged_at > now:
                raise ValueError(
                    "event catalog time is after trusted ledger time"
                )
            if cataloged_at < latest_input_available:
                raise ValueError("event catalog predates holdout input availability")
            try:
                if isinstance(plan, NeuralPriorHoldoutPlan):
                    plan_row = connection.execute(
                        "SELECT plan_json FROM neural_prior_holdout_plans "
                        "WHERE plan_digest = ?",
                        (plan.plan_digest,),
                    ).fetchone()
                    if plan_row is None or plan_row[0] != json.dumps(
                        asdict(plan), sort_keys=True
                    ):
                        raise ValueError(
                            "event catalog holdout plan is not registered"
                        )
                    connection.execute(
                        "INSERT INTO neural_prior_event_catalog_results "
                        "(plan_digest, result_digest, result_json, cataloged_at, "
                        "created_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            plan.plan_digest,
                            result.result_digest,
                            result_json,
                            result.cataloged_at,
                            now.isoformat(),
                        ),
                    )
                else:
                    connection.execute(
                        "INSERT INTO neural_prior_training_event_catalog_results "
                        "(plan_digest, result_digest, plan_json, result_json, "
                        "cataloged_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            catalog_plan.plan_digest,
                            result.result_digest,
                            json.dumps(
                                catalog_plan.payload
                                | {"plan_digest": catalog_plan.plan_digest},
                                sort_keys=True,
                            ),
                            result_json,
                            result.cataloged_at,
                            now.isoformat(),
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "physical event catalog is already registered"
                ) from error
        return result.result_digest

    def append_resolved_source_coverage_artifact(
        self,
        plan: OperationalIssuanceDomainPlan,
        input_plan: NeuralPriorInputPlan,
        resolved: ResolvedSourceCoverageArtifact,
        operational_domain: OperationalIssuanceDomainArtifact,
    ) -> str:
        """Pre-issue append of trusted input-time mosaic source resolution."""

        validate_resolved_source_coverage_artifact(resolved)
        if (
            plan.radar_source_kind != "mosaic"
            or resolved.issuance_domain_plan_digest != plan.plan_digest
            or resolved.case_id != plan.case_id
            or input_plan.grid_contract_digest != plan.grid_contract_digest
            or resolved.input_available_at != input_plan.input_available_time
            or resolved.decision_deadline != input_plan.decision_deadline
            or resolved.publication_time != input_plan.publication_time
            or resolved.data_ingestor_id != plan.data_ingestor_id
            or resolved.data_ingestor_public_key_hex
            != plan.data_ingestor_public_key_hex
            or resolved.source_radar_registry_digest
            != plan.source_radar_registry_digest
            or resolved.source_radar_count != plan.source_radar_count
            or operational_domain.plan_digest != plan.plan_digest
            or operational_domain.resolved_source_coverage_artifact_digest
            != resolved.artifact_digest
        ):
            raise ValueError("resolved source coverage disagrees with its issue plan")
        now = datetime.now(timezone.utc)
        resolved_at = datetime.fromisoformat(
            resolved.resolved_at.replace("Z", "+00:00")
        )
        deadline = datetime.fromisoformat(
            input_plan.decision_deadline.replace("Z", "+00:00")
        )
        if not resolved_at <= now <= deadline:
            raise ValueError("resolved source coverage was not appended pre-issue")
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO neural_prior_resolved_source_coverage_artifacts "
                    "(artifact_digest, operational_domain_artifact_digest, "
                    "issuance_domain_plan_digest, input_plan_digest, case_id, "
                    "payload_json, resolved_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        resolved.artifact_digest,
                        operational_domain.artifact_digest,
                        plan.plan_digest,
                        input_plan.plan_digest,
                        plan.case_id,
                        json.dumps(
                            resolved.payload
                            | {"artifact_digest": resolved.artifact_digest},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        resolved.resolved_at,
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "resolved source coverage is already registered"
                ) from error
        return resolved.artifact_digest

    @staticmethod
    def _record_global_sampling_registry_entry(
        connection: sqlite3.Connection,
        *,
        registry_id: str,
        registry_sequence_number: int,
        previous_registry_root_digest: str,
        committed_registry_root_digest: str,
        receipt_digest: str,
        entry_kind: str,
        family_digest: str,
        created_at: str,
    ) -> None:
        """Append one non-equivocating entry to the mirrored global registry."""

        if entry_kind not in {"training_raw", "slot_reservation", "raw_resolution"}:
            raise ValueError("global sampling registry entry kind is invalid")
        retained = connection.execute(
            "SELECT registry_id,registry_sequence_number,"
            "previous_registry_root_digest,committed_registry_root_digest,"
            "entry_kind,family_digest FROM global_sampling_registry_entries "
            "WHERE receipt_digest = ?",
            (receipt_digest,),
        ).fetchone()
        expected = (
            registry_id,
            registry_sequence_number,
            previous_registry_root_digest,
            committed_registry_root_digest,
            entry_kind,
            family_digest,
        )
        if retained is not None:
            if tuple(retained) != expected:
                raise ValueError("global sampling registry entry equivocated")
            return
        head = connection.execute(
            "SELECT registry_sequence_number,committed_registry_root_digest "
            "FROM global_sampling_registry_entries WHERE registry_id = ? "
            "ORDER BY registry_sequence_number DESC LIMIT 1",
            (registry_id,),
        ).fetchone()
        expected_sequence = 1 if head is None else int(head[0]) + 1
        expected_previous = (
            GLOBAL_SAMPLING_REGISTRY_GENESIS_DIGEST
            if head is None
            else str(head[1])
        )
        if (
            registry_sequence_number != expected_sequence
            or previous_registry_root_digest != expected_previous
        ):
            raise ValueError("global sampling registry entry is not contiguous")
        connection.execute(
            "INSERT INTO global_sampling_registry_entries "
            "(registry_id,registry_sequence_number,"
            "previous_registry_root_digest,committed_registry_root_digest,"
            "receipt_digest,entry_kind,family_digest,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                registry_id,
                registry_sequence_number,
                previous_registry_root_digest,
                committed_registry_root_digest,
                receipt_digest,
                entry_kind,
                family_digest,
                created_at,
            ),
        )

    @staticmethod
    def _record_raw_volume_resolution_membership(
        connection: sqlite3.Connection,
        *,
        global_resolution_receipt_digest: str,
        raw_observation_slot_digest: str,
        raw_volume_identity_digest: str,
        case_id: str,
        family_digest: str,
        resolved_at: str,
    ) -> None:
        """Reserve one stable raw identity and attach one rolling-window use."""

        connection.execute(
            "INSERT OR IGNORE INTO "
            "promotion_raw_volume_identity_reservations "
            "(raw_volume_identity_digest,family_digest,"
            "global_resolution_receipt_digest,reserved_at) VALUES (?,?,?,?)",
            (
                raw_volume_identity_digest,
                family_digest,
                global_resolution_receipt_digest,
                resolved_at,
            ),
        )
        retained_identity = connection.execute(
            "SELECT family_digest FROM "
            "promotion_raw_volume_identity_reservations "
            "WHERE raw_volume_identity_digest = ?",
            (raw_volume_identity_digest,),
        ).fetchone()
        if retained_identity != (family_digest,):
            raise ValueError(
                "canonical raw volume is reserved by another family"
            )
        connection.execute(
            "INSERT OR IGNORE INTO raw_observation_slot_identity_bindings "
            "(raw_observation_slot_digest,raw_volume_identity_digest,"
            "family_digest,first_resolved_at) VALUES (?,?,?,?)",
            (
                raw_observation_slot_digest,
                raw_volume_identity_digest,
                family_digest,
                resolved_at,
            ),
        )
        retained_slot = connection.execute(
            "SELECT raw_volume_identity_digest,family_digest FROM "
            "raw_observation_slot_identity_bindings "
            "WHERE raw_observation_slot_digest = ?",
            (raw_observation_slot_digest,),
        ).fetchone()
        if retained_slot != (raw_volume_identity_digest, family_digest):
            raise ValueError("raw observation slot resolution equivocated")
        connection.execute(
            "INSERT INTO raw_volume_resolution_memberships "
            "(global_resolution_receipt_digest,raw_observation_slot_digest,"
            "raw_volume_identity_digest,case_id,family_digest,resolved_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                global_resolution_receipt_digest,
                raw_observation_slot_digest,
                raw_volume_identity_digest,
                case_id,
                family_digest,
                resolved_at,
            ),
        )

    def append_operational_analysis_input_provenance_plan(
        self,
        plan: OperationalAnalysisInputProvenancePlan,
        *,
        analysis_processor_trust_store_path: str | Path,
    ) -> str:
        """Register one ordinary operational cycle before its first raw slot."""

        if type(plan) is not OperationalAnalysisInputProvenancePlan:
            raise TypeError("current operational provenance plan is required")
        trust = _load_promotion_deployment_authority_trust_store(
            analysis_processor_trust_store_path
        )
        _trusted_authority_key(
            trust,
            authority_id=plan.analysis_processor_id,
            public_key_hex=plan.analysis_processor_public_key_hex,
            role="analysis_processor",
            issued_at=plan.registered_at,
        )
        if plan.analysis_processor_trust_store_digest != trust.content_digest:
            raise ValueError("operational provenance processor trust disagrees")
        canonical = json.dumps(
            plan.payload | {"plan_digest": plan.plan_digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(timezone.utc)
            earliest_slot = min(
                datetime.fromisoformat(
                    item.acquisition_valid_time.replace("Z", "+00:00")
                )
                for item in plan.raw_observation_slot_plans
            )
            if (
                now >= earliest_slot
                or now
                >= datetime.fromisoformat(
                    plan.input_plan.observation_valid_time.replace("Z", "+00:00")
                )
            ):
                raise ValueError(
                    "operational provenance plan was not durably prospective"
                )
            connection.execute(
                "INSERT INTO operational_analysis_input_provenance_plans "
                "(plan_digest,plan_id,input_plan_digest,payload_json,"
                "registered_at,created_at) VALUES (?,?,?,?,?,?)",
                (
                    plan.plan_digest,
                    plan.plan_id,
                    plan.input_plan.plan_digest,
                    canonical,
                    plan.registered_at,
                    now.isoformat(),
                ),
            )
            final_now = datetime.now(timezone.utc)
            if final_now >= earliest_slot:
                raise ValueError(
                    "operational provenance plan crossed its first raw slot"
                )
        return plan.plan_digest

    def append_analysis_input_provenance(
        self,
        plan: NeuralPriorHoldoutPlan,
        *,
        case_id: str,
        run: ForecastRunContract,
        resolved_raw_observations: tuple[
            RawObservationResolutionReceipt, ...
        ],
        global_resolution: GlobalRawVolumeResolutionReceipt,
        derivation: AnalysisInputDerivationArtifact,
        resolved_source_coverage: ResolvedSourceCoverageArtifact | None,
        background_frames_dbz: Tensor | None = None,
        raw_ingestor_trust_store_path: str | Path,
        analysis_processor_trust_store_path: str | Path,
        provenance_commit_signer: DeploymentAuthoritySigner,
    ) -> str:
        """Durably commit raw bytes and their derivation before the case deadline."""

        validate_neural_prior_holdout_plan(plan)
        run.validate_integrity()
        planned_case = plan.case(case_id)
        input_plan = next(
            item
            for item in plan.input_plans
            if item.plan_digest == planned_case.input_plan_digest
        )
        sampling_unit = next(
            item
            for item in plan.meteorological_sampling_units
            if item.sampling_unit_digest
            == planned_case.meteorological_sampling_unit_digest
        )
        range_band = next(
            item
            for item in plan.range_band_contracts
            if item.contract_digest == planned_case.range_band_contract_digest
        )
        range_geometry = next(
            item
            for item in plan.range_geometry_contracts
            if item.contract_digest == range_band.range_geometry_contract_digest
        )
        issuance_plan = next(
            item
            for item in plan.operational_issuance_domain_plans
            if item.plan_digest
            == planned_case.operational_issuance_domain_plan_digest
        )
        slot_by_digest = {
            item.slot_digest: item
            for item in plan.raw_observation_slot_plans
            if item.slot_digest in sampling_unit.raw_observation_slot_digests
        }
        current_raw_ingestor_trust = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        receipt_by_slot = {
            item.slot_plan_digest: item for item in resolved_raw_observations
        }
        for slot_digest, receipt in receipt_by_slot.items():
            _validate_current_raw_ingestor_receipt(
                receipt,
                slot_by_digest[slot_digest],
                pinned_trust_store=plan.raw_ingestor_trust_store,
                current_trust_store=current_raw_ingestor_trust,
            )
        ordered_receipts = tuple(
            sorted(
                resolved_raw_observations,
                key=lambda item: (
                    item.acquisition_valid_time,
                    item.radar_site_digest,
                ),
            )
        )
        raw_sites = {
            item.radar_site_digest
            for item in ordered_receipts
        }
        slot_source_rules = {
            slot_by_digest[item.slot_plan_digest].source_selection_rule_digest
            for item in ordered_receipts
        }
        if run.operational_data_identity_json is None:
            raise ValueError("analysis provenance requires operational source identity")
        operational_identity = OperationalDataIdentity.from_json(
            run.operational_data_identity_json
        )
        if type(range_geometry) is RangeGeometryContract:
            if (
                raw_sites != {range_geometry.radar_site_digest}
                or operational_identity.radar_source_kind != "single_site"
                or operational_identity.radar_site_digest
                != range_geometry.radar_site_digest
                or operational_identity.radar_site_location_digest
                != range_geometry.radar_site_location_digest
                or slot_source_rules
                != {issuance_plan.radar_source_contract_digest}
            ):
                raise ValueError(
                    "single-site raw provenance disagrees with radar geometry"
                )
        elif type(range_geometry) is MosaicRangeGeometryContract:
            if (
                raw_sites != set(range_geometry.radar_site_digests)
                or operational_identity.radar_source_kind != "mosaic"
                or operational_identity.source_selection_policy_digest
                != range_geometry.source_selection_policy_digest
                or slot_source_rules
                != {range_geometry.source_selection_policy_digest}
            ):
                raise ValueError(
                    "mosaic raw provenance disagrees with radar registry"
                )
        else:
            raise TypeError("current range geometry contract is required")
        bindings = tuple(
            sorted(
                (
                    item.slot_plan_digest,
                    item.resolution_identity_digest,
                )
                for item in ordered_receipts
            )
        )
        reservation = plan.promotion_experiment_family.global_sampling_reservation
        if (
            not ordered_receipts
            or set(receipt_by_slot) != set(slot_by_digest)
            or len(receipt_by_slot) != len(ordered_receipts)
            or global_resolution.slot_identity_bindings != bindings
            or global_resolution.experiment_scope_digest
            != reservation.experiment_scope_digest
            or global_resolution.reservation_receipt_digest
            != reservation.receipt_digest
            or global_resolution.registry_id != reservation.registry_id
            or global_resolution.authority_id != reservation.authority_id
            or global_resolution.authority_public_key_hex
            != reservation.authority_public_key_hex
            or derivation.case_id != case_id
            or derivation.input_plan_digest != input_plan.plan_digest
            or derivation.global_raw_resolution_receipt_digest
            != global_resolution.receipt_digest
            or derivation.resolved_raw_observation_receipt_digests
            != tuple(sorted(item.receipt_digest for item in ordered_receipts))
            or derivation.canonical_raw_volume_identity_digests
            != tuple(
                sorted(
                    item.raw_volume_identity.identity_digest
                    for item in ordered_receipts
                    if type(item) is ResolvedRawObservationReceipt
                )
            )
            or derivation.processor_id != plan.analysis_processor_id
            or derivation.processor_public_key_hex
            != plan.analysis_processor_public_key_hex
            or derivation.decoder_version_digest
            != _json_digest(
                {
                    "contract": "canonical-binary32-radar-decoder-v1",
                    "raw_volume_contract": (
                        "canonical-raw-grid-volume-artifact-v4"
                    ),
                }
            )
            or derivation.qc_algorithm_digest
            != _json_digest(
                {
                    "contract": (
                        "native-flags-canonical-masked-input-qc-v3"
                    ),
                    "registered_qc_pipeline_digest": (
                        input_plan.qc_pipeline_digest
                    ),
                }
            )
            or derivation.qc_policy_digest != input_plan.mask_policy_digest
            or derivation.regrid_algorithm_digest
            != _json_digest(
                {
                    "contract": "exact-source-grid-identity-regrid-v1",
                    "source_grid_digest": input_plan.grid_contract_digest,
                    "target_grid_digest": derivation.grid_contract_digest,
                }
            )
            or derivation.background_cycle_rule_digest
            != input_plan.background_cycle_rule_digest
            or run.input_plan_digest != input_plan.plan_digest
            or run.analysis_input_derivation_artifact_digest
            != derivation.artifact_digest
            or run.analysis_input_derivation_artifact_json
            != json.dumps(
                derivation.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            or derivation.input_bundle_digest != run.input_bundle_digest
            or derivation.full_analysis_input_digest
            != run.full_analysis_input_digest
            or derivation.grid_contract_digest
            != run.grid_time_contract_digest
        ):
            raise ValueError("analysis input provenance disagrees with its plan")
        if resolved_source_coverage is not None:
            validate_resolved_source_coverage_artifact(resolved_source_coverage)
            if (
                resolved_source_coverage.case_id != case_id
                or derivation.source_selection_evidence_digest
                != resolved_source_coverage.artifact_digest
            ):
                raise ValueError("analysis source coverage preimage disagrees")
        elif derivation.source_selection_evidence_digest != _json_digest(
            {"contract": "single-site-source-selection-v1"}
        ):
            raise ValueError("single-site analysis source selection disagrees")
        derived = _derive_analysis_inputs_from_raw_products(
            input_plan=input_plan,
            resolved_raw_observations=ordered_receipts,
            resolved_source_coverage=resolved_source_coverage,
        )
        background_digest = (
            None
            if background_frames_dbz is None
            else tensor_digest(background_frames_dbz)
        )
        (
            background_valid_times,
            background_source_identity_digest,
            background_identities,
        ) = _background_input_identity_digests(
            input_plan=input_plan,
            run=run,
            background_frames_dbz=background_frames_dbz,
        )
        if (
            tensor_digest(derived[0]) != derivation.input_frames_digest
            or tensor_digest(derived[1]) != derivation.observation_masks_digest
            or tensor_digest(derived[2])
            != derivation.observation_quality_weight_digest
            or tensor_digest(derived[3]) != derivation.observation_std_dbz_digest
            or background_digest != derivation.background_frames_digest
            or background_valid_times != derivation.background_valid_times
            or background_source_identity_digest
            != derivation.background_source_identity_digest
            or background_identities
            != derivation.background_input_identity_digests
        ):
            raise ValueError("analysis input provenance does not replay its outputs")

        arrays: dict[str, Tensor] = {
            **_raw_resolution_encoded_arrays(
                ordered_receipts,
                derived_frames=derived[0],
            ),
            "derived_input_frames": derived[0],
            "derived_qc_valid_mask": derived[1],
            "derived_quality_weight": derived[2],
            "derived_observation_std_dbz": derived[3],
        }
        if background_frames_dbz is not None:
            arrays["background_frames_dbz"] = background_frames_dbz
        if resolved_source_coverage is not None:
            arrays.update(
                {
                    "source_radar_index_map": (
                        resolved_source_coverage._source_radar_index_map
                    ),
                    "input_history_source_radar_index_map": (
                        resolved_source_coverage
                        ._input_history_source_radar_index_map
                    ),
                    "outage_mask": resolved_source_coverage._outage_mask,
                    "dynamic_qc_valid_mask": (
                        resolved_source_coverage._dynamic_qc_valid_mask
                    ),
                    "nominal_source_coverage_mask": (
                        resolved_source_coverage._nominal_source_coverage_mask
                    ),
                    "resolved_source_coverage_mask": (
                        resolved_source_coverage._resolved_mask
                    ),
                }
            )
        metadata = {
            "contract": "analysis-input-provenance-bundle-v1",
            "holdout_plan_digest": plan.plan_digest,
            "case_id": case_id,
            "raw_ingestor_trust_store_digest": (
                current_raw_ingestor_trust.content_digest
            ),
            "input_plan": input_plan.payload | {"plan_digest": input_plan.plan_digest},
            "resolved_raw_observations": [
                item.payload | {"receipt_digest": item.receipt_digest}
                for item in ordered_receipts
            ],
            "global_resolution": global_resolution.payload
            | {"receipt_digest": global_resolution.receipt_digest},
            "analysis_input_derivation": derivation.payload
            | {"artifact_digest": derivation.artifact_digest},
            "resolved_source_coverage": (
                None
                if resolved_source_coverage is None
                else resolved_source_coverage.payload
                | {"artifact_digest": resolved_source_coverage.artifact_digest}
            ),
        }
        canonical_payload = json.dumps(
            derivation.payload
            | {
                "artifact_digest": derivation.artifact_digest,
                "input_plan": input_plan.payload
                | {"plan_digest": input_plan.plan_digest},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        target = self.analysis_input_provenance_dir / derivation.artifact_digest
        with self._connect() as connection:
            retained = connection.execute(
                "SELECT provenance_kind,payload_json,status FROM "
                "analysis_input_provenance_commits WHERE artifact_digest = ?",
                (derivation.artifact_digest,),
            ).fetchone()
        if retained is not None:
            if tuple(retained[:2]) != ("holdout", canonical_payload):
                raise ValueError("holdout provenance commit equivocated")
            return self.reconcile_prepared_analysis_input_provenance(
                derivation.artifact_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=(
                    analysis_processor_trust_store_path
                ),
                provenance_commit_signer=provenance_commit_signer,
            )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{derivation.artifact_digest}.",
                dir=self.analysis_input_provenance_dir,
            )
        )
        published = False
        try:
            arrays_path = temporary / "source_and_derived_arrays.npz"
            numpy_arrays = {
                name: tensor.detach().cpu().contiguous().numpy()
                for name, tensor in arrays.items()
            }
            np.savez_compressed(
                arrays_path,
                **cast(dict[str, Any], numpy_arrays),
            )
            metadata_path = temporary / "provenance.json"
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            checksums = {
                "source_and_derived_arrays.npz": _file_digest(arrays_path),
                "provenance.json": _file_digest(metadata_path),
            }
            checksums_path = temporary / "checksums.json"
            checksums_path.write_text(_json_text(checksums), encoding="utf-8")
            for path in (arrays_path, metadata_path, checksums_path):
                _fsync_file(path)
            _fsync_directory(temporary)

            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(timezone.utc)
                deadline = datetime.fromisoformat(
                    input_plan.decision_deadline.replace("Z", "+00:00")
                )
                if not (
                    datetime.fromisoformat(
                        global_resolution.resolved_at.replace("Z", "+00:00")
                    )
                    <= datetime.fromisoformat(
                        derivation.processed_at.replace("Z", "+00:00")
                    )
                    <= now
                    <= deadline
                ):
                    raise ValueError(
                        "analysis input provenance missed its durable deadline"
                    )
                final_raw_ingestor_trust = _load_raw_ingestor_trust_store(
                    raw_ingestor_trust_store_path
                )
                if (
                    final_raw_ingestor_trust.content_digest
                    != current_raw_ingestor_trust.content_digest
                ):
                    raise ValueError(
                        "raw-ingestor trust store changed during provenance commit"
                    )
                for slot_digest, receipt in receipt_by_slot.items():
                    _validate_current_raw_ingestor_receipt(
                        receipt,
                        slot_by_digest[slot_digest],
                        pinned_trust_store=plan.raw_ingestor_trust_store,
                        current_trust_store=final_raw_ingestor_trust,
                    )
                plan_row = connection.execute(
                    "SELECT plan_json FROM neural_prior_holdout_plans "
                    "WHERE plan_digest = ?",
                    (plan.plan_digest,),
                ).fetchone()
                if plan_row is None or plan_row[0] != json.dumps(
                    asdict(plan), sort_keys=True
                ):
                    raise ValueError(
                        "analysis input provenance requires its registered plan"
                    )
                head = connection.execute(
                    "SELECT registry_sequence_number,committed_registry_root_digest "
                    "FROM global_sampling_registry_entries WHERE registry_id = ? "
                    "ORDER BY registry_sequence_number DESC LIMIT 1",
                    (global_resolution.registry_id,),
                ).fetchone()
                if (
                    head is None
                    or global_resolution.registry_sequence_number
                    != int(head[0]) + 1
                    or global_resolution.previous_registry_root_digest
                    != str(head[1])
                ):
                    raise ValueError(
                        "global raw-volume resolution is not a committed successor"
                    )
                if target.exists():
                    if (
                        target.is_symlink()
                        or not target.is_dir()
                        or _file_digest(
                            target / "source_and_derived_arrays.npz"
                        ) != checksums["source_and_derived_arrays.npz"]
                        or _file_digest(target / "provenance.json")
                        != checksums["provenance.json"]
                        or (target / "checksums.json").read_text("utf-8")
                        != _json_text(checksums)
                    ):
                        quarantine = self.analysis_input_provenance_dir / (
                            f".{derivation.artifact_digest}.quarantine."
                            f"{uuid.uuid4().hex}"
                        )
                        os.replace(target, quarantine)
                        _fsync_directory(self.analysis_input_provenance_dir)
                        raise ValueError(
                            "orphan analysis provenance bytes equivocated"
                        )
                    shutil.rmtree(temporary)
                else:
                    os.rename(temporary, target)
                    _fsync_directory(self.analysis_input_provenance_dir)
                published = True
                EpisodeLedger._record_global_sampling_registry_entry(
                    connection,
                    registry_id=global_resolution.registry_id,
                    registry_sequence_number=(
                        global_resolution.registry_sequence_number
                    ),
                    previous_registry_root_digest=(
                        global_resolution.previous_registry_root_digest
                    ),
                    committed_registry_root_digest=(
                        global_resolution.committed_registry_root_digest
                    ),
                    receipt_digest=global_resolution.receipt_digest,
                    entry_kind="raw_resolution",
                    family_digest=(
                        plan.promotion_experiment_family.family_digest
                    ),
                    created_at=now.isoformat(),
                )
                for slot_digest, raw_identity_digest in bindings:
                    EpisodeLedger._record_raw_volume_resolution_membership(
                        connection,
                        global_resolution_receipt_digest=(
                            global_resolution.receipt_digest
                        ),
                        raw_observation_slot_digest=slot_digest,
                        raw_volume_identity_digest=raw_identity_digest,
                        case_id=case_id,
                        family_digest=(
                            plan.promotion_experiment_family.family_digest
                        ),
                        resolved_at=now.isoformat(),
                    )
                authority_trust = (
                    _load_promotion_deployment_authority_trust_store(
                        analysis_processor_trust_store_path
                    )
                )
                ledger_instance_digest = (
                    self._analysis_provenance_ledger_instance_digest(connection)
                )
                holdout_memberships = tuple(
                    (
                        slot_digest,
                        raw_identity_digest,
                        case_id,
                        plan.promotion_experiment_family.family_digest,
                    )
                    for slot_digest, raw_identity_digest in sorted(bindings)
                )
                side_effect_digest = _json_digest(
                    {
                        "contract": (
                            "analysis-provenance-ledger-side-effects-v1"
                        ),
                        "provenance_kind": "holdout",
                        "registry_id": global_resolution.registry_id,
                        "registry_sequence_number": (
                            global_resolution.registry_sequence_number
                        ),
                        "committed_registry_root_digest": (
                            global_resolution.committed_registry_root_digest
                        ),
                        "raw_resolution_receipt_digest": (
                            global_resolution.receipt_digest
                        ),
                        "memberships": [
                            list(item) for item in holdout_memberships
                        ],
                    }
                )
                (
                    preparation_receipt_json,
                    preparation_receipt_digest,
                ) = self._issue_analysis_provenance_preparation_receipt(
                    artifact_digest=derivation.artifact_digest,
                    provenance_kind="holdout",
                    provenance_plan_digest=plan.plan_digest,
                    input_plan_digest=input_plan.plan_digest,
                    raw_resolution_receipt_digest=(
                        global_resolution.receipt_digest
                    ),
                    payload_json=canonical_payload,
                    payload_committed_at=now.isoformat(),
                    deadline=input_plan.decision_deadline,
                    ledger_instance_digest=ledger_instance_digest,
                    side_effect_digest=side_effect_digest,
                    authority_trust_store=authority_trust,
                    signer=provenance_commit_signer,
                )
                if datetime.now(timezone.utc) > deadline:
                    raise ValueError(
                        "analysis provenance ledger receipt missed its deadline"
                    )
                connection.execute(
                    "INSERT INTO neural_prior_analysis_input_provenance "
                    "(artifact_digest,holdout_plan_digest,case_id,input_plan_digest,"
                    "global_resolution_receipt_digest,payload_json,arrays_sha256,"
                    "metadata_sha256,path,raw_ingestor_trust_store_digest,"
                    "raw_trust_validated_at,committed_at,usable,status,"
                    "payload_committed_at,preparation_receipt_json,"
                    "preparation_receipt_digest,activated_at,expired_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?,0,'prepared',?,"
                    "?,?,NULL,NULL)",
                    (
                        derivation.artifact_digest,
                        plan.plan_digest,
                        case_id,
                        input_plan.plan_digest,
                        global_resolution.receipt_digest,
                        json.dumps(
                            derivation.payload
                            | {
                                "artifact_digest": derivation.artifact_digest,
                                "input_plan": input_plan.payload
                                | {"plan_digest": input_plan.plan_digest},
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        checksums["source_and_derived_arrays.npz"],
                        checksums["provenance.json"],
                        str(target),
                        final_raw_ingestor_trust.content_digest,
                        now.isoformat(),
                        now.isoformat(),
                        preparation_receipt_json,
                        preparation_receipt_digest,
                    ),
                )
                connection.execute(
                    "INSERT INTO analysis_input_provenance_commits "
                    "(artifact_digest,provenance_kind,provenance_plan_digest,"
                    "case_id,input_plan_digest,raw_resolution_receipt_digest,"
                    "payload_json,arrays_sha256,metadata_sha256,path,"
                    "raw_ingestor_trust_store_digest,raw_trust_validated_at,"
                    "committed_at,usable,status,payload_committed_at,"
                    "preparation_receipt_json,preparation_receipt_digest,"
                    "activated_at,expired_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?,0,'prepared',?,"
                    "?,?,NULL,NULL)",
                    (
                        derivation.artifact_digest,
                        "holdout",
                        plan.plan_digest,
                        case_id,
                        input_plan.plan_digest,
                        global_resolution.receipt_digest,
                        json.dumps(
                            derivation.payload
                            | {
                                "artifact_digest": derivation.artifact_digest,
                                "input_plan": input_plan.payload
                                | {"plan_digest": input_plan.plan_digest},
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        checksums["source_and_derived_arrays.npz"],
                        checksums["provenance.json"],
                        str(target),
                        final_raw_ingestor_trust.content_digest,
                        now.isoformat(),
                        now.isoformat(),
                        preparation_receipt_json,
                        preparation_receipt_digest,
                    ),
                )
                connection.commit()
            self.reconcile_prepared_analysis_input_provenance(
                derivation.artifact_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=(
                    analysis_processor_trust_store_path
                ),
                provenance_commit_signer=provenance_commit_signer,
            )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and target.exists():
                with self._connect() as connection:
                    retained = connection.execute(
                        "SELECT 1 FROM neural_prior_analysis_input_provenance "
                        "WHERE artifact_digest = ?",
                        (derivation.artifact_digest,),
                    ).fetchone()
                if retained is None:
                    shutil.rmtree(target)
            raise
        return derivation.artifact_digest

    def _validate_analysis_input_provenance_directory(
        self,
        *,
        artifact_digest: str,
        path_text: str,
        arrays_sha256: str,
        metadata_sha256: str,
    ) -> Path:
        """Rehash one ledger-owned provenance directory without trusting links."""

        path, _, _ = self._snapshot_analysis_input_provenance_directory(
            artifact_digest=artifact_digest,
            path_text=path_text,
            arrays_sha256=arrays_sha256,
            metadata_sha256=metadata_sha256,
        )
        return path

    def _snapshot_analysis_input_provenance_directory(
        self,
        *,
        artifact_digest: str,
        path_text: str,
        arrays_sha256: str,
        metadata_sha256: str,
    ) -> tuple[Path, bytes, str]:
        """Read immutable provenance members once through pinned descriptors."""

        expected = self.analysis_input_provenance_dir / artifact_digest
        path = Path(path_text)
        if path != expected:
            raise ValueError("analysis provenance durable path changed")
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            directory = os.open(path, directory_flags)
        except OSError as error:
            raise ValueError("analysis provenance directory is unsafe") from error

        def read_member(name: str, maximum_bytes: int) -> bytes:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=directory)
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid not in {0, os.getuid()}
                    or before.st_mode & 0o022
                    or before.st_size > maximum_bytes
                ):
                    raise ValueError("analysis provenance member is unsafe")
                chunks: list[bytes] = []
                retained = 0
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    retained += len(chunk)
                    if retained > maximum_bytes:
                        raise ValueError("analysis provenance member is too large")
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    stat.S_IMODE(before.st_mode),
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    stat.S_IMODE(after.st_mode),
                ):
                    raise ValueError("analysis provenance member changed")
                return b"".join(chunks)
            finally:
                os.close(descriptor)

        try:
            directory_stat = os.fstat(directory)
            if (
                not stat.S_ISDIR(directory_stat.st_mode)
                or directory_stat.st_uid not in {0, os.getuid()}
                or directory_stat.st_mode & 0o022
                or set(os.listdir(directory))
                != {
                    "source_and_derived_arrays.npz",
                    "provenance.json",
                    "checksums.json",
                }
            ):
                raise ValueError("analysis provenance directory is unsafe")
            arrays_bytes = read_member(
                "source_and_derived_arrays.npz",
                _MAXIMUM_ANALYSIS_PROVENANCE_FILE_BYTES,
            )
            metadata_bytes = read_member(
                "provenance.json",
                _MAXIMUM_ANALYSIS_PROVENANCE_FILE_BYTES,
            )
            checksums_bytes = read_member("checksums.json", 1024 * 1024)
        except (OSError, ValueError) as error:
            raise ValueError("analysis provenance durable bytes changed") from error
        finally:
            os.close(directory)
        if (
            hashlib.sha256(arrays_bytes).hexdigest() != arrays_sha256
            or hashlib.sha256(metadata_bytes).hexdigest() != metadata_sha256
        ):
            raise ValueError("analysis provenance durable bytes changed")
        try:
            metadata_text = metadata_bytes.decode("utf-8")
            checksums_text = checksums_bytes.decode("utf-8")
            checksums = json.loads(checksums_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("analysis provenance checksums are invalid") from error
        expected_checksums = {
            "source_and_derived_arrays.npz": arrays_sha256,
            "provenance.json": metadata_sha256,
        }
        if (
            checksums != expected_checksums
            or checksums_text != _json_text(expected_checksums)
        ):
            raise ValueError("analysis provenance checksums are invalid")
        return path, arrays_bytes, metadata_text

    @staticmethod
    def _analysis_provenance_raw_resolution_time(
        *,
        provenance_kind: str,
        metadata_text: str,
        expected_receipt_digest: str,
    ) -> str:
        """Recover the signed/typed raw-resolution time from durable metadata."""

        try:
            metadata = json.loads(metadata_text)
            if not isinstance(metadata, dict):
                raise TypeError
            if provenance_kind == "holdout":
                raw = metadata["global_resolution"]
                if not isinstance(raw, dict):
                    raise TypeError
                values = dict(raw)
                retained_digest = values.pop("receipt_digest")
                bindings = values.get("slot_identity_bindings")
                if not isinstance(bindings, list):
                    raise TypeError
                values["slot_identity_bindings"] = tuple(
                    (str(item[0]), str(item[1]))
                    for item in bindings
                    if isinstance(item, list) and len(item) == 2
                )
                if len(values["slot_identity_bindings"]) != len(bindings):
                    raise ValueError
                resolution = GlobalRawVolumeResolutionReceipt(
                    **cast(Any, values)
                )
                if (
                    retained_digest != expected_receipt_digest
                    or resolution.receipt_digest != expected_receipt_digest
                    or raw
                    != resolution.payload
                    | {"receipt_digest": resolution.receipt_digest}
                ):
                    raise ValueError
                return resolution.resolved_at
            if provenance_kind == "operational":
                raw = metadata["raw_resolution"]
                if not isinstance(raw, dict):
                    raise TypeError
                resolution = (
                    _operational_raw_volume_resolution_receipt_from_json(
                        _json_text(raw),
                        expected_digest=expected_receipt_digest,
                    )
                )
                return resolution.resolved_at
            raise ValueError
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "analysis provenance raw resolution is invalid"
            ) from error

    @staticmethod
    def _operational_provenance_plan_from_json(
        text: str,
        *,
        expected_digest: str,
    ) -> OperationalAnalysisInputProvenancePlan:
        """Decode the complete registered operational plan preimage."""

        try:
            payload = json.loads(text)
            if not isinstance(payload, dict) or text != _json_text(payload):
                raise TypeError
            values = dict(payload)
            retained_digest = values.pop("plan_digest")

            input_values = dict(values["input_plan"])
            retained_input_digest = input_values.pop("plan_digest")
            input_values["valid_times"] = tuple(input_values["valid_times"])
            input_plan = NeuralPriorInputPlan(**cast(Any, input_values))
            if retained_input_digest != input_plan.plan_digest:
                raise ValueError
            values["input_plan"] = input_plan

            slots: list[RawObservationSlotPlan] = []
            for retained_slot in values["raw_observation_slot_plans"]:
                slot_values = dict(retained_slot)
                retained_slot_digest = slot_values.pop("slot_digest")
                slot = RawObservationSlotPlan(**cast(Any, slot_values))
                if retained_slot_digest != slot.slot_digest:
                    raise ValueError
                slots.append(slot)
            values["raw_observation_slot_plans"] = tuple(slots)

            trust_values = dict(values["raw_ingestor_trust_store"])
            retained_trust_digest = trust_values.pop("content_digest")
            trust_values["authorities"] = tuple(
                tuple(item) for item in trust_values["authorities"]
            )
            raw_trust = RawIngestorTrustStore(**cast(Any, trust_values))
            if retained_trust_digest != raw_trust.content_digest:
                raise ValueError
            values["raw_ingestor_trust_store"] = raw_trust

            geometry_values = dict(values["range_geometry_contract"])
            retained_geometry_digest = geometry_values.pop("contract_digest")
            for name in (
                "range_regime_labels",
                "radial_distance_edges_m",
                "radar_site_digests",
                "radar_site_location_digests",
            ):
                if name in geometry_values:
                    geometry_values[name] = tuple(geometry_values[name])
            if "radar_projected_xy_m" in geometry_values:
                geometry_values["radar_projected_xy_m"] = tuple(
                    tuple(point)
                    for point in geometry_values["radar_projected_xy_m"]
                )
            geometry: RangeGeometryContract | MosaicRangeGeometryContract
            if geometry_values.get("contract") == (
                "mosaic-horizontal-range-geometry-contract-v3"
            ):
                geometry = MosaicRangeGeometryContract(
                    **cast(Any, geometry_values)
                )
            else:
                geometry = RangeGeometryContract(**cast(Any, geometry_values))
            if retained_geometry_digest != geometry.contract_digest:
                raise ValueError
            values["range_geometry_contract"] = geometry

            issuance_values = dict(values["operational_issuance_domain_plan"])
            retained_issuance_digest = issuance_values.pop("plan_digest")
            issuance_values["lead_minutes"] = tuple(
                issuance_values["lead_minutes"]
            )
            issuance = OperationalIssuanceDomainPlan(
                **cast(Any, issuance_values)
            )
            if retained_issuance_digest != issuance.plan_digest:
                raise ValueError
            values["operational_issuance_domain_plan"] = issuance
            plan = OperationalAnalysisInputProvenancePlan(
                **cast(Any, values)
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "registered operational provenance plan is invalid"
            ) from error
        if (
            retained_digest != expected_digest
            or plan.plan_digest != expected_digest
            or text != _json_text(plan.payload | {"plan_digest": plan.plan_digest})
        ):
            raise ValueError("registered operational provenance plan changed")
        return plan

    @staticmethod
    def _issue_analysis_provenance_preparation_receipt(
        *,
        artifact_digest: str,
        provenance_kind: str,
        provenance_plan_digest: str,
        input_plan_digest: str,
        raw_resolution_receipt_digest: str,
        payload_json: str,
        payload_committed_at: str,
        deadline: str,
        ledger_instance_digest: str,
        side_effect_digest: str,
        authority_trust_store: _PromotionDeploymentAuthorityTrustStore,
        signer: DeploymentAuthoritySigner,
    ) -> tuple[str, str]:
        if re.fullmatch(r"[0-9a-f]{64}", ledger_instance_digest) is None:
            raise ValueError("provenance ledger instance digest is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", side_effect_digest) is None:
            raise ValueError("provenance side-effect digest is invalid")
        prepared_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        _trusted_authority_key(
            authority_trust_store,
            authority_id=signer.authority_id,
            public_key_hex=signer.public_key_hex,
            role="ledger_issuance",
            issued_at=prepared_at,
        )
        if ledger_instance_digest not in (
            authority_trust_store.ledger_instance_digests.get(
                signer.authority_id,
                frozenset(),
            )
        ):
            raise ValueError("provenance signer is not bound to this ledger")
        payload_commit = _canonical_utc_datetime(
            payload_committed_at, "payload_committed_at"
        )
        if not (
            payload_commit
            <= _canonical_utc_datetime(prepared_at, "prepared_at")
            <= _canonical_utc_datetime(deadline, "decision_deadline")
        ):
            raise ValueError("provenance payload missed its durable deadline")
        unsigned: dict[str, object] = {
            "contract": "analysis-input-provenance-preparation-receipt-v3",
            "artifact_digest": artifact_digest,
            "provenance_kind": provenance_kind,
            "provenance_plan_digest": provenance_plan_digest,
            "input_plan_digest": input_plan_digest,
            "raw_resolution_receipt_digest": raw_resolution_receipt_digest,
            "payload_json_digest": hashlib.sha256(
                payload_json.encode("utf-8")
            ).hexdigest(),
            "payload_committed_at": payload_committed_at,
            "prepared_at": prepared_at,
            "ledger_instance_digest": ledger_instance_digest,
            "side_effect_digest": side_effect_digest,
            "authority_id": signer.authority_id,
            "authority_public_key_hex": signer.public_key_hex,
            "authority_signature_hex": "",
        }
        payload = dict(unsigned)
        payload["authority_signature_hex"] = signer.sign(
            _json_digest(unsigned).encode("ascii")
        ).hex()
        receipt_digest = _json_digest(payload)
        return (
            _json_text(payload | {"receipt_digest": receipt_digest}),
            receipt_digest,
        )

    @staticmethod
    def _validate_analysis_provenance_preparation_receipt(
        receipt_json: str,
        receipt_digest: str,
        *,
        artifact_digest: str,
        provenance_kind: str,
        provenance_plan_digest: str,
        input_plan_digest: str,
        raw_resolution_receipt_digest: str,
        payload_json: str,
        payload_committed_at: str,
        deadline: str,
        ledger_instance_digest: str,
        side_effect_digest: str,
        authority_trust_store: _PromotionDeploymentAuthorityTrustStore,
    ) -> None:
        try:
            payload = json.loads(receipt_json)
            if not isinstance(payload, dict):
                raise TypeError
            retained = dict(payload)
            stored_digest = retained.pop("receipt_digest")
            signature = str(retained["authority_signature_hex"])
            unsigned = dict(retained)
            unsigned["authority_signature_hex"] = ""
            if (
                set(retained)
                != {
                    "contract",
                    "artifact_digest",
                    "provenance_kind",
                    "provenance_plan_digest",
                    "input_plan_digest",
                    "raw_resolution_receipt_digest",
                    "payload_json_digest",
                    "payload_committed_at",
                    "prepared_at",
                    "ledger_instance_digest",
                    "side_effect_digest",
                    "authority_id",
                    "authority_public_key_hex",
                    "authority_signature_hex",
                }
                or receipt_json != _json_text(payload)
                or stored_digest != receipt_digest
                or _json_digest(retained) != receipt_digest
                or retained["contract"]
                != "analysis-input-provenance-preparation-receipt-v3"
                or retained["artifact_digest"] != artifact_digest
                or retained["provenance_kind"] != provenance_kind
                or retained["provenance_plan_digest"]
                != provenance_plan_digest
                or retained["input_plan_digest"] != input_plan_digest
                or retained["raw_resolution_receipt_digest"]
                != raw_resolution_receipt_digest
                or retained["payload_json_digest"]
                != hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                or retained["payload_committed_at"] != payload_committed_at
                or retained["ledger_instance_digest"]
                != ledger_instance_digest
                or retained["side_effect_digest"] != side_effect_digest
                or _canonical_utc_datetime(
                    str(retained["payload_committed_at"]),
                    "payload_committed_at",
                )
                > _canonical_utc_datetime(
                    str(retained["prepared_at"]),
                    "prepared_at",
                )
                or _canonical_utc_datetime(
                    str(retained["prepared_at"]),
                    "prepared_at",
                )
                > _canonical_utc_datetime(deadline, "decision_deadline")
            ):
                raise ValueError
            authority_id = str(retained["authority_id"])
            authority_public_key_hex = str(
                retained["authority_public_key_hex"]
            )
            key = _trusted_authority_key(
                authority_trust_store,
                authority_id=authority_id,
                public_key_hex=authority_public_key_hex,
                role="ledger_issuance",
                issued_at=str(retained["prepared_at"]),
            )
            if ledger_instance_digest not in (
                authority_trust_store.ledger_instance_digests.get(
                    authority_id,
                    frozenset(),
                )
            ):
                raise ValueError
            key.verify(
                bytes.fromhex(signature),
                _json_digest(unsigned).encode("ascii"),
            )
        except (
            InvalidSignature,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "analysis provenance preparation receipt is invalid"
            ) from error

    def _registered_analysis_provenance_context(
        self,
        *,
        artifact_digest: str,
        provenance_kind: str,
        provenance_plan_digest: str,
        case_id: str,
        input_plan_digest: str,
        raw_resolution_receipt_digest: str,
        payload_json: str,
        analysis_processor_trust_store_path: str | Path,
    ) -> tuple[
        NeuralPriorInputPlan,
        AnalysisInputDerivationArtifact,
        str,
        str,
        _PromotionDeploymentAuthorityTrustStore,
        NeuralPriorHoldoutPlan | OperationalAnalysisInputProvenancePlan,
    ]:
        """Derive provenance authority only from a registered root-bound plan."""

        try:
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):
                raise TypeError
            derivation_payload = dict(payload)
            retained_artifact_digest = derivation_payload.pop(
                "artifact_digest"
            )
            embedded_input_plan = derivation_payload.pop("input_plan")
            if not isinstance(embedded_input_plan, dict):
                raise TypeError
            derivation = _analysis_input_derivation_from_json(
                _json_text(derivation_payload),
                expected_digest=artifact_digest,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "prepared provenance derivation is invalid"
            ) from error
        if retained_artifact_digest != artifact_digest:
            raise ValueError("prepared provenance artifact identity changed")

        expected_trust_digest: str | None = None
        if provenance_kind == "holdout":
            try:
                registered = self.load_neural_prior_holdout_plan(
                    provenance_plan_digest
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "prepared provenance lacks its registered holdout plan"
                ) from error
            if type(registered) is not NeuralPriorHoldoutPlan:
                raise ValueError(
                    "prepared provenance requires a current holdout plan"
                )
            planned_case = registered.case(case_id)
            try:
                registered_input_plan = next(
                    item
                    for item in registered.input_plans
                    if item.plan_digest == input_plan_digest
                )
            except StopIteration as error:
                raise ValueError(
                    "prepared provenance input plan is not registered"
                ) from error
            if planned_case.input_plan_digest != input_plan_digest:
                raise ValueError(
                    "prepared provenance case disagrees with its plan"
                )
            expected_authority_id = registered.analysis_processor_id
            expected_authority_public_key_hex = (
                registered.analysis_processor_public_key_hex
            )
            provenance_plan: (
                NeuralPriorHoldoutPlan
                | OperationalAnalysisInputProvenancePlan
            ) = registered
        elif provenance_kind == "operational":
            with self._connect() as connection:
                plan_row = connection.execute(
                    "SELECT input_plan_digest,payload_json FROM "
                    "operational_analysis_input_provenance_plans "
                    "WHERE plan_digest = ?",
                    (provenance_plan_digest,),
                ).fetchone()
            try:
                if plan_row is None:
                    raise KeyError
                operational_plan = self._operational_provenance_plan_from_json(
                    str(plan_row[1]),
                    expected_digest=provenance_plan_digest,
                )
                registered_input_plan = operational_plan.input_plan
                expected_case_id = operational_plan.plan_id
                expected_authority_id = operational_plan.analysis_processor_id
                expected_authority_public_key_hex = (
                    operational_plan.analysis_processor_public_key_hex
                )
                expected_trust_digest = (
                    operational_plan.analysis_processor_trust_store_digest
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise ValueError(
                    "prepared provenance lacks its registered operational plan"
                ) from error
            if (
                str(plan_row[0]) != input_plan_digest
                or registered_input_plan.plan_digest != input_plan_digest
                or expected_case_id != case_id
            ):
                raise ValueError(
                    "prepared provenance operational plan changed"
                )
            provenance_plan = operational_plan
        else:
            raise ValueError("prepared provenance kind is invalid")

        embedded_input = dict(embedded_input_plan)
        embedded_input_digest = embedded_input.pop("plan_digest", None)
        valid_times = embedded_input.get("valid_times")
        if not isinstance(valid_times, list):
            raise ValueError("prepared provenance input plan is invalid")
        embedded_input["valid_times"] = tuple(valid_times)
        try:
            decoded_embedded_input = NeuralPriorInputPlan(
                **cast(Any, embedded_input)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "prepared provenance input plan is invalid"
            ) from error
        if (
            embedded_input_digest != registered_input_plan.plan_digest
            or decoded_embedded_input != registered_input_plan
            or derivation.case_id != case_id
            or derivation.input_plan_digest != input_plan_digest
            or derivation.global_raw_resolution_receipt_digest
            != raw_resolution_receipt_digest
            or derivation.processor_id != expected_authority_id
            or derivation.processor_public_key_hex
            != expected_authority_public_key_hex
            or payload_json
            != _json_text(
                derivation.payload
                | {
                    "artifact_digest": derivation.artifact_digest,
                    "input_plan": registered_input_plan.payload
                    | {"plan_digest": registered_input_plan.plan_digest},
                }
            )
        ):
            raise ValueError(
                "prepared provenance disagrees with its registered plan"
            )
        authority_trust = _load_promotion_deployment_authority_trust_store(
            analysis_processor_trust_store_path
        )
        if (
            expected_trust_digest is not None
            and authority_trust.content_digest != expected_trust_digest
        ):
            raise ValueError(
                "prepared provenance processor trust snapshot changed"
            )
        _trusted_authority_key(
            authority_trust,
            authority_id=expected_authority_id,
            public_key_hex=expected_authority_public_key_hex,
            role="analysis_processor",
            issued_at=derivation.processed_at,
        )
        return (
            registered_input_plan,
            derivation,
            expected_authority_id,
            expected_authority_public_key_hex,
            authority_trust,
            provenance_plan,
        )

    @staticmethod
    def _decode_provenance_raw_observations(
        raw_payloads: object,
        arrays: Mapping[str, Tensor],
    ) -> tuple[RawObservationResolutionReceipt, ...]:
        """Rebuild signed raw receipts from one pinned NPZ byte snapshot."""

        if not isinstance(raw_payloads, list) or not raw_payloads:
            raise ValueError("analysis provenance raw products are incomplete")
        ordered_payloads = sorted(
            raw_payloads,
            key=lambda item: (
                (
                    item["raw_grid_volume"]["acquisition_valid_time"]
                    if isinstance(item, dict)
                    and item.get("contract")
                    == "resolved-raw-observation-receipt-v1"
                    else item["acquisition_valid_time"]
                ),
                (
                    item["raw_grid_volume"]["radar_site_digest"]
                    if isinstance(item, dict)
                    and item.get("contract")
                    == "resolved-raw-observation-receipt-v1"
                    else item["radar_site_digest"]
                ),
            ),
        )
        raw_roles = (
            "raw_source_reflectivity_bits",
            "raw_source_qc_flags",
            "raw_source_quality_bits",
            "raw_source_observation_std_bits",
        )
        present_count = sum(
            isinstance(item, dict)
            and item.get("contract")
            == "resolved-raw-observation-receipt-v1"
            for item in ordered_payloads
        )
        if any(
            role not in arrays or arrays[role].shape[0] != present_count
            for role in raw_roles
        ):
            raise ValueError("analysis provenance raw product count is invalid")
        receipts: list[RawObservationResolutionReceipt] = []
        raw_index = 0
        for retained_receipt in ordered_payloads:
            if not isinstance(retained_receipt, dict):
                raise ValueError("analysis provenance raw receipt is invalid")
            if retained_receipt.get("contract") == (
                "missing-raw-observation-receipt-v1"
            ):
                missing_values = dict(retained_receipt)
                retained_receipt_digest = missing_values.pop(
                    "receipt_digest", None
                )
                retained_identity_digest = missing_values.pop(
                    "resolution_identity_digest", None
                )
                missing = MissingRawObservationReceipt(
                    **cast(Any, missing_values)
                )
                if (
                    missing.receipt_digest != retained_receipt_digest
                    or missing.resolution_identity_digest
                    != retained_identity_digest
                ):
                    raise ValueError("analysis provenance missing receipt changed")
                receipts.append(missing)
                continue
            try:
                retained_volume = dict(retained_receipt["raw_grid_volume"])
                rebuilt_volume = (
                    CanonicalRawGridVolumeArtifact.from_encoded_tensors(
                        raw_reflectivity_bits=(
                            arrays["raw_source_reflectivity_bits"][raw_index]
                        ),
                        raw_qc_flags=arrays["raw_source_qc_flags"][raw_index],
                        raw_quality_bits=(
                            arrays["raw_source_quality_bits"][raw_index]
                        ),
                        raw_observation_std_bits=(
                            arrays["raw_source_observation_std_bits"][raw_index]
                        ),
                        radar_site_digest=cast(
                            str, retained_volume["radar_site_digest"]
                        ),
                        acquisition_valid_time=cast(
                            str, retained_volume["acquisition_valid_time"]
                        ),
                        canonical_scan_identity_digest=cast(
                            str,
                            retained_volume["canonical_scan_identity_digest"],
                        ),
                        radar_product_digest=cast(
                            str, retained_volume["radar_product_digest"]
                        ),
                        grid_contract_digest=cast(
                            str, retained_volume["grid_contract_digest"]
                        ),
                    )
                )
                identity_values = dict(retained_receipt["raw_volume_identity"])
                retained_identity_digest = identity_values.pop(
                    "identity_digest", None
                )
                identity = CanonicalRawVolumeIdentity(
                    **cast(Any, identity_values)
                )
                attestation_values = dict(
                    retained_receipt["raw_volume_attestation"]
                )
                retained_attestation_digest = attestation_values.pop(
                    "attestation_digest", None
                )
                attestation = RawVolumeAttestation(
                    **cast(Any, attestation_values)
                )
                receipt = ResolvedRawObservationReceipt(
                    slot_plan_digest=cast(
                        str, retained_receipt["slot_plan_digest"]
                    ),
                    raw_grid_volume=rebuilt_volume,
                    raw_volume_identity=identity,
                    raw_volume_attestation=attestation,
                    contract=cast(str, retained_receipt["contract"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "analysis provenance raw receipt is invalid"
                ) from error
            if (
                retained_volume
                != rebuilt_volume.payload
                | {"artifact_digest": rebuilt_volume.artifact_digest}
                or identity.identity_digest != retained_identity_digest
                or attestation.attestation_digest
                != retained_attestation_digest
                or receipt.receipt_digest
                != retained_receipt.get("receipt_digest")
            ):
                raise ValueError("analysis provenance raw receipt changed")
            receipts.append(receipt)
            raw_index += 1
        return tuple(receipts)

    def _validate_recoverable_analysis_provenance_replay(
        self,
        *,
        provenance_kind: str,
        provenance_plan_digest: str,
        case_id: str,
        input_plan: NeuralPriorInputPlan,
        derivation: AnalysisInputDerivationArtifact,
        raw_resolution_receipt_digest: str,
        arrays_bytes: bytes,
        metadata_text: str,
        current_raw_trust: RawIngestorTrustStore,
        provenance_plan: (
            NeuralPriorHoldoutPlan
            | OperationalAnalysisInputProvenancePlan
        ),
    ) -> None:
        """Replay the same raw/slot/QC invariants used by a normal append."""

        try:
            metadata = json.loads(metadata_text)
            if (
                not isinstance(metadata, dict)
                or metadata_text != _json_text(metadata)
            ):
                raise TypeError
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("analysis provenance metadata is invalid") from error

        if provenance_kind == "holdout":
            if type(provenance_plan) is not NeuralPriorHoldoutPlan:
                raise ValueError("holdout provenance plan type changed")
            plan = provenance_plan
            planned_case = plan.case(case_id)
            sampling = next(
                item
                for item in plan.meteorological_sampling_units
                if item.sampling_unit_digest
                == planned_case.meteorological_sampling_unit_digest
            )
            slots = {
                item.slot_digest: item
                for item in plan.raw_observation_slot_plans
                if item.slot_digest in sampling.raw_observation_slot_digests
            }
            pinned_raw_trust = plan.raw_ingestor_trust_store
            range_band = next(
                item
                for item in plan.range_band_contracts
                if item.contract_digest == planned_case.range_band_contract_digest
            )
            geometry = next(
                item
                for item in plan.range_geometry_contracts
                if item.contract_digest
                == range_band.range_geometry_contract_digest
            )
            expected_keys = {
                "contract",
                "holdout_plan_digest",
                "case_id",
                "raw_ingestor_trust_store_digest",
                "input_plan",
                "resolved_raw_observations",
                "global_resolution",
                "analysis_input_derivation",
                "resolved_source_coverage",
            }
            if (
                set(metadata) != expected_keys
                or metadata["contract"] != "analysis-input-provenance-bundle-v1"
                or metadata["holdout_plan_digest"] != plan.plan_digest
                or metadata["case_id"] != case_id
                or metadata["raw_ingestor_trust_store_digest"]
                != current_raw_trust.content_digest
            ):
                raise ValueError("holdout provenance metadata changed")
        elif provenance_kind == "operational":
            if type(provenance_plan) is not OperationalAnalysisInputProvenancePlan:
                raise ValueError("operational provenance plan type changed")
            operational_plan = provenance_plan
            slots = {
                item.slot_digest: item
                for item in operational_plan.raw_observation_slot_plans
            }
            pinned_raw_trust = operational_plan.raw_ingestor_trust_store
            geometry = operational_plan.range_geometry_contract
            expected_keys = {
                "contract",
                "provenance_kind",
                "provenance_plan",
                "input_plan",
                "resolved_raw_observations",
                "raw_resolution",
                "analysis_input_derivation",
                "resolved_source_coverage",
            }
            if (
                set(metadata) != expected_keys
                or metadata["contract"] != "analysis-input-provenance-bundle-v2"
                or metadata["provenance_kind"] != "operational"
                or _json_text(metadata["provenance_plan"])
                != _json_text(
                    operational_plan.payload
                    | {"plan_digest": operational_plan.plan_digest}
                )
            ):
                raise ValueError("operational provenance metadata changed")
        else:
            raise ValueError("analysis provenance kind is invalid")

        if _json_text(metadata["input_plan"]) != _json_text(
            input_plan.payload | {"plan_digest": input_plan.plan_digest}
        ) or _json_text(metadata["analysis_input_derivation"]) != _json_text(
            derivation.payload | {"artifact_digest": derivation.artifact_digest}
        ):
            raise ValueError("analysis provenance registered preimage changed")

        coverage_payload = metadata["resolved_source_coverage"]
        expected_array_names = {
            "raw_source_reflectivity_bits",
            "raw_source_qc_flags",
            "raw_source_quality_bits",
            "raw_source_observation_std_bits",
            "derived_input_frames",
            "derived_qc_valid_mask",
            "derived_quality_weight",
            "derived_observation_std_dbz",
        }
        if derivation.background_frames_digest is not None:
            expected_array_names.add("background_frames_dbz")
        coverage_roles = {
            "source_radar_index_map",
            "input_history_source_radar_index_map",
            "outage_mask",
            "dynamic_qc_valid_mask",
            "nominal_source_coverage_mask",
            "resolved_source_coverage_mask",
        }
        if coverage_payload is not None:
            expected_array_names |= coverage_roles
        try:
            with zipfile.ZipFile(io.BytesIO(arrays_bytes)) as archive:
                infos = archive.infolist()
                retained_names = {
                    item.filename.removesuffix(".npy")
                    for item in infos
                    if item.filename.endswith(".npy")
                }
                if (
                    len(infos) > 16
                    or any(item.is_dir() for item in infos)
                    or len(retained_names) != len(infos)
                    or retained_names != expected_array_names
                    or sum(item.file_size for item in infos)
                    > _MAXIMUM_ANALYSIS_PROVENANCE_EXPANDED_BYTES
                ):
                    raise ValueError
            with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as archive:
                arrays = {
                    name: torch.from_numpy(np.array(archive[name], copy=True))
                    for name in archive.files
                }
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError("analysis provenance tensor archive is invalid") from error

        receipts = self._decode_provenance_raw_observations(
            metadata["resolved_raw_observations"], arrays
        )
        receipts_by_slot = {item.slot_plan_digest: item for item in receipts}
        if (
            not receipts
            or len(receipts_by_slot) != len(receipts)
            or set(receipts_by_slot) != set(slots)
        ):
            raise ValueError("analysis provenance slot coverage changed")
        for slot_digest, receipt in receipts_by_slot.items():
            _validate_current_raw_ingestor_receipt(
                receipt,
                slots[slot_digest],
                pinned_trust_store=pinned_raw_trust,
                current_trust_store=current_raw_trust,
            )
        ordered_receipts = tuple(
            sorted(
                receipts,
                key=lambda item: (
                    item.acquisition_valid_time,
                    item.radar_site_digest,
                ),
            )
        )
        expected_bindings = tuple(
            sorted(
                (
                    item.slot_plan_digest,
                    item.resolution_identity_digest,
                )
                for item in ordered_receipts
            )
        )
        if provenance_kind == "holdout":
            resolution_values = dict(metadata["global_resolution"])
            retained_resolution_digest = resolution_values.pop(
                "receipt_digest", None
            )
            resolution_values["slot_identity_bindings"] = tuple(
                tuple(item)
                for item in resolution_values["slot_identity_bindings"]
            )
            resolution = GlobalRawVolumeResolutionReceipt(
                **cast(Any, resolution_values)
            )
            assert type(provenance_plan) is NeuralPriorHoldoutPlan
            reservation = (
                provenance_plan.promotion_experiment_family
                .global_sampling_reservation
            )
            if (
                retained_resolution_digest != raw_resolution_receipt_digest
                or resolution.receipt_digest != raw_resolution_receipt_digest
                or resolution.slot_identity_bindings != expected_bindings
                or resolution.reservation_receipt_digest
                != reservation.receipt_digest
                or resolution.experiment_scope_digest
                != reservation.experiment_scope_digest
                or resolution.registry_id != reservation.registry_id
                or resolution.authority_id != reservation.authority_id
                or resolution.authority_public_key_hex
                != reservation.authority_public_key_hex
            ):
                raise ValueError("holdout raw resolution changed")
        else:
            resolution = _operational_raw_volume_resolution_receipt_from_json(
                _json_text(metadata["raw_resolution"]),
                expected_digest=raw_resolution_receipt_digest,
            )
            assert type(provenance_plan) is OperationalAnalysisInputProvenancePlan
            if (
                resolution.provenance_plan_digest != provenance_plan_digest
                or resolution.input_plan_digest != input_plan.plan_digest
                or resolution.slot_identity_bindings != expected_bindings
                or any(
                    item.provenance_plan_digest != provenance_plan_digest
                    or item.authority_id
                    != provenance_plan.analysis_processor_id
                    or item.authority_public_key_hex
                    != provenance_plan.analysis_processor_public_key_hex
                    for item in resolution.history_entries
                )
            ):
                raise ValueError("operational raw resolution changed")

        resolved_at = _canonical_utc_datetime(
            resolution.resolved_at,
            "raw_resolution_resolved_at",
        )
        if any(
            _canonical_utc_datetime(receipt.observed_at, "raw_observed_at")
            > resolved_at
            for receipt in ordered_receipts
        ):
            raise ValueError("raw observation arrived after its resolution")

        coverage: ResolvedSourceCoverageArtifact | None = None
        if coverage_payload is not None:
            if not isinstance(coverage_payload, dict):
                raise ValueError("analysis source coverage is invalid")
            coverage_values = dict(coverage_payload)
            retained_coverage_digest = coverage_values.pop(
                "artifact_digest", None
            )
            coverage = object.__new__(ResolvedSourceCoverageArtifact)
            for name, value in coverage_values.items():
                if name in {"source_radar_site_digests", "resolved_cell_counts"}:
                    value = tuple(cast(list[object], value))
                object.__setattr__(coverage, name, value)
            for name, value in (
                ("_source_radar_index_map", arrays["source_radar_index_map"]),
                (
                    "_input_history_source_radar_index_map",
                    arrays["input_history_source_radar_index_map"],
                ),
                ("_outage_mask", arrays["outage_mask"]),
                ("_dynamic_qc_valid_mask", arrays["dynamic_qc_valid_mask"]),
                (
                    "_nominal_source_coverage_mask",
                    arrays["nominal_source_coverage_mask"],
                ),
                (
                    "_resolved_mask",
                    arrays["resolved_source_coverage_mask"],
                ),
            ):
                object.__setattr__(coverage, name, value)
            object.__setattr__(
                coverage, "artifact_digest", retained_coverage_digest
            )
            validate_resolved_source_coverage_artifact(coverage)
            if (
                type(geometry) is not MosaicRangeGeometryContract
                or coverage.case_id != case_id
                or coverage.grid_contract_digest != input_plan.grid_contract_digest
                or coverage.source_radar_registry_digest
                != geometry.source_radar_registry_digest
                or derivation.source_selection_evidence_digest
                != coverage.artifact_digest
            ):
                raise ValueError("analysis mosaic source coverage changed")
        elif (
            type(geometry) is not RangeGeometryContract
            or derivation.source_selection_evidence_digest
            != _json_digest({"contract": "single-site-source-selection-v1"})
        ):
            raise ValueError("analysis single-site source coverage changed")

        derived_frames, derived_masks, derived_quality, derived_std = (
            _derive_analysis_inputs_from_raw_products(
                input_plan=input_plan,
                resolved_raw_observations=ordered_receipts,
                resolved_source_coverage=coverage,
            )
        )
        source_available = (
            torch.ones_like(derived_masks, dtype=torch.bool)
            if coverage is None
            else coverage.input_history_source_radar_index_map >= 0
        )
        if (
            not torch.equal(arrays["derived_input_frames"], derived_frames)
            or not torch.equal(arrays["derived_qc_valid_mask"], derived_masks)
            or not torch.equal(arrays["derived_quality_weight"], derived_quality)
            or not torch.equal(arrays["derived_observation_std_dbz"], derived_std)
            or derivation.resolved_raw_observation_receipt_digests
            != tuple(sorted(item.receipt_digest for item in ordered_receipts))
            or derivation.canonical_raw_volume_identity_digests
            != tuple(
                sorted(
                    item.raw_volume_identity.identity_digest
                    for item in ordered_receipts
                    if type(item) is ResolvedRawObservationReceipt
                )
            )
            or derivation.input_frames_digest != tensor_digest(derived_frames)
            or derivation.observation_masks_digest != tensor_digest(derived_masks)
            or derivation.observation_quality_weight_digest
            != tensor_digest(derived_quality)
            or derivation.observation_std_dbz_digest != tensor_digest(derived_std)
            or derivation.source_available_mask_digest
            != tensor_digest(source_available)
            or derivation.learned_model_input_features_digest
            != tensor_digest(
                learned_radar_input_features(
                    derived_frames,
                    derived_masks,
                    derived_quality,
                    derived_std,
                    source_available,
                )
            )
            or (
                derivation.background_frames_digest is None
                and "background_frames_dbz" in arrays
            )
            or (
                derivation.background_frames_digest is not None
                and tensor_digest(arrays["background_frames_dbz"])
                != derivation.background_frames_digest
            )
        ):
            raise ValueError(
                "analysis provenance raw products do not reproduce input"
            )

    @staticmethod
    def _analysis_provenance_ledger_instance_digest(
        connection: sqlite3.Connection,
    ) -> str:
        row = connection.execute(
            "SELECT ledger_instance_digest FROM "
            "deployment_certificate_chain_head WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise ValueError("analysis provenance ledger instance is unavailable")
        digest = str(row[0])
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "analysis provenance ledger instance digest is invalid"
            )
        return digest

    def _validate_analysis_provenance_side_effects(
        self,
        *,
        provenance_kind: str,
        provenance_plan_digest: str,
        case_id: str,
        raw_resolution_receipt_digest: str,
        metadata_text: str,
        provenance_plan: (
            NeuralPriorHoldoutPlan
            | OperationalAnalysisInputProvenancePlan
        ),
    ) -> tuple[str, str]:
        """Verify the registry/history rows committed with a provenance bundle."""

        try:
            metadata = json.loads(metadata_text)
            if not isinstance(metadata, dict):
                raise TypeError
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("analysis provenance metadata is invalid") from error
        with self._connect() as connection:
            ledger_instance_digest = (
                self._analysis_provenance_ledger_instance_digest(connection)
            )
            if provenance_kind == "holdout":
                if type(provenance_plan) is not NeuralPriorHoldoutPlan:
                    raise ValueError("holdout provenance plan type changed")
                values = dict(metadata["global_resolution"])
                retained_digest = values.pop("receipt_digest", None)
                bindings = values.get("slot_identity_bindings")
                if not isinstance(bindings, list):
                    raise ValueError("holdout resolution bindings are invalid")
                values["slot_identity_bindings"] = tuple(
                    tuple(item) for item in bindings
                )
                resolution = GlobalRawVolumeResolutionReceipt(
                    **cast(Any, values)
                )
                family = provenance_plan.promotion_experiment_family
                registry_row = connection.execute(
                    "SELECT registry_id,registry_sequence_number,"
                    "previous_registry_root_digest,"
                    "committed_registry_root_digest,receipt_digest,"
                    "entry_kind,family_digest FROM "
                    "global_sampling_registry_entries WHERE receipt_digest = ?",
                    (raw_resolution_receipt_digest,),
                ).fetchone()
                expected_registry = (
                    resolution.registry_id,
                    resolution.registry_sequence_number,
                    resolution.previous_registry_root_digest,
                    resolution.committed_registry_root_digest,
                    resolution.receipt_digest,
                    "raw_resolution",
                    family.family_digest,
                )
                membership_rows = connection.execute(
                    "SELECT raw_observation_slot_digest,"
                    "raw_volume_identity_digest,case_id,family_digest FROM "
                    "raw_volume_resolution_memberships WHERE "
                    "global_resolution_receipt_digest = ? ORDER BY "
                    "raw_observation_slot_digest",
                    (raw_resolution_receipt_digest,),
                ).fetchall()
                expected_memberships = tuple(
                    (
                        slot_digest,
                        identity_digest,
                        case_id,
                        family.family_digest,
                    )
                    for slot_digest, identity_digest in sorted(
                        resolution.slot_identity_bindings
                    )
                )
                if (
                    retained_digest != raw_resolution_receipt_digest
                    or resolution.receipt_digest
                    != raw_resolution_receipt_digest
                    or registry_row is None
                    or tuple(registry_row) != expected_registry
                    or tuple(tuple(row) for row in membership_rows)
                    != expected_memberships
                ):
                    raise ValueError(
                        "holdout provenance registry side effects changed"
                    )
                for slot_digest, identity_digest in (
                    resolution.slot_identity_bindings
                ):
                    slot_row = connection.execute(
                        "SELECT raw_volume_identity_digest,family_digest FROM "
                        "raw_observation_slot_identity_bindings WHERE "
                        "raw_observation_slot_digest = ?",
                        (slot_digest,),
                    ).fetchone()
                    identity_row = connection.execute(
                        "SELECT family_digest FROM "
                        "promotion_raw_volume_identity_reservations WHERE "
                        "raw_volume_identity_digest = ?",
                        (identity_digest,),
                    ).fetchone()
                    if slot_row != (
                        identity_digest,
                        family.family_digest,
                    ) or identity_row != (family.family_digest,):
                        raise ValueError(
                            "holdout provenance reservation side effects changed"
                        )
                side_effect_payload: dict[str, object] = {
                    "contract": "analysis-provenance-ledger-side-effects-v1",
                    "provenance_kind": "holdout",
                    "registry_id": resolution.registry_id,
                    "registry_sequence_number": (
                        resolution.registry_sequence_number
                    ),
                    "committed_registry_root_digest": (
                        resolution.committed_registry_root_digest
                    ),
                    "raw_resolution_receipt_digest": resolution.receipt_digest,
                    "memberships": [list(item) for item in expected_memberships],
                }
            elif provenance_kind == "operational":
                if type(provenance_plan) is not (
                    OperationalAnalysisInputProvenancePlan
                ):
                    raise ValueError("operational provenance plan type changed")
                resolution = (
                    _operational_raw_volume_resolution_receipt_from_json(
                        _json_text(metadata["raw_resolution"]),
                        expected_digest=raw_resolution_receipt_digest,
                    )
                )
                for target in resolution.history_entries:
                    rows = connection.execute(
                        "SELECT sequence_number,entry_digest,"
                        "previous_entry_digest,provenance_plan_digest,"
                        "resolution_identity_digest,resolution_kind,"
                        "transition,entry_json,raw_resolution_receipt_digest,"
                        "recorded_at FROM operational_raw_resolution_history "
                        "WHERE slot_digest = ? ORDER BY sequence_number",
                        (target.slot_digest,),
                    ).fetchall()
                    previous = OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
                    previous_identity: str | None = None
                    previous_kind: str | None = None
                    anchor = connection.execute(
                        "SELECT slot_digest,anchor_digest,"
                        "provenance_artifact_digest,"
                        "raw_resolution_receipt_digest,"
                        "resolution_identity_digest,resolution_kind,"
                        "anchored_at FROM "
                        "operational_raw_resolution_legacy_anchors "
                        "WHERE slot_digest = ?",
                        (target.slot_digest,),
                    ).fetchone()
                    if anchor is not None:
                        previous, previous_identity, previous_kind = (
                            self._validate_operational_raw_resolution_legacy_anchor(
                                connection,
                                anchor,
                            )
                        )
                    found = False
                    for expected_sequence, row in enumerate(rows, start=1):
                        sequence, entry, retained_receipt = (
                            self._validate_operational_raw_history_row(
                                connection,
                                row,
                                fallback_authority_id=(
                                    provenance_plan.analysis_processor_id
                                ),
                                fallback_authority_public_key_hex=(
                                    provenance_plan
                                    .analysis_processor_public_key_hex
                                ),
                            )
                        )
                        if (
                            sequence != expected_sequence
                            or entry.previous_entry_digest != previous
                        ):
                            raise ValueError(
                                "operational raw-resolution history is broken"
                            )
                        if previous_identity is None:
                            expected_transition = "original"
                        elif entry.resolution_identity_digest == previous_identity:
                            expected_transition = "reuse"
                        elif previous_kind == "missing" and (
                            entry.resolution_kind == "resolved"
                        ):
                            expected_transition = "correction"
                        elif previous_kind == "resolved" and (
                            entry.resolution_kind == "resolved"
                        ):
                            expected_transition = "supersession"
                        elif previous_kind == "resolved" and (
                            entry.resolution_kind == "missing"
                        ):
                            expected_transition = "cancellation"
                        else:
                            raise ValueError(
                                "operational raw-resolution transition is invalid"
                            )
                        if entry.transition != expected_transition:
                            raise ValueError(
                                "operational raw-resolution transition changed"
                            )
                        if entry.entry_digest == target.entry_digest:
                            if (
                                entry != target
                                or retained_receipt
                                != raw_resolution_receipt_digest
                            ):
                                raise ValueError(
                                    "operational provenance history changed"
                                )
                            found = True
                        previous = entry.entry_digest
                        previous_identity = entry.resolution_identity_digest
                        previous_kind = entry.resolution_kind
                    if not found:
                        raise ValueError(
                            "operational provenance history is not committed"
                        )
                side_effect_payload = {
                    "contract": "analysis-provenance-ledger-side-effects-v1",
                    "provenance_kind": "operational",
                    "raw_resolution_receipt_digest": resolution.receipt_digest,
                    "history_entry_digests": sorted(
                        item.entry_digest for item in resolution.history_entries
                    ),
                }
            else:
                raise ValueError("analysis provenance kind is invalid")
        return ledger_instance_digest, _json_digest(side_effect_payload)

    def reconcile_prepared_analysis_input_provenance(
        self,
        artifact_digest: str,
        *,
        raw_ingestor_trust_store_path: str | Path,
        analysis_processor_trust_store_path: str | Path,
        provenance_commit_signer: DeploymentAuthoritySigner | None = None,
    ) -> str:
        """Idempotently activate a fully committed prepared provenance row.

        The payload transaction has already validated the raw receipts and
        processor signature. Recovery independently rehashes its immutable
        files, requires an exact-ledger preparation authorization over the
        registry/history side effects before the deadline, and checks the same current
        root-owned raw-ingestor trust snapshot before activation.
        """

        if not re.fullmatch(r"[0-9a-f]{64}", artifact_digest):
            raise ValueError("analysis provenance digest is invalid")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT provenance_kind,payload_json,arrays_sha256,"
                "metadata_sha256,path,raw_ingestor_trust_store_digest,"
                "payload_committed_at,status,usable,provenance_plan_digest,"
                "case_id,input_plan_digest,raw_resolution_receipt_digest,"
                "preparation_receipt_json,preparation_receipt_digest FROM "
                "analysis_input_provenance_commits WHERE artifact_digest = ?",
                (artifact_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown analysis provenance: {artifact_digest}")
        provenance_kind = str(row[0])
        status = str(row[7])
        if not (
            (status == "active" and int(row[8]) == 1)
            or (status == "prepared" and int(row[8]) == 0)
        ):
            raise ValueError("analysis provenance is not recoverable")
        _, arrays_bytes, metadata_text = (
            self._snapshot_analysis_input_provenance_directory(
                artifact_digest=artifact_digest,
                path_text=str(row[4]),
                arrays_sha256=str(row[2]),
                metadata_sha256=str(row[3]),
            )
        )
        raw_resolution_time = self._analysis_provenance_raw_resolution_time(
            provenance_kind=provenance_kind,
            metadata_text=metadata_text,
            expected_receipt_digest=str(row[12]),
        )
        (
            input_plan,
            derivation,
            processor_id,
            processor_public_key_hex,
            processor_trust,
            provenance_plan,
        ) = self._registered_analysis_provenance_context(
            artifact_digest=artifact_digest,
            provenance_kind=provenance_kind,
            provenance_plan_digest=str(row[9]),
            case_id=str(row[10]),
            input_plan_digest=str(row[11]),
            raw_resolution_receipt_digest=str(row[12]),
            payload_json=str(row[1]),
            analysis_processor_trust_store_path=(
                analysis_processor_trust_store_path
            ),
        )
        expected_trust_digest = str(row[5])
        current_trust = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        if current_trust.content_digest != expected_trust_digest:
            expired_at = datetime.now(timezone.utc).isoformat()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE analysis_input_provenance_commits SET "
                    "status = 'expired',usable = 0,expired_at = ? "
                    "WHERE artifact_digest = ? AND status IN ('prepared','active')",
                    (expired_at, artifact_digest),
                )
                if provenance_kind == "holdout":
                    connection.execute(
                        "UPDATE neural_prior_analysis_input_provenance SET "
                        "status = 'expired',usable = 0,expired_at = ? "
                        "WHERE artifact_digest = ? AND "
                        "status IN ('prepared','active')",
                        (expired_at, artifact_digest),
                    )
            raise ValueError("prepared provenance trust snapshot changed")
        self._validate_recoverable_analysis_provenance_replay(
            provenance_kind=provenance_kind,
            provenance_plan_digest=str(row[9]),
            case_id=str(row[10]),
            input_plan=input_plan,
            derivation=derivation,
            raw_resolution_receipt_digest=str(row[12]),
            arrays_bytes=arrays_bytes,
            metadata_text=metadata_text,
            current_raw_trust=current_trust,
            provenance_plan=provenance_plan,
        )
        ledger_instance_digest, side_effect_digest = (
            self._validate_analysis_provenance_side_effects(
                provenance_kind=provenance_kind,
                provenance_plan_digest=str(row[9]),
                case_id=str(row[10]),
                raw_resolution_receipt_digest=str(row[12]),
                metadata_text=metadata_text,
                provenance_plan=provenance_plan,
            )
        )
        payload_commit = _canonical_utc_datetime(
            str(row[6]), "payload_committed_at"
        )
        deadline = _canonical_utc_datetime(
            input_plan.decision_deadline, "decision_deadline"
        )
        if not (
            _canonical_utc_datetime(
                raw_resolution_time,
                "raw_resolution_resolved_at",
            )
            <= _canonical_utc_datetime(
                derivation.processed_at,
                "derivation_processed_at",
            )
            <= payload_commit
            <= deadline
        ):
            raise ValueError("prepared provenance missed its durable deadline")
        preparation_receipt_json = (
            None if row[13] is None else str(row[13])
        )
        preparation_receipt_digest = (
            None if row[14] is None else str(row[14])
        )
        if (
            preparation_receipt_json is None
            or preparation_receipt_digest is None
        ):
            if status != "prepared" or provenance_commit_signer is None:
                raise ValueError(
                    "prepared provenance requires a signed ledger preparation receipt"
                )
            if datetime.now(timezone.utc) > deadline:
                raise ValueError(
                    "prepared provenance lacks a predeadline ledger receipt"
                )
            (
                preparation_receipt_json,
                preparation_receipt_digest,
            ) = self._issue_analysis_provenance_preparation_receipt(
                artifact_digest=artifact_digest,
                provenance_kind=provenance_kind,
                provenance_plan_digest=str(row[9]),
                input_plan_digest=str(row[11]),
                raw_resolution_receipt_digest=str(row[12]),
                payload_json=str(row[1]),
                payload_committed_at=str(row[6]),
                deadline=input_plan.decision_deadline,
                ledger_instance_digest=ledger_instance_digest,
                side_effect_digest=side_effect_digest,
                authority_trust_store=processor_trust,
                signer=provenance_commit_signer,
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    "UPDATE analysis_input_provenance_commits SET "
                    "preparation_receipt_json=?,preparation_receipt_digest=? "
                    "WHERE artifact_digest=? AND status='prepared' AND "
                    "preparation_receipt_json IS NULL AND "
                    "preparation_receipt_digest IS NULL",
                    (
                        preparation_receipt_json,
                        preparation_receipt_digest,
                        artifact_digest,
                    ),
                )
                if updated.rowcount != 1:
                    raise ValueError("provenance preparation receipt raced")
                if provenance_kind == "holdout":
                    updated_holdout = connection.execute(
                        "UPDATE neural_prior_analysis_input_provenance SET "
                        "preparation_receipt_json=?,"
                        "preparation_receipt_digest=? WHERE artifact_digest=? "
                        "AND status='prepared' AND "
                        "preparation_receipt_json IS NULL AND "
                        "preparation_receipt_digest IS NULL",
                        (
                            preparation_receipt_json,
                            preparation_receipt_digest,
                            artifact_digest,
                        ),
                    )
                    if updated_holdout.rowcount != 1:
                        raise ValueError(
                            "holdout provenance preparation receipt raced"
                        )
        self._validate_analysis_provenance_preparation_receipt(
            preparation_receipt_json,
            preparation_receipt_digest,
            artifact_digest=artifact_digest,
            provenance_kind=provenance_kind,
            provenance_plan_digest=str(row[9]),
            input_plan_digest=str(row[11]),
            raw_resolution_receipt_digest=str(row[12]),
            payload_json=str(row[1]),
            payload_committed_at=str(row[6]),
            deadline=input_plan.decision_deadline,
            ledger_instance_digest=ledger_instance_digest,
            side_effect_digest=side_effect_digest,
            authority_trust_store=processor_trust,
        )
        try:
            receipt_payload = json.loads(preparation_receipt_json)
            if not isinstance(receipt_payload, dict):
                raise TypeError
            prepared_at = str(receipt_payload["prepared_at"])
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "analysis provenance preparation receipt is invalid"
            ) from error
        _trusted_authority_key(
            processor_trust,
            authority_id=processor_id,
            public_key_hex=processor_public_key_hex,
            role="analysis_processor",
            issued_at=prepared_at,
        )
        if payload_commit > _canonical_utc_datetime(prepared_at, "prepared_at"):
            raise ValueError("prepared provenance receipt predates its payload")
        if status == "active":
            return artifact_digest
        activated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            final_trust = _load_raw_ingestor_trust_store(
                raw_ingestor_trust_store_path
            )
            if final_trust.content_digest != expected_trust_digest:
                raise ValueError("raw-ingestor trust changed during recovery")
            updated = connection.execute(
                "UPDATE analysis_input_provenance_commits SET "
                "status = 'active',usable = 1,"
                "raw_trust_validated_at = ?,activated_at = ? "
                "WHERE artifact_digest = ? AND status = 'prepared' AND usable = 0",
                (activated_at, activated_at, artifact_digest),
            )
            if updated.rowcount != 1:
                raise ValueError("prepared provenance activation raced")
            if provenance_kind == "holdout":
                updated_holdout = connection.execute(
                    "UPDATE neural_prior_analysis_input_provenance SET "
                    "status = 'active',usable = 1,"
                    "raw_trust_validated_at = ?,activated_at = ? "
                    "WHERE artifact_digest = ? AND status = 'prepared' "
                    "AND usable = 0",
                    (activated_at, activated_at, artifact_digest),
                )
                if updated_holdout.rowcount != 1:
                    raise ValueError("holdout provenance activation raced")
        post_trust = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        if post_trust.content_digest != expected_trust_digest:
            expired_at = datetime.now(timezone.utc).isoformat()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE analysis_input_provenance_commits SET "
                    "status = 'expired',usable = 0,expired_at = ? "
                    "WHERE artifact_digest = ? AND status = 'active'",
                    (expired_at, artifact_digest),
                )
                if provenance_kind == "holdout":
                    connection.execute(
                        "UPDATE neural_prior_analysis_input_provenance SET "
                        "status = 'expired',usable = 0,expired_at = ? "
                        "WHERE artifact_digest = ? AND status = 'active'",
                        (expired_at, artifact_digest),
                    )
            raise ValueError("raw-ingestor trust changed after recovery")
        return artifact_digest

    @staticmethod
    def _operational_raw_history_plan_authority(
        connection: sqlite3.Connection,
        *,
        provenance_plan_digest: str,
        fallback_authority_id: str | None = None,
        fallback_authority_public_key_hex: str | None = None,
    ) -> tuple[str, str]:
        row = connection.execute(
            "SELECT payload_json FROM "
            "operational_analysis_input_provenance_plans "
            "WHERE plan_digest = ?",
            (provenance_plan_digest,),
        ).fetchone()
        if row is None:
            if (
                fallback_authority_id is None
                or fallback_authority_public_key_hex is None
            ):
                raise ValueError(
                    "operational raw-resolution history plan is unavailable"
                )
            return fallback_authority_id, fallback_authority_public_key_hex
        try:
            payload = json.loads(str(row[0]))
            if not isinstance(payload, dict):
                raise TypeError
            values = dict(payload)
            stored_digest = values.pop("plan_digest")
            authority_id = values["analysis_processor_id"]
            authority_key = values["analysis_processor_public_key_hex"]
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "operational raw-resolution history plan is invalid"
            ) from error
        if (
            stored_digest != provenance_plan_digest
            or _json_digest(values) != provenance_plan_digest
            or str(row[0]) != canonical
            or not isinstance(authority_id, str)
            or not authority_id
            or authority_id.strip() != authority_id
            or not isinstance(authority_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", authority_key) is None
        ):
            raise ValueError(
                "operational raw-resolution history plan changed"
            )
        return authority_id, authority_key

    @staticmethod
    def _validate_operational_raw_history_row(
        connection: sqlite3.Connection,
        row: sqlite3.Row | tuple[object, ...],
        *,
        fallback_authority_id: str | None = None,
        fallback_authority_public_key_hex: str | None = None,
    ) -> tuple[int, OperationalRawResolutionHistoryEntry, str]:
        try:
            sequence_number = int(cast(Any, row[0]))
            entry_digest = str(row[1])
            previous_entry_digest = str(row[2])
            provenance_plan_digest = str(row[3])
            resolution_identity_digest = str(row[4])
            resolution_kind = str(row[5])
            transition = str(row[6])
            entry_json = str(row[7])
            raw_resolution_receipt_digest = str(row[8])
            recorded_at = str(row[9])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "operational raw-resolution history row is invalid"
            ) from error
        entry = _operational_raw_resolution_history_entry_from_json(
            entry_json,
            expected_digest=entry_digest,
        )
        authority_id, authority_key = (
            EpisodeLedger._operational_raw_history_plan_authority(
                connection,
                provenance_plan_digest=entry.provenance_plan_digest,
                fallback_authority_id=fallback_authority_id,
                fallback_authority_public_key_hex=(
                    fallback_authority_public_key_hex
                ),
            )
        )
        if (
            sequence_number <= 0
            or previous_entry_digest != entry.previous_entry_digest
            or provenance_plan_digest != entry.provenance_plan_digest
            or resolution_identity_digest != entry.resolution_identity_digest
            or resolution_kind != entry.resolution_kind
            or transition != entry.transition
            or entry.authority_id != authority_id
            or entry.authority_public_key_hex != authority_key
            or re.fullmatch(r"[0-9a-f]{64}", raw_resolution_receipt_digest)
            is None
        ):
            raise ValueError(
                "operational raw-resolution history row changed"
            )
        _canonical_utc_datetime(recorded_at, "recorded_at")
        return sequence_number, entry, raw_resolution_receipt_digest

    def _validate_operational_raw_resolution_legacy_anchor(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row | tuple[object, ...],
    ) -> tuple[str, str, str]:
        """Rebuild one legacy anchor from its immutable provenance bytes."""

        try:
            slot_digest = str(row[0])
            anchor_digest = str(row[1])
            provenance_artifact_digest = str(row[2])
            raw_resolution_receipt_digest = str(row[3])
            resolution_identity_digest = str(row[4])
            resolution_kind = str(row[5])
            anchored_at = str(row[6])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "operational raw-resolution legacy anchor is invalid"
            ) from error
        for name, value in (
            ("legacy slot", slot_digest),
            ("legacy anchor", anchor_digest),
            ("legacy provenance", provenance_artifact_digest),
            ("legacy raw resolution", raw_resolution_receipt_digest),
            ("legacy resolution identity", resolution_identity_digest),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} digest is invalid")
        if resolution_kind not in {"resolved", "missing"}:
            raise ValueError("legacy resolution kind is invalid")
        _canonical_utc_datetime(anchored_at, "legacy anchored_at")
        expected_anchor_digest = _json_digest(
            {
                "contract": "operational-raw-resolution-legacy-anchor-v1",
                "slot_digest": slot_digest,
                "resolution_identity_digest": resolution_identity_digest,
                "resolution_kind": resolution_kind,
                "provenance_artifact_digest": provenance_artifact_digest,
                "raw_resolution_receipt_digest": raw_resolution_receipt_digest,
            }
        )
        commit = connection.execute(
            "SELECT provenance_kind,path,arrays_sha256,metadata_sha256,"
            "raw_resolution_receipt_digest,payload_committed_at FROM "
            "analysis_input_provenance_commits WHERE artifact_digest = ?",
            (provenance_artifact_digest,),
        ).fetchone()
        if (
            anchor_digest != expected_anchor_digest
            or commit is None
            or str(commit[0]) != "operational"
            or str(commit[4]) != raw_resolution_receipt_digest
            or str(commit[5]) != anchored_at
        ):
            raise ValueError(
                "operational raw-resolution legacy anchor changed"
            )
        provenance_path = self._validate_analysis_input_provenance_directory(
            artifact_digest=provenance_artifact_digest,
            path_text=str(commit[1]),
            arrays_sha256=str(commit[2]),
            metadata_sha256=str(commit[3]),
        )
        try:
            metadata_text = (provenance_path / "provenance.json").read_text(
                "utf-8"
            )
            metadata = json.loads(metadata_text)
            if not isinstance(metadata, dict) or metadata_text != _json_text(metadata):
                raise TypeError
            resolution_payload = dict(metadata["raw_resolution"])
            observations = metadata["resolved_raw_observations"]
            if not isinstance(observations, list) or any(
                not isinstance(item, dict) for item in observations
            ):
                raise TypeError
            if resolution_payload.get("contract") == (
                "operational-raw-volume-resolution-receipt-v2"
            ):
                resolution = _operational_raw_volume_resolution_receipt_from_json(
                    _json_text(resolution_payload),
                    expected_digest=raw_resolution_receipt_digest,
                )
                bindings = resolution.slot_identity_bindings
            else:
                stored_digest = resolution_payload.pop("receipt_digest")
                if (
                    set(resolution_payload)
                    != {
                        "contract",
                        "provenance_plan_digest",
                        "input_plan_digest",
                        "slot_identity_bindings",
                        "resolved_at",
                    }
                    or resolution_payload.get("contract")
                    != "operational-raw-volume-resolution-receipt-v1"
                    or stored_digest != raw_resolution_receipt_digest
                    or _json_digest(resolution_payload)
                    != raw_resolution_receipt_digest
                ):
                    raise ValueError
                raw_bindings = resolution_payload["slot_identity_bindings"]
                if not isinstance(raw_bindings, list):
                    raise TypeError
                bindings = tuple(
                    sorted((str(item[0]), str(item[1])) for item in raw_bindings)
                )
                if (
                    [list(item) for item in bindings] != raw_bindings
                    or len({item[0] for item in bindings}) != len(bindings)
                ):
                    raise ValueError
            observation = next(
                (
                    item
                    for item in observations
                    if item.get("slot_plan_digest") == slot_digest
                ),
                None,
            )
            if observation is None or sum(
                item.get("slot_plan_digest") == slot_digest
                for item in observations
            ) != 1:
                raise ValueError
            observed_kind = (
                "missing"
                if observation.get("contract")
                == "missing-raw-observation-receipt-v1"
                else "resolved"
            )
            observed_identity = str(
                observation["resolution_identity_digest"]
                if observed_kind == "missing"
                else observation["raw_volume_identity"]["identity_digest"]
            )
        except (
            KeyError,
            OSError,
            StopIteration,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "operational raw-resolution legacy anchor preimage is invalid"
            ) from error
        if (
            (slot_digest, resolution_identity_digest) not in bindings
            or observed_identity != resolution_identity_digest
            or observed_kind != resolution_kind
        ):
            raise ValueError(
                "operational raw-resolution legacy anchor preimage changed"
            )
        return anchor_digest, resolution_identity_digest, resolution_kind

    def _record_operational_raw_resolution_history(
        self,
        connection: sqlite3.Connection,
        *,
        entry: OperationalRawResolutionHistoryEntry,
        raw_resolution_receipt_digest: str,
        recorded_at: str,
        expected_authority_id: str | None = None,
        expected_authority_public_key_hex: str | None = None,
    ) -> None:
        """Append one cross-cycle slot interpretation without equivocation."""

        entry_json = json.dumps(
            entry.payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        entry = _operational_raw_resolution_history_entry_from_json(
            entry_json,
            expected_digest=entry.entry_digest,
        )
        if expected_authority_id is not None and (
            entry.authority_id != expected_authority_id
            or entry.authority_public_key_hex
            != expected_authority_public_key_hex
        ):
            raise ValueError(
                "operational raw correction authority disagrees with its plan"
            )

        retained = connection.execute(
            "SELECT sequence_number,entry_digest,previous_entry_digest,"
            "provenance_plan_digest,resolution_identity_digest,resolution_kind,"
            "transition,entry_json,raw_resolution_receipt_digest,recorded_at "
            "FROM operational_raw_resolution_history "
            "WHERE slot_digest = ? ORDER BY sequence_number DESC LIMIT 1",
            (entry.slot_digest,),
        ).fetchone()
        recorded = _canonical_utc_datetime(recorded_at, "recorded_at")
        issued = _canonical_utc_datetime(entry.issued_at, "issued_at")
        if issued > recorded:
            raise ValueError("operational raw-resolution history is future-dated")
        if retained is None:
            sequence_number = 1
            legacy_anchor = connection.execute(
                "SELECT slot_digest,anchor_digest,provenance_artifact_digest,"
                "raw_resolution_receipt_digest,resolution_identity_digest,"
                "resolution_kind,anchored_at FROM "
                "operational_raw_resolution_legacy_anchors "
                "WHERE slot_digest = ?",
                (entry.slot_digest,),
            ).fetchone()
            if legacy_anchor is None:
                expected_previous = OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
                expected_transition = "original"
            else:
                (
                    expected_previous,
                    previous_identity,
                    previous_kind,
                ) = self._validate_operational_raw_resolution_legacy_anchor(
                    connection,
                    legacy_anchor,
                )
                if entry.resolution_identity_digest == previous_identity:
                    expected_transition = "reuse"
                elif (
                    previous_kind == "missing"
                    and entry.resolution_kind == "resolved"
                ):
                    expected_transition = "correction"
                elif (
                    previous_kind == "resolved"
                    and entry.resolution_kind == "resolved"
                ):
                    expected_transition = "supersession"
                elif (
                    previous_kind == "resolved"
                    and entry.resolution_kind == "missing"
                ):
                    expected_transition = "cancellation"
                else:
                    raise ValueError(
                        "operational raw-resolution transition is invalid"
                    )
        else:
            (
                retained_sequence,
                previous_entry,
                _retained_receipt_digest,
            ) = EpisodeLedger._validate_operational_raw_history_row(
                connection,
                retained,
                fallback_authority_id=expected_authority_id,
                fallback_authority_public_key_hex=(
                    expected_authority_public_key_hex
                ),
            )
            sequence_number = retained_sequence + 1
            expected_previous = previous_entry.entry_digest
            previous_identity = previous_entry.resolution_identity_digest
            previous_kind = previous_entry.resolution_kind
            if issued < _canonical_utc_datetime(
                previous_entry.issued_at,
                "previous issued_at",
            ):
                raise ValueError(
                    "operational raw-resolution history chronology regressed"
                )
            if entry.resolution_identity_digest == previous_identity:
                expected_transition = "reuse"
            elif previous_kind == "missing" and entry.resolution_kind == "resolved":
                expected_transition = "correction"
            elif previous_kind == "resolved" and entry.resolution_kind == "resolved":
                expected_transition = "supersession"
            elif previous_kind == "resolved" and entry.resolution_kind == "missing":
                expected_transition = "cancellation"
            else:
                raise ValueError("operational raw-resolution transition is invalid")
        if (
            entry.previous_entry_digest != expected_previous
            or entry.transition != expected_transition
        ):
            raise ValueError("operational raw-resolution history equivocated")
        connection.execute(
            "INSERT INTO operational_raw_resolution_history "
            "(slot_digest,sequence_number,entry_digest,previous_entry_digest,"
            "provenance_plan_digest,resolution_identity_digest,resolution_kind,"
            "transition,entry_json,raw_resolution_receipt_digest,recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry.slot_digest,
                sequence_number,
                entry.entry_digest,
                entry.previous_entry_digest,
                entry.provenance_plan_digest,
                entry.resolution_identity_digest,
                entry.resolution_kind,
                entry.transition,
                entry_json,
                raw_resolution_receipt_digest,
                recorded_at,
            ),
        )

    def load_operational_raw_resolution_history(
        self,
        slot_digest: str,
        *,
        expected_authority_id: str | None = None,
        expected_authority_public_key_hex: str | None = None,
    ) -> tuple[OperationalRawResolutionHistoryEntry, ...]:
        """Load and independently verify the append-only history for one slot."""

        if not re.fullmatch(r"[0-9a-f]{64}", slot_digest):
            raise ValueError("operational raw slot digest is invalid")
        with self._connect() as connection:
            anchor = connection.execute(
                "SELECT slot_digest,anchor_digest,provenance_artifact_digest,"
                "raw_resolution_receipt_digest,resolution_identity_digest,"
                "resolution_kind,anchored_at FROM "
                "operational_raw_resolution_legacy_anchors "
                "WHERE slot_digest = ?",
                (slot_digest,),
            ).fetchone()
            rows = connection.execute(
                "SELECT sequence_number,entry_digest,previous_entry_digest,"
                "provenance_plan_digest,resolution_identity_digest,"
                "resolution_kind,transition,entry_json,"
                "raw_resolution_receipt_digest,recorded_at "
                "FROM operational_raw_resolution_history "
                "WHERE slot_digest = ? ORDER BY sequence_number",
                (slot_digest,),
            ).fetchall()
            entries: list[OperationalRawResolutionHistoryEntry] = []
            previous = OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST
            if anchor is not None:
                previous, _identity, _kind = (
                    self._validate_operational_raw_resolution_legacy_anchor(
                        connection,
                        anchor,
                    )
                )
            for expected_sequence, row in enumerate(rows, start=1):
                sequence, entry, _receipt_digest = (
                    self._validate_operational_raw_history_row(
                        connection,
                        row,
                        fallback_authority_id=expected_authority_id,
                        fallback_authority_public_key_hex=(
                            expected_authority_public_key_hex
                        ),
                    )
                )
                if (
                    sequence != expected_sequence
                    or entry.previous_entry_digest != previous
                    or entry.slot_digest != slot_digest
                ):
                    raise ValueError(
                        "operational raw-resolution history chain is broken"
                    )
                entries.append(entry)
                previous = entry.entry_digest
        return tuple(entries)

    def operational_raw_resolution_predecessor(
        self,
        slot_digest: str,
        *,
        expected_authority_id: str | None = None,
        expected_authority_public_key_hex: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Return the signed predecessor and prior interpretation for a slot."""

        if not re.fullmatch(r"[0-9a-f]{64}", slot_digest):
            raise ValueError("operational raw slot digest is invalid")
        with self._connect() as connection:
            retained = connection.execute(
                "SELECT sequence_number,entry_digest,previous_entry_digest,"
                "provenance_plan_digest,resolution_identity_digest,"
                "resolution_kind,transition,entry_json,"
                "raw_resolution_receipt_digest,recorded_at "
                "FROM operational_raw_resolution_history "
                "WHERE slot_digest = ? ORDER BY sequence_number DESC LIMIT 1",
                (slot_digest,),
            ).fetchone()
            if retained is not None:
                _sequence, entry, _receipt_digest = (
                    self._validate_operational_raw_history_row(
                        connection,
                        retained,
                        fallback_authority_id=expected_authority_id,
                        fallback_authority_public_key_hex=(
                            expected_authority_public_key_hex
                        ),
                    )
                )
                return (
                    entry.entry_digest,
                    entry.resolution_identity_digest,
                    entry.resolution_kind,
                )
            anchor = connection.execute(
                "SELECT slot_digest,anchor_digest,provenance_artifact_digest,"
                "raw_resolution_receipt_digest,resolution_identity_digest,"
                "resolution_kind,anchored_at FROM "
                "operational_raw_resolution_legacy_anchors "
                "WHERE slot_digest = ?",
                (slot_digest,),
            ).fetchone()
            if anchor is not None:
                return self._validate_operational_raw_resolution_legacy_anchor(
                    connection,
                    anchor,
                )
        return OPERATIONAL_RAW_RESOLUTION_GENESIS_DIGEST, None, None

    def append_operational_analysis_input_provenance(
        self,
        plan: OperationalAnalysisInputProvenancePlan,
        *,
        run: ForecastRunContract,
        resolved_raw_observations: tuple[
            RawObservationResolutionReceipt, ...
        ],
        raw_resolution: OperationalRawVolumeResolutionReceipt,
        derivation: AnalysisInputDerivationArtifact,
        resolved_source_coverage: ResolvedSourceCoverageArtifact | None,
        background_frames_dbz: Tensor | None = None,
        raw_ingestor_trust_store_path: str | Path,
        analysis_processor_trust_store_path: str | Path,
        provenance_commit_signer: DeploymentAuthoritySigner,
    ) -> str:
        """Commit provenance for a future production cycle without a holdout."""

        if type(plan) is not OperationalAnalysisInputProvenancePlan:
            raise TypeError("current operational provenance plan is required")
        run.validate_integrity()
        raw_resolution_json = json.dumps(
            raw_resolution.payload
            | {"receipt_digest": raw_resolution.receipt_digest},
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_resolution = _operational_raw_volume_resolution_receipt_from_json(
            raw_resolution_json,
            expected_digest=raw_resolution.receipt_digest,
        )
        input_plan = plan.input_plan
        slot_by_digest = {
            item.slot_digest: item for item in plan.raw_observation_slot_plans
        }
        receipts_by_slot = {
            item.slot_plan_digest: item for item in resolved_raw_observations
        }
        if len(receipts_by_slot) != len(resolved_raw_observations):
            raise ValueError("operational provenance has duplicate raw slots")
        current_raw_trust = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        analysis_trust = _load_promotion_deployment_authority_trust_store(
            analysis_processor_trust_store_path
        )
        if (
            current_raw_trust.content_digest
            != plan.raw_ingestor_trust_store.content_digest
            or analysis_trust.content_digest
            != plan.analysis_processor_trust_store_digest
        ):
            raise ValueError("operational provenance trust snapshot changed")
        _trusted_authority_key(
            analysis_trust,
            authority_id=plan.analysis_processor_id,
            public_key_hex=plan.analysis_processor_public_key_hex,
            role="analysis_processor",
            issued_at=derivation.processed_at,
        )
        for slot_digest, receipt in receipts_by_slot.items():
            slot = slot_by_digest.get(slot_digest)
            if slot is None:
                raise ValueError("operational provenance contains an unplanned slot")
            _validate_current_raw_ingestor_receipt(
                receipt,
                slot,
                pinned_trust_store=plan.raw_ingestor_trust_store,
                current_trust_store=current_raw_trust,
            )
        ordered_receipts = tuple(
            sorted(
                resolved_raw_observations,
                key=lambda item: (
                    item.acquisition_valid_time,
                    item.radar_site_digest,
                ),
            )
        )
        bindings = tuple(sorted(
            (
                item.slot_plan_digest,
                item.resolution_identity_digest,
            )
            for item in ordered_receipts
        ))
        history_by_slot = {
            item.slot_digest: item for item in raw_resolution.history_entries
        }
        if len(history_by_slot) != len(raw_resolution.history_entries):
            raise ValueError("operational raw-resolution history has duplicate slots")
        for slot_digest, receipt in receipts_by_slot.items():
            history = history_by_slot.get(slot_digest)
            expected_kind = (
                "resolved"
                if type(receipt) is ResolvedRawObservationReceipt
                else "missing"
            )
            if (
                history is None
                or history.provenance_plan_digest != plan.plan_digest
                or history.resolution_identity_digest
                != receipt.resolution_identity_digest
                or history.resolution_kind != expected_kind
                or not (
                    _canonical_utc_datetime(receipt.observed_at, "observed_at")
                    <= _canonical_utc_datetime(history.issued_at, "issued_at")
                    <= _canonical_utc_datetime(
                        raw_resolution.resolved_at, "resolved_at"
                    )
                )
            ):
                raise ValueError("operational raw-resolution history changed")
            _trusted_authority_key(
                analysis_trust,
                authority_id=history.authority_id,
                public_key_hex=history.authority_public_key_hex,
                role="analysis_processor",
                issued_at=history.issued_at,
            )
            if (
                history.authority_id != plan.analysis_processor_id
                or history.authority_public_key_hex
                != plan.analysis_processor_public_key_hex
            ):
                raise ValueError(
                    "operational raw correction authority disagrees with its plan"
                )
        if (
            set(receipts_by_slot) != set(slot_by_digest)
            or raw_resolution.provenance_plan_digest != plan.plan_digest
            or raw_resolution.input_plan_digest != input_plan.plan_digest
            or raw_resolution.slot_identity_bindings != bindings
            or derivation.case_id != plan.plan_id
            or derivation.input_plan_digest != input_plan.plan_digest
            or derivation.global_raw_resolution_receipt_digest
            != raw_resolution.receipt_digest
            or derivation.resolved_raw_observation_receipt_digests
            != tuple(sorted(item.receipt_digest for item in ordered_receipts))
            or derivation.canonical_raw_volume_identity_digests
            != tuple(
                sorted(
                    item.raw_volume_identity.identity_digest
                    for item in ordered_receipts
                    if type(item) is ResolvedRawObservationReceipt
                )
            )
            or derivation.processor_id != plan.analysis_processor_id
            or derivation.processor_public_key_hex
            != plan.analysis_processor_public_key_hex
            or derivation.qc_policy_digest != input_plan.mask_policy_digest
            or derivation.background_cycle_rule_digest
            != input_plan.background_cycle_rule_digest
            or run.input_plan_digest != input_plan.plan_digest
            or run.analysis_input_derivation_artifact_digest
            != derivation.artifact_digest
            or run.analysis_input_derivation_artifact_json
            != json.dumps(
                derivation.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            or derivation.input_bundle_digest != run.input_bundle_digest
            or derivation.full_analysis_input_digest
            != run.full_analysis_input_digest
            or derivation.grid_contract_digest != run.grid_time_contract_digest
        ):
            raise ValueError("operational analysis provenance disagrees with its plan")
        if run.operational_data_identity_json is None:
            raise ValueError("operational provenance requires source identity")
        source_identity = OperationalDataIdentity.from_json(
            run.operational_data_identity_json
        )
        geometry = plan.range_geometry_contract
        raw_sites = {
            item.radar_site_digest
            for item in ordered_receipts
        }
        if type(geometry) is RangeGeometryContract:
            if (
                raw_sites != {geometry.radar_site_digest}
                or source_identity.radar_source_kind != "single_site"
                or source_identity.radar_site_digest != geometry.radar_site_digest
                or source_identity.radar_site_location_digest
                != geometry.radar_site_location_digest
                or resolved_source_coverage is not None
                or derivation.source_selection_evidence_digest
                != _json_digest({"contract": "single-site-source-selection-v1"})
            ):
                raise ValueError("single-site operational provenance disagrees")
        else:
            mosaic = cast(MosaicRangeGeometryContract, geometry)
            if (
                raw_sites != set(mosaic.radar_site_digests)
                or source_identity.radar_source_kind != "mosaic"
                or source_identity.source_selection_policy_digest
                != mosaic.source_selection_policy_digest
                or resolved_source_coverage is None
            ):
                raise ValueError("mosaic operational provenance disagrees")
            validate_resolved_source_coverage_artifact(resolved_source_coverage)
            if (
                resolved_source_coverage.case_id != plan.plan_id
                or derivation.source_selection_evidence_digest
                != resolved_source_coverage.artifact_digest
            ):
                raise ValueError("operational source coverage changed")
        derived = _derive_analysis_inputs_from_raw_products(
            input_plan=input_plan,
            resolved_raw_observations=ordered_receipts,
            resolved_source_coverage=resolved_source_coverage,
        )
        background_digest = (
            None
            if background_frames_dbz is None
            else tensor_digest(background_frames_dbz)
        )
        background_times, background_source, background_identities = (
            _background_input_identity_digests(
                input_plan=input_plan,
                run=run,
                background_frames_dbz=background_frames_dbz,
            )
        )
        if (
            tensor_digest(derived[0]) != derivation.input_frames_digest
            or tensor_digest(derived[1]) != derivation.observation_masks_digest
            or tensor_digest(derived[2])
            != derivation.observation_quality_weight_digest
            or tensor_digest(derived[3]) != derivation.observation_std_dbz_digest
            or background_digest != derivation.background_frames_digest
            or background_times != derivation.background_valid_times
            or background_source != derivation.background_source_identity_digest
            or background_identities != derivation.background_input_identity_digests
        ):
            raise ValueError("operational provenance does not replay its outputs")
        arrays: dict[str, Tensor] = {
            **_raw_resolution_encoded_arrays(
                ordered_receipts,
                derived_frames=derived[0],
            ),
            "derived_input_frames": derived[0],
            "derived_qc_valid_mask": derived[1],
            "derived_quality_weight": derived[2],
            "derived_observation_std_dbz": derived[3],
        }
        if background_frames_dbz is not None:
            arrays["background_frames_dbz"] = background_frames_dbz
        if resolved_source_coverage is not None:
            arrays.update({
                "source_radar_index_map": (
                    resolved_source_coverage._source_radar_index_map
                ),
                "input_history_source_radar_index_map": (
                    resolved_source_coverage._input_history_source_radar_index_map
                ),
                "outage_mask": resolved_source_coverage._outage_mask,
                "dynamic_qc_valid_mask": (
                    resolved_source_coverage._dynamic_qc_valid_mask
                ),
                "nominal_source_coverage_mask": (
                    resolved_source_coverage._nominal_source_coverage_mask
                ),
                "resolved_source_coverage_mask": (
                    resolved_source_coverage._resolved_mask
                ),
            })
        metadata = {
            "contract": "analysis-input-provenance-bundle-v2",
            "provenance_kind": "operational",
            "provenance_plan": plan.payload | {"plan_digest": plan.plan_digest},
            "input_plan": input_plan.payload | {"plan_digest": input_plan.plan_digest},
            "resolved_raw_observations": [
                item.payload | {"receipt_digest": item.receipt_digest}
                for item in ordered_receipts
            ],
            "raw_resolution": raw_resolution.payload
            | {"receipt_digest": raw_resolution.receipt_digest},
            "analysis_input_derivation": derivation.payload
            | {"artifact_digest": derivation.artifact_digest},
            "resolved_source_coverage": (
                None
                if resolved_source_coverage is None
                else resolved_source_coverage.payload
                | {"artifact_digest": resolved_source_coverage.artifact_digest}
            ),
        }
        target = self.analysis_input_provenance_dir / derivation.artifact_digest
        canonical_payload = json.dumps(
            derivation.payload
            | {
                "artifact_digest": derivation.artifact_digest,
                "input_plan": input_plan.payload
                | {"plan_digest": input_plan.plan_digest},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            retained = connection.execute(
                "SELECT provenance_kind,payload_json,status FROM "
                "analysis_input_provenance_commits "
                "WHERE artifact_digest = ?",
                (derivation.artifact_digest,),
            ).fetchone()
        if retained is not None:
            if tuple(retained[:2]) != ("operational", canonical_payload):
                raise ValueError("operational provenance commit equivocated")
            return self.reconcile_prepared_analysis_input_provenance(
                derivation.artifact_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=(
                    analysis_processor_trust_store_path
                ),
                provenance_commit_signer=provenance_commit_signer,
            )
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{derivation.artifact_digest}.",
            dir=self.analysis_input_provenance_dir,
        ))
        published = False
        try:
            arrays_path = temporary / "source_and_derived_arrays.npz"
            np.savez_compressed(
                arrays_path,
                **cast(dict[str, Any], {
                    name: value.detach().cpu().contiguous().numpy()
                    for name, value in arrays.items()
                }),
            )
            metadata_path = temporary / "provenance.json"
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            checksums_path = temporary / "checksums.json"
            checksums = {
                "source_and_derived_arrays.npz": _file_digest(arrays_path),
                "provenance.json": _file_digest(metadata_path),
            }
            checksums_path.write_text(_json_text(checksums), encoding="utf-8")
            for path in (arrays_path, metadata_path, checksums_path):
                _fsync_file(path)
            _fsync_directory(temporary)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(timezone.utc)
                deadline = datetime.fromisoformat(
                    input_plan.decision_deadline.replace("Z", "+00:00")
                )
                if not (
                    datetime.fromisoformat(
                        raw_resolution.resolved_at.replace("Z", "+00:00")
                    )
                    <= datetime.fromisoformat(
                        derivation.processed_at.replace("Z", "+00:00")
                    )
                    <= now
                    <= deadline
                ):
                    raise ValueError(
                        "operational provenance missed its durable deadline"
                    )
                plan_row = connection.execute(
                    "SELECT payload_json FROM "
                    "operational_analysis_input_provenance_plans "
                    "WHERE plan_digest = ?",
                    (plan.plan_digest,),
                ).fetchone()
                if plan_row != (json.dumps(
                    plan.payload | {"plan_digest": plan.plan_digest},
                    sort_keys=True,
                    separators=(",", ":"),
                ),):
                    raise ValueError("operational provenance plan is not registered")
                final_raw_trust = _load_raw_ingestor_trust_store(
                    raw_ingestor_trust_store_path
                )
                if final_raw_trust.content_digest != current_raw_trust.content_digest:
                    raise ValueError("raw-ingestor trust changed during commit")
                for slot_digest, receipt in receipts_by_slot.items():
                    _validate_current_raw_ingestor_receipt(
                        receipt,
                        slot_by_digest[slot_digest],
                        pinned_trust_store=plan.raw_ingestor_trust_store,
                        current_trust_store=final_raw_trust,
                    )
                if target.exists():
                    if (
                        target.is_symlink()
                        or not target.is_dir()
                        or _file_digest(
                            target / "source_and_derived_arrays.npz"
                        ) != checksums["source_and_derived_arrays.npz"]
                        or _file_digest(target / "provenance.json")
                        != checksums["provenance.json"]
                        or (target / "checksums.json").read_text("utf-8")
                        != _json_text(checksums)
                    ):
                        quarantine = self.analysis_input_provenance_dir / (
                            f".{derivation.artifact_digest}.quarantine."
                            f"{uuid.uuid4().hex}"
                        )
                        os.replace(target, quarantine)
                        _fsync_directory(self.analysis_input_provenance_dir)
                        raise ValueError(
                            "orphan operational provenance bytes equivocated"
                        )
                    shutil.rmtree(temporary)
                else:
                    os.rename(temporary, target)
                    _fsync_directory(self.analysis_input_provenance_dir)
                published = True
                for history in raw_resolution.history_entries:
                    self._record_operational_raw_resolution_history(
                        connection,
                        entry=history,
                        raw_resolution_receipt_digest=(
                            raw_resolution.receipt_digest
                        ),
                        recorded_at=now.isoformat(),
                        expected_authority_id=plan.analysis_processor_id,
                        expected_authority_public_key_hex=(
                            plan.analysis_processor_public_key_hex
                        ),
                    )
                ledger_instance_digest = (
                    self._analysis_provenance_ledger_instance_digest(connection)
                )
                side_effect_digest = _json_digest(
                    {
                        "contract": (
                            "analysis-provenance-ledger-side-effects-v1"
                        ),
                        "provenance_kind": "operational",
                        "raw_resolution_receipt_digest": (
                            raw_resolution.receipt_digest
                        ),
                        "history_entry_digests": sorted(
                            item.entry_digest
                            for item in raw_resolution.history_entries
                        ),
                    }
                )
                (
                    preparation_receipt_json,
                    preparation_receipt_digest,
                ) = self._issue_analysis_provenance_preparation_receipt(
                    artifact_digest=derivation.artifact_digest,
                    provenance_kind="operational",
                    provenance_plan_digest=plan.plan_digest,
                    input_plan_digest=input_plan.plan_digest,
                    raw_resolution_receipt_digest=(
                        raw_resolution.receipt_digest
                    ),
                    payload_json=canonical_payload,
                    payload_committed_at=now.isoformat(),
                    deadline=input_plan.decision_deadline,
                    ledger_instance_digest=ledger_instance_digest,
                    side_effect_digest=side_effect_digest,
                    authority_trust_store=analysis_trust,
                    signer=provenance_commit_signer,
                )
                if datetime.now(timezone.utc) > deadline:
                    raise ValueError(
                        "operational provenance ledger receipt missed its deadline"
                    )
                connection.execute(
                    "INSERT INTO analysis_input_provenance_commits "
                    "(artifact_digest,provenance_kind,provenance_plan_digest,"
                    "case_id,input_plan_digest,raw_resolution_receipt_digest,"
                    "payload_json,arrays_sha256,metadata_sha256,path,"
                    "raw_ingestor_trust_store_digest,raw_trust_validated_at,"
                    "committed_at,usable,status,payload_committed_at,"
                    "preparation_receipt_json,preparation_receipt_digest,"
                    "activated_at,expired_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?,0,'prepared',?,"
                    "?,?,NULL,NULL)",
                    (
                        derivation.artifact_digest,
                        "operational",
                        plan.plan_digest,
                        plan.plan_id,
                        input_plan.plan_digest,
                        raw_resolution.receipt_digest,
                        canonical_payload,
                        checksums["source_and_derived_arrays.npz"],
                        checksums["provenance.json"],
                        str(target),
                        final_raw_trust.content_digest,
                        now.isoformat(),
                        now.isoformat(),
                        preparation_receipt_json,
                        preparation_receipt_digest,
                    ),
                )
            self.reconcile_prepared_analysis_input_provenance(
                derivation.artifact_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=(
                    analysis_processor_trust_store_path
                ),
                provenance_commit_signer=provenance_commit_signer,
            )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and target.exists():
                with self._connect() as connection:
                    retained = connection.execute(
                        "SELECT 1 FROM analysis_input_provenance_commits "
                        "WHERE artifact_digest = ?",
                        (derivation.artifact_digest,),
                    ).fetchone()
                if retained is None:
                    shutil.rmtree(target)
            raise
        return derivation.artifact_digest

    def append_neural_prior_holdout_scoring_input_artifact(
        self,
        plan: NeuralPriorHoldoutPlan,
        result: PhysicalEventCatalogResult,
        artifact: HoldoutScoringInputArtifact,
    ) -> str:
        """Append the exact forecast/verification set before scoring launches."""

        validate_neural_prior_holdout_plan(plan)
        validate_physical_event_catalog_result(
            result,
            plan.physical_event_catalog_plan,
        )
        if (
            artifact.artifact_digest != _json_digest(artifact.payload)
            or artifact.holdout_plan_digest != plan.plan_digest
            or artifact.promotion_decision_rule_digest
            != plan.promotion_decision_rule_digest
            or artifact.candidate_prior_digest != plan.candidate_family_digests[0]
            or artifact.parent_prior_digest != plan.parent_prior_digest
            or set(artifact.ordered_case_ids) != {item.case_id for item in plan.cases}
        ):
            raise ValueError("scoring input artifact disagrees with holdout plan")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            catalog_row = connection.execute(
                "SELECT result_digest FROM neural_prior_event_catalog_results "
                "WHERE plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if catalog_row is None or catalog_row[0] != result.result_digest:
                raise ValueError("scoring input requires its registered event catalog")
            rule_row = connection.execute(
                "SELECT rule_digest FROM neural_prior_holdout_plan_rule_bindings "
                "WHERE holdout_plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if (
                rule_row is None
                or rule_row[0] != plan.promotion_decision_rule_digest
            ):
                raise ValueError("scoring input requires its pre-outcome rule binding")
            domain_digest_by_case = dict(
                zip(
                    artifact.ordered_case_ids,
                    artifact.operational_issuance_domain_artifact_digests,
                    strict=True,
                )
            )
            for planned_case in plan.cases:
                issuance_plan = next(
                    item
                    for item in plan.operational_issuance_domain_plans
                    if item.plan_digest
                    == planned_case.operational_issuance_domain_plan_digest
                )
                if issuance_plan.radar_source_kind != "mosaic":
                    continue
                coverage_row = connection.execute(
                    "SELECT artifact_digest FROM "
                    "neural_prior_resolved_source_coverage_artifacts "
                    "WHERE operational_domain_artifact_digest = ? "
                    "AND issuance_domain_plan_digest = ? AND case_id = ?",
                    (
                        domain_digest_by_case[planned_case.case_id],
                        issuance_plan.plan_digest,
                        planned_case.case_id,
                    ),
                ).fetchone()
                if coverage_row is None:
                    raise ValueError(
                        "mosaic scoring input requires pre-issue source coverage"
                    )
            try:
                now = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    "INSERT INTO neural_prior_holdout_scoring_input_artifacts "
                    "(artifact_digest, holdout_plan_digest, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        artifact.artifact_digest,
                        plan.plan_digest,
                        json.dumps(
                            artifact.payload
                            | {"artifact_digest": artifact.artifact_digest},
                            sort_keys=True,
                        ),
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "holdout scoring input artifact is already registered"
                ) from error
        return artifact.artifact_digest

    def append_neural_prior_scoring_replay_bundle(
        self,
        scoring_input_artifact: HoldoutScoringInputArtifact,
        cases: tuple[ScoringReplayCaseArtifact, ...],
        *,
        algorithm_source_manifest_digest: str,
        raw_ingestor_trust_store_path: str | Path,
        analysis_processor_trust_store_path: str | Path,
        mps_backend_certification_policy: (
            MPSBackendCertificationPolicy | None
        ) = None,
        mps_backend_certification: (
            MPSBackendCertificationEvidence | None
        ) = None,
    ) -> ScoringReplayBundleManifest:
        """Recompute scores in product code, then durably retain exact inputs."""

        if not cases or any(type(item) is not ScoringReplayCaseArtifact for item in cases):
            raise TypeError("scoring replay requires typed product case artifacts")
        ordered_cases = tuple(sorted(cases, key=lambda item: item.case_id))
        current_scoring_raw_trust = _validate_current_scoring_raw_ingestor_receipts(
            ordered_cases,
            raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
        )
        for replay_case in ordered_cases:
            self.reconcile_prepared_analysis_input_provenance(
                replay_case.analysis_input_derivation.artifact_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=(
                    analysis_processor_trust_store_path
                ),
            )
        plan = ordered_cases[0].plan
        if any(item.plan.plan_digest != plan.plan_digest for item in ordered_cases):
            raise ValueError("scoring replay cases use different holdout plans")
        family = plan.promotion_experiment_family
        reservation = family.global_sampling_reservation
        resolved_bindings: dict[str, str] = {}
        resolutions_by_digest: dict[str, GlobalRawVolumeResolutionReceipt] = {}
        for case in ordered_cases:
            case_bindings = tuple(
                sorted(
                    (
                        item.slot_plan_digest,
                        item.resolution_identity_digest,
                    )
                    for item in case.resolved_raw_observations
                )
            )
            resolution = case.global_raw_resolution_receipt
            if resolution.slot_identity_bindings != case_bindings:
                raise ValueError(
                    "scoring replay case raw-volume resolution is incomplete"
                )
            resolutions_by_digest[resolution.receipt_digest] = resolution
            for slot_digest, identity_digest in case_bindings:
                retained_identity = resolved_bindings.setdefault(
                    slot_digest,
                    identity_digest,
                )
                if retained_identity != identity_digest:
                    raise ValueError("scoring replay raw-volume resolution equivocated")
        global_resolutions = tuple(
            sorted(
                resolutions_by_digest.values(),
                key=lambda item: item.registry_sequence_number,
            )
        )
        expected_sequence = reservation.registry_sequence_number + 1
        expected_previous = reservation.committed_registry_root_digest
        for resolution in global_resolutions:
            if (
                resolution.experiment_scope_digest
                != reservation.experiment_scope_digest
                or resolution.reservation_receipt_digest
                != reservation.receipt_digest
                or resolution.registry_id != reservation.registry_id
                or resolution.authority_id != reservation.authority_id
                or resolution.authority_public_key_hex
                != reservation.authority_public_key_hex
                or resolution.registry_sequence_number != expected_sequence
                or resolution.previous_registry_root_digest != expected_previous
            ):
                raise ValueError(
                    "scoring replay raw resolutions do not form one registry chain"
                )
            expected_sequence += 1
            expected_previous = resolution.committed_registry_root_digest
        if set(resolved_bindings) != set(family.raw_observation_slot_digests):
            raise ValueError("scoring replay raw-volume resolution is incomplete")
        case_tensors = {
            item.case_id: item.replay_tensors() for item in ordered_cases
        }
        execution_device = _semantic_replay_execution_device(
            ordered_cases,
            case_tensors,
        )
        (
            scoring_certification_policy_digest,
            scoring_certification_evidence_digest,
        ) = _validate_scoring_backend_certification(
            execution_device,
            mps_backend_certification_policy,
            mps_backend_certification,
        )
        semantic_case_digests = tuple(
            item.semantic_input_digest for item in ordered_cases
        )
        ordered = tuple(
            recompute_prior_holdout_evaluation_from_bundle(item)
            for item in ordered_cases
        )
        case_ids = tuple(item.case_id for item in ordered)
        dynamic_source_case_ids = tuple(
            item.case_id
            for item in ordered_cases
            if item.resolved_source_coverage is not None
        )
        background_case_ids = tuple(
            item.case_id
            for item in ordered_cases
            if item.background_frames_dbz is not None
        )
        for case_id in case_ids:
            _validate_scoring_replay_case_tensors(
                case_tensors[case_id],
                dynamic_source=case_id in dynamic_source_case_ids,
                background_present=case_id in background_case_ids,
            )
        total_expanded_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensors in case_tensors.values()
            for tensor in tensors.values()
        )
        available_disk_bytes = shutil.disk_usage(self.scoring_replays_dir).free
        active_runtime = numerical_runtime_manifest(execution_device)
        current_algorithm_source_manifest_digest = algorithm_bundle_digest()
        if (
            scoring_input_artifact.artifact_digest
            != _json_digest(scoring_input_artifact.payload)
            or case_ids != scoring_input_artifact.ordered_case_ids
            or total_expanded_bytes
            > _MAXIMUM_SCORING_REPLAY_TOTAL_EXPANDED_BYTES
            or available_disk_bytes < total_expanded_bytes * 2
            or algorithm_source_manifest_digest
            != current_algorithm_source_manifest_digest
        ):
            raise ValueError("scoring replay inputs are incomplete")
        temporary = Path(
            tempfile.mkdtemp(prefix=".scoring-replay.", dir=self.scoring_replays_dir)
        )
        published = False
        registered = False
        target: Path | None = None
        try:
            records: list[ScoringReplayTensorRecord] = []
            shard_paths: dict[str, Path] = {}
            for case_index, case_id in enumerate(case_ids):
                for role, source_tensor in sorted(case_tensors[case_id].items()):
                    tensor = source_tensor.detach().cpu().contiguous()
                    if tensor.layout is not torch.strided or tensor.numel() == 0:
                        raise ValueError("scoring replay tensor is invalid")
                    try:
                        array = tensor.numpy().copy()
                    except TypeError as error:
                        raise ValueError(
                            "scoring replay tensor dtype is unsupported"
                        ) from error
                    if array.nbytes > _MAXIMUM_ACTION_ARTIFACT_FILE_BYTES:
                        raise ValueError("scoring replay tensor exceeds shard budget")
                    member = "tensor"
                    candidate_path = (
                        temporary / f".tensor-{case_index:06d}-{role}.npz"
                    )
                    np.savez_compressed(
                        candidate_path,
                        **cast(dict[str, Any], {member: array}),
                    )
                    archive_sha256 = _file_digest(candidate_path)
                    shard_path = (
                        temporary / f"tensor_{archive_sha256}.npz"
                    )
                    retained_path = shard_paths.get(archive_sha256)
                    if retained_path is None:
                        os.replace(candidate_path, shard_path)
                        shard_paths[archive_sha256] = shard_path
                    else:
                        candidate_path.unlink()
                    records.append(
                        ScoringReplayTensorRecord(
                            case_id=case_id,
                            role=role,
                            archive_member=member,
                            dtype=str(tensor.dtype).removeprefix("torch."),
                            shape=tuple(int(value) for value in tensor.shape),
                            tensor_digest=tensor_digest(tensor),
                            archive_sha256=archive_sha256,
                        )
                    )
            shard_sha256s = tuple(sorted(shard_paths))
            if len(shard_sha256s) > _MAXIMUM_SCORING_REPLAY_SHARDS:
                raise ValueError("scoring replay exceeds shard-count budget")
            shard_set_digest = _json_digest(
                {
                    "contract": "neural-prior-scoring-replay-shard-set-v1",
                    "ordered_shard_sha256s": list(shard_sha256s),
                }
            )
            evaluation_path = temporary / "evaluations.json"
            evaluation_path.write_text(
                json.dumps(
                    [_evaluation_audit_payload(item) for item in ordered],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            raw_provenance_path = temporary / "raw_provenance.json"
            raw_provenance_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": item.case_id,
                            "input_plan": (
                                next(
                                    plan
                                    for plan in item.plan.input_plans
                                    if plan.plan_digest
                                    == item.analysis_input_derivation.input_plan_digest
                                ).payload
                                | {
                                    "plan_digest": (
                                        item.analysis_input_derivation.input_plan_digest
                                    )
                                }
                            ),
                            "resolved_raw_observations": [
                                receipt.payload
                                | {"receipt_digest": receipt.receipt_digest}
                                for receipt in item.resolved_raw_observations
                            ],
                            "global_raw_resolution_receipt": (
                                item.global_raw_resolution_receipt.payload
                                | {
                                    "receipt_digest": (
                                        item.global_raw_resolution_receipt.receipt_digest
                                    )
                                }
                            ),
                            "analysis_input_derivation": (
                                item.analysis_input_derivation.payload
                                | {
                                    "artifact_digest": (
                                        item.analysis_input_derivation.artifact_digest
                                    )
                                }
                            ),
                            "background_run_lineage": (
                                None
                                if item.background_frames_dbz is None
                                else {
                                    "background_valid_times": list(
                                        cast(
                                            tuple[str, str, str],
                                            cast(
                                                Any,
                                                item.candidate_forecast.run.grid_time_contract,
                                            ).background_valid_times,
                                        )
                                    ),
                                    "operational_data_identity_json": (
                                        item.candidate_forecast.run.operational_data_identity_json
                                    ),
                                    "operational_data_identity_digest": (
                                        item.candidate_forecast.run.operational_data_identity_digest
                                    ),
                                }
                            ),
                            "resolved_source_coverage": (
                                None
                                if item.resolved_source_coverage is None
                                else item.resolved_source_coverage.payload
                                | {
                                    "artifact_digest": (
                                        item.resolved_source_coverage.artifact_digest
                                    )
                                }
                            ),
                            "range_geometry_contract": (
                                _scoring_replay_range_geometry(item).payload
                                | {
                                    "contract_digest": (
                                        _scoring_replay_range_geometry(
                                            item
                                        ).contract_digest
                                    )
                                }
                            ),
                        }
                        for item in ordered_cases
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            verification_provenance_path = (
                temporary / "verification_provenance.json"
            )
            verification_provenance_path.write_text(
                json.dumps(
                    [
                        _current_verification_provenance_payload(item)
                        for item in ordered_cases
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            manifest = ScoringReplayBundleManifest(
                scoring_input_artifact_digest=(
                    scoring_input_artifact.artifact_digest
                ),
                ordered_case_ids=case_ids,
                ordered_evaluation_digests=tuple(
                    item.evaluation_digest for item in ordered
                ),
                semantic_case_digests=semantic_case_digests,
                dynamic_source_case_ids=dynamic_source_case_ids,
                background_case_ids=background_case_ids,
                algorithm_source_manifest_digest=(
                    algorithm_source_manifest_digest
                ),
                runtime_compatibility_digest=(
                    active_runtime.compatibility_digest
                ),
                runtime_exact_digest=active_runtime.exact_digest,
                scoring_backend_certification_policy_digest=(
                    scoring_certification_policy_digest
                ),
                scoring_backend_certification_evidence_digest=(
                    scoring_certification_evidence_digest
                ),
                tensor_records=tuple(records),
                tensor_archive_sha256=shard_set_digest,
                evaluation_payload_sha256=_file_digest(evaluation_path),
                raw_provenance_payload_sha256=(
                    _file_digest(raw_provenance_path)
                ),
                verification_provenance_payload_sha256=(
                    _file_digest(verification_provenance_path)
                ),
                raw_ingestor_trust_store_digest=(
                    current_scoring_raw_trust.content_digest
                ),
                tensor_shard_sha256s=shard_sha256s,
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    manifest.payload | {"bundle_digest": manifest.bundle_digest},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            target = self.scoring_replays_dir / manifest.bundle_digest
            if target.exists():
                raise FileExistsError("scoring replay bundle already exists")
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                final_scoring_raw_trust = _validate_current_scoring_raw_ingestor_receipts(
                    ordered_cases,
                    raw_ingestor_trust_store_path=(
                        raw_ingestor_trust_store_path
                    ),
                )
                if (
                    final_scoring_raw_trust.content_digest
                    != current_scoring_raw_trust.content_digest
                ):
                    raise ValueError(
                        "raw-ingestor trust store changed during replay commit"
                    )
                input_row = connection.execute(
                    "SELECT payload_json FROM "
                    "neural_prior_holdout_scoring_input_artifacts "
                    "WHERE artifact_digest = ?",
                    (scoring_input_artifact.artifact_digest,),
                ).fetchone()
                expected_input_json = json.dumps(
                    scoring_input_artifact.payload
                    | {"artifact_digest": scoring_input_artifact.artifact_digest},
                    sort_keys=True,
                )
                if input_row is None or input_row[0] != expected_input_json:
                    raise ValueError(
                        "scoring replay requires its registered scoring input"
                    )
                start_rows = connection.execute(
                    "SELECT receipt_json FROM trusted_process_start_receipts_v2 "
                    "WHERE process_kind = 'candidate_scoring'"
                ).fetchall()
                if not any(
                    json.loads(row[0]).get("subject_digests")
                    == [scoring_input_artifact.artifact_digest]
                    for row in start_rows
                ):
                    raise ValueError(
                        "scoring replay requires its ledger scoring start"
                    )
                for replay_case in ordered_cases:
                    training_inputs = (
                        (
                            replay_case.manifest
                            .training_raw_registry_receipt_payload_json,
                            replay_case.manifest.training_raw_registry_receipt_digest,
                        ),
                        (
                            replay_case.regime_classifier_manifest
                            .training_raw_registry_receipt_payload_json,
                            replay_case.regime_classifier_manifest
                            .training_raw_registry_receipt_digest,
                        ),
                    )
                    if training_inputs[0][1] != training_inputs[1][1]:
                        raise ValueError(
                            "candidate and classifier require one family-wide "
                            "training raw registry receipt"
                        )
                    for payload_json, receipt_digest in training_inputs:
                        training_receipt = _training_raw_registry_receipt_from_json(
                            payload_json,
                            expected_digest=receipt_digest,
                        )
                        expected_training_json = json.dumps(
                            training_receipt.payload
                            | {"receipt_digest": training_receipt.receipt_digest},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        retained_training = connection.execute(
                            "SELECT payload_json FROM training_raw_registry_entries "
                            "WHERE receipt_digest = ?",
                            (training_receipt.receipt_digest,),
                        ).fetchone()
                        if retained_training != (expected_training_json,):
                            raise ValueError(
                                "scoring replay requires precommitted training raw lineage"
                            )
                for replay_case in ordered_cases:
                    global_resolution = (
                        replay_case.global_raw_resolution_receipt
                    )
                    retained_resolution = connection.execute(
                        "SELECT registry_sequence_number,previous_registry_root_digest,"
                        "committed_registry_root_digest FROM "
                        "global_sampling_registry_entries WHERE receipt_digest = ?",
                        (global_resolution.receipt_digest,),
                    ).fetchone()
                    expected_resolution_entry = (
                        global_resolution.registry_sequence_number,
                        global_resolution.previous_registry_root_digest,
                        global_resolution.committed_registry_root_digest,
                    )
                    if (
                        retained_resolution is None
                        or tuple(retained_resolution) != expected_resolution_entry
                    ):
                        raise ValueError(
                            "scoring replay requires a precommitted raw resolution"
                        )
                    provenance_row = connection.execute(
                        "SELECT payload_json,arrays_sha256,metadata_sha256,path,"
                        "committed_at,usable,raw_ingestor_trust_store_digest,"
                        "raw_trust_validated_at FROM "
                        "neural_prior_analysis_input_provenance "
                        "WHERE artifact_digest = ? AND holdout_plan_digest = ? "
                        "AND case_id = ? AND input_plan_digest = ? "
                        "AND global_resolution_receipt_digest = ?",
                        (
                            replay_case.analysis_input_derivation.artifact_digest,
                            replay_case.plan.plan_digest,
                            replay_case.case_id,
                            replay_case.analysis_input_derivation.input_plan_digest,
                            global_resolution.receipt_digest,
                        ),
                    ).fetchone()
                    planned = replay_case.plan.case(replay_case.case_id)
                    retained_input_plan = next(
                        item
                        for item in replay_case.plan.input_plans
                        if item.plan_digest == planned.input_plan_digest
                    )
                    expected_derivation_json = json.dumps(
                        replay_case.analysis_input_derivation.payload
                        | {
                            "artifact_digest": (
                                replay_case.analysis_input_derivation.artifact_digest
                            ),
                            "input_plan": retained_input_plan.payload
                            | {"plan_digest": retained_input_plan.plan_digest},
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if (
                        provenance_row is None
                        or provenance_row[0] != expected_derivation_json
                        or int(provenance_row[5]) != 1
                        or provenance_row[6]
                        != final_scoring_raw_trust.content_digest
                        or provenance_row[7] is None
                        or datetime.fromisoformat(str(provenance_row[4]))
                        > datetime.fromisoformat(
                            retained_input_plan.decision_deadline.replace(
                                "Z", "+00:00"
                            )
                        )
                    ):
                        raise ValueError(
                            "scoring replay requires pre-deadline input provenance"
                        )
                    provenance_path = Path(str(provenance_row[3]))
                    if (
                        not provenance_path.is_dir()
                        or _file_digest(
                            provenance_path / "source_and_derived_arrays.npz"
                        )
                        != provenance_row[1]
                        or _file_digest(provenance_path / "provenance.json")
                        != provenance_row[2]
                    ):
                        raise ValueError(
                            "analysis input provenance durable bytes changed"
                        )
                    for slot_digest, raw_identity_digest in (
                        global_resolution.slot_identity_bindings
                    ):
                        retained = connection.execute(
                            "SELECT reservation.family_digest,membership.case_id "
                            "FROM promotion_raw_volume_identity_reservations "
                            "AS reservation JOIN "
                            "raw_volume_resolution_memberships AS membership "
                            "ON membership.raw_volume_identity_digest = "
                            "reservation.raw_volume_identity_digest "
                            "WHERE reservation.raw_volume_identity_digest = ? "
                            "AND membership.global_resolution_receipt_digest = ? "
                            "AND membership.raw_observation_slot_digest = ?",
                            (
                                raw_identity_digest,
                                global_resolution.receipt_digest,
                                slot_digest,
                            ),
                        ).fetchone()
                        if retained != (
                            family.family_digest,
                            replay_case.case_id,
                        ):
                            raise ValueError(
                                "canonical raw volume lacks its committed reservation"
                            )
                _publish_durable_directory(
                    temporary=temporary,
                    target=target,
                    durable_files=(
                        *tuple(shard_paths[value] for value in shard_sha256s),
                        evaluation_path,
                        raw_provenance_path,
                        verification_provenance_path,
                        manifest_path,
                    ),
                    parent=self.scoring_replays_dir,
                )
                published = True
                connection.execute(
                    "INSERT INTO neural_prior_scoring_replay_bundles "
                    "(bundle_digest, scoring_input_artifact_digest, "
                    "manifest_json, path, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        manifest.bundle_digest,
                        manifest.scoring_input_artifact_digest,
                        json.dumps(
                            manifest.payload
                            | {"bundle_digest": manifest.bundle_digest},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        str(target.relative_to(self.root)),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self._prepare_raw_trust_artifact_activation(
                    connection,
                    artifact_kind="scoring_replay_bundle",
                    artifact_digest=manifest.bundle_digest,
                    raw_ingestor_trust_store_digest=(
                        current_scoring_raw_trust.content_digest
                    ),
                )
            registered = True
            self._activate_raw_trust_artifact(
                artifact_kind="scoring_replay_bundle",
                artifact_digest=manifest.bundle_digest,
                raw_ingestor_trust_store_digest=(
                    current_scoring_raw_trust.content_digest
                ),
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
            )
            return manifest
        except Exception:
            if (
                published
                and not registered
                and target is not None
                and target.exists()
            ):
                shutil.rmtree(target)
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load_neural_prior_scoring_replay_bundle(
        self,
        bundle_digest: str,
        *,
        cases: tuple[ScoringReplayCaseArtifact, ...] | None = None,
        _require_raw_trust_activation: bool = True,
    ) -> LoadedScoringReplayBundle:
        """Rehash members and optionally run product-owned semantic scoring."""

        if re.fullmatch(r"[0-9a-f]{64}", bundle_digest) is None:
            raise ValueError("scoring replay bundle digest is invalid")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT manifest_json, path FROM "
                "neural_prior_scoring_replay_bundles WHERE bundle_digest = ?",
                (bundle_digest,),
            ).fetchone()
        if row is None:
            raise ValueError("scoring replay bundle is not registered")
        target = (self.root / row[1]).resolve()
        if target.parent != self.scoring_replays_dir:
            raise ValueError("scoring replay bundle path is invalid")
        retained_manifest_payload = json.loads(row[0])
        current_bundle = (
            isinstance(retained_manifest_payload, dict)
            and retained_manifest_payload.get("contract")
            == SEMANTIC_SCORING_REPLAY_CONTRACT
        )
        if current_bundle and _require_raw_trust_activation:
            retained_raw_trust_digest = retained_manifest_payload.get(
                "raw_ingestor_trust_store_digest"
            )
            if not isinstance(retained_raw_trust_digest, str):
                raise ValueError("current scoring replay raw trust is missing")
            with self._connect() as connection:
                self._require_raw_trust_artifact_usable(
                    connection,
                    artifact_kind="scoring_replay_bundle",
                    artifact_digest=bundle_digest,
                    raw_ingestor_trust_store_digest=retained_raw_trust_digest,
                )
        retained_contract = retained_manifest_payload.get("contract")
        raw_provenance_contracts = {
            "neural-prior-scoring-replay-bundle-v9",
            "neural-prior-scoring-replay-bundle-v11",
            "neural-prior-scoring-replay-bundle-v12",
            "neural-prior-scoring-replay-bundle-v13",
            "neural-prior-scoring-replay-bundle-v14",
            SEMANTIC_SCORING_REPLAY_CONTRACT,
        }
        expected_artifact_members = {"manifest.json", "evaluations.json"}
        retained_shards = retained_manifest_payload.get(
            "tensor_shard_sha256s", []
        )
        if current_bundle:
            if (
                not isinstance(retained_shards, list)
                or not retained_shards
                or len(retained_shards) > _MAXIMUM_SCORING_REPLAY_SHARDS
                or any(
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in retained_shards
                )
                or retained_shards != sorted(set(retained_shards))
            ):
                raise ValueError("current scoring replay shard set is invalid")
            expected_artifact_members.update(
                f"tensor_{value}.npz" for value in retained_shards
            )
        else:
            expected_artifact_members.add("replay_arrays.npz")
        if retained_contract in raw_provenance_contracts:
            expected_artifact_members.add("raw_provenance.json")
        if retained_contract in {
            "neural-prior-scoring-replay-bundle-v14",
            SEMANTIC_SCORING_REPLAY_CONTRACT,
        }:
            expected_artifact_members.add("verification_provenance.json")
        validate_artifact_directory(
            target,
            expected_members=frozenset(expected_artifact_members),
            maximum_members=len(expected_artifact_members),
            maximum_file_bytes=_MAXIMUM_ACTION_ARTIFACT_FILE_BYTES,
        )
        manifest = _decode_scoring_replay_bundle_manifest(
            (target / "manifest.json").read_text("utf-8"),
            expected_digest=bundle_digest,
        )
        if row[0] != json.dumps(
            manifest.payload | {"bundle_digest": manifest.bundle_digest},
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise ValueError("scoring replay manifest disagrees with the ledger")
        archive_path = target / "replay_arrays.npz"
        evaluation_path = target / "evaluations.json"
        raw_provenance_path = target / "raw_provenance.json"
        verification_provenance_path = target / "verification_provenance.json"
        manifest_type = type(manifest)
        has_raw_provenance = manifest_type in {
            ScoringReplayBundleManifest,
            LegacyScoringReplayBundleManifestAuditV9,
            LegacyScoringReplayBundleManifestAuditV11,
            LegacyScoringReplayBundleManifestAuditV12,
            LegacyScoringReplayBundleManifestAuditV13,
            LegacyScoringReplayBundleManifestAuditV14,
        }
        has_verification_provenance = manifest_type in {
            ScoringReplayBundleManifest,
            LegacyScoringReplayBundleManifestAuditV14,
        }
        raw_provenance_checksum_mismatch = False
        if has_raw_provenance:
            raw_manifest = cast(ScoringReplayBundleManifest, manifest)
            raw_provenance_checksum_mismatch = (
                _file_digest(raw_provenance_path)
                != raw_manifest.raw_provenance_payload_sha256
            )
        verification_provenance_checksum_mismatch = False
        if has_verification_provenance:
            verification_manifest = cast(
                ScoringReplayBundleManifest,
                manifest,
            )
            verification_provenance_checksum_mismatch = (
                _file_digest(verification_provenance_path)
                != verification_manifest.verification_provenance_payload_sha256
            )
        if (
            _file_digest(evaluation_path)
            != manifest.evaluation_payload_sha256
            or raw_provenance_checksum_mismatch
            or verification_provenance_checksum_mismatch
        ):
            raise ValueError("scoring replay bundle member checksum mismatch")
        if manifest_type is ScoringReplayBundleManifest:
            current_manifest = cast(ScoringReplayBundleManifest, manifest)
            tensors: Mapping[tuple[str, str], Tensor] = (
                _ScoringReplayTensorShardStore(
                    target,
                    current_manifest.tensor_records,
                )
            )
            for key in tensors:
                tensors[key]
        else:
            if _file_digest(archive_path) != manifest.tensor_archive_sha256:
                raise ValueError("scoring replay bundle member checksum mismatch")
            expected_members = frozenset(
                record.archive_member for record in manifest.tensor_records
            )
            preflight_npz_archive(
                archive_path,
                expected_members=expected_members,
                maximum_members=len(expected_members),
                maximum_expanded_bytes=_MAXIMUM_ACTION_ARTIFACT_EXPANDED_BYTES,
            )
            loaded_tensors: dict[tuple[str, str], Tensor] = {}
            with np.load(archive_path, allow_pickle=False) as archive:
                for record in manifest.tensor_records:
                    array = archive[record.archive_member].copy()
                    tensor = torch.from_numpy(array)
                    if (
                        str(tensor.dtype).removeprefix("torch.") != record.dtype
                        or tuple(tensor.shape) != record.shape
                        or tensor_digest(tensor) != record.tensor_digest
                    ):
                        raise ValueError("scoring replay tensor digest mismatch")
                    loaded_tensors[(record.case_id, record.role)] = tensor
            tensors = loaded_tensors
        verification_reconstructed = False
        if manifest_type is ScoringReplayBundleManifest:
            current_manifest = cast(ScoringReplayBundleManifest, manifest)
            for case_id in current_manifest.ordered_case_ids:
                _validate_scoring_replay_case_tensors(
                    {
                        record.role: tensors[(record.case_id, record.role)]
                        for record in current_manifest.tensor_records
                        if record.case_id == case_id
                    },
                    dynamic_source=(
                        case_id in current_manifest.dynamic_source_case_ids
                    ),
                    background_present=(
                        case_id in current_manifest.background_case_ids
                    ),
                )
            _validate_current_raw_provenance_payload(
                raw_provenance_path.read_text("utf-8"),
                manifest=current_manifest,
                tensors=tensors,
            )
            _validate_current_verification_provenance_payload(
                verification_provenance_path.read_text("utf-8"),
                manifest=current_manifest,
                tensors=tensors,
            )
            verification_reconstructed = True
        raw_evaluations = json.loads(evaluation_path.read_text("utf-8"))
        evaluations = _decode_evaluation_audit_payloads(raw_evaluations)
        if any(
            not isinstance(item, PriorHoldoutEvaluation) for item in evaluations
        ):
            raise ValueError("scoring replay contains audit-only evaluations")
        retained = tuple(
            cast(PriorHoldoutEvaluation, item) for item in evaluations
        )
        if (
            tuple(item.case_id for item in retained)
            != manifest.ordered_case_ids
            or tuple(item.evaluation_digest for item in retained)
            != manifest.ordered_evaluation_digests
        ):
            raise ValueError("scoring replay evaluation digest mismatch")
        semantic_replay_verified = False
        if cases is not None:
            if type(manifest) is not ScoringReplayBundleManifest:
                raise ValueError(
                    "legacy scoring replay bundle is audit-only"
                )
            if any(type(item) is not ScoringReplayCaseArtifact for item in cases):
                raise TypeError("semantic replay cases must use the product type")
            ordered_cases = tuple(sorted(cases, key=lambda item: item.case_id))
            if (
                tuple(item.case_id for item in ordered_cases)
                != manifest.ordered_case_ids
                or tuple(item.semantic_input_digest for item in ordered_cases)
                != manifest.semantic_case_digests
            ):
                raise ValueError("semantic scoring replay input disagrees")
            for item in ordered_cases:
                expected_tensors = item.replay_tensors()
                stored_tensors = {
                    record.role: tensors[(record.case_id, record.role)]
                    for record in manifest.tensor_records
                    if record.case_id == item.case_id
                }
                if set(expected_tensors) != set(stored_tensors) or any(
                    tensor_digest(expected_tensors[role])
                    != tensor_digest(stored_tensors[role])
                    for role in expected_tensors
                ):
                    raise ValueError("semantic scoring replay tensor disagrees")
            recomputed = tuple(
                recompute_prior_holdout_evaluation_from_bundle(item)
                for item in ordered_cases
            )
            if tuple(item.evaluation_digest for item in recomputed) != (
                manifest.ordered_evaluation_digests
            ):
                raise ValueError("semantic scoring replay recomputation disagrees")
            semantic_replay_verified = True
        current_runtime = numerical_runtime_manifest(torch.device("cpu"))
        exact_scientific_runtime = (
            manifest_type is ScoringReplayBundleManifest
            and manifest.algorithm_source_manifest_digest
            == algorithm_bundle_digest()
            and manifest.runtime_exact_digest == current_runtime.exact_digest
        )
        semantic_replay_verified = (
            semantic_replay_verified and exact_scientific_runtime
        )
        return LoadedScoringReplayBundle(
            manifest=manifest,
            evaluations=retained,
            tensors=tensors,
            verification_bytes_verified=True,
            verification_reconstructed=verification_reconstructed,
            verification_semantic_replay_verified=(
                verification_reconstructed and exact_scientific_runtime
            ),
            semantic_replay_verified=semantic_replay_verified,
        )

    def append_trusted_process_start_receipt(
        self,
        plan: NeuralPriorHoldoutPlan | PhysicalEventCatalogPlan,
        result: PhysicalEventCatalogResult,
        receipt: TrustedProcessStartReceipt,
        *,
        scoring_input_artifact: HoldoutScoringInputArtifact | None = None,
        training_dataset_derivation: TrainingDatasetDerivationArtifact | None = None,
        training_target_source_trust_store_path: str | Path | None = None,
        scheduler_trust_store_path: str | Path,
    ) -> str:
        """Append a root-authorized training or scoring launch after cataloging."""

        catalog_plan = (
            plan.physical_event_catalog_plan
            if isinstance(plan, NeuralPriorHoldoutPlan)
            else plan
        )
        scheduler_trust = _load_scheduler_trust_store(
            scheduler_trust_store_path
        )
        _validate_scheduler_authority(catalog_plan, scheduler_trust)
        validate_trusted_process_start_receipt(
            receipt,
            catalog_plan,
            catalog_result=result,
        )
        if isinstance(plan, NeuralPriorHoldoutPlan):
            if (
                receipt.process_kind != "candidate_scoring"
                or scoring_input_artifact is None
                or scoring_input_artifact.artifact_digest
                != _json_digest(scoring_input_artifact.payload)
                or scoring_input_artifact.holdout_plan_digest != plan.plan_digest
                or receipt.subject_digests
                != (scoring_input_artifact.artifact_digest,)
                or receipt.process_algorithm_digest
                != plan.scoring_algorithm_digest
                or receipt.process_runtime_digest != plan.scoring_runtime_digest
                or receipt.execution_contract_digest
                != plan.scoring_execution_contract_digest
            ):
                raise ValueError("scoring start receipt disagrees with holdout plan")
        else:
            if receipt.process_kind != "candidate_training":
                raise ValueError("training catalog requires a training start receipt")
            if training_target_source_trust_store_path is None:
                raise ValueError(
                    "training start requires the current target-source trust store"
                )
            current_target_source_trust_store = (
                _load_training_target_source_trust_store(
                    training_target_source_trust_store_path
                )
            )
            if (
                training_dataset_derivation is None
                or training_dataset_derivation.training_dataset_digest
                not in receipt.subject_digests
                or training_dataset_derivation.training_tensor_snapshot_set_digest
                not in receipt.subject_digests
                or training_dataset_derivation.normalization_derivation_artifact_digest
                not in receipt.subject_digests
            ):
                raise ValueError(
                    "training start requires its dataset derivation and current trust"
                )
            training_dataset_derivation.validate_target_source_trust_at(
                current_target_source_trust_store,
                at=receipt.started_at,
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            catalog_table = (
                "neural_prior_event_catalog_results"
                if isinstance(plan, NeuralPriorHoldoutPlan)
                else "neural_prior_training_event_catalog_results"
            )
            catalog_lookup_digest = (
                plan.plan_digest
                if isinstance(plan, NeuralPriorHoldoutPlan)
                else catalog_plan.plan_digest
            )
            catalog_row = connection.execute(
                f"SELECT result_digest, created_at FROM {catalog_table} "
                "WHERE plan_digest = ?",
                (catalog_lookup_digest,),
            ).fetchone()
            if catalog_row is None or catalog_row[0] != result.result_digest:
                raise ValueError("process start requires a registered event catalog")
            input_append_time: datetime | None = None
            if isinstance(plan, NeuralPriorHoldoutPlan):
                assert scoring_input_artifact is not None
                input_row = connection.execute(
                    "SELECT payload_json, created_at FROM "
                    "neural_prior_holdout_scoring_input_artifacts "
                    "WHERE artifact_digest = ? AND holdout_plan_digest = ?",
                    (scoring_input_artifact.artifact_digest, plan.plan_digest),
                ).fetchone()
                expected_input_json = json.dumps(
                    scoring_input_artifact.payload
                    | {"artifact_digest": scoring_input_artifact.artifact_digest},
                    sort_keys=True,
                )
                if input_row is None or input_row[0] != expected_input_json:
                    raise ValueError("scoring start requires its ledger input artifact")
                input_append_time = datetime.fromisoformat(input_row[1])
                rule_row = connection.execute(
                    "SELECT rule_digest FROM neural_prior_holdout_plan_rule_bindings "
                    "WHERE holdout_plan_digest = ?",
                    (plan.plan_digest,),
                ).fetchone()
                if (
                    rule_row is None
                    or rule_row[0]
                    != scoring_input_artifact.promotion_decision_rule_digest
                ):
                    raise ValueError("scoring start requires its preregistered decision rule")
            now = datetime.now(timezone.utc)
            started_at = datetime.fromisoformat(
                receipt.started_at.replace("Z", "+00:00")
            )
            catalog_append_time = datetime.fromisoformat(catalog_row[1])
            if started_at > now:
                raise ValueError("scoring start receipt cannot claim a future start")
            if started_at <= catalog_append_time:
                raise ValueError("process must start after catalog ledger append")
            if isinstance(plan, NeuralPriorHoldoutPlan):
                assert input_append_time is not None
                if started_at < input_append_time:
                    raise ValueError(
                        "process cannot start before scoring input ledger append"
                    )
            else:
                assert training_dataset_derivation is not None
                assert training_target_source_trust_store_path is not None
                final_target_source_trust = (
                    _load_training_target_source_trust_store(
                        training_target_source_trust_store_path
                    )
                )
                training_dataset_derivation.validate_target_source_trust_at(
                    final_target_source_trust,
                    at=receipt.started_at,
                )
            predecessor = connection.execute(
                "SELECT receipt_digest, scheduler_sequence_number "
                "FROM trusted_process_start_receipts_v2 "
                "WHERE scheduler_id = ? ORDER BY scheduler_sequence_number DESC "
                "LIMIT 1",
                (receipt.scheduler_id,),
            ).fetchone()
            expected_sequence = 1 if predecessor is None else int(predecessor[1]) + 1
            expected_previous = None if predecessor is None else str(predecessor[0])
            if (
                receipt.scheduler_sequence_number != expected_sequence
                or receipt.previous_receipt_digest != expected_previous
            ):
                raise ValueError("trusted process receipt chain is not contiguous")
            try:
                connection.execute(
                    "INSERT INTO trusted_process_start_receipts_v2 "
                    "(receipt_digest, catalog_plan_digest, catalog_result_digest, "
                    "process_kind, scheduler_id, scheduler_sequence_number, job_id, "
                    "launch_nonce, previous_receipt_digest, receipt_json, started_at, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        receipt.receipt_digest,
                        catalog_plan.plan_digest,
                        result.result_digest,
                        receipt.process_kind,
                        receipt.scheduler_id,
                        receipt.scheduler_sequence_number,
                        receipt.job_id,
                        receipt.launch_nonce,
                        receipt.previous_receipt_digest,
                        json.dumps(
                            receipt.payload
                            | {"receipt_digest": receipt.receipt_digest},
                            sort_keys=True,
                        ),
                        receipt.started_at,
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "trusted process-start receipt is already registered"
                ) from error
        return receipt.receipt_digest

    def append_trusted_process_completion_receipt(
        self,
        start: TrustedProcessStartReceipt,
        completion: TrustedProcessCompletionReceipt,
        *,
        process_log_artifact: ProcessLogArtifact,
        scheduler_trust_store_path: str | Path,
        scoring_artifact: HoldoutScoringArtifact | None = None,
        scoring_replay_cases: tuple[ScoringReplayCaseArtifact, ...] | None = None,
        raw_ingestor_trust_store_path: str | Path | None = None,
        training_dataset_derivation: TrainingDatasetDerivationArtifact | None = None,
        training_target_source_trust_store_path: str | Path | None = None,
        mps_backend_certification_policy: (
            MPSBackendCertificationPolicy | None
        ) = None,
        mps_backend_certification: (
            MPSBackendCertificationEvidence | None
        ) = None,
    ) -> str:
        """Append the immutable output and log produced by one trusted launch."""

        validate_trusted_process_completion_receipt(completion, start)
        validate_process_log_artifact(process_log_artifact)
        if (
            process_log_artifact.artifact_digest
            != _json_digest(process_log_artifact.payload)
            or process_log_artifact.start_receipt_digest != start.receipt_digest
            or process_log_artifact.process_kind != start.process_kind
            or completion.process_log_digest
            != process_log_artifact.artifact_digest
        ):
            raise ValueError("process completion does not seal its process log")
        ordered_replay_cases: tuple[ScoringReplayCaseArtifact, ...] = ()
        if start.process_kind == "candidate_scoring":
            if (
                scoring_artifact is None
                or scoring_artifact.artifact_digest
                != _json_digest(scoring_artifact.payload)
                or scoring_artifact.scoring_start_receipt_digest
                != start.receipt_digest
                or completion.output_artifact_digest
                != scoring_artifact.artifact_digest
            ):
                raise ValueError(
                    "scoring completion requires its canonical scoring artifact"
                )
            if not scoring_replay_cases:
                raise ValueError("scoring completion requires semantic replay cases")
            if raw_ingestor_trust_store_path is None:
                raise ValueError(
                    "scoring completion requires the current raw-ingestor trust store"
                )
            ordered_replay_cases = tuple(
                sorted(scoring_replay_cases, key=lambda item: item.case_id)
            )
            completion_raw_trust = _validate_current_scoring_raw_ingestor_receipts(
                ordered_replay_cases,
                raw_ingestor_trust_store_path=(
                    raw_ingestor_trust_store_path
                ),
            )
            replay = self.load_neural_prior_scoring_replay_bundle(
                scoring_artifact.scoring_replay_bundle_digest,
                cases=scoring_replay_cases,
            )
            replay_case_tensors = {
                item.case_id: item.replay_tensors()
                for item in ordered_replay_cases
            }
            execution_device = _semantic_replay_execution_device(
                ordered_replay_cases,
                replay_case_tensors,
            )
            (
                scoring_certification_policy_digest,
                scoring_certification_evidence_digest,
            ) = _validate_scoring_backend_certification(
                execution_device,
                mps_backend_certification_policy,
                mps_backend_certification,
            )
            if (
                not replay.semantic_replay_verified
                or not isinstance(
                    replay.manifest,
                    ScoringReplayBundleManifest,
                )
                or replay.manifest.contract != SEMANTIC_SCORING_REPLAY_CONTRACT
                or replay.manifest.replay_method != SEMANTIC_SCORING_REPLAY_METHOD
                or scoring_artifact.scoring_replay_contract
                != replay.manifest.contract
                or scoring_artifact.scoring_replay_method
                != replay.manifest.replay_method
                or scoring_artifact.semantic_replay_generation_digest
                != SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST
                or replay.manifest.scoring_input_artifact_digest
                != scoring_artifact.scoring_input_artifact_digest
                or replay.manifest.ordered_case_ids
                != scoring_artifact.ordered_case_ids
                or replay.manifest.ordered_evaluation_digests
                != scoring_artifact.ordered_evaluation_digests
                or replay.manifest.runtime_exact_digest
                != scoring_artifact.scoring_runtime_digest
                or replay.manifest.runtime_exact_digest
                != numerical_runtime_manifest(execution_device).exact_digest
                or replay.manifest.scoring_backend_certification_policy_digest
                != scoring_certification_policy_digest
                or replay.manifest.scoring_backend_certification_evidence_digest
                != scoring_certification_evidence_digest
                or scoring_artifact.scoring_backend_certification_policy_digest
                != scoring_certification_policy_digest
                or scoring_artifact.scoring_backend_certification_evidence_digest
                != scoring_certification_evidence_digest
                or replay.manifest.algorithm_source_manifest_digest
                != algorithm_bundle_digest()
                or replay.manifest.raw_ingestor_trust_store_digest
                != completion_raw_trust.content_digest
                or scoring_artifact.raw_ingestor_trust_store_digest
                != completion_raw_trust.content_digest
            ):
                raise ValueError(
                    "scoring completion replay bundle disagrees with its output"
                )
        elif (
            scoring_artifact is not None
            or raw_ingestor_trust_store_path is not None
            or mps_backend_certification_policy is not None
            or mps_backend_certification is not None
        ):
            raise ValueError("training completion cannot seal scoring evidence")
        scheduler_trust = _load_scheduler_trust_store(
            scheduler_trust_store_path
        )
        key = scheduler_trust.keys.get(completion.scheduler_id)
        if (
            key is None
            or scheduler_trust.content_digest
            != completion.scheduler_trust_store_digest
            or key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            ).hex()
            != completion.scheduler_public_key_hex
        ):
            raise ValueError("process completion scheduler is not root-approved")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if start.process_kind == "candidate_scoring":
                assert raw_ingestor_trust_store_path is not None
                final_completion_raw_trust = (
                    _validate_current_scoring_raw_ingestor_receipts(
                        ordered_replay_cases,
                        raw_ingestor_trust_store_path=(
                            raw_ingestor_trust_store_path
                        ),
                    )
                )
                if (
                    scoring_artifact is None
                    or scoring_artifact.raw_ingestor_trust_store_digest
                    != final_completion_raw_trust.content_digest
                ):
                    raise ValueError(
                        "raw-ingestor trust changed during scoring completion"
                    )
            start_row = connection.execute(
                "SELECT receipt_json, created_at FROM "
                "trusted_process_start_receipts_v2 WHERE receipt_digest = ?",
                (start.receipt_digest,),
            ).fetchone()
            expected_start_json = json.dumps(
                start.payload | {"receipt_digest": start.receipt_digest},
                sort_keys=True,
            )
            if start_row is None or start_row[0] != expected_start_json:
                raise ValueError("process completion requires its ledger start row")
            if start.process_kind == "candidate_training":
                if training_target_source_trust_store_path is None:
                    raise ValueError(
                        "training completion requires the current target-source trust store"
                    )
                current_target_source_trust_store = (
                    _load_training_target_source_trust_store(
                        training_target_source_trust_store_path
                    )
                )
                if (
                    training_dataset_derivation is None
                    or training_dataset_derivation.training_dataset_digest
                    not in start.subject_digests
                    or training_dataset_derivation.training_tensor_snapshot_set_digest
                    not in start.subject_digests
                    or training_dataset_derivation.normalization_derivation_artifact_digest
                    not in start.subject_digests
                ):
                    raise ValueError(
                        "training completion requires its dataset derivation and current trust"
                    )
                training_dataset_derivation.validate_target_source_trust_at(
                    current_target_source_trust_store,
                    at=completion.completed_at,
                )
            now = datetime.now(timezone.utc)
            completed_at = datetime.fromisoformat(
                completion.completed_at.replace("Z", "+00:00")
            )
            if completed_at > now:
                raise ValueError("process completion cannot claim a future time")
            if completed_at <= datetime.fromisoformat(start_row[1]):
                raise ValueError("process completion predates start ledger append")
            try:
                connection.execute(
                    "INSERT INTO trusted_process_log_artifacts "
                    "(artifact_digest, start_receipt_digest, process_kind, "
                    "payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        process_log_artifact.artifact_digest,
                        start.receipt_digest,
                        start.process_kind,
                        json.dumps(
                            process_log_artifact.payload
                            | {
                                "artifact_digest": (
                                    process_log_artifact.artifact_digest
                                )
                            },
                            sort_keys=True,
                        ),
                        now.isoformat(),
                    ),
                )
                if scoring_artifact is not None:
                    connection.execute(
                        "INSERT INTO neural_prior_holdout_scoring_artifacts "
                        "(artifact_digest, holdout_plan_digest, "
                        "candidate_manifest_digest, scoring_start_receipt_digest, "
                        "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            scoring_artifact.artifact_digest,
                            scoring_artifact.holdout_plan_digest,
                            scoring_artifact.candidate_manifest_digest,
                            scoring_artifact.scoring_start_receipt_digest,
                            json.dumps(
                                scoring_artifact.payload
                                | {
                                    "artifact_digest": (
                                        scoring_artifact.artifact_digest
                                    )
                                },
                                sort_keys=True,
                            ),
                            now.isoformat(),
                        ),
                    )
                connection.execute(
                    "INSERT INTO trusted_process_completion_receipts "
                    "(receipt_digest, start_receipt_digest, process_kind, "
                    "output_artifact_digest, process_log_digest, receipt_json, "
                    "completed_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        completion.receipt_digest,
                        start.receipt_digest,
                        completion.process_kind,
                        completion.output_artifact_digest,
                        completion.process_log_digest,
                        json.dumps(
                            completion.payload
                            | {"receipt_digest": completion.receipt_digest},
                            sort_keys=True,
                        ),
                        completion.completed_at,
                        now.isoformat(),
                    ),
                )
                if scoring_artifact is not None:
                    self._prepare_raw_trust_artifact_activation(
                        connection,
                        artifact_kind="scoring_completion",
                        artifact_digest=completion.receipt_digest,
                        raw_ingestor_trust_store_digest=(
                            scoring_artifact.raw_ingestor_trust_store_digest
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "trusted process-completion receipt is already registered"
                ) from error
        if scoring_artifact is not None:
            assert raw_ingestor_trust_store_path is not None
            self._activate_raw_trust_artifact(
                artifact_kind="scoring_completion",
                artifact_digest=completion.receipt_digest,
                raw_ingestor_trust_store_digest=(
                    scoring_artifact.raw_ingestor_trust_store_digest
                ),
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
            )
        return completion.receipt_digest

    def append_neural_prior_promotion(
        self,
        evidence: NeuralPriorPromotionEvidence,
        manifest: NeuralPriorCandidateManifest,
        plan: NeuralPriorHoldoutPlan,
        evaluations: tuple[PriorHoldoutEvaluation, ...],
        *,
        scoring_input_artifact: HoldoutScoringInputArtifact,
        scoring_artifact: HoldoutScoringArtifact,
        scoring_process_log: ProcessLogArtifact,
        scoring_completion_receipt: TrustedProcessCompletionReceipt,
        scoring_replay_cases: tuple[ScoringReplayCaseArtifact, ...],
        policy: NeuralPriorPromotionPolicy,
        policy_trust_store_path: str | Path,
        raw_ingestor_trust_store_path: str | Path,
        training_target_source_trust_store_path: str | Path,
    ) -> str:
        """Append one promotion over every preregistered holdout case."""

        ordered_replay_cases = tuple(
            sorted(scoring_replay_cases, key=lambda item: item.case_id)
        )
        promotion_raw_trust = _validate_current_scoring_raw_ingestor_receipts(
            ordered_replay_cases,
            raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
        )
        training_target_source_trust_store = (
            _load_training_target_source_trust_store(
                training_target_source_trust_store_path
            )
        )
        replay = self.load_neural_prior_scoring_replay_bundle(
            scoring_artifact.scoring_replay_bundle_digest,
            cases=scoring_replay_cases,
        )
        if (
            not replay.semantic_replay_verified
            or not isinstance(
                replay.manifest,
                ScoringReplayBundleManifest,
            )
            or replay.manifest.scoring_input_artifact_digest
            != scoring_input_artifact.artifact_digest
            or replay.manifest.ordered_evaluation_digests
            != tuple(
                item.evaluation_digest
                for item in sorted(evaluations, key=lambda item: item.case_id)
            )
            or replay.manifest.scoring_backend_certification_policy_digest
            != scoring_artifact.scoring_backend_certification_policy_digest
            or replay.manifest.scoring_backend_certification_evidence_digest
            != scoring_artifact.scoring_backend_certification_evidence_digest
            or evidence.scoring_backend_certification_policy_digest
            != scoring_artifact.scoring_backend_certification_policy_digest
            or evidence.scoring_backend_certification_evidence_digest
            != scoring_artifact.scoring_backend_certification_evidence_digest
            or replay.manifest.raw_ingestor_trust_store_digest
            != promotion_raw_trust.content_digest
            or scoring_artifact.raw_ingestor_trust_store_digest
            != promotion_raw_trust.content_digest
            or evidence.raw_ingestor_trust_store_digest
            != promotion_raw_trust.content_digest
        ):
            raise ValueError("promotion scoring replay bundle is inconsistent")
        recomputed = compute_neural_prior_promotion(
            manifest,
            plan,
            evaluations,
            scoring_input_artifact=scoring_input_artifact,
            scoring_artifact=scoring_artifact,
            scoring_process_log=scoring_process_log,
            scoring_completion_receipt=scoring_completion_receipt,
            policy=policy,
            policy_trust_store_path=policy_trust_store_path,
            current_training_target_source_trust_store=(
                training_target_source_trust_store
            ),
        )
        validate_neural_prior_holdout_plan(plan)
        validate_neural_prior_candidate_manifest(manifest)
        if (
            recomputed.promotion_evidence_digest
            != evidence.promotion_evidence_digest
            or evidence.training_target_source_trust_store_digest
            != training_target_source_trust_store.content_digest
        ):
            raise ValueError("neural-prior promotion evidence is not reproducible")
        validate_neural_prior_promotion(evidence)
        with self._connect() as connection:
            final_target_source_trust = (
                _load_training_target_source_trust_store(
                    training_target_source_trust_store_path
                )
            )
            if (
                final_target_source_trust.content_digest
                != evidence.training_target_source_trust_store_digest
            ):
                raise ValueError(
                    "training target-source trust changed during promotion append"
                )
            final_promotion_raw_trust = (
                _validate_current_scoring_raw_ingestor_receipts(
                    ordered_replay_cases,
                    raw_ingestor_trust_store_path=(
                        raw_ingestor_trust_store_path
                    ),
                )
            )
            if (
                evidence.raw_ingestor_trust_store_digest
                != final_promotion_raw_trust.content_digest
            ):
                raise ValueError(
                    "raw-ingestor trust changed during promotion append"
                )
            input_row = connection.execute(
                "SELECT payload_json FROM "
                "neural_prior_holdout_scoring_input_artifacts "
                "WHERE artifact_digest = ? AND holdout_plan_digest = ?",
                (scoring_input_artifact.artifact_digest, plan.plan_digest),
            ).fetchone()
            expected_input_json = json.dumps(
                scoring_input_artifact.payload
                | {"artifact_digest": scoring_input_artifact.artifact_digest},
                sort_keys=True,
            )
            if input_row is None or input_row[0] != expected_input_json:
                raise ValueError("promotion scoring input artifact is not registered")
            rule_row = connection.execute(
                "SELECT rule_digest FROM neural_prior_holdout_plan_rule_bindings "
                "WHERE holdout_plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if (
                rule_row is None
                or rule_row[0]
                != scoring_input_artifact.promotion_decision_rule_digest
                or rule_row[0] != policy.decision_rule_digest
            ):
                raise ValueError("promotion decision rule is not preregistered")
            family_row = connection.execute(
                "SELECT family_digest FROM "
                "neural_prior_holdout_plan_experiment_bindings "
                "WHERE holdout_plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if (
                family_row is None
                or family_row[0]
                != plan.promotion_experiment_family.family_digest
                or family_row[0] != evidence.promotion_experiment_family_digest
            ):
                raise ValueError("promotion experiment family is not preregistered")
            plan_row = connection.execute(
                "SELECT plan_json, created_at FROM neural_prior_holdout_plans "
                "WHERE plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            if plan_row is None or plan_row[0] != json.dumps(
                asdict(plan), sort_keys=True
            ):
                raise ValueError("promotion holdout plan is not pre-registered")
            catalog_row = connection.execute(
                "SELECT result_digest, result_json "
                "FROM neural_prior_event_catalog_results "
                "WHERE plan_digest = ?",
                (plan.plan_digest,),
            ).fetchone()
            expected_catalog_json = json.dumps(
                manifest.physical_event_catalog_result.payload
                | {
                    "result_digest": (
                        manifest.physical_event_catalog_result.result_digest
                    )
                },
                sort_keys=True,
            )
            if catalog_row is None or tuple(catalog_row) != (
                manifest.physical_event_catalog_result.result_digest,
                expected_catalog_json,
            ):
                raise ValueError(
                    "promotion physical event catalog result is not registered"
                )
            training_catalog_row = connection.execute(
                "SELECT result_digest, plan_json, result_json FROM "
                "neural_prior_training_event_catalog_results "
                "WHERE plan_digest = ?",
                (manifest.training_physical_event_catalog_plan.plan_digest,),
            ).fetchone()
            expected_training_plan_json = json.dumps(
                manifest.training_physical_event_catalog_plan.payload
                | {
                    "plan_digest": (
                        manifest.training_physical_event_catalog_plan.plan_digest
                    )
                },
                sort_keys=True,
            )
            expected_training_result_json = json.dumps(
                manifest.training_physical_event_catalog_result.payload
                | {
                    "result_digest": (
                        manifest.training_physical_event_catalog_result.result_digest
                    )
                },
                sort_keys=True,
            )
            if training_catalog_row is None or tuple(training_catalog_row) != (
                manifest.training_physical_event_catalog_result.result_digest,
                expected_training_plan_json,
                expected_training_result_json,
            ):
                raise ValueError(
                    "promotion training event catalog is not registered"
                )
            for start, completion in (
                (
                    manifest.candidate_training_start_receipt,
                    manifest.candidate_training_completion_receipt,
                ),
                (
                    manifest.candidate_scoring_start_receipt,
                    scoring_completion_receipt,
                ),
            ):
                start_row = connection.execute(
                    "SELECT receipt_json FROM trusted_process_start_receipts_v2 "
                    "WHERE receipt_digest = ?",
                    (start.receipt_digest,),
                ).fetchone()
                completion_row = connection.execute(
                    "SELECT receipt_json FROM trusted_process_completion_receipts "
                    "WHERE receipt_digest = ? AND start_receipt_digest = ?",
                    (completion.receipt_digest, start.receipt_digest),
                ).fetchone()
                expected_start_json = json.dumps(
                    start.payload | {"receipt_digest": start.receipt_digest},
                    sort_keys=True,
                )
                expected_completion_json = json.dumps(
                    completion.payload
                    | {"receipt_digest": completion.receipt_digest},
                    sort_keys=True,
                )
                if (
                    start_row is None
                    or start_row[0] != expected_start_json
                    or completion_row is None
                    or completion_row[0] != expected_completion_json
                ):
                    raise ValueError(
                        "promotion trusted process receipt chain is not registered"
                    )
                if start.process_kind == "candidate_scoring":
                    self._require_raw_trust_artifact_usable(
                        connection,
                        artifact_kind="scoring_completion",
                        artifact_digest=completion.receipt_digest,
                        raw_ingestor_trust_store_digest=(
                            evidence.raw_ingestor_trust_store_digest
                        ),
                    )
            scoring_artifact_row = connection.execute(
                "SELECT payload_json FROM "
                "neural_prior_holdout_scoring_artifacts "
                "WHERE artifact_digest = ? AND scoring_start_receipt_digest = ?",
                (
                    scoring_artifact.artifact_digest,
                    manifest.candidate_scoring_start_receipt.receipt_digest,
                ),
            ).fetchone()
            scoring_log_row = connection.execute(
                "SELECT payload_json FROM trusted_process_log_artifacts "
                "WHERE artifact_digest = ? AND start_receipt_digest = ?",
                (
                    scoring_process_log.artifact_digest,
                    manifest.candidate_scoring_start_receipt.receipt_digest,
                ),
            ).fetchone()
            expected_scoring_artifact_json = json.dumps(
                scoring_artifact.payload
                | {"artifact_digest": scoring_artifact.artifact_digest},
                sort_keys=True,
            )
            expected_scoring_log_json = json.dumps(
                scoring_process_log.payload
                | {"artifact_digest": scoring_process_log.artifact_digest},
                sort_keys=True,
            )
            if (
                scoring_artifact_row is None
                or scoring_artifact_row[0] != expected_scoring_artifact_json
                or scoring_log_row is None
                or scoring_log_row[0] != expected_scoring_log_json
            ):
                raise ValueError(
                    "promotion scoring artifacts are not registered"
                )
            plan_created = datetime.fromisoformat(plan_row[1])
            if plan.mode == "prospective" and any(
                plan_created
                >= datetime.fromisoformat(item.issue_time.replace("Z", "+00:00"))
                for item in plan.cases
            ):
                raise ValueError("holdout plan was registered after forecast issue")
            receipt_rows = tuple(
                connection.execute(
                    "SELECT receipt_digest, receipt_json, created_at "
                    "FROM realized_intervention_receipts"
                )
            )
            recorded = {row[0]: json.loads(row[1]) for row in receipt_rows}
            recorded_approvals = {
                row[0]
                for row in connection.execute(
                    "SELECT approval_evidence_digest "
                    "FROM variational_learning_approvals"
                )
            }
            if set(manifest.training_learning_approval_digests) - recorded_approvals:
                raise ValueError("candidate training approvals are not recorded")
            training_interventions = set(manifest.training_intervention_digests)
            if training_interventions - set(recorded):
                raise ValueError("candidate training interventions are not recorded")
            for digest in training_interventions:
                if recorded[digest]["actual_input_bundle_digest"] not in (
                    manifest.training_input_bundle_digests
                ):
                    raise ValueError("candidate training input lineage is inconsistent")
                training_decision = connection.execute(
                    "SELECT decision_json FROM prospective_intervention_decisions "
                    "WHERE decision_digest = ?",
                    (recorded[digest]["decision_digest"],),
                ).fetchone()
                if training_decision is None:
                    raise ValueError("candidate training decision is not recorded")
                decision_payload = json.loads(training_decision[0])
                if decision_payload["decision_basis_digest"] not in (
                    manifest.training_learning_approval_digests
                ):
                    raise ValueError("training receipt is not linked to its approval")
            promotion_row: dict[str, object] = {
                "promotion_evidence_digest": evidence.promotion_evidence_digest,
                "candidate_prior_digest": evidence.candidate_prior_digest,
                "parent_prior_digest": evidence.parent_prior_digest,
                "candidate_manifest_digest": evidence.candidate_manifest_digest,
                "candidate_manifest_json": json.dumps(
                    asdict(manifest), sort_keys=True
                ),
                "holdout_plan_digest": plan.plan_digest,
                "policy_digest": evidence.policy_digest,
                "trust_store_digest": evidence.trust_store_digest,
                "evaluation_digests_json": json.dumps(
                    list(evidence.evaluation_digests)
                ),
                "evaluation_payloads_json": json.dumps(
                    [_evaluation_audit_payload(item) for item in evaluations],
                    sort_keys=True,
                ),
                "intervention_digests_json": json.dumps([]),
                "realized_intervention_count": evidence.holdout_case_count,
                "material_outcome_count": evidence.material_case_count,
                "distinct_case_count": evidence.distinct_case_count,
                "distinct_storm_count": evidence.distinct_storm_count,
                "distinct_day_count": evidence.distinct_day_count,
                "distinct_radar_count": evidence.distinct_radar_count,
                "distinct_regime_count": evidence.distinct_regime_count,
                "distinct_range_regime_count": (
                    evidence.distinct_range_regime_count
                ),
                "beneficial_fraction": evidence.beneficial_fraction,
                "beneficial_fraction_lower_bound": (
                    evidence.beneficial_fraction_lower_bound
                ),
                "harmful_fraction": evidence.harmful_fraction,
                "harmful_fraction_upper_bound": evidence.harmful_fraction_upper_bound,
                "mean_normalized_improvement": (
                    evidence.mean_normalized_improvement
                ),
                "mean_improvement_lower_bound": evidence.mean_improvement_lower_bound,
                "maximum_normalized_degradation": (
                    evidence.maximum_normalized_degradation
                ),
                "prior_gaussian_nll_increase_upper_bound": 0.0,
                "prior_support_brier_increase_upper_bound": (
                    evidence.prior_support_brier_increase_upper_bound
                ),
                "prior_underdispersion_increase_upper_bound": 0.0,
                "prior_conditional_underdispersion_increase_upper_bound": (
                    evidence.prior_conditional_underdispersion_increase_upper_bound
                ),
                "prior_echo_support_miss_increase_upper_bound": (
                    evidence.prior_echo_support_miss_increase_upper_bound
                ),
                "prior_echo_object_miss_increase_upper_bound": (
                    evidence.prior_echo_object_miss_increase_upper_bound
                ),
                "prior_echo_intensity_nll_increase_upper_bound": (
                    evidence.prior_echo_intensity_nll_increase_upper_bound
                ),
                "prior_clear_sky_false_echo_increase_upper_bound": (
                    evidence.prior_clear_sky_false_echo_increase_upper_bound
                ),
                "prior_echo_component_status": evidence.prior_echo_component_status,
                "prior_clear_sky_component_status": (
                    evidence.prior_clear_sky_component_status
                ),
                "prior_echo_case_count": evidence.prior_echo_case_count,
                "prior_clear_sky_case_count": evidence.prior_clear_sky_case_count,
                "prior_echo_cluster_count": evidence.prior_echo_cluster_count,
                "prior_clear_sky_cluster_count": (
                    evidence.prior_clear_sky_cluster_count
                ),
                "simultaneous_inference_test_count": (
                    evidence.simultaneous_inference_test_count
                ),
                "eligible": int(evidence.eligible),
                "rejection_reasons_json": json.dumps(
                    list(evidence.rejection_reasons)
                ),
                "evidence_contract": evidence.contract,
                "evidence_payload_json": json.dumps(
                    evidence._payload(), sort_keys=True
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            columns = tuple(promotion_row)
            try:
                connection.execute(
                    "INSERT INTO neural_prior_experiment_family_consumptions "
                    "(family_digest, promotion_evidence_digest, consumed_at) "
                    "VALUES (?, ?, ?)",
                    (
                        evidence.promotion_experiment_family_digest,
                        evidence.promotion_evidence_digest,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "promotion experiment family is already consumed"
                ) from error
            connection.execute(
                "INSERT INTO neural_prior_promotions "
                f"({','.join(columns)}) VALUES "
                f"({','.join('?' for _ in columns)})",
                tuple(promotion_row[name] for name in columns),
            )
            self._prepare_raw_trust_artifact_activation(
                connection,
                artifact_kind="promotion_evidence",
                artifact_digest=evidence.promotion_evidence_digest,
                raw_ingestor_trust_store_digest=(
                    evidence.raw_ingestor_trust_store_digest
                ),
            )
            for sampling_unit_digest in (
                plan.promotion_experiment_family.meteorological_sampling_unit_digests
            ):
                reservation = connection.execute(
                    "SELECT family_digest FROM "
                    "promotion_sampling_unit_reservations "
                    "WHERE sampling_unit_digest = ?",
                    (sampling_unit_digest,),
                ).fetchone()
                if reservation != (evidence.promotion_experiment_family_digest,):
                    raise ValueError(
                        "promotion sampling unit lacks its family reservation"
                    )
                try:
                    connection.execute(
                        "INSERT INTO promotion_sampling_unit_consumptions "
                        "(sampling_unit_digest,family_digest,"
                        "promotion_evidence_digest,consumed_at) VALUES (?,?,?,?)",
                        (
                            sampling_unit_digest,
                            evidence.promotion_experiment_family_digest,
                            evidence.promotion_evidence_digest,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise FileExistsError(
                        "meteorological sampling unit is already consumed"
                    ) from error
            raw_identity_rows = connection.execute(
                "SELECT raw_volume_identity_digest FROM "
                "promotion_raw_volume_identity_reservations "
                "WHERE family_digest = ? ORDER BY raw_volume_identity_digest",
                (evidence.promotion_experiment_family_digest,),
            ).fetchall()
            if not raw_identity_rows:
                raise ValueError(
                    "promotion raw volumes lack their family reservation"
                )
            for (raw_identity_digest,) in raw_identity_rows:
                try:
                    connection.execute(
                        "INSERT INTO promotion_raw_volume_identity_consumptions "
                        "(raw_volume_identity_digest,family_digest,"
                        "promotion_evidence_digest,consumed_at) VALUES (?,?,?,?)",
                        (
                            raw_identity_digest,
                            evidence.promotion_experiment_family_digest,
                            evidence.promotion_evidence_digest,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise FileExistsError(
                        "canonical raw volume is already consumed"
                    ) from error
        self._activate_raw_trust_artifact(
            artifact_kind="promotion_evidence",
            artifact_digest=evidence.promotion_evidence_digest,
            raw_ingestor_trust_store_digest=(
                evidence.raw_ingestor_trust_store_digest
            ),
            raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
        )
        return evidence.promotion_evidence_digest

    def issue_neural_prior_promotion_deployment_certificate(
        self,
        promotion_evidence_digest: str,
        *,
        scoring_replay_cases: tuple[ScoringReplayCaseArtifact, ...],
        ledger_signer: DeploymentAuthoritySigner,
        deployment_signer: DeploymentAuthoritySigner,
        authority_trust_store_path: str | Path,
        raw_ingestor_trust_store_path: str | Path,
        training_target_source_trust_store_path: str | Path,
    ) -> LedgeredPromotionDeploymentCertificate:
        """Issue the sole deployment-capable view of a ledgered promotion."""

        evidence = self.load_neural_prior_promotion(promotion_evidence_digest)
        if type(evidence) is not NeuralPriorPromotionEvidence:
            raise ValueError("legacy promotion evidence is audit-only")
        assert isinstance(evidence, NeuralPriorPromotionEvidence)
        if not evidence.deployment_eligible:
            raise ValueError("ineligible promotion cannot receive a certificate")
        if any(
            value is not None
            for value in (
                evidence.scoring_backend_certification_policy_digest,
                evidence.scoring_backend_certification_evidence_digest,
            )
        ):
            raise ValueError("automatic deployment requires CPU-scored evidence")
        authority_trust = _load_promotion_deployment_authority_trust_store(
            authority_trust_store_path
        )
        ordered_replay_cases = tuple(
            sorted(scoring_replay_cases, key=lambda item: item.case_id)
        )
        current_raw_ingestor_trust = (
            _validate_current_scoring_raw_ingestor_receipts(
                ordered_replay_cases,
                raw_ingestor_trust_store_path=(
                    raw_ingestor_trust_store_path
                ),
            )
        )
        current_target_source_trust = (
            _load_training_target_source_trust_store(
                training_target_source_trust_store_path
            )
        )
        if (
            current_target_source_trust.content_digest
            != evidence.training_target_source_trust_store_digest
        ):
            raise ValueError(
                "training target-source trust changed before certificate issuance"
            )
        if (
            ledger_signer.authority_id == deployment_signer.authority_id
            or ledger_signer.public_key_hex == deployment_signer.public_key_hex
        ):
            raise ValueError(
                "ledger and promotion certificates require separate authority keys"
            )
        replay = self.load_neural_prior_scoring_replay_bundle(
            evidence.scoring_replay_bundle_digest,
            cases=scoring_replay_cases,
        )
        if (
            not replay.semantic_replay_verified
            or not isinstance(replay.manifest, ScoringReplayBundleManifest)
            or replay.manifest.bundle_digest
            != evidence.scoring_replay_bundle_digest
            or tuple(item.evaluation_digest for item in replay.evaluations)
            != evidence.evaluation_digests
        ):
            raise ValueError(
                "deployment certificate requires intact scoring replay preimages"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.row_factory = sqlite3.Row
            final_raw_ingestor_trust = (
                _validate_current_scoring_raw_ingestor_receipts(
                    ordered_replay_cases,
                    raw_ingestor_trust_store_path=(
                        raw_ingestor_trust_store_path
                    ),
                )
            )
            if (
                final_raw_ingestor_trust.content_digest
                != current_raw_ingestor_trust.content_digest
            ):
                raise ValueError(
                    "raw-ingestor trust store changed during certificate issuance"
                )
            final_target_source_trust = (
                _load_training_target_source_trust_store(
                    training_target_source_trust_store_path
                )
            )
            if (
                final_target_source_trust.content_digest
                != current_target_source_trust.content_digest
                or final_target_source_trust.content_digest
                != evidence.training_target_source_trust_store_digest
            ):
                raise ValueError(
                    "training target-source trust changed during certificate issuance"
                )
            promotion_row = connection.execute(
                "SELECT evidence_payload_json FROM neural_prior_promotions "
                "WHERE promotion_evidence_digest = ?",
                (promotion_evidence_digest,),
            ).fetchone()
            replay_row = connection.execute(
                "SELECT bundle_digest FROM neural_prior_scoring_replay_bundles "
                "WHERE bundle_digest = ?",
                (evidence.scoring_replay_bundle_digest,),
            ).fetchone()
            scoring_row = connection.execute(
                "SELECT artifact_digest,payload_json FROM "
                "neural_prior_holdout_scoring_artifacts "
                "WHERE artifact_digest = ?",
                (evidence.scoring_artifact_digest,),
            ).fetchone()
            completion_row = connection.execute(
                "SELECT receipt_digest,output_artifact_digest,receipt_json FROM "
                "trusted_process_completion_receipts "
                "WHERE receipt_digest = ?",
                (evidence.scoring_completion_receipt_digest,),
            ).fetchone()
            head_row = connection.execute(
                "SELECT ledger_instance_digest,sequence_number,"
                "certificate_digest,ledger_chain_head_digest FROM "
                "deployment_certificate_chain_head WHERE singleton = 1"
            ).fetchone()
            if (
                promotion_row is None
                or replay_row is None
                or scoring_row is None
                or completion_row is None
                or head_row is None
                or promotion_row["evidence_payload_json"]
                != json.dumps(evidence._payload(), sort_keys=True)
            ):
                raise ValueError(
                    "deployment certificate requires complete ledger preimages"
                )
            for artifact_kind, artifact_digest in (
                ("promotion_evidence", evidence.promotion_evidence_digest),
                ("scoring_replay_bundle", evidence.scoring_replay_bundle_digest),
                (
                    "scoring_completion",
                    evidence.scoring_completion_receipt_digest,
                ),
            ):
                self._require_raw_trust_artifact_usable(
                    connection,
                    artifact_kind=artifact_kind,
                    artifact_digest=artifact_digest,
                    raw_ingestor_trust_store_digest=(
                        evidence.raw_ingestor_trust_store_digest
                    ),
                )
            scoring_artifact = _decode_holdout_scoring_artifact(
                scoring_row["payload_json"],
                evidence.scoring_artifact_digest,
            )
            if type(scoring_artifact) is not HoldoutScoringArtifact:
                raise ValueError("current deployment requires current scoring artifact")
            completion_receipt = _decode_completion_receipt(
                completion_row["receipt_json"],
                evidence.scoring_completion_receipt_digest,
            )
            if (
                scoring_artifact.scoring_replay_bundle_digest
                != evidence.scoring_replay_bundle_digest
                or completion_row["output_artifact_digest"]
                != evidence.scoring_artifact_digest
                or completion_receipt.output_artifact_digest
                != evidence.scoring_artifact_digest
            ):
                raise ValueError(
                    "deployment certificate scoring preimages disagree"
                )
            ledger_issued_at = ledger_signer.signing_time()
            promotion_issued_at = deployment_signer.signing_time()
            try:
                completed_time = datetime.fromisoformat(
                    completion_receipt.completed_at.replace("Z", "+00:00")
                )
                ledger_issue_time = datetime.fromisoformat(
                    ledger_issued_at.replace("Z", "+00:00")
                )
                promotion_issue_time = datetime.fromisoformat(
                    promotion_issued_at.replace("Z", "+00:00")
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    "deployment certificate issuance chronology is invalid"
                ) from error
            trusted_ledger_now = datetime.now(timezone.utc)
            if not (
                completed_time
                <= ledger_issue_time
                <= promotion_issue_time
                <= trusted_ledger_now
            ):
                raise ValueError(
                    "deployment certificate issuance chronology is invalid"
                )
            sequence_number = int(head_row["sequence_number"]) + 1
            ledger_receipt = _issue_ledger_issuance_receipt(
                ledger_instance_digest=head_row["ledger_instance_digest"],
                sequence_number=sequence_number,
                previous_certificate_digest=head_row["certificate_digest"],
                promotion_evidence_digest=evidence.promotion_evidence_digest,
                scoring_replay_bundle_digest=evidence.scoring_replay_bundle_digest,
                scoring_replay_archive_sha256=(
                    replay.manifest.tensor_archive_sha256
                ),
                scoring_evaluation_payload_sha256=(
                    replay.manifest.evaluation_payload_sha256
                ),
                scoring_artifact_digest=evidence.scoring_artifact_digest,
                scoring_completion_receipt_digest=(
                    evidence.scoring_completion_receipt_digest
                ),
                scoring_completion_completed_at=completion_receipt.completed_at,
                issued_at=ledger_issued_at,
                signer=ledger_signer,
                authority_trust_store=authority_trust,
            )
            certificate = _issue_ledgered_promotion_deployment_certificate(
                evidence,
                issued_at=promotion_issued_at,
                ledger_issuance_receipt=ledger_receipt,
                signer=deployment_signer,
                authority_trust_store=authority_trust,
                raw_ingestor_trust_store_digest=(
                    final_raw_ingestor_trust.content_digest
                ),
            )
            payload_json = json.dumps(
                certificate.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            try:
                connection.execute(
                    "INSERT INTO neural_prior_promotion_deployment_certificates_v3 "
                    "(certificate_digest,ledger_instance_digest,sequence_number,"
                    "promotion_evidence_digest,"
                    "previous_certificate_digest,ledger_chain_head_digest,"
                    "payload_json,issued_at,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        certificate.certificate_digest,
                        certificate.ledger_instance_digest,
                        certificate.sequence_number,
                        certificate.promotion_evidence_digest,
                        certificate.previous_certificate_digest,
                        certificate.ledger_chain_head_digest,
                        payload_json,
                        certificate.issued_at,
                        promotion_issued_at,
                    ),
                )
                self._prepare_raw_trust_artifact_activation(
                    connection,
                    artifact_kind="promotion_deployment_certificate",
                    artifact_digest=certificate.certificate_digest,
                    raw_ingestor_trust_store_digest=(
                        certificate.raw_ingestor_trust_store_digest
                    ),
                )
                updated = connection.execute(
                    "UPDATE deployment_certificate_chain_head SET "
                    "sequence_number = ?, certificate_digest = ?, "
                    "ledger_chain_head_digest = ?, updated_at = ? "
                    "WHERE singleton = 1 AND sequence_number = ? AND "
                    "certificate_digest = ?",
                    (
                        certificate.sequence_number,
                        certificate.certificate_digest,
                        certificate.ledger_chain_head_digest,
                        promotion_issued_at,
                        int(head_row["sequence_number"]),
                        head_row["certificate_digest"],
                    ),
                )
                if updated.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "deployment certificate chain head changed"
                    )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "promotion deployment certificate already exists"
                ) from error
        self._activate_raw_trust_artifact(
            artifact_kind="promotion_deployment_certificate",
            artifact_digest=certificate.certificate_digest,
            raw_ingestor_trust_store_digest=(
                certificate.raw_ingestor_trust_store_digest
            ),
            raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
        )
        return certificate

    def issue_operational_deployment_decision(
        self,
        decision_payload: dict[str, object],
        *,
        deployment_bundle_release_approval: DeploymentBundleReleaseApproval,
        deployment_runtime_activation_receipt: (
            DeploymentRuntimeActivationReceipt
        ),
        promotion_deployment_certificate: LedgeredPromotionDeploymentCertificate,
        promotion_evidence: NeuralPriorPromotionEvidence,
        policy: DeployedNeuralPriorPolicy,
        policy_trust_store_digest: str,
        ledger_signer: DeploymentAuthoritySigner,
        operational_signer: DeploymentAuthoritySigner,
        authority_trust_store_path: str | Path,
        raw_ingestor_trust_store_path: str | Path = (
            "/etc/advar/raw-ingestors.json"
        ),
        training_target_source_trust_store_path: str | Path = (
            "/etc/advar/training-target-sources.json"
        ),
        regime_evidence: RegimeClassificationEvidence | None = None,
        range_partition_evidence: RangePartitionEvidence | None = None,
        range_geometry_contract: (
            RangeGeometryContract | MosaicRangeGeometryContract | None
        ) = None,
        regime_classifier: NeuralPriorRegimeClassifier | None = None,
        input_run: ForecastRunContract | None = None,
        range_grid_x_m: Tensor | None = None,
        range_grid_y_m: Tensor | None = None,
    ) -> OperationalDeploymentDecisionCertificate:
        """Issue or resume one failure-atomic operational decision."""

        authority_trust = _load_promotion_deployment_authority_trust_store(
            authority_trust_store_path
        )
        _validate_deployment_bundle_release_approval(
            deployment_bundle_release_approval,
            authority_trust_store=authority_trust,
            required_valid_through=str(
                decision_payload.get("publication_time", "")
            ),
            require_deployable=True,
        )
        _validate_deployment_runtime_activation_receipt(
            deployment_runtime_activation_receipt,
            release_approval=deployment_bundle_release_approval,
            authority_trust_store=authority_trust,
            required_valid_through=str(decision_payload.get("publication_time", "")),
            require_current_runtime=True,
        )
        if (
            decision_payload.get("deployment_bundle_release_approval")
            != deployment_bundle_release_approval.payload
            | {
                "approval_digest": (
                    deployment_bundle_release_approval.approval_digest
                )
            }
        ):
            raise ValueError(
                "operational decision requires its exact bundle release approval"
            )
        if (
            decision_payload.get("deployment_runtime_activation_receipt")
            != deployment_runtime_activation_receipt.payload
            | {
                "receipt_digest": (
                    deployment_runtime_activation_receipt.receipt_digest
                )
            }
        ):
            raise ValueError(
                "operational decision requires its exact runtime activation"
            )
        current_raw_ingestor_trust = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        current_target_source_trust = (
            _load_training_target_source_trust_store(
                training_target_source_trust_store_path
            )
        )
        if (
            current_target_source_trust.content_digest
            != promotion_evidence.training_target_source_trust_store_digest
            or current_target_source_trust.content_digest
            != promotion_deployment_certificate
            .training_target_source_trust_store_digest
            or decision_payload.get(
                "training_target_source_trust_store_digest"
            )
            != current_target_source_trust.content_digest
        ):
            raise ValueError(
                "operational decision target-source trust is no longer current"
            )
        with self._connect() as connection:
            self._require_raw_trust_artifact_usable(
                connection,
                artifact_kind="promotion_deployment_certificate",
                artifact_digest=(
                    promotion_deployment_certificate.certificate_digest
                ),
                raw_ingestor_trust_store_digest=(
                    promotion_deployment_certificate
                    .raw_ingestor_trust_store_digest
                ),
            )
        role_authority_ids = (
            ledger_signer.authority_id,
            operational_signer.authority_id,
            promotion_deployment_certificate.authority_id,
            deployment_bundle_release_approval.authority_id,
            deployment_runtime_activation_receipt.authority_id,
        )
        role_public_keys = (
            ledger_signer.public_key_hex,
            operational_signer.public_key_hex,
            promotion_deployment_certificate.authority_public_key_hex,
            deployment_bundle_release_approval.authority_public_key_hex,
            deployment_runtime_activation_receipt.authority_public_key_hex,
        )
        if (
            len(set(role_authority_ids)) != len(role_authority_ids)
            or len(set(role_public_keys)) != len(role_public_keys)
        ):
            raise ValueError(
                "release, runtime, promotion, ledger, and operational roles "
                "require separate keys"
            )
        canonical_decision = json.dumps(
            decision_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        operational_cycle_id = str(
            decision_payload.get("operational_cycle_id", "")
        )
        input_plan_digest = str(decision_payload.get("input_plan_digest", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", operational_cycle_id) or not re.fullmatch(
            r"[0-9a-f]{64}", input_plan_digest
        ):
            raise ValueError("operational decision identity is invalid")
        if operational_cycle_id != _json_digest(
            {
                "contract": "advar-operational-cycle-v2",
                "input_plan_digest": input_plan_digest,
                "full_analysis_input_digest": decision_payload.get(
                    "full_analysis_input_digest"
                ),
                "analysis_input_derivation_artifact_digest": (
                    decision_payload.get(
                        "analysis_input_derivation_artifact_digest"
                    )
                ),
                "global_raw_resolution_receipt_digest": decision_payload.get(
                    "global_raw_resolution_receipt_digest"
                ),
                "resolved_raw_volume_identity_set_digest": (
                    decision_payload.get(
                        "resolved_raw_volume_identity_set_digest"
                    )
                ),
            }
        ):
            raise ValueError("operational cycle provenance is invalid")
        deadline = datetime.fromisoformat(
            str(decision_payload["decision_deadline"]).replace("Z", "+00:00")
        )
        publication_time = datetime.fromisoformat(
            str(decision_payload["publication_time"]).replace("Z", "+00:00")
        )
        derivation_digest = str(
            decision_payload.get("analysis_input_derivation_artifact_digest", "")
        )
        global_resolution_digest = str(
            decision_payload.get("global_raw_resolution_receipt_digest", "")
        )
        raw_identity_set_digest = str(
            decision_payload.get("resolved_raw_volume_identity_set_digest", "")
        )
        provenance_commitment_digest = str(
            decision_payload.get(
                "analysis_input_provenance_commitment_digest",
                "",
            )
        )
        if (
            decision_payload.get("analysis_processor_trust_store_digest")
            != authority_trust.content_digest
            or decision_payload.get("raw_ingestor_trust_store_digest")
            != current_raw_ingestor_trust.content_digest
            or promotion_deployment_certificate.raw_ingestor_trust_store_digest
            != current_raw_ingestor_trust.content_digest
            or provenance_commitment_digest
            != _json_digest(
                {
                    "contract": (
                        "operational-analysis-input-provenance-commitment-v2"
                    ),
                    "analysis_input_derivation_artifact_digest": (
                        derivation_digest
                    ),
                    "global_raw_resolution_receipt_digest": (
                        global_resolution_digest
                    ),
                    "resolved_raw_volume_identity_set_digest": (
                        raw_identity_set_digest
                    ),
                    "analysis_processor_trust_store_digest": (
                        authority_trust.content_digest
                    ),
                    "raw_ingestor_trust_store_digest": (
                        current_raw_ingestor_trust.content_digest
                    ),
                }
            )
        ):
            raise ValueError("operational analysis provenance is untrusted")
        try:
            self.reconcile_prepared_analysis_input_provenance(
                derivation_digest,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                analysis_processor_trust_store_path=authority_trust_store_path,
            )
        except KeyError as error:
            raise ValueError(
                "operational decision requires committed analysis provenance"
            ) from error
        published_certificate: OperationalDeploymentDecisionCertificate | None = None
        with self._connect() as connection:
            provenance_row = connection.execute(
                "SELECT payload_json,arrays_sha256,metadata_sha256,path,usable,"
                "raw_ingestor_trust_store_digest,raw_trust_validated_at FROM "
                "analysis_input_provenance_commits "
                "WHERE artifact_digest = ? AND input_plan_digest = ? "
                "AND raw_resolution_receipt_digest = ?",
                (
                    derivation_digest,
                    input_plan_digest,
                    global_resolution_digest,
                ),
            ).fetchone()
        if (
            provenance_row is None
            or int(provenance_row[4]) != 1
            or provenance_row[5] != current_raw_ingestor_trust.content_digest
            or provenance_row[6] is None
        ):
            raise ValueError(
                "operational decision requires committed analysis provenance"
            )
        provenance_path = Path(str(provenance_row[3]))
        expected_provenance_path = (
            self.analysis_input_provenance_dir / derivation_digest
        )
        arrays_path = provenance_path / "source_and_derived_arrays.npz"
        metadata_path = provenance_path / "provenance.json"
        if (
            provenance_path != expected_provenance_path
            or provenance_path.is_symlink()
            or not provenance_path.is_dir()
            or not arrays_path.is_file()
            or not metadata_path.is_file()
            or _file_digest(arrays_path) != str(provenance_row[1])
            or _file_digest(metadata_path) != str(provenance_row[2])
        ):
            raise ValueError(
                "operational decision requires durable analysis provenance"
            )
        try:
            provenance_payload = json.loads(str(provenance_row[0]))
            raw_identities = provenance_payload[
                "canonical_raw_volume_identity_digests"
            ]
            input_plan_payload = dict(provenance_payload["input_plan"])
            retained_input_plan_digest = input_plan_payload.pop(
                "plan_digest"
            )
            valid_times = input_plan_payload.get("valid_times")
            if not isinstance(valid_times, list):
                raise TypeError
            input_plan_payload["valid_times"] = tuple(valid_times)
            committed_input_plan = NeuralPriorInputPlan(
                **cast(Any, input_plan_payload)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "committed analysis provenance payload is invalid"
            ) from error
        if (
            not isinstance(provenance_payload, dict)
            or not isinstance(raw_identities, list)
            or any(type(item) is not str for item in raw_identities)
            or provenance_payload.get("artifact_digest") != derivation_digest
            or provenance_payload.get("input_plan_digest") != input_plan_digest
            or provenance_payload.get("global_raw_resolution_receipt_digest")
            != global_resolution_digest
        ):
            raise ValueError("committed analysis provenance changed")
        if (
            provenance_payload.get("full_analysis_input_digest")
            != decision_payload.get("full_analysis_input_digest")
        ):
            raise ValueError(
                "committed analysis provenance full input changed"
            )
        if (
            committed_input_plan.plan_digest != retained_input_plan_digest
            or committed_input_plan.plan_digest != input_plan_digest
            or committed_input_plan.observation_valid_time
            != decision_payload.get("observation_valid_time")
            or committed_input_plan.input_available_time
            != decision_payload.get("input_available_time")
            or committed_input_plan.decision_deadline
            != decision_payload.get("decision_deadline")
            or committed_input_plan.publication_time
            != decision_payload.get("publication_time")
        ):
            raise ValueError(
                "committed analysis provenance input plan changed"
            )
        if raw_identity_set_digest != _json_digest(
            {
                "contract": "resolved-raw-volume-identity-set-v1",
                "identity_digests": sorted(raw_identities),
            }
        ):
            raise ValueError(
                "committed analysis provenance raw identity set changed"
            )
        if decision_payload.get("routing_semantic_replay_verified") is True:
            if (
                type(regime_classifier) is not NeuralPriorRegimeClassifier
                or type(input_run) is not ForecastRunContract
                or type(range_grid_x_m) is not Tensor
                or type(range_grid_y_m) is not Tensor
                or type(regime_evidence) is not RegimeClassificationEvidence
                or type(range_partition_evidence) is not RangePartitionEvidence
                or type(range_geometry_contract)
                not in (RangeGeometryContract, MosaicRangeGeometryContract)
            ):
                raise ValueError(
                    "operational routing requires product-owned replay inputs"
                )
            input_run.validate_integrity()
            if (
                input_run.input_plan_digest != input_plan_digest
                or input_run.full_analysis_input_digest
                != decision_payload.get("full_analysis_input_digest")
                or input_run.analysis_input_derivation_artifact_digest
                != derivation_digest
            ):
                raise ValueError(
                    "operational routing run disagrees with committed provenance"
                )
            try:
                with np.load(arrays_path, allow_pickle=False) as archive:
                    committed_frames = torch.from_numpy(
                        np.array(archive["derived_input_frames"], copy=True)
                    )
                    committed_qc_mask = torch.from_numpy(
                        np.array(archive["derived_qc_valid_mask"], copy=True)
                    )
                    committed_quality = torch.from_numpy(
                        np.array(archive["derived_quality_weight"], copy=True)
                    )
                    committed_observation_std = torch.from_numpy(
                        np.array(
                            archive["derived_observation_std_dbz"],
                            copy=True,
                        )
                    )
                    committed_source_history = (
                        torch.from_numpy(
                            np.array(
                                archive[
                                    "input_history_source_radar_index_map"
                                ],
                                copy=True,
                            )
                        )
                        if "input_history_source_radar_index_map"
                        in archive.files
                        else None
                    )
                    committed_source_map = (
                        torch.from_numpy(
                            np.array(
                                archive["source_radar_index_map"], copy=True
                            )
                        )
                        if "source_radar_index_map" in archive.files
                        else None
                    )
                    committed_outage_mask = (
                        torch.from_numpy(
                            np.array(archive["outage_mask"], copy=True)
                        )
                        if "outage_mask" in archive.files
                        else None
                    )
                    committed_dynamic_qc_mask = (
                        torch.from_numpy(
                            np.array(
                                archive["dynamic_qc_valid_mask"], copy=True
                            )
                        )
                        if "dynamic_qc_valid_mask" in archive.files
                        else None
                    )
            except (KeyError, OSError, ValueError) as error:
                raise ValueError(
                    "committed routing replay arrays are invalid"
                ) from error
            if tensor_digest(committed_frames) != input_run.input_frames_digest:
                raise ValueError(
                    "committed routing frames disagree with the forecast run"
                )
            committed_source_available = (
                torch.ones_like(committed_qc_mask, dtype=torch.bool)
                if committed_source_history is None
                else committed_source_history >= 0
            )
            replayed_regime = NeuralPriorRegimeClassifier.classify(
                regime_classifier,
                committed_frames,
                input_run=input_run,
                qc_valid_mask=committed_qc_mask,
                quality_weight=committed_quality,
                observation_std_dbz=committed_observation_std,
                source_available_mask=committed_source_available,
            )
            if (
                replayed_regime.payload != regime_evidence.payload
                or replayed_regime.evidence_digest
                != regime_evidence.evidence_digest
            ):
                raise ValueError(
                    "operational regime evidence was not produced by replay"
                )
            assert range_geometry_contract is not None
            assert range_grid_x_m is not None
            assert range_grid_y_m is not None
            if type(range_geometry_contract) is RangeGeometryContract:
                replayed_range = resolve_range_geometry(
                    range_geometry_contract,
                    grid_x_m=range_grid_x_m,
                    grid_y_m=range_grid_y_m,
                )
                if any(
                    item is not None
                    for item in (
                        committed_source_map,
                        committed_outage_mask,
                        committed_dynamic_qc_mask,
                    )
                ):
                    raise ValueError(
                        "single-site routing retained mosaic provenance"
                    )
            else:
                mosaic_geometry = cast(
                    MosaicRangeGeometryContract,
                    range_geometry_contract,
                )
                if (
                    committed_source_map is None
                    or committed_outage_mask is None
                    or committed_dynamic_qc_mask is None
                    or committed_outage_mask.dtype is not torch.bool
                    or committed_dynamic_qc_mask.dtype is not torch.bool
                    or committed_outage_mask.shape
                    != committed_source_map.shape
                    or committed_dynamic_qc_mask.shape
                    != committed_source_map.shape
                ):
                    raise ValueError(
                        "mosaic routing provenance is incomplete"
                    )
                replayed_range, _ = resolve_mosaic_range_geometry(
                    mosaic_geometry,
                    grid_x_m=range_grid_x_m,
                    grid_y_m=range_grid_y_m,
                    source_radar_index_map=committed_source_map,
                )
                replayed_range = restrict_range_partition_domain(
                    replayed_range,
                    valid_range_domain_mask=(
                        (committed_source_map >= 0)
                        & ~committed_outage_mask
                        & committed_dynamic_qc_mask
                    ),
                )
            if (
                replayed_range.payload != range_partition_evidence.payload
                or replayed_range.evidence_digest
                != range_partition_evidence.evidence_digest
            ):
                raise ValueError(
                    "operational range evidence was not produced by replay"
                )
        _validate_ledgered_promotion_deployment_certificate(
            promotion_deployment_certificate,
            authority_trust_store=authority_trust,
            promotion_evidence=promotion_evidence,
        )
        policy.validate_integrity()
        _replay_operational_deployment_selection(
            decision_payload,
            promotion_deployment_certificate=(
                promotion_deployment_certificate
            ),
            promotion_evidence=promotion_evidence,
            policy=policy,
            policy_trust_store_digest=policy_trust_store_digest,
            regime_evidence=regime_evidence,
            range_partition_evidence=range_partition_evidence,
            range_geometry_contract=range_geometry_contract,
        )
        _require_current_raw_ingestor_trust_store_digest(
            raw_ingestor_trust_store_path,
            current_raw_ingestor_trust.content_digest,
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                current_raw_ingestor_trust.content_digest,
            )
            connection.row_factory = sqlite3.Row
            retained = connection.execute(
                "SELECT * FROM operational_decision_issuance_states "
                "WHERE operational_cycle_id = ?",
                (operational_cycle_id,),
            ).fetchone()
            retained_release = connection.execute(
                "SELECT approval_json FROM deployment_bundle_release_approvals "
                "WHERE approval_digest = ?",
                (deployment_bundle_release_approval.approval_digest,),
            ).fetchone()
            canonical_release_approval = json.dumps(
                deployment_bundle_release_approval.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            if retained_release is None:
                connection.execute(
                    "INSERT INTO deployment_bundle_release_approvals "
                    "(approval_digest,deployment_bundle_digest,"
                    "bundle_manifest_digest,authority_id,approval_json,"
                    "approved_at,expires_at,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        deployment_bundle_release_approval.approval_digest,
                        deployment_bundle_release_approval
                        .deployment_bundle_digest,
                        deployment_bundle_release_approval
                        .bundle_manifest_digest,
                        deployment_bundle_release_approval.authority_id,
                        canonical_release_approval,
                        deployment_bundle_release_approval.approved_at,
                        deployment_bundle_release_approval.expires_at,
                        deployment_bundle_release_approval.approved_at,
                    ),
                )
            elif retained_release[0] != canonical_release_approval:
                raise ValueError("deployment bundle release approval equivocated")
            retained_runtime = connection.execute(
                "SELECT receipt_json FROM deployment_runtime_activations "
                "WHERE receipt_digest = ?",
                (deployment_runtime_activation_receipt.receipt_digest,),
            ).fetchone()
            activation_head = connection.execute(
                "SELECT h.sequence_number,h.receipt_digest,"
                "h.host_identity_digest,a.runtime_tree_digest "
                "FROM deployment_runtime_activation_heads AS h "
                "JOIN deployment_runtime_activations AS a "
                "ON a.receipt_digest = h.receipt_digest "
                "WHERE h.deployment_instance_digest = ?",
                (
                    deployment_runtime_activation_receipt
                    .deployment_instance_digest,
                ),
            ).fetchone()
            canonical_runtime_receipt = json.dumps(
                deployment_runtime_activation_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            if retained_runtime is None:
                expected_sequence = (
                    1 if activation_head is None else int(activation_head[0]) + 1
                )
                expected_previous = (
                    DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST
                    if activation_head is None
                    else str(activation_head[1])
                )
                if (
                    deployment_runtime_activation_receipt
                    .activation_sequence_number
                    != expected_sequence
                    or deployment_runtime_activation_receipt
                    .previous_activation_receipt_digest
                    != expected_previous
                    or (
                        activation_head is not None
                        and str(activation_head[2])
                        != deployment_runtime_activation_receipt
                        .host_identity_digest
                    )
                ):
                    raise ValueError(
                        "deployment runtime activation does not extend the current head"
                    )
                reused_runtime = None
                if (
                    activation_head is not None
                    and deployment_runtime_activation_receipt.runtime_tree_digest
                    != str(activation_head[3])
                ):
                    reused_runtime = connection.execute(
                        "SELECT 1 FROM deployment_runtime_activations "
                        "WHERE deployment_instance_digest = ? AND "
                        "runtime_tree_digest = ? AND receipt_digest != ? LIMIT 1",
                        (
                            deployment_runtime_activation_receipt
                            .deployment_instance_digest,
                            deployment_runtime_activation_receipt
                            .runtime_tree_digest,
                            expected_previous,
                        ),
                    ).fetchone()
                if (
                    reused_runtime is not None
                    and deployment_runtime_activation_receipt
                    .rollback_reason_digest is None
                ):
                    raise ValueError(
                        "runtime rollback requires a signed rollback reason"
                    )
                connection.execute(
                    "INSERT INTO deployment_runtime_activations "
                    "(receipt_digest,release_approval_digest,"
                    "deployment_bundle_digest,"
                    "runtime_tree_digest,interpreter_closure_digest,"
                    "deployment_instance_digest,host_identity_digest,"
                    "activation_sequence_number,previous_activation_receipt_digest,"
                    "rollback_reason_digest,receipt_json,activated_at,"
                    "expires_at,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        deployment_runtime_activation_receipt.receipt_digest,
                        deployment_runtime_activation_receipt
                        .release_approval_digest,
                        deployment_runtime_activation_receipt
                        .deployment_bundle_digest,
                        deployment_runtime_activation_receipt.runtime_tree_digest,
                        deployment_runtime_activation_receipt
                        .interpreter_closure_digest,
                        deployment_runtime_activation_receipt
                        .deployment_instance_digest,
                        deployment_runtime_activation_receipt.host_identity_digest,
                        deployment_runtime_activation_receipt
                        .activation_sequence_number,
                        deployment_runtime_activation_receipt
                        .previous_activation_receipt_digest,
                        deployment_runtime_activation_receipt
                        .rollback_reason_digest,
                        canonical_runtime_receipt,
                        deployment_runtime_activation_receipt.activated_at,
                        deployment_runtime_activation_receipt.expires_at,
                        deployment_runtime_activation_receipt.activated_at,
                    ),
                )
                if activation_head is None:
                    connection.execute(
                        "INSERT INTO deployment_runtime_activation_heads "
                        "(deployment_instance_digest,host_identity_digest,"
                        "sequence_number,receipt_digest,updated_at) "
                        "VALUES (?,?,?,?,?)",
                        (
                            deployment_runtime_activation_receipt
                            .deployment_instance_digest,
                            deployment_runtime_activation_receipt
                            .host_identity_digest,
                            deployment_runtime_activation_receipt
                            .activation_sequence_number,
                            deployment_runtime_activation_receipt.receipt_digest,
                            deployment_runtime_activation_receipt.activated_at,
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE deployment_runtime_activation_heads SET "
                        "sequence_number = ?,receipt_digest = ?,updated_at = ? "
                        "WHERE deployment_instance_digest = ? AND "
                        "sequence_number = ? AND receipt_digest = ?",
                        (
                            deployment_runtime_activation_receipt
                            .activation_sequence_number,
                            deployment_runtime_activation_receipt.receipt_digest,
                            deployment_runtime_activation_receipt.activated_at,
                            deployment_runtime_activation_receipt
                            .deployment_instance_digest,
                            int(activation_head[0]),
                            str(activation_head[1]),
                        ),
                    )
                    if connection.execute("SELECT changes()").fetchone()[0] != 1:
                        raise ValueError(
                            "deployment runtime activation head changed concurrently"
                        )
            elif retained_runtime[0] != canonical_runtime_receipt:
                raise ValueError("deployment runtime activation equivocated")
            elif (
                activation_head is None
                or str(activation_head[1])
                != deployment_runtime_activation_receipt.receipt_digest
                or int(activation_head[0])
                != deployment_runtime_activation_receipt
                .activation_sequence_number
            ):
                raise ValueError(
                    "only the current runtime activation head may authorize decisions"
                )
            if retained is None:
                prepared_at = datetime.now(timezone.utc)
                if prepared_at > deadline:
                    raise ValueError(
                        "operational decision missed its preparation deadline"
                    )
                connection.execute(
                    "INSERT INTO operational_decision_issuance_states "
                    "(operational_cycle_id,input_plan_digest,"
                    "promotion_certificate_digest,release_approval_digest,"
                    "runtime_activation_receipt_digest,"
                    "decision_payload_json,status,prepared_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        operational_cycle_id,
                        input_plan_digest,
                        promotion_deployment_certificate.certificate_digest,
                        deployment_bundle_release_approval.approval_digest,
                        deployment_runtime_activation_receipt.receipt_digest,
                        canonical_decision,
                        "prepared",
                        prepared_at.isoformat(),
                        prepared_at.isoformat(),
                    ),
                )
            else:
                if (
                    retained["input_plan_digest"] != input_plan_digest
                    or retained["promotion_certificate_digest"]
                    != promotion_deployment_certificate.certificate_digest
                    or retained["release_approval_digest"]
                    != deployment_bundle_release_approval.approval_digest
                    or retained["runtime_activation_receipt_digest"]
                    != deployment_runtime_activation_receipt.receipt_digest
                    or retained["decision_payload_json"] != canonical_decision
                ):
                    raise ValueError("operational cycle id was reused")
                if retained["status"] == "expired":
                    raise ValueError("operational decision issuance expired")
                if retained["status"] == "published":
                    stored = connection.execute(
                        "SELECT payload_json FROM "
                        "operational_deployment_decisions_v2 "
                        "WHERE certificate_digest = ?",
                        (retained["certificate_digest"],),
                    ).fetchone()
                    if stored is None:
                        raise ValueError(
                            "published operational decision is incomplete"
                        )
                    payload = json.loads(str(stored[0]))
                    if not isinstance(payload, dict):
                        raise ValueError(
                            "published operational decision is invalid"
                        )
                    published_certificate = (
                        _operational_deployment_decision_certificate_from_payload(
                            payload
                        )
                    )

        if published_certificate is not None:
            retained_activation = self._ensure_operational_decision_activation_receipt(
                published_certificate,
                ledger_signer=ledger_signer,
                authority_trust=authority_trust,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                raw_ingestor_trust_store_digest=(
                    current_raw_ingestor_trust.content_digest
                ),
            )
            self._ensure_operational_decision_commit_authorization_receipt(
                published_certificate,
                retained_activation,
                operational_cycle_id=operational_cycle_id,
                ledger_signer=ledger_signer,
                authority_trust=authority_trust,
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
                raw_ingestor_trust_store_digest=(
                    current_raw_ingestor_trust.content_digest
                ),
            )
            return published_certificate

        certificate: OperationalDeploymentDecisionCertificate
        receipt_payload: dict[str, object]
        retained_publication_receipt_json: str | None = None
        retained_publication_receipt_digest: str | None = None
        publication_payload_committed_at_text: str | None = None
        retained_activation_receipt_json: str | None = None
        retained_activation_receipt_digest: str | None = None
        retained_activation_authorized_at: str | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                current_raw_ingestor_trust.content_digest,
            )
            connection.row_factory = sqlite3.Row
            state = connection.execute(
                "SELECT * FROM operational_decision_issuance_states "
                "WHERE operational_cycle_id = ?",
                (operational_cycle_id,),
            ).fetchone()
            if state is None:
                raise ValueError("operational decision preparation disappeared")
            if state["status"] == "decision_recorded":
                stored = connection.execute(
                    "SELECT d.payload_json,p.receipt_json FROM "
                    "operational_deployment_decisions_v2 AS d JOIN "
                    "operational_decision_commit_proofs AS p "
                    "ON p.receipt_digest = d.receipt_digest "
                    "WHERE d.certificate_digest = ?",
                    (state["certificate_digest"],),
                ).fetchone()
                if stored is None:
                    raise ValueError(
                        "recorded operational decision is incomplete"
                    )
                certificate_payload = json.loads(str(stored[0]))
                receipt_payload = json.loads(str(stored[1]))
                if not isinstance(certificate_payload, dict) or not isinstance(
                    receipt_payload, dict
                ):
                    raise ValueError("recorded operational decision is invalid")
                certificate = (
                    _operational_deployment_decision_certificate_from_payload(
                        certificate_payload
                    )
                )
            elif state["status"] == "prepared":
                promotion_row = connection.execute(
                    "SELECT certificate_digest,ledger_instance_digest FROM "
                    "neural_prior_promotion_deployment_certificates_v3 "
                    "WHERE certificate_digest = ?",
                    (promotion_deployment_certificate.certificate_digest,),
                ).fetchone()
                head = connection.execute(
                    "SELECT ledger_instance_digest,sequence_number,"
                    "certificate_digest FROM operational_decision_chain_head "
                    "WHERE singleton = 1"
                ).fetchone()
                if (
                    promotion_row is None
                    or head is None
                    or promotion_row["ledger_instance_digest"]
                    != promotion_deployment_certificate.ledger_instance_digest
                    or head["ledger_instance_digest"]
                    != promotion_deployment_certificate.ledger_instance_digest
                ):
                    raise ValueError(
                        "operational decision requires its ledgered promotion certificate"
                    )
                accepted_at = datetime.now(timezone.utc)
                if accepted_at > deadline:
                    raise ValueError(
                        "operational decision missed its acceptance deadline"
                    )
                accepted_at_text = accepted_at.isoformat().replace(
                    "+00:00", "Z"
                )
                sequence_number = int(head["sequence_number"]) + 1
                ledger_instance_digest = str(head["ledger_instance_digest"])
                previous = str(head["certificate_digest"])
                commit_entry_digest, chain_root = (
                    _operational_decision_commit_digests(
                        decision_payload,
                        ledger_instance_digest=ledger_instance_digest,
                        sequence_number=sequence_number,
                        previous_operational_decision_digest=previous,
                        accepted_at=accepted_at_text,
                    )
                )
                proof_committed_at = datetime.now(timezone.utc)
                if proof_committed_at > deadline:
                    raise ValueError(
                        "operational decision proof missed its deadline"
                    )
                receipt = _issue_operational_decision_ledger_receipt(
                    decision_payload,
                    ledger_instance_digest=ledger_instance_digest,
                    sequence_number=sequence_number,
                    previous_operational_decision_digest=previous,
                    accepted_at=accepted_at_text,
                    committed_at=proof_committed_at.isoformat(),
                    commit_entry_digest=commit_entry_digest,
                    committed_chain_root_digest=chain_root,
                    signer=ledger_signer,
                    authority_trust_store=authority_trust,
                )
                certificate = _issue_operational_deployment_decision_certificate(
                    decision_payload,
                    deployment_bundle_release_approval=(
                        deployment_bundle_release_approval
                    ),
                    deployment_runtime_activation_receipt=(
                        deployment_runtime_activation_receipt
                    ),
                    promotion_deployment_certificate=(
                        promotion_deployment_certificate
                    ),
                    promotion_evidence=promotion_evidence,
                    policy=policy,
                    policy_trust_store_digest=policy_trust_store_digest,
                    regime_evidence=regime_evidence,
                    range_partition_evidence=range_partition_evidence,
                    range_geometry_contract=range_geometry_contract,
                    ledger_receipt=receipt,
                    signer=operational_signer,
                    authority_trust_store=authority_trust,
                )
                _require_current_raw_ingestor_trust_store_digest(
                    raw_ingestor_trust_store_path,
                    current_raw_ingestor_trust.content_digest,
                )
                if datetime.now(timezone.utc) > deadline:
                    raise ValueError(
                        "operational decision signature missed its deadline"
                    )
                receipt_payload = receipt.payload
                certificate_payload = certificate.payload
                connection.execute(
                    "INSERT INTO operational_decision_commits "
                    "(commit_entry_digest,committed_chain_root_digest,"
                    "ledger_instance_digest,sequence_number,"
                    "previous_commit_root_digest,promotion_certificate_digest,"
                    "input_plan_digest,decision_payload_json,accepted_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        commit_entry_digest,
                        chain_root,
                        ledger_instance_digest,
                        sequence_number,
                        previous,
                        promotion_deployment_certificate.certificate_digest,
                        input_plan_digest,
                        canonical_decision,
                        receipt.accepted_at,
                    ),
                )
                connection.execute(
                    "INSERT INTO operational_decision_commit_proofs "
                    "(receipt_digest,commit_entry_digest,receipt_json,"
                    "committed_at,created_at) VALUES (?,?,?,?,?)",
                    (
                        receipt.receipt_digest,
                        commit_entry_digest,
                        json.dumps(
                            receipt_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        receipt.committed_at,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.execute(
                    "INSERT INTO operational_deployment_decisions_v2 "
                    "(certificate_digest,ledger_instance_digest,sequence_number,"
                    "previous_certificate_digest,promotion_certificate_digest,"
                    "release_approval_digest,runtime_activation_receipt_digest,"
                    "input_plan_digest,"
                    "commit_entry_digest,receipt_digest,payload_json,recorded_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        certificate.certificate_digest,
                        certificate.ledger_instance_digest,
                        certificate.ledger_sequence_number,
                        certificate.previous_operational_decision_digest,
                        certificate.promotion_deployment_certificate_digest,
                        certificate.deployment_bundle_release_approval_digest,
                        certificate.deployment_runtime_activation_receipt_digest,
                        certificate.input_plan_digest,
                        commit_entry_digest,
                        receipt.receipt_digest,
                        json.dumps(
                            certificate_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        certificate.issued_at,
                    ),
                )
                connection.execute(
                    "UPDATE operational_decision_issuance_states SET "
                    "status = 'decision_recorded',certificate_digest = ?,"
                    "updated_at = ? WHERE operational_cycle_id = ?",
                    (
                        certificate.certificate_digest,
                        datetime.now(timezone.utc).isoformat(),
                        operational_cycle_id,
                    ),
                )
                updated = connection.execute(
                    "UPDATE operational_decision_chain_head SET "
                    "sequence_number = ?,certificate_digest = ?,updated_at = ? "
                    "WHERE singleton = 1 AND sequence_number = ? AND "
                    "certificate_digest = ?",
                    (
                        certificate.ledger_sequence_number,
                        str(receipt_payload["committed_chain_root_digest"]),
                        datetime.now(timezone.utc).isoformat(),
                        int(head["sequence_number"]),
                        head["certificate_digest"],
                    ),
                )
                if updated.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "operational decision chain head changed"
                    )
            else:
                raise ValueError("operational decision state is invalid")

        retained_committed_at = state["decision_row_committed_at"]
        if retained_committed_at is None:
            decision_row_committed_at = datetime.now(timezone.utc)
            if decision_row_committed_at > deadline:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                    expired_at=decision_row_committed_at,
                )
                raise ValueError(
                    "operational decision recording committed after its deadline"
                )
            decision_row_committed_at_text = (
                decision_row_committed_at.isoformat().replace("+00:00", "Z")
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    "UPDATE operational_decision_issuance_states SET "
                    "decision_row_committed_at = ?,updated_at = ? "
                    "WHERE operational_cycle_id = ? AND "
                    "status = 'decision_recorded' AND "
                    "decision_row_committed_at IS NULL",
                    (
                        decision_row_committed_at_text,
                        decision_row_committed_at_text,
                        operational_cycle_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "operational decision commit timestamp changed"
                    )
        else:
            try:
                decision_row_committed_at = datetime.fromisoformat(
                    str(retained_committed_at).replace("Z", "+00:00")
                )
            except ValueError as error:
                raise ValueError(
                    "recorded operational decision timestamp is invalid"
                ) from error
            decision_row_committed_at_text = (
                decision_row_committed_at.isoformat().replace("+00:00", "Z")
            )
            if decision_row_committed_at > deadline:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                    expired_at=decision_row_committed_at,
                )
                raise ValueError(
                    "operational decision recording committed after its deadline"
                )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.row_factory = sqlite3.Row
            state = connection.execute(
                "SELECT status,certificate_digest FROM "
                "operational_decision_issuance_states "
                "WHERE operational_cycle_id = ?",
                (operational_cycle_id,),
            ).fetchone()
            if (
                state is None
                or state["status"] != "decision_recorded"
                or state["certificate_digest"] != certificate.certificate_digest
            ):
                raise ValueError("operational decision publication state changed")
            staged_publication = connection.execute(
                "SELECT u.decision_row_committed_at,"
                "u.publication_payload_committed_at,u.activation_committed_at,"
                "u.usable,u.receipt_digest,u.receipt_json,"
                "a.receipt_digest AS activation_receipt_digest,"
                "a.receipt_json AS activation_receipt_json FROM "
                "operational_decision_publications AS u LEFT JOIN "
                "operational_decision_activation_receipts AS a "
                "ON a.certificate_digest = u.certificate_digest "
                "WHERE u.certificate_digest = ?",
                (certificate.certificate_digest,),
            ).fetchone()
            now = datetime.now(timezone.utc)
            retained_payload_commit = (
                None
                if staged_publication is None
                else staged_publication["publication_payload_committed_at"]
            )
            if now >= publication_time:
                raise ValueError(
                    "operational decision missed its publication time"
                )
            if now > deadline and retained_payload_commit is None:
                raise ValueError(
                    "operational decision publication missed its deadline"
                )
            if staged_publication is None:
                connection.execute(
                    "INSERT INTO operational_decision_publications "
                    "(certificate_digest,decision_row_committed_at,"
                    "publication_payload_committed_at,activation_committed_at,"
                    "usable,receipt_digest,receipt_json,created_at) "
                    "VALUES (?,?,NULL,NULL,0,NULL,NULL,?)",
                    (
                        certificate.certificate_digest,
                        decision_row_committed_at_text,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            elif (
                str(staged_publication["decision_row_committed_at"])
                != decision_row_committed_at_text
                or int(staged_publication["usable"]) != 0
                or (
                    staged_publication["receipt_digest"] is None
                    and staged_publication["receipt_json"] is not None
                )
                or (
                    staged_publication["receipt_digest"] is not None
                    and staged_publication["receipt_json"] is None
                )
                or (
                    staged_publication["publication_payload_committed_at"]
                    is not None
                    and staged_publication["receipt_digest"] is None
                )
                or (
                    staged_publication["activation_committed_at"] is None
                    and (
                        staged_publication["activation_receipt_digest"]
                        is not None
                        or staged_publication["activation_receipt_json"]
                        is not None
                    )
                )
                or (
                    staged_publication["activation_committed_at"] is not None
                    and (
                        staged_publication["activation_receipt_digest"] is None
                        or staged_publication["activation_receipt_json"] is None
                        or staged_publication[
                            "publication_payload_committed_at"
                        ]
                        is None
                    )
                )
            ):
                raise ValueError(
                    "retained operational publication state changed"
                )
            if staged_publication is not None:
                retained_publication_receipt_json = cast(
                    str | None,
                    staged_publication["receipt_json"],
                )
                retained_publication_receipt_digest = cast(
                    str | None,
                    staged_publication["receipt_digest"],
                )
                publication_payload_committed_at_text = cast(
                    str | None,
                    staged_publication["publication_payload_committed_at"],
                )
                retained_activation_receipt_json = cast(
                    str | None,
                    staged_publication["activation_receipt_json"],
                )
                retained_activation_receipt_digest = cast(
                    str | None,
                    staged_publication["activation_receipt_digest"],
                )
                retained_activation_authorized_at = cast(
                    str | None,
                    staged_publication["activation_committed_at"],
                )

        if retained_publication_receipt_json is None:
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                current_raw_ingestor_trust.content_digest,
            )
            publication_receipt = (
                _issue_operational_decision_publication_receipt(
                    certificate,
                    decision_row_committed_at=decision_row_committed_at_text,
                    signer=ledger_signer,
                    authority_trust_store=authority_trust,
                )
            )
            post_signature = datetime.now(timezone.utc)
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                current_raw_ingestor_trust.content_digest,
            )
            if post_signature > deadline:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                    expired_at=post_signature,
                )
                raise ValueError(
                    "operational decision publication missed its deadline"
                )
            canonical_publication_receipt = json.dumps(
                publication_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _require_current_raw_ingestor_trust_store_digest(
                    raw_ingestor_trust_store_path,
                    current_raw_ingestor_trust.content_digest,
                )
                attached = connection.execute(
                    "UPDATE operational_decision_publications SET "
                    "receipt_digest = ?,receipt_json = ? "
                    "WHERE certificate_digest = ? AND usable = 0 AND "
                    "publication_payload_committed_at IS NULL AND "
                    "activation_committed_at IS NULL AND receipt_digest IS NULL "
                    "AND receipt_json IS NULL",
                    (
                        publication_receipt.receipt_digest,
                        canonical_publication_receipt,
                        certificate.certificate_digest,
                    ),
                )
                if attached.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "operational publication receipt attachment changed"
                    )
        else:
            try:
                retained_payload = json.loads(retained_publication_receipt_json)
                if not isinstance(retained_payload, dict):
                    raise ValueError
                publication_receipt = (
                    _operational_decision_publication_receipt_from_payload(
                        retained_payload
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    "retained operational publication receipt is invalid"
                ) from error
            _validate_operational_decision_publication_receipt(
                publication_receipt,
                certificate=certificate,
                authority_trust_store=authority_trust,
            )
            if (
                publication_receipt.receipt_digest
                != retained_publication_receipt_digest
            ):
                raise ValueError(
                    "retained operational publication receipt changed"
                )

        if publication_payload_committed_at_text is None:
            payload_committed_at = datetime.now(timezone.utc)
            publication_payload_committed_at_text = (
                payload_committed_at.isoformat().replace("+00:00", "Z")
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _require_current_raw_ingestor_trust_store_digest(
                    raw_ingestor_trust_store_path,
                    current_raw_ingestor_trust.content_digest,
                )
                marked = connection.execute(
                    "UPDATE operational_decision_publications SET "
                    "publication_payload_committed_at = ? "
                    "WHERE certificate_digest = ? AND usable = 0 AND "
                    "publication_payload_committed_at IS NULL AND "
                    "activation_committed_at IS NULL AND receipt_digest = ?",
                    (
                        publication_payload_committed_at_text,
                        certificate.certificate_digest,
                        publication_receipt.receipt_digest,
                    ),
                )
                if marked.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "operational publication payload marker changed"
                    )
            if payload_committed_at > deadline:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                    expired_at=payload_committed_at,
                )
                raise ValueError(
                    "operational decision publication committed after its deadline"
                )
        else:
            payload_committed_at = datetime.fromisoformat(
                publication_payload_committed_at_text.replace("Z", "+00:00")
            )
            if payload_committed_at > deadline:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                    expired_at=payload_committed_at,
                )
                raise ValueError(
                    "operational decision publication committed after its deadline"
                )

        # Hold the SQLite writer lock before fixing the signed authorization
        # instant. Commit the authorization receipt while the publication is
        # still unusable, observe that durable staging commit, and only then
        # expose published/usable together with the observation receipt.
        publication_guard_interval_seconds = 0.05
        publication_guard = timedelta(
            seconds=publication_guard_interval_seconds
        )
        committed_chain_root_digest = str(
            receipt_payload["committed_chain_root_digest"]
        )
        _require_current_raw_ingestor_trust_store_digest(
            raw_ingestor_trust_store_path,
            current_raw_ingestor_trust.content_digest,
        )
        if retained_activation_receipt_json is None:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    activation_authorized = datetime.now(timezone.utc)
                    if activation_authorized + publication_guard >= publication_time:
                        raise ValueError(
                            "operational publication guard interval was missed"
                        )
                    activation_authorized_at = (
                        activation_authorized.isoformat().replace("+00:00", "Z")
                    )
                    _require_current_raw_ingestor_trust_store_digest(
                        raw_ingestor_trust_store_path,
                        current_raw_ingestor_trust.content_digest,
                    )
                    activation_receipt = (
                        _issue_operational_decision_activation_receipt(
                            certificate,
                            publication_receipt,
                            publication_payload_committed_at=(
                                publication_payload_committed_at_text
                            ),
                            activation_authorized_at=activation_authorized_at,
                            publication_guard_interval_seconds=(
                                publication_guard_interval_seconds
                            ),
                            committed_chain_root_digest=(
                                committed_chain_root_digest
                            ),
                            signer=ledger_signer,
                            authority_trust_store=authority_trust,
                        )
                    )
                    if (
                        datetime.now(timezone.utc) + publication_guard
                        >= publication_time
                    ):
                        raise ValueError(
                            "operational activation signing crossed its guard"
                        )
                    _require_current_raw_ingestor_trust_store_digest(
                        raw_ingestor_trust_store_path,
                        current_raw_ingestor_trust.content_digest,
                    )
                    canonical_activation_receipt = json.dumps(
                        activation_receipt.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    inserted = connection.execute(
                        "INSERT INTO operational_decision_activation_receipts "
                        "(certificate_digest,receipt_digest,receipt_json,created_at) "
                        "SELECT ?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM "
                        "operational_decision_activation_receipts WHERE "
                        "certificate_digest = ?)",
                        (
                            certificate.certificate_digest,
                            activation_receipt.receipt_digest,
                            canonical_activation_receipt,
                            activation_authorized_at,
                            certificate.certificate_digest,
                        ),
                    )
                    staged = connection.execute(
                        "UPDATE operational_decision_publications SET "
                        "activation_committed_at = ? "
                        "WHERE certificate_digest = ? AND usable = 0 AND "
                        "publication_payload_committed_at = ? AND "
                        "activation_committed_at IS NULL AND receipt_digest = ?",
                        (
                            activation_authorized_at,
                            certificate.certificate_digest,
                            publication_payload_committed_at_text,
                            publication_receipt.receipt_digest,
                        ),
                    )
                    if inserted.rowcount != 1 or staged.rowcount != 1:
                        raise sqlite3.IntegrityError(
                            "operational activation authorization changed"
                        )
            except ValueError:
                self._expire_operational_decision_issuance(
                    operational_cycle_id,
                    certificate.certificate_digest,
                )
                raise
        else:
            try:
                retained_activation_payload = json.loads(
                    retained_activation_receipt_json
                )
                if not isinstance(retained_activation_payload, dict):
                    raise ValueError
                activation_receipt = (
                    _operational_decision_activation_receipt_from_payload(
                        retained_activation_payload
                    )
                )
                _validate_operational_decision_activation_receipt(
                    activation_receipt,
                    certificate=certificate,
                    publication_receipt=publication_receipt,
                    authority_trust_store=authority_trust,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    "retained operational activation receipt is invalid"
                ) from error
            if (
                activation_receipt.receipt_digest
                != retained_activation_receipt_digest
                or activation_receipt.activation_authorized_at
                != retained_activation_authorized_at
            ):
                raise ValueError("retained operational activation changed")

        self._ensure_operational_decision_commit_authorization_receipt(
            certificate,
            activation_receipt,
            operational_cycle_id=operational_cycle_id,
            ledger_signer=ledger_signer,
            authority_trust=authority_trust,
            raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
            raw_ingestor_trust_store_digest=(
                current_raw_ingestor_trust.content_digest
            ),
        )
        try:
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                current_raw_ingestor_trust.content_digest,
            )
        except ValueError:
            self._expire_operational_decision_issuance(
                operational_cycle_id,
                certificate.certificate_digest,
            )
            raise
        retained_activation_receipt = (
            self._ensure_operational_decision_activation_receipt(
                certificate,
                ledger_signer=ledger_signer,
                authority_trust=authority_trust,
                raw_ingestor_trust_store_path=(
                    raw_ingestor_trust_store_path
                ),
                raw_ingestor_trust_store_digest=(
                    current_raw_ingestor_trust.content_digest
                ),
            )
        )
        if (
            retained_activation_receipt.receipt_digest
            != activation_receipt.receipt_digest
        ):
            raise ValueError("operational activation receipt changed after commit")
        try:
            if (
                _load_training_target_source_trust_store(
                    training_target_source_trust_store_path
                ).content_digest
                != current_target_source_trust.content_digest
            ):
                raise ValueError(
                    "operational target-source trust changed during issuance"
                )
        except ValueError:
            self._expire_operational_decision_issuance(
                operational_cycle_id,
                certificate.certificate_digest,
            )
            raise
        return certificate

    def _ensure_operational_decision_commit_authorization_receipt(
        self,
        certificate: OperationalDeploymentDecisionCertificate,
        activation_receipt: OperationalDecisionActivationReceipt,
        *,
        operational_cycle_id: str,
        ledger_signer: DeploymentAuthoritySigner,
        authority_trust: _PromotionDeploymentAuthorityTrustStore,
        raw_ingestor_trust_store_path: str | Path,
        raw_ingestor_trust_store_digest: str,
    ) -> OperationalDecisionCommitAuthorizationReceipt:
        """Authorize the terminal transition under a bounded writer lock."""

        with self._connect() as connection:
            retained = connection.execute(
                "SELECT o.receipt_digest,o.receipt_json,u.usable,s.status "
                "FROM operational_decision_commit_authorization_receipts AS o "
                "JOIN operational_decision_publications AS u "
                "ON u.certificate_digest = o.certificate_digest "
                "JOIN operational_decision_issuance_states AS s "
                "ON s.certificate_digest = o.certificate_digest "
                "WHERE o.certificate_digest = ?",
                (certificate.certificate_digest,),
            ).fetchone()
        if retained is not None:
            try:
                payload = json.loads(str(retained[1]))
                if not isinstance(payload, dict):
                    raise ValueError
                receipt = (
                    _operational_decision_commit_authorization_receipt_from_payload(
                        payload
                    )
                )
                _validate_operational_decision_commit_authorization_receipt(
                    receipt,
                    certificate=certificate,
                    activation_receipt=activation_receipt,
                    authority_trust_store=authority_trust,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    "operational commit authorization is invalid"
                ) from error
            if (
                receipt.receipt_digest != retained[0]
                or int(retained[2]) != 1
                or retained[3] != "published"
            ):
                raise ValueError("operational commit authorization changed")
            return receipt

        _require_current_raw_ingestor_trust_store_digest(
            raw_ingestor_trust_store_path,
            raw_ingestor_trust_store_digest,
        )
        publication_time = _canonical_utc_datetime(
            certificate.publication_time,
            "publication_time",
        )
        publication_guard = timedelta(
            seconds=activation_receipt.publication_guard_interval_seconds
        )
        with self._connect() as connection:
            lock_budget = (
                publication_time
                - publication_guard
                - datetime.now(timezone.utc)
            ).total_seconds()
            if lock_budget <= 0.0:
                raise ValueError(
                    "operational terminal commit missed its publication guard"
                )
            connection.execute(
                "PRAGMA busy_timeout = "
                f"{max(1, min(1000, int(lock_budget * 1000)))}"
            )
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                raise ValueError(
                    "operational terminal writer lock is unavailable"
                ) from error
            commit_authorized = datetime.now(timezone.utc)
            if commit_authorized + publication_guard >= publication_time:
                raise ValueError(
                    "operational terminal commit missed its publication guard"
                )
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                raw_ingestor_trust_store_digest,
            )
            receipt = _issue_operational_decision_commit_authorization_receipt(
                certificate,
                activation_receipt,
                terminal_commit_authorized_at=(
                    commit_authorized.isoformat().replace("+00:00", "Z")
                ),
                signer=ledger_signer,
                authority_trust_store=authority_trust,
            )
            if datetime.now(timezone.utc) + publication_guard >= publication_time:
                raise ValueError(
                    "operational terminal authorization signing crossed its guard"
                )
            inserted = connection.execute(
                "INSERT INTO operational_decision_commit_authorization_receipts "
                "(certificate_digest,receipt_digest,receipt_json,"
                "terminal_commit_authorized_at,created_at) VALUES (?,?,?,?,?)",
                (
                    certificate.certificate_digest,
                    receipt.receipt_digest,
                    json.dumps(
                        receipt.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    receipt.terminal_commit_authorized_at,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            finalized = connection.execute(
                "UPDATE operational_decision_publications SET usable = 1 "
                "WHERE certificate_digest = ? AND usable = 0 AND "
                "activation_committed_at = ? AND "
                "publication_payload_committed_at IS NOT NULL AND "
                "receipt_digest IS NOT NULL",
                (
                    certificate.certificate_digest,
                    activation_receipt.activation_authorized_at,
                ),
            )
            published = connection.execute(
                "UPDATE operational_decision_issuance_states SET "
                "status = 'published',updated_at = ? "
                "WHERE operational_cycle_id = ? AND "
                "certificate_digest = ? AND status = 'decision_recorded'",
                (
                    activation_receipt.activation_authorized_at,
                    operational_cycle_id,
                    certificate.certificate_digest,
                ),
            )
            if (
                inserted.rowcount != 1
                or finalized.rowcount != 1
                or published.rowcount != 1
            ):
                raise sqlite3.IntegrityError(
                    "operational publication finalization changed"
                )
            connection.commit()
        if datetime.now(timezone.utc) >= publication_time:
            self._expire_operational_decision_issuance(
                operational_cycle_id,
                certificate.certificate_digest,
            )
            raise ValueError(
                "operational terminal commit completed after publication"
            )
        return receipt

    def _ensure_operational_decision_activation_receipt(
        self,
        certificate: OperationalDeploymentDecisionCertificate,
        *,
        ledger_signer: DeploymentAuthoritySigner,
        authority_trust: _PromotionDeploymentAuthorityTrustStore,
        raw_ingestor_trust_store_path: str | Path,
        raw_ingestor_trust_store_digest: str,
    ) -> OperationalDecisionActivationReceipt:
        """Issue or load the immutable proof of the terminal publication state."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT u.receipt_json AS publication_receipt_json,"
                "u.publication_payload_committed_at,u.activation_committed_at,"
                "u.usable,s.status,c.committed_chain_root_digest,"
                "a.receipt_json AS activation_receipt_json,"
                "a.receipt_digest AS activation_receipt_digest "
                "FROM operational_decision_publications AS u "
                "JOIN operational_decision_issuance_states AS s "
                "ON s.certificate_digest = u.certificate_digest "
                "JOIN operational_deployment_decisions_v2 AS d "
                "ON d.certificate_digest = u.certificate_digest "
                "JOIN operational_decision_commits AS c "
                "ON c.commit_entry_digest = d.commit_entry_digest "
                "LEFT JOIN operational_decision_activation_receipts AS a "
                "ON a.certificate_digest = u.certificate_digest "
                "WHERE u.certificate_digest = ?",
                (certificate.certificate_digest,),
            ).fetchone()
        if (
            row is None
            or int(row["usable"]) != 1
            or row["status"] != "published"
            or row["publication_payload_committed_at"] is None
            or row["activation_committed_at"] is None
        ):
            raise ValueError("operational decision has not reached final activation")
        try:
            publication_payload = json.loads(str(row["publication_receipt_json"]))
            if not isinstance(publication_payload, dict):
                raise ValueError
            publication_receipt = (
                _operational_decision_publication_receipt_from_payload(
                    publication_payload
                )
            )
            _validate_operational_decision_publication_receipt(
                publication_receipt,
                certificate=certificate,
                authority_trust_store=authority_trust,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("operational publication receipt is invalid") from error
        retained_json = cast(str | None, row["activation_receipt_json"])
        if retained_json is not None:
            try:
                retained_payload = json.loads(retained_json)
                if not isinstance(retained_payload, dict):
                    raise ValueError
                receipt = _operational_decision_activation_receipt_from_payload(
                    retained_payload
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError("operational activation receipt is invalid") from error
            _validate_operational_decision_activation_receipt(
                receipt,
                certificate=certificate,
                publication_receipt=publication_receipt,
                authority_trust_store=authority_trust,
            )
            if receipt.receipt_digest != row["activation_receipt_digest"]:
                raise ValueError("operational activation receipt changed")
            return receipt
        raise ValueError("operational activation receipt is missing")

    def _expire_operational_decision_issuance(
        self,
        operational_cycle_id: str,
        certificate_digest: str,
        *,
        expired_at: datetime | None = None,
    ) -> None:
        """Retain an immutable decision with a fail-closed expiry marker."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT certificate_digest FROM "
                "operational_deployment_decisions_v2 "
                "WHERE certificate_digest = ?",
                (certificate_digest,),
            ).fetchone()
            now = (
                datetime.now(timezone.utc) if expired_at is None else expired_at
            ).isoformat()
            if row is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO operational_decision_publications "
                    "(certificate_digest,decision_row_committed_at,"
                    "activation_committed_at,usable,receipt_digest,receipt_json,"
                    "created_at) VALUES (?,?,NULL,0,NULL,NULL,?)",
                    (certificate_digest, now, now),
                )
                connection.execute(
                    "UPDATE operational_decision_publications SET usable = 0 "
                    "WHERE certificate_digest = ? AND usable = 1",
                    (certificate_digest,),
                )
            connection.execute(
                "UPDATE operational_decision_issuance_states SET "
                "status = 'expired',updated_at = ? "
                "WHERE operational_cycle_id = ?",
                (now, operational_cycle_id),
            )

    def _issue_operational_deployment_decision_legacy(
        self,
        decision_payload: dict[str, object],
        *,
        deployment_bundle_release_approval: DeploymentBundleReleaseApproval,
        deployment_runtime_activation_receipt: (
            DeploymentRuntimeActivationReceipt
        ),
        promotion_deployment_certificate: LedgeredPromotionDeploymentCertificate,
        promotion_evidence: NeuralPriorPromotionEvidence,
        policy: DeployedNeuralPriorPolicy,
        policy_trust_store_digest: str,
        ledger_signer: DeploymentAuthoritySigner,
        operational_signer: DeploymentAuthoritySigner,
        authority_trust_store_path: str | Path,
    ) -> OperationalDeploymentDecisionCertificate:
        """Commit a decision core, then countersign its durable ledger proof."""

        authority_trust = _load_promotion_deployment_authority_trust_store(
            authority_trust_store_path
        )
        if (
            ledger_signer.authority_id == operational_signer.authority_id
            or ledger_signer.public_key_hex == operational_signer.public_key_hex
            or ledger_signer.authority_id
            == promotion_deployment_certificate.authority_id
            or ledger_signer.public_key_hex
            == promotion_deployment_certificate.authority_public_key_hex
        ):
            raise ValueError(
                "promotion, ledger, and operational roles require separate keys"
            )
        accepted_at: str
        commit_entry_digest: str
        committed_chain_root_digest: str
        sequence_number: int
        ledger_instance_digest: str
        previous_operational_decision_digest: str
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.row_factory = sqlite3.Row
            promotion_row = connection.execute(
                "SELECT certificate_digest,ledger_instance_digest FROM "
                "neural_prior_promotion_deployment_certificates_v3 "
                "WHERE certificate_digest = ?",
                (promotion_deployment_certificate.certificate_digest,),
            ).fetchone()
            head = connection.execute(
                "SELECT ledger_instance_digest,sequence_number,"
                "certificate_digest FROM operational_decision_chain_head "
                "WHERE singleton = 1"
            ).fetchone()
            if (
                promotion_row is None
                or head is None
                or promotion_row["ledger_instance_digest"]
                != promotion_deployment_certificate.ledger_instance_digest
                or head["ledger_instance_digest"]
                != promotion_deployment_certificate.ledger_instance_digest
            ):
                raise ValueError(
                    "operational decision requires its ledgered promotion certificate"
                )
            accepted_at = datetime.now(timezone.utc).isoformat()
            sequence_number = int(head["sequence_number"]) + 1
            ledger_instance_digest = str(head["ledger_instance_digest"])
            previous_operational_decision_digest = str(
                head["certificate_digest"]
            )
            commit_entry_digest, committed_chain_root_digest = (
                _operational_decision_commit_digests(
                    decision_payload,
                    ledger_instance_digest=ledger_instance_digest,
                    sequence_number=sequence_number,
                    previous_operational_decision_digest=(
                        previous_operational_decision_digest
                    ),
                    accepted_at=accepted_at,
                )
            )
            deadline = datetime.fromisoformat(
                str(decision_payload["decision_deadline"]).replace("Z", "+00:00")
            )
            if datetime.fromisoformat(accepted_at) > deadline:
                raise ValueError("operational decision missed its acceptance deadline")
            connection.execute(
                "INSERT INTO operational_decision_commits "
                "(commit_entry_digest,committed_chain_root_digest,"
                "ledger_instance_digest,sequence_number,previous_commit_root_digest,"
                "promotion_certificate_digest,input_plan_digest,"
                "decision_payload_json,accepted_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    commit_entry_digest,
                    committed_chain_root_digest,
                    ledger_instance_digest,
                    sequence_number,
                    previous_operational_decision_digest,
                    promotion_deployment_certificate.certificate_digest,
                    decision_payload["input_plan_digest"],
                    json.dumps(
                        decision_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    accepted_at,
                ),
            )
            updated = connection.execute(
                "UPDATE operational_decision_chain_head SET "
                "sequence_number = ?,certificate_digest = ?,updated_at = ? "
                "WHERE singleton = 1 AND sequence_number = ? AND "
                "certificate_digest = ?",
                (
                    sequence_number,
                    committed_chain_root_digest,
                    accepted_at,
                    int(head["sequence_number"]),
                    head["certificate_digest"],
                ),
            )
            if updated.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "operational decision chain head changed"
                )
            connection.commit()

        committed_at = datetime.now(timezone.utc).isoformat()
        if datetime.fromisoformat(committed_at) > deadline:
            raise ValueError(
                "operational decision committed after its deadline"
            )
        receipt = _issue_operational_decision_ledger_receipt(
                decision_payload,
                ledger_instance_digest=ledger_instance_digest,
                sequence_number=sequence_number,
                previous_operational_decision_digest=(
                    previous_operational_decision_digest
                ),
                accepted_at=accepted_at,
                committed_at=committed_at,
                commit_entry_digest=commit_entry_digest,
                committed_chain_root_digest=committed_chain_root_digest,
                signer=ledger_signer,
                authority_trust_store=authority_trust,
            )
        if datetime.now(timezone.utc) > deadline:
            raise ValueError(
                "operational ledger proof completed after its deadline"
            )
        certificate = _issue_operational_deployment_decision_certificate(
            decision_payload,
            deployment_bundle_release_approval=(
                deployment_bundle_release_approval
            ),
            deployment_runtime_activation_receipt=(
                    deployment_runtime_activation_receipt
                ),
                promotion_deployment_certificate=promotion_deployment_certificate,
                promotion_evidence=promotion_evidence,
                policy=policy,
                policy_trust_store_digest=policy_trust_store_digest,
                regime_evidence=None,
                range_partition_evidence=None,
                range_geometry_contract=None,
                ledger_receipt=receipt,
                signer=operational_signer,
                authority_trust_store=authority_trust,
            )
        if datetime.now(timezone.utc) > deadline:
            raise ValueError(
                "operational decision signature completed after its deadline"
            )
        payload_json = json.dumps(
                certificate.payload,
                sort_keys=True,
                separators=(",", ":"),
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if datetime.now(timezone.utc) > deadline:
                    raise ValueError(
                        "operational decision recording started after its deadline"
                    )
                retained_commit = connection.execute(
                    "SELECT committed_chain_root_digest,decision_payload_json "
                    "FROM operational_decision_commits WHERE commit_entry_digest = ?",
                    (commit_entry_digest,),
                ).fetchone()
                if retained_commit != (
                    committed_chain_root_digest,
                    json.dumps(
                        decision_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ):
                    raise ValueError("operational decision commit preimage changed")
                connection.execute(
                    "INSERT INTO operational_decision_commit_proofs "
                    "(receipt_digest,commit_entry_digest,receipt_json,"
                    "committed_at,created_at) VALUES (?,?,?,?,?)",
                    (
                        receipt.receipt_digest,
                        commit_entry_digest,
                        json.dumps(
                            receipt.payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        receipt.committed_at,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.execute(
                    "INSERT INTO operational_deployment_decisions_v2 "
                    "(certificate_digest,ledger_instance_digest,sequence_number,"
                    "previous_certificate_digest,promotion_certificate_digest,"
                    "release_approval_digest,runtime_activation_receipt_digest,"
                    "input_plan_digest,"
                    "commit_entry_digest,receipt_digest,payload_json,recorded_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        certificate.certificate_digest,
                        certificate.ledger_instance_digest,
                        certificate.ledger_sequence_number,
                        certificate.previous_operational_decision_digest,
                        certificate.promotion_deployment_certificate_digest,
                        certificate.deployment_bundle_release_approval_digest,
                        certificate.deployment_runtime_activation_receipt_digest,
                        certificate.input_plan_digest,
                        commit_entry_digest,
                        receipt.receipt_digest,
                        payload_json,
                        certificate.issued_at,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    "operational deployment decision is already recorded"
                ) from error
            connection.commit()
        decision_row_committed_at = datetime.now(timezone.utc)
        usable = int(decision_row_committed_at <= deadline)
        publication_receipt: OperationalDecisionPublicationReceipt | None = None
        publication_error: ValueError | None = None
        if usable:
            try:
                publication_receipt = (
                    _issue_operational_decision_publication_receipt(
                        certificate,
                        decision_row_committed_at=(
                            decision_row_committed_at.isoformat()
                        ),
                        signer=ledger_signer,
                        authority_trust_store=authority_trust,
                    )
                )
            except ValueError as error:
                usable = 0
                publication_error = error
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO operational_decision_publications "
                "(certificate_digest,decision_row_committed_at,usable,"
                "receipt_digest,receipt_json,created_at) VALUES (?,?,?,?,?,?)",
                (
                    certificate.certificate_digest,
                    decision_row_committed_at.isoformat(),
                    usable,
                    (
                        None
                        if publication_receipt is None
                        else publication_receipt.receipt_digest
                    ),
                    (
                        None
                        if publication_receipt is None
                        else json.dumps(
                            publication_receipt.payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    decision_row_committed_at.isoformat(),
                ),
            )
            connection.commit()
        if not usable:
            if publication_error is not None:
                raise publication_error
            raise ValueError(
                "operational decision recording committed after its deadline"
            )
        return certificate

    def _validate_committed_operational_decision(
        self,
        certificate: OperationalDeploymentDecisionCertificate,
        decision_payload: dict[str, object],
        *,
        expected_index_path: Path,
        authority_trust_store_path: str | Path,
    ) -> tuple[
        OperationalDecisionPublicationReceipt,
        OperationalDecisionActivationReceipt,
        OperationalDecisionCommitAuthorizationReceipt,
    ]:
        """Prove that an automatic decision is an on-time row in this ledger."""

        EpisodeLedger._require_initialized(self)
        if (
            type(expected_index_path) is not _NATIVE_PATH_TYPE
            or self.index_path != expected_index_path
        ):
            raise ValueError(
                "automatic deployment ledger index is not root-approved"
            )
        canonical_decision = json.dumps(
            decision_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical_certificate = json.dumps(
            certificate.payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical_receipt = certificate.operational_ledger_receipt_payload_json
        authority_trust = _load_promotion_deployment_authority_trust_store(
            authority_trust_store_path
        )
        _validate_operational_deployment_decision_certificate(
            certificate,
            decision_payload=decision_payload,
            authority_trust_store=authority_trust,
        )
        with EpisodeLedger._connect_approved_index(
            expected_index_path
        ) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT d.*,p.receipt_json,p.committed_at AS proof_committed_at,"
                "c.decision_payload_json,c.committed_chain_root_digest,"
                "c.ledger_instance_digest AS commit_ledger_instance_digest,"
                "c.sequence_number AS commit_sequence_number,"
                "c.previous_commit_root_digest,c.accepted_at,"
                "u.decision_row_committed_at,u.usable,u.receipt_digest AS "
                "publication_receipt_digest,u.receipt_json AS "
                "publication_receipt_json,u.publication_payload_committed_at,"
                "u.activation_committed_at,"
                "s.status AS issuance_status,s.updated_at AS issuance_updated_at,"
                "a.receipt_digest AS activation_receipt_digest,"
                "a.receipt_json AS activation_receipt_json,"
                "o.receipt_digest AS commit_authorization_receipt_digest,"
                "o.receipt_json AS commit_authorization_receipt_json,"
                "o.terminal_commit_authorized_at,"
                "q.approval_json AS release_approval_json,"
                "q.expires_at AS release_approval_expires_at,"
                "r.receipt_json AS runtime_activation_receipt_json,"
                "r.expires_at AS runtime_activation_expires_at,"
                "h.receipt_digest AS runtime_activation_head_digest,"
                "h.sequence_number AS runtime_activation_head_sequence "
                "FROM operational_deployment_decisions_v2 AS d "
                "JOIN operational_decision_commit_proofs AS p "
                "ON p.receipt_digest = d.receipt_digest "
                "JOIN operational_decision_commits AS c "
                "ON c.commit_entry_digest = d.commit_entry_digest "
                "JOIN operational_decision_publications AS u "
                "ON u.certificate_digest = d.certificate_digest "
                "JOIN operational_decision_issuance_states AS s "
                "ON s.certificate_digest = d.certificate_digest "
                "JOIN operational_decision_activation_receipts AS a "
                "ON a.certificate_digest = d.certificate_digest "
                "JOIN operational_decision_commit_authorization_receipts AS o "
                "ON o.certificate_digest = d.certificate_digest "
                "JOIN deployment_runtime_activations AS r "
                "ON r.receipt_digest = d.runtime_activation_receipt_digest "
                "JOIN deployment_bundle_release_approvals AS q "
                "ON q.approval_digest = d.release_approval_digest "
                "JOIN deployment_runtime_activation_heads AS h "
                "ON h.deployment_instance_digest = r.deployment_instance_digest "
                "WHERE d.certificate_digest = ?",
                (certificate.certificate_digest,),
            ).fetchone()
        try:
            publication_payload = json.loads(
                "" if row is None else str(row["publication_receipt_json"])
            )
            if not isinstance(publication_payload, dict):
                raise ValueError
            publication_receipt = (
                _operational_decision_publication_receipt_from_payload(
                    publication_payload
                )
            )
            activation_payload = json.loads(
                "" if row is None else str(row["activation_receipt_json"])
            )
            if not isinstance(activation_payload, dict):
                raise ValueError
            activation_receipt = (
                _operational_decision_activation_receipt_from_payload(
                    activation_payload
                )
            )
            observation_payload = json.loads(
                "" if row is None else str(
                    row["commit_authorization_receipt_json"]
                )
            )
            if not isinstance(observation_payload, dict):
                raise ValueError
            commit_authorization_receipt = (
                _operational_decision_commit_authorization_receipt_from_payload(
                    observation_payload
                )
            )
            ledger_receipt_payload = json.loads(canonical_receipt)
            if not isinstance(ledger_receipt_payload, dict):
                raise ValueError
            ledger_receipt = _operational_decision_ledger_receipt_from_payload(
                ledger_receipt_payload
            )
            runtime_payload = json.loads(
                "" if row is None else str(
                    row["runtime_activation_receipt_json"]
                )
            )
            if not isinstance(runtime_payload, dict):
                raise ValueError
            runtime_receipt = (
                _deployment_runtime_activation_receipt_from_payload(
                    runtime_payload
                )
            )
            release_payload = json.loads(
                "" if row is None else str(row["release_approval_json"])
            )
            if not isinstance(release_payload, dict):
                raise ValueError
            release_approval = (
                _deployment_bundle_release_approval_from_payload(
                    release_payload
                )
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                "operational decision lacks its committed publication receipt"
            ) from error
        if (
            row is None
            or int(row["usable"]) != 1
            or row["issuance_status"] != "published"
            or row["payload_json"] != canonical_certificate
            or row["decision_payload_json"] != canonical_decision
            or row["receipt_json"] != canonical_receipt
            or row["receipt_digest"]
            != certificate.operational_ledger_receipt_digest
            or row["ledger_instance_digest"]
            != certificate.ledger_instance_digest
            or int(row["sequence_number"])
            != certificate.ledger_sequence_number
            or row["previous_certificate_digest"]
            != certificate.previous_operational_decision_digest
            or row["promotion_certificate_digest"]
            != certificate.promotion_deployment_certificate_digest
            or row["release_approval_digest"]
            != certificate.deployment_bundle_release_approval_digest
            or release_approval.approval_digest
            != certificate.deployment_bundle_release_approval_digest
            or row["release_approval_expires_at"]
            != release_approval.expires_at
            or row["runtime_activation_receipt_digest"]
            != certificate.deployment_runtime_activation_receipt_digest
            or runtime_receipt.receipt_digest
            != certificate.deployment_runtime_activation_receipt_digest
            or row["runtime_activation_expires_at"]
            != runtime_receipt.expires_at
            or row["runtime_activation_head_digest"]
            != runtime_receipt.receipt_digest
            or int(row["runtime_activation_head_sequence"])
            != runtime_receipt.activation_sequence_number
            or runtime_receipt.release_approval_digest
            != release_approval.approval_digest
            or row["input_plan_digest"] != certificate.input_plan_digest
            or row["publication_receipt_digest"]
            != publication_receipt.receipt_digest
            or row["activation_receipt_digest"]
            != activation_receipt.receipt_digest
            or row["commit_authorization_receipt_digest"]
            != commit_authorization_receipt.receipt_digest
            or row["terminal_commit_authorized_at"]
            != commit_authorization_receipt.terminal_commit_authorized_at
            or row["committed_chain_root_digest"]
            != ledger_receipt.committed_chain_root_digest
            or row["proof_committed_at"] != ledger_receipt.committed_at
            or row["commit_ledger_instance_digest"]
            != ledger_receipt.ledger_instance_digest
            or int(row["commit_sequence_number"])
            != ledger_receipt.sequence_number
            or row["previous_commit_root_digest"]
            != ledger_receipt.previous_operational_decision_digest
            or row["accepted_at"] != ledger_receipt.accepted_at
            or row["decision_row_committed_at"]
            != publication_receipt.decision_row_committed_at
            or row["publication_payload_committed_at"] is None
            or row["activation_committed_at"] is None
            or row["issuance_updated_at"] != row["activation_committed_at"]
            or row["activation_committed_at"]
            != activation_receipt.activation_authorized_at
            or datetime.fromisoformat(
                certificate.issued_at.replace("Z", "+00:00")
            )
            > datetime.fromisoformat(
                str(row["decision_row_committed_at"]).replace("Z", "+00:00")
            )
            or datetime.fromisoformat(
                str(row["decision_row_committed_at"]).replace("Z", "+00:00")
            )
            > datetime.fromisoformat(
                publication_receipt.issued_at.replace("Z", "+00:00")
            )
            or datetime.fromisoformat(
                publication_receipt.issued_at.replace("Z", "+00:00")
            )
            > datetime.fromisoformat(
                str(row["publication_payload_committed_at"]).replace(
                    "Z", "+00:00"
                )
            )
            or datetime.fromisoformat(
                str(row["publication_payload_committed_at"]).replace(
                    "Z", "+00:00"
                )
            )
            > datetime.fromisoformat(
                certificate.decision_deadline.replace("Z", "+00:00")
            )
            or datetime.fromisoformat(
                str(row["publication_payload_committed_at"]).replace(
                    "Z", "+00:00"
                )
            )
            > datetime.fromisoformat(
                str(row["activation_committed_at"]).replace("Z", "+00:00")
            )
        ):
            raise ValueError(
                "operational decision is not a usable commit in its bound ledger"
            )
        _validate_operational_decision_activation_receipt(
            activation_receipt,
            certificate=certificate,
            publication_receipt=publication_receipt,
            authority_trust_store=authority_trust,
        )
        _validate_operational_decision_commit_authorization_receipt(
            commit_authorization_receipt,
            certificate=certificate,
            activation_receipt=activation_receipt,
            authority_trust_store=authority_trust,
        )
        _validate_deployment_runtime_activation_receipt(
            runtime_receipt,
            release_approval=release_approval,
            authority_trust_store=authority_trust,
            required_valid_through=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            require_current_runtime=True,
        )
        return (
            publication_receipt,
            activation_receipt,
            commit_authorization_receipt,
        )

    def committed_operational_decision_client(
        self,
        *,
        ledger_signer: DeploymentAuthoritySigner,
        operational_signer: DeploymentAuthoritySigner,
        authority_trust_store_path: str | Path,
        raw_ingestor_trust_store_path: str | Path,
        training_target_source_trust_store_path: str | Path = (
            "/etc/advar/training-target-sources.json"
        ),
    ) -> _EpisodeLedgerOperationalDecisionClient:
        """Bind signer capabilities inside the only automatic ledger client."""

        def issue(
            decision_payload: dict[str, object],
            *,
            deployment_bundle_release_approval: DeploymentBundleReleaseApproval,
            deployment_runtime_activation_receipt: (
                DeploymentRuntimeActivationReceipt
            ),
            promotion_deployment_certificate: LedgeredPromotionDeploymentCertificate,
            promotion_evidence: NeuralPriorPromotionEvidence,
            policy: DeployedNeuralPriorPolicy,
            policy_trust_store_digest: str,
            regime_evidence: RegimeClassificationEvidence,
            range_partition_evidence: RangePartitionEvidence,
            range_geometry_contract: (
                RangeGeometryContract | MosaicRangeGeometryContract
            ),
            regime_classifier: NeuralPriorRegimeClassifier,
            input_run: ForecastRunContract,
            range_grid_x_m: Tensor,
            range_grid_y_m: Tensor,
        ) -> OperationalDeploymentDecisionCertificate:
            return self.issue_operational_deployment_decision(
                decision_payload,
                deployment_bundle_release_approval=(
                    deployment_bundle_release_approval
                ),
                deployment_runtime_activation_receipt=(
                    deployment_runtime_activation_receipt
                ),
                promotion_deployment_certificate=promotion_deployment_certificate,
                promotion_evidence=promotion_evidence,
                policy=policy,
                policy_trust_store_digest=policy_trust_store_digest,
                ledger_signer=ledger_signer,
                operational_signer=operational_signer,
                authority_trust_store_path=authority_trust_store_path,
                raw_ingestor_trust_store_path=(
                    raw_ingestor_trust_store_path
                ),
                training_target_source_trust_store_path=(
                    training_target_source_trust_store_path
                ),
                regime_evidence=regime_evidence,
                range_partition_evidence=range_partition_evidence,
                range_geometry_contract=range_geometry_contract,
                regime_classifier=regime_classifier,
                input_run=input_run,
                range_grid_x_m=range_grid_x_m,
                range_grid_y_m=range_grid_y_m,
            )

        result = object.__new__(_EpisodeLedgerOperationalDecisionClient)
        object.__setattr__(result, "_issuer", issue)
        object.__setattr__(result, "_ledger", self)
        object.__setattr__(
            result,
            "_authority_trust_store_path",
            str(Path(authority_trust_store_path).expanduser().resolve()),
        )
        object.__setattr__(
            result,
            "_raw_ingestor_trust_store_path",
            str(Path(raw_ingestor_trust_store_path).expanduser().resolve()),
        )
        object.__setattr__(
            result,
            "_training_target_source_trust_store_path",
            str(
                Path(training_target_source_trust_store_path)
                .expanduser()
                .resolve()
            ),
        )
        return result

    def load_neural_prior_promotion_deployment_certificate(
        self,
        certificate_digest: str,
        *,
        authority_trust_store_path: str | Path,
        _require_raw_trust_activation: bool = True,
    ) -> LedgeredPromotionDeploymentCertificate:
        """Load a certificate and revalidate its evidence preimage and root."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM neural_prior_promotion_deployment_certificates_v3 "
                "WHERE certificate_digest = ?",
                (certificate_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown deployment certificate: {certificate_digest}")
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError as error:
            raise ValueError("invalid stored deployment certificate") from error
        if not isinstance(payload, dict):
            raise ValueError("invalid stored deployment certificate")
        certificate = _ledgered_promotion_deployment_certificate_from_payload(
            payload
        )
        if _require_raw_trust_activation:
            with self._connect() as connection:
                self._require_raw_trust_artifact_usable(
                    connection,
                    artifact_kind="promotion_deployment_certificate",
                    artifact_digest=certificate.certificate_digest,
                    raw_ingestor_trust_store_digest=(
                        certificate.raw_ingestor_trust_store_digest
                    ),
                )
        evidence = self.load_neural_prior_promotion(
            certificate.promotion_evidence_digest
        )
        if type(evidence) is not NeuralPriorPromotionEvidence:
            raise ValueError("certificate references legacy promotion evidence")
        assert isinstance(evidence, NeuralPriorPromotionEvidence)
        _validate_ledgered_promotion_deployment_certificate(
            certificate,
            authority_trust_store=(
                _load_promotion_deployment_authority_trust_store(
                    authority_trust_store_path
                )
            ),
            promotion_evidence=evidence,
        )
        if (
            certificate.certificate_digest != certificate_digest
            or row["promotion_evidence_digest"]
            != certificate.promotion_evidence_digest
            or row["ledger_chain_head_digest"]
            != certificate.ledger_chain_head_digest
            or row["previous_certificate_digest"]
            != certificate.previous_certificate_digest
            or row["ledger_instance_digest"]
            != certificate.ledger_instance_digest
            or row["sequence_number"] != certificate.sequence_number
        ):
            raise ValueError("stored deployment certificate lineage disagrees")
        with self._connect() as connection:
            predecessor = connection.execute(
                "SELECT ledger_instance_digest,sequence_number FROM "
                "neural_prior_promotion_deployment_certificates_v3 "
                "WHERE certificate_digest = ?",
                (certificate.previous_certificate_digest,),
            ).fetchone()
        if certificate.sequence_number == 1:
            predecessor_ok = (
                certificate.previous_certificate_digest
                == _PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
            )
        else:
            predecessor_ok = predecessor is not None and (
                predecessor[0] == certificate.ledger_instance_digest
                and int(predecessor[1]) == certificate.sequence_number - 1
            )
        if not predecessor_ok:
            raise ValueError("stored deployment certificate chain is forked")
        return certificate

    def load_neural_prior_promotion(
        self,
        promotion_evidence_digest: str,
        *,
        _require_raw_trust_activation: bool = True,
    ) -> (
        NeuralPriorPromotionEvidence
        | LegacyNeuralPriorPromotionEvidenceAuditV3
        | LegacyNeuralPriorPromotionEvidenceAuditV4
        | LegacyNeuralPriorPromotionEvidenceAuditV5
        | LegacyNeuralPriorPromotionEvidenceAuditV6
        | LegacyNeuralPriorPromotionEvidenceAuditV7
        | LegacyNeuralPriorPromotionEvidenceAuditV8
        | LegacyNeuralPriorPromotionEvidenceAuditV9
        | LegacyNeuralPriorPromotionEvidenceAuditV10
        | LegacyNeuralPriorPromotionEvidenceAuditV11
        | LegacyNeuralPriorPromotionEvidenceAuditV12
        | LegacyNeuralPriorPromotionEvidenceAuditV13
        | LegacyNeuralPriorPromotionEvidenceAuditV14
        | LegacyNeuralPriorPromotionEvidenceAuditV15
        | LegacyNeuralPriorPromotionEvidenceAuditV16
        | LegacyNeuralPriorPromotionEvidenceAuditV17
        | LegacyNeuralPriorPromotionEvidenceAuditV18
        | LegacyNeuralPriorPromotionEvidenceAuditV19
        | LegacyNeuralPriorPromotionEvidenceAuditV20
        | LegacyNeuralPriorPromotionEvidenceAuditV21
        | LegacyNeuralPriorPromotionEvidenceAuditV22
        | LegacyNeuralPriorPromotionEvidenceAuditV23
        | LegacyNeuralPriorPromotionEvidenceAuditV24
        | LegacyNeuralPriorPromotionEvidenceAuditV25
        | LegacyNeuralPriorPromotionEvidenceAuditV26
        | LegacyNeuralPriorPromotionEvidenceAuditV27
        | LegacyNeuralPriorPromotionEvidenceAuditV28
        | LegacyNeuralPriorPromotionEvidenceAuditV29
        | LegacyNeuralPriorPromotionEvidenceAuditV30
    ):
        """Load and validate one immutable prior-promotion decision."""

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM neural_prior_promotions
                WHERE promotion_evidence_digest = ?
                """,
                (promotion_evidence_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown neural-prior promotion: {promotion_evidence_digest}"
            )
        common_payload: dict[str, object] = {
            "candidate_prior_digest": row["candidate_prior_digest"],
            "parent_prior_digest": row["parent_prior_digest"],
            "candidate_manifest_digest": row["candidate_manifest_digest"],
            "policy_digest": row["policy_digest"],
            "trust_store_digest": row["trust_store_digest"],
            "evaluation_digests": tuple(
                json.loads(row["evaluation_digests_json"])
            ),
            "holdout_case_count": row["realized_intervention_count"],
            "material_case_count": row["material_outcome_count"],
            "distinct_case_count": row["distinct_case_count"],
            "distinct_storm_count": row["distinct_storm_count"],
            "distinct_day_count": row["distinct_day_count"],
            "distinct_radar_count": row["distinct_radar_count"],
            "distinct_regime_count": row["distinct_regime_count"],
            "distinct_range_regime_count": row["distinct_range_regime_count"],
            "beneficial_fraction": row["beneficial_fraction"],
            "beneficial_fraction_lower_bound": (
                row["beneficial_fraction_lower_bound"]
            ),
            "harmful_fraction": row["harmful_fraction"],
            "harmful_fraction_upper_bound": row["harmful_fraction_upper_bound"],
            "mean_normalized_improvement": row["mean_normalized_improvement"],
            "mean_improvement_lower_bound": row["mean_improvement_lower_bound"],
            "maximum_normalized_degradation": (
                row["maximum_normalized_degradation"]
            ),
            "eligible": bool(row["eligible"]),
            "rejection_reasons": tuple(
                json.loads(row["rejection_reasons_json"])
            ),
            "contract": row["evidence_contract"],
        }
        contract = row["evidence_contract"]
        if contract == "neural-prior-promotion-evidence-v3":
            evidence: (
                NeuralPriorPromotionEvidence
                | LegacyNeuralPriorPromotionEvidenceAuditV3
                | LegacyNeuralPriorPromotionEvidenceAuditV4
                | LegacyNeuralPriorPromotionEvidenceAuditV5
                | LegacyNeuralPriorPromotionEvidenceAuditV6
                | LegacyNeuralPriorPromotionEvidenceAuditV7
                | LegacyNeuralPriorPromotionEvidenceAuditV8
                | LegacyNeuralPriorPromotionEvidenceAuditV9
                | LegacyNeuralPriorPromotionEvidenceAuditV10
                | LegacyNeuralPriorPromotionEvidenceAuditV11
                | LegacyNeuralPriorPromotionEvidenceAuditV12
                | LegacyNeuralPriorPromotionEvidenceAuditV13
                | LegacyNeuralPriorPromotionEvidenceAuditV14
                | LegacyNeuralPriorPromotionEvidenceAuditV15
                | LegacyNeuralPriorPromotionEvidenceAuditV16
                | LegacyNeuralPriorPromotionEvidenceAuditV17
                | LegacyNeuralPriorPromotionEvidenceAuditV18
                | LegacyNeuralPriorPromotionEvidenceAuditV19
                | LegacyNeuralPriorPromotionEvidenceAuditV20
                | LegacyNeuralPriorPromotionEvidenceAuditV21
                | LegacyNeuralPriorPromotionEvidenceAuditV22
                | LegacyNeuralPriorPromotionEvidenceAuditV23
                | LegacyNeuralPriorPromotionEvidenceAuditV24
                | LegacyNeuralPriorPromotionEvidenceAuditV25
                | LegacyNeuralPriorPromotionEvidenceAuditV26
                | LegacyNeuralPriorPromotionEvidenceAuditV27
                | LegacyNeuralPriorPromotionEvidenceAuditV28
                | LegacyNeuralPriorPromotionEvidenceAuditV29
                | LegacyNeuralPriorPromotionEvidenceAuditV30
            ) = LegacyNeuralPriorPromotionEvidenceAuditV3(
                promotion_evidence_digest=promotion_evidence_digest,
                payload_json=json.dumps(
                    common_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        else:
            v4_payload = {
                **common_payload,
                "prior_gaussian_nll_increase_upper_bound": (
                    row["prior_gaussian_nll_increase_upper_bound"]
                ),
                "prior_support_brier_increase_upper_bound": (
                    row["prior_support_brier_increase_upper_bound"]
                ),
                "prior_underdispersion_increase_upper_bound": (
                    row["prior_underdispersion_increase_upper_bound"]
                ),
            }
            if contract == "neural-prior-promotion-evidence-v4":
                evidence = LegacyNeuralPriorPromotionEvidenceAuditV4(
                    promotion_evidence_digest=promotion_evidence_digest,
                    payload_json=json.dumps(
                        v4_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            else:
                v5_payload = {
                    **common_payload,
                    "prior_echo_intensity_nll_increase_upper_bound": (
                        row["prior_echo_intensity_nll_increase_upper_bound"]
                    ),
                    "prior_support_brier_increase_upper_bound": (
                        row["prior_support_brier_increase_upper_bound"]
                    ),
                    "prior_clear_sky_false_echo_increase_upper_bound": (
                        row["prior_clear_sky_false_echo_increase_upper_bound"]
                    ),
                    "prior_underdispersion_increase_upper_bound": (
                        row["prior_underdispersion_increase_upper_bound"]
                    ),
                }
                if contract == "neural-prior-promotion-evidence-v5":
                    evidence = LegacyNeuralPriorPromotionEvidenceAuditV5(
                        promotion_evidence_digest=promotion_evidence_digest,
                        payload_json=json.dumps(
                            v5_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                else:
                    v6_payload = {
                        **v5_payload,
                        "prior_echo_component_status": (
                            row["prior_echo_component_status"]
                        ),
                        "prior_clear_sky_component_status": (
                            row["prior_clear_sky_component_status"]
                        ),
                        "prior_echo_case_count": row["prior_echo_case_count"],
                        "prior_clear_sky_case_count": (
                            row["prior_clear_sky_case_count"]
                        ),
                        "prior_echo_cluster_count": (
                            row["prior_echo_cluster_count"]
                        ),
                        "prior_clear_sky_cluster_count": (
                            row["prior_clear_sky_cluster_count"]
                        ),
                        "simultaneous_inference_test_count": (
                            row["simultaneous_inference_test_count"]
                        ),
                    }
                    v6_payload["contract"] = contract
                    if contract == "neural-prior-promotion-evidence-v6":
                        evidence = LegacyNeuralPriorPromotionEvidenceAuditV6(
                            promotion_evidence_digest=promotion_evidence_digest,
                            payload_json=json.dumps(
                                v6_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    elif contract in (
                        "neural-prior-promotion-evidence-v7",
                        "neural-prior-promotion-evidence-v8",
                        "neural-prior-promotion-evidence-v9",
                        "neural-prior-promotion-evidence-v10",
                        "neural-prior-promotion-evidence-v11",
                        "neural-prior-promotion-evidence-v12",
                        "neural-prior-promotion-evidence-v13",
                        "neural-prior-promotion-evidence-v14",
                        "neural-prior-promotion-evidence-v15",
                        "neural-prior-promotion-evidence-v16",
                        "neural-prior-promotion-evidence-v17",
                        "neural-prior-promotion-evidence-v18",
                        "neural-prior-promotion-evidence-v19",
                        "neural-prior-promotion-evidence-v20",
                        "neural-prior-promotion-evidence-v21",
                        "neural-prior-promotion-evidence-v22",
                        "neural-prior-promotion-evidence-v23",
                        "neural-prior-promotion-evidence-v24",
                        "neural-prior-promotion-evidence-v25",
                        "neural-prior-promotion-evidence-v26",
                        "neural-prior-promotion-evidence-v27",
                        "neural-prior-promotion-evidence-v28",
                        "neural-prior-promotion-evidence-v29",
                        "neural-prior-promotion-evidence-v31",
                    ):
                        raw_payload = json.loads(row["evidence_payload_json"])
                        if not isinstance(raw_payload, dict):
                            raise ValueError(
                                "invalid neural-prior promotion evidence payload"
                            )
                        raw_payload["evaluation_digests"] = tuple(
                            raw_payload["evaluation_digests"]
                        )
                        raw_payload["rejection_reasons"] = tuple(
                            raw_payload["rejection_reasons"]
                        )
                        if contract in (
                            "neural-prior-promotion-evidence-v9",
                            "neural-prior-promotion-evidence-v10",
                            "neural-prior-promotion-evidence-v11",
                            "neural-prior-promotion-evidence-v12",
                            "neural-prior-promotion-evidence-v13",
                            "neural-prior-promotion-evidence-v14",
                            "neural-prior-promotion-evidence-v15",
                            "neural-prior-promotion-evidence-v16",
                            "neural-prior-promotion-evidence-v17",
                            "neural-prior-promotion-evidence-v18",
                            "neural-prior-promotion-evidence-v19",
                            "neural-prior-promotion-evidence-v20",
                            "neural-prior-promotion-evidence-v21",
                            "neural-prior-promotion-evidence-v22",
                            "neural-prior-promotion-evidence-v23",
                            "neural-prior-promotion-evidence-v24",
                            "neural-prior-promotion-evidence-v25",
                            "neural-prior-promotion-evidence-v26",
                            "neural-prior-promotion-evidence-v27",
                            "neural-prior-promotion-evidence-v28",
                            "neural-prior-promotion-evidence-v29",
                            "neural-prior-promotion-evidence-v31",
                        ):
                            raw_payload["regime_classifier_evidence_digests"] = tuple(
                                raw_payload["regime_classifier_evidence_digests"]
                            )
                        raw_payload[
                            "certified_applicability_regime_groups"
                        ] = tuple(
                            tuple(item)
                            for item in raw_payload[
                                "certified_applicability_regime_groups"
                            ]
                        )
                        if contract in (
                            "neural-prior-promotion-evidence-v11",
                            "neural-prior-promotion-evidence-v12",
                            "neural-prior-promotion-evidence-v13",
                            "neural-prior-promotion-evidence-v14",
                            "neural-prior-promotion-evidence-v15",
                            "neural-prior-promotion-evidence-v16",
                            "neural-prior-promotion-evidence-v17",
                            "neural-prior-promotion-evidence-v18",
                            "neural-prior-promotion-evidence-v19",
                            "neural-prior-promotion-evidence-v20",
                            "neural-prior-promotion-evidence-v21",
                            "neural-prior-promotion-evidence-v22",
                            "neural-prior-promotion-evidence-v23",
                            "neural-prior-promotion-evidence-v24",
                            "neural-prior-promotion-evidence-v25",
                            "neural-prior-promotion-evidence-v26",
                            "neural-prior-promotion-evidence-v27",
                            "neural-prior-promotion-evidence-v28",
                            "neural-prior-promotion-evidence-v29",
                            "neural-prior-promotion-evidence-v31",
                        ):
                            raw_payload["range_band_skill_bounds"] = tuple(
                                tuple(item)
                                for item in raw_payload[
                                    "range_band_skill_bounds"
                                ]
                            )
                        if contract in (
                            "neural-prior-promotion-evidence-v12",
                            "neural-prior-promotion-evidence-v13",
                            "neural-prior-promotion-evidence-v14",
                            "neural-prior-promotion-evidence-v15",
                            "neural-prior-promotion-evidence-v16",
                            "neural-prior-promotion-evidence-v17",
                            "neural-prior-promotion-evidence-v18",
                            "neural-prior-promotion-evidence-v19",
                            "neural-prior-promotion-evidence-v20",
                            "neural-prior-promotion-evidence-v21",
                            "neural-prior-promotion-evidence-v22",
                            "neural-prior-promotion-evidence-v23",
                            "neural-prior-promotion-evidence-v24",
                            "neural-prior-promotion-evidence-v25",
                            "neural-prior-promotion-evidence-v26",
                            "neural-prior-promotion-evidence-v27",
                            "neural-prior-promotion-evidence-v28",
                            "neural-prior-promotion-evidence-v29",
                            "neural-prior-promotion-evidence-v31",
                        ):
                            raw_payload[
                                "range_band_skill_inference_diagnostics"
                            ] = tuple(
                                tuple(item)
                                for item in raw_payload[
                                    "range_band_skill_inference_diagnostics"
                                ]
                            )
                        if contract in (
                            "neural-prior-promotion-evidence-v13",
                            "neural-prior-promotion-evidence-v14",
                            "neural-prior-promotion-evidence-v15",
                            "neural-prior-promotion-evidence-v16",
                            "neural-prior-promotion-evidence-v17",
                            "neural-prior-promotion-evidence-v18",
                            "neural-prior-promotion-evidence-v19",
                            "neural-prior-promotion-evidence-v20",
                            "neural-prior-promotion-evidence-v21",
                            "neural-prior-promotion-evidence-v22",
                            "neural-prior-promotion-evidence-v23",
                            "neural-prior-promotion-evidence-v24",
                            "neural-prior-promotion-evidence-v25",
                            "neural-prior-promotion-evidence-v26",
                            "neural-prior-promotion-evidence-v27",
                            "neural-prior-promotion-evidence-v28",
                            "neural-prior-promotion-evidence-v29",
                            "neural-prior-promotion-evidence-v31",
                        ):
                            raw_payload[
                                "certified_range_geometry_contract_digests"
                            ] = tuple(
                                raw_payload[
                                    "certified_range_geometry_contract_digests"
                                ]
                            )
                        indexed_v7 = {
                            **common_payload,
                            "prior_echo_intensity_nll_increase_upper_bound": (
                                row[
                                    "prior_echo_intensity_nll_increase_upper_bound"
                                ]
                            ),
                            "prior_support_brier_increase_upper_bound": (
                                row["prior_support_brier_increase_upper_bound"]
                            ),
                            "prior_clear_sky_false_echo_increase_upper_bound": (
                                row[
                                    "prior_clear_sky_false_echo_increase_upper_bound"
                                ]
                            ),
                            "prior_conditional_underdispersion_increase_upper_bound": (
                                row[
                                    "prior_conditional_underdispersion_increase_upper_bound"
                                ]
                            ),
                            "prior_echo_support_miss_increase_upper_bound": (
                                row[
                                    "prior_echo_support_miss_increase_upper_bound"
                                ]
                            ),
                            "prior_echo_object_miss_increase_upper_bound": (
                                row[
                                    "prior_echo_object_miss_increase_upper_bound"
                                ]
                            ),
                            "prior_echo_component_status": row[
                                "prior_echo_component_status"
                            ],
                            "prior_clear_sky_component_status": row[
                                "prior_clear_sky_component_status"
                            ],
                            "prior_echo_case_count": row["prior_echo_case_count"],
                            "prior_clear_sky_case_count": row[
                                "prior_clear_sky_case_count"
                            ],
                            "prior_echo_cluster_count": row[
                                "prior_echo_cluster_count"
                            ],
                            "prior_clear_sky_cluster_count": row[
                                "prior_clear_sky_cluster_count"
                            ],
                            "simultaneous_inference_test_count": row[
                                "simultaneous_inference_test_count"
                            ],
                        }
                        if any(
                            raw_payload.get(name) != value
                            for name, value in indexed_v7.items()
                        ):
                            raise ValueError(
                                "neural-prior promotion index disagrees with payload"
                            )
                        if contract == "neural-prior-promotion-evidence-v7":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV7(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v8":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV8(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v9":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV9(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v10":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV10(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v11":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV11(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v12":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV12(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v13":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV13(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v14":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV14(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v15":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV15(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v16":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV16(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v17":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV17(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v18":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV18(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v19":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV19(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v20":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV20(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v21":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV21(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v22":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV22(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v23":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV23(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v24":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV24(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v25":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV25(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v26":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV26(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v27":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV27(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v28":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV28(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v29":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV29(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        elif contract == "neural-prior-promotion-evidence-v30":
                            evidence = LegacyNeuralPriorPromotionEvidenceAuditV30(
                                promotion_evidence_digest=promotion_evidence_digest,
                                payload_json=json.dumps(
                                    raw_payload,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        else:
                            raw_payload["range_metric_cell_bounds"] = tuple(
                                tuple(item)
                                for item in raw_payload["range_metric_cell_bounds"]
                            )
                            raw_payload[
                                "range_metric_end_to_end_cell_bounds"
                            ] = tuple(
                                tuple(item)
                                for item in raw_payload[
                                    "range_metric_end_to_end_cell_bounds"
                                ]
                            )
                            raw_payload["range_issuance_cell_bounds"] = tuple(
                                tuple(item)
                                for item in raw_payload[
                                    "range_issuance_cell_bounds"
                                ]
                            )
                            evidence = NeuralPriorPromotionEvidence(
                                **cast(Any, raw_payload)
                            )
                    else:
                        raise ValueError(
                            "unsupported neural-prior promotion evidence"
                        )
                if evidence.promotion_evidence_digest != promotion_evidence_digest:
                    raise ValueError("neural-prior promotion digest mismatch")
        payloads = json.loads(row["evaluation_payloads_json"])
        evaluations = _decode_evaluation_audit_payloads(payloads)
        if tuple(item.evaluation_digest for item in evaluations) != (
            tuple(cast(tuple[str, ...], common_payload["evaluation_digests"]))
        ):
            raise ValueError("neural-prior promotion evaluation audit mismatch")
        manifest = _decode_candidate_manifest(
            row["candidate_manifest_json"],
            expected_digest=row["candidate_manifest_digest"],
        )
        if type(evidence) is NeuralPriorPromotionEvidence:
            if _require_raw_trust_activation:
                with self._connect() as connection:
                    self._require_raw_trust_artifact_usable(
                        connection,
                        artifact_kind="promotion_evidence",
                        artifact_digest=evidence.promotion_evidence_digest,
                        raw_ingestor_trust_store_digest=(
                            evidence.raw_ingestor_trust_store_digest
                        ),
                    )
            if not isinstance(manifest, NeuralPriorCandidateManifest) or any(
                not isinstance(item, PriorHoldoutEvaluation)
                for item in evaluations
            ):
                raise ValueError("current promotion requires current typed inputs")
            if (
                evidence.candidate_manifest_digest != manifest.manifest_digest
                or evidence.candidate_prior_digest
                != manifest.candidate_prior_digest
                or evidence.parent_prior_digest != manifest.parent_prior_digest
            ):
                raise ValueError(
                    "promotion evidence disagrees with its candidate manifest"
                )
            with self._connect() as connection:
                scoring_row = connection.execute(
                    "SELECT payload_json FROM "
                    "neural_prior_holdout_scoring_artifacts "
                    "WHERE artifact_digest = ?",
                    (evidence.scoring_artifact_digest,),
                ).fetchone()
                log_row = connection.execute(
                    "SELECT payload_json FROM trusted_process_log_artifacts "
                    "WHERE artifact_digest = ?",
                    (evidence.scoring_process_log_digest,),
                ).fetchone()
                completion_row = connection.execute(
                    "SELECT receipt_json FROM trusted_process_completion_receipts "
                    "WHERE receipt_digest = ?",
                    (evidence.scoring_completion_receipt_digest,),
                ).fetchone()
            if scoring_row is None or log_row is None or completion_row is None:
                raise ValueError("promotion scoring artifacts are unavailable")
            scoring_artifact = _decode_holdout_scoring_artifact(
                scoring_row[0], evidence.scoring_artifact_digest
            )
            if type(scoring_artifact) is not HoldoutScoringArtifact:
                raise ValueError("current promotion requires current scoring artifact")
            with self._connect() as connection:
                scoring_input_row = connection.execute(
                    "SELECT payload_json FROM "
                    "neural_prior_holdout_scoring_input_artifacts "
                    "WHERE artifact_digest = ?",
                    (scoring_artifact.scoring_input_artifact_digest,),
                ).fetchone()
            if scoring_input_row is None:
                raise ValueError("promotion scoring input artifact is unavailable")
            scoring_input_artifact = _decode_holdout_scoring_input_artifact(
                scoring_input_row[0],
                scoring_artifact.scoring_input_artifact_digest,
            )
            process_log = _decode_process_log_artifact(
                log_row[0], evidence.scoring_process_log_digest
            )
            completion = _decode_completion_receipt(
                completion_row[0], evidence.scoring_completion_receipt_digest
            )
            scoring_plan = self.load_neural_prior_holdout_plan(
                scoring_artifact.holdout_plan_digest
            )
            if not isinstance(scoring_plan, NeuralPriorHoldoutPlan):
                raise ValueError("current scoring artifact requires current plan")
            validate_holdout_scoring_artifact(
                scoring_artifact,
                manifest,
                scoring_plan,
                scoring_input_artifact,
                cast(tuple[PriorHoldoutEvaluation, ...], evaluations),
            )
            validate_trusted_process_completion_receipt(
                completion,
                manifest.candidate_scoring_start_receipt,
            )
            if (
                process_log.start_receipt_digest
                != manifest.candidate_scoring_start_receipt.receipt_digest
                or process_log.process_kind != "candidate_scoring"
                or completion.output_artifact_digest
                != scoring_artifact.artifact_digest
                or completion.process_log_digest != process_log.artifact_digest
            ):
                raise ValueError("promotion scoring artifact replay failed")
        return evidence

    def load_neural_prior_promotion_evaluations(
        self,
        promotion_evidence_digest: str,
    ) -> tuple[
        PriorHoldoutEvaluation | LegacyPromotionEvaluationAudit,
        ...,
    ]:
        """Load typed metrics or an explicit read-only legacy audit."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT evaluation_payloads_json FROM neural_prior_promotions "
                "WHERE promotion_evidence_digest = ?",
                (promotion_evidence_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown neural-prior promotion: {promotion_evidence_digest}"
            )
        return _decode_evaluation_audit_payloads(json.loads(row[0]))

    def _initialize_index(self) -> None:
        with self._connect() as connection:
            previous_schema_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    issue_time TEXT NOT NULL,
                    radar_id TEXT NOT NULL,
                    contract_hash TEXT NOT NULL,
                    model_commit TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    promotion_eligible INTEGER NOT NULL CHECK(
                        promotion_eligible = 0
                    ),
                    impact_available INTEGER NOT NULL,
                    indirect_observation_sensitivity_available INTEGER NOT NULL
                        DEFAULT 0 CHECK(
                            indirect_observation_sensitivity_available = 0
                        ),
                    manifest_sha256 TEXT NOT NULL,
                    arrays_sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_episode_impacts_schema(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS variational_learning_approvals (
                    learning_result_digest TEXT PRIMARY KEY,
                    approval_evidence_digest TEXT NOT NULL,
                    evidence_contract TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    fsoi_digest TEXT NOT NULL,
                    full_step_analysis_digest TEXT NOT NULL,
                    half_step_analysis_digest TEXT NOT NULL,
                    full_step_forecast_digest TEXT NOT NULL,
                    half_step_forecast_digest TEXT NOT NULL,
                    first_order_validation_digest TEXT NOT NULL,
                    learning_impact_digest TEXT NOT NULL,
                    approved_action_digest TEXT,
                    nominal_input_bundle_digest TEXT,
                    nominal_full_analysis_input_digest TEXT,
                    selection_mode TEXT NOT NULL DEFAULT 'direct',
                    candidate_id TEXT,
                    candidate_rank INTEGER,
                    candidate_score REAL,
                    candidate_perturbation_digest TEXT,
                    ranking_digest TEXT,
                    ranking_policy_digest TEXT,
                    ranking_objective TEXT,
                    whitener_operations_per_apply INTEGER NOT NULL DEFAULT 0,
                    observed_whitener_apply_count INTEGER NOT NULL DEFAULT 0,
                    observed_whitener_total_operations INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS realized_observation_interventions (
                    intervention_digest TEXT PRIMARY KEY,
                    intervention_id TEXT NOT NULL UNIQUE,
                    intervention_type TEXT NOT NULL,
                    action_digest TEXT NOT NULL,
                    applied_time TEXT NOT NULL,
                    actual_input_before_digest TEXT NOT NULL,
                    actual_input_after_digest TEXT NOT NULL,
                    observed_outcome_digest TEXT NOT NULL,
                    execution_policy_digest TEXT NOT NULL,
                    execution_trust_store_digest TEXT NOT NULL,
                    predicted_normalized_benefit REAL NOT NULL,
                    resolved_normalized_benefit REAL NOT NULL,
                    learning_result_digest TEXT NOT NULL,
                    learning_approval_evidence_digest TEXT NOT NULL,
                    counterfactual_perturbation_digest TEXT NOT NULL,
                    linearization_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL DEFAULT '',
                    radar_id TEXT NOT NULL DEFAULT '',
                    issue_time TEXT NOT NULL DEFAULT '',
                    input_bundle_before_digest TEXT NOT NULL DEFAULT '',
                    input_bundle_after_digest TEXT NOT NULL DEFAULT '',
                    resolved_issuance_validation_digest TEXT NOT NULL DEFAULT '',
                    evidence_contract TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (learning_result_digest)
                        REFERENCES variational_learning_approvals(
                            learning_result_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrospective_counterfactual_replays (
                    replay_digest TEXT PRIMARY KEY,
                    replay_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reusable_intervention_policies (
                    policy_digest TEXT PRIMARY KEY,
                    policy_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prospective_intervention_decisions (
                    decision_digest TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL UNIQUE,
                    decision_json TEXT NOT NULL,
                    operator_approval_digest TEXT NOT NULL DEFAULT '',
                    operator_approval_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS realized_intervention_receipts (
                    receipt_digest TEXT PRIMARY KEY,
                    decision_digest TEXT NOT NULL UNIQUE,
                    executor_key_id TEXT NOT NULL DEFAULT '',
                    executor_sequence_number INTEGER NOT NULL DEFAULT 0,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (decision_digest)
                        REFERENCES prospective_intervention_decisions(
                            decision_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_plans (
                    plan_digest TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL UNIQUE,
                    plan_json TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_event_catalog_results (
                    plan_digest TEXT PRIMARY KEY,
                    result_digest TEXT NOT NULL UNIQUE,
                    result_json TEXT NOT NULL,
                    cataloged_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (plan_digest)
                        REFERENCES neural_prior_holdout_plans(plan_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_process_start_receipts (
                    receipt_digest TEXT PRIMARY KEY,
                    plan_digest TEXT NOT NULL,
                    result_digest TEXT NOT NULL,
                    process_kind TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (plan_digest, process_kind),
                    FOREIGN KEY (plan_digest)
                        REFERENCES neural_prior_holdout_plans(plan_digest),
                    FOREIGN KEY (result_digest)
                        REFERENCES neural_prior_event_catalog_results(result_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_training_event_catalog_results (
                    plan_digest TEXT PRIMARY KEY,
                    result_digest TEXT NOT NULL UNIQUE,
                    plan_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    cataloged_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotion_decision_rules (
                    rule_digest TEXT PRIMARY KEY,
                    holdout_plan_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotion_rule_definitions (
                    rule_digest TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_plan_rule_bindings (
                    holdout_plan_digest TEXT PRIMARY KEY,
                    rule_digest TEXT NOT NULL,
                    bound_at TEXT NOT NULL,
                    FOREIGN KEY (holdout_plan_digest)
                        REFERENCES neural_prior_holdout_plans(plan_digest),
                    FOREIGN KEY (rule_digest)
                        REFERENCES neural_prior_promotion_rule_definitions(rule_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotion_experiment_families (
                    family_digest TEXT PRIMARY KEY,
                    holdout_cohort_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_plan_experiment_bindings (
                    holdout_plan_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    trial_digest TEXT NOT NULL,
                    bound_at TEXT NOT NULL,
                    UNIQUE (family_digest, trial_digest),
                    FOREIGN KEY (holdout_plan_digest)
                        REFERENCES neural_prior_holdout_plans(plan_digest),
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_experiment_family_consumptions (
                    family_digest TEXT PRIMARY KEY,
                    promotion_evidence_digest TEXT NOT NULL UNIQUE,
                    consumed_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_sampling_unit_reservations (
                    sampling_unit_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_sampling_unit_consumptions (
                    sampling_unit_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    promotion_evidence_digest TEXT NOT NULL,
                    consumed_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        ),
                    FOREIGN KEY (promotion_evidence_digest)
                        REFERENCES neural_prior_promotions(
                            promotion_evidence_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_raw_registry_entries (
                    receipt_digest TEXT PRIMARY KEY,
                    registry_id TEXT NOT NULL,
                    registry_sequence_number INTEGER NOT NULL,
                    previous_registry_root_digest TEXT NOT NULL,
                    committed_registry_root_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (registry_id, registry_sequence_number)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS global_sampling_registry_entries (
                    registry_id TEXT NOT NULL,
                    registry_sequence_number INTEGER NOT NULL,
                    previous_registry_root_digest TEXT NOT NULL,
                    committed_registry_root_digest TEXT NOT NULL UNIQUE,
                    receipt_digest TEXT NOT NULL UNIQUE,
                    entry_kind TEXT NOT NULL CHECK (
                        entry_kind IN (
                            'training_raw',
                            'slot_reservation',
                            'raw_resolution'
                        )
                    ),
                    family_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (registry_id, registry_sequence_number),
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_raw_observation_slot_reservations (
                    raw_observation_slot_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    global_receipt_digest TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_raw_volume_identity_reservations (
                    raw_volume_identity_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    global_resolution_receipt_digest TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_observation_slot_identity_bindings (
                    raw_observation_slot_digest TEXT PRIMARY KEY,
                    raw_volume_identity_digest TEXT NOT NULL,
                    family_digest TEXT NOT NULL,
                    first_resolved_at TEXT NOT NULL,
                    FOREIGN KEY (raw_volume_identity_digest)
                        REFERENCES promotion_raw_volume_identity_reservations(
                            raw_volume_identity_digest
                        ),
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_volume_resolution_memberships (
                    global_resolution_receipt_digest TEXT NOT NULL,
                    raw_observation_slot_digest TEXT NOT NULL,
                    raw_volume_identity_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    family_digest TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    PRIMARY KEY (
                        global_resolution_receipt_digest,
                        raw_observation_slot_digest
                    ),
                    UNIQUE (
                        family_digest,
                        case_id,
                        raw_observation_slot_digest
                    ),
                    FOREIGN KEY (raw_volume_identity_digest)
                        REFERENCES promotion_raw_volume_identity_reservations(
                            raw_volume_identity_digest
                        ),
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_raw_volume_identity_consumptions (
                    raw_volume_identity_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    promotion_evidence_digest TEXT NOT NULL,
                    consumed_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        ),
                    FOREIGN KEY (promotion_evidence_digest)
                        REFERENCES neural_prior_promotions(
                            promotion_evidence_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_raw_observation_reservations (
                    raw_observation_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    global_receipt_digest TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_raw_observation_consumptions (
                    raw_observation_digest TEXT PRIMARY KEY,
                    family_digest TEXT NOT NULL,
                    promotion_evidence_digest TEXT NOT NULL,
                    consumed_at TEXT NOT NULL,
                    FOREIGN KEY (family_digest)
                        REFERENCES neural_prior_promotion_experiment_families(
                            family_digest
                        ),
                    FOREIGN KEY (promotion_evidence_digest)
                        REFERENCES neural_prior_promotions(
                            promotion_evidence_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_resolved_source_coverage_artifacts (
                    artifact_digest TEXT PRIMARY KEY,
                    operational_domain_artifact_digest TEXT NOT NULL UNIQUE,
                    issuance_domain_plan_digest TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (issuance_domain_plan_digest, case_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_analysis_input_provenance (
                    artifact_digest TEXT PRIMARY KEY,
                    holdout_plan_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL,
                    global_resolution_receipt_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    arrays_sha256 TEXT NOT NULL,
                    metadata_sha256 TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    raw_ingestor_trust_store_digest TEXT NOT NULL,
                    raw_trust_validated_at TEXT,
                    committed_at TEXT NOT NULL,
                    usable INTEGER NOT NULL DEFAULT 0 CHECK (usable IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'prepared' CHECK (
                        status IN ('prepared', 'active', 'expired')
                    ),
                    payload_committed_at TEXT,
                    preparation_receipt_json TEXT,
                    preparation_receipt_digest TEXT,
                    activated_at TEXT,
                    expired_at TEXT,
                    UNIQUE (holdout_plan_digest, case_id),
                    FOREIGN KEY (holdout_plan_digest)
                        REFERENCES neural_prior_holdout_plans(plan_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_raw_resolution_history (
                    slot_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL CHECK(sequence_number > 0),
                    entry_digest TEXT NOT NULL UNIQUE,
                    previous_entry_digest TEXT NOT NULL,
                    provenance_plan_digest TEXT NOT NULL,
                    resolution_identity_digest TEXT NOT NULL,
                    resolution_kind TEXT NOT NULL CHECK(
                        resolution_kind IN ('resolved', 'missing')
                    ),
                    transition TEXT NOT NULL CHECK(
                        transition IN (
                            'original', 'reuse', 'correction',
                            'supersession', 'cancellation'
                        )
                    ),
                    entry_json TEXT NOT NULL,
                    raw_resolution_receipt_digest TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (slot_digest, sequence_number)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_raw_resolution_legacy_anchors (
                    slot_digest TEXT PRIMARY KEY,
                    anchor_digest TEXT NOT NULL UNIQUE,
                    provenance_artifact_digest TEXT NOT NULL,
                    raw_resolution_receipt_digest TEXT NOT NULL,
                    resolution_identity_digest TEXT NOT NULL,
                    resolution_kind TEXT NOT NULL CHECK(
                        resolution_kind IN ('resolved', 'missing')
                    ),
                    anchored_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_analysis_input_provenance_plans (
                    plan_digest TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL UNIQUE,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_input_provenance_commits (
                    artifact_digest TEXT PRIMARY KEY,
                    provenance_kind TEXT NOT NULL CHECK(
                        provenance_kind IN ('holdout', 'operational')
                    ),
                    provenance_plan_digest TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    raw_resolution_receipt_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    arrays_sha256 TEXT NOT NULL,
                    metadata_sha256 TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    raw_ingestor_trust_store_digest TEXT NOT NULL,
                    raw_trust_validated_at TEXT,
                    committed_at TEXT NOT NULL,
                    usable INTEGER NOT NULL DEFAULT 0 CHECK (usable IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'prepared' CHECK (
                        status IN ('prepared', 'active', 'expired')
                    ),
                    payload_committed_at TEXT,
                    preparation_receipt_json TEXT,
                    preparation_receipt_digest TEXT,
                    activated_at TEXT,
                    expired_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    operational_analysis_input_provenance_plans_no_update
                BEFORE UPDATE ON operational_analysis_input_provenance_plans
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational provenance plans are immutable'
                    );
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    operational_analysis_input_provenance_plans_no_delete
                BEFORE DELETE ON operational_analysis_input_provenance_plans
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational provenance plans are immutable'
                    );
                END
                """
            )
            provenance_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(neural_prior_analysis_input_provenance)"
                ).fetchall()
            }
            if "raw_ingestor_trust_store_digest" not in provenance_columns:
                connection.execute(
                    "ALTER TABLE neural_prior_analysis_input_provenance "
                    "ADD COLUMN raw_ingestor_trust_store_digest TEXT"
                )
            if "raw_trust_validated_at" not in provenance_columns:
                connection.execute(
                    "ALTER TABLE neural_prior_analysis_input_provenance "
                    "ADD COLUMN raw_trust_validated_at TEXT"
                )
            for name, declaration in (
                ("status", "TEXT NOT NULL DEFAULT 'prepared'"),
                ("payload_committed_at", "TEXT"),
                ("preparation_receipt_json", "TEXT"),
                ("preparation_receipt_digest", "TEXT"),
                ("activated_at", "TEXT"),
                ("expired_at", "TEXT"),
            ):
                if name not in provenance_columns:
                    connection.execute(
                        "ALTER TABLE neural_prior_analysis_input_provenance "
                        f"ADD COLUMN {name} {declaration}"
                    )
            commit_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(analysis_input_provenance_commits)"
                ).fetchall()
            }
            for name, declaration in (
                ("status", "TEXT NOT NULL DEFAULT 'prepared'"),
                ("payload_committed_at", "TEXT"),
                ("preparation_receipt_json", "TEXT"),
                ("preparation_receipt_digest", "TEXT"),
                ("activated_at", "TEXT"),
                ("expired_at", "TEXT"),
            ):
                if name not in commit_columns:
                    connection.execute(
                        "ALTER TABLE analysis_input_provenance_commits "
                        f"ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                "DROP TRIGGER IF EXISTS "
                "operational_raw_resolution_legacy_anchors_insert"
            )
            connection.execute(
                """
                CREATE TRIGGER operational_raw_resolution_legacy_anchors_insert
                BEFORE INSERT ON operational_raw_resolution_legacy_anchors
                WHEN NOT EXISTS (
                    SELECT 1 FROM analysis_input_provenance_commits AS p
                    WHERE p.artifact_digest = NEW.provenance_artifact_digest
                      AND p.provenance_kind = 'operational'
                      AND p.raw_resolution_receipt_digest =
                          NEW.raw_resolution_receipt_digest
                      AND p.payload_committed_at = NEW.anchored_at
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'legacy raw-resolution anchor requires provenance'
                    );
                END
                """
            )
            connection.execute(
                "DROP TRIGGER IF EXISTS analysis_input_provenance_commits_update"
            )
            for table in (
                "neural_prior_analysis_input_provenance",
                "analysis_input_provenance_commits",
            ):
                immutable_identity = (
                    "OLD.holdout_plan_digest = NEW.holdout_plan_digest "
                    "AND OLD.case_id = NEW.case_id "
                    "AND OLD.input_plan_digest = NEW.input_plan_digest "
                    "AND OLD.global_resolution_receipt_digest = "
                    "NEW.global_resolution_receipt_digest"
                    if table == "neural_prior_analysis_input_provenance"
                    else
                    "OLD.provenance_kind = NEW.provenance_kind "
                    "AND OLD.provenance_plan_digest = "
                    "NEW.provenance_plan_digest "
                    "AND OLD.case_id = NEW.case_id "
                    "AND OLD.input_plan_digest = NEW.input_plan_digest "
                    "AND OLD.raw_resolution_receipt_digest = "
                    "NEW.raw_resolution_receipt_digest"
                )
                connection.execute(
                    f"DROP TRIGGER IF EXISTS {table}_prepared_insert"
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER {table}_prepared_insert
                    BEFORE INSERT ON {table}
                    WHEN NOT (
                        NEW.status = 'prepared'
                        AND NEW.usable = 0
                        AND NEW.raw_trust_validated_at IS NULL
                        AND NEW.activated_at IS NULL
                        AND NEW.expired_at IS NULL
                        AND NEW.payload_committed_at IS NEW.committed_at
                        AND (
                            (
                                NEW.preparation_receipt_json IS NULL
                                AND NEW.preparation_receipt_digest IS NULL
                            )
                            OR (
                                NEW.preparation_receipt_json IS NOT NULL
                                AND NEW.preparation_receipt_digest IS NOT NULL
                            )
                        )
                        AND julianday(NEW.committed_at) IS NOT NULL
                        AND (
                            substr(NEW.committed_at, -1) = 'Z'
                            OR substr(NEW.committed_at, -6) = '+00:00'
                        )
                        AND length(NEW.raw_ingestor_trust_store_digest) = 64
                        AND NEW.raw_ingestor_trust_store_digest NOT GLOB
                            '*[^0-9a-f]*'
                    )
                    BEGIN
                        SELECT RAISE(
                            ABORT,
                            'analysis provenance must be inserted prepared'
                        );
                    END
                    """
                )
                connection.execute(
                    f"DROP TRIGGER IF EXISTS {table}_state_update"
                )
                connection.execute(
                    f"UPDATE {table} SET payload_committed_at = committed_at "
                    "WHERE payload_committed_at IS NULL"
                )
                connection.execute(
                    f"UPDATE {table} SET status = CASE "
                    "WHEN usable = 1 THEN 'active' "
                    "WHEN raw_trust_validated_at IS NULL THEN 'prepared' "
                    "ELSE 'expired' END, "
                    "activated_at = CASE WHEN usable = 1 THEN "
                    "raw_trust_validated_at ELSE activated_at END, "
                    "expired_at = CASE WHEN usable = 0 AND "
                    "raw_trust_validated_at IS NOT NULL THEN committed_at "
                    "ELSE expired_at END"
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER {table}_state_update
                    BEFORE UPDATE ON {table}
                    WHEN NOT (
                        OLD.artifact_digest = NEW.artifact_digest
                        AND {immutable_identity}
                        AND OLD.payload_json = NEW.payload_json
                        AND OLD.arrays_sha256 = NEW.arrays_sha256
                        AND OLD.metadata_sha256 = NEW.metadata_sha256
                        AND OLD.path = NEW.path
                        AND OLD.raw_ingestor_trust_store_digest =
                            NEW.raw_ingestor_trust_store_digest
                        AND OLD.committed_at = NEW.committed_at
                        AND OLD.payload_committed_at = NEW.payload_committed_at
                        AND (
                            (
                                OLD.status = 'prepared'
                                AND NEW.status = 'prepared'
                                AND OLD.usable = 0 AND NEW.usable = 0
                                AND OLD.raw_trust_validated_at
                                    IS NEW.raw_trust_validated_at
                                AND OLD.activated_at IS NEW.activated_at
                                AND OLD.expired_at IS NEW.expired_at
                                AND OLD.preparation_receipt_json IS NULL
                                AND OLD.preparation_receipt_digest IS NULL
                                AND NEW.preparation_receipt_json IS NOT NULL
                                AND NEW.preparation_receipt_digest IS NOT NULL
                            )
                            OR (
                                OLD.status = 'prepared'
                                AND NEW.status = 'active'
                                AND OLD.usable = 0 AND NEW.usable = 1
                                AND OLD.activated_at IS NULL
                                AND NEW.activated_at IS NOT NULL
                                AND NEW.raw_trust_validated_at
                                    IS NEW.activated_at
                                AND julianday(NEW.activated_at) IS NOT NULL
                                AND (
                                    substr(NEW.activated_at, -1) = 'Z'
                                    OR substr(NEW.activated_at, -6) = '+00:00'
                                )
                                AND julianday(NEW.activated_at) >=
                                    julianday(OLD.payload_committed_at)
                                AND OLD.preparation_receipt_json
                                    IS NEW.preparation_receipt_json
                                AND OLD.preparation_receipt_digest
                                    IS NEW.preparation_receipt_digest
                                AND NEW.preparation_receipt_json IS NOT NULL
                                AND NEW.preparation_receipt_digest IS NOT NULL
                                AND NEW.expired_at IS NULL
                            )
                            OR (
                                OLD.status IN ('prepared', 'active')
                                AND NEW.status = 'expired'
                                AND NEW.usable = 0
                                AND NEW.activated_at IS OLD.activated_at
                                AND NEW.raw_trust_validated_at
                                    IS OLD.raw_trust_validated_at
                                AND OLD.preparation_receipt_json
                                    IS NEW.preparation_receipt_json
                                AND OLD.preparation_receipt_digest
                                    IS NEW.preparation_receipt_digest
                                AND OLD.expired_at IS NULL
                                AND NEW.expired_at IS NOT NULL
                                AND julianday(NEW.expired_at) IS NOT NULL
                                AND (
                                    substr(NEW.expired_at, -1) = 'Z'
                                    OR substr(NEW.expired_at, -6) = '+00:00'
                                )
                                AND julianday(NEW.expired_at) >=
                                    julianday(OLD.payload_committed_at)
                            )
                        )
                    )
                    BEGIN
                        SELECT RAISE(
                            ABORT,
                            'analysis provenance state transition is invalid'
                        );
                    END
                    """
                )
                connection.execute(
                    f"DROP TRIGGER IF EXISTS {table}_no_delete"
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(
                            ABORT,
                            'analysis provenance rows are immutable'
                        );
                    END
                    """
                )
            if previous_schema_version < 38:
                legacy_rows = connection.execute(
                    "SELECT artifact_digest,raw_resolution_receipt_digest,path,"
                    "payload_committed_at,arrays_sha256,metadata_sha256,"
                    "provenance_plan_digest FROM "
                    "analysis_input_provenance_commits "
                    "WHERE provenance_kind = 'operational'"
                ).fetchall()
                for legacy_row in legacy_rows:
                    provenance_path = (
                        self._validate_analysis_input_provenance_directory(
                            artifact_digest=str(legacy_row[0]),
                            path_text=str(legacy_row[2]),
                            arrays_sha256=str(legacy_row[4]),
                            metadata_sha256=str(legacy_row[5]),
                        )
                    )
                    metadata_path = provenance_path / "provenance.json"
                    try:
                        metadata_text = metadata_path.read_text("utf-8")
                        legacy_metadata = json.loads(metadata_text)
                        if (
                            not isinstance(legacy_metadata, dict)
                            or metadata_text != _json_text(legacy_metadata)
                        ):
                            raise TypeError
                        raw_resolution_payload = dict(
                            legacy_metadata["raw_resolution"]
                        )
                        raw_observations = legacy_metadata[
                            "resolved_raw_observations"
                        ]
                        if (
                            not isinstance(raw_observations, list)
                            or any(
                                not isinstance(item, dict)
                                for item in raw_observations
                            )
                        ):
                            raise TypeError
                        raw_resolution_text = _json_text(
                            raw_resolution_payload
                        )
                        if raw_resolution_payload.get("contract") == (
                            "operational-raw-volume-resolution-receipt-v2"
                        ):
                            current_resolution = (
                                _operational_raw_volume_resolution_receipt_from_json(
                                    raw_resolution_text,
                                    expected_digest=str(legacy_row[1]),
                                )
                            )
                            bindings = current_resolution.slot_identity_bindings
                            resolution_plan_digest = (
                                current_resolution.provenance_plan_digest
                            )
                            resolution_input_plan_digest = (
                                current_resolution.input_plan_digest
                            )
                        else:
                            stored_receipt_digest = raw_resolution_payload.pop(
                                "receipt_digest"
                            )
                            if set(raw_resolution_payload) != {
                                "contract",
                                "provenance_plan_digest",
                                "input_plan_digest",
                                "slot_identity_bindings",
                                "resolved_at",
                            } or raw_resolution_payload.get("contract") != (
                                "operational-raw-volume-resolution-receipt-v1"
                            ):
                                raise ValueError
                            raw_bindings = raw_resolution_payload[
                                "slot_identity_bindings"
                            ]
                            if (
                                not isinstance(raw_bindings, list)
                                or not raw_bindings
                                or any(
                                    not isinstance(item, list)
                                    or len(item) != 2
                                    for item in raw_bindings
                                )
                            ):
                                raise TypeError
                            bindings = tuple(
                                sorted(
                                    (str(item[0]), str(item[1]))
                                    for item in raw_bindings
                                )
                            )
                            if (
                                len({item[0] for item in bindings})
                                != len(bindings)
                                or [list(item) for item in bindings]
                                != raw_bindings
                            ):
                                raise ValueError
                            for slot_digest, identity_digest in bindings:
                                if (
                                    re.fullmatch(r"[0-9a-f]{64}", slot_digest)
                                    is None
                                    or re.fullmatch(
                                        r"[0-9a-f]{64}", identity_digest
                                    )
                                    is None
                                ):
                                    raise ValueError
                            raw_resolution_payload["resolved_at"] = (
                                _canonical_utc_datetime(
                                    str(raw_resolution_payload["resolved_at"]),
                                    "resolved_at",
                                )
                                .isoformat()
                                .replace("+00:00", "Z")
                            )
                            calculated_receipt_digest = _json_digest(
                                raw_resolution_payload
                            )
                            if (
                                stored_receipt_digest
                                != calculated_receipt_digest
                                or calculated_receipt_digest
                                != str(legacy_row[1])
                            ):
                                raise ValueError
                            resolution_plan_digest = str(
                                raw_resolution_payload[
                                    "provenance_plan_digest"
                                ]
                            )
                            resolution_input_plan_digest = str(
                                raw_resolution_payload["input_plan_digest"]
                            )
                        plan_row = connection.execute(
                            "SELECT payload_json FROM "
                            "operational_analysis_input_provenance_plans "
                            "WHERE plan_digest = ?",
                            (str(legacy_row[6]),),
                        ).fetchone()
                        if plan_row is None:
                            raise ValueError
                        plan_payload = json.loads(str(plan_row[0]))
                        if (
                            not isinstance(plan_payload, dict)
                            or str(plan_row[0]) != _json_text(plan_payload)
                            or legacy_metadata.get("provenance_plan")
                            != plan_payload
                        ):
                            raise ValueError
                        plan_values = dict(plan_payload)
                        stored_plan_digest = plan_values.pop("plan_digest")
                        if (
                            stored_plan_digest != str(legacy_row[6])
                            or _json_digest(plan_values)
                            != str(legacy_row[6])
                            or resolution_plan_digest != str(legacy_row[6])
                        ):
                            raise ValueError
                        input_plan_payload = plan_payload["input_plan"]
                        if (
                            not isinstance(input_plan_payload, dict)
                            or resolution_input_plan_digest
                            != input_plan_payload.get("plan_digest")
                        ):
                            raise ValueError
                        derivation_payload = legacy_metadata[
                            "analysis_input_derivation"
                        ]
                        if not isinstance(derivation_payload, dict):
                            raise TypeError
                        derivation_values = dict(derivation_payload)
                        stored_derivation_digest = derivation_values.pop(
                            "artifact_digest"
                        )
                        if stored_derivation_digest != str(legacy_row[0]):
                            raise ValueError
                        derivation = _analysis_input_derivation_from_json(
                            _json_text(derivation_values),
                            expected_digest=str(legacy_row[0]),
                        )
                        if (
                            derivation.processor_id
                            != plan_payload["analysis_processor_id"]
                            or derivation.processor_public_key_hex
                            != plan_payload[
                                "analysis_processor_public_key_hex"
                            ]
                        ):
                            raise ValueError
                        kinds = {
                            str(item["slot_plan_digest"]): (
                                "missing"
                                if item.get("contract")
                                == "missing-raw-observation-receipt-v1"
                                else "resolved"
                            )
                            for item in raw_observations
                        }
                        observation_identities = {
                            str(item["slot_plan_digest"]): str(
                                item["resolution_identity_digest"]
                                if item.get("contract")
                                == "missing-raw-observation-receipt-v1"
                                else item["raw_volume_identity"][
                                    "identity_digest"
                                ]
                            )
                            for item in raw_observations
                        }
                        if (
                            set(kinds) != {item[0] for item in bindings}
                            or any(
                                observation_identities[slot_digest]
                                != identity_digest
                                for slot_digest, identity_digest in bindings
                            )
                        ):
                            raise ValueError
                    except (
                        KeyError,
                        OSError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        raise ValueError(
                            "legacy operational provenance cannot be anchored"
                        ) from error
                    for slot_digest, identity_digest in bindings:
                        anchor_payload = {
                            "contract": (
                                "operational-raw-resolution-legacy-anchor-v1"
                            ),
                            "slot_digest": slot_digest,
                            "resolution_identity_digest": identity_digest,
                            "resolution_kind": kinds[str(slot_digest)],
                            "provenance_artifact_digest": str(legacy_row[0]),
                            "raw_resolution_receipt_digest": str(legacy_row[1]),
                        }
                        anchor_digest = _json_digest(anchor_payload)
                        retained_anchor = connection.execute(
                            "SELECT anchor_digest,resolution_identity_digest,"
                            "resolution_kind FROM "
                            "operational_raw_resolution_legacy_anchors "
                            "WHERE slot_digest = ?",
                            (slot_digest,),
                        ).fetchone()
                        expected_anchor = (
                            anchor_digest,
                            identity_digest,
                            kinds[str(slot_digest)],
                        )
                        if retained_anchor is not None:
                            if tuple(retained_anchor) != expected_anchor:
                                raise ValueError(
                                    "legacy operational raw slot equivocated"
                                )
                            continue
                        connection.execute(
                            "INSERT INTO "
                            "operational_raw_resolution_legacy_anchors "
                            "(slot_digest,anchor_digest,"
                            "provenance_artifact_digest,"
                            "raw_resolution_receipt_digest,"
                            "resolution_identity_digest,resolution_kind,"
                            "anchored_at) VALUES (?,?,?,?,?,?,?)",
                            (
                                slot_digest,
                                anchor_digest,
                                legacy_row[0],
                                legacy_row[1],
                                identity_digest,
                                kinds[str(slot_digest)],
                                legacy_row[3],
                            ),
                        )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_scoring_input_artifacts (
                    artifact_digest TEXT PRIMARY KEY,
                    holdout_plan_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_process_start_receipts_v2 (
                    receipt_digest TEXT PRIMARY KEY,
                    catalog_plan_digest TEXT NOT NULL,
                    catalog_result_digest TEXT NOT NULL,
                    process_kind TEXT NOT NULL,
                    scheduler_id TEXT NOT NULL,
                    scheduler_sequence_number INTEGER NOT NULL,
                    job_id TEXT NOT NULL UNIQUE,
                    launch_nonce TEXT NOT NULL UNIQUE,
                    previous_receipt_digest TEXT,
                    receipt_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (scheduler_id, scheduler_sequence_number),
                    UNIQUE (catalog_plan_digest, process_kind)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_process_log_artifacts (
                    artifact_digest TEXT PRIMARY KEY,
                    start_receipt_digest TEXT NOT NULL UNIQUE,
                    process_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (start_receipt_digest)
                        REFERENCES trusted_process_start_receipts_v2(receipt_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_holdout_scoring_artifacts (
                    artifact_digest TEXT PRIMARY KEY,
                    holdout_plan_digest TEXT NOT NULL,
                    candidate_manifest_digest TEXT NOT NULL,
                    scoring_start_receipt_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (scoring_start_receipt_digest)
                        REFERENCES trusted_process_start_receipts_v2(receipt_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_scoring_replay_bundles (
                    bundle_digest TEXT PRIMARY KEY,
                    scoring_input_artifact_digest TEXT NOT NULL UNIQUE,
                    manifest_json TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (scoring_input_artifact_digest)
                        REFERENCES neural_prior_holdout_scoring_input_artifacts(
                            artifact_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_process_completion_receipts (
                    receipt_digest TEXT PRIMARY KEY,
                    start_receipt_digest TEXT NOT NULL UNIQUE,
                    process_kind TEXT NOT NULL,
                    output_artifact_digest TEXT NOT NULL,
                    process_log_digest TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (start_receipt_digest)
                        REFERENCES trusted_process_start_receipts_v2(receipt_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_trust_artifact_activations (
                    artifact_kind TEXT NOT NULL CHECK (
                        artifact_kind IN (
                            'scoring_replay_bundle',
                            'scoring_completion',
                            'promotion_evidence',
                            'promotion_deployment_certificate'
                        )
                    ),
                    artifact_digest TEXT NOT NULL CHECK (
                        length(artifact_digest) = 64
                        AND artifact_digest NOT GLOB '*[^0-9a-f]*'
                    ),
                    raw_ingestor_trust_store_digest TEXT NOT NULL CHECK (
                        length(raw_ingestor_trust_store_digest) = 64
                        AND raw_ingestor_trust_store_digest
                            NOT GLOB '*[^0-9a-f]*'
                    ),
                    usable INTEGER NOT NULL DEFAULT 0 CHECK (usable IN (0, 1)),
                    prepared_at TEXT NOT NULL,
                    activated_at TEXT,
                    expired_at TEXT,
                    PRIMARY KEY (artifact_kind, artifact_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotions (
                    promotion_evidence_digest TEXT PRIMARY KEY,
                    candidate_prior_digest TEXT NOT NULL UNIQUE,
                    parent_prior_digest TEXT NOT NULL,
                    candidate_manifest_digest TEXT NOT NULL,
                    candidate_manifest_json TEXT NOT NULL,
                    holdout_plan_digest TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    trust_store_digest TEXT NOT NULL,
                    evaluation_digests_json TEXT NOT NULL,
                    evaluation_payloads_json TEXT NOT NULL,
                    intervention_digests_json TEXT NOT NULL,
                    realized_intervention_count INTEGER NOT NULL,
                    material_outcome_count INTEGER NOT NULL,
                    distinct_case_count INTEGER NOT NULL,
                    distinct_storm_count INTEGER NOT NULL,
                    distinct_day_count INTEGER NOT NULL,
                    distinct_radar_count INTEGER NOT NULL,
                    distinct_regime_count INTEGER NOT NULL,
                    distinct_range_regime_count INTEGER NOT NULL,
                    beneficial_fraction REAL NOT NULL,
                    beneficial_fraction_lower_bound REAL NOT NULL,
                    harmful_fraction REAL NOT NULL,
                    harmful_fraction_upper_bound REAL NOT NULL,
                    mean_normalized_improvement REAL NOT NULL,
                    mean_improvement_lower_bound REAL NOT NULL,
                    maximum_normalized_degradation REAL NOT NULL,
                    prior_gaussian_nll_increase_upper_bound REAL NOT NULL,
                    prior_support_brier_increase_upper_bound REAL NOT NULL,
                    prior_underdispersion_increase_upper_bound REAL NOT NULL,
                    prior_conditional_underdispersion_increase_upper_bound REAL NOT NULL,
                    prior_echo_support_miss_increase_upper_bound REAL NOT NULL,
                    prior_echo_object_miss_increase_upper_bound REAL NOT NULL,
                    prior_echo_intensity_nll_increase_upper_bound REAL NOT NULL,
                    prior_clear_sky_false_echo_increase_upper_bound REAL NOT NULL,
                    prior_echo_component_status TEXT NOT NULL,
                    prior_clear_sky_component_status TEXT NOT NULL,
                    prior_echo_case_count INTEGER NOT NULL,
                    prior_clear_sky_case_count INTEGER NOT NULL,
                    prior_echo_cluster_count INTEGER NOT NULL,
                    prior_clear_sky_cluster_count INTEGER NOT NULL,
                    simultaneous_inference_test_count INTEGER NOT NULL,
                    eligible INTEGER NOT NULL,
                    rejection_reasons_json TEXT NOT NULL,
                    evidence_contract TEXT NOT NULL,
                    evidence_payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotion_deployment_certificates (
                    certificate_digest TEXT PRIMARY KEY,
                    promotion_evidence_digest TEXT NOT NULL UNIQUE,
                    previous_certificate_digest TEXT UNIQUE,
                    ledger_chain_head_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_evidence_digest)
                        REFERENCES neural_prior_promotions(
                            promotion_evidence_digest
                        ),
                    FOREIGN KEY (previous_certificate_digest)
                        REFERENCES neural_prior_promotion_deployment_certificates(
                            certificate_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS neural_prior_promotion_deployment_certificates_v3 (
                    certificate_digest TEXT PRIMARY KEY,
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL UNIQUE CHECK(sequence_number > 0),
                    promotion_evidence_digest TEXT NOT NULL UNIQUE,
                    previous_certificate_digest TEXT NOT NULL UNIQUE,
                    ledger_chain_head_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_evidence_digest)
                        REFERENCES neural_prior_promotions(
                            promotion_evidence_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_certificate_chain_head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL CHECK(sequence_number >= 0),
                    certificate_digest TEXT NOT NULL,
                    ledger_chain_head_digest TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            if connection.execute(
                "SELECT 1 FROM deployment_certificate_chain_head WHERE singleton = 1"
            ).fetchone() is None:
                now = datetime.now(timezone.utc).isoformat()
                ledger_instance_digest = _json_digest(
                    {
                        "contract": "advar-deployment-certificate-ledger-instance-v1",
                        "nonce": os.urandom(32).hex(),
                        "created_at": now,
                    }
                )
                connection.execute(
                    "INSERT INTO deployment_certificate_chain_head "
                    "(singleton,ledger_instance_digest,sequence_number,"
                    "certificate_digest,ledger_chain_head_digest,updated_at) "
                    "VALUES (1,?,?,?,?,?)",
                    (
                        ledger_instance_digest,
                        0,
                        _PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
                        _PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
                        now,
                    ),
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_bundle_release_approvals (
                    approval_digest TEXT PRIMARY KEY,
                    deployment_bundle_digest TEXT NOT NULL,
                    bundle_manifest_digest TEXT NOT NULL,
                    authority_id TEXT NOT NULL,
                    approval_json TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_runtime_activations (
                    receipt_digest TEXT PRIMARY KEY,
                    release_approval_digest TEXT NOT NULL,
                    deployment_bundle_digest TEXT NOT NULL,
                    runtime_tree_digest TEXT NOT NULL,
                    interpreter_closure_digest TEXT NOT NULL,
                    deployment_instance_digest TEXT NOT NULL,
                    host_identity_digest TEXT NOT NULL,
                    activation_sequence_number INTEGER NOT NULL CHECK (
                        activation_sequence_number > 0
                    ),
                    previous_activation_receipt_digest TEXT NOT NULL,
                    rollback_reason_digest TEXT,
                    receipt_json TEXT NOT NULL,
                    activated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        deployment_instance_digest,
                        activation_sequence_number
                    ),
                    FOREIGN KEY (release_approval_digest)
                        REFERENCES deployment_bundle_release_approvals(
                            approval_digest
                        )
                )
                """
            )
            runtime_activation_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(deployment_runtime_activations)"
                ).fetchall()
            }
            for column_name, column_type in (
                ("release_approval_digest", "TEXT"),
                ("previous_activation_receipt_digest", "TEXT"),
                ("rollback_reason_digest", "TEXT"),
            ):
                if column_name not in runtime_activation_columns:
                    connection.execute(
                        "ALTER TABLE deployment_runtime_activations ADD COLUMN "
                        f"{column_name} {column_type}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_runtime_activation_heads (
                    deployment_instance_digest TEXT PRIMARY KEY,
                    host_identity_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL CHECK(sequence_number > 0),
                    receipt_digest TEXT NOT NULL UNIQUE,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (receipt_digest)
                        REFERENCES deployment_runtime_activations(receipt_digest)
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO deployment_runtime_activation_heads (
                    deployment_instance_digest,host_identity_digest,
                    sequence_number,receipt_digest,updated_at
                )
                SELECT a.deployment_instance_digest,a.host_identity_digest,
                       a.activation_sequence_number,a.receipt_digest,a.activated_at
                FROM deployment_runtime_activations AS a
                WHERE a.activation_sequence_number = (
                    SELECT MAX(b.activation_sequence_number)
                    FROM deployment_runtime_activations AS b
                    WHERE b.deployment_instance_digest =
                          a.deployment_instance_digest
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_issuance_states (
                    operational_cycle_id TEXT PRIMARY KEY,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    promotion_certificate_digest TEXT NOT NULL,
                    release_approval_digest TEXT NOT NULL,
                    runtime_activation_receipt_digest TEXT NOT NULL,
                    decision_payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'prepared',
                            'decision_recorded',
                            'published',
                            'expired'
                        )
                    ),
                    prepared_at TEXT NOT NULL,
                    decision_row_committed_at TEXT,
                    certificate_digest TEXT UNIQUE,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_certificate_digest)
                        REFERENCES neural_prior_promotion_deployment_certificates_v3(
                            certificate_digest
                        ),
                    FOREIGN KEY (release_approval_digest)
                        REFERENCES deployment_bundle_release_approvals(
                            approval_digest
                        ),
                    FOREIGN KEY (runtime_activation_receipt_digest)
                        REFERENCES deployment_runtime_activations(receipt_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_commits (
                    commit_entry_digest TEXT PRIMARY KEY,
                    committed_chain_root_digest TEXT NOT NULL UNIQUE,
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL UNIQUE CHECK(sequence_number > 0),
                    previous_commit_root_digest TEXT NOT NULL UNIQUE,
                    promotion_certificate_digest TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    decision_payload_json TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_certificate_digest)
                        REFERENCES neural_prior_promotion_deployment_certificates_v3(
                            certificate_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_commit_proofs (
                    receipt_digest TEXT PRIMARY KEY,
                    commit_entry_digest TEXT NOT NULL UNIQUE,
                    receipt_json TEXT NOT NULL,
                    committed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (commit_entry_digest)
                        REFERENCES operational_decision_commits(commit_entry_digest)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_deployment_decisions_v2 (
                    certificate_digest TEXT PRIMARY KEY,
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL UNIQUE CHECK(sequence_number > 0),
                    previous_certificate_digest TEXT NOT NULL UNIQUE,
                    promotion_certificate_digest TEXT NOT NULL,
                    release_approval_digest TEXT NOT NULL,
                    runtime_activation_receipt_digest TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    commit_entry_digest TEXT NOT NULL UNIQUE,
                    receipt_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_certificate_digest)
                        REFERENCES neural_prior_promotion_deployment_certificates_v3(
                            certificate_digest
                        ),
                    FOREIGN KEY (release_approval_digest)
                        REFERENCES deployment_bundle_release_approvals(
                            approval_digest
                        ),
                    FOREIGN KEY (runtime_activation_receipt_digest)
                        REFERENCES deployment_runtime_activations(receipt_digest),
                    FOREIGN KEY (commit_entry_digest)
                        REFERENCES operational_decision_commits(commit_entry_digest),
                    FOREIGN KEY (receipt_digest)
                        REFERENCES operational_decision_commit_proofs(receipt_digest)
                )
                """
            )
            issuance_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(operational_decision_issuance_states)"
                ).fetchall()
            }
            if "runtime_activation_receipt_digest" not in issuance_columns:
                connection.execute(
                    "ALTER TABLE operational_decision_issuance_states ADD COLUMN "
                    "runtime_activation_receipt_digest TEXT"
                )
            if "release_approval_digest" not in issuance_columns:
                connection.execute(
                    "ALTER TABLE operational_decision_issuance_states ADD COLUMN "
                    "release_approval_digest TEXT"
                )
            decision_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(operational_deployment_decisions_v2)"
                ).fetchall()
            }
            if "runtime_activation_receipt_digest" not in decision_columns:
                connection.execute(
                    "ALTER TABLE operational_deployment_decisions_v2 ADD COLUMN "
                    "runtime_activation_receipt_digest TEXT"
                )
            if "release_approval_digest" not in decision_columns:
                connection.execute(
                    "ALTER TABLE operational_deployment_decisions_v2 ADD COLUMN "
                    "release_approval_digest TEXT"
                )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_bundle_release_approvals_immutable_update
                BEFORE UPDATE ON deployment_bundle_release_approvals
                BEGIN
                    SELECT RAISE(ABORT, 'deployment release approvals are immutable');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_bundle_release_approvals_immutable_delete
                BEFORE DELETE ON deployment_bundle_release_approvals
                BEGIN
                    SELECT RAISE(ABORT, 'deployment release approvals are immutable');
                END
                """
            )
            connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activations_append_guard
                BEFORE INSERT ON deployment_runtime_activations
                WHEN NEW.activation_sequence_number != COALESCE((
                        SELECT MAX(activation_sequence_number) + 1
                        FROM deployment_runtime_activations
                        WHERE deployment_instance_digest =
                              NEW.deployment_instance_digest
                    ), 1)
                    OR NEW.previous_activation_receipt_digest != COALESCE((
                        SELECT receipt_digest
                        FROM deployment_runtime_activations
                        WHERE deployment_instance_digest =
                              NEW.deployment_instance_digest
                        ORDER BY activation_sequence_number DESC LIMIT 1
                    ), '{DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST}')
                    OR (
                        EXISTS (
                            SELECT 1 FROM deployment_runtime_activations
                            WHERE deployment_instance_digest =
                                  NEW.deployment_instance_digest
                              AND runtime_tree_digest = NEW.runtime_tree_digest
                        )
                        AND NEW.runtime_tree_digest != COALESCE((
                            SELECT runtime_tree_digest
                            FROM deployment_runtime_activations
                            WHERE deployment_instance_digest =
                                  NEW.deployment_instance_digest
                            ORDER BY activation_sequence_number DESC LIMIT 1
                        ), NEW.runtime_tree_digest)
                        AND NEW.rollback_reason_digest IS NULL
                    )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid deployment runtime activation append');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activations_immutable_update
                BEFORE UPDATE ON deployment_runtime_activations
                BEGIN
                    SELECT RAISE(ABORT, 'deployment runtime activations are immutable');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activations_immutable_delete
                BEFORE DELETE ON deployment_runtime_activations
                BEGIN
                    SELECT RAISE(ABORT, 'deployment runtime activations are immutable');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activation_heads_insert_guard
                BEFORE INSERT ON deployment_runtime_activation_heads
                WHEN EXISTS (
                        SELECT 1 FROM deployment_runtime_activation_heads
                        WHERE deployment_instance_digest =
                              NEW.deployment_instance_digest
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM deployment_runtime_activations
                        WHERE receipt_digest = NEW.receipt_digest
                          AND deployment_instance_digest =
                              NEW.deployment_instance_digest
                          AND host_identity_digest = NEW.host_identity_digest
                          AND activation_sequence_number = NEW.sequence_number
                    )
                    OR NEW.sequence_number != (
                        SELECT MAX(activation_sequence_number)
                        FROM deployment_runtime_activations
                        WHERE deployment_instance_digest =
                              NEW.deployment_instance_digest
                    )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid deployment runtime activation head');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activation_heads_update_guard
                BEFORE UPDATE ON deployment_runtime_activation_heads
                WHEN NEW.deployment_instance_digest != OLD.deployment_instance_digest
                    OR NEW.host_identity_digest != OLD.host_identity_digest
                    OR NEW.sequence_number != OLD.sequence_number + 1
                    OR NOT EXISTS (
                        SELECT 1 FROM deployment_runtime_activations
                        WHERE receipt_digest = NEW.receipt_digest
                          AND deployment_instance_digest =
                              OLD.deployment_instance_digest
                          AND host_identity_digest = OLD.host_identity_digest
                          AND activation_sequence_number = NEW.sequence_number
                          AND previous_activation_receipt_digest =
                              OLD.receipt_digest
                    )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid deployment runtime activation head');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                deployment_runtime_activation_heads_immutable_delete
                BEFORE DELETE ON deployment_runtime_activation_heads
                BEGIN
                    SELECT RAISE(ABORT, 'deployment runtime activation heads are immutable');
                END
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_publications (
                    certificate_digest TEXT PRIMARY KEY,
                    decision_row_committed_at TEXT NOT NULL,
                    publication_payload_committed_at TEXT,
                    activation_committed_at TEXT,
                    usable INTEGER NOT NULL CHECK(usable IN (0, 1)),
                    receipt_digest TEXT UNIQUE,
                    receipt_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (certificate_digest)
                        REFERENCES operational_deployment_decisions_v2(
                            certificate_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_activation_receipts (
                    certificate_digest TEXT PRIMARY KEY,
                    receipt_digest TEXT NOT NULL UNIQUE,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (certificate_digest)
                        REFERENCES operational_deployment_decisions_v2(
                            certificate_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                    operational_decision_commit_authorization_receipts (
                    certificate_digest TEXT PRIMARY KEY,
                    receipt_digest TEXT NOT NULL UNIQUE,
                    receipt_json TEXT NOT NULL,
                    terminal_commit_authorized_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (certificate_digest)
                        REFERENCES operational_deployment_decisions_v2(
                            certificate_digest
                        )
                )
                """
            )
            publication_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(operational_decision_publications)"
                ).fetchall()
            }
            if "activation_committed_at" not in publication_columns:
                connection.execute(
                    "ALTER TABLE operational_decision_publications "
                    "ADD COLUMN activation_committed_at TEXT"
                )
            if "publication_payload_committed_at" not in publication_columns:
                connection.execute(
                    "ALTER TABLE operational_decision_publications "
                    "ADD COLUMN publication_payload_committed_at TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_deployment_decisions (
                    certificate_digest TEXT PRIMARY KEY,
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL UNIQUE CHECK(sequence_number > 0),
                    previous_certificate_digest TEXT NOT NULL UNIQUE,
                    promotion_certificate_digest TEXT NOT NULL,
                    input_plan_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (promotion_certificate_digest)
                        REFERENCES neural_prior_promotion_deployment_certificates_v3(
                            certificate_digest
                        )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operational_decision_chain_head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    ledger_instance_digest TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL CHECK(sequence_number >= 0),
                    certificate_digest TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            if connection.execute(
                "SELECT 1 FROM operational_decision_chain_head WHERE singleton = 1"
            ).fetchone() is None:
                deployment_head = connection.execute(
                    "SELECT ledger_instance_digest,updated_at FROM "
                    "deployment_certificate_chain_head WHERE singleton = 1"
                ).fetchone()
                assert deployment_head is not None
                connection.execute(
                    "INSERT INTO operational_decision_chain_head "
                    "(singleton,ledger_instance_digest,sequence_number,"
                    "certificate_digest,updated_at) VALUES (1,?,?,?,?)",
                    (
                        deployment_head[0],
                        0,
                        OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST,
                        deployment_head[1],
                    ),
                )
            _ensure_variational_learning_approval_schema(connection)
            _ensure_realized_intervention_schema(connection)
            _ensure_prospective_receipt_schema(connection)
            _ensure_prospective_decision_schema(connection)
            _ensure_neural_prior_promotion_schema(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS episodes_contract_time
                ON episodes(contract_hash, issue_time)
                """
            )
            connection.execute(
                f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}"
            )
            for table in (
                "episodes",
                "episode_impacts",
                "variational_learning_approvals",
                "realized_observation_interventions",
                "retrospective_counterfactual_replays",
                "reusable_intervention_policies",
                "prospective_intervention_decisions",
                "realized_intervention_receipts",
                "neural_prior_holdout_plans",
                "neural_prior_event_catalog_results",
                "neural_prior_process_start_receipts",
                "neural_prior_training_event_catalog_results",
                "neural_prior_promotion_decision_rules",
                "neural_prior_promotion_rule_definitions",
                "neural_prior_holdout_plan_rule_bindings",
                "neural_prior_promotion_experiment_families",
                "neural_prior_holdout_plan_experiment_bindings",
                "neural_prior_experiment_family_consumptions",
                "promotion_sampling_unit_reservations",
                "promotion_sampling_unit_consumptions",
                "training_raw_registry_entries",
                "global_sampling_registry_entries",
                "promotion_raw_observation_slot_reservations",
                "promotion_raw_volume_identity_reservations",
                "raw_observation_slot_identity_bindings",
                "raw_volume_resolution_memberships",
                "promotion_raw_volume_identity_consumptions",
                "promotion_raw_observation_reservations",
                "promotion_raw_observation_consumptions",
                "operational_raw_resolution_history",
                "operational_raw_resolution_legacy_anchors",
                "neural_prior_resolved_source_coverage_artifacts",
                "neural_prior_holdout_scoring_input_artifacts",
                "trusted_process_start_receipts_v2",
                "trusted_process_log_artifacts",
                "neural_prior_holdout_scoring_artifacts",
                "neural_prior_scoring_replay_bundles",
                "trusted_process_completion_receipts",
                "neural_prior_promotions",
                "neural_prior_promotion_deployment_certificates_v3",
                "deployment_runtime_activations",
                "operational_decision_commits",
                "operational_decision_commit_proofs",
                "operational_deployment_decisions_v2",
                "operational_deployment_decisions",
            ):
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} rows are immutable');
                    END
                    """
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} rows are immutable');
                    END
                    """
                )
            connection.execute(
                "DROP TRIGGER IF EXISTS raw_trust_artifact_activations_no_insert"
            )
            connection.execute(
                """
                CREATE TRIGGER raw_trust_artifact_activations_no_insert
                BEFORE INSERT ON raw_trust_artifact_activations
                WHEN NEW.artifact_kind NOT IN (
                        'scoring_replay_bundle',
                        'scoring_completion',
                        'promotion_evidence',
                        'promotion_deployment_certificate'
                    )
                    OR length(NEW.artifact_digest) != 64
                    OR NEW.artifact_digest GLOB '*[^0-9a-f]*'
                    OR length(NEW.raw_ingestor_trust_store_digest) != 64
                    OR NEW.raw_ingestor_trust_store_digest GLOB '*[^0-9a-f]*'
                    OR NEW.usable != 0
                    OR NEW.activated_at IS NOT NULL
                    OR NEW.expired_at IS NOT NULL
                    OR julianday(NEW.prepared_at) IS NULL
                    OR NOT (
                        NEW.prepared_at GLOB '*Z'
                        OR NEW.prepared_at GLOB '*+00:00'
                    )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'raw trust artifact must begin prepared and unusable'
                    );
                END
                """
            )
            connection.execute(
                "DROP TRIGGER IF EXISTS raw_trust_artifact_activations_no_update"
            )
            connection.execute(
                """
                CREATE TRIGGER raw_trust_artifact_activations_no_update
                BEFORE UPDATE ON raw_trust_artifact_activations
                WHEN NOT (
                    OLD.artifact_kind = NEW.artifact_kind
                    AND OLD.artifact_digest = NEW.artifact_digest
                    AND OLD.raw_ingestor_trust_store_digest =
                        NEW.raw_ingestor_trust_store_digest
                    AND OLD.prepared_at = NEW.prepared_at
                    AND (
                        (
                            OLD.usable = 0 AND NEW.usable = 1
                            AND OLD.activated_at IS NULL
                            AND NEW.activated_at IS NOT NULL
                            AND julianday(NEW.activated_at) IS NOT NULL
                            AND (
                                NEW.activated_at GLOB '*Z'
                                OR NEW.activated_at GLOB '*+00:00'
                            )
                            AND julianday(NEW.activated_at)
                                >= julianday(OLD.prepared_at)
                            AND OLD.expired_at IS NULL
                            AND NEW.expired_at IS NULL
                        )
                        OR (
                            OLD.usable = 1 AND NEW.usable = 0
                            AND OLD.activated_at = NEW.activated_at
                            AND OLD.expired_at IS NULL
                            AND NEW.expired_at IS NOT NULL
                            AND julianday(NEW.expired_at) IS NOT NULL
                            AND (
                                NEW.expired_at GLOB '*Z'
                                OR NEW.expired_at GLOB '*+00:00'
                            )
                            AND julianday(NEW.expired_at)
                                >= julianday(OLD.activated_at)
                        )
                        OR (
                            OLD.usable = 0 AND NEW.usable = 0
                            AND OLD.activated_at IS NULL
                            AND NEW.activated_at IS NULL
                            AND OLD.expired_at IS NULL
                            AND NEW.expired_at IS NOT NULL
                            AND julianday(NEW.expired_at) IS NOT NULL
                            AND (
                                NEW.expired_at GLOB '*Z'
                                OR NEW.expired_at GLOB '*+00:00'
                            )
                            AND julianday(NEW.expired_at)
                                >= julianday(OLD.prepared_at)
                        )
                    )
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'raw trust artifact activation transition is invalid'
                    );
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    raw_trust_artifact_activations_no_delete
                BEFORE DELETE ON raw_trust_artifact_activations
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'raw trust artifact activations are immutable'
                    );
                END
                """
            )
            connection.execute(
                "DROP TRIGGER IF EXISTS operational_decision_publications_no_update"
            )
            connection.execute(
                """
                CREATE TRIGGER operational_decision_publications_no_update
                BEFORE UPDATE ON operational_decision_publications
                WHEN NOT (
                    (
                        OLD.usable = 0 AND NEW.usable = 0
                        AND OLD.activation_committed_at IS NULL
                        AND NEW.activation_committed_at IS NULL
                        AND OLD.publication_payload_committed_at IS NULL
                        AND NEW.publication_payload_committed_at IS NULL
                        AND OLD.receipt_digest IS NULL
                        AND NEW.receipt_digest IS NOT NULL
                        AND OLD.receipt_json IS NULL
                        AND NEW.receipt_json IS NOT NULL
                    )
                    OR (
                        OLD.usable = 0 AND NEW.usable = 0
                        AND OLD.activation_committed_at IS NULL
                        AND NEW.activation_committed_at IS NULL
                        AND OLD.publication_payload_committed_at IS NULL
                        AND NEW.publication_payload_committed_at IS NOT NULL
                        AND NEW.receipt_digest IS OLD.receipt_digest
                        AND NEW.receipt_json IS OLD.receipt_json
                    )
                    OR (
                        OLD.usable = 0 AND NEW.usable = 0
                        AND OLD.activation_committed_at IS NULL
                        AND NEW.activation_committed_at IS NOT NULL
                        AND NEW.publication_payload_committed_at
                            IS OLD.publication_payload_committed_at
                        AND OLD.publication_payload_committed_at IS NOT NULL
                        AND NEW.receipt_digest IS OLD.receipt_digest
                        AND OLD.receipt_digest IS NOT NULL
                        AND NEW.receipt_json IS OLD.receipt_json
                        AND OLD.receipt_json IS NOT NULL
                        AND (
                            SELECT status FROM
                                operational_decision_issuance_states
                            WHERE certificate_digest = NEW.certificate_digest
                        ) = 'decision_recorded'
                        AND EXISTS (
                            SELECT 1 FROM
                                operational_decision_activation_receipts
                            WHERE certificate_digest = NEW.certificate_digest
                        )
                    )
                    OR (
                        OLD.usable = 0 AND NEW.usable = 1
                        AND OLD.activation_committed_at IS NOT NULL
                        AND NEW.activation_committed_at
                            IS OLD.activation_committed_at
                        AND NEW.publication_payload_committed_at
                            IS OLD.publication_payload_committed_at
                        AND OLD.publication_payload_committed_at IS NOT NULL
                        AND NEW.receipt_digest IS OLD.receipt_digest
                        AND OLD.receipt_digest IS NOT NULL
                        AND NEW.receipt_json IS OLD.receipt_json
                        AND OLD.receipt_json IS NOT NULL
                        AND (
                            SELECT status FROM
                                operational_decision_issuance_states
                            WHERE certificate_digest = NEW.certificate_digest
                        ) = 'decision_recorded'
                        AND EXISTS (
                            SELECT 1 FROM
                                operational_decision_activation_receipts
                            WHERE certificate_digest = NEW.certificate_digest
                        )
                        AND EXISTS (
                            SELECT 1 FROM
                                operational_decision_commit_authorization_receipts
                            WHERE certificate_digest = NEW.certificate_digest
                        )
                    )
                    OR (
                        OLD.usable = 1 AND NEW.usable = 0
                        AND NEW.activation_committed_at
                            IS OLD.activation_committed_at
                        AND NEW.publication_payload_committed_at
                            IS OLD.publication_payload_committed_at
                        AND NEW.receipt_digest IS OLD.receipt_digest
                        AND NEW.receipt_json IS OLD.receipt_json
                    )
                )
                OR NEW.certificate_digest != OLD.certificate_digest
                OR NEW.decision_row_committed_at
                    != OLD.decision_row_committed_at
                OR NEW.created_at != OLD.created_at
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational decision publications may only fail closed'
                    );
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    operational_decision_publications_no_delete
                BEFORE DELETE ON operational_decision_publications
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational_decision_publications rows are immutable'
                    );
                END
                """
            )
            connection.execute(
                "DROP TRIGGER IF EXISTS "
                "operational_decision_activation_receipts_no_update"
            )
            connection.execute(
                """
                CREATE TRIGGER operational_decision_activation_receipts_no_update
                BEFORE UPDATE ON operational_decision_activation_receipts
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational activation receipts are immutable'
                    );
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    operational_decision_activation_receipts_no_delete
                BEFORE DELETE ON operational_decision_activation_receipts
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'operational activation receipts are immutable'
                    );
                END
                """
            )
            for action in ("UPDATE", "DELETE"):
                trigger = (
                    "operational_decision_commit_authorizations_no_"
                    f"{action.lower()}"
                )
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                connection.execute(
                    f"""
                    CREATE TRIGGER {trigger}
                    BEFORE {action}
                    ON operational_decision_commit_authorization_receipts
                    BEGIN
                        SELECT RAISE(
                            ABORT,
                            'operational commit authorizations are immutable'
                        );
                    END
                    """
                )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episode_impacts_no_late_insert
                BEFORE INSERT ON episode_impacts
                WHEN EXISTS (
                    SELECT 1 FROM episodes
                    WHERE episode_id = NEW.episode_id
                )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'episode_impacts rows are immutable'
                    );
                END
                """
            )

    @staticmethod
    def _prepare_raw_trust_artifact_activation(
        connection: sqlite3.Connection,
        *,
        artifact_kind: str,
        artifact_digest: str,
        raw_ingestor_trust_store_digest: str,
    ) -> None:
        if (
            artifact_kind not in _RAW_TRUST_ACTIVATION_KINDS
            or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", raw_ingestor_trust_store_digest
            )
            is None
        ):
            raise ValueError("raw trust artifact activation identity is invalid")
        connection.execute(
            "INSERT INTO raw_trust_artifact_activations "
            "(artifact_kind,artifact_digest,"
            "raw_ingestor_trust_store_digest,usable,prepared_at) "
            "VALUES (?,?,?,?,?)",
            (
                artifact_kind,
                artifact_digest,
                raw_ingestor_trust_store_digest,
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _require_raw_trust_artifact_usable(
        connection: sqlite3.Connection,
        *,
        artifact_kind: str,
        artifact_digest: str,
        raw_ingestor_trust_store_digest: str,
    ) -> None:
        row = connection.execute(
            "SELECT raw_ingestor_trust_store_digest,usable,prepared_at,"
            "activated_at,expired_at FROM raw_trust_artifact_activations "
            "WHERE artifact_kind = ? AND artifact_digest = ?",
            (artifact_kind, artifact_digest),
        ).fetchone()
        try:
            prepared_at = datetime.fromisoformat(
                str(row[2]).replace("Z", "+00:00")
            )
            activated_at = datetime.fromisoformat(
                str(row[3]).replace("Z", "+00:00")
            )
            chronology_valid = (
                prepared_at.utcoffset() == timedelta(0)
                and activated_at.utcoffset() == timedelta(0)
                and prepared_at <= activated_at
            )
        except (IndexError, TypeError, ValueError):
            chronology_valid = False
        if (
            row is None
            or row[0] != raw_ingestor_trust_store_digest
            or int(row[1]) != 1
            or not chronology_valid
            or row[3] is None
            or row[4] is not None
        ):
            raise ValueError("raw trust artifact is not durably usable")

    def _expire_raw_trust_artifact_activation(
        self,
        *,
        artifact_kind: str,
        artifact_digest: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE raw_trust_artifact_activations SET usable = 0, "
                "expired_at = ? WHERE artifact_kind = ? AND "
                "artifact_digest = ? AND expired_at IS NULL",
                (
                    datetime.now(timezone.utc).isoformat(),
                    artifact_kind,
                    artifact_digest,
                ),
            )

    def _activate_raw_trust_artifact(
        self,
        *,
        artifact_kind: str,
        artifact_digest: str,
        raw_ingestor_trust_store_digest: str,
        raw_ingestor_trust_store_path: str | Path,
    ) -> None:
        _require_current_raw_ingestor_trust_store_digest(
            raw_ingestor_trust_store_path,
            raw_ingestor_trust_store_digest,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                raw_ingestor_trust_store_digest,
            )
            activated_at = datetime.now(timezone.utc).isoformat()
            updated = connection.execute(
                "UPDATE raw_trust_artifact_activations SET usable = 1, "
                "activated_at = ? WHERE artifact_kind = ? AND "
                "artifact_digest = ? AND raw_ingestor_trust_store_digest = ? "
                "AND usable = 0 AND activated_at IS NULL AND expired_at IS NULL",
                (
                    activated_at,
                    artifact_kind,
                    artifact_digest,
                    raw_ingestor_trust_store_digest,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("raw trust artifact activation is unavailable")
        try:
            _require_current_raw_ingestor_trust_store_digest(
                raw_ingestor_trust_store_path,
                raw_ingestor_trust_store_digest,
            )
        except Exception:
            self._expire_raw_trust_artifact_activation(
                artifact_kind=artifact_kind,
                artifact_digest=artifact_digest,
            )
            raise

    def reconcile_prepared_raw_trust_activations(
        self,
        *,
        raw_ingestor_trust_store_path: str | Path,
        authority_trust_store_path: str | Path,
    ) -> tuple[tuple[str, str], ...]:
        """Resume product-owned activations left prepared by a process crash."""

        current = _load_raw_ingestor_trust_store(
            raw_ingestor_trust_store_path
        )
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT artifact_kind,artifact_digest,"
                "raw_ingestor_trust_store_digest FROM "
                "raw_trust_artifact_activations WHERE usable = 0 "
                "AND activated_at IS NULL AND expired_at IS NULL "
                "ORDER BY prepared_at,artifact_kind,artifact_digest"
            ).fetchall()
        activated: list[tuple[str, str]] = []
        for artifact_kind, artifact_digest, trust_digest in rows:
            retained_trust_digest: str | None = None
            try:
                if artifact_kind == "scoring_replay_bundle":
                    replay = self.load_neural_prior_scoring_replay_bundle(
                        str(artifact_digest),
                        _require_raw_trust_activation=False,
                    )
                    if isinstance(replay.manifest, ScoringReplayBundleManifest):
                        retained_trust_digest = (
                            replay.manifest.raw_ingestor_trust_store_digest
                        )
                elif artifact_kind == "scoring_completion":
                    retained_trust_digest = (
                        self._validate_prepared_scoring_completion(
                            str(artifact_digest)
                        )
                    )
                elif artifact_kind == "promotion_evidence":
                    evidence = self.load_neural_prior_promotion(
                        str(artifact_digest),
                        _require_raw_trust_activation=False,
                    )
                    if type(evidence) is NeuralPriorPromotionEvidence:
                        retained_trust_digest = (
                            evidence.raw_ingestor_trust_store_digest
                        )
                elif artifact_kind == "promotion_deployment_certificate":
                    certificate = (
                        self.load_neural_prior_promotion_deployment_certificate(
                            str(artifact_digest),
                            authority_trust_store_path=(
                                authority_trust_store_path
                            ),
                            _require_raw_trust_activation=False,
                        )
                    )
                    retained_trust_digest = (
                        certificate.raw_ingestor_trust_store_digest
                    )
            except Exception:
                retained_trust_digest = None
            if (
                retained_trust_digest != trust_digest
                or trust_digest != current.content_digest
            ):
                self._expire_raw_trust_artifact_activation(
                    artifact_kind=str(artifact_kind),
                    artifact_digest=str(artifact_digest),
                )
                continue
            self._activate_raw_trust_artifact(
                artifact_kind=str(artifact_kind),
                artifact_digest=str(artifact_digest),
                raw_ingestor_trust_store_digest=str(trust_digest),
                raw_ingestor_trust_store_path=raw_ingestor_trust_store_path,
            )
            activated.append((str(artifact_kind), str(artifact_digest)))
        return tuple(activated)

    def _validate_prepared_scoring_completion(
        self,
        receipt_digest: str,
    ) -> str:
        """Rehash the typed scoring output sealed by a prepared completion."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT c.receipt_json,s.receipt_json,a.payload_json,"
                "l.payload_json FROM trusted_process_completion_receipts AS c "
                "JOIN trusted_process_start_receipts_v2 AS s "
                "ON s.receipt_digest = c.start_receipt_digest "
                "JOIN neural_prior_holdout_scoring_artifacts AS a "
                "ON a.artifact_digest = c.output_artifact_digest "
                "JOIN trusted_process_log_artifacts AS l "
                "ON l.artifact_digest = c.process_log_digest "
                "WHERE c.receipt_digest = ? AND c.process_kind = "
                "'candidate_scoring'",
                (receipt_digest,),
            ).fetchone()
        if row is None:
            raise ValueError("prepared scoring completion preimage is unavailable")
        completion = _decode_completion_receipt(row[0], receipt_digest)
        start = _decode_start_receipt(row[1], completion.start_receipt_digest)
        scoring = _decode_holdout_scoring_artifact(
            row[2], completion.output_artifact_digest
        )
        if type(scoring) is not HoldoutScoringArtifact:
            raise ValueError("current completion requires current scoring artifact")
        process_log = _decode_process_log_artifact(
            row[3], completion.process_log_digest
        )
        validate_trusted_process_completion_receipt(completion, start)
        if (
            process_log.start_receipt_digest != start.receipt_digest
            or process_log.process_kind != "candidate_scoring"
            or completion.output_artifact_digest != scoring.artifact_digest
            or completion.process_log_digest != process_log.artifact_digest
        ):
            raise ValueError("prepared scoring completion preimage changed")
        replay = self.load_neural_prior_scoring_replay_bundle(
            scoring.scoring_replay_bundle_digest
        )
        if (
            not isinstance(replay.manifest, ScoringReplayBundleManifest)
            or replay.manifest.raw_ingestor_trust_store_digest
            != scoring.raw_ingestor_trust_store_digest
        ):
            raise ValueError("prepared scoring completion replay changed")
        return scoring.raw_ingestor_trust_store_digest

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _connect_approved_index(index_path: Path) -> sqlite3.Connection:
        """Open only the exact externally registered regular SQLite file."""

        if type(index_path) is not _NATIVE_PATH_TYPE:
            raise TypeError("approved EpisodeLedger index must be a native Path")
        before = os.lstat(index_path)
        current_uid = getattr(os, "geteuid", lambda: before.st_uid)()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, current_uid}
            or before.st_mode & 0o022
        ):
            raise ValueError(
                "approved EpisodeLedger index must be an owned immutable locator"
            )
        connection = sqlite3.connect(index_path, timeout=30.0)
        try:
            after = os.lstat(index_path)
            if (
                not stat.S_ISREG(after.st_mode)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or after.st_uid != before.st_uid
                or after.st_mode & 0o022
            ):
                raise ValueError(
                    "approved EpisodeLedger index changed while it was opened"
                )
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
        except Exception:
            connection.close()
            raise
        return connection

    def _is_indexed(self, episode_id: str) -> bool:
        with self._connect() as connection:
            return _row_exists(connection, episode_id)

    def _indexed_target(
        self,
        episode_id: str,
    ) -> tuple[Path, sqlite3.Row]:
        _validate_episode_id(episode_id)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT contract_hash, manifest_sha256, arrays_sha256,
                       indirect_observation_sensitivity_available, path
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown episode: {episode_id}")

        expected = Path("episodes") / episode_id
        relative = Path(row["path"])
        if relative != expected:
            raise ValueError("indexed episode path violates the storage contract")
        unresolved = self.root / relative
        if unresolved.is_symlink():
            raise ValueError("indexed episode directory cannot be a symlink")
        target = unresolved.resolve()
        if target.parent != self.episodes_dir or not target.is_dir():
            raise ValueError("indexed episode path is missing or unsafe")
        return target, row

    def _insert_episode(
        self,
        connection: sqlite3.Connection,
        episode: SensitivityEpisode,
        target: Path,
        checksums: dict[str, str],
    ) -> None:
        snapshot = episode.snapshot
        connection.execute(
            """
            INSERT INTO episodes (
                episode_id, issue_time, radar_id, contract_hash,
                model_commit, trust_score, promotion_eligible,
                impact_available,
                indirect_observation_sensitivity_available,
                manifest_sha256, arrays_sha256, path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.issue_time,
                episode.radar_id,
                episode.contract.digest,
                episode.contract.model_commit,
                snapshot.trust_score,
                0,
                int(snapshot.impact_available),
                0,
                checksums["manifest.json"],
                checksums["sensitivity_arrays.npz"],
                str(target.relative_to(self.root)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _insert_scalar_impacts(
        self,
        connection: sqlite3.Connection,
        episode: SensitivityEpisode,
    ) -> None:
        snapshot = episode.snapshot
        rows: list[tuple[Any, ...]] = []
        for lead_index, lead in enumerate(snapshot.lead_minutes):
            for metric_index, metric in enumerate(snapshot.metric_names):
                rows.append(
                    (
                        episode.episode_id,
                        lead,
                        metric,
                        0,
                        "partial_direct_latest_dbz_fixed_control",
                        _sqlite_number(
                            snapshot.forecast_scores[lead_index, metric_index]
                        ),
                        _sqlite_number(
                            snapshot.direct.norm[lead_index, metric_index]
                        ),
                        _sqlite_number(
                            None
                            if snapshot.direct.impact is None
                            else snapshot.direct.impact[
                                lead_index, metric_index
                            ]
                        ),
                        _sqlite_number(
                            None
                            if snapshot.direct.reward is None
                            else snapshot.direct.reward[
                                lead_index, metric_index
                            ]
                        ),
                    )
                )
        connection.executemany(
            """
            INSERT INTO episode_impacts (
                episode_id, lead_minutes, metric_name,
                input_offset_minutes, direct_path_status,
                forecast_score, direct_sensitivity_norm,
                direct_impact, direct_normalized_reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _ensure_episode_impacts_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create or transactionally upgrade the scalar impact table."""

    row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND name = 'episode_impacts'
        """
    ).fetchone()
    if row is None:
        _create_episode_impacts_table(connection)
        return

    columns = {
        value[1]: value
        for value in connection.execute(
            "PRAGMA table_info(episode_impacts)"
        ).fetchall()
    }
    nullable_scores = all(
        name in columns and columns[name][3] == 0
        for name in ("forecast_score", "direct_sensitivity_norm")
    )
    deferred_foreign_key = (
        "DEFERRABLE INITIALLY DEFERRED" in row[0].upper()
    )
    if nullable_scores and deferred_foreign_key:
        return

    for trigger in (
        "episode_impacts_no_update",
        "episode_impacts_no_delete",
        "episode_impacts_no_late_insert",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.execute(
        "ALTER TABLE episode_impacts RENAME TO episode_impacts_legacy"
    )
    _create_episode_impacts_table(connection)
    connection.execute(
        """
        INSERT INTO episode_impacts (
            episode_id, lead_minutes, metric_name,
            input_offset_minutes, direct_path_status,
            forecast_score, direct_sensitivity_norm,
            direct_impact, direct_normalized_reward
        )
        SELECT
            episode_id, lead_minutes, metric_name,
            input_offset_minutes, direct_path_status,
            forecast_score, direct_sensitivity_norm,
            direct_impact, direct_normalized_reward
        FROM episode_impacts_legacy
        """
    )
    connection.execute("DROP TABLE episode_impacts_legacy")


def _ensure_variational_learning_approval_schema(
    connection: sqlite3.Connection,
) -> None:
    """Add ranked-selection lineage columns to an existing append-only index."""

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(variational_learning_approvals)"
        ).fetchall()
    }
    definitions = {
        "selection_mode": "TEXT NOT NULL DEFAULT 'direct'",
        "candidate_id": "TEXT",
        "candidate_rank": "INTEGER",
        "candidate_score": "REAL",
        "candidate_perturbation_digest": "TEXT",
        "ranking_digest": "TEXT",
        "ranking_policy_digest": "TEXT",
        "ranking_objective": "TEXT",
        "approved_action_digest": "TEXT",
        "nominal_input_bundle_digest": "TEXT",
        "nominal_full_analysis_input_digest": "TEXT",
        "whitener_operations_per_apply": "INTEGER NOT NULL DEFAULT 0",
        "observed_whitener_apply_count": "INTEGER NOT NULL DEFAULT 0",
        "observed_whitener_total_operations": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE variational_learning_approvals "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_realized_intervention_schema(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(realized_observation_interventions)"
        ).fetchall()
    }
    definitions = {
        "execution_policy_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "execution_trust_store_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "predicted_normalized_benefit": "REAL NOT NULL DEFAULT 0",
        "resolved_normalized_benefit": "REAL NOT NULL DEFAULT 0",
        "case_id": "TEXT NOT NULL DEFAULT ''",
        "radar_id": "TEXT NOT NULL DEFAULT ''",
        "issue_time": "TEXT NOT NULL DEFAULT ''",
        "input_bundle_before_digest": "TEXT NOT NULL DEFAULT ''",
        "input_bundle_after_digest": "TEXT NOT NULL DEFAULT ''",
        "resolved_issuance_validation_digest": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE realized_observation_interventions "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_prospective_receipt_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(realized_intervention_receipts)"
        ).fetchall()
    }
    definitions = {
        "executor_key_id": "TEXT NOT NULL DEFAULT ''",
        "executor_sequence_number": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE realized_intervention_receipts "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_prospective_decision_schema(connection: sqlite3.Connection) -> None:
    """Add signed-review columns without fabricating evidence for old rows."""

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(prospective_intervention_decisions)"
        ).fetchall()
    }
    definitions = {
        "operator_approval_digest": "TEXT NOT NULL DEFAULT ''",
        "operator_approval_json": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE prospective_intervention_decisions "
                f"ADD COLUMN {name} {definition}"
            )


def _ensure_neural_prior_promotion_schema(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(neural_prior_promotions)"
        ).fetchall()
    }
    definitions = {
        "candidate_manifest_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "candidate_manifest_json": "TEXT NOT NULL DEFAULT '{}'",
        "holdout_plan_digest": f"TEXT NOT NULL DEFAULT '{'0' * 64}'",
        "distinct_case_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_storm_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_day_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_radar_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_regime_count": "INTEGER NOT NULL DEFAULT 0",
        "distinct_range_regime_count": "INTEGER NOT NULL DEFAULT 0",
        "beneficial_fraction_lower_bound": "REAL NOT NULL DEFAULT 0",
        "harmful_fraction_upper_bound": "REAL NOT NULL DEFAULT 1",
        "mean_improvement_lower_bound": "REAL NOT NULL DEFAULT 0",
        "prior_gaussian_nll_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_support_brier_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_underdispersion_increase_upper_bound": "REAL NOT NULL DEFAULT 0",
        "prior_conditional_underdispersion_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_echo_support_miss_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_echo_object_miss_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_echo_intensity_nll_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_clear_sky_false_echo_increase_upper_bound": (
            "REAL NOT NULL DEFAULT 0"
        ),
        "prior_echo_component_status": "TEXT NOT NULL DEFAULT ''",
        "prior_clear_sky_component_status": "TEXT NOT NULL DEFAULT ''",
        "prior_echo_case_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_clear_sky_case_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_echo_cluster_count": "INTEGER NOT NULL DEFAULT 0",
        "prior_clear_sky_cluster_count": "INTEGER NOT NULL DEFAULT 0",
        "simultaneous_inference_test_count": "INTEGER NOT NULL DEFAULT 0",
        "evaluation_payloads_json": "TEXT NOT NULL DEFAULT '[]'",
        "evidence_payload_json": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in definitions.items():
        if name not in columns:
            connection.execute(
                "ALTER TABLE neural_prior_promotions "
                f"ADD COLUMN {name} {definition}"
            )


def _evaluation_audit_payload(
    evaluation: PriorHoldoutEvaluation,
) -> dict[str, object]:
    """Serialize the small paired-evaluation payload for later audit."""

    result: dict[str, object] = {}
    for name, value in evaluation.__dict__.items():
        if isinstance(value, Tensor):
            owned = value.detach().cpu()
            result[name] = {
                "kind": "tensor",
                "dtype": str(owned.dtype).removeprefix("torch."),
                "shape": list(owned.shape),
                "digest": tensor_digest(owned),
                "values": owned.tolist(),
            }
        elif name == "range_band_evaluations":
            result[name] = [
                _range_band_evaluation_audit_payload(item) for item in value
            ]
        elif isinstance(value, tuple):
            result[name] = list(value)
        elif hasattr(value, "value"):
            result[name] = value.value
        else:
            result[name] = value
    return result


def _range_band_evaluation_audit_payload(
    evaluation: RangeBandEvaluation,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in evaluation.__dict__.items():
        if isinstance(value, Tensor):
            owned = value.detach().cpu()
            result[name] = {
                "kind": "tensor",
                "dtype": str(owned.dtype).removeprefix("torch."),
                "shape": list(owned.shape),
                "digest": tensor_digest(owned),
                "values": owned.tolist(),
            }
        elif isinstance(value, tuple):
            result[name] = [
                list(item) if isinstance(item, tuple) else item
                for item in value
            ]
        else:
            result[name] = value
    return result


def _decode_audit_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "dtype",
        "shape",
        "digest",
        "values",
    } or value["kind"] != "tensor":
        raise ValueError(f"invalid promotion audit tensor: {name}")
    dtype = getattr(torch, str(value["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"invalid promotion audit dtype: {name}")
    tensor = torch.tensor(value["values"], dtype=dtype)
    if list(tensor.shape) != value["shape"] or tensor_digest(tensor) != value["digest"]:
        raise ValueError(f"promotion audit tensor digest mismatch: {name}")
    return tensor


def _decode_evaluation_audit_payloads(
    value: object,
) -> tuple[PriorHoldoutEvaluation | LegacyPromotionEvaluationAudit, ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("invalid promotion evaluation audit payload")
    if value and (
        isinstance(value[0].get("metric_change"), list)
        or value[0].get("contract") != "prior-holdout-evaluation-v24"
    ):
        audits: list[LegacyPromotionEvaluationAudit] = []
        for raw in value:
            assert isinstance(raw, dict)
            digest = raw.get("evaluation_digest")
            if not isinstance(digest, str):
                raise ValueError("legacy promotion evaluation digest is missing")
            audits.append(
                LegacyPromotionEvaluationAudit(
                    evaluation_digest=digest,
                    payload_json=json.dumps(
                        raw,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        return tuple(audits)
    evaluations: list[PriorHoldoutEvaluation] = []
    tensor_names = {
        "metric_change",
        "candidate_issuance_effect",
        "parent_issuance_effect",
        "end_to_end_metric_change",
        "metric_available",
        "coverage_candidate",
        "coverage_parent",
        "coverage_common",
        "newly_issued_fraction",
        "withdrawn_fraction",
    }
    tuple_names = {
        "lead_minutes",
        "metric_names",
        "metric_support_contract_digests",
        "verification_valid_times",
        "classified_range_regimes",
        "reference_active_range_regimes",
        "classifier_regime_labels",
        "classifier_range_regime_labels",
        "classifier_regime_probabilities",
        "classifier_range_regime_probabilities",
    }
    for raw in value:
        assert isinstance(raw, dict)
        values = dict(raw)
        stored_digest = values.pop("evaluation_digest", None)
        for name in tensor_names:
            values[name] = _decode_audit_tensor(name, values[name])
        for name in tuple_names:
            values[name] = tuple(values[name])
        band_evaluations: list[RangeBandEvaluation] = []
        raw_bands = values.pop("range_band_evaluations", None)
        if not isinstance(raw_bands, list):
            raise ValueError("range-band evaluation audit payload is missing")
        for raw_band in raw_bands:
            if not isinstance(raw_band, dict):
                raise ValueError("invalid range-band evaluation audit payload")
            band_values = dict(raw_band)
            band_digest = band_values.pop("evaluation_digest", None)
            for name in (
                "metric_change",
                "end_to_end_metric_change",
                "metric_available",
                "metric_valid_area_km2",
                "withdrawn_fraction_by_lead",
                "newly_issued_fraction_by_lead",
                "background_fallback_increase_by_lead",
                "confidence_weighted_coverage_change_by_lead",
            ):
                band_values[name] = _decode_audit_tensor(
                    f"range_band.{name}", band_values[name]
                )
            band_values["uncertainty_component_differences"] = tuple(
                tuple(item)
                for item in band_values["uncertainty_component_differences"]
            )
            band_values["candidate_uncertainty_component_scores"] = tuple(
                tuple(item)
                for item in band_values[
                    "candidate_uncertainty_component_scores"
                ]
            )
            band_values["parent_uncertainty_component_scores"] = tuple(
                tuple(item)
                for item in band_values[
                    "parent_uncertainty_component_scores"
                ]
            )
            band_values["uncertainty_component_sample_counts"] = tuple(
                tuple(item)
                for item in band_values[
                    "uncertainty_component_sample_counts"
                ]
            )
            band_values["metric_valid_area_km2_by_lead"] = tuple(
                band_values["metric_valid_area_km2_by_lead"]
            )
            for name in (
                "issuance_domain_cell_count_by_lead",
                "issuance_domain_area_km2_by_lead",
                "parent_issued_count_by_lead",
                "candidate_issued_count_by_lead",
                "withdrawn_count_by_lead",
                "newly_issued_count_by_lead",
                "parent_fallback_count_by_lead",
                "candidate_fallback_count_by_lead",
                "parent_confidence_weighted_issued_area_by_lead",
                "candidate_confidence_weighted_issued_area_by_lead",
            ):
                band_values[name] = tuple(band_values[name])
            band_values.pop("contract", None)
            band = RangeBandEvaluation(**band_values)
            if band.evaluation_digest != band_digest:
                raise ValueError("range-band evaluation digest mismatch")
            band_evaluations.append(band)
        values["range_band_evaluations"] = tuple(band_evaluations)
        values.pop("contract", None)
        evaluation = _new_prior_holdout_evaluation(**values)
        if evaluation.evaluation_digest != stored_digest:
            raise ValueError("promotion evaluation digest mismatch")
        evaluations.append(evaluation)
    return tuple(evaluations)


def _decode_process_log_artifact(
    text: str,
    expected_digest: str,
) -> ProcessLogArtifact:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid process-log artifact payload")
    values = dict(value)
    stored_digest = values.pop("artifact_digest", None)
    values["entries"] = tuple(values["entries"])
    artifact = ProcessLogArtifact(**cast(Any, values))
    if stored_digest != expected_digest or artifact.artifact_digest != expected_digest:
        raise ValueError("process-log artifact digest mismatch")
    return artifact


def _decode_holdout_scoring_input_artifact(
    text: str,
    expected_digest: str,
) -> HoldoutScoringInputArtifact:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid holdout scoring input artifact payload")
    values = dict(value)
    stored_digest = values.pop("artifact_digest", None)
    for name in (
        "ordered_case_ids",
        "completed_holdout_case_digests",
        "candidate_forecast_digests",
        "parent_forecast_digests",
        "candidate_prior_application_digests",
        "parent_prior_application_digests",
        "candidate_inference_evidence_digests",
        "parent_inference_evidence_digests",
        "verification_digests",
        "metric_contract_digests",
        "operational_issuance_domain_artifact_digests",
    ):
        values[name] = tuple(values[name])
    artifact = object.__new__(HoldoutScoringInputArtifact)
    for name, retained in values.items():
        object.__setattr__(artifact, name, retained)
    object.__setattr__(artifact, "artifact_digest", _json_digest(artifact.payload))
    if stored_digest != expected_digest or artifact.artifact_digest != expected_digest:
        raise ValueError("holdout scoring input artifact digest mismatch")
    return artifact


def _decode_holdout_scoring_artifact(
    text: str,
    expected_digest: str,
) -> (
    HoldoutScoringArtifact
    | LegacyHoldoutScoringArtifactAuditV10
    | LegacyHoldoutScoringArtifactAuditV11
    | LegacyHoldoutScoringArtifactAuditV12
    | LegacyHoldoutScoringArtifactAuditV13
):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid holdout scoring artifact payload")
    values = dict(value)
    stored_digest = values.pop("artifact_digest", None)
    if values.get("contract") == "neural-prior-holdout-scoring-artifact-v10":
        if stored_digest != expected_digest:
            raise ValueError("holdout scoring artifact digest mismatch")
        return LegacyHoldoutScoringArtifactAuditV10(
            artifact_digest=expected_digest,
            payload_json=json.dumps(value, sort_keys=True, separators=(",", ":")),
        )
    if values.get("contract") == "neural-prior-holdout-scoring-artifact-v11":
        if stored_digest != expected_digest:
            raise ValueError("holdout scoring artifact digest mismatch")
        return LegacyHoldoutScoringArtifactAuditV11(
            artifact_digest=expected_digest,
            payload_json=json.dumps(value, sort_keys=True, separators=(",", ":")),
        )
    if values.get("contract") == "neural-prior-holdout-scoring-artifact-v12":
        if stored_digest != expected_digest:
            raise ValueError("holdout scoring artifact digest mismatch")
        return LegacyHoldoutScoringArtifactAuditV12(
            artifact_digest=expected_digest,
            payload_json=json.dumps(value, sort_keys=True, separators=(",", ":")),
        )
    if values.get("contract") == "neural-prior-holdout-scoring-artifact-v13":
        if stored_digest != expected_digest:
            raise ValueError("holdout scoring artifact digest mismatch")
        return LegacyHoldoutScoringArtifactAuditV13(
            artifact_digest=expected_digest,
            payload_json=json.dumps(value, sort_keys=True, separators=(",", ":")),
        )
    if values.get("contract") != "neural-prior-holdout-scoring-artifact-v14":
        raise ValueError("legacy holdout scoring artifacts are audit-only")
    for name in (
        "ordered_case_ids",
        "ordered_evaluation_digests",
        "candidate_forecast_digests",
        "parent_forecast_digests",
        "verification_digests",
        "metric_contract_digests",
    ):
        values[name] = tuple(values[name])
    artifact = _new_holdout_scoring_artifact(**values)
    if stored_digest != expected_digest or artifact.artifact_digest != expected_digest:
        raise ValueError("holdout scoring artifact digest mismatch")
    return artifact


def _decode_scoring_replay_bundle_manifest(
    text: str,
    *,
    expected_digest: str,
) -> (
    ScoringReplayBundleManifest
    | LegacyScoringReplayBundleManifestAuditV1
    | LegacyScoringReplayBundleManifestAuditV2
    | LegacyScoringReplayBundleManifestAuditV3
    | LegacyScoringReplayBundleManifestAuditV4
    | LegacyScoringReplayBundleManifestAuditV5
    | LegacyScoringReplayBundleManifestAuditV6
    | LegacyScoringReplayBundleManifestAuditV7
    | LegacyScoringReplayBundleManifestAuditV8
    | LegacyScoringReplayBundleManifestAuditV9
    | LegacyScoringReplayBundleManifestAuditV11
    | LegacyScoringReplayBundleManifestAuditV12
    | LegacyScoringReplayBundleManifestAuditV13
    | LegacyScoringReplayBundleManifestAuditV14
):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid scoring replay bundle manifest")
    values = dict(value)
    stored_digest = values.pop("bundle_digest", None)
    raw_records = values.pop("tensor_records", None)
    if not isinstance(raw_records, list) or any(
        not isinstance(item, dict) for item in raw_records
    ):
        raise ValueError("invalid scoring replay tensor records")
    records = tuple(
        ScoringReplayTensorRecord(
            case_id=str(item["case_id"]),
            role=str(item["role"]),
            archive_member=str(item["archive_member"]),
            dtype=str(item["dtype"]),
            shape=tuple(int(entry) for entry in item["shape"]),
            tensor_digest=str(item["tensor_digest"]),
            archive_sha256=(
                None
                if item.get("archive_sha256") is None
                else str(item["archive_sha256"])
            ),
        )
        for item in raw_records
    )
    values["ordered_case_ids"] = tuple(values["ordered_case_ids"])
    values["ordered_evaluation_digests"] = tuple(
        values["ordered_evaluation_digests"]
    )
    values["tensor_records"] = records
    if "tensor_shard_sha256s" in values:
        values["tensor_shard_sha256s"] = tuple(
            values["tensor_shard_sha256s"]
        )
    if values.get("contract") == "neural-prior-scoring-replay-bundle-v1":
        manifest: (
            ScoringReplayBundleManifest
            | LegacyScoringReplayBundleManifestAuditV1
            | LegacyScoringReplayBundleManifestAuditV2
            | LegacyScoringReplayBundleManifestAuditV3
            | LegacyScoringReplayBundleManifestAuditV4
            | LegacyScoringReplayBundleManifestAuditV5
            | LegacyScoringReplayBundleManifestAuditV6
            | LegacyScoringReplayBundleManifestAuditV7
            | LegacyScoringReplayBundleManifestAuditV8
            | LegacyScoringReplayBundleManifestAuditV9
            | LegacyScoringReplayBundleManifestAuditV11
            | LegacyScoringReplayBundleManifestAuditV12
            | LegacyScoringReplayBundleManifestAuditV13
            | LegacyScoringReplayBundleManifestAuditV14
        ) = LegacyScoringReplayBundleManifestAuditV1(
            **cast(Any, values)
        )
    else:
        values["semantic_case_digests"] = tuple(
            values["semantic_case_digests"]
        )
        values["dynamic_source_case_ids"] = tuple(
            values["dynamic_source_case_ids"]
        )
        values["background_case_ids"] = tuple(
            values["background_case_ids"]
        )
        if values.get("contract") == "neural-prior-scoring-replay-bundle-v2":
            manifest = LegacyScoringReplayBundleManifestAuditV2(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v3":
            manifest = LegacyScoringReplayBundleManifestAuditV3(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v4":
            manifest = LegacyScoringReplayBundleManifestAuditV4(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v5":
            manifest = LegacyScoringReplayBundleManifestAuditV5(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v6":
            manifest = LegacyScoringReplayBundleManifestAuditV6(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v7":
            manifest = LegacyScoringReplayBundleManifestAuditV7(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v8":
            manifest = LegacyScoringReplayBundleManifestAuditV8(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v9":
            manifest = LegacyScoringReplayBundleManifestAuditV9(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v11":
            manifest = LegacyScoringReplayBundleManifestAuditV11(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v12":
            manifest = LegacyScoringReplayBundleManifestAuditV12(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v13":
            manifest = LegacyScoringReplayBundleManifestAuditV13(
                **cast(Any, values)
            )
        elif values.get("contract") == "neural-prior-scoring-replay-bundle-v14":
            manifest = LegacyScoringReplayBundleManifestAuditV14(
                **cast(Any, values)
            )
        else:
            manifest = ScoringReplayBundleManifest(**cast(Any, values))
    if (
        stored_digest != expected_digest
        or manifest.bundle_digest != expected_digest
    ):
        raise ValueError("scoring replay bundle manifest digest mismatch")
    return manifest


def _decode_completion_receipt(
    text: str,
    expected_digest: str,
) -> TrustedProcessCompletionReceipt:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid process-completion receipt payload")
    values = dict(value)
    stored_digest = values.pop("receipt_digest", None)
    values["subject_digests"] = tuple(values["subject_digests"])
    receipt = _new_trusted_process_completion_receipt(**values)
    if stored_digest != expected_digest or receipt.receipt_digest != expected_digest:
        raise ValueError("process-completion receipt digest mismatch")
    return receipt


def _decode_start_receipt(
    text: str,
    expected_digest: str,
) -> TrustedProcessStartReceipt:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid process-start receipt payload")
    values = dict(value)
    stored_digest = values.pop("receipt_digest", None)
    values["subject_digests"] = tuple(values["subject_digests"])
    receipt = _new_trusted_process_start_receipt(**values)
    if stored_digest != expected_digest or receipt.receipt_digest != expected_digest:
        raise ValueError("process-start receipt digest mismatch")
    return receipt


def _decode_candidate_manifest(
    text: str,
    *,
    expected_digest: str,
) -> (
    NeuralPriorCandidateManifest
    | LegacyNeuralPriorCandidateManifestAuditV2
    | LegacyNeuralPriorCandidateManifestAuditV3
    | LegacyNeuralPriorCandidateManifestAuditV4
    | LegacyNeuralPriorCandidateManifestAuditV5
    | LegacyNeuralPriorCandidateManifestAuditV6
    | LegacyNeuralPriorCandidateManifestAuditV7
    | LegacyNeuralPriorCandidateManifestAuditV8
    | LegacyNeuralPriorCandidateManifestAuditV9
    | LegacyNeuralPriorCandidateManifestAuditV10
    | LegacyNeuralPriorCandidateManifestAuditV11
    | LegacyNeuralPriorCandidateManifestAuditV12
    | LegacyNeuralPriorCandidateManifestAuditV13
    | LegacyNeuralPriorCandidateManifestAuditV14
    | LegacyNeuralPriorCandidateManifestAuditV15
    | LegacyNeuralPriorCandidateManifestAuditV16
    | LegacyNeuralPriorCandidateManifestAuditV17
    | LegacyNeuralPriorCandidateManifestAuditV18
):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("invalid candidate manifest audit payload")
    values = dict(value)
    stored_digest = values.pop("manifest_digest", None)
    if values.get("contract") == "neural-prior-candidate-manifest-v2":
        audit = LegacyNeuralPriorCandidateManifestAuditV2(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit
    if values.get("contract") == "neural-prior-candidate-manifest-v3":
        audit_v3 = LegacyNeuralPriorCandidateManifestAuditV3(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v3.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v3
    if values.get("contract") == "neural-prior-candidate-manifest-v4":
        audit_v4 = LegacyNeuralPriorCandidateManifestAuditV4(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v4.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v4
    if values.get("contract") == "neural-prior-candidate-manifest-v5":
        audit_v5 = LegacyNeuralPriorCandidateManifestAuditV5(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v5.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v5
    if values.get("contract") == "neural-prior-candidate-manifest-v6":
        audit_v6 = LegacyNeuralPriorCandidateManifestAuditV6(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v6.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v6
    if values.get("contract") == "neural-prior-candidate-manifest-v7":
        audit_v7 = LegacyNeuralPriorCandidateManifestAuditV7(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v7.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v7
    if values.get("contract") == "neural-prior-candidate-manifest-v8":
        audit_v8 = LegacyNeuralPriorCandidateManifestAuditV8(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v8.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v8
    if values.get("contract") == "neural-prior-candidate-manifest-v9":
        audit_v9 = LegacyNeuralPriorCandidateManifestAuditV9(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v9.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v9
    if values.get("contract") == "neural-prior-candidate-manifest-v10":
        audit_v10 = LegacyNeuralPriorCandidateManifestAuditV10(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v10.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v10
    if values.get("contract") == "neural-prior-candidate-manifest-v11":
        audit_v11 = LegacyNeuralPriorCandidateManifestAuditV11(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v11.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v11
    if values.get("contract") == "neural-prior-candidate-manifest-v12":
        audit_v12 = LegacyNeuralPriorCandidateManifestAuditV12(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v12.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v12
    if values.get("contract") == "neural-prior-candidate-manifest-v13":
        audit_v13 = LegacyNeuralPriorCandidateManifestAuditV13(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v13.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v13
    if values.get("contract") == "neural-prior-candidate-manifest-v14":
        audit_v14 = LegacyNeuralPriorCandidateManifestAuditV14(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v14.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v14
    if values.get("contract") == "neural-prior-candidate-manifest-v15":
        audit_v15 = LegacyNeuralPriorCandidateManifestAuditV15(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v15.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v15
    if values.get("contract") == "neural-prior-candidate-manifest-v16":
        audit_v16 = LegacyNeuralPriorCandidateManifestAuditV16(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if audit_v16.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v16
    if values.get("contract") == "neural-prior-candidate-manifest-v17":
        audit_v17 = LegacyNeuralPriorCandidateManifestAuditV17(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ),
        )
        if audit_v17.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v17
    if values.get("contract") == "neural-prior-candidate-manifest-v18":
        audit_v18 = LegacyNeuralPriorCandidateManifestAuditV18(
            manifest_digest=str(stored_digest),
            payload_json=json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ),
        )
        if audit_v18.manifest_digest != expected_digest:
            raise ValueError("candidate manifest ledger digest mismatch")
        return audit_v18
    values["holdout_cases"] = tuple(
        NeuralPriorHoldoutCase(
            **cast(
                Any,
                dict(item)
                | {
                    "resolved_raw_volume_identity_digests": tuple(
                        item["resolved_raw_volume_identity_digests"]
                    )
                },
            )
        )
        for item in values["holdout_cases"]
    )
    reference_evidences: list[RegimeReferenceEvidence] = []
    for item in values["regime_reference_evidences"]:
        evidence_values = dict(item)
        stored_evidence_digest = evidence_values.pop("evidence_digest", None)
        evidence = _new_regime_reference_evidence(**evidence_values)
        if evidence.evidence_digest != stored_evidence_digest:
            raise ValueError("regime-reference evidence digest mismatch")
        reference_evidences.append(evidence)
    values["regime_reference_evidences"] = tuple(reference_evidences)
    event_catalogs: list[PhysicalEventCatalogEvidence] = []
    for item in values["physical_event_catalog_evidences"]:
        catalog_values = dict(item)
        stored_event_digest = catalog_values.pop("event_digest", None)
        catalog_values["object_track_artifact"] = (
            _decode_physical_event_track_artifact(
                catalog_values["object_track_artifact"]
            )
        )
        for name in (
            "member_case_ids",
            "member_full_analysis_input_digests",
            "spatial_envelope_xy_m",
            "start_centroid_xy_m",
            "end_centroid_xy_m",
            "mean_velocity_xy_mps",
            "participating_radar_ids",
        ):
            catalog_values[name] = tuple(catalog_values[name])
        catalog = _new_physical_event_catalog_evidence(**catalog_values)
        if catalog.event_digest != stored_event_digest:
            raise ValueError("physical event-catalog digest mismatch")
        event_catalogs.append(catalog)
    values["physical_event_catalog_evidences"] = tuple(event_catalogs)
    result_values = dict(values["physical_event_catalog_result"])
    stored_result_digest = result_values.pop("result_digest", None)
    result_event_digests = tuple(
        str(item["event_digest"])
        for item in result_values.pop("event_evidences")
    )
    if result_event_digests != tuple(item.event_digest for item in event_catalogs):
        raise ValueError("physical event-catalog result members disagree")
    result_values["event_evidences"] = tuple(event_catalogs)
    result_values["case_spatial_membership_evidences"] = (
        _decode_physical_event_case_spatial_evidences(
            result_values["case_spatial_membership_evidences"]
        )
    )
    catalog_result = _new_physical_event_catalog_result(**result_values)
    if catalog_result.result_digest != stored_result_digest:
        raise ValueError("physical event-catalog result digest mismatch")
    values["physical_event_catalog_result"] = catalog_result
    training_plan_values = dict(values["training_physical_event_catalog_plan"])
    stored_training_plan_digest = training_plan_values.pop("plan_digest", None)
    training_plan_values["holdout_case_ids"] = tuple(
        training_plan_values["holdout_case_ids"]
    )
    training_catalog_plan = PhysicalEventCatalogPlan(**training_plan_values)
    if training_catalog_plan.plan_digest != stored_training_plan_digest:
        raise ValueError("training physical event-catalog plan digest mismatch")
    values["training_physical_event_catalog_plan"] = training_catalog_plan
    training_result_values = dict(
        values["training_physical_event_catalog_result"]
    )
    stored_training_result_digest = training_result_values.pop(
        "result_digest", None
    )
    training_events: list[PhysicalEventCatalogEvidence] = []
    for item in training_result_values.pop("event_evidences"):
        event_values = dict(item)
        stored_event_digest = event_values.pop("event_digest", None)
        event_values["object_track_artifact"] = (
            _decode_physical_event_track_artifact(
                event_values["object_track_artifact"]
            )
        )
        for name in (
            "member_case_ids",
            "member_full_analysis_input_digests",
            "spatial_envelope_xy_m",
            "start_centroid_xy_m",
            "end_centroid_xy_m",
            "mean_velocity_xy_mps",
            "participating_radar_ids",
        ):
            event_values[name] = tuple(event_values[name])
        event = _new_physical_event_catalog_evidence(**event_values)
        if event.event_digest != stored_event_digest:
            raise ValueError("training physical event-catalog digest mismatch")
        training_events.append(event)
    training_result_values["event_evidences"] = tuple(training_events)
    training_result_values["case_spatial_membership_evidences"] = (
        _decode_physical_event_case_spatial_evidences(
            training_result_values["case_spatial_membership_evidences"]
        )
    )
    training_result = _new_physical_event_catalog_result(
        **training_result_values
    )
    if training_result.result_digest != stored_training_result_digest:
        raise ValueError("training physical event-catalog result digest mismatch")
    values["training_physical_event_catalog_result"] = training_result
    for name in (
        "candidate_training_start_receipt",
        "candidate_scoring_start_receipt",
    ):
        receipt_values = dict(values[name])
        stored_receipt_digest = receipt_values.pop("receipt_digest", None)
        receipt_values["subject_digests"] = tuple(
            receipt_values["subject_digests"]
        )
        receipt = _new_trusted_process_start_receipt(**receipt_values)
        if receipt.receipt_digest != stored_receipt_digest:
            raise ValueError("trusted process-start receipt digest mismatch")
        values[name] = receipt
    for name, start_name in (
        (
            "candidate_training_completion_receipt",
            "candidate_training_start_receipt",
        ),
    ):
        receipt_values = dict(values[name])
        stored_receipt_digest = receipt_values.pop("receipt_digest", None)
        receipt_values["subject_digests"] = tuple(
            receipt_values["subject_digests"]
        )
        receipt = _new_trusted_process_completion_receipt(**receipt_values)
        validate_trusted_process_completion_receipt(
            receipt,
            values[start_name],
        )
        if receipt.receipt_digest != stored_receipt_digest:
            raise ValueError("trusted process-completion receipt digest mismatch")
        values[name] = receipt
    for name in (
        "training_learning_approval_digests",
        "training_intervention_digests",
        "training_case_ids",
        "training_input_bundle_digests",
        "training_full_analysis_input_digests",
        "training_physical_event_digests",
        "training_raw_volume_identity_digests",
        "training_sampling_unit_digests",
        "training_storm_ids",
        "training_days",
        "training_radars",
        "training_regimes",
    ):
        values[name] = tuple(values[name])
    values["training_time_windows"] = tuple(
        tuple(item) for item in values["training_time_windows"]
    )
    manifest = NeuralPriorCandidateManifest(**values)
    if manifest.manifest_digest != stored_digest:
        raise ValueError("candidate manifest digest mismatch")
    if manifest.manifest_digest != expected_digest:
        raise ValueError("candidate manifest ledger digest mismatch")
    return manifest


def _decode_physical_event_track_artifact(
    value: object,
) -> PhysicalEventTrackArtifact:
    if not isinstance(value, dict):
        raise ValueError("physical-event track artifact payload is missing")
    values = dict(value)
    stored_digest = values.pop("artifact_digest", None)
    values["timestamps"] = tuple(values["timestamps"])
    values["centroid_xy_m"] = tuple(
        tuple(item) for item in values["centroid_xy_m"]
    )
    values["object_mask_digests"] = tuple(values["object_mask_digests"])
    values["source_radar_ids"] = tuple(values["source_radar_ids"])
    values["association_edge_digests"] = tuple(
        values["association_edge_digests"]
    )
    artifact = PhysicalEventTrackArtifact(**cast(Any, values))
    if artifact.artifact_digest != stored_digest:
        raise ValueError("physical-event track artifact digest mismatch")
    return artifact


def _decode_physical_event_case_spatial_evidences(
    values: object,
) -> tuple[PhysicalEventCaseSpatialEvidence, ...]:
    result: list[PhysicalEventCaseSpatialEvidence] = []
    for item in cast(list[object], values):
        evidence_values = dict(cast(dict[str, object], item))
        stored_digest = evidence_values.pop("evidence_digest", None)
        evidence_values["observed_spatial_envelope_xy_m"] = tuple(
            cast(list[float], evidence_values["observed_spatial_envelope_xy_m"])
        )
        evidence_values["event_spatial_envelope_xy_m"] = tuple(
            cast(list[float], evidence_values["event_spatial_envelope_xy_m"])
        )
        evidence = PhysicalEventCaseSpatialEvidence(
            **cast(Any, evidence_values)
        )
        if evidence.evidence_digest != stored_digest:
            raise ValueError(
                "physical event case spatial-evidence digest mismatch"
            )
        result.append(evidence)
    return tuple(result)


def _create_episode_impacts_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE episode_impacts (
            episode_id TEXT NOT NULL,
            lead_minutes INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            input_offset_minutes INTEGER NOT NULL,
            direct_path_status TEXT NOT NULL,
            forecast_score REAL,
            direct_sensitivity_norm REAL,
            direct_impact REAL,
            direct_normalized_reward REAL,
            PRIMARY KEY (
                episode_id, lead_minutes, metric_name,
                input_offset_minutes
            ),
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                DEFERRABLE INITIALLY DEFERRED
        )
        """
    )


def _owned_episode_copy(episode: SensitivityEpisode) -> SensitivityEpisode:
    """Take one private snapshot before creating any persistent artifact."""

    replacements: dict[str, Any] = {}
    for field in fields(episode.snapshot):
        value = getattr(episode.snapshot, field.name)
        if isinstance(value, Tensor):
            replacements[field.name] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            replacements[field.name] = dict(value)
    direct_replacements: dict[str, Tensor] = {}
    for field in fields(episode.snapshot.direct):
        value = getattr(episode.snapshot.direct, field.name)
        if isinstance(value, Tensor):
            direct_replacements[field.name] = value.detach().cpu().clone()
    replacements["direct"] = replace(
        episode.snapshot.direct,
        **direct_replacements,
    )
    owned_snapshot = replace(episode.snapshot, **replacements)
    return replace(
        episode,
        snapshot=owned_snapshot,
        action_features=tuple(episode.action_features),
    )


def _snapshot_arrays(episode: SensitivityEpisode) -> dict[str, NDArray[Any]]:
    snapshot = episode.snapshot
    tensor_arrays = {
        "context_features": snapshot.context_features,
        "analysis_control": snapshot.analysis_control,
        "forecast_scores": snapshot.forecast_scores,
        "metric_available": snapshot.metric_available,
        "control_sensitivity": snapshot.control_sensitivity,
        "forecast_sensitivity": snapshot.forecast_sensitivity,
        "forecast_cap_active_mask": snapshot.forecast_cap_active_mask,
        "forecast_confidence": snapshot.forecast_confidence,
        "path_evidence_by_metric": snapshot.path_evidence_by_metric,
        "observation_source_fraction_by_metric": (
            snapshot.observation_source_fraction_by_metric
        ),
        "observation_verified_evidence_by_metric": (
            snapshot.observation_verified_evidence_by_metric
        ),
        "background_verified_evidence_by_metric": (
            snapshot.background_verified_evidence_by_metric
        ),
        "direct_observation_sensitivity": snapshot.direct.maps,
        "direct_observation_sensitivity_norm": snapshot.direct.norm,
        "tile_direct_sensitivity_norm": snapshot.direct.tile_norm,
        "latest_sensitivity_mask": snapshot.latest_sensitivity_mask,
    }
    optional_arrays = {
        "tile_whitened_direct_sensitivity_norm": (
            snapshot.direct.whitened_tile_norm
        ),
        "direct_observation_impact": snapshot.direct.impact,
        "tile_direct_observation_impact": snapshot.direct.tile_impact,
        "observation_std_dbz": snapshot.observation_std_dbz,
        "observation_innovation_dbz": snapshot.observation_innovation_dbz,
        "observation_innovation_mask": snapshot.observation_innovation_mask,
    }
    tensor_arrays.update(
        {
            name: value
            for name, value in optional_arrays.items()
            if value is not None
        }
    )
    arrays = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in tensor_arrays.items()
    }
    arrays["action_features"] = np.asarray(
        episode.action_features,
        dtype=np.float64,
    )
    return arrays


def _episode_manifest(
    episode: SensitivityEpisode,
    arrays: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    snapshot = episode.snapshot
    return {
        "schema_version": _EPISODE_SCHEMA_VERSION,
        "episode_id": episode.episode_id,
        "issue_time": episode.issue_time,
        "radar_id": episode.radar_id,
        "forecast_run_digest": episode.forecast_run_digest,
        "verification_contract": snapshot.verification_contract,
        "verification_bundle_digest": snapshot.verification_bundle_digest,
        "verification_lineage_complete": (
            snapshot.verification_lineage_complete
        ),
        "verification_valid_times": (
            None
            if snapshot.verification_valid_times is None
            else list(snapshot.verification_valid_times)
        ),
        "verification_grid_contract_digest": (
            snapshot.verification_grid_contract_digest
        ),
        "verification_radar_product_digest": (
            snapshot.verification_radar_product_digest
        ),
        "verification_qc_pipeline_digest": (
            snapshot.verification_qc_pipeline_digest
        ),
        "contract": asdict(episode.contract),
        "contract_hash": episode.contract.digest,
        "metric_names": list(snapshot.metric_names),
        "lead_minutes": list(snapshot.lead_minutes),
        "full_map_lead_minutes": list(snapshot.full_map_lead_minutes),
        "tile_size": snapshot.tile_size,
        "tile_shape_yx": list(snapshot.tile_shape_yx),
        "context_feature_names": list(snapshot.context_feature_names),
        "sensitivity_scope": _sensitivity_scope(),
        "units": {
            "forecast_sensitivity": "d_error/d_linear_echo",
            "control_sensitivity": (
                "d_error/d_[dy_px,dx_px,log_growth_per_step]"
            ),
            "direct_observation_sensitivity": "d_error/d_dbz",
            "direct_observation_impact": "error_change",
            "forecast_confidence": "dimensionless_evidence_score",
            "path_evidence_by_metric": "dimensionless",
            "observation_source_fraction_by_metric": "dimensionless",
            "observation_verified_evidence_by_metric": "dimensionless",
            "background_verified_evidence_by_metric": "dimensionless",
        },
        "whitened_tile_norm_available": (
            snapshot.whitened_tile_norm_available
        ),
        "indirect_observation_sensitivity_available": False,
        "total_observation_sensitivity_available": False,
        "impact_available": snapshot.impact_available,
        "reward_available": snapshot.reward_available,
        "baseline_lineage_available": False,
        "reward_epsilon": snapshot.reward_epsilon,
        "trust_components": snapshot.trust_components,
        "trust_score": snapshot.trust_score,
        "promotion_eligible": False,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        },
    }


def _sensitivity_scope() -> dict[str, str]:
    return {
        "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
        "indirect_analysis_path": (
            "unavailable_implicit_variational_fso_not_implemented"
        ),
        "total_observation_sensitivity": "unavailable",
    }


def _verify_manifest(
    manifest: dict[str, Any],
    episode_id: str,
    row: sqlite3.Row,
) -> None:
    schema_version = manifest.get("schema_version")
    if (
        type(schema_version) is not int
        or not 1 <= schema_version <= _EPISODE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported episode schema")
    if manifest.get("episode_id") != episode_id:
        raise ValueError("manifest episode_id disagrees with the index")
    forecast_run_digest = manifest.get("forecast_run_digest")
    if schema_version >= 4:
        if (
            not isinstance(forecast_run_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", forecast_run_digest) is None
        ):
            raise ValueError("manifest forecast_run_digest is invalid")
    elif "forecast_run_digest" in manifest:
        raise ValueError(
            "forecast run provenance does not match the episode schema"
        )
    verification_keys = {
        "verification_contract",
        "verification_bundle_digest",
        "verification_lineage_complete",
        "verification_valid_times",
        "verification_grid_contract_digest",
        "verification_radar_product_digest",
        "verification_qc_pipeline_digest",
    }
    if schema_version >= 17:
        _validate_verification_manifest(manifest)
    elif verification_keys & set(manifest):
        raise ValueError(
            "verification provenance does not match the episode schema"
        )
    if schema_version >= 18:
        tile_shape = manifest.get("tile_shape_yx")
        if (
            not isinstance(tile_shape, list)
            or len(tile_shape) != 2
            or any(type(value) is not int or value <= 0 for value in tile_shape)
        ):
            raise ValueError("manifest physical tile shape is invalid")
    elif "tile_shape_yx" in manifest:
        raise ValueError("physical tile shape does not match episode schema")
    contract_values = manifest.get("contract")
    if not isinstance(contract_values, dict):
        raise ValueError("manifest contains an invalid contract")
    if schema_version < 3:
        if (
            set(contract_values) != _LEGACY_MODEL_CONTRACT_FIELDS_V1_V2
            or not all(contract_values.values())
        ):
            raise ValueError("manifest contains an invalid contract")
        contract_hash = _json_digest(
            {
                "contract_schema_version": schema_version,
                **contract_values,
            }
        )
    else:
        try:
            contract = ModelContract(**contract_values)
        except (TypeError, ValueError) as error:
            raise ValueError("manifest contains an invalid contract") from error
        contract_hash = _model_contract_digest(
            contract,
            schema_version=_MODEL_CONTRACT_SCHEMA_BY_EPISODE_SCHEMA.get(
                schema_version,
                _MODEL_CONTRACT_SCHEMA_VERSION,
            ),
        )
    if contract_hash != manifest.get("contract_hash"):
        raise ValueError("manifest contract hash is invalid")
    if contract_hash != row["contract_hash"]:
        raise ValueError("contract hash disagrees with the index")
    if schema_version >= 17 and manifest.get(
        "verification_lineage_complete"
    ):
        if manifest.get("verification_grid_contract_digest") != (
            contract_values.get("grid_time_contract_digest")
        ):
            raise ValueError(
                "manifest verification grid disagrees with the forecast grid"
            )
        expected_times = _expected_verification_times(
            manifest.get("issue_time"),
            manifest.get("lead_minutes"),
        )
        if tuple(manifest.get("verification_valid_times", ())) != (
            expected_times
        ):
            raise ValueError(
                "manifest verification times disagree with issue and leads"
            )
    if manifest.get("indirect_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain indirect sensitivity")
    if manifest.get("total_observation_sensitivity_available") is not False:
        raise ValueError("M0 episodes cannot contain total sensitivity")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("M0 episodes cannot be promoted automatically")
    trust_components = manifest.get("trust_components")
    expected_trust_components = (
        _TRUST_COMPONENTS_V16
        if schema_version >= 16
        else _TRUST_COMPONENTS_V13
    )
    if schema_version >= 13 and (
        not isinstance(trust_components, dict)
        or set(trust_components) != expected_trust_components
    ):
        raise ValueError("manifest trust component contract is invalid")
    if schema_version >= 15 and (
        manifest.get("baseline_lineage_available") is not False
        or manifest.get("reward_available") is not False
    ):
        raise ValueError(
            "schema 15 and later require fail-closed baseline reward lineage"
        )
    reward_epsilon = manifest.get("reward_epsilon")
    if (
        not isinstance(reward_epsilon, (int, float))
        or isinstance(reward_epsilon, bool)
        or not math.isfinite(reward_epsilon)
        or reward_epsilon <= 0
    ):
        raise ValueError("manifest reward_epsilon must be positive")
    if row["indirect_observation_sensitivity_available"] != 0:
        raise ValueError("index contains an invalid indirect sensitivity flag")
    if not isinstance(manifest.get("arrays"), dict):
        raise ValueError("manifest array schema is missing")
    _verify_manifest_layout(manifest)


def _validate_verification_manifest(manifest: dict[str, Any]) -> None:
    digest = manifest.get("verification_bundle_digest")
    if not isinstance(digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", digest
    ) is None:
        raise ValueError("manifest verification bundle digest is invalid")
    complete = manifest.get("verification_lineage_complete")
    if type(complete) is not bool:
        raise ValueError("manifest verification lineage flag is invalid")
    valid_times = manifest.get("verification_valid_times")
    lineage_digests = tuple(
        manifest.get(name)
        for name in (
            "verification_grid_contract_digest",
            "verification_radar_product_digest",
            "verification_qc_pipeline_digest",
        )
    )
    if not complete:
        if manifest.get("verification_contract") != (
            "legacy-verification-tensor-v1"
        ):
            raise ValueError("manifest incomplete verification contract is invalid")
        if valid_times is not None or any(
            value is not None for value in lineage_digests
        ):
            raise ValueError("manifest incomplete verification claims lineage")
        return
    if manifest.get("verification_contract") != (
        "radar-verification-bundle-v1"
    ):
        raise ValueError("manifest complete verification contract is invalid")
    if (
        not isinstance(valid_times, list)
        or not valid_times
        or any(not isinstance(value, str) or not value for value in valid_times)
    ):
        raise ValueError("manifest verification valid times are invalid")
    parsed_times = tuple(
        _canonical_utc_datetime(value, "verification valid time")
        for value in valid_times
    )
    canonical_times = [
        value.isoformat().replace("+00:00", "Z") for value in parsed_times
    ]
    if valid_times != canonical_times or any(
        later <= earlier
        for earlier, later in zip(parsed_times, parsed_times[1:])
    ):
        raise ValueError(
            "manifest verification valid times must be canonical and increasing"
        )
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in lineage_digests
    ):
        raise ValueError("manifest verification lineage digest is invalid")


def _canonical_utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 string")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a timezone-aware ISO-8601 string"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _expected_verification_times(
    issue_time: object,
    lead_minutes: object,
) -> tuple[str, ...]:
    issue = _canonical_utc_datetime(issue_time, "issue_time")
    if (
        not isinstance(lead_minutes, (tuple, list))
        or not lead_minutes
        or any(type(value) is not int or value <= 0 for value in lead_minutes)
    ):
        raise ValueError("lead_minutes must contain positive integers")
    return tuple(
        (issue + timedelta(minutes=value))
        .isoformat()
        .replace("+00:00", "Z")
        for value in lead_minutes
    )


def _verify_manifest_layout(manifest: dict[str, Any]) -> None:
    arrays = manifest["arrays"]
    schema_version = manifest["schema_version"]
    legacy_core = {
        "context_features",
        "analysis_control",
        "forecast_scores",
        "metric_available",
        "control_sensitivity",
        "forecast_sensitivity",
        "forecast_cap_active_mask",
        "direct_observation_sensitivity",
        "direct_observation_sensitivity_norm",
        "tile_direct_sensitivity_norm",
        "latest_sensitivity_mask",
        "action_features",
    }
    legacy_evidence_arrays = {
        "forecast_confidence",
        "path_evidence_by_metric",
        "observation_evidence_by_metric",
    }
    joint_evidence_arrays = {
        "forecast_confidence",
        "path_evidence_by_metric",
        "observation_source_fraction_by_metric",
        "observation_verified_evidence_by_metric",
        "background_verified_evidence_by_metric",
    }
    if schema_version >= 16:
        required_evidence_arrays = joint_evidence_arrays
    elif schema_version >= 14:
        required_evidence_arrays = legacy_evidence_arrays
    else:
        required_evidence_arrays = set()
    core = legacy_core | required_evidence_arrays
    legacy_optional = {
        "tile_whitened_direct_sensitivity_norm",
        "direct_observation_impact",
        "tile_direct_observation_impact",
        "direct_normalized_reward",
        "observation_std_dbz",
        "observation_innovation_dbz",
        "observation_innovation_mask",
        "baseline_scores",
    }
    optional = set(legacy_optional)
    if schema_version < 14:
        optional |= legacy_evidence_arrays | joint_evidence_arrays
    if schema_version >= 15:
        optional -= {"direct_normalized_reward", "baseline_scores"}
    array_names = set(arrays)
    if schema_version == 1:
        context_names = _SCHEMA_ONE_CONTEXT_FEATURE_NAMES
        leads = manifest.get("lead_minutes")
        if (
            not isinstance(leads, list)
            or not leads
            or type(leads[0]) is not int
            or leads[0] <= 0
        ):
            raise ValueError("schema 1 lead_minutes are invalid")
        if array_names not in (
            core | legacy_optional,
            core | legacy_optional | joint_evidence_arrays,
            core | optional,
        ):
            raise ValueError("manifest arrays do not match episode schema 1")
        interval = leads[0]
        expected_scope = {
            f"input_{-2 * interval}_minutes": "no_direct_forecast_path",
            f"input_{-interval}_minutes": "no_direct_forecast_path",
            "input_0_minutes": "partial_direct_latest_dbz_fixed_control",
            "indirect_analysis_path": (
                "unavailable_implicit_variational_fso_not_implemented"
            ),
            "total_observation_sensitivity": "unavailable",
        }
    elif schema_version in (2, 3, 4):
        context_names = _SCHEMA_TWO_TO_FOUR_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 5:
        context_names = _SCHEMA_FIVE_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 5"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 6:
        context_names = _SCHEMA_SIX_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 6"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 7:
        context_names = _SCHEMA_SEVEN_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 7"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 8:
        context_names = _SCHEMA_EIGHT_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 8"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version == 9:
        context_names = _SCHEMA_NINE_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 9"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version in (10, 11):
        context_names = _SCHEMA_TEN_CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                "manifest arrays do not match episode schema 10"
            )
        expected_scope = _sensitivity_scope()
    elif schema_version in (12, 13):
        context_names = CONTEXT_FEATURE_NAMES_V13
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
        expected_scope = _sensitivity_scope()
    else:
        context_names = CONTEXT_FEATURE_NAMES
        if not core <= array_names <= core | optional:
            raise ValueError(
                f"manifest arrays do not match episode schema {schema_version}"
            )
        expected_scope = _sensitivity_scope()

    if manifest.get("context_feature_names") != list(context_names):
        raise ValueError("context features do not match the episode schema")
    if _declared_array_shape(arrays, "context_features") != (
        len(context_names),
    ):
        raise ValueError("context feature shape does not match the episode schema")
    ranks = {
        "direct_observation_sensitivity": (5, 4, 2),
        "direct_observation_sensitivity_norm": (3, 2, 2),
        "tile_direct_sensitivity_norm": (5, 4, 2),
        "tile_whitened_direct_sensitivity_norm": (5, 4, 2),
        "direct_observation_impact": (3, 2, 2),
        "tile_direct_observation_impact": (5, 4, 2),
        "direct_normalized_reward": (3, 2, 2),
        "observation_std_dbz": (3, 2, 0),
        "observation_innovation_dbz": (3, 2, 0),
        "observation_innovation_mask": (3, 2, 0),
    }
    for name, (v1_rank, v2_rank, input_axis) in ranks.items():
        if name not in arrays:
            continue
        shape = _declared_array_shape(arrays, name)
        expected_rank = v1_rank if schema_version == 1 else v2_rank
        if len(shape) != expected_rank:
            raise ValueError(
                f"{name} does not match episode schema {schema_version}"
            )
        if schema_version == 1 and shape[input_axis] != 3:
            raise ValueError(f"{name} does not preserve three input times")
    if manifest.get("sensitivity_scope") != expected_scope:
        raise ValueError("sensitivity_scope does not match the episode schema")


def _declared_array_shape(
    arrays: dict[str, Any],
    name: str,
) -> tuple[int, ...]:
    declaration = arrays.get(name)
    if not isinstance(declaration, dict):
        raise ValueError(f"manifest array declaration is missing: {name}")
    shape = declaration.get("shape")
    if not isinstance(shape, list) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError(f"manifest array shape is invalid: {name}")
    return tuple(shape)


def _verify_array_schema(
    arrays_path: Path,
    expected_schema: dict[str, Any],
) -> None:
    with np.load(arrays_path, allow_pickle=False) as data:
        if set(data.files) != set(expected_schema):
            raise ValueError("NPZ array names disagree with the manifest")
        for name in data.files:
            array = data[name]
            expected = expected_schema[name]
            if list(array.shape) != expected.get("shape"):
                raise ValueError(f"array shape mismatch: {name}")
            if str(array.dtype) != expected.get("dtype"):
                raise ValueError(f"array dtype mismatch: {name}")


def _validate_m0_snapshot(snapshot: SensitivitySnapshot) -> None:
    for name, digest in (
        ("forecast_run_digest", snapshot.forecast_run_digest),
        ("nowcast_config_digest", snapshot.nowcast_config_digest),
        ("sensitivity_config_digest", snapshot.sensitivity_config_digest),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name} must be a SHA-256 digest")
    grid_digest = snapshot.grid_time_contract_digest
    if grid_digest is not None and re.fullmatch(
        r"[0-9a-f]{64}", grid_digest
    ) is None:
        raise ValueError(
            "grid_time_contract_digest must be a SHA-256 digest"
        )
    verification_manifest = {
        "verification_contract": snapshot.verification_contract,
        "verification_bundle_digest": snapshot.verification_bundle_digest,
        "verification_lineage_complete": (
            snapshot.verification_lineage_complete
        ),
        "verification_valid_times": (
            None
            if snapshot.verification_valid_times is None
            else list(snapshot.verification_valid_times)
        ),
        "verification_grid_contract_digest": (
            snapshot.verification_grid_contract_digest
        ),
        "verification_radar_product_digest": (
            snapshot.verification_radar_product_digest
        ),
        "verification_qc_pipeline_digest": (
            snapshot.verification_qc_pipeline_digest
        ),
    }
    _validate_verification_manifest(verification_manifest)
    if not math.isfinite(snapshot.trust_score):
        raise ValueError("trust_score must be finite")
    if snapshot.baseline_scores is not None or snapshot.reward_available:
        raise ValueError(
            "normalized reward requires a verified baseline lineage contract"
        )
    if (
        not isinstance(snapshot.reward_epsilon, (int, float))
        or isinstance(snapshot.reward_epsilon, bool)
        or not math.isfinite(snapshot.reward_epsilon)
        or snapshot.reward_epsilon <= 0
    ):
        raise ValueError("reward_epsilon must be positive")

    leads = snapshot.lead_minutes
    metrics = snapshot.metric_names
    selected_leads = snapshot.full_map_lead_minutes
    if not leads or any(
        type(value) is not int or value <= 0 for value in leads
    ):
        raise ValueError("lead_minutes must be positive integers")
    if tuple(sorted(set(leads))) != leads:
        raise ValueError("lead_minutes must be unique and increasing")
    interval = leads[0]
    if leads != tuple(interval * step for step in range(1, len(leads) + 1)):
        raise ValueError("lead_minutes must use one uniform interval")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metric_names must be non-empty and unique")
    if any(not isinstance(name, str) or not name for name in metrics):
        raise ValueError("metric names must be non-empty strings")
    if (
        not snapshot.context_feature_names
        or len(snapshot.context_feature_names)
        != len(set(snapshot.context_feature_names))
    ):
        raise ValueError("context feature names must be non-empty and unique")
    if (
        len(set(selected_leads)) != len(selected_leads)
        or not set(selected_leads).issubset(leads)
        or any(type(value) is not int for value in selected_leads)
    ):
        raise ValueError("full-map leads must be unique forecast leads")
    if type(snapshot.tile_size) is not int or snapshot.tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if (
        len(snapshot.tile_shape_yx) != 2
        or any(type(value) is not int or value <= 0 for value in snapshot.tile_shape_yx)
        or snapshot.tile_size != max(snapshot.tile_shape_yx)
    ):
        raise ValueError("tile_shape_yx must contain two positive dimensions")

    _require_bool_tensor(
        "latest_sensitivity_mask",
        snapshot.latest_sensitivity_mask,
    )
    if snapshot.latest_sensitivity_mask.ndim != 2:
        raise ValueError("latest_sensitivity_mask must be two-dimensional")
    height, width = snapshot.latest_sensitivity_mask.shape
    lead_count = len(leads)
    metric_count = len(metrics)
    selected_count = len(selected_leads)
    tile_rows = math.ceil(height / snapshot.tile_shape_yx[0])
    tile_columns = math.ceil(width / snapshot.tile_shape_yx[1])

    core_tensors = {
        "context_features": (len(snapshot.context_feature_names),),
        "analysis_control": (3,),
        "forecast_scores": (lead_count, metric_count),
        "metric_available": (lead_count, metric_count),
        "control_sensitivity": (lead_count, metric_count, 3),
        "forecast_sensitivity": (
            selected_count,
            metric_count,
            height,
            width,
        ),
        "forecast_cap_active_mask": (selected_count, height, width),
        "forecast_confidence": (lead_count, height, width),
        "path_evidence_by_metric": (lead_count, metric_count),
        "observation_source_fraction_by_metric": (
            lead_count,
            metric_count,
        ),
        "observation_verified_evidence_by_metric": (
            lead_count,
            metric_count,
        ),
        "background_verified_evidence_by_metric": (
            lead_count,
            metric_count,
        ),
        "direct_observation_sensitivity": (
            selected_count,
            metric_count,
            height,
            width,
        ),
        "direct_observation_sensitivity_norm": (
            lead_count,
            metric_count,
        ),
        "tile_direct_sensitivity_norm": (
            lead_count,
            metric_count,
            tile_rows,
            tile_columns,
        ),
        "latest_sensitivity_mask": (height, width),
    }
    core_values = {
        "context_features": snapshot.context_features,
        "analysis_control": snapshot.analysis_control,
        "forecast_scores": snapshot.forecast_scores,
        "metric_available": snapshot.metric_available,
        "control_sensitivity": snapshot.control_sensitivity,
        "forecast_sensitivity": snapshot.forecast_sensitivity,
        "forecast_cap_active_mask": snapshot.forecast_cap_active_mask,
        "forecast_confidence": snapshot.forecast_confidence,
        "path_evidence_by_metric": snapshot.path_evidence_by_metric,
        "observation_source_fraction_by_metric": (
            snapshot.observation_source_fraction_by_metric
        ),
        "observation_verified_evidence_by_metric": (
            snapshot.observation_verified_evidence_by_metric
        ),
        "background_verified_evidence_by_metric": (
            snapshot.background_verified_evidence_by_metric
        ),
        "direct_observation_sensitivity": snapshot.direct.maps,
        "direct_observation_sensitivity_norm": snapshot.direct.norm,
        "tile_direct_sensitivity_norm": snapshot.direct.tile_norm,
        "latest_sensitivity_mask": snapshot.latest_sensitivity_mask,
    }
    for name, expected in core_tensors.items():
        value = core_values[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")

    for name, value in core_values.items():
        if name in {
            "metric_available",
            "forecast_cap_active_mask",
            "latest_sensitivity_mask",
        }:
            _require_bool_tensor(name, value)
        else:
            _require_float_tensor(name, value)

    _require_finite("context_features", snapshot.context_features)
    _require_finite("analysis_control", snapshot.analysis_control)
    if not bool(
        torch.all(
            (snapshot.forecast_confidence >= 0)
            & (snapshot.forecast_confidence <= 1)
        )
    ):
        raise ValueError("forecast_confidence must be in [0, 1]")

    available = snapshot.metric_available
    if not bool(torch.all(torch.isfinite(snapshot.forecast_scores[available]))):
        raise ValueError("available forecast scores must be finite")
    if not bool(torch.all(torch.isnan(snapshot.forecast_scores[~available]))):
        raise ValueError("unavailable forecast scores must be NaN")
    if not bool(
        torch.all(torch.isfinite(snapshot.control_sensitivity[available]))
    ):
        raise ValueError("available control sensitivities must be finite")
    if not bool(
        torch.all(torch.isnan(snapshot.control_sensitivity[~available]))
    ):
        raise ValueError("unavailable control sensitivities must be NaN")
    evidence_values = (
        snapshot.path_evidence_by_metric,
        snapshot.observation_source_fraction_by_metric,
        snapshot.observation_verified_evidence_by_metric,
        snapshot.background_verified_evidence_by_metric,
    )
    for value in evidence_values:
        finite = torch.isfinite(value)
        if not bool(torch.all((value[finite] >= 0) & (value[finite] <= 1))):
            raise ValueError("metric evidence must be in [0, 1]")
        if bool(torch.any(finite & ~available)):
            raise ValueError("unavailable metrics cannot retain evidence")
    joint_evidence_available = (
        torch.isfinite(snapshot.path_evidence_by_metric)
        & torch.isfinite(snapshot.observation_verified_evidence_by_metric)
        & torch.isfinite(snapshot.background_verified_evidence_by_metric)
    )
    if not torch.allclose(
        (
            snapshot.observation_verified_evidence_by_metric
            + snapshot.background_verified_evidence_by_metric
        )[joint_evidence_available],
        snapshot.path_evidence_by_metric[joint_evidence_available],
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("source evidence channels do not close")

    latest_norm = snapshot.direct.norm
    latest_tile_norm = snapshot.direct.tile_norm
    if not bool(torch.all(torch.isfinite(latest_norm[available]))):
        raise ValueError("available latest-frame norms must be finite")
    if not bool(torch.all(torch.isnan(latest_norm[~available]))):
        raise ValueError("unavailable latest-frame norms must be NaN")
    combined_tile_norm = torch.sqrt(
        torch.sum(latest_tile_norm.square(), dim=(-1, -2))
    )
    if not torch.allclose(
        combined_tile_norm[available],
        latest_norm[available],
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("tile norms and whole-field norms disagree")
    if not bool(torch.all(torch.isnan(latest_tile_norm[~available]))):
        raise ValueError("unavailable latest-frame tile norms must be NaN")
    if snapshot.whitened_tile_norm_available:
        whitened_latest = snapshot.direct.whitened_tile_norm
        if whitened_latest is None:
            raise ValueError("whitened summaries require whitened tile norms")
        expected = (lead_count, metric_count, tile_rows, tile_columns)
        if tuple(whitened_latest.shape) != expected:
            raise ValueError(
                "tile_whitened_direct_sensitivity_norm "
                f"must have shape {expected}"
            )
        _require_float_tensor(
            "tile_whitened_direct_sensitivity_norm",
            whitened_latest,
        )
        observation_std = snapshot.observation_std_dbz
        if observation_std is None:
            raise ValueError("whitened summaries require observation_std_dbz")
        if tuple(observation_std.shape) != (height, width):
            raise ValueError("observation_std_dbz must match the latest frame")
        _require_float_tensor("observation_std_dbz", observation_std)
        _require_finite("observation_std_dbz", observation_std)
        if bool(torch.any(observation_std <= 0)):
            raise ValueError("observation_std_dbz must be positive")
        if not bool(torch.all(torch.isfinite(whitened_latest[available]))):
            raise ValueError("available whitened tile norms must be finite")
        if not bool(torch.all(torch.isnan(whitened_latest[~available]))):
            raise ValueError("unavailable whitened tile norms must be NaN")
    elif snapshot.observation_std_dbz is not None:
        raise ValueError(
            "observation_std_dbz requires whitened sensitivity summaries"
        )
    for position, lead in enumerate(selected_leads):
        lead_index = leads.index(lead)
        for metric_index in range(metric_count):
            is_available = bool(available[lead_index, metric_index])
            forecast_map = snapshot.forecast_sensitivity[
                position, metric_index
            ]
            direct_map = snapshot.direct.maps[position, metric_index]
            if is_available:
                _require_finite("forecast sensitivity map", forecast_map)
                _require_finite("direct sensitivity map", direct_map)
                expected_norm = torch.linalg.vector_norm(direct_map)
                stored_norm = latest_norm[lead_index, metric_index]
                if not torch.allclose(
                    expected_norm,
                    stored_norm,
                    rtol=1.0e-5,
                    atol=1.0e-7,
                ):
                    raise ValueError("direct map and norm disagree")
            elif not bool(
                torch.all(torch.isnan(forecast_map))
                and torch.all(torch.isnan(direct_map))
            ):
                raise ValueError("unavailable sensitivity maps must be NaN")

    innovation = snapshot.observation_innovation_dbz
    innovation_mask = snapshot.observation_innovation_mask
    if (innovation is None) != (innovation_mask is None):
        raise ValueError("innovation value and mask availability must agree")
    if innovation is not None and innovation_mask is not None:
        if tuple(innovation.shape) != (height, width):
            raise ValueError("observation_innovation_dbz must match latest frame")
        if tuple(innovation_mask.shape) != (height, width):
            raise ValueError("observation_innovation_mask must match latest frame")
        _require_float_tensor("observation_innovation_dbz", innovation)
        _require_bool_tensor("observation_innovation_mask", innovation_mask)
        if not bool(torch.all(torch.isfinite(innovation[innovation_mask]))):
            raise ValueError("valid innovations must be finite")
        if not bool(torch.all(torch.isnan(innovation[~innovation_mask]))):
            raise ValueError("invalid innovations must be NaN")

    if snapshot.impact_available:
        if snapshot.observation_innovation_mask is None or not bool(
            torch.any(
                snapshot.observation_innovation_mask
                & snapshot.latest_sensitivity_mask
            )
        ):
            raise ValueError("impact requires a valid latest-frame innovation")
        if not bool(torch.any(snapshot.metric_available)):
            raise ValueError("impact requires an available metric")
        latest_impact = snapshot.direct.impact
        latest_tile_impact = snapshot.direct.tile_impact
        if latest_impact is None or latest_tile_impact is None:
            raise ValueError("direct impact requires impact and tile impact")
        if tuple(latest_impact.shape) != (lead_count, metric_count):
            raise ValueError("direct_observation_impact has invalid shape")
        expected = (lead_count, metric_count, tile_rows, tile_columns)
        if tuple(latest_tile_impact.shape) != expected:
            raise ValueError("tile_direct_observation_impact has invalid shape")
        _require_float_tensor("direct_observation_impact", latest_impact)
        _require_float_tensor(
            "tile_direct_observation_impact",
            latest_tile_impact,
        )
        if not bool(torch.all(torch.isfinite(latest_impact[available]))):
            raise ValueError("available latest-frame impacts must be finite")
        if not bool(torch.all(torch.isnan(latest_impact[~available]))):
            raise ValueError("unavailable latest-frame impacts must be NaN")
        if not torch.allclose(
            latest_tile_impact.sum(dim=(-1, -2))[available],
            latest_impact[available],
            rtol=1.0e-5,
            atol=1.0e-7,
        ):
            raise ValueError("tile impacts and whole-field impacts disagree")
    elif snapshot.direct.tile_impact is not None:
        raise ValueError("tile impact cannot exist without direct impact")
    if any(
        not math.isfinite(value)
        for value in snapshot.trust_components.values()
    ):
        raise ValueError("trust components must be finite")
    if set(snapshot.trust_components) != _TRUST_COMPONENTS_V16:
        raise ValueError("trust component contract is invalid")
    expected_trust = math.prod(snapshot.trust_components.values())
    if not math.isclose(
        snapshot.trust_score,
        expected_trust,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("trust_score must equal the component product")


def _require_float_tensor(name: str, value: Any) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")


def _require_bool_tensor(name: str, value: Any) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def _row_exists(
    connection: sqlite3.Connection,
    episode_id: str,
) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        is not None
    )


def _sqlite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _model_contract_digest(
    contract: ModelContract,
    *,
    schema_version: int,
) -> str:
    values = asdict(contract)
    if schema_version < 10:
        values.pop("grid_time_contract_digest")
    return _json_digest(
        {
            "contract_schema_version": schema_version,
            **values,
        }
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_durable_directory(
    *,
    temporary: Path,
    target: Path,
    durable_files: tuple[Path, ...],
    parent: Path,
) -> None:
    """Publish complete bytes before a SQLite row can reference them."""

    for path in durable_files:
        _fsync_file(path)
    _fsync_directory(temporary)
    renamed = False
    try:
        temporary.rename(target)
        renamed = True
        _fsync_directory(parent)
    except Exception:
        if renamed and target.exists():
            shutil.rmtree(target)
            _fsync_directory(parent)
        raise


def _validate_episode_id(episode_id: str) -> None:
    if (
        not episode_id
        or len(episode_id) > 128
        or episode_id in {".", ".."}
        or not _SAFE_ID.fullmatch(episode_id)
    ):
        raise ValueError(
            "episode_id must be 1-128 safe filename characters and not . or .."
        )
