"""Small authoritative registry for current scientific contract capability."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import cast

from ._digest import json_digest


_METRIC_EVIDENCE_LIFECYCLE_PROBE = (
    "tests/test_nowcast.py::NowcastTests::"
    "test_metric_domain_evidence_is_content_addressed_and_bounds_area"
)
_PROMOTION_LIFECYCLE_PROBE = (
    "tests/test_promotion.py::NeuralPriorPromotionTests::"
    "test_semantic_replay_generation_stops_before_operational_deployment"
)
_DEPLOYMENT_LINEAGE_LIFECYCLE_PROBE = (
    "tests/test_nowcast.py::NowcastTests::"
    "test_current_run_rejects_any_operational_deployment_claim"
)
_VERIFICATION_FSO_LIFECYCLE_PROBE = (
    "tests/test_sensitivity.py::VariationalFSOTests::"
    "test_approved_learning_policy_certifies_physical_branch"
)
_SEMANTIC_REPLAY_LIFECYCLE_PROBE = (
    "tests/test_promotion.py::NeuralPriorPromotionTests::"
    "test_full_product_semantic_replay_reaches_promotion_without_scorer_patch"
)
_METRIC_EVIDENCE_AUDIT_PROBE = (
    "tests/test_nowcast.py::NowcastTests::"
    "test_legacy_metric_domain_report_is_audit_only"
)
_PROMOTION_AUDIT_PROBE = (
    "tests/test_ledger.py::EpisodeLedgerTests::"
    "test_pr141_promotion_v31_loads_from_the_ledger_as_audit_only"
)
_DEPLOYMENT_LINEAGE_AUDIT_PROBE = (
    "tests/test_run_artifact.py::ForecastRunArtifactTests::"
    "test_v52_deployment_geometry_loads_as_audit_only"
)
_VERIFICATION_FSO_AUDIT_PROBE = (
    "tests/test_sensitivity.py::VariationalFSOTests::"
    "test_verification_bundle_capabilities_have_one_current_generation"
)
_SEMANTIC_REPLAY_AUDIT_PROBE = (
    "tests/test_ledger.py::EpisodeLedgerTests::"
    "test_replay_v19_through_v26_preserve_frozen_tensor_roles"
)
_FORECAST_RUN_AUDIT_PROBE = (
    "tests/test_run_artifact.py::ForecastRunArtifactTests::"
    "test_v52_deployment_geometry_loads_as_audit_only"
)
_HOLDOUT_PLAN_LIFECYCLE_PROBE = (
    "tests/test_promotion.py::NeuralPriorPromotionTests::"
    "test_physical_applicability_contract_versions_are_new_generations"
)
_HOLDOUT_PLAN_AUDIT_PROBE = (
    "tests/test_ledger.py::EpisodeLedgerTests::"
    "test_holdout_plans_v33_through_v36_load_as_cold_audit_fixtures"
)


@dataclass(frozen=True)
class ContractCapabilities:
    """Describe which generations can be built, audited, or acted upon."""

    current: str
    predecessor: str | None
    issuable: frozenset[str]
    audit_readable: frozenset[str]
    scientific_eligible: frozenset[str]
    operationally_accepted: frozenset[str]
    lifecycle_probe: str
    audit_generation_probes: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if self.current not in self.issuable:
            raise ValueError("current contract generation must be issuable")
        if not self.issuable <= self.audit_readable:
            raise ValueError("every issuable generation must be audit-readable")
        if not self.scientific_eligible <= self.issuable:
            raise ValueError("scientific generations must be constructible")
        if not self.operationally_accepted <= self.issuable:
            raise ValueError("operational generations must be constructible")
        if (
            self.predecessor is not None
            and self.predecessor not in self.audit_readable
        ):
            raise ValueError("predecessor generation must be audit-readable")
        probe_parts = self.lifecycle_probe.split("::")
        if (
            not self.lifecycle_probe
            or self.lifecycle_probe.strip() != self.lifecycle_probe
            or len(probe_parts) != 3
            or not probe_parts[0].startswith("tests/test_")
            or not probe_parts[0].endswith(".py")
            or any(not part for part in probe_parts)
        ):
            raise ValueError("contract lifecycle probe must be named")
        audit_probe_contracts = tuple(
            contract for contract, _ in self.audit_generation_probes
        )
        if (
            len(audit_probe_contracts) != len(set(audit_probe_contracts))
            or set(audit_probe_contracts) != (self.audit_readable - self.issuable)
        ):
            raise ValueError(
                "every audit-only generation must have one executable probe"
            )
        for contract, probe in self.audit_generation_probes:
            if not contract or not probe:
                raise ValueError("audit-generation probe is invalid")
            probe_parts = probe.split("::")
            if (
                len(probe_parts) != 3
                or not probe_parts[0].startswith("tests/test_")
                or not probe_parts[0].endswith(".py")
                or any(not part for part in probe_parts)
            ):
                raise ValueError("audit-generation probe must be named")


@dataclass(frozen=True)
class AuditGenerationFixture:
    """One frozen audit-readable generation and its executable decoder probe."""

    family: str
    contract: str
    fixture_path: str
    expected_type: str
    scientific_action_allowed: bool
    operational_action_allowed: bool
    decoder_probe: str

    def validate(self) -> None:
        probe_parts = self.decoder_probe.split("::")
        if (
            not self.family
            or not self.contract
            or not self.fixture_path.startswith("tests/fixtures/audit_generations/")
            or not self.fixture_path.endswith(".json")
            or not self.expected_type
            or type(self.scientific_action_allowed) is not bool
            or type(self.operational_action_allowed) is not bool
            or len(probe_parts) != 3
            or not probe_parts[0].startswith("tests/test_")
            or not probe_parts[0].endswith(".py")
            or any(not part for part in probe_parts)
        ):
            raise ValueError("audit generation fixture is invalid")


@dataclass(frozen=True)
class FrozenAuditGeneration:
    """Cold-decoded immutable bytes for one audit-readable generation.

    This envelope is deliberately non-actionable.  Family-specific probes may
    additionally reconstruct a richer historical type, but this common layer
    guarantees canonical bytes, original payload digest, and action denial for
    every generation declared audit-readable.
    """

    family: str
    contract: str
    expected_type: str
    decoder_probe: str
    payload: dict[str, object]
    payload_digest: str
    scientific_action_allowed: bool
    operational_action_allowed: bool

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        metadata: AuditGenerationFixture,
    ) -> "FrozenAuditGeneration":
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("cold audit fixture is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("cold audit fixture must be a JSON object")
        canonical = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        payload = value.get("payload")
        if (
            raw != canonical
            or set(value)
            != {
                "contract",
                "decoder_probe",
                "expected_type",
                "family",
                "fixture_contract",
                "operational_action_allowed",
                "payload",
                "payload_digest",
                "scientific_action_allowed",
            }
            or value.get("fixture_contract") != "audit-generation-cold-fixture-v1"
            or value.get("family") != metadata.family
            or value.get("contract") != metadata.contract
            or value.get("expected_type") != metadata.expected_type
            or value.get("decoder_probe") != metadata.decoder_probe
            or value.get("scientific_action_allowed")
            is not metadata.scientific_action_allowed
            or value.get("operational_action_allowed")
            is not metadata.operational_action_allowed
            or not isinstance(payload, dict)
            or payload.get("contract") != metadata.contract
            or payload.get("family") != metadata.family
            or value.get("payload_digest") != json_digest(payload)
        ):
            raise ValueError("cold audit fixture digest or contract is invalid")
        return cls(
            family=metadata.family,
            contract=metadata.contract,
            expected_type=metadata.expected_type,
            decoder_probe=metadata.decoder_probe,
            payload=cast(dict[str, object], payload),
            payload_digest=cast(str, value["payload_digest"]),
            scientific_action_allowed=metadata.scientific_action_allowed,
            operational_action_allowed=metadata.operational_action_allowed,
        )

    def require_scientific_action(self) -> None:
        if not self.scientific_action_allowed:
            raise ValueError("cold audit generation is not scientifically actionable")

    def require_operational_action(self) -> None:
        if not self.operational_action_allowed:
            raise OperationalDeploymentUnsupportedError(
                "cold audit generation is not operationally actionable"
            )


class OperationalDeploymentUnsupportedError(RuntimeError):
    """Scientific artifacts cannot authorize an operational deployment."""


CONTRACT_CAPABILITIES: dict[str, ContractCapabilities] = {
    "radar_metric_domain_evidence": ContractCapabilities(
        current="radar-metric-domain-evidence-v4",
        predecessor="radar-metric-domain-evidence-v3",
        issuable=frozenset({"radar-metric-domain-evidence-v4"}),
        audit_readable=frozenset(
            {
                "radar-metric-domain-evidence-v1",
                "radar-metric-domain-evidence-v2",
                "radar-metric-domain-evidence-v3",
                "radar-metric-domain-evidence-v4",
            }
        ),
        scientific_eligible=frozenset({"radar-metric-domain-evidence-v4"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_METRIC_EVIDENCE_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _METRIC_EVIDENCE_AUDIT_PROBE)
            for generation in (
                "radar-metric-domain-evidence-v1",
                "radar-metric-domain-evidence-v2",
                "radar-metric-domain-evidence-v3",
            )
        ),
    ),
    "neural_prior_promotion_evidence": ContractCapabilities(
        current="neural-prior-promotion-evidence-v32",
        predecessor="neural-prior-promotion-evidence-v31",
        issuable=frozenset({"neural-prior-promotion-evidence-v32"}),
        audit_readable=frozenset(
            {
                "neural-prior-promotion-evidence-v31",
                "neural-prior-promotion-evidence-v32",
            }
        ),
        scientific_eligible=frozenset({"neural-prior-promotion-evidence-v32"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_PROMOTION_LIFECYCLE_PROBE,
        audit_generation_probes=((
            "neural-prior-promotion-evidence-v31",
            _PROMOTION_AUDIT_PROBE,
        ),),
    ),
    "deployed_neural_prior_policy": ContractCapabilities(
        current="deployed-neural-prior-policy-v17",
        predecessor=None,
        issuable=frozenset({"deployed-neural-prior-policy-v17"}),
        audit_readable=frozenset({"deployed-neural-prior-policy-v17"}),
        scientific_eligible=frozenset(),
        operationally_accepted=frozenset(),
        lifecycle_probe=_PROMOTION_LIFECYCLE_PROBE,
    ),
    "neural_prior_deployment_lineage": ContractCapabilities(
        current="neural-prior-deployment-lineage-v19",
        predecessor="neural-prior-deployment-lineage-v18-audit",
        issuable=frozenset({"neural-prior-deployment-lineage-v19"}),
        audit_readable=frozenset(
            {
                "neural-prior-deployment-lineage-v18-audit",
                "neural-prior-deployment-lineage-v19-audit",
                "neural-prior-deployment-lineage-v19",
            }
        ),
        scientific_eligible=frozenset(),
        operationally_accepted=frozenset(),
        lifecycle_probe=_DEPLOYMENT_LINEAGE_LIFECYCLE_PROBE,
        audit_generation_probes=(
            (
                "neural-prior-deployment-lineage-v18-audit",
                _DEPLOYMENT_LINEAGE_AUDIT_PROBE,
            ),
            (
                "neural-prior-deployment-lineage-v19-audit",
                _DEPLOYMENT_LINEAGE_AUDIT_PROBE,
            ),
        ),
    ),
    "radar_spatial_grid_identity": ContractCapabilities(
        current="radar-spatial-grid-identity-v6",
        predecessor="radar-spatial-grid-identity-v5",
        issuable=frozenset({"radar-spatial-grid-identity-v6"}),
        audit_readable=frozenset(
            {
                "radar-spatial-grid-identity-v5",
                "radar-spatial-grid-identity-v6",
            }
        ),
        scientific_eligible=frozenset({"radar-spatial-grid-identity-v6"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=((
            "radar-spatial-grid-identity-v5",
            _VERIFICATION_FSO_AUDIT_PROBE,
        ),),
    ),
    "mosaic_observation_source_registry": ContractCapabilities(
        current="mosaic-observation-source-registry-v7",
        predecessor="mosaic-observation-source-registry-v6",
        issuable=frozenset({"mosaic-observation-source-registry-v7"}),
        audit_readable=frozenset(
            {
                "mosaic-observation-source-registry-v6",
                "mosaic-observation-source-registry-v7",
            }
        ),
        scientific_eligible=frozenset(
            {"mosaic-observation-source-registry-v7"}
        ),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=((
            "mosaic-observation-source-registry-v6",
            _VERIFICATION_FSO_AUDIT_PROBE,
        ),),
    ),
    "radar_observation_geometry": ContractCapabilities(
        current="radar-observation-geometry-v7",
        predecessor="radar-observation-geometry-v6",
        issuable=frozenset({"radar-observation-geometry-v7"}),
        audit_readable=frozenset(
            {"radar-observation-geometry-v6", "radar-observation-geometry-v7"}
        ),
        scientific_eligible=frozenset({"radar-observation-geometry-v7"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=((
            "radar-observation-geometry-v6",
            _VERIFICATION_FSO_AUDIT_PROBE,
        ),),
    ),
    "verification_observation_error_plan": ContractCapabilities(
        current="verification-observation-error-plan-v16",
        predecessor="verification-observation-error-plan-v15",
        issuable=frozenset({"verification-observation-error-plan-v16"}),
        audit_readable=frozenset(
            {
                "verification-observation-error-plan-v14",
                "verification-observation-error-plan-v15",
                "verification-observation-error-plan-v16",
            }
        ),
        scientific_eligible=frozenset({"verification-observation-error-plan-v16"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _VERIFICATION_FSO_AUDIT_PROBE)
            for generation in (
                "verification-observation-error-plan-v14",
                "verification-observation-error-plan-v15",
            )
        ),
    ),
    "verification_bundle": ContractCapabilities(
        current="radar-verification-bundle-v22",
        predecessor="radar-verification-bundle-v21",
        issuable=frozenset({"radar-verification-bundle-v22"}),
        audit_readable=frozenset(
            {
                "radar-verification-bundle-v20",
                "radar-verification-bundle-v21",
                "radar-verification-bundle-v22",
            }
        ),
        scientific_eligible=frozenset({"radar-verification-bundle-v22"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _VERIFICATION_FSO_AUDIT_PROBE)
            for generation in (
                "radar-verification-bundle-v20",
                "radar-verification-bundle-v21",
            )
        ),
    ),
    "variational_fso": ContractCapabilities(
        current="p1-variational-fso-v28",
        predecessor="p1-variational-fso-v27",
        issuable=frozenset({"p1-variational-fso-v28"}),
        audit_readable=frozenset(
            {
                "p1-variational-fso-v26",
                "p1-variational-fso-v27",
                "p1-variational-fso-v28",
            }
        ),
        scientific_eligible=frozenset({"p1-variational-fso-v28"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _VERIFICATION_FSO_AUDIT_PROBE)
            for generation in (
                "p1-variational-fso-v26",
                "p1-variational-fso-v27",
            )
        ),
    ),
    "variational_fsoi": ContractCapabilities(
        current="p1-linearized-observation-impact-v24",
        predecessor="p1-linearized-observation-impact-v23",
        issuable=frozenset({"p1-linearized-observation-impact-v24"}),
        audit_readable=frozenset(
            {
                "p1-linearized-observation-impact-v22",
                "p1-linearized-observation-impact-v23",
                "p1-linearized-observation-impact-v24",
            }
        ),
        scientific_eligible=frozenset({"p1-linearized-observation-impact-v24"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _VERIFICATION_FSO_AUDIT_PROBE)
            for generation in (
                "p1-linearized-observation-impact-v22",
                "p1-linearized-observation-impact-v23",
            )
        ),
    ),
    "semantic_scoring_replay": ContractCapabilities(
        current="neural-prior-scoring-replay-bundle-v27",
        predecessor="neural-prior-scoring-replay-bundle-v26",
        issuable=frozenset({"neural-prior-scoring-replay-bundle-v27"}),
        audit_readable=frozenset(
            {
                "neural-prior-scoring-replay-bundle-v25",
                "neural-prior-scoring-replay-bundle-v26",
                "neural-prior-scoring-replay-bundle-v27",
            }
        ),
        scientific_eligible=frozenset({"neural-prior-scoring-replay-bundle-v27"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_SEMANTIC_REPLAY_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _SEMANTIC_REPLAY_AUDIT_PROBE)
            for generation in (
                "neural-prior-scoring-replay-bundle-v25",
                "neural-prior-scoring-replay-bundle-v26",
            )
        ),
    ),
    "forecast_run_artifact": ContractCapabilities(
        current="forecast-run-v72",
        predecessor="forecast-run-v71",
        issuable=frozenset({"forecast-run-v72"}),
        audit_readable=frozenset(
            {
                "forecast-run-v68",
                "forecast-run-v69",
                "forecast-run-v70",
                "forecast-run-v71",
                "forecast-run-v72",
            }
        ),
        scientific_eligible=frozenset({"forecast-run-v72"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_DEPLOYMENT_LINEAGE_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _FORECAST_RUN_AUDIT_PROBE)
            for generation in (
                "forecast-run-v68",
                "forecast-run-v69",
                "forecast-run-v70",
                "forecast-run-v71",
            )
        ),
    ),
    "neural_prior_holdout_plan": ContractCapabilities(
        current="neural-prior-holdout-plan-v37",
        predecessor="neural-prior-holdout-plan-v36",
        issuable=frozenset({"neural-prior-holdout-plan-v37"}),
        audit_readable=frozenset(
            {
                "neural-prior-holdout-plan-v33",
                "neural-prior-holdout-plan-v34",
                "neural-prior-holdout-plan-v35",
                "neural-prior-holdout-plan-v36",
                "neural-prior-holdout-plan-v37",
            }
        ),
        scientific_eligible=frozenset({"neural-prior-holdout-plan-v37"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_HOLDOUT_PLAN_LIFECYCLE_PROBE,
        audit_generation_probes=tuple(
            (generation, _HOLDOUT_PLAN_AUDIT_PROBE)
            for generation in (
                "neural-prior-holdout-plan-v33",
                "neural-prior-holdout-plan-v34",
                "neural-prior-holdout-plan-v35",
                "neural-prior-holdout-plan-v36",
            )
        ),
    ),
}


def _audit_fixture(
    family: str,
    contract: str,
    expected_type: str,
    decoder_probe: str,
) -> AuditGenerationFixture:
    filename = contract.replace("/", "_") + ".json"
    capabilities = CONTRACT_CAPABILITIES[family]
    return AuditGenerationFixture(
        family=family,
        contract=contract,
        fixture_path=f"tests/fixtures/audit_generations/{filename}",
        expected_type=expected_type,
        scientific_action_allowed=contract in capabilities.scientific_eligible,
        operational_action_allowed=contract in capabilities.operationally_accepted,
        decoder_probe=decoder_probe,
    )


# Historical entries are intentionally explicit; current entries are derived
# from the one registered authority.  The exact-set invariant below catches a
# newly declared historical generation without frozen bytes, a missing current
# envelope, and any stale extra fixture.
AUDIT_GENERATION_FIXTURES: tuple[AuditGenerationFixture, ...] = (
    *(
        _audit_fixture(
            "radar_metric_domain_evidence",
            f"radar-metric-domain-evidence-v{generation}",
            f"LegacyRadarMetricDomainEvidenceAuditV{generation}",
            _METRIC_EVIDENCE_AUDIT_PROBE,
        )
        for generation in (1, 2, 3)
    ),
    _audit_fixture(
        "neural_prior_promotion_evidence",
        "neural-prior-promotion-evidence-v31",
        "LegacyNeuralPriorPromotionEvidenceAuditV31",
        _PROMOTION_AUDIT_PROBE,
    ),
    _audit_fixture(
        "neural_prior_deployment_lineage",
        "neural-prior-deployment-lineage-v18-audit",
        "ForecastResult",
        _DEPLOYMENT_LINEAGE_AUDIT_PROBE,
    ),
    _audit_fixture(
        "neural_prior_deployment_lineage",
        "neural-prior-deployment-lineage-v19-audit",
        "ForecastResult",
        _DEPLOYMENT_LINEAGE_AUDIT_PROBE,
    ),
    _audit_fixture(
        "radar_spatial_grid_identity",
        "radar-spatial-grid-identity-v5",
        "RadarSpatialGridIdentity",
        _VERIFICATION_FSO_AUDIT_PROBE,
    ),
    _audit_fixture(
        "mosaic_observation_source_registry",
        "mosaic-observation-source-registry-v6",
        "MosaicObservationSourceRegistry",
        _VERIFICATION_FSO_AUDIT_PROBE,
    ),
    _audit_fixture(
        "radar_observation_geometry",
        "radar-observation-geometry-v6",
        "RadarObservationGeometryContract",
        _VERIFICATION_FSO_AUDIT_PROBE,
    ),
    *(
        _audit_fixture(
            "verification_observation_error_plan",
            f"verification-observation-error-plan-v{generation}",
            "VerificationObservationErrorPlan",
            _VERIFICATION_FSO_AUDIT_PROBE,
        )
        for generation in (14, 15)
    ),
    *(
        _audit_fixture(
            "verification_bundle",
            f"radar-verification-bundle-v{generation}",
            "VerificationBundle",
            _VERIFICATION_FSO_AUDIT_PROBE,
        )
        for generation in (20, 21)
    ),
    *(
        _audit_fixture(
            "variational_fso",
            f"p1-variational-fso-v{generation}",
            "VariationalFSO",
            _VERIFICATION_FSO_AUDIT_PROBE,
        )
        for generation in (26, 27)
    ),
    *(
        _audit_fixture(
            "variational_fsoi",
            f"p1-linearized-observation-impact-v{generation}",
            "VariationalFSOI",
            _VERIFICATION_FSO_AUDIT_PROBE,
        )
        for generation in (22, 23)
    ),
    *(
        _audit_fixture(
            "semantic_scoring_replay",
            f"neural-prior-scoring-replay-bundle-v{generation}",
            f"LegacyScoringReplayBundleManifestAuditV{generation}",
            _SEMANTIC_REPLAY_AUDIT_PROBE,
        )
        for generation in (25, 26)
    ),
    *(
        _audit_fixture(
            "forecast_run_artifact",
            f"forecast-run-v{generation}",
            "ForecastResult",
            _FORECAST_RUN_AUDIT_PROBE,
        )
        for generation in (68, 69, 70, 71)
    ),
    *(
        _audit_fixture(
            "neural_prior_holdout_plan",
            f"neural-prior-holdout-plan-v{generation}",
            f"LegacyNeuralPriorHoldoutPlanV{generation}Audit",
            _HOLDOUT_PLAN_AUDIT_PROBE,
        )
        for generation in (33, 34, 35, 36)
    ),
    *(
        _audit_fixture(
            family,
            capabilities.current,
            {
                "radar_metric_domain_evidence": "RadarMetricDomainEvidence",
                "neural_prior_promotion_evidence": "NeuralPriorPromotionEvidence",
                "deployed_neural_prior_policy": "DeployedNeuralPriorPolicy",
                "neural_prior_deployment_lineage": "ForecastResult",
                "radar_spatial_grid_identity": "RadarSpatialGridIdentity",
                "mosaic_observation_source_registry": (
                    "MosaicObservationSourceRegistry"
                ),
                "radar_observation_geometry": "RadarObservationGeometryContract",
                "verification_observation_error_plan": (
                    "VerificationObservationErrorPlan"
                ),
                "verification_bundle": "VerificationBundle",
                "variational_fso": "VariationalFSO",
                "variational_fsoi": "VariationalFSOI",
                "semantic_scoring_replay": "ScoringReplayBundleManifest",
                "forecast_run_artifact": "ForecastResult",
                "neural_prior_holdout_plan": "NeuralPriorHoldoutPlan",
            }[family],
            capabilities.lifecycle_probe,
        )
        for family, capabilities in CONTRACT_CAPABILITIES.items()
    ),
)


for _capabilities in CONTRACT_CAPABILITIES.values():
    _capabilities.validate()
for _fixture in AUDIT_GENERATION_FIXTURES:
    _fixture.validate()
_registered_audit_generations = {
    (family, contract)
    for family, capabilities in CONTRACT_CAPABILITIES.items()
    for contract in capabilities.audit_readable
}
_fixture_audit_generations = {
    (fixture.family, fixture.contract) for fixture in AUDIT_GENERATION_FIXTURES
}
if (
    len(_fixture_audit_generations) != len(AUDIT_GENERATION_FIXTURES)
    or _fixture_audit_generations != _registered_audit_generations
    or len({fixture.fixture_path for fixture in AUDIT_GENERATION_FIXTURES})
    != len(AUDIT_GENERATION_FIXTURES)
):
    raise ValueError(
        "frozen audit fixtures must exactly cover audit-readable generations"
    )
for _fixture in AUDIT_GENERATION_FIXTURES:
    _capabilities = CONTRACT_CAPABILITIES[_fixture.family]
    if (
        _fixture.scientific_action_allowed
        is not (_fixture.contract in _capabilities.scientific_eligible)
        or _fixture.operational_action_allowed
        is not (_fixture.contract in _capabilities.operationally_accepted)
    ):
        raise ValueError("frozen audit fixture action policy disagrees with registry")


def current_contract(family: str) -> str:
    """Return the single registered current generation for one family."""

    try:
        return CONTRACT_CAPABILITIES[family].current
    except KeyError as error:
        raise ValueError(f"unknown contract family: {family}") from error


CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE_CONTRACT = current_contract(
    "radar_metric_domain_evidence"
)
CURRENT_NEURAL_PRIOR_PROMOTION_EVIDENCE_CONTRACT = current_contract(
    "neural_prior_promotion_evidence"
)
CURRENT_DEPLOYED_NEURAL_PRIOR_POLICY_CONTRACT = current_contract(
    "deployed_neural_prior_policy"
)
CURRENT_RADAR_SPATIAL_GRID_IDENTITY_CONTRACT = current_contract(
    "radar_spatial_grid_identity"
)
CURRENT_MOSAIC_OBSERVATION_SOURCE_REGISTRY_CONTRACT = current_contract(
    "mosaic_observation_source_registry"
)
CURRENT_RADAR_OBSERVATION_GEOMETRY_CONTRACT = current_contract(
    "radar_observation_geometry"
)
CURRENT_VERIFICATION_OBSERVATION_ERROR_PLAN_CONTRACT = current_contract(
    "verification_observation_error_plan"
)
CURRENT_VERIFICATION_BUNDLE_CONTRACT = current_contract(
    "verification_bundle"
)
CURRENT_VARIATIONAL_FSO_CONTRACT = current_contract("variational_fso")
CURRENT_VARIATIONAL_FSOI_CONTRACT = current_contract("variational_fsoi")
CURRENT_SEMANTIC_SCORING_REPLAY_CONTRACT = current_contract(
    "semantic_scoring_replay"
)
CURRENT_FORECAST_RUN_ARTIFACT_VERSION = current_contract(
    "forecast_run_artifact"
)
CURRENT_NEURAL_PRIOR_HOLDOUT_PLAN_CONTRACT = current_contract(
    "neural_prior_holdout_plan"
)


def render_contract_capability_table() -> str:
    """Render the exact README table checked by the scientific test suite."""

    header = (
        "| Contract family | Current | Predecessor | Issuable | Audit-readable "
        "| Scientific | Operational |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for family, capability in CONTRACT_CAPABILITIES.items():
        values = (
            family,
            capability.current,
            capability.predecessor or "—",
            ", ".join(sorted(capability.issuable)) or "∅",
            ", ".join(sorted(capability.audit_readable)) or "∅",
            ", ".join(sorted(capability.scientific_eligible)) or "∅",
            ", ".join(sorted(capability.operationally_accepted)) or "∅",
        )
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join((header, *rows))
