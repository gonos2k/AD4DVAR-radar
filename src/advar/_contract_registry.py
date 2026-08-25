"""Small authoritative registry for current scientific contract capability."""

from __future__ import annotations

from dataclasses import dataclass


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


class OperationalDeploymentUnsupportedError(RuntimeError):
    """Scientific artifacts cannot authorize an operational deployment."""


CONTRACT_CAPABILITIES: dict[str, ContractCapabilities] = {
    "radar_metric_domain_evidence": ContractCapabilities(
        current="radar-metric-domain-evidence-v3",
        predecessor="radar-metric-domain-evidence-v2",
        issuable=frozenset({"radar-metric-domain-evidence-v3"}),
        audit_readable=frozenset(
            {
                "radar-metric-domain-evidence-v1",
                "radar-metric-domain-evidence-v2",
                "radar-metric-domain-evidence-v3",
            }
        ),
        scientific_eligible=frozenset({"radar-metric-domain-evidence-v3"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_METRIC_EVIDENCE_LIFECYCLE_PROBE,
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
    ),
    "radar_spatial_grid_identity": ContractCapabilities(
        current="radar-spatial-grid-identity-v6",
        predecessor="radar-spatial-grid-identity-v5",
        issuable=frozenset({"radar-spatial-grid-identity-v6"}),
        audit_readable=frozenset(
            {"radar-spatial-grid-identity-v5", "radar-spatial-grid-identity-v6"}
        ),
        scientific_eligible=frozenset({"radar-spatial-grid-identity-v6"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
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
    ),
    "verification_observation_error_plan": ContractCapabilities(
        current="verification-observation-error-plan-v13",
        predecessor="verification-observation-error-plan-v12",
        issuable=frozenset({"verification-observation-error-plan-v13"}),
        audit_readable=frozenset(
            {
                "verification-observation-error-plan-v12",
                "verification-observation-error-plan-v13",
            }
        ),
        scientific_eligible=frozenset(
            {"verification-observation-error-plan-v13"}
        ),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
    ),
    "verification_bundle": ContractCapabilities(
        current="radar-verification-bundle-v19",
        predecessor="radar-verification-bundle-v18",
        issuable=frozenset({"radar-verification-bundle-v19"}),
        audit_readable=frozenset(
            {"radar-verification-bundle-v18", "radar-verification-bundle-v19"}
        ),
        scientific_eligible=frozenset({"radar-verification-bundle-v19"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
    ),
    "variational_fso": ContractCapabilities(
        current="p1-variational-fso-v25",
        predecessor="p1-variational-fso-v24",
        issuable=frozenset({"p1-variational-fso-v25"}),
        audit_readable=frozenset(
            {"p1-variational-fso-v24", "p1-variational-fso-v25"}
        ),
        scientific_eligible=frozenset({"p1-variational-fso-v25"}),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
    ),
    "variational_fsoi": ContractCapabilities(
        current="p1-linearized-observation-impact-v21",
        predecessor="p1-linearized-observation-impact-v20",
        issuable=frozenset({"p1-linearized-observation-impact-v21"}),
        audit_readable=frozenset(
            {
                "p1-linearized-observation-impact-v20",
                "p1-linearized-observation-impact-v21",
            }
        ),
        scientific_eligible=frozenset(
            {"p1-linearized-observation-impact-v21"}
        ),
        operationally_accepted=frozenset(),
        lifecycle_probe=_VERIFICATION_FSO_LIFECYCLE_PROBE,
    ),
    "semantic_scoring_replay": ContractCapabilities(
        current="neural-prior-scoring-replay-bundle-v24",
        predecessor="neural-prior-scoring-replay-bundle-v23",
        issuable=frozenset({"neural-prior-scoring-replay-bundle-v24"}),
        audit_readable=frozenset(
            {
                "neural-prior-scoring-replay-bundle-v23",
                "neural-prior-scoring-replay-bundle-v24",
            }
        ),
        scientific_eligible=frozenset(
            {"neural-prior-scoring-replay-bundle-v24"}
        ),
        operationally_accepted=frozenset(),
        lifecycle_probe=_SEMANTIC_REPLAY_LIFECYCLE_PROBE,
    ),
}


for _capabilities in CONTRACT_CAPABILITIES.values():
    _capabilities.validate()


CURRENT_RADAR_METRIC_DOMAIN_EVIDENCE_CONTRACT = CONTRACT_CAPABILITIES[
    "radar_metric_domain_evidence"
].current
CURRENT_NEURAL_PRIOR_PROMOTION_EVIDENCE_CONTRACT = CONTRACT_CAPABILITIES[
    "neural_prior_promotion_evidence"
].current
CURRENT_DEPLOYED_NEURAL_PRIOR_POLICY_CONTRACT = CONTRACT_CAPABILITIES[
    "deployed_neural_prior_policy"
].current


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
