"""Authoritative current contract generations and declared capabilities."""

from __future__ import annotations

from dataclasses import dataclass


class OperationalDeploymentUnsupportedError(RuntimeError):
    """Scientific artifacts cannot authorize operational deployment."""


@dataclass(frozen=True)
class ContractCapabilities:
    """Compact runtime authority for one contract family."""

    current: str
    predecessor: str | None
    issuable: frozenset[str]
    audit_readable: frozenset[str]
    scientific_eligible: frozenset[str]
    operationally_accepted: frozenset[str]

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


def _capability(
    *,
    current: str,
    predecessor: str | None = None,
    scientific: bool = True,
    operational: bool = False,
) -> ContractCapabilities:
    issuable = frozenset({current})
    readable = {current}
    if predecessor is not None:
        readable.add(predecessor)
    return ContractCapabilities(
        current=current,
        predecessor=predecessor,
        issuable=issuable,
        audit_readable=frozenset(readable),
        scientific_eligible=issuable if scientific else frozenset(),
        operationally_accepted=issuable if operational else frozenset(),
    )


CONTRACT_CAPABILITIES: dict[str, ContractCapabilities] = {
    "radar_metric_domain_evidence": _capability(
        current="radar-metric-domain-evidence-v4",
        predecessor="radar-metric-domain-evidence-v3",
    ),
    "neural_prior_promotion_evidence": _capability(
        current="neural-prior-promotion-evidence-v32",
        predecessor="neural-prior-promotion-evidence-v31",
    ),
    "deployed_neural_prior_policy": _capability(
        current="deployed-neural-prior-policy-v17",
        scientific=False,
    ),
    "neural_prior_deployment_lineage": _capability(
        current="neural-prior-deployment-lineage-v19",
        predecessor="neural-prior-deployment-lineage-v18-audit",
        scientific=False,
    ),
    "radar_spatial_grid_identity": _capability(
        current="radar-spatial-grid-identity-v6",
        predecessor="radar-spatial-grid-identity-v5",
    ),
    "mosaic_observation_source_registry": _capability(
        current="mosaic-observation-source-registry-v7",
        predecessor="mosaic-observation-source-registry-v6",
    ),
    "radar_observation_geometry": _capability(
        current="radar-observation-geometry-v7",
        predecessor="radar-observation-geometry-v6",
    ),
    "verification_observation_error_plan": _capability(
        current="verification-observation-error-plan-v16",
        predecessor="verification-observation-error-plan-v15",
    ),
    "verification_bundle": _capability(
        current="radar-verification-bundle-v22",
        predecessor="radar-verification-bundle-v21",
    ),
    "variational_fso": _capability(
        current="p1-variational-fso-v28",
        predecessor="p1-variational-fso-v27",
    ),
    "variational_fsoi": _capability(
        current="p1-linearized-observation-impact-v24",
        predecessor="p1-linearized-observation-impact-v23",
    ),
    "semantic_scoring_replay": _capability(
        current="neural-prior-scoring-replay-bundle-v27",
        predecessor="neural-prior-scoring-replay-bundle-v26",
    ),
    "forecast_run_artifact": _capability(
        current="forecast-run-v72",
        predecessor="forecast-run-v71",
    ),
    "neural_prior_holdout_plan": _capability(
        current="neural-prior-holdout-plan-v37",
        predecessor="neural-prior-holdout-plan-v36",
    ),
}


for _capabilities in CONTRACT_CAPABILITIES.values():
    _capabilities.validate()


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
    """Render the exact README table checked by the test suite."""

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
