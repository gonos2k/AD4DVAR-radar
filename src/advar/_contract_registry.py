"""Small authoritative registry for current scientific contract capability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractCapabilities:
    """Describe which generations can be built, audited, or acted upon."""

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


CONTRACT_CAPABILITIES: dict[str, ContractCapabilities] = {
    "radar_metric_domain_evidence": ContractCapabilities(
        current="radar-metric-domain-evidence-v2",
        predecessor="radar-metric-domain-evidence-v1",
        issuable=frozenset({"radar-metric-domain-evidence-v2"}),
        audit_readable=frozenset(
            {
                "radar-metric-domain-evidence-v1",
                "radar-metric-domain-evidence-v2",
            }
        ),
        scientific_eligible=frozenset({"radar-metric-domain-evidence-v2"}),
        operationally_accepted=frozenset(),
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
    ),
    "deployed_neural_prior_policy": ContractCapabilities(
        current="deployed-neural-prior-policy-v17",
        predecessor=None,
        issuable=frozenset({"deployed-neural-prior-policy-v17"}),
        audit_readable=frozenset({"deployed-neural-prior-policy-v17"}),
        scientific_eligible=frozenset(),
        operationally_accepted=frozenset(),
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
