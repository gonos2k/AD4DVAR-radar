from __future__ import annotations

import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import advar.promotion as promotion


def _private_key(seed: int) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes((seed,)) * 32)


def _authority_store() -> tuple[
    promotion._PromotionDeploymentAuthorityTrustStore,
    dict[str, Ed25519PrivateKey],
]:
    keys = {
        "ledger": _private_key(1),
        "release": _private_key(2),
        "runtime": _private_key(3),
        "operational": _private_key(4),
        "promotion": _private_key(5),
    }
    store = promotion._PromotionDeploymentAuthorityTrustStore(
        keys={name: key.public_key() for name, key in keys.items()},
        content_digest="1" * 64,
        roles={
            "ledger": frozenset({"ledger_issuance"}),
            "release": frozenset({"release_approval"}),
            "runtime": frozenset({"runtime_activation"}),
            "operational": frozenset({"operational_decision"}),
            "promotion": frozenset({"promotion_certificate"}),
        },
        not_before={
            name: "2026-01-01T00:00:00+00:00" for name in keys
        },
        not_after={name: "2031-01-01T00:00:00+00:00" for name in keys},
        revoked_at={name: None for name in keys},
        ledger_instance_digests={
            "ledger": frozenset({"d" * 64}),
            "release": frozenset(),
            "runtime": frozenset(),
            "operational": frozenset(),
            "promotion": frozenset(),
        },
    )
    return store, keys


def _signed_payload(
    signer: Ed25519PrivateKey,
    payload: dict[str, object],
    *,
    domain: bytes = b"",
) -> str:
    return signer.sign(
        domain
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hex()


def _ledger_receipt(
    store: promotion._PromotionDeploymentAuthorityTrustStore,
    signer: Ed25519PrivateKey,
) -> promotion.LedgerIssuanceReceipt:
    authority = promotion.Ed25519DeploymentAuthoritySigner(
        "ledger", signer, fixed_signing_time="2026-08-09T00:00:01Z"
    )
    return promotion._issue_ledger_issuance_receipt(
        ledger_instance_digest="d" * 64,
        sequence_number=1,
        previous_certificate_digest=(
            promotion.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST
        ),
        promotion_evidence_digest="a" * 64,
        scoring_replay_bundle_digest="b" * 64,
        scoring_replay_archive_sha256="c" * 64,
        scoring_evaluation_payload_sha256="e" * 64,
        scoring_artifact_digest="f" * 64,
        scoring_completion_receipt_digest="0" * 64,
        scoring_completion_completed_at="2026-08-09T00:00:00Z",
        issued_at=authority.signing_time(),
        signer=authority,
        authority_trust_store=store,
    )


def _release_and_runtime(
    store: promotion._PromotionDeploymentAuthorityTrustStore,
    keys: dict[str, Ed25519PrivateKey],
    *,
    release_expiry: str = "2026-08-10T00:00:00Z",
    runtime_expiry: str | None = None,
) -> tuple[
    promotion.DeploymentBundleReleaseApproval,
    promotion.DeploymentRuntimeActivationReceipt,
    promotion.Ed25519DeploymentAuthoritySigner,
]:
    release_signer = promotion.Ed25519DeploymentAuthoritySigner(
        "release", keys["release"], fixed_signing_time="2026-08-09T00:00:01Z"
    )
    release = promotion._issue_deployment_bundle_release_approval(
        deployment_bundle_digest="a" * 64,
        bundle_manifest_digest="b" * 64,
        source_commit="c" * 40,
        repository="gonos2k/AD4DVAR-radar",
        source_ref="refs/tags/v0.93.0",
        platform="linux-x86_64-cpu",
        runtime_mode="deployable",
        expires_at=release_expiry,
        signer=release_signer,
        authority_trust_store=store,
    )
    runtime_signer = promotion.Ed25519DeploymentAuthoritySigner(
        "runtime", keys["runtime"], fixed_signing_time="2026-08-09T00:00:02Z"
    )
    runtime = promotion._issue_deployment_runtime_activation_receipt(
        release_approval=release,
        runtime_tree_digest="d" * 64,
        interpreter_closure_digest="e" * 64,
        installation_attestation_sha256="f" * 64,
        deployment_instance_digest="0" * 64,
        host_identity_digest="1" * 64,
        runtime_mode="deployable",
        activation_sequence_number=1,
        previous_activation_receipt_digest=(
            promotion.DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST
        ),
        expires_at=release_expiry if runtime_expiry is None else runtime_expiry,
        signer=runtime_signer,
        authority_trust_store=store,
    )
    return release, runtime, runtime_signer


def _operational_fixture() -> tuple[
    promotion.OperationalDeploymentDecisionCertificate,
    dict[str, object],
    promotion._PromotionDeploymentAuthorityTrustStore,
    dict[str, Ed25519PrivateKey],
]:
    store, keys = _authority_store()
    release, runtime, _ = _release_and_runtime(store, keys)
    promotion_key = keys["promotion"]
    promotion_ledger_receipt = _ledger_receipt(store, keys["ledger"])
    decision_payload: dict[str, object] = {
        "selection": {
            "selected_prior_digest": "2" * 64,
            "selected_role": "parent",
            "fallback_reason": "unverified_routing_evidence",
        },
        "promotion_deployment_certificate": {
            "certificate_digest": "3" * 64,
            "authority_id": "promotion",
            "authority_public_key_hex": promotion_key.public_key()
            .public_bytes_raw()
            .hex(),
            "issued_at": "2026-08-08T00:00:00Z",
            "ledger_issuance_receipt_payload_json": json.dumps(
                promotion_ledger_receipt.payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        "deployment_policy": {
            "promotion_evidence_digest": "4" * 64,
            "policy_digest": "5" * 64,
        },
        "policy_trust_store": {"content_digest": "6" * 64},
        "full_analysis_input_digest": "7" * 64,
        "analysis_input_derivation_artifact_digest": "8" * 64,
        "global_raw_resolution_receipt_digest": "9" * 64,
        "resolved_raw_volume_identity_set_digest": "a" * 64,
        "analysis_processor_trust_store_digest": "b" * 64,
        "raw_ingestor_trust_store_digest": "c" * 64,
        "analysis_input_provenance_commitment_digest": "d" * 64,
        "input_plan_digest": "e" * 64,
        "observation_valid_time": "2026-08-09T00:00:00Z",
        "input_available_time": "2026-08-09T00:00:01Z",
        "decision_deadline": "2026-08-09T00:00:10Z",
        "publication_time": "2026-08-09T00:00:11Z",
        "operational_cycle_id": "f" * 64,
        "deployment_bundle_release_approval": release.payload
        | {"approval_digest": release.approval_digest},
        "deployment_runtime_activation_receipt": runtime.payload
        | {"receipt_digest": runtime.receipt_digest},
    }
    ledger_signer = promotion.Ed25519DeploymentAuthoritySigner(
        "ledger", keys["ledger"], fixed_signing_time="2026-08-09T00:00:05Z"
    )
    commit_entry, chain_root = promotion._operational_decision_commit_digests(
        decision_payload,
        ledger_instance_digest="d" * 64,
        sequence_number=1,
        previous_operational_decision_digest=(
            promotion.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
        ),
        accepted_at="2026-08-09T00:00:03Z",
    )
    ledger_receipt = promotion._issue_operational_decision_ledger_receipt(
        decision_payload,
        ledger_instance_digest="d" * 64,
        sequence_number=1,
        previous_operational_decision_digest=(
            promotion.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
        ),
        accepted_at="2026-08-09T00:00:03Z",
        committed_at="2026-08-09T00:00:04Z",
        commit_entry_digest=commit_entry,
        committed_chain_root_digest=chain_root,
        signer=ledger_signer,
        authority_trust_store=store,
    )
    values: dict[str, object] = {
        "decision_payload_digest": promotion.json_digest(decision_payload),
        "promotion_deployment_certificate_digest": "3" * 64,
        "promotion_evidence_digest": "4" * 64,
        "deployment_policy_digest": "5" * 64,
        "deployment_policy_trust_store_digest": "6" * 64,
        "full_analysis_input_digest": "7" * 64,
        "analysis_input_derivation_artifact_digest": "8" * 64,
        "global_raw_resolution_receipt_digest": "9" * 64,
        "resolved_raw_volume_identity_set_digest": "a" * 64,
        "analysis_processor_trust_store_digest": "b" * 64,
        "raw_ingestor_trust_store_digest": "c" * 64,
        "analysis_input_provenance_commitment_digest": "d" * 64,
        "deployment_bundle_release_approval_digest": release.approval_digest,
        "deployment_runtime_activation_receipt_digest": runtime.receipt_digest,
        "deployment_bundle_digest": runtime.deployment_bundle_digest,
        "runtime_tree_digest": runtime.runtime_tree_digest,
        "interpreter_closure_digest": runtime.interpreter_closure_digest,
        "deployment_instance_digest": runtime.deployment_instance_digest,
        "host_identity_digest": runtime.host_identity_digest,
        "input_plan_digest": "e" * 64,
        "observation_valid_time": "2026-08-09T00:00:00Z",
        "input_available_time": "2026-08-09T00:00:01Z",
        "decision_deadline": "2026-08-09T00:00:10Z",
        "publication_time": "2026-08-09T00:00:11Z",
        "operational_cycle_id": "f" * 64,
        "selected_prior_digest": "2" * 64,
        "selected_role": "parent",
        "fallback_reason": "unverified_routing_evidence",
        "operational_ledger_receipt_payload_json": json.dumps(
            ledger_receipt.payload, sort_keys=True, separators=(",", ":")
        ),
        "operational_ledger_receipt_digest": ledger_receipt.receipt_digest,
        "ledger_instance_digest": "d" * 64,
        "ledger_sequence_number": 1,
        "previous_operational_decision_digest": (
            promotion.OPERATIONAL_DECISION_LEDGER_GENESIS_DIGEST
        ),
        "issued_at": "2026-08-09T00:00:06Z",
        "authority_id": "operational",
        "authority_public_key_hex": keys["operational"].public_key()
        .public_bytes_raw()
        .hex(),
        "authority_trust_store_digest": store.content_digest,
        "authority_signature_hex": "",
        "contract": "operational-deployment-decision-certificate-v8",
    }
    values["authority_signature_hex"] = _signed_payload(
        keys["operational"],
        {
            key: value
            for key, value in values.items()
            if key != "authority_signature_hex"
        },
    )
    certificate = promotion._operational_deployment_decision_certificate_from_payload(
        values
    )
    return certificate, decision_payload, store, keys


class A5OfflineBoundsTests(unittest.TestCase):
    def test_exact_signed_ledger_genesis_fixture_is_accepted(self) -> None:
        store, keys = _authority_store()
        receipt = _ledger_receipt(store, keys["ledger"])

        self.assertEqual(receipt.sequence_number, 1)
        self.assertEqual(
            receipt.previous_certificate_digest,
            promotion.PROMOTION_DEPLOYMENT_CERTIFICATE_GENESIS_DIGEST,
        )
        promotion._validate_ledger_issuance_receipt(
            receipt, authority_trust_store=store
        )

    def test_signed_sequence_two_genesis_fixture_is_rejected(self) -> None:
        store, keys = _authority_store()
        receipt = _ledger_receipt(store, keys["ledger"])
        values = dict(receipt.payload)
        values["sequence_number"] = 2
        values["checkpoint_digest"] = promotion._ledger_checkpoint_digest(
            ledger_instance_digest=values["ledger_instance_digest"],
            sequence_number=2,
            previous_certificate_digest=values["previous_certificate_digest"],
            promotion_evidence_digest=values["promotion_evidence_digest"],
            scoring_replay_bundle_digest=values["scoring_replay_bundle_digest"],
            scoring_replay_archive_sha256=values["scoring_replay_archive_sha256"],
            scoring_evaluation_payload_sha256=values[
                "scoring_evaluation_payload_sha256"
            ],
            scoring_artifact_digest=values["scoring_artifact_digest"],
            scoring_completion_receipt_digest=values[
                "scoring_completion_receipt_digest"
            ],
            scoring_completion_completed_at=values[
                "scoring_completion_completed_at"
            ],
        )
        values["signer_signature_hex"] = _signed_payload(
            keys["ledger"],
            {key: value for key, value in values.items() if key != "signer_signature_hex"},
        )
        forged = promotion._ledger_issuance_receipt_from_payload(values)

        with self.assertRaisesRegex(ValueError, "integrity"):
            promotion._validate_ledger_issuance_receipt(
                forged, authority_trust_store=store
            )

    def test_exact_signed_runtime_expiry_matches_release_expiry(self) -> None:
        store, keys = _authority_store()
        release, runtime, _ = _release_and_runtime(store, keys)

        self.assertEqual(runtime.expires_at, release.expires_at)
        promotion._validate_deployment_runtime_activation_receipt(
            runtime,
            release_approval=release,
            authority_trust_store=store,
            required_valid_through="2026-08-09T00:01:00Z",
        )

    def test_signed_runtime_expiry_beyond_release_is_rejected(self) -> None:
        store, keys = _authority_store()
        release, runtime, runtime_signer = _release_and_runtime(store, keys)
        values = dict(runtime.payload)
        values["expires_at"] = "2026-08-11T00:00:00Z"
        values["authority_signature_hex"] = _signed_payload(
            runtime_signer.private_key,
            {
                key: value
                for key, value in values.items()
                if key != "authority_signature_hex"
            },
            domain=promotion._RUNTIME_ACTIVATION_SIGNATURE_DOMAIN,
        )
        forged = promotion._deployment_runtime_activation_receipt_from_payload(
            values
        )

        with self.assertRaisesRegex(ValueError, "invalid"):
            promotion._validate_deployment_runtime_activation_receipt(
                forged,
                release_approval=release,
                authority_trust_store=store,
                required_valid_through="2026-08-09T00:01:00Z",
            )

    def test_exact_signed_operational_certificate_uses_current_trust_store(self) -> None:
        certificate, decision_payload, store, _ = _operational_fixture()

        self.assertEqual(
            certificate.authority_trust_store_digest, store.content_digest
        )
        promotion._validate_operational_deployment_decision_certificate(
            certificate,
            decision_payload=decision_payload,
            authority_trust_store=store,
        )

    def test_signed_operational_certificate_with_wrong_trust_digest_is_rejected(
        self,
    ) -> None:
        certificate, decision_payload, store, keys = _operational_fixture()
        values = dict(certificate.payload)
        values["authority_trust_store_digest"] = "2" * 64
        values["authority_signature_hex"] = _signed_payload(
            keys["operational"],
            {
                key: value
                for key, value in values.items()
                if key != "authority_signature_hex"
            },
        )
        forged = promotion._operational_deployment_decision_certificate_from_payload(
            values
        )

        with self.assertRaisesRegex(ValueError, "integrity"):
            promotion._validate_operational_deployment_decision_certificate(
                forged,
                decision_payload=decision_payload,
                authority_trust_store=store,
            )


if __name__ == "__main__":
    unittest.main()
