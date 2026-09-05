from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import advar.promotion as promotion


@dataclass(frozen=True)
class _AuthorityFixture:
    trust: promotion._PromotionDeploymentAuthorityTrustStore
    release_key: Ed25519PrivateKey
    runtime_key: Ed25519PrivateKey


def _authority_fixture() -> _AuthorityFixture:
    release_key = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
    runtime_key = Ed25519PrivateKey.from_private_bytes(b"\x06" * 32)
    trust = promotion._PromotionDeploymentAuthorityTrustStore(
        keys={
            "review-release": release_key.public_key(),
            "review-runtime": runtime_key.public_key(),
        },
        content_digest="7" * 64,
        roles={
            "review-release": frozenset({"release_approval"}),
            "review-runtime": frozenset({"runtime_activation"}),
        },
        not_before={
            "review-release": "2026-01-01T00:00:00+00:00",
            "review-runtime": "2026-01-01T00:00:00+00:00",
        },
        not_after={
            "review-release": "2031-01-01T00:00:00+00:00",
            "review-runtime": "2031-01-01T00:00:00+00:00",
        },
        revoked_at={"review-release": None, "review-runtime": None},
        ledger_instance_digests={
            "review-release": frozenset(),
            "review-runtime": frozenset(),
        },
    )
    return _AuthorityFixture(trust, release_key, runtime_key)


def _signed_release(authority: _AuthorityFixture) -> promotion.DeploymentBundleReleaseApproval:
    return promotion._issue_deployment_bundle_release_approval(
        deployment_bundle_digest="a" * 64,
        bundle_manifest_digest="b" * 64,
        source_commit="c" * 40,
        repository="review/repository",
        source_ref="refs/tags/review",
        platform="linux-x86_64-cpu",
        runtime_mode="deployable",
        expires_at="2031-01-01T00:00:00Z",
        signer=promotion.Ed25519DeploymentAuthoritySigner(
            "review-release",
            authority.release_key,
            fixed_signing_time="2026-08-09T00:00:44Z",
        ),
        authority_trust_store=authority.trust,
    )


def _signed_runtime(
    authority: _AuthorityFixture,
    release: promotion.DeploymentBundleReleaseApproval,
    *,
    sequence: int = 1,
) -> promotion.DeploymentRuntimeActivationReceipt:
    return promotion._issue_deployment_runtime_activation_receipt(
        release_approval=release,
        runtime_tree_digest="d" * 64,
        interpreter_closure_digest="e" * 64,
        installation_attestation_sha256="f" * 64,
        deployment_instance_digest="1" * 64,
        host_identity_digest="2" * 64,
        runtime_mode="deployable",
        activation_sequence_number=sequence,
        previous_activation_receipt_digest=(
            promotion.DEPLOYMENT_RUNTIME_ACTIVATION_GENESIS_DIGEST
            if sequence == 1
            else "3" * 64
        ),
        expires_at="2031-01-01T00:00:00Z",
        signer=promotion.Ed25519DeploymentAuthoritySigner(
            "review-runtime",
            authority.runtime_key,
            fixed_signing_time="2026-08-09T00:00:45Z",
        ),
        authority_trust_store=authority.trust,
    )


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    )


class ReviewLegacyLoaderTests(unittest.TestCase):
    def test_legacy_loaders_round_trip_canonical_signed_payloads(self) -> None:
        authority = _authority_fixture()
        release = _signed_release(authority)
        runtime = _signed_runtime(authority, release)

        with tempfile.TemporaryDirectory() as directory:
            release_path = Path(directory) / "release.json"
            runtime_path = Path(directory) / "runtime.json"
            _write_canonical(
                release_path,
                release.payload | {"approval_digest": release.approval_digest},
            )
            _write_canonical(
                runtime_path,
                runtime.payload | {"receipt_digest": runtime.receipt_digest},
            )

            real_fstat = os.fstat

            def root_owned_metadata(descriptor: int) -> SimpleNamespace:
                metadata = real_fstat(descriptor)
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_uid=0,
                    st_size=metadata.st_size,
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino,
                    st_mtime_ns=metadata.st_mtime_ns,
                )

            with (
                patch.object(
                    promotion,
                    "_load_promotion_deployment_authority_trust_store",
                    return_value=authority.trust,
                ),
                patch.object(promotion, "validate_current_runtime_closure"),
                patch.object(promotion.os, "fstat", side_effect=root_owned_metadata),
            ):
                loaded_release = promotion.load_deployment_bundle_release_approval(
                    release_path,
                    authority_trust_store_path="/etc/advar/deployment-authorities.json",
                    required_valid_through="2026-09-06T00:00:00Z",
                )
                loaded_runtime = promotion.load_deployment_runtime_activation_receipt(
                    runtime_path,
                    release_approval_path=release_path,
                    authority_trust_store_path="/etc/advar/deployment-authorities.json",
                    required_valid_through="2026-09-06T00:00:00Z",
                )

        self.assertEqual(loaded_release.approval_digest, release.approval_digest)
        self.assertEqual(loaded_runtime.receipt_digest, runtime.receipt_digest)
        self.assertEqual(loaded_runtime.activation_sequence_number, 1)

    def test_runtime_activation_requires_positive_exact_integer(self) -> None:
        authority = _authority_fixture()
        release = _signed_release(authority)
        with self.assertRaisesRegex(ValueError, "chronology"):
            _signed_runtime(authority, release, sequence=1.5)  # type: ignore[arg-type]

        runtime = _signed_runtime(authority, release, sequence=1)
        promotion._validate_deployment_runtime_activation_receipt(
            runtime,
            release_approval=release,
            authority_trust_store=authority.trust,
            required_valid_through="2026-09-06T00:00:00Z",
        )
        self.assertIs(type(runtime.activation_sequence_number), int)
        self.assertEqual(runtime.activation_sequence_number, 1)

    def test_low_confidence_replay_records_low_confidence_reason(self) -> None:
        class FakeEvidence:
            deployment_eligible = True
            certified_applicability_regime_groups = (("known", "near"),)
            certified_range_geometry_contract_digests = ("g" * 64,)
            candidate_prior_digest = "c" * 64
            parent_prior_digest = "p" * 64
            deployment_regime_classifier_digest = "r" * 64
            deployment_regime_classifier_manifest_digest = "m" * 64
            semantic_replay_generation_digest = (
                promotion.SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST
            )
            promotion_evidence_digest = "e" * 64

        class FakePolicy:
            candidate_prior_digest = "c" * 64
            parent_prior_digest = "p" * 64
            promotion_evidence_digest = "e" * 64
            promotion_deployment_certificate_digest = "x" * 64
            regime_classifier_digest = "r" * 64
            regime_classifier_manifest_digest = "m" * 64
            range_geometry_contract_digest = "g" * 64
            semantic_replay_generation_digest = (
                promotion.SEMANTIC_SCORING_REPLAY_GENERATION_DIGEST
            )
            minimum_regime_confidence = 0.8
            minimum_weather_top1_top2_gap = 0.05
            minimum_deployment_confidence_margin = 0.05
            policy_digest = "q" * 64

            @property
            def payload(self) -> dict[str, object]:
                return {
                    "candidate_prior_digest": self.candidate_prior_digest,
                    "parent_prior_digest": self.parent_prior_digest,
                    "promotion_evidence_digest": self.promotion_evidence_digest,
                    "promotion_deployment_certificate_digest": self.promotion_deployment_certificate_digest,
                    "promotion_deployment_authority_trust_store_digest": "a" * 64,
                    "regime_classifier_digest": self.regime_classifier_digest,
                    "regime_classifier_manifest_digest": self.regime_classifier_manifest_digest,
                    "range_geometry_contract_digest": self.range_geometry_contract_digest,
                    "semantic_replay_generation_digest": self.semantic_replay_generation_digest,
                    "mps_backend_certification_digest": None,
                    "mps_backend_certification_policy_digest": None,
                    "minimum_regime_confidence": self.minimum_regime_confidence,
                    "minimum_weather_top1_top2_gap": self.minimum_weather_top1_top2_gap,
                    "minimum_deployment_confidence_margin": self.minimum_deployment_confidence_margin,
                    "contract": "deployed-neural-prior-policy-v17",
                }

        class FakeCertificate:
            certificate_digest = "x" * 64
            payload = {"certificate_digest": certificate_digest}

        regime = {
            "classifier_digest": "r" * 64,
            "regime": "known",
            "regime_confidence": 0.2,
            "weather_top1_top2_gap": 0.9,
            "is_ood": False,
            "full_analysis_input_digest": "i" * 64,
        }
        regime["evidence_digest"] = promotion.json_digest(regime)
        ranges = {
            "active_range_regimes": ["near"],
            "range_geometry_contract_digest": "g" * 64,
        }
        ranges["evidence_digest"] = promotion.json_digest(ranges)
        policy = FakePolicy()
        certificate = FakeCertificate()
        decision = {
            "regime_classification_evidence": regime,
            "range_partition_evidence": ranges,
            "selection": {
                "fallback_reason": "low_regime_confidence",
                "selected_role": "parent",
                "selected_prior_digest": "p" * 64,
                "deployment_confidence_margin": -0.6,
            },
            "deployment_policy": policy.payload | {"policy_digest": policy.policy_digest},
            "promotion_deployment_certificate": certificate.payload,
            "policy_trust_store": {"content_digest": "t" * 64},
            "routing_semantic_replay_verified": True,
        }
        with patch.object(promotion, "NeuralPriorPromotionEvidence", FakeEvidence), patch.object(
            promotion, "DeployedNeuralPriorPolicy", FakePolicy
        ):
            reason, role, _, _ = promotion._replay_operational_deployment_selection(
                decision,
                promotion_deployment_certificate=certificate,
                promotion_evidence=FakeEvidence(),
                policy=policy,
                policy_trust_store_digest="t" * 64,
                allow_committed_routing_evidence=True,
            )
        self.assertEqual((reason, role), ("low_regime_confidence", "parent"))

    def test_resigned_malformed_regime_reference_digest_is_rejected(self) -> None:
        try:
            from test_promotion import NeuralPriorPromotionTests
        except ModuleNotFoundError:
            from tests.test_promotion import NeuralPriorPromotionTests

        fixture = NeuralPriorPromotionTests("runTest")
        plan = fixture.plan().regime_reference_plans[0]
        evidence = fixture.reference_evidence(1)
        for field, invalid in (
            ("full_analysis_input_digest", "malformed"),
            ("verification_bundle_digest", "malformed"),
            ("observed_regime", ""),
            ("observed_storm_id", " label "),
        ):
            with self.subTest(field=field):
                values = dict(evidence.payload)
                values[field] = invalid
                unsigned = dict(values)
                unsigned["labeler_signature"] = ""
                values["labeler_signature"] = fixture.regime_labeler_key().sign(
                    promotion.json_digest(unsigned).encode("ascii")
                ).hex()
                malformed = promotion._new_regime_reference_evidence(**values)
                with self.assertRaisesRegex(ValueError, field):
                    promotion.validate_regime_reference_evidence(malformed, plan)


if __name__ == "__main__":
    unittest.main()
