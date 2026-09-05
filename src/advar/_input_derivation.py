"""Validate input provenance payloads independently of forecast numerics."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ._digest import json_digest, validate_sha256_digest


def validate_analysis_input_derivation_lineage(
    artifact_json: str | None,
    artifact_digest: str | None,
) -> None:
    if artifact_json is None and artifact_digest is None:
        return
    if artifact_json is None or artifact_digest is None:
        raise ValueError(
            "analysis input derivation JSON and digest must be provided together"
        )
    validate_sha256_digest(
        "analysis_input_derivation_artifact_digest",
        artifact_digest,
    )
    try:
        payload = json.loads(artifact_json)
    except json.JSONDecodeError as error:
        raise ValueError("invalid analysis input derivation JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("analysis input derivation payload digest mismatch")
    contract = payload.get("contract")
    expected_fields = {
        "contract",
        "case_id",
        "input_plan_digest",
        "resolved_raw_observation_receipt_digests",
        "canonical_raw_volume_identity_digests",
        "global_raw_resolution_receipt_digest",
        "decoder_version_digest",
        "qc_algorithm_digest",
        "qc_policy_digest",
        "source_selection_evidence_digest",
        "regrid_algorithm_digest",
        "grid_contract_digest",
        "background_cycle_rule_digest",
        "background_valid_times",
        "background_source_identity_digest",
        "background_input_identity_digests",
        "input_frames_digest",
        "observation_masks_digest",
        "observation_quality_weight_digest",
        "observation_std_dbz_digest",
        "background_frames_digest",
        "input_bundle_digest",
        "full_analysis_input_digest",
        "processed_at",
        "processor_id",
        "processor_public_key_hex",
        "processor_signature_hex",
    }
    if contract == "analysis-input-derivation-artifact-v5":
        expected_fields.update(
            {
                "source_available_mask_digest",
                "learned_model_input_features_digest",
            }
        )
    digest_fields = expected_fields - {
        "contract",
        "case_id",
        "resolved_raw_observation_receipt_digests",
        "canonical_raw_volume_identity_digests",
        "background_valid_times",
        "background_source_identity_digest",
        "background_input_identity_digests",
        "background_frames_digest",
        "processed_at",
        "processor_id",
        "processor_public_key_hex",
        "processor_signature_hex",
    }
    raw_receipts = payload.get("resolved_raw_observation_receipt_digests")
    raw_identities = payload.get("canonical_raw_volume_identity_digests")
    background_times = payload.get("background_valid_times")
    background_identities = payload.get("background_input_identity_digests")
    if (
        set(payload) != expected_fields
        or contract not in (
            "analysis-input-derivation-artifact-v4",
            "analysis-input-derivation-artifact-v5",
        )
        or not isinstance(payload.get("case_id"), str)
        or not str(payload.get("case_id", "")).strip()
        or str(payload.get("case_id")) != str(payload.get("case_id")).strip()
        or not isinstance(payload.get("processor_id"), str)
        or not str(payload.get("processor_id", "")).strip()
        or str(payload.get("processor_id"))
        != str(payload.get("processor_id")).strip()
        or not isinstance(raw_receipts, list)
        or not raw_receipts
        or raw_receipts != sorted(raw_receipts)
        or len(set(raw_receipts)) != len(raw_receipts)
        or not isinstance(raw_identities, list)
        or not raw_identities
        or raw_identities != sorted(raw_identities)
        or len(set(raw_identities)) != len(raw_identities)
        or not isinstance(background_times, list)
        or not isinstance(background_identities, list)
        or len(background_times) != len(background_identities)
        or any(
            not isinstance(value, str)
            for value in (*raw_receipts, *raw_identities, *background_identities)
        )
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (*raw_receipts, *raw_identities, *background_identities)
        )
        or any(
            not isinstance(payload.get(name), str)
            for name in digest_fields
        )
        or any(
            len(str(payload[name])) != 64
            or any(character not in "0123456789abcdef" for character in str(payload[name]))
            for name in digest_fields
        )
        or (
            (payload.get("background_frames_digest") is None)
            != (not background_times)
        )
        or (
            (payload.get("background_source_identity_digest") is None)
            != (not background_times)
        )
        or json.dumps(payload, sort_keys=True, separators=(",", ":"))
        != artifact_json
        or json_digest(payload) != artifact_digest
    ):
        raise ValueError("analysis input derivation payload digest mismatch")
    for value in (
        payload.get("background_frames_digest"),
        payload.get("background_source_identity_digest"),
    ):
        if value is not None:
            validate_sha256_digest("background derivation digest", value)
    try:
        processed = datetime.fromisoformat(
            str(payload["processed_at"]).replace("Z", "+00:00")
        )
        if processed.tzinfo is None or (
            processed.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            != payload["processed_at"]
        ):
            raise ValueError
        for value in background_times:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None or (
                parsed.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                != value
            ):
                raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(
            "analysis input derivation time is not canonical"
        ) from error


def validate_analysis_input_derivation_signature(payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    signature_hex = unsigned.pop("processor_signature_hex", None)
    try:
        key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(payload["processor_public_key_hex"]))
        )
        key.verify(
            bytes.fromhex(str(signature_hex)),
            json_digest(unsigned).encode("ascii"),
        )
    except (InvalidSignature, KeyError, TypeError, ValueError) as error:
        raise ValueError("analysis input derivation signature is invalid") from error
