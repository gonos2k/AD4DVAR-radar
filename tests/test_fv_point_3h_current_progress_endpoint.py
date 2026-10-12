from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_PATH = (ROOT / "graphify-out/fv-root-cause-20260919/residual_direction_20261012/endpoint.py")
SPEC = importlib.util.spec_from_file_location("current_progress_endpoint", ENDPOINT_PATH)
assert SPEC is not None and SPEC.loader is not None
endpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(endpoint)


def test_receipt_endpoint_closes_nine_progress_steps_without_forward_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("receipt-only endpoint invoked an FV observation")

    monkeypatch.setattr(endpoint.arithmetic.tangent, "_observe", forbidden)
    admitted = endpoint.load_endpoint()

    assert endpoint.arithmetic.tangent._tensor_sha(admitted["control"]) == endpoint.FINAL_SHA
    assert admitted["G"].shape == (27,)
    assert admitted["repeat"]["control"] == admitted["control"].tolist()
    assert admitted["theta"] == admitted["repeat"]["theta"]
    assert admitted["input_after"]["control_sha256"] == endpoint.FINAL_SHA
    assert admitted["budget"]["internal_overrun_seconds"] > 0.0
    assert admitted["budget"]["outer_budget_passed"] is True
    assert admitted["pins"]["raw_step_sha256"] == endpoint.PROGRESS_TAR_STEP_SHA
    assert admitted["pins"]["source_member_count"] == 228
    assert admitted["pins"]["archive_pin_count"] == 296


def test_endpoint_rejects_a_changed_progress_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    original = endpoint._read

    def changed_manifest(path: Path, digest: str) -> bytes:
        if path == endpoint.RECEIPT_MANIFEST:
            return b"{}"
        return original(path, digest)

    monkeypatch.setattr(endpoint, "_read", changed_manifest)
    with pytest.raises(ValueError, match="manifest|pinned"):
        endpoint.load_endpoint()
