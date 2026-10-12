from __future__ import annotations

import pytest

from examples.weather_scenarios import fv_point_3h_netflux_comparison_endpoint as endpoint


def test_receipt_endpoint_admits_d19e_without_model_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("receipt-only endpoint invoked production evaluation")

    monkeypatch.setattr(endpoint.comparison, "CurrentBackend", forbidden)
    monkeypatch.setattr(endpoint.comparison.tangent, "_observe", forbidden)
    admitted = endpoint.load_endpoint()

    assert endpoint.comparison.tangent._tensor_sha(admitted["control"]) == endpoint.BASE_SHA
    assert admitted["full_G"].shape == (27,)
    assert admitted["repeat"]["control"] == admitted["control"].tolist()
    assert admitted["theta"] == admitted["repeat"]["theta"]
    assert admitted["input_after"]["control_sha256"] == endpoint.BASE_SHA
    assert admitted["parent_plan"]["experiment_kind"] == "same_point_fixed_version_gradient_robust_GN"
    assert admitted["receipt_files"][endpoint.PLAN.relative_to(endpoint.ROOT).as_posix()] == endpoint.PLAN_SHA


def test_endpoint_rejects_a_changed_receipt_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    original = endpoint._read

    def changed_manifest(path, digest):
        if path == endpoint.RECEIPT_MANIFEST:
            return b"{}"
        return original(path, digest)

    monkeypatch.setattr(endpoint, "_read", changed_manifest)
    with pytest.raises(ValueError, match="manifest|pinned"):
        endpoint.load_endpoint()
