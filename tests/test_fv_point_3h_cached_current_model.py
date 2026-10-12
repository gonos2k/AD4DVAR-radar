"""Regression for PR302's pinned offline current-model operator cache."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (ROOT / "graphify-out/fv-root-cause-20260919/subspace_active_qr_20261012"
              / "current_model_audit.py")
SPEC = importlib.util.spec_from_file_location("fv_cached_current_model_audit", AUDIT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the cached current-model audit")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_cached_current_model_closes_pins_and_rejects_unresolved_gn_gain(tmp_path: Path):
    report = AUDIT.run(tmp_path / "current-model.json")

    assert report["status"] == "offline_cached_current_model_audit_complete"
    assert report["base"]["control_sha256"] == AUDIT.BASE_SHA256
    assert report["base"]["control_dimension"] == 26
    assert report["base"]["full_G_dimension"] == 27
    assert report["cached_operator_audit"]["jacobian_rows"] == 24
    assert report["cached_operator_audit"]["control_dimension"] == 26
    assert report["cached_operator_audit"]["legacy_GN_B_shape"] == [12, 26]
    assert report["cached_operator_audit"]["legacy_GN_S_shape"] == [12, 12]
    assert report["cached_operator_audit"]["HVPs_started_completed"] == [4, 4]
    assert report["cached_operator_audit"]["candidate_evaluations"] == [0, 0]
    assert report["cached_operator_audit"]["commits"] == 0
    assert set(report["cached_side_HVPs"]["baseline_tangent"]) == {"-1", "1"}

    gate = report["incremental_GN_gate_by_radius"]
    assert [item["radius_multiple"] for item in gate] == [1.0, 1.25, 1.5, 2.0]
    assert all(item["residual_derivative_rank"] == 2 for item in gate)
    assert all(item["selected_active_set"] == "axis_0" for item in gate)
    assert all(item["normalized_coefficients"][1] == 0.0 for item in gate)
    assert all(item["added_predicted_R_decrease"] == 0.0 for item in gate)
    assert all(not item["GN_gate_passed"] for item in gate)
