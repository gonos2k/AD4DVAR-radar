"""The precision audit must compare the original normalized Armijo metric."""
import math
import json
import pytest

from examples.weather_scenarios import fv_slice_precision_probe as probe
from examples.weather_scenarios.fv_slice_precision_probe import armijo_ratio


def test_norm_decrease_can_still_fail_normalized_armijo():
    # Norm decreases, but not by the sufficient-decrease factor at slope=-1.
    candidate_norm = 0.99995
    ratio = armijo_ratio(candidate_norm, 1.0, -1.0, 1.0)
    assert candidate_norm < 1.0
    assert ratio > 1.0
    assert math.isclose(ratio * math.sqrt(0.9998), candidate_norm, rel_tol=2e-16)


def test_runtime_mismatch_refuses_before_problem_construction(tmp_path, monkeypatch):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"environment": {"python": "other", "torch": "other", "device": "CPU FP64"}}))
    monkeypatch.setattr(probe, "RAW", raw)
    monkeypatch.setattr(probe, "RAW_SHA", probe.sha(raw))
    def forbidden_problem():
        raise AssertionError("runtime mismatch reached the FV problem")
    monkeypatch.setattr(probe.slices, "_current_problem", forbidden_problem)
    with pytest.raises(probe.slices.SliceIdentityRefusal, match="runtime differs"):
        probe.main(tmp_path / "result.json")
