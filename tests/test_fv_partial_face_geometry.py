"""Fail-closed geometry accounting for the archived partial FV root attempt."""

import json
from pathlib import Path

import pytest
import torch

from examples.weather_scenarios import fv_partial_face_geometry as geometry
from tests.test_fv_research_partial_observation import _problem


def test_diagnostic_cannot_write_inside_raw_archive():
    destination = geometry.ARCHIVE / "unwritten_geometry.json"
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside the raw archive"):
        geometry.run(destination, expected_plan_sha256="x", expected_probe_sha256="y")
    assert not destination.exists()


def test_diagnostic_cannot_overwrite_existing_temporary_symlink(tmp_path):
    target = tmp_path / "user_data.json"
    target.write_text("preserve")
    destination = tmp_path / "geometry.json"
    temporary = tmp_path / "geometry.json.tmp"
    temporary.symlink_to(target)
    with pytest.raises(ValueError, match="must be fresh"):
        geometry.run(destination, expected_plan_sha256="x", expected_probe_sha256="y")
    assert target.read_text() == "preserve"
    assert temporary.is_symlink()


def test_changed_transport_source_is_not_attributed_to_archived_geometry(monkeypatch, tmp_path):
    actual_sha = geometry._sha
    transport_path = geometry.ROOT / "src/advar/transport.py"
    def altered_sha(path: Path) -> str:
        return "0" * 64 if path == transport_path else actual_sha(path)
    monkeypatch.setattr(geometry, "_sha", altered_sha)
    monkeypatch.setattr(geometry.prior, "_preflight_identity", lambda: pytest.fail(
        "source mismatch must fail before the FV preflight"))
    with pytest.raises(ValueError, match="archived geometric source changed"):
        geometry.run(
            tmp_path / "geometry.json",
            expected_plan_sha256=actual_sha(geometry.PLAN),
            expected_probe_sha256=actual_sha(Path(geometry.__file__)),
        )


def test_archived_seed_face_sign_and_ratio_are_recomputed_from_production_map():
    problem, _, _ = _problem()
    raw = json.loads((geometry.ARCHIVE / "partial_sector_root.json").read_text())
    control = torch.tensor(raw["gauss_newton"]["control"], dtype=torch.float64)
    face = geometry._face_geometry(
        control, problem.frozen.fv_transport,
        problem.frozen.nowcast_config.interval_minutes,
    )
    assert face["minimizing_face"] == {"kind": "qy", "index": [2, 0]}
    assert face["minimizer_count_at_roundoff_scale"] == 1
    assert face["minimum_sign"] == -1
    assert abs(face["ratio"] - raw["seed_branch"]["minimum_scaled_face_flux_margin"]) < 1e-12
