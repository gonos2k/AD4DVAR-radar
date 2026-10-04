"""Archive-only renderer regressions; no FV construction or execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from examples.weather_scenarios.fv_archived_response_contract import render_archived_response
from examples.weather_scenarios import fv_archived_response_contract as archived_contract

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
ARCHIVES = (
    EVIDENCE / "point_single_qx_response_attempt2/response.json",
    EVIDENCE / "partial_active_face_response_attempt1/response.json",
    EVIDENCE / "point_centered_response_attempt1/point_centered_response.json",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_profile_fixture(root: Path, archive: Path) -> Path:
    relative_archive = archive.relative_to(ROOT)
    archive_dir = relative_archive.parent
    name = archive_dir.name
    if name == "point_single_qx_response_attempt2":
        required = ("response.json", "preflight.json", "parent.json", "response.resource.json")
    elif name == "partial_active_face_response_attempt1":
        required = ("response.json", "preflight.json", "response.resource.json")
    else:
        required = ("point_centered_response.json", "manifest.json", "point_centered_response.run.json",
                    "point_centered_response.resource.json")
    destination_dir = root / archive_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        shutil.copyfile(ROOT / archive_dir / filename, destination_dir / filename)
    if name == "point_centered_response_attempt1":
        manifest = json.loads((destination_dir / "manifest.json").read_text())
        for filename in manifest["sha256"]:
            if filename.endswith(".py.txt"):
                shutil.copyfile(ROOT / archive_dir / filename, destination_dir / filename)
    support_paths = []
    if name == "point_single_qx_response_attempt2":
        support_paths = ["graphify-out/fv-root-cause-20260919/point_single_qx_normal_attempt2/normal.json"]
    elif name == "partial_active_face_response_attempt1":
        support_paths = [
            "graphify-out/fv-root-cause-20260919/partial_active_face_normal_attempt3/normal.json",
            "examples/weather_scenarios/fv_active_face_stationarity_probe.py",
        ]
    else:
        support_paths = ["graphify-out/fv-root-cause-20260919/FV_POINT_CENTERED_PRIOR_RESPONSE_EVIDENCE.json"]
    for support in support_paths:
        source, destination = ROOT / support, root / support
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root / relative_archive


@pytest.mark.parametrize("archive", ARCHIVES, ids=("qx-face", "two-hole-face", "centered-smooth"))
def test_archived_render_preserves_raw_archive_and_references(archive: Path):
    before = _digest(archive)
    raw = json.loads(archive.read_text())
    result = render_archived_response(archive)
    assert _digest(archive) == before == result["profile"]["archive_sha256"]
    assert result["profile"]["source_before_after_equal"] is True
    assert result["identities"]["source"]["references"]
    assert all(len(value) == 64 for value in result["identities"]["source"]["references"].values())
    assert result["claims"] == {"full_root_supported": False, "physical_validation_supported": False}
    assert result["assessment"]["physical"] == "not_validated"
    assert result["independent_validation"]
    assert [row["status"] for row in result["independent_validation"]] == ["passed", "passed"]
    for row in result["independent_validation"]:
        assert len(row["endpoint_control_sha256"]) == 2
        assert len(row["endpoint_parameters_sha256"]) == 2
        base_parameters_sha = raw.get("parameters_sha256", raw.get("nominal", {}).get("parameters_sha256"))
        assert row["parameters_sha256"] == base_parameters_sha
        assert row["step_sizes"] and row["direction_sha256"] == raw.get(
            "direction_sha256", raw.get("nominal", {}).get("direction_sha256")
        )
    assert result["response"]["total"][result["response"]["direction_name"]] == pytest.approx(
        raw.get("response", raw.get("base_response", {})).get("total", raw.get("response", {}).get("total"))
        if "nominal" not in raw else raw["response"]["total"]
    )


def test_selected_face_profiles_keep_their_distinct_static_charts_and_dims():
    qx = render_archived_response(ARCHIVES[0])
    two_hole = render_archived_response(ARCHIVES[1])
    assert qx["kind"] == two_hole["kind"] == "selected_face_conditional"
    assert qx["geometry"]["selected_face"] == {"axis": "x", "row": 3, "column": 4, "value": 0.0}
    assert two_hole["geometry"]["selected_face"] == {"axis": "y", "row": 2, "column": 0, "value": 0.0}
    assert qx["geometry"]["chart"]["pivot_flow_index"] == 0
    assert two_hole["geometry"]["chart"]["pivot_flow_index"] == 1
    assert qx["coordinates"]["ambient_dimension"] == two_hole["coordinates"]["ambient_dimension"] == 26
    assert qx["coordinates"]["tangent_dimension"] == two_hole["coordinates"]["tangent_dimension"] == 25
    assert qx["identities"]["parameters"]["dimension"] == 13
    assert two_hole["identities"]["parameters"]["dimension"] == 61
    assert qx["assessment"]["execution"] == "completed"
    assert two_hole["assessment"]["execution"] == "not_recorded"
    assert two_hole["resource"]["wrapper_status_source"] == "not_recorded; child record only"
    assert qx["response"]["direct"]["middle_four_qx_face_plus1"] == pytest.approx(0.0)
    assert qx["response"]["indirect"]["middle_four_qx_face_plus1"] == pytest.approx(
        json.loads(ARCHIVES[0].read_text())["base_response"]["indirect"]
    )
    for envelope, archive in ((qx, ARCHIVES[0]), (two_hole, ARCHIVES[1])):
        raw = json.loads(archive.read_text())
        ref = envelope["geometry"]["normal_evidence"]["references"][0]
        normal = raw.get("base_normal", raw.get("base_normal_recomputed"))
        canonical = json.dumps(normal, sort_keys=True, separators=(",", ":")).encode()
        assert ref["sha256"] == hashlib.sha256(canonical).hexdigest()
        assert ref["archive_sha256"] == envelope["profile"]["archive_sha256"]
        assert ref["json_pointer"] in {"/base_normal", "/base_normal_recomputed"}
        assert ref["control_sha256"] == raw["base_control_sha256"]
        assert ref["parameters_sha256"] == raw["parameters_sha256"]
        assert ref["tangent_sha256"] == raw["base_tangent_sha256"]
        historical = envelope["response"]["historical_prequalification_normal_reference"]
        assert historical["sha256"] == raw["normal_result_sha256"]
        assert historical["status"] == "historical_prequalification_only_not_bound_to_current_base_point"
        assert "control_sha256" not in historical
        assert envelope["geometry"]["normal_evidence"]["parameters_sha256"] == raw["parameters_sha256"]
    assert "four middle-time point-observation values in dBZ" in qx["independent_validation"][0]["scope"]


def test_centered_profile_keeps_its_constructed_prior_and_smooth_full_control_scope():
    result = render_archived_response(ARCHIVES[2])
    assert result["kind"] == "smooth_stationary"
    assert result["geometry"] == {"selected_face": None, "chart": None, "normal_evidence": None}
    assert result["coordinates"]["ambient_dimension"] == result["coordinates"]["tangent_dimension"] == 26
    assert result["prior"]["mean_dimension"] == 6
    assert "not original zero-prior" in result["prior"]["contract"]
    assert result["assessment"]["execution"] == "completed"


@pytest.mark.parametrize("archive", ARCHIVES)
def test_stationarity_uses_archived_profile_gate_and_source_sha(archive: Path):
    result = render_archived_response(archive)
    stationarity = result["stationarity"]
    criterion = stationarity["criterion"]
    source_refs = result["identities"]["source"]["references"]
    assert stationarity["tolerance"] == criterion["tolerance"] == 1e-10
    assert stationarity["value"] < stationarity["tolerance"]
    assert stationarity["status"] == "reported_pass_at_archived_fixed_gate"
    assert criterion["comparison"] == "pass iff gradient_max < tolerance"
    assert criterion["gate_source"]["sha256"] == source_refs[criterion["gate_source"]["path"]]
    local = criterion["local_response_source"]
    if "evidence_record_path" in local:
        evidence_path = ROOT / local["evidence_record_path"]
        evidence = json.loads(evidence_path.read_text())
        assert _digest(evidence_path) == local["evidence_record_sha256"]
        assert local["sha256"] == evidence["sha256"][local["path"]]
    else:
        assert local["sha256"] == source_refs[local["path"]]


def test_renderer_refuses_unregistered_archive_profiles():
    with pytest.raises(ValueError, match="unknown FV response archive"):
        render_archived_response(EVIDENCE / "point_single_qx_response_attempt1/response.json")


def test_renderer_rejects_changed_registered_archive_bytes(tmp_path: Path):
    archive = ARCHIVES[0]
    relative = archive.relative_to(ROOT)
    altered = tmp_path / relative
    altered.parent.mkdir(parents=True)
    altered.write_bytes(archive.read_bytes() + b" ")
    with pytest.raises(ValueError, match="registered archived response SHA-256 mismatch"):
        render_archived_response(altered, root=tmp_path)


def test_inline_normal_evidence_is_point_bound_not_the_historical_file():
    result = render_archived_response(ARCHIVES[0])
    inline = result["geometry"]["normal_evidence"]["references"][0]
    historical = result["response"]["historical_prequalification_normal_reference"]
    assert inline["json_pointer"] == "/base_normal"
    assert inline["control_sha256"] == result["identities"]["control"]["sha256"]
    assert historical["sha256"] != inline["sha256"]
    assert historical["status"].endswith("not_bound_to_current_base_point")


@pytest.mark.parametrize("archive", ARCHIVES, ids=("qx-face", "two-hole-face", "centered-smooth"))
def test_full_registered_profile_renders_from_alternate_caller_root(tmp_path: Path, monkeypatch, archive: Path):
    root_a = tmp_path / "importer-a"
    root_a.mkdir()
    root_b = tmp_path / "caller-b"
    copied_archive = _copy_profile_fixture(root_b, archive)
    monkeypatch.setattr(archived_contract, "_ROOT", root_a)

    result = render_archived_response(copied_archive, root=root_b)

    assert result["profile"]["archive_sha256"] == _digest(copied_archive)
    assert result["profile"]["source_before_after_equal"] is True
    assert result["stationarity"]["criterion"]["gate_source"]["path"]
    assert result["independent_validation"]
    if result["kind"] == "selected_face_conditional":
        assert result["geometry"]["normal_evidence"]["parameters_sha256"] == result[
            "identities"]["parameters"]["sha256"]


def test_alternate_root_source_mismatch_rejects_even_when_importer_root_has_original(tmp_path: Path,
                                                                                     monkeypatch):
    root_b = tmp_path / "caller-b"
    archive = _copy_profile_fixture(root_b, ARCHIVES[1])
    source = root_b / "examples/weather_scenarios/fv_active_face_stationarity_probe.py"
    source.write_bytes(source.read_bytes() + b"\n# changed in caller root\n")
    monkeypatch.setattr(archived_contract, "_ROOT", ROOT)

    with pytest.raises(ValueError, match="differs from the archived source SHA"):
        render_archived_response(archive, root=root_b)


@pytest.mark.parametrize("archive", ARCHIVES, ids=("qx-face", "two-hole-face", "centered-smooth"))
def test_absolute_archive_paths_through_root_symlink_render_from_canonical_root(tmp_path: Path,
                                                                               monkeypatch, archive: Path):
    root_a = tmp_path / "empty-importer-a"
    root_a.mkdir()
    root_b = tmp_path / "caller-b"
    canonical_archive = _copy_profile_fixture(root_b, archive)
    alias = tmp_path / "caller-b-alias"
    alias.symlink_to(root_b, target_is_directory=True)
    monkeypatch.setattr(archived_contract, "_ROOT", root_a)

    result = render_archived_response(alias / canonical_archive.relative_to(root_b), root=alias)

    assert result["profile"]["archive_sha256"] == _digest(canonical_archive)
    assert result["identities"]["source"]["before_after_equal"] is True


def test_relative_caller_root_and_archive_path_are_supported(tmp_path: Path, monkeypatch):
    root_a = tmp_path / "empty-importer-a"
    root_a.mkdir()
    root_b = tmp_path / "caller-b"
    archive = _copy_profile_fixture(root_b, ARCHIVES[0])
    relative_root = Path(os.path.relpath(root_b, Path.cwd()))
    relative_archive = archive.relative_to(root_b)
    monkeypatch.setattr(archived_contract, "_ROOT", root_a)

    result = render_archived_response(relative_archive, root=relative_root)

    assert result["profile"]["archive_sha256"] == _digest(archive)


def test_archive_symlink_escaping_caller_root_is_rejected(tmp_path: Path):
    root = tmp_path / "caller-root"
    relative = ARCHIVES[0].relative_to(ROOT)
    link = root / relative
    link.parent.mkdir(parents=True)
    link.symlink_to(ARCHIVES[0])
    with pytest.raises(ValueError, match="archive must be inside the repository root"):
        render_archived_response(link, root=root)


@pytest.mark.parametrize(("archive_index", "dependency"), [
    (0, "parent.json"),
    (0, "response.resource.json"),
    (2, "point_centered_response.run.json"),
])
def test_alternate_root_changed_registered_sidecar_is_rejected(tmp_path: Path,
                                                               archive_index: int,
                                                               dependency: str):
    root_b = tmp_path / "caller-b"
    archive = _copy_profile_fixture(root_b, ARCHIVES[archive_index])
    sidecar = archive.parent / dependency
    data = json.loads(sidecar.read_text())
    data["execution_status"] = "refused"
    sidecar.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="registered archive dependency SHA-256 mismatch"):
        render_archived_response(archive, root=root_b)
