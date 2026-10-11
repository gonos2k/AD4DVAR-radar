"""Receipt-only endpoint admission for the net-flux numerical evaluator."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as old
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent

ROOT = cycle.ROOT
WORK = cycle.WORK
PLAN = cycle.PLAN
PLAN_SHA = "7f26c76bad511e1345c2d3c27b2073908c574587fc3aefd433fb44b98a932441"
PLAN_PATH = PLAN.relative_to(ROOT).as_posix()
PARENT_PLAN = cycle.PARENT_PLAN
PARENT_PLAN_SHA = cycle.PARENT_SHA
SOURCE_TAR = WORK / "source.tar.gz"
SOURCE_TAR_SHA = "e84e0e2612dc372139a9e67ed377c71f0fa54c59db1cfd1f8e7a26d1bdb0e251"
RECEIPT_MANIFEST = WORK / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA = "6593e3b5da7eb9886f8c9abc63bd6538105c61c2810b31d9379b519a6b9f3db2"
RECEIPTS_TAR = WORK / "receipts.tar.gz"
RECEIPTS_TAR_SHA = "fc3c2bbb9f0612131d2d2d295eb1f7464fbd02d91f347925cbf482739e025412"
CASE = cycle.CASE
CASE_MANIFEST = cycle.CASE_MANIFEST
BASE_SHA = "0aafdf8069fcb8d5adc4ec813646707a3d1e083804631b17facb24664098655b"
BASE_RAW_SHA = "620ba8f831eb249175dea6b2dbe26657927a1006f4070d861df04476c564e7ed"
PARENT_CONTROL_SHA = "556a25d910cad707b108cc8c6770af9ebf6b34f5a84196736e9fee82ed298a8b"
NUMERICAL_VERSION = cycle.VERSION
TRANSPORT = cycle.TRANSPORT
OLD_TRANSPORT_SHA = cycle.OLD_TRANSPORT_SHA
NEW_TRANSPORT_SHA = cycle.NEW_TRANSPORT_SHA
RECEIPTS = (PLAN, PARENT_PLAN, SOURCE_TAR, RECEIPT_MANIFEST, RECEIPTS_TAR,
            CASE_MANIFEST, CASE, cycle.LEGACY_TRANSPORT)
SELF = "examples/weather_scenarios/fv_point_3h_netflux_endpoint.py"
TEST = "tests/test_fv_point_3h_netflux_endpoint.py"
FACE_WEIGHTS = torch.tensor([0.0, -0.08, -0.28, -0.14, -0.84], dtype=torch.float64)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, expected_sha: str | None = None) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"net-flux receipt path is not an in-repository file: {path}")
    data = path.read_bytes()
    if expected_sha is not None and _sha_bytes(data) != expected_sha:
        raise ValueError(f"net-flux pinned receipt changed: {path.name}")
    return data


def _load_json(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _bundle_members(data: bytes, expected_names: set[str]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    try:
        import io
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name in members:
                    raise ValueError("receipt tar contains a non-file or duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("receipt tar member has no data")
                members[member.name] = stream.read()
    except (tarfile.TarError, OSError) as error:
        raise ValueError("receipt tar is unreadable") from error
    if set(members) != expected_names:
        raise ValueError("receipt tar member set differs from its manifest")
    return members


def _validate_child(plan: dict[str, Any], raw: dict[str, Any], run: dict[str, Any],
                    resource: dict[str, Any], members: dict[str, bytes],
                    case_manifest: dict[str, Any]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    policy = plan["policy"]
    if (raw.get("plan_sha256") != PLAN_SHA or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "commit_cap_reached"
            or raw.get("numerical_version") != NUMERICAL_VERSION
            or raw.get("base_requalified") is not True
            or raw.get("base_control_sha256") != PARENT_CONTROL_SHA
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or raw.get("current_control") != raw.get("last_confirmed_control")
            or raw.get("current_theta") != raw.get("last_confirmed_theta")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or raw.get("candidate_committed") is not True
            or raw.get("optimizer_steps_applied") != 1
            or raw.get("hvp_started") != 2 or raw.get("hvp_completed") != 2
            or raw.get("candidate_evaluations_started") != 1
            or raw.get("candidate_evaluations_completed") != 1
            or raw.get("root_claim") is not False or raw.get("minimum_claim") is not False
            or raw.get("response_claim") is not False or raw.get("score_claim") is not False
            or raw.get("terminal_E_recomputed") is not False
            or raw.get("postcommit_readiness_performed") is not False
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or raw.get("deadline_passed") is not True
            or raw.get("runtime_before") != raw.get("runtime_after")
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("source_before") != {**plan["source_files"], **plan["archive_files"],
                PLAN_PATH: PLAN_SHA}
            or run.get("execution_status") != "completed"
            or run.get("child_sha256") != BASE_RAW_SHA
            or run.get("resource") != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != policy["outer_seconds"]
            or resource.get("rss_limit_bytes") != policy["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", policy["rss_bytes"]) >= policy["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or not 0.0 < float(resource["elapsed_seconds"]) <= policy["outer_seconds"]):
        raise ValueError("net-flux endpoint execution/resource/prefix closure failed")

    base = raw.get("base_requalification")
    if (not isinstance(base, dict)
            or base.get("status") != "new_evaluator_current_P2_requalified"
            or base.get("control_sha256") != PARENT_CONTROL_SHA
            or base.get("old_transport_source_sha256") != OLD_TRANSPORT_SHA
            or base.get("new_transport_source_sha256") != NEW_TRANSPORT_SHA
            or base.get("source_unchanged") is not True
            or base.get("input_unchanged") is not True
            or base.get("runtime_unchanged") is not True
            or base.get("new_repeat_matches_first_full_G") is not True
            or base.get("full_G_check", {}).get("passed") is not True):
        raise ValueError("net-flux fresh base P2/source-version requalification failed")
    if (base.get("input_before") != case_manifest.get("old_input_identity")
            or raw.get("input_before") != case_manifest.get("old_input_identity")
            or raw.get("runtime_before") != case_manifest.get("runtime_identity")
            or base.get("runtime_before") != case_manifest.get("runtime_identity")
            or base.get("input_after") != case_manifest.get("old_input_identity")
            or base.get("source_identity", {}).get(TRANSPORT) != NEW_TRANSPORT_SHA):
        raise ValueError("net-flux old-input asset/source-version identity failed")

    steps = raw.get("closed_steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("net-flux endpoint must close exactly one accepted prefix step")
    step = steps[0]
    point = raw.get("current_point")
    if (not isinstance(step, dict) or not isinstance(point, dict)
            or step.get("point_index") != 0 or step.get("base_control_sha256") != PARENT_CONTROL_SHA
            or raw.get("last_confirmed_numerical_version") != NUMERICAL_VERSION
            or point.get("previous_P2_match", {}).get("passed") is not True
            or point.get("minimum_mixture_P2_match", {}).get("passed") is not True
            or point.get("outward_derivatives", {}).get("release_required") is not False
            or point.get("direction_audit", {}).get("common_face_gradient_jump") is not True
            or point.get("hvp_started") != 2 or point.get("hvp_completed") != 2):
        raise ValueError("net-flux accepted-prefix/current-P2 closure failed")
    gates = point.get("model_gates", {})
    if not all(gates.get(key) is True for key in (
            "finite", "both_side_gradients_descend", "merit_descends",
            "envelope_slope_matches_residual_dot")):
        raise ValueError("net-flux current direction model gates do not close")

    proposal, repeat = step.get("proposal"), step.get("final_repeat")
    full_check = step.get("full_G_check")
    selected = point.get("selected_candidate")
    if (not isinstance(proposal, dict) or not isinstance(repeat, dict)
            or not isinstance(full_check, dict) or not isinstance(selected, dict)
            or proposal.get("conditional_Q0") is not True
            or proposal.get("J_armijo_passed") is not True
            or proposal.get("F_squared_armijo_passed") is not True
            or not math.isfinite(float(proposal.get("J_armijo_bound", float("nan"))))
            or not math.isfinite(float(proposal.get("F_squared_armijo_bound", float("nan"))))
            or not math.isfinite(float(repeat.get("objective", float("nan"))))
            or not math.isfinite(float(repeat.get("F_squared", float("nan"))))
            or float(repeat["objective"]) > float(proposal["J_armijo_bound"])
            or float(repeat["F_squared"]) > float(proposal["F_squared_armijo_bound"])
            or selected.get("J_armijo_passed") is not True
            or selected.get("F_squared_armijo_passed") is not True
            or selected.get("branch_pair_passed") is not True
            or selected.get("face_audit_passed") is not True
            or selected.get("finite") is not True
            or not tangent.fresh_final_closure(proposal, repeat)
            or full_check.get("passed") is not True
            or full_check.get("component_match", {}).get("passed") is not True
            or full_check.get("merit_match") is not True):
        raise ValueError("net-flux accepted candidate/repeat/full-G closure failed")

    receipt_path = selected.get("candidate_receipt")
    candidate_name = f"attempt1/point_00/candidate_01.json"
    if (receipt_path != f"graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/{candidate_name}"
            or selected.get("candidate_receipt_sha256") != _sha_bytes(members[candidate_name])):
        raise ValueError("net-flux candidate receipt linkage failed")
    raw_control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=raw_control.dtype)
    if (raw_control.shape != (26,) or not bool(torch.isfinite(raw_control).all())
            or tangent._tensor_sha(raw_control) != BASE_SHA
            or repeat.get("control") != raw.get("current_control")
            or repeat.get("theta") != raw.get("current_theta")
            or full_g.shape != (27,) or not bool(torch.isfinite(full_g).all())):
        raise ValueError("net-flux final control/theta/full-G is malformed")
    closure = old.fresh_full_G(repeat, full_g, FACE_WEIGHTS)
    actual_g = torch.as_tensor(closure.get("G"), dtype=full_g.dtype)
    if (closure.get("passed") is not True or actual_g.shape != (27,)
            or old.finite._full_G_match(actual_g, full_g).get("passed") is not True
            or raw.get("current_full_G") != step.get("full_G_check", {}).get("G")):
        raise ValueError("net-flux final repeat does not reconstruct the stored full-G")

    before, after = raw.get("input_before"), raw.get("input_after")
    fixed_keys = ("parameters_sha256", "terminal_truth_sha256", "archived_input",
                  "original_flow_fractions")
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or not all(before.get(key) == after.get(key) for key in fixed_keys)
            or after.get("control_sha256") != BASE_SHA):
        raise ValueError("net-flux fixed observation/parameter/truth identity changed")
    return raw_control, full_g, repeat


def load_endpoint() -> dict[str, Any]:
    """Admit the prior net-flux cycle from pinned bytes only; no FV or seed calls."""
    plan_bytes = _read(PLAN, PLAN_SHA)
    plan = _load_json(plan_bytes, PLAN.name)
    parent_bytes = _read(PARENT_PLAN, PARENT_PLAN_SHA)
    parent = _load_json(parent_bytes, PARENT_PLAN.name)
    source_tar = _read(SOURCE_TAR, SOURCE_TAR_SHA)
    manifest_bytes = _read(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA)
    receipt_manifest = _load_json(manifest_bytes, RECEIPT_MANIFEST.name)
    receipts_tar = _read(RECEIPTS_TAR, RECEIPTS_TAR_SHA)
    case_manifest = _load_json(_read(CASE_MANIFEST), CASE_MANIFEST.name)
    _read(CASE, cycle.CASE_SHA)
    _read(cycle.LEGACY_TRANSPORT, OLD_TRANSPORT_SHA)

    source_names = set(plan.get("source_files", {}))
    archive_names = set(plan.get("archive_files", {}))
    expected_sources = set(parent["source_files"]) | cycle.NEW_SOURCES
    expected_archives = set(parent["archive_files"]) | {
        path.relative_to(ROOT).as_posix() for path in cycle.EXTRA_ARCHIVES}
    if (plan.get("experiment_kind") != "netflux_same_control_requalification_and_one_gradient_cycle"
            or plan.get("base_control_sha256") != PARENT_CONTROL_SHA
            or plan.get("numerical_change") != cycle.NUMERICAL_CHANGE
            or plan.get("policy") != cycle.POLICY
            or source_names != expected_sources or archive_names != expected_archives
            or plan["source_files"].get(TRANSPORT) != NEW_TRANSPORT_SHA
            or parent["source_files"].get(TRANSPORT) != OLD_TRANSPORT_SHA
            or case_manifest.get("case_asset_sha256") != cycle.CASE_SHA
            or case_manifest.get("legacy_transport_sha256") != OLD_TRANSPORT_SHA
            or case_manifest.get("roundtrip_input_unchanged") is not True):
        raise ValueError("net-flux endpoint plan/source-version/fixed-case scope changed")
    for name, digest in parent["source_files"].items():
        expected = NEW_TRANSPORT_SHA if name == TRANSPORT else digest
        if plan["source_files"].get(name) != expected:
            raise ValueError(f"net-flux inherited source transition is not allowed: {name}")
    for name, digest in parent["archive_files"].items():
        if plan["archive_files"].get(name) != digest:
            raise ValueError(f"net-flux inherited archive changed: {name}")
    for name, digest in {**plan["source_files"], **plan["archive_files"]}.items():
        if _sha_bytes(_read(ROOT / name)) != digest:
            raise ValueError(f"net-flux endpoint plan pin mismatch: {name}")

    source_members = _bundle_members(source_tar, source_names)
    if any(_sha_bytes(source_members[name]) != digest
           for name, digest in plan["source_files"].items()):
        raise ValueError("net-flux source bundle member hash mismatch")

    receipt_files = receipt_manifest.get("files")
    if not isinstance(receipt_files, dict) or set(receipt_files) != {
            "attempt1/step.run.json", "attempt1/step.json", "attempt1/step.resource.json",
            "attempt1/step.log", "attempt1/base_requalification.json",
            "attempt1/point_00/candidate_01.json"}:
        raise ValueError("net-flux receipt manifest member set changed")
    receipt_members = _bundle_members(receipts_tar, set(receipt_files))
    for name, entry in receipt_files.items():
        if (_sha_bytes(receipt_members[name]) != entry.get("sha256")
                or len(receipt_members[name]) != entry.get("bytes")):
            raise ValueError(f"net-flux receipt bundle member mismatch: {name}")
    raw_bytes = receipt_members["attempt1/step.json"]
    if _sha_bytes(raw_bytes) != BASE_RAW_SHA:
        raise ValueError("net-flux raw child digest mismatch")
    raw = _load_json(raw_bytes, "attempt1/step.json")
    run = _load_json(receipt_members["attempt1/step.run.json"], "step.run.json")
    resource = _load_json(receipt_members["attempt1/step.resource.json"], "step.resource.json")
    if run.get("resource") != resource or run.get("child_sha256") != BASE_RAW_SHA:
        raise ValueError("net-flux run/guard receipt mismatch")
    control, full_g, repeat = _validate_child(
        plan, raw, run, resource, receipt_members, case_manifest)
    return {"raw": raw, "control": control, "theta": float(raw["current_theta"]),
        "full_G": full_g, "repeat": repeat, "input_after": raw["input_after"],
        "runtime_after": raw["runtime_after"], "parent_plan": plan,
        "receipt_files": {path.relative_to(ROOT).as_posix(): _sha_bytes(_read(path))
                          for path in RECEIPTS}}
