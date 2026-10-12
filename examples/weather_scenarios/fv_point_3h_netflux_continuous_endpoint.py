"""Receipt-only admission of the closed continuous net-flux prefix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_netflux_continuation as continuation
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle
from examples.weather_scenarios import fv_point_3h_netflux_endpoint as cycle_endpoint

tangent, finite = continuation.tangent, continuation.finite

ROOT = continuation.ROOT
WORK = continuation.WORK
PLAN = continuation.PLAN
PLAN_SHA = "5fd879837c2e5898b16b01007aabae34d0eeea7da8b6d094d6d03b9dad366e07"
PLAN_PATH = PLAN.relative_to(ROOT).as_posix()
SOURCE_TAR = WORK / "source.tar.gz"
SOURCE_TAR_SHA = "99c353119ecf9a91c797f9ee7bf7f98098207516d52e78ef5cfec45ed30ea0e4"
RECEIPT_MANIFEST = WORK / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA = "d4bfbddf86bbfa6b0df4318bf69b7069d9b56a1f258df58b3665373ca93586cb"
RECEIPTS_TAR = WORK / "receipts.tar.gz"
RECEIPTS_TAR_SHA = "508463f4a40a270a43e63ecbacb006e060db172abb973913c15e8277417cf0db"
BASE_SHA = "b6470069c9b8992bf19fa85ab848f5ef424a1e2e8e33a4909bcb0d036ef88660"
BASE_RAW_SHA = "a014b58164c3b623cecdec8560be57724e260cb0e04399766a801540193d84ab"
PARENT_CONTROL_SHA = cycle_endpoint.BASE_SHA
NUMERICAL_VERSION = continuation.VERSION
PLAN_PARENT_SHA = continuation.PARENT_SHA
TRANSPORT = cycle.TRANSPORT
TRANSPORT_SHA = cycle.NEW_TRANSPORT_SHA
RECEIPTS = (PLAN, SOURCE_TAR, RECEIPT_MANIFEST, RECEIPTS_TAR, *cycle_endpoint.RECEIPTS)
SELF = "examples/weather_scenarios/fv_point_3h_netflux_continuous_endpoint.py"
TEST = "tests/test_fv_point_3h_netflux_continuous_endpoint.py"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, expected: str | None = None) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"continuous endpoint path is not an in-repository file: {path}")
    data = path.read_bytes()
    if expected is not None and _sha_bytes(data) != expected:
        raise ValueError(f"continuous endpoint pinned file changed: {path.name}")
    return data


def _json(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"continuous endpoint {label} is not a JSON object")
    return value


def _tar_members(data: bytes, expected: set[str], label: str) -> dict[str, bytes]:
    import io
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name in members:
                    raise ValueError(f"{label} contains a non-file or duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{label} member has no data")
                members[member.name] = stream.read()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if set(members) != expected:
        raise ValueError(f"{label} member set differs from its manifest")
    return members


def _validate_child(plan: dict[str, Any], raw: dict[str, Any], run: dict[str, Any],
                    resource: dict[str, Any], members: dict[str, bytes],
                    start: dict[str, Any]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    policy = plan["policy"]
    expected_source = {**plan["source_files"], **plan["archive_files"], PLAN_PATH: PLAN_SHA}
    steps = raw.get("closed_steps")
    if (raw.get("plan_sha256") != PLAN_SHA or raw.get("numerical_version") != NUMERICAL_VERSION
            or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "budget_exhausted"
            or raw.get("termination_reason") != "600-second internal deadline reached"
            or raw.get("deadline_passed") is not False
            or raw.get("failure_stage") is not None
            or raw.get("base_control_sha256") != PARENT_CONTROL_SHA
            or raw.get("base_endpoint_admitted") is not True
            or raw.get("last_confirmed_numerical_version") != NUMERICAL_VERSION
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or raw.get("current_control") != raw.get("last_confirmed_control")
            or raw.get("current_theta") != raw.get("last_confirmed_theta")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or raw.get("optimizer_steps_applied") != 11
            or raw.get("candidate_committed") is not True
            or raw.get("candidate_evaluations_started") != 11
            or raw.get("candidate_evaluations_completed") != 11
            or raw.get("hvp_started") != 24 or raw.get("hvp_completed") != 23
            or raw.get("root_claim") is not False or raw.get("minimum_claim") is not False
            or raw.get("response_claim") is not False or raw.get("score_claim") is not False
            or raw.get("terminal_E_recomputed") is not False
            or raw.get("postcommit_readiness_performed") is not False
            or raw.get("source_before") != expected_source
            or raw.get("source_after") != expected_source
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or raw.get("runtime_before") != raw.get("runtime_after")
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
            or not 0.0 < float(resource["elapsed_seconds"]) <= policy["outer_seconds"]
            or not isinstance(raw.get("elapsed_seconds"), (int, float))
            or float(raw["elapsed_seconds"]) > policy["outer_seconds"]):
        raise ValueError("continuous endpoint budget/run/source/input/runtime closure failed")
    if not isinstance(steps, list) or len(steps) != 11:
        raise ValueError("continuous endpoint does not contain the required 11-step closed prefix")

    current_point = raw.get("current_point")
    if (not isinstance(current_point, dict)
            or current_point.get("point_index") != 11
            or current_point.get("base_control_sha256") != BASE_SHA
            or current_point.get("previous_P2_match", {}).get("passed") is not True
            or current_point.get("minimum_mixture_P2_match", {}).get("passed") is not True
            or current_point.get("outward_derivatives", {}).get("release_required") is not False
            or current_point.get("hvp_started") != 2
            or current_point.get("hvp_completed") != 1
            or "side_HVP" in current_point or "model_gates" in current_point
            or "trials" in current_point or "selected_candidate" in current_point):
        raise ValueError("continuous endpoint terminal partial point is not the expected incomplete HVP pair")
    if (sum(int(step.get("hvp_started", 0)) for step in steps) + int(current_point["hvp_started"]) != 24
            or sum(int(step.get("hvp_completed", 0)) for step in steps)
                + int(current_point["hvp_completed"]) != 23):
        raise ValueError("continuous endpoint HVP totals do not close over committed prefix and partial point")

    previous_g = start["full_G"]
    previous_control_sha = PARENT_CONTROL_SHA
    previous_repeat = start["repeat"]
    for index, step in enumerate(steps):
        if (not isinstance(step, dict) or step.get("point_index") != index
                or step.get("hvp_started") != 2 or step.get("hvp_completed") != 2
                or step.get("previous_P2_match", {}).get("passed") is not True
                or step.get("minimum_mixture_P2_match", {}).get("passed") is not True
                or step.get("outward_derivatives", {}).get("release_required") is not False
                or not isinstance(step.get("model_gates"), dict)
                or not all(step["model_gates"].get(k) is True for k in (
                    "finite", "both_side_gradients_descend", "merit_descends",
                    "envelope_slope_matches_residual_dot"))):
            raise ValueError(f"continuous endpoint point {index} P2/HVP/model closure failed")
        repeat, step_g, control_sha = continuation.previous.endpoint._close_step(
            step, previous_g, previous_control_sha)
        trials = step.get("trials", [])
        selected = step.get("selected_candidate", {})
        receipt_path = selected.get("candidate_receipt")
        candidate_name = f"attempt1/point_{index:02d}/candidate_{index + 1:02d}.json"
        expected_path = (WORK.relative_to(ROOT) / candidate_name).as_posix()
        if (len(trials) != 1 or selected.get("candidate_receipt") != expected_path
                or selected.get("candidate_receipt_sha256") != _sha_bytes(members[candidate_name])
                or selected.get("J_armijo_passed") is not True
                or selected.get("F_squared_armijo_passed") is not True
                or selected.get("finite") is not True
                or selected.get("branch_pair_passed") is not True
                or selected.get("face_audit_passed") is not True):
            raise ValueError(f"continuous endpoint point {index} candidate receipt/Armijo closure failed")
        previous_g, previous_control_sha, previous_repeat = step_g, control_sha, repeat

    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=control.dtype)
    if (control.shape != (26,) or not bool(torch.isfinite(control).all())
            or tangent._tensor_sha(control) != BASE_SHA
            or full_g.shape != (27,) or not bool(torch.isfinite(full_g).all())
            or raw.get("current_control") != raw.get("last_confirmed_control")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or previous_control_sha != BASE_SHA
            or raw.get("current_theta") != previous_repeat.get("theta")
            or raw.get("current_full_G") != steps[-1].get("full_G_check", {}).get("G")
            or raw.get("current_point", {}).get("full_G") != raw.get("current_full_G")
            or raw.get("current_point", {}).get("J") != previous_repeat.get("objective")
            or raw.get("current_point", {}).get("R") != previous_repeat.get("F_squared")):
        raise ValueError("continuous endpoint final committed control/full-G differs from the closed prefix")

    before, after = raw.get("input_before"), raw.get("input_after")
    stable = ("parameters_sha256", "terminal_truth_sha256", "archived_input", "original_flow_fractions")
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or before != start["input_after"]
            or not all(before.get(k) == after.get(k) for k in stable)
            or after.get("control_sha256") != BASE_SHA):
        raise ValueError("continuous endpoint fixed-input/old-case identity failed")
    return control, full_g, previous_repeat


def load_endpoint() -> dict[str, Any]:
    """Load the last confirmed continuous point from committed receipt bytes only."""
    plan = continuation.load_plan(PLAN, PLAN_SHA)
    source_bundle = _read(SOURCE_TAR, SOURCE_TAR_SHA)
    source_members = _tar_members(source_bundle, set(plan["source_files"]), "continuous source bundle")
    if any(_sha_bytes(source_members[name]) != digest for name, digest in plan["source_files"].items()):
        raise ValueError("continuous source-bundle member hash mismatch")
    receipt_manifest = _json(_read(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA), "receipt manifest")
    receipt_files = receipt_manifest.get("files")
    if not isinstance(receipt_files, dict) or len(receipt_files) != 15:
        raise ValueError("continuous receipt manifest has an unexpected member set")
    receipts = _tar_members(_read(RECEIPTS_TAR, RECEIPTS_TAR_SHA), set(receipt_files), "continuous receipts")
    for name, entry in receipt_files.items():
        if (_sha_bytes(receipts[name]) != entry.get("sha256")
                or len(receipts[name]) != entry.get("bytes")):
            raise ValueError(f"continuous receipt bundle member mismatch: {name}")

    raw_bytes = receipts["attempt1/step.json"]
    if _sha_bytes(raw_bytes) != BASE_RAW_SHA:
        raise ValueError("continuous endpoint raw child digest mismatch")
    raw = _json(raw_bytes, "continuous raw child")
    run = _json(receipts["attempt1/step.run.json"], "continuous run receipt")
    resource = _json(receipts["attempt1/step.resource.json"], "continuous resource receipt")
    if run.get("resource") != resource or run.get("child_sha256") != BASE_RAW_SHA:
        raise ValueError("continuous run/resource receipt linkage mismatch")
    start = continuation.endpoint.load_endpoint()
    control, full_g, repeat = _validate_child(plan, raw, run, resource, receipts, start)
    return {"raw": raw, "control": control, "theta": float(repeat["theta"]),
        "full_G": full_g, "repeat": repeat, "input_after": raw["input_after"],
        "runtime_after": raw["runtime_after"], "parent_plan": plan,
        "receipt_files": {path.relative_to(ROOT).as_posix(): _sha_bytes(_read(path))
                          for path in RECEIPTS}}
