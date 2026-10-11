"""Receipt-only admission of the closed 556a current-gradient endpoint."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PARENT_PLAN = EVIDENCE / "GRADIENT_CONTINUATION_PLAN_20261011.json"
PARENT_PLAN_SHA = "a3e5b0cfb9d639a236efc0d126bfec91c2fee891108f14340cdfe28a3bf14b92"
PARENT_ARCHIVE = EVIDENCE / "GRADIENT_CONTINUATION_ARCHIVE_20261011.json"
ATTEMPT = EVIDENCE / "current_gradient_continuation_20261011_attempt1"
BASE_GZIP = ATTEMPT / "step.json.gz"
BASE_RUN = ATTEMPT / "step.run.json"
BASE_RESOURCE = ATTEMPT / "step.resource.json"
BASE_SHA = "556a25d910cad707b108cc8c6770af9ebf6b34f5a84196736e9fee82ed298a8b"
BASE_RAW_SHA = "c64cde977e9a28dc28addd81d6d116aa7de6b1e005a6e1643622c9a7b28e5e37"
RAW = ATTEMPT / "step.json"  # Virtual manifest member; bytes come from BASE_GZIP.
RECEIPTS = (PARENT_PLAN, PARENT_ARCHIVE, BASE_GZIP, BASE_RUN, BASE_RESOURCE)
SELF = "examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py"
TEST = "tests/test_fv_point_3h_current_gradient_endpoint.py"

from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as finite  # noqa: E402
from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as helpers  # noqa: E402
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent  # noqa: E402

FACE_WEIGHTS = torch.tensor([0.0, -0.08, -0.28, -0.14, -0.84], dtype=torch.float64)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"receipt path is not a regular in-repository file: {path}")
    return path.read_bytes()


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = _read_bytes(path)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a JSON object: {path.name}")
    return data, value


def _close_step(step: dict[str, Any], previous_g: Tensor | None,
                previous_control_sha: str) -> tuple[dict[str, Any], Tensor, str]:
    proposal, repeat, full_check = (step.get("proposal"), step.get("final_repeat"),
        step.get("full_G_check"))
    p2 = step.get("previous_P2_match")
    if (not isinstance(proposal, dict) or not isinstance(repeat, dict)
            or not isinstance(full_check, dict) or not isinstance(p2, dict)
            or p2.get("passed") is not True
            or p2.get("component_match", {}).get("passed") is not True
            or p2.get("side_gradient_match") is not True or p2.get("trace_match") is not True
            or not float(p2.get("theta_error", float("inf"))) <= float(p2.get("theta_budget", 0.0))
            or not float(p2.get("objective_error", float("inf"))) <= float(p2.get("objective_budget", 0.0))
            or not tangent.fresh_final_closure(proposal, repeat)
            or proposal.get("J_armijo_passed") is not True
            or proposal.get("F_squared_armijo_passed") is not True
            or proposal.get("conditional_Q0") is not True
            or float(repeat["objective"]) > float(proposal["J_armijo_bound"])
            or float(repeat["F_squared"]) > float(proposal["F_squared_armijo_bound"])):
        raise ValueError("closed step failed original-J/P2/previous-P2 qualification")

    control = torch.as_tensor(repeat["control"], dtype=torch.float64)
    control_sha = tangent._tensor_sha(control)
    if control.shape != (26,) or not bool(torch.isfinite(control).all()):
        raise ValueError("closed repeat control is malformed or nonfinite")
    expected = torch.as_tensor(full_check.get("G"), dtype=control.dtype)
    reconstructed = helpers.fresh_full_G(repeat, expected, FACE_WEIGHTS)
    actual = torch.as_tensor(reconstructed.get("G"), dtype=control.dtype)
    point_g = torch.as_tensor(step.get("full_G"), dtype=control.dtype)
    previous_match = True
    if previous_g is not None:
        previous_match = finite._full_G_match(point_g, previous_g).get("passed") is True
    if (step.get("base_control_sha256") != previous_control_sha
            or proposal.get("control_sha256") != control_sha
            or not reconstructed.get("passed")
            or actual.shape != (27,) or not bool(torch.isfinite(actual).all())
            or point_g.shape != (27,) or not bool(torch.isfinite(point_g).all())
            or not previous_match
            or full_check.get("component_match", {}).get("passed") is not True
            or full_check.get("merit_match") is not True or full_check.get("passed") is not True
            ):
        raise ValueError("closed step full-G or previous-prefix closure failed")
    return repeat, actual, control_sha


def validate_endpoint(plan: dict[str, Any], archive: dict[str, Any], raw: dict[str, Any],
        run: dict[str, Any], resource: dict[str, Any], *, raw_bytes: bytes,
        gzip_bytes: bytes, run_bytes: bytes, resource_bytes: bytes) -> dict[str, Any]:
    """Validate all three persisted commits without preparing or evaluating FV."""
    if _sha_bytes(_read_bytes(PARENT_PLAN)) != PARENT_PLAN_SHA:
        raise ValueError("current-gradient parent plan digest mismatch")
    expected_source = {**plan.get("source_files", {}), **plan.get("archive_files", {}),
        PARENT_PLAN.relative_to(ROOT).as_posix(): PARENT_PLAN_SHA}
    if (not plan.get("source_files") or not plan.get("archive_files")
            or archive.get("plan_sha256") != PARENT_PLAN_SHA
            or archive.get("raw_sha256") != BASE_RAW_SHA
            or raw.get("plan_sha256") != PARENT_PLAN_SHA
            or raw.get("source_before") != expected_source
            or raw.get("source_after") != expected_source):
        raise ValueError("archived endpoint parent source pins do not close")

    raw_sha, gzip_sha = _sha_bytes(raw_bytes), _sha_bytes(gzip_bytes)
    if json.loads(raw_bytes) != raw:
        raise ValueError("parsed archived child differs from its raw receipt bytes")
    manifest = archive.get("manifest", {})
    for path, data in ((RAW, raw_bytes), (BASE_RUN, run_bytes), (BASE_RESOURCE, resource_bytes)):
        entry = manifest.get(path.relative_to(ROOT).as_posix())
        if (not isinstance(entry, dict) or entry.get("sha256") != _sha_bytes(data)
                or entry.get("bytes") != len(data)):
            raise ValueError(f"archived receipt manifest mismatch: {path.name}")
    if (raw_sha != BASE_RAW_SHA or archive.get("gzip_sha256") != gzip_sha
            or archive.get("raw_bytes") != len(raw_bytes)
            or archive.get("gzip_bytes") != len(gzip_bytes)
            or gzip.decompress(gzip_bytes) != raw_bytes):
        raise ValueError("archived raw/gzip receipt identity mismatch")

    policy = plan.get("policy", {})
    elapsed = resource.get("elapsed_seconds")
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != BASE_RAW_SHA
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != policy.get("outer_seconds")
            or resource.get("rss_limit_bytes") != policy.get("rss_bytes")
            or resource.get("sampled_peak_rss_bytes", policy.get("rss_bytes")) >= policy.get("rss_bytes", 0)
            or not isinstance(elapsed, (int, float))
            or not 0 < float(elapsed) <= float(policy.get("outer_seconds", 0))):
        raise ValueError("archived run/resource receipt closure failed")

    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=torch.float64)
    steps = raw.get("closed_steps")
    if (raw.get("execution_status") != "completed" or raw.get("numerical_status") != "commit_cap_reached"
            or raw.get("candidate_committed") is not True or raw.get("optimizer_steps_applied") != 3
            or raw.get("current_control_sha256") != BASE_SHA
            or raw.get("last_confirmed_control_sha256") != BASE_SHA
            or tangent._tensor_sha(control) != BASE_SHA
            or raw.get("current_control") != raw.get("last_confirmed_control")
            or raw.get("current_theta") != raw.get("last_confirmed_theta")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or full_g.shape != (27,) or not bool(torch.isfinite(full_g).all())
            or not isinstance(steps, list) or len(steps) != 3
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("runtime_before") != raw.get("runtime_after")):
        raise ValueError("archived final 556 control/source/input/runtime closure failed")

    before, after = raw.get("input_before"), raw.get("input_after")
    stable_input_keys = ("parameters_sha256", "terminal_truth_sha256", "archived_input",
        "original_flow_fractions")
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or not all(before.get(key) == after.get(key) for key in stable_input_keys)
            or after.get("control_sha256") != BASE_SHA):
        raise ValueError("archived terminal fixed-input identity does not close")

    previous_g = None
    previous_control_sha = raw.get("base_control_sha256")
    if not isinstance(previous_control_sha, str):
        raise ValueError("archived endpoint base control identity is malformed")
    final_repeat = None
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError("closed step is malformed")
        if step.get("point_index") != index:
            raise ValueError("closed-step order does not match its point index")
        final_repeat, step_g, previous_control_sha = _close_step(
            step, previous_g, previous_control_sha)
        previous_g = step_g
    if final_repeat is None or previous_g is None:
        raise ValueError("archived endpoint has no closed terminal step")
    if (final_repeat.get("control") != raw.get("current_control")
            or final_repeat.get("theta") != raw.get("current_theta")
            or not finite._full_G_match(previous_g, full_g).get("passed")):
        raise ValueError("archived terminal P2/full-G prefix differs from the confirmed endpoint")
    return {"raw": raw, "control": control, "theta": float(raw["current_theta"]),
        "full_G": full_g, "repeat": final_repeat, "input_after": after,
        "runtime_after": raw["runtime_after"]}


def load_endpoint() -> dict[str, Any]:
    """Load the 556a endpoint from fixed receipts only; this function makes no FV calls."""
    plan_bytes, plan = _read_json(PARENT_PLAN)
    archive_bytes, archive = _read_json(PARENT_ARCHIVE)
    gzip_bytes = _read_bytes(BASE_GZIP)
    raw_bytes = gzip.decompress(gzip_bytes)
    run_bytes, run = _read_json(BASE_RUN)
    resource_bytes, resource = _read_json(BASE_RESOURCE)
    if _sha_bytes(plan_bytes) != PARENT_PLAN_SHA:
        raise ValueError("current-gradient parent plan digest mismatch")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("archived endpoint child is not a JSON object")
    endpoint = validate_endpoint(plan, archive, raw, run, resource, raw_bytes=raw_bytes,
        gzip_bytes=gzip_bytes, run_bytes=run_bytes, resource_bytes=resource_bytes)
    return {**endpoint, "parent_plan": plan, "receipt_files": {
        path.relative_to(ROOT).as_posix(): _sha_bytes(_read_bytes(path)) for path in RECEIPTS}}
