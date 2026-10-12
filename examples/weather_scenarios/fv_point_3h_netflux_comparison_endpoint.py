"""Receipt-only admission of the d19e net-flux direction-comparison endpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as current_endpoint
from examples.weather_scenarios import fv_point_3h_netflux_direction_comparison as comparison
from examples.weather_scenarios import fv_point_3h_netflux_endpoint as original_endpoint

ROOT = comparison.ROOT
WORK = comparison.WORK
PLAN = comparison.PLAN
PLAN_SHA = "36686ccf05c55ab9dff1d6c0c4658e14049a108eba402b76c86fc0b9ac904b35"
SOURCE_TAR = WORK / "source.tar.gz"
SOURCE_TAR_SHA = "556fec30ca5b0a06c0b00601c8652b616b02705a09d7ac82fcea3203fd0c46bc"
RECEIPT_MANIFEST = WORK / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA = "2d1467b35d1bd99bacbf4f0cf16a9596c57e31ae963f31b29481d2ccffa233a7"
RECEIPTS_TAR = WORK / "receipts.tar.gz"
RECEIPTS_TAR_SHA = "55f220c1f5da91a2a76175984e419049470acb324aef8b34b1eacd36e77a4040"
BASE_SHA = "d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e"
BASE_RAW_SHA = "1c79690de1d8e42ff4accf5e6b3bd62dac5c55c0607d3cc8b771fb6b6a8b4703"
PARENT_CONTROL_SHA = comparison.BASE_SHA
NUMERICAL_VERSION = comparison.VERSION
SELF = "examples/weather_scenarios/fv_point_3h_netflux_comparison_endpoint.py"
TEST = "tests/test_fv_point_3h_netflux_comparison_endpoint.py"
RECEIPTS = (PLAN, SOURCE_TAR, RECEIPT_MANIFEST, RECEIPTS_TAR, *comparison.endpoint.RECEIPTS)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"comparison receipt is not an object: {label}")
    return value


def _read(path: Path, digest: str | None = None) -> bytes:
    return comparison.endpoint._read(path, digest)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(plan: dict[str, Any], raw: dict[str, Any], run: dict[str, Any],
              resource: dict[str, Any], members: dict[str, bytes],
              manifest: dict[str, Any], base: dict[str, Any]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    policy = plan["policy"]
    expected_source = {**plan["source_files"], **plan["archive_files"],
        PLAN.relative_to(ROOT).as_posix(): PLAN_SHA}
    before, after = raw.get("input_before"), raw.get("input_after")
    arms = raw.get("arms")
    steps = raw.get("closed_steps")
    _require(
        raw.get("plan_sha256") == PLAN_SHA
        and raw.get("numerical_version") == NUMERICAL_VERSION
        and raw.get("execution_status") == "completed"
        and raw.get("numerical_status") == "same_point_comparison_one_candidate_committed"
        and raw.get("comparison_complete") is True
        and raw.get("candidate_committed") is True
        and raw.get("optimizer_steps_applied") == 1
        and raw.get("current_control_sha256") == BASE_SHA
        and raw.get("last_confirmed_control_sha256") == BASE_SHA
        and raw.get("current_control") == raw.get("last_confirmed_control")
        and raw.get("current_theta") == raw.get("last_confirmed_theta")
        and raw.get("current_full_G") == raw.get("last_confirmed_full_G")
        and raw.get("candidate_evaluations_started") == 11
        and raw.get("candidate_evaluations_completed") == 11
        and raw.get("hvp_started") == 4 and raw.get("hvp_completed") == 4
        and raw.get("GN_rows_started") == 24 and raw.get("GN_rows_completed") == 24
        and raw.get("GN_solves_completed") == 1
        and raw.get("root_claim") is False and raw.get("minimum_claim") is False
        and raw.get("response_claim") is False and raw.get("score_claim") is False
        and raw.get("terminal_E_recomputed") is False
        and raw.get("postcommit_readiness_performed") is False
        and raw.get("source_unchanged") is True
        and raw.get("fixed_input_unchanged") is True
        and raw.get("runtime_unchanged") is True
        and raw.get("source_before") == expected_source
        and raw.get("source_after") == expected_source
        and raw.get("runtime_before") == raw.get("runtime_after")
        and isinstance(before, dict) and isinstance(after, dict)
        and before.get("control_sha256") == PARENT_CONTROL_SHA
        and after.get("control_sha256") == BASE_SHA
        and all(before.get(key) == after.get(key) for key in (
            "archived_input", "parameters_sha256", "terminal_truth_sha256", "original_flow_fractions"))
        and run.get("execution_status") == "completed"
        and run.get("child_sha256") == BASE_RAW_SHA
        and run.get("resource") == resource
        and resource.get("exit_code") == 0
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("received_sigterm") is False
        and resource.get("wall_limit_seconds") == policy["outer_seconds"]
        and resource.get("rss_limit_bytes") == policy["rss_bytes"]
        and resource.get("sampled_peak_rss_bytes", policy["rss_bytes"]) < policy["rss_bytes"]
        and isinstance(resource.get("elapsed_seconds"), (int, float))
        and 0 < float(resource["elapsed_seconds"]) <= policy["outer_seconds"],
        "comparison execution/resource/source/input closure failed")
    if not isinstance(arms, list) or len(arms) != 2 or any(not isinstance(arm, dict) for arm in arms):
        raise ValueError("both independent comparison arms are not closed")
    if ([arm.get("name") for arm in arms] != list(comparison.ARM_ORDER)
            or any(arm.get("status") != "independent_P2_fullG_eligible" for arm in arms)):
        raise ValueError("both independent comparison arms are not closed")
    winner = comparison.choose_repeated_arm(arms, torch.float64)
    if winner is None or raw.get("selected_direction") != winner["name"]:
        raise ValueError("selected direction does not match the repeated-arm rule")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise ValueError("comparison must contain one committed step")

    closed = steps[0]
    _require(closed.get("point_index") == 0
        and closed.get("base_control_sha256") == PARENT_CONTROL_SHA
        and closed.get("selected_candidate") == winner.get("candidate")
        and closed.get("proposal") == winner.get("proposal")
        and closed.get("final_repeat") == winner.get("repeat")
        and closed.get("full_G_check") == winner.get("full_G_check"),
        "closed step is not linked to the selected eligible arm")
    # Reuse the established arithmetic closure for proposal, repeat, full-G, and control lineage.
    repeat, step_g, control_sha = current_endpoint._close_step(
        closed, base["full_G"], PARENT_CONTROL_SHA)
    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=torch.float64)
    _require(control.shape == (26,) and bool(torch.isfinite(control).all())
        and comparison.tangent._tensor_sha(control) == BASE_SHA
        and control_sha == BASE_SHA
        and repeat.get("control") == raw.get("current_control")
        and repeat.get("theta") == raw.get("current_theta")
        and full_g.shape == (27,) and bool(torch.isfinite(full_g).all())
        and torch.equal(step_g, full_g)
        and closed.get("full_G_check", {}).get("G") == raw.get("current_full_G"),
        "d19e endpoint control/theta/full-G differs from its closed step")

    candidate_names = {f"attempt1/{arm['name']}/candidate_{i:02d}.json"
        for arm in arms for i in range(1, len(arm.get("trials", [])) + 1)}
    _require(candidate_names <= set(manifest["files"]), "candidate receipt set is incomplete")
    for arm in arms:
        candidate = arm["candidate"]
        receipt_path = candidate.get("candidate_receipt")
        _require(isinstance(receipt_path, str) and receipt_path.startswith(
            "graphify-out/fv-root-cause-20260919/netflux_direction_comparison_20261012/"),
            "candidate receipt path is outside the pinned comparison")
        relative = Path(receipt_path).relative_to(WORK.relative_to(ROOT)).as_posix()
        payload = _json(members[relative], relative)
        _require(candidate.get("candidate_receipt_sha256") == _sha_bytes(members[relative])
            and payload.get("G") == candidate.get("G")
            and payload.get("alpha") == candidate.get("alpha")
            and payload.get("objective") == candidate.get("objective")
            and payload.get("F_squared") == candidate.get("F_squared"),
            "selected arm candidate payload does not match the receipt bundle")
    return control, full_g, repeat


def load_endpoint() -> dict[str, Any]:
    """Load d19e from pinned receipts and arithmetic checks; never call the FV model."""
    plan = comparison.load_plan(PLAN, PLAN_SHA)
    source_bytes = _read(SOURCE_TAR, SOURCE_TAR_SHA)
    source_members = original_endpoint._bundle_members(source_bytes, set(plan["source_files"]))
    _require(all(_sha_bytes(source_members[name]) == digest
        for name, digest in plan["source_files"].items()), "comparison source-bundle hashes differ")
    manifest_bytes = _read(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA)
    manifest = _json(manifest_bytes, RECEIPT_MANIFEST.name)
    receipt_files = manifest.get("files")
    if not isinstance(receipt_files, dict) or len(receipt_files) != 16:
        raise ValueError("comparison receipt manifest changed")
    _require(manifest.get("bundle_sha256") == RECEIPTS_TAR_SHA
        and manifest.get("source_bundle_sha256") == SOURCE_TAR_SHA,
        "comparison receipt manifest changed")
    receipt_bytes = _read(RECEIPTS_TAR, RECEIPTS_TAR_SHA)
    members = original_endpoint._bundle_members(receipt_bytes, set(receipt_files))
    for name, entry in receipt_files.items():
        _require(_sha_bytes(members[name]) == entry.get("sha256")
            and len(members[name]) == entry.get("bytes"), f"comparison receipt member mismatch: {name}")
    raw_bytes = members["attempt1/step.json"]
    _require(_sha_bytes(raw_bytes) == BASE_RAW_SHA, "comparison raw-child hash mismatch")
    raw = _json(raw_bytes, "attempt1/step.json")
    run = _json(members["attempt1/step.run.json"], "attempt1/step.run.json")
    resource = _json(members["attempt1/step.resource.json"], "attempt1/step.resource.json")
    _require(run.get("resource") == resource, "comparison run/resource linkage mismatch")
    base = comparison.endpoint.load_endpoint()
    control, full_g, repeat = _validate(plan, raw, run, resource, members, manifest, base)
    return {"raw": raw, "control": control, "full_G": full_g, "repeat": repeat,
        "theta": float(repeat["theta"]), "input_after": raw["input_after"],
        "runtime_after": raw["runtime_after"], "parent_plan": plan,
        "receipt_files": {path.relative_to(ROOT).as_posix(): _sha_bytes(_read(
            path, SOURCE_TAR_SHA if path == SOURCE_TAR else RECEIPTS_TAR_SHA if path == RECEIPTS_TAR
            else PLAN_SHA if path == PLAN else RECEIPT_MANIFEST_SHA if path == RECEIPT_MANIFEST else None))
            for path in RECEIPTS}}
