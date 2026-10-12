"""Receipt-only admission of the closed 7839 residual-direction progress run."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

import torch

from examples.weather_scenarios import fv_point_3h_current_gradient_endpoint as arithmetic
from examples.weather_scenarios import fv_point_3h_current_gradient_continuation as math_helpers

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
EVIDENCE = WORK.parent / "subspace_active_qr_20261012"
PLAN = EVIDENCE / "PROGRESS_PLAN.json"
PLAN_SHA = "5410fd4ae986d07a8b9a1d112c106fd79acdc6d228d18153d6adb24f361eb866"
SOURCE_TAR = EVIDENCE / "source.tar.gz"
SOURCE_TAR_SHA = "0166b2a66e55c4a3e182b96f24a79f4d903147dd1d5076f4871b30d1fa281b3a"
RECEIPT_MANIFEST = EVIDENCE / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA = "9c773db8df7bf5b6d4d47c0e23a2e8039b29956bbe475ad3ec57acaf2c331b6f"
RECEIPTS_TAR = EVIDENCE / "receipts.tar.gz"
RECEIPTS_TAR_SHA = "d180241d688c7cd98db7757aa9f03d6560b176d6e5d1773998dff14a08d1a728"
CLOSED_ENDPOINT_BUNDLE = EVIDENCE / "CLOSED_ENDPOINT_BUNDLE.json"
CLOSED_ENDPOINT_BUNDLE_SHA = "8fac8166543193a326b413e3b1a5aedf10e6a301924515cb782d5c5094e77849"
CLOSED_ENDPOINT_TAR = EVIDENCE / "closed_endpoint.tar.gz"
CLOSED_ENDPOINT_TAR_SHA = "2543da031743c0901b3df0cd585250ca0366935a27c0f45509e7466513443fe9"
CLOSED_ENDPOINT_SNAPSHOT_SHA = "0044aee455f4eec8b6b082583c0849f9fa70e0f1ca8a85aa29b641db1410ec54"
PARENT_CLOSED_PLAN_SHA = "466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03"
PARENT_CLOSED_MANIFEST_SHA = "57c812eefa96a3b0edecafc2d394c5ca1e99bd9d39492d4e658e46286d4ff4ca"
PARENT_CLOSED_SOURCE_SHA = "ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed"
PARENT_CLOSED_RECEIPT_SHA = "0ddb163057c4a4d7650c009f77401e604fb7aab0dcf7c2d7e3186656f91cdca0"
PARENT_CLOSED_STEP_SHA = "00b097714e7b6e76c207ebe87c2fa49a96ce822b8c6cb1889ae91c714ca73e92"
PROGRESS_TAR_STEP_SHA = "0d9c090c633cb57cf1c5da5d2b33e9433f53f8397c0db4ece7fba87db831584c"
BASE_SHA = "7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a"
FINAL_SHA = "2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca"
PARENT_PLAN = ROOT / "graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012/EXPANSION_PLAN.json"
PARENT_PLAN_SHA = "466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03"
STEP_MEMBER = "progress1/step.json"
RECEIPTS = (PLAN, SOURCE_TAR, RECEIPT_MANIFEST, RECEIPTS_TAR,
            CLOSED_ENDPOINT_BUNDLE, CLOSED_ENDPOINT_TAR, PARENT_PLAN)

# This search driver and test-only files are the only allowed source drift after
# the archived run. The FV and numerical kernel sources remain byte-pinned.
ALLOWED_SOURCE_DRIFT = {
    "examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py",
    "tests/test_fv_point_3h_netflux_comparison_endpoint.py",
}
SIDE_WEIGHTS = arithmetic.FACE_WEIGHTS


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, digest: str) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned progress input is not an in-repository regular file: {path}")
    data = path.read_bytes()
    if _sha(data) != digest:
        raise ValueError(f"pinned progress input changed: {path.name}")
    return data


def _object(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _tar_files(data: bytes, label: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (not member.isfile() or member.name in files or name.is_absolute()
                        or ".." in name.parts):
                    raise ValueError(f"{label} has an unsafe, duplicate, or non-file member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{label} has an unreadable member")
                files[member.name] = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"{label} is unreadable") from error
    return files


def _check_source_closure(plan: dict[str, Any], raw: dict[str, Any]) -> dict[str, str]:
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or len(sources) != 228:
        raise ValueError("progress source pin count changed")
    if not isinstance(archives, dict) or len(archives) != 296:
        raise ValueError("progress archive pin count changed")
    source_bytes = _tar_files(_read(SOURCE_TAR, SOURCE_TAR_SHA), "progress source bundle")
    if set(source_bytes) != set(sources):
        raise ValueError("progress source member set changed")
    for name, digest in sources.items():
        if _sha(source_bytes[name]) != digest:
            raise ValueError(f"progress source member hash mismatch: {name}")

    drift: dict[str, str] = {}
    for name, digest in sources.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"current source is missing or unsafe: {name}")
        current = _sha(path.read_bytes())
        if current != digest:
            if name not in ALLOWED_SOURCE_DRIFT and not name.startswith("tests/"):
                raise ValueError(f"unapproved source drift, including possible kernel change: {name}")
            drift[name] = current
    for name, digest in archives.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or _sha(path.read_bytes()) != digest:
            raise ValueError(f"current archived input differs from its pin: {name}")

    plan_path = PLAN.relative_to(ROOT).as_posix()
    expected = {**sources, **archives, plan_path: PLAN_SHA}
    if raw.get("source_before") != expected or raw.get("source_after") != expected:
        raise ValueError("progress execution source before/after pins do not close")
    if raw.get("source_unchanged") is not True:
        raise ValueError("progress execution reports source mutation")
    return drift


def _validate_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    if (bundle.get("bundle_sha256") != _sha(_read(CLOSED_ENDPOINT_TAR, CLOSED_ENDPOINT_TAR_SHA))
            or bundle.get("files", {}).get("CLOSED_ENDPOINT_SNAPSHOT.json", {}).get("sha256")
            != CLOSED_ENDPOINT_SNAPSHOT_SHA):
        raise ValueError("closed 7839 endpoint bundle pins changed")
    members = _tar_files(_read(CLOSED_ENDPOINT_TAR, CLOSED_ENDPOINT_TAR_SHA),
                         "closed endpoint bundle")
    if set(members) != {"CLOSED_ENDPOINT_SNAPSHOT.json"}:
        raise ValueError("closed endpoint bundle member set changed")
    snapshot_bytes = members["CLOSED_ENDPOINT_SNAPSHOT.json"]
    entry = bundle["files"]["CLOSED_ENDPOINT_SNAPSHOT.json"]
    if _sha(snapshot_bytes) != CLOSED_ENDPOINT_SNAPSHOT_SHA or len(snapshot_bytes) != entry.get("bytes"):
        raise ValueError("closed endpoint snapshot hash or length changed")
    loose = EVIDENCE / "CLOSED_ENDPOINT_SNAPSHOT.json"
    if _read(loose, CLOSED_ENDPOINT_SNAPSHOT_SHA) != snapshot_bytes:
        raise ValueError("loose closed endpoint snapshot differs from its packed bytes")
    snapshot = _object(snapshot_bytes, "closed endpoint snapshot")
    base = snapshot.get("base", {})
    admission = snapshot.get("admission", {})
    if (snapshot.get("status") != "offline_cached_current_model_audit_complete"
            or admission.get("plan_sha256") != PARENT_CLOSED_PLAN_SHA
            or admission.get("manifest_sha256") != PARENT_CLOSED_MANIFEST_SHA
            or admission.get("source_bundle_sha256") != PARENT_CLOSED_SOURCE_SHA
            or admission.get("receipt_bundle_sha256") != PARENT_CLOSED_RECEIPT_SHA
            or admission.get("step_sha256") != PARENT_CLOSED_STEP_SHA
            or admission.get("source_member_count") != 222
            or admission.get("archive_pin_count") != 291
            or admission.get("receipt_member_count") != 5
            or base.get("control_sha256") != BASE_SHA):
        raise ValueError("closed endpoint is not the initial 7839 control")
    control = torch.as_tensor(base.get("control"), dtype=torch.float64)
    full_g = torch.as_tensor(base.get("full_G"), dtype=torch.float64)
    repeat = base.get("repeat", {})
    if (control.shape != (26,) or not bool(torch.isfinite(control).all())
            or arithmetic.tangent._tensor_sha(control) != BASE_SHA
            or full_g.shape != (27,) or not bool(torch.isfinite(full_g).all())
            or repeat.get("control") != base.get("control")
            or repeat.get("theta") != base.get("theta")
            or repeat.get("objective") != base.get("J")
            or repeat.get("F_squared") != base.get("R")):
        raise ValueError("closed endpoint base control, G, merit, or repeat changed")
    return snapshot


def _close_progress(raw: dict[str, Any], snapshot: dict[str, Any]) -> tuple[
        torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    steps = raw.get("closed_steps")
    if not isinstance(steps, list) or len(steps) != 9:
        raise ValueError("progress receipt must contain nine closed steps")
    if not torch.equal(SIDE_WEIGHTS, arithmetic.FACE_WEIGHTS):
        raise ValueError("full-G closure side weights differ from the pinned numerical helper")
    base = snapshot["base"]
    previous_g = torch.as_tensor(base["full_G"], dtype=torch.float64)
    previous_sha = BASE_SHA
    final_repeat: dict[str, Any] = {}
    final_g = previous_g
    for step in steps:
        final_repeat, final_g, previous_sha = arithmetic._close_step(step, previous_g, previous_sha)
        previous_g = final_g
    control = torch.as_tensor(raw.get("current_control"), dtype=torch.float64)
    full_g = torch.as_tensor(raw.get("current_full_G"), dtype=torch.float64)
    if (previous_sha != FINAL_SHA or arithmetic.tangent._tensor_sha(control) != FINAL_SHA
            or not torch.equal(final_g, full_g) or final_repeat.get("control") != raw.get("current_control")
            or final_repeat.get("theta") != raw.get("current_theta")
            or raw.get("current_control") != raw.get("last_confirmed_control")
            or raw.get("current_full_G") != raw.get("last_confirmed_full_G")
            or raw.get("current_theta") != raw.get("last_confirmed_theta")):
        raise ValueError("nine-step final control, theta, or full-G does not close")
    return control, full_g, float(final_repeat["theta"]), final_repeat


def load_endpoint() -> dict[str, Any]:
    """Admit the cached progress receipt and close its nine-step math lineage."""
    plan = _object(_read(PLAN, PLAN_SHA), PLAN.name)
    if (plan.get("experiment_kind") != "cached7839_original_JR_bounded_continuation"
            or plan.get("numerical_version") != "minmod_net_flux_fp64_v1"
            or plan.get("base_control_sha256") != BASE_SHA):
        raise ValueError("progress plan identity changed")
    manifest = _object(_read(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA), RECEIPT_MANIFEST.name)
    receipt_files = manifest.get("files")
    if (not isinstance(receipt_files, dict) or len(receipt_files) != 13
            or manifest.get("bundle_sha256") != RECEIPTS_TAR_SHA
            or manifest.get("source_bundle_sha256") != SOURCE_TAR_SHA):
        raise ValueError("progress receipt manifest or bundle identity changed")
    members = _tar_files(_read(RECEIPTS_TAR, RECEIPTS_TAR_SHA), "progress receipts")
    if set(members) != set(receipt_files):
        raise ValueError("progress receipt member set changed")
    for name, entry in receipt_files.items():
        if _sha(members[name]) != entry.get("sha256") or len(members[name]) != entry.get("bytes"):
            raise ValueError(f"progress receipt member hash or length mismatch: {name}")
    raw_bytes = members[STEP_MEMBER]
    if _sha(raw_bytes) != PROGRESS_TAR_STEP_SHA:
        raise ValueError("progress raw step digest changed")
    raw = _object(raw_bytes, STEP_MEMBER)
    run = _object(members["progress1/step.run.json"], "step.run.json")
    resource = _object(members["progress1/step.resource.json"], "step.resource.json")
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != PROGRESS_TAR_STEP_SHA
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or raw.get("plan_sha256") != PLAN_SHA or raw.get("execution_status") != "completed"
            or raw.get("candidate_evaluations_started") != 9
            or raw.get("candidate_evaluations_completed") != 9
            or raw.get("optimizer_steps_applied") != 9 or raw.get("hvp_started") != 17
            or raw.get("hvp_completed") != 16 or raw.get("score_claim") is not False
            or raw.get("root_claim") is not False or raw.get("minimum_claim") is not False
            or raw.get("response_claim") is not False or raw.get("terminal_E_recomputed") is not False
            or raw.get("postcommit_readiness_performed") is not False):
        raise ValueError("progress execution status or claim accounting changed")

    drift = _check_source_closure(plan, raw)
    snapshot = _validate_snapshot(_object(_read(CLOSED_ENDPOINT_BUNDLE,
        CLOSED_ENDPOINT_BUNDLE_SHA), CLOSED_ENDPOINT_BUNDLE.name))
    control, full_g, theta, repeat = _close_progress(raw, snapshot)
    input_before, input_after = raw.get("input_before", {}), raw.get("input_after", {})
    if (raw.get("base_endpoint_admitted") is not True or raw.get("base_control_sha256") != BASE_SHA
            or input_before != snapshot["base"].get("context")
            or input_after.get("control_sha256") != FINAL_SHA
            or input_before.get("archived_input") != input_after.get("archived_input")
            or input_before.get("parameters_sha256") != input_after.get("parameters_sha256")
            or input_before.get("terminal_truth_sha256") != input_after.get("terminal_truth_sha256")
            or input_before.get("original_flow_fractions") != input_after.get("original_flow_fractions")
            or input_before.get("changed_control_indices") != input_after.get("changed_control_indices")
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_before") != raw.get("runtime_after")
            or raw.get("runtime_unchanged") is not True):
        raise ValueError("progress fixed input/runtime lineage changed")

    parent_plan = _object(_read(PARENT_PLAN, PARENT_PLAN_SHA), PARENT_PLAN.name)
    outer_seconds = float(plan["policy"]["outer_seconds"])
    internal_seconds = float(plan["policy"]["internal_seconds"])
    charged = float(raw["prior_preparation_seconds"])
    observed = float(resource["elapsed_seconds"])
    total = charged + observed
    if (resource.get("wall_limit_seconds") != outer_seconds - charged
            or observed > float(resource["wall_limit_seconds"]) or total > outer_seconds
            or resource.get("rss_limit_bytes") != plan["policy"]["rss_bytes"]
            or resource.get("sampled_peak_rss_bytes", resource["rss_limit_bytes"])
            >= resource["rss_limit_bytes"] or resource.get("received_sigterm") is not False):
        raise ValueError("progress receipt exceeded its outer resource budget")
    budget = {"internal_seconds": internal_seconds, "charged_preparation_seconds": charged,
        "guarded_seconds": observed, "total_seconds": total,
        "internal_overrun_seconds": max(0.0, total - internal_seconds),
        "outer_seconds": outer_seconds, "outer_budget_passed": total <= outer_seconds,
        "rss_limit_bytes": resource.get("rss_limit_bytes"),
        "sampled_peak_rss_bytes": resource.get("sampled_peak_rss_bytes")}
    pins = {"plan_sha256": PLAN_SHA, "source_bundle_sha256": SOURCE_TAR_SHA,
        "receipt_bundle_sha256": RECEIPTS_TAR_SHA, "receipt_manifest_sha256": RECEIPT_MANIFEST_SHA,
        "closed_endpoint_bundle_sha256": CLOSED_ENDPOINT_BUNDLE_SHA,
        "closed_endpoint_snapshot_sha256": CLOSED_ENDPOINT_SNAPSHOT_SHA,
        "raw_step_sha256": PROGRESS_TAR_STEP_SHA, "source_member_count": 228,
        "archive_pin_count": 296, "receipt_member_count": 13,
        "current_source_drift": drift}
    return {"control": control, "G": full_g, "full_G": full_g, "theta": theta,
        "repeat": repeat, "input_after": raw["input_after"], "runtime_after": raw["runtime_after"],
        "parent_plan": parent_plan, "pins": pins, "budget": budget, "progress_plan": plan,
        "raw": raw, "resource": resource}
