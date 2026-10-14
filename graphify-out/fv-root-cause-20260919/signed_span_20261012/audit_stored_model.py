"""Audit the frozen PR304 residual model; this module never evaluates FV."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace

WORK = Path(__file__).resolve().parent
RUN = ROOT / "graphify-out/fv-root-cause-20260919/residual_direction_20261012"
ACTIVE = ROOT / "graphify-out/fv-root-cause-20260919/subspace_active_qr_20261012"
PLAN = RUN / "PLAN.json"
PLAN_SHA = "cd9db66869652c8807264dcdb9172f94e4c65118fc3982465e516e706502fa64"
SOURCE_TAR = RUN / "source.tar.gz"
SOURCE_TAR_SHA = "da936a87c96bcb5970afbed650ba00760949763652f77febfebdd9b44d6ac2b8"
RECEIPT_MANIFEST = RUN / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA = "c7f55b3b44b83c895f8340c22181546cb307333f56f79ea0f58df380bdc29263"
RECEIPTS_TAR = RUN / "receipts.tar.gz"
RECEIPTS_TAR_SHA = "5fae12a5f5a8a39c1556a6e62180c399d654f837882026f650bd2c2cb7b1aa11"
ACTIVE_RECEIPT_MANIFEST_SHA = "9c773db8df7bf5b6d4d47c0e23a2e8039b29956bbe475ad3ec57acaf2c331b6f"
ACTIVE_RECEIPTS_TAR_SHA = "d180241d688c7cd98db7757aa9f03d6560b176d6e5d1773998dff14a08d1a728"
RADIUS = 3.751435583391765e-5
BASE_SHA = "2a4dba130d28273a63e97d0de39a11f64fdee09b0ad754559bc8902ba6cddbca"
STEP_SHA = "d8d36a361834439968a188ba669d5ea6352a0cc0f0e39288ec4a290b016dd014"
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SIGNED_PLAN = WORK / "PLAN.json"
SIGNED_PLAN_SHA_FILE = WORK / "PLAN_SHA256.txt"
APPROVED_CURRENT_DRIFT = {
    "examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py",
    "examples/weather_scenarios/fv_point_3h_residual_direction.py",
    "tests/test_fv_point_3h_residual_direction.py",
    "tests/test_fv_point_3h_subspace_model.py",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pinned(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"unsafe or out-of-repository input: {path}")
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"pinned input hash mismatch: {path}")
    return data


def as_object(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def tar_files(data: bytes, label: str) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name)
            if (not member.isfile() or member.name in result or parts.is_absolute()
                    or ".." in parts.parts):
                raise ValueError(f"{label} has unsafe, duplicate, or non-file member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"{label} member is unreadable: {member.name}")
            result[member.name] = stream.read()
    return result


def audit_bundle(plan: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    sources = plan.get("source_files")
    archives = plan.get("archive_files")
    if not isinstance(sources, dict) or len(sources) != 234:
        raise ValueError("unexpected source pin count")
    if not isinstance(archives, dict) or len(archives) != 302:
        raise ValueError("unexpected inherited archive pin count")

    source_members = tar_files(read_pinned(SOURCE_TAR, SOURCE_TAR_SHA), "PR304 source tar")
    if set(source_members) != set(sources):
        raise ValueError("source tar member set differs from frozen plan")
    current_drift: dict[str, dict[str, str]] = {}
    for name, expected in sources.items():
        if sha(source_members[name]) != expected:
            raise ValueError(f"source member digest mismatch: {name}")
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"current source is missing or unsafe: {name}")
        current = sha(path.read_bytes())
        if current != expected:
            if name not in APPROVED_CURRENT_DRIFT:
                raise ValueError(f"unapproved live source drift: {name}")
            current_drift[name] = {"frozen": expected, "current": current}
    for name, expected in archives.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != expected:
            raise ValueError(f"current inherited archive differs from frozen pin: {name}")

    manifest = as_object(read_pinned(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA),
                         RECEIPT_MANIFEST.name)
    entries = manifest.get("files")
    if (not isinstance(entries, dict) or len(entries) != 10
            or manifest.get("bundle_sha256") != RECEIPTS_TAR_SHA
            or manifest.get("source_bundle_sha256") != SOURCE_TAR_SHA):
        raise ValueError("PR304 receipt manifest identity/count mismatch")
    receipt_members = tar_files(read_pinned(RECEIPTS_TAR, RECEIPTS_TAR_SHA),
                                "PR304 receipt tar")
    if set(receipt_members) != set(entries):
        raise ValueError("PR304 receipt member set differs from manifest")
    for name, entry in entries.items():
        data = receipt_members[name]
        if sha(data) != entry.get("sha256") or len(data) != entry.get("bytes"):
            raise ValueError(f"PR304 receipt member digest/length mismatch: {name}")
        if (RUN / name).read_bytes() != data:
            raise ValueError(f"PR304 loose receipt differs from packed bytes: {name}")
    signed_transition: dict[str, Any] = {}
    if SIGNED_PLAN.is_file() and SIGNED_PLAN_SHA_FILE.is_file():
        signed_bytes = SIGNED_PLAN.read_bytes()
        signed_digest = sha(signed_bytes)
        if signed_digest != SIGNED_PLAN_SHA_FILE.read_text().strip():
            raise ValueError("signed-span plan digest disagrees with its freeze seal")
        signed_plan = as_object(signed_bytes, SIGNED_PLAN.name)
        signed_transition = signed_plan.get("source_transition", {})
        for name, hashes in current_drift.items():
            entry = signed_transition.get(name)
            if (not isinstance(entry, dict) or entry.get("before") != hashes["frozen"]
                    or entry.get("after") != hashes["current"]
                    or signed_plan.get("source_files", {}).get(name) != hashes["current"]):
                raise ValueError(f"signed-span plan does not seal approved source transition: {name}")
    return receipt_members, {"plan_sha256": PLAN_SHA, "source_tar_sha256": SOURCE_TAR_SHA,
        "receipt_manifest_sha256": RECEIPT_MANIFEST_SHA,
        "receipt_tar_sha256": RECEIPTS_TAR_SHA,
        "source_members": len(sources), "archive_pins": len(archives),
        "receipt_members": len(entries), "all_frozen_bundle_and_archive_pins_match": True,
        "approved_current_source_drift": current_drift,
        "signed_plan_transition_checked": bool(signed_transition)}


def load_parent_progress() -> tuple[dict[str, Any], dict[str, Any]]:
    parent_manifest_path = ACTIVE / "RECEIPT_MANIFEST.json"
    parent_receipt_path = ACTIVE / "receipts.tar.gz"
    manifest = as_object(read_pinned(parent_manifest_path, ACTIVE_RECEIPT_MANIFEST_SHA),
                         parent_manifest_path.name)
    members = tar_files(read_pinned(parent_receipt_path, ACTIVE_RECEIPTS_TAR_SHA),
                        "PR303 progress receipts")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or set(members) != set(entries):
        raise ValueError("PR303 base receipt bundle differs from its manifest")
    for name, entry in entries.items():
        if sha(members[name]) != entry.get("sha256") or len(members[name]) != entry.get("bytes"):
            raise ValueError(f"PR303 base member digest/length mismatch: {name}")
    step_bytes = members["progress1/step.json"]
    step = as_object(step_bytes, "PR303 progress step")
    if (sha(step_bytes) != "0d9c090c633cb57cf1c5da5d2b33e9433f53f8397c0db4ece7fba87db831584c"
            or step.get("current_control_sha256") != BASE_SHA
            or step.get("current_control") != step.get("last_confirmed_control")
            or step.get("current_full_G") != step.get("last_confirmed_full_G")
            or step.get("fixed_input_unchanged") is not True
            or step.get("runtime_unchanged") is not True
            or not step.get("closed_steps")):
        raise ValueError("PR303 closed 2a4d endpoint lineage is not pinned")
    return step, {"parent_receipt_tar_sha256": ACTIVE_RECEIPTS_TAR_SHA,
        "parent_receipt_manifest_sha256": ACTIVE_RECEIPT_MANIFEST_SHA,
        "parent_step_sha256": sha(step_bytes), "parent_closed_steps": len(step["closed_steps"])}


def load_stored_model() -> dict[str, Any]:
    """Return pinned PR304 vectors and the closed 2a4d side gradients."""
    plan = as_object(read_pinned(PLAN, PLAN_SHA), PLAN.name)
    receipt_members, pins = audit_bundle(plan)
    raw = as_object(receipt_members["attempt1/step.json"], "PR304 raw step")
    progress, parent_pins = load_parent_progress()
    last = progress["closed_steps"][-1]
    repeat = last["final_repeat"]
    models = {model["name"]: model for model in raw["fresh_models"]}
    G = torch.as_tensor(progress["current_full_G"], dtype=torch.float64)
    D = torch.stack([torch.as_tensor(models[name]["direction"], dtype=torch.float64)
                     for name in ("baseline_tangent", "residual_merit")], dim=1)
    V = torch.stack([torch.as_tensor(models[name]["residual_direction"], dtype=torch.float64)
                     for name in ("baseline_tangent", "residual_merit")], dim=1)
    gm = torch.as_tensor(repeat["side_gradients"]["-1"], dtype=torch.float64)
    gp = torch.as_tensor(repeat["side_gradients"]["1"], dtype=torch.float64)
    if (progress.get("current_control_sha256") != BASE_SHA
            or repeat.get("control") != progress.get("current_control")
            or raw.get("base_control_sha256") != BASE_SHA
            or not torch.equal(G, torch.as_tensor(raw["closed_steps"][0]["full_G"], dtype=G.dtype))
            or gm.shape != (26,) or gp.shape != (26,) or G.shape != (27,)
            or D.shape != (26, 2) or V.shape != (27, 2)):
        raise ValueError("stored PR303/PR304 same-point vectors do not close")
    return {"plan": plan, "raw": raw, "progress": progress, "base_repeat": repeat,
        "control": torch.as_tensor(progress["current_control"], dtype=torch.float64),
        "theta": float(progress["current_theta"]), "G": G, "D": D, "V": V,
        "parent_J": float(progress["current_point"]["J"]),
        "parent_R": float(progress["current_point"]["R"]),
        "gm": gm, "gp": gp, "fresh_models": models,
        "pins": {**pins, **parent_pins}}


def kkt_summary(G: torch.Tensor, D: torch.Tensor, V: torch.Tensor,
                coefficients: torch.Tensor, signs: torch.Tensor,
                radius: float, *, constrained: bool = True) -> dict[str, Any]:
    residual = G + V @ coefficients
    coefficient_gradient = V.T @ residual
    direction = D @ coefficients
    control_norm = torch.linalg.vector_norm(direction)
    sign_basis = D * signs.reshape(1, 2)
    beta = coefficients * signs
    beta_gradient = (V * signs.reshape(1, 2)).T @ residual
    qbeta = sign_basis.T @ (sign_basis @ beta)
    radius_budget = 128.0 * torch.finfo(G.dtype).eps * max(
        abs(float(control_norm)), radius, torch.finfo(G.dtype).tiny)
    radius_active = bool(abs(float(control_norm) - radius) <= radius_budget)
    active = beta > 0.0
    multiplier = 0.0
    if radius_active and bool(active.any()):
        active_q = qbeta[active]
        denominator = torch.dot(active_q, active_q)
        if bool(denominator > torch.finfo(G.dtype).tiny):
            multiplier = float(-torch.dot(beta_gradient[active], active_q) / denominator)
    lagrangian_gradient = beta_gradient + multiplier * qbeta
    positive_stationarity = (float(lagrangian_gradient[active].abs().max())
                             if bool(active.any()) else 0.0)
    lower_bound_violation = (
        float((-lagrangian_gradient[~active]).clamp_min(0).max())
        if constrained and bool((~active).any()) else 0.0)
    return {"coefficients_original": coefficients.tolist(),
        "coefficients_in_sign_quadrant": beta.tolist(),
        "sign_quadrant": signs.tolist(),
        "coefficient_gradient_Vt_residual": coefficient_gradient.tolist(),
        "quadrant_gradient": beta_gradient.tolist(),
        "radius_multiplier_estimate": multiplier,
        "radius_multiplier_negative_violation": max(0.0, -multiplier) if radius_active else 0.0,
        "radius_active": radius_active,
        "radius_roundoff_budget": radius_budget,
        "positive_stationarity_linf": positive_stationarity,
        "lower_bound_kkt_violation": lower_bound_violation,
        "control_direction": direction.tolist(),
        "control_norm": float(control_norm),
        "radius": radius,
        "radius_slack": radius - float(control_norm),
        "predicted_R": float(torch.dot(residual, residual)),
        "residual": residual.tolist()}


def main() -> None:
    plan_bytes = read_pinned(PLAN, PLAN_SHA)
    plan = as_object(plan_bytes, PLAN.name)
    if (plan.get("experiment_kind") != "fresh2a4d_gradient_residual_direction_comparison"
            or plan.get("base_control_sha256") != BASE_SHA
            or plan.get("policy", {}).get("radius") != RADIUS):
        raise ValueError("PR304 problem/radius identity mismatch")
    receipt_members, pins = audit_bundle(plan)
    raw = as_object(receipt_members["attempt1/step.json"], "PR304 raw step")
    if raw.get("plan_sha256") != PLAN_SHA or raw.get("execution_status") != "completed":
        raise ValueError("PR304 run status/plan link mismatch")
    parent, parent_pins = load_parent_progress()

    parent_last = parent["closed_steps"][-1]
    parent_repeat = parent_last["final_repeat"]
    gm = torch.as_tensor(parent_repeat["side_gradients"]["-1"], dtype=torch.float64)
    gp = torch.as_tensor(parent_repeat["side_gradients"]["1"], dtype=torch.float64)
    G = torch.as_tensor(parent["current_full_G"], dtype=torch.float64)
    trial_base_G = torch.as_tensor(raw["closed_steps"][0]["full_G"], dtype=torch.float64)
    if (parent_repeat["control"] != parent["current_control"]
            or parent.get("current_control_sha256") != BASE_SHA
            or raw.get("base_control_sha256") != BASE_SHA
            or raw["closed_steps"][0].get("base_control_sha256") != BASE_SHA
            or not torch.equal(G, trial_base_G)
            or gm.shape != (26,) or gp.shape != (26,) or G.shape != (27,)):
        raise ValueError("stored PR303 gradients/G are not exactly the PR304 2a4d base")

    models = {model["name"]: model for model in raw["fresh_models"]}
    names = ("baseline_tangent", "residual_merit")
    if set(models) != set(names):
        raise ValueError("PR304 stored model columns are incomplete")
    D = torch.stack([torch.as_tensor(models[name]["direction"], dtype=torch.float64)
                     for name in names], dim=1)
    V = torch.stack([torch.as_tensor(models[name]["residual_direction"], dtype=torch.float64)
                     for name in names], dim=1)
    if D.shape != (26, 2) or V.shape != (27, 2):
        raise ValueError("PR304 D/V stored dimensions changed")

    quadrants: list[dict[str, Any]] = []
    for signs_tuple in SIGNS:
        signs = torch.tensor(signs_tuple, dtype=G.dtype)
        Dq, Vq = D * signs.reshape(1, 2), V * signs.reshape(1, 2)
        result = solve_two_direction_subspace(G, Dq, Vq, RADIUS)
        coefficients = result.coefficients * signs
        summary = kkt_summary(G, D, V, coefficients, signs, RADIUS)
        direct_direction = D @ coefficients
        direct_residual_direction = V @ coefficients
        predicted = torch.linalg.vector_norm(G + direct_residual_direction).square()
        if (not torch.allclose(result.direction, direct_direction, atol=1e-14, rtol=1e-12)
                or not torch.allclose(result.residual_direction, direct_residual_direction,
                    atol=1e-14, rtol=1e-12)
                or not torch.allclose(result.predicted_R, predicted, atol=1e-14, rtol=1e-12)):
            raise ValueError("solver sign-quadrant transformation identity failed")
        summary.update({"solver_active_set": result.diagnostics["selected_active_set"],
            "solver_kkt_status": result.diagnostics["full_kkt_status"],
            "residual_derivative_rank": int(result.diagnostics["residual_derivative_rank"]),
            "identity_checks": {"direction_Da": True, "residual_direction_Va": True,
                                "predicted_R_recomputed": True}})
        summary["side_gradient_dot_Da"] = {
            "gminus_dot_Da": float(torch.dot(gm, direct_direction)),
            "gplus_dot_Da": float(torch.dot(gp, direct_direction))}
        quadrants.append(summary)

    nonnegative = quadrants[0]
    signed_best = min(quadrants, key=lambda item: item["predicted_R"])
    if (nonnegative["sign_quadrant"] != [1.0, 1.0]
            or nonnegative["coefficients_original"] != raw["arms"][1]["candidate"]["model_coefficients"]):
        # candidate coefficients include chart alpha and can differ from the
        # frozen model minimizer; preserve both but do not conflate them.
        nonnegative["matches_recorded_trial_coefficients"] = False
    else:
        nonnegative["matches_recorded_trial_coefficients"] = True

    q, r = torch.linalg.qr(V, mode="reduced")
    qr_coefficients = torch.linalg.solve_triangular(
        r, -(q.T @ G).reshape(2, 1), upper=True).reshape(2)
    qr_summary = kkt_summary(G, D, V, qr_coefficients,
                             torch.ones(2, dtype=G.dtype), RADIUS, constrained=False)
    qr_summary["within_radius"] = qr_summary["control_norm"] <= RADIUS
    qr_summary["least_squares_gradient_linf"] = float(
        torch.linalg.vector_norm(torch.as_tensor(
            qr_summary["coefficient_gradient_Vt_residual"], dtype=G.dtype), ord=float("inf")))
    qr_summary["V_condition_number"] = float(torch.linalg.cond(V))
    qr_summary["side_gradient_dot_Da"] = {
        "gminus_dot_Da": float(torch.dot(gm, D @ qr_coefficients)),
        "gplus_dot_Da": float(torch.dot(gp, D @ qr_coefficients))}
    qr_summary["solver"] = "unconstrained signed QR least squares (no orthant/radius projection)"

    output = {"scope": "stored PR304 model arithmetic only; no FV/HVP/candidate evaluations",
        "pins": {**pins, **parent_pins}, "base": {"control_sha256": BASE_SHA,
            "control": parent["current_control"], "G": G.tolist(),
            "J": parent["current_point"]["J"] if isinstance(parent.get("current_point"), dict) else None,
            "R": float(torch.dot(G, G)), "theta": parent["current_theta"]},
        "model_columns": {"names": list(names), "D": D.tolist(), "V": V.tolist(),
            "D_sha256_float64": sha(D.contiguous().numpy().tobytes()),
            "V_sha256_float64": sha(V.contiguous().numpy().tobytes()),
            "G_sha256_float64": sha(G.contiguous().numpy().tobytes()),
            "gm": gm.tolist(), "gp": gp.tolist(), "side_gradient_source": "final 2a4d repeat in PR303 receipt"},
        "radius": RADIUS, "nonnegative_cone": nonnegative,
        "signed_quadrants": quadrants, "best_signed_quadrant": {
            "sign_quadrant": signed_best["sign_quadrant"],
            "coefficients_original": signed_best["coefficients_original"],
            "predicted_R": signed_best["predicted_R"],
            "control_norm": signed_best["control_norm"],
            "side_gradient_dot_Da": signed_best["side_gradient_dot_Da"]},
        "unconstrained_signed_qr_ls": qr_summary,
        "actual_FV_or_HVP_calls": 0, "actual_candidate_calls": 0}
    out_path = WORK / "AUDIT.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out_path.relative_to(ROOT)),
        "plan_sha256": PLAN_SHA, "nonnegative_predicted_R": nonnegative["predicted_R"],
        "best_signed_predicted_R": signed_best["predicted_R"],
        "unconstrained_signed_control_norm": qr_summary["control_norm"],
        "radius": RADIUS}, sort_keys=True))


if __name__ == "__main__":
    main()
