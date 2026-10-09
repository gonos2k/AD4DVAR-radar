"""Requalify one archived nonsmooth direction against the current branch pair."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "CANDIDATE_REQUALIFICATION_PLAN_20261009.json"
ARCHIVED_STEP = EVIDENCE / "nonsmooth_coupled_20261009_attempt1/step.json.gz"
ARCHIVED_RUN = EVIDENCE / "nonsmooth_coupled_20261009_attempt1/step.run.json"
ARCHIVED_RESOURCE = EVIDENCE / "nonsmooth_coupled_20261009_attempt1/step.resource.json"
ARCHIVE_MANIFEST = EVIDENCE / "KINK_ARCHIVE_20261009.json"
FIXED_PLAN = EVIDENCE / "NONSMOOTH_COUPLED_FACE_PLAN_20261009.json"
SELF = "examples/weather_scenarios/fv_point_3h_candidate_requalification.py"
TEST = "tests/test_fv_point_3h_candidate_requalification.py"
POLICY = {
    "guarded_launches": 1, "internal_seconds": 240.0, "outer_seconds": 300.0,
    "rss_bytes": 1024**3, "hvp_calls": 2, "basis_hvps_per_side": 0,
    "parity_hvps": 0, "dense_solves": 0, "max_candidates": 8,
    "radius": 0.05, "optimizer_steps": 1, "pcg_solves": 0,
    "gradient_scale": 1.0, "face_constraint_scale": "max_abs_face_weight",
    "root_claim": False, "minimum_claim": False, "response_claim": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(probe_module: Any | None = None) -> Any:
    """Load shared probe helpers without a module-import cycle."""
    if probe_module is not None:
        return probe_module
    return importlib.import_module(
        "examples.weather_scenarios.fv_point_3h_nonsmooth_coupled_probe")


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("candidate-requalification plan identity mismatch")
    plan = json.loads(path.read_text())
    fixed = json.loads(FIXED_PLAN.read_text())
    actual_source_manifest = json.loads((EVIDENCE / "kink_actual_source_20261009/manifest.json").read_text())
    if (plan.get("experiment_kind") != "nonsmooth_candidate_requalification"
            or plan.get("policy") != POLICY
            or plan.get("producing_plan") != FIXED_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != "83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299"
            or _sha(FIXED_PLAN) != plan.get("producing_plan_sha256")
            or plan.get("base_control_sha256") != "95a55578eea1d3b9600a210e7bdba37c21d7d7a1baf07244afc513a46df8c842"
            or plan.get("producer_source_snapshots") != actual_source_manifest.get("snapshots")):
        raise ValueError("candidate-requalification plan kind or bounded policy changed")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or not isinstance(archives, dict):
        raise ValueError("candidate-requalification source maps are missing")
    expected_sources = set(fixed["source_files"]) | {SELF, TEST}
    actual_manifest_path = EVIDENCE / "kink_actual_source_20261009/manifest.json"
    actual_probe = EVIDENCE / "kink_actual_source_20261009/probe.py"
    actual_test = EVIDENCE / "kink_actual_source_20261009/tests.py"
    expected_archives = set(fixed["archive_files"]) | {
        FIXED_PLAN.relative_to(ROOT).as_posix(),
        ARCHIVED_STEP.relative_to(ROOT).as_posix(),
        ARCHIVED_RUN.relative_to(ROOT).as_posix(),
        ARCHIVED_RESOURCE.relative_to(ROOT).as_posix(),
        ARCHIVE_MANIFEST.relative_to(ROOT).as_posix(),
        actual_manifest_path.relative_to(ROOT).as_posix(),
        actual_probe.relative_to(ROOT).as_posix(),
        actual_test.relative_to(ROOT).as_posix(),
    }
    if set(sources) != expected_sources or len(sources) != 124 \
            or set(archives) != expected_archives or len(archives) != 86:
        raise ValueError("candidate-requalification source/archive scope changed")
    for name, old_digest in fixed["source_files"].items():
        if name not in actual_source_manifest["snapshots"] and sources.get(name) != old_digest:
            raise ValueError(f"inherited actual J/H source pin changed: {name}")
    for name, old_digest in fixed["archive_files"].items():
        if archives.get(name) != old_digest:
            raise ValueError(f"inherited archive pin changed: {name}")
    for name, snapshot in actual_source_manifest["snapshots"].items():
        archive_name = snapshot["archive_path"]
        if (fixed["source_files"].get(name) != snapshot["sha256"]
                or archives.get(archive_name) != snapshot["sha256"]
                or _sha(ROOT / archive_name) != snapshot["sha256"]):
            raise ValueError(f"archived producer source snapshot changed: {name}")
    if sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST):
        raise ValueError("candidate-requalification code or regression tests differ from plan")
    for name, digest_value in {**sources, **archives}.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or source.is_symlink() or _sha(source) != digest_value:
            raise ValueError(f"candidate-requalification source/archive pin mismatch: {name}")
    for archived_path in (FIXED_PLAN, ARCHIVED_STEP, ARCHIVED_RUN, ARCHIVED_RESOURCE,
                          ARCHIVE_MANIFEST, actual_manifest_path, actual_probe, actual_test):
        name = archived_path.relative_to(ROOT).as_posix()
        if archives.get(name) != _sha(archived_path):
            raise ValueError(f"candidate-requalification archive receipt differs from plan: {name}")
    for field, expected in (("direction_archive", ARCHIVED_STEP),
                            ("direction_run", ARCHIVED_RUN),
                            ("direction_resource", ARCHIVED_RESOURCE)):
        if plan.get(field) != expected.relative_to(ROOT).as_posix():
            raise ValueError(f"candidate-requalification {field} path changed")
    for field, expected in (("direction_archive_sha256", ARCHIVED_STEP),
                            ("direction_run_sha256", ARCHIVED_RUN),
                            ("direction_resource_sha256", ARCHIVED_RESOURCE)):
        if plan.get(field) != _sha(expected):
            raise ValueError(f"candidate-requalification {field} pin changed")
    archive_entry = json.loads(ARCHIVE_MANIFEST.read_text())["archives"][
        "graphify-out/fv-root-cause-20260919/nonsmooth_coupled_20261009_attempt1/step.json"]
    if plan.get("direction_child_sha256") != archive_entry.get("raw_sha256"):
        raise ValueError("candidate-requalification raw child pin differs from archive manifest")
    return plan


def current_branch_pair_gate(minus: dict[str, Any], plus: dict[str, Any], *,
                             dtype: torch.dtype, stages: int = 360) -> dict[str, Any]:
    """Require strict current-side traces to agree with each other."""
    eps = torch.finfo(dtype).eps
    bound = 128 * eps * max(float(minus.get("maximum_face_flux", 0.0)),
                            float(plus.get("maximum_face_flux", 0.0)), 1e-300)
    target_max = max(max(map(abs, minus.get("target_fluxes", ())), default=0.0),
                     max(map(abs, plus.get("target_fluxes", ())), default=0.0))
    same_choices = minus.get("choices") == plus.get("choices")
    same_signs = minus.get("face_signs") == plus.get("face_signs")
    complete = all(row.get(key) == stages for row in (minus, plus)
        for key in ("stage_count", "observed_stage_count", "expected_stage_count"))
    passed = (complete
        and not minus.get("nonfinite_or_tie") and not plus.get("nonfinite_or_tie")
        and same_choices and same_signs and target_max <= bound)
    return {"passed": passed, "target_flux_bound": bound,
        "target_flux_max_abs": target_max, "same_current_choices": same_choices,
        "same_current_face_signs": same_signs, "signature_changed_from_base": None}


def load_archived_direction(step_path: Path = ARCHIVED_STEP, *,
                            base_control_sha256: str, old_plan_sha256: str,
                            expected_child_sha256: str | None = None) -> dict[str, Any]:
    """Read the compressed refusal receipt and validate its reusable direction."""
    payload = gzip.decompress(step_path.read_bytes()) if step_path.suffix == ".gz" else step_path.read_bytes()
    child_sha = hashlib.sha256(payload).hexdigest()
    if expected_child_sha256 is not None and child_sha != expected_child_sha256:
        raise ValueError("archived child bytes differ from the parent run receipt")
    run = json.loads(ARCHIVED_RUN.read_text())
    resource = json.loads(ARCHIVED_RESOURCE.read_text())
    archive_manifest = json.loads(ARCHIVE_MANIFEST.read_text())
    entry = archive_manifest.get("archives", {}).get(
        "graphify-out/fv-root-cause-20260919/nonsmooth_coupled_20261009_attempt1/step.json")
    run_resource = run.get("resource", {})
    if (not isinstance(entry, dict) or entry.get("raw_sha256") != child_sha
            or entry.get("gzip_path") != step_path.relative_to(ROOT).as_posix()
            or entry.get("gzip_sha256") != _sha(step_path)
            or run.get("execution_status") != "completed"
            or run.get("child_sha256") != child_sha
            or run.get("numerical_status") != "coupled_step_refused"
            or run.get("child_read_error") is not None
            or run_resource != resource
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 660.0
            or resource.get("rss_limit_bytes") != 1024**3
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > 660.0
            or not isinstance(resource.get("sampled_peak_rss_bytes"), (int, float))
            or resource["sampled_peak_rss_bytes"] >= 1024**3):
        raise ValueError("archived parent/resource/archive-manifest closure is invalid")
    raw = json.loads(payload)
    old_plan_hashes = raw.get("plan_hashes", {})
    old_source_union = {**old_plan_hashes.get("source_files", {}),
        **old_plan_hashes.get("archive_files", {}),
        FIXED_PLAN.relative_to(ROOT).as_posix(): old_plan_sha256}
    hvp_history = raw.get("hvp_history", [])
    parity_kinds = {(row.get("side"), row.get("operator"), row.get("status"))
        for row in hvp_history if row.get("phase") == "parity"}
    basis_columns = {(row.get("side"), row.get("column")) for row in raw.get("hvp_columns", [])}
    expected_columns = {(side, column) for side in (-1, 1) for column in range(26)}
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("plan_sha256") != old_plan_sha256
            or raw.get("base_control_sha256") != base_control_sha256
            or raw.get("current_control_sha256") != base_control_sha256
            or raw.get("candidate_committed") is not False
            or raw.get("active_candidate_committed") is not False
            or raw.get("optimizer_steps_applied") != 0
            or raw.get("hvp_calls_started") != 56 or raw.get("hvp_calls_completed") != 56
            or len(hvp_history) != 56 or len(raw.get("hvp_columns", [])) != 52
            or basis_columns != expected_columns
            or sum(row.get("phase") == "parity" for row in hvp_history) != 4
            or sum(row.get("phase") == "basis" for row in hvp_history) != 52
            or parity_kinds != {(side, operator, "completed") for side in (-1, 1)
                                for operator in ("native", "selected_extension")}
            or any(row.get("operator", "selected_extension") != "selected_extension" for row in hvp_history
                   if row.get("phase") == "basis")
            or any(row.get("status") != "completed" for row in hvp_history)
            or raw.get("hessian_scope") !=
                "full 26x26 ambient FP64 one-sided Hessians; exact selected branch contexts"
            or not raw.get("source_unchanged") or not raw.get("fixed_input_unchanged")
            or not raw.get("runtime_unchanged")
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("source_before") != old_source_union
            or raw.get("input_before") != raw.get("input_after")
            or raw.get("runtime") != raw.get("runtime_after")
            or raw.get("current_control_sha256") != base_control_sha256
            or not isinstance(raw.get("iterations"), list)
            or len(raw["iterations"]) != 1
            or raw["iterations"][0].get("accepted") is not None):
        raise ValueError("archived producer is not a completed closed refusal at the shared base")
    delta = torch.as_tensor(raw.get("delta"), dtype=torch.float64)
    hminus = torch.as_tensor(raw.get("hessian_minus"), dtype=torch.float64)
    hplus = torch.as_tensor(raw.get("hessian_plus"), dtype=torch.float64)
    matrix = torch.as_tensor(raw.get("matrix"), dtype=torch.float64)
    residual = torch.as_tensor(raw.get("residual"), dtype=torch.float64)
    if (delta.shape != (27,) or hminus.shape != (26, 26) or hplus.shape != (26, 26)
            or matrix.shape != (27, 27) or residual.shape != (27,)
            or not bool(torch.isfinite(delta).all() & torch.isfinite(hminus).all()
                        & torch.isfinite(hplus).all() & torch.isfinite(matrix).all()
                        & torch.isfinite(residual).all())
            or not math.isfinite(float(raw.get("theta", float("nan"))))
            or not 0.0 <= float(raw["theta"]) <= 1.0):
        raise ValueError("archived direction or same-point reference Hessians are invalid")
    return {"raw": raw, "delta": delta, "hessian_minus": hminus,
        "hessian_plus": hplus, "child_sha256": child_sha}


def validate_same_point_evidence(archived: dict[str, Any], control: Tensor,
                                 traces: dict[int, dict[str, Any]],
                                 side_gradients: dict[int, Tensor],
                                 source_before: dict[str, str],
                                 input_before: dict[str, Any],
                                 runtime_before: dict[str, str]) -> dict[str, Any]:
    """Bind current J/g/branch/input/source evidence to the archived base point."""
    raw = archived["raw"]
    if probe_sha := raw.get("face_control_sha256"):
        if probe_sha != hashlib.sha256(control.detach().contiguous().cpu().numpy().tobytes()).hexdigest():
            raise ValueError("fresh zero-face chart control differs from archived face control")
    else:
        raise ValueError("archived receipt has no same-point face-control hash")
    saved_traces = raw.get("face_qualification", {}).get("traces", {})
    for side in (-1, 1):
        saved = saved_traces.get(str(side), {}).get("signature_sha256")
        if saved is None or traces[side].get("signature_sha256") != saved:
            raise ValueError("fresh base-side branch signature differs from archived same point")
    old_sources = raw["source_before"]
    changed_controls = {
        "examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py",
        "tests/test_fv_point_3h_nonsmooth_coupled_probe.py"}
    if any(source_before.get(name) != value for name, value in old_sources.items()
           if name not in changed_controls):
        raise ValueError("fresh actual J/H source map differs from archived same point")
    if raw.get("input_before") != input_before or raw.get("runtime") != runtime_before:
        raise ValueError("fresh fixed input/runtime differs from archived same point")
    old_matrix = torch.as_tensor(raw["matrix"], dtype=control.dtype)
    old_residual = torch.as_tensor(raw["residual"], dtype=control.dtype)
    old_jump = old_matrix[:26, 26]
    old_theta = float(raw["theta"])
    old_mixed = old_residual[:26]
    reconstructed = {
        -1: old_mixed - old_theta * old_jump,
        1: old_mixed + (1 - old_theta) * old_jump,
    }
    gradient_errors: dict[str, float] = {}
    for side in (-1, 1):
        scale = max(float(side_gradients[side].abs().max()),
                    float(reconstructed[side].abs().max()), torch.finfo(control.dtype).tiny)
        error = float((side_gradients[side] - reconstructed[side]).abs().max())
        gradient_errors[str(side)] = error
        if error > 128 * torch.finfo(control.dtype).eps * scale:
            raise ValueError("fresh side gradient differs from archived F/jump/theta reconstruction")
    return {"face_control_sha256": probe_sha, "base_side_signatures_match": True,
        "actual_jh_sources_match": True, "input_runtime_match": True,
        "reconstructed_side_gradient_max_errors": gradient_errors,
        "gradient_tolerance": 128 * torch.finfo(control.dtype).eps}


def validate_current_direction(probe: Any, record: dict[str, Any], output: Path,
                               deadline: float, problem: Any, control: Tensor,
                               parameters: Tensor, archived: dict[str, Any],
                               gminus: Tensor, gplus: Tensor, normal: Tensor,
                               qscale: Tensor) -> dict[str, Any]:
    """Check the archived step with one fresh current HVP on each side."""
    delta = archived["delta"]
    weights = probe._face_weights(problem, **probe.FACE)
    pivot = probe._pivot(control, weights)
    path_direction, _ = probe._path_direction(control, weights, pivot, delta)
    gfn = torch.func.grad(problem.objective, argnums=0)
    products: dict[int, Tensor] = {}
    for side in (-1, 1):
        def calculate(sign: int = side) -> Tensor:
            with probe.transport.selected_face_extension(
                    probe.FACE["axis"], probe.FACE["row"], probe.FACE["column"], sign):
                return torch.func.jvp(lambda c: gfn(c, parameters),
                    (control,), (path_direction,))[1]
        products[side] = probe._counted_hvp(record, output, deadline,
            {"phase": "current_direction_validation", "side": side,
             "operator": "selected_face_extension", "scope": "archived direction only"}, calculate)
    theta = probe._min_norm_mix(gminus, gplus)[0]
    theta_old = float(archived["raw"]["theta"])
    top = ((1 - theta) * products[-1] + theta * products[1]
        + (gplus - gminus) * delta[-1])
    q = probe._face_value(control, weights)
    bottom = torch.dot(normal, path_direction) / qscale
    actual_product = torch.cat((top, bottom.reshape(1)))
    fresh_residual = probe._residual_measure((1 - theta) * gminus + theta * gplus, q, qscale)
    remainder = actual_product + fresh_residual
    actual_rhs_relative = torch.linalg.vector_norm(remainder) / torch.linalg.vector_norm(
        fresh_residual).clamp_min(torch.finfo(control.dtype).tiny)
    archived_matrix = torch.as_tensor(archived["raw"]["matrix"], dtype=control.dtype)
    archived_residual = torch.as_tensor(archived["raw"]["residual"], dtype=control.dtype)
    archived_remainder = archived_matrix @ delta + archived_residual
    archived_denominator = (torch.linalg.vector_norm(archived_matrix)
        * torch.linalg.vector_norm(delta) + torch.linalg.vector_norm(archived_residual))
    archived_numerator = torch.linalg.vector_norm(archived_remainder)
    archived_backward_error = archived_numerator / archived_denominator.clamp_min(
        torch.finfo(control.dtype).tiny)
    actual_rhs_norm = torch.linalg.vector_norm(remainder)
    actual_rhs_denominator = torch.linalg.vector_norm(fresh_residual)
    passed = bool(torch.isfinite(actual_rhs_relative) & torch.isfinite(archived_backward_error)
        & (actual_rhs_relative <= 1e-10))
    return {"passed": passed, "theta_current": theta, "theta_archived": theta_old,
        "actual_rhs_relative": float(actual_rhs_relative),
        "actual_rhs_residual_norm": float(actual_rhs_norm),
        "actual_rhs_denominator_norm": float(actual_rhs_denominator),
        "actual_rhs_residual": remainder,
        "archived_matrix_backward_error": float(archived_backward_error),
        "archived_matrix_backward_error_numerator": float(archived_numerator),
        "archived_matrix_backward_error_denominator": float(archived_denominator),
        "archived_matrix_norm": float(torch.linalg.vector_norm(archived_matrix)),
        "actual_product": actual_product, "fresh_residual": fresh_residual,
        "path_direction": path_direction, "scope": "current archived direction validation only"}


def prepare_current_direction(probe: Any, record: dict[str, Any], output: Path,
                               deadline: float, problem: Any, control: Tensor,
                               parameters: Tensor, archived: dict[str, Any],
                               gminus: Tensor, gplus: Tensor, normal: Tensor,
                               qscale: Tensor, q: Tensor) -> tuple[Tensor, Tensor, float, Tensor, dict[str, Any]]:
    """Build common-search inputs after the current two-HVP qualification."""
    check = validate_current_direction(probe, record, output, deadline, problem,
        control, parameters, archived, gminus, gplus, normal, qscale)
    if not check["passed"]:
        raise ValueError("archived direction failed fresh current-side HVP residual gate")
    theta = check["theta_current"]
    matrix, residual, _ = probe._coupled_matrix(archived["hessian_minus"],
        archived["hessian_plus"], gminus, gplus, normal, q, theta, qscale)
    info = {"passed": True,
        "method": "archived direction verified by fresh current HVPs",
        "unknowns": 27, "actual_rhs_relative": check["actual_rhs_relative"],
        "actual_rhs_residual_norm": check["actual_rhs_residual_norm"],
        "actual_rhs_denominator_norm": check["actual_rhs_denominator_norm"],
        "actual_rhs_residual": check["actual_rhs_residual"].tolist(),
        "archived_matrix_backward_error": check["archived_matrix_backward_error"],
        "archived_matrix_backward_error_numerator": check["archived_matrix_backward_error_numerator"],
        "archived_matrix_backward_error_denominator": check["archived_matrix_backward_error_denominator"],
        "current_hvp_calls": 2, "dense_solves": 0,
        "hessian_scope": "same-point archived reference diagnostic; no current Hessian claim"}
    return matrix, residual, theta, archived["delta"], {"solve": info, "check": check}


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path,
                    closure_state: dict[str, Any], *, probe_module: Any | None = None) -> dict[str, Any]:
    """Enter the shared probe with the requalification path selected."""
    probe = _probe(probe_module)
    return probe._run_child_impl(plan_path, plan_sha, output, closure_state,
        requalification=True)
