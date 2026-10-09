"""Compare the baseline and coupled robust-GN tangent directions at PR272c."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_probe as mixing
from examples.weather_scenarios import fv_point_3h_tangent_mixing_minimum_resume as resume
from examples.weather_scenarios import fv_point_3h_tangent_precondition_comparison as precondition

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "TANGENT_COUPLED_GN_COMPARISON_PLAN_20261010.json"
PRODUCING_PLAN = EVIDENCE / "TANGENT_MIXING_MINIMUM_RESUME_PLAN_20261010.json"
PRODUCING_PLAN_SHA = "a3f24f93e84ec107c36a8888fbe2e644f9a5be3a71bff855ec219bf9be7c0bba"
PRODUCER_ARCHIVE = EVIDENCE / "MIXMIN_RESUME_ARCHIVE_20261010.json"
BASE_ARCHIVE = EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.json.gz"
BASE_RUN = EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.run.json"
BASE_RESOURCE = EVIDENCE / "tangent_mixing_minimum_resume_20261010_attempt1/step.resource.json"
SOURCE_MANIFEST = EVIDENCE / "gncoupled_source_20261010/manifest.json"
PRODUCER_ARCHIVE_SHA = "d4ddc48cd6861020ff6d73767ddd3dbacc9b7c6ea87d4bea25545e3463380bdb"
BASE_RAW_SHA = "2a36ae30ec9749af1b212ef3395c8125fd46a18ca77956807ddc73ca5e9cb0ff"
BASE_GZIP_SHA = "4d6d2eb1b81562ab50dc98b0540e0480155ea40a5c9783bcf87d7f4ff4d8e844"
BASE_RUN_SHA = "6b99e9e93d60dab10b7377d9c27fa6ab18e9ca1a4e3a35bc8e6e609ca52dbfbf"
BASE_RESOURCE_SHA = "cbca6ceebda5ac70206429c1522e34ae396ba8ca3cf459806c9e5c48237f1972"
BASE_CONTROL_SHA = "c8fba1f93ee0b6c83edc5b56792de158921c0a64bab41eef5403c2a514b1fc33"
BASE_THETA = 0.4818866600367756
BASE_OBJECTIVE = 0.06123349298793274
BASE_CARRIED_F_SQUARED = 0.004605971728402105
FACE = mixing.FACE
SELF = "examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py"
TEST = "tests/test_fv_point_3h_tangent_coupled_gn_comparison.py"
POLICY: dict[str, Any] = {
    "guarded_launches": 1, "internal_seconds": 600.0, "outer_seconds": 660.0,
    "rss_bytes": 1024**3, "direction_models": 2,
    "candidate_grid_per_direction": 16, "radius": 0.05, "face_scale": 0.84,
    "jacobian_row_vjp_calls": 24, "hvp_calls": 4,
    "max_accepted_iterations": 1, "optimizer_steps": 1,
    "pcg_solves": 0, "dense_solves": 1, "root_claim": False,
    "minimum_claim": False, "response_claim": False, "score_claim": False,
}
METHOD = {
    "observation_rows": 12,
    "residual_space": "quality_standardized_then_same_time_correlation_whitened_point_dbz",
    "robust_loss": "pseudo_huber_delta_2",
    "field_smoothness_weight": 0.0,
    "neural_prior": False,
    "theta_rule": "unique_resolved_strict_interior_argmin_over_theta_of_squared_mixed_side_gradient",
    "merit": "minimum_squared_mixed_side_gradient_plus_scaled_face_residual",
    "coupled_system": "S=I12+B*B.T; B=sqrt(D)*Aminus*P; d=-t+B.T*solve(S,B*t)",
    "direction_models": ["baseline_tangent", "robust_gn_coupled"],
    "candidate_theta": "recomputed_from_candidate_side_gradients_no_clipping",
    "candidate_selection": "actual_F_squared_then_native_J_then_baseline_with_128_epsilon_ties",
    "residual_curvature_included": False,
    "full_control_hessian_formed": False,
}
OBSERVATION_ROWS = 12
ANALYSIS_STAGES = 360
J_C1 = F_C1 = 1e-4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError("coupled-GN plan identity mismatch")
    if (_sha(PRODUCING_PLAN) != PRODUCING_PLAN_SHA
            or _sha(PRODUCER_ARCHIVE) != PRODUCER_ARCHIVE_SHA):
        raise ValueError("PR272c producer pins changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    snapshots = json.loads(SOURCE_MANIFEST.read_text())["snapshots"]
    plan = json.loads(path.read_text())
    prior_sources, prior_archives = prior["source_files"], prior["archive_files"]
    expected_sources = set(prior_sources) | {SELF, TEST}
    expected_archives = set(prior_archives) | {
        PRODUCING_PLAN.relative_to(ROOT).as_posix(), PRODUCER_ARCHIVE.relative_to(ROOT).as_posix(),
        BASE_ARCHIVE.relative_to(ROOT).as_posix(), BASE_RUN.relative_to(ROOT).as_posix(),
        BASE_RESOURCE.relative_to(ROOT).as_posix(), SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        *[snapshot["archive_path"] for snapshot in snapshots.values()],
    }
    if (plan.get("experiment_kind") != "current_tangent_coupled_gn_comparison"
            or plan.get("policy") != POLICY or plan.get("method") != METHOD
            or plan.get("producing_plan") != PRODUCING_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCING_PLAN_SHA
            or plan.get("base_control_sha256") != BASE_CONTROL_SHA
            or plan.get("initial_theta") != BASE_THETA
            or plan.get("base_objective") != BASE_OBJECTIVE
            or plan.get("base_carried_F_squared") != BASE_CARRIED_F_SQUARED
            or plan.get("face") != FACE
            or plan.get("producer_source_snapshots") != snapshots
            or set(plan.get("source_files", {})) != expected_sources
            or len(plan["source_files"]) != 137
            or set(plan.get("archive_files", {})) != expected_archives
            or len(plan["archive_files"]) != 141):
        raise ValueError("coupled-GN plan scope or policy changed")
    sources, archives = plan["source_files"], plan["archive_files"]
    if any(archives.get(name) != value for name, value in prior_archives.items()):
        raise ValueError("inherited PR272c archive pin changed")
    snapshot_names = set(snapshots)
    if any(sources.get(name) != value for name, value in prior_sources.items()
            if name not in snapshot_names):
        raise ValueError("inherited PR272c source pin changed")
    if any(prior_sources.get(name) != snapshot["sha256"] for name, snapshot in snapshots.items()):
        raise ValueError("pre-coupled source snapshot differs from PR272c plan")
    if any(sources.get(name) != _sha(ROOT / name) for name in snapshots):
        raise ValueError("current coupled source pins differ from successor plan")
    for name, snapshot in snapshots.items():
        if (archives.get(snapshot["archive_path"]) != snapshot["sha256"]
                or _sha(ROOT / snapshot["archive_path"]) != snapshot["sha256"]):
            raise ValueError(f"pre-edit shared source snapshot changed: {name}")
    for name, value in {**sources, **archives}.items():
        target = ROOT / name
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()) or _sha(target) != value:
            raise ValueError(f"coupled-GN source/archive pin mismatch: {name}")
    if (plan.get("base_archive") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("base_archive_sha256") != BASE_GZIP_SHA
            or plan.get("base_run") != BASE_RUN.relative_to(ROOT).as_posix()
            or plan.get("base_run_sha256") != BASE_RUN_SHA
            or plan.get("base_resource") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or plan.get("base_resource_sha256") != BASE_RESOURCE_SHA
            or plan.get("base_archive_manifest") != PRODUCER_ARCHIVE.relative_to(ROOT).as_posix()
            or plan.get("base_archive_manifest_sha256") != PRODUCER_ARCHIVE_SHA
            or plan.get("base_child_sha256") != BASE_RAW_SHA):
        raise ValueError("coupled-GN predecessor receipt pins changed")
    return plan


def _load_current_base(plan: dict[str, Any]) -> dict[str, Any]:
    """Close the PR272c three-point resume chain from saved receipts only."""
    if (_sha(PRODUCING_PLAN) != PRODUCING_PLAN_SHA
            or _sha(PRODUCER_ARCHIVE) != PRODUCER_ARCHIVE_SHA
            or _sha(BASE_RUN) != BASE_RUN_SHA or _sha(BASE_RESOURCE) != BASE_RESOURCE_SHA):
        raise ValueError("PR272c plan/run/resource receipt digests changed")
    prior = json.loads(PRODUCING_PLAN.read_text())
    manifest = json.loads(PRODUCER_ARCHIVE.read_text())
    compressed = BASE_ARCHIVE.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    child_sha = hashlib.sha256(raw_bytes).hexdigest()
    if (child_sha != BASE_RAW_SHA or _sha(BASE_ARCHIVE) != BASE_GZIP_SHA
            or manifest.get("raw_sha256") != BASE_RAW_SHA
            or manifest.get("gzip_sha256") != BASE_GZIP_SHA
            or manifest.get("run_sha256") != BASE_RUN_SHA
            or manifest.get("resource_sha256") != BASE_RESOURCE_SHA
            or manifest.get("guarded_launch_count") != 1):
        raise ValueError("PR272c lossless archive manifest does not close")
    raw = json.loads(raw_bytes)
    parent = json.loads(BASE_RUN.read_text())
    resource = json.loads(BASE_RESOURCE.read_text())
    source_receipt = {**prior["source_files"], **prior["archive_files"],
        PRODUCING_PLAN.relative_to(ROOT).as_posix(): PRODUCING_PLAN_SHA}
    if (manifest.get("raw_path") != "graphify-out/fv-root-cause-20260919/tangent_mixing_minimum_resume_20261010_attempt1/step.json"
            or manifest.get("gzip_path") != BASE_ARCHIVE.relative_to(ROOT).as_posix()
            or manifest.get("raw_bytes") != len(raw_bytes)
            or manifest.get("gzip_bytes") != len(compressed)
            or manifest.get("run_path") != BASE_RUN.relative_to(ROOT).as_posix()
            or manifest.get("resource_path") != BASE_RESOURCE.relative_to(ROOT).as_posix()
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "tangent_mixing_minimum_resume_cap_reached"
            or parent.get("child_read_error") is not None or parent.get("child_sha256") != child_sha
            or parent.get("resource") != resource
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != resume.POLICY["outer_seconds"]
            or resource.get("rss_limit_bytes") != resume.POLICY["rss_bytes"]
            or not isinstance(resource.get("elapsed_seconds"), (int, float))
            or resource["elapsed_seconds"] > resume.POLICY["outer_seconds"]
            or resource.get("sampled_peak_rss_bytes", resume.POLICY["rss_bytes"])
                >= resume.POLICY["rss_bytes"]
            or raw.get("plan_sha256") != PRODUCING_PLAN_SHA
            or raw.get("plan_hashes", {}).get("source_files") != prior["source_files"]
            or raw.get("plan_hashes", {}).get("archive_files") != prior["archive_files"]
            or raw.get("source_before") != source_receipt or raw.get("source_after") != source_receipt
            or raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "tangent_mixing_minimum_resume_cap_reached"
            or raw.get("base_control_sha256") != resume.BASE_CONTROL_SHA
            or raw.get("current_control_sha256") != BASE_CONTROL_SHA
            or raw.get("candidate_committed") is not True or raw.get("active_candidate_committed") is not False
            or raw.get("accepted_iterations") != 3 or raw.get("optimizer_steps_applied") != 3
            or raw.get("hvp_calls_started") != raw.get("hvp_calls_completed")
            or raw.get("hvp_calls_completed") != 6
            or raw.get("jacobian_rows_started") != 0 or raw.get("jacobian_rows_completed") != 0
            or raw.get("source_unchanged") is not True or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True or raw.get("deadline_passed") is not True
            or raw.get("last_confirmed_control_sha256") != BASE_CONTROL_SHA
            or raw.get("last_confirmed_theta") != BASE_THETA or raw.get("last_confirmed_iterations") != 3
            or raw.get("current_theta") != BASE_THETA
            or not geometry.model._fixed_input(raw.get("input_after", {}),
                raw.get("input_before", {}), BASE_CONTROL_SHA)):
        raise ValueError("PR272c endpoint/input/resource/source closure is invalid")
    if not tangent._mixing_minimum_resume_chain_closed(
            raw, prior, resume.BASE_CONTROL_SHA, raw.get("hvp_history", [])):
        raise ValueError("PR272c three-step accepted control/theta/HVP chain is invalid")
    final = raw.get("last_confirmed_closure")
    if not isinstance(final, dict) or final != raw.get("iterations", [{}])[-1].get("final_repeat"):
        raise ValueError("PR272c terminal final-repeat receipt is not the last committed endpoint")
    control = torch.as_tensor(final["control"], dtype=torch.float64)
    if tangent._tensor_sha(control) != BASE_CONTROL_SHA:
        raise ValueError("PR272c endpoint control digest is invalid")
    return {"raw": raw, "control": control, "theta": BASE_THETA,
        "objective": float(final["objective"]), "accepted": final, "child_sha256": child_sha}


def coupled_woodbury_direction(normal: Tensor, tangent_residual: Tensor,
        jacobian_minus: Tensor, robust_curvature: Tensor, chart: Tensor, pivot: int,
        *, on_solve_start: Any = None, on_solve_complete: Any = None
        ) -> tuple[Tensor, dict[str, Any]]:
    """Apply the 12-row Woodbury GN model, with only one row-space solve."""
    size, rows = tangent_residual.numel(), robust_curvature.numel()
    values = (normal, tangent_residual, jacobian_minus, robust_curvature, chart)
    if (not all(bool(torch.isfinite(value).all()) for value in values)
            or normal.shape != (size,) or jacobian_minus.shape != (rows, size)
            or chart.shape != (size, size - 1) or rows < 1
            or robust_curvature.shape != (rows,) or not bool((robust_curvature >= 0).all())):
        raise ValueError("coupled GN products violate the finite 12x26 profile")
    normal_norm = torch.linalg.vector_norm(normal)
    if not bool(torch.isfinite(normal_norm) & (normal_norm > torch.finfo(normal.dtype).tiny)):
        raise ValueError("coupled GN selected-face normal is unresolved")
    unit = normal / normal_norm
    projector = torch.eye(size, dtype=normal.dtype, device=normal.device) - torch.outer(unit, unit)
    tangent = projector @ tangent_residual
    bmatrix = robust_curvature.sqrt()[:, None] * (jacobian_minus @ projector)
    system = torch.eye(rows, dtype=normal.dtype, device=normal.device) + bmatrix @ bmatrix.T
    system = 0.5 * (system + system.T)
    rhs = bmatrix @ tangent
    if on_solve_start is not None:
        on_solve_start()
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0 or not bool(torch.isfinite(factor).all()):
        raise ValueError("coupled GN row-space system is not positive definite")
    solved = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
    if on_solve_complete is not None:
        on_solve_complete()
    direction_unrepaired = -tangent + bmatrix.T @ solved
    woodbury_correction = bmatrix.T @ solved
    retained = [index for index in range(size) if index != pivot]
    direction = chart @ direction_unrepaired[retained]
    solve_residual = torch.linalg.vector_norm(system @ solved - rhs)
    solve_scale = max(float(torch.linalg.vector_norm(rhs)),
        float(torch.linalg.matrix_norm(system)) * float(torch.linalg.vector_norm(solved)),
        torch.finfo(normal.dtype).tiny)
    solve_budget = 128.0 * torch.finfo(normal.dtype).eps * solve_scale
    repair_norm = torch.linalg.vector_norm(direction - direction_unrepaired)
    repair_budget = 128.0 * torch.finfo(normal.dtype).eps * max(
        float(torch.linalg.vector_norm(direction)),
        float(torch.linalg.vector_norm(direction_unrepaired)), torch.finfo(normal.dtype).tiny)
    normal_residual = torch.dot(normal, direction)
    normal_budget = 128.0 * torch.finfo(normal.dtype).eps * float(normal_norm) \
        * float(torch.linalg.vector_norm(direction))
    descent = torch.dot(tangent, direction)
    descent_budget = 128.0 * torch.finfo(normal.dtype).eps * max(
        float(torch.linalg.vector_norm(tangent)) * float(torch.linalg.vector_norm(direction)),
        torch.finfo(normal.dtype).tiny)
    if (not bool(torch.isfinite(direction).all())
            or float(solve_residual) > solve_budget
            or float(repair_norm) > repair_budget
            or abs(float(normal_residual)) > normal_budget
            or float(descent) >= -descent_budget):
        raise ValueError("coupled GN solve, tangent repair or descent check failed")
    diagnostics = {"dense_solve_dimension": rows, "S_dimension": rows, "dense_solves": 1,
        "S": system.tolist(), "B": bmatrix.tolist(), "t": tangent.tolist(),
        "B_t": rhs.tolist(), "solve_vector": solved.tolist(),
        "woodbury_correction": woodbury_correction.tolist(),
        "uncorrected_direction": direction_unrepaired.tolist(),
        "direction": direction.tolist(), "solve_residual": float(solve_residual),
        "solve_residual_budget": solve_budget,
        "positive_definite_from_cholesky": True,
        "projected_tangent_residual": float(normal_residual),
        "tangent_repair_l2": float(repair_norm),
        "tangent_repair_budget": repair_budget,
        "tangent_descent_product": float(descent), "tangent_descent_budget": descent_budget}
    return direction, diagnostics


def _direction_model_factory(context: dict[str, Any]):
    baseline_factory = mixing._working_model(context)
    def build(control: Tensor, reference_theta: float, gminus: Tensor, gplus: Tensor,
              normal: Tensor, chart: Tensor, pivot: int) -> list[dict[str, Any]]:
        problem, parameters = context["problem"], context["parameters"]
        frozen = problem.frozen
        if (problem.layout.get("controls") != 26 or problem.layout.get("parameters") != 13
                or tuple(problem.observation_dbz.shape) != (3, 4)
                or problem.observation_status is not None or not bool(problem._detected_mask.all())
                or frozen.neural_prior_std_dbz is not None
                or frozen.neural_prior_valid_mask is not None
                or frozen.neural_prior_dependency is not None
                or frozen.analysis_config.field_smoothness_weight != 0.0
                or float(frozen.analysis_config.pseudo_huber_delta) != 2.0):
            raise ValueError("coupled GN requires the fixed all-detected plain-prior point profile")
        contract = problem.contract(parameters)
        prior_residual = precondition.v._control_prior_residual(control, contract)
        if not torch.equal(prior_residual, control):
            raise ValueError("coupled GN requires the original identity control prior")
        baseline_specs = baseline_factory(control, reference_theta, gminus, gplus, normal, chart, pivot)
        baseline = baseline_specs[0]
        theta_star = float(baseline["working_theta"])
        record, output, deadline, plan = (
            context["record"], context["output"], context["deadline"], context["plan"])
        residual_minus, rows_minus = precondition._counted_residual_rows(context, control, -1, theta_star)
        residual_plus, rows_plus = precondition._counted_residual_rows(context, control, 1, theta_star)
        residual_budget, residual_scale = precondition._residual_pair_roundoff_budget(
            problem, parameters, residual_minus, residual_plus)
        residual_error = (residual_minus - residual_plus).abs().reshape_as(residual_budget)
        if bool((residual_error > residual_budget).any()):
            raise ValueError("coupled GN side residual rows differ beyond construction-scaled roundoff")
        data_cost = precondition.v._pseudo_huber_cost(residual_minus, float(
            frozen.analysis_config.pseudo_huber_delta)).sum()
        smooth_cost = precondition.v._field_smoothness_prior_cost(control, contract)
        reconstructed_j = data_cost + 0.5 * torch.dot(prior_residual, prior_residual) + smooth_cost
        observed = context["model_state"]["observed"]
        native_j = float(observed["native_j"])
        side_objectives = {side: float(observed["side"][side][0]) for side in (-1, 1)}
        if (not precondition._close(float(reconstructed_j), native_j, control.dtype)
                or any(not precondition._close(float(reconstructed_j), value, control.dtype)
                    for value in side_objectives.values())
                or float(smooth_cost) != 0.0):
            raise ValueError("coupled-GN whitened-row and fixed-prior decomposition does not match native J")
        unit = normal / torch.linalg.vector_norm(normal)
        projector = torch.eye(26, dtype=control.dtype, device=control.device) - torch.outer(unit, unit)
        row_error, row_budget, ambient_minus, ambient_plus, projected_norms = (
            precondition._projected_row_parity(rows_minus, rows_plus, projector))
        if bool((row_error > row_budget).any()):
            raise ValueError("coupled GN projected side residual rows do not agree")
        delta = float(frozen.analysis_config.pseudo_huber_delta)
        robust_curvature = precondition.pseudo_huber_second_derivative(residual_minus, delta)
        theta_check = float(tangent.minimum_mixture_weight(gminus, gplus))
        theta_check_budget = 128.0 * torch.finfo(control.dtype).eps \
            * max(abs(theta_star), abs(theta_check), torch.finfo(control.dtype).tiny)
        if abs(theta_star - theta_check) > theta_check_budget:
            raise ValueError("baseline and coupled arms do not share the same current theta-star")
        mixture = (1.0 - theta_star) * gminus + theta_star * gplus
        tangent_residual = projector @ mixture
        record["dense_solves_started"] = int(record.get("dense_solves_started", 0))
        record["dense_solves_completed"] = int(record.get("dense_solves_completed", 0))
        def solve_start() -> None:
            if record["dense_solves_started"] >= int(plan["policy"]["dense_solves"]):
                raise ValueError("coupled GN dense solve budget exhausted")
            record["dense_solves_started"] += 1
            tangent._write(output, record)
        def solve_complete() -> None:
            record["dense_solves_completed"] += 1
            tangent._write(output, record)
        gn_direction, solve_audit = coupled_woodbury_direction(normal,
            tangent_residual, rows_minus, robust_curvature, chart, pivot,
            on_solve_start=solve_start, on_solve_complete=solve_complete)
        record["dense_solve_audit"] = solve_audit
        tangent._write(output, record)
        projected_minus = rows_minus @ projector
        diagnostics = {"scope": "coupled robust-GN Woodbury tangent search metric; residual curvature omitted",
            "theta_star": theta_star, "pseudo_huber_delta": delta,
            "reconstructed_native_J": float(reconstructed_j),
            "native_J": native_j, "side_objectives": {str(s): side_objectives[s] for s in (-1, 1)},
            "theta_star_pair_error": abs(theta_star - theta_check),
            "theta_star_pair_budget": theta_check_budget,
            "robust_curvature_D": robust_curvature.tolist(),
            "residual_minus": residual_minus.tolist(), "residual_plus": residual_plus.tolist(),
            "residual_pair_max_error": float(residual_error.max()),
            "residual_pair_max_budget": float(residual_budget.max()),
            "residual_pair_construction_scale": residual_scale.tolist(),
            "projected_row_errors": row_error.tolist(), "projected_row_budgets": row_budget.tolist(),
            "ambient_minus_row_norms": ambient_minus.tolist(), "ambient_plus_row_norms": ambient_plus.tolist(),
            "projected_minus_row_norms": projected_norms[0].tolist(),
            "projected_plus_row_norms": projected_norms[1].tolist(),
            "row_vjps_completed": record["jacobian_rows_completed"], **solve_audit}

        def gn_model_builder(gm: Tensor, gp: Tensor, hm: Tensor, hp: Tensor,
                n: Tensor, z: Tensor, carried_theta: float, q: float,
                actual_pivot: int) -> tangent.TangentModel:
            return mixing.minimum_tangent_model(gm, gp, hm, hp, n, z,
                theta_star, q, actual_pivot, float(plan["policy"]["face_scale"]),
                direction_override=gn_direction)

        gn_spec = {"name": "robust_gn_coupled", "direction": gn_direction,
            "direction_override": gn_direction, "working_theta": theta_star,
            "candidate_mixing_minimum": True, "model_builder": gn_model_builder,
            "diagnostics": diagnostics}
        baseline["name"] = "baseline_tangent"
        baseline["diagnostics"] = {**baseline.get("diagnostics", {}),
            "row_parity_validated": True, "theta_star": theta_star,
            "row_vjps_completed": record["jacobian_rows_completed"]}
        return [baseline, gn_spec]
    return build


def _run_child_impl(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child_impl(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_direction_model_factory)


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    return tangent._run_child(plan_path, plan_sha, output,
        plan_loader=_load_plan, base_loader=_load_current_base,
        base_control_sha256=BASE_CONTROL_SHA,
        direction_model_factory=_direction_model_factory,
        initial_theta=BASE_THETA)


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    return tangent.run(plan_path, plan_sha, output, resource, log,
        plan_loader=_load_plan, child_script=Path(__file__),
        base_control_sha256=BASE_CONTROL_SHA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.json")
    parser.add_argument("--resource", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path,
        default=EVIDENCE / "tangent_coupled_gn_comparison_20261010_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
            args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
