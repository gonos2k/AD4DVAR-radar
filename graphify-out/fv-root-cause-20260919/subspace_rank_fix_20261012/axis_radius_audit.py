"""Offline KKT and radius audit from the PR301 receipt bundle; no model calls."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PR301 = EVIDENCE / "netflux_subspace_20261012"
WORK = EVIDENCE / "subspace_rank_fix_20261012"
PLAN = PR301 / "PLAN.json"
PLAN_SHA256 = "a0fef5718772032b4567d24ca75fbaf39cff8b58e30f7e4b249dbabb4a5c7d26"
RECEIPT_MANIFEST = PR301 / "RECEIPT_MANIFEST.json"
RECEIPT_MANIFEST_SHA256 = "879b35fd94d9e77036badf3bad47dba8f4885794a3a4841206bc3be3104a960c"
SOURCE_TAR = PR301 / "source.tar.gz"
SOURCE_TAR_SHA256 = "b332268a74f0c5ed9f9e05b3d890288d2e953215b1422253eec2426cb183cab3"
RECEIPTS_TAR = PR301 / "receipts.tar.gz"
RECEIPTS_TAR_SHA256 = "cb05abcc41fd0b3f1a3cb86208caec566eb3fb5f363014094fd3b6c9a3f7bfc0"
RAW_STEP_SHA256 = "0802769b3dd6624c11e67ff9a222594f2ec25c728ea704021903419e64c6a4cb"
START_CONTROL_SHA256 = "d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e"
RADIUS0 = 1.8757177916958825e-5
OUTPUT = WORK / "d19e_axis_radius_audit.json"


@dataclass(frozen=True)
class ResidualGeometry:
    scales: Tensor
    D_normalized: Tensor
    V_normalized: Tensor
    M: Tensor
    Q: Tensor
    c: Tensor
    V_singular_values: Tensor
    rank_tolerance: float
    V_rank: int


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pinned(path: Path, expected: str) -> bytes:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"audit input is not an in-repository regular file: {path}")
    data = path.read_bytes()
    if _sha(data) != expected:
        raise ValueError(f"audit input hash mismatch: {path.name}")
    return data


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _tar_members(data: bytes, expected_names: set[str], label: str) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name in members:
                    raise ValueError(f"{label} contains a non-file or duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"{label} member has no contents")
                members[member.name] = stream.read()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if set(members) != expected_names:
        raise ValueError(f"{label} member set differs from its frozen manifest")
    return members


def load_archived_step() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate bundled source/receipt bytes, without a live plan or endpoint loader."""
    plan = _json_object(_read_pinned(PLAN, PLAN_SHA256), PLAN.name)
    manifest = _json_object(_read_pinned(RECEIPT_MANIFEST, RECEIPT_MANIFEST_SHA256),
        RECEIPT_MANIFEST.name)
    source_files, archive_files = plan.get("source_files"), plan.get("archive_files")
    if (not isinstance(source_files, dict) or len(source_files) != 216
            or not isinstance(archive_files, dict) or len(archive_files) != 287):
        raise ValueError("PR301 source/archive pin counts differ from the frozen plan")
    source_members = _tar_members(_read_pinned(SOURCE_TAR, SOURCE_TAR_SHA256),
        set(source_files), "PR301 source bundle")
    for name, digest in source_files.items():
        if _sha(source_members[name]) != digest:
            raise ValueError(f"PR301 archived source hash mismatch: {name}")
    receipt_files = manifest.get("files")
    if (not isinstance(receipt_files, dict) or len(receipt_files) != 11
            or manifest.get("source_bundle_sha256") != SOURCE_TAR_SHA256
            or manifest.get("bundle_sha256") != RECEIPTS_TAR_SHA256):
        raise ValueError("PR301 receipt manifest or bundle pins changed")
    receipt_members = _tar_members(_read_pinned(RECEIPTS_TAR, RECEIPTS_TAR_SHA256),
        set(receipt_files), "PR301 receipts")
    for name, entry in receipt_files.items():
        if (_sha(receipt_members[name]) != entry.get("sha256")
                or len(receipt_members[name]) != entry.get("bytes")):
            raise ValueError(f"PR301 archived receipt hash/length mismatch: {name}")
    raw_bytes = receipt_members["attempt1/step.json"]
    if _sha(raw_bytes) != RAW_STEP_SHA256:
        raise ValueError("PR301 raw step hash mismatch")
    raw = _json_object(raw_bytes, "attempt1/step.json")
    run = _json_object(receipt_members["attempt1/step.run.json"], "step.run.json")
    resource = _json_object(receipt_members["attempt1/step.resource.json"], "step.resource.json")
    if (run.get("execution_status") != "completed" or run.get("child_sha256") != RAW_STEP_SHA256
            or run.get("resource") != resource or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("sampled_peak_rss_bytes", 2**63) >= 2 * 1024**3
            or resource.get("elapsed_seconds", math.inf) > 660.0
            or raw.get("hvp_started") != 4 or raw.get("hvp_completed") != 4
            or raw.get("candidate_evaluations_started") != 2
            or raw.get("candidate_evaluations_completed") != 2
            or raw.get("candidate_committed") is not True
            or raw.get("selected_direction") != "baseline_tangent"
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("runtime_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("runtime_before") != raw.get("runtime_after")):
        raise ValueError("PR301 child/run/resource closure is incomplete")
    input_before, input_after = raw.get("input_before"), raw.get("input_after")
    fixed_keys = ("archived_input", "parameters_sha256", "terminal_truth_sha256", "original_flow_fractions")
    if (not isinstance(input_before, dict) or not isinstance(input_after, dict)
            or input_before.get("control_sha256") != START_CONTROL_SHA256
            or input_after.get("control_sha256") != raw.get("current_control_sha256")
            or any(input_before.get(key) != input_after.get(key) for key in fixed_keys)):
        raise ValueError("PR301 fixed-input/control lineage is inconsistent")
    operator = raw.get("GN_operator")
    if (not isinstance(operator, dict) or operator.get("jacobian_rows_started") != 24
            or operator.get("jacobian_rows_completed") != 24
            or operator.get("dense_solves_started") != 1 or operator.get("dense_solves_completed") != 1):
        raise ValueError("PR301 GN operator budget did not close")
    for arm in raw.get("arms", []):
        candidate = arm.get("candidate", {})
        receipt_path = candidate.get("candidate_receipt")
        if not isinstance(receipt_path, str):
            raise ValueError("PR301 arm is missing its candidate receipt link")
        relative = Path(receipt_path).relative_to(PR301.relative_to(ROOT)).as_posix()
        payload_bytes = receipt_members[relative]
        payload = _json_object(payload_bytes, relative)
        if (candidate.get("candidate_receipt_sha256") != _sha(payload_bytes)
                or candidate.get("control") != payload.get("control")
                or candidate.get("G") != payload.get("G")
                or candidate.get("DGd") != payload.get("DGd")):
            raise ValueError("PR301 candidate link differs from bundled receipt bytes")
    if (run.get("child_sha256") != RAW_STEP_SHA256 or run.get("resource") != resource
            or raw.get("base_control_sha256") != START_CONTROL_SHA256
            or raw.get("execution_status") != "completed"
            or raw.get("candidate_committed") is not True):
        raise ValueError("PR301 run/step/control lineage is not closed")
    return plan, raw, {
        "plan_sha256": PLAN_SHA256,
        "source_bundle_sha256": SOURCE_TAR_SHA256,
        "receipt_manifest_sha256": RECEIPT_MANIFEST_SHA256,
        "receipt_bundle_sha256": RECEIPTS_TAR_SHA256,
        "raw_step_sha256": RAW_STEP_SHA256,
        "source_member_count": str(len(source_members)),
        "archive_pin_count": str(len(archive_files)),
        "receipt_member_count": str(len(receipt_members)),
    }


def residual_geometry(G: Tensor, D: Tensor, V: Tensor) -> ResidualGeometry:
    """Build normalized residual and control Gram matrices used by the solver."""
    if (G.ndim != 1 or D.ndim != 2 or V.shape != (G.numel(), 2)
            or D.shape[1] != 2 or D.shape[0] == 0
            or G.dtype != torch.float64 or D.dtype != G.dtype or V.dtype != G.dtype
            or G.device.type != "cpu" or D.device != G.device or V.device != G.device
            or not all(bool(torch.isfinite(x).all()) for x in (G, D, V))):
        raise ValueError("KKT audit requires matching finite CPU FP64 G/D/V arrays")
    scales = torch.linalg.vector_norm(D, dim=0)
    if bool((scales <= torch.finfo(G.dtype).tiny).any()):
        raise ValueError("both control directions need a resolved nonzero norm")
    Dn, Vn = D / scales, V / scales
    M, Q = Vn.T @ Vn, Dn.T @ Dn
    c = Vn.T @ G
    singular_values = torch.linalg.svdvals(Vn)
    tol = 64.0 * torch.finfo(G.dtype).eps * max(Vn.shape) * singular_values[0]
    rank = int((singular_values > tol).sum())
    return ResidualGeometry(scales, Dn, Vn, M, Q, c, singular_values, float(tol), rank)


def rank_deficient_lambda_zero(G: Tensor, Dn: Tensor, Vn: Tensor, radius: float,
                               *, rtol: float = 64.0 * torch.finfo(torch.float64).eps) -> dict[str, Any]:
    """Find the minimum-movement nonnegative member of a rank-one lambda=0 set."""
    M, Q, c = Vn.T @ Vn, Dn.T @ Dn, Vn.T @ G
    singular_values = torch.linalg.svdvals(Vn)
    tolerance = rtol * max(Vn.shape) * singular_values[0]
    if int((singular_values > tolerance).sum()) != 1:
        raise ValueError("rank_deficient_lambda_zero expects rank(V)=1")
    beta_p = torch.linalg.pinv(M, rtol=rtol * max(M.shape)) @ (-c)
    _, _, vh = torch.linalg.svd(M)
    null = vh[-1]
    # Feasible t interval for beta=beta_p+t*null and beta>=0.
    lower, upper = -math.inf, math.inf
    for i in range(2):
        if abs(float(null[i])) <= rtol:
            if float(beta_p[i]) < -rtol:
                raise ValueError("rank-one lambda=0 equations have no nonnegative solution")
        elif null[i] > 0:
            lower = max(lower, -float(beta_p[i] / null[i]))
        else:
            upper = min(upper, -float(beta_p[i] / null[i]))
    if lower > upper:
        raise ValueError("rank-one lambda=0 solution misses the nonnegative orthant")
    qn = torch.dot(null, Q @ null)
    if not bool(torch.isfinite(qn)) or float(qn) <= torch.finfo(G.dtype).tiny:
        raise ValueError("rank-one null direction has no resolved control-space metric")
    t_min = -torch.dot(null, Q @ beta_p) / qn
    t = min(max(float(t_min), lower), upper)
    beta = beta_p + t * null
    movement_sq = torch.dot(beta, Q @ beta)
    equation_error = torch.linalg.vector_norm(M @ beta + c)
    residual = G + Vn @ beta
    equation_scale = max(float(torch.linalg.vector_norm(M @ beta)),
        float(torch.linalg.vector_norm(c)), torch.finfo(G.dtype).tiny)
    equation_budget = 256.0 * rtol * equation_scale
    return {"beta": beta, "movement_norm": float(torch.sqrt(torch.clamp_min(movement_sq, 0.0))),
        "predicted_R": float(torch.dot(residual, residual)),
        "equation_error_l2": float(equation_error),
        "equation_budget_l2": equation_budget,
        "feasible_at_lambda_zero": bool(float(movement_sq) <= radius * radius
            and bool((beta >= -rtol).all())
            and float(equation_error) <= equation_budget),
        "radius": radius, "rank_tolerance": float(tolerance)}


def axis0_kkt(M: Tensor, Q: Tensor, c: Tensor, radius: float) -> dict[str, Any]:
    """KKT multipliers on the positive axis-0, radius-active branch."""
    q00 = float(Q[0, 0])
    if not all(math.isfinite(float(x)) for x in (q00, radius)) or q00 <= 0 or radius <= 0:
        raise ValueError("axis-0 KKT audit needs positive finite radius and Q00")
    beta0 = radius / math.sqrt(q00)
    eps, tiny = torch.finfo(M.dtype).eps, torch.finfo(M.dtype).tiny
    m00b, c0 = float(M[0, 0]) * beta0, float(c[0])
    lambda_numerator = -(m00b + c0)
    lambda_denominator = q00 * beta0
    lam = lambda_numerator / lambda_denominator
    q10b_lam = lam * float(Q[1, 0]) * beta0
    mu1_terms = (float(M[1, 0]) * beta0, float(c[1]), q10b_lam)
    mu1 = math.fsum(mu1_terms)
    free_terms = (m00b, c0, lam * q00 * beta0)
    stationarity_0 = math.fsum(free_terms)
    radius_terms = (q00 * beta0 * beta0, radius * radius)
    radius_residual = radius_terms[0] - radius_terms[1]
    lambda_budget = 128.0 * eps * max(abs(m00b), abs(c0), tiny)
    mu1_budget = 128.0 * eps * max(*(abs(x) for x in mu1_terms), tiny)
    stationarity_budget = 128.0 * eps * max(*(abs(x) for x in free_terms), tiny)
    radius_budget = 128.0 * eps * max(*(abs(x) for x in radius_terms), tiny)
    return {"beta0": beta0, "beta1": 0.0, "lambda": lam, "mu1": mu1,
        "lambda_numerator": lambda_numerator, "lambda_sign_budget": lambda_budget,
        "mu1_sign_budget": mu1_budget,
        "stationarity_free_coordinate": stationarity_0,
        "stationarity_budget": stationarity_budget,
        "radius_residual": radius_residual, "radius_budget": radius_budget,
        "lambda_sign_resolved": abs(lambda_numerator) > lambda_budget,
        "mu1_sign_resolved": abs(mu1) > mu1_budget,
        "axis0_kkt": beta0 > 0.0 and lambda_numerator >= -lambda_budget
            and mu1 >= -mu1_budget and abs(stationarity_0) <= stationarity_budget
            and abs(radius_residual) <= radius_budget}


def axis0_activation(M: Tensor, Q: Tensor, c: Tensor) -> dict[str, Any]:
    """Analytic radii where axis-0 loses KKT validity or its radius multiplier vanishes."""
    q00, q10 = float(Q[0, 0]), float(Q[1, 0])
    m00, m10 = float(M[0, 0]), float(M[1, 0])
    c0, c1 = float(c[0]), float(c[1])
    if q00 <= 0.0 or m00 <= 0.0 or c0 >= 0.0:
        raise ValueError("axis-0 activation requires a positive descent axis")
    beta_lambda0 = -c0 / m00
    delta_lambda0_axis = beta_lambda0 * math.sqrt(q00)
    slope_terms = (m10, q10 * m00 / q00)
    intercept_terms = (c1, q10 * c0 / q00)
    slope = slope_terms[0] - slope_terms[1]
    intercept = intercept_terms[0] - intercept_terms[1]
    eps, tiny = torch.finfo(M.dtype).eps, torch.finfo(M.dtype).tiny
    slope_budget = 128.0 * eps * max(*(abs(x) for x in slope_terms), tiny)
    intercept_budget = 128.0 * eps * max(*(abs(x) for x in intercept_terms), tiny)
    slope_resolved = abs(slope) > slope_budget
    intercept_resolved = abs(intercept) > intercept_budget
    downward = slope < -slope_budget
    beta_candidate = -intercept / slope if slope_resolved and downward else None
    positive_candidate = (beta_candidate is not None and beta_candidate > 0.0
        and intercept > intercept_budget)
    delta_candidate = (float(beta_candidate) * math.sqrt(q00)
        if positive_candidate and beta_candidate is not None else None)
    lambda_at_critical = None
    root_mu = None
    lambda_budget_at_critical = None
    if positive_candidate and beta_candidate is not None:
        beta_value = float(beta_candidate)
        lambda_num = -(m00 * beta_value + c0)
        lambda_at_critical = lambda_num / (q00 * beta_value)
        lambda_budget_at_critical = 128.0 * eps * max(abs(m00 * beta_value), abs(c0), tiny)
        root_mu = math.fsum((intercept, slope * beta_value))
    beta_critical_for_root = beta_candidate if beta_candidate is not None else 0.0
    activates_before_lambda_zero = bool(delta_candidate is not None
        and delta_candidate < delta_lambda0_axis
        and lambda_at_critical is not None
        and lambda_at_critical * q00 * beta_critical_for_root > (lambda_budget_at_critical or 0.0))
    reason = ("unresolved_mu1_slope" if not slope_resolved else
        "mu1_slope_not_downward" if not downward else
        "unresolved_mu1_intercept" if not intercept_resolved else
        "no_positive_mu1_root" if not positive_candidate else
        "root_after_axis_lambda_zero" if not activates_before_lambda_zero else "resolved_axis_activation")
    return {"mu1_intercept": intercept, "mu1_slope_per_beta0": slope,
        "mu1_intercept_budget": intercept_budget, "mu1_slope_budget": slope_budget,
        "mu1_intercept_resolved": intercept_resolved, "mu1_slope_resolved": slope_resolved,
        "mu1_slope_downward": downward, "critical_radius_status": reason,
        "beta0_at_axis_lambda_zero": beta_lambda0,
        "axis_lambda_zero_radius": delta_lambda0_axis,
        "beta0_at_mu1_zero": beta_candidate if activates_before_lambda_zero else None,
        "unconstrained_positive_mu1_root_radius": delta_candidate,
        "critical_axis_activation_radius": delta_candidate if activates_before_lambda_zero else None,
        "lambda_at_mu1_zero": lambda_at_critical,
        "lambda_sign_budget_at_mu1_zero": lambda_budget_at_critical,
        "mu1_at_critical_radius": root_mu,
        "inactive_axis_activates_before_axis_lambda_zero": activates_before_lambda_zero}


def _axis0_mu1(M: Tensor, Q: Tensor, c: Tensor, radius: float) -> float:
    beta0 = radius / math.sqrt(float(Q[0, 0]))
    lam = -(float(M[0, 0]) * beta0 + float(c[0])) / (float(Q[0, 0]) * beta0)
    return float(M[1, 0]) * beta0 + float(c[1]) + lam * float(Q[1, 0]) * beta0


def _vector_list(value: Tensor) -> list[float]:
    return [float(x) for x in value.tolist()]


def build_audit() -> dict[str, Any]:
    plan, step, pins = load_archived_step()
    closed = step["closed_steps"][0]
    if closed.get("base_control_sha256") != START_CONTROL_SHA256:
        raise ValueError("archived fresh models are not anchored at d19e")
    G = torch.tensor(closed["full_G"], dtype=torch.float64)
    models = step.get("fresh_models")
    if not isinstance(models, list) or [m.get("name") for m in models] != ["baseline_tangent", "robust_gn_coupled"]:
        raise ValueError("archived d19e fresh-model order changed")
    D = torch.stack([torch.tensor(m["direction"], dtype=G.dtype) for m in models], dim=1)
    V = torch.stack([torch.tensor(m["residual_direction"], dtype=G.dtype) for m in models], dim=1)
    geom = residual_geometry(G, D, V)
    scales, Dn, Vn = geom.scales, geom.D_normalized, geom.V_normalized
    M, Q, c = geom.M, geom.Q, geom.c
    actual = axis0_kkt(M, Q, c, RADIUS0)
    activation = axis0_activation(M, Q, c)
    activation_radius = activation["critical_axis_activation_radius"]
    activation_radius_value = (float(activation_radius)
        if isinstance(activation_radius, (int, float)) else None)
    matrix_condition = float(torch.linalg.cond(M)) if geom.V_rank == 2 else None
    critical_radius_bracket = None
    critical_mu_bracket = None
    if activation_radius_value is not None and matrix_condition is not None:
        root_tolerance = (256.0 * torch.finfo(G.dtype).eps * matrix_condition
            * activation_radius_value)
        critical_radius_bracket = [max(0.0, activation_radius_value - root_tolerance),
            activation_radius_value + root_tolerance]
        critical_mu_bracket = [_axis0_mu1(M, Q, c, r) for r in critical_radius_bracket]
    if activation_radius_value is not None:
        axis0_active_interval = [0.0, activation_radius_value]
    else:
        axis_lambda_zero_radius = activation["axis_lambda_zero_radius"]
        axis0_active_interval = [0.0, float(axis_lambda_zero_radius)]
    full_rank = geom.V_rank == 2
    beta_unconstrained = None
    if full_rank:
        orthogonal, triangular = torch.linalg.qr(Vn, mode="reduced")
        beta_unconstrained = torch.linalg.solve_triangular(triangular,
            -(orthogonal.T @ G).reshape(2, 1), upper=True).reshape(2)
    full_l0_radius = (float(torch.sqrt(torch.dot(beta_unconstrained,
        Q @ beta_unconstrained))) if beta_unconstrained is not None
        and bool((beta_unconstrained >= 0.0).all()) else None)

    from examples.weather_scenarios.fv_point_3h_subspace_model import solve_two_direction_subspace

    scale_at_activation = (activation_radius_value / RADIUS0
        if activation_radius_value is not None else math.nan)
    sweep_factors = sorted(set((0.75, 1.0, 1.2, 1.3, 1.34,
        max(1.35, scale_at_activation * 0.99), scale_at_activation,
        scale_at_activation * 1.01, 1.4, 1.5, 2.0)))
    sweep: list[dict[str, Any]] = []
    for factor in sweep_factors:
        radius = RADIUS0 * factor
        solution = solve_two_direction_subspace(G, D, V, radius)
        beta = solution.coefficients * scales
        base_R = float(torch.dot(G, G))
        axis_radius = radius / math.sqrt(float(Q[0, 0]))
        beta_axis0 = min(max(-float(c[0]) / float(M[0, 0]), 0.0), axis_radius)
        residual_axis = G + Vn[:, 0] * beta_axis0
        axis_R = float(torch.dot(residual_axis, residual_axis))
        sweep.append({"radius_factor": factor, "radius": radius,
            "coefficients_raw": _vector_list(solution.coefficients),
            "coefficients_normalized": _vector_list(beta),
            "selected_active_set": solution.diagnostics["selected_active_set"],
            "full_kkt_status": solution.diagnostics["full_kkt_status"],
            "movement_norm": float(solution.diagnostics["movement_norm"]),
            "predicted_R": float(solution.predicted_R),
            "gradient_axis_R": axis_R,
            "model_gain_vs_gradient_axis": axis_R - float(solution.predicted_R),
            "model_gain_relative_to_axis_R": (axis_R - float(solution.predicted_R)) / axis_R
                if axis_R > 0.0 else None,
            "second_coefficient_positive": float(beta[1]) > 0.0})

    scale_tol = geom.rank_tolerance
    rank_def_fixture = {
        "G": [-1.0, 0.0, 0.0],
        "D_normalized": [[1.0, 0.0], [0.0, 1.0]],
        "V_normalized": [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        "purpose": "rank-one residual columns with a feasible lambda=0 nonnegative minimizer",
    }
    rank_def = rank_deficient_lambda_zero(
        torch.tensor(rank_def_fixture["G"], dtype=G.dtype),
        torch.tensor(rank_def_fixture["D_normalized"], dtype=G.dtype),
        torch.tensor(rank_def_fixture["V_normalized"], dtype=G.dtype), radius=0.8)
    gradient_candidate = step["arms"][0]["candidate"]
    actual_fidelity = gradient_candidate["fidelity"]
    actual_branch_match = gradient_candidate.get("branch_signature_matches_base") is True
    current_solver_path = Path(__import__("examples.weather_scenarios.fv_point_3h_subspace_model",
        fromlist=["solve_two_direction_subspace"]).__file__).resolve()
    solver_sha = _sha(current_solver_path.read_bytes())
    d_crit = activation_radius_value
    candidate_1_5 = next(row for row in sweep if math.isclose(row["radius_factor"], 1.5))
    predicted_gain_budget = 128.0 * torch.finfo(G.dtype).eps * max(
        abs(candidate_1_5["predicted_R"]), abs(candidate_1_5["gradient_axis_R"]),
        torch.finfo(G.dtype).tiny)
    second_coefficient_budget = 128.0 * torch.finfo(G.dtype).eps * max(
        abs(candidate_1_5["coefficients_normalized"][0]), torch.finfo(G.dtype).tiny)
    resolved_1_5_gain = (candidate_1_5["model_gain_vs_gradient_axis"] > predicted_gain_budget
        and candidate_1_5["coefficients_normalized"][1] > second_coefficient_budget)
    warranted = bool(full_rank and actual["axis0_kkt"]
        and d_crit is not None and d_crit > RADIUS0
        and 1.05 * RADIUS0 < d_crit < 2.0 * RADIUS0
        and float(actual_fidelity["eta_vec"]) < 0.5
        and float(actual_fidelity["rho_R"]) >= 0.25 and actual_branch_match
        and resolved_1_5_gain)
    return {
        "scope": "receipt-only d19e local KKT/radius analysis; no FV/HVP/score calls",
        "plan_sha256": pins["plan_sha256"],
        "source_bundle_sha256": pins["source_bundle_sha256"],
        "receipt_manifest_sha256": pins["receipt_manifest_sha256"],
        "receipt_bundle_sha256": pins["receipt_bundle_sha256"],
        "raw_step_sha256": pins["raw_step_sha256"],
        "source_member_count": 216, "archive_pin_count": 287, "receipt_member_count": 11,
        "current_solver_sha256": solver_sha,
        "base_control_sha256": START_CONTROL_SHA256,
        "G_norm": float(torch.linalg.vector_norm(G)),
        "current_radius": RADIUS0,
        "normalization_scales": _vector_list(scales),
        "D_singular_values": _vector_list(torch.linalg.svdvals(Dn)),
        "V_singular_values": _vector_list(geom.V_singular_values),
        "V_rank_tolerance": scale_tol,
        "V_rank": geom.V_rank,
        "V_condition_number": float(geom.V_singular_values[0] / geom.V_singular_values[-1]),
        "M": M.tolist(), "Q": Q.tolist(), "c": _vector_list(c),
        "axis0_current_radius_KKT": actual,
        "axis0_activation": activation,
        "axis0_gradient_active_interval": axis0_active_interval,
        "critical_radius_bracket_fp64": critical_radius_bracket,
        "inactive_multiplier_at_bracket": critical_mu_bracket,
        "M_condition_number": matrix_condition,
        "stored_actual_candidate_evidence": {"radius": gradient_candidate["actual_path_norm"],
            "branch_signature_matches_base": actual_branch_match,
            "eta_vec": actual_fidelity["eta_vec"], "rho_R": actual_fidelity["rho_R"],
            "actual_R": gradient_candidate["F_squared"],
            "predicted_R": actual_fidelity["predicted_R"]},
        "full_positive_lambda0_solution": ({"beta": _vector_list(beta_unconstrained),
            "movement_radius": full_l0_radius} if beta_unconstrained is not None else None),
        "radius_sweep": sweep,
        "bounded_expansion_screen": {"candidate_radius_factor": 1.5,
            "candidate_radius": RADIUS0 * 1.5,
            "predicted_gain_vs_gradient_axis": candidate_1_5["model_gain_vs_gradient_axis"],
            "predicted_gain_roundoff_budget": predicted_gain_budget,
            "second_coefficient": candidate_1_5["coefficients_normalized"][1],
            "second_coefficient_roundoff_budget": second_coefficient_budget,
            "positive_gain_resolved": resolved_1_5_gain},
        "rank_deficient_lambda0_fixture": {**rank_def_fixture,
            "result": {k: (_vector_list(v) if isinstance(v, Tensor) else v)
                for k, v in rank_def.items()}},
        "limited_new_point_experiment_worthwhile": warranted,
        "recommendation": ("A single bounded fresh-point test near 1.5x the starting radius is mathematically worth considering: the cached d19e local KKT model predicts activation of the second direction at the reported critical radius, and the original-radius actual vector model was accurate. Requalify the next base and freshly rebuild both directions before any such test; the stored d19e result does not authorize radius expansion at a later point." if warranted else
            "The cached local KKT conditions do not justify an expanded-radius experiment."),
        "limits": ["KKT sweep is for the frozen d19e linear residual model only.",
            "No finite candidate, chart retraction, branch change, or J/R outcome was evaluated above the stored radius.",
            "The rank-one fixture tests lambda=0 feasibility separately; d19e itself is full rank in V."],
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    result = build_audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
