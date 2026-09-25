"""Guard and independently classify the constructed point FV reanalysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from advar.matrix_free import vjp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_centered_prior_response_probe as probe
from examples.weather_scenarios.fv_point_centered_prior_case import (
    CenteredBranchRefusal, make_case, tensor_sha,
)


WALL_SECONDS = 600
SAMPLED_RSS_BYTES = 1024**3


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _branch(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("euler_stages") == 54
        and isinstance(value.get("signature_sha256"), str)
        and len(value["signature_sha256"]) == 64
        and all(ch in "0123456789abcdef" for ch in value["signature_sha256"])
        and all(_finite(value.get(k)) and value[k] > 1e-4 for k in (
            "minimum_scaled_slope_margin", "minimum_scaled_face_flux_margin"))
    )


def _vector(value: object, length: int) -> torch.Tensor | None:
    if (not isinstance(value, list) or len(value) != length
            or not all(_finite(item) for item in value)):
        return None
    return torch.tensor(value, dtype=torch.float64)


def _close(reported: object, actual: float) -> bool:
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        return False
    return math.isfinite(reported) and math.isclose(reported, actual, rel_tol=1e-11, abs_tol=1e-12)


def _same_branch(reported: object, actual: dict[str, Any]) -> bool:
    return (isinstance(reported, dict)
            and reported.get("signature_sha256") == actual["signature_sha256"]
            and reported.get("euler_stages") == actual["euler_stages"]
            and _close(reported.get("minimum_scaled_slope_margin"),
                       actual["minimum_scaled_slope_margin"])
            and _close(reported.get("minimum_scaled_face_flux_margin"),
                       actual["minimum_scaled_face_flux_margin"]))


def _valid_result(child: dict[str, Any], case: Any, *, require_pass: bool = True) -> bool:
    nominal = child.get("nominal")
    response = child.get("response")
    tangent = child.get("tangent")
    endpoints = child.get("endpoints")
    pairs = child.get("pairs")
    if not isinstance(nominal, dict) or not isinstance(response, dict) or not isinstance(tangent, dict):
        return False
    if not isinstance(endpoints, list) or len(endpoints) != 4 or not isinstance(pairs, list) or len(pairs) != 2:
        return False
    c = _vector(nominal.get("control"), 26)
    p = _vector(nominal.get("parameters"), 13)
    d = _vector(nominal.get("direction"), 13)
    mu = _vector(nominal.get("dynamics_prior_mean"), 6)
    cdot = _vector(tangent.get("control_direction"), 26)
    direct = _vector(response.get("direct_gradient"), 13)
    indirect = _vector(response.get("indirect_gradient"), 13)
    total = _vector(response.get("total_gradient"), 13)
    adjoint = _vector(response.get("adjoint"), 26)
    if any(item is None for item in (c, p, d, mu, cdot, direct, indirect, total, adjoint)):
        return False
    assert c is not None and p is not None and d is not None and mu is not None
    assert cdot is not None and direct is not None and indirect is not None and total is not None
    assert adjoint is not None
    gradient_fn = torch.func.grad(case.objective, argnums=0)
    score_control, actual_direct = torch.func.grad(case.score, argnums=(0, 1))(c, p)
    _, transpose_product = vjp(lambda value: gradient_fn(value, p), c, adjoint)
    score_norm = float(torch.linalg.vector_norm(score_control))
    adjoint_residual = float(torch.linalg.vector_norm(transpose_product - score_control))
    adjoint_relative = adjoint_residual / score_norm if score_norm else adjoint_residual
    _, parameter_product = vjp(lambda value: gradient_fn(c, value), p, adjoint)
    actual_indirect = -parameter_product
    actual_cross = torch.func.jvp(lambda value: gradient_fn(c, value), (p,), (d,))[1]
    actual_direct_scalar = float(torch.dot(actual_direct, d))
    actual_indirect_scalar = -float(torch.dot(adjoint, actual_cross))
    tangent_rhs = -actual_cross
    tangent_product = torch.func.jvp(lambda value: gradient_fn(value, p), (c,), (cdot,))[1]
    tangent_norm = float(torch.linalg.vector_norm(tangent_rhs))
    tangent_residual = float(torch.linalg.vector_norm(tangent_product - tangent_rhs))
    tangent_relative = tangent_residual / tangent_norm if tangent_norm else tangent_residual
    columns = [torch.func.jvp(lambda value: gradient_fn(value, p), (c,), (basis,))[1]
               for basis in torch.eye(c.numel(), dtype=c.dtype)]
    hessian = torch.stack(columns, dim=1)
    if not bool(torch.isfinite(hessian).all()):
        return False
    hessian_scale = float(torch.linalg.matrix_norm(hessian))
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T)) / max(
        hessian_scale, torch.finfo(c.dtype).tiny)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    eigen_min, eigen_max = float(eigenvalues[0]), float(eigenvalues[-1])
    nominal_objective = float(case.objective(c, p))
    nominal_score = float(case.score(c, p))
    nominal_gradient_max = float(gradient_fn(c, p).abs().max())
    try:
        nominal_branch, _ = case.branch_check(c, p)
    except CenteredBranchRefusal:
        return False
    nominal_branch_summary = probe._branch_summary(nominal_branch)
    audit = nominal.get("hessian_audit")
    if not (
        child.get("phase") == "finished"
        and nominal.get("control_sha256") == tensor_sha(c) == tensor_sha(case.control)
        and nominal.get("parameters_sha256") == tensor_sha(p) == tensor_sha(case.parameters)
        and nominal.get("direction_sha256") == tensor_sha(d) == tensor_sha(case.direction)
        and nominal.get("dynamics_prior_mean_sha256") == tensor_sha(mu) == tensor_sha(case.dynamics_prior_mean)
        and _close(nominal.get("objective"), nominal_objective)
        and _close(nominal.get("score"), nominal_score)
        and _close(nominal.get("gradient_max"), nominal_gradient_max)
        and 0 <= nominal_gradient_max < 1e-10
        and _branch(nominal.get("branch"))
        and _same_branch(nominal["branch"], nominal_branch_summary)
        and nominal_branch_summary["signature_sha256"] == probe._branch_summary(case.nominal_branch)["signature_sha256"]
        and isinstance(audit, dict)
        and audit.get("hvp_columns") == 26
        and _finite(symmetry) and symmetry <= 1e-10
        and _finite(eigen_min) and eigen_min > 0
        and _finite(eigen_max) and eigen_max >= eigen_min
        and _close(audit.get("symmetry_relative"), symmetry)
        and _close(audit.get("lambda_min"), eigen_min)
        and _close(audit.get("lambda_max"), eigen_max)
        and _finite(adjoint_relative) and 0 <= adjoint_relative <= 1e-10
        and _close(response.get("true_adjoint_relative_residual"), adjoint_relative)
        and _finite(response.get("pcg_relative_residual"))
        and 0 <= response["pcg_relative_residual"] <= 1e-10
        and type(response.get("pcg_iterations")) is int
        and 0 <= response["pcg_iterations"] <= 104
        and _finite(response.get("total")) and _finite(response.get("direct"))
        and _finite(response.get("indirect"))
        and torch.allclose(direct, actual_direct, rtol=1e-10, atol=1e-12)
        and torch.allclose(indirect, actual_indirect, rtol=1e-10, atol=1e-12)
        and _close(response["direct"], float(torch.dot(direct, d)))
        and _close(response["indirect"], float(torch.dot(indirect, d)))
        and _close(response["total"], float(torch.dot(total, d)))
        and _close(response["direct"], actual_direct_scalar)
        and _close(response["indirect"], actual_indirect_scalar)
        and abs(response["direct"] + response["indirect"] - response["total"]) <= 1e-12
        and torch.allclose(direct + indirect, total, rtol=1e-10, atol=1e-12)
        and _finite(response.get("total_projection"))
        and abs(float(torch.dot(total, d)) - response["total_projection"]) <= 1e-12
        and _finite(response.get("projection_absolute_error"))
        and _finite(response.get("projection_limit"))
        and math.isclose(response["projection_limit"],
                         max(1e-6 * abs(response["total"]), 1e-12),
                         rel_tol=1e-12, abs_tol=1e-15)
        and math.isclose(response["projection_absolute_error"],
                         abs(response["total_projection"] - response["total"]),
                         rel_tol=1e-12, abs_tol=1e-15)
        and response["projection_absolute_error"] <= max(1e-6 * abs(response["total"]), 1e-12)
        and _finite(tangent_relative) and 0 <= tangent_relative <= 1e-10
        and _close(tangent.get("true_relative_residual"), tangent_relative)
        and tangent.get("control_direction_sha256") == tensor_sha(cdot)
        and tangent.get("rhs_sha256") == tensor_sha(tangent_rhs)
        and type(tangent.get("iterations")) is int
        and 0 <= tangent["iterations"] <= 104
    ):
        return False
    errors = []
    pair_passes = []
    for index, h in enumerate(probe.STEP_SIZES):
        scores = {}
        for name, sign in (("plus", 1.0), ("minus", -1.0)):
            endpoint = endpoints[2 * index + (0 if sign > 0 else 1)]
            if not isinstance(endpoint, dict):
                return False
            endpoint_c = _vector(endpoint.get("control"), 26)
            if endpoint_c is None:
                return False
            endpoint_p = p + sign * h * d
            objective = float(case.objective(endpoint_c, endpoint_p))
            score = float(case.score(endpoint_c, endpoint_p))
            gradient_max = float(gradient_fn(endpoint_c, endpoint_p).abs().max())
            try:
                branch, _ = case.branch_check(endpoint_c, endpoint_p)
            except CenteredBranchRefusal:
                return False
            actual_branch = probe._branch_summary(branch)
            if not (
                endpoint.get("h") == h and endpoint.get("sign") == name
                and endpoint.get("status") == "eligible"
                and endpoint.get("parameters_sha256") == tensor_sha(endpoint_p)
                and endpoint.get("predictor_sha256") == tensor_sha(c + sign * h * cdot)
                and endpoint.get("control_sha256") == tensor_sha(endpoint_c)
                and _close(endpoint.get("objective"), objective)
                and _close(endpoint.get("score"), score)
                and _close(endpoint.get("gradient_max"), gradient_max)
                and 0 <= gradient_max < 1e-10
                and _branch(endpoint.get("branch"))
                and _same_branch(endpoint["branch"], actual_branch)
                and actual_branch["signature_sha256"] == nominal_branch_summary["signature_sha256"]
            ):
                return False
            scores[name] = score
        pair = pairs[index]
        central = (scores["plus"] - scores["minus"]) / (2 * h)
        error = abs(central - response["total"])
        limit = 1e-4 * abs(response["total"]) if abs(response["total"]) >= 1e-8 else 1e-10
        if not isinstance(pair, dict) or not (
            pair.get("h") == h and pair.get("passed") is (error < limit)
            and _finite(pair.get("central_difference"))
            and abs(pair["central_difference"] - central) <= 1e-12
            and _finite(pair.get("absolute_error"))
            and abs(pair["absolute_error"] - error) <= 1e-12
        ):
            return False
        errors.append(error)
        pair_passes.append(error < limit)
    validated = all(pair_passes) and errors[1] < errors[0]
    return validated if require_pass else not validated


def run(directory: Path) -> dict[str, object]:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("centered point output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "point_centered_response.json"
    command = [sys.executable, "-m", "examples.weather_scenarios.fv_point_centered_prior_response_probe",
               "--output", str(output)]
    source_before = probe._sources()
    plan_before, archive_before = probe._sha(probe.PLAN), probe._sha(probe.ARCHIVE)
    case = make_case()
    input_before = case.identity
    resource = run_guarded(command, wall_seconds=WALL_SECONDS,
                           rss_bytes=SAMPLED_RSS_BYTES,
                           report_path=directory / "point_centered_response.resource.json",
                           log_path=directory / "point_centered_response.log")
    child = None
    read_error = None
    try:
        child = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        read_error = f"{type(error).__name__}: {error}"
    resource_ok = (
        resource.get("exit_code") in (0, 2)
        and resource.get("resource_termination") is None
        and resource.get("monitor_error") is None
        and resource.get("command") == command
        and resource.get("wall_limit_seconds") == WALL_SECONDS
        and _finite(resource.get("elapsed_seconds"))
        and 0 <= resource["elapsed_seconds"] <= WALL_SECONDS
        and resource.get("rss_limit_bytes") == SAMPLED_RSS_BYTES
        and type(resource.get("rss_samples")) is int and resource["rss_samples"] > 0
        and type(resource.get("sampled_peak_rss_bytes")) is int
        and 0 < resource["sampled_peak_rss_bytes"] <= SAMPLED_RSS_BYTES
    )
    identity_ok = (
        isinstance(child, dict)
        and child.get("source_unchanged") is True
        and child.get("input_unchanged") is True
        and child.get("plan_unchanged") is True
        and child.get("archive_unchanged") is True
        and child.get("source_before") == child.get("source_after") == source_before == probe._sources()
        and child.get("input_before") == child.get("input_after") == input_before == make_case().identity
        and child.get("plan_sha256") == plan_before == probe.PLAN_SHA256 == probe._sha(probe.PLAN)
        and child.get("archive_sha256") == archive_before == probe._sha(probe.ARCHIVE)
        and type(child.get("pid")) is int and child["pid"] == resource.get("child_pid")
    )
    numerical_status = child.get("numerical_status") if isinstance(child, dict) else "not_reached"
    valid_result = bool(isinstance(child, dict) and numerical_status == "eligible"
                        and child.get("response_validation") == "passed"
                        and resource.get("exit_code") == 0
                        and _valid_result(child, case))
    valid_validation_failure = bool(
        isinstance(child, dict) and numerical_status == "validation_failed"
        and child.get("response_validation") == "failed"
        and resource.get("exit_code") == 2
        and _valid_result(child, case, require_pass=False)
    )
    valid_refusal = bool(isinstance(child, dict)
                         and numerical_status == "endpoint_refused"
                         and child.get("response_validation") == "not_performed"
                         and resource.get("exit_code") == 2
                         and any(isinstance(item, dict) and item.get("status") == "refused"
                                 and isinstance(item.get("refusal"), str)
                                 for item in child.get("endpoints", [])))
    result: dict[str, object] = {
        "execution_status": "completed" if resource_ok and identity_ok and (valid_result or valid_validation_failure or valid_refusal) else "failed",
        "numerical_status": numerical_status,
        "response_validation": child.get("response_validation") if isinstance(child, dict) else "not_performed",
        "child_read_error": read_error,
        "resource": resource,
        "scope": "constructed centered-prior correlated point case; no original-input convergence or physical skill claim",
    }
    (directory / "point_centered_response.run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
