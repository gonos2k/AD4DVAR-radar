"""One guarded original-H Newton step using a frozen block-Schur preconditioner."""
from __future__ import annotations

import argparse, hashlib, json, math, platform, re, time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from advar import matrix_free
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as frozen

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
DIAGNOSTIC, PLAN = EVIDENCE / "point_3h_terminal_block_schur_attempt1/block_schur.json", EVIDENCE / "S4_POINT_TERMINAL_SCHUR_NEWTON_STEP_PLAN_20261003.md"
DIAGNOSTIC_SHA = "ca04e5f9712667cee95de69121d889dd8f2d33f81192b70e3724ee6fa52f224b"
TAIL_RAW_SHA = frozen.TAIL_RAW_SHA
PLAN_SHA = "432b988b478069e754c51b62a28b78775a6d017276bb2bc95b4a54bbc4e103d9"
SELF, TEST = "examples/weather_scenarios/fv_point_3h_schur_newton_step.py", "tests/test_fv_point_3h_schur_newton_step.py"
SELF_CANONICAL_SHA = "a4cbe1a42e889359ac2b0e9573b8a20c778ac18d1f69d43b597ae6748845440e"
TEST_SHA = "b33037cfb44e158a2cee5d912b653445a442350f0fbe3f4b34ba36caa68926c5"
_SELF_PIN = re.compile(rb'(?m)^SELF_CANONICAL_SHA = "[0-9a-f]{64}"$')
PCG_RTOL, PCG_MAX, INTERNAL_SECONDS = 1e-10, 104, 300.0
ARMIJO_C1, ALPHAS = 1e-4, tuple(2.0**-i for i in range(16))
EPS = torch.finfo(torch.float64).eps
SOURCE_PATHS = tuple(dict.fromkeys((*frozen.SOURCE_PATHS, SELF, TEST)))

class StepRefusal(ValueError):
    """A known numerical or strict-point gate refused the one-step attempt."""

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_sha(value: Tensor) -> str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def require_pins() -> None:
    source, count = _SELF_PIN.subn(b'SELF_CANONICAL_SHA = "<canonical-self-pin>"', (ROOT / SELF).read_bytes())
    if count != 1 or hashlib.sha256(source).hexdigest() != SELF_CANONICAL_SHA: raise ValueError("S4 Newton-step canonical producer pin changed")
    if sha(ROOT / TEST) != TEST_SHA or sha(PLAN) != PLAN_SHA: raise ValueError("S4 Newton-step test or plan pin changed")

def source_hashes() -> dict[str, str]: return {name: sha(ROOT / name) for name in SOURCE_PATHS}

def current_endpoint_control() -> Tensor:
    if sha(frozen.TAIL_RAW) != TAIL_RAW_SHA: raise ValueError("pinned accepted-tail control source changed")
    return frozen.endpoint_control(json.loads(frozen.TAIL_RAW.read_text()))

def _spd_factor(matrix: Tensor, label: str) -> tuple[Tensor, Tensor, dict[str, float]]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.dtype != torch.float64 or matrix.device.type != "cpu" or not bool(torch.isfinite(matrix).all()): raise StepRefusal(f"{label} must be a finite square CPU FP64 matrix")
    scale = torch.linalg.matrix_norm(matrix)
    symmetry = torch.linalg.matrix_norm(matrix - matrix.T) / scale if bool(scale > 0) else scale
    if not bool(torch.isfinite(scale) & (scale > 0) & torch.isfinite(symmetry)) or float(symmetry) > 1e-10: raise StepRefusal(f"{label} symmetry/scale gate failed")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    floor = 128 * EPS * max(float(scale), torch.finfo(torch.float64).tiny)
    if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[0]) <= floor: raise StepRefusal(f"{label} is not resolved positive definite")
    try:
        factor = torch.linalg.cholesky(symmetric)
    except RuntimeError as error:
        raise StepRefusal(f"{label} Cholesky failed: {type(error).__name__}: {error}") from error
    return symmetric, factor, {"lambda_min": float(eigenvalues[0]), "lambda_max": float(eigenvalues[-1]), "positive_floor": floor}

def block_inverse_preconditioner(hessian: Tensor) -> tuple[Callable[[Tensor], Tensor], dict[str, Any]]:
    """Build M⁻¹ for the 20+6 block factorization; never use M as the H operator."""
    if hessian.shape != (26, 26) or hessian.dtype != torch.float64 or hessian.device.type != "cpu" or not bool(torch.isfinite(hessian).all()): raise StepRefusal("frozen Hessian must be finite CPU FP64 26x26")
    scale = torch.linalg.matrix_norm(hessian)
    symmetry = torch.linalg.matrix_norm(hessian - hessian.T) / scale if bool(scale > 0) else scale
    if not bool(torch.isfinite(scale) & (scale > 0) & torch.isfinite(symmetry)) or float(symmetry) > 1e-8: raise StepRefusal("frozen full Hessian symmetry/scale gate failed")
    h = 0.5 * (hessian + hessian.T)
    h, _, h_audit = _spd_factor(h, "full frozen Hessian")
    a, b, d = h[:20, :20], h[:20, 20:], h[20:, 20:]
    a, la, a_audit = _spd_factor(a, "Hff")
    x = torch.cholesky_solve(b, la)
    xres = torch.linalg.matrix_norm(a @ x - b) / torch.maximum(torch.linalg.matrix_norm(a @ x), torch.linalg.matrix_norm(b)).clamp_min(torch.finfo(torch.float64).tiny)
    if not bool(torch.isfinite(xres)) or float(xres) > 1e-10:
        raise StepRefusal("Hff inverse-Hfd preconditioner solve residual failed")
    schur_raw = d - b.T @ x
    schur, ls, s_audit = _spd_factor(schur_raw, "Schur complement")

    def apply(rhs: Tensor) -> Tensor:
        if (rhs.shape != (26,) or rhs.dtype != torch.float64 or rhs.device.type != "cpu"
                or not bool(torch.isfinite(rhs).all())):
            raise ValueError("preconditioner input must be finite CPU FP64 length 26")
        rf, rd = rhs[:20], rhs[20:]
        af = torch.cholesky_solve(rf[:, None], la)[:, 0]
        zd = torch.cholesky_solve((rd - b.T @ af)[:, None], ls)[:, 0]
        zf = af - x @ zd
        result = torch.cat((zf, zd))
        if not bool(torch.isfinite(result).all()):
            raise StepRefusal("block inverse preconditioner returned nonfinite values")
        return result

    return apply, {"full_symmetry_relative": float(symmetry), "full": h_audit, "hff": a_audit, "hff_inverse_hfd_relative_residual": float(xres), "schur": s_audit, "schur_symmetry_relative": float(torch.linalg.matrix_norm(schur_raw-schur_raw.T)/torch.linalg.matrix_norm(schur_raw))}

def solve_newton_direction(operator: Callable[[Tensor], Tensor], gradient: Tensor, preconditioner: Callable[[Tensor], Tensor], deadline: float | None = None) -> tuple[Tensor, dict[str, Any]]:
    if gradient.shape != (26,) or gradient.dtype != torch.float64 or gradient.device.type != "cpu" or not bool(torch.isfinite(gradient).all()): raise ValueError("Newton gradient must be finite CPU FP64 length 26")
    if float(torch.linalg.vector_norm(gradient)) == 0: raise StepRefusal("zero gradient has no descent Newton direction")

    def checked_operator(value: Tensor) -> Tensor:
        if deadline is not None and time.monotonic() >= deadline: raise StepRefusal("300-second internal budget exhausted during PCG")
        result = operator(value)
        if result.shape != (26,) or result.dtype != torch.float64 or result.device.type != "cpu": raise ValueError("original-H HVP returned an invalid vector contract")
        if not bool(torch.isfinite(result).all()): raise StepRefusal("original-H HVP is nonfinite")
        return result

    try:
        result = matrix_free.pcg(checked_operator, -gradient, preconditioner=preconditioner,
                                 rtol=PCG_RTOL, max_iterations=PCG_MAX)
    except RuntimeError as error:
        known = {"operator must be symmetric positive definite", "preconditioner must be positive definite", "PCG step is not finite", "residual norm is not finite", "true residual norm is not finite"}
        if str(error) not in known: raise
        raise StepRefusal(f"PCG numerical refusal: {error}") from error
    step = result.solution
    residual = checked_operator(step) + gradient
    relative = float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient))
    products = gradient * step
    dot = torch.dot(gradient, step)
    dot_budget = 128 * EPS * torch.sum(products.abs())
    if not result.converged or not bool(torch.isfinite(step).all()) or not math.isfinite(relative) or relative > PCG_RTOL: raise StepRefusal("PCG or fresh original-H true residual gate failed")
    if not bool(torch.isfinite(dot) & torch.isfinite(dot_budget)) or not bool(dot < -dot_budget): raise StepRefusal("Newton direction is not descending beyond component-dot rounding budget")
    return step, {"converged": result.converged, "iterations": result.iterations,
                  "pcg_relative_residual": result.relative_residual, "true_residual": residual.tolist(),
                  "true_relative_residual": relative, "g_dot_s": float(dot),
                  "g_dot_s_rounding_budget": float(dot_budget), "rtol": PCG_RTOL, "max_iterations": PCG_MAX}

def armijo_holds(base_j: float, candidate_j: float, alpha: float, slope: float) -> bool:
    return all(math.isfinite(v) for v in (base_j, candidate_j, alpha, slope)) and alpha > 0 and slope < 0 and candidate_j <= base_j + ARMIJO_C1 * alpha * slope

def evaluate_trial(control: Tensor, parameters: Tensor, step: Tensor, alpha: float, base_j: float, slope: float, objective: Callable[[Tensor, Tensor], Tensor], gradient_fn: Callable[[Tensor, Tensor], Tensor], branch_fn: Callable[[Tensor, Tensor], tuple[dict[str, Any], dict[str, Any]]]) -> tuple[Tensor, dict[str, Any]]:
    candidate = control + alpha * step
    value = objective(candidate, parameters)
    record: dict[str, Any] = {"alpha": alpha, "control": candidate.tolist()}
    if not isinstance(value, Tensor) or value.shape != () or value.dtype != torch.float64 or value.device.type != "cpu": raise ValueError("trial objective must return a CPU FP64 scalar tensor")
    if not bool(torch.isfinite(value)): record.update(status="nonfinite_objective"); return candidate, record
    j = float(value)
    record.update(objective=j, armijo_passed=armijo_holds(base_j, j, alpha, slope))
    if not record["armijo_passed"]: record["status"] = "armijo_rejected"; return candidate, record
    branch, margins = branch_fn(candidate, parameters)
    record.update(branch=branch, margins=margins)
    if branch.get("status") != "passed_strict_branch" or branch.get("euler_stages") != 3600 or branch.get("choice_stage_count") != 3600 or branch.get("face_sign_stage_count") != 3600 or not seed._valid_margins(margins, complete=True): record["status"] = "strict_point_refused"; return candidate, record
    gradient = gradient_fn(candidate, parameters)
    if not isinstance(gradient, Tensor) or gradient.shape != (26,) or gradient.dtype != torch.float64 or gradient.device.type != "cpu": raise ValueError("trial gradient must be CPU FP64 length 26")
    if not bool(torch.isfinite(gradient).all()): record["status"] = "nonfinite_candidate_gradient"; return candidate, record
    phi = torch.dot(gradient, gradient) / 2
    if not bool(torch.isfinite(phi)): record["status"] = "nonfinite_candidate_phi"; return candidate, record
    record.update(status="accepted", gradient=gradient.tolist(), phi=float(phi), gradient_blocks=frozen.gradient_blocks(gradient))
    return candidate, record

def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("S4 Newton-step output must be fresh")
    require_pins()
    started, deadline = time.monotonic(), time.monotonic() + INTERNAL_SECONDS
    report: dict[str, Any] = {"phase": "preflight", "numerical_status": "not_reached",
        "scope": "one original-H Newton step with a frozen block-Schur preconditioner only",
        "full_root_claim": False, "score_computed": False, "adjoint_computed": False,
        "response_computed": False, "optimizer_steps_max": 1, "diagnostic_sha256": DIAGNOSTIC_SHA,
        "plan_sha256": PLAN_SHA, "source_before": source_hashes(), "trials": []}
    save = lambda: _write(output, report)
    if sha(DIAGNOSTIC) != DIAGNOSTIC_SHA or sha(PLAN) != PLAN_SHA or sha(frozen.TAIL_RAW) != TAIL_RAW_SHA:
        raise ValueError("pinned S4 diagnostic or Newton-step plan changed")
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    if (diagnostic.get("phase") != "finished" or diagnostic.get("numerical_status") != "diagnostic_completed"
            or diagnostic.get("hessian_status") != "schur_evaluated"
            or diagnostic.get("input_unchanged") is not True or diagnostic.get("source_unchanged") is not True
            or diagnostic.get("score_computed") is not False or diagnostic.get("response_computed") is not False
            or diagnostic.get("tail_raw_sha256") != TAIL_RAW_SHA):
        raise ValueError("pinned S4 diagnostic is not an eligible frozen-H source")
    runtime = {"device": "CPU FP64", "python": platform.python_version(), "torch": torch.__version__}
    if runtime != diagnostic.get("runtime") or runtime != diagnostic.get("runtime_after"):
        raise ValueError("S4 Newton-step runtime differs from frozen diagnostic")
    sources = diagnostic["source_before"]
    if sources != diagnostic["source_after"] or any(report["source_before"].get(k) != v for k, v in sources.items()):
        raise ValueError("S4 frozen-H dependency source map changed")
    problem, original, _, parameters, truth, _identity = seed._prepare_fixed_seed()
    control = current_endpoint_control()
    input_identity = seed._input_identity(problem, original, control, parameters, truth)
    if (input_identity != diagnostic["input_before"]["identity"]
            or tensor_sha(control) != diagnostic["endpoint_control_sha256"]
            or tensor_sha(parameters) != diagnostic["parameters_sha256"]):
        raise ValueError("fixed point S4 control, parameters, or input identity changed")
    report.update(phase="ready_input", input_before=diagnostic["input_before"], runtime=runtime,
        control=control.tolist(), control_sha256=tensor_sha(control),
        parameters=parameters.tolist(), parameters_sha256=tensor_sha(parameters))
    save()
    try:
        branch, margins = tail._full_current_branch(problem, control, parameters)
        if (branch.get("status") != "passed_strict_branch" or branch.get("signature_sha256") != diagnostic["branch_signature_sha256"]
                or not seed._valid_margins(margins, complete=True)):
            raise StepRefusal("initial control no longer passes its pinned strict 3600-stage point branch")
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = tail._fresh_merit(problem, control, parameters, gradient_fn)
        if fresh is None: raise StepRefusal("fresh original objective or gradient is nonfinite")
        objective, gradient, phi = fresh
        metric_checks = {
            "J": frozen.metric_check(float(diagnostic["objective"]), float(objective)),
            "Phi": frozen.metric_check(float(diagnostic["phi"]), float(phi)),
            "gradient_l2": frozen.metric_check(float(diagnostic["gradient_blocks"]["full_l2"]),
                                                float(torch.linalg.vector_norm(gradient))),
        }
        if not all(row["passed"] for row in metric_checks.values()): raise StepRefusal("fresh S4 objective/gradient norms differ from pinned diagnostic")
        saved_gradient = torch.tensor(diagnostic["gradient"], dtype=torch.float64)
        gradient_budget = 128 * EPS * max(float(gradient.abs().max()), float(saved_gradient.abs().max()), torch.finfo(torch.float64).tiny)
        if saved_gradient.shape != gradient.shape or bool(((gradient-saved_gradient).abs() > gradient_budget).any()):
            raise StepRefusal("fresh original objective gradient differs from pinned S4 diagnostic")
        hessian = torch.tensor(diagnostic["hessian"], dtype=torch.float64)
        if tensor_sha(hessian) != diagnostic.get("hessian_sha256"):
            raise ValueError("frozen Hessian bytes differ from the pinned S4 report")
        preconditioner, preconditioner_audit = block_inverse_preconditioner(hessian)
        report.update(phase="ready", objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
            gradient_blocks=frozen.gradient_blocks(gradient), metric_checks=metric_checks,
            branch=branch, branch_margins=margins, preconditioner=preconditioner_audit,
            frozen_hessian_sha256=diagnostic["hessian_sha256"])
        save()

        def operator(vector: Tensor) -> Tensor:
            if time.monotonic() >= deadline: raise StepRefusal("300-second internal budget exhausted during Newton HVP")
            return torch.func.jvp(lambda c: gradient_fn(c, parameters), (control,), (vector,))[1]

        step, solve = solve_newton_direction(operator, gradient, preconditioner, deadline)
        report.update(step=step.tolist(), solve=solve, phase="line_search")
        save()
        base_j, slope = float(objective), float(torch.dot(gradient, step))
        for alpha in ALPHAS:
            if time.monotonic() >= deadline:
                report["trials"].append({"alpha": alpha, "status": "internal_budget_exhausted"})
                break
            candidate, trial = evaluate_trial(control, parameters, step, alpha, base_j, slope,
                                               problem.objective, gradient_fn,
                                               lambda c, p: tail._full_current_branch(problem, c, p))
            trial["branch_signature_changed"] = (None if "branch" not in trial else
                trial["branch"].get("signature_sha256") != branch.get("signature_sha256"))
            if time.monotonic() >= deadline: trial["status"] = "internal_budget_exhausted_after_trial"
            report["trials"].append(trial)
            if trial["status"] == "internal_budget_exhausted_after_trial": break
            if trial["status"] == "accepted":
                report.update(phase="finished", numerical_status="one_step_accepted",
                    optimizer_steps=1, accepted_alpha=alpha, accepted_control=candidate.tolist(),
                    accepted_control_sha256=tensor_sha(candidate), accepted_objective=trial["objective"],
                    accepted_phi=trial["phi"], accepted_gradient=trial["gradient"],
                    accepted_gradient_blocks=trial["gradient_blocks"],
                    accepted_branch=trial["branch"], accepted_margins=trial["margins"])
                break
        else:
            report.update(phase="finished", numerical_status="armijo_grid_exhausted", optimizer_steps=0)
        if report.get("phase") != "finished":
            report.update(phase="finished", numerical_status="internal_budget_refusal", optimizer_steps=0)
    except StepRefusal as error:
        report.update(phase="finished", numerical_status="step_refused", optimizer_steps=0,
                      refusal=str(error))
    after, runtime_after = source_hashes(), {"device": "CPU FP64", "python": platform.python_version(),
                                               "torch": torch.__version__}
    input_after = seed._input_identity(problem, original, control, parameters, truth)
    if (after != report["source_before"] or runtime_after != runtime or input_after != input_identity
            or sha(DIAGNOSTIC) != DIAGNOSTIC_SHA or sha(PLAN) != PLAN_SHA
            or sha(frozen.TAIL_RAW) != TAIL_RAW_SHA):
        raise ValueError("S4 Newton-step source/input/runtime pin changed during work")
    report.update(source_after=after, source_unchanged=True, input_after=diagnostic["input_before"],
                  input_unchanged=True, runtime_after=runtime_after,
                  elapsed_seconds=time.monotonic() - started)
    save()
    return report

def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
