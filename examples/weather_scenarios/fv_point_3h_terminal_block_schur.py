"""Read-only gradient/Hessian block diagnostics at the pinned S4 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_merit_continuation as tail_probe
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed_probe

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
TAIL_DIR = EVIDENCE / "point_3h_merit_tail_attempt1"
TAIL_RAW = TAIL_DIR / "point_3h_merit_tail.json"
TAIL_EVIDENCE = EVIDENCE / "FV_POINT_3H_MERIT_TAIL_EVIDENCE.json"
TAIL_PLAN = EVIDENCE / "FV_POINT_3H_MERIT_TAIL_PLAN.md"
PLAN = EVIDENCE / "S4_POINT_TERMINAL_BLOCK_SCHUR_PLAN_20261003.md"
TAIL_RAW_SHA = "c961be69fa8da93a0e4a65991136440043ff7fcc46a32f6dc068bf0daaeb6264"
TAIL_EVIDENCE_SHA = "18634b105b0901c1311d9fd8a225f44b39869cb7c209bdb1d15b23b70114d6bd"
TAIL_PLAN_SHA = "8663ccc4046ca0ddde489b57a51127259747b335978776556b7e9a02f8f679ff"
PLAN_SHA = "4c53cb940816569a3e1bf7beb982714fd4cd993200b48edd6549552d1fbb8600"
CONTROL_SHA = "602b08828b92508dfaed3605f4626882a1cf70646730da1cb866416dd15cf0b9"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
PROBLEM_SHA = "16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3"
TRUTH_SHA = "b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610"
BRANCH_SHA = "18a36b01bc9a8b7c21ea1970f79f8bd379a52ceb67b7fae72b2e79c1c5d0eed8"
RUNTIME = {"device": "CPU FP64", "python": "3.12.13", "torch": "2.13.0"}
NF, ND, NC, NSTAGE = 20, 6, 26, 3600
INTERNAL_SECONDS, HFF_COND_MAX = 540.0, 1.0e8
SYMMETRY_TOL, EIGENPAIR_TOL, SOLVE_TOL = 1.0e-8, 1.0e-8, 1.0e-10
CURVATURE_REL_TOL, ROUNDING_MULT = 1.0e-8, 128.0
SELF = "examples/weather_scenarios/fv_point_3h_terminal_block_schur.py"
TEST = "tests/test_fv_point_3h_terminal_block_schur.py"
SELF_CANONICAL_SHA = "68d74fdf110d0191c2d8fa19470d6ed1b3a444eb5cc8920d9c2acd8b150fb74b"
TEST_SHA = "6dd911242d825d48c050d92f45c97d93ff8c48d593bf459df9e296929e70a491"
_SELF_PIN = re.compile(rb'(?m)^SELF_CANONICAL_SHA = "[0-9a-f]{64}"$')
SOURCE_PATHS = tuple(dict.fromkeys((*tail_probe.SOURCE_PATHS, SELF, TEST)))

class DiagnosticRefusal(ValueError):
    """A numerical block/Schur qualification refusal."""

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_sha(value: Tensor) -> str: return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def runtime_identity() -> dict[str, str]: return {"device": "CPU FP64", "python": platform.python_version(), "torch": torch.__version__}


def require_source_pins() -> None:
    source, count = _SELF_PIN.subn(b'SELF_CANONICAL_SHA = "<canonical-self-pin>"', (ROOT / SELF).read_bytes())
    if count != 1 or hashlib.sha256(source).hexdigest() != SELF_CANONICAL_SHA: raise ValueError("S4 canonical source pin changed")
    if sha(ROOT / TEST) != TEST_SHA or sha(PLAN) != PLAN_SHA:
        raise ValueError("S4 test/plan pin changed")

def source_hashes() -> dict[str, str]: return {name: sha(ROOT / name) for name in SOURCE_PATHS}

def metric_check(saved: float, fresh: float) -> dict[str, float | bool]:
    difference = abs(saved - fresh); budget = 128 * torch.finfo(torch.float64).eps * max(abs(saved), abs(fresh), torch.finfo(torch.float64).tiny)
    return {"saved": saved, "fresh": fresh, "difference": difference, "budget": budget,
            "passed": math.isfinite(difference) and difference <= budget}

def endpoint_control(tail: dict[str, Any]) -> Tensor:
    final = tail.get("accepted_candidate")
    if not (isinstance(final, dict) and final.get("control_sha256") == CONTROL_SHA
            and final.get("branch_signature_sha256") == BRANCH_SHA
            and tail.get("execution_status") == "completed"
            and tail.get("numerical_status") == "accepted_epoch_limit"
            and tail.get("response_computed") is False and tail.get("score_computed") is False):
        raise ValueError("tail does not contain the declared nonresponse endpoint")
    control = torch.tensor(final["control"], dtype=torch.float64)
    if control.shape != (NC,) or tensor_sha(control) != CONTROL_SHA or not bool(torch.isfinite(control).all()):
        raise ValueError("tail endpoint control/hash is malformed")
    return control

def gradient_blocks(gradient: Tensor) -> dict[str, Any]:
    if (gradient.shape != (NC,) or gradient.dtype != torch.float64
            or gradient.device.type != "cpu" or not bool(torch.isfinite(gradient).all())):
        raise ValueError("gradient must be finite CPU FP64 length 26")
    def norms(values: Tensor) -> dict[str, float]:
        return {"l2": float(values.norm()), "rms": float(values.norm() / math.sqrt(values.numel())), "inf": float(values.abs().max())}
    return {"full_l2": float(gradient.norm()), "full_inf": float(gradient.abs().max()),
            "field": norms(gradient[:NF]), "dynamics": norms(gradient[NF:]),
            "flow": norms(gradient[NF:NF + 5]),
            "growth": {"value": float(gradient[-1]), **norms(gradient[-1:])}}

def _relative_residual(residual: Tensor, lhs: Tensor, rhs: Tensor) -> float:
    return float(torch.linalg.matrix_norm(residual) / torch.maximum(
        torch.maximum(torch.linalg.matrix_norm(lhs), torch.linalg.matrix_norm(rhs)),
        torch.tensor(torch.finfo(torch.float64).tiny, dtype=torch.float64)))

def eigenpair_audit(product: Tensor, eigenvalue: Tensor, vector: Tensor, scale: Tensor) -> dict[str, Any]:
    residual = product - eigenvalue * vector
    residual_l2 = float(residual.norm()) if bool(torch.isfinite(residual).all()) else None
    ratio = residual_l2 / float(scale) if residual_l2 is not None and bool(torch.isfinite(scale)) and float(scale) > 0 else None
    relative = ratio if ratio is not None and math.isfinite(ratio) else None
    passed = relative is not None and math.isfinite(relative) and relative <= EIGENPAIR_TOL
    return {"passed": passed, "relative_residual": relative,
            "tolerance": EIGENPAIR_TOL, "residual_l2": residual_l2}


def schur_diagnostic(hessian: Tensor, gradient: Tensor) -> dict[str, Any]:
    """Report f=20,d=6 blocks; never solves a control step."""
    if (hessian.shape != (NC, NC) or hessian.dtype != torch.float64
            or hessian.device.type != "cpu" or not bool(torch.isfinite(hessian).all())):
        raise DiagnosticRefusal("Hessian must be finite CPU FP64 26x26")
    if gradient.shape != (NC,) or not bool(torch.isfinite(gradient).all()):
        raise DiagnosticRefusal("gradient must be finite length 26")
    norm = torch.linalg.matrix_norm(hessian)
    scale = float(norm)
    if not math.isfinite(scale) or scale <= 0:
        raise DiagnosticRefusal("Hessian has no positive finite scale")
    symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T) / norm)
    if not math.isfinite(symmetry) or symmetry > SYMMETRY_TOL:
        raise DiagnosticRefusal("full Hessian symmetry gate failed")
    h = 0.5 * (hessian + hessian.T)
    hf, b, hd = h[:NF, :NF], h[:NF, NF:], h[NF:, NF:]
    gf, gd = gradient[:NF], gradient[NF:]
    try:
        eig, eigf, eigd = torch.linalg.eigvalsh(h), torch.linalg.eigvalsh(hf), torch.linalg.eigvalsh(hd)
    except RuntimeError as error:
        raise DiagnosticRefusal(f"block eigensolver failed: {type(error).__name__}: {error}") from error
    if not all(bool(torch.isfinite(x).all()) for x in (eig, eigf, eigd)):
        raise DiagnosticRefusal("block eigensolver returned nonfinite values")
    ff_norm, b_norm, dd_norm = (float(torch.linalg.matrix_norm(x)) for x in (hf, b, hd))
    if not all(math.isfinite(value) for value in (ff_norm, b_norm, dd_norm)):
        raise DiagnosticRefusal("block Frobenius norm is not finite")
    fscale = float(eigf.abs().max())
    floor = max(CURVATURE_REL_TOL * fscale,
                ROUNDING_MULT * torch.finfo(torch.float64).eps * max(ff_norm, torch.finfo(torch.float64).tiny))
    lambda_min, lambda_max = float(eigf[0]), float(eigf[-1])
    raw_condition = lambda_max / lambda_min if lambda_min > 0 else None
    condition = raw_condition if raw_condition is not None and math.isfinite(raw_condition) else None
    result: dict[str, Any] = {"status": "hff_schur_refused", "full_hessian_eigenvalues": eig.tolist(),
        "full_symmetry_relative": symmetry, "hff_eigenvalues": eigf.tolist(), "hdd_eigenvalues": eigd.tolist(),
        "hff_positive_floor": floor, "hff_condition": condition, "hff_condition_limit": HFF_COND_MAX,
        "block_frobenius_norms": {"hff": ff_norm, "hfd": b_norm, "hdd": dd_norm},
        "normalized_coupling": b_norm / max(math.sqrt(ff_norm) * math.sqrt(dd_norm),
                                              math.sqrt(torch.finfo(torch.float64).tiny)),
        "gradient_blocks": gradient_blocks(gradient), "schur_complement": None,
        "linearized_eliminated_dynamics_residual": None}
    if lambda_min <= floor or condition is None or not math.isfinite(condition) or condition > HFF_COND_MAX:
        result["refusal"] = "Hff is not resolved SPD within the condition limit"
        return result
    try:
        x, y = torch.linalg.solve(hf, b), torch.linalg.solve(hf, gf)
    except RuntimeError as error:
        result["refusal"] = f"Hff solve failed: {type(error).__name__}: {error}"
        return result
    rx, ry = hf @ x - b, hf @ y - gf
    xrel = _relative_residual(rx, hf @ x, b)
    yden = torch.maximum(torch.maximum(torch.linalg.vector_norm(hf @ y),
                                        torch.linalg.vector_norm(gf)),
                         torch.tensor(torch.finfo(torch.float64).tiny, dtype=torch.float64))
    yrel = float(torch.linalg.vector_norm(ry) / yden)
    result["field_solve_residuals"] = {
        "hff_inverse_hfd": xrel if math.isfinite(xrel) else None,
        "hff_inverse_gf": yrel if math.isfinite(yrel) else None,
    }
    if (not bool(torch.isfinite(x).all() & torch.isfinite(y).all())
            or not math.isfinite(xrel) or not math.isfinite(yrel)
            or max(xrel, yrel) > SOLVE_TOL):
        result["refusal"] = "Hff solve residual failed"
        return result
    schur = hd - b.T @ x
    snorm = torch.linalg.matrix_norm(schur)
    srel = float(torch.linalg.matrix_norm(schur - schur.T) / snorm) if float(snorm) > 0 else math.inf
    if not bool(torch.isfinite(schur).all()) or not math.isfinite(srel):
        result["refusal"] = "Schur complement is nonfinite or unresolved"
        return result
    result.update(status="schur_evaluated", schur_complement=schur.tolist(),
                  schur_symmetry_relative=srel,
                  schur_eigenvalues=torch.linalg.eigvalsh(0.5 * (schur + schur.T)).tolist(),
                  linearized_eliminated_dynamics_residual=(gd - b.T @ y).tolist(),
                  scope="linearized Newton residual after algebraic field elimination; not a profiled gradient unless gf is stationary")
    if srel > SYMMETRY_TOL:
        result.update(status="schur_symmetry_refused", refusal="Schur symmetry gate failed")
    return result

def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("S4 block diagnostic output must be fresh")
    require_source_pins()
    runtime_before = runtime_identity()
    if (runtime_before != RUNTIME or sha(TAIL_RAW) != TAIL_RAW_SHA
            or sha(TAIL_EVIDENCE) != TAIL_EVIDENCE_SHA or sha(TAIL_PLAN) != TAIL_PLAN_SHA):
        raise ValueError("pinned S4 tail raw/evidence/plan changed")
    tail = json.loads(TAIL_RAW.read_text())
    if tail.get("environment") != runtime_before:
        raise ValueError("S4 runtime differs from the pinned tail")
    source_before = source_hashes()
    if tail.get("source_before") != tail.get("source_after") or any(
        source_before.get(name) != digest for name, digest in tail["source_before"].items()
    ):
        raise ValueError("S4 tail dependency sources changed")
    control = endpoint_control(tail)
    problem, original, _, parameters, truth, base_identity = seed_probe._prepare_fixed_seed()
    identity = seed_probe._input_identity(problem, original, control, parameters, truth)
    tail_input = tail.get("input_after", {})
    if (identity != tail_input or identity.get("parameters_sha256") != PARAMETERS_SHA
            or identity.get("terminal_truth_sha256") != TRUTH_SHA
            or identity.get("archived_input", {}).get("problem", {}).get("fixed_problem_sha256") != PROBLEM_SHA
            or base_identity.get("archived_input") != identity.get("archived_input")
            or tensor_sha(parameters) != PARAMETERS_SHA):
        raise ValueError("reconstructed S4 point problem/parameters differ from tail")
    before = {"identity": identity, "endpoint_control_sha256": tensor_sha(control)}
    report: dict[str, Any] = {
        "phase": "running", "numerical_status": "not_reached",
        "scope": "fresh endpoint gradient/Hessian blocks only; no optimizer, score, response or physical claim",
        "tail_raw_sha256": TAIL_RAW_SHA, "tail_evidence_sha256": TAIL_EVIDENCE_SHA,
        "tail_plan_sha256": TAIL_PLAN_SHA, "endpoint_control_sha256": CONTROL_SHA,
        "branch_signature_sha256": BRANCH_SHA, "parameters_sha256": PARAMETERS_SHA,
        "runtime": runtime_before, "source_before": source_before, "input_before": before,
        "full_root_claim": False, "optimizer_step_applied": False,
        "score_computed": False, "adjoint_computed": False, "response_computed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    def save() -> None:
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)
    save()
    columns: list[Tensor] = []
    branch_before = margins_before = None
    try:
        deadline = time.monotonic() + INTERNAL_SECONDS
        branch_before, margins_before = tail_probe._full_current_branch(problem, control, parameters)
        report.update(branch_before=branch_before, branch_margins_before=margins_before)
        if (branch_before.get("status") != "passed_strict_branch"
                or branch_before.get("euler_stages") != NSTAGE
                or branch_before.get("signature_sha256") != BRANCH_SHA
                or not seed_probe._valid_margins(margins_before, complete=True)):
            raise DiagnosticRefusal("current endpoint no longer passes the pinned strict branch/margin gate")
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = tail_probe._fresh_merit(problem, control, parameters, gradient_fn)
        if fresh is None:
            raise DiagnosticRefusal("fresh current endpoint objective/gradient is nonfinite")
        objective, gradient, phi = fresh
        g_l2, g_inf = float(gradient.norm()), float(gradient.abs().max())
        prior_attempt = next((row for row in reversed(tail["candidate_attempts"])
                              if row.get("control_sha256") == CONTROL_SHA and row.get("accepted") is True), None)
        if prior_attempt is None:
            raise ValueError("final tail candidate metrics are missing")
        checks = {name: metric_check(float(prior_attempt[key]), actual) for name, key, actual in (
            ("J", "objective", float(objective)), ("Phi", "phi", float(phi)),
            ("gradient_l2", "gradient_l2", g_l2), ("gradient_inf", "gradient_inf", g_inf))}
        if not all(bool(item["passed"]) for item in checks.values()):
            raise ValueError("fresh endpoint metrics differ from the immutable tail record")
        report.update(phase="endpoint_checked", numerical_status="endpoint_metrics_reproduced",
                      objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
                      gradient_blocks=gradient_blocks(gradient), saved_metric_checks=checks,
                      branch=branch_before, branch_margins=margins_before)
        save()
        hessian_started = time.monotonic()
        eye = torch.eye(NC, dtype=control.dtype)
        for index in range(NC):
            if time.monotonic() >= deadline:
                report.update(phase="finished", numerical_status="diagnostic_budget_refusal",
                              hessian_status="budget_exhausted", hvp_columns_completed=len(columns))
                break
            col = torch.func.jvp(lambda c: gradient_fn(c, parameters), (control,), (eye[index],))[1]
            if col.shape != (NC,) or not bool(torch.isfinite(col).all()):
                raise DiagnosticRefusal("current endpoint exact HVP is invalid/nonfinite")
            columns.append(col)
        if len(columns) == NC and time.monotonic() < deadline:
            hessian = torch.stack(columns, dim=1)
            hnorm = torch.linalg.matrix_norm(hessian)
            if not bool(torch.isfinite(hnorm)) or float(hnorm) <= 0:
                raise DiagnosticRefusal("current endpoint Hessian has no finite positive scale")
            symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T) / hnorm)
            symmetric = 0.5 * (hessian + hessian.T)
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
            except RuntimeError as error:
                raise DiagnosticRefusal(f"full Hessian eigensolver failed: {error}") from error
            if (not bool(torch.isfinite(eigenvalues).all() & torch.isfinite(eigenvectors).all())
                    or not math.isfinite(symmetry) or symmetry > SYMMETRY_TOL):
                raise DiagnosticRefusal("current endpoint Hessian eigensymmetry gate failed")
            if time.monotonic() >= deadline:
                report.update(phase="finished", numerical_status="diagnostic_budget_refusal",
                              hessian_status="budget_exhausted_before_eigenpair", hvp_columns_completed=NC)
            else:
                fresh_min_hvp = torch.func.jvp(
                    lambda c: gradient_fn(c, parameters), (control,), (eigenvectors[:, 0],))[1]
                eigen_audit = eigenpair_audit(fresh_min_hvp, eigenvalues[0], eigenvectors[:, 0], hnorm)
                if not eigen_audit["passed"]:
                    report.update(phase="finished", numerical_status="diagnostic_refusal",
                                  hessian_status="minimum_eigenpair_refused", eigenpair_audit=eigen_audit)
                else:
                    blocks = schur_diagnostic(hessian, gradient)
                    report.update(phase="hessian_checked", numerical_status="diagnostic_completed",
                        hessian_status=blocks["status"], hessian=hessian.tolist(), hessian_sha256=tensor_sha(hessian),
                        hvp_columns_completed=NC, hvp_calls_total=NC + 1, eigenpair_audit=eigen_audit,
                        full_hessian_symmetry_relative=symmetry, full_hessian_eigenvalues=eigenvalues.tolist(),
                        block_schur=blocks, hessian_seconds=time.monotonic() - hessian_started)
        elif len(columns) == NC:
            report.update(phase="finished", numerical_status="diagnostic_budget_refusal",
                          hessian_status="budget_exhausted_before_eigensystem", hvp_columns_completed=NC)
    except DiagnosticRefusal as error:
        report.update(phase="diagnostic_refused", numerical_status="diagnostic_refusal",
                      hessian_status="numerical_refusal", reason=str(error),
                      hvp_columns_completed=len(columns))
    branch_after, margins_after = tail_probe._full_current_branch(problem, control, parameters)
    input_after = seed_probe._input_identity(problem, original, control, parameters, truth)
    after = source_hashes()
    runtime_after = runtime_identity()
    if branch_before is None:
        raise ValueError("initial branch audit did not produce a record")
    if branch_before.get("status") == "passed_strict_branch":
        stable_branch = (branch_after.get("status") == "passed_strict_branch"
                         and branch_after.get("signature_sha256") == BRANCH_SHA
                         and seed_probe._valid_margins(margins_after, complete=True))
    else:
        stable_branch = branch_after == branch_before and margins_after == margins_before
    if (not stable_branch or input_after != identity or after != source_before or runtime_after != runtime_before
            or sha(TAIL_RAW) != TAIL_RAW_SHA or sha(TAIL_EVIDENCE) != TAIL_EVIDENCE_SHA
            or sha(TAIL_PLAN) != TAIL_PLAN_SHA or sha(PLAN) != PLAN_SHA):
        raise ValueError("S4 endpoint branch/source/input changed during diagnostic")
    require_source_pins()
    report.update(phase="finished", source_after=after,
                  input_after={"identity": input_after,
                               "endpoint_control_sha256": tensor_sha(control)},
                  runtime_after=runtime_after, branch_after=branch_after,
                  branch_margins_after=margins_after, source_unchanged=True, input_unchanged=True)
    save()
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
