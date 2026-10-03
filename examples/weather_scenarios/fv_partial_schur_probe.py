"""One fresh 20+6 exact-Hessian Schur diagnostic at the PR227 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from advar import matrix_free
from examples.weather_scenarios import fv_partial_field_correction_probe as field
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from tests.test_fv_research_partial_observation import _problem

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "graphify-out/fv-root-cause-20260919/partial_field_correction_attempt1/field_correction.json"
PLAN = ROOT / "graphify-out/fv-root-cause-20260919/PR196_228_SCHUR_DIAGNOSTIC_PLAN_20261003.md"
TEST = ROOT / "tests/test_fv_partial_schur_probe.py"
RAW_SHA256 = "8ad5b5e3d56a074ad79a28fa0f23c0cbdae745c26efed5101cd826fe2d9442cf"
PLAN_SHA256 = "a54b05464a0e2315df9163595dc405d8c011c01e0a0849d5767dae1ad2a0da26"
TEST_SHA256 = "9d1fe1a28eec25b5a8f24de8a7ce394e8d2f7ecb5dcadadf093f82cef7a89fe4"
CONTROL_SHA256 = "a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26"
PARAMETERS_SHA256 = "e3a45fac86e622332c6afd3472b46d58ef97cd37db1445260c86923a3c423bf9"
INPUT_SHA256 = "838fcf77a43b209b53dfe62f6a1657d0f8eba5a41797e431451059f3f8d18158"
REL_TOL = 1e-10


class SchurScientificRefusal(ValueError):
    """Numerical evidence refuses elimination or a dense Schur solve."""

    def __init__(self, reason: str, evidence: dict[str, Any]) -> None:
        super().__init__(reason)
        self.evidence = evidence


def schur_algebra(hessian: Tensor, gradient: Tensor) -> dict[str, Tensor | float]:
    """Dense 20+6 elimination; raise on unsafe field or singular Schur blocks."""
    if hessian.shape != (26, 26) or gradient.shape != (26,):
        raise ValueError("expected a 26x26 Hessian and 26-vector gradient")
    if (hessian.dtype != torch.float64 or gradient.dtype != torch.float64
            or hessian.device.type != "cpu" or gradient.device.type != "cpu"):
        raise ValueError("Schur algebra requires CPU FP64 inputs")
    if not bool(torch.isfinite(hessian).all() & torch.isfinite(gradient).all()):
        raise ValueError("Hessian and gradient must be finite")
    tiny = torch.finfo(hessian.dtype).tiny
    sym = _norm_ratio(hessian - hessian.T, hessian)
    if sym > REL_TOL:
        raise ValueError("Hessian symmetry tolerance failed")
    hff, hfd = hessian[:20, :20], hessian[:20, 20:]
    hdf, hdd = hessian[20:, :20], hessian[20:, 20:]
    gf, gd = gradient[:20], gradient[20:]
    evff = torch.linalg.eigvalsh(hff)
    condff = float(torch.linalg.cond(hff))
    if (float(evff[0]) <= 0 or not math.isfinite(condff)
            or condff >= 1 / math.sqrt(torch.finfo(hessian.dtype).eps)):
        raise SchurScientificRefusal("field Hessian is not numerically positive definite", {
            "hessian_eigenvalues": torch.linalg.eigvalsh(0.5 * (hessian + hessian.T)),
            "hff_eigenvalues": evff, "hff_condition": condff if math.isfinite(condff) else None})
    ff_gf = torch.linalg.solve(hff, gf)
    ff_hfd = torch.linalg.solve(hff, hfd)
    gf_solve_abs, gf_solve_rel = _backward_residual(hff, ff_gf, gf)
    hfd_solve_abs, hfd_solve_rel = _backward_residual(hff, ff_hfd, hfd)
    if max(gf_solve_rel, hfd_solve_rel) > REL_TOL:
        raise ValueError("field-block solve residual tolerance failed")
    schur = hdd - hdf @ ff_hfd
    rhs = -gd + hdf @ ff_gf
    schur_symmetry = _norm_ratio(schur - schur.T, schur)
    if schur_symmetry > REL_TOL:
        raise ValueError("Schur complement symmetry invariant failed")
    evs = torch.linalg.eigvalsh(0.5 * (schur + schur.T))
    conds = float(torch.linalg.cond(schur))
    if not math.isfinite(conds) or conds >= 1 / math.sqrt(torch.finfo(hessian.dtype).eps):
        raise SchurScientificRefusal("Schur complement is singular or numerically ill-conditioned", {
            "hessian_eigenvalues": torch.linalg.eigvalsh(0.5 * (hessian + hessian.T)),
            "hff_eigenvalues": evff, "schur_eigenvalues": evs,
            "schur_condition": conds if math.isfinite(conds) else None,
            "schur_symmetry_relative": schur_symmetry,
            "hff_gradient_solve_absolute": gf_solve_abs,
            "hff_gradient_solve_relative": gf_solve_rel,
            "hff_cross_solve_absolute": hfd_solve_abs,
            "hff_cross_solve_relative": hfd_solve_rel})
    dd = torch.linalg.solve(schur, rhs)  # Dense six-variable diagnostic, no CG.
    df = -torch.linalg.solve(hff, gf + hfd @ dd)
    step = torch.cat((df, dd))
    direct = torch.linalg.solve(hessian, -gradient)
    schur_solve_abs, schur_solve_rel = _backward_residual(schur, dd, rhs)
    if schur_solve_rel > REL_TOL:
        raise ValueError("Schur solve residual tolerance failed")
    hxg = hessian @ step + gradient
    block_residual = torch.cat((hff @ df + hfd @ dd + gf, hdf @ df + hdd @ dd + gd))
    norm = _scaled_norm
    full_scale = max(tiny, norm(hessian) * norm(step) + norm(gradient))
    block_scale = max(tiny, norm(hessian) * norm(step) + norm(gradient))
    return {"hessian_eigenvalues": torch.linalg.eigvalsh(0.5 * (hessian + hessian.T)),
            "hff_eigenvalues": evff, "schur_eigenvalues": evs, "schur": schur, "rhs": rhs,
            "step": step, "direct_step": direct, "symmetry_relative": sym,
            "hff_condition": condff, "schur_condition": conds,
            "schur_symmetry_relative": schur_symmetry,
            "full_newton_absolute_difference": norm(step - direct),
            "full_newton_relative_difference": norm(step - direct) / max(norm(direct), tiny),
            "hff_gradient_solve_absolute": gf_solve_abs, "hff_gradient_solve_relative": gf_solve_rel,
            "hff_cross_solve_absolute": hfd_solve_abs, "hff_cross_solve_relative": hfd_solve_rel,
            "schur_solve_absolute": schur_solve_abs, "schur_solve_relative": schur_solve_rel,
            "true_hx_plus_g_absolute": norm(hxg), "true_hx_plus_g_relative": norm(hxg) / full_scale,
            "block_residual_absolute": norm(block_residual),
            "block_residual_relative": norm(block_residual) / block_scale}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _branch(problem: Any, control: Tensor, parameters: Tensor) -> dict[str, Any]:
    raw, _, face = preflight.branch_with_face_margin(problem, control, parameters)
    return field._branch_summary(raw, face)


def _relative(a: Tensor, b: Tensor) -> float:
    scale = float(b.abs().max()) if b.numel() else 0.0
    if scale == 0.0:
        return 0.0 if not bool((a != 0).any()) else math.inf
    return _norm_ratio(a / scale - b / scale, b / scale)


def _norm_ratio(numerator: Tensor, denominator: Tensor) -> float:
    scale = float(denominator.abs().max()) if denominator.numel() else 0.0
    if scale == 0.0:
        return 0.0 if not bool((numerator != 0).any()) else math.inf
    return float(torch.linalg.vector_norm(numerator / scale)
                 / torch.linalg.vector_norm(denominator / scale))


def _scaled_norm(value: Tensor) -> float:
    scale = float(value.abs().max()) if value.numel() else 0.0
    return scale * float(torch.linalg.vector_norm(value / scale)) if scale else 0.0


def _backward_residual(matrix: Tensor, solution: Tensor, rhs: Tensor) -> tuple[float, float]:
    absolute = _scaled_norm(matrix @ solution - rhs)
    scale = _scaled_norm(matrix) * _scaled_norm(solution) + _scaled_norm(rhs)
    return absolute, absolute / max(scale, torch.finfo(matrix.dtype).tiny)


def _assemble_hessian(objective: Any, control: Tensor) -> Tensor:
    eye = torch.eye(26, dtype=control.dtype, device=control.device)
    return torch.stack([matrix_free.hvp(objective, control, eye[:, i]) for i in range(26)], dim=1)


def run_diagnostic() -> dict[str, Any]:
    """Run exactly one endpoint linearization and 26 Hessian-vector products."""
    raw_before, plan_before, test_before = _sha(RAW), _sha(PLAN), _sha(TEST)
    if (raw_before != RAW_SHA256 or plan_before != PLAN_SHA256 or test_before != TEST_SHA256):
        raise ValueError("pinned endpoint raw, plan, or regression source changed")
    if platform.python_version() != "3.12.13" or torch.__version__ != "2.13.0":
        raise ValueError("diagnostic runtime differs from pinned CPU FP64 runtime")
    record = json.loads(RAW.read_text())
    sources = record["source_before"]
    if sources != record["source_after"] or field._sources() != sources:
        raise ValueError("one or more of the 21 endpoint source pins changed")
    self_before = _sha(Path(__file__))
    input_before = prior._preflight_identity()
    if input_before["current_problem_identity"]["fixed_problem_sha256"] != INPUT_SHA256:
        raise ValueError("fixed input identity changed")
    problem, _, parameters = _problem()
    control = torch.tensor(record["final_control"], dtype=torch.float64)
    if (problem.identity != input_before["current_problem_identity"]
            or _tensor_sha(control) != CONTROL_SHA256
            or _tensor_sha(parameters) != PARAMETERS_SHA256
            or _tensor_sha(parameters) != record["parameters_sha256"]):
        raise ValueError("endpoint, parameters, or problem identity changed")
    if (control.device.type != "cpu" or parameters.device.type != "cpu"
            or control.dtype != torch.float64 or parameters.dtype != torch.float64):
        raise ValueError("diagnostic requires CPU FP64")
    branch_before = _branch(problem, control, parameters)
    if branch_before != record["final_branch"]:
        raise ValueError("endpoint branch differs from pinned field-correction branch")
    objective = lambda x: problem.objective(x, parameters)
    value_before = float(objective(control))
    gradient_before = torch.func.grad(objective)(control)
    audit = record["final_full_gradient"]
    field_inf = float(gradient_before[:20].abs().max())
    # Match the archived reduction exactly; stationarity is a separate gate.
    field_l2 = float(torch.linalg.vector_norm(gradient_before[:20]))
    if (value_before != record["final_objective"]
            or not torch.equal(gradient_before[20:], torch.tensor(audit["dynamics_components"], dtype=torch.float64))
            or field_inf != audit["field_inf"] or field_l2 != audit["field_l2"]
            or float(gradient_before.abs().max()) != audit["full_inf"]):
        raise ValueError("fresh endpoint J or gradient differs from pinned raw audit")
    if field_inf > 1e-10:
        raise ValueError("field block no longer meets its declared stationarity gate")
    hessian = _assemble_hessian(objective, control)
    refusal: SchurScientificRefusal | None = None
    try:
        algebra = schur_algebra(hessian, gradient_before)
    except SchurScientificRefusal as error:
        algebra, refusal = error.evidence, error
    branch_after = _branch(problem, control, parameters)
    value_after, gradient_after = float(objective(control)), torch.func.grad(objective)(control)
    sources_after = field._sources()
    raw_after, plan_after, test_after = _sha(RAW), _sha(PLAN), _sha(TEST)
    if (raw_after != raw_before or plan_after != plan_before or test_after != test_before
            or sources_after != sources or prior._preflight_identity() != input_before
            or branch_after != branch_before or value_after != value_before
            or not torch.equal(gradient_after, gradient_before)):
        raise ValueError("pinned source, raw, plan, test, input, branch, J, or gradient changed during linearization")
    if refusal is None and any(float(algebra[k]) > REL_TOL for k in (
            "full_newton_relative_difference", "hff_gradient_solve_relative",
            "hff_cross_solve_relative", "schur_solve_relative",
            "true_hx_plus_g_relative", "block_residual_relative")):
        raise ValueError("Schur/block Newton residual verification failed")
    self_after = _sha(Path(__file__))
    if self_after != self_before:
        raise ValueError("diagnostic source changed during linearization")
    return {"status": "scientific_refusal" if refusal else "diagnostic_completed",
            "scientific_refusal_reason": str(refusal) if refusal else None,
            "control_sha256": CONTROL_SHA256,
            "parameters_sha256": PARAMETERS_SHA256, "fixed_input_sha256": INPUT_SHA256,
            "plan_sha256_before_after": [plan_before, plan_after],
            "test_sha256_before_after": [test_before, test_after],
            "endpoint_raw_sha256_before_after": [raw_before, raw_after],
            "source_before_after": sources, "source_self_sha256_before_after": [self_before, self_after],
            "input_before_after": input_before, "branch": branch_before, "objective": value_before,
            "archive_comparison": "exact pinned CPU FP64 J, dynamics gradient, field-gradient norms",
            "field_gradient_inf": field_inf, "field_stationarity_tolerance": 1e-10,
            "gradient": gradient_before.tolist(), "hvp_columns": 26,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
            **{k: (v.tolist() if isinstance(v, Tensor) else v) for k, v in algebra.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("Schur diagnostic output and temporary path must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(run_diagnostic(), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
