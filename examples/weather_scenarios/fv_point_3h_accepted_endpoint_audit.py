"""Fresh read-only gradient/Hessian block audit at one accepted S4 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
STEP_DIR = EVIDENCE / "point_3h_schur_newton_step_attempt1"
RAW, PREFLIGHT = STEP_DIR / "newton_step.json", STEP_DIR / "preflight.json"
STEP_PLAN = EVIDENCE / "S4_POINT_TERMINAL_SCHUR_NEWTON_STEP_PLAN_20261003.md"
PLAN = EVIDENCE / "S4_POINT_ACCEPTED_ENDPOINT_AUDIT_PLAN_20261003.md"
DEFAULT_OUT = EVIDENCE / "point_3h_accepted_endpoint_audit_attempt1/audit.json"
RAW_SHA = "b71411024191756d27bc30982800207a392e31d5b60862229ab24c181aa3fdbd"
PREFLIGHT_SHA = "d757df0a311bca4f574a55492ee0fd38435e65310dd8bbeab0980bed88b8432a"
STEP_PLAN_SHA = "432b988b478069e754c51b62a28b78775a6d017276bb2bc95b4a54bbc4e103d9"
PLAN_SHA = "97e51e332f19b86e1ab6b67c9068f93b2a91c373967c25343a828d19b4245e63"
SELF = "examples/weather_scenarios/fv_point_3h_accepted_endpoint_audit.py"
TEST = "tests/test_fv_point_3h_accepted_endpoint_audit.py"
SELF_PIN = "1b9dc387ba56a705f51dcbfd2e6c6adb87067ef407b564d550784833088596b4"
TEST_SHA = "c95de9d9be3ce504601b4f98d5e76c7dc811517761a7ffcc73d30dff08ea83a9"
SELF_RE = re.compile(rb'(?m)^SELF_PIN = "[0-9a-f]{64}"$')
NC, NF, NSTAGE = 26, 20, 3600
INTERNAL_SECONDS = 540.0
ACCEPTED_CONTROL_SHA = "d74b35f3cb8c9f70d3eeb98b2fffc2efc5492e818fe6de12abece6276f648ba8"


class AuditRefusal(ValueError):
    """Expected numerical or qualification refusal, serialized as terminal evidence."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def self_test_pins() -> None:
    content, count = SELF_RE.subn(b'SELF_PIN = "<canonical-self-pin>"', (ROOT / SELF).read_bytes())
    if count != 1 or hashlib.sha256(content).hexdigest() != SELF_PIN or sha(ROOT / TEST) != TEST_SHA:
        raise ValueError("accepted-endpoint audit source/test pin changed")
    if sha(PLAN) != PLAN_SHA or sha(STEP_PLAN) != STEP_PLAN_SHA:
        raise ValueError("accepted-endpoint audit plan pin changed")


def _metric(saved: float, fresh: float) -> dict[str, float | bool]:
    budget = 128 * torch.finfo(torch.float64).eps * max(
        abs(saved), abs(fresh), torch.finfo(torch.float64).tiny)
    delta = abs(saved - fresh)
    return {"saved": saved, "fresh": fresh, "difference": delta,
            "budget": budget, "passed": math.isfinite(delta) and delta <= budget}


def _source_hashes(names: list[str]) -> dict[str, str]:
    return {name: sha(ROOT / name) for name in names}


def _source_subset_matches(child: dict[str, str], parent: dict[str, str]) -> bool:
    return all(parent.get(name) == digest for name, digest in child.items())


def _fixed_input_matches(candidate: dict[str, Any], base: dict[str, Any], control_sha: str) -> bool:
    fixed_keys = ("parameters_sha256", "terminal_truth_sha256", "archived_input")
    return (all(candidate.get(key) == base.get(key) for key in fixed_keys)
            and candidate.get("control_sha256") == control_sha)


def _save(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def run(output: Path = DEFAULT_OUT) -> dict[str, Any]:
    if output.exists():
        raise ValueError("accepted-endpoint audit output must be fresh")
    self_test_pins()
    blocks.require_source_pins()
    if sha(RAW) != RAW_SHA or sha(PREFLIGHT) != PREFLIGHT_SHA or sha(STEP_PLAN) != STEP_PLAN_SHA:
        raise ValueError("pinned one-step input/preflight/plan changed")
    raw, preflight = json.loads(RAW.read_text()), json.loads(PREFLIGHT.read_text())
    runtime = blocks.runtime_identity()
    if (runtime != blocks.RUNTIME or raw.get("runtime") != runtime or raw.get("runtime_after") != runtime
            or raw.get("source_unchanged") is not True or raw.get("input_unchanged") is not True
            or raw.get("numerical_status") != "one_step_accepted" or raw.get("phase") != "finished"
            or raw.get("optimizer_steps") != 1 or raw.get("accepted_alpha") != 1.0
            or raw.get("response_computed") is not False or raw.get("score_computed") is not False
            or raw.get("adjoint_computed") is not False or raw.get("full_root_claim") is not False):
        raise ValueError("one-step artifact is not the pinned single accepted nonresponse result")
    parent_sources = preflight.get("source_sha256")
    step_sources = raw.get("source_before")
    if (not isinstance(parent_sources, dict) or not isinstance(step_sources, dict)
            or step_sources != raw.get("source_after")
            or not _source_subset_matches(step_sources, parent_sources)):
        raise ValueError("one-step preflight/child source maps do not agree")
    deps = sorted(parent_sources)
    before_sources = _source_hashes(deps)
    if before_sources != parent_sources:
        raise ValueError("original one-step dependencies no longer match their frozen source hashes")
    p = torch.tensor(raw["parameters"], dtype=torch.float64)
    control = torch.tensor(raw["accepted_control"], dtype=torch.float64)
    cached_g = torch.tensor(raw["accepted_gradient"], dtype=torch.float64)
    if (p.shape != (13,) or control.shape != (NC,) or cached_g.shape != (NC,)
            or not bool(torch.isfinite(p).all() & torch.isfinite(control).all() & torch.isfinite(cached_g).all())
            or tensor_sha(p) != raw.get("parameters_sha256")
            or tensor_sha(control) != ACCEPTED_CONTROL_SHA
            or raw.get("accepted_control_sha256") != ACCEPTED_CONTROL_SHA
            or raw.get("control_sha256") != "602b08828b92508dfaed3605f4626882a1cf70646730da1cb866416dd15cf0b9"):
        raise ValueError("accepted endpoint control or original parameter vector is malformed")
    if raw.get("accepted_branch", {}).get("status") != "passed_strict_branch" or raw["accepted_branch"].get("euler_stages") != NSTAGE:
        raise ValueError("archived accepted endpoint lacks strict 3600-stage branch")
    problem, original, _, parameters, truth, identity0 = seed._prepare_fixed_seed()
    identity = seed._input_identity(problem, original, control, parameters, truth)
    base_identity = raw.get("input_after", {}).get("identity", {})
    if (tensor_sha(parameters) != raw["parameters_sha256"]
            or not _fixed_input_matches(identity, base_identity, ACCEPTED_CONTROL_SHA)
            or identity0.get("archived_input") != identity.get("archived_input")):
        raise ValueError("reconstructed fixed original problem/parameters differ from accepted endpoint")
    own = [SELF, TEST, str(PLAN.relative_to(ROOT)), str(RAW.relative_to(ROOT)),
           str(PREFLIGHT.relative_to(ROOT)), str(STEP_PLAN.relative_to(ROOT)),
           blocks.SELF, blocks.TEST, str(blocks.PLAN.relative_to(ROOT))]
    paths = sorted(set(deps + own))
    source_before = _source_hashes(paths)
    initial = {"phase": "running", "numerical_status": "not_reached",
        "scope": "fresh one-point original-problem gradient/Hessian block audit; no step, root, response, score or physical claim",
        "step_raw_sha256": RAW_SHA, "step_preflight_sha256": PREFLIGHT_SHA,
        "step_plan_sha256": STEP_PLAN_SHA, "plan_sha256": PLAN_SHA,
        "accepted_control": control.tolist(), "accepted_control_sha256": tensor_sha(control),
        "original_parameters": parameters.tolist(), "parameters_sha256": tensor_sha(parameters),
        "runtime": runtime, "source_before": source_before, "input_before": identity,
        "optimizer_step_applied": False, "full_root_claim": False,
        "response_computed": False, "score_computed": False, "adjoint_computed": False}
    _save(output, initial)
    report: dict[str, Any] = initial
    branch0 = margins0 = None
    columns: list[Tensor] = []
    try:
        deadline = time.monotonic() + INTERNAL_SECONDS
        branch0, margins0 = tail._full_current_branch(problem, control, parameters)
        if (branch0.get("status") != "passed_strict_branch" or branch0.get("euler_stages") != NSTAGE
                or branch0.get("signature_sha256") != raw["accepted_branch"].get("signature_sha256")
                or not seed._valid_margins(margins0, complete=True)):
            raise AuditRefusal("accepted endpoint failed fresh strict branch/margin replay")
        grad_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = tail._fresh_merit(problem, control, parameters, grad_fn)
        if fresh is None:
            raise AuditRefusal("fresh accepted endpoint J/Phi/gradient is nonfinite")
        objective, gradient, phi = fresh
        gm = [_metric(float(cached_g[i]), float(gradient[i])) for i in range(NC)]
        metrics: dict[str, Any] = {
            "J": _metric(float(raw["accepted_objective"]), float(objective)),
            "Phi": _metric(float(raw["accepted_phi"]), float(phi)),
            "gradient_components": gm,
        }
        if not metrics["J"]["passed"] or not metrics["Phi"]["passed"] or not all(x["passed"] for x in gm):
            raise AuditRefusal("fresh J/Phi/full-gradient differ from accepted one-step record")
        report.update(phase="endpoint_checked", numerical_status="accepted_endpoint_reproduced",
            objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
            gradient_blocks=blocks.gradient_blocks(gradient), saved_metric_checks=metrics,
            branch=branch0, branch_margins=margins0)
        _save(output, report)
        eye = torch.eye(NC, dtype=torch.float64)
        for i in range(NC):
            if time.monotonic() >= deadline:
                raise AuditRefusal("540-second internal budget exhausted during exact Hessian columns")
            col = torch.func.jvp(lambda c: grad_fn(c, parameters), (control,), (eye[i],))[1]
            if col.shape != (NC,) or not bool(torch.isfinite(col).all()):
                raise AuditRefusal("fresh exact HVP column is invalid/nonfinite")
            columns.append(col)
        hessian = torch.stack(columns, dim=1)
        hnorm = torch.linalg.matrix_norm(hessian)
        if not bool(torch.isfinite(hnorm)) or float(hnorm) <= 0:
            raise AuditRefusal("fresh Hessian has no positive finite scale")
        symmetry = float(torch.linalg.matrix_norm(hessian - hessian.T) / hnorm)
        if not math.isfinite(symmetry) or symmetry > blocks.SYMMETRY_TOL:
            raise AuditRefusal("fresh Hessian symmetry qualification failed")
        try:
            eig, vec = torch.linalg.eigh(0.5 * (hessian + hessian.T))
        except RuntimeError as error:
            raise AuditRefusal(f"fresh Hessian eigensolver failed: {error}") from error
        if not bool(torch.isfinite(eig).all() & torch.isfinite(vec).all()) or time.monotonic() >= deadline:
            raise AuditRefusal("fresh Hessian eigensystem is nonfinite or budget exhausted")
        min_product = torch.func.jvp(lambda c: grad_fn(c, parameters), (control,), (vec[:, 0],))[1]
        eigen_audit = blocks.eigenpair_audit(min_product, eig[0], vec[:, 0], hnorm)
        if not eigen_audit["passed"]:
            raise AuditRefusal("independent minimum-eigenpair HVP residual failed")
        schur = blocks.schur_diagnostic(hessian, gradient)
        report.update(phase="hessian_checked", numerical_status="diagnostic_completed",
            hessian_status=schur["status"], hessian=hessian.tolist(), hessian_sha256=tensor_sha(hessian),
            hvp_columns_completed=NC, hvp_calls_total=NC + 1, full_hessian_symmetry_relative=symmetry,
            full_hessian_eigenvalues=eig.tolist(), eigenpair_audit=eigen_audit, block_schur=schur)
    except AuditRefusal as error:
        report.update(phase="diagnostic_refused", numerical_status="diagnostic_refusal",
                      reason=str(error), hvp_columns_completed=len(columns))
    branch1, margins1 = tail._full_current_branch(problem, control, parameters)
    identity_after = seed._input_identity(problem, original, control, parameters, truth)
    source_after = _source_hashes(paths)
    if branch0 is None:
        raise ValueError("initial branch audit produced no record")
    if branch0.get("status") == "passed_strict_branch":
        stable = (branch1.get("status") == "passed_strict_branch"
                  and branch1.get("signature_sha256") == branch0.get("signature_sha256")
                  and seed._valid_margins(margins1, complete=True))
    else:
        stable = branch1 == branch0 and margins1 == margins0
    self_test_pins()
    if (not stable or identity_after != identity or source_after != source_before
            or blocks.runtime_identity() != runtime or sha(RAW) != RAW_SHA
            or sha(PREFLIGHT) != PREFLIGHT_SHA or sha(PLAN) != PLAN_SHA):
        raise ValueError("accepted-endpoint branch/source/input/runtime changed during audit")
    report.update(phase="finished", source_after=source_after, input_after=identity_after,
        runtime_after=blocks.runtime_identity(), branch_after=branch1,
        branch_margins_after=margins1, source_unchanged=True, input_unchanged=True)
    _save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().output)
