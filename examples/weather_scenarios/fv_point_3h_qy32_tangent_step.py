"""One bounded chart-tangent original-J step at the committed e29 point."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_qy32_diagnostic as qy
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guard_policy
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "QY32_TANGENT_STEP_PLAN_20261007.json"
OLD_PLAN = EVIDENCE / "QY32_TWO_SIDED_DIAGNOSTIC_PLAN_20261007.json"
OLD_PLAN_SHA = "bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f"
GZIP_RAW = EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.json.gz"
GZIP_SHA = "1b81004e56e31e051221ddebd6af6bbab8f8809f1eb888796cc1cb37476fea3e"
RAW_SHA = "d05e5b8ed86f9067b4d29d5f9b9bf7ccd4ea7c23d8157a94410c0d59914d15c3"
PARENT = EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.run.json"
RESOURCE = EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.resource.json"
ARCHIVE_META = EVIDENCE / "qy32_diagnostic_attempt1/archive.json"
SELF = "examples/weather_scenarios/fv_point_3h_qy32_tangent_step.py"
TEST = "tests/test_fv_point_3h_qy32_tangent_step.py"
INTERNAL_SECONDS, WALL_SECONDS, RSS_BYTES = 240.0, 300.0, 1024**3
RADIUS, C1_J, C1_PHI, MAX_CANDIDATES, HVP_CAP = 0.05, 1e-4, 1e-4, 16, 1
CONTROL_SHA = qy.ENDPOINT_SHA
PARAMETERS_SHA = qy.PARAMETERS_SHA
EXPECTED_DIRECTION_NORM = 0.7477478026576
EXPECTED_G_D = -0.5587502088011976


def policy() -> dict[str, Any]:
    return {"radius": RADIUS, "j_armijo_c1": C1_J, "phi_armijo_c1": C1_PHI,
            "max_candidates": MAX_CANDIDATES, "hvp_cap": HVP_CAP, "pcg_solves": 0,
            "internal_seconds": INTERNAL_SECONDS, "outer_seconds": WALL_SECONDS,
            "rss_bytes": RSS_BYTES, "guarded_launches": 1}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _metric(saved: float, fresh: float) -> dict[str, float | bool]:
    budget = 128 * torch.finfo(torch.float64).eps * max(
        abs(saved), abs(fresh), torch.finfo(torch.float64).tiny)
    difference = abs(saved - fresh)
    return {"saved": saved, "fresh": fresh, "difference": difference,
            "roundoff_budget": budget, "passed": math.isfinite(difference) and difference <= budget}


def tangent_direction(z0: Tensor, average_tangent_gradient: Tensor) -> tuple[Tensor, Tensor]:
    """Solve the chart metric system for the negative common-tangent direction."""
    if (z0.shape != (26, 25) or average_tangent_gradient.shape != (25,)
            or z0.dtype != torch.float64 or average_tangent_gradient.dtype != torch.float64
            or not bool(torch.isfinite(z0).all() & torch.isfinite(average_tangent_gradient).all())):
        raise ValueError("tangent solve requires finite FP64 Z[26,25] and q_t[25]")
    metric = z0.T @ z0
    if not bool(torch.isfinite(metric).all()):
        raise ValueError("chart tangent metric is nonfinite")
    try:
        dt = torch.linalg.solve(metric, -average_tangent_gradient)
    except RuntimeError as error:
        raise ValueError("chart tangent metric solve failed") from error
    if not bool(torch.isfinite(dt).all()):
        raise ValueError("chart tangent direction is nonfinite")
    return dt, z0 @ dt


def chart_candidate(base: Tensor, eta: float, retained_step: Tensor, alpha: float) -> Tensor:
    """Apply a retained-coordinate step and restore the pivot at fixed eta."""
    if (base.shape != (26,) or retained_step.shape != (25,) or not math.isfinite(alpha)
            or alpha < 0):
        raise ValueError("chart candidate requires a 26-control base and nonnegative finite scale")
    retained = [index for index in range(26) if index != qy.PIVOT]
    trial = base.clone()
    trial[retained] = base[retained] + alpha * retained_step
    return qy.chart(trial, eta)


def resolved_direction_checks(gradient: Tensor, direction: Tensor,
                              h_direction: Tensor) -> dict[str, Any]:
    """Require resolved descent in J and its gradient-squared merit function."""
    j_products = gradient * direction
    phi_products = gradient * h_direction
    g_dot_d = float(torch.dot(gradient, direction))
    g_dot_h_d = float(torch.dot(gradient, h_direction))
    j_budget = float(128 * torch.finfo(torch.float64).eps * j_products.abs().sum())
    phi_budget = float(128 * torch.finfo(torch.float64).eps * phi_products.abs().sum())
    if not (math.isfinite(g_dot_d) and math.isfinite(g_dot_h_d)
            and g_dot_d < -j_budget and g_dot_h_d < -phi_budget):
        raise guard_policy.StepRefusal("chart tangent direction lacks resolved descent in J and Phi")
    return {"g_dot_d": g_dot_d, "g_dot_d_roundoff_budget": j_budget,
            "g_dot_Hd": g_dot_h_d, "g_dot_Hd_roundoff_budget": phi_budget}


def dual_armijo(base_j: float, base_phi: float, trial_j: float, trial_phi: float,
                alpha: float, g_dot_d: float, g_dot_h_d: float) -> dict[str, Any]:
    """Evaluate the two full-gradient merit Armijo inequalities independently."""
    j_limit = base_j + C1_J * alpha * g_dot_d
    phi_limit = base_phi + C1_PHI * alpha * g_dot_h_d
    return {"j_limit": j_limit, "phi_limit": phi_limit,
            "J_armijo_passed": trial_j <= j_limit,
            "Phi_armijo_passed": trial_phi <= phi_limit,
            "accepted": trial_j <= j_limit and trial_phi <= phi_limit}


def bounded_start_alpha(base: Tensor, eta: float, retained_step: Tensor,
                        direction_norm: float) -> float:
    """Choose the nominal radius scale, bisecting only the static chart geometry."""
    def admitted(scale: float) -> bool:
        try:
            point = chart_candidate(base, eta, retained_step, scale)
        except ValueError:
            return False
        return float(torch.linalg.vector_norm(point - base)) <= RADIUS

    alpha = min(1.0, RADIUS / direction_norm)
    if admitted(alpha):
        return alpha
    low, high = 0.0, alpha
    for _ in range(60):
        middle = (low + high) / 2
        if admitted(middle):
            low = middle
        else:
            high = middle
    return low


def counted_hvp(gradient_fn: Any, control: Tensor, parameters: Tensor,
                direction: Tensor, deadline: float) -> tuple[Any, dict[str, int]]:
    """Wrap one matrix-free HVP with a strict one-product counter and deadline."""
    counts = {"started": 0, "completed": 0}

    def apply() -> Tensor:
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired before the one live HVP")
        if counts["started"] >= HVP_CAP:
            raise ValueError("single live-HVP cap reached")
        counts["started"] += 1
        result = torch.func.jvp(gradient_fn, (control, parameters),
                                (direction, torch.zeros_like(parameters)))[1]
        counts["completed"] += 1
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired after the one live HVP")
        return result

    return apply, counts


def _mark_not_committed(record: dict[str, Any], reason: str) -> None:
    if record.get("optimizer_steps_applied") == 0 and record.get("trials"):
        last = record["trials"][-1]
        if last.get("status") == "accepted":
            last.update(status="candidate_not_committed", accepted=False,
                        commit_refusal=reason)


def _record_direction_refusal(record: dict[str, Any], reason: str, *, closed: bool) -> None:
    record.update(phase="finished", execution_status="completed" if closed else "failed",
                  numerical_status="direction_refusal" if closed else "integrity_refusal",
                  refusal=reason, optimizer_steps_applied=0, candidate_committed=False)
    _mark_not_committed(record, reason)


def _pinned_path(name: str, digest: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("new tangent plan pins must be repository relative")
    path = ROOT / name
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
            or not path.is_file() or _sha(path) != digest):
        raise ValueError(f"tangent source/archive pin changed: {name}")
    return path


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    path = path.resolve()
    if (path != PLAN.resolve() or path.is_symlink()
            or not path.resolve().is_relative_to(ROOT.resolve()) or _sha(path) != digest):
        raise ValueError("tangent-step plan identity mismatch")
    plan = json.loads(path.read_text())
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if (plan.get("experiment_kind") != "qy32_chart_tangent_step"
            or plan.get("policy") != policy() or not isinstance(sources, dict)
            or not isinstance(archives, dict)):
        raise ValueError("tangent plan policy or pin maps are invalid")
    old_plan, _, _ = qy.load_base(OLD_PLAN, OLD_PLAN_SHA)
    if (plan.get("producing_plan") != OLD_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != OLD_PLAN_SHA
            or not set(old_plan["source_files"]) <= set(sources)
            or any(sources[name] != value for name, value in old_plan["source_files"].items())):
        raise ValueError("tangent plan does not inherit the pinned Qy32 source set")
    if sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST):
        raise ValueError("tangent plan must pin its producer and focused tests")
    for name, value in sources.items():
        _pinned_path(name, value)
    for name, value in archives.items():
        _pinned_path(name, value)
    inherited_archives = old_plan["archive_files"]
    if any(archives.get(name) != value for name, value in inherited_archives.items()):
        raise ValueError("tangent plan changed an inherited Qy32 archive pin")
    required = {OLD_PLAN.relative_to(ROOT).as_posix(): OLD_PLAN_SHA,
                GZIP_RAW.relative_to(ROOT).as_posix(): GZIP_SHA,
                PARENT.relative_to(ROOT).as_posix(): _sha(PARENT),
                RESOURCE.relative_to(ROOT).as_posix(): _sha(RESOURCE),
                ARCHIVE_META.relative_to(ROOT).as_posix(): _sha(ARCHIVE_META)}
    if any(archives.get(name) != value for name, value in required.items()):
        raise ValueError("tangent plan omits or changes the compressed diagnostic receipts")
    return plan


def _load_diagnostic() -> dict[str, Any]:
    parent = json.loads(PARENT.read_text())
    resource = json.loads(RESOURCE.read_text())
    archive_meta = json.loads(ARCHIVE_META.read_text())
    execution = guard_policy.execution_status(resource)
    if (execution != "completed" or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != RAW_SHA
            or parent.get("numerical_status") != "finite_chart_samples"
            or archive_meta.get("raw_sha256") != RAW_SHA
            or archive_meta.get("archive_sha256") != GZIP_SHA
            or _sha(GZIP_RAW) != GZIP_SHA):
        raise ValueError("compressed Qy32 source diagnostic receipts are incomplete")
    compressed = GZIP_RAW.read_bytes()
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != RAW_SHA:
        raise ValueError("decompressed Qy32 diagnostic hash differs from d05 pin")
    raw = json.loads(payload)
    del payload, compressed
    if (raw.get("phase") != "finished" or raw.get("execution_status") != "completed"
            or raw.get("numerical_status") != "finite_chart_samples"
            or raw.get("plan_sha256") != OLD_PLAN_SHA
            or raw.get("base_control_sha256") != CONTROL_SHA
            or raw.get("parameters_sha256") != PARAMETERS_SHA
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("runtime") != raw.get("runtime_after")
            or [item.get("requested_eta") for item in raw.get("samples", [])]
            != [-2e-6, -1e-6, 1e-6, 2e-6]):
        raise ValueError("compressed Qy32 diagnostic receipt is not a closed four-side report")
    return raw


def _direction_from_diagnostic(base: Tensor, base_eta: float,
                               raw: dict[str, Any]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    samples = {float(row["requested_eta"]): row for row in raw["samples"]}
    negative = torch.tensor(samples[-1e-6]["geometry"]["tangent_covector"], dtype=torch.float64)
    positive = torch.tensor(samples[1e-6]["geometry"]["tangent_covector"], dtype=torch.float64)
    if any(value.shape != (25,) or not bool(torch.isfinite(value).all())
           for value in (negative, positive)):
        raise ValueError("diagnostic tangent covectors must be finite length-25 values")
    average = (negative + positive) / 2
    z0, _ = qy.chart_jacobian(base, 0.0)
    zeta, _ = qy.chart_jacobian(base, base_eta)
    dt, _ = tangent_direction(z0, average)
    direction = zeta @ dt
    expected_pairing = {-2e-6: -0.5579511898, -1e-6: -0.5585182949,
                        1e-6: -0.5597352579, 2e-6: -0.5603851153}
    pairings = {}
    for eta, sample in samples.items():
        covector = torch.tensor(sample["geometry"]["tangent_covector"], dtype=torch.float64)
        if covector.shape != (25,) or not bool(torch.isfinite(covector).all()):
            raise ValueError("all four archived tangent covectors must be finite length-25 values")
        pairing = float(covector @ dt)
        if abs(pairing - expected_pairing[eta]) > 3e-8:
            raise ValueError("archived tangent covector pairing differs from independent check")
        pairings[f"{eta:+.0e}"] = pairing
    direction_norm = float(torch.linalg.vector_norm(direction))
    stats = {"average_tangent_gradient": average.tolist(),
             "retained_direction": dt.tolist(), "direction": direction.tolist(),
             "direction_norm": direction_norm,
             "source_eta_covectors": {"-1e-6": negative.tolist(), "+1e-6": positive.tolist()},
             "all_eta_covector_pairings": pairings,
             "metric_scope": "G0=Z0.T@Z0 at chart(base, eta=0); covector mean from symmetric nonzero eta samples"}
    if abs(direction_norm - EXPECTED_DIRECTION_NORM) > 2e-8:
        raise ValueError("reconstructed chart direction norm differs from independent pin")
    return dt, direction, stats


def _check_fixed_input(candidate: dict[str, Any], base_identity: dict[str, Any], control_sha: str) -> bool:
    return (candidate.get("control_sha256") == control_sha
            and candidate.get("parameters_sha256") == base_identity.get("parameters_sha256")
            and candidate.get("terminal_truth_sha256") == base_identity.get("terminal_truth_sha256")
            and candidate.get("archived_input") == base_identity.get("archived_input"))


def _compact_measure(measure: dict[str, Any]) -> dict[str, Any]:
    row = dict(measure)
    signature = row.pop("branch_signature", None)
    row["branch_signature_sha256"] = (None if signature is None else hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    return row


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    record: dict[str, Any] = {"phase": "running", "execution_status": "running",
        "numerical_status": "setup_not_complete", "plan_sha256": plan_sha,
        "base_control_sha256": CONTROL_SHA, "parameters_sha256": PARAMETERS_SHA,
        "policy": policy(), "trials": [], "hvp_calls": 0, "hvp_calls_completed": 0,
        "pcg_solves": 0, "optimizer_steps_applied": 0,
        "candidate_committed": False, "response_computed": False,
        "score_computed": False, "source_before": None, "source_after": None,
        "source_unchanged": None, "fixed_input_unchanged": None,
        "input_before": None, "input_after": None, "runtime": None, "runtime_after": None}
    guard_policy._write(output, record)
    source_names: set[str] = set()
    problem = None
    original = None
    control = None
    parameters = None
    truth = None
    identity: dict[str, Any] | None = None

    def finalize(final_control: Tensor | None) -> None:
        if source_names:
            after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            record["source_after"] = after
            record["source_unchanged"] = record["source_before"] == after
        if all(value is not None for value in (problem, original, control, parameters, truth)):
            result_control = final_control if final_control is not None else control
            input_after = seed._input_identity(cast(Any, problem), cast(Tensor, original),
                cast(Tensor, result_control), cast(Tensor, parameters), cast(Tensor, truth))
            record["input_after"] = input_after
            record["fixed_input_unchanged"] = _check_fixed_input(
                input_after, identity or {}, _tensor_sha(cast(Tensor, result_control)))
        record["runtime_after"] = guard_policy.blocks.runtime_identity()
        record["elapsed_seconds"] = time.monotonic() - started

    try:
        plan = _load_plan(plan_path, plan_sha)
        source_names = set(plan["source_files"]) | set(plan["archive_files"]) | {
            plan_path.resolve().relative_to(ROOT.resolve()).as_posix()}
        expected = {**plan["source_files"], **plan["archive_files"],
                    plan_path.resolve().relative_to(ROOT.resolve()).as_posix(): plan_sha}
        before = {name: _sha(ROOT / name) for name in sorted(source_names)}
        if before != expected:
            raise ValueError("preflight source/archive snapshot differs from pinned hashes")
        record["source_before"] = before
        old, e29, _preconditioner = qy.load_base(OLD_PLAN, OLD_PLAN_SHA)
        raw_diag = _load_diagnostic()
        problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
        control = torch.tensor(e29["accepted_control"], dtype=torch.float64)
        current_identity = seed._input_identity(problem, original, control, parameters, truth)
        if (current_identity != raw_diag.get("input_identity")
                or _tensor_sha(control) != CONTROL_SHA or _tensor_sha(parameters) != PARAMETERS_SHA):
            raise ValueError("reconstructed e29 full-objective problem or input differs from Qy32 receipt")
        identity = current_identity
        runtime = guard_policy.blocks.runtime_identity()
        if runtime != raw_diag.get("runtime"):
            raise ValueError("runtime differs from compressed Qy32 source diagnostic")
        p = parameters
        current_problem = problem
        eta0 = float(qy.production_qy32(current_problem, control))
        dt, direction, direction_stats = _direction_from_diagnostic(control, eta0, raw_diag)
        del raw_diag
        record.update(direction=direction_stats, eta0=eta0,
                      input_before=current_identity, runtime=runtime,
                      phase="baseline", numerical_status="not_reached")
        guard_policy._write(output, record)
        gradient_fn = torch.func.grad(current_problem.objective, argnums=0)
        base = qy._measure(current_problem, p, control, eta0, gradient_fn,
                           with_gradient=True, deadline=deadline)
        base_state = e29.get("current_state", {})
        base_checks: dict[str, Any] = {"objective": _metric(float(base_state["objective"]), base["objective"]),
                       "phi": _metric(float(base_state["phi"]), base["phi"]),
                       "gradient_inf": _metric(float(base_state["gradient_inf"]), base["gradient_inf"])}
        saved_gradient = torch.tensor(base_state["gradient"], dtype=torch.float64)
        fresh_gradient = torch.tensor(base["gradient"], dtype=torch.float64)
        base_checks["gradient_components"] = [
            _metric(float(a), float(b)) for a, b in zip(saved_gradient, fresh_gradient, strict=True)]
        if (base["branch"].get("status") != "passed_strict_branch"
                or base["branch"]["signature_sha256"] != base_state.get("branch", {}).get("signature_sha256")
                or any(not check["passed"] for key, check in base_checks.items()
                       if key != "gradient_components")
                or not all(item["passed"] for item in base_checks["gradient_components"])):
            raise ValueError("fresh baseline J/g/branch does not match e29 accepted receipt")
        base.pop("branch_signature", None)
        base.pop("static_flux_signs", None)
        g = fresh_gradient
        d = direction
        live_hvp, hvp_counts = counted_hvp(gradient_fn, control, p, d, deadline)
        try:
            hd = live_hvp()
        finally:
            record.update(hvp_calls=hvp_counts["started"],
                          hvp_calls_completed=hvp_counts["completed"])
        if hd.shape != (26,) or not bool(torch.isfinite(hd).all()):
            raise ValueError("single current-point HVP is nonfinite")
        record.update(base_objective=base["objective"], base_phi=base["phi"],
                      base_gradient=g.tolist(), base_geometry=base["geometry"],
                      base_branch=base["branch"], base_branch_partition=base["branch_partition"],
                      base_metric_checks=base_checks, H_direction=hd.tolist())
        guard_policy._write(output, record)
        descent = resolved_direction_checks(g, d, hd)
        if abs(descent["g_dot_d"] - EXPECTED_G_D) > 2e-8:
            raise ValueError("fresh baseline slope differs from independent tangent check")
        record.update(direction_checks=descent, phase="line_search")
        guard_policy._write(output, record)
        norm_d = float(torch.linalg.vector_norm(d))
        alpha0 = bounded_start_alpha(control, eta0, dt, norm_d)
        accepted: Tensor | None = None
        for index in range(MAX_CANDIDATES):
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second budget expired during chart line search")
            alpha = alpha0 * (0.5 ** index)
            try:
                candidate = chart_candidate(control, eta0, dt, alpha)
            except ValueError as error:
                record["trials"].append({"index": index, "alpha": alpha,
                    "status": "rejected", "reason": f"static chart domain refused: {error}",
                    "objective_evaluated": False, "new_hvp": 0})
                guard_policy._write(output, record)
                continue
            displacement = float(torch.linalg.vector_norm(candidate - control))
            row: dict[str, Any] = {"index": index, "alpha": alpha,
                                   "control_sha256": _tensor_sha(candidate),
                                   "displacement_l2": displacement,
                                   "chart_linear_displacement_l2": alpha * norm_d,
                                   "chart_curvature_remainder_l2": float(torch.linalg.vector_norm(
                                       candidate - control - alpha * d)),
                                   "new_hvp": 0}
            if displacement > RADIUS:
                row.update(status="rejected", reason="control-norm radius exceeded before FV evaluation")
                record["trials"].append(row)
                guard_policy._write(output, record)
                continue
            try:
                measured = qy._measure(current_problem, p, candidate, eta0, gradient_fn,
                                       with_gradient=True, deadline=deadline)
            except TimeoutError:
                raise
            row.update(objective=measured["objective"], phi=measured["phi"],
                       gradient=measured["gradient"], gradient_inf=measured["gradient_inf"],
                       branch=measured["branch"], branch_margins=measured["branch_margins"],
                       branch_partition=measured["branch_partition"],
                       geometry=measured["geometry"],
                       branch_signature_sha256=measured["branch"].get("signature_sha256"),
                       qy32=measured["realized_qy32"],
                       first_order_path_model={
                           "predicted_J": base["objective"] + alpha * descent["g_dot_d"],
                           "predicted_Phi": base["phi"] + alpha * descent["g_dot_Hd"],
                           "predicted_gradient": (g + alpha * hd).tolist(),
                           "gradient_prediction_error": (
                               torch.tensor(measured["gradient"], dtype=torch.float64)
                               - g - alpha * hd).tolist(),
                           "scope": "first-order model along nonlinear fixed-eta chart path"})
            strict = measured["branch"].get("status") == "passed_strict_branch"
            armijo = dual_armijo(base["objective"], base["phi"], measured["objective"],
                                 measured["phi"], alpha, descent["g_dot_d"], descent["g_dot_Hd"])
            row.update(armijo, strict_point_passed=strict)
            row["status"] = "accepted" if strict and armijo["accepted"] else "rejected"
            record["trials"].append(row)
            guard_policy._write(output, record)
            measured.pop("branch_signature", None)
            measured.pop("static_flux_signs", None)
            if row["status"] == "accepted":
                accepted = candidate
                record.update(numerical_status="candidate_tentative", candidate_committed=False)
                guard_policy._write(output, record)
                break
        if accepted is None:
            record.update(phase="finished", execution_status="completed",
                          numerical_status="no_chart_step_accepted")
        else:
            final_measure = qy._measure(current_problem, p, accepted, eta0, gradient_fn,
                                        with_gradient=True, deadline=deadline)
            last = record["trials"][-1]
            final_gradient = torch.tensor(final_measure["gradient"], dtype=torch.float64)
            recheck_passed = bool(final_measure["branch"].get("status") == "passed_strict_branch"
                    and final_measure["branch"].get("signature_sha256") == last.get("branch_signature_sha256")
                    and _metric(float(last["objective"]), final_measure["objective"])["passed"]
                    and _metric(float(last["phi"]), final_measure["phi"])["passed"]
                    and all(_metric(a, float(b))["passed"] for a, b in
                            zip(last["gradient"], final_measure["gradient"], strict=True)))
            if not recheck_passed:
                raise ValueError("fresh final accepted-point recheck differs from tentative chart step")
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second budget expired after final endpoint recheck")
            final_control_sha = _tensor_sha(accepted)
            source_after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            input_after = seed._input_identity(current_problem, original, accepted, p, truth)
            runtime_after = guard_policy.blocks.runtime_identity()
            intact = (source_after == before and _check_fixed_input(input_after, identity, final_control_sha)
                      and runtime_after == runtime)
            if time.monotonic() >= deadline:
                raise TimeoutError("240-second budget expired after final source/input/runtime closure")
            if not intact:
                raise ValueError("final tangent candidate failed source/input/runtime closure")
            record.update(phase="finished", execution_status="completed",
                          numerical_status="one_qy32_tangent_dual_armijo_step_accepted",
                          optimizer_steps_applied=1, candidate_committed=True,
                          accepted_control=accepted.tolist(), accepted_control_sha256=final_control_sha,
                          accepted_objective=final_measure["objective"], accepted_phi=final_measure["phi"],
                          accepted_gradient=final_gradient.tolist(), accepted_branch=_compact_measure(final_measure)["branch"],
                          accepted_branch_partition=final_measure["branch_partition"],
                          accepted_geometry=final_measure["geometry"],
                          accepted_displacement_l2=float(torch.linalg.vector_norm(accepted - control)),
                          accepted_gradient_l2=float(torch.linalg.vector_norm(final_gradient)),
                          accepted_gradient_inf=float(final_gradient.abs().max()),
                          source_after=source_after, source_unchanged=True,
                          input_after=input_after, fixed_input_unchanged=True,
                          runtime_after=runtime_after)
            final_measure.pop("branch_signature", None)
            final_measure.pop("static_flux_signs", None)
        if record.get("source_after") is None:
            finalize(None)
        if time.monotonic() >= deadline:
            raise TimeoutError("240-second budget expired before final receipt closure")
        if (record.get("source_unchanged") is not True
                or record.get("fixed_input_unchanged") is not True
                or record.get("runtime_after") != record.get("runtime")):
            raise ValueError("final tangent report failed source/input/runtime closure")
        if record.get("phase") != "finished":
            record.update(phase="finished", execution_status="completed")
        record["elapsed_seconds"] = time.monotonic() - started
        guard_policy._write(output, record)
        return record
    except TimeoutError as error:
        finalize(None)
        closure = (record.get("source_unchanged") is True
                   and record.get("fixed_input_unchanged") is True
                   and record.get("runtime_after") == record.get("runtime"))
        record.update(phase="finished", execution_status="completed" if closure else "failed",
                      numerical_status="budget_refusal" if closure else "integrity_refusal",
                      refusal=str(error), candidate_committed=False, optimizer_steps_applied=0)
        _mark_not_committed(record, str(error))
        guard_policy._write(output, record)
        if not closure:
            raise RuntimeError("budget-partial tangent receipt failed closure") from error
        return record
    except guard_policy.StepRefusal as error:
        finalize(None)
        closure = (record.get("source_unchanged") is True
                   and record.get("fixed_input_unchanged") is True
                   and record.get("runtime_after") == record.get("runtime"))
        _record_direction_refusal(record, str(error), closed=closure)
        guard_policy._write(output, record)
        if not closure:
            raise RuntimeError("direction refusal receipt failed closure") from error
        return record
    except Exception as error:
        finalize(None)
        record.update(phase="finished", execution_status="failed",
                      numerical_status="tangent_step_error", refusal=f"{type(error).__name__}: {error}",
                      candidate_committed=False, optimizer_steps_applied=0)
        _mark_not_committed(record, str(error))
        guard_policy._write(output, record)
        raise


def run(plan_path: Path, plan_sha: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if (plan_path != PLAN.resolve() or _sha(plan_path) != plan_sha):
        raise ValueError("caller tangent-step plan identity mismatch")
    _load_plan(plan_path, plan_sha)
    parent_path = output.with_suffix(".run.json")
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, parent_path)):
        raise ValueError("tangent-step report/resource/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
               "--plan", str(plan_path), "--plan-sha256", plan_sha, "--output", str(output)]
    resource_result = run_guarded_diagnostic(command, wall_seconds=WALL_SECONDS,
        rss_bytes=RSS_BYTES, report_path=resource, log_path=log)
    execution = guard_policy.execution_status(resource_result)
    parent: dict[str, Any] = {"execution_status": execution, "resource": resource_result,
                             "child_sha256": None, "child_read_error": None,
                             "numerical_status": "not_reached"}
    guard_policy._write(parent_path, parent)
    try:
        child = json.loads(output.read_text())
        if not isinstance(child, dict):
            raise ValueError("tangent-step child report must be a JSON object")
        parent.update(child_sha256=_sha(output), numerical_status=child.get("numerical_status"))
        closed = (child.get("phase") == "finished" and child.get("execution_status") == "completed"
                  and child.get("source_unchanged") is True and child.get("fixed_input_unchanged") is True
                  and child.get("runtime_after") == child.get("runtime")
                  and child.get("plan_sha256") == plan_sha and child.get("base_control_sha256") == CONTROL_SHA)
        if execution == "completed" and not closed:
            parent.update(execution_status="failed", execution_failure_reason="child closure/commit refused")
        if child.get("optimizer_steps_applied") == 1 and child.get("candidate_committed") is not True:
            parent.update(execution_status="failed", execution_failure_reason="candidate lacks final commit receipt")
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if execution == "completed" else execution,
                      child_read_error=f"{type(error).__name__}: {error}")
        guard_policy._write(parent_path, parent)
        raise RuntimeError(f"tangent child report could not be verified; durable parent: {parent_path}") from error
    guard_policy._write(parent_path, parent)
    if parent.get("execution_status") == "failed":
        raise RuntimeError(f"tangent child failed; durable receipts: {parent_path}, {output}")
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "qy32_tangent_step_attempt1/step.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "qy32_tangent_step_attempt1/step.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "qy32_tangent_step_attempt1/step.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output,
                             args.resource, args.log), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
