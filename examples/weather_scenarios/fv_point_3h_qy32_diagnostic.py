"""Bounded two-sided full-objective diagnostic around the Qy[3, 2] face.

This probes a fixed control chart only. It applies no optimization step and
makes no claim that the face is stationary or physically meaningful.
"""
from __future__ import annotations

import argparse
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

from examples.weather_scenarios import fv_diagnostic_guard as guard
from examples.weather_scenarios import fv_point_3h_inexact_continuation as inexact_parent
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_bounded_coupled_step as guarded_policy
from advar import transport, variational

EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "QY32_TWO_SIDED_DIAGNOSTIC_PLAN_20261007.json"
PRODUCING_PLAN = EVIDENCE / "F462_INEXACT_CONTINUATION_PLAN_20261007.json"
PRODUCING_PLAN_SHA = "499e89862d86c8e4258e5199194674714c9001c59fca00e5436c935d98a3e33b"
STEP = EVIDENCE / "f462_inexact_continuation_20261007_attempt1/step.json"
SELF = "examples/weather_scenarios/fv_point_3h_qy32_diagnostic.py"
TEST = "tests/test_fv_point_3h_qy32_diagnostic.py"
ENDPOINT_SHA = "e29c348d51e7ec2de23f3f74d1522a34f58bf0bb72fd56e63d9b89cadea15e37"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
STEP_SHA = "520573d9710d4cf261a7b28a9f8c5266805d349c598d9c0f815b108120a27b0d"
RUN_SHA = "9c1baeb693ef5a09e8e29af0fe5b86a13ef2cc253cc3df85386d8bbc729b6d9e"
RESOURCE_SHA = "66fbf90c3198c3298d6d4d7caade4fe99fa7f943dd2dfbc901a2406209e78ec7"
PIVOT = 24
ETA_VALUES = (-2.0e-6, -1.0e-6, 1.0e-6, 2.0e-6)
EXPECTED_FACE_WEIGHTS = (-0.08, -0.21, -0.10, -0.45)


def diagnostic_policy() -> dict[str, Any]:
    """Return the single-launch diagnostic resource and sampling contract."""
    return {"outer_seconds": 300.0, "internal_seconds": 240.0,
            "rss_bytes": 1024**3, "guarded_launches": 1,
            "eta_values": list(ETA_VALUES), "cost_only_eta_zero": True,
            "optimizer_steps": 0, "hvp_calls": 0, "pcg_solves": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_base(plan_path: Path, plan_sha256: str) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Load the pinned e29 endpoint after validating its producing continuation."""
    plan_path = plan_path.resolve()
    if (plan_path.is_symlink() or not plan_path.resolve().is_relative_to(ROOT.resolve())
            or _sha(plan_path) != plan_sha256 or plan_path != PLAN):
        raise ValueError("diagnostic producing-plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    if (plan.get("experiment_kind") != "qy32_two_sided_diagnostic"
            or plan.get("producing_plan") != PRODUCING_PLAN.relative_to(ROOT).as_posix()
            or plan.get("producing_plan_sha256") != PRODUCING_PLAN_SHA
            or plan.get("policy") != diagnostic_policy()):
        raise ValueError("diagnostic plan must name the fixed F462 producing continuation")
    sources, archives = plan.get("source_files"), plan.get("archive_files")
    if not isinstance(sources, dict) or not isinstance(archives, dict):
        raise ValueError("diagnostic plan must pin source and archive files")
    producer = json.loads(PRODUCING_PLAN.read_text())
    if not set(producer["source_files"]) <= set(sources):
        raise ValueError("diagnostic plan omits inherited full-objective and continuation sources")
    if any(sources[name] != digest for name, digest in producer["source_files"].items()):
        raise ValueError("diagnostic plan changed an inherited source pin")
    for name, digest in sources.items():
        path = ROOT / name
        if (Path(name).is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())
                or not path.is_file() or _sha(path) != digest):
            raise ValueError(f"diagnostic source pin changed: {name}")
    for name, digest in archives.items():
        path = ROOT / name
        if (Path(name).is_absolute() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())
                or not path.is_file() or _sha(path) != digest):
            raise ValueError(f"diagnostic archive pin changed: {name}")
    if sources.get(SELF) != _sha(ROOT / SELF) or sources.get(TEST) != _sha(ROOT / TEST):
        raise ValueError("diagnostic plan must pin its producer and focused tests")
    if archives.get(PRODUCING_PLAN.relative_to(ROOT).as_posix()) != PRODUCING_PLAN_SHA:
        raise ValueError("diagnostic archive pins omit the producing continuation plan")
    # Validate the declared producing run and its inherited F82 cache contract.
    _plan, _prior, preconditioner, _prior_plan = inexact_parent._validated_loader(
        PRODUCING_PLAN, PRODUCING_PLAN_SHA)
    raw_path = STEP
    for path, expected in ((raw_path, STEP_SHA), (raw_path.with_suffix(".run.json"), RUN_SHA),
                           (raw_path.with_suffix(".resource.json"), RESOURCE_SHA)):
        if (archives.get(path.relative_to(ROOT).as_posix()) != expected
                or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
                or _sha(path) != expected):
            raise ValueError(f"e29 producing artifact changed: {path.name}")
    raw = json.loads(raw_path.read_text())
    parent = json.loads(raw_path.with_suffix(".run.json").read_text())
    resource = json.loads(raw_path.with_suffix(".resource.json").read_text())
    iterations = raw.get("iterations")
    last = iterations[-1] if isinstance(iterations, list) and iterations else {}
    state = raw.get("current_state", {})
    accepted_trials = [trial for trial in raw.get("trials", [])
                       if trial.get("status") == "accepted"]
    if (raw.get("accepted_control_sha256") != ENDPOINT_SHA
            or raw.get("execution_status") != "completed"
            or raw.get("optimizer_steps_applied") != 2
            or raw.get("numerical_status") != "budget_refusal"
            or raw.get("plan_sha256") != PRODUCING_PLAN_SHA
            or raw.get("parameters_sha256") != PARAMETERS_SHA
            or raw.get("source_unchanged") is not True
            or raw.get("fixed_input_unchanged") is not True
            or raw.get("source_before") != raw.get("source_after")
            or raw.get("runtime") != raw.get("runtime_after")
            or raw.get("input_before", {}).get("archived_input")
            != raw.get("input_after", {}).get("archived_input")
            or not isinstance(iterations, list) or len(iterations) != 2
            or [item.get("status") for item in iterations] != ["accepted", "accepted"]
            or len(accepted_trials) != 2
            or last.get("accepted_control_sha256") != ENDPOINT_SHA
            or last.get("accepted_control") != raw.get("accepted_control")
            or last.get("accepted_objective") != state.get("objective")
            or last.get("accepted_phi") != state.get("phi")
            or last.get("accepted_gradient") != state.get("gradient")
            or last.get("accepted_gradient_inf") != state.get("gradient_inf")
            or state.get("branch", {}).get("status") != "passed_strict_branch"
            or state.get("branch", {}).get("euler_stages") != 3600
            or parent.get("execution_status") != "completed"
            or parent.get("child_sha256") != STEP_SHA
            or parent.get("completed_iterations") != 2
            or parent.get("hvp_calls") != raw.get("hvp_calls")
            or resource.get("exit_code") != 0
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("wall_limit_seconds") != 780.0
            or resource.get("rss_limit_bytes") != 1024**3
            or resource.get("elapsed_seconds", 781.0) > 780.0
            or resource.get("sampled_peak_rss_bytes", 1024**3 + 1) > 1024**3):
        raise ValueError("e29 endpoint is not the normally completed pinned two-step result")
    return plan, raw, preconditioner


def qy32(control: Tensor, weights: Tensor | None = None) -> Tensor:
    """Evaluate the declared Qy[3,2] face from the four active controls."""
    if control.shape != (26,) or control.dtype != torch.float64:
        raise ValueError("Qy[3,2] requires a length-26 FP64 control")
    w = control.new_tensor(EXPECTED_FACE_WEIGHTS) if weights is None else weights
    if w.shape != (4,):
        raise ValueError("Qy[3,2] weights must have length four")
    return torch.dot(w, torch.tanh(control[21:25]))


def production_fluxes(problem: Any, control: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate face flux arrays through the production coefficient and flux operators."""
    spec = problem.frozen.fv_transport
    if spec is None or control.shape != (26,):
        raise ValueError("production face evaluation requires the fixed FV control profile")
    nowcast = problem.frozen.nowcast_config
    coefficients = variational.bounded_fv_coefficients(
        control[20:25], psi_basis=spec.psi_basis,
        coefficient_limits=spec.coefficient_limits,
        dt_seconds=nowcast.interval_minutes * 60.0 / spec.substeps_per_interval,
        spacing_yx=spec.spacing_yx, reconstruction=spec.reconstruction,
        max_courant=spec.max_courant)
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    return transport.face_volume_fluxes(psi)


def production_qy32(problem: Any, control: Tensor) -> Tensor:
    """Evaluate Qy[3,2] through the same bounded coefficients and flux operator."""
    return production_fluxes(problem, control)[1][3, 2]


def face_weights(problem: Any) -> Tensor:
    """Derive the face coefficients from the loaded FV phase profile."""
    spec = problem.frozen.fv_transport
    if spec is None or spec.psi_basis.shape != (5, 5, 6):
        raise ValueError("loaded profile no longer matches the declared 4 by 5 FV cell grid")
    basis = spec.psi_basis
    weights = (-torch.diff(basis[:, 3, :], dim=-1)[:, 2] * spec.coefficient_limits)[1:]
    expected = weights.new_tensor(EXPECTED_FACE_WEIGHTS)
    if not torch.allclose(weights, expected, atol=2e-15, rtol=0.0):
        raise ValueError("Qy[3,2] geometry differs from the pinned phase profile")
    return weights


def chart(control: Tensor, eta: float) -> Tensor:
    """Restore c24 so the declared face has value eta at fixed other controls."""
    if control.shape != (26,) or not math.isfinite(eta):
        raise ValueError("chart needs a finite eta and a length-26 control")
    result = control.clone()
    numerator = (-eta - 0.08 * torch.tanh(control[21])
                 - 0.21 * torch.tanh(control[22]) - 0.10 * torch.tanh(control[23]))
    argument = numerator / 0.45
    if not bool(torch.isfinite(argument)) or not bool(argument.abs() < 1.0):
        raise ValueError("requested face coordinate lies outside the pivot chart domain")
    result[PIVOT] = torch.atanh(argument)
    return result


def chart_jacobian(control: Tensor, eta: float) -> tuple[Tensor, Tensor]:
    """Return retained-coordinate Jacobian Z and transverse chart vector."""
    point = chart(control, eta)
    retained = [index for index in range(26) if index != PIVOT]
    z = torch.zeros((26, 25), dtype=control.dtype, device=control.device)
    z[retained, torch.arange(25)] = 1.0
    slope = -1.0 / (0.45 * (1.0 - torch.tanh(point[PIVOT]).square()))
    # c24 depends on retained c21:c23; all other retained columns are direct.
    z[PIVOT, PIVOT - 3] = -0.08 * (1.0 - torch.tanh(point[21]).square()) / (
        0.45 * (1.0 - torch.tanh(point[PIVOT]).square()))
    z[PIVOT, PIVOT - 2] = -0.21 * (1.0 - torch.tanh(point[22]).square()) / (
        0.45 * (1.0 - torch.tanh(point[PIVOT]).square()))
    z[PIVOT, PIVOT - 1] = -0.10 * (1.0 - torch.tanh(point[23]).square()) / (
        0.45 * (1.0 - torch.tanh(point[PIVOT]).square()))
    gamma_eta = torch.zeros_like(control)
    gamma_eta[PIVOT] = slope
    return z, gamma_eta


def geometry(control: Tensor, eta: float, gradient: Tensor) -> dict[str, Any]:
    """Report tangent projection and chart-dependent/transverse derivatives."""
    weights = control.new_tensor(EXPECTED_FACE_WEIGHTS)
    normal = torch.zeros_like(control)
    normal[21:25] = weights * (1.0 - torch.tanh(control[21:25]).square())
    z, gamma_eta = chart_jacobian(control, eta)
    q, r = torch.linalg.qr(z, mode="reduced")
    tangent = q @ (q.T @ gradient)
    metric = z.T @ z
    eigenvalues = torch.linalg.eigvalsh(metric)
    norm_n = torch.linalg.vector_norm(normal)
    tangent_covector = z.T @ gradient
    z_condition = float(torch.linalg.cond(z))
    normal_tangent = gradient - normal * (torch.dot(normal, gradient) / torch.dot(normal, normal))
    return {"tangent_gradient_norm": float(torch.linalg.vector_norm(tangent)),
            "tangent_gradient_norm_projection": float(torch.linalg.vector_norm(normal_tangent)),
            "tangent_covector": tangent_covector.tolist(),
            "tangent_covector_inf": float(tangent_covector.abs().max()),
            "euclidean_unit_normal_gradient": float(torch.dot(normal, gradient) / norm_n),
            "intrinsic_normal_flux_slope": float(torch.dot(normal, gradient) / torch.dot(normal, normal)),
            "j_eta_fixed_t": float(torch.dot(gradient, gamma_eta)),
            "j_t_fixed_eta": tangent_covector.tolist(),
            "normal_norm": float(norm_n),
            "normal_dot_tangent_max_abs": float((normal @ z).abs().max()),
            "normal_dot_chart_eta": float(torch.dot(normal, gamma_eta)),
            "chart_jacobian_condition": z_condition,
            "tangent_metric_condition": float(eigenvalues[-1] / eigenvalues[0]),
            "tangent_qr_diagonal": torch.diagonal(r).abs().tolist()}


def _partition(signature: dict[str, Any], analysis_stages: int) -> dict[str, Any]:
    choices, signs = signature.get("choices"), signature.get("face_signs")
    if (not isinstance(choices, list) or not isinstance(signs, list)
            or len(choices) != len(signs) or len(choices) != 3600
            or analysis_stages <= 0 or analysis_stages >= len(choices)):
        raise ValueError("branch signature cannot be partitioned for the fixed profile")
    def part(start: int, stop: int) -> dict[str, Any]:
        return {"choices": choices[start:stop], "face_signs": signs[start:stop]}
    return {"analysis_stages": analysis_stages,
            "analysis_sha256": _digest(part(0, analysis_stages)),
            "future_stages": len(choices) - analysis_stages,
            "future_sha256": _digest(part(analysis_stages, len(choices))),
            "full_sha256": _digest(part(0, len(choices)))}


def _measure(problem: Any, parameters: Tensor, control: Tensor, eta: float,
             gradient_fn: Any, *, with_gradient: bool, deadline: float) -> dict[str, Any]:
    qx, qy = production_fluxes(problem, control)
    q = qy[3, 2]
    q_formula = qy32(control, face_weights(problem))
    result: dict[str, Any] = {"requested_eta": eta, "realized_qy32": float(q),
                              "formula_qy32": float(q_formula),
                              "control": control.tolist(), "control_sha256": _tensor_sha(control),
                              "static_flux_signs": {"qx": torch.sign(qx).to(torch.int8).tolist(),
                                                    "qy": torch.sign(qy).to(torch.int8).tolist()}}
    if not math.isclose(float(q), eta, rel_tol=0.0, abs_tol=2e-15):
        raise ValueError("chart point and actual FV face flux do not agree")
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired before objective")
    objective = problem.objective(control, parameters)
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired after objective")
    if objective.shape != () or not bool(torch.isfinite(objective)):
        raise ValueError("original full-control objective is not finite")
    result["objective"] = float(objective)
    if not with_gradient:
        result["gradient_computed"] = False
        return result
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired before gradient")
    gradient = gradient_fn(control, parameters)
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired after gradient")
    phi = torch.dot(gradient, gradient) / 2.0
    if gradient.shape != (26,) or not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(phi)):
        raise ValueError("original full-control gradient or Phi is not finite")
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired before strict branch")
    collector = seed._MarginCollector()
    try:
        with torch.no_grad(), seed.transport.observe_minmod_stages(collector):
            signature, scope = problem.branch_check(control, parameters)
    except ValueError as error:
        margins = collector.report()
        result.update(gradient_computed=True, gradient=gradient.tolist(),
                      gradient_sha256=_tensor_sha(gradient),
                      gradient_norm=float(torch.linalg.vector_norm(gradient)),
                      gradient_inf=float(gradient.abs().max()), phi=float(phi),
                      geometry=geometry(control, eta, gradient),
                      branch={"status": "branch_refused", "reason": str(error)},
                      branch_margins=margins, branch_scope=None, branch_partition=None,
                      branch_signature=None)
        return result
    if time.monotonic() >= deadline:
        raise TimeoutError("diagnostic internal time budget expired after strict branch")
    margins = collector.report()
    passed = (signature.get("euler_stages") == 3600
              and len(signature.get("choices", [])) == 3600
              and len(signature.get("face_signs", [])) == 3600
              and seed._valid_margins(margins, complete=True))
    branch = {"status": "passed_strict_branch" if passed else "branch_or_margin_refused",
              "euler_stages": signature.get("euler_stages"),
              "choice_stage_count": len(signature.get("choices", [])),
              "face_sign_stage_count": len(signature.get("face_signs", [])),
              "signature_sha256": _digest({"choices": signature.get("choices"),
                                           "face_signs": signature.get("face_signs")})}
    analysis_stages = 2 * (len(problem.layout["observation_times_seconds"]) - 1) * problem.frozen.fv_transport.substeps_per_interval
    result.update(gradient_computed=True, gradient=gradient.tolist(),
                  gradient_sha256=_tensor_sha(gradient), gradient_norm=float(torch.linalg.vector_norm(gradient)),
                  gradient_inf=float(gradient.abs().max()), phi=float(phi),
                  geometry=geometry(control, eta, gradient), branch=branch,
                  branch_margins=margins, branch_scope=scope,
                  branch_partition=_partition(signature, analysis_stages),
                  branch_signature={"choices": signature.get("choices"),
                                    "face_signs": signature.get("face_signs")})
    return result


def _run_child(plan_path: Path, plan_sha256: str, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + diagnostic_policy()["internal_seconds"]
    report: dict[str, Any] = {
        "schema": "advar.point3h-qy32-two-sided-diagnostic.v1", "phase": "running",
        "execution_status": "running", "numerical_status": "setup_not_complete",
        "plan_sha256": plan_sha256, "base_control_sha256": ENDPOINT_SHA,
        "parameters_sha256": PARAMETERS_SHA, "policy": diagnostic_policy(),
        "source_before": None, "source_after": None, "source_unchanged": None,
        "input_before": None, "input_after": None, "fixed_input_unchanged": None,
        "runtime": None, "runtime_after": None, "eta_zero_cost_only": None,
        "samples": [], "optimizer_steps_applied": 0, "hvp_calls": 0, "pcg_solves": 0,
        "score_computed": False, "response_computed": False,
        "stationary_root_claim": False, "normal_root_claim": False,
    }
    guarded_policy._write(output, report)
    source_names: set[str] = set()
    problem: Any | None = None
    original: Tensor | None = None
    base: Tensor | None = None
    parameters: Tensor | None = None
    truth: Tensor | None = None

    def finalize() -> None:
        if source_names:
            source_after = {name: _sha(ROOT / name) for name in sorted(source_names)}
            report["source_after"] = source_after
            report["source_unchanged"] = report["source_before"] == source_after
        if all(value is not None for value in (problem, original, base, parameters, truth)):
            report["input_after"] = seed._input_identity(
                cast(Any, problem), cast(Tensor, original), cast(Tensor, base),
                cast(Tensor, parameters), cast(Tensor, truth))
        else:
            report["input_after"] = report.get("input_before")
        report["fixed_input_unchanged"] = (
            None if report["input_before"] is None
            else report["input_after"] == report["input_before"])
        report["runtime_after"] = guarded_policy.blocks.runtime_identity()
        report["elapsed_seconds"] = time.monotonic() - started

    try:
        plan, raw, _preconditioner = load_base(plan_path, plan_sha256)
        source_names = set(plan["source_files"]) | set(plan["archive_files"]) | {
            plan_path.relative_to(ROOT).as_posix()}
        expected = {**plan["source_files"], **plan["archive_files"],
                    plan_path.relative_to(ROOT).as_posix(): plan_sha256}
        before = {name: _sha(ROOT / name) for name in sorted(source_names)}
        if before != expected:
            raise ValueError("prelaunch source/archive snapshot differs from pinned hashes")
        report["source_before"] = before
        problem, original, _seed_control, parameters, truth, _identity = seed._prepare_fixed_seed()
        base = torch.tensor(raw["accepted_control"], dtype=torch.float64)
        endpoint_input = seed._input_identity(problem, original, base, parameters, truth)
        runtime = guarded_policy.blocks.runtime_identity()
        if (parameters.shape != (13,) or parameters.dtype != torch.float64
                or _tensor_sha(parameters) != PARAMETERS_SHA
                or _tensor_sha(base) != ENDPOINT_SHA
                or endpoint_input != raw.get("input_after")
                or runtime != raw.get("runtime")):
            raise ValueError("reconstructed original fixed problem/runtime differs from e29 receipt")
        report.update(input_before=endpoint_input, runtime=runtime,
                      input_identity=endpoint_input)
        weights = face_weights(problem)
        if not math.isclose(float(production_qy32(problem, base)), -5.91061e-7,
                             rel_tol=0.0, abs_tol=3e-12):
            raise ValueError("e29 base is not the pinned near-zero Qy[3,2] endpoint")
        report["geometry_weights"] = weights.tolist()
        current_problem = cast(Any, problem)
        current_parameters = cast(Tensor, parameters)
        gradient_fn = torch.func.grad(current_problem.objective, argnums=0)
        zero = _measure(current_problem, current_parameters, chart(base, 0.0), 0.0, gradient_fn,
                        with_gradient=False, deadline=deadline)
        report["eta_zero_cost_only"] = zero
        guarded_policy._write(output, report)
        for eta in ETA_VALUES:
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic internal time budget expired")
            sample = _measure(current_problem, current_parameters, chart(base, eta), eta, gradient_fn,
                              with_gradient=True, deadline=deadline)
            report["samples"].append(sample)
            guarded_policy._write(output, report)
        if time.monotonic() >= deadline:
            raise TimeoutError("diagnostic internal time budget expired after final sample")
        samples = report["samples"]
        reference_signature = samples[0]["branch_signature"]
        reference_flux = samples[0]["static_flux_signs"]
        for sample in samples:
            signature = sample["branch_signature"]
            sample["branch_stage_changes_vs_eta_minus_2e-6"] = (
                None if reference_signature is None or signature is None else {
                    key: sum(left != right for left, right in
                             zip(reference_signature[key], signature[key], strict=True))
                    for key in ("choices", "face_signs")})
            flux = sample["static_flux_signs"]
            qx_changes = sum(left != right for left_row, right_row in
                             zip(reference_flux["qx"], flux["qx"], strict=True)
                             for left, right in zip(left_row, right_row, strict=True))
            qy_changes = sum(left != right for row_index, (left_row, right_row) in
                             enumerate(zip(reference_flux["qy"], flux["qy"], strict=True))
                             for col_index, (left, right) in enumerate(zip(left_row, right_row, strict=True))
                             if (row_index, col_index) != (3, 2))
            sample["other_static_face_sign_changes_vs_eta_minus_2e-6"] = {
                "qx_faces": qx_changes, "qy_faces_excluding_qy32": qy_changes}
        finalize()
        report.update(phase="finished", execution_status="completed",
                      numerical_status="finite_chart_samples")
        if report["source_unchanged"] is not True or report["fixed_input_unchanged"] is not True \
                or report["runtime_after"] != report["runtime"]:
            raise ValueError("source, fixed input, or runtime changed during diagnostic")
        guarded_policy._write(output, report)
        return report
    except TimeoutError as error:
        if problem is not None:
            finalize()
        closure_ok = (report["source_unchanged"] is True
                      and report["fixed_input_unchanged"] is True
                      and report["runtime_after"] == report["runtime"])
        report.update(phase="finished",
                      execution_status="completed" if closure_ok else "failed",
                      numerical_status="budget_refusal" if closure_ok else "diagnostic_error",
                      refusal=str(error) if closure_ok else f"{error}; partial receipt closure failed")
        guarded_policy._write(output, report)
        if not closure_ok:
            raise RuntimeError("partial diagnostic receipt failed its source/input/runtime closure") from error
        return report
    except Exception as error:
        if problem is not None:
            finalize()
        report.update(phase="finished", execution_status="failed",
                      numerical_status="diagnostic_error",
                      refusal=f"{type(error).__name__}: {error}")
        guarded_policy._write(output, report)
        raise


def run(plan_path: Path, plan_sha256: str, output: Path, resource: Path,
        log: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if (not plan_path.is_relative_to(ROOT.resolve()) or _sha(plan_path) != plan_sha256):
        raise ValueError("caller diagnostic plan identity mismatch")
    load_base(plan_path, plan_sha256)
    parent_path = output.with_suffix(".run.json")
    if any(path.exists() or path.is_symlink() for path in (output, resource, log, parent_path)):
        raise ValueError("diagnostic output, parent, resource, and log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), "--child",
               "--plan", str(plan_path), "--plan-sha256", plan_sha256,
               "--output", str(output)]
    resource_result = guard.run_guarded_diagnostic(
        command, wall_seconds=300.0, rss_bytes=1024**3,
        report_path=resource, log_path=log)
    execution = guarded_policy.execution_status(resource_result)
    parent: dict[str, Any] = {"execution_status": execution, "resource": resource_result,
                             "child_read_error": None, "numerical_status": "not_reached",
                             "child_sha256": None}
    # Save process/resource evidence before opening the child report.
    guarded_policy._write(parent_path, parent)
    child: dict[str, Any] | None = None
    try:
        loaded = json.loads(output.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("diagnostic child report must be a JSON object")
        child = loaded
        parent.update(child_sha256=_sha(output), numerical_status=child.get("numerical_status", "not_reached"))
        complete_child = (child.get("phase") == "finished"
                          and child.get("execution_status") == "completed"
                          and child.get("plan_sha256") == plan_sha256
                          and child.get("base_control_sha256") == ENDPOINT_SHA
                          and child.get("source_unchanged") is True
                          and child.get("fixed_input_unchanged") is True
                          and child.get("runtime_after") == child.get("runtime"))
        if child.get("numerical_status") == "finite_chart_samples":
            complete_child = (complete_child and len(child.get("samples", [])) == len(ETA_VALUES)
                              and child.get("eta_zero_cost_only", {}).get("gradient_computed") is False)
        elif child.get("numerical_status") == "budget_refusal":
            complete_child = complete_child and len(child.get("samples", [])) <= len(ETA_VALUES)
        if execution == "completed" and not complete_child:
            parent.update(execution_status="failed",
                          execution_failure_reason="child_completion_or_integrity_not_verified")
    except (OSError, ValueError) as error:
        parent.update(execution_status="failed" if execution == "completed" else execution,
                      child_read_error=f"{type(error).__name__}: {error}")
    guarded_policy._write(parent_path, parent)
    if child is None:
        raise RuntimeError(f"diagnostic child report could not be verified; parent receipt: {parent_path}")
    if (parent.get("execution_status") == "failed"
            or (execution == "completed" and child.get("execution_status") != "completed")
            or child.get("numerical_status") == "diagnostic_error"):
        raise RuntimeError(f"diagnostic child failed; durable receipts: {parent_path}, {output}")
    return {"parent": parent, "child": child}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.json")
    parser.add_argument("--resource", type=Path, default=EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.resource.json")
    parser.add_argument("--log", type=Path, default=EVIDENCE / "qy32_diagnostic_attempt1/diagnostic.log")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    plan_sha = args.plan_sha256
    args.plan = args.plan.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.child:
        _run_child(args.plan, plan_sha, args.output)
    else:
        print(json.dumps(run(args.plan, plan_sha, args.output, args.resource, args.log),
                         indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
