"""Fresh Hessian diagnostic at the accepted f82c full-control endpoint.

This module records curvature only. It does not take a step or compute a
response, score, stationary root, or physical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_diagnostic_guard
from examples.weather_scenarios import fv_hessian_checkpoint as checkpoint_probe
from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
STEP_DIR = EVIDENCE / "bounded_coupled_original_j_20261005_attempt1"
STEP_RAW = STEP_DIR / "step.json"
STEP_PARENT = STEP_DIR / "step.run.json"
STEP_RESOURCE = STEP_DIR / "step.resource.json"
PLAN = EVIDENCE / "F82C_CURVATURE_PLAN_20261005.json"
SELF = "examples/weather_scenarios/fv_point_3h_accepted_curvature.py"
TEST = "tests/test_fv_point_3h_accepted_curvature.py"
R9_ARCHIVE = EVIDENCE / "coupled_original_j_continuation_attempt1/coupled_original_j_continuation.json"
STEP_RAW_SHA = "ec289275e29d106685cb3a76b10b48ed96676411b84582b649ac944040d7d361"
STEP_PARENT_SHA = "e9b0645b1687ced1c57d62dd8a723dc82d9dd6255a8b5523bdad2c96b90bcc81"
STEP_RESOURCE_SHA = "643691ce857a4b404cc5870be3d86e654063f5bf33cee64174c06a62bbb604db"
CONTROL_SHA = "f82cc3412a9ec8cb3896412d4a7bf9703e96cccae9693bc1f61b947914e5b359"
PARAMETERS_SHA = "8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed"
EXPECTED = {"objective": 0.07811087971491998, "phi": 1.7670579890832934,
            "gradient_inf": 1.1815341609110253}
WALL_SECONDS, INTERNAL_SECONDS, RSS_BYTES = 300.0, 240.0, 1024**3
KERNEL_ATTEMPTS = 3


class CurvatureRefusal(RuntimeError):
    """A measured endpoint failed its declared strict-branch or metric gate."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _write(path: Path, record: dict[str, Any]) -> None:
    fv_diagnostic_guard.atomic_write_text(
        path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _required_sources(r9: dict[str, Any]) -> set[str]:
    return set(r9["source_before"]) | {
        SELF, TEST,
        "examples/weather_scenarios/fv_hessian_checkpoint.py",
        "tests/test_fv_hessian_checkpoint.py",
        "examples/weather_scenarios/fv_diagnostic_guard.py",
        "tests/test_fv_diagnostic_guard.py",
        "examples/weather_scenarios/fv_point_3h_accepted_endpoint_audit.py",
        "examples/weather_scenarios/fv_point_3h_merit_continuation.py",
        "examples/weather_scenarios/fv_point_3h_seed_linear_probe.py",
        "examples/weather_scenarios/fv_point_3h_terminal_block_schur.py",
        "tests/test_fv_point_3h_terminal_block_schur.py",
    }


def _accepted_trial(step: dict[str, Any]) -> dict[str, Any]:
    accepted = [item for item in step.get("trials", []) if item.get("status") == "accepted"]
    if (len(accepted) != 1 or accepted[0].get("control_sha256") != CONTROL_SHA
            or step.get("accepted_control_sha256") != CONTROL_SHA
            or accepted[0].get("branch", {}).get("status") != "passed_strict_branch"
            or accepted[0].get("strict_point_passed") is not True
            or accepted[0].get("armijo_passed") is not True):
        raise ValueError("step report does not have exactly one accepted strict f82c endpoint")
    return accepted[0]


def _endpoint_metrics(accepted: dict[str, Any], objective: Tensor,
                      gradient: Tensor, phi: Tensor) -> dict[str, Any]:
    saved_gradient = torch.tensor(accepted.get("gradient"), dtype=torch.float64)
    if (objective.shape != () or phi.shape != () or gradient.shape != (26,)
            or objective.dtype != torch.float64 or phi.dtype != torch.float64
            or gradient.dtype != torch.float64 or objective.device.type != "cpu"
            or phi.device.type != "cpu" or gradient.device.type != "cpu"
            or saved_gradient.shape != (26,) or not bool(torch.isfinite(objective))
            or not bool(torch.isfinite(phi)) or not bool(torch.isfinite(gradient).all())
            or not bool(torch.isfinite(saved_gradient).all())):
        raise ValueError("endpoint metric comparison needs finite CPU FP64 scalar/scalar/26-vector")
    checks: dict[str, Any] = {
        "objective": audit._metric(float(accepted["objective"]), float(objective)),
        "phi": audit._metric(float(accepted["phi"]), float(phi)),
        "gradient_inf": audit._metric(float(accepted["gradient_inf"]), float(gradient.abs().max())),
        "gradient_components": [audit._metric(float(a), float(b)) for a, b in
                                zip(saved_gradient, gradient, strict=True)],
    }
    return checks


def _endpoint_metrics_pass(checks: dict[str, Any]) -> bool:
    return (all(isinstance(checks.get(key), dict) and checks[key].get("passed") is True
                for key in ("objective", "phi", "gradient_inf"))
            and isinstance(checks.get("gradient_components"), list)
            and len(checks["gradient_components"]) == 26
            and all(isinstance(item, dict) and item.get("passed") is True
                    for item in checks["gradient_components"]))


def _branch_partition(signature: dict[str, Any], observation_times: tuple[float, ...],
                      substeps_per_interval: int) -> dict[str, Any]:
    choices, signs = signature.get("choices"), signature.get("face_signs")
    analysis_stages = 2 * (len(observation_times) - 1) * substeps_per_interval
    if (not isinstance(choices, list) or not isinstance(signs, list)
            or len(choices) != len(signs) or analysis_stages <= 0
            or len(choices) <= analysis_stages):
        raise CurvatureRefusal("strict branch signature cannot be partitioned into analysis/future stages")

    def digest(values: dict[str, Any]) -> str:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    analysis = {"choices": choices[:analysis_stages], "face_signs": signs[:analysis_stages]}
    future = {"choices": choices[analysis_stages:], "face_signs": signs[analysis_stages:]}
    complete = {"choices": choices, "face_signs": signs}
    return {"analysis_stages": analysis_stages,
            "future_stages": len(choices) - analysis_stages,
            "full_signature_sha256": digest(complete),
            "analysis_signature_sha256": digest(analysis),
            "future_signature_sha256": digest(future)}


def _load_inputs(plan_path: Path, plan_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not plan_path.resolve().is_relative_to(ROOT) or sha(plan_path) != plan_sha256:
        raise ValueError("caller curvature plan identity mismatch")
    for path, expected in ((STEP_RAW, STEP_RAW_SHA), (STEP_PARENT, STEP_PARENT_SHA),
                           (STEP_RESOURCE, STEP_RESOURCE_SHA)):
        if sha(path) != expected:
            raise ValueError(f"pinned accepted-step artifact changed: {path.name}")
    step, parent, resource = (json.loads(path.read_text())
                              for path in (STEP_RAW, STEP_PARENT, STEP_RESOURCE))
    r9 = json.loads(R9_ARCHIVE.read_text())
    plan = json.loads(plan_path.read_text())
    archives = plan.get("archive_files")
    if (not isinstance(archives, dict)
            or archives.get(STEP_RAW.relative_to(ROOT).as_posix()) != STEP_RAW_SHA
            or archives.get(STEP_PARENT.relative_to(ROOT).as_posix()) != STEP_PARENT_SHA
            or archives.get(STEP_RESOURCE.relative_to(ROOT).as_posix()) != STEP_RESOURCE_SHA
            or sha(R9_ARCHIVE) != archives.get(R9_ARCHIVE.relative_to(ROOT).as_posix())):
        raise ValueError("curvature plan does not pin the accepted step and original objective archive")
    pins = plan.get("source_files")
    if not isinstance(pins, dict) or not _required_sources(r9) <= set(pins):
        raise ValueError("curvature plan is missing source and focused-test pins")
    for name, digest in pins.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != digest:
            raise ValueError(f"curvature source pin changed: {name}")
    if any((ROOT / name).is_symlink() for name in pins):
        raise ValueError("curvature source pins cannot name symlinks")
    if (step.get("phase") != "finished" or step.get("source_unchanged") is not True
            or step.get("source_before") != step.get("source_after")
            or step.get("numerical_status") != "one_original_J_step_accepted"
            or step.get("optimizer_steps_applied") != 1
            or step.get("response_computed") is not False
            or step.get("score_computed") is not False
            or parent.get("execution_status") != "completed"
            or parent.get("child_read_error") is not None
            or parent.get("child_sha256") != STEP_RAW_SHA
            or resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("received_sigterm") is not False
            or resource.get("monitor_error") is not None
            or not math.isfinite(resource.get("elapsed_seconds", math.nan))
            or resource["elapsed_seconds"] > WALL_SECONDS
            or resource.get("wall_limit_seconds") != WALL_SECONDS
            or resource.get("rss_limit_bytes") != RSS_BYTES
        or resource.get("sampled_peak_rss_bytes", math.inf) > RSS_BYTES):
        raise ValueError("pinned f82c step did not finish successfully under its declared guard")
    if (step.get("accepted_control_sha256") != CONTROL_SHA
            or step.get("parameters_sha256") != PARAMETERS_SHA
            or step.get("fixed_input_unchanged") is not True
            or step.get("runtime") != step.get("runtime_after")):
        raise ValueError("accepted f82c control/parameters/runtime identity changed")
    policy = plan.get("policy", {})
    if (plan.get("control_sha256") != CONTROL_SHA or plan.get("parameters_sha256") != PARAMETERS_SHA
            or policy.get("max_kernel_attempts") != KERNEL_ATTEMPTS
            or policy.get("kernel_attempt_seconds") != INTERNAL_SECONDS
            or policy.get("internal_deadline_seconds") != INTERNAL_SECONDS
            or policy.get("per_launch_wall_seconds") != WALL_SECONDS
            or policy.get("sampled_child_rss_bytes") != RSS_BYTES
            or policy.get("required_columns") != 26
            or policy.get("independent_audit_products") != 1):
        raise ValueError("curvature plan policy or endpoint tensor pins changed")
    return plan, step, _accepted_trial(step), r9


def run(*, plan_path: Path, plan_sha256: str, output: Path,
        checkpoint_directory: Path) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    if output.exists() or output.is_symlink():
        raise ValueError("curvature output must be fresh")
    if checkpoint_directory.resolve().is_relative_to(output.parent.resolve()):
        raise ValueError("resumable checkpoint directory must be outside the per-attempt output directory")
    plan, step, accepted, r9 = _load_inputs(plan_path, plan_sha256)
    plan_name = plan_path.resolve().relative_to(ROOT).as_posix()
    source_names = sorted(set(r9["source_before"]) | set(plan["source_files"])
                          | set(plan["archive_files"]) | {plan_name})
    before = {name: sha(ROOT / name) for name in source_names}
    if before[plan_name] != plan_sha256:
        raise ValueError("curvature plan changed after preflight")
    if any(before[name] != digest for name, digest in r9["source_before"].items()):
        raise ValueError("one or more original full-objective dependencies changed")
    control = torch.tensor(step["accepted_control"], dtype=torch.float64)
    parameters = torch.tensor(step["parameters"], dtype=torch.float64)
    if tensor_sha(control) != CONTROL_SHA or tensor_sha(parameters) != PARAMETERS_SHA:
        raise ValueError("f82c endpoint tensors do not match their pinned identities")
    runtime = blocks.runtime_identity()
    if runtime != step.get("runtime"):
        raise ValueError("runtime differs from the accepted-step child")
    problem, original, _seed_control, fixed_parameters, truth, base_identity = seed._prepare_fixed_seed()
    if not torch.equal(parameters, fixed_parameters):
        raise ValueError("reconstructed original parameter vector changed")
    input_before = seed._input_identity(problem, original, control, fixed_parameters, truth)
    if input_before != step.get("input_after") or base_identity.get("archived_input") != input_before.get("archived_input"):
        raise ValueError("reconstructed fixed problem/input differs from the accepted endpoint")

    report: dict[str, Any] = {
        "schema": "advar.point3h-accepted-curvature.v1", "phase": "running",
        "execution_status": "running", "numerical_status": "curvature_not_measured",
        "scope": "fresh original full-control J Hessian at accepted f82c point; no step/response/score/physical claim",
        "step_sha256": STEP_RAW_SHA, "step_parent_sha256": STEP_PARENT_SHA,
        "step_resource_sha256": STEP_RESOURCE_SHA,
        "original_objective_archive_sha256": sha(R9_ARCHIVE), "plan_sha256": plan_sha256,
        "source_before": before, "source_after": None, "source_unchanged": None,
        "runtime": runtime, "runtime_after": None, "input_before": input_before,
        "input_after": None, "control": control.tolist(), "control_sha256": tensor_sha(control),
        "parameters": parameters.tolist(), "parameters_sha256": tensor_sha(parameters),
        "endpoint_branch": accepted["branch"], "endpoint_margins": accepted["margins"],
        "branch_signature_changed_from_step_base": accepted.get("branch_signature_changed"),
        "objective": None, "phi": None, "gradient": None, "gradient_blocks": None,
        "curvature": None, "checkpoint_progress": None,
        "hvp_columns_completed": 0, "hvp_calls_total": 0,
        "response_computed": False, "optimizer_step_applied": False,
        "score_computed": False, "full_root_claim": False, "minimum_claim": False,
        "physical_validated": False, "internal_budget_seconds": INTERNAL_SECONDS,
        "kernel_attempt_seconds": INTERNAL_SECONDS, "kernel_attempts_max": KERNEL_ATTEMPTS,
        "outer_wall_seconds": WALL_SECONDS, "rss_limit_bytes": RSS_BYTES,
    }
    _write(output, report)

    def check_setup_budget(label: str) -> None:
        if time.monotonic() >= deadline:
            report.update(numerical_status="setup_budget_refused",
                          refusal=f"internal deadline expired {label}")
            raise CurvatureRefusal(f"internal deadline expired {label}")

    try:
        check_setup_budget("before strict endpoint branch")
        branch, margins = tail._full_current_branch(problem, control, fixed_parameters)
        check_setup_budget("after strict endpoint branch")
        if (branch.get("status") != "passed_strict_branch" or branch.get("euler_stages") != 3600
                or branch.get("choice_stage_count") != 3600 or branch.get("face_sign_stage_count") != 3600
                or not seed._valid_margins(margins, complete=True)
                or branch.get("signature_sha256") != accepted["branch"].get("signature_sha256")):
            raise CurvatureRefusal("fresh endpoint strict branch/margins differ from accepted f82c report")
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        check_setup_budget("before endpoint J/g/Phi evaluation")
        fresh = tail._fresh_merit(problem, control, fixed_parameters, gradient_fn)
        check_setup_budget("after endpoint J/g/Phi evaluation")
        if fresh is None:
            raise CurvatureRefusal("fresh endpoint original J/g/Phi is nonfinite")
        objective, gradient, phi = fresh
        metrics = _endpoint_metrics(accepted, objective, gradient, phi)
        report.update(objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
                      gradient_inf=float(gradient.abs().max()), gradient_sha256=tensor_sha(gradient),
                      gradient_blocks=blocks.gradient_blocks(gradient), endpoint_metric_checks=metrics,
                      endpoint_branch=branch, endpoint_margins=margins)
        _write(output, report)
        if (not _endpoint_metrics_pass(metrics)
                or not audit._metric(EXPECTED["objective"], float(objective))["passed"]
                or not audit._metric(EXPECTED["phi"], float(phi))["passed"]
                or not audit._metric(EXPECTED["gradient_inf"], float(gradient.abs().max()))["passed"]):
            report.update(numerical_status="endpoint_metric_refusal", endpoint_metric_checks=metrics)
            raise CurvatureRefusal("fresh endpoint J/Phi/full-gradient differs from pinned f82c measurements")
        check_setup_budget("before complete branch signature capture")
        full_signature, branch_scope = problem.branch_check(control, fixed_parameters)
        check_setup_budget("after complete branch signature capture")
        branch_partition = _branch_partition(
            full_signature, tuple(problem.layout["observation_times_seconds"]),
            problem.frozen.fv_transport.substeps_per_interval)
        if branch_partition["full_signature_sha256"] != branch.get("signature_sha256"):
            raise CurvatureRefusal("independent complete branch signature differs from strict endpoint check")
        report.update(objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
                      gradient_inf=float(gradient.abs().max()), gradient_blocks=blocks.gradient_blocks(gradient),
                      gradient_sha256=tensor_sha(gradient), endpoint_branch=branch,
                      branch_scope=branch_scope, branch_partition=branch_partition,
                      endpoint_margins=margins)
        _write(output, report)

        def receipt_provider() -> dict[str, Any]:
            live = seed._input_identity(problem, original, control, fixed_parameters, truth)
            current_sources = {name: sha(ROOT / name) for name in source_names}
            return {
                "objective": {"definition": "problem.objective full-control J(c,p), including original 26 priors",
                              "point_objective": float(objective), "point_phi": float(phi)},
                "problem": live,
                "coordinates": {"control_sha256": tensor_sha(control),
                                "parameters_sha256": tensor_sha(fixed_parameters),
                                "dimension": 26, "blocks": {"field": [0, 20], "dynamics": [20, 26]}},
                "source": current_sources,
                "runtime": blocks.runtime_identity(),
                "branch": {"signature_sha256": branch["signature_sha256"],
                           "status": branch["status"], "euler_stages": branch["euler_stages"],
                           "partition": branch_partition, "margins": margins},
                "analysis_context": {"plan_sha256": plan_sha256, "step_sha256": STEP_RAW_SHA,
                                     "operator": "original full-control Hessian of J(c,p)",
                                     "purpose": "accepted-endpoint curvature only"},
            }

        check_setup_budget("before Hessian checkpoint attempt")
        try:
            result = checkpoint_probe.checkpoint_hessian(
                problem.objective, control, fixed_parameters, gradient,
                receipt_provider=receipt_provider, checkpoint_dir=checkpoint_directory,
                deadline=deadline, max_attempts=KERNEL_ATTEMPTS,
                attempt_seconds=INTERNAL_SECONDS)
        except checkpoint_probe.BudgetRefusal as error:
            report.update(numerical_status="curvature_budget_refused", checkpoint_progress=error.state,
                          hvp_columns_completed=error.state.get("completed_columns", 0),
                          hvp_calls_total=error.state.get("hvp_calls_total", 0),
                          refusal=f"{type(error).__name__}: {error}")
        except checkpoint_probe.NumericalRefusal as error:
            report.update(numerical_status="curvature_numerical_refusal", checkpoint_progress=error.state,
                          hvp_columns_completed=error.state.get("completed_columns", 0),
                          hvp_calls_total=error.state.get("hvp_calls_total", 0),
                          refusal=f"{type(error).__name__}: {error}")
        except checkpoint_probe.ReceiptRefusal as error:
            report.update(numerical_status="curvature_receipt_refusal", checkpoint_progress=error.state,
                          hvp_columns_completed=error.state.get("completed_columns", 0),
                          hvp_calls_total=error.state.get("hvp_calls_total", 0),
                          refusal=f"{type(error).__name__}: {error}")
        else:
            report.update(numerical_status="curvature_completed",
                          checkpoint_progress={"status": "completed", "attempts_reserved": result["attempts_reserved"],
                                               "hvp_columns": result["hvp_columns"],
                                               "hvp_calls_total": result["hvp_calls_total"],
                                               "checkpoint_dir": result["checkpoint_dir"]},
                          hvp_columns_completed=result["hvp_columns"],
                          hvp_calls_total=result["hvp_calls_total"],
                          curvature={"hessian_sha256": result["hessian_sha256"],
                                     "hessian": result["hessian"],
                                     "hvp_columns": result["hvp_columns"],
                                     "hvp_calls_total": result["hvp_calls_total"],
                                     "symmetry_relative": result["symmetry_relative"],
                                     "eigenvalues": result["eigenvalues"],
                                     "minimum_eigenpair_audit": result["minimum_eigenpair_audit"],
                                     "block_schur": result["block_schur"]})
    except CurvatureRefusal as error:
        refusal = f"{type(error).__name__}: {error}"
        if report["numerical_status"] == "curvature_not_measured":
            report["numerical_status"] = "endpoint_refused"
        report["refusal"] = refusal

    input_after = seed._input_identity(problem, original, control, fixed_parameters, truth)
    after = {name: sha(ROOT / name) for name in source_names}
    runtime_after = blocks.runtime_identity()
    intact = (before == after and runtime_after == runtime and input_after == input_before)
    report.update(phase="finished" if intact else "integrity_refused",
                  execution_status="completed" if intact else "failed",
                  source_after=after, source_unchanged=before == after,
                  input_after=input_after, runtime_after=runtime_after,
                  fixed_input_unchanged=input_after == input_before,
                  elapsed_seconds=time.monotonic() - started)
    if not intact:
        report.update(numerical_status="integrity_refusal", refusal="source/input/runtime changed during curvature collection")
    _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.output_directory.resolve()
    if not args.child:
        raise ValueError("the experiment coordinator must invoke this child under the guarded runner")
    run(plan_path=args.plan.resolve(), plan_sha256=args.plan_sha256,
        output=directory / "audit.json", checkpoint_directory=args.checkpoint_directory.resolve())


if __name__ == "__main__":
    main()
