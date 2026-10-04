"""Read-only current-point J/gradient/branch and full-H/Schur diagnostic."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_accepted_endpoint_audit as audit
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail
from examples.weather_scenarios import fv_point_3h_precision_hessian_audit as hess_probe
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios import fv_hessian_checkpoint as checkpoint_probe
from examples.weather_scenarios.fv_diagnostic_guard import atomic_write_text, run_guarded_diagnostic

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "PR238_ENDPOINT_DIAGNOSTIC_PLAN_20261004.md"
SELF = "examples/weather_scenarios/fv_point_3h_endpoint_diagnostics.py"
TEST = "tests/test_fv_point_3h_endpoint_diagnostics.py"
GUARD = "examples/weather_scenarios/fv_diagnostic_guard.py"
GUARD_TEST = "tests/test_fv_diagnostic_guard.py"
CHECKPOINT = "examples/weather_scenarios/fv_hessian_checkpoint.py"
CHECKPOINT_TEST = "tests/test_fv_hessian_checkpoint.py"
NC, NF, NPARAM, NSTAGE = 26, 20, 13, 3600
INTERNAL_SECONDS = 240.0
OUTER_SECONDS = 300.0
RSS_LIMIT_BYTES = 1024**3
STATIONARITY_TOLERANCE = 1e-10


class EndpointRefusal(RuntimeError):
    """Expected pointwise numerical or qualification refusal."""

    def __init__(self, message: str, *, checkpoint_result: dict[str, Any] | None = None,
                 checkpoint_kind: str | None = None) -> None:
        super().__init__(message)
        self.checkpoint_result = checkpoint_result
        self.checkpoint_kind = checkpoint_kind


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    value = json.loads(raw_bytes)
    if not isinstance(value, dict):
        raise ValueError(f"archive sidecar must contain a JSON object: {path}")
    return value, hashlib.sha256(raw_bytes).hexdigest()


def extract_last_accepted_trial(raw: dict[str, Any]) -> dict[str, Any]:
    """Select and cross-check the archive's terminal accepted original-J trial."""
    trials = raw.get("trials")
    if not isinstance(trials, list):
        raise ValueError("continuation archive has no trial list")
    accepted = [row for row in trials if isinstance(row, dict) and row.get("status") == "accepted"]
    if not accepted:
        raise ValueError("continuation archive has no accepted trial")
    row = accepted[-1]
    control = torch.tensor(row.get("control"), dtype=torch.float64)
    gradient = torch.tensor(row.get("gradient"), dtype=torch.float64)
    if (control.shape != (NC,) or gradient.shape != (NC,)
            or not bool(torch.isfinite(control).all() & torch.isfinite(gradient).all())
            or tensor_sha(control) != raw.get("last_accepted_control_sha256")
            or raw.get("last_accepted_control") != row.get("control")
            or row.get("armijo_passed") is not True
            or row.get("strict_point_passed") is not True
            or row.get("branch", {}).get("status") != "passed_strict_branch"):
        raise ValueError("last accepted trial/control does not match its archive identity")
    if (row.get("epoch") is None or raw.get("optimizer_steps") != len(accepted)
            or row.get("epoch") != raw.get("optimizer_steps")):
        raise ValueError("accepted trial count and final epoch are inconsistent")
    return row


def _metric_check(saved: float, fresh: float) -> dict[str, float | bool]:
    return audit._metric(float(saved), float(fresh))


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten_scalars(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def _fixed_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"source map path escapes repository: {relative}")
    return path


def _source_hashes(root: Path, paths: list[str]) -> dict[str, str]:
    return {name: sha(_fixed_path(root, name)) for name in paths}


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _checkpoint_receipt_provider(*, root: Path, paths: list[str], problem: Any,
                                 original: Any, control: Tensor, parameters: Tensor,
                                 truth: Tensor, base_identity: dict[str, Any],
                                 objective: Tensor, phi: Tensor, gradient: Tensor, plan_sha256: str,
                                 gradient_blocks: dict[str, Any], branch: dict[str, Any],
                                 margins: dict[str, Any]):
    def receipt() -> dict[str, Any]:
        source_hashes = _source_hashes(root, paths)
        input_identity = seed._input_identity(problem, original, control, parameters, truth)
        runtime = blocks.runtime_identity()
        return {
            "objective": {"name": "original full-control J including the declared prior",
                          "value": float(objective),
                          "operator_scope": "Jcc of the original full-control objective",
                          "source_sha256": source_hashes.get("src/advar/fv_point_research_problem.py")},
            "problem": {"input_identity": input_identity,
                        "terminal_truth_sha256": input_identity.get("terminal_truth_sha256"),
                        "archived_problem_identity": base_identity.get("archived_input")},
            "coordinates": {"kind": "original full-control coordinates", "dimension": NC,
                            "field_dimension": NF, "dynamics_dimension": NC - NF,
                            "block_layout": {"field": [0, NF], "dynamics": [NF, NC]}},
            "source": {"hashes": source_hashes, "plan_sha256": plan_sha256},
            "runtime": runtime,
            "branch": {"status": branch.get("status"),
                       "signature_sha256": branch.get("signature_sha256"),
                       "euler_stages": branch.get("euler_stages"),
                       "choice_stage_count": branch.get("choice_stage_count"),
                       "face_sign_stage_count": branch.get("face_sign_stage_count"),
                       "margins": margins},
            "analysis_context": {"endpoint_control_sha256": tensor_sha(control),
                "parameters_sha256": tensor_sha(parameters), "objective": float(objective),
                "phi": float(phi), "gradient_sha256": tensor_sha(gradient),
                "gradient_blocks": gradient_blocks,
                "prior_contract": "original full 26-control zero-centered prior retained",
                "full_case_qualification": {"strict_branch_status": branch.get("status"),
                    "strict_branch_signature_sha256": branch.get("signature_sha256"),
                    "full_gradient_inf": float(gradient.abs().max()),
                    "stationarity_tolerance": STATIONARITY_TOLERANCE,
                    "stationarity_passed": float(gradient.abs().max()) < STATIONARITY_TOLERANCE},
                "hessian_scope": "fresh original full-H of J; field/dynamics split is diagnostic only"},
        }
    return receipt


def _exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_parent_and_resource(raw: dict[str, Any], parent: dict[str, Any],
                                  resource: dict[str, Any]) -> None:
    source_before = raw.get("source_before")
    source_after = raw.get("source_after")
    if not isinstance(source_before, dict) or source_before != source_after:
        raise ValueError("continuation archive source maps are absent or changed")
    source_after = source_before
    if (parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != raw.get("numerical_status")
            or parent.get("numerical_status") != "epoch_limit"
            or parent.get("optimizer_steps") != raw.get("optimizer_steps")
            or parent.get("child_read_error") is not None
            or parent.get("source_bytes_unchanged") is not True
            or parent.get("input_identity_unchanged") is not True
            or parent.get("r8_unchanged") is not True
            or parent.get("response_computed") is not False
            or parent.get("adjoint_computed") is not False
            or parent.get("full_root_claim") is not False):
        raise ValueError("archive parent does not establish completed unchanged continuation")
    if (parent.get("source_sha256_before") != source_before
            or parent.get("source_sha256_after") != source_after
            or parent.get("resource") != resource):
        raise ValueError("archive parent source/resource bindings do not match their sidecars")
    encoded = parent.get("source_bytes_base64_after")
    if not isinstance(encoded, dict) or set(encoded) != set(source_after):
        raise ValueError("archive parent does not carry the complete source-byte map")
    for name, payload in encoded.items():
        if not isinstance(payload, str):
            raise ValueError("parent source-byte entry is not base64 text")
        try:
            contents = base64.b64decode(payload, validate=True)
        except ValueError as error:
            raise ValueError(f"parent source-byte entry is invalid base64: {name}") from error
        if hashlib.sha256(contents).hexdigest() != source_after[name]:
            raise ValueError(f"parent source bytes do not match their SHA-256: {name}")
    if (resource.get("exit_code") != 0 or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("elapsed_seconds", math.inf) > 1500
            or resource.get("wall_limit_seconds") != 1500
            or resource.get("rss_limit_bytes") != RSS_LIMIT_BYTES
            or resource.get("sampled_peak_rss_bytes", math.inf) > RSS_LIMIT_BYTES):
        raise ValueError("archive resource sidecar does not establish bounded completed execution")
    if (raw.get("phase") != "finished" or raw.get("numerical_status") != "epoch_limit"
            or raw.get("optimizer_steps") != 4 or raw.get("input_unchanged") is not True
            or raw.get("source_unchanged") is not True
            or raw.get("response_computed") is not False or raw.get("score_computed") is not False
            or raw.get("full_root_claim") is not False or raw.get("minimum_claim") is not False):
        raise ValueError("raw archive is outside the completed four-epoch nonresponse schema")


def run(*, archive: Path, archive_sha256: str, parent_sha256: str,
        resource_sha256: str, plan: Path, output: Path,
        checkpoint_directory: Path | None = None,
        checkpoint_max_attempts: int = 3) -> dict[str, Any]:
    """Audit one last accepted control without changing it or running optimization."""
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    root = ROOT.resolve()
    archive_path = archive.resolve()
    plan_path = plan.resolve()
    checkpoint_path = checkpoint_directory.resolve() if checkpoint_directory is not None else None
    if checkpoint_path is not None:
        output_parent = output.resolve().parent
        if (not checkpoint_path.is_relative_to(root)
                or checkpoint_path.is_relative_to(output_parent)
                or output_parent.is_relative_to(checkpoint_path)):
            raise ValueError("checkpoint directory must be root-contained and separate from the attempt output directory")
    if _exists_or_symlink(output):
        raise ValueError("endpoint diagnostic output must be fresh")
    if not archive_path.is_relative_to(root) or not plan_path.is_relative_to(root):
        raise ValueError("archive and diagnostic plan must be inside the repository")
    if not plan_path.is_file():
        raise ValueError("endpoint diagnostic plan is missing")
    raw, actual_archive_sha = _read_object(archive_path)
    if actual_archive_sha != archive_sha256:
        raise ValueError("caller-pinned continuation archive SHA-256 mismatch")
    parent_path = archive_path.with_suffix(".run.json")
    resource_path = archive_path.with_suffix(".resource.json")
    parent, actual_parent_sha = _read_object(parent_path)
    resource, actual_resource_sha = _read_object(resource_path)
    if actual_parent_sha != parent_sha256 or actual_resource_sha != resource_sha256:
        raise ValueError("caller-pinned parent/resource SHA-256 mismatch")
    _validate_parent_and_resource(raw, parent, resource)
    accepted = extract_last_accepted_trial(raw)
    if not isinstance(raw.get("parameters"), list) or len(raw["parameters"]) != NPARAM:
        raise ValueError("archive original parameter vector must be length 13")
    archive_sources = raw["source_before"]
    if not all(isinstance(k, str) and isinstance(v, str) and len(v) == 64
               for k, v in archive_sources.items()):
        raise ValueError("archive source map contains an invalid path/hash")
    producer_plan = "graphify-out/fv-root-cause-20260919/S4_COUPLED_ORIGINAL_J_CONTINUATION_PLAN_20261003.md"
    if raw.get("plan_sha256") != archive_sources.get(producer_plan):
        raise ValueError("continuation's producer-plan source hash does not match its raw plan SHA")
    plan_sha = sha(plan_path)
    plan_relative = plan_path.relative_to(root).as_posix()
    archive_relative = archive_path.relative_to(root).as_posix()
    parent_relative = parent_path.relative_to(root).as_posix()
    resource_relative = resource_path.relative_to(root).as_posix()
    checkpoint_sources = {CHECKPOINT, CHECKPOINT_TEST} if checkpoint_path is not None else set()
    paths = sorted(set(archive_sources) | {SELF, TEST, GUARD, GUARD_TEST, plan_relative, archive_relative,
                                           parent_relative, resource_relative} | checkpoint_sources)
    source_before = _source_hashes(root, paths)
    if any(source_before.get(name) != digest for name, digest in archive_sources.items()):
        raise ValueError("current source tree differs from the continuation's archived source hashes")
    if (source_before[archive_relative] != archive_sha256
            or source_before[parent_relative] != parent_sha256
            or source_before[resource_relative] != resource_sha256):
        raise ValueError("source map does not include the caller-pinned archive sidecars")

    endpoint = torch.tensor(accepted["control"], dtype=torch.float64)
    parameters = torch.tensor(raw["parameters"], dtype=torch.float64)
    if (endpoint.shape != (NC,) or parameters.shape != (NPARAM,)
            or not bool(torch.isfinite(endpoint).all() & torch.isfinite(parameters).all())
            or tensor_sha(parameters) != raw.get("parameters_sha256")
            or raw.get("last_accepted_control_sha256") != raw.get("input_after", {}).get("control_sha256")):
        raise ValueError("archive endpoint/parameter tensor identity is invalid")
    runtime = blocks.runtime_identity()
    if runtime != raw.get("runtime") or runtime != raw.get("runtime_after"):
        raise ValueError("runtime differs from the completed continuation archive")
    problem, original, _seed_control, fixed_parameters, truth, base_identity = seed._prepare_fixed_seed()
    if (not torch.equal(parameters, fixed_parameters)
            or tensor_sha(fixed_parameters) != raw.get("parameters_sha256")):
        raise ValueError("reconstructed original parameters differ from the continuation archive")
    start_control = torch.tensor(raw.get("start_control"), dtype=torch.float64)
    if (start_control.shape != (NC,) or not bool(torch.isfinite(start_control).all())
            or tensor_sha(start_control) != raw.get("start_control_sha256")):
        raise ValueError("archive start control/hash is malformed")
    start_identity = seed._input_identity(problem, original, start_control, fixed_parameters, truth)
    endpoint_identity = seed._input_identity(problem, original, endpoint, fixed_parameters, truth)
    if (start_identity != raw.get("input_before") or endpoint_identity != raw.get("input_after")
            or not audit._fixed_input_matches(endpoint_identity, start_identity, tensor_sha(endpoint))
            or base_identity.get("archived_input") != endpoint_identity.get("archived_input")):
        raise ValueError("reconstructed endpoint changed original data/time/prior/input identity")
    initial = {
        "schema": "advar.point3h-endpoint-diagnostic.v1", "phase": "running",
        "numerical_status": "not_reached", "scope": "one read-only current-point J/g/branch/Hessian/Schur diagnostic",
        "archive_path": archive_relative, "archive_sha256": actual_archive_sha,
        "parent_sha256": actual_parent_sha, "resource_sha256": actual_resource_sha,
        "archive_producer_plan_sha256": raw["plan_sha256"],
        "diagnostic_plan_path": plan_relative, "diagnostic_plan_sha256": plan_sha,
        "start_control_sha256": raw.get("start_control_sha256"),
        "endpoint_control": endpoint.tolist(), "endpoint_control_sha256": tensor_sha(endpoint),
        "parameters": fixed_parameters.tolist(), "parameters_sha256": tensor_sha(fixed_parameters),
        "runtime": runtime, "source_before": source_before, "input_before": endpoint_identity,
        "internal_budget_seconds": INTERNAL_SECONDS, "outer_wall_limit_seconds": OUTER_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES, "optimizer_step_applied": False,
        "score_computed": False, "adjoint_computed": False, "response_computed": False,
        "full_root_claim": False, "minimum_claim": False, "physical_validated": False,
        "checkpoint": ({"status": "pending", "directory": checkpoint_path.relative_to(root).as_posix(),
                        "max_attempts": checkpoint_max_attempts, "attempt_seconds": INTERNAL_SECONDS,
                        "quota_scope": "Hessian-kernel invocations only; metadata/branch/J/g work is outside this quota"}
                       if checkpoint_path is not None else {"status": "not_requested"}),
    }
    _write(output, initial)
    report = dict(initial)
    try:
        if time.monotonic() >= deadline:
            raise EndpointRefusal("240-second internal budget exhausted before the endpoint branch audit")
        branch, margins = tail._full_current_branch(problem, endpoint, fixed_parameters)
        if time.monotonic() >= deadline:
            raise EndpointRefusal("240-second internal budget exhausted after the endpoint branch audit")
        report.update(phase="branch_checked", branch=branch, margins=margins)
        _write(output, report)
        if (branch.get("status") != "passed_strict_branch" or branch.get("euler_stages") != NSTAGE
                or branch.get("choice_stage_count") != NSTAGE
                or branch.get("face_sign_stage_count") != NSTAGE
                or branch.get("signature_sha256") != accepted["branch"].get("signature_sha256")
                or not seed._valid_margins(margins, complete=True)):
            raise EndpointRefusal("current endpoint does not reproduce its archived strict branch/margins")
        gradient_fn = torch.func.grad(problem.objective, argnums=0)
        fresh = tail._fresh_merit(problem, endpoint, fixed_parameters, gradient_fn)
        if time.monotonic() >= deadline:
            raise EndpointRefusal("240-second internal budget exhausted after fresh J/gradient evaluation")
        if fresh is None:
            raise EndpointRefusal("fresh current-point objective/gradient/Phi is nonfinite")
        objective, gradient, phi = fresh
        fresh_blocks = blocks.gradient_blocks(gradient)
        saved_gradient = torch.tensor(accepted["gradient"], dtype=torch.float64)
        component_checks = [_metric_check(float(a), float(b))
                            for a, b in zip(saved_gradient, gradient, strict=True)]
        saved_block_scalars = _flatten_scalars(accepted["gradient_blocks"])
        fresh_block_scalars = _flatten_scalars(fresh_blocks)
        if set(saved_block_scalars) != set(fresh_block_scalars):
            raise ValueError("accepted/fresh gradient-block schemas differ")
        block_checks = {name: _metric_check(value, fresh_block_scalars[name])
                        for name, value in saved_block_scalars.items()}
        scalar_checks: dict[str, dict[str, float | bool]] = {
            "objective": _metric_check(float(accepted["objective"]), float(objective)),
            "phi": _metric_check(float(accepted["phi"]), float(phi)),
        }
        checks: dict[str, Any] = {
            **scalar_checks, "gradient_components": component_checks,
            "gradient_blocks": block_checks,
            "all_passed": (scalar_checks["objective"]["passed"] and scalar_checks["phi"]["passed"]
                           and all(row["passed"] for row in component_checks)
                           and bool(block_checks) and all(row["passed"] for row in block_checks.values())),
        }
        report.update(phase="endpoint_checked", branch=branch, margins=margins,
                      objective=float(objective), phi=float(phi), gradient=gradient.tolist(),
                      gradient_blocks=fresh_blocks, saved_metric_checks=checks,
                      case_qualification={"branch_and_saved_J_gradient_metrics_reproduced": checks["all_passed"],
                          "full_control_stationarity_gate_passed": float(gradient.abs().max()) < STATIONARITY_TOLERANCE,
                          "qualification_scope": "endpoint case evidence; independent of Hessian curvature"},
                      hessian_objective_scope="original full-control J including all declared prior terms",
                      stationarity_gate={"tolerance": STATIONARITY_TOLERANCE,
                                         "full_gradient_inf": float(gradient.abs().max()),
                                         "passed_scalar_gate": float(gradient.abs().max()) < STATIONARITY_TOLERANCE,
                                         "scope": "scalar threshold only; no root/minimum/response claim"})
        _write(output, report)
        if not checks["all_passed"]:
            raise EndpointRefusal("fresh J/Phi/full-gradient/block metrics differ from the last accepted trial")
        if time.monotonic() >= deadline:
            raise EndpointRefusal("240-second internal budget exhausted before fresh Hessian construction")
        if checkpoint_path is None:
            try:
                hessian = hess_probe.fresh_hessian(problem.objective, endpoint, fixed_parameters, gradient, deadline)
            except hess_probe.AuditRefusal as error:
                raise EndpointRefusal(str(error)) from error
        else:
            receipt_provider = _checkpoint_receipt_provider(root=root, paths=paths,
                problem=problem, original=original, control=endpoint, parameters=fixed_parameters,
                truth=truth, base_identity=base_identity, objective=objective, phi=phi,
                gradient=gradient, plan_sha256=plan_sha, gradient_blocks=fresh_blocks,
                branch=branch, margins=margins)
            try:
                hessian = checkpoint_probe.checkpoint_hessian(problem.objective, endpoint, fixed_parameters,
                    gradient, receipt_provider=receipt_provider, checkpoint_dir=checkpoint_path,
                    deadline=deadline, max_attempts=checkpoint_max_attempts,
                    attempt_seconds=INTERNAL_SECONDS)
                report["checkpoint"] = {"status": "completed", "directory": checkpoint_path.relative_to(root).as_posix(),
                    "max_attempts": checkpoint_max_attempts, "attempt_seconds": INTERNAL_SECONDS,
                    "attempts_reserved": hessian.get("attempts_reserved")}
            except checkpoint_probe.BudgetRefusal as error:
                partial = dict(error.state)
                report["checkpoint"] = partial
                raise EndpointRefusal(str(error), checkpoint_result=partial,
                                      checkpoint_kind="budget") from error
            except checkpoint_probe.NumericalRefusal as error:
                partial = dict(error.state)
                report["checkpoint"] = partial
                raise EndpointRefusal(str(error), checkpoint_result=partial,
                                      checkpoint_kind="numerical") from error
        if hessian.get("hvp_columns") != NC or hessian.get("hvp_calls_total") != NC + 1:
            raise ValueError("fresh Hessian helper did not report 26 basis HVPs and one independent audit HVP")
        schur = hessian.get("block_schur")
        if not isinstance(schur, dict):
            raise ValueError("fresh Hessian helper returned no block/Schur result")
        report.update(phase="hessian_checked", numerical_status="not_reached",
                      fresh_hessian=hessian, hvp_columns_completed=NC,
                      hvp_calls_total=NC + 1,
                      block_schur_status=schur.get("status"),
                      block_schur_rhs="fresh full gradient; includes nonzero field-gradient correction",
                      hessian_objective_scope="original full-control J including all declared prior terms",
                      optimizer_step_applied=False, score_computed=False,
                      adjoint_computed=False, response_computed=False)
        if schur.get("status") != "schur_evaluated":
            raise EndpointRefusal(
                f"fresh block/Schur refusal: status={schur.get('status')}; refusal={schur.get('refusal')}"
            )
        report["numerical_status"] = "endpoint_diagnostic_completed"
    except EndpointRefusal as error:
        if error.checkpoint_result is not None:
            partial = error.checkpoint_result
            checkpoint_status = ("checkpoint_budget_refusal" if error.checkpoint_kind == "budget"
                                 else "checkpoint_numerical_refusal")
            report.update(phase="diagnostic_refused", numerical_status=checkpoint_status,
                refusal=f"{type(error).__name__}: {error}",
                checkpoint_hessian_status=partial.get("status", "audit_pending"),
                hvp_columns_completed=partial.get("hvp_columns", partial.get("completed_columns", "not_recorded")),
                hvp_calls_total=partial.get("hvp_calls_total", "not_recorded"))
        else:
            report.update(phase="diagnostic_refused", numerical_status="diagnostic_refusal",
                refusal=f"{type(error).__name__}: {error}",
                hvp_columns_completed=report.get("hvp_columns_completed", "not_recorded"),
                hvp_calls_total=report.get("hvp_calls_total", "not_recorded"))
    source_after = _source_hashes(root, paths)
    input_after = seed._input_identity(problem, original, endpoint, fixed_parameters, truth)
    runtime_after = blocks.runtime_identity()
    if source_after != source_before or input_after != endpoint_identity or runtime_after != runtime:
        report.update(phase="integrity_refused", numerical_status="integrity_refusal",
                      source_after=source_after, source_unchanged=source_after == source_before,
                      input_after=input_after, input_unchanged=input_after == endpoint_identity,
                      runtime_after=runtime_after,
                      refusal="source/input/runtime changed during read-only endpoint diagnostic")
        _write(output, report)
        raise ValueError("endpoint diagnostic source/input/runtime integrity check failed")
    if report.get("phase") != "diagnostic_refused":
        report["phase"] = "finished"
    report.update(source_after=source_after, source_unchanged=True,
                  input_after=input_after, input_unchanged=True,
                  runtime_after=runtime_after, elapsed_seconds=time.monotonic() - started)
    _write(output, report)
    return report


def guarded_run(*, archive: Path, archive_sha256: str, parent_sha256: str,
                resource_sha256: str, plan: Path, output: Path,
                cancel_requested: Any = None,
                checkpoint_directory: Path | None = None,
                checkpoint_max_attempts: int = 3) -> dict[str, Any]:
    """Run one root-contained child and preserve separate guard/diagnostic outcomes."""
    root, evidence = ROOT.resolve(), EVIDENCE.resolve()
    if not output.resolve().is_relative_to(evidence):
        raise ValueError("guarded diagnostic output must be inside the evidence tree")
    parent_path, resource_path, log_path = (output.with_suffix(suffix) for suffix in
                                             (".run.json", ".resource.json", ".log"))
    preflight_path = output.parent / "preflight.json"
    claim_path = output.parent / ".guarded-launch.lock"
    siblings = (output, parent_path, resource_path, log_path, preflight_path, claim_path)
    canonical_siblings = [path.resolve() for path in siblings]
    if len(set(canonical_siblings)) != len(canonical_siblings):
        raise ValueError("guarded output and sidecar paths must be distinct")
    guarded_paths = tuple(path for item in siblings for path in
                          (item, item.with_suffix(item.suffix + ".tmp")))
    atomic_temps = (candidate for item in siblings for candidate in
                    (output.parent.glob(f".{item.name}.*.tmp") if output.parent.is_dir() else ()))
    if any(_exists_or_symlink(path) for path in guarded_paths) or any(
            _exists_or_symlink(path) for path in atomic_temps):
        raise ValueError("guarded diagnostic output siblings must all be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_path, plan_path = archive.resolve(), plan.resolve()
    if not archive_path.is_relative_to(root) or not plan_path.is_relative_to(root):
        raise ValueError("archive and diagnostic plan must be inside the repository")
    checkpoint_path = checkpoint_directory.resolve() if checkpoint_directory is not None else None
    if checkpoint_path is not None and (not checkpoint_path.is_relative_to(root)
            or checkpoint_path.is_relative_to(output.parent.resolve())
            or output.parent.resolve().is_relative_to(checkpoint_path)):
        raise ValueError("checkpoint directory must be root-contained and separate from the attempt output directory")
    raw, actual_archive_sha = _read_object(archive_path)
    parent_input, actual_parent_sha = _read_object(archive_path.with_suffix(".run.json"))
    resource_input, actual_resource_sha = _read_object(archive_path.with_suffix(".resource.json"))
    if (actual_archive_sha != archive_sha256 or actual_parent_sha != parent_sha256
            or actual_resource_sha != resource_sha256):
        raise ValueError("guarded diagnostic caller pins do not match the archive sidecars")
    _validate_parent_and_resource(raw, parent_input, resource_input)
    accepted = extract_last_accepted_trial(raw)
    plan_relative = plan_path.relative_to(root).as_posix()
    archive_relative = archive_path.relative_to(root).as_posix()
    parent_input_relative = archive_path.with_suffix(".run.json").relative_to(root).as_posix()
    resource_input_relative = archive_path.with_suffix(".resource.json").relative_to(root).as_posix()
    checkpoint_sources = {CHECKPOINT, CHECKPOINT_TEST} if checkpoint_path is not None else set()
    paths = sorted(set(raw["source_before"]) | {SELF, TEST, GUARD, GUARD_TEST, plan_relative,
        archive_relative, parent_input_relative, resource_input_relative} | checkpoint_sources)
    before = _source_hashes(root, paths)
    if any(before.get(name) != digest for name, digest in raw["source_before"].items()):
        raise ValueError("guarded preflight found changed archive dependencies")
    if (before[archive_relative] != archive_sha256 or before[parent_input_relative] != parent_sha256
            or before[resource_input_relative] != resource_sha256):
        raise ValueError("guarded preflight source map omits pinned archive sidecars")
    runtime_before = blocks.runtime_identity()
    if runtime_before != raw.get("runtime"):
        raise ValueError("guarded preflight runtime differs from the completed continuation")
    try:
        with claim_path.open("x", encoding="utf-8") as claim:
            claim.write(_canonical_json({"archive_sha256": archive_sha256,
                "endpoint_control_sha256": raw["last_accepted_control_sha256"],
            "parameters_sha256": raw["parameters_sha256"],
                "checkpoint_directory": checkpoint_path.relative_to(root).as_posix() if checkpoint_path else None,
                "checkpoint_max_attempts": checkpoint_max_attempts if checkpoint_path else None,
                "checkpoint_quota_scope": "Hessian-kernel invocations only; metadata/branch/J/g work is outside this quota",
                "source_before_sha256": hashlib.sha256(_canonical_json(before).encode()).hexdigest()}) + "\n")
    except FileExistsError as error:
        raise ValueError("guarded diagnostic attempt directory is already claimed; no retry") from error
    endpoint_sha = raw["last_accepted_control_sha256"]
    command = [str(root / ".venv/bin/python"), "-m", "examples.weather_scenarios.fv_point_3h_endpoint_diagnostics",
        "--archive", str(archive_path), "--archive-sha256", archive_sha256,
        "--parent-sha256", parent_sha256, "--resource-sha256", resource_sha256,
        "--plan", str(plan_path), "--output", str(output)]
    if checkpoint_path is not None:
        command.extend(("--checkpoint-directory", str(checkpoint_path),
                        "--checkpoint-max-attempts", str(checkpoint_max_attempts),
                        "--checkpoint-child"))
    _write(preflight_path, {"scope": "metadata-only guarded preflight; child performs the endpoint diagnostic",
        "command": command, "source_before": before, "archive_sha256": archive_sha256,
        "attempt_claim_sha256": sha(claim_path),
        "endpoint_control_sha256": endpoint_sha, "parameters_sha256": raw["parameters_sha256"],
        "saved_objective": accepted["objective"], "wall_seconds": OUTER_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "checkpoint_directory": checkpoint_path.relative_to(root).as_posix() if checkpoint_path else None,
        "checkpoint_max_attempts": checkpoint_max_attempts if checkpoint_path else None,
        "checkpoint_quota_scope": ("Hessian-kernel invocations only; metadata/branch/J/g work is outside this quota"
                                    if checkpoint_path else None)})
    try:
        resource = run_guarded_diagnostic(command, wall_seconds=OUTER_SECONDS, rss_bytes=RSS_LIMIT_BYTES,
            report_path=resource_path, log_path=log_path, cancel_requested=cancel_requested)
    except Exception as error:
        after = _source_hashes(root, paths)
        runtime_after = blocks.runtime_identity()
        try:
            resource_failure, _ = _read_object(resource_path)
        except (OSError, ValueError):
            resource_failure = None
        _write(parent_path, {"execution_status": "failed", "failure_reason":
            f"guard invocation failed: {type(error).__name__}: {error}", "numerical_status": "not_reached",
            "child_terminal_phase": None, "child_read_error": "guard did not return", "resource": resource_failure,
            "source_before": before, "source_after": after, "source_unchanged": before == after,
            "runtime_before": runtime_before, "runtime_after": runtime_after,
            "endpoint_identity_matches_archive": False, "endpoint_control_sha256": endpoint_sha,
            "parameters_sha256": raw["parameters_sha256"], "archive_sha256": archive_sha256,
            "checkpoint": {"status": "not_reached", "directory": checkpoint_path.relative_to(root).as_posix()
                           if checkpoint_path else None},
            "archive_unchanged": sha(archive_path) == archive_sha256,
            "optimizer_step_applied": False, "response_computed": False, "physical_validated": False})
        raise
    after = _source_hashes(root, paths)
    runtime_after = blocks.runtime_identity()
    try:
        child, child_sha = _read_object(output)
        child_read_error = None
    except (OSError, ValueError) as error:
        child, child_sha, child_read_error = None, None, str(error)
    endpoint_matches = bool(child is not None
        and child.get("endpoint_control_sha256") == endpoint_sha
        and child.get("endpoint_control") == raw["last_accepted_control"]
        and child.get("parameters_sha256") == raw["parameters_sha256"]
        and child.get("parameters") == raw["parameters"]
        and child.get("input_before") == raw["input_after"]
        and child.get("input_after") == raw["input_after"])
    child_sources_match = bool(child is not None
        and child.get("source_before") == before and child.get("source_after") == after)
    child_runtime_matches = bool(child is not None
        and child.get("runtime") == runtime_before == raw.get("runtime")
        and child.get("runtime_after") == runtime_after == runtime_before)
    complete = bool(endpoint_matches and child is not None
        and child.get("phase") in {"finished", "diagnostic_refused"}
        and child.get("source_unchanged") is True and child_sources_match
        and child.get("input_unchanged") is True and child_runtime_matches and before == after)
    if resource.get("received_sigterm"):
        execution, failure = "failed", "guard received SIGTERM; child cancellation was requested"
    elif resource.get("resource_termination") == "cancelled":
        execution, failure = "failed", "guard cancellation was requested"
    elif resource.get("monitor_error") is not None or resource.get("resource_termination") not in {
            None, "wall_time_limit", "rss_limit"}:
        execution, failure = "failed", str(resource.get("monitor_error") or resource["resource_termination"])
    elif resource.get("resource_termination") in {"wall_time_limit", "rss_limit"}:
        execution, failure = "resource_limited", str(resource["resource_termination"])
    elif resource.get("child_process_group_cleanup_error") is not None:
        execution, failure = "failed", str(resource["child_process_group_cleanup_error"])
    elif resource.get("child_process_group_cleanup_sent") is True:
        execution, failure = "failed", "guard terminated residual child processes after child exit"
    elif resource.get("child_process_group_cleanup_sent") is not False:
        execution, failure = "failed", "guard did not verify a clean child process group"
    elif (not isinstance(resource.get("elapsed_seconds"), (int, float))
            or isinstance(resource.get("elapsed_seconds"), bool)
            or not math.isfinite(resource["elapsed_seconds"])):
        execution, failure = "failed", "guard resource report has no finite elapsed_seconds value"
    elif resource["elapsed_seconds"] > OUTER_SECONDS:
        execution, failure = "resource_limited", (
            f"measured elapsed_seconds={resource['elapsed_seconds']} exceeded {OUTER_SECONDS}-second wall cap")
    elif resource.get("exit_code") == 0 and resource.get("monitor_error") is None and complete:
        execution, failure = "completed", None
    else:
        execution, failure = "failed", "guarded child or parent integrity gate failed"
    record = {"execution_status": execution, "failure_reason": failure,
        "numerical_status": child.get("numerical_status", "not_reached") if child else "not_reached",
        "child_terminal_phase": child.get("phase") if child else None, "child_sha256": child_sha,
        "child_read_error": child_read_error, "resource": resource, "source_before": before,
        "source_after": after, "source_unchanged": before == after,
        "child_source_maps_match": child_sources_match,
        "runtime_before": runtime_before, "runtime_after": runtime_after,
        "child_runtime_matches": child_runtime_matches,
        "checkpoint": child.get("checkpoint", {"status": "not_requested"}) if child else {"status": "not_reached"},
        "endpoint_identity_matches_archive": endpoint_matches, "endpoint_control_sha256": endpoint_sha,
        "parameters_sha256": raw["parameters_sha256"], "input_before": child.get("input_before") if child else None,
        "input_after": child.get("input_after") if child else None, "archive_sha256": archive_sha256,
        "archive_unchanged": sha(archive_path) == archive_sha256,
        "optimizer_step_applied": False, "response_computed": False, "physical_validated": False}
    _write(parent_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guarded", action="store_true",
                        help="run the read-only child under the new wall/RSS guard")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--checkpoint-max-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.guarded:
        if args.checkpoint_child:
            parser.error("--checkpoint-child is reserved for an externally guarded child")
        result = guarded_run(archive=args.archive, archive_sha256=args.archive_sha256,
            parent_sha256=args.parent_sha256, resource_sha256=args.resource_sha256,
            plan=args.plan, output=args.output, checkpoint_directory=args.checkpoint_directory,
            checkpoint_max_attempts=args.checkpoint_max_attempts)
        print(json.dumps({"execution_status": result["execution_status"],
                          "numerical_status": result["numerical_status"],
                          "child_terminal_phase": result["child_terminal_phase"]}))
        if result["execution_status"] != "completed":
            raise SystemExit(1)
    else:
        if args.checkpoint_child and args.checkpoint_directory is None:
            parser.error("--checkpoint-child requires --checkpoint-directory")
        if args.checkpoint_directory is not None and not args.checkpoint_child:
            parser.error("checkpoint CLI requires --guarded; --checkpoint-child is for its launched child")
        run(archive=args.archive, archive_sha256=args.archive_sha256,
            parent_sha256=args.parent_sha256, resource_sha256=args.resource_sha256,
            plan=args.plan, output=args.output, checkpoint_directory=args.checkpoint_directory,
            checkpoint_max_attempts=args.checkpoint_max_attempts)


if __name__ == "__main__":
    main()
