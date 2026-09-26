"""One guarded root-only refinement from the frozen PR212 alternate seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import matrix_free
from advar.local_refinement import (RefinementCallbackError,
                                    RefinementNumericalRefusal,
                                    RefinementTrial, refine_stationary)
from examples.weather_scenarios import fv_partial_alternate_seed_gate as seed_gate
from examples.weather_scenarios import fv_partial_reanalysis_preflight as preflight
from examples.weather_scenarios import fv_partial_sector_root_probe as prior
from examples.weather_scenarios import fv_point_basin_probe as basin
from examples.weather_scenarios.fv_point_sector_policy import _digest, _key
from tests.test_fv_research_partial_observation import _problem


EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_PARTIAL_ALTERNATE_ROOT_PLAN.md"
SEED_ARCHIVE = EVIDENCE / "partial_alternate_seed_attempt1"
SEED_MANIFEST_SHA256 = "959ab714de6f862a85fb0ee6ef8353fbfc45098916217a83a0d6acb7ef8f8fa5"
SEED_CONTROL_SHA256 = seed_gate.SELECTED_CONTROL_SHA256
SEED_SIGNATURE_SHA256 = seed_gate.SELECTED_SIGNATURE_SHA256
SOURCE_PATHS = tuple(dict.fromkeys((
    *prior.SOURCE_PATHS,
    *seed_gate.SOURCE_PATHS,
    "examples/weather_scenarios/fv_partial_alternate_root_probe.py",
    "examples/weather_scenarios/fv_partial_alternate_root_runner.py",
)))
SEED_RECORDS = (
    "alternate_seed.json", "alternate_seed.run.json",
    "alternate_seed.resource.json", "alternate_seed.log",
)
SEED_GATE_PLAN_SHA256 = "a3fb9ed7879ece170c4398e41bca230c5d76cb3d8790e7eaec06a20b60de5500"
SEED_GATE_PROBE_SHA256 = "443d7225b5322c9a2d9e89f22b1993d2c878f5c3e98186d6e7b9cf8a7d3dfff0"
SEED_GATE_RUNNER_SHA256 = "5c7cf933ffa0dfc4a2a51e4e35193580702e3671fa6d51323a335e9d0bfd5cd9"
SEED_CONTROL_MARGIN = 1e-4
STATIONARITY_TOLERANCE = 1e-10
PCG_TRUE_RESIDUAL_TOLERANCE = 1e-10


class _SeedGateRefusal(ValueError):
    """The immutable PR212 prerequisite evidence is invalid."""


class _SeedIdentityError(ValueError):
    """Current source, input, runtime, or fresh measurements differ from PR212."""


class _FinalQualificationRefusal(ValueError):
    """A final stationarity, exact-Hessian, or branch qualification failed."""


def _archived_seed_control() -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Recover the selected PR209 vector from its hash-pinned raw trial archive."""
    archive = seed_gate.ARCHIVE
    manifest_path = archive / "manifest.json"
    if _sha(manifest_path) != seed_gate.ARCHIVE_MANIFEST_SHA256:
        raise _SeedIdentityError("PR209 source archive manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    file_hashes = manifest.get("file_sha256")
    if not isinstance(file_hashes, dict):
        raise _SeedIdentityError("PR209 source archive lacks file hashes")
    for name in seed_gate.RAW_FILES:
        path = archive / name
        relative = str(path.relative_to(ROOT))
        if relative not in file_hashes or _sha(path) != file_hashes[relative]:
            raise _SeedIdentityError(f"PR209 raw root record changed: {name}")
    prior_child = json.loads((archive / "partial_sector_root.json").read_text())
    if (prior_child.get("numerical_status") != "root_refused"
            or prior_child.get("source_before") != prior_child.get("source_after")
            or prior_child.get("source_before") != manifest.get("source_before_after_sha256")
            or prior_child.get("input_before") != prior_child.get("input_after")):
        raise _SeedIdentityError("PR209 root archive source or input record changed")
    current_sources = _sources()
    if any(current_sources.get(name) != digest for name, digest in
           prior_child["source_before"].items()
           if name not in seed_gate.STATUS_ONLY_SOURCES):
        raise _SeedIdentityError("PR209 archived objective source differs from current source")
    selected = seed_gate.select_seed(prior_child)["selected"]
    control = torch.tensor(selected["candidate_control"], dtype=torch.float64)
    if (control.shape != (26,) or not bool(torch.isfinite(control).all())
            or _tensor_sha(control) != SEED_CONTROL_SHA256):
        raise _SeedIdentityError("PR209 selected control differs from frozen PR212 seed")
    return control, selected, prior_child


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _sources() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _seed_record_hashes() -> dict[str, str]:
    records = {
        "partial_alternate_seed_attempt1/manifest.json":
            SEED_ARCHIVE / "manifest.json",
        **{f"partial_alternate_seed_attempt1/{name}": SEED_ARCHIVE / name
           for name in SEED_RECORDS},
        "partial_sector_root_attempt1/manifest.json":
            seed_gate.ARCHIVE / "manifest.json",
        **{f"partial_sector_root_attempt1/{name}": seed_gate.ARCHIVE / name
           for name in seed_gate.RAW_FILES},
    }
    return {name: _sha(path) for name, path in records.items()}


def _branch_summary(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "euler_stages": branch["euler_stages"],
        "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
        "minimum_scaled_face_flux_margin": branch["minimum_scaled_face_flux_margin"],
        "signature_sha256": _digest(_key(branch)),
    }


def _positive_branch(branch: object, *, threshold: float = 0.0) -> bool:
    return (isinstance(branch, dict)
            and branch.get("euler_stages") == 54
            and type(branch.get("minimum_scaled_slope_margin")) in (int, float)
            and math.isfinite(branch["minimum_scaled_slope_margin"])
            and branch["minimum_scaled_slope_margin"] > threshold
            and type(branch.get("minimum_scaled_face_flux_margin")) in (int, float)
            and math.isfinite(branch["minimum_scaled_face_flux_margin"])
            and branch["minimum_scaled_face_flux_margin"] > threshold
            and isinstance(branch.get("signature_sha256"), str)
            and len(branch["signature_sha256"]) == 64)


def _valid_curvature(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    names = ("symmetry_relative", "lambda_min", "lambda_max", "lambda_ratio")
    return (all(type(value.get(name)) in (int, float)
                and math.isfinite(value[name]) for name in names)
            and type(value.get("hvp_columns")) is int
            and value["hvp_columns"] == 26
            and 0 <= value["symmetry_relative"] <= 1e-10
            and value["lambda_min"] > 0
            and value["lambda_max"] > 0
            and value["lambda_ratio"] > math.sqrt(torch.finfo(torch.float64).eps))


def _seed_evidence() -> tuple[dict[str, Any], Tensor]:
    manifest_path = SEED_ARCHIVE / "manifest.json"
    if _sha(manifest_path) != SEED_MANIFEST_SHA256:
        raise _SeedGateRefusal("PR212 seed manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    file_hashes = manifest.get("file_sha256")
    if not isinstance(file_hashes, dict):
        raise _SeedGateRefusal("PR212 seed manifest lacks raw file hashes")
    for name in SEED_RECORDS:
        path = SEED_ARCHIVE / name
        relative = str(path.relative_to(ROOT))
        if relative not in file_hashes or _sha(path) != file_hashes[relative]:
            raise _SeedGateRefusal(f"PR212 seed record changed: {name}")
    seed = json.loads((SEED_ARCHIVE / "alternate_seed.json").read_text())
    parent = json.loads((SEED_ARCHIVE / "alternate_seed.run.json").read_text())
    resource = json.loads((SEED_ARCHIVE / "alternate_seed.resource.json").read_text())
    if (manifest.get("archive_manifest_sha256") != seed_gate.ARCHIVE_MANIFEST_SHA256
            or manifest.get("selected_control_sha256") != SEED_CONTROL_SHA256
            or manifest.get("reviewed_plan_sha256") != SEED_GATE_PLAN_SHA256
            or manifest.get("reviewed_probe_sha256") != SEED_GATE_PROBE_SHA256
            or manifest.get("reviewed_runner_sha256") != SEED_GATE_RUNNER_SHA256
            or seed.get("numerical_status") != "seed_locally_spd"
            or seed.get("phase") != "finished"
            or seed.get("archive_manifest_sha256") != seed_gate.ARCHIVE_MANIFEST_SHA256
            or seed.get("selected_control_sha256") != SEED_CONTROL_SHA256
            or seed.get("selected_signature_sha256") != SEED_SIGNATURE_SHA256
            or seed.get("selected_iteration") != 5
            or seed.get("selected_backtrack") != 3
            or seed.get("selected_previous_policy_reason") != "merit_switch_refused"
            or seed.get("new_product_gn_runs") != 0
            or seed.get("new_newton_refinement_runs") != 0
            or seed.get("new_adjoint_reanalysis_runs") != 0
            or seed.get("response_validation") != "not_performed"
            or seed.get("physical_validation") != "not_performed"
            or seed.get("raw_unchanged") is not True
            or seed.get("curvature_control_sha256") != SEED_CONTROL_SHA256
            or not _valid_curvature(seed.get("curvature"))
            or parent.get("execution_status") != "completed"
            or parent.get("numerical_status") != "seed_locally_spd"
            or parent.get("response_validation") != "not_performed"
            or parent.get("physical_validation") != "not_performed"
            or resource.get("exit_code") != 0
            or resource.get("child_pid") != seed.get("pid")
            or resource.get("resource_termination") is not None
            or resource.get("monitor_error") is not None
            or resource.get("wall_limit_seconds") != 120
            or resource.get("rss_limit_bytes") != 1024**3
            or type(resource.get("rss_samples")) is not int
            or resource["rss_samples"] <= 0
            or type(resource.get("sampled_peak_rss_bytes")) is not int
            or not 0 < resource["sampled_peak_rss_bytes"] <= 1024**3
            or type(resource.get("elapsed_seconds")) not in (int, float)
            or not 0 <= resource["elapsed_seconds"] <= 120):
        raise _SeedGateRefusal("PR212 selected seed or gate records failed validation")
    if (seed_gate._sha(seed_gate.PLAN) != SEED_GATE_PLAN_SHA256
            or seed_gate._sha(Path(seed_gate.__file__)) != SEED_GATE_PROBE_SHA256
            or seed_gate._sha(Path(seed_gate.__file__).with_name(
                "fv_partial_alternate_seed_runner.py")) != SEED_GATE_RUNNER_SHA256
            ):
        raise _SeedGateRefusal("pinned PR212 seed-gate source or plan changed")
    runtime = seed.get("environment")
    if runtime != {"python": platform.python_version(), "torch": torch.__version__,
                   "device": "CPU FP64"}:
        raise _SeedIdentityError("PR212 seed runtime differs from current runtime")
    current_sources = _sources()
    archived_sources = seed.get("source_before")
    if (seed.get("source_before") != seed.get("source_after")
            or not isinstance(archived_sources, dict)
            or seed.get("input_before") != seed.get("input_after")
            or seed.get("input_before") != prior._preflight_identity()
            or current_sources.get("examples/weather_scenarios/fv_partial_sector_root_probe.py")
            != seed_gate.PRIOR_STATUS_SOURCE_SHA256
            or current_sources.get("examples/weather_scenarios/fv_partial_sector_root_runner.py")
            != seed_gate.PRIOR_RUNNER_STATUS_SOURCE_SHA256
            or any(current_sources.get(name) != digest
                   for name, digest in archived_sources.items()
                   if name not in seed_gate.STATUS_ONLY_SOURCES)):
        raise _SeedIdentityError("PR212 source or two-hole input identities changed")
    control, selected_row, prior_child = _archived_seed_control()
    if (prior_child["input_before"] != seed.get("input_before")
            or selected_row.get("candidate_control_sha256")
            != seed.get("selected_control_sha256")
            or selected_row.get("branch", {}).get("signature_sha256")
            != seed.get("selected_signature_sha256")):
        raise _SeedIdentityError("PR209 raw candidate and PR212 selected seed do not match")
    return seed, control


def measured_switch_acceptance(
    trial: RefinementTrial,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[bool, str] | None:
    """Use endpoint Phi decrease only when the full limiter/face signature changes."""
    current = _key(trial.current_branch)
    candidate = _key(trial.candidate_branch)
    if current == candidate:
        return None
    safe_norm = math.sqrt(torch.finfo(torch.float64).max)
    old_norm, new_norm = trial.current_gradient_norm, trial.candidate_gradient_norm
    old = (0.5 * old_norm * old_norm if math.isfinite(old_norm)
           and abs(old_norm) <= safe_norm else math.inf)
    new = (0.5 * new_norm * new_norm if math.isfinite(new_norm)
           and abs(new_norm) <= safe_norm else math.inf)
    finite = math.isfinite(old) and math.isfinite(new)
    delta = old - new if finite else math.nan
    floor = (128 * torch.finfo(torch.float64).eps
             * max(abs(old), abs(new), torch.finfo(torch.float64).tiny)
             if finite else math.nan)
    accepted = finite and math.isfinite(delta) and math.isfinite(floor) and delta > floor
    reason = "measured_phi_decrease" if accepted else "measured_phi_refused"
    if record is not None:
        record({
            "iteration": trial.iteration, "backtrack": trial.backtrack,
            "step_scale": trial.step_scale,
            "current_signature_sha256": _digest(current),
            "candidate_signature_sha256": _digest(candidate),
            "old_phi": old if finite else None, "new_phi": new if finite else None,
            "measured_phi_decrease": delta if finite and math.isfinite(delta) else None,
            "required_decrease": floor if finite and math.isfinite(floor) else None,
            "accepted": accepted, "reason": reason,
        })
    return accepted, reason


def run(output: Path, *, expected_plan_sha256: str,
        expected_probe_sha256: str,
        expected_runner_sha256: str) -> dict[str, Any]:
    raw_archives = (seed_gate.ARCHIVE.resolve(strict=True),
                    SEED_ARCHIVE.resolve(strict=True))
    temporary = output.with_suffix(output.suffix + ".tmp")
    if (any(output.resolve(strict=False).is_relative_to(path) for path in raw_archives)
            or any(temporary.resolve(strict=False).is_relative_to(path)
                   for path in raw_archives)
            or output.exists() or output.is_symlink()
            or temporary.exists() or temporary.is_symlink()):
        raise ValueError("alternate root output must be fresh and outside raw archives")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (_sha(PLAN) != expected_plan_sha256
            or _sha(Path(__file__)) != expected_probe_sha256
            or _sha(Path(__file__).with_name("fv_partial_alternate_root_runner.py"))
            != expected_runner_sha256):
        raise ValueError("reviewed alternate-root plan or source changed before run")
    started = time.monotonic()
    seed, control = _seed_evidence()
    source_before = _sources()
    input_before = prior._preflight_identity()
    seed_record_hashes_before = _seed_record_hashes()
    problem, _, parameters = _problem()
    if problem.identity != input_before["current_problem_identity"]:
        raise ValueError("current two-hole problem identity changed")
    report: dict[str, Any] = {
        "pid": os.getpid(), "phase": "seed_recheck", "numerical_status": "running",
        "scope": "one exploratory PR212-seed root-only refinement; no GN/response/reanalysis",
        "response_validation": "not_performed", "physical_validation": "not_performed",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "plan_sha256": expected_plan_sha256,
        "plan_sha256_before": expected_plan_sha256,
        "reviewed_probe_sha256": expected_probe_sha256,
        "reviewed_runner_sha256": expected_runner_sha256,
        "seed_manifest_sha256": SEED_MANIFEST_SHA256,
        "seed_control_sha256": SEED_CONTROL_SHA256,
        "seed_signature_sha256": SEED_SIGNATURE_SHA256,
        "seed_input_identity": seed["input_before"],
        "source_before": source_before, "input_before": input_before,
        "seed_record_hashes_before": seed_record_hashes_before,
        "control_sha256": _tensor_sha(control), "parameters_sha256": _tensor_sha(parameters),
        "new_product_gn_runs": 0, "new_adjoint_reanalysis_runs": 0,
        "new_vjp_runs": 0, "new_nonlinear_reanalysis_runs": 0,
        "new_newton_refinement_runs": 0,
        "branch_calls": [], "trial_records": [], "policy_records": [],
        "linear_solves": [],
    }

    def save() -> None:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        temporary.replace(output)

    def branch_check(candidate: Tensor, p: Tensor) -> tuple[dict[str, Any], str]:
        entry: dict[str, Any] = {"phase": report["phase"],
                                 "control_sha256": _tensor_sha(candidate),
                                 "status": "trace_pending"}
        try:
            branch, scope, face = preflight.branch_with_face_margin(problem, candidate, p)
            complete = {**branch, "minimum_scaled_face_flux_margin": face}
            summary = _branch_summary(complete)
            if summary["euler_stages"] != 54:
                raise RuntimeError("partial FV branch oracle did not return 54 Euler stages")
            if not _positive_branch(summary):
                raise ValueError("partial FV strict branch has nonpositive scaled margins")
            entry.update(status="positive_strict_branch", **summary)
            return complete, scope
        except ValueError as error:
            if str(error) in {
                "minmod joint oracle left its strict smooth branch",
                "partial FV strict branch has nonpositive scaled margins",
            }:
                entry.update(status="oracle_branch_refused", reason=str(error))
                raise
            entry.update(status="callback_error", reason=f"{type(error).__name__}: {error}")
            raise RefinementCallbackError(f"unexpected branch oracle failure: {error}") from error
        except Exception as error:
            entry.update(status="callback_error", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["branch_calls"].append(entry)
            save()

    def record_policy(entry: dict[str, Any]) -> None:
        report["policy_records"].append(entry)
        save()

    def record_trial(entry: dict[str, Any]) -> None:
        candidate_values = entry.get("candidate_control")
        if candidate_values is not None:
            candidate = torch.tensor(candidate_values, dtype=torch.float64)
            entry["candidate_control_sha256"] = _tensor_sha(candidate)
            if not bool(torch.isfinite(candidate).all()):
                entry["candidate_control"] = None
                entry["candidate_nonfinite"] = True
        branch = entry.get("branch")
        if isinstance(branch, dict):
            entry["branch"] = _branch_summary(branch)
        report["trial_records"].append(entry)
        save()

    def monitor(original: Any) -> Any:
        def solve(operator: Any, rhs: Tensor, **kwargs: Any) -> Any:
            iteration = 1 + sum(item.get("phase") == "root_refinement"
                               for item in report["linear_solves"])
            entry: dict[str, Any] = {
                "phase": report["phase"], "iteration": iteration, "hvp_calls": 0,
                "rtol": kwargs.get("rtol"),
                "max_iterations": kwargs.get("max_iterations"),
            }

            def measured(vector: Tensor) -> Tensor:
                entry["hvp_calls"] += 1
                return operator(vector)

            tick = time.monotonic()
            try:
                result = original(measured, rhs, **kwargs)
                true_residual = torch.linalg.vector_norm(
                    measured(result.solution) - rhs
                ) / torch.linalg.vector_norm(rhs)
                entry.update(converged=result.converged, iterations=result.iterations,
                             reported_relative_residual=result.relative_residual,
                             true_relative_residual=float(true_residual))
                return result
            except (RuntimeError, ValueError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                entry["seconds"] = time.monotonic() - tick
                report["linear_solves"].append(entry)
                save()
        return solve

    save()
    try:
        report["phase"] = "seed_branch"
        fresh_gradient = torch.func.grad(problem.objective, argnums=0)(control, parameters)
        fresh_objective = float(problem.objective(control, parameters))
        if (not bool(torch.isfinite(fresh_gradient).all())
                or not math.isfinite(fresh_objective)
                or not math.isclose(float(fresh_gradient.abs().max()), seed["gradient_max"],
                                    rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(float(torch.linalg.vector_norm(fresh_gradient)),
                                    seed["gradient_norm"], rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(fresh_objective, seed["objective"],
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise _SeedIdentityError("fresh PR212 seed objective or gradient differs")
        branch, _ = branch_check(control, parameters)
        seed_branch = _branch_summary(branch)
        if seed_branch["signature_sha256"] != SEED_SIGNATURE_SHA256:
            raise _SeedIdentityError("fresh PR212 seed limiter/face signature differs")
        for name in ("minimum_scaled_slope_margin", "minimum_scaled_face_flux_margin"):
            archived_value = seed["seed_branch"][name]
            if not math.isclose(seed_branch[name], archived_value,
                                rel_tol=1e-10, abs_tol=1e-12):
                raise _SeedIdentityError(f"fresh PR212 seed {name} differs")
        report.update(seed_objective=fresh_objective,
                      seed_gradient_norm=float(torch.linalg.vector_norm(fresh_gradient)),
                      seed_gradient_max=float(fresh_gradient.abs().max()),
                      seed_branch=seed_branch,
                      seed_curvature=seed["curvature"],
                      seed_curvature_control_sha256=SEED_CONTROL_SHA256)
        report["phase"] = "root_refinement"
        report["new_newton_refinement_runs"] = 1
        save()
        try:
            with matrix_free.observe_pcg_calls(monitor):
                refined = refine_stationary(
                    problem.objective, control, parameters,
                    branch_check=branch_check,
                    trial_acceptance=lambda trial: measured_switch_acceptance(
                        trial, record_policy),
                    trial_observer=record_trial,
                    max_iterations=8, max_backtracks=16, pcg_max_iterations=104,
                )
        except RefinementNumericalRefusal as error:
            report.update(phase="root_refinement", numerical_status="root_refused",
                          refusal=f"{type(error).__name__}: {error}")
        else:
            final_gradient = torch.func.grad(problem.objective, argnums=0)(
                refined.control, parameters)
            final_gradient_max = float(final_gradient.abs().max())
            if not math.isfinite(final_gradient_max) or final_gradient_max >= STATIONARITY_TOLERANCE:
                raise _FinalQualificationRefusal("final gradient infinity norm is not below 1e-10")
            report["phase"] = "final_curvature"
            save()
            try:
                final_curvature = basin._hessian_audit(problem, refined.control, parameters)
            except ValueError as error:
                if str(error).startswith("basin terminal exact "):
                    raise _FinalQualificationRefusal(str(error)) from error
                raise
            report["phase"] = "final_branch"
            try:
                final_branch, _ = branch_check(refined.control, parameters)
            except ValueError as error:
                if str(error) in {
                    "minmod joint oracle left its strict smooth branch",
                    "partial FV strict branch has nonpositive scaled margins",
                }:
                    raise _FinalQualificationRefusal(
                        f"final strict branch refused: {error}") from error
                raise
            final_summary = _branch_summary(final_branch)
            qualified = (
                final_summary["minimum_scaled_slope_margin"] > SEED_CONTROL_MARGIN
                and final_summary["minimum_scaled_face_flux_margin"] > SEED_CONTROL_MARGIN
            )
            report.update(
                phase="finished",
                numerical_status="root_margin_qualified" if qualified else "root_low_margin",
                response_margin_qualified=qualified,
                control=refined.control.tolist(),
                final_control_sha256=_tensor_sha(refined.control),
                final_objective=float(problem.objective(refined.control, parameters)),
                final_gradient_norm=float(torch.linalg.vector_norm(final_gradient)),
                final_gradient_max=final_gradient_max,
                final_branch=final_summary, final_curvature=final_curvature,
                final_curvature_control_sha256=_tensor_sha(refined.control),
                refinement_iterations=refined.iterations,
                refinement_hvp_count=refined.hvp_count,
            )
    except _SeedIdentityError as error:
        report.update(phase="seed_recheck", numerical_status="identity_error",
                      refusal=f"{type(error).__name__}: {error}")
    except _FinalQualificationRefusal as error:
        report.update(numerical_status="root_refused",
                      refusal=f"{type(error).__name__}: {error}")
    except RefinementCallbackError as error:
        report.update(numerical_status="callback_error",
                      refusal=f"{type(error).__name__}: {error}")
    except Exception as error:
        report.update(numerical_status="callback_error",
                      refusal=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_after"] = _sources()
        report["input_after"] = prior._preflight_identity()
        report["plan_sha256_after"] = _sha(PLAN)
        report["plan_unchanged"] = (
            report["plan_sha256_before"] == report["plan_sha256_after"]
            == expected_plan_sha256)
        report["seed_record_hashes_after"] = _seed_record_hashes()
        report["seed_archive_unchanged"] = (
            _sha(SEED_ARCHIVE / "manifest.json") == SEED_MANIFEST_SHA256
            and all(_sha(SEED_ARCHIVE / name) == json.loads(
                (SEED_ARCHIVE / "manifest.json").read_text())["file_sha256"][
                    str((SEED_ARCHIVE / name).relative_to(ROOT))]
                    for name in SEED_RECORDS))
        report["seed_archives_unchanged"] = (
            report["seed_record_hashes_before"] == report["seed_record_hashes_after"])
        report["source_unchanged"] = report["source_before"] == report["source_after"]
        report["input_unchanged"] = report["input_before"] == report["input_after"]
        report["parameters_unchanged"] = _tensor_sha(parameters) == report["parameters_sha256"]
        if not all(report.get(name) is True for name in (
            "plan_unchanged", "seed_archive_unchanged", "source_unchanged",
            "seed_archives_unchanged", "input_unchanged", "parameters_unchanged",
        )):
            report.update(phase="identity_recheck", numerical_status="identity_error")
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()
    result = run(args.output, expected_plan_sha256=args.expected_plan_sha256,
                 expected_probe_sha256=args.expected_probe_sha256,
                 expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps({name: result.get(name) for name in (
        "phase", "numerical_status", "final_gradient_max", "refusal",
        "elapsed_seconds",
    )}, indent=2, sort_keys=True))
    raise SystemExit(0 if result.get("numerical_status") in (
        "root_margin_qualified", "root_low_margin") else 2)
