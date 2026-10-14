"""Restore one stored native candidate and observe its forecast; no optimization."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
WORK = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012"
CONTINUATION = WORK / "continuation3"
ATTEMPT = CONTINUATION / "attempt1"
PLAN = CONTINUATION / "CONTINUATION_PLAN.json"
PLAN_SHA = CONTINUATION / "CONTINUATION_PLAN_SHA256.txt"
CANDIDATE = ATTEMPT / "face_chart_newton_resume.json"
PREFLIGHT = ATTEMPT / "current_face_corrected_preflight.json"
CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
CASE_SHA = "1e810c23c73b93d89b2a10b46d9ddea44a8217b07dcd3ebfebb462858e3fa9b7"
TRANSPORT = ROOT / "src/advar/transport.py"
TRANSPORT_BEFORE = "b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a"
TRANSPORT_AFTER = "c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9"
REFERENCE = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.json"
REFERENCE_ARRAYS = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.npz"
REFERENCE_RESOURCE = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_reference.resource.json"
OUTPUT = ATTEMPT / "candidate_8eff6b1f_forecast.json"
FIELDS = ATTEMPT / "candidate_8eff6b1f_forecast_fields.npz"
EXPECTED_CONTROL_SHA = "8eff6b1f2285ff626fcd8815b43e0b772cdef050501511837e4f51fcbaadb47c"
LEADS_MINUTES = (10, 30, 60, 180)
WALL_LIMIT_SECONDS = 30.0


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned file missing or unsafe: {path}")
    return _sha_bytes(path.read_bytes())


def _tensor_sha(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("candidate tensors must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(plan: dict[str, Any], digest: str) -> dict[str, str]:
    if (_sha(PLAN) != digest or PLAN_SHA.read_text().strip() != digest
            or _sha(CASE) != CASE_SHA or _sha(TRANSPORT) != TRANSPORT_AFTER):
        raise ValueError("continuation plan, fixed case, or support-transition source pin mismatch")
    transition = plan.get("source_transition", {}).get("src/advar/transport.py")
    if transition != {"before": TRANSPORT_BEFORE, "after": TRANSPORT_AFTER}:
        raise ValueError("support-only transport source transition differs from the frozen continuation plan")
    hashes: dict[str, str] = {}
    for name, expected in plan["source_files"].items():
        actual = _sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"continuation source_files pin mismatch: {name}")
        hashes[name] = actual
    for name, expected in plan["extra_source_files"].items():
        actual = _sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"continuation extra source pin mismatch: {name}")
        hashes[name] = actual
    for name, expected in plan["archive_files"].items():
        actual = _sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"continuation archive_files pin mismatch: {name}")
        hashes[name] = actual
    bundle = CONTINUATION / "source.tar.gz"
    if _sha(bundle) != plan.get("source_bundle_sha256"):
        raise ValueError("continuation source bundle hash mismatch")
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: archive.extractfile(member).read()
            for member in archive.getmembers() if member.isfile()}
    source_files = {**plan["source_files"], **plan["extra_source_files"]}
    if len(members) != len(source_files) or set(members) != set(source_files):
        raise ValueError("continuation source bundle member set mismatch")
    for name, expected in source_files.items():
        if _sha_bytes(members[name]) != expected:
            raise ValueError(f"continuation source bundle member mismatch: {name}")
    return hashes


def run() -> dict[str, Any]:
    if OUTPUT.exists() or FIELDS.exists():
        raise ValueError("forecast restoration receipt must be fresh")
    started = time.monotonic()
    deadline = started + WALL_LIMIT_SECONDS
    initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runner_sha_before = _sha(SELF)
    plan_digest = _sha(PLAN)
    plan = _read(PLAN)
    pins_before = _validate(plan, plan_digest)
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"}
    if runtime != plan.get("runtime_identity"):
        raise ValueError(f"forecast restoration runtime differs from plan: {runtime}")

    candidate = _read(CANDIDATE)
    preflight = _read(PREFLIGHT)
    control = torch.as_tensor(candidate.get("control"), dtype=torch.float64, device="cpu")
    if (control.shape != (26,) or _tensor_sha(control) != EXPECTED_CONTROL_SHA
            or candidate.get("control_sha256") != EXPECTED_CONTROL_SHA
            or candidate.get("continuation_plan_sha256") != plan_digest
            or candidate.get("status") != "nonpositive_or_unresolved_face_curvature"
            or preflight.get("point_sha256") != EXPECTED_CONTROL_SHA
            or preflight.get("source_archive_closure") is not True
            or candidate.get("source_transition") != {
                "path": "src/advar/transport.py", "before": TRANSPORT_BEFORE,
                "after": TRANSPORT_AFTER}):
        raise ValueError("stored face-chart candidate/preflight does not match the pinned 8eff6b1f point")

    case = torch.load(CASE, map_location="cpu", weights_only=False)
    problem = case["problem"]
    parameters = case["parameters"]
    truth = case["truth"]
    if (_tensor_sha(parameters) != candidate.get("input_parameters_sha256")
            or truth.shape != (4, 5)):
        raise ValueError("fixed parameters or terminal truth differ from the face-chart candidate")
    reference = _read(REFERENCE)
    ref_resource = _read(REFERENCE_RESOURCE)
    if (reference.get("status") != "completed"
            or reference.get("closure", {}).get("all_signed_source_hashes_unchanged") is not True
            or reference.get("closure", {}).get("all_signed_archive_hashes_unchanged") is not True
            or reference.get("closure", {}).get("input_identity_unchanged") is not True
            or reference.get("closure", {}).get("runtime_identity_unchanged") is not True
            or reference.get("provenance_digests", {}).get("fixed_case_sha256") != CASE_SHA
            or ref_resource.get("exit_code") != 0):
        raise ValueError("saved independent semidiscrete truth/data receipt does not close")
    reference_arrays_path = ROOT / reference["artifacts"]["arrays"]
    if reference_arrays_path.resolve() != REFERENCE_ARRAYS.resolve():
        raise ValueError("independent reference array path differs from its receipt")
    independent = np.load(reference_arrays_path, allow_pickle=False)
    archived_truth = np.ascontiguousarray(independent["archived_terminal_dbz"])
    if not np.array_equal(archived_truth, np.ascontiguousarray(truth.numpy())):
        raise ValueError("saved fixed truth differs from previously computed independent reference")

    if time.monotonic() >= deadline:
        raise TimeoutError("30-second shared restoration/forecast allocation expired before native J")
    with torch.no_grad():
        native_j = problem.objective(control, parameters)
    if native_j.ndim != 0 or not bool(torch.isfinite(native_j)):
        raise ValueError("restored candidate native J must be a finite scalar")
    native_j_value = float(native_j)
    if abs(native_j_value - float(candidate["native_J"])) > 64 * torch.finfo(torch.float64).eps * max(abs(native_j_value), 1.0):
        raise ValueError("independent native J evaluation differs from stored candidate J")
    if time.monotonic() >= deadline:
        raise TimeoutError("native J exhausted the 30-second shared restoration/forecast allocation")

    from advar import variational as v
    from advar.physics import echo_to_dbz
    frozen = problem.contract(parameters)
    try:
        analyzed = v.analysis_trajectory(control, frozen)
        if (analyzed.psi_coefficients is None or analyzed.support_frames is None
                or time.monotonic() >= deadline):
            raise TimeoutError("analysis restoration exceeded its bounded allocation")
        spec = frozen.fv_transport
        if spec is None:
            raise ValueError("pinned fixed case does not provide FV transport")
        forecast_echo, forecast_support = v.finite_volume_trajectory(
            analyzed.frames_linear[-1], analyzed.support_frames[-1],
            analyzed.psi_coefficients, analyzed.log_growth_per_step,
            psi_basis=spec.psi_basis, leads=frozen.nowcast_config.forecast_steps,
            substeps_per_interval=spec.substeps_per_interval,
            interval_seconds=frozen.nowcast_config.interval_minutes * 60.0,
            spacing_yx=spec.spacing_yx,
            boundary_echo=problem.future_boundary_echo,
            boundary_support=problem.future_boundary_support,
            reconstruction=spec.reconstruction, max_courant=spec.max_courant,
            replay=spec.replay,
        )
    except Exception as error:
        failure = {
            "status": "forecast_failed", "candidate_control_sha256": EXPECTED_CONTROL_SHA,
            "candidate_receipt_sha256": _sha(CANDIDATE), "minimum_claim": False,
            "selection_use": False, "native_J_independent_evaluation": native_j_value,
            "input_parameters_sha256": _tensor_sha(parameters), "truth_sha256": _tensor_sha(truth),
            "fixed_case_sha256": _sha(CASE), "continuation_plan_sha256": plan_digest,
            "runtime": runtime, "transport_support_only_transition": {
                "path": "src/advar/transport.py", "before": TRANSPORT_BEFORE,
                "after": TRANSPORT_AFTER}, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.monotonic() - started,
            "allocation_wall_seconds": WALL_LIMIT_SECONDS,
        }
        OUTPUT.write_text(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return failure
    if time.monotonic() >= deadline:
        raise TimeoutError("forecast restoration exceeded its bounded allocation")
    forecast_dbz = echo_to_dbz(forecast_echo, min_dbz=frozen.nowcast_config.min_dbz)
    analysis_dbz = echo_to_dbz(analyzed.frames_linear[-1], min_dbz=frozen.nowcast_config.min_dbz)
    interval_minutes = frozen.nowcast_config.interval_minutes
    lead_axis = tuple(interval_minutes * index for index in range(forecast_dbz.shape[0]))
    if not set(LEADS_MINUTES).issubset(lead_axis):
        raise ValueError(f"forecast does not cover requested relative leads: {lead_axis}")
    selected = {str(lead): forecast_dbz[lead_axis.index(lead)] for lead in LEADS_MINUTES}
    terminal = forecast_dbz[lead_axis.index(180)]
    terminal_diff = terminal - truth
    terminal_e = float(terminal_diff.square().mean())
    terminal_mae = float(terminal_diff.abs().mean())

    fields = {"control": control, "parameters": parameters,
        "analysis_echo": analyzed.frames_linear[-1], "analysis_support": analyzed.support_frames[-1],
        "analysis_dbz": analysis_dbz, "forecast_echo": forecast_echo,
        "forecast_support": forecast_support, "forecast_dbz": forecast_dbz,
        "terminal_truth_dbz": truth,
        **{f"lead_{lead:03d}_dbz": selected[str(lead)] for lead in LEADS_MINUTES}}
    np.savez_compressed(FIELDS, **{key: value.detach().cpu().numpy() for key, value in fields.items()})

    pins_after = _validate(plan, plan_digest)
    runner_sha_after = _sha(SELF)
    elapsed = time.monotonic() - started
    if elapsed > WALL_LIMIT_SECONDS or pins_before != pins_after or runner_sha_before != runner_sha_after:
        raise ValueError("forecast restoration exceeded its budget or changed pinned inputs")
    ledger_before = candidate.get("ledger", {})
    cumulative = plan["cumulative_budget"]
    successful_before = (int(cumulative["previous_successful_native_J_evaluations"])
        + int(cumulative["budget_after_equivalence_probes"]))
    charged_before = float(ledger_before.get("charged_s", 0.0))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    peak_bytes = int(peak * rss_scale)
    result = {
        "status": "completed", "candidate_control_sha256": EXPECTED_CONTROL_SHA,
        "candidate_receipt_sha256": _sha(CANDIDATE),
        "candidate_phase_status": candidate.get("status"),
        "minimum_claim": False, "score_selection_use": False,
        "native_J_independent_evaluation": native_j_value,
        "native_J_stored": float(candidate["native_J"]),
        "input_parameters_sha256": _tensor_sha(parameters),
        "truth_sha256": _tensor_sha(truth), "fixed_case_sha256": _sha(CASE),
        "independent_reference_json_sha256": _sha(REFERENCE),
        "independent_reference_arrays_sha256": _sha(REFERENCE_ARRAYS),
        "runtime": runtime, "continuation_plan_sha256": plan_digest,
        "continuation_source_bundle_sha256": _sha(CONTINUATION / "source.tar.gz"),
        "transport_support_only_transition": {
            "path": "src/advar/transport.py", "before": TRANSPORT_BEFORE,
            "after": TRANSPORT_AFTER, "verified_current_sha256": _sha(TRANSPORT)},
        "source_hashes_before_after_equal": pins_before == pins_after,
        "runner_source_sha256_before_after": runner_sha_before,
        "source_pin_count": len(pins_before),
        "current_face_preflight_sha256": _sha(PREFLIGHT),
        "analysis_lead": "end of interval 2; restored from the saved 26-control candidate",
        "forecast_leads_minutes_after_analysis": list(LEADS_MINUTES),
        "absolute_model_minutes_from_initial_condition": [20 + lead for lead in LEADS_MINUTES],
        "terminal_future_E_observation_only": {"lead_minutes_after_analysis": 180,
            "mean_squared_dbz_error": terminal_e, "mean_absolute_dbz_error": terminal_mae,
            "selection_use": False},
        "forecast_fields_npz": FIELDS.name, "forecast_fields_sha256": _sha(FIELDS),
        "analysis_state_shape": list(analysis_dbz.shape),
        "forecast_fields_shape": list(forecast_dbz.shape),
        "successful_native_J_evaluations_added": 1,
        "cumulative_budget": {
            "no_new_grant": True,
            "native_search_wall_cap_seconds": cumulative["total_original_search_wall_cap_seconds"],
            "charged_seconds_before": charged_before,
            "this_forecast_seconds": elapsed,
            "charged_seconds_after": charged_before + elapsed,
            "remaining_seconds_after": float(cumulative["total_original_search_wall_cap_seconds"])
                - charged_before - elapsed,
            "successful_native_J_evaluations_before": successful_before,
            "this_run_native_J_evaluations": 1,
            "successful_native_J_evaluations_after": successful_before + 1,
            "successful_native_J_evaluations_remaining": int(cumulative["max_successful_native_J_evaluations"])
                - successful_before - 1,
        },
        "forecast_trajectories": 1, "forecast_seconds": elapsed,
        "allocation_wall_seconds": WALL_LIMIT_SECONDS,
        "sampled_process_peak_rss_bytes": peak_bytes,
        "initial_process_peak_rss_bytes": int(initial_rss * rss_scale),
        "output_fields": sorted(fields),
        "supersedes": "candidate_8eff6b1f_forecast_preliminary_superseded.json (lead indexing error; retained for audit)",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
