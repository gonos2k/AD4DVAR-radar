"""One sealed observation run for the current native24 chord endpoint."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import sys
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
PLAN = HERE / "OBSERVER_PLAN.json"
PLAN_SHA = HERE / "OBSERVER_PLAN_SHA256.txt"
OUTPUT = HERE / "final_native_forecast_20261013.json"
FIELDS = HERE / "final_native_forecast_20261013.npz"
ENDPOINT_DIR = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012/two_face_intersection_20261013/continuation2_20261013"
ENDPOINT = ENDPOINT_DIR / "native24_chord_extension_result.json"
ENDPOINT_CLOSURE = ENDPOINT_DIR / "native24_chord_readonly_terminal_closure.json"
ENDPOINT_PLAN = ENDPOINT_DIR / "CHORD_PLAN_v2.json"
CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
REFERENCE = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.json"
REFERENCE_ARRAYS = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.npz"
REFERENCE_RESOURCE = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_reference.resource.json"
CURRENT_TRANSPORT_SHA = "f6c680da7ca130fcbc505a03b73741591dc023f57a7745e47cfd49dd72622fab"
TERMINAL_CONTROL_SHA = "6ca56498f82002634a5103fecd41b1ec9d67e8ff745ecbe9b554906043c0c4d5"
TERMINAL_NATIVE_J = 0.0601191808341518
BASE_NATIVE_J_CALLS = 493
GLOBAL_NATIVE_J_CAP = 512
WALL_CAP_SECONDS = 3600.0
LEADS_MINUTES = tuple(range(10, 181, 10))
SUMMARY_LEADS = (10, 30, 60, 180)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"pinned file missing or unsafe: {path}")
    return _sha_bytes(path.read_bytes())


def _tensor_sha(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("observer tensors must be finite CPU FP64")
    return _sha_bytes(value.detach().contiguous().numpy().tobytes())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "CPU FP64"}


def _verify_inputs(plan: dict[str, Any], plan_sha: str) -> tuple[dict[str, Any], torch.Tensor,
        torch.Tensor, torch.Tensor, np.ndarray, dict[str, Any], dict[str, Any]]:
    if _sha(PLAN) != plan_sha or PLAN_SHA.read_text().strip() != plan_sha:
        raise ValueError("observer plan digest/sidecar mismatch")
    if plan.get("status") != "frozen_current_endpoint_observer_plan":
        raise ValueError("observer plan is not frozen")
    if plan.get("runtime_identity") != _runtime():
        raise ValueError("observer runtime differs from the frozen plan")
    if plan.get("runner_sha256") != _sha(SELF):
        raise ValueError("observer runner differs from its frozen source pin")
    for name, expected in {**plan["source_hashes"], **plan["archive_hashes"]}.items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"observer source/archive pin mismatch: {name}")

    endpoint = _read(ENDPOINT)
    closure = _read(ENDPOINT_CLOSURE)
    endpoint_plan = _read(ENDPOINT_PLAN)
    endpoint_plan_sha = _sha(ENDPOINT_PLAN)
    endpoint_plan_sidecar = ENDPOINT_DIR / "CHORD_PLAN_v2_SHA256.txt"
    if (endpoint.get("plan_sha256") != endpoint_plan_sha
            or endpoint_plan_sha != plan.get("endpoint_plan_sha256")
            or endpoint_plan_sidecar.read_text().strip() != endpoint_plan_sha
            or endpoint.get("status") != "one_chord_nonstationary"
            or endpoint.get("phase") != "terminal_first_order_not_met"
            or endpoint.get("current_control_sha256") != TERMINAL_CONTROL_SHA
            or endpoint.get("current_point", {}).get("control_sha256") != TERMINAL_CONTROL_SHA
            or abs(float(endpoint.get("native_J", math.inf)) - TERMINAL_NATIVE_J) > 1e-14):
        raise ValueError("current chord endpoint does not match the frozen native24 endpoint")
    if (closure.get("execution", {}).get("status") != "one_chord_nonstationary"
            or closure.get("execution", {}).get("source_archive_closure") is not True
            or closure.get("qualification", {}).get("minimum_claim") is not False
            or closure.get("qualification", {}).get("exact_terminal_hessian") is not False
            or closure.get("hashes", {}).get("raw_result_sha256") != _sha(ENDPOINT)
            or closure.get("hashes", {}).get("terminal_control_sha256") != TERMINAL_CONTROL_SHA
            or closure.get("accounting", {}).get("native_J_total") != BASE_NATIVE_J_CALLS
            or abs(float(closure.get("accounting", {}).get("corrected_remaining_seconds", -1.0))
                - float(plan["budget"]["remaining_seconds_before"])) > 1e-9):
        raise ValueError("read-only current endpoint closure or cumulative ledger mismatch")
    transition = endpoint.get("transport_extension_transition")
    if (not isinstance(transition, dict) or transition.get("path") != "src/advar/transport.py"
            or transition.get("after") != CURRENT_TRANSPORT_SHA):
        raise ValueError("current endpoint does not carry the tuple-based transport source pin")
    if endpoint_plan.get("transport_extension_transition") != transition:
        raise ValueError("current endpoint plan/result source transition differs")

    case = torch.load(CASE, map_location="cpu", weights_only=False)
    problem = case["problem"]
    control = torch.as_tensor(endpoint["current_control"], dtype=torch.float64, device="cpu")
    parameters = case["parameters"]
    original = case["original"]
    truth = case["truth"]
    if (_tensor_sha(control) != TERMINAL_CONTROL_SHA
            or _tensor_sha(parameters) != plan["input_hashes"]["parameters_sha256"]
            or _tensor_sha(original) != plan["input_hashes"]["original_control_sha256"]
            or _tensor_sha(truth) != plan["input_hashes"]["archived_terminal_truth_sha256"]
            or _tensor_sha(problem.background_dbz) != plan["input_hashes"]["background_dbz_sha256"]
            or _tensor_sha(problem.background_pattern) != plan["input_hashes"]["background_pattern_sha256"]
            or _tensor_sha(problem.observation_dbz) != plan["input_hashes"]["observation_dbz_sha256"]
            or _tensor_sha(problem.observation_std_dbz) != plan["input_hashes"]["observation_std_dbz_sha256"]
            or _tensor_sha(problem.verification_dbz) != plan["input_hashes"]["verification_dbz_sha256"]
            or problem.identity.get("fixed_problem_sha256") != plan["fixed_problem_sha256"]):
        raise ValueError("current endpoint control or fixed-case inputs differ from the observer plan")

    ref = _read(REFERENCE)
    resource_receipt = _read(REFERENCE_RESOURCE)
    if (ref.get("status") != "completed"
            or ref.get("closure", {}).get("all_signed_source_hashes_unchanged") is not True
            or ref.get("closure", {}).get("all_signed_archive_hashes_unchanged") is not True
            or ref.get("closure", {}).get("input_identity_unchanged") is not True
            or ref.get("closure", {}).get("runtime_identity_unchanged") is not True
            or ref.get("provenance_digests", {}).get("fixed_case_sha256") != _sha(CASE)
            or resource_receipt.get("exit_code") != 0
            or ref.get("initial_field_source") != "pinned problem.background_dbz full 4x5 input, checked equal to frozen.initial_background_dbz; forward_case.make_case copies this zero-flow case initial field before truth generation; converted to q by archived min_dbz"):
        raise ValueError("stored independent DOP853 truth/reference provenance does not close")
    arrays_path = ROOT / ref["artifacts"]["arrays"]
    if arrays_path.resolve() != REFERENCE_ARRAYS.resolve():
        raise ValueError("independent truth array path differs from its signed receipt")
    arrays = np.load(arrays_path, allow_pickle=False)
    independent_truth = np.ascontiguousarray(arrays["independent_dbz"])
    times = np.ascontiguousarray(arrays["times_seconds"])
    if (independent_truth.shape != (18, 4, 5)
            or not np.array_equal(times, np.arange(1800.0, 12001.0, 600.0))
            or ref.get("times_seconds") != times.tolist()
            or not np.array_equal(np.ascontiguousarray(arrays["archived_terminal_dbz"]),
                np.ascontiguousarray(truth.numpy()))):
        raise ValueError("stored 18-lead independent truth or terminal archive does not match the fixed case")
    generating_fractions = torch.tanh(original[20:25]).tolist()
    generating_growth = (case["problem"].frozen.nowcast_config.max_log_growth_per_step
        * torch.tanh(original[25]))
    if (not np.allclose(generating_fractions, ref.get("flow_fractions_from_pinned_original_control"),
                rtol=0.0, atol=2e-15)
            or abs(float(generating_growth) - float(ref.get("growth_per_600_seconds", math.inf))) > 2e-15):
        raise ValueError("stored independent trajectory generating flow/growth differs from case.original")
    if _tensor_sha(torch.as_tensor(independent_truth, dtype=torch.float64)) != plan["input_hashes"]["independent_truth_sha256"]:
        raise ValueError("independent truth tensor hash differs from the observer plan")

    current_plan_budget = plan["budget"]
    if (current_plan_budget.get("native_J_successes_before") != BASE_NATIVE_J_CALLS
            or current_plan_budget.get("phase_native_J_limit") != 1
            or not 0 < float(current_plan_budget.get("phase_seconds", 0.0)) <= 30.0
            or BASE_NATIVE_J_CALLS + int(current_plan_budget["phase_native_J_limit"]) > GLOBAL_NATIVE_J_CAP
            or float(current_plan_budget.get("charged_seconds_before", math.inf))
                + float(current_plan_budget.get("phase_seconds", math.inf)) > WALL_CAP_SECONDS
            or float(current_plan_budget.get("remaining_seconds_before", 0.0))
                != float(closure["accounting"]["corrected_remaining_seconds"])):
        raise ValueError("observer phase would reset or exceed the shared time/J budget")
    return case, control, parameters, original, independent_truth, ref, closure


def _initial_dbz(control: torch.Tensor, frozen: Any) -> torch.Tensor:
    import advar.variational as variational
    size = frozen.active_field_index.numel()
    field = torch.zeros_like(frozen.initial_background_dbz).flatten().scatter(
        0, frozen.active_field_index, control[:size]).reshape_as(frozen.initial_background_dbz)
    return variational._initial_analysis_dbz(field, frozen)


def _metrics(forecast: np.ndarray, truth: np.ndarray, persistence: np.ndarray) -> dict[str, Any]:
    error = forecast - truth
    persist_error = persistence - truth
    lead_mae = np.mean(np.abs(error), axis=(1, 2))
    lead_mse = np.mean(np.square(error), axis=(1, 2))
    persist_mae = np.mean(np.abs(persist_error), axis=(1, 2))
    persist_mse = np.mean(np.square(persist_error), axis=(1, 2))
    result: dict[str, Any] = {
        "truth_source": "stored independent NumPy minmod RHS + SciPy DOP853 trajectory generated from the original control; not continuous-PDE or held-out weather truth",
        "forecast_lead_minutes_after_analysis": list(LEADS_MINUTES),
        "absolute_model_minutes_from_initial_condition": list(range(30, 201, 10)),
        "forecast_mean_mae_dbz_all_18_leads": float(lead_mae.mean()),
        "forecast_aggregate_mae_dbz_all_cells_leads": float(np.abs(error).mean()),
        "forecast_aggregate_mse_dbz2_all_cells_leads": float(np.square(error).mean()),
        "forecast_max_abs_error_dbz_all_leads": float(np.abs(error).max()),
        "persistence_mean_mae_dbz_all_18_leads": float(persist_mae.mean()),
        "persistence_aggregate_mse_dbz2_all_cells_leads": float(np.square(persist_error).mean()),
        "forecast_minus_persistence_mean_mae_dbz": float(lead_mae.mean() - persist_mae.mean()),
        "forecast_per_lead_mae_dbz": lead_mae.tolist(),
        "forecast_per_lead_mse_dbz2": lead_mse.tolist(),
        "persistence_per_lead_mae_dbz": persist_mae.tolist(),
        "persistence_per_lead_mse_dbz2": persist_mse.tolist(),
        "first_persistence_mae_crossover_minutes": next((LEADS_MINUTES[i]
            for i in range(len(LEADS_MINUTES)) if lead_mae[i] <= persist_mae[i]), None),
        "selected_observation_leads": {},
        "selection_use": False,
    }
    for lead in SUMMARY_LEADS:
        index = lead // 10 - 1
        result["selected_observation_leads"][str(lead)] = {
            "forecast_mae_dbz": float(lead_mae[index]),
            "forecast_mse_dbz2": float(lead_mse[index]),
            "forecast_max_abs_error_dbz": float(np.abs(error[index]).max()),
            "persistence_mae_dbz": float(persist_mae[index]),
            "persistence_mse_dbz2": float(persist_mse[index]),
            "absolute_model_minute": 30 + lead - 10,
            "selection_use": False,
        }
    return result


def run() -> dict[str, Any]:
    if OUTPUT.exists() or FIELDS.exists():
        raise ValueError("current endpoint observer output must be fresh")
    start = time.monotonic()
    wall_cap = 30.0
    deadline = start + wall_cap
    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    plan_sha = _sha(PLAN)
    plan = _read(PLAN)
    runner_sha_before = _sha(SELF)
    source_hashes_before = {**plan["source_hashes"], **plan["archive_hashes"]}
    evaluation_counts = {"native_J_started": 0, "native_J_completed": 0,
        "analysis_passes_inside_native_J": 0,
        "analysis_trajectory_calls": 0, "forecast_trajectory_calls": 0,
        "forecast_steps": 0, "analysis_interval_outputs": 0,
        "forecast_interval_outputs": 0, "total_model_trajectory_passes": 0,
        "total_interval_outputs": 0, "score_calls": 0}
    try:
        case, control, parameters, original, independent_truth, ref, closure = _verify_inputs(plan, plan_sha)
        problem = case["problem"]
        frozen = problem.contract(parameters)
        spec = frozen.fv_transport
        if spec is None:
            raise ValueError("pinned current case has no FV transport")
        from advar import transport, variational
        from advar.physics import echo_to_dbz
        if (transport._selected_face.get() is not None or spec.replay is not True
                or plan["profile_identity"].get("original_replay") is not True
                or problem.identity.get("fixed_problem_sha256") != plan["fixed_problem_sha256"]):
            raise ValueError("observer requires ambient normal selector, fixed problem, and original replay=True")

        evaluation_counts["native_J_started"] = 1
        if time.monotonic() >= deadline:
            raise TimeoutError("observer wall allocation expired before independent native J")
        with torch.no_grad():
            native_j_tensor = problem.objective(control, parameters)
        if native_j_tensor.ndim != 0 or not bool(torch.isfinite(native_j_tensor)):
            raise ValueError("independent current endpoint native J is not a finite scalar")
        native_j = float(native_j_tensor)
        evaluation_counts["native_J_completed"] = 1
        evaluation_counts["analysis_passes_inside_native_J"] = 1
        if abs(native_j - TERMINAL_NATIVE_J) > 64 * torch.finfo(torch.float64).eps * max(1.0, abs(native_j)):
            raise ValueError("independent current native J differs from stored endpoint J")
        if transport._selected_face.get() is not None:
            raise ValueError("native J left a selected-face extension active")

        if time.monotonic() >= deadline:
            raise TimeoutError("observer wall allocation expired before analysis replay")
        with torch.no_grad():
            analyzed = variational.analysis_trajectory(control, frozen)
        evaluation_counts["analysis_trajectory_calls"] = 1
        if (analyzed.psi_coefficients is None or analyzed.support_frames is None
                or analyzed.log_growth_per_step is None):
            raise ValueError("analysis replay did not produce the pinned FV dynamics")
        if analyzed.frames_linear.shape[0] != 3:
            raise ValueError("analysis replay did not return the two expected intervals")
        evaluation_counts["analysis_interval_outputs"] = 2
        if time.monotonic() >= deadline:
            raise TimeoutError("observer wall allocation expired before the 18-lead forecast")
        forecast_echo, forecast_support = variational.finite_volume_trajectory(
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
        evaluation_counts["forecast_trajectory_calls"] = 1
        evaluation_counts["forecast_steps"] = int(frozen.nowcast_config.forecast_steps)
        if time.monotonic() >= deadline:
            raise TimeoutError("18-lead forecast exceeded the observer wall allocation")
        forecast_dbz_all = echo_to_dbz(forecast_echo, min_dbz=frozen.nowcast_config.min_dbz)
        if forecast_dbz_all.shape[0] != 19 or len(LEADS_MINUTES) != 18:
            raise ValueError("forecast must contain t=0 plus all 18 ten-minute leads")
        evaluation_counts["forecast_interval_outputs"] = forecast_dbz_all.shape[0] - 1
        evaluation_counts["total_model_trajectory_passes"] = 3
        evaluation_counts["total_interval_outputs"] = 2 + 2 + (forecast_dbz_all.shape[0] - 1)
        forecast_dbz = forecast_dbz_all[1:]
        truth_np = independent_truth
        forecast_np = forecast_dbz.detach().cpu().numpy().copy()
        if forecast_np.shape != truth_np.shape:
            raise ValueError("forecast and stored independent truth trajectory shapes differ")
        analysis_dbz = echo_to_dbz(analyzed.frames_linear[-1], min_dbz=frozen.nowcast_config.min_dbz)
        persistence_np = np.broadcast_to(analysis_dbz.detach().cpu().numpy(), truth_np.shape).copy()
        metrics = _metrics(forecast_np, truth_np, persistence_np)

        original_initial = _initial_dbz(original, frozen)
        candidate_initial = _initial_dbz(control, frozen)
        field_error = candidate_initial - original_initial
        size = frozen.active_field_index.numel()
        dynamic = control[size:]
        original_dynamic = original[size:]
        coefficient_count = spec.coefficient_limits.numel()
        from advar.variational import bounded_fv_coefficients
        coefficient_kwargs = {"psi_basis": spec.psi_basis,
            "coefficient_limits": spec.coefficient_limits,
            "dt_seconds": frozen.nowcast_config.interval_minutes * 60.0 / spec.substeps_per_interval,
            "spacing_yx": spec.spacing_yx, "reconstruction": spec.reconstruction,
            "max_courant": spec.max_courant}
        current_coeff = bounded_fv_coefficients(dynamic[:coefficient_count], **coefficient_kwargs)
        original_coeff = bounded_fv_coefficients(original_dynamic[:coefficient_count], **coefficient_kwargs)
        current_psi = torch.einsum("k,kij->ij", current_coeff, spec.psi_basis)
        original_psi = torch.einsum("k,kij->ij", original_coeff, spec.psi_basis)
        current_qx, current_qy = transport.face_volume_fluxes(current_psi)
        original_qx, original_qy = transport.face_volume_fluxes(original_psi)
        current_flux = torch.cat((current_qx.reshape(-1), current_qy.reshape(-1)))
        original_flux = torch.cat((original_qx.reshape(-1), original_qy.reshape(-1)))
        flux_denominator = max(float(torch.linalg.vector_norm(original_flux)), torch.finfo(torch.float64).tiny)
        current_growth = frozen.nowcast_config.max_log_growth_per_step * torch.tanh(dynamic[coefficient_count])
        original_growth = frozen.nowcast_config.max_log_growth_per_step * torch.tanh(original_dynamic[coefficient_count])
        state_comparison = {
            "comparison_control_source": "case.original generating control versus current terminal native24 chord endpoint",
            "initial_background_dbz_sha256": _tensor_sha(frozen.initial_background_dbz),
            "background_dbz_sha256": _tensor_sha(problem.background_dbz),
            "background_pattern_sha256": _tensor_sha(problem.background_pattern),
            "background_beta_parameter": float(parameters[-1]),
            "control_echo_block_l2_difference": float(torch.linalg.vector_norm(control[:size] - original[:size])),
            "control_echo_block_max_abs_difference": float((control[:size] - original[:size]).abs().max()),
            "initial_dbz_candidate_vs_original_control": {
                "mae_dbz": float(field_error.abs().mean()),
                "rmse_dbz": float(field_error.square().mean().sqrt()),
                "max_abs_dbz": float(field_error.abs().max()),
            },
            "flow_coefficients_candidate": current_coeff.tolist(),
            "flow_coefficients_original": original_coeff.tolist(),
            "flow_coefficients_relative_l2_error": float(torch.linalg.vector_norm(current_coeff - original_coeff)
                / torch.linalg.vector_norm(original_coeff).clamp_min(torch.finfo(torch.float64).tiny)),
            "flow_flux_relative_l2_error": float(torch.linalg.vector_norm(current_flux - original_flux)
                / flux_denominator),
            "growth_log_per_step_candidate": float(current_growth),
            "growth_log_per_step_original": float(original_growth),
            "growth_log_per_step_difference": float(current_growth - original_growth),
            "growth_relative_error": float((current_growth - original_growth).abs()
                / original_growth.abs().clamp_min(torch.finfo(torch.float64).tiny)),
            "selection_use": False,
        }

        fields = {
            "current_control": control, "original_control": original,
            "parameters": parameters, "terminal_truth_archived": case["truth"],
            "independent_truth_dbz_18leads": torch.as_tensor(truth_np, dtype=torch.float64),
            "candidate_initial_dbz": candidate_initial, "original_control_initial_dbz": original_initial,
            "analyzed_terminal_dbz": analysis_dbz, "analysis_echo_trajectory": analyzed.frames_linear,
            "analysis_support_trajectory": analyzed.support_frames,
            "forecast_echo_t0_plus_18": forecast_echo,
            "forecast_support_t0_plus_18": forecast_support,
            "forecast_dbz_18leads": forecast_dbz,
            "persistence_dbz_18leads": torch.as_tensor(persistence_np, dtype=torch.float64),
            "current_psi_coefficients": current_coeff, "original_psi_coefficients": original_coeff,
            "current_psi": current_psi, "original_psi": original_psi,
            "current_qx": current_qx, "current_qy": current_qy,
            "original_qx": original_qx, "original_qy": original_qy,
        }
        np.savez_compressed(FIELDS, **{name: value.detach().cpu().numpy() for name, value in fields.items()})
        pins_after = {name: _sha(ROOT / name) for name in source_hashes_before}
        elapsed = time.monotonic() - start
        if (pins_after != source_hashes_before or _sha(SELF) != runner_sha_before
                or elapsed > float(plan["budget"]["phase_seconds"])):
            raise ValueError("observer source closure or shared wall allowance failed")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if platform.system() == "Darwin" else 1024
        successful_before = int(plan["budget"]["native_J_successes_before"])
        result = {
            "status": "completed", "endpoint_control_sha256": TERMINAL_CONTROL_SHA,
            "endpoint_result_sha256": _sha(ENDPOINT), "endpoint_closure_sha256": _sha(ENDPOINT_CLOSURE),
            "endpoint_plan_sha256": _sha(ENDPOINT_PLAN), "observer_plan_sha256": plan_sha,
            "observer_runner_sha256": runner_sha_before, "source_hashes_before_after_equal": True,
            "source_hashes": source_hashes_before,
            "transport_source_transition": plan["source_transition"],
            "runtime_identity": _runtime(), "fixed_problem_sha256": problem.identity["fixed_problem_sha256"],
            "input_hashes": plan["input_hashes"], "profile_identity_from_endpoint": plan["profile_identity"],
            "original_replay": spec.replay, "ambient_selected_face_extension": None,
            "candidate_native_J_independent": native_j, "candidate_native_J_stored": TERMINAL_NATIVE_J,
            "forecast_metrics": metrics, "state_errors_vs_original": state_comparison,
            "evaluation_counts": evaluation_counts, "score_calls": 0,
            "future_truth_use": "observation-only; excluded from candidate selection and minimum claims",
            "minimum_claim": False, "response_claim": False, "selection_use": False,
            "forecast_trajectories": 1, "forecast_fields_npz": FIELDS.name,
            "forecast_fields_sha256": _sha(FIELDS), "forecast_fields": sorted(fields),
            "forecast_shape": list(forecast_dbz.shape),
            "analysis_shape": list(analysis_dbz.shape), "elapsed_seconds": elapsed,
            "sampled_peak_rss_bytes": int(peak * scale), "initial_process_peak_rss_bytes": int(initial_peak * scale),
            "cumulative_budget": {
                "original_wall_cap_seconds": WALL_CAP_SECONDS,
                "charged_seconds_before": plan["budget"]["charged_seconds_before"],
                "this_phase_seconds": elapsed,
                "charged_seconds_after": float(plan["budget"]["charged_seconds_before"]) + elapsed,
                "remaining_seconds_after": float(plan["budget"]["remaining_seconds_before"]) - elapsed,
                "native_J_calls_before": successful_before,
                "this_phase_native_J_calls": evaluation_counts["native_J_completed"],
                "native_J_calls_after": successful_before + evaluation_counts["native_J_completed"],
                "native_J_calls_remaining_of_512": GLOBAL_NATIVE_J_CAP - successful_before - evaluation_counts["native_J_completed"],
                "no_new_grant": True,
            },
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return result
    except Exception as error:
        failure = {"status": "observer_failed", "error_type": type(error).__name__,
            "error": str(error), "observer_plan_sha256": plan_sha,
            "endpoint_control_sha256": TERMINAL_CONTROL_SHA,
            "evaluation_counts": evaluation_counts, "score_calls": 0,
            "minimum_claim": False, "selection_use": False,
            "elapsed_seconds": time.monotonic() - start,
            "cumulative_budget_before_seconds": plan.get("budget", {}).get("charged_seconds_before"),
            "cumulative_native_J_before": plan.get("budget", {}).get("native_J_successes_before")}
        OUTPUT.write_text(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return failure


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
