"""Correctly compare saved candidate fields with the saved generator truth; no FV calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ATTEMPT = Path(__file__).resolve().parent / "attempt1"
RECEIPT = ATTEMPT / "candidate_8eff6b1f_forecast.json"
FIELDS_PATH = ATTEMPT / "candidate_8eff6b1f_forecast_fields.npz"
REFERENCE_JSON = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.json"
REFERENCE_NPZ = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012/independent_semidiscrete_reference.npz"
FIXED_CASE = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
SUMMARY = ATTEMPT / "candidate_8eff6b1f_reference_comparison.json"
STATE_AUX = ATTEMPT / "candidate_8eff6b1f_state_auxiliary.npz"
PREVIOUS_SUMMARY = ATTEMPT / "candidate_8eff6b1f_reference_comparison_cross_control_superseded.json"
PREVIOUS_STATE = ATTEMPT / "candidate_8eff6b1f_state_auxiliary_preinterpretation.npz"
LEADS = (10, 30, 60, 180)
FIELD_INDICES = {10: 1, 30: 3, 60: 6, 180: 18}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _tensor_sha(value: torch.Tensor) -> str:
    return _sha_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _flow_flux(control: torch.Tensor, frozen: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from advar import transport

    spec = frozen.fv_transport
    coefficients = spec.coefficient_limits * torch.tanh(control[20:25])
    psi = torch.einsum("k,kij->ij", coefficients, spec.psi_basis)
    qx, qy = transport.face_volume_fluxes(psi)
    growth = frozen.nowcast_config.max_log_growth_per_step * torch.tanh(control[25])
    return coefficients, torch.cat((qx.flatten(), qy.flatten())), growth


def run() -> dict:
    old_summary = ATTEMPT / "candidate_8eff6b1f_reference_comparison.json"
    if PREVIOUS_SUMMARY.exists() or PREVIOUS_STATE.exists():
        raise ValueError("previous interpretation has already been preserved")
    if old_summary.exists() and not PREVIOUS_SUMMARY.exists():
        old_summary.rename(PREVIOUS_SUMMARY)
    if STATE_AUX.exists() and not PREVIOUS_STATE.exists():
        STATE_AUX.rename(PREVIOUS_STATE)

    receipt = _read(RECEIPT)
    independent = _read(REFERENCE_JSON)
    candidate_fields = np.load(FIELDS_PATH, allow_pickle=False)
    reference = np.load(REFERENCE_NPZ, allow_pickle=False)
    case = torch.load(FIXED_CASE, map_location="cpu", weights_only=False)
    problem, original = case["problem"], case["original"]
    parameters, archived_truth = case["parameters"], case["truth"]
    control = torch.from_numpy(candidate_fields["control"].copy())
    frozen = problem.contract(parameters)
    from advar import variational as v
    from advar.physics import dbz_to_echo

    if (receipt.get("candidate_control_sha256") != _tensor_sha(control)
            or receipt.get("input_parameters_sha256") != _tensor_sha(parameters)
            or _tensor_sha(original) == _tensor_sha(control)
            or independent.get("status") != "completed"
            or independent.get("closure", {}).get("all_signed_source_hashes_unchanged") is not True
            or independent.get("closure", {}).get("all_signed_archive_hashes_unchanged") is not True
            or independent.get("candidate_control_sha256") != "96108bdcf2cd6abc05878bf2cfc9e2531f37cf82e152ad1a9eaf0a67de746411"):
        raise ValueError("saved candidate/reference identity or closure does not match the expected receipts")
    if not np.array_equal(reference["archived_terminal_dbz"], archived_truth.numpy()):
        raise ValueError("fixed terminal truth differs from the previously saved reference receipt")
    fractions_original = torch.tanh(original[20:25])
    growth_original = frozen.nowcast_config.max_log_growth_per_step * torch.tanh(original[25])
    if (not torch.equal(original[:20], torch.zeros_like(original[:20]))
            or not np.allclose(fractions_original.numpy(), (0.7, -0.6, 0.5, 0.4, -0.3), rtol=0, atol=2e-15)
            or abs(float(growth_original) - 0.08) > 2e-15):
        raise ValueError("independent truth generator does not match the documented original control")

    field_count = frozen.active_field_index.numel()
    field_control = torch.zeros_like(frozen.initial_background_dbz).flatten().scatter(
        0, frozen.active_field_index, control[:field_count]).reshape_as(frozen.initial_background_dbz)
    candidate_initial_dbz = v._initial_analysis_dbz(field_control, frozen)
    generator_initial_dbz = problem.background_dbz
    candidate_initial_echo = dbz_to_echo(candidate_initial_dbz, min_dbz=frozen.nowcast_config.min_dbz)
    generator_initial_echo = dbz_to_echo(generator_initial_dbz, min_dbz=frozen.nowcast_config.min_dbz)
    candidate_coeff, candidate_flux, candidate_growth = _flow_flux(control, frozen)
    original_coeff, original_flux, original_growth = _flow_flux(original, frozen)
    flow_delta = candidate_flux - original_flux
    original_flux_norm = torch.linalg.vector_norm(original_flux)
    if not bool(original_flux_norm > 0):
        raise ValueError("pinned original generator has no physical face flux")
    initial_delta = candidate_initial_dbz - generator_initial_dbz

    forecast = candidate_fields["forecast_dbz"]
    truth = reference["independent_dbz"]
    reference_leads = ((reference["times_seconds"] / 60.0) - 20.0).astype(int)
    if (forecast.shape != (19, 4, 5) or truth.shape != (18, 4, 5)
            or tuple(reference_leads.tolist()) != tuple(range(10, 181, 10))):
        raise ValueError("saved candidate forecasts and truth reference do not share the 10-minute lead schedule")

    rows: list[dict[str, object]] = []
    for lead in LEADS:
        forecast_index = FIELD_INDICES[lead]
        reference_index = int(np.flatnonzero(reference_leads == lead)[0])
        candidate_frame = forecast[forecast_index]
        reference_frame = truth[reference_index]
        persistence_frame = np.broadcast_to(candidate_fields["analysis_dbz"], reference_frame.shape)
        difference = candidate_frame - reference_frame
        persistence_difference = persistence_frame - reference_frame
        rows.append({
            "lead_minutes_after_analysis": lead,
            "candidate_forecast_index": forecast_index,
            "independent_truth_index": reference_index,
            "absolute_model_time_seconds": float(reference["times_seconds"][reference_index]),
            "candidate_truth_mae_dbz": float(np.mean(np.abs(difference))),
            "candidate_truth_mse_dbz2": float(np.mean(difference * difference)),
            "persistence_truth_mae_dbz": float(np.mean(np.abs(persistence_difference))),
            "persistence_truth_mse_dbz2": float(np.mean(persistence_difference * persistence_difference)),
            "candidate_forecast_field_sha256": _sha_bytes(np.ascontiguousarray(candidate_frame).tobytes()),
        })

    aux_fields = {
        "candidate_initial_analysis_dbz": candidate_initial_dbz,
        "candidate_initial_analysis_echo": candidate_initial_echo,
        "generator_initial_dbz": generator_initial_dbz,
        "generator_initial_echo": generator_initial_echo,
        "initial_support": frozen.initial_support_mask,
        "analysis_end_echo": torch.from_numpy(candidate_fields["analysis_echo"].copy()),
        "analysis_end_dbz": torch.from_numpy(candidate_fields["analysis_dbz"].copy()),
        "analysis_end_support": torch.from_numpy(candidate_fields["analysis_support"].copy()),
        "candidate_flow_coefficients": candidate_coeff,
        "generator_flow_coefficients": original_coeff,
        "candidate_flow_flux_vector": candidate_flux,
        "generator_flow_flux_vector": original_flux,
        "candidate_growth_per_interval": candidate_growth.reshape(1),
        "generator_growth_per_interval": original_growth.reshape(1),
    }
    np.savez_compressed(STATE_AUX, **{key: value.detach().cpu().numpy() for key, value in aux_fields.items()})

    result = {
        "status": "derived_from_saved_arrays_no_new_forward",
        "candidate_control_sha256": receipt["candidate_control_sha256"],
        "truth_generator_control_sha256": _tensor_sha(original),
        "prior_reference_candidate_control_sha256": independent["candidate_control_sha256"],
        "prior_reference_candidate_sha256_role": "metadata for the forecast previously compared by the independent-reference producer; not the truth generator control",
        "candidate_and_truth_generator_controls_match": _tensor_sha(control) == _tensor_sha(original),
        "selection_use": False, "new_truth_generation": False,
        "native_J_evaluations": 0, "forecast_trajectories": 0,
        "reference_truth_generation": {
            "source": "pinned fixed-case original control and independent NumPy minmod-MUSCL DOP853 run",
            "initial_field": "fixed problem.background_dbz (original first 20 field controls are zero)",
            "flow_fractions": fractions_original.tolist(),
            "flow_coefficients": original_coeff.tolist(),
            "growth_per_600_seconds": float(original_growth),
            "integrator": independent["integrator"],
            "saved_reference_json_sha256": _sha(REFERENCE_JSON),
            "saved_reference_arrays_sha256": _sha(REFERENCE_NPZ),
            "terminal_reference_vs_archived_truth_mae_dbz": independent["archived_same_kernel_terminal_truth_comparison"]["mean_abs_dbz_difference"],
        },
        "candidate_state_error_vs_truth_generator": {
            "initial_dbz_mae": float(initial_delta.abs().mean()),
            "initial_dbz_mse": float(initial_delta.square().mean()),
            "initial_dbz_max_abs": float(initial_delta.abs().max()),
            "candidate_initial_dbz_range": [float(candidate_initial_dbz.min()), float(candidate_initial_dbz.max())],
            "generator_initial_dbz_range": [float(generator_initial_dbz.min()), float(generator_initial_dbz.max())],
            "flow_fraction_candidate": torch.tanh(control[20:25]).tolist(),
            "flow_fraction_generator": fractions_original.tolist(),
            "flow_coefficients_candidate": candidate_coeff.tolist(),
            "flow_coefficients_generator": original_coeff.tolist(),
            "flow_flux_relative_l2_error": float(torch.linalg.vector_norm(flow_delta) / original_flux_norm),
            "flow_flux_max_abs_error": float(flow_delta.abs().max()),
            "growth_per_600_seconds_candidate": float(candidate_growth),
            "growth_per_600_seconds_generator": float(original_growth),
            "growth_difference": float(candidate_growth - original_growth),
        },
        "forecast_fields_sha256": _sha(FIELDS_PATH),
        "state_auxiliary_npz": STATE_AUX.name,
        "state_auxiliary_sha256": _sha(STATE_AUX),
        "lead_comparisons_against_independent_truth": rows,
        "terminal_truth_at_180_minutes": {
            "mae_dbz": rows[-1]["candidate_truth_mae_dbz"],
            "mse_dbz2": rows[-1]["candidate_truth_mse_dbz2"],
            "terminal_reference_is_true_generator_trajectory": True,
            "selection_use": False,
        },
        "caveat": "The prior independent-reference receipt's 96108 control hash identifies the candidate forecast that producer compared against its independent truth. The independent_dbz array itself is generated from the pinned original control (zero field controls, flow fractions 0.7,-0.6,0.5,0.4,-0.3, growth 0.08 per 600 seconds).",
    }
    SUMMARY.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
