"""Recompute forecast/reference errors from saved arrays only; no FV calls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "graphify-out/fv-root-cause-20260919/signed_span_20261012"
ATTEMPT = WORK / "forecast_validation_attempt1"
OUT = ATTEMPT / "reference_comparison"
POINTS = ("2a4d", "fb0a", "96108")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def main() -> None:
    validation_plan = read_json(WORK / "FORECAST_PLAN.json")
    validation_plan_sha = sha(WORK / "FORECAST_PLAN.json")
    if validation_plan_sha != (WORK / "FORECAST_PLAN_SHA256.txt").read_text().strip():
        raise ValueError("forecast validation plan seal mismatch")
    base_plan = read_json(WORK / "PLAN.json")
    inherited = base_plan
    if (len(validation_plan["source_files"]) != 237
            or validation_plan["source_files"] != inherited["source_files"]
            or validation_plan["inherited_archive_count"] != 306
            or validation_plan["signed_span_plan_sha256"] != sha(WORK / "PLAN.json")):
        raise ValueError("comparison is not bound to the frozen 237-source signed run")
    source_mismatches = [name for name, digest in validation_plan["source_files"].items()
        if sha(ROOT / name) != digest]
    archive_mismatches = [name for name, digest in validation_plan["archive_files"].items()
        if sha(ROOT / name) != digest]
    if source_mismatches or archive_mismatches:
        raise ValueError(f"pinned source/archive mismatch: sources={source_mismatches[:3]}, archives={archive_mismatches[:3]}")

    producer_path = ATTEMPT / "forecast_validation.json"
    run_path = ATTEMPT / "forecast_validation.run.json"
    fields_path = ATTEMPT / "forecast_fields.pt"
    producer, run = read_json(producer_path), read_json(run_path)
    fields = torch.load(fields_path, map_location="cpu", weights_only=True)
    if (producer.get("status") != "completed" or producer.get("score_calls_completed") != 3
            or producer.get("forecast_calls") != 3 or producer.get("optimizer_steps") != 0
            or producer.get("candidate_evaluations") != 0 or producer.get("hvp_calls") != 0
            or producer.get("selection_use") is not False
            or run.get("execution_status") != "completed"
            or run.get("child_sha256") != sha(producer_path)
            or producer.get("forecast_fields_sha256") != sha(fields_path)
            or not all(producer.get(key) is True for key in (
                "source_unchanged", "input_unchanged", "runtime_unchanged"))):
        raise ValueError("three-score producer receipt does not close")
    if tuple(fields["field_frames_dbz"][POINTS[0]].shape) != (18, 4, 5) or set(fields["controls"]) != set(POINTS):
        raise ValueError("saved forecast fields do not match the 18×4×5 three-point contract")
    if {p: producer["score_results"][p]["control_sha256"] for p in POINTS} != fields["controls"]:
        raise ValueError("saved forecast field controls differ from score producer controls")

    reference_json_path = WORK / "independent_semidiscrete_reference.json"
    reference_npz_path = WORK / "independent_semidiscrete_reference.npz"
    reference_resource_path = WORK / "independent_reference.resource.json"
    reference_script_path = WORK / "independent_semidiscrete_reference.py"
    reference = read_json(reference_json_path)
    resource = read_json(reference_resource_path)
    arrays = np.load(reference_npz_path, allow_pickle=False)
    candidate_receipt_path = ROOT / reference["candidate_receipt"]
    fixed_case_path = ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"
    candidate_receipt = read_json(candidate_receipt_path)
    candidate_control = torch.as_tensor(candidate_receipt["control"], dtype=torch.float64)
    candidate_control_sha = hashlib.sha256(
        candidate_control.contiguous().numpy().tobytes()).hexdigest()
    if (reference.get("status") != "completed"
            or reference.get("candidate_control_sha256") != fields["controls"]["96108"]
            or reference.get("pinned_case_sha256") != sha(fixed_case_path)
            or reference.get("provenance_digests", {}).get("reference_script_sha256") != sha(reference_script_path)
            or reference.get("provenance_digests", {}).get("candidate_receipt_sha256") != sha(candidate_receipt_path)
            or reference.get("provenance_digests", {}).get("fixed_case_sha256") != sha(fixed_case_path)
            or reference.get("closure", {}).get("all_signed_source_hashes_unchanged") is not True
            or reference.get("closure", {}).get("all_signed_archive_hashes_unchanged") is not True
            or resource.get("exit_code") != 0):
        raise ValueError("independent reference receipt/provenance does not close")
    signed_plan_sha = (WORK / "PLAN_SHA256.txt").read_text().strip()
    if (reference.get("signed_plan_sha256") != signed_plan_sha
            or reference.get("signed_source_count") != 237
            or reference.get("signed_archive_count") != 306
            or candidate_receipt.get("control_sha256") != fields["controls"]["96108"]
            or candidate_control_sha != fields["controls"]["96108"]):
        raise ValueError("independent reference is not for this signed control and frozen source scope")

    independent_dbz = np.ascontiguousarray(arrays["independent_dbz"])
    times = np.ascontiguousarray(arrays["times_seconds"])
    leads = np.asarray(fields["lead_minutes_after_analysis"], dtype=np.float64)
    if (independent_dbz.shape != (18, 4, 5) or times.shape != (18,)
            or not np.array_equal(times, 1200.0 + 60.0 * leads)
            or not np.array_equal(leads, np.arange(10, 181, 10, dtype=np.float64))):
        raise ValueError("reference and saved forecasts differ in lead times/domain")
    if not np.array_equal(np.asarray(arrays["candidate_forecast_dbz"]),
            fields["field_frames_dbz"]["96108"].numpy()):
        raise ValueError("saved signed forecast differs from the reference producer candidate forecast")

    case = torch.load(fixed_case_path, map_location="cpu", weights_only=False)
    truth_tensor = case["truth"]
    truth = np.ascontiguousarray(truth_tensor.numpy())
    archived_truth = np.ascontiguousarray(arrays["archived_terminal_dbz"])
    truth_hash = hashlib.sha256(truth_tensor.contiguous().numpy().tobytes()).hexdigest()
    if truth.shape != (4, 5) or not np.array_equal(truth, archived_truth):
        raise ValueError("independent reference's archived terminal truth differs from pinned fixed case")
    original_truth_mse: dict[str, float] = {}
    original_truth_mae: dict[str, float] = {}
    per_point: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    for point in POINTS:
        forecast = np.ascontiguousarray(fields["field_frames_dbz"][point].numpy())
        diff = forecast - independent_dbz
        abs_diff = np.abs(diff)
        lead_mse = np.mean(diff * diff, axis=(1, 2))
        lead_mae = np.mean(abs_diff, axis=(1, 2))
        lead_max = np.max(abs_diff, axis=(1, 2))
        terminal_residual = forecast[-1] - truth
        score_value = producer["score_results"][point]["terminal_dbz_mse"]
        truth_mse = float(np.mean(terminal_residual * terminal_residual))
        tolerance = 128 * np.finfo(np.float64).eps * max(abs(score_value), abs(truth_mse), np.finfo(np.float64).tiny)
        if abs(score_value - truth_mse) > tolerance:
            raise ValueError(f"saved terminal field does not reproduce score at {point}")
        original_truth_mse[point] = truth_mse
        original_truth_mae[point] = float(np.mean(np.abs(terminal_residual)))
        per_point[point] = {
            "control_sha256": fields["controls"][point],
            "mean_18lead_mse_dbz2": float(np.mean(diff * diff)),
            "mean_18lead_mae_dbz": float(np.mean(abs_diff)),
            "max_abs_error_18leads_dbz": float(np.max(abs_diff)),
            "terminal_mse_dbz2_vs_independent_reference": float(lead_mse[-1]),
            "terminal_mae_dbz_vs_independent_reference": float(lead_mae[-1]),
            "terminal_max_abs_error_dbz_vs_independent_reference": float(lead_max[-1]),
            "per_lead_mse_dbz2": lead_mse.tolist(),
            "per_lead_mae_dbz": lead_mae.tolist(),
            "per_lead_max_abs_error_dbz": lead_max.tolist(),
            "terminal_mse_dbz2_vs_archived_truth": truth_mse,
            "terminal_mae_dbz_vs_archived_truth": original_truth_mae[point],
        }
        for i, seconds in enumerate(times):
            rows.append({"point": point, "lead_minutes_after_analysis": int(leads[i]),
                "absolute_time_seconds": float(seconds), "mse_vs_reference_dbz2": float(lead_mse[i]),
                "mae_vs_reference_dbz": float(lead_mae[i]),
                "max_abs_error_vs_reference_dbz": float(lead_max[i]),
                "persistence_mse_dbz2": "", "persistence_mae_dbz": "",
                "forecast_minus_persistence_mae_dbz": ""})

    persistence_key = "analyzed_frame_persistence_dbz"
    persistence = np.ascontiguousarray(arrays[persistence_key])
    if persistence.shape != independent_dbz.shape:
        raise ValueError("96108 persistence array does not share reference shape")
    pdiff = persistence - independent_dbz
    persistence_mae = np.mean(np.abs(pdiff), axis=(1, 2))
    persistence_mse = np.mean(pdiff * pdiff, axis=(1, 2))
    selected_mae = np.asarray(per_point["96108"]["per_lead_mae_dbz"])
    mae_difference = selected_mae - persistence_mae
    loses_indices = np.flatnonzero(mae_difference > 0.0)
    first_loses = int(loses_indices[0]) if loses_indices.size else None
    selected = per_point["96108"]
    selected["same_96108_analysis_frame_persistence"] = {
        "mean_18lead_mse_dbz2": float(np.mean(persistence_mse)),
        "mean_18lead_mae_dbz": float(np.mean(persistence_mae)),
        "terminal_mse_dbz2": float(persistence_mse[-1]),
        "terminal_mae_dbz": float(persistence_mae[-1]),
        "per_lead_mse_dbz2": persistence_mse.tolist(),
        "per_lead_mae_dbz": persistence_mae.tolist(),
    }
    selected["forecast_minus_persistence_mae_dbz"] = {
        "mean_18leads": float(np.mean(selected_mae - persistence_mae)),
        "terminal": float(selected_mae[-1] - persistence_mae[-1]),
        "first_lead_where_forecast_mae_exceeds_persistence": (
            None if first_loses is None else int(leads[first_loses])),
        "previous_lead_minutes": (None if first_loses in (None, 0) else int(leads[first_loses - 1])),
    }
    for row in rows:
        if row["point"] == "96108":
            i = row["lead_minutes_after_analysis"] // 10 - 1
            row["persistence_mse_dbz2"] = float(persistence_mse[i])
            row["persistence_mae_dbz"] = float(persistence_mae[i])
            row["forecast_minus_persistence_mae_dbz"] = float(mae_difference[i])

    recovery_path = WORK / "PHYSICAL_RECOVERY.json"
    recovery = read_json(recovery_path)
    if recovery.get("candidate_control") != fields["controls"]["96108"]:
        raise ValueError("algebra-only physical-recovery evidence is for another control")
    inputs = {
        "forecast_plan_sha256": validation_plan_sha,
        "forecast_producer_sha256": sha(producer_path),
        "forecast_run_sha256": sha(run_path),
        "forecast_fields_sha256": sha(fields_path),
        "reference_json_sha256": sha(reference_json_path),
        "reference_npz_sha256": sha(reference_npz_path),
        "reference_script_sha256": sha(reference_script_path),
        "reference_resource_sha256": sha(reference_resource_path),
        "signed_plan_sha256": signed_plan_sha,
        "signed_step_sha256": sha(WORK / "attempt1/step.json"),
        "candidate_receipt_sha256": sha(candidate_receipt_path),
        "fixed_case_sha256": sha(fixed_case_path),
        "archived_terminal_truth_tensor_sha256": truth_hash,
        "independent_dbz_array_sha256": hashlib.sha256(independent_dbz.tobytes()).hexdigest(),
        "times_seconds_sha256": hashlib.sha256(times.tobytes()).hexdigest(),
        "physical_recovery_sha256": sha(recovery_path),
        "comparison_script_sha256": sha(Path(__file__)),
    }
    comparison = {
        "status": "completed_read_only_same_domain_comparison",
        "new_fv_calls": 0,
        "shape": [18, 4, 5],
        "score_domain": "full 4x5 terminal verification grid",
        "timing": {"absolute_times_seconds": times.tolist(),
            "lead_minutes_after_analysis": leads.astype(int).tolist()},
        "controls": fields["controls"],
        "metrics_vs_independent_semidiscrete_reference": per_point,
        "terminal_score_recomputed_from_saved_frame_vs_archived_truth": {
            "mse_dbz2": original_truth_mse, "mae_dbz": original_truth_mae,
            "matches_score_producer": True},
        "reference_scope": reference["scope"],
        "truth_scope": "one pinned terminal 4x5 field; no original 18-lead truth sequence",
        "limits": {"leadwise_observed_skill_claim": False,
            "continuous_pde_truth_claim": False,"held_out_weather_skill_claim": False,
            "persistence_comparable_only_for_96108": True},
        "physical_call_accounting": {
            "GREEN_score_forecasts_in_forecast_validation_run": 3,
            "RED_exploratory_candidate_FV_forwards": 4,
            "final_reference_producer_FV_forwards": 1,
            "RED_reference_producer_FV_forward_total": 5,
            "reference_rerun_guard_seconds": float(resource["elapsed_seconds"]),
            "reference_rerun_peak_rss_bytes": int(resource["sampled_peak_rss_bytes"]),
            "this_comparison_new_FV_forwards": 0,
            "adjacent_12_by_12_analytic_cases_counted_as_forecasts": 0,
            "note": "Per-run counts are separate: 3 GREEN scoring forwards; 4 RED exploratory plus 1 final reference-producer forward. The comparison itself only reads saved arrays."},
        "physical_recovery_link": {"path": recovery_path.relative_to(ROOT).as_posix(),
            "scope": recovery["scope"],
            "candidate_J": recovery["candidate_J"],
            "generator_prior_lower_bound_on_J": recovery["generator_prior_lower_bound_on_J"],
            "ratio": recovery["generator_J_lower_bound_over_candidate_J"],
            "no_new_FV": True},
        "input_hashes": inputs,
    }
    json_path = OUT / "METRICS.json"
    csv_path = OUT / "PER_LEAD.csv"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics": json_path.relative_to(ROOT).as_posix(),
        "metrics_sha256": sha(json_path), "csv": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": sha(csv_path),
        "mean_mae": {p: per_point[p]["mean_18lead_mae_dbz"] for p in POINTS},
        "mean_mse": {p: per_point[p]["mean_18lead_mse_dbz2"] for p in POINTS},
        "mae_crossover_lead_min": selected["forecast_minus_persistence_mae_dbz"][
            "first_lead_where_forecast_mae_exceeds_persistence"]}, sort_keys=True))


if __name__ == "__main__":
    main()
