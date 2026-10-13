"""Plot saved 8eff native forecast against truth and analysis persistence; no model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
NATIVE = EVIDENCE / "native_reference_20261012/continuation3/attempt1"
REFERENCE = EVIDENCE / "signed_span_20261012/independent_semidiscrete_reference.npz"
FORECAST = NATIVE / "candidate_8eff6b1f_forecast.json"
REFERENCE_COMPARISON = NATIVE / "candidate_8eff6b1f_reference_comparison.json"
OUT = EVIDENCE / "html_case_review_20261012"
SVG = OUT / "native_forecast_8eff6b1f_mae.svg"
PNG = OUT / "native_forecast_8eff6b1f_mae.png"
SUMMARY = OUT / "native_forecast_8eff6b1f_mae.json"
CURRENT_OBSERVER = EVIDENCE / "native_reference_20261012/current_endpoint_observer_20261013"
CURRENT_FORECAST = CURRENT_OBSERVER / "final_native_forecast_20261013.json"
CURRENT_VERIFY = CURRENT_OBSERVER / "OBSERVER_VERIFICATION.json"
CURRENT_FIELDS = CURRENT_OBSERVER / "final_native_forecast_20261013.npz"
CURRENT_SVG = OUT / "native_forecast_6ca56498_mae.svg"
CURRENT_PNG = OUT / "native_forecast_6ca56498_mae.png"
CURRENT_SUMMARY = OUT / "native_forecast_6ca56498_mae.json"
CURRENT_A_RESULT = (EVIDENCE / "native_reference_20261012/two_face_intersection_20261013"
                    / "continuation2_20261013/native24_chord_extension_result.json")
INDEPENDENT_REFERENCE = EVIDENCE / "signed_span_20261012/independent_semidiscrete_reference.npz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> dict[str, object]:
    forecast = json.loads(FORECAST.read_text())
    comparison = json.loads(REFERENCE_COMPARISON.read_text())
    fields_path = NATIVE / forecast["forecast_fields_npz"]
    if (_sha(fields_path) != forecast["forecast_fields_sha256"]
            or _sha(REFERENCE) != comparison["reference_truth_generation"]["saved_reference_arrays_sha256"]
            or forecast["candidate_control_sha256"] != comparison["candidate_control_sha256"]
            or forecast["score_selection_use"] is not False
            or comparison["selection_use"] is not False
            or comparison["new_truth_generation"] is not False):
        raise ValueError("native forecast/independent truth receipts do not match their saved hashes")

    with np.load(fields_path) as fields, np.load(REFERENCE) as reference:
        forecast_dbz = np.asarray(fields["forecast_dbz"], dtype=np.float64)
        analysis_dbz = np.asarray(fields["analysis_dbz"], dtype=np.float64)
        truth_dbz = np.asarray(reference["independent_dbz"], dtype=np.float64)
        times_seconds = np.asarray(reference["times_seconds"], dtype=np.float64)
    if (forecast_dbz.shape != (19, 4, 5) or analysis_dbz.shape != (4, 5)
            or truth_dbz.shape != (18, 4, 5) or times_seconds.shape != (18,)):
        raise ValueError("saved candidate/reference arrays do not have the expected 18-lead 4×5 shapes")
    leads_minutes = (times_seconds - 1200.0) / 60.0
    requested_leads = np.arange(10.0, 181.0, 10.0)
    if not np.array_equal(leads_minutes, requested_leads):
        raise ValueError("saved independent truth lead times do not align with the native forecast")

    candidate = forecast_dbz[1:]
    persistence = np.broadcast_to(analysis_dbz, truth_dbz.shape)
    candidate_mae = np.abs(candidate - truth_dbz).mean(axis=(1, 2))
    persistence_mae = np.abs(persistence - truth_dbz).mean(axis=(1, 2))
    candidate_mse = np.square(candidate - truth_dbz).mean(axis=(1, 2))
    persistence_mse = np.square(persistence - truth_dbz).mean(axis=(1, 2))
    for row in comparison["lead_comparisons_against_independent_truth"]:
        index = int(row["lead_minutes_after_analysis"] // 10) - 1
        if (index < 0 or index >= len(leads_minutes)
                or row["candidate_forecast_index"] != index + 1
                or row["independent_truth_index"] != index
                or not np.isclose(candidate_mae[index], row["candidate_truth_mae_dbz"], rtol=0, atol=2e-12)
                or not np.isclose(persistence_mae[index], row["persistence_truth_mae_dbz"], rtol=0, atol=2e-12)):
            raise ValueError("selected lead receipt does not match the 18-row plotted MAE arrays")

    first_worse = int(np.flatnonzero(candidate_mae > persistence_mae)[0]) if np.any(candidate_mae > persistence_mae) else None
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.plot(leads_minutes, candidate_mae, color="#2563eb", linewidth=2.4, marker="o",
              markersize=4.2, label="8eff native forecast")
    axis.plot(leads_minutes, persistence_mae, color="#d97706", linewidth=2.1, marker="s",
              markersize=3.8, label="analysis-end persistence")
    if first_worse is not None:
        first_worse_lead = int(leads_minutes[first_worse])
        axis.axvline(first_worse_lead, color="#b91c1c", linewidth=1, linestyle="--", alpha=0.65)
        axis.annotate(f"first worse: +{first_worse_lead} min", (first_worse_lead, candidate_mae[first_worse]),
                      xytext=(7, 9), textcoords="offset points", color="#991b1b", fontsize=9)
    axis.set_title("8eff native forecast · MAE against the generating truth")
    axis.set_xlabel("Lead after analysis window (minutes)")
    axis.set_ylabel("Mean absolute dBZ error")
    axis.set_xticks(np.arange(10, 181, 20))
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(0.99, 0.02, "single synthetic case · E observer only · not used for selection",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=8.5, color="#555555")
    matplotlib.rcParams["svg.hashsalt"] = "advar-8eff-native-forecast-mae-v1"
    figure.savefig(SVG, format="svg", metadata={"Date": None})
    figure.savefig(PNG, format="png", dpi=180)
    plt.close(figure)

    summary: dict[str, object] = {
        "scope": "saved 8eff forecast/reference arrays only; no FV, score, or HVP calls",
        "candidate_control_sha256": forecast["candidate_control_sha256"],
        "truth_generator_control_sha256": comparison["truth_generator_control_sha256"],
        "prior_96108_role": comparison["prior_reference_candidate_sha256_role"],
        "forecast_fields_sha256": _sha(fields_path),
        "independent_reference_arrays_sha256": _sha(REFERENCE),
        "lead_minutes_after_analysis": [int(value) for value in leads_minutes],
        "candidate_mae_dbz": candidate_mae.tolist(),
        "analysis_persistence_mae_dbz": persistence_mae.tolist(),
        "candidate_mse_dbz2": candidate_mse.tolist(),
        "analysis_persistence_mse_dbz2": persistence_mse.tolist(),
        "first_lead_candidate_worse_than_persistence_minutes": (
            None if first_worse is None else int(leads_minutes[first_worse])),
        "truth_generator": comparison["reference_truth_generation"],
        "selection_use": False,
        "svg_sha256": _sha(SVG),
        "png_sha256": _sha(PNG),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def render_current_endpoint() -> dict[str, object]:
    """Render the latest 6ca endpoint from observer arrays without new evaluation."""
    receipt = json.loads(CURRENT_FORECAST.read_text())
    verification = json.loads(CURRENT_VERIFY.read_text())
    current = json.loads(CURRENT_A_RESULT.read_text())
    if (_sha(CURRENT_FIELDS) != receipt.get("forecast_fields_sha256")
            or verification.get("status") != "passed"
            or verification.get("metric_recomputed_from_saved_npz") is not True
            or verification.get("selection_use") is not False
            or verification.get("score_calls") != 0
            or verification.get("fields_sha256") != receipt.get("forecast_fields_sha256")
            or verification.get("result_sha256") != _sha(CURRENT_FORECAST)
            or receipt.get("status") != "completed"
            or receipt.get("selection_use") is not False
            or receipt.get("minimum_claim") is not False
            or receipt.get("response_claim") is not False
            or receipt.get("score_calls") != 0
            or receipt.get("candidate_native_J_independent") != receipt.get("candidate_native_J_stored")
            or receipt.get("endpoint_control_sha256") != current.get("current_control_sha256")
            or receipt.get("candidate_native_J_stored") != current.get("native_J")):
        raise ValueError("latest native observer or endpoint receipts failed their closure checks")

    with np.load(CURRENT_FIELDS) as fields, np.load(INDEPENDENT_REFERENCE) as reference:
        candidate = np.asarray(fields["forecast_dbz_18leads"], dtype=np.float64)
        persistence = np.asarray(fields["persistence_dbz_18leads"], dtype=np.float64)
        truth = np.asarray(fields["independent_truth_dbz_18leads"], dtype=np.float64)
        original_control = np.asarray(fields["original_control"], dtype=np.float64)
        archived_truth = np.asarray(reference["independent_dbz"], dtype=np.float64)
        truth_times = np.asarray(reference["times_seconds"], dtype=np.float64)
    if (candidate.shape != (18, 4, 5) or persistence.shape != candidate.shape
            or truth.shape != candidate.shape or archived_truth.shape != candidate.shape
            or original_control.shape != (26,) or not np.array_equal(truth, archived_truth)):
        raise ValueError("latest native forecast arrays do not align with the stored true-generator trajectory")
    truth_generator_control_sha = hashlib.sha256(original_control.tobytes()).hexdigest()
    if truth_generator_control_sha != "adc5bfd01e98e32b09ccc963f096a6f6a3c7307d600a0ba20f8ddc8dcad3721e":
        raise ValueError("latest observer fields changed the fixed truth-generator control")
    leads_minutes = np.asarray(receipt["forecast_metrics"]["forecast_lead_minutes_after_analysis"], dtype=np.float64)
    if (leads_minutes.shape != (18,) or not np.array_equal(leads_minutes, np.arange(10.0, 181.0, 10.0))
            or not np.array_equal((truth_times - 1200.0) / 60.0, leads_minutes)):
        raise ValueError("latest native forecast lead rows are not aligned to the 20-minute analysis end")

    candidate_mae = np.abs(candidate - truth).mean(axis=(1, 2))
    persistence_mae = np.abs(persistence - truth).mean(axis=(1, 2))
    candidate_mse = np.square(candidate - truth).mean(axis=(1, 2))
    persistence_mse = np.square(persistence - truth).mean(axis=(1, 2))
    metrics = receipt["forecast_metrics"]
    if (not np.allclose(candidate_mae, metrics["forecast_per_lead_mae_dbz"], rtol=0, atol=2e-12)
            or not np.allclose(persistence_mae, metrics["persistence_per_lead_mae_dbz"], rtol=0, atol=2e-12)
            or not np.isclose(float(candidate_mae.mean()), metrics["forecast_mean_mae_dbz_all_18_leads"], rtol=0, atol=2e-12)):
        raise ValueError("latest native per-lead MAE differs from the verified observer receipt")
    first_worse = np.flatnonzero(candidate_mae > persistence_mae)
    first_worse_lead = None if first_worse.size == 0 else int(leads_minutes[int(first_worse[0])])

    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.plot(leads_minutes, candidate_mae, color="#2563eb", linewidth=2.4, marker="o",
              markersize=4.2, label="6ca current native forecast")
    axis.plot(leads_minutes, persistence_mae, color="#d97706", linewidth=2.1, marker="s",
              markersize=3.8, label="analysis-end persistence")
    if first_worse_lead is not None:
        index = int(first_worse[0])
        axis.axvline(first_worse_lead, color="#b91c1c", linewidth=1, linestyle="--", alpha=0.65)
        axis.annotate(f"first worse: +{first_worse_lead} min",
                      (first_worse_lead, candidate_mae[index]), xytext=(7, 9),
                      textcoords="offset points", color="#991b1b", fontsize=9)
    axis.set_title("6ca current native forecast · MAE against the generating truth")
    axis.set_xlabel("Lead after analysis window (minutes)")
    axis.set_ylabel("Mean absolute dBZ error")
    axis.set_xticks(np.arange(10, 181, 20))
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(0.99, 0.02, "one synthetic case · E observer only · selection disabled",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=8.5, color="#555555")
    matplotlib.rcParams["svg.hashsalt"] = "advar-6ca-current-forecast-mae-v1"
    figure.savefig(CURRENT_SVG, format="svg", metadata={"Date": None})
    figure.savefig(CURRENT_PNG, format="png", dpi=180)
    plt.close(figure)

    summary: dict[str, object] = {
        "scope": "latest 6ca observer arrays and stored DOP853 truth only; no FV/J/HVP/score calls",
        "candidate_control_sha256": receipt["endpoint_control_sha256"],
        "truth_generator_control_sha256": truth_generator_control_sha,
        "endpoint_plan_sha256": receipt["endpoint_plan_sha256"],
        "forecast_receipt_sha256": _sha(CURRENT_FORECAST),
        "observer_verification_sha256": _sha(CURRENT_VERIFY),
        "forecast_fields_sha256": _sha(CURRENT_FIELDS),
        "independent_reference_arrays_sha256": _sha(INDEPENDENT_REFERENCE),
        "lead_minutes_after_analysis": [int(value) for value in leads_minutes],
        "candidate_mae_dbz": candidate_mae.tolist(),
        "persistence_mae_dbz": persistence_mae.tolist(),
        "candidate_mse_dbz2": candidate_mse.tolist(),
        "persistence_mse_dbz2": persistence_mse.tolist(),
        "candidate_mean_mae_dbz": float(candidate_mae.mean()),
        "persistence_mean_mae_dbz": float(persistence_mae.mean()),
        "first_lead_candidate_worse_than_persistence_minutes": first_worse_lead,
        "first_lead_candidate_better_than_persistence_minutes": (
            None if not np.any(candidate_mae < persistence_mae)
            else int(leads_minutes[int(np.flatnonzero(candidate_mae < persistence_mae)[0])])),
        "truth_generator": receipt["forecast_metrics"]["truth_source"],
        "selection_use": False,
        "svg_sha256": _sha(CURRENT_SVG),
        "png_sha256": _sha(CURRENT_PNG),
    }
    CURRENT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    results = {"8eff": render(), "6ca": render_current_endpoint()}
    print(json.dumps({label: {key: result[key] for key in (
        "candidate_control_sha256", "truth_generator_control_sha256",
        "first_lead_candidate_worse_than_persistence_minutes", "svg_sha256", "png_sha256")}
        for label, result in results.items()}, indent=2))
