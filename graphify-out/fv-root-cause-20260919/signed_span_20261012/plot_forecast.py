"""Export the measured lead-error curve for the case-review HTML."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent
data = json.loads((WORK / "forecast_validation_attempt1/reference_comparison/METRICS.json").read_text())
metrics = data["metrics_vs_independent_semidiscrete_reference"]["96108"]
lead = data["timing"]["lead_minutes_after_analysis"]
persistence = metrics["same_96108_analysis_frame_persistence"]["per_lead_mae_dbz"]
fig, axis = plt.subplots(figsize=(9.0, 4.3), layout="constrained")
axis.plot(lead, metrics["per_lead_mae_dbz"], "o-", color="#7a1f2b", label="Signed-candidate forecast", markersize=4)
axis.plot(lead, persistence, "s-", color="#1b4c63", label="Hold the same analyzed frame", markersize=3)
axis.axvline(110, color="#7b816d", linestyle=":", linewidth=1)
axis.text(112, 0.25, "Forecast worse from +110 min", fontsize=9, color="#5c6354")
axis.set(xlabel="Lead after the 20-minute analysis window (min)", ylabel="Full 4 x 5 grid MAE (dBZ)",
         title="Independent semidiscrete reference: local J/R gain does not establish forecast skill",
         xlim=(10, 180), ylim=(0, 2.6))
axis.set_xticks([10, 40, 70, 100, 130, 160, 180])
axis.grid(alpha=0.2)
axis.legend(frameon=False, loc="upper left")
fig.savefig(WORK / "forecast_mae.svg")
fig.savefig(WORK / "forecast_mae.png", dpi=150)
plt.close(fig)
