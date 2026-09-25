"""Read-only numerical parity and identity distinction for two fixed exclusions."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
missing = json.loads((EVIDENCE / "point_missing_centered_attempt1/point_centered_response.json").read_text())
qc = json.loads((EVIDENCE / "point_qc_centered_attempt1/point_centered_response.json").read_text())

if missing["input_before"] == qc["input_before"]:
    raise SystemExit("status-1 missing and status-2 QC identities must differ")
if [missing["numerical_status"], qc["numerical_status"]] != ["eligible", "eligible"]:
    raise SystemExit("both fixed-mask response records must be eligible")
for section, keys in (
    ("nominal", ("objective", "score", "gradient_max", "hessian_audit", "branch")),
    ("response", ("direct", "indirect", "total", "direct_gradient",
                  "indirect_gradient", "total_gradient", "true_adjoint_relative_residual")),
    ("tangent", ("control_direction", "true_relative_residual")),
):
    for key in keys:
        if missing[section][key] != qc[section][key]:
            raise SystemExit(f"fixed-mask numerical mismatch: {section}.{key}")
for index in range(4):
    for key in ("control", "objective", "score", "gradient_max", "branch"):
        if missing["endpoints"][index][key] != qc["endpoints"][index][key]:
            raise SystemExit(f"fixed-mask endpoint mismatch: {index}.{key}")
if missing["pairs"] != qc["pairs"]:
    raise SystemExit("fixed-mask signed pairs differ")
print("status identities differ; selected numerical response, endpoints and signed pairs match")
