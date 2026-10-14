"""Independently compare terminal retained-graph gradients to native replay."""
from pathlib import Path
import hashlib
import json
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from advar import transport
from examples.weather_scenarios.fv_point_3h_netflux_current_cycle import load_fixed_case

HERE = Path(__file__).resolve().parent
RAW = HERE / "attempt1/RESULT.json"
OUTPUT = HERE / "TERMINAL_REPLAY_CHECK.json"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if OUTPUT.exists():
        raise ValueError("terminal replay check already exists")
    record = json.loads(RAW.read_text())
    plan = record["plan"]
    for name, digest in {**plan["source_hashes"], **plan["input_hashes"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("source/input closure failed before terminal comparison")
    control = torch.tensor(record["current_control"], dtype=torch.float64)
    references = record["terminal"]["exact_current_gradient_by_flux_sector"]
    declared = {"executor_sha256": sha(Path(__file__)), "raw_sha256": sha(RAW),
        "scope": "same terminal control native replay=True/retained replay=False gradient closure; no candidate selection",
        "cumulative_gradient_before": record["cumulative_gradient_started"], "additional_gradient_cap": 4,
        "original_phase_gradient_cap": plan["budget"]["gradient_cap"], "direct_J_calls": 0,
        "Hessian_calls": 0, "score_calls": 0, "seconds_cap": 30}
    if record["counts"]["gradient_started"] + 4 > plan["budget"]["gradient_cap"]:
        raise ValueError("native comparison exceeds declared phase derivative budget")
    (HERE / "TERMINAL_REPLAY_PLAN.json").write_text(json.dumps(declared, indent=2) + "\n")
    case = load_fixed_case()
    start = time.monotonic()
    result: dict[str, Any] = {"status": "in_progress", "gradient_started": 0, "gradient_completed": 0, "rows": []}
    for sides, reference in zip(((-1, -1), (-1, 1), (1, -1), (1, 1)), references):
        if time.monotonic() - start >= 30:
            raise TimeoutError("native terminal comparison time cap")
        result["gradient_started"] += 1
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        with transport.selected_face_extensions(("y", 4, 3, sides[0]), ("y", 2, 1, sides[1])):
            gradient = torch.func.grad(case["problem"].objective, argnums=0)(control, case["parameters"])
        result["gradient_completed"] += 1
        gap = float((gradient - torch.tensor(reference, dtype=control.dtype)).abs().max())
        budget = 128 * 360 * torch.finfo(control.dtype).eps * float(gradient.norm())
        result["rows"].append({"sides": list(sides), "gradient": gradient.tolist(),
            "retained_native_gap_inf": gap, "component_operation_budget": budget, "passed": gap <= budget})
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    result.update(status="passed" if all(row["passed"] for row in result["rows"]) else "failed",
        elapsed_seconds=time.monotonic() - start, raw_unchanged=sha(RAW) == declared["raw_sha256"],
        source_unchanged=all(sha(ROOT / name) == digest for name, digest in plan["source_hashes"].items()),
        cumulative_gradient_after=declared["cumulative_gradient_before"] + result["gradient_started"],
        minimum_claim=False)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "gradient_completed": result["gradient_completed"], "maximum_gap": max(row["retained_native_gap_inf"] for row in result["rows"])}))


if __name__ == "__main__":
    run()
