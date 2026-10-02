"""Enclose the same rejected-trial objective difference; no PyTorch replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mpmath import iv, mp
from examples.weather_scenarios import fv_slice_precision_reference as reference

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
INPUT_SHA = "50267f00fed59074b8d848552fbdde70b3eeda0e25a1f5813fe2dd86d2825442"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output: Path) -> None:
    if output.exists():
        raise ValueError("interval output must be fresh")
    source = EVIDENCE / "partial_slice_precision_attempt2/precision.json"
    plan = EVIDENCE / "R4_INTERVAL_DIAGNOSTIC_PLAN_20261002.md"
    source_digest, plan_digest = sha(source), sha(plan)
    if source_digest != INPUT_SHA:
        raise ValueError("captured precision result identity differs")
    data = json.loads(source.read_text())
    sources = {name: sha(ROOT / name) for name in data["source_before"]}
    if sources != data["source_before"]:
        raise ValueError("original measured sources changed")
    probes = {str(Path(path).relative_to(ROOT)): sha(Path(path)) for path in (__file__, reference.__file__)}
    results = [reference.evaluate(data["fixture"], control, dps=80, interval=True)
               for control in (data["control_base"], data["control_reconstructed_trial"])]
    for row, original in zip(results, data["torch"], strict=True):
        if row["branch_choices"] != original["branch_choices"]:
            raise ValueError("interval branches differ from the measured model")
    previous = iv.dps
    try:
        iv.dps = 80
        bounds = [iv.make_mpf(tuple(tuple(part) for part in row["objective_interval_binary"])) for row in results]
        difference = bounds[1] - bounds[0]
        binary_difference = [list(bound) for bound in difference._mpi_]
        with mp.workdps(100):
            lower, upper = [mp.make_mpf(tuple(bound)) for bound in binary_difference]
            delta50, delta80 = [mp.mpf(data["precision"][dps]["delta"]) for dps in ("50", "80")]
            original_j = data["torch"][0]["objective"]
            trial_j = data["torch"][1]["objective"]
            allowance = mp.mpf(128 * 2**-52 * max(abs(original_j), abs(trial_j)))
            metrics = {
                "lower": str(lower), "upper": str(upper), "width": str(upper - lower),
                "strictly_negative": bool(upper < 0),
                "reference_points_enclosed": bool(lower <= delta50 <= upper and lower <= delta80 <= upper),
                "original_allowance": str(allowance),
                "below_original_allowance": bool(upper < allowance),
            }
    finally:
        iv.dps = previous
    if (sha(source) != source_digest or sha(plan) != plan_digest
            or any(sha(ROOT / name) != digest for name, digest in {**sources, **probes}.items())):
        raise ValueError("diagnostic sources or inputs changed during enclosure")
    output.write_text(json.dumps({
        "scope": "mpmath interval enclosure for captured binary FP64 controls and constants; original trial stays rejected",
        "input_precision_sha256": source_digest, "plan_sha256": plan_digest,
        "original_source_sha256": sources, "diagnostic_source_sha256": probes,
        "metrics": metrics, "difference_interval_binary": binary_difference,
        "endpoints": results, "branch_count_per_endpoint": 36,
        "full_root_claim": False, "original_trial_accepted": False,
        "response_validation": "not_performed",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
