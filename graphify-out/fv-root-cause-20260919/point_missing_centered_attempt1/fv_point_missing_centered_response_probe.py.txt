"""Bounded signed response for one declared missing point-observation row."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_centered_prior_response_probe as base
from examples.weather_scenarios.fv_point_missing_centered_case import (
    MISSING_PARAMETER_INDEX, make_missing_case,
)


PLAN = base.EVIDENCE / "FV_POINT_MISSING_CENTERED_RESPONSE_PLAN.md"
PLAN_SHA256 = "fab3e7c1175c5918992477cdae93fdbbc6c7960609603b5bfdc8e15cd88273f7"
SOURCE_PATHS = tuple(dict.fromkeys((*base.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_missing_centered_case.py",
    "examples/weather_scenarios/fv_point_missing_centered_response_probe.py",
    "examples/weather_scenarios/fv_point_missing_centered_response_runner.py",
)))


def run(output: Path) -> dict[str, Any]:
    return base.run(
        output, case_factory=make_missing_case,
        plan=PLAN, plan_sha256=PLAN_SHA256,
        source_paths=SOURCE_PATHS,
        inactive_indices=(MISSING_PARAMETER_INDEX,),
        scope="constructed centered-prior point FV: 11 detected of 12; middle-time point 1, parameter 5, status 1 genuinely missing with canonical inactive fill; no original-input or physical-skill claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"] == "eligible" else 2)
