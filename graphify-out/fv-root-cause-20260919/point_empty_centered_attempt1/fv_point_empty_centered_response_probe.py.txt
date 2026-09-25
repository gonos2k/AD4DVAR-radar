"""Bounded signed response with the first point-observation time empty."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_centered_prior_response_probe as base
from examples.weather_scenarios.fv_point_empty_centered_case import (
    INACTIVE_PARAMETERS, make_empty_time_case,
)


PLAN = base.EVIDENCE / "FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_PLAN.md"
PLAN_SHA256 = "b05aadabec9c40d110586767f7b923f1a3baeed507908ee276c63fcfdedb0fc1"
SOURCE_PATHS = tuple(dict.fromkeys((*base.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_empty_centered_case.py",
    "examples/weather_scenarios/fv_point_empty_centered_response_probe.py",
    "examples/weather_scenarios/fv_point_empty_centered_response_runner.py",
)))


def run(output: Path) -> dict[str, Any]:
    return base.run(
        output, case_factory=make_empty_time_case,
        plan=PLAN, plan_sha256=PLAN_SHA256,
        source_paths=SOURCE_PATHS,
        inactive_indices=INACTIVE_PARAMETERS,
        required_nonzero_indices=(12,),
        scope="constructed centered-prior point FV: first observation time empty, 8 detected of 12, indices 0-3 status 1 genuinely missing with canonical inactive fill; complete model time/state and external theta background retained; no physical-skill claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"] == "eligible" else 2)
