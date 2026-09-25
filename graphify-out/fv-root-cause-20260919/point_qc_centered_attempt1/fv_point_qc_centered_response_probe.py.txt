"""Bounded signed response for one externally QC-excluded point."""
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
    EXCLUDED_PARAMETER_INDEX, make_qc_case,
)


PLAN = base.EVIDENCE / "FV_POINT_QC_CENTERED_RESPONSE_PLAN.md"
PLAN_SHA256 = "4d247b648abd0834cdeeb6df0fc9fd2ec30554dc5407c5fe3faa3a52d706a39d"
SOURCE_PATHS = tuple(dict.fromkeys((*base.SOURCE_PATHS,
    "examples/weather_scenarios/fv_point_missing_centered_case.py",
    "examples/weather_scenarios/fv_point_qc_centered_response_probe.py",
    "examples/weather_scenarios/fv_point_qc_centered_response_runner.py",
)))


def run(output: Path) -> dict[str, Any]:
    return base.run(
        output, case_factory=make_qc_case,
        plan=PLAN, plan_sha256=PLAN_SHA256,
        source_paths=SOURCE_PATHS,
        inactive_indices=(EXCLUDED_PARAMETER_INDEX,),
        scope="constructed centered-prior point FV: 11 detected of 12; middle-time point 1, parameter 5, status 2 externally QC rejected with canonical inactive fill; no QC algorithm or physical-skill claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.output)
    raise SystemExit(0 if result["numerical_status"] == "eligible" else 2)
