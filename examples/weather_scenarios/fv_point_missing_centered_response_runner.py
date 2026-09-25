"""Guard one missing-point centered-prior signed FV reanalysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_centered_prior_response_runner as base
from examples.weather_scenarios import fv_point_missing_centered_response_probe as probe
from examples.weather_scenarios.fv_point_missing_centered_case import (
    MISSING_PARAMETER_INDEX, make_missing_case,
)


def run(directory: Path) -> dict[str, object]:
    return base.run(
        directory, case_factory=make_missing_case,
        child_module="examples.weather_scenarios.fv_point_missing_centered_response_probe",
        plan=probe.PLAN, plan_sha256=probe.PLAN_SHA256,
        source_paths=probe.SOURCE_PATHS,
        inactive_indices=(MISSING_PARAMETER_INDEX,),
        scope="constructed centered-prior point FV: 11 detected of 12; middle-time point 1, parameter 5, status 1 missing with canonical inactive fill; no original-input convergence or physical-skill claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
