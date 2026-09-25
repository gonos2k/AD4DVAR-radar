"""Guard one wholly empty first-time centered-prior point FV response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_centered_prior_response_runner as base
from examples.weather_scenarios import fv_point_empty_centered_response_probe as probe
from examples.weather_scenarios.fv_point_empty_centered_case import (
    INACTIVE_PARAMETERS, make_empty_time_case,
)


def run(directory: Path) -> dict[str, object]:
    return base.run(
        directory, case_factory=make_empty_time_case,
        child_module="examples.weather_scenarios.fv_point_empty_centered_response_probe",
        plan=probe.PLAN, plan_sha256=probe.PLAN_SHA256,
        source_paths=probe.SOURCE_PATHS,
        inactive_indices=INACTIVE_PARAMETERS,
        required_nonzero_indices=(12,),
        scope="constructed centered-prior point FV: first observation time empty, 8 detected of 12, indices 0-3 status 1 missing with canonical inactive fill; complete model time/state and external theta background retained; no physical-skill claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["execution_status"] == "completed" else 1)
