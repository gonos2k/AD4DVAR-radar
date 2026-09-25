"""Read-only PR200 result-gate replay after shared point-runner parameterization."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios.fv_point_centered_prior_case import make_case
from examples.weather_scenarios.fv_point_centered_prior_response_runner import _valid_result


saved = json.loads((ROOT / "graphify-out/fv-root-cause-20260919/point_centered_response_attempt1/point_centered_response.json").read_text())
if (saved.get("numerical_status") != "eligible"
        or saved.get("response_validation") != "passed"
        or not _valid_result(saved, make_case())):
    raise SystemExit("archived PR200 full-valid result failed the current gate")
print("archived PR200 full-valid result passes current _valid_result numerical predicate")
