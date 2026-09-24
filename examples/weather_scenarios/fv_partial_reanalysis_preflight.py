"""Cheap source-bound preflight for a proposed partial-observation FV run.

This checks inputs and a warm-start branch. It does not optimize, solve an
adjoint, or certify a response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "graphify-out/fv-root-cause-20260919/FV_PARTIAL_REANALYSIS_PLAN.md"
WARM_START_REPORT = ROOT / "graphify-out/fv-root-cause-20260919/minmod_middle_time_bias_final.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_fv_research_partial_observation import MISSING, _problem
from advar import transport as t
from advar.fv_research_problem import FVResearchProblem


def _digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def branch_with_face_margin(
    problem: FVResearchProblem, control: Tensor, parameters: Tensor,
) -> tuple[dict[str, Any], str, float]:
    """Measure the smallest relative face flux during the caller's strict trace."""
    margins: list[float] = []

    def record(echo: Tensor, qx: Tensor, qy: Tensor) -> None:
        flux = torch.cat((qx.flatten(), qy.flatten())).abs()
        margins.append(float(flux.min() / flux.max()))

    # The supplied tracer wraps this observer and still owns minmod branch
    # eligibility. The outer observer adds a quantitative flux margin.
    with t.observe_minmod_stages(record):
        branch, scope = problem.branch_check(control, parameters)
    if len(margins) != branch["euler_stages"] or not all(map(math.isfinite, margins)):
        raise ValueError("face-flux margin trace is incomplete or nonfinite")
    return branch, scope, min(margins)


def run() -> dict[str, Any]:
    problem, warm_start, parameters = _problem()
    observations = problem.observations
    whitener = problem.frozen.observation_whitener
    if (int(observations.valid_mask.sum()) != 58
            or int(observations.missing_mask.sum()) != 2
            or not bool(observations.detected_mask[0].all())
            or bool(observations.censored_mask.any())
            or bool(observations.qc_rejected_mask.any())
            or not bool(problem.frozen.initial_support_mask.all())
            or whitener.mode is None
            or any(bool(whitener.mode[index] != 0) for index in MISSING)):
        raise ValueError("partial reanalysis preflight input contract changed")
    if not torch.equal(
        problem.verification,
        (problem.forecast(warm_start, parameters) + 0.1 * problem.pattern).detach(),
    ):
        raise ValueError("verification is not the declared fixed synthetic field")

    branch, _, face_margin = branch_with_face_margin(problem, warm_start, parameters)
    margin = branch["minimum_scaled_slope_margin"]
    if branch["euler_stages"] != 54 or margin <= 1.0e-4 or face_margin <= 1.0e-4:
        raise ValueError("warm-start strict branch/margin preflight failed")
    direction = torch.zeros_like(parameters)
    direction[20:40] = observations.valid_mask[1].to(parameters.dtype).flatten()
    if int(torch.count_nonzero(direction)) != 19:
        raise ValueError("valid-only middle-time direction changed")
    gradient = torch.func.grad(problem.objective, argnums=0)(warm_start, parameters)
    objective = problem.objective(warm_start, parameters)
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(objective)):
        raise ValueError("warm-start objective or gradient is nonfinite")

    sources = [
        Path(__file__),
        ROOT / "tests/test_fv_research_partial_observation.py",
        ROOT / "examples/weather_scenarios/fv_minmod_inverse_probe.py",
        ROOT / "examples/weather_scenarios/fv86_resource_runner.py",
        ROOT / "examples/weather_scenarios/fv_partial_reanalysis_runner.py",
        ROOT / "examples/weather_scenarios/fv_partial_reanalysis_probe.py",
        *(ROOT / "src/advar" / name for name in (
            "fv_research_problem.py", "local_refinement.py", "local_response.py",
            "matrix_free.py", "variational.py", "transport.py", "physics.py",
            "nowcast.py",
        )),
    ]
    return {
        "status": "preflight_only",
        "numerical_solver_runs": 0,
        "command": ".venv/bin/python examples/weather_scenarios/fv_partial_reanalysis_preflight.py --output graphify-out/fv-root-cause-20260919/fv_partial_reanalysis_preflight.json",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "CPU FP64"},
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "warm_start_report_sha256": hashlib.sha256(WARM_START_REPORT.read_bytes()).hexdigest(),
        "problem_identity": problem.identity,
        "observation_counts": {"valid": 58, "missing": 2, "censored": 0, "qc_rejected": 0},
        "missing_indices": [list(index) for index in MISSING],
        "common_bias_std_dbz": problem.frozen.analysis_config.observation_common_bias_std_dbz,
        "verification_constructor": "tests.test_fv_research_partial_observation._problem: warm-start partial forecast dBZ + 0.1 * fixed pattern, then detach",
        "tensor_sha256": {
            "observation_dbz": _digest_tensor(observations.dbz),
            "valid_mask": _digest_tensor(observations.valid_mask),
            "missing_mask": _digest_tensor(observations.missing_mask),
            "whitener_mode": _digest_tensor(whitener.mode),
            "parameters": _digest_tensor(parameters),
            "warm_start_control": _digest_tensor(warm_start),
            "verification": _digest_tensor(problem.verification),
            "direction": _digest_tensor(direction),
        },
        "warm_start_gradient_max": float(gradient.abs().max()),
        "warm_start_objective": float(objective),
        "warm_start_branch": {
            "euler_stages": branch["euler_stages"],
            "minimum_scaled_slope_margin": margin,
            "minimum_scaled_face_flux_margin": face_margin,
            "signature_sha256": hashlib.sha256(json.dumps(
                {"choices": branch["choices"], "face_signs": branch["face_signs"]},
                sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"preflight only: {args.output}")
