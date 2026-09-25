"""No-solver preflight for one full-valid correlated point FV response."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch
from torch import Tensor

from advar import transport
from advar.fv_point_research_problem import FVPointResearchProblem
from examples.weather_scenarios.fv_point_research_case import make_case


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
PLAN = EVIDENCE / "FV_POINT_STATIONARY_RESPONSE_PLAN.md"
SOURCE_PATHS = (
    "examples/weather_scenarios/fv_point_research_case.py",
    "examples/weather_scenarios/fv_point_response_preflight.py",
    "examples/weather_scenarios/fv_point_response_preflight_runner.py",
    "examples/weather_scenarios/fv_minmod_inverse_probe.py",
    "examples/weather_scenarios/fv_sensitivity_probe.py",
    "examples/weather_scenarios/fv86_resource_runner.py",
    "src/advar/fv_point_research_problem.py",
    "src/advar/fv_research_problem.py",
    "src/advar/fv_point_sampler.py",
    "src/advar/local_refinement.py",
    "src/advar/local_response.py",
    "src/advar/matrix_free.py",
    "src/advar/variational.py",
    "src/advar/transport.py",
    "src/advar/physics.py",
    "src/advar/nowcast.py",
)
ARCHIVED_WARM = EVIDENCE / "minmod_middle_time_bias_final.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha(ROOT / name) for name in SOURCE_PATHS}


def _branch(problem: FVPointResearchProblem, control: Tensor, parameters: Tensor) -> tuple[dict[str, Any], str, float]:
    face_margins: list[float] = []

    def record(_echo: Tensor, qx: Tensor, qy: Tensor) -> None:
        flux = torch.cat((qx.reshape(-1), qy.reshape(-1))).abs()
        face_margins.append(float(flux.min() / flux.max()))

    with transport.observe_minmod_stages(record):
        signature, scope = problem.branch_check(control, parameters)
    if (len(face_margins) != signature["euler_stages"]
            or not all(math.isfinite(value) for value in face_margins)):
        raise ValueError("point response branch/face trace is incomplete")
    return signature, scope, min(face_margins)


def fixed_problem() -> tuple[FVPointResearchProblem, Tensor, Tensor, Tensor]:
    problem, warm, parameters = make_case()
    correlation = torch.eye(4, dtype=torch.float64)
    correlation[0, 2] = correlation[2, 0] = 0.3
    correlation[1, 3] = correlation[3, 1] = -0.2
    problem = replace(problem, observation_correlation=correlation)
    direction = torch.zeros_like(parameters)
    direction[4:8] = 1.0
    return problem, warm, parameters, direction


def _input_identity(
    problem: FVPointResearchProblem, warm: Tensor, parameters: Tensor, direction: Tensor,
) -> dict[str, Any]:
    correlation = problem.observation_correlation
    assert correlation is not None
    return {
        "problem_identity": problem.identity,
        "tensor_sha256": {
            "warm_control": _tensor_sha(warm),
            "parameters": _tensor_sha(parameters),
            "direction": _tensor_sha(direction),
            "verification": _tensor_sha(problem.verification_dbz),
            "observation_coordinates": _tensor_sha(problem.observation_coordinates),
            "observation_correlation": _tensor_sha(correlation),
        },
    }


def run() -> dict[str, Any]:
    source_before = _source_hashes()
    archive_before = _sha(ARCHIVED_WARM)
    plan_before = _sha(PLAN)
    problem, warm, parameters, direction = fixed_problem()
    input_before = _input_identity(problem, warm, parameters, direction)
    if (problem.layout["controls"] != 26 or problem.layout["parameters"] != 13
            or problem.layout["euler_stages"] != 54
            or problem.observation_status is not None
            or not bool(problem.frozen.initial_support_mask.all())
            or int(torch.count_nonzero(direction)) != 4
            or not bool((direction[4:8] == 1).all())
            or bool((direction[:4] != 0).any())
            or bool((direction[8:] != 0).any())):
        raise ValueError("point response fixed input layout changed")
    branch, scope, face_margin = _branch(problem, warm, parameters)
    if (branch["euler_stages"] != 54
            or branch["minimum_scaled_slope_margin"] <= 1e-4
            or face_margin <= 1e-4):
        raise ValueError("point response warm strict branch or margin failed")
    objective = problem.objective(warm, parameters)
    score = problem.score(warm, parameters)
    gradient = torch.func.grad(problem.objective, argnums=0)(warm, parameters)
    if (not bool(torch.isfinite(objective)) or not bool(torch.isfinite(score))
            or not bool(torch.isfinite(gradient).all())):
        raise ValueError("point response warm objective, score or gradient is nonfinite")
    source_after = _source_hashes()
    if (source_after != source_before or _sha(ARCHIVED_WARM) != archive_before
            or _sha(PLAN) != plan_before
            or _input_identity(problem, warm, parameters, direction) != input_before):
        raise ValueError("point response preflight source or fixed input changed")
    signature = {key: branch[key] for key in ("choices", "face_signs")}
    return {
        "status": "preflight_only", "numerical_solver_runs": 0,
        "pid": os.getpid(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "CPU FP64"},
        "plan_sha256": plan_before, "archived_warm_sha256": archive_before,
        "source_sha256": source_before, "source_unchanged": True,
        **input_before,
        "layout": problem.layout, "support": problem.support,
        "warm_objective": float(objective), "warm_score": float(score),
        "warm_gradient": gradient.detach().tolist(),
        "warm_gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "warm_gradient_max": float(gradient.abs().max()),
        "warm_branch": {
            "euler_stages": branch["euler_stages"],
            "minimum_scaled_slope_margin": branch["minimum_scaled_slope_margin"],
            "minimum_scaled_face_flux_margin": face_margin,
            **signature,
            "signature_sha256": hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        },
        "branch_scope": scope,
        "response_validation": "not_performed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(arguments.output)
