"""Portable test inputs for fixed-response source and lifecycle contracts.

The helpers below make current-runtime test fixtures. Archived controls and
directions are loaded as numerical candidate data; regenerated parameters,
verification fields, and synthetic worker records are never presented as an
archived execution certificate.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import sys
import time
from typing import Any, cast

import torch
from torch import Tensor
from advar.fv_research_problem import FVResearchProblem
from advar.transport import BoundarySchedule

from examples.weather_scenarios import fv_concurrent_response_probe as probe


def portable_case(
    case_id: str,
) -> tuple[FVResearchProblem, Tensor, Tensor, Tensor, int]:
    """Build a fresh current-runtime case with an archived control seed."""
    if case_id == "fv4x5":
        saved = probe._archive("minmod_middle_time_bias_final.json")
        observations, frozen, future_echo, future_support = (
            probe.small_fixture.make_spatial_case()
        )
        future_echo = cast(BoundarySchedule, future_echo)
        future_support = cast(BoundarySchedule, future_support)
        control = torch.tensor(saved["nominal_control"], dtype=torch.float64)
        parameters = torch.cat((observations.dbz.flatten(), observations.dbz.new_tensor([0.02])))
        pattern = torch.linspace(-0.2, 0.3, 20, dtype=control.dtype).reshape(4, 5)
        contract = replace(
            frozen,
            initial_background_dbz=observations.dbz[0] + parameters[-1] * pattern,
        )
        forecast = probe.v.forecast_fv_analysis(
            control,
            contract,
            leads=1,
            boundary_start_interval=2,
            boundary_echo=future_echo,
            boundary_support=future_support,
        ).frames_linear[-1]
        verification = (
            probe.echo_to_dbz(forecast, min_dbz=frozen.nowcast_config.min_dbz)
            + 0.1 * pattern
        ).detach()
        problem = probe.FVResearchProblem(
            observations,
            frozen,
            future_echo,
            future_support,
            pattern,
            verification,
            probe.small_fixture.inspect_branches,
        )
        direction = torch.tensor(
            saved["tangents"]["middle_time_bias"]["parameter_direction"],
            dtype=torch.float64,
        )
        return problem, control, parameters, direction, 54

    if case_id == "fv8x10":
        saved = probe._archive("fv86_seed_a.json")
        archived_reanalysis = probe._archive("fv86_reanalysis.json")
        case = probe.large_fixture.make_case()
        control = torch.tensor(saved["workflow"]["control"], dtype=torch.float64)
        problem = probe.large_fixture.make_problem(case)
        direction = torch.tensor(
            archived_reanalysis["direction"]["values"], dtype=torch.float64,
        )
        return problem, control, case.parameters, direction, 108

    raise ValueError(f"unknown portable response case: {case_id}")


def synthetic_job_identity() -> tuple[dict[str, Any], float]:
    """Return test-only identity metadata, not a measured or archived input."""
    identity = {
        "problem": "a" * 64,
        "control": {"shape": [26], "dtype": "torch.float64", "sha256": "b" * 64},
        "parameters": {"shape": [61], "dtype": "torch.float64", "sha256": "c" * 64},
        "verification": {"shape": [4, 5], "dtype": "torch.float64", "sha256": "d" * 64},
        "direction": {"shape": [61], "dtype": "torch.float64", "sha256": "e" * 64},
    }
    return identity, 0.08


def synthetic_worker_report(
    identity: dict[str, Any], source_hashes: dict[str, str], archive_hashes: dict[str, str],
    expected_total: float, *, pid: int = 123, attempt_id: str | None = None,
) -> dict[str, Any]:
    """Build a lifecycle protocol fixture; it contains no archived worker output."""
    started = time.monotonic() - 10.0
    ended = started + 3.0
    intervals = [[started + 0.5, started + 0.75], [started + 1.5, started + 1.75]]
    events = [started + 0.625, started + 1.625]
    direct, indirect = expected_total * 0.6, expected_total * 0.4
    branch_digest = hashlib.sha256(b"synthetic test branch choices").hexdigest()
    face_digest = hashlib.sha256(b"synthetic test face signs").hexdigest()
    response = {
        "branch_euler_stages": 54,
        "stage_observer_events": 54,
        "branch_choices_sha256": branch_digest,
        "archived_branch_choices_sha256": branch_digest,
        "branch_face_signs_sha256": face_digest,
        "archived_branch_face_signs_sha256": face_digest,
        "gradient_max": 0.0,
        "true_adjoint_residual": 0.0,
        "true_adjoint_relative_residual": 0.0,
        "pcg_relative_residual": 1e-12,
        "pcg_iterations": 2,
        "hvp_count": 2,
        "direct": direct,
        "indirect": indirect,
        "total": expected_total,
        "archived_total": expected_total,
        "relative_difference": 0.0,
        "pcg_monitor": {
            "hvp_calls": 2,
            "hvp_event_times": events,
            "hvp_intervals": intervals,
            "converged": True,
            "iterations": 2,
            "relative_residual": 1e-12,
        },
    }
    child = {
        "case_id": "fv4x5",
        "status": "completed",
        "phase": "finished",
        "pid": pid,
        "executable": sys.executable,
        "source_before": dict(source_hashes),
        "source_after": dict(source_hashes),
        "source_unchanged": True,
        "inputs_unchanged": True,
        # This names the pinned archive set required by the schema; response
        # values above are synthetic and are not copied from that archive.
        "archived_report_sha256": dict(archive_hashes),
        "input_identity": dict(identity),
        "input_identity_after": dict(identity),
        "response_started_monotonic": started,
        "response_ended_monotonic": ended,
        "response": response,
    }
    if attempt_id is not None:
        child["attempt_id"] = attempt_id
    return child
