"""An empty first observation time leaves model time and theta intact."""

import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from examples.weather_scenarios import fv_point_empty_centered_response_probe as probe
from examples.weather_scenarios import fv_point_empty_centered_response_runner as runner
from examples.weather_scenarios.fv_point_empty_centered_case import (
    EMPTY_TIME, INACTIVE_PARAMETERS, make_empty_time_case,
)


def test_empty_first_time_keeps_complete_state_schedule_and_external_background():
    case = make_empty_time_case()
    assert EMPTY_TIME == 0 and INACTIVE_PARAMETERS == (0, 1, 2, 3)
    assert case.problem.empty_observation_time == 0
    assert case.problem.observation_status is not None
    assert case.problem.observation_status.tolist() == [
        [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],
    ]
    assert int(case.problem._detected_mask.sum()) == 8
    assert case.problem._masked_whiteners[0] is None
    assert all(item is not None and item.shape == (4, 4)
               for item in case.problem._masked_whiteners[1:])
    assert case.problem.layout["euler_stages"] == 54
    assert bool(case.problem.frozen.initial_support_mask.all())
    assert torch.equal(case.direction.nonzero().flatten(), torch.tensor([4, 5, 6, 7]))
    for index in INACTIVE_PARAMETERS:
        assert case.parameters[index] == case.problem.frozen.nowcast_config.min_dbz
    gradient = torch.func.grad(case.objective, argnums=0)(case.control, case.parameters)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    branch, _ = case.branch_check(case.control, case.parameters)
    assert branch["euler_stages"] == 54
    assert branch["minimum_scaled_slope_margin"] > 1e-4
    assert branch["minimum_scaled_face_flux_margin"] > 1e-4
    direct = torch.func.grad(case.score, argnums=1)(case.control, case.parameters)
    assert abs(float(direct[-1])) > 1e-12


def test_four_inactive_first_time_payloads_have_zero_local_effect():
    case = make_empty_time_case()
    c, p = case.control, case.parameters
    gradient = torch.func.grad(case.objective, argnums=0)
    score_gradient = torch.func.grad(case.score, argnums=1)(c, p)
    for index in INACTIVE_PARAMETERS:
        changed = p.clone()
        changed[index] += 0.25
        torch.testing.assert_close(case.objective(c, changed), case.objective(c, p), rtol=0, atol=0)
        torch.testing.assert_close(case.score(c, changed), case.score(c, p), rtol=0, atol=0)
        torch.testing.assert_close(gradient(c, changed), gradient(c, p), rtol=0, atol=0)
        unit = torch.zeros_like(p)
        unit[index] = 1.0
        mixed = torch.func.jvp(lambda q: gradient(c, q), (p,), (unit,))[1]
        torch.testing.assert_close(mixed, torch.zeros_like(mixed), rtol=0, atol=0)
        assert score_gradient[index] == 0


def test_empty_profile_wrappers_bind_plan_mask_theta_and_scope(monkeypatch, tmp_path):
    assert hashlib.sha256(probe.PLAN.read_bytes()).hexdigest() == probe.PLAN_SHA256
    calls = []
    def fake_probe(output, **kwargs):
        calls.append((output, kwargs))
        return {"numerical_status": "eligible"}
    def fake_runner(directory, **kwargs):
        calls.append((directory, kwargs))
        return {"execution_status": "completed"}
    monkeypatch.setattr(probe.base, "run", fake_probe)
    monkeypatch.setattr(runner.base, "run", fake_runner)
    probe.run(tmp_path / "child.json")
    runner.run(tmp_path / "parent")
    assert len(calls) == 2
    for _, config in calls:
        assert config["case_factory"] is make_empty_time_case
        assert config["inactive_indices"] == (0, 1, 2, 3)
        assert config["required_nonzero_indices"] == (12,)
        assert config["plan"] == probe.PLAN
        assert config["plan_sha256"] == probe.PLAN_SHA256
        assert config["source_paths"] == probe.SOURCE_PATHS
        assert "first observation time empty" in config["scope"]
        assert "8 detected of 12" in config["scope"]
        assert "external theta background retained" in config["scope"]


def test_empty_profile_direct_cli_imports_from_unrelated_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("fv_point_empty_centered_response_probe.py",
                 "fv_point_empty_centered_response_runner.py"):
        script = root / "examples/weather_scenarios" / name
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output" in completed.stdout or "--directory" in completed.stdout
