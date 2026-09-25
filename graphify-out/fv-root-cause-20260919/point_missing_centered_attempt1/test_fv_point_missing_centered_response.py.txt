"""One missing point leaves the full FV state and active likelihood intact."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from examples.weather_scenarios import fv_point_missing_centered_response_probe as probe
from examples.weather_scenarios import fv_point_missing_centered_response_runner as runner
from examples.weather_scenarios.fv_point_missing_centered_case import (
    MISSING_PARAMETER_INDEX, make_missing_case,
)


def test_missing_middle_point_is_inactive_but_time_and_state_remain():
    case = make_missing_case()
    assert MISSING_PARAMETER_INDEX == 5
    assert case.problem.observation_status is not None
    assert case.problem.observation_status.tolist() == [
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
    ]
    assert int(case.problem._detected_mask.sum()) == 11
    assert case.parameters[5] == case.problem.frozen.nowcast_config.min_dbz
    assert case.direction.nonzero().flatten().tolist() == [4, 6, 7]
    assert case.problem.layout["euler_stages"] == 54
    assert case.problem.observation_correlation is not None
    assert case.problem._masked_whiteners[1] is not None
    assert case.problem._masked_whiteners[1].shape == (3, 3)
    valid = torch.tensor([0, 2, 3])
    principal = case.problem.observation_correlation[valid][:, valid]
    eigenvalues, eigenvectors = torch.linalg.eigh(principal)
    expected_whitener = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    torch.testing.assert_close(case.problem._masked_whiteners[1], expected_whitener,
                               rtol=1e-12, atol=1e-12)
    gradient = torch.func.grad(case.objective, argnums=0)(case.control, case.parameters)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    branch, _ = case.branch_check(case.control, case.parameters)
    assert branch["minimum_scaled_slope_margin"] > 1e-4
    assert branch["minimum_scaled_face_flux_margin"] > 1e-4


def test_missing_parameter_has_zero_objective_score_and_mixed_derivatives():
    case = make_missing_case()
    p = case.parameters
    shifted = p.clone()
    shifted[5] += 0.25
    c = case.control
    torch.testing.assert_close(case.objective(c, shifted), case.objective(c, p), rtol=0, atol=0)
    torch.testing.assert_close(case.score(c, shifted), case.score(c, p), rtol=0, atol=0)
    gradient = torch.func.grad(case.objective, argnums=0)
    torch.testing.assert_close(gradient(c, shifted), gradient(c, p), rtol=0, atol=0)
    unit = torch.zeros_like(p)
    unit[5] = 1.0
    mixed = torch.func.jvp(lambda q: gradient(c, q), (p,), (unit,))[1]
    torch.testing.assert_close(mixed, torch.zeros_like(mixed), rtol=0, atol=0)
    direct = torch.func.grad(case.score, argnums=1)(c, p)
    assert direct[5] == 0


def test_profile_wrappers_bind_plan_source_and_inactive_slot(monkeypatch, tmp_path):
    assert hashlib.sha256(probe.PLAN.read_bytes()).hexdigest() == probe.PLAN_SHA256
    assert len(probe.SOURCE_PATHS) == len(set(probe.SOURCE_PATHS))
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
        assert config["inactive_indices"] == (5,)
        assert config["plan"] == probe.PLAN
        assert config["plan_sha256"] == probe.PLAN_SHA256
        assert config["source_paths"] == probe.SOURCE_PATHS
        assert config["case_factory"] is make_missing_case
        assert "missing" in config["scope"]
        assert "11 detected of 12" in config["scope"]
        assert "parameter 5" in config["scope"]
        assert "status 1" in config["scope"]
        assert "canonical inactive fill" in config["scope"]
        assert "full-valid" not in config["scope"]


def test_plan_distinguishes_missing_profile_from_full_profile():
    plan = probe.PLAN.read_text()
    assert "genuinely missing" in plan
    assert "original zero-centered-prior" in plan
    case = make_missing_case()
    assert json.dumps(case.identity, sort_keys=True) != ""


def test_direct_cli_wrappers_import_from_another_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in (
        "fv_point_centered_prior_response_probe.py",
        "fv_point_missing_centered_response_probe.py",
        "fv_point_missing_centered_response_runner.py",
    ):
        script = root / "examples/weather_scenarios" / name
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output" in completed.stdout or "--directory" in completed.stdout
