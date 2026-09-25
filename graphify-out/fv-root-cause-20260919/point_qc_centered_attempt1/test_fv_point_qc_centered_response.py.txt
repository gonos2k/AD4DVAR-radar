"""External QC exclusion remains distinct from missing or clear sky."""

import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from examples.weather_scenarios import fv_point_qc_centered_response_probe as probe
from examples.weather_scenarios import fv_point_qc_centered_response_runner as runner
from examples.weather_scenarios.fv_point_missing_centered_case import (
    EXCLUDED_PARAMETER_INDEX, make_missing_case, make_qc_case,
)


def test_external_qc_identity_and_principal_whitener():
    qc, missing = make_qc_case(), make_missing_case()
    assert EXCLUDED_PARAMETER_INDEX == 5
    assert qc.problem.observation_status is not None
    assert qc.problem.observation_status.tolist() == [
        [0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0],
    ]
    assert int(qc.problem._detected_mask.sum()) == 11
    assert qc.parameters[5] == qc.problem.frozen.nowcast_config.min_dbz
    assert qc.direction.nonzero().flatten().tolist() == [4, 6, 7]
    assert qc.problem.identity != missing.problem.identity
    torch.testing.assert_close(qc.parameters, missing.parameters, rtol=0, atol=0)
    assert qc.problem.observation_correlation is not None
    valid = torch.tensor([0, 2, 3])
    principal = qc.problem.observation_correlation[valid][:, valid]
    eigenvalues, eigenvectors = torch.linalg.eigh(principal)
    expected = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    assert qc.problem._masked_whiteners[1] is not None
    torch.testing.assert_close(qc.problem._masked_whiteners[1], expected,
                               rtol=1e-12, atol=1e-12)
    gradient = torch.func.grad(qc.objective, argnums=0)(qc.control, qc.parameters)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    branch, _ = qc.branch_check(qc.control, qc.parameters)
    assert branch["euler_stages"] == 54
    assert branch["minimum_scaled_slope_margin"] > 1e-4
    assert branch["minimum_scaled_face_flux_margin"] > 1e-4


def test_qc_payload_has_no_fixed_mask_objective_score_or_mixed_effect():
    case = make_qc_case()
    c, p = case.control, case.parameters
    shifted = p.clone()
    shifted[5] += 0.25
    torch.testing.assert_close(case.objective(c, shifted), case.objective(c, p), rtol=0, atol=0)
    torch.testing.assert_close(case.score(c, shifted), case.score(c, p), rtol=0, atol=0)
    gradient = torch.func.grad(case.objective, argnums=0)
    torch.testing.assert_close(gradient(c, shifted), gradient(c, p), rtol=0, atol=0)
    unit = torch.zeros_like(p)
    unit[5] = 1.0
    mixed = torch.func.jvp(lambda q: gradient(c, q), (p,), (unit,))[1]
    torch.testing.assert_close(mixed, torch.zeros_like(mixed), rtol=0, atol=0)
    assert torch.func.grad(case.score, argnums=1)(c, p)[5] == 0


def test_qc_wrappers_bind_distinct_scope_and_plan(monkeypatch, tmp_path):
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
        assert config["case_factory"] is make_qc_case
        assert config["inactive_indices"] == (5,)
        assert config["plan"] == probe.PLAN
        assert config["plan_sha256"] == probe.PLAN_SHA256
        assert config["source_paths"] == probe.SOURCE_PATHS
        assert "status 2 externally QC rejected" in config["scope"]
        assert "11 detected of 12" in config["scope"]
        assert "status 1" not in config["scope"]
        assert "full-valid" not in config["scope"]


def test_qc_direct_cli_imports_from_unrelated_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("fv_point_qc_centered_response_probe.py",
                 "fv_point_qc_centered_response_runner.py"):
        script = root / "examples/weather_scenarios" / name
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=tmp_path,
            capture_output=True, text=True, timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output" in completed.stdout or "--directory" in completed.stdout
