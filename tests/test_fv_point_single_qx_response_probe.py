"""Pure wiring checks for the bounded original-point qx response probe."""
import json
from pathlib import Path
import torch
from types import SimpleNamespace
from typing import Any, cast
import pytest

from examples.weather_scenarios.fv_point_single_qx_response_probe import (
    DIRECTION_NAME,
    NORMAL_RESULT_SHA256,
    PLAN_SHA256,
    RELEASE_SHA256,
    _fixture,
    _source_inventory,
    _validate_runtimes,
    _refine,
    qx_direction,
    parameter_endpoint,
)
from examples.weather_scenarios import fv_point_response_preflight as preflight


def test_qx_response_direction_is_exactly_the_four_middle_observation_parameters():
    parameters = torch.linspace(-0.2, 0.3, 13, dtype=torch.float64)
    direction = qx_direction(parameters)
    expected = torch.zeros(13, dtype=torch.float64)
    expected[4:8] = 1.0
    assert DIRECTION_NAME == "middle_four_qx_face_plus1"
    assert torch.equal(direction, expected)
    assert torch.equal(parameters, torch.linspace(-0.2, 0.3, 13, dtype=torch.float64))


def test_parameter_endpoints_are_actual_symmetric_steps_and_preserve_zero_slots():
    parameters = torch.linspace(-0.2, 0.3, 13, dtype=torch.float64)
    direction = qx_direction(parameters)
    original = parameters.clone()
    for step in (0.00025, 0.000125):
        minus = parameter_endpoint(parameters, direction, -1, step)
        plus = parameter_endpoint(parameters, direction, 1, step)
        torch.testing.assert_close((minus + plus) / 2, parameters, rtol=0, atol=1e-16)
        torch.testing.assert_close((plus - minus) / (2 * step), direction, rtol=1e-12, atol=1e-12)
        assert torch.equal(minus[:4], parameters[:4])
        assert torch.equal(minus[8:], parameters[8:])
        assert torch.equal(plus[:4], parameters[:4])
        assert torch.equal(plus[8:], parameters[8:])
    assert torch.equal(parameters, original)


def test_release_plan_and_root_normal_prerequisite_pins_are_fixed():
    assert RELEASE_SHA256 == "e3f4f083ac9bfb002d38e4e2a0352672f179775d18b22fbf26fd294bedd2ad24"
    assert PLAN_SHA256 == "1aae37e936ce9ecd0a8439fbd78a87470dca68169773dab5152844f1c8b24408"
    assert NORMAL_RESULT_SHA256 == "ae6e65aaab17de0ab5e8d578dcba6d61e220e4365e46fd096a0dcc507974fb38"


def test_runtime_gate_uses_release_and_normal_report_schemas_separately():
    from examples.weather_scenarios import fv_point_single_qx_response_probe as probe

    release = {"runtime": {"python": probe.platform.python_version(),
                            "torch": torch.__version__, "device": "CPU FP64"}}
    normal = {"runtime": {"python": probe.platform.python_version(), "mpmath": probe.mpmath.__version__}}
    assert _validate_runtimes(release, normal) == release["runtime"]
    normal["runtime"]["torch"] = torch.__version__
    with pytest.raises(ValueError, match="runtime differs"):
        _validate_runtimes(release, normal)


def test_actual_parameter_fixture_refreshes_only_observations_and_background():
    class Problem:
        observation_dbz = torch.zeros((3, 4), dtype=torch.float64)

        def contract(self, parameters):
            background = torch.ones((4, 5), dtype=torch.float64) * parameters[-1]
            return SimpleNamespace(initial_background_dbz=background)

    fixed = {"point_dbz": torch.zeros((3, 4)).tolist(),
             "background_dbz": torch.zeros((4, 5)).tolist(),
             "whitener": [[1.0, 0.0], [0.0, 1.0]], "psi_basis": [1],
             "boundary_echo": [7]}
    parameters = torch.arange(13, dtype=torch.float64)
    result = _fixture(cast(Any, Problem()), parameters, fixed)
    assert result["point_dbz"] == parameters[:12].reshape(3, 4).tolist()
    assert result["background_dbz"] == (torch.ones((4, 5), dtype=torch.float64) * parameters[-1]).tolist()
    assert result["whitener"] == fixed["whitener"]
    assert result["psi_basis"] == fixed["psi_basis"]
    assert result["boundary_echo"] == fixed["boundary_echo"]


def test_actual_point_problem_fixture_tracks_middle_observations_and_theta_only_background():
    from examples.weather_scenarios.fv_point_single_qx_response_probe import _fixture

    problem, _, parameters, direction = preflight.fixed_problem()
    capture_path = (Path(__file__).resolve().parents[1] / "graphify-out/fv-root-cause-20260919"
                    / "R2_POINT_SINGLE_QX_NORMAL_INPUT_20261003_RELEASE2.json")
    captured = json.loads(capture_path.read_text())["fixture"]
    step = 0.00025
    for sign in (-1, 1):
        actual = parameter_endpoint(parameters, direction, sign, step)
        fixture = _fixture(problem, actual, captured)
        assert fixture["point_dbz"] == actual[:-1].reshape_as(problem.observation_dbz).tolist()
        assert fixture["background_dbz"] == problem.contract(actual).initial_background_dbz.tolist()
        assert fixture["whitener"] == captured["whitener"]
        assert fixture["std_dbz"] == captured["std_dbz"]
    theta_changed = parameters.clone()
    theta_changed[-1] += step
    fixture = _fixture(problem, theta_changed, captured)
    assert fixture["point_dbz"] == captured["point_dbz"]
    assert fixture["background_dbz"] == problem.contract(theta_changed).initial_background_dbz.tolist()
    assert fixture["whitener"] == captured["whitener"]


def test_source_inventory_unions_all_prerequisite_maps():
    paths = _source_inventory({"release.py": "a"}, {"normal.py": "b"}, {"capture.json": "c"})
    assert {"release.py", "normal.py", "capture.json"}.issubset(paths)


def test_unclassified_refinement_error_propagates(monkeypatch):
    import examples.weather_scenarios.fv_point_single_qx_response_probe as probe

    def programming_error(*args, **kwargs):
        raise ValueError("programming defect")

    monkeypatch.setattr(probe, "refine_stationary", programming_error)
    binding = SimpleNamespace(objective=lambda c, p: c.sum())
    with pytest.raises(ValueError, match="programming defect"):
        _refine(None, cast(Any, binding), torch.zeros(25, dtype=torch.float64),
                torch.zeros(13, dtype=torch.float64), "signature")
