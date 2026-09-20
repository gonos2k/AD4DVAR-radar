"""Small forwarding checks for the explicit GN-to-response workflow."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fv_gn_response_probe", ROOT / "examples/weather_scenarios/fv_gn_response_probe.py"
)
assert SPEC and SPEC.loader
PRODUCER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRODUCER
SPEC.loader.exec_module(PRODUCER)


@dataclass(frozen=True)
class _Frozen:
    initial_background_dbz: torch.Tensor


def test_fresh_gn_forwards_exact_solver_control_after_verification(monkeypatch, tmp_path):
    events: list[str] = []
    forwarded: dict[str, object] = {}
    dtype = torch.float64
    observations = SimpleNamespace(dbz=torch.full((3, 4, 5), 20.0, dtype=dtype))
    frozen = _Frozen(torch.full((4, 5), 20.0, dtype=dtype))
    boundary = object()
    support = object()
    solver_control = torch.arange(26, dtype=dtype)
    parameters = torch.cat((observations.dbz.flatten(), observations.dbz.new_tensor([0.02])))

    class FakeForecast:
        frames_linear = [torch.zeros((4, 5), dtype=dtype)]

    class FakeV:
        def initial_control(self, _frozen):
            return torch.zeros(26, dtype=dtype)

        def forecast_fv_analysis(self, *_args, **_kwargs):
            events.append("forecast")
            return FakeForecast()

        def solve_analysis(self, _obs, _contract, *, control):
            assert events == ["forecast", "verification"]
            events.append("solve")
            assert control.shape == solver_control.shape
            return SimpleNamespace(
                control=solver_control,
                reason="fake",
                outer_iterations=1,
                pcg_iterations=2,
            )

    def echo_to_dbz(frame, **_kwargs):
        events.append("verification")
        return frame

    def digest(value):
        return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()

    def make_functions(*_args):
        return (
            lambda c, p: (c.square().sum() + p.square().sum()) * 0.5,
            lambda c, p: (c.square().sum() + p.square().sum()) * 0.5,
            lambda c, p: ("branch", "fake"),
        )

    fixture = SimpleNamespace(
        __file__=str(ROOT / "examples/weather_scenarios/fv_minmod_inverse_probe.py"),
        make_spatial_case=lambda: (observations, frozen, boundary, support),
        echo_to_dbz=echo_to_dbz,
    )
    bridge = SimpleNamespace(
        __file__=str(ROOT / "examples/weather_scenarios/fv_minmod_matrix_free_probe.py"),
        EVIDENCE=tmp_path,
        v=FakeV(),
        _digest=digest,
        _load=lambda name: fixture if name == "fv_minmod_inverse_probe" else oracle,
        check_archived_sources=lambda *_args, **_kwargs: None,
        make_research_functions=make_functions,
    )
    oracle = SimpleNamespace(
        __file__=str(ROOT / "examples/weather_scenarios/fv_sensitivity_probe.py")
    )

    saved = {
        "cache_payload_identity": {"core_source_sha256": {}},
        "input_identity": {"parameters_sha256": digest(parameters)},
        "nominal_control": [0.0] * 26,
        "nominal_branch": {"choices": [], "face_signs": []},
        "tangents": {"zero": {"parameter_direction": [0.0] * 61}},
    }
    (tmp_path / "minmod_middle_time_bias_final.json").write_text(json.dumps(saved))
    (tmp_path / "minmod_spatial_inverse.json").write_text(json.dumps({"source_sha256": {}}))

    def prepare_response(objective, score, control, p, directions, **kwargs):
        events.append("prepare")
        forwarded.update({"control": control, "p": p, "directions": directions})
        assert control is solver_control
        assert events == ["forecast", "verification", "solve", "prepare"]
        return {"status": "ineligible", "response": None}

    workflow = SimpleNamespace(
        __file__=str(ROOT / "examples/weather_scenarios/fv_analysis_response.py"),
        prepare_response=prepare_response,
    )
    monkeypatch.setattr(PRODUCER, "EVIDENCE", tmp_path)
    monkeypatch.setattr(
        PRODUCER, "load",
        lambda name: bridge if name == "fv_minmod_matrix_free_probe" else workflow,
    )

    output = tmp_path / "result.json"
    report = PRODUCER.run(output, fresh_gn=True)

    assert report["fresh_gn_runs"] == 1
    assert report["workflow"]["status"] == "ineligible"
    assert forwarded["control"] is solver_control
    assert events == ["forecast", "verification", "solve", "prepare"]
