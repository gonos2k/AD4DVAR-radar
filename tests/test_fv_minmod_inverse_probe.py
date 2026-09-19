"""Small branch checks for the local inverse experiment, not its full rerun."""
import importlib.util
from pathlib import Path

import pytest
import torch
from advar import variational as v

_spec = importlib.util.spec_from_file_location(
    "minmod_inverse_probe", Path(__file__).parents[1]
    / "examples/weather_scenarios/fv_minmod_inverse_probe.py",
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_joint_inverse_traces_both_rk_stages_across_replay_boundary():
    _, frozen, boundary, support = probe.make_case()
    control = v.initial_control(frozen)
    control[-3:] = control.new_tensor([.1, .1, .01])
    result = probe.inspect_branches(lambda: v.forecast_fv_analysis(
        control, frozen, leads=1, boundary_start_interval=2,
        boundary_echo=boundary, boundary_support=support,
    ))
    assert result["euler_stages"] == 3 * 9 * 2
    assert result["minimum_scaled_slope_margin"] > 1e-4


def test_joint_inverse_rejects_nominal_zero_controlled_flux():
    _, frozen, _, _ = probe.make_case()
    control = v.initial_control(frozen)
    control[-1] = .01
    with pytest.raises(ValueError, match="strict smooth branch"):
        probe.inspect_branches(lambda: v.analysis_trajectory(control, frozen))


def test_joint_inverse_rejects_controlled_zero_growth():
    _, frozen, _, _ = probe.make_case()
    control = v.initial_control(frozen)
    control[-3:-1] = .1
    with pytest.raises(ValueError, match="strict positive growth"):
        probe.inspect_branches(lambda: v.analysis_trajectory(control, frozen))


def test_joint_inverse_rejects_active_limiter_tie():
    q = torch.arange(1., 10., dtype=torch.float64).reshape(3, 3)
    edges = (q[:, 0], q[:, -1], q[0], q[-1])
    with pytest.raises(ValueError, match="strict smooth branch"):
        probe.inspect_branches(lambda: probe.t._euler_minmod(
            q, q.new_ones(3, 4), q.new_ones(4, 3), edges,
            q.new_tensor(.1), q.new_tensor(1.),
        ))
