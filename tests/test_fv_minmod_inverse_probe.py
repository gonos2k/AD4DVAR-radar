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


def test_branch_signature_keeps_signed_faces_and_per_cell_slope_choices():
    dtype = torch.float64
    height, width = 4, 5
    y = torch.arange(height + 1, dtype=dtype)[:, None].expand(height + 1, width + 1)
    x = torch.arange(width + 1, dtype=dtype)[None, :].expand(height + 1, width + 1)
    basis = torch.stack((y, x, x * y, 0.5 * (x.square() - y.square()), x.square() * y))
    coefficients = torch.tensor([-0.055, -0.08, 0.07, 0.04, 0.0], dtype=dtype)
    qx, qy = probe.t.face_volume_fluxes(torch.einsum("k,kij->ij", coefficients, basis))
    assert bool((qx < 0).any())
    assert bool((qx > 0).any())
    assert bool((qy < 0).any())
    assert bool((qy > 0).any())
    assert bool((qx == 0).any()) is False
    assert bool((qy == 0).any()) is False

    echo = torch.tensor(
        [
            [30.0, 31.0, 31.8, 33.0, 34.0],
            [31.0, 32.1, 33.0, 34.1, 35.0],
            [32.0, 32.9, 34.0, 34.9, 36.0],
            [33.0, 34.1, 35.3, 36.1, 37.3],
        ],
        dtype=dtype,
    )
    edges = tuple(torch.ones(n, dtype=dtype) for n in (height, height, width, width))
    result = probe.inspect_branches(
        lambda: probe.t._euler_minmod(
            echo, qx, qy, edges, echo.new_tensor(0.1), echo.new_tensor(100.0)
        )
    )
    choices = result["choices"][0]
    assert choices[0]["choose_left"] == [[False, True, False], [True, False, True]]
    assert choices[1]["choose_left"] == [[False, False, False], [True, True, True]]
    assert choices[0]["slope_sign"] == [[1, 1, 1], [1, 1, 1]]
    assert len(result["face_signs"][0]["qx"]) == height
    assert len(result["face_signs"][0]["qx"][0]) == width + 1
    other = echo.new_tensor([
        [30, 31, 31.8, 33, 34], [31, 31.9, 33, 33.9, 35],
        [32, 33.1, 34, 35.1, 36], [33, 34.2, 35.3, 36.2, 37.1],
    ])
    changed = probe.inspect_branches(lambda: probe.t._euler_minmod(
        other, qx, qy, edges, echo.new_tensor(.1), echo.new_tensor(100.),
    ))
    original_mask = torch.tensor(choices[0]["choose_left"])
    changed_mask = torch.tensor(changed["choices"][0][0]["choose_left"])
    assert not bool(original_mask.all()) and not bool(changed_mask.all())
    assert torch.equal(changed_mask, ~original_mask)
    assert result["choices"] != changed["choices"]
    assert result["face_signs"] == changed["face_signs"]


def test_strict_opposite_slopes_have_a_smooth_zero_limiter():
    echo = torch.tensor([[1., 2., 1.], [2., 4., 2.], [4., 7., 4.]], dtype=torch.float64)
    edges = tuple(echo.new_ones(3) for _ in range(4))
    trace = probe.inspect_branches(lambda: probe.t._euler_minmod(
        echo, echo.new_ones(3, 4), echo.new_ones(4, 3), edges,
        echo.new_tensor(.1), echo.new_tensor(1.),
    ))
    choice = trace["choices"][0][0]
    assert choice["slope_sign"] == [[0]]
    assert choice["left_sign"] == [[1]]
    assert choice["right_sign"] == [[-1]]
