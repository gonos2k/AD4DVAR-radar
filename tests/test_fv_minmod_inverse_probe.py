"""Small branch checks for the local inverse experiment, not its full rerun."""
import importlib.util
from pathlib import Path

import pytest
import torch
from advar import variational as v
from advar.transport import BoundaryEdges, BoundarySchedule
from examples.weather_scenarios import fv_multilead_research_case as multilead

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


@pytest.mark.parametrize("growth_control", [-0.023187493904503132, 0.0, 0.023187493904503132])
def test_joint_inverse_traces_decay_maintain_and_growth_on_smooth_branches(growth_control):
    problem, _, control, parameters = multilead.make_case()
    control = control.clone()
    control[-1] = growth_control
    direction = torch.zeros_like(control)
    direction[-1] = 1
    nominal, _ = problem.branch_check(control, parameters)
    step = 1e-4
    for endpoint in (control - step * direction, control + step * direction):
        branch, _ = problem.branch_check(endpoint, parameters)
        assert branch["choices"] == nominal["choices"]
        assert branch["face_signs"] == nominal["face_signs"]
    derivative = torch.func.jvp(
        lambda value: problem.score(value, parameters),
        (control,), (direction,),
    )[1]
    central = (
        problem.score(control + step * direction, parameters)
        - problem.score(control - step * direction, parameters)
    ) / (2 * step)
    torch.testing.assert_close(derivative, central, rtol=2e-6, atol=1e-9)


def test_zero_growth_objective_hvp_has_two_sided_second_order_convergence():
    problem, _, control, parameters = multilead.make_case()
    control = control.clone()
    control[-1] = 0
    direction = torch.zeros_like(control)
    direction[-1] = 1
    gradient = torch.func.grad(problem.objective, argnums=0)
    hvp = torch.func.jvp(lambda value: gradient(value, parameters),
                         (control,), (direction,))[1]
    errors = []
    for step in (1e-4, 5e-5):
        central = (
            gradient(control + step * direction, parameters)
            - gradient(control - step * direction, parameters)
        ) / (2 * step)
        errors.append(float((central - hvp).norm() / hvp.norm()))
    assert errors[1] < 0.3 * errors[0]
    assert errors[1] < 2e-7


def test_flat_echo_is_forward_available_but_strict_local_response_is_refused():
    observations, frozen, _, _ = probe.make_spatial_case()
    spec = frozen.fv_transport
    assert spec is not None
    initial = observations.dbz.new_full((4, 5), 30.0)
    edges: BoundaryEdges = (initial[:, 0], initial[:, -1], initial[0], initial[-1])
    known: BoundaryEdges = (
        torch.ones_like(edges[0]), torch.ones_like(edges[1]),
        torch.ones_like(edges[2]), torch.ones_like(edges[3]),
    )
    boundary: BoundarySchedule = ((edges, edges),) * 9
    support: BoundarySchedule = ((known, known),) * 9
    coefficients = spec.coefficient_limits * initial.new_tensor([.7, -.6, .5, .4, -.3])

    def trajectory():
        return probe.t.finite_volume_trajectory(
            initial, torch.ones_like(initial), coefficients, initial.new_zeros(()),
            psi_basis=spec.psi_basis, leads=1, substeps_per_interval=9,
            interval_seconds=60.0, spacing_yx=spec.spacing_yx,
            boundary_echo=boundary, boundary_support=support, reconstruction="minmod",
        )

    frames, _ = trajectory()
    assert bool(torch.isfinite(frames).all())
    assert bool((frames >= 0).all())
    torch.testing.assert_close(frames[-1], initial, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="strict smooth branch"):
        probe.inspect_branches(trajectory)


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
