from pathlib import Path
import json

import torch

from advar import transport


def test_material_support_preserves_constants_and_flow_derivative():
    dtype = torch.float64
    psi = torch.linspace(-0.2, 0.7, 30, dtype=dtype).reshape(5, 6)
    dt, area = torch.tensor(0.1, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    for value in (0.0, 0.4, 1.0):
        support = torch.full((4, 5), value, dtype=dtype)
        edges = (support[:, 0], support[:, -1], support[0], support[-1])

        def advance(potential):
            qx, qy = transport.face_volume_fluxes(potential)
            return transport._euler_donorcell(support, qx, qy, edges, dt, area,
                                             material_support=True)

        perturbation = torch.sin(torch.arange(30, dtype=dtype)).reshape(5, 6)
        result, derivative = torch.func.jvp(advance, (psi,), (perturbation,))
        assert torch.equal(result, support)
        assert torch.equal(derivative, torch.zeros_like(support))


def test_material_support_keeps_missing_inflow_and_support_derivatives():
    dtype = torch.float64
    support = torch.full((3, 4), 0.5, dtype=dtype)
    qx = torch.full((3, 5), 0.1, dtype=dtype)
    qy = torch.zeros((4, 4), dtype=dtype)
    edges = (torch.zeros(3, dtype=dtype), torch.ones(3, dtype=dtype),
             torch.ones(4, dtype=dtype), torch.ones(4, dtype=dtype))
    dt, area = torch.tensor(0.25, dtype=dtype), torch.tensor(1.0, dtype=dtype)

    def advance(state):
        return transport._euler_donorcell(state, qx, qy, edges, dt, area,
                                         material_support=True)

    result, derivative = torch.func.jvp(advance, (support,), (torch.ones_like(support),))
    expected = support.clone()
    expected[:, 0] = 0.4875
    torch.testing.assert_close(result, expected, atol=1e-16, rtol=0)
    assert bool((result[:, 0] < support[:, 0]).all())
    expected_derivative = torch.ones_like(support)
    expected_derivative[:, 0] = 0.975
    torch.testing.assert_close(derivative, expected_derivative, atol=1e-16, rtol=0)


def test_material_support_matches_conservative_update_and_jvp_for_divfree_flow():
    dtype = torch.float64
    rows = torch.arange(5, dtype=dtype)[:, None]
    columns = torch.arange(6, dtype=dtype)[None, :]
    potential = 0.013 * torch.sin(0.7 * rows + 0.2 * columns)
    # Make one face exactly zero so the selected extension's chosen derivative
    # branch is exercised while qx/qy still come from one streamfunction.
    potential = potential.clone()
    potential[2, 3] = potential[1, 3]
    support = (0.2 + 0.6 * torch.sigmoid(
        torch.arange(20, dtype=dtype).reshape(4, 5) * 0.17 - 1.3
    )).requires_grad_()
    dpotential = torch.cos(torch.arange(30, dtype=dtype).reshape(5, 6) * 0.23)
    dsupport = torch.sin(torch.arange(20, dtype=dtype).reshape(4, 5) * 0.31)
    dt, area = torch.tensor(0.07, dtype=dtype), torch.tensor(1.3, dtype=dtype)

    def updates(state: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qx, qy = transport.face_volume_fluxes(psi)
        transport._validate_divergence(qx, qy)
        edges = (state[:, 0], state[:, -1], state[0], state[-1])
        conservative = transport._euler_donorcell(state, qx, qy, edges, dt, area)
        material = transport._euler_donorcell(
            state, qx, qy, edges, dt, area, material_support=True,
        )
        return conservative, material

    with transport.selected_face_extension("x", 1, 3, 1):
        (conservative, material), (dconservative, dmaterial) = torch.func.jvp(
            updates, (support, potential), (dsupport, dpotential),
        )
    torch.testing.assert_close(material, conservative, atol=2e-15, rtol=2e-13)
    torch.testing.assert_close(dmaterial, dconservative, atol=2e-14, rtol=2e-12)


def test_native_powell_valid_control_no_longer_loses_known_support():
    from examples.weather_scenarios.fv_point_3h_netflux_current_cycle import load_fixed_case
    root = Path(__file__).resolve().parents[1]
    receipt = root / "graphify-out/fv-root-cause-20260919/native_reference_20261012/zero_powell_replay.json"
    control = torch.tensor(json.loads(receipt.read_text())["failed_control"], dtype=torch.float64)
    case = load_fixed_case()
    problem = case["problem"]
    with torch.no_grad():
        cost = problem.objective(control, case["parameters"])
    assert bool(torch.isfinite(cost))
    # The original strict support validator remains in force on every step.
    assert torch.equal(problem.frozen.initial_support_mask,
                       torch.ones_like(problem.frozen.initial_support_mask))
