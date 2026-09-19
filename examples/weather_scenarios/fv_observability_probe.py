"""Small local observability probe for the production minmod FV step.

The result is a discrete Jacobian experiment on a 6x7 Cartesian grid.  It is
not a recovery result, a prior experiment, or evidence for the 128x128 D7
case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from advar.transport import face_volume_fluxes, finite_volume_step


HEIGHT = 6
WIDTH = 7
INTERVAL = 1.0
SUBSTEPS = 4
DT = INTERVAL / SUBSTEPS
PSI_CONTROL_COUNT = 8
GROWTH_CONTROL_INDEX = PSI_CONTROL_COUNT
INITIAL_Q_OFFSET = PSI_CONTROL_COUNT + 1
INITIAL_Q_CONTROL_COUNT = HEIGHT * WIDTH
CONTROL_COUNT = INITIAL_Q_OFFSET + INITIAL_Q_CONTROL_COUNT
PSI_SCALE_M2_PER_S = 0.02
GROWTH_SCALE_PER_INTERVAL = 0.03
INITIAL_Q_SCALE = 1.0


def _edges(height: int, width: int, value: float, *, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(torch.full((size,), value, dtype=dtype) for size in (height, height, width, width))


def _coarse_psi(values: torch.Tensor) -> torch.Tensor:
    coarse = values.new_zeros((3, 3))
    coarse.reshape(-1)[1:] = values
    return F.interpolate(
        coarse[None, None],
        size=(HEIGHT + 1, WIDTH + 1),
        mode="bilinear",
        align_corners=True,
    )[0, 0]


def _asymmetric_initial_q(dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(HEIGHT, dtype=dtype),
        torch.arange(WIDTH, dtype=dtype),
        indexing="ij",
    )
    return 0.25 + 0.07 * y + 0.11 * x + 0.015 * y * x


def _case_inputs(name: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if name == "knownclear":
        initial = torch.zeros((HEIGHT, WIDTH), dtype=dtype)
        boundary_value = 0.0
        psi_base = torch.zeros(PSI_CONTROL_COUNT, dtype=dtype)
    elif name == "knownconstant_inflow":
        initial = torch.ones((HEIGHT, WIDTH), dtype=dtype)
        boundary_value = 1.0
        psi_base = torch.zeros(PSI_CONTROL_COUNT, dtype=dtype)
    elif name == "asymmetricpositive_knownzero_exterior":
        initial = _asymmetric_initial_q(dtype)
        boundary_value = 0.0
        y, x = torch.meshgrid(
            torch.arange(3, dtype=dtype),
            torch.arange(3, dtype=dtype),
            indexing="ij",
        )
        psi_base = (0.01 * y - 0.008 * x).reshape(-1)[1:]
    else:
        raise ValueError(f"unknown observability case: {name}")
    return initial, psi_base, boundary_value, 0.0


def _observations(control: torch.Tensor, name: str) -> torch.Tensor:
    initial, psi_base, boundary_value, growth_base = _case_inputs(name, control.dtype)
    psi_control = psi_base + PSI_SCALE_M2_PER_S * control[:PSI_CONTROL_COUNT]
    growth = growth_base + GROWTH_SCALE_PER_INTERVAL * control[GROWTH_CONTROL_INDEX]
    echo = initial + INITIAL_Q_SCALE * control[INITIAL_Q_OFFSET:].reshape(HEIGHT, WIDTH)
    psi = _coarse_psi(psi_control)
    qx, qy = face_volume_fluxes(psi)
    boundary_echo = _edges(HEIGHT, WIDTH, boundary_value, dtype=control.dtype)
    boundary_support = _edges(HEIGHT, WIDTH, 1.0, dtype=control.dtype)
    stages_echo = (boundary_echo, boundary_echo)
    stages_support = (boundary_support, boundary_support)
    frames = [echo]
    for step_index in range(1, 2 * SUBSTEPS + 1):
        result = finite_volume_step(
            echo,
            torch.ones_like(echo),
            qx,
            qy,
            dt_seconds=DT,
            spacing_yx=(1.0, 1.0),
            log_growth=growth * DT,
            boundary_echo=stages_echo,
            boundary_support=stages_support,
            max_courant=0.5,
            reconstruction="minmod",
        )
        echo = result.echo
        if step_index % SUBSTEPS == 0:
            frames.append(echo)
    return torch.stack(frames).reshape(-1)


def _sequential_jacobian(control: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    nominal = _observations(control, name)
    columns = []
    for index in range(control.numel()):
        direction = torch.zeros_like(control)
        direction[index] = 1.0
        _, column = torch.func.jvp(
            lambda value: _observations(value, name),
            (control,),
            (direction,),
        )
        columns.append(column)
    return nominal, torch.stack(columns, dim=1)


def _rank_info(matrix: torch.Tensor, *, tolerance: float) -> dict[str, object]:
    matrix = matrix.to(dtype=torch.float64)
    singular = torch.linalg.svdvals(matrix)
    rank = int(torch.count_nonzero(singular > tolerance))
    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "singular_values": [float(value) for value in singular],
        "tolerance": tolerance,
    }


def _shared_rank_tolerance(jacobian: torch.Tensor) -> tuple[float, float]:
    """Use one physical-data scale for conditional and nuisance ranks."""
    singular = torch.linalg.svdvals(jacobian.to(dtype=torch.float64))
    smax = float(singular[0]) if singular.numel() else 0.0
    tolerance = torch.finfo(torch.float64).eps * max(jacobian.shape) * smax
    return smax, tolerance


def _project_off(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    base = base.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)
    if base.shape[1] == 0:
        return target
    coefficients = torch.linalg.lstsq(base, target).solution
    return target - base @ coefficients


def run_probe(output_path: str | Path | None = None) -> dict[str, object]:
    torch.set_num_threads(1)
    dtype = torch.float64
    control = torch.zeros(CONTROL_COUNT, dtype=dtype)
    records = []
    for name in (
        "knownclear",
        "knownconstant_inflow",
        "asymmetricpositive_knownzero_exterior",
    ):
        observations, jacobian = _sequential_jacobian(control, name)
        full_jacobian_smax, shared_tolerance = _shared_rank_tolerance(jacobian)
        flow = jacobian[:, :PSI_CONTROL_COUNT]
        growth = jacobian[:, GROWTH_CONTROL_INDEX : GROWTH_CONTROL_INDEX + 1]
        initial_q = jacobian[:, INITIAL_Q_OFFSET:]
        dynamics = torch.cat((flow, growth), dim=1)
        projected_flow = _project_off(initial_q, flow)
        projected_growth = _project_off(initial_q, growth)
        projected_dynamics = torch.cat((projected_flow, projected_growth), dim=1)
        records.append(
            {
                "case": name,
                "observation_shape": list(observations.shape),
                "initial_frame_min": float(observations[: HEIGHT * WIDTH].min()),
                "initial_frame_max": float(observations[: HEIGHT * WIDTH].max()),
                "full_physical_jacobian_smax": full_jacobian_smax,
                "shared_rank_tolerance_shape": list(jacobian.shape),
                "shared_rank_tolerance": shared_tolerance,
                "conditional_flow_jacobian": _rank_info(flow, tolerance=shared_tolerance),
                "growth_column": _rank_info(growth, tolerance=shared_tolerance),
                "conditional_dynamics_jacobian": _rank_info(dynamics, tolerance=shared_tolerance),
                "initial_q_jacobian": _rank_info(initial_q, tolerance=shared_tolerance),
                "projected_flow_jacobian": _rank_info(projected_flow, tolerance=shared_tolerance),
                "projected_growth_column": _rank_info(projected_growth, tolerance=shared_tolerance),
                "projected_dynamics_jacobian": _rank_info(projected_dynamics, tolerance=shared_tolerance),
                "joint_data_only_identifiability": _rank_info(
                    torch.cat((initial_q, projected_dynamics), dim=1),
                    tolerance=shared_tolerance,
                ),
                "flow_prescribed_not_recovered": True,
                "stationarity_claimed": False,
            }
        )
    report: dict[str, object] = {
        "status": "complete",
        "scope": "local_discrete_minmod_fv_observability",
        "grid_shape_yx": [HEIGHT, WIDTH],
        "observation_times": [0.0, 1.0, 2.0],
        "substeps_per_interval": SUBSTEPS,
        "dt_seconds": DT,
        "dtype": str(dtype).removeprefix("torch."),
        "device": "cpu",
        "control_layout": {
            "coarse_psi_controls": PSI_CONTROL_COUNT,
            "g_column_index": GROWTH_CONTROL_INDEX,
            "initial_q_controls": INITIAL_Q_CONTROL_COUNT,
            "total": CONTROL_COUNT,
            "gauge": "coarse_psi[0,0]=0",
        },
        "physical_control_scale": {
            "psi_m2_per_s": PSI_SCALE_M2_PER_S,
            "growth_per_interval": GROWTH_SCALE_PER_INTERVAL,
            "initial_echo_proxy": INITIAL_Q_SCALE,
        },
        "common_scalar_field_std": INITIAL_Q_SCALE,
        "rank_definition": "SVD rank with shared per-case tolerance eps_float64 * max(full_J.shape) * smax(full_J); no prior or regularization",
        "claims": {
            "flow_prescribed": True,
            "flow_recovered_from_data": False,
            "stationarity_verified": False,
            "full_d7_128x128": False,
        },
        "cases": records,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graphify-out/fv-integration-20260910/observability.json"),
    )
    args = parser.parse_args()
    report = run_probe(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
