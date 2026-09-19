"""Focused regression for the isolated prescribed-flow convergence probe."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

_SPEC = spec_from_file_location(
    "fv_grid_convergence_probe",
    Path(__file__).parents[1] / "examples/weather_scenarios/fv_grid_convergence_probe.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("FV grid convergence probe module is unavailable")
_PROBE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


@pytest.fixture(scope="module")
def convergence_report() -> dict:
    return _PROBE.run_probe((32, 64))


@pytest.mark.parametrize("case_name", ["translation", "rotation", "area_preserving_strain"])
def test_spatial_and_jvp_errors_decrease_on_refinement(case_name: str, convergence_report: dict) -> None:
    report = convergence_report
    rows = [row for row in report["results"] if row["case"] == case_name]
    assert [row["size"] for row in rows] == [32, 64]
    assert all(row["actual_max_cfl"] <= report["max_courant"] for row in rows)
    assert all(row["max_transformed_budget_residual"] < 1e-12 for row in rows)
    assert rows[1]["leads"][-1]["relative_echo_l2"] < rows[0]["leads"][-1]["relative_echo_l2"]
    assert rows[1]["derivative"]["jvp_relative_to_independent_oracle"] < rows[0]["derivative"]["jvp_relative_to_independent_oracle"]


def test_shape_moments_include_uniform_cell_interior():
    # Four cells cover [-2,2]^2: each coordinate has variance 4/3 m^2.
    shape = _PROBE._shape(torch.ones((2, 2), dtype=torch.float64), 2.0, 0.5)
    assert shape["centroid_yx_m"] == pytest.approx([0.0, 0.0])
    assert shape["variance_yx_m2"] == pytest.approx([4 / 3, 4 / 3])
    assert shape["maximum_echo"] == 1.0
    assert shape["area_above_threshold_m2"] == 16.0


def test_empty_shape_has_no_defined_centroid_or_width():
    shape = _PROBE._shape(torch.zeros((2, 2), dtype=torch.float64), 2.0, 0.5)
    assert shape["centroid_yx_m"] is None
    assert shape["variance_yx_m2"] is None
    assert shape["maximum_echo"] == 0.0
    assert shape["area_above_threshold_m2"] == 0.0


def test_minmod_one_lead_forwards_reconstruction_to_primal_and_jvp(
    monkeypatch,
):
    step_reconstructions: list[str] = []
    advance_reconstructions: list[str] = []
    original_step = _PROBE.transport.finite_volume_step
    original_advance = _PROBE._advance

    def step_wrapper(*args, **kwargs):
        step_reconstructions.append(kwargs.get("reconstruction", "donorcell"))
        return original_step(*args, **kwargs)

    def advance_wrapper(*args, **kwargs):
        reconstruction = (
            args[-1] if len(args) >= 7 else kwargs.get("reconstruction", "donorcell")
        )
        advance_reconstructions.append(reconstruction)
        return original_advance(*args, **kwargs)

    monkeypatch.setattr(_PROBE.transport, "finite_volume_step", step_wrapper)
    monkeypatch.setattr(_PROBE, "_advance", advance_wrapper)
    report = _PROBE.run_probe((16,), leads=1, reconstruction="minmod")

    assert report["scheme"] == "current minmod SSPRK2"
    assert all(row["reconstruction"] == "minmod" for row in report["results"])
    assert advance_reconstructions == ["minmod"] * len(_PROBE.CASES)
    expected_steps = sum(2 * row["substeps_per_lead"] for row in report["results"])
    assert len(step_reconstructions) == expected_steps
    assert set(step_reconstructions) == {"minmod"}
