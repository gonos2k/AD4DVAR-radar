"""Focused regression for the isolated prescribed-flow convergence probe."""

from __future__ import annotations

import pytest

from examples.weather_scenarios.fv_grid_convergence_probe import run_probe


@pytest.fixture(scope="module")
def convergence_report() -> dict:
    return run_probe((32, 64))


@pytest.mark.parametrize("case_name", ["translation", "rotation", "area_preserving_strain"])
def test_spatial_and_jvp_errors_decrease_on_refinement(case_name: str, convergence_report: dict) -> None:
    report = convergence_report
    rows = [row for row in report["results"] if row["case"] == case_name]
    assert [row["size"] for row in rows] == [32, 64]
    assert all(row["actual_max_cfl"] <= report["max_courant"] for row in rows)
    assert all(row["max_transformed_budget_residual"] < 1e-12 for row in rows)
    assert rows[1]["leads"][-1]["relative_echo_l2"] < rows[0]["leads"][-1]["relative_echo_l2"]
    assert rows[1]["derivative"]["jvp_relative_to_independent_oracle"] < rows[0]["derivative"]["jvp_relative_to_independent_oracle"]
