import torch
from types import SimpleNamespace

from examples.weather_scenarios.fv_point_3h_limiter_ray_diagnostic import (
    LimiterTrace, ROOT, compare_traces, exact_selectors, local_module_sources,
)


def test_exact_selectors_distinguish_active_ties_and_negative_choices():
    left = torch.tensor([1., 2., -1., -2., 1., 0., 1., 1. + 1e-14], dtype=torch.float64)
    right = torch.tensor([2., 1., -2., -1., 1., 1., -1., 1.], dtype=torch.float64)
    assert exact_selectors(left, right).tolist() == [1, 2, 1, 2, 3, 0, 0, 2]


def test_flux_sign_change_is_not_reported_as_limiter_change():
    old, new = LimiterTrace(), LimiterTrace()
    field = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    qx, qy = torch.ones(4, 6), torch.ones(5, 5)
    old(field, qx, qy)
    changed = qy.clone()
    changed[2, 1] = -1
    new(field, qx, changed)
    result = compare_traces(old, new)
    assert result["limiter_changes"] == []
    assert result["flux_sign_changes"] == [
        {"stage": 0, "axis": "y", "face": [2, 1], "before": 1., "after": -1.}]


def test_changed_selector_is_localized_to_stage_axis_and_interior_cell():
    old, new = LimiterTrace(), LimiterTrace()
    field = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    altered = field.clone()
    altered[1, 1] += 0.25
    qx, qy = torch.ones(4, 6), torch.ones(5, 5)
    old(field, qx, qy)
    new(altered, qx, qy)
    result = compare_traces(old, new)
    change = next(c for c in result["limiter_changes"] if c["axis"] == "x" and c["cell"] == [1, 1])
    assert change["stage"] == 0
    assert (change["before"], change["after"]) == (3, 2)
    assert change["operands_after"] == [1.25, 0.75]
    assert result["flux_sign_changes"] == []


def test_synthetic_torch_filename_does_not_become_a_repository_source():
    path = "examples/weather_scenarios/fv_point_3h_limiter_ray_diagnostic.py"
    modules = [SimpleNamespace(__file__="_classes.py"),
               SimpleNamespace(__file__=str(ROOT / path))]
    assert local_module_sources(modules) == {path}
