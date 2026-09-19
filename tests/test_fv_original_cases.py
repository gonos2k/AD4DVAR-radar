"""Fixed-domain metric guards for the bounded original-page FV adapter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import pytest


_SPEC = spec_from_file_location(
    "fv_original_cases",
    Path(__file__).parents[1] / "examples/weather_scenarios/fv_original_cases.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_fixed_metric_keeps_persistence_on_full_domain():
    truth = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)
    persistence = torch.zeros_like(truth)
    good = _MODULE._metric_fixed(truth + 1.0, persistence, truth, None)
    altered = _MODULE._metric_fixed(truth + 2.0, persistence, truth, None)
    assert good is not None and altered is not None
    assert good["domain_pixels"] == 4 and good["missing_pixels"] == 0
    assert good["persistence_mae"] == altered["persistence_mae"] == 4.0


def test_any_missing_candidate_is_unscored_and_marked():
    truth = torch.ones((2, 2), dtype=torch.float64)
    persistence = torch.zeros_like(truth)
    forecast = truth.clone()
    forecast[0, 1] = float("nan")
    valid = torch.ones_like(truth, dtype=torch.bool)
    metric = _MODULE._metric_fixed(forecast, persistence, truth, None, valid)
    assert metric is not None
    assert metric["mae"] is None
    assert metric["domain_pixels"] == 4
    assert metric["missing_pixels"] == 1
    assert metric["scored_pixels"] == 0
    assert metric["persistence_mae"] == 1.0
    assert all(item["hits"] is None for item in metric["detection"].values())


def test_failed_metric_records_both_leads_with_fixed_domain():
    truth = torch.ones((2, 2, 2), dtype=torch.float64)
    failed = _MODULE._failed_metrics(torch.zeros((2, 2), dtype=torch.float64), truth)
    assert [metric["lead_minutes"] for metric in failed] == [10, 20]
    assert all(metric["mae"] is None for metric in failed)
    assert all(metric["missing_pixels"] == 4 for metric in failed)
    assert len(_MODULE._null_grid()) == _MODULE.GRID
    assert all(len(row) == _MODULE.GRID for row in _MODULE._null_grid())


def test_outer_budget_rejects_nonpositive_without_running_cases():
    with pytest.raises(ValueError, match="maximum_outer_iterations"):
        _MODULE.build_payload(0)
