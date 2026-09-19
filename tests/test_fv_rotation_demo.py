"""The larger demo preserves fixed scoring and separate time schedules."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/weather_scenarios'))
import fv_rotation_demo as demo


def test_missing_candidate_does_not_shrink_large_demo_score_domain():
    truth = torch.ones((demo.LEADS, 2, 2), dtype=torch.float64) * 20
    valid = torch.ones_like(truth, dtype=torch.bool)
    valid[0, 0, 0] = False
    metrics = demo._metrics(truth, truth, truth[0], valid)
    assert metrics[0]['domain_pixels'] == 4
    assert metrics[0]['missing_pixels'] == 1
    assert metrics[0]['mae'] is None
    assert metrics[0]['scored_pixels'] == 0
    assert metrics[-1]['lead_minutes'] == 180
    assert metrics[-1]['mae'] == 0


def test_analysis_and_forecast_support_have_distinct_time_extents():
    analysis = demo._support_schedule(2)
    forecast = demo._support_schedule(demo.LEADS)
    assert len(analysis) == 2 * demo.SUBSTEPS
    assert len(forecast) == 18 * demo.SUBSTEPS
    assert analysis[0][0][0].shape == (240,)
    assert demo.GRID * demo.SPACING_M == 48_000
