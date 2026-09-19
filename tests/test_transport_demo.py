"""Protect display averaging and failed-checkpoint reporting in teaching tools."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/weather_scenarios'))
from build_transport_demo import display_field, boundary_examples
import existing_p1_probe


def test_display_coarsening_preserves_echo_integral():
    field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    coarse = torch.tensor(display_field(field), dtype=field.dtype)
    torch.testing.assert_close(coarse, torch.tensor([2.5, 4.5, 10.5, 12.5], dtype=field.dtype))
    assert float(coarse.sum()) * 4 == float(field.sum())


def test_boundary_demo_distinguishes_unknown_clear_and_partial():
    unknown, clear, partial = boundary_examples()
    assert unknown['echo'] == clear['echo'] == 0
    assert unknown['support'] == 0 < clear['support']
    assert abs(partial['echo'] - .3325) < 1e-14
    assert abs(partial['support'] - .0475) < 1e-14


def test_learning_probe_does_not_report_success_when_copy_fails(tmp_path, monkeypatch):
    @dataclass
    class Result:
        objective_after: float = .1

    def train(*, checkpoint_path):
        checkpoint_path.write_bytes(b'isolated-test-checkpoint')
        return Result()

    destination = tmp_path / 'directory'
    destination.mkdir()
    output = tmp_path / 'report.json'
    monkeypatch.setattr(existing_p1_probe, 'run_mean_only_learning', train)
    monkeypatch.setattr(existing_p1_probe, '_destination', lambda _: destination)
    monkeypatch.setattr(sys, 'argv', ['probe', '--output', str(output),
                                     '--checkpoint-destination', str(destination)])
    assert existing_p1_probe.main() == 1
    result = json.loads(output.read_text())
    assert result['status'] == 'failed'
    assert result['failure']['type'] == 'IsADirectoryError'
