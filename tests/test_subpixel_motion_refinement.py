"""Independent measurements for the opt-in subpixel motion experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from advar.nowcast import NowcastConfig


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "weather_scenarios"
    / "motion_experiment.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "advar_motion_experiment",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MOTION_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOTION_EXPERIMENT)


def _analytic_frame(
    y: torch.Tensor,
    x: torch.Tensor,
    step: float,
    shift_yx: tuple[float, float],
    width_yx: tuple[float, float],
    amplitude: float,
) -> torch.Tensor:
    shift_y, shift_x = shift_yx
    width_y, width_x = width_yx
    return -10.0 + amplitude * torch.exp(
        -0.5
        * (
            ((y - (24.0 + step * shift_y)) / width_y).square()
            + ((x - (24.0 + step * shift_x)) / width_x).square()
        )
    )


class SubpixelMotionRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NowcastConfig()
        self.y, self.x = torch.meshgrid(
            torch.arange(48, dtype=torch.float64),
            torch.arange(48, dtype=torch.float64),
            indexing="ij",
        )

    def _measure(
        self,
        shift: tuple[float, float],
        *,
        width: tuple[float, float] = (4.0, 5.5),
        amplitude: float = 39.0,
    ) -> dict[str, object]:
        previous = _analytic_frame(
            self.y, self.x, 0.0, shift, width, amplitude
        )
        current = _analytic_frame(
            self.y, self.x, 1.0, shift, width, amplitude
        )
        return _MOTION_EXPERIMENT.local_dft_motion_experiment(
            previous,
            current,
            self.config,
        )

    def test_signed_fractional_shifts_improve_independent_error(self) -> None:
        for shift in (
            (0.3, 0.0),
            (-0.3, 0.0),
            (0.0, 0.3),
            (0.0, -0.3),
            (0.3, -0.3),
        ):
            result = self._measure(shift)
            target = torch.tensor(shift, dtype=torch.float64)
            baseline_error = torch.linalg.vector_norm(
                result["baseline_shift_yx"] - target  # type: ignore[operator]
            )
            candidate_error = torch.linalg.vector_norm(
                result["candidate_shift_yx"] - target  # type: ignore[operator]
            )
            self.assertLess(float(candidate_error), 0.05)
            self.assertLess(float(candidate_error), float(baseline_error))
            self.assertTrue(result["candidate_alignment_improved"])
            self.assertGreaterEqual(
                float(result["objective_after"]),
                float(result["objective_before"])
                - self.config.contract_absolute_tolerance,
            )

    def test_width_and_contrast_changes_do_not_change_shift_sign(self) -> None:
        for width, amplitude in (
            ((1.5, 4.0), 39.0),
            ((8.0, 2.0), 39.0),
            ((4.0, 5.5), 60.0),
        ):
            result = self._measure(
                (0.3, -0.3),
                width=width,
                amplitude=amplitude,
            )
            candidate = result["candidate_shift_yx"]  # type: ignore[assignment]
            self.assertTrue(torch.all(torch.isfinite(candidate)))
            self.assertGreater(float(candidate[0]), 0.0)
            self.assertLess(float(candidate[1]), 0.0)
            torch.testing.assert_close(
                candidate,
                torch.tensor((0.3, -0.3), dtype=torch.float64),
                atol=0.1,
                rtol=0.0,
            )

    def test_broad_fast_motion_remains_a_visible_candidate_limit(self) -> None:
        result = self._measure(
            (0.9, -0.4),
            width=(10.0, 10.0),
        )
        target = torch.tensor((0.9, -0.4), dtype=torch.float64)
        candidate_error = torch.linalg.vector_norm(
            result["candidate_shift_yx"] - target  # type: ignore[operator]
        )
        self.assertGreater(float(candidate_error), 0.1)
        self.assertTrue(result["candidate_alignment_improved"])


if __name__ == "__main__":
    unittest.main()
