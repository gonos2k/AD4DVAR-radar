"""Keep nonfinite chart refusal receipts JSON-safe without changing the search."""
from __future__ import annotations

import math
from typing import Any, Callable

from torch import Tensor

from examples.weather_scenarios import fv_point_3h_search_progress as frozen


def search_candidates(model: Any, current: Tensor,
        evaluate: Callable[[Tensor, float, float], dict[str, Any]], *,
        chart_candidate: Callable[[Tensor, Tensor, float], Tensor],
        progress: Callable[[list[dict[str, Any]], dict[str, Any]], None],
        radius: float = 0.05, limit: int = 24
        ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    def persist(trials: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        if trials:
            movement = trials[-1].get("actual_path_norm")
            if isinstance(movement, float) and not math.isfinite(movement):
                # The frozen search already refused this slot; null keeps the
                # undefined length recordable without replacing it with zero.
                trials[-1]["actual_path_norm"] = None
        progress(trials, meta)

    return frozen.search_candidates(model, current, evaluate,
        chart_candidate=chart_candidate, progress=persist, radius=radius, limit=limit)
