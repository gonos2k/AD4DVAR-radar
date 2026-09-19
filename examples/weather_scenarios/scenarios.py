"""Deterministic analytic radar scenarios for the weather-scenarios demo."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch

from advar.nowcast import NowcastConfig, nowcast


INTERVAL_MINUTES = 10
HEIGHT = WIDTH = 48
MIN_DBZ = -10.0
MAX_DBZ = 60.0
OBSERVATION_STEPS = (-2, -1, 0)
TRUTH_STEPS = tuple(range(1, 19))
DIAGNOSTIC_THRESHOLDS_DBZ = (10.0, 20.0)


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float64),
        torch.arange(WIDTH, dtype=torch.float64),
        indexing="ij",
    )


def _blob(
    y: torch.Tensor,
    x: torch.Tensor,
    center_y: float | torch.Tensor,
    center_x: float | torch.Tensor,
    scale_y: float,
    scale_x: float,
    amplitude: float,
) -> torch.Tensor:
    return amplitude * torch.exp(
        -0.5
        * (((y - center_y) / scale_y).square() + ((x - center_x) / scale_x).square())
    )


def analytic_field(case_id: str, step: int) -> torch.Tensor:
    """Return one dBZ field without using the production remapper."""
    y, x = _grid()
    if case_id == "translation":
        value = _blob(y, x, 24.0 + 0.7 * step, 22.0 - 0.45 * step, 4.5, 6.0, 42.0)
        value += _blob(y, x, 16.0 + 0.7 * step, 31.0 - 0.45 * step, 2.4, 2.0, 18.0)
    elif case_id == "growth_decay":
        factor = math.exp(-0.045 * step)
        value = factor * _blob(y, x, 24.0, 23.0, 5.0, 7.0, 38.0)
    elif case_id == "rotation":
        angle = 0.10 * step
        cosine, sine = math.cos(angle), math.sin(angle)
        dy, dx = y - 24.0, x - 23.0
        rotated_x = cosine * dx + sine * dy
        rotated_y = -sine * dx + cosine * dy
        value = 38.0 * torch.exp(
            -0.5 * ((rotated_x / 8.0).square() + (rotated_y / 2.0).square())
        )
        secondary_y = 24.0 + cosine * (-6.0) - sine * 6.0
        secondary_x = 23.0 + sine * (-6.0) + cosine * 6.0
        value += _blob(y, x, secondary_y, secondary_x, 2.0, 2.8, 20.0)
    elif case_id == "splitting_opposingcells":
        separation = 2.0 + 0.75 * step
        value = _blob(y, x, 24.0, 23.0 - separation, 3.0, 3.5, 29.0)
        value += _blob(y, x, 24.0, 23.0 + separation, 3.0, 3.5, 29.0)
    elif case_id == "newcell_birth":
        value = _blob(y, x, 27.0 + 0.3 * step, 23.0, 4.0, 5.5, 39.0)
        if step > 0:
            value += _blob(y, x, 12.0, 35.0 - 0.2 * step, 2.5, 2.5, 32.0)
    elif case_id == "boundary_outflow":
        value = _blob(y, x, 4.0 - 1.15 * step, 24.0, 3.2, 4.0, 43.0)
    elif case_id == "missing_coverage":
        value = _blob(y, x, 24.0 + 0.7 * step, 22.0 - 0.45 * step, 4.5, 6.0, 42.0)
        value += _blob(y, x, 16.0 + 0.7 * step, 31.0 - 0.45 * step, 2.4, 2.0, 18.0)
    else:
        raise ValueError(f"unknown scenario: {case_id}")
    return (MIN_DBZ + value).clamp(MIN_DBZ, MAX_DBZ)


SCENARIO_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "translation",
        "name": "평행 이동",
        "description": "두 반사도 봉우리가 일정한 속도로 이동합니다.",
        "expectation": "P0가 하나의 전역 이동을 추정합니다.",
        "lesson": "일관된 이동은 persistence보다 긴 lead에서 유리할 수 있습니다.",
        "model_limit": "회전·분열·새 세포를 전역 이동 하나로 표현하지 않습니다.",
    },
    {
        "id": "growth_decay",
        "name": "감쇠",
        "description": "고정된 세포의 반사도 강도가 매 10분 감쇠합니다.",
        "expectation": "10분 간격의 log-growth 감쇠 추정을 확인합니다.",
        "lesson": "위치는 맞아도 강도 경향이 틀리면 오차가 남습니다.",
        "model_limit": "강도 변화는 한 개의 전역 log-growth per step으로 요약됩니다.",
    },
    {
        "id": "rotation",
        "name": "비대칭 회전",
        "description": "길쭉한 세포와 보조 lobe의 방향이 회전합니다.",
        "expectation": "예측은 유한해야 하지만 회전 skill은 보장하지 않습니다.",
        "lesson": "유효한 수치와 표현 가능한 물리 과정은 별개입니다.",
        "model_limit": "P0는 회전을 전역 평행 이동으로 근사합니다.",
    },
    {
        "id": "splitting_opposingcells",
        "name": "분열·반대 이동",
        "description": "두 세포가 서로 반대 방향으로 벌어집니다.",
        "expectation": "충돌하는 pair evidence는 fallback과 불확실성으로 드러납니다.",
        "lesson": "global state 하나가 다중 세포 운동을 모두 보존하지 않습니다.",
        "model_limit": "단일 displacement는 두 속도를 동시에 표현할 수 없습니다.",
    },
    {
        "id": "newcell_birth",
        "name": "새 세포 발생",
        "description": "관측 창 뒤에 새로운 반사도 세포가 발생합니다.",
        "expectation": "기존 관측으로 새 세포를 미리 맞힐 것을 요구하지 않습니다.",
        "lesson": "결측과 미래 사건을 같은 실패로 해석하지 않습니다.",
        "model_limit": "P0에는 명시적인 cell birth 과정이 없습니다.",
    },
    {
        "id": "boundary_outflow",
        "name": "경계 유출",
        "description": "세포가 북쪽 경계를 빠져나갑니다.",
        "expectation": "도메인 밖 echo는 반대편으로 wrap하지 않습니다.",
        "lesson": "경계 유출은 물리적 소멸과 구분되는 도메인 효과입니다.",
        "model_limit": "유한 격자 밖의 실제 강수는 평가영역에 없습니다.",
    },
    {
        "id": "missing_coverage",
        "name": "부분 관측",
        "description": "세 시각 모두 같은 영역이 결측입니다.",
        "expectation": "결측 셀은 null 관측으로 남고 score domain에서 제외됩니다.",
        "lesson": "유효 마스크와 관측 coverage를 함께 읽어야 합니다.",
        "model_limit": "Persistence도 최신 관측의 결측을 유지하며, 함께 유효한 화소에서만 평가합니다.",
    },
)


def _json_grid(values: torch.Tensor) -> list[list[float | None]]:
    return [
        [None if not math.isfinite(float(value)) else round(float(value), 2) for value in row]
        for row in values.detach().cpu()
    ]


def _metric(
    forecast: torch.Tensor,
    persistence: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor | None,
    common: torch.Tensor | None = None,
) -> dict[str, Any]:
    finite_forecast = torch.isfinite(forecast)
    finite_persistence = torch.isfinite(persistence)
    finite_truth = torch.isfinite(truth)
    domain = valid & finite_forecast & finite_persistence & finite_truth
    count = int(domain.sum())
    total = forecast.numel()
    issued = valid & finite_forecast
    issued_count = int(issued.sum())
    if common is None:
        common = finite_persistence & finite_truth
    common = common.to(dtype=torch.bool)
    detection: dict[str, dict[str, int | float | None]] = {}
    for threshold in DIAGNOSTIC_THRESHOLDS_DBZ:
        label = str(int(threshold))
        forecast_positive = finite_forecast & (forecast >= threshold)
        truth_positive = finite_truth & (truth >= threshold)
        hit = int((domain & forecast_positive & truth_positive).sum())
        miss = int((domain & ~forecast_positive & truth_positive).sum())
        false_alarm = int((domain & forecast_positive & ~truth_positive).sum())
        csi_denominator = hit + miss + false_alarm
        pod_denominator = hit + miss
        detection[label] = {
            "hits": hit,
            "misses": miss,
            "false_alarms": false_alarm,
            "csi": None if csi_denominator == 0 else round(hit / csi_denominator, 4),
            "pod": None if pod_denominator == 0 else round(hit / pod_denominator, 4),
            "truth_echo_pixels": int(truth_positive.sum()),
            "scored_truth_echo_pixels": int((domain & truth_positive).sum()),
            "excluded_truth_echo_pixels": int((truth_positive & ~domain).sum()),
            "truth_positive_outside_common": int((truth_positive & ~common).sum()),
        }
    return {
        "lead_minutes": 0,
        "mae": None if count == 0 else round(float((forecast[domain] - truth[domain]).abs().mean()), 4),
        "persistence_mae": None if count == 0 else round(float((persistence[domain] - truth[domain]).abs().mean()), 4),
        "scored_pixels": count,
        "scored_fraction": round(count / total, 4),
        "valid_fraction": round(float(valid.float().mean()), 4),
        "confidence": None if count == 0 or confidence is None else round(float(confidence[domain].mean()), 4),
        "issued_pixels": issued_count,
        "issued_missing_pixels": total - issued_count,
        "unscored_pixels": total - count,
        "detection": detection,
    }


def _tendency_reason(metadata: Any) -> str:
    if metadata.tendency_source.value == "BACKGROUND":
        return "background_tendency"
    if metadata.tendency_source.value == "NONE":
        return "no_usable_pair_support"
    if metadata.motion_pair_conflict or metadata.growth_pair_conflict:
        return "observation_pair_conflict_blended"
    if metadata.tendency_source.value == "OBSERVATION":
        return "observation_tendency"
    return "no_usable_tendency"


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_dataset(*, source_commit: str | None = None) -> dict[str, Any]:
    """Run default P0 on every case and return the browser-facing payload."""
    config = NowcastConfig()
    scenarios: list[dict[str, Any]] = []
    for spec in SCENARIO_SPECS:
        full = torch.stack(tuple(analytic_field(spec["id"], step) for step in range(-2, 19)))
        source = full[:3]
        truth = full[3:]
        observations = source.clone()
        persistence_source = "latest_observation"
        if spec["id"] == "missing_coverage":
            observations[:, 15:31, 18:34] = float("nan")
        persistence_source = "latest_observation"
        result = nowcast(observations, config)
        persistence = observations[-1]
        common = torch.isfinite(observations).all(dim=0)
        metrics: list[dict[str, Any]] = []
        for lead, truth_frame in enumerate(truth):
            metric = _metric(
                result.forecast_dbz[lead],
                persistence,
                truth_frame,
                result.valid_mask[lead],
                result.forecast_confidence[lead],
                common,
            )
            metric["lead_minutes"] = (lead + 1) * INTERVAL_MINUTES
            metrics.append(metric)
        scenarios.append(
            {
                **spec,
                "persistence_source": persistence_source,
                "observations": [_json_grid(frame) for frame in observations],
                "truth": [_json_grid(frame) for frame in truth],
                "forecast": [_json_grid(frame) for frame in result.forecast_dbz],
                "persistence": _json_grid(persistence),
                "metrics": metrics,
                "state": {
                    "displacement_yx": [round(float(v), 6) for v in result.state.displacement_yx],
                    "log_growth_per_step": round(float(result.state.log_growth_per_step), 6),
                    "data_status": result.metadata.data_status.value,
                    "tendency_source": result.metadata.tendency_source.value,
                    "dynamics_source": result.metadata.dynamics_source.value,
                    "motion_pair_count": result.metadata.motion_pair_count,
                    "growth_pair_count": result.metadata.growth_pair_count,
                    "motion_pair_selection": result.metadata.motion_pair_selection.value,
                    "growth_pair_selection": result.metadata.growth_pair_selection.value,
                    "reason": _tendency_reason(result.metadata),
                },
            }
        )
    return {
        "meta": {
            "interval_minutes": INTERVAL_MINUTES,
            "grid_shape": [HEIGHT, WIDTH],
            "min_dbz": MIN_DBZ,
            "max_dbz": MAX_DBZ,
            "source_commit": source_commit or _source_commit(),
            "generator": "analytic-weather-scenarios-v1",
            "model": "ADVAR P0 default NowcastConfig; synthetic theory demo",
            "metric_note": "metrics are computed from raw tensors before JSON rounding; grids are rounded to 2 dBZ decimals and nonfinite values are null. detection truth_echo_pixels is all finite truth positives, scored_truth_echo_pixels is the score domain, excluded_truth_echo_pixels is outside that domain, and truth_positive_outside_common is outside the common observed mask",
        },
        "scenarios": scenarios,
    }


def dumps_dataset(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    print(dumps_dataset(build_dataset()))
