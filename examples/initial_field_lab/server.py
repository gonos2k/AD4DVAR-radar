#!/usr/bin/env python3
"""Local homepage for comparing ADVAR P0 initial fields on one fixed case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from advar import NowcastConfig, nowcast
from advar.nowcast import ForecastResult


ASSET_ROOT = Path(__file__).resolve().parent
CASE_HEIGHT = 48
CASE_WIDTH = 48
FORECAST_STEP_MINUTES = 10
MAXIMUM_REQUEST_BYTES = 16 * 1024


@dataclass(frozen=True)
class ExperimentSettings:
    use_background: bool = True
    background_age_minutes: float = 10.0
    shift_y: int = 0
    shift_x: int = 0
    intensity_bias_dbz: float = 0.0
    coverage_percent: int = 100
    lead_minutes: int = 30

    @classmethod
    def from_payload(cls, payload: object) -> ExperimentSettings:
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")

        use_background = payload.get("use_background", True)
        if not isinstance(use_background, bool):
            raise ValueError("use_background must be boolean")

        return cls(
            use_background=use_background,
            background_age_minutes=_number(
                payload,
                "background_age_minutes",
                minimum=0.0,
                maximum=60.0,
                default=10.0,
            ),
            shift_y=_integer(
                payload,
                "shift_y",
                minimum=-6,
                maximum=6,
                default=0,
            ),
            shift_x=_integer(
                payload,
                "shift_x",
                minimum=-6,
                maximum=6,
                default=0,
            ),
            intensity_bias_dbz=_number(
                payload,
                "intensity_bias_dbz",
                minimum=-12.0,
                maximum=12.0,
                default=0.0,
            ),
            coverage_percent=_integer(
                payload,
                "coverage_percent",
                minimum=25,
                maximum=100,
                default=100,
            ),
            lead_minutes=_integer(
                payload,
                "lead_minutes",
                minimum=10,
                maximum=180,
                default=30,
                multiple=10,
            ),
        )


def _number(
    payload: dict[str, object],
    name: str,
    *,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _integer(
    payload: dict[str, object],
    name: str,
    *,
    minimum: int,
    maximum: int,
    default: int,
    multiple: int = 1,
) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum or value % multiple:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum} "
            f"in steps of {multiple}"
        )
    return value


def _gaussian_field(step: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(CASE_HEIGHT, dtype=torch.float32),
        torch.arange(CASE_WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    center_y = 18.0 + 1.2 * step
    center_x = 31.0 - 0.8 * step
    core = torch.exp(-((y - center_y).square() / 30.0 + (x - center_x).square() / 42.0))
    shoulder = torch.exp(
        -(
            (y - (center_y + 8.0)).square() / 52.0
            + (x - (center_x - 10.0)).square() / 34.0
        )
    )
    return -10.0 + 52.0 * core + 23.0 * shoulder


def _fixed_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.stack(tuple(_gaussian_field(step) for step in (-2, -1, 0)))
    truth = torch.stack(tuple(_gaussian_field(step) for step in range(1, 19)))

    observed = source.clone()
    qc_mask = torch.ones_like(observed, dtype=torch.bool)
    qc_mask[:, 15:31, 19:34] = False
    observed[~qc_mask] = torch.nan
    return observed, source, truth


def _translate_without_wrap(
    values: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    result = torch.full_like(values, torch.nan)
    height, width = values.shape[-2:]

    source_y_start = max(0, -shift_y)
    source_y_stop = min(height, height - shift_y)
    source_x_start = max(0, -shift_x)
    source_x_stop = min(width, width - shift_x)
    if source_y_start >= source_y_stop or source_x_start >= source_x_stop:
        return result

    target_y_start = source_y_start + shift_y
    target_y_stop = source_y_stop + shift_y
    target_x_start = source_x_start + shift_x
    target_x_stop = source_x_stop + shift_x
    result[..., target_y_start:target_y_stop, target_x_start:target_x_stop] = values[
        ..., source_y_start:source_y_stop, source_x_start:source_x_stop
    ]
    return result


def _initial_field(
    source: torch.Tensor,
    settings: ExperimentSettings,
) -> torch.Tensor | None:
    if not settings.use_background:
        return None

    background = _translate_without_wrap(
        source,
        settings.shift_y,
        settings.shift_x,
    )
    background = torch.where(
        torch.isfinite(background),
        background + settings.intensity_bias_dbz,
        background,
    )
    covered_columns = math.ceil(CASE_WIDTH * settings.coverage_percent / 100.0)
    background[..., :, covered_columns:] = torch.nan
    return background


def _json_field(values: torch.Tensor) -> list[list[float | None]]:
    result: list[list[float | None]] = []
    for row in values.detach().cpu().tolist():
        result.append(
            [
                None if not math.isfinite(value) else round(float(value), 3)
                for value in row
            ]
        )
    return result


def _finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return round(float(finite.mean()), 4)


def _run_case(
    settings: ExperimentSettings,
) -> tuple[ForecastResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    frames, source, truth = _fixed_case()
    background = _initial_field(source, settings)
    result = nowcast(
        frames,
        NowcastConfig(),
        background_frames_dbz=background,
        background_age_minutes=(
            settings.background_age_minutes if background is not None else None
        ),
    )

    lead_index = settings.lead_minutes // FORECAST_STEP_MINUTES - 1
    latest_observation = frames[-1]
    latest_background = (
        torch.full_like(latest_observation, torch.nan)
        if background is None
        else background[-1]
    )
    return result, latest_observation, latest_background, truth[lead_index]


def _persistence(observation: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(observation), observation, background)


def _score_forecast(
    forecast: torch.Tensor,
    persistence: torch.Tensor,
    target: torch.Tensor,
    domain: torch.Tensor,
) -> dict[str, int | float | None]:
    """Score exactly the supplied domain; never silently drop missing pixels."""
    scored_pixels = int(domain.sum())
    missing = domain & ~(
        torch.isfinite(forecast)
        & torch.isfinite(persistence)
        & torch.isfinite(target)
    )
    missing_pixels = int(missing.sum())
    forecast_mae = persistence_mae = skill = math.nan
    if scored_pixels and not missing_pixels:
        forecast_mae = float((forecast[domain] - target[domain]).abs().mean())
        persistence_mae = float((persistence[domain] - target[domain]).abs().mean())
        skill = persistence_mae - forecast_mae
    return {
        "forecast_mae_dbz": _round_or_none(forecast_mae),
        "persistence_mae_dbz": _round_or_none(persistence_mae),
        "skill_dbz": _round_or_none(skill),
        "scored_pixels": scored_pixels if not missing_pixels else 0,
        "scored_fraction": (
            round(float(domain.float().mean()), 4) if not missing_pixels else 0.0
        ),
        "missing_pixels": missing_pixels,
    }


def _compare_forecasts(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    persistence: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    # A fixes the domain and persistence. Missing B values cannot shrink it.
    domain = (
        torch.isfinite(reference)
        & torch.isfinite(persistence)
        & torch.isfinite(target)
    )
    reference_scores = _score_forecast(reference, persistence, target, domain)
    candidate_scores = _score_forecast(candidate, persistence, target, domain)
    improvement = math.nan
    if int(domain.sum()) and not candidate_scores["missing_pixels"]:
        improvement = float(
            (
                (reference[domain] - target[domain]).abs()
                - (candidate[domain] - target[domain]).abs()
            ).mean()
        )
    return {
        "reference": reference_scores,
        "candidate": candidate_scores,
        "improvement_dbz": _round_or_none(improvement),
        "domain_pixels": int(domain.sum()),
        "domain_fraction": round(float(domain.float().mean()), 4),
    }


def run_experiment(
    settings: ExperimentSettings,
    reference_settings: ExperimentSettings | None = None,
) -> dict[str, object]:
    if (
        reference_settings is not None
        and reference_settings.lead_minutes != settings.lead_minutes
    ):
        raise ValueError("A/B comparison requires the same lead_minutes")
    result, latest_observation, latest_background, target = _run_case(settings)
    lead_index = settings.lead_minutes // FORECAST_STEP_MINUTES - 1
    valid = result.valid_mask[lead_index].detach().cpu()
    forecast = result.forecast_dbz[lead_index].detach().cpu()
    forecast = forecast.masked_fill(~valid, torch.nan)
    persistence = _persistence(latest_observation, latest_background)
    common = (
        torch.isfinite(forecast)
        & torch.isfinite(persistence)
        & torch.isfinite(target)
    )
    scores = _score_forecast(forecast, persistence, target, common)
    comparison = None
    if reference_settings is not None:
        reference_result, reference_observation, reference_background, _ = _run_case(
            reference_settings,
        )
        reference_valid = reference_result.valid_mask[lead_index].detach().cpu()
        reference_forecast = reference_result.forecast_dbz[lead_index].detach().cpu()
        reference_forecast = reference_forecast.masked_fill(~reference_valid, torch.nan)
        reference_persistence = _persistence(
            reference_observation, reference_background,
        )
        comparison = _compare_forecasts(
            reference_forecast, forecast, reference_persistence, target,
        )
        comparison["settings"] = reference_settings.__dict__

    valid_fraction = float(result.valid_mask[lead_index].float().mean())
    background_coverage = float(torch.isfinite(latest_background).float().mean())
    confidence = result.forecast_confidence[lead_index].detach().cpu()
    confidence_on_valid = confidence[result.valid_mask[lead_index].detach().cpu()]

    metadata = result.metadata
    return {
        "case": {
            "name": "이동하는 이중 강수셀",
            "description": "관측 결측영역을 고정하고 초기장만 바꾸는 48×48 합성 사례",
            "lead_minutes": settings.lead_minutes,
            "observation_coverage": round(
                float(torch.isfinite(latest_observation).float().mean()),
                4,
            ),
            "background_coverage": round(background_coverage, 4),
        },
        "settings": settings.__dict__,
        "comparison": comparison,
        "fields": {
            "observation": _json_field(latest_observation),
            "background": _json_field(latest_background),
            "forecast": _json_field(forecast),
            "truth": _json_field(target),
        },
        "metrics": {
            **scores,
            "valid_fraction": round(valid_fraction, 4),
            "mean_confidence": _finite_mean(confidence_on_valid),
            "background_contribution_fraction": round(
                metadata.background_contribution_fraction,
                4,
            ),
        },
        "state": {
            "displacement_yx": [
                round(float(value), 3)
                for value in result.state.displacement_yx.detach().cpu()
            ],
            "log_growth_per_step": round(
                float(result.state.log_growth_per_step),
                5,
            ),
            "data_status": metadata.data_status.value,
            "motion_pair_count": metadata.motion_pair_count,
            "growth_pair_count": metadata.growth_pair_count,
            "background_used": metadata.background_used,
        },
        "process": [
            {
                "name": "고정 관측",
                "detail": f"최신장 관측률 {float(torch.isfinite(latest_observation).float().mean()):.1%}",
            },
            {
                "name": "초기장 보완",
                "detail": (
                    f"배경 기여 {metadata.background_contribution_fraction:.1%}"
                    if metadata.background_used
                    else "배경 미사용"
                ),
            },
            {
                "name": "운동·성장",
                "detail": (
                    f"motion {metadata.motion_pair_count}쌍 / growth {metadata.growth_pair_count}쌍"
                ),
            },
            {
                "name": "직접 예측",
                "detail": f"+{settings.lead_minutes}분, 유효영역 {valid_fraction:.1%}",
            },
            {
                "name": "공통영역 검증",
                "detail": (
                    f"동일 입력 persistence 비교 · {scores['scored_pixels']}화소"
                ),
            },
        ],
    }


def _round_or_none(value: float) -> float | None:
    return None if not math.isfinite(value) else round(value, 4)


class LabHandler(BaseHTTPRequestHandler):
    _assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    }

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/default":
            self._send_json(run_experiment(ExperimentSettings()))
            return
        asset = self._assets.get(path)
        if asset is None:
            self.send_error(404)
            return
        filename, content_type = asset
        data = (ASSET_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAXIMUM_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            settings = ExperimentSettings.from_payload(payload)
            reference_payload = payload.get("reference")
            reference = (
                None if reference_payload is None
                else ExperimentSettings.from_payload(reference_payload)
            )
            self._send_json(run_experiment(settings, reference))
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json({"error": str(error)}, status=400)

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the default experiment without starting the server",
    )
    arguments = parser.parse_args()
    if arguments.check:
        result = run_experiment(ExperimentSettings())
        print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
        return 0
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")

    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), LabHandler)
    print(f"ADVAR initial-field lab: http://127.0.0.1:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
