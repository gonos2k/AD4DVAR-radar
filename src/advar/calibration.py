"""Immutable provenance for an operational hindcast calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

from ._digest import json_digest


OPERATIONAL_CALIBRATION_MANIFEST_VERSION = (
    "operational-calibration-manifest-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_MANIFEST_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CalibrationMetric:
    name: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("calibration metric names must be nonempty")
        if self.name.strip() != self.name:
            raise ValueError("calibration metric names must be canonical")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise ValueError("calibration metric values must be finite")


@dataclass(frozen=True)
class OperationalCalibrationManifest:
    calibration_id: str
    expected_profile_digest: str
    radar_class: str
    training_period: tuple[str, str]
    validation_period: tuple[str, str]
    validation_metrics: tuple[CalibrationMetric, ...]
    schema_version: str = OPERATIONAL_CALIBRATION_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATIONAL_CALIBRATION_MANIFEST_VERSION:
            raise ValueError("unsupported calibration manifest schema")
        for name, value in (
            ("calibration_id", self.calibration_id),
            ("radar_class", self.radar_class),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
            if value.strip() != value:
                raise ValueError(f"{name} must be canonical")
        if (
            not isinstance(self.expected_profile_digest, str)
            or _SHA256.fullmatch(self.expected_profile_digest) is None
        ):
            raise ValueError("expected_profile_digest must be lowercase SHA-256")

        training = _canonical_period("training_period", self.training_period)
        validation = _canonical_period(
            "validation_period",
            self.validation_period,
        )
        object.__setattr__(self, "training_period", training)
        object.__setattr__(self, "validation_period", validation)
        if _parse_time(training[1]) > _parse_time(validation[0]):
            raise ValueError("training and validation periods cannot overlap")

        if (
            not isinstance(self.validation_metrics, tuple)
            or not self.validation_metrics
            or not all(
                isinstance(metric, CalibrationMetric)
                for metric in self.validation_metrics
            )
        ):
            raise ValueError("validation_metrics must be nonempty")
        names = tuple(metric.name for metric in self.validation_metrics)
        if len(set(names)) != len(names):
            raise ValueError("validation metric names must be unique")
        if names != tuple(sorted(names)):
            raise ValueError("validation metrics must use canonical name order")

    @property
    def value(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "expected_profile_digest": self.expected_profile_digest,
            "radar_class": self.radar_class,
            "training_period": list(self.training_period),
            "validation_period": list(self.validation_period),
            "validation_metrics": [
                {"name": metric.name, "value": metric.value}
                for metric in self.validation_metrics
            ],
        }

    @property
    def json(self) -> str:
        return json.dumps(
            self.value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def digest(self) -> str:
        return json_digest(self.value)

    @classmethod
    def from_json(cls, text: str) -> OperationalCalibrationManifest:
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("invalid calibration manifest JSON") from error
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "calibration_id",
            "expected_profile_digest",
            "radar_class",
            "training_period",
            "validation_period",
            "validation_metrics",
        }:
            raise ValueError("invalid calibration manifest fields")
        metrics = value["validation_metrics"]
        if not isinstance(metrics, list):
            raise ValueError("validation_metrics must be a list")
        parsed_metrics: list[CalibrationMetric] = []
        for metric in metrics:
            if not isinstance(metric, dict) or set(metric) != {"name", "value"}:
                raise ValueError("invalid calibration metric")
            name = metric["name"]
            raw_value = metric["value"]
            if (
                not isinstance(name, str)
                or isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
            ):
                raise ValueError("invalid calibration metric")
            parsed_metrics.append(CalibrationMetric(name, float(raw_value)))
        return cls(
            schema_version=_required_string(
                "schema_version",
                value["schema_version"],
            ),
            calibration_id=_required_string(
                "calibration_id",
                value["calibration_id"],
            ),
            expected_profile_digest=_required_string(
                "expected_profile_digest",
                value["expected_profile_digest"],
            ),
            radar_class=_required_string("radar_class", value["radar_class"]),
            training_period=_string_pair(
                "training_period",
                value["training_period"],
            ),
            validation_period=_string_pair(
                "validation_period",
                value["validation_period"],
            ),
            validation_metrics=tuple(parsed_metrics),
        )

    @classmethod
    def load(cls, path: Path) -> OperationalCalibrationManifest:
        if not path.is_file():
            raise ValueError("calibration manifest must be a regular file")
        if path.stat().st_size > _MAXIMUM_MANIFEST_BYTES:
            raise ValueError("calibration manifest exceeds the size limit")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError("cannot read calibration manifest") from error
        manifest = cls.from_json(text)
        if text != manifest.json:
            raise ValueError("calibration manifest JSON must be canonical")
        return manifest


def _string_pair(name: str, value: object) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{name} must contain two timestamps")
    return value[0], value[1]


def _required_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _canonical_period(
    name: str,
    value: tuple[str, str],
) -> tuple[str, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must contain two timestamps")
    start, end = (_parse_time(item) for item in value)
    if start >= end:
        raise ValueError(f"{name} must have positive duration")
    return start.isoformat(), end.isoformat()


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("calibration timestamps must be nonempty strings")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("calibration timestamps must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("calibration timestamps must include timezone offsets")
    return parsed.astimezone(timezone.utc)
