"""Immutable certification provenance for an operational calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from ._digest import json_digest


OPERATIONAL_CALIBRATION_MANIFEST_VERSION = (
    "operational-calibration-manifest-v2"
)
ALGORITHM_BUNDLE_VERSION = "advar-algorithm-bundle-v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_MANIFEST_BYTES = 1024 * 1024
_PROFILE_KINDS = frozenset(("p0", "p1"))
_METRIC_DIRECTIONS = frozenset(("maximize", "minimize"))


def algorithm_bundle_digest(package_root: Path | None = None) -> str:
    """Content-address every Python module shipped in the advar package."""

    root = Path(__file__).resolve().parent if package_root is None else package_root
    files = tuple(
        path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    if not files or not all(path.is_file() for path in files):
        raise ValueError("algorithm bundle must contain Python modules")
    return json_digest(
        {
            "version": ALGORITHM_BUNDLE_VERSION,
            "files": [
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            ],
        }
    )


@dataclass(frozen=True)
class CalibrationMetric:
    name: str
    definition_digest: str
    direction: str
    acceptance_threshold: float
    value: float

    def __post_init__(self) -> None:
        _canonical_string("calibration metric name", self.name)
        _sha256("calibration metric definition_digest", self.definition_digest)
        if self.direction not in _METRIC_DIRECTIONS:
            raise ValueError("calibration metric direction is invalid")
        for name, value in (
            ("acceptance_threshold", self.acceptance_threshold),
            ("value", self.value),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"calibration metric {name} must be finite")
        if self.direction == "maximize":
            accepted = self.value >= self.acceptance_threshold
        else:
            accepted = self.value <= self.acceptance_threshold
        if not accepted:
            raise ValueError("calibration metric does not meet its threshold")

    @property
    def contract_value(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "definition_digest": self.definition_digest,
            "direction": self.direction,
            "acceptance_threshold": self.acceptance_threshold,
        }

    @property
    def value_record(self) -> dict[str, Any]:
        return {**self.contract_value, "value": self.value}


@dataclass(frozen=True)
class CalibrationRegime:
    name: str
    case_count: int

    def __post_init__(self) -> None:
        _canonical_string("calibration regime name", self.name)
        if (
            isinstance(self.case_count, bool)
            or not isinstance(self.case_count, int)
            or self.case_count <= 0
        ):
            raise ValueError("calibration regime case_count must be positive")


@dataclass(frozen=True)
class OperationalDataIdentity:
    radar_class: str
    qc_pipeline_digest: str
    observation_error_model_digest: str
    background_model_digest: str
    radar_product_digest: str | None = None
    background_cycle_rule_digest: str | None = None
    mask_policy_digest: str | None = None
    radar_source_kind: str | None = None
    radar_site_digest: str | None = None
    radar_site_location_digest: str | None = None
    radar_source_contract_digest: str | None = None
    source_radar_index_map_digest: str | None = None
    effective_horizontal_range_map_digest: str | None = None
    source_selection_policy_digest: str | None = None

    def __post_init__(self) -> None:
        _canonical_string("radar_class", self.radar_class)
        for name, value in (
            ("qc_pipeline_digest", self.qc_pipeline_digest),
            (
                "observation_error_model_digest",
                self.observation_error_model_digest,
            ),
            ("background_model_digest", self.background_model_digest),
        ):
            _sha256(name, value)
        plan_values = (
            self.radar_product_digest,
            self.background_cycle_rule_digest,
            self.mask_policy_digest,
        )
        if any(value is not None for value in plan_values):
            if any(value is None for value in plan_values):
                raise ValueError("input-plan data identity must be complete")
            for name, value in (
                ("radar_product_digest", self.radar_product_digest),
                (
                    "background_cycle_rule_digest",
                    self.background_cycle_rule_digest,
                ),
                ("mask_policy_digest", self.mask_policy_digest),
            ):
                assert value is not None
                _sha256(name, value)
        source_values = (
            self.radar_source_kind,
            self.radar_site_digest,
            self.radar_site_location_digest,
            self.radar_source_contract_digest,
            self.source_radar_index_map_digest,
            self.effective_horizontal_range_map_digest,
            self.source_selection_policy_digest,
        )
        if any(value is not None for value in source_values):
            if self.radar_source_kind is None:
                inferred_kind = (
                    "single_site"
                    if self.radar_site_digest is not None
                    or self.radar_site_location_digest is not None
                    else "mosaic"
                )
                object.__setattr__(self, "radar_source_kind", inferred_kind)
            if self.radar_source_kind == "single_site":
                required = (
                    self.radar_site_digest,
                    self.radar_site_location_digest,
                    self.radar_source_contract_digest,
                )
                forbidden = (
                    self.source_radar_index_map_digest,
                    self.effective_horizontal_range_map_digest,
                    self.source_selection_policy_digest,
                )
            elif self.radar_source_kind == "mosaic":
                required = (
                    self.radar_source_contract_digest,
                    self.source_radar_index_map_digest,
                    self.effective_horizontal_range_map_digest,
                    self.source_selection_policy_digest,
                )
                forbidden = (
                    self.radar_site_digest,
                    self.radar_site_location_digest,
                )
            else:
                raise ValueError("radar source kind must be single_site or mosaic")
            if any(value is None for value in required) or any(
                value is not None for value in forbidden
            ):
                raise ValueError("radar source identity must be complete")
            for name, value in (
                ("radar_site_digest", self.radar_site_digest),
                ("radar_site_location_digest", self.radar_site_location_digest),
                ("radar_source_contract_digest", self.radar_source_contract_digest),
                ("source_radar_index_map_digest", self.source_radar_index_map_digest),
                (
                    "effective_horizontal_range_map_digest",
                    self.effective_horizontal_range_map_digest,
                ),
                ("source_selection_policy_digest", self.source_selection_policy_digest),
            ):
                if value is not None:
                    _sha256(name, value)

    @property
    def value(self) -> dict[str, str]:
        result = {
            "radar_class": self.radar_class,
            "qc_pipeline_digest": self.qc_pipeline_digest,
            "observation_error_model_digest": (
                self.observation_error_model_digest
            ),
            "background_model_digest": self.background_model_digest,
        }
        if self.radar_product_digest is not None:
            assert self.background_cycle_rule_digest is not None
            assert self.mask_policy_digest is not None
            result.update(
                {
                    "radar_product_digest": self.radar_product_digest,
                    "background_cycle_rule_digest": (
                        self.background_cycle_rule_digest
                    ),
                    "mask_policy_digest": self.mask_policy_digest,
                }
            )
        if self.radar_source_kind is not None:
            result["radar_source_kind"] = self.radar_source_kind
            assert self.radar_source_contract_digest is not None
            result["radar_source_contract_digest"] = (
                self.radar_source_contract_digest
            )
            if self.radar_source_kind == "single_site":
                assert self.radar_site_digest is not None
                assert self.radar_site_location_digest is not None
                result.update(
                    {
                        "radar_site_digest": self.radar_site_digest,
                        "radar_site_location_digest": (
                            self.radar_site_location_digest
                        ),
                    }
                )
            else:
                assert self.source_radar_index_map_digest is not None
                assert self.effective_horizontal_range_map_digest is not None
                assert self.source_selection_policy_digest is not None
                result.update(
                    {
                        "source_radar_index_map_digest": (
                            self.source_radar_index_map_digest
                        ),
                        "effective_horizontal_range_map_digest": (
                            self.effective_horizontal_range_map_digest
                        ),
                        "source_selection_policy_digest": (
                            self.source_selection_policy_digest
                        ),
                    }
                )
        return result

    @property
    def json(self) -> str:
        return json.dumps(
            self.value,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        return json_digest(self.value)

    @classmethod
    def from_json(cls, text: str) -> OperationalDataIdentity:
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("invalid operational data identity JSON") from error
        base_fields = {
            "radar_class",
            "qc_pipeline_digest",
            "observation_error_model_digest",
            "background_model_digest",
        }
        plan_fields = {
            "radar_product_digest",
            "background_cycle_rule_digest",
            "mask_policy_digest",
        }
        single_source_fields = {
            "radar_source_kind",
            "radar_site_digest",
            "radar_site_location_digest",
            "radar_source_contract_digest",
        }
        mosaic_source_fields = {
            "radar_source_kind",
            "radar_source_contract_digest",
            "source_radar_index_map_digest",
            "effective_horizontal_range_map_digest",
            "source_selection_policy_digest",
        }
        allowed = {
            frozenset(base_fields),
            frozenset(base_fields | plan_fields),
            frozenset(base_fields | single_source_fields),
            frozenset(base_fields | mosaic_source_fields),
            frozenset(base_fields | plan_fields | single_source_fields),
            frozenset(base_fields | plan_fields | mosaic_source_fields),
        }
        if not isinstance(value, dict) or frozenset(value) not in allowed:
            raise ValueError("invalid operational data identity fields")
        identity = cls(
            radar_class=_required_string("radar_class", value["radar_class"]),
            qc_pipeline_digest=_required_string(
                "qc_pipeline_digest", value["qc_pipeline_digest"]
            ),
            observation_error_model_digest=_required_string(
                "observation_error_model_digest",
                value["observation_error_model_digest"],
            ),
            background_model_digest=_required_string(
                "background_model_digest", value["background_model_digest"]
            ),
            radar_product_digest=(
                _required_string(
                    "radar_product_digest", value["radar_product_digest"]
                )
                if "radar_product_digest" in value
                else None
            ),
            background_cycle_rule_digest=(
                _required_string(
                    "background_cycle_rule_digest",
                    value["background_cycle_rule_digest"],
                )
                if "background_cycle_rule_digest" in value
                else None
            ),
            mask_policy_digest=(
                _required_string("mask_policy_digest", value["mask_policy_digest"])
                if "mask_policy_digest" in value
                else None
            ),
            radar_source_kind=(
                _required_string("radar_source_kind", value["radar_source_kind"])
                if "radar_source_kind" in value
                else None
            ),
            radar_site_digest=(
                _required_string("radar_site_digest", value["radar_site_digest"])
                if "radar_site_digest" in value
                else None
            ),
            radar_site_location_digest=(
                _required_string(
                    "radar_site_location_digest",
                    value["radar_site_location_digest"],
                )
                if "radar_site_location_digest" in value
                else None
            ),
            radar_source_contract_digest=(
                _required_string(
                    "radar_source_contract_digest",
                    value["radar_source_contract_digest"],
                )
                if "radar_source_contract_digest" in value
                else None
            ),
            source_radar_index_map_digest=(
                _required_string(
                    "source_radar_index_map_digest",
                    value["source_radar_index_map_digest"],
                )
                if "source_radar_index_map_digest" in value
                else None
            ),
            effective_horizontal_range_map_digest=(
                _required_string(
                    "effective_horizontal_range_map_digest",
                    value["effective_horizontal_range_map_digest"],
                )
                if "effective_horizontal_range_map_digest" in value
                else None
            ),
            source_selection_policy_digest=(
                _required_string(
                    "source_selection_policy_digest",
                    value["source_selection_policy_digest"],
                )
                if "source_selection_policy_digest" in value
                else None
            ),
        )
        if identity.json != text:
            raise ValueError("operational data identity JSON must be canonical")
        return identity


@dataclass(frozen=True)
class OperationalCalibrationManifest:
    calibration_id: str
    profile_kind: str
    expected_runtime_profile_digest: str
    expected_algorithm_bundle_digest: str
    calibration_dataset_digest: str
    validation_dataset_digest: str
    data_identity: OperationalDataIdentity
    training_period: tuple[str, str]
    validation_period: tuple[str, str]
    validation_case_count: int
    validation_regimes: tuple[CalibrationRegime, ...]
    validation_metrics: tuple[CalibrationMetric, ...]
    schema_version: str = OPERATIONAL_CALIBRATION_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPERATIONAL_CALIBRATION_MANIFEST_VERSION:
            raise ValueError("unsupported calibration manifest schema")
        _canonical_string("calibration_id", self.calibration_id)
        if not isinstance(self.data_identity, OperationalDataIdentity):
            raise ValueError("data_identity must be operational provenance")
        if self.profile_kind not in _PROFILE_KINDS:
            raise ValueError("profile_kind must be p0 or p1")
        for name, value in (
            (
                "expected_runtime_profile_digest",
                self.expected_runtime_profile_digest,
            ),
            (
                "expected_algorithm_bundle_digest",
                self.expected_algorithm_bundle_digest,
            ),
            ("calibration_dataset_digest", self.calibration_dataset_digest),
            ("validation_dataset_digest", self.validation_dataset_digest),
        ):
            _sha256(name, value)
        if self.calibration_dataset_digest == self.validation_dataset_digest:
            raise ValueError("calibration and validation datasets must differ")

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
            isinstance(self.validation_case_count, bool)
            or not isinstance(self.validation_case_count, int)
            or self.validation_case_count <= 0
        ):
            raise ValueError("validation_case_count must be positive")
        _validate_canonical_records(
            "validation regimes",
            self.validation_regimes,
            CalibrationRegime,
        )
        if sum(regime.case_count for regime in self.validation_regimes) != (
            self.validation_case_count
        ):
            raise ValueError(
                "validation regime counts must equal validation_case_count"
            )
        _validate_canonical_records(
            "validation metrics",
            self.validation_metrics,
            CalibrationMetric,
        )

    @property
    def metric_contract_digest(self) -> str:
        return json_digest(
            {
                "version": "calibration-metric-contract-v1",
                "metrics": [
                    metric.contract_value for metric in self.validation_metrics
                ],
            }
        )

    @property
    def value(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "profile_kind": self.profile_kind,
            "expected_runtime_profile_digest": (
                self.expected_runtime_profile_digest
            ),
            "expected_algorithm_bundle_digest": (
                self.expected_algorithm_bundle_digest
            ),
            "calibration_dataset_digest": self.calibration_dataset_digest,
            "validation_dataset_digest": self.validation_dataset_digest,
            "data_identity": self.data_identity.value,
            "training_period": list(self.training_period),
            "validation_period": list(self.validation_period),
            "validation_case_count": self.validation_case_count,
            "validation_regimes": [
                {"name": regime.name, "case_count": regime.case_count}
                for regime in self.validation_regimes
            ],
            "metric_contract_digest": self.metric_contract_digest,
            "validation_metrics": [
                metric.value_record for metric in self.validation_metrics
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
        required = {
            "schema_version",
            "calibration_id",
            "profile_kind",
            "expected_runtime_profile_digest",
            "expected_algorithm_bundle_digest",
            "calibration_dataset_digest",
            "validation_dataset_digest",
            "data_identity",
            "training_period",
            "validation_period",
            "validation_case_count",
            "validation_regimes",
            "metric_contract_digest",
            "validation_metrics",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("invalid calibration manifest fields")
        regimes = _parse_regimes(value["validation_regimes"])
        metrics = _parse_metrics(value["validation_metrics"])
        manifest = cls(
            schema_version=_required_string(
                "schema_version", value["schema_version"]
            ),
            calibration_id=_required_string(
                "calibration_id", value["calibration_id"]
            ),
            profile_kind=_required_string(
                "profile_kind", value["profile_kind"]
            ),
            expected_runtime_profile_digest=_required_string(
                "expected_runtime_profile_digest",
                value["expected_runtime_profile_digest"],
            ),
            expected_algorithm_bundle_digest=_required_string(
                "expected_algorithm_bundle_digest",
                value["expected_algorithm_bundle_digest"],
            ),
            calibration_dataset_digest=_required_string(
                "calibration_dataset_digest",
                value["calibration_dataset_digest"],
            ),
            validation_dataset_digest=_required_string(
                "validation_dataset_digest",
                value["validation_dataset_digest"],
            ),
            data_identity=_parse_data_identity(value["data_identity"]),
            training_period=_string_pair(
                "training_period", value["training_period"]
            ),
            validation_period=_string_pair(
                "validation_period", value["validation_period"]
            ),
            validation_case_count=_required_int(
                "validation_case_count", value["validation_case_count"]
            ),
            validation_regimes=regimes,
            validation_metrics=metrics,
        )
        metric_digest = _required_string(
            "metric_contract_digest", value["metric_contract_digest"]
        )
        if metric_digest != manifest.metric_contract_digest:
            raise ValueError("metric contract digest mismatch")
        return manifest

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


def _parse_regimes(value: object) -> tuple[CalibrationRegime, ...]:
    if not isinstance(value, list):
        raise ValueError("validation_regimes must be a list")
    parsed: list[CalibrationRegime] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "case_count"}:
            raise ValueError("invalid calibration regime")
        parsed.append(
            CalibrationRegime(
                _required_string("regime name", item["name"]),
                _required_int("regime case_count", item["case_count"]),
            )
        )
    return tuple(parsed)


def _parse_data_identity(value: object) -> OperationalDataIdentity:
    if not isinstance(value, dict):
        raise ValueError("data_identity must be an object")
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return OperationalDataIdentity.from_json(text)


def _parse_metrics(value: object) -> tuple[CalibrationMetric, ...]:
    if not isinstance(value, list):
        raise ValueError("validation_metrics must be a list")
    required = {
        "name",
        "definition_digest",
        "direction",
        "acceptance_threshold",
        "value",
    }
    parsed: list[CalibrationMetric] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("invalid calibration metric")
        parsed.append(
            CalibrationMetric(
                name=_required_string("metric name", item["name"]),
                definition_digest=_required_string(
                    "metric definition_digest", item["definition_digest"]
                ),
                direction=_required_string(
                    "metric direction", item["direction"]
                ),
                acceptance_threshold=_required_float(
                    "metric acceptance_threshold",
                    item["acceptance_threshold"],
                ),
                value=_required_float("metric value", item["value"]),
            )
        )
    return tuple(parsed)


def _validate_canonical_records(
    name: str,
    records: tuple[Any, ...],
    expected_type: type[Any],
) -> None:
    if (
        not isinstance(records, tuple)
        or not records
        or not all(isinstance(record, expected_type) for record in records)
    ):
        raise ValueError(f"{name} must be nonempty")
    names = tuple(record.name for record in records)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} names must be unique")
    if names != tuple(sorted(names)):
        raise ValueError(f"{name} must use canonical name order")


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


def _required_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _canonical_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty")
    if value.strip() != value:
        raise ValueError(f"{name} must be canonical")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
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
