#!/usr/bin/env python3
"""Generate the committed EPSG:5179 metric-domain evidence report.

This script is intentionally outside the runtime package.  It uses the PROJ
command-line tools to sample the registered EPSG area-of-use lattice, records
the engine/database identity, and writes one deterministic JSON report that can
be verified at runtime without installing PROJ.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from typing import Any


PROJECTION = "EPSG:5179"
MINIMUM_LONGITUDE_DEG = Decimal("122.71")
MAXIMUM_LONGITUDE_DEG = Decimal("134.28")
MINIMUM_LATITUDE_DEG = Decimal("28.60")
MAXIMUM_LATITUDE_DEG = Decimal("40.27")
REGISTERED_MAXIMUM_LINEAR_SCALE_ERROR = Decimal("0.006")
REGISTERED_MAXIMUM_AREA_SCALE_ERROR = Decimal("0.012036")
COVERAGE_BOUNDARY_SAMPLE_COUNT_PER_EDGE = 10_001
GENERATOR_CONTRACT = "generate-metric-domain-evidence-v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _check_generator_source_binding(output: Path) -> str:
    """Verify the committed report names these exact generator bytes."""

    if not output.is_file():
        raise RuntimeError("metric-domain evidence output does not exist")
    report_bytes = output.read_bytes()
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("metric-domain evidence output is invalid") from error
    generator = report.get("generator") if isinstance(report, dict) else None
    if (
        not isinstance(generator, dict)
        or generator.get("contract") != GENERATOR_CONTRACT
        or generator.get("source_sha256")
        != _file_sha256(Path(__file__).resolve())
        or report_bytes != _canonical_bytes(report) + b"\n"
    ):
        raise RuntimeError(
            "metric-domain evidence generator source binding is invalid"
        )
    return hashlib.sha256(report_bytes).hexdigest()


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    # Foreign data paths can silently pair a binary with an incompatible DB.
    environment.pop("PROJ_LIB", None)
    environment.pop("PROJ_DATA", None)
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=environment,
        input=input_text,
        text=True,
    )


def _proj_version(proj: str) -> str:
    environment = dict(os.environ)
    environment.pop("PROJ_LIB", None)
    environment.pop("PROJ_DATA", None)
    completed = subprocess.run(
        [proj],
        capture_output=True,
        env=environment,
        text=True,
    )
    match = re.search(r"Rel\.\s+([0-9.]+)", completed.stderr)
    if match is None:
        raise RuntimeError("could not determine the PROJ engine version")
    return match.group(1)


def _proj_database_metadata(projinfo: str) -> tuple[dict[str, str], Path]:
    completed = _run([projinfo, "--searchpaths"])
    database_path = next(
        (
            Path(line.strip()) / "proj.db"
            for line in completed.stdout.splitlines()
            if line.strip() and (Path(line.strip()) / "proj.db").is_file()
        ),
        None,
    )
    if database_path is None:
        raise RuntimeError("could not locate the PROJ database")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT key, value FROM metadata "
            "WHERE key IN ("
            "'PROJ.VERSION','EPSG.VERSION','EPSG.DATE','PROJ_DATA.VERSION',"
            "'DATABASE.LAYOUT.VERSION.MAJOR','DATABASE.LAYOUT.VERSION.MINOR'"
            ")"
        ).fetchall()
    metadata = {str(key): str(value) for key, value in rows}
    required = {
        "PROJ.VERSION",
        "EPSG.VERSION",
        "EPSG.DATE",
        "PROJ_DATA.VERSION",
        "DATABASE.LAYOUT.VERSION.MAJOR",
        "DATABASE.LAYOUT.VERSION.MINOR",
    }
    if set(metadata) != required:
        raise RuntimeError("PROJ database metadata is incomplete")
    return metadata, database_path.resolve()


def _epsg_projection_definition(
    projinfo: str,
) -> tuple[str, str]:
    """Resolve the factor projection and canonical CRS from one EPSG DB."""

    proj_output = _run(
        [projinfo, PROJECTION, "-o", "PROJ", "--single-line"]
    ).stdout
    definition = next(
        (
            line.strip()
            for line in proj_output.splitlines()
            if line.startswith("+proj=")
        ),
        None,
    )
    if definition is None:
        raise RuntimeError("could not resolve the EPSG projection definition")
    factor_tokens = [
        token for token in definition.split() if token != "+type=crs"
    ]
    factor_definition = " ".join(factor_tokens)

    projjson_output = _run(
        [projinfo, PROJECTION, "-o", "PROJJSON"]
    ).stdout
    marker = "PROJJSON:\n"
    if marker not in projjson_output:
        raise RuntimeError("could not resolve the EPSG PROJJSON definition")
    projjson = json.loads(projjson_output.split(marker, 1)[1])
    if not isinstance(projjson, dict):
        raise RuntimeError("EPSG PROJJSON definition is invalid")
    identifier = projjson.get("id")
    bbox = projjson.get("bbox")
    if (
        not isinstance(identifier, dict)
        or identifier.get("authority") != "EPSG"
        or identifier.get("code") != 5179
        or not isinstance(bbox, dict)
        or Decimal(str(bbox.get("west_longitude")))
        != MINIMUM_LONGITUDE_DEG
        or Decimal(str(bbox.get("east_longitude")))
        != MAXIMUM_LONGITUDE_DEG
        or Decimal(str(bbox.get("south_latitude")))
        != MINIMUM_LATITUDE_DEG
        or Decimal(str(bbox.get("north_latitude")))
        != MAXIMUM_LATITUDE_DEG
    ):
        raise RuntimeError("EPSG:5179 authority or area of use is unexpected")
    return factor_definition, _digest(projjson)


def _validate_toolchain_identity(
    *,
    proj: str,
    projinfo: str,
    database_path: Path,
    engine_version: str,
    database_version: str,
) -> dict[str, str]:
    proj_path = Path(proj).resolve()
    projinfo_path = Path(projinfo).resolve()
    if (
        proj_path.parent != projinfo_path.parent
        or database_path.parents[2] != proj_path.parent.parent
        or not proj_path.is_file()
        or not projinfo_path.is_file()
        or not os.access(proj_path, os.X_OK)
        or not os.access(projinfo_path, os.X_OK)
        or not database_path.is_file()
        or engine_version != database_version
    ):
        raise RuntimeError(
            "PROJ binaries and database do not form one exact toolchain"
        )
    return {
        "proj_binary_sha256": _file_sha256(proj_path),
        "projinfo_binary_sha256": _file_sha256(projinfo_path),
        "proj_database_sha256": _file_sha256(database_path),
    }


def _lattice_values(
    minimum: Decimal,
    maximum: Decimal,
    count: int,
) -> tuple[Decimal, ...]:
    if count < 2:
        raise ValueError("sample count per axis must be at least two")
    step = (maximum - minimum) / Decimal(count - 1)
    return tuple(minimum + step * index for index in range(count))


def _project_points(
    *,
    proj: str,
    projection_definition: str,
    points: tuple[tuple[Decimal, Decimal], ...],
    inverse: bool = False,
) -> tuple[tuple[float, float], ...]:
    input_text = "".join(
        f"{first:.12f} {second:.12f}\n" for first, second in points
    )
    command = [proj]
    if inverse:
        command.append("-I")
    command.extend(
        ["-f", "%.12f", *projection_definition.split()]
    )
    completed = _run(command, input_text=input_text)
    rows = completed.stdout.splitlines()
    if len(rows) != len(points):
        raise RuntimeError("PROJ returned the wrong number of coordinate rows")
    result: list[tuple[float, float]] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 2:
            raise RuntimeError("PROJ returned an invalid coordinate row")
        first, second = float(fields[0]), float(fields[1])
        if not math.isfinite(first) or not math.isfinite(second):
            raise RuntimeError("PROJ returned a non-finite coordinate")
        result.append((first, second))
    return tuple(result)


def _validated_projected_coverage(
    *,
    proj: str,
    projection_definition: str,
) -> dict[str, Any]:
    """Derive an inward-rounded projected bbox inside the EPSG area of use."""

    longitudes = _lattice_values(
        MINIMUM_LONGITUDE_DEG,
        MAXIMUM_LONGITUDE_DEG,
        COVERAGE_BOUNDARY_SAMPLE_COUNT_PER_EDGE,
    )
    latitudes = _lattice_values(
        MINIMUM_LATITUDE_DEG,
        MAXIMUM_LATITUDE_DEG,
        COVERAGE_BOUNDARY_SAMPLE_COUNT_PER_EDGE,
    )
    bottom = tuple((longitude, MINIMUM_LATITUDE_DEG) for longitude in longitudes)
    top = tuple((longitude, MAXIMUM_LATITUDE_DEG) for longitude in longitudes)
    left = tuple((MINIMUM_LONGITUDE_DEG, latitude) for latitude in latitudes)
    right = tuple((MAXIMUM_LONGITUDE_DEG, latitude) for latitude in latitudes)
    geographic_boundary = bottom + top + left + right
    projected_boundary = _project_points(
        proj=proj,
        projection_definition=projection_definition,
        points=geographic_boundary,
    )
    edge_count = COVERAGE_BOUNDARY_SAMPLE_COUNT_PER_EDGE
    projected_bottom = projected_boundary[:edge_count]
    projected_top = projected_boundary[edge_count : 2 * edge_count]
    projected_left = projected_boundary[2 * edge_count : 3 * edge_count]
    projected_right = projected_boundary[3 * edge_count :]
    minimum_easting_m = float(
        math.ceil(max(easting for easting, _ in projected_left))
    )
    maximum_easting_m = float(
        math.floor(min(easting for easting, _ in projected_right))
    )
    minimum_northing_m = float(
        math.ceil(max(northing for _, northing in projected_bottom))
    )
    maximum_northing_m = float(
        math.floor(min(northing for _, northing in projected_top))
    )
    if (
        minimum_easting_m >= maximum_easting_m
        or minimum_northing_m >= maximum_northing_m
    ):
        raise RuntimeError("projected evidence coverage is empty")

    coverage_eastings = _lattice_values(
        Decimal(str(minimum_easting_m)),
        Decimal(str(maximum_easting_m)),
        edge_count,
    )
    coverage_northings = _lattice_values(
        Decimal(str(minimum_northing_m)),
        Decimal(str(maximum_northing_m)),
        edge_count,
    )
    coverage_boundary = (
        tuple(
            (easting, Decimal(str(minimum_northing_m)))
            for easting in coverage_eastings
        )
        + tuple(
            (easting, Decimal(str(maximum_northing_m)))
            for easting in coverage_eastings
        )
        + tuple(
            (Decimal(str(minimum_easting_m)), northing)
            for northing in coverage_northings
        )
        + tuple(
            (Decimal(str(maximum_easting_m)), northing)
            for northing in coverage_northings
        )
    )
    inverse_boundary = _project_points(
        proj=proj,
        projection_definition=projection_definition,
        points=coverage_boundary,
        inverse=True,
    )
    inverse_longitudes = tuple(value[0] for value in inverse_boundary)
    inverse_latitudes = tuple(value[1] for value in inverse_boundary)
    if (
        min(inverse_longitudes) < float(MINIMUM_LONGITUDE_DEG)
        or max(inverse_longitudes) > float(MAXIMUM_LONGITUDE_DEG)
        or min(inverse_latitudes) < float(MINIMUM_LATITUDE_DEG)
        or max(inverse_latitudes) > float(MAXIMUM_LATITUDE_DEG)
    ):
        raise RuntimeError(
            "projected evidence coverage escapes the EPSG area of use"
        )
    return {
        "contract": "epsg5179-sampled-inscribed-projected-bbox-v1",
        "boundary_sample_count_per_edge": edge_count,
        "inward_rounding": "integer-metre-ceil-min-floor-max-v1",
        "minimum_easting_m": minimum_easting_m,
        "maximum_easting_m": maximum_easting_m,
        "minimum_northing_m": minimum_northing_m,
        "maximum_northing_m": maximum_northing_m,
        "source_geographic_boundary_digest": _digest(
            [
                {"longitude_deg": float(longitude), "latitude_deg": float(latitude)}
                for longitude, latitude in geographic_boundary
            ]
        ),
        "projected_source_boundary_digest": _digest(
            [
                {"projected_easting_m": easting, "projected_northing_m": northing}
                for easting, northing in projected_boundary
            ]
        ),
        "inverse_coverage_boundary_digest": _digest(
            [
                {"longitude_deg": longitude, "latitude_deg": latitude}
                for longitude, latitude in inverse_boundary
            ]
        ),
        "inverse_minimum_longitude_deg": min(inverse_longitudes),
        "inverse_maximum_longitude_deg": max(inverse_longitudes),
        "inverse_minimum_latitude_deg": min(inverse_latitudes),
        "inverse_maximum_latitude_deg": max(inverse_latitudes),
    }


def _generate_report(
    *,
    proj: str,
    projinfo: str,
    sample_count_per_axis: int,
) -> dict[str, Any]:
    database, database_path = _proj_database_metadata(projinfo)
    engine_version = _proj_version(proj)
    toolchain_hashes = _validate_toolchain_identity(
        proj=proj,
        projinfo=projinfo,
        database_path=database_path,
        engine_version=engine_version,
        database_version=database["PROJ.VERSION"],
    )
    projection_definition, projjson_digest = _epsg_projection_definition(
        projinfo
    )
    projected_coverage = _validated_projected_coverage(
        proj=proj,
        projection_definition=projection_definition,
    )
    longitudes = _lattice_values(
        MINIMUM_LONGITUDE_DEG,
        MAXIMUM_LONGITUDE_DEG,
        sample_count_per_axis,
    )
    latitudes = _lattice_values(
        MINIMUM_LATITUDE_DEG,
        MAXIMUM_LATITUDE_DEG,
        sample_count_per_axis,
    )
    points = tuple(
        (longitude, latitude)
        for latitude in latitudes
        for longitude in longitudes
    )
    input_text = "".join(
        f"{longitude:.12f} {latitude:.12f}\n"
        for longitude, latitude in points
    )
    completed = _run(
        [proj, "-f", "%.12f", "-S", *projection_definition.split()],
        input_text=input_text,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != len(points):
        raise RuntimeError("PROJ returned the wrong number of factor rows")

    samples: list[dict[str, float]] = []
    factor_pattern = re.compile(
        r"^([^\t]+)\t([^\t]+)\t<([^>]+)>$"
    )
    for (longitude, latitude), line in zip(points, lines, strict=True):
        match = factor_pattern.fullmatch(line)
        if match is None:
            raise RuntimeError(f"unexpected PROJ factor row: {line!r}")
        factors = tuple(float(value) for value in match.group(3).split())
        if len(factors) != 6:
            raise RuntimeError("PROJ factor row has the wrong width")
        samples.append(
            {
                "longitude_deg": float(longitude),
                "latitude_deg": float(latitude),
                "projected_easting_m": float(match.group(1)),
                "projected_northing_m": float(match.group(2)),
                "meridional_scale": factors[0],
                "parallel_scale": factors[1],
                "areal_scale": factors[2],
            }
        )

    points_payload = [
        {
            "longitude_deg": sample["longitude_deg"],
            "latitude_deg": sample["latitude_deg"],
        }
        for sample in samples
    ]
    projected_points_payload = [
        {
            "projected_easting_m": sample["projected_easting_m"],
            "projected_northing_m": sample["projected_northing_m"],
        }
        for sample in samples
    ]
    meridional_scales = [sample["meridional_scale"] for sample in samples]
    parallel_scales = [sample["parallel_scale"] for sample in samples]
    areal_scales = [sample["areal_scale"] for sample in samples]
    maximum_linear_error = max(
        *(abs(value - 1.0) for value in meridional_scales),
        *(abs(value - 1.0) for value in parallel_scales),
    )
    maximum_area_error = max(abs(value - 1.0) for value in areal_scales)
    if maximum_linear_error > float(REGISTERED_MAXIMUM_LINEAR_SCALE_ERROR):
        raise RuntimeError("observed linear scale error exceeds its budget")
    if maximum_area_error > float(REGISTERED_MAXIMUM_AREA_SCALE_ERROR):
        raise RuntimeError("observed area scale error exceeds its budget")

    return {
        "contract": "radar-metric-domain-geodetic-report-v1",
        "generator": {
            "contract": GENERATOR_CONTRACT,
            "source_sha256": _file_sha256(Path(__file__).resolve()),
            "canonical_output": "sorted-compact-json-utf8-newline-v1",
        },
        "canonical_projection": PROJECTION,
        "projection_definition": projection_definition,
        "epsg_crs_projjson_digest": projjson_digest,
        "geodetic_engine": {
            "name": "PROJ",
            "version": engine_version,
            "proj_database_version": database["PROJ.VERSION"],
            "epsg_database_version": database["EPSG.VERSION"],
            "epsg_database_date": database["EPSG.DATE"],
            "proj_data_version": database["PROJ_DATA.VERSION"],
            "database_layout_version": (
                f"{database['DATABASE.LAYOUT.VERSION.MAJOR']}."
                f"{database['DATABASE.LAYOUT.VERSION.MINOR']}"
            ),
            **toolchain_hashes,
        },
        "validated_projected_coverage": projected_coverage,
        "sampling": {
            "contract": "epsg5179-area-of-use-geographic-lattice-v1",
            "minimum_longitude_deg": float(MINIMUM_LONGITUDE_DEG),
            "maximum_longitude_deg": float(MAXIMUM_LONGITUDE_DEG),
            "minimum_latitude_deg": float(MINIMUM_LATITUDE_DEG),
            "maximum_latitude_deg": float(MAXIMUM_LATITUDE_DEG),
            "longitude_count": sample_count_per_axis,
            "latitude_count": sample_count_per_axis,
            "point_order": "latitude-major-longitude-minor-v1",
            "sampled_geographic_points_digest": _digest(points_payload),
        },
        "factor_digests": {
            "sampled_projected_points_digest": _digest(
                projected_points_payload
            ),
            "meridional_scale_digest": _digest(meridional_scales),
            "parallel_scale_digest": _digest(parallel_scales),
            "areal_scale_digest": _digest(areal_scales),
        },
        "observed_maximum_linear_scale_error": maximum_linear_error,
        "observed_maximum_area_scale_error": maximum_area_error,
        "registered_maximum_linear_scale_error": float(
            REGISTERED_MAXIMUM_LINEAR_SCALE_ERROR
        ),
        "registered_maximum_area_scale_error": float(
            REGISTERED_MAXIMUM_AREA_SCALE_ERROR
        ),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/advar/data/epsg5179_metric_domain_evidence_v1.json"),
    )
    parser.add_argument("--proj", default=shutil.which("proj") or "proj")
    parser.add_argument(
        "--projinfo",
        default=shutil.which("projinfo") or "projinfo",
    )
    parser.add_argument("--sample-count-per-axis", type=int, default=17)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument(
        "--check",
        action="store_true",
        help="fail unless generated canonical bytes equal the output file",
    )
    validation.add_argument(
        "--check-source-only",
        action="store_true",
        help=(
            "verify the committed report names this exact generator source "
            "without requiring PROJ"
        ),
    )
    arguments = parser.parse_args()
    if arguments.check_source_only:
        print(_check_generator_source_binding(arguments.output))
        return 0
    report = _generate_report(
        proj=arguments.proj,
        projinfo=arguments.projinfo,
        sample_count_per_axis=arguments.sample_count_per_axis,
    )
    generated = _canonical_bytes(report) + b"\n"
    if arguments.check:
        if not arguments.output.is_file():
            raise RuntimeError("metric-domain evidence output does not exist")
        if arguments.output.read_bytes() != generated:
            raise RuntimeError("metric-domain evidence output is not reproducible")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(generated)
    print(hashlib.sha256(generated).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
