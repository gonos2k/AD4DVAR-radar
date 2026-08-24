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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
) -> None:
    proj_path = Path(proj).resolve()
    projinfo_path = Path(projinfo).resolve()
    if (
        proj_path.parent != projinfo_path.parent
        or database_path.parents[2] != proj_path.parent.parent
        or engine_version != database_version
    ):
        raise RuntimeError(
            "PROJ binaries and database do not form one exact toolchain"
        )


def _lattice_values(
    minimum: Decimal,
    maximum: Decimal,
    count: int,
) -> tuple[Decimal, ...]:
    if count < 2:
        raise ValueError("sample count per axis must be at least two")
    step = (maximum - minimum) / Decimal(count - 1)
    return tuple(minimum + step * index for index in range(count))


def _generate_report(
    *,
    proj: str,
    projinfo: str,
    sample_count_per_axis: int,
) -> dict[str, Any]:
    database, database_path = _proj_database_metadata(projinfo)
    engine_version = _proj_version(proj)
    _validate_toolchain_identity(
        proj=proj,
        projinfo=projinfo,
        database_path=database_path,
        engine_version=engine_version,
        database_version=database["PROJ.VERSION"],
    )
    projection_definition, projjson_digest = _epsg_projection_definition(
        projinfo
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
        },
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
    arguments = parser.parse_args()
    report = _generate_report(
        proj=arguments.proj,
        projinfo=arguments.projinfo,
        sample_count_per_axis=arguments.sample_count_per_axis,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_bytes(report) + b"\n")
    print(hashlib.sha256(arguments.output.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
