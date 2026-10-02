"""Audit two interval normal derivatives at one approximate restricted root."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mpmath import mp
from examples.weather_scenarios import fv_active_face_normal_reference as reference

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "graphify-out/fv-root-cause-20260919"
INPUT = EVIDENCE / "R4_ACTIVE_FACE_NORMAL_INPUT_20261003.json"
INPUT_SHA = "5694b56e958058adb4004291ea03dc4b5c85841a5acd1627603f1ac0d016174e"
PLAN = EVIDENCE / "R4_ACTIVE_FACE_NORMAL_PLAN_20261003.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_trace(native: list[dict[str, Any]], side: int) -> list[dict[str, Any]]:
    result = []
    for row in native[:36]:
        item: dict[str, Any] = {"step": row["step"], "stage": row["stage"]}
        for axis in ("x", "y"):
            signs = [value for line in row[axis]["slope_sign"] for value in line]
            left = [value for line in row[axis]["choose_left"] for value in line]
            item[axis] = [0 if sign == 0 else (1 if choose_left else 2)
                          for sign, choose_left in zip(signs, left, strict=True)]
        item["qx_sign"] = [value for line in row["qx_sign"] for value in line]
        qy = [value for line in row["qy_sign"] for value in line]
        if qy[10] != 0:
            raise ValueError("native selected face was not zero")
        qy[10] = side
        item["qy_sign"] = qy
        result.append(item)
    return result


def main(output: Path) -> None:
    if output.exists():
        raise ValueError("normal output must be fresh")
    if sha(INPUT) != INPUT_SHA:
        raise ValueError("captured approximate root changed")
    plan_digest = sha(PLAN)
    code = {str(Path(path).relative_to(ROOT)): sha(Path(path))
            for path in (__file__, reference.__file__)}
    data = json.loads(INPUT.read_text())
    stationarity = EVIDENCE / "partial_active_face_stationarity_attempt1/stationarity.json"
    if sha(stationarity) != data["stationarity_sha256"]:
        raise ValueError("source stationarity evidence changed")
    sides = [reference.evaluate_side(data["fixture"], data["tangent"], side, dps=80)
             for side in (-1, 1)]
    for row in sides:
        if row["branch_choices"] != expected_trace(data["trace"], row["side"]):
            raise ValueError("normal extension changed another face/minmod branch")
    with mp.workdps(100):
        intervals = [[mp.make_mpf(tuple(bound)) for bound in row["sigma_eta_interval_binary"]]
                     for row in sides]
        slopes = [{"side": row["side"], "lower": str(bounds[0]), "upper": str(bounds[1])}
                  for row, bounds in zip(sides, intervals, strict=True)]
        oriented = bool(intervals[0][1] < 0 < intervals[1][0])
    if (sha(INPUT) != INPUT_SHA or sha(PLAN) != plan_digest
            or sha(stationarity) != data["stationarity_sha256"]
            or any(sha(ROOT / path) != digest for path, digest in code.items())):
        raise ValueError("normal reference sources/inputs changed")
    output.write_text(json.dumps({
        "scope": "two one-sided J derivatives at the approximate restricted root; no exact root/neighborhood/minimum/response proof",
        "input_sha256": INPUT_SHA, "plan_sha256": plan_digest,
        "source_sha256": code, "sides": sides, "normal_intervals": slopes,
        "strict_opposite_orientation_at_approximate_point": oriented,
        "branch_audit": "36 analysis stages; 48 other-face signs plus selected side sign and 12 minmod choices per stage",
        "full_root_claim": False, "normal_optimizer_certificate": "not_performed",
        "response_validation": "not_performed",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
