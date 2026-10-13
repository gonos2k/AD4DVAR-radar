"""Replay cached SciPy Powell calls, then optionally probe its first unseen point."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
SEARCH = WORK / "attempt1"
LOG = SEARCH / "evaluations.jsonl"
REPLAY = WORK / "zero_powell_replay.json"
FAILURE = WORK / "zero_powell_support_failure.json"
PLAN = WORK / "PLAN.json"
sys.path.insert(0, str(ROOT))


class FirstUnseen(Exception):
    def __init__(self, vector: np.ndarray, callback_index: int, seen: int) -> None:
        self.vector = np.asarray(vector, dtype=np.float64).copy()
        self.callback_index = callback_index
        self.seen = seen


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def replay() -> dict[str, Any]:
    from scipy.optimize import minimize
    import scipy

    rows = [json.loads(line) for line in LOG.read_text().splitlines()]
    zero_rows = [row for row in rows if row["start"] == "zero"]
    cache: dict[bytes, float] = {}
    for row in zero_rows:
        key = np.ascontiguousarray(row["control"], dtype=np.float64).tobytes()
        value = float(row["J"])
        if key in cache and cache[key] != value:
            raise ValueError("duplicate cached control has nondeterministic J")
        cache[key] = value
    first = next(row for row in zero_rows if row["kind"] == "initial_native_J")
    x0 = np.asarray(first["control"], dtype=np.float64)
    start_j = float(first["J"])
    radius = float(np.sqrt(2.0 * start_j))
    callback_index = 0

    def cached_objective(vector: np.ndarray) -> float:
        nonlocal callback_index
        callback_index += 1
        key = np.ascontiguousarray(vector, dtype=np.float64).tobytes()
        if key not in cache:
            raise FirstUnseen(np.asarray(vector), callback_index, len(cache))
        return cache[key]

    try:
        minimize(cached_objective, x0, method="Powell", bounds=[(-radius, radius)] * len(x0),
            options={"maxfev": 1000, "maxiter": 100, "xtol": 1e-11, "ftol": 1e-13, "disp": False})
    except FirstUnseen as unseen:
        result = {"status": "first_unseen_reproduced", "plan_sha256": (WORK / "PLAN_SHA256.txt").read_text().strip(),
            "sciPy_version": scipy.__version__, "python_version": sys.version.split()[0],
            "replay_cache_log_sha256": _sha(LOG), "cached_zero_rows": len(zero_rows),
            "distinct_cached_vectors": len(cache), "initial_J": start_j,
            "box_radius": radius, "powell_options": {"maxfev": 1000,"maxiter": 100,"xtol": 1e-11,"ftol": 1e-13},
            "first_unseen_powell_callback": unseen.callback_index,
            "cached_distinct_vectors_consumed": unseen.seen,
            "failed_control": unseen.vector.tolist(),
            "failed_control_sha256": _tensor_sha(unseen.vector),
            "failed_control_inf_norm": float(np.max(np.abs(unseen.vector))),
            "failed_control_l2_norm": float(np.linalg.norm(unseen.vector)),
            "failed_control_within_component_box": bool(np.all(np.abs(unseen.vector) <= radius)),
            "failed_control_identity_prior": float(0.5 * np.dot(unseen.vector, unseen.vector)),
            "new_FV_evaluations_during_replay": 0}
        REPLAY.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return result
    raise ValueError("cached Powell replay completed without reaching the first unseen point")


def support_probe(control_list: list[float]) -> dict[str, Any]:
    import torch
    from examples.weather_scenarios import fv_point_3h_netflux_future_observation as obs
    from advar import transport
    from advar.transport import _volume_parts

    case = obs._load_case()
    problem, _, parameters, truth = obs._fixed_case_tensors(case, {
        "parameters_sha256": obs.PARAMETERS_SHA,
        "truth_sha256": obs.TRUTH_SHA})
    control = torch.as_tensor(control_list, dtype=torch.float64)
    if (control.shape != (26,) or not bool(torch.isfinite(control).all())
            or _tensor_sha(control.numpy()) != json.loads(REPLAY.read_text())["failed_control_sha256"]):
        raise ValueError("replayed Powell failure control identity mismatch")
    checks: list[dict[str, Any]] = []
    original_check = transport._check_minmod_support

    def inspect_support(support, qx, qy, support_edges):
        eps = 16 * torch.finfo(support.dtype).eps
        missed = support < 1.0 - eps
        edge_entries = []
        qxp, qxm = _volume_parts(qx, axis="x")
        qyp, qym = _volume_parts(qy, axis="y")
        incoming = (qxp[:, 0] > 0, -qxm[:, -1] > 0,
            qyp[0, :] > 0, -qym[-1, :] > 0)
        for stage_i, stage in enumerate(support_edges):
            for side_i, (edge, mask) in enumerate(zip(stage, incoming)):
                edge_entries.append({"stage": stage_i, "side_index": side_i,
                    "edge_shape": list(edge.shape), "edge_min": float(edge.min()),
                    "edge_max": float(edge.max()), "incoming_faces": int(mask.sum()),
                    "incoming_unknown_faces": int((mask & (edge < 1.0 - eps)).sum())})
        record = {"check_index": len(checks) + 1,"support_shape": list(support.shape),
            "support_min": float(support.min()),"support_max": float(support.max()),
            "support_unknown_cells": int(missed.sum()),
            "unknown_cell_indices": torch.nonzero(missed, as_tuple=False).tolist(),
            "qx_shape": list(qx.shape),"qy_shape": list(qy.shape),
            "qx_min": float(qx.min()),"qx_max": float(qx.max()),
            "qy_min": float(qy.min()),"qy_max": float(qy.max()),
            "boundary_support": edge_entries}
        checks.append(record)
        try:
            original_check(support, qx, qy, support_edges)
        except ValueError as error:
            record["validator_error"] = str(error)
            record["check_failed"] = True
            raise

    transport._check_minmod_support = inspect_support
    try:
        with torch.no_grad():
            value = problem.objective(control, parameters)
        result = {"status": "unexpectedly_succeeded", "J": float(value)}
    except ValueError as error:
        result = {"status": "native_support_refusal_reproduced", "error": str(error),
            "check_count": len(checks), "failing_check": checks[-1] if checks else None}
    finally:
        transport._check_minmod_support = original_check
    result.update({"plan_sha256": (WORK / "PLAN_SHA256.txt").read_text().strip(),
        "replay_receipt_sha256": _sha(REPLAY),
        "fixed_case_sha256": _sha(ROOT / "graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/fixed_case.pt"),
        "control_sha256": _tensor_sha(control.numpy()),
        "objective_forward_attempts": 1,
        "new_HVP_calls": 0,"new_E_calls": 0})
    FAILURE.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-support", action="store_true")
    args = parser.parse_args()
    if args.probe_support:
        record = json.loads(REPLAY.read_text())
        print(json.dumps(support_probe(record["failed_control"]), sort_keys=True))
    else:
        print(json.dumps(replay(), sort_keys=True))


if __name__ == "__main__":
    main()
