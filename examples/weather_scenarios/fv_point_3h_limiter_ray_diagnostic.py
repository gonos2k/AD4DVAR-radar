"""Observe exact limiter choices and native derivatives on one stored ray.

This is a diagnostic, with no optimizer, accepted point, or minimum certificate.
Exact selector choices are separate from tolerance-based branch qualification.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import resource
import sys
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012"
OUT = HERE / "limiter_ray_20261014"
ENDPOINT = HERE / "two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json"
METRIC = ENDPOINT.with_name("native24_resume_qualification.json")
ALPHAS = (0.0, 2.0**-20, 2.0**-16, 2.0**-12, 2.0**-10, 2.0**-7)


def exact_selectors(left: Tensor, right: Tensor) -> Tensor:
    """0=inactive, 1=left, 2=right, 3=exact active tie (AD averages)."""
    positive = (left > 0) & (right > 0)
    negative = (left < 0) & (right < 0)
    active = positive | negative
    choose_left = torch.where(positive, left < right, left > right)
    return torch.where(active, torch.where(left == right, 3,
                       torch.where(choose_left, 1, 2)), 0).to(torch.int8)


class LimiterTrace:
    def __init__(self) -> None:
        self.stages: list[dict[str, Any]] = []

    def __call__(self, q: Tensor, qx: Tensor, qy: Tensor) -> None:
        record: dict[str, Any] = {"shape": list(q.shape), "scale": float(q.abs().max()), "qx_sign": qx.sign().tolist(),
                  "qy_sign": qy.sign().tolist()}
        for axis, left, right in (
            ("x", q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
            ("y", q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]),
        ):
            record[axis] = {"selector": exact_selectors(left, right).tolist(),
                            "left": left.tolist(), "right": right.tolist()}
        self.stages.append(record)


def compare_traces(before: LimiterTrace, after: LimiterTrace) -> dict[str, Any]:
    if len(before.stages) != len(after.stages):
        raise ValueError("ray traces have different stage counts")
    changes: list[dict[str, Any]] = []
    flux_changes: list[dict[str, Any]] = []
    for stage, (old, new) in enumerate(zip(before.stages, after.stages)):
        if old["shape"] != new["shape"]:
            raise ValueError("ray traces have different grids")
        for axis in ("x", "y"):
            a, b = old[axis], new[axis]
            for row, (old_row, new_row) in enumerate(zip(a["selector"], b["selector"])):
                for col, (left_choice, right_choice) in enumerate(zip(old_row, new_row)):
                    if left_choice != right_choice:
                        changes.append({"stage": stage, "axis": axis, "cell": [row + 1, col + 1],
                            "before": left_choice, "after": right_choice,
                            "operands_before": [a["left"][row][col], a["right"][row][col]],
                            "operands_after": [b["left"][row][col], b["right"][row][col]],
                            "q_scale_before": old["scale"], "q_scale_after": new["scale"]})
            for row, (old_row, new_row) in enumerate(zip(old[f"q{axis}_sign"], new[f"q{axis}_sign"])):
                for col, (old_sign, new_sign) in enumerate(zip(old_row, new_row)):
                    if old_sign != new_sign:
                        flux_changes.append({"stage": stage, "axis": axis, "face": [row, col],
                                             "before": old_sign, "after": new_sign})
    return {"limiter_changes": changes, "limiter_change_count": len(changes),
            "flux_sign_changes": flux_changes, "flux_sign_change_count": len(flux_changes)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_module_sources(modules: Iterable[object]) -> set[str]:
    paths: set[str] = set()
    for module in modules:
        name = getattr(module, "__file__", None)
        if name and Path(name).is_absolute() and Path(name).is_file():
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                relative = path.relative_to(ROOT)
                if relative.parts[0] != ".venv":
                    paths.add(str(relative))
    return paths


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load diagnostic helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, record: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run() -> dict[str, Any]:
    if OUT.exists():
        raise ValueError("diagnostic directory already exists; preserve previous attempts")
    start = time.monotonic()
    deadline = start + 60.0
    q = _load("limiter_ray_qualifier", HERE / "qualify_candidate.py")
    geo = _load("limiter_ray_geometry", HERE / "two_face_intersection_20261013/diagnose.py")
    from advar import transport
    accounting = json.loads((HERE / "FINAL_ACCOUNTING.json").read_text())
    prior = json.loads(ENDPOINT.read_text())
    metric_record = json.loads(METRIC.read_text())
    case = q.cycle.load_fixed_case()
    control = torch.tensor(prior["current_control"], dtype=torch.float64)
    problem, parameters = case["problem"], case["parameters"]
    fv = problem.frozen.fv_transport
    matrix, pivots, free, _ = geo._chart(fv.coefficient_limits, fv.psi_basis)
    lift, coordinates, projected, _ = geo._make_intersection_chart(control, matrix, pivots, free)
    if not torch.equal(projected, control) or q._tensor_sha(control) != prior["current_control_sha256"]:
        raise ValueError("endpoint does not roundtrip its stored chart")
    gradient = torch.tensor(prior["current_point"]["tangent_hull"]["vector"], dtype=torch.float64)
    metric = torch.tensor(metric_record["fresh_hessians"][-1]["hessian"], dtype=torch.float64)
    direction = torch.linalg.solve(metric, -gradient)
    old_ray = json.loads((HERE / "coupled_ray_20261013/RESULT.json").read_text())
    for row in old_ray["rows"]:
        if row["alpha"] in ALPHAS:
            if q._tensor_sha(lift(coordinates + row["alpha"] * direction)) != row["control_sha256"]:
                raise ValueError("reconstructed direction differs from the stored coupled ray")
    observer_plan = json.loads((HERE / "current_endpoint_observer_20261013/OBSERVER_PLAN.json").read_text())
    source_paths = set(observer_plan["source_hashes"]) | {
        str(Path(__file__).relative_to(ROOT)), str((HERE / "qualify_candidate.py").relative_to(ROOT)),
        str((HERE / "two_face_intersection_20261013/diagnose.py").relative_to(ROOT)),
        str((HERE / "reference_search.py").relative_to(ROOT)),
    }
    # Capture all loaded local Python helpers, including imports reached by archived modules.
    source_paths.update(local_module_sources(tuple(sys.modules.values())))
    sources = {name: _sha(ROOT / name) for name in sorted(source_paths)}
    inputs = {str(path.relative_to(ROOT)): _sha(path) for path in (
        ENDPOINT, METRIC, HERE / "FINAL_ACCOUNTING.json", q.cycle.CASE,
        HERE / "coupled_ray_20261013/RESULT.json",
    )}
    if accounting["direct_native_J_qualification_observer_total"] + len(ALPHAS) != 512:
        raise ValueError("six remaining direct-J calls no longer available")
    if accounting["remaining_estimated_seconds"] < 60:
        raise ValueError("insufficient inherited diagnostic time budget")
    plan: dict[str, Any] = {"scope": "fixed existing coupled ray; exact limiter and flux choices separately; no candidate selection",
        "source_hashes": sources, "input_hashes": inputs, "alphas": list(ALPHAS),
        "control_sha256": q._tensor_sha(control), "parameters_sha256": q._tensor_sha(parameters),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU FP64"},
        "direction": direction.tolist(), "stored_tangent_slope": float(gradient @ direction),
        "metric_is_current_hessian": False,
        "repeated_costs_reason": "alpha 0, 2^-10 and 2^-7 need raw stage traces absent from old receipts; also check native/source closure",
        "budget": {"wall_seconds": 60, "direct_J_cap": 6,
        "wall_policy": "check between calls and record actual elapsed time; in-flight calls cannot be interrupted",
        "gradient_cap": 6, "direct_J_before": 506, "direct_J_cumulative_cap": 512,
        "gradient_before": accounting["gradient_current_qualification_total"],
        "charged_seconds_before": accounting["charged_after_current_observers_seconds"],
        "remaining_estimated_seconds_before": accounting["remaining_estimated_seconds"],
        "RSS_cap_bytes": 2 * 1024**3}, "minimum_claim": False, "score_calls": 0, "Hessian_calls": 0}
    OUT.mkdir()
    _write(OUT / "PLAN.json", plan)
    (OUT / "PLAN_SHA256.txt").write_text(_sha(OUT / "PLAN.json") + "\n")
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {"status": "in_progress", "plan_sha256": _sha(OUT / "PLAN.json"), "rows": rows,
        "direct_J_started": 0, "direct_J_completed": 0, "gradient_started": 0,
        "gradient_completed": 0, "minimum_claim": False, "accepted_points": 0,
        "Hessian_calls": 0, "score_calls": 0}
    _write(OUT / "RESULT.json", result)
    baseline_trace = None
    try:
        with tarfile.open(OUT / "source.tar.gz", "w:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        for alpha in ALPHAS:
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic wall limit reached")
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 2 * 1024**3:
                raise MemoryError("diagnostic RSS cap reached")
            location = coordinates + alpha * direction
            point = lift(location)
            trace = LimiterTrace()
            result["direct_J_started"] += 1
            _write(OUT / "RESULT.json", result)
            with torch.no_grad(), transport.observe_minmod_stages(trace):
                value = float(problem.objective(point, parameters))
            result["direct_J_completed"] += 1
            if len(trace.stages) != 360:
                raise ValueError("analysis trace does not contain 360 stages")
            if baseline_trace is None:
                baseline_trace = trace
                if abs(value - prior["native_J"]) > 1e-14:
                    raise ValueError("native baseline differs from stored endpoint")
            row: dict[str, Any] = {"alpha": alpha, "control_sha256": q._tensor_sha(point), "native_J": value,
                   "delta_J": value - rows[0]["native_J"] if rows else 0.0,
                   "stage_count": len(trace.stages), **compare_traces(baseline_trace, trace)}
            rows.append(row)
            _write(OUT / "RESULT.json", result)
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic deadline before native gradient")
            result["gradient_started"] += 1
            _write(OUT / "RESULT.json", result)
            fresh = torch.func.grad(lambda z: problem.objective(lift(z), parameters))(location)
            result["gradient_completed"] += 1
            if not bool(torch.isfinite(fresh).all()):
                raise ValueError("native tangent gradient is nonfinite")
            row.update(tangent_gradient=fresh.tolist(), autograd_tangent_gradient_dot_direction=float(fresh @ direction),
                       derivative_interpretation="selected AD branch; base value is not an independently certified right derivative",
                       step_control_norm=float(torch.linalg.vector_norm(point - control)))
            if alpha:
                row["secant_slope"] = row["delta_J"] / alpha
                # Scalar subtraction resolution only; not an FV backward-error certificate.
                row["scalar_roundoff_slope_scale"] = 8 * torch.finfo(control.dtype).eps * abs(value) / alpha
            _write(OUT / "RESULT.json", result)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="interrupted", error=f"{type(error).__name__}: {error}")
    finally:
        elapsed = time.monotonic() - start
        result.update(elapsed_seconds=elapsed,
            direct_J_cumulative_started=plan["budget"]["direct_J_before"] + result["direct_J_started"],
            direct_J_cumulative_completed=plan["budget"]["direct_J_before"] + result["direct_J_completed"],
            gradient_cumulative_started=plan["budget"]["gradient_before"] + result["gradient_started"],
            gradient_cumulative_completed=plan["budget"]["gradient_before"] + result["gradient_completed"],
            inherited_charged_seconds_after=plan["budget"]["charged_seconds_before"] + elapsed,
            wall_limit_met=elapsed <= 60, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_unchanged=all(_sha(ROOT / name) == digest for name, digest in sources.items()),
            inputs_unchanged=all(_sha(ROOT / name) == digest for name, digest in inputs.items()),
            parameters_unchanged=q._tensor_sha(parameters) == plan["parameters_sha256"])
        _write(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    record = run()
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
        "rows": [{key: row.get(key) for key in ("alpha", "delta_J", "secant_slope",
                 "autograd_tangent_gradient_dot_direction", "limiter_change_count", "flux_sign_change_count")}
                 for row in record["rows"]]}))
