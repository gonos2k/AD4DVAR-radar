"""Bounded in-memory growth-factor rounding diagnostic; never patches product files."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from advar import fv_sensitivity, transport, variational


HERE = Path(__file__).resolve().parent
CURRENT_STEP = transport.finite_volume_step


def _small_log_factor(log_growth: torch.Tensor) -> torch.Tensor:
    if bool(torch.abs(log_growth) < 0.125):
        return torch.addcmul(torch.ones_like(log_growth), torch.ones_like(log_growth), torch.expm1(log_growth))
    return torch.exp(log_growth)


def _scaled_growth(value: torch.Tensor, log_growth: torch.Tensor) -> torch.Tensor:
    if bool(torch.abs(log_growth) < 0.125):
        return torch.addcmul(value, value, torch.expm1(log_growth))
    return torch.exp(log_growth) * value


def _candidate_step() -> object:
    source = inspect.getsource(CURRENT_STEP).replace(
        "def finite_volume_step(", "def candidate_step(", 1
    )
    replacements = {
        "grown_echo = growth * echo":
        "grown_echo = _scaled_growth(echo, growth_log)",
        "growth * small_edges[0]":
        "_scaled_growth(small_edges[0], growth_log)",
        "growth * small_edges[1]":
        "_scaled_growth(small_edges[1], growth_log)",
        "growth * small_edges[2]":
        "_scaled_growth(small_edges[2], growth_log)",
        "growth * small_edges[3]":
        "_scaled_growth(small_edges[3], growth_log)",
        "growth * euler(\n                torch.zeros_like(echo), qx, qy, large_edges, dt, area\n            )":
        "_scaled_growth(euler(\n                torch.zeros_like(echo), qx, qy, large_edges, dt, area\n            ), growth_log)",
        "stage1 = growth * transformed_stage1":
        "stage1 = _scaled_growth(transformed_stage1, growth_log)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"candidate source pattern missing: {old}")
        source = source.replace(old, new)
    namespace = transport.__dict__.copy()
    namespace.update(_small_log_factor=_small_log_factor, _scaled_growth=_scaled_growth)
    exec(compile(source, "<growth-expm1-candidate>", "exec"), namespace)
    return namespace["candidate_step"]


def _gradient(control: torch.Tensor, observations, frozen) -> dict[str, object]:
    value = torch.func.grad(variational.robust_objective)(control, observations, frozen)
    index = int(value.abs().argmax())
    return {
        "gradient_max": float(value.abs().max()),
        "gradient_norm": float(value.norm()),
        "dominant_index": index,
        "dominant_value": float(value[index]),
    }


def _scalar_checks() -> dict[str, object]:
    logs = torch.tensor([-0.125, -1.0e-6, 0.0, 1.0e-6, 0.125], dtype=torch.float64)
    stable = torch.stack([_small_log_factor(value) for value in logs])
    exact = torch.exp(logs)
    small = logs.abs() < 0.125
    return {
        "logs": logs.tolist(),
        "max_small_abs_error_vs_exp": float((stable[small] - exact[small]).abs().max()),
        "max_small_relative_error_vs_exp": float(
            ((stable[small] - exact[small]).abs() / exact[small]).max()
        ),
        "minimum_factor": float(stable.min()),
        "all_factors_positive": bool((stable > 0).all()),
        "fallback_boundary_matches_exp": bool(stable[0] == exact[0] and stable[-1] == exact[-1]),
        "per_step_240_log_growth": -0.00011335020969746059 / 64.0,
    }


def main() -> None:
    torch.set_num_threads(1)
    started = time.perf_counter()
    nominal = torch.load(HERE / "rotation240_refined.pt", weights_only=False)
    impact = torch.load(HERE / "rotation240_impact_0.001.pt", weights_only=False)
    impact_observations = replace(nominal["observations"], dbz=impact["observations"])
    cases = {
        "nominal_refined": (
            nominal["control"], nominal["observations"],
            replace(nominal["frozen"],
                    input_frames_dbz=nominal["observations"].dbz,
                    initial_background_dbz=nominal["observations"].dbz[0]),
        ),
        "impact_0.001_c4": (
            impact["control"], impact_observations,
            replace(nominal["frozen"],
                    input_frames_dbz=impact_observations.dbz,
                    initial_background_dbz=impact_observations.dbz[0]),
        ),
    }
    report: dict[str, object] = {
        "scope": "in-memory expm1 growth-factor diagnostic; no product edit or adjoint claim",
        "candidate": "addcmul(value, value, expm1(log_growth)) for |log_growth|<0.125; original exp factors/guards retained",
        "gate": "unchanged max(abs(gradient)) <= 1e-8",
        "scalar_checks": _scalar_checks(),
        "source_sha256": {
            "src/advar/transport.py": hashlib.sha256(Path(transport.__file__).read_bytes()).hexdigest(),
            "src/advar/fv_sensitivity.py": hashlib.sha256(Path(fv_sensitivity.__file__).read_bytes()).hexdigest(),
            "candidate_step_source": hashlib.sha256(inspect.getsource(CURRENT_STEP).encode()).hexdigest(),
        },
        "results": {},
    }
    candidate = _candidate_step()
    try:
        transport.finite_volume_step = CURRENT_STEP
        report["baseline_current"] = {
            name: {"gradient": _gradient(control, observations, frozen)}
            for name, (control, observations, frozen) in cases.items()
        }
        transport.finite_volume_step = candidate
        candidate_results: dict[str, object] = {}
        report["candidate_addcmul"] = candidate_results
        for name, (control, observations, frozen) in cases.items():
            before = _gradient(control, observations, frozen)
            accepted: list[str] = []

            def save_step(value: torch.Tensor, record: dict[str, float | int]) -> None:
                path = HERE / f"growth_addcmul_{name}_step{len(accepted) + 1}.pt"
                torch.save({"control": value, "record": record}, path)
                accepted.append(str(path.name))

            try:
                final, records = fv_sensitivity.refine_fv_stationarity(
                    control, observations, frozen,
                    gradient_tolerance=1e-8,
                    maximum_iterations=4,
                    on_step=save_step,
                )
                after = _gradient(final, observations, frozen)
                output_name = f"growth_addcmul_{name}_final.pt"
                torch.save({"control": final, "records": records, "before": before, "after": after}, HERE / output_name)
                candidate_results[name] = {
                    "before": before, "after": after, "records": records,
                    "accepted_controls": accepted, "candidate_control": output_name,
                }
            except Exception as error:
                candidate_results[name] = {
                    "before": before, "accepted_controls": accepted,
                    "error": f"{type(error).__name__}: {error}",
                }
            (HERE / "growth_expm1_candidate.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        transport.finite_volume_step = CURRENT_STEP
    report["wall_seconds"] = time.perf_counter() - started
    (HERE / "growth_expm1_candidate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
