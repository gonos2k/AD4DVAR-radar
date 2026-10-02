"""Read-only precision diagnosis of one previously rejected FV slice trial.

This reconstructs a trial from an archived Newton vector. It does not accept
the trial, optimize controls, or publish a response.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform

import mpmath as mp
import torch

from advar import variational as v
from advar.physics import echo_to_dbz
from advar.local_refinement import _ARMIJO_CONSTANT
from advar.transport import observe_minmod_stages
from examples.weather_scenarios import fv_partial_signed_face_slices as slices
from examples.weather_scenarios import fv_slice_precision_reference as reference

RAW = slices.EVIDENCE / "partial_signed_face_slices_attempt1/signed_face_slices.json"
RAW_SHA = "2e331247833dcf9cda71891bc99283551e5490bb9701fbde3b29eb18fcfbaee9"
PLAN = slices.EVIDENCE / "R4_PRECISION_DIAGNOSTIC_PLAN_20261002.md"
AMENDMENT = slices.EVIDENCE / "R4_PRECISION_ATTEMPT2_AMENDMENT_20261002.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def armijo_ratio(candidate_norm: float, current_norm: float, slope: float, scale: float) -> float:
    limit = math.sqrt(max(0.0, 1.0 + 2.0 * _ARMIJO_CONSTANT * scale * slope))
    return candidate_norm / current_norm / limit if limit > 0 else math.inf


def branch_choices(q, qx, qy):
    def choice(a, b):
        return torch.where(a * b <= 0, 0, torch.where(a.abs() < b.abs(), 1,
                           torch.where(b.abs() < a.abs(), 2, 3))).flatten().tolist()
    return {
        "x": choice(q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
        "y": choice(q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]),
        "qx_sign": qx.sign().flatten().tolist(),
        "qy_sign": qy.sign().flatten().tolist(),
    }


def fixture_for(problem, parameters):
    frozen = problem.contract(parameters)
    spec = frozen.fv_transport
    assert spec is not None and spec.reconstruction == "minmod"
    observations = replace(problem.observations, dbz=problem._observation_values(parameters))
    assert not observations.censored_mask.any()
    assert torch.equal(observations.detected_mask, observations.valid_mask)
    assert frozen.neural_prior_std_dbz is None
    assert observations.common_bias_mode_weights is None
    assert observations.common_bias_group_index is None
    assert frozen.analysis_config.observation_common_bias_tile_size_px == 0
    assert frozen.observation_whitener.per_frame
    assert frozen.initial_support_mask.all()
    assert spec.boundary_echo is not None and spec.boundary_support is not None
    assert all(bool(edge.eq(1).all()) for pair in spec.boundary_support for stage in pair for edge in stage)
    floor = frozen.nowcast_config.min_dbz
    cfg = frozen.analysis_config
    fixture = {
        "background_dbz": frozen.initial_background_dbz.tolist(),
        "psi_basis": spec.psi_basis.tolist(),
        "coefficient_limits": spec.coefficient_limits.tolist(),
        "growth_limit": frozen.nowcast_config.max_log_growth_per_step,
        "floor_dbz": floor, "echo_floor": 10.0 ** (floor / 10.0),
        "echo_exponent_factor": math.log(10.0) / 10.0,
        "dbz_log_factor": 10.0 / math.log(10.0),
        "transform_scale": cfg.echo_transform_scale_dbz,
        "increment_ratio": cfg.initial_increment_scale_dbz / cfg.echo_transform_scale_dbz,
        "transform_epsilon": cfg.transform_epsilon,
        "substeps": spec.substeps_per_interval,
        "dt": frozen.nowcast_config.interval_minutes * 60.0 / spec.substeps_per_interval,
        "area": spec.spacing_yx[0] * spec.spacing_yx[1],
        "boundary_echo": [[[edge.tolist() for edge in stage] for stage in pair] for pair in spec.boundary_echo],
        "observation_dbz": observations.dbz.tolist(),
        "valid_mask": observations.valid_mask.tolist(),
        "detected_mask": observations.detected_mask.tolist(),
        "std_dbz": observations.std_dbz.tolist(),
        "quality_weight": observations.quality_weight.tolist(),
        "whitener_mode": frozen.observation_whitener.mode.tolist(),
        "bias_std": cfg.observation_common_bias_std_dbz, "per_frame": True,
        "smooth_left_index": frozen.smooth_edge_left_index.tolist(),
        "smooth_right_index": frozen.smooth_edge_right_index.tolist(),
        "smooth_physical_weight": frozen.smooth_edge_physical_weight.tolist(),
        "smooth_weight": cfg.field_smoothness_weight,
        "robust_delta": cfg.pseudo_huber_delta,
    }
    return fixture, observations, frozen


def main(output: Path):
    if output.exists():
        raise ValueError("precision output must be fresh")
    if sha(RAW) != RAW_SHA:
        raise ValueError("archived slice bytes changed")
    saved = json.loads(RAW.read_text())
    slices._validate_runtime(saved["environment"])
    before = {name: sha(slices.ROOT / name) for name in saved["source_before"]}
    if before != saved["source_before"]:
        raise ValueError("measured source identity differs")
    diagnostic_sources = {str(Path(path).relative_to(slices.ROOT)): sha(Path(path))
                          for path in (__file__, reference.__file__)}
    plan_sha = sha(PLAN)
    amendment_sha = sha(AMENDMENT)
    problem, parameters, identity = slices._current_problem()
    if slices._tensor_sha(parameters) != saved["parameters_sha256"]:
        raise ValueError("fixed parameters differ from archived slice input")
    chart = slices._chart_for_problem(problem)
    row = next(row for row in saved["slices"] if row["factor"] == 2)
    solve = row["linear_solves"][-1]
    tangent = torch.tensor(solve["input_tangent"], dtype=torch.float64)
    step = torch.tensor(solve["solution"], dtype=torch.float64)
    eta = tangent.new_tensor(solve["eta"])
    controls = [slices._slice_control(chart, tangent + scale * step, eta) for scale in (0.0, 1.0)]
    assert slices._tensor_sha(controls[0]) == row["last_accepted_control_sha256"]
    fixture, observations, frozen = fixture_for(problem, parameters)
    torch_results = []
    for control in controls:
        production_eta = slices._face_flux(problem, control) / chart.face_scale
        if abs(float(production_eta - eta)) > slices._eta_roundoff_tolerance(chart, control):
            raise ValueError("reconstructed control changed fixed face coordinate")
        strict_branch = slices._branch_summary(problem, control, parameters)
        if strict_branch["signature_sha256"] != solve["input_signature"]:
            raise ValueError("reconstructed endpoint changed the full strict branch")
        trace = []
        with observe_minmod_stages(lambda q, qx, qy: trace.append(branch_choices(q, qx, qy))):
            residual = v.whitened_observation_residual(control, observations, frozen)
        value = v._robust_objective_from_residual(control, residual, observations, frozen)
        trace = [{"step": index // 2, "stage": index % 2, **entry} for index, entry in enumerate(trace)]
        prediction = echo_to_dbz(v._analysis_trajectory(control, frozen).frames_linear,
                                 min_dbz=frozen.nowcast_config.min_dbz)
        torch_results.append({"objective": float(value), "residual": residual.tolist(),
                              "predicted_dbz": prediction.tolist(),
                              "control_sha256": slices._tensor_sha(control), "production_eta": float(production_eta),
                              "branch_choices": trace, "strict_branch": strict_branch})
    assert len(torch_results[0]["branch_choices"]) == 36
    slices._assert_scalar_match("base objective", row["accepted_steps"][-1]["objective"], torch_results[0]["objective"])
    slices._assert_scalar_match("reconstructed trial objective", row["rejected_trials"][0]["objective"], torch_results[1]["objective"])
    precision_results = {}
    for dps in (50, 80):
        results = [reference.evaluate(fixture, control.tolist(), dps=dps) for control in controls]
        for i, result in enumerate(results):
            if result["branch_choices"] != torch_results[i]["branch_choices"]:
                raise ValueError("precision reference selected a different branch")
            with mp.workdps(dps):
                errors = [abs(mp.mpf(a)-mp.mpf(b)) for ra, rb in zip(result["predicted_dbz"],
                          torch_results[i]["predicted_dbz"], strict=True)
                          for la, lb in zip(ra, rb, strict=True) for a, b in zip(la, lb, strict=True)]
                result["torch_prediction_max_absolute_difference"] = str(max(errors))
        with mp.workdps(dps):
            delta = mp.mpf(results[1]["objective"]) - mp.mpf(results[0]["objective"])
            delta_string = str(delta)
        precision_results[str(dps)] = {"delta": delta_string, "endpoints": results}
    gradient = torch.func.grad(slices._slice_objective(problem.objective, chart, eta))
    integral_rows = []
    for alpha, weight in ((0.0, 1 / 6), (0.5, 4 / 6), (1.0, 1 / 6)):
        g = gradient(tangent + alpha * step, parameters)
        integral_rows.append({"alpha": alpha, "weight": weight, "gradient_step": float(torch.dot(g, step)),
                              "gradient_max": float(g.abs().max()), "gradient_l2": float(torch.linalg.vector_norm(g))})
        if alpha == 1.0:
            slices._assert_scalar_match("reconstructed trial gradient",
                row["rejected_trials"][0]["gradient_max"], float(g.abs().max()))
    reconstructed_armijo_ratio = armijo_ratio(integral_rows[-1]["gradient_l2"],
        integral_rows[0]["gradient_l2"], row["rejected_trials"][0]["normalized_slope"], 1.0)
    slices._assert_scalar_match("reconstructed trial Armijo ratio",
                               row["rejected_trials"][0]["armijo_ratio"], reconstructed_armijo_ratio)
    with mp.workdps(80):
        residuals = [[mp.mpf(value) for frame in result["residual"] for line in frame for value in line]
                     for result in torch_results]
        valid = observations.valid_mask.flatten().tolist()
        delta = mp.mpf(frozen.analysis_config.pseudo_huber_delta)
        robust_difference = mp.fsum((b-a)*(b+a)/(mp.sqrt(1+(a/delta)**2)+mp.sqrt(1+(b/delta)**2))
                                    for a, b, active in zip(*residuals, valid, strict=True) if active)
        prior_difference = mp.fsum((mp.mpf(b)-mp.mpf(a))*(mp.mpf(b)+mp.mpf(a))/2
                                   for a, b in zip(controls[0].tolist(), controls[1].tolist(), strict=True))
        # This pinned input has no field smoothness cost; refuse new profiles.
        assert frozen.analysis_config.field_smoothness_weight == 0
        promoted_components = {"robust_delta": str(robust_difference), "prior_delta": str(prior_difference),
                               "total_delta": str(robust_difference + prior_difference),
                               "scope": "accurate reduction of already rounded FP64 residuals; no forward error correction"}
    after = {name: sha(slices.ROOT / name) for name in before}
    if (after != before or sha(PLAN) != plan_sha or sha(AMENDMENT) != amendment_sha
            or any(sha(slices.ROOT / name) != digest for name, digest in diagnostic_sources.items())):
        raise ValueError("source changed during calculation")
    result = {
        "scope": "one reconstructed policy-rejected trial; fixed FP64 input constants; independent high-precision discrete objective; no new optimization, acceptance or response",
        "raw_sha256": RAW_SHA, "source_before": before, "source_after": after,
        "plan_sha256": plan_sha,
        "attempt_amendment_sha256": amendment_sha,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "mpmath": mp.__version__, "device": "CPU", "decimal_digits": [50, 80]},
        "diagnostic_source_sha256": diagnostic_sources, "input_identity": identity,
        "parameters_sha256": slices._tensor_sha(parameters), "fixture": fixture,
        "control_base": controls[0].tolist(), "control_reconstructed_trial": controls[1].tolist(),
        "torch": torch_results, "precision": precision_results,
        "gradient_simpson_diagnostic": integral_rows,
        "reconstructed_armijo_ratio": reconstructed_armijo_ratio,
        "promoted_fp64_component_difference": promoted_components,
        "gradient_simpson_delta": math.fsum(row["weight"] * row["gradient_step"] for row in integral_rows),
        "full_root_claim": False, "response_validation": "not_performed",
        "original_trial_accepted": False,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
