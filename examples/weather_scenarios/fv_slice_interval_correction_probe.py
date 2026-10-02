"""One newly declared tangent correction with a precision-aware objective gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from advar import matrix_free
from advar.local_refinement import refine_stationary, RefinementNumericalRefusal
from examples.weather_scenarios import fv_partial_signed_face_slices as slices
from examples.weather_scenarios import fv_slice_precision_probe as precision
from examples.weather_scenarios import fv_slice_precision_reference as reference
from examples.weather_scenarios import fv_slice_interval_acceptance as acceptance

PLAN = slices.EVIDENCE / "R4_INTERVAL_CORRECTION_PLAN_20261003.md"
PRECISION = slices.EVIDENCE / "partial_slice_precision_attempt2/precision.json"
PRECISION_SHA = "50267f00fed59074b8d848552fbdde70b3eeda0e25a1f5813fe2dd86d2825442"


def main(output: Path):
    if output.exists():
        raise ValueError("correction output must be fresh")
    if precision.sha(precision.RAW) != precision.RAW_SHA or precision.sha(PRECISION) != PRECISION_SHA:
        raise ValueError("archived inputs changed")
    archived = json.loads(precision.RAW.read_text())
    captured = json.loads(PRECISION.read_text())
    slices._validate_runtime(archived["environment"])
    before = {path: precision.sha(slices.ROOT / path) for path in archived["source_before"]}
    if before != archived["source_before"]:
        raise ValueError("original measured sources changed")
    measured = {str(Path(path).relative_to(slices.ROOT)): precision.sha(Path(path))
                for path in (__file__, precision.__file__, reference.__file__, acceptance.__file__)}
    plan_sha = precision.sha(PLAN)
    problem, parameters, identity = slices._current_problem()
    if slices._tensor_sha(parameters) != archived["parameters_sha256"]:
        raise ValueError("parameters changed")
    fixed_parameters = parameters.clone()
    chart = slices._chart_for_problem(problem)
    row = next(row for row in archived["slices"] if row["factor"] == 2)
    solve = row["linear_solves"][-1]
    tangent = torch.tensor(solve["input_tangent"], dtype=torch.float64)
    eta = tangent.new_tensor(solve["eta"])
    start = slices._slice_control(chart, tangent, eta)
    if slices._tensor_sha(start) != row["last_accepted_control_sha256"]:
        raise ValueError("start differs from archived last accepted point")
    fixture, _, _ = precision.fixture_for(problem, parameters)
    if fixture != captured["fixture"]:
        raise ValueError("captured fixed problem differs")
    signature = solve["input_signature"]
    strict = slices._strict_slice_branch(problem, chart, eta, signature)
    state = {"tangent": tangent.clone(), "signature": signature}

    def branch(t, p):
        if not torch.equal(p, fixed_parameters):
            raise ValueError("fixed parameters changed")
        summary, scope = strict(t, p)
        state["tangent"] = t.detach().clone()
        return {**summary, "physical_control": slices._slice_control(chart, t, eta).detach().tolist()}, scope

    objective = slices._slice_objective(problem.objective, chart, eta)
    trials, gates, solves = [], [], []
    gate = acceptance.make_gate(fixture, captured["torch"][0]["branch_choices"], gates)
    report: dict[str, Any] = {"scope": "one precision-aware correction on fixed +2eta slice; no full root or response",
              "input_precision_sha256": PRECISION_SHA, "plan_sha256": plan_sha,
              "original_source_before": before, "diagnostic_source_sha256": measured,
              "input_identity": identity, "environment": slices._runtime(),
              "historical_trial_accepted": False, "full_root_claim": False,
              "response_validation": "not_performed", "trial_history": trials,
              "interval_objective_gates": gates, "linear_solves": solves}
    try:
        with matrix_free.observe_pcg_calls(slices._true_residual_monitor(solves, float(eta), state)):
            result = refine_stationary(objective, tangent, parameters,
                branch_check=branch, max_iterations=1, max_backtracks=16,
                pcg_max_iterations=80, trial_acceptance=gate, trial_observer=trials.append)
        final_control = slices._slice_control(chart, result.control, eta)
        final_branch = slices._branch_summary(problem, final_control, parameters)
        final_gradient = torch.func.grad(objective)(result.control, parameters)
        fresh_max = float(final_gradient.abs().max())
        slices._assert_scalar_match("new tangent gradient", result.gradient_max, fresh_max)
        if fresh_max >= slices.STATIONARITY_TOLERANCE:
            raise RefinementNumericalRefusal("fresh tangent-gradient gate failed")
        endpoint = slices._endpoint("interval-correction-new-endpoint", "new_endpoint", final_control,
            eta, problem, chart, parameters, final_branch, slice_gradient=final_gradient)
        audited = slices._audit_linear_solves({"eta": float(eta), "linear_solves": solves,
            "status": "tangent_stationary_candidate", "start_branch": {"signature_sha256": signature}},
            problem, chart, parameters)
        report.update(status="tangent_stationary_candidate", final_endpoint=endpoint,
                      audited_linear_solves=audited, newton_iterations=result.iterations,
                      hvp_count=result.hvp_count, tangent_gradient_max=fresh_max)
    except RefinementNumericalRefusal as error:
        report.update(status="numerical_refusal", reason=str(error))
    after = {path: precision.sha(slices.ROOT / path) for path in before}
    if (after != before or precision.sha(PLAN) != plan_sha or not torch.equal(parameters, fixed_parameters)
            or any(precision.sha(slices.ROOT / path) != digest for path, digest in measured.items())
            or precision.sha(PRECISION) != PRECISION_SHA):
        raise ValueError("inputs or sources changed during correction")
    report["original_source_after"] = after
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
