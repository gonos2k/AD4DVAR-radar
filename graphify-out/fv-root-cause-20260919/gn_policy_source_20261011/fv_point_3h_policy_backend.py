"""Reusable fixed-seed FV backend for bounded point-policy comparisons.

This module deliberately stops at evaluation and independent repeat receipts:
policy selection and any point promotion belong to the calling runner.
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as finite
from examples.weather_scenarios import fv_point_3h_gn_local_commit as gni
from examples.weather_scenarios import fv_point_3h_gn_prepared_search as prepared
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios import fv_point_3h_tangent_coupled_gn_comparison as coupled
from advar import transport


class FVBackend:
    """Evaluate point-local robust-GN models and candidate policy facts.

    ``build`` performs the only current-point operator work (24 row VJPs, one
    row-space solve, and two selected-face HVPs). Candidate evaluations never
    calculate HVPs and never alter/promote the carried point.
    """

    face: dict[str, Any] = dict(finite.FACE)
    face_scale = finite.FACE_SCALE
    radius = finite.RADIUS
    max_builds = 3
    rows_per_build = 24
    hvps_per_build = 2

    def __init__(self, plan: dict[str, Any], plan_path: Path, plan_sha: str,
                 output_dir: Path, deadline: float):
        self.plan = plan
        self.plan_path = Path(plan_path).resolve()
        self.plan_sha = plan_sha
        self.output_dir = Path(output_dir)
        self.deadline = float(deadline)
        self.build_count = 0
        self.row_count = 0
        self.solve_count = 0
        self.hvp_count = 0
        self.problem, self.original, _, self.parameters, self.truth, _ = seed._prepare_fixed_seed()
        self.timings = {"identity_seconds": 0.0, "own_write_seconds": 0.0,
            "candidate_observe_seconds": 0.0, "repeat_seconds": 0.0,
            "row_solve_seconds_inclusive": 0.0, "pure_hvp_seconds": 0.0}
        self.parent = finite._load_parent()
        self.control = torch.as_tensor(self.parent["current_control"], dtype=torch.float64)
        if tangent._tensor_sha(self.control) != finite.BASE_SHA:
            raise ValueError("policy backend did not load the frozen original P2 control")
        self.input_anchor = shared._input_identity(
            self.problem, self.original, self.control, self.parameters, self.truth)
        self.runtime_anchor = shared._runtime()
        self.source_anchor = shared._source_hashes(self.plan, self.plan_path, self.plan_sha)
        self.weights = geometry._face_weights(
            self.problem, axis=self.face["axis"], row=self.face["row"],
            column=self.face["column"])
        if (self.input_anchor != self.parent["input_after"]
                or self.runtime_anchor != self.parent["runtime_after"]):
            raise ValueError("fixed seed input/runtime differs from the closed P2 parent")
        self._check(self.control)

    def _check(self, control: Tensor | None = None) -> dict[str, float]:
        started = time.perf_counter()
        shared._deadline(self.deadline, float(self.plan["policy"]["internal_seconds"]))
        source_now = shared._source_hashes(self.plan, self.plan_path, self.plan_sha)
        if source_now != self.source_anchor:
            raise ValueError("policy backend source identity changed")
        checked_control = self.control if control is None else control
        expected_sha = tangent._tensor_sha(checked_control)
        identity = shared._input_identity(
            self.problem, self.original, checked_control, self.parameters, self.truth)
        if not geometry.model._fixed_input(identity, self.input_anchor, expected_sha):
            raise ValueError("policy backend fixed input identity changed")
        if shared._runtime() != self.runtime_anchor:
            raise ValueError("policy backend runtime identity changed")
        shared._deadline(self.deadline, float(self.plan["policy"]["internal_seconds"]))
        elapsed = time.perf_counter() - started
        self.timings["identity_seconds"] += elapsed
        return {"identity_check_seconds": elapsed}

    def _write_operator(self, path: Path, record: dict[str, Any]) -> None:
        started = time.perf_counter()
        tangent._write(path, record)
        self.timings["own_write_seconds"] += time.perf_counter() - started

    @staticmethod
    def _full_g_match(actual: Tensor, expected: Tensor) -> dict[str, Any]:
        return finite._full_G_match(actual, expected)

    def build(self, control: Tensor, closure: dict[str, Any], full_G: Tensor,
              tag: str) -> dict[str, Any]:
        if self.build_count >= self.max_builds:
            raise ValueError("policy backend point-build cap of three exceeded")
        self.build_count += 1
        point_started = time.perf_counter()
        control = torch.as_tensor(control, dtype=torch.float64)
        if control.shape != self.control.shape or not bool(torch.isfinite(control).all()):
            raise ValueError("policy backend point control has an invalid shape or nonfinite value")
        self._check(control)
        input_before = shared._input_identity(
            self.problem, self.original, control, self.parameters, self.truth)
        source_before, runtime_before = self.source_anchor, self.runtime_anchor
        observed = tangent._observe(shared, self.problem, control, self.parameters, self.weights)
        theta = float(tangent.minimum_mixture_weight(
            observed["side"][-1][1], observed["side"][1][1]))
        G = torch.cat(((1.0 - theta) * observed["side"][-1][1]
                       + theta * observed["side"][1][1],
                       (observed["q"] / self.face_scale).reshape(1)))
        expected_G = torch.as_tensor(full_G, dtype=control.dtype)
        full_G_check = self._full_g_match(G, expected_G)
        if not full_G_check.get("passed"):
            raise ValueError("fresh current full-G vector differs componentwise from supplied point")
        eps = 128 * torch.finfo(control.dtype).eps
        gradient_match = shared._gradient_pair_match(
            {s: observed["side"][s][1] for s in (-1, 1)},
            {s: torch.as_tensor(closure["side_gradients"][str(s)], dtype=control.dtype)
             for s in (-1, 1)})
        native_j_match = abs(float(observed["native_j"]) - float(closure["objective"])) <= eps * max(
            abs(float(observed["native_j"])), abs(float(closure["objective"])),
            torch.finfo(control.dtype).tiny)
        theta_match = abs(theta - float(closure["theta"])) <= eps * max(
            abs(theta), abs(float(closure["theta"])), torch.finfo(control.dtype).tiny)
        residual = float(torch.dot(G, G))
        residual_match = abs(residual - float(closure["F_squared"])) <= eps * max(
            abs(residual), abs(float(closure["F_squared"])), torch.finfo(control.dtype).tiny)
        trace_match = all(observed["traces"][s]["signature_sha256"]
            == closure["branch_trace"][str(s)]["signature_sha256"] for s in (-1, 1))
        qualifications = {"gradient_pair_match": gradient_match,
            "branch_pair_passed": observed["pair"]["passed"], "face_audit_passed": observed["face_ok"],
            "side_objectives_match_native": observed["side_objectives_match_native"],
            "side_gradients_finite": observed["side_gradients_finite"],
            "native_objective_matches_closure": native_j_match,
            "theta_matches_closure": theta_match, "F_squared_matches_closure": residual_match,
            "trace_matches_closure": trace_match}
        if not all((gradient_match, observed["pair"]["passed"], observed["face_ok"],
                    observed["side_objectives_match_native"], observed["side_gradients_finite"],
                    native_j_match, theta_match, residual_match, trace_match)):
            raise ValueError("fresh point J/G/theta/strict pair/trace qualification failed")
        pivot, normal = geometry._pivot(control, self.weights), geometry._face_normal(control, self.weights)
        _, chart, _ = geometry._chart_jacobian(control, self.weights, pivot, 0.0)
        record: dict[str, Any] = {"point_tag": str(tag), "point_index": self.build_count - 1,
            "base_control_sha256": tangent._tensor_sha(control), "policy": self.plan["policy"],
            "jacobian_rows_started": 0, "jacobian_rows_completed": 0,
            "jacobian_row_history": [], "dense_solves_started": 0,
            "dense_solves_completed": 0, "hvp_calls_started": 0,
            "hvp_calls_completed": 0, "hvp_history": [],
            "base_J": float(observed["native_j"]), "base_R": residual,
            "base_theta": theta, "base_full_G": G.tolist()}
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag)).strip("._") or "point"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        operator_path = self.output_dir / f"{self.build_count:02d}_{safe_tag}_operator.json"
        context = {"problem": self.problem, "parameters": self.parameters,
            "record": record, "output": operator_path, "deadline": self.deadline,
            "weights": self.weights, "plan": self.plan, "plan_path": self.plan_path,
            "plan_sha": self.plan_sha, "model_state": {"observed": observed}}
        factory_started = time.perf_counter()
        specs = coupled._direction_model_factory(context)(control, theta,
            observed["side"][-1][1], observed["side"][1][1], normal, chart, pivot)
        operator_phase_seconds = time.perf_counter() - factory_started
        self.timings["row_solve_seconds_inclusive"] += operator_phase_seconds
        self.row_count += int(record["jacobian_rows_completed"])
        self.solve_count += int(record["dense_solves_completed"])
        if (record["jacobian_rows_started"] != self.rows_per_build
                or record["jacobian_rows_completed"] != self.rows_per_build
                or record["dense_solves_started"] != 1 or record["dense_solves_completed"] != 1
                or self.row_count > 72 or self.solve_count > 3):
            raise ValueError("current coupled-GN row/solve budget does not match 24/1 per point")
        gn_specs = [spec for spec in specs if spec.get("name") == "robust_gn_coupled"]
        if len(gn_specs) != 1:
            raise ValueError("current coupled-GN factory did not return one coupled model")
        spec = gn_specs[0]
        direction = spec["direction"]
        grad_fn = torch.func.grad(self.problem.objective, argnums=0)
        HVP: dict[int, Tensor] = {}
        hvp_times: dict[int, float] = {}
        for side in (-1, 1):
            self._check(control)
            label = {"phase": "policy_backend_current_point", "side": side,
                "base_control_sha256": tangent._tensor_sha(control), "point_tag": str(tag),
                "direction_sha256": tangent._tensor_sha(direction), "reference_theta": theta,
                "working_theta": float(spec["working_theta"]),
                "operator": "selected_face_extension", "status": "started"}
            if self.hvp_count >= 6:
                raise ValueError("policy backend aggregate HVP cap of six exceeded")
            self.hvp_count += 1
            record["hvp_calls_started"] += 1
            record["hvp_history"].append(label)
            self._write_operator(operator_path, record)
            started = time.perf_counter()
            with transport.selected_face_extension(self.face["axis"], self.face["row"],
                                                   self.face["column"], side):
                value = torch.func.jvp(lambda point: grad_fn(point, self.parameters),
                                       (control,), (direction,))[1]
            hvp_times[side] = time.perf_counter() - started
            self.timings["pure_hvp_seconds"] += hvp_times[side]
            if value.shape != control.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("current selected-face HVP is nonfinite or has wrong shape")
            HVP[side] = value
            record["hvp_calls_completed"] += 1
            record["hvp_history"][-1] = {**label, "status": "completed"}
            self._write_operator(operator_path, record)
            self._check(control)
        try:
            model = spec["model_builder"](observed["side"][-1][1], observed["side"][1][1],
                HVP[-1], HVP[1], normal, chart, theta, float(observed["q"]), pivot)
        except ValueError as error:
            raise ValueError(f"current coupled-GN tangent model unsupported: {error}") from error
        model_closure = finite._base_model_closure(model, G, theta)
        if not torch.equal(model.direction, direction) or model_closure["passed"] is not True:
            raise ValueError("coupled model changed direction or current full-G/theta")
        gates = model.gates
        if not all(gates.get(key) is True for key in (
                "finite", "both_side_gradients_descend", "envelope_slope_matches_residual_dot")):
            raise ValueError("coupled model finite, two-cost descent, or envelope gate refused")
        cost_slopes = (float(torch.dot(observed["side"][-1][1], direction)),
                       float(torch.dot(observed["side"][1][1], direction)))
        model_alphas = tangent.candidate_alphas(model, radius=self.radius, limit=24)
        record.update({"active_HVP_minus": HVP[-1].tolist(), "active_HVP_plus": HVP[1].tolist(),
            "side_gradients": {str(s): observed["side"][s][1].tolist() for s in (-1, 1)},
            "side_objectives": {str(s): float(observed["side"][s][0]) for s in (-1, 1)},
            "native_J": float(observed["native_j"]),
            "direction": direction.tolist(), "direction_sha256": tangent._tensor_sha(direction),
            "gn_diagnostics": spec["diagnostics"], "model_residual": model.residual.tolist(),
            "model_residual_direction": model.residual_direction.tolist(),
            "model_alpha_candidates": model_alphas,
            "model_predicted_F_squared": {str(alpha): float(torch.dot(
                model.residual + alpha * model.residual_direction,
                model.residual + alpha * model.residual_direction)) for alpha in model_alphas},
            "model_theta_prime": model.delta_theta, "model_gates": model.gates,
            "base_model_closure": model_closure})
        self._write_operator(operator_path, record)
        identity_checks = self._check(control)
        timing = {"point_total_seconds": time.perf_counter() - point_started,
            "row_and_solve_phase_seconds_inclusive": operator_phase_seconds,
            "pure_hvp_seconds": sum(hvp_times.values()), **identity_checks}
        return {"observed": observed, "theta": theta, "G": G, "R": residual,
            "J": float(observed["native_j"]), "direction": direction,
            "model": model, "pivot": pivot, "normal": normal, "chart": chart,
            "operator_record": str(operator_path), "operator_receipt": record,
            "closure": closure, "qualifications": qualifications,
            "full_G_match": full_G_check, "cost_slopes": cost_slopes,
            "timing": timing, "source_before": source_before,
            "input_before": input_before, "runtime_before": runtime_before,
            "branch_signature_equal_base": all(observed["traces"][s]["signature_sha256"]
                == closure["branch_trace"][str(s)]["signature_sha256"] for s in (-1, 1))}

    def chart(self, bundle: dict[str, Any], current: Tensor,
              direction: Tensor, alpha: float) -> Tensor:
        return prepared.local._chart_candidate(current, self.weights, direction,
                                               bundle["pivot"], alpha)

    def evaluate(self, bundle: dict[str, Any], candidate: Tensor,
                 alpha: float) -> dict[str, Any]:
        candidate = torch.as_tensor(candidate, dtype=torch.float64)
        self._check(candidate)
        started = time.perf_counter()
        trial = tangent._observe(shared, self.problem, candidate, self.parameters, self.weights)
        observe_seconds = time.perf_counter() - started
        self.timings["candidate_observe_seconds"] += observe_seconds
        self._check(candidate)
        facts: dict[str, Any] = {"alpha": float(alpha),
            "branch_pair_passed": trial["pair"]["passed"],
            "face_audit_passed": trial["face_ok"],
            "side_objectives_match_native": trial["side_objectives_match_native"],
            "side_gradients_finite": trial["side_gradients_finite"],
            "side_gradients": {str(s): trial["side"][s][1].tolist() for s in (-1, 1)},
            "branch_trace": {str(s): trial["traces"][s] for s in (-1, 1)},
            "branch_signature_matches_base": all(trial["traces"][s]["signature_sha256"]
                == bundle["observed"]["traces"][s]["signature_sha256"] for s in (-1, 1)),
            "candidate_observe_seconds": observe_seconds}
        try:
            theta = float(tangent.minimum_mixture_weight(
                trial["side"][-1][1], trial["side"][1][1]))
        except ValueError:
            theta = math.nan
        eps = 128 * torch.finfo(candidate.dtype).eps
        valid_theta = math.isfinite(theta) and eps < theta < 1.0 - eps
        facts["mixing_minimum_valid"] = valid_theta
        facts["theta"] = theta if valid_theta else None
        if not valid_theta:
            facts.update(finite=False, J_armijo_passed=False,
                         F_squared_armijo_passed=False, candidate_observe_seconds=observe_seconds)
            return finite._safe_candidate_facts(facts)
        G = torch.cat(((1.0 - theta) * trial["side"][-1][1]
                       + theta * trial["side"][1][1],
                       (trial["q"] / self.face_scale).reshape(1)))
        actual_j, actual_r = float(trial["native_j"]), float(torch.dot(G, G))
        slope = max(bundle["cost_slopes"])
        model = bundle["model"]
        j_bound, r_bound = prepared._armijo_bounds(bundle["J"], bundle["R"], float(alpha),
                                                  slope, model.merit_product)
        bounds_finite = math.isfinite(j_bound) and math.isfinite(r_bound)
        finite_values = bool(torch.isfinite(G).all()) and math.isfinite(actual_j) and math.isfinite(actual_r)
        facts.update({"objective": actual_j, "F_squared": actual_r, "G": G.tolist(),
            "J_armijo_bound": j_bound,
            "J_armijo_passed": bounds_finite and actual_j <= j_bound,
            "F_squared_armijo_bound": r_bound,
            "F_squared_armijo_passed": bounds_finite and actual_r <= r_bound,
            "finite": finite_values and bounds_finite,
            "model_error_norm": float(torch.linalg.vector_norm(
                G - (model.residual + float(alpha) * model.residual_direction))),
            "candidate_observe_seconds": observe_seconds})
        return finite._safe_candidate_facts(facts)

    def repeat(self, bundle: dict[str, Any], trial: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = torch.as_tensor(trial["control"], dtype=torch.float64)
        self._check(candidate)
        proposal_observation = {"native_j": trial["objective"],
            "face_ok": trial["face_audit_passed"],
            "pair": {"passed": trial["branch_pair_passed"]},
            "side_objectives_match_native": trial["side_objectives_match_native"],
            "side_gradients_finite": trial["side_gradients_finite"],
            "side": {s: (None, torch.as_tensor(trial["side_gradients"][str(s)], dtype=candidate.dtype))
                     for s in (-1, 1)},
            "traces": {s: trial["branch_trace"][str(s)] for s in (-1, 1)}}
        proposal = gni._candidate_proposal(candidate, proposal_observation,
            float(trial["theta"]), float(trial["F_squared"]), float(trial["alpha"]),
            float(trial["J_armijo_bound"]), float(trial["F_squared_armijo_bound"]))
        proposal["selected_arm"] = "prepared_gn"
        proposal["selected_direction_sha256"] = tangent._tensor_sha(bundle["direction"])
        input_anchor = shared._input_identity(
            self.problem, self.original, self.control, self.parameters, self.truth)
        proposal_input = shared._input_identity(
            self.problem, self.original, candidate, self.parameters, self.truth)
        if not geometry.model._fixed_input(proposal_input, input_anchor,
                                           tangent._tensor_sha(candidate)):
            raise ValueError("candidate fixed-input identity changed before independent repeat")
        repeat_started = time.perf_counter()
        repeat = gni._final_repeat(shared, geometry, tangent, self.plan,
            self.plan_path, self.plan_sha, self.problem, self.original,
            self.parameters, self.truth, self.weights, proposal, input_anchor,
            self.runtime_anchor, self.source_anchor, self.deadline)
        self.timings["repeat_seconds"] += time.perf_counter() - repeat_started
        self._check(candidate)
        return proposal, repeat
