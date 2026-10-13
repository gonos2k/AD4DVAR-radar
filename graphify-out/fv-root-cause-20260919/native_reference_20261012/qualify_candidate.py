"""Qualify an explicit native-J 26-control candidate on its actual analysis branch.

This reuses the immutable current-cycle inputs, the original point objective,
the strict minmod predicates, matrix-free exact J-HVPs, local_refinement, and
the terminal block/Schur arithmetic. It never scores a candidate or reuses a
historical endpoint control. Active nonsmooth candidates remain in the report
with their native J/control and are marked unsupported for active-set minimum
qualification; this module does not manufacture an active-face KKT system.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import matrix_free, transport, variational as v
from advar.local_refinement import RefinementNumericalRefusal, refine_stationary
from examples.weather_scenarios import fv_point_3h_netflux_current_cycle as cycle
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_first_branch_probe as branch_oracle
from examples.weather_scenarios import fv_point_3h_terminal_block_schur as blocks
from examples.weather_scenarios import fv_point_3h_merit_continuation as tail

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
SEARCH_PATH = HERE / "reference_search.py"
_search_spec = importlib.util.spec_from_file_location("native_reference_search", SEARCH_PATH)
if _search_spec is None or _search_spec.loader is None:
    raise RuntimeError("native reference search producer cannot be loaded")
search = importlib.util.module_from_spec(_search_spec)
sys.modules[_search_spec.name] = search
_search_spec.loader.exec_module(search)
CONTROL_SIZE, PARAMETERS_SIZE, ANALYSIS_STAGES = 26, 13, 360
STATIONARITY_TOL = 1.0e-10
BRANCH_PROBE_STEPS = (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12)
INTERNAL_SECONDS = 540.0
WALL_SECONDS = 600.0
RSS_LIMIT_BYTES = 2 * 1024**3
FROZEN_TRANSPORT_SHA256 = "b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a"
APPROVED_SUPPORT_TRANSPORT_SHA256 = "c9dde24c12237d5ffee6b8ca0766b4d5619abf6110f3e858ee447bf49c1f7da9"


class QualificationRefusal(ValueError):
    """A concrete input, branch, resource, or curvature qualification refusal."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _selector_counts(left: Tensor, right: Tensor, tolerance: Tensor) -> tuple[dict[str, int], Tensor]:
    positive = (left > tolerance) & (right > tolerance)
    negative = (left < -tolerance) & (right < -tolerance)
    active = positive | negative
    difference = (left - right).abs()
    tie = active & (difference <= tolerance)
    choose_left = active & torch.where(positive, left < right, left > right)
    choose_right = active & ~choose_left & ~tie
    inactive = ~active
    counts = {"left": int(choose_left.sum()), "right": int(choose_right.sum()),
              "inactive_zero": int(inactive.sum()), "active_tie": int(tie.sum())}
    selector = torch.where(choose_left, 1, torch.where(choose_right, 2,
                          torch.where(tie, 3, 0))).to(torch.int8)
    return counts, selector


class AnalysisBranchAudit:
    """Summarize actual 360-stage minmod selectors and their scale margins."""

    def __init__(self) -> None:
        self.stages = 0
        self.digest = hashlib.sha256()
        self.selector_counts = {
            axis: {key: 0 for key in ("left", "right", "inactive_zero", "active_tie")}
            for axis in ("x", "y")
        }
        self.face_sign_counts = {
            axis: {key: 0 for key in ("negative", "positive", "zero")}
            for axis in ("x", "y")
        }
        self.face_entities: set[str] = set()
        self.slope_entities: set[str] = set()
        self.violation_counts: dict[str, int] = {}
        self.minima: dict[str, dict[str, Any] | None] = {
            name: None for name in (
                "x_operand_over_q_scale", "x_active_gap_over_q_scale",
                "y_operand_over_q_scale", "y_active_gap_over_q_scale",
                "face_abs_over_global_max_flux",
            )
        }
        self.nonfinite = False

    def _minimum(self, name: str, ratio: Tensor, stage: int) -> None:
        if ratio.numel() == 0:
            return
        if not bool(torch.isfinite(ratio).all()):
            self.nonfinite = True
            return
        flat = ratio.reshape(-1)
        at = int(torch.argmin(flat))
        value = float(flat[at])
        old = self.minima[name]
        if old is None or value < old["value"]:
            self.minima[name] = {
                "value": value, "stage_index": stage,
                "stage_context": branch_oracle._stage_context(stage),
                "first_flat_index": at,
            }

    def __call__(self, q: Tensor, qx: Tensor, qy: Tensor) -> None:
        stage = self.stages
        self.stages += 1
        diagnostic = branch_oracle.classify_stage(stage, q, qx, qy)
        for violation in diagnostic.get("violations", []):
            name = str(violation["predicate"])
            self.violation_counts[name] = self.violation_counts.get(name, 0) + int(violation["count"])

        scale = q.abs().max()
        if not bool(torch.isfinite(scale)) or not bool(scale > 0):
            self.nonfinite = True
            return
        tol = 128 * torch.finfo(q.dtype).eps * scale
        records: dict[str, Any] = {"stage": stage, "x": [], "y": [], "qx": None, "qy": None}
        for axis, left, right in (
            ("x", q[1:-1, 1:-1] - q[1:-1, :-2],
             q[1:-1, 2:] - q[1:-1, 1:-1]),
            ("y", q[1:-1, 1:-1] - q[:-2, 1:-1],
             q[2:, 1:-1] - q[1:-1, 1:-1]),
        ):
            counts, selectors = _selector_counts(left, right, tol)
            for key, value in counts.items():
                self.selector_counts[axis][key] += value
            active = (((left > tol) & (right > tol)) | ((left < -tol) & (right < -tol)))
            operand_tie = (left.abs() <= tol) | (right.abs() <= tol)
            gap_tie = active & ((left - right).abs() <= tol)
            for row, col in torch.nonzero(operand_tie | gap_tie, as_tuple=False).tolist():
                # Slope tests cover only interior center cells, matching the strict oracle.
                self.slope_entities.add(f"{axis}:{row + 1}:{col + 1}")
            self._minimum(f"{axis}_operand_over_q_scale",
                          torch.minimum(left.abs(), right.abs()) / scale, stage)
            self._minimum(f"{axis}_active_gap_over_q_scale",
                          (left - right).abs()[active] / scale, stage)
            records[axis] = selectors.tolist()

        flux_scale = torch.cat((qx.reshape(-1), qy.reshape(-1))).abs().max()
        if not bool(torch.isfinite(flux_scale)) or not bool(flux_scale > 0):
            self.nonfinite = True
            return
        face_tol = 128 * torch.finfo(q.dtype).eps * flux_scale
        for axis, faces in (("x", qx), ("y", qy)):
            signs = faces.sign().to(torch.int8)
            self.face_sign_counts[axis]["negative"] += int((signs < 0).sum())
            self.face_sign_counts[axis]["positive"] += int((signs > 0).sum())
            self.face_sign_counts[axis]["zero"] += int((faces.abs() <= face_tol).sum())
            for row, col in torch.nonzero(faces.abs() <= face_tol, as_tuple=False).tolist():
                self.face_entities.add(f"{axis}:{row}:{col}")
            records[f"q{axis}"] = signs.tolist()
            self._minimum("face_abs_over_global_max_flux", faces.abs() / flux_scale, stage)
        self.digest.update(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())
        self.digest.update(b"\n")

    def report(self) -> dict[str, Any]:
        if self.stages != ANALYSIS_STAGES:
            classification = "incomplete"
        elif self.nonfinite:
            classification = "nonfinite"
        elif not self.face_entities and not self.slope_entities and not self.violation_counts:
            classification = "smooth_regular"
        elif len(self.face_entities) == 1 and not self.slope_entities:
            classification = "singleface"
        elif len(self.face_entities) > 1:
            classification = "multiplefaces"
        else:
            classification = "limiter_or_mixed_nonsmooth"
        return {
            "classification": classification,
            "stage_count": self.stages,
            "expected_analysis_stage_count": ANALYSIS_STAGES,
            "actual_selector_signature_sha256": self.digest.hexdigest(),
            "selector_counts": self.selector_counts,
            "face_sign_counts_over_stages": self.face_sign_counts,
            "unique_nearzero_face_entities": sorted(self.face_entities),
            "unique_near_tie_slope_entities": sorted(self.slope_entities),
            "violation_counts": self.violation_counts,
            "normalized_branch_margins": self.minima,
            "strict_pointwise_branch": classification == "smooth_regular",
        }


def _analysis_branch(problem: Any, control: Tensor, parameters: Tensor,
                     *, deadline: float) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        raise TimeoutError("qualification deadline before branch audit")
    audit = AnalysisBranchAudit()
    with torch.no_grad(), transport.observe_minmod_stages(audit):
        # The original objective traverses only its two observed 10-minute
        # intervals; future forecast stages do not define J or its Hessian.
        value = problem.objective(control, parameters)
    result = audit.report()
    result["native_J_on_audited_path"] = float(value) if bool(torch.isfinite(value)) else None
    result["objective_finite"] = bool(torch.isfinite(value))
    return result


def _analysis_branch_only(problem: Any, control: Tensor, parameters: Tensor,
                          *, deadline: float) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        raise TimeoutError("qualification deadline before branch stability probe")
    audit = AnalysisBranchAudit()
    with torch.no_grad(), transport.observe_minmod_stages(audit):
        v.analysis_trajectory(control, problem.contract(parameters))
    return audit.report()


def _small_step_stability(problem: Any, control: Tensor, parameters: Tensor,
                          reference: dict[str, Any], *, deadline: float) -> dict[str, Any]:
    """Check exact reached selectors at symmetric small coordinate FD probes.

    These are branch-stability probes only: they are not optimizer directions
    and do not contribute a gradient, Jacobian, or Hessian estimate.
    """
    reference_sha = reference["actual_selector_signature_sha256"]
    attempts: list[dict[str, Any]] = []
    for step in BRANCH_PROBE_STEPS:
        changed: list[dict[str, Any]] = []
        evaluated = 0
        for index in range(CONTROL_SIZE):
            h = step * max(1.0, abs(float(control[index])))
            for side in (-1, 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("qualification deadline during small-step branch probes")
                probe = control.clone()
                probe[index] += side * h
                result = _analysis_branch_only(problem, probe, parameters, deadline=deadline)
                evaluated += 1
                if (result["classification"] != "smooth_regular"
                        or result["actual_selector_signature_sha256"] != reference_sha):
                    changed.append({"coordinate": index, "side": side,
                                    "classification": result["classification"],
                                    "selector_signature_sha256": result["actual_selector_signature_sha256"]})
        attempts.append({"relative_step": step, "point_evaluations": evaluated,
                         "selector_changes": changed, "stable": not changed})
        if not changed:
            return {"status": "axiswise_stable", "first_stable_relative_step": step,
                    "domain": "all 26 coordinate axes, both signs; not a full-box certificate",
                    "attempts": attempts}
    return {"status": "no_stable_axiswise_step_in_tested_range",
            "domain": "all 26 coordinate axes, both signs; not a full-box certificate",
            "attempts": attempts}


def _candidate_value(record: dict[str, Any], start: str | None) -> tuple[dict[str, Any], Tensor]:
    selected = record
    if isinstance(record.get("starts"), dict):
        if start is None or start not in record["starts"]:
            raise QualificationRefusal("multi-start result requires an explicit --start")
        selected = record["starts"][start]
    control_value = selected.get("control", selected.get("best_control", selected.get("candidate_control")))
    if not isinstance(control_value, list) or len(control_value) != CONTROL_SIZE:
        raise QualificationRefusal("candidate JSON must contain an explicit native 26-control vector")
    control = torch.as_tensor(control_value, dtype=torch.float64, device="cpu")
    if control.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(control).all()):
        raise QualificationRefusal("candidate control must be finite CPU FP64 length 26")
    expected = selected.get("control_sha256", selected.get("best_control_sha256"))
    if expected is not None and expected != _tensor_sha(control):
        raise QualificationRefusal("candidate control digest does not match its receipt")
    if selected is not record and selected.get("status") != "completed":
        raise QualificationRefusal("selected native-J start is not completed")
    return selected, control


def _check_signed_sources(plan: dict[str, Any]) -> dict[str, str]:
    paths = {**plan["source_files"], **plan["archive_files"]}
    actual: dict[str, str] = {}
    for name, expected in paths.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise QualificationRefusal(f"pinned input is missing or symlinked: {name}")
        digest = _sha(path)
        if digest != expected:
            raise QualificationRefusal(f"native reference signed pin mismatch: {name}")
        actual[name] = digest
    return actual


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _execution_plan(*, continuation_plan_path: Path | None,
                    expected_continuation_sha256: str | None) -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
    """Validate the frozen Powell plan plus its sole approved transport transition."""
    base_sha = _sha(search.PLAN)
    if (base_sha != "b6f3a08e02ec6da684ea8ea7c9611e6eedf3b8de9de8a9ebc6830347dd13d9ac"
            or search.PLAN_SHA_FILE.read_text().strip() != base_sha):
        raise QualificationRefusal("frozen native-reference plan identity changed")
    base = json.loads(search.PLAN.read_text())
    parent_sha = _sha(search.SIGNED_PLAN)
    parent = json.loads(search.SIGNED_PLAN.read_text())
    if (parent_sha != search.SIGNED_PLAN_SHA
            or base.get("source_files") != parent.get("source_files")
            or base.get("parent_signed_span_plan_sha256") != search.SIGNED_PLAN_SHA
            or base.get("observer_source_sha256") != _sha(ROOT / search.SELF)):
        raise QualificationRefusal("native plan no longer inherits the sealed signed-span source set")
    before = base["source_files"]
    archives = base["archive_files"]
    transition = None
    extra_actual: dict[str, str] = {}
    if continuation_plan_path is None:
        if expected_continuation_sha256 is not None:
            raise QualificationRefusal("continuation digest supplied without continuation plan")
        actual_sources = dict(before)
    else:
        if expected_continuation_sha256 is None:
            raise QualificationRefusal("continuation plan requires a caller-pinned digest")
        if continuation_plan_path.is_symlink() or not continuation_plan_path.is_file():
            raise QualificationRefusal("continuation plan must be a regular nonsymlink file")
        if _sha(continuation_plan_path) != expected_continuation_sha256:
            raise QualificationRefusal("continuation plan digest mismatch")
        plan_sha_file = continuation_plan_path.with_name("CONTINUATION_PLAN_SHA256.txt")
        if not plan_sha_file.is_file() or plan_sha_file.read_text().strip() != expected_continuation_sha256:
            raise QualificationRefusal("continuation plan sidecar digest mismatch")
        data = json.loads(continuation_plan_path.read_text())
        if (data.get("base_native_plan_sha256") != base_sha
                and data.get("native_reference_plan_sha256") != base_sha
                and data.get("base_plan_sha256") != base_sha
                and data.get("parent_native_plan_sha256") != base_sha):
            raise QualificationRefusal("continuation plan does not preserve the frozen Powell plan")
        transitions = data.get("source_transition", data.get("transport_transition"))
        if not isinstance(transitions, dict) or set(transitions) != {"src/advar/transport.py"}:
            raise QualificationRefusal("only the approved support-kernel source transition may be loaded")
        transition = {"path": "src/advar/transport.py", **transitions["src/advar/transport.py"]}
        before_sha = transition.get("before_sha256", transition.get("before", transition.get("old_sha256")))
        after_sha = transition.get("after_sha256", transition.get("after", transition.get("new_sha256")))
        if (before_sha != before.get("src/advar/transport.py")
                or before_sha != FROZEN_TRANSPORT_SHA256):
            raise QualificationRefusal("continuation transition does not start at the frozen transport source")
        actual_transport = _sha(ROOT / "src/advar/transport.py")
        if after_sha != APPROVED_SUPPORT_TRANSPORT_SHA256 or after_sha != actual_transport:
            raise QualificationRefusal("current support-kernel hash is not the continuation-approved hash")
        transitioned_sources = dict(before)
        transitioned_sources["src/advar/transport.py"] = after_sha
        if data.get("source_files") != transitioned_sources:
            raise QualificationRefusal("continuation must change only the authorized transport source hash")
        extra_sources = data.get("extra_source_files")
        if not isinstance(extra_sources, dict):
            raise QualificationRefusal("continuation plan must pin its added audit sources")
        if data.get("source_member_count") != len(before) + len(extra_sources):
            raise QualificationRefusal("continuation source member count is inconsistent")
        for name, expected in extra_sources.items():
            path = ROOT / name
            if path.is_symlink() or not path.is_file() or _sha(path) != expected:
                raise QualificationRefusal(f"continuation audit source pin mismatch: {name}")
            extra_actual[f"extra_source:{name}"] = expected
        if (data.get("parent_source_count") != len(before)
                or data.get("archive_count_inherited") != len(archives)
                or data.get("parent_archive_count") != len(archives)):
            raise QualificationRefusal("continuation plan counts do not inherit the sealed native plan")
        transitioned_archives = data.get("archive_files")
        if not isinstance(transitioned_archives, dict) or any(
                transitioned_archives.get(name) != expected for name, expected in archives.items()):
            raise QualificationRefusal("continuation archive map changed inherited member hashes")
        if (data.get("archive_count_new") != len(transitioned_archives) - len(archives)
                or data.get("source_tar_base") != "original native plan sources with only transport.py updated"):
            raise QualificationRefusal("continuation archive/source transition scope mismatch")
        continuation_tar = continuation_plan_path.parent / "source.tar.gz"
        expected_tar = data.get("source_bundle_sha256")
        if not isinstance(expected_tar, str) or _sha(continuation_tar) != expected_tar:
            raise QualificationRefusal("continuation source bundle hash mismatch")
        actual_sources = dict(before)
        actual_sources["src/advar/transport.py"] = after_sha

    actual: dict[str, str] = {}
    for name, expected in actual_sources.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise QualificationRefusal(f"authorized source pin mismatch: {name}")
        actual[name] = expected
    actual.update(extra_actual)
    archives_to_verify = archives
    if transition is not None:
        continuation_data = json.loads(continuation_plan_path.read_text())
        archives_to_verify = continuation_data["archive_files"]
    for name, expected in archives_to_verify.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise QualificationRefusal(f"frozen archive pin mismatch: {name}")
        actual[f"archive:{name}"] = expected
    return base, actual, transition


def _legacy_qualify_candidate_file(candidate_path: Path, output_path: Path, *, start: str | None = None,
                                   total_seconds: float = INTERNAL_SECONDS) -> dict[str, Any]:
    if output_path.exists():
        raise QualificationRefusal("qualification output must be fresh")
    if not math.isfinite(total_seconds) or total_seconds <= 0 or total_seconds > INTERNAL_SECONDS:
        raise ValueError("total_seconds must be in (0,540]")
    started = time.monotonic()
    deadline = started + total_seconds
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise QualificationRefusal("candidate receipt must be a regular nonsymlink file")
    candidate_path = candidate_path.resolve(strict=True)
    candidate_sha_before = _sha(candidate_path)
    self_sha_before = _sha(SELF)
    plan_sha = _sha(search.PLAN)
    plan = search.load_plan(plan_sha)
    pinned_sources_before = _check_signed_sources(plan)
    case = cycle.load_fixed_case()
    problem, original = case["problem"], case["original"]
    parameters, truth = case["parameters"], case["truth"]
    candidate_record = json.loads(candidate_path.read_text())
    if not isinstance(candidate_record, dict):
        raise QualificationRefusal("candidate input must be a JSON object")
    selected, control = _candidate_value(candidate_record, start)
    if selected.get("plan_sha256", candidate_record.get("plan_sha256")) != plan_sha:
        raise QualificationRefusal("candidate is not bound to the completed native-J plan")
    if plan.get("parameters_sha256") != _tensor_sha(parameters):
        raise QualificationRefusal("native-J plan parameters differ from the pinned point case")
    if plan.get("terminal_truth_sha256") != _tensor_sha(truth):
        raise QualificationRefusal("native-J plan truth differs from the pinned point case")
    if selected.get("parameters_sha256", plan["parameters_sha256"]) != plan["parameters_sha256"]:
        raise QualificationRefusal("candidate receipt names different fixed parameters")
    if "parameters" in selected:
        candidate_parameters = torch.as_tensor(selected["parameters"], dtype=torch.float64)
        if not torch.equal(candidate_parameters, parameters):
            raise QualificationRefusal("candidate JSON parameters differ from the pinned case")

    input_before = cycle.shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = cycle.shared._runtime()
    record: dict[str, Any] = {
        "phase": "candidate_loaded", "status": "running",
        "candidate_json": str(candidate_path), "candidate_json_sha256": candidate_sha_before,
        "candidate_control_sha256": _tensor_sha(control),
        "candidate_status": selected.get("status", candidate_record.get("status")),
        "native_J_scope": "full 26-control problem.objective(c,p), fixed pinned p",
        "minimum_system": {"F": "grad_c J(c,p)", "F_c": "exact Hessian of original native J",
                           "B": "not computed; if smooth regular root, B=-F_c^{-1}F_p"},
        "source_plan_sha256": plan_sha,
        "signed_source_count": len(plan["source_files"]),
        "signed_archive_count": len(plan["archive_files"]),
        "input_before": input_before, "runtime_before": runtime_before,
        "optimizer_step_applied": False, "score_computed": False,
        "gradient_tolerance": STATIONARITY_TOL,
        "hvp_columns_completed": 0, "hvp_calls_total": 0,
        "control": control.tolist(),
        "old_Q_and_mixed_G": "not_applicable; no global Q=0 or cached mixed gradient used",
    }
    _write(output_path, record)

    base_control = control.clone()
    j_value = problem.objective(control, parameters)
    if j_value.shape != () or not bool(torch.isfinite(j_value)):
        record.update(phase="finished", status="function_invalid",
                      refusal="native objective is not finite scalar")
        _write(output_path, record)
        return record
    base_branch = _analysis_branch(problem, control, parameters, deadline=deadline)
    if not base_branch["objective_finite"]:
        raise QualificationRefusal("native J became nonfinite on its observed-stage audit")
    native_j = float(j_value)
    if base_branch["native_J_on_audited_path"] != native_j:
        budget = 128 * torch.finfo(torch.float64).eps * max(abs(native_j), 1.0)
        if abs(base_branch["native_J_on_audited_path"] - native_j) > budget:
            raise QualificationRefusal("native J differs between scalar and stage-audited evaluation")
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    gradient = gradient_fn(control, parameters)
    if gradient.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(gradient).all()):
        record.update(phase="finished", status="gradient_invalid", native_J=native_j,
                      branch=base_branch, control=base_control.tolist())
        _write(output_path, record)
        return record
    grad_inf = float(gradient.abs().max())
    record.update(phase="native_point_measured", native_J=native_j,
                  native_gradient=gradient.tolist(), gradient_inf=grad_inf,
                  branch=base_branch, control=base_control.tolist())
    _write(output_path, record)

    if base_branch["classification"] != "smooth_regular":
        record.update(phase="finished", status="unsupported_active_qualification",
                      qualification_reason="pointwise nonsmooth selector(s); no generic full-26 active-set KKT/reduced-Hessian implementation",
                      function_valid_candidate_retained=True, native_J=native_j,
                      control=base_control.tolist(), gradient_scope="autograd branch value; not an active-set stationarity certificate",
                      minimum_claim=False, response_computed=False)
    else:
        base_signature = base_branch["actual_selector_signature_sha256"]
        stability = _small_step_stability(problem, control, parameters, base_branch, deadline=deadline)
        record["small_step_branch_stability"] = stability
        if grad_inf >= STATIONARITY_TOL:
            def bounded_objective(c: Tensor, p: Tensor) -> Tensor:
                if time.monotonic() >= deadline:
                    raise TimeoutError("qualification budget exhausted in local refinement")
                return problem.objective(c, p)

            def actual_branch_check(c: Tensor, p: Tensor) -> tuple[Any, str]:
                detail = _analysis_branch(problem, c, p, deadline=deadline)
                if (detail["classification"] != "smooth_regular"
                        or detail["actual_selector_signature_sha256"] != base_signature):
                    raise ValueError("local refinement left the candidate's actual smooth branch")
                return detail, "actual native-J analysis branch; no frozen expected-branch identity"

            try:
                refined = refine_stationary(
                    bounded_objective, control, parameters,
                    branch_check=actual_branch_check,
                    max_iterations=8, max_backtracks=16,
                    pcg_max_iterations=4 * CONTROL_SIZE,
                )
                control = refined.control
                gradient = gradient_fn(control, parameters)
                native_j = float(problem.objective(control, parameters))
                grad_inf = float(gradient.abs().max())
                record["refinement"] = {
                    "status": "returned", "iterations": refined.iterations,
                    "hvp_calls": refined.hvp_count,
                    "history": refined.history,
                    "refined_control_sha256": _tensor_sha(control),
                }
                record.update(native_J=native_j, native_gradient=gradient.tolist(),
                              gradient_inf=grad_inf, control=control.tolist())
            except (RefinementNumericalRefusal, TimeoutError, ValueError) as error:
                record["refinement"] = {"status": "refused", "reason": str(error)}

        if time.monotonic() >= deadline:
            record.update(phase="finished", status="qualification_budget_refusal",
                          refusal="reserved internal qualification budget exhausted before exact Hessian",
                          function_valid_candidate_retained=True, minimum_claim=False)
        else:
            # Re-audit the actual selectors after local refinement and before HVP columns.
            final_branch = _analysis_branch(problem, control, parameters, deadline=deadline)
            record["final_branch_before_hessian"] = final_branch
            if (final_branch["classification"] != "smooth_regular"
                    or final_branch["actual_selector_signature_sha256"] != base_signature):
                record.update(phase="finished", status="branch_changed_before_hessian",
                              function_valid_candidate_retained=True, minimum_claim=False)
            else:
                stability = _small_step_stability(problem, control, parameters, final_branch, deadline=deadline)
                record["small_step_branch_stability_after_refinement"] = stability
                current_gradient = gradient_fn(control, parameters)
                if float(current_gradient.abs().max()) >= STATIONARITY_TOL:
                    record.update(phase="finished", status="stationarity_not_reached",
                                  qualification_reason="existing local refinement returned without the unchanged full-gradient threshold",
                                  function_valid_candidate_retained=True, minimum_claim=False,
                                  native_J=float(problem.objective(control, parameters)),
                                  native_gradient=current_gradient.tolist(),
                                  gradient_inf=float(current_gradient.abs().max()),
                                  control=control.tolist())
                elif stability["status"] != "axiswise_stable":
                    record.update(phase="finished", status="branch_stability_support_refusal",
                                  qualification_reason="no stable actual selector signature across tested symmetric coordinate probes",
                                  function_valid_candidate_retained=True, minimum_claim=False,
                                  native_J=float(problem.objective(control, parameters)),
                                  native_gradient=current_gradient.tolist(),
                                  gradient_inf=float(current_gradient.abs().max()),
                                  control=control.tolist())
                else:
                    eye = torch.eye(CONTROL_SIZE, dtype=control.dtype)
                    columns: list[Tensor] = []
                    hessian_started = time.monotonic()
                    for index in range(CONTROL_SIZE):
                        if time.monotonic() >= deadline:
                            break
                        column = torch.func.jvp(
                            lambda c: gradient_fn(c, parameters),
                            (control,), (eye[index],),
                        )[1]
                        if column.shape != (CONTROL_SIZE,) or not bool(torch.isfinite(column).all()):
                            raise QualificationRefusal("exact native-J HVP column is nonfinite or malformed")
                        columns.append(column)
                        record.update(phase="exact_hessian_columns",
                                      hvp_columns_completed=len(columns),
                                      hvp_calls_total=len(columns))
                        _write(output_path, record)
                    if len(columns) != CONTROL_SIZE or time.monotonic() >= deadline:
                        record.update(phase="finished", status="qualification_budget_refusal",
                                      refusal="internal budget ended during exact 26-column Hessian",
                                      function_valid_candidate_retained=True, minimum_claim=False,
                                      hvp_columns_completed=len(columns), hessian_seconds=time.monotonic()-hessian_started)
                    else:
                        hessian = torch.stack(columns, dim=1)
                        hnorm = torch.linalg.matrix_norm(hessian)
                        symmetry = float(torch.linalg.matrix_norm(hessian-hessian.T) /
                                         torch.clamp_min(hnorm, torch.finfo(hessian.dtype).tiny))
                        symmetric = 0.5 * (hessian + hessian.T)
                        eig, vectors = torch.linalg.eigh(symmetric)
                        scale = float(eig.abs().max())
                        floor = max(blocks.CURVATURE_REL_TOL * scale,
                                    blocks.ROUNDING_MULT * torch.finfo(hessian.dtype).eps *
                                    max(float(hnorm), torch.finfo(hessian.dtype).tiny))
                        minimum_audit = torch.func.jvp(
                            lambda c: gradient_fn(c, parameters),
                            (control,), (vectors[:, 0],),
                        )[1]
                        eigenpair = blocks.eigenpair_audit(
                            minimum_audit, eig[0], vectors[:, 0], hnorm,
                        )
                        schur = blocks.schur_diagnostic(hessian, current_gradient)
                        schur_eig = torch.as_tensor(schur.get("schur_eigenvalues") or [], dtype=hessian.dtype)
                        schur_norm = float(torch.linalg.vector_norm(schur_eig)) if schur_eig.numel() else 0.0
                        schur_floor = max(blocks.CURVATURE_REL_TOL * float(schur_eig.abs().max()) if schur_eig.numel() else 0.0,
                                          blocks.ROUNDING_MULT * torch.finfo(hessian.dtype).eps *
                                          max(schur_norm, torch.finfo(hessian.dtype).tiny))
                        reduced_positive = (schur.get("status") == "schur_evaluated"
                                            and schur_eig.numel() == CONTROL_SIZE - 20
                                            and bool(schur_eig.min() > schur_floor))
                        after_branch = _analysis_branch(problem, control, parameters, deadline=deadline)
                        branch_stable = (
                            after_branch["classification"] == "smooth_regular"
                            and after_branch["actual_selector_signature_sha256"] == base_signature
                        )
                        pd = (math.isfinite(symmetry) and symmetry <= blocks.SYMMETRY_TOL
                              and bool(eig[0] > floor) and eigenpair["passed"])
                        minimum_claim = (
                            grad_inf < STATIONARITY_TOL and pd and reduced_positive and branch_stable
                            and stability["status"] == "axiswise_stable"
                        )
                        record.update(phase="finished",
                            status="smooth_local_minimum_qualified" if minimum_claim else "smooth_point_not_minimum_qualified",
                            minimum_system={"F": "grad_c J(c,p)", "F_c": "exact native-J Hessian",
                                "B": "not computed; smooth implicit system is H B + F_p = 0"},
                            native_J=float(problem.objective(control, parameters)),
                            native_gradient=current_gradient.tolist(),
                            gradient_inf=grad_inf, control=control.tolist(),
                            branch_after_hessian=after_branch, branch_stable=branch_stable,
                            hessian=hessian.tolist(), hessian_sha256=_tensor_sha(hessian),
                            hessian_symmetry_relative=symmetry,
                            hessian_eigenvalues=eig.tolist(),
                            positive_curvature_floor=floor,
                            full_hessian_positive_definite=pd,
                            reduced_curvature_passed=reduced_positive,
                            reduced_curvature_floor=schur_floor,
                            minimum_eigenpair_audit=eigenpair,
                            reduced_block_schur=schur,
                            exact_hvp_columns=CONTROL_SIZE, exact_hvp_calls=CONTROL_SIZE + 1,
                            hessian_seconds=time.monotonic()-hessian_started,
                            stationarity_passed=grad_inf < STATIONARITY_TOL,
                            small_step_branch_stability=stability,
                            minimum_claim=minimum_claim,
                            response_computed=False,
                            scope="one smooth actual analysis branch; local full-space Hessian only; no global or forecast-minimum claim")
    # Source, case, candidate, runtime, and full identity closure surrounds all
    # qualification work. It deliberately does not compare to a historical
    # endpoint control or require its expected branch signature.
    pinned_sources_after = _check_signed_sources(plan)
    input_after = cycle.shared._input_identity(problem, original, control, parameters, truth)
    runtime_after = cycle.shared._runtime()
    closure = {
        "signed_source_and_archive_hashes_unchanged": pinned_sources_before == pinned_sources_after,
        "candidate_json_unchanged": candidate_sha_before == _sha(candidate_path),
        "qualification_source_unchanged": self_sha_before == _sha(SELF),
        "fixed_parameters_and_problem_unchanged": input_before.get("parameters_sha256") == input_after.get("parameters_sha256")
            and input_before.get("terminal_truth_sha256") == input_after.get("terminal_truth_sha256")
            and input_before.get("problem_sha256") == input_after.get("problem_sha256"),
        "runtime_unchanged": runtime_before == runtime_after,
    }
    record.update(source_hash_count=len(pinned_sources_after), closure=closure,
                  input_after=input_after, runtime_after=runtime_after,
                  elapsed_seconds=time.monotonic()-started,
                  optimizer_step_applied=False, score_computed=False)
    if not all(closure.values()):
        record.update(status="provenance_closure_failed", minimum_claim=False)
    _write(output_path, record)
    return record


def qualify(control: Tensor, parameters: Tensor, *, remaining_seconds: float,
            expected_parameters_sha256: str | None = None,
            continuation_plan_path: Path | None = None,
            expected_continuation_sha256: str | None = None,
            full_hessian: bool = False,
            checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Measure native J/F and optionally checkpoint the exact 26-column J-Hessian.

    The only permitted source transition is the caller-pinned continuation
    plan's transport support fix. Caller-supplied p requires its own digest.
    """
    if (not math.isfinite(remaining_seconds) or remaining_seconds <= 0
            or remaining_seconds > 600.0):
        raise ValueError("remaining_seconds must be in (0,600]")
    if type(full_hessian) is not bool:
        raise TypeError("full_hessian must be a bool")
    if full_hessian and checkpoint_path is None:
        raise QualificationRefusal("full Hessian requires a per-column checkpoint path")
    if (control.shape != (CONTROL_SIZE,) or control.dtype is not torch.float64
            or control.device.type != "cpu" or not bool(torch.isfinite(control).all())):
        raise QualificationRefusal("control must be finite CPU FP64 length 26")
    if (parameters.shape != (PARAMETERS_SIZE,) or parameters.dtype is not torch.float64
            or parameters.device.type != "cpu" or not bool(torch.isfinite(parameters).all())):
        raise QualificationRefusal("parameters must be finite CPU FP64 length 13")
    started = time.monotonic()
    deadline = started + min(remaining_seconds, INTERNAL_SECONDS)
    qualification_source_sha = _sha(SELF)
    plan, source_before, source_transition = _execution_plan(
        continuation_plan_path=continuation_plan_path,
        expected_continuation_sha256=expected_continuation_sha256,
    )
    plan_sha = _sha(search.PLAN)
    case = cycle.load_fixed_case()
    problem, original = case["problem"], case["original"]
    fixed_parameters, truth = case["parameters"], case["truth"]
    parameters_sha = _tensor_sha(parameters)
    if expected_parameters_sha256 is None:
        if parameters_sha != plan["parameters_sha256"]:
            raise QualificationRefusal("arbitrary parameters require caller-pinned expected_parameters_sha256")
        expected_parameters_sha256 = plan["parameters_sha256"]
    if parameters_sha != expected_parameters_sha256 or _tensor_sha(truth) != plan["terminal_truth_sha256"]:
        raise QualificationRefusal("caller parameters or fixed truth differ from their sealed digests")
    input_before = cycle.shared._input_identity(problem, original, control, parameters, truth)
    runtime_before = cycle.shared._runtime()

    audit = AnalysisBranchAudit()
    with torch.no_grad(), transport.observe_minmod_stages(audit):
        native_j_tensor = problem.objective(control, parameters)
    branch = audit.report()
    branch["native_J_on_audited_path"] = float(native_j_tensor) if bool(torch.isfinite(native_j_tensor)) else None
    branch["objective_finite"] = bool(torch.isfinite(native_j_tensor))
    if (native_j_tensor.shape != () or not bool(torch.isfinite(native_j_tensor))
            or not branch["objective_finite"]):
        raise QualificationRefusal("native point objective is not a finite scalar")
    if time.monotonic() >= deadline:
        raise TimeoutError("60-second early qualification deadline during native objective")
    gradient_fn = torch.func.grad(problem.objective, argnums=0)
    gradient = gradient_fn(control, parameters)
    if gradient.shape != control.shape or not bool(torch.isfinite(gradient).all()):
        raise QualificationRefusal("native objective gradient is invalid")
    gradient_inf = float(gradient.abs().max())
    result: dict[str, Any] = {
        "phase": "native_point_measured",
        "status": "unsupported_active_qualification" if branch["classification"] != "smooth_regular"
            else "smooth_stationary_pending_curvature" if gradient_inf < STATIONARITY_TOL
            else "smooth_nonstationary_candidate",
        "control_sha256": _tensor_sha(control), "parameters_sha256": parameters_sha,
        "native_J": float(native_j_tensor), "native_gradient": gradient.tolist(),
        "gradient_inf": gradient_inf, "gradient_tolerance": STATIONARITY_TOL,
        "branch": branch, "minimum_claim": False,
        "source_plan_sha256": plan_sha,
        "continuation_plan_sha256": expected_continuation_sha256,
        "source_transition": source_transition,
        "hvp_calls_started": 0, "hvp_calls_completed": 0,
        "hvp_columns_started": 0, "hvp_columns_completed": 0,
        "score_computed": False, "optimizer_step_applied": False,
    }
    if full_hessian:
        checkpoint_identity = {
            "control_sha256": _tensor_sha(control), "parameters_sha256": parameters_sha,
            "source_plan_sha256": plan_sha,
            "continuation_plan_sha256": expected_continuation_sha256,
            "branch_signature_sha256": branch["actual_selector_signature_sha256"],
            "qualification_source_sha256": qualification_source_sha,
        }
        result.update(checkpoint_identity, columns=[], hvp_events=[])
        _write(Path(checkpoint_path), result)
    if full_hessian and branch["classification"] != "smooth_regular":
        result.update(function_valid_candidate_retained=True,
            qualification_reason="nonsmooth native objective selectors; no generic active-set KKT/Hessian was constructed")
    elif full_hessian:
        if continuation_plan_path is None or expected_continuation_sha256 is None:
            raise QualificationRefusal("full Hessian requires the explicitly approved continuation plan")
        if checkpoint_path is None:
            raise QualificationRefusal("full Hessian requires a per-column checkpoint path")
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_symlink():
            raise QualificationRefusal("Hessian checkpoint path may not be a symlink")
        identity = checkpoint_identity
        if checkpoint_path.is_symlink():
            raise QualificationRefusal("Hessian checkpoint path may not be a symlink")
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text())
            if any(saved.get(key) != value for key, value in identity.items()):
                raise QualificationRefusal("partial Hessian checkpoint is bound to a different point or source transition")
            columns = saved.get("columns")
            if not isinstance(columns, list) or len(columns) > CONTROL_SIZE:
                raise QualificationRefusal("partial Hessian checkpoint columns are malformed")
            hessian_columns = [torch.as_tensor(column, dtype=control.dtype) for column in columns]
            if any(column.shape != control.shape or not bool(torch.isfinite(column).all())
                   for column in hessian_columns):
                raise QualificationRefusal("partial Hessian checkpoint contains an invalid returned column")
            result.update(saved)
            interrupted = result.get("current_hvp")
            if isinstance(interrupted, dict) and interrupted.get("status") == "started":
                result.setdefault("hvp_events", []).append({
                    "column": interrupted.get("column"), "status": "unreturned_before_resume",
                    "call": interrupted.get("call"),
                })
                result["current_hvp"] = None
        else:
            hessian_columns = []
            result.update(identity, columns=[], hvp_events=[])
            _write(checkpoint_path, result)
        eye = torch.eye(CONTROL_SIZE, dtype=control.dtype)
        for index in range(len(hessian_columns), CONTROL_SIZE):
            if time.monotonic() >= deadline:
                result.update(phase="finished", status="hessian_budget_refusal",
                              refusal="remaining shared qualification budget expired before another HVP")
                break
            call_index = int(result.get("hvp_calls_started", 0)) + 1
            result.update(phase="hessian_column_started", hvp_calls_started=call_index,
                          hvp_columns_started=index + 1,
                          current_hvp={"column": index, "status": "started", "call": call_index})
            result.setdefault("hvp_events", []).append({"column": index, "status": "started", "call": call_index})
            _write(checkpoint_path, result)
            try:
                column = torch.func.jvp(
                    lambda c: gradient_fn(c, parameters), (control,), (eye[index],),
                )[1]
                if column.shape != control.shape or not bool(torch.isfinite(column).all()):
                    raise QualificationRefusal("exact native-J HVP column is nonfinite or malformed")
            except Exception as error:
                result.update(phase="finished", status="hessian_column_refusal",
                    current_hvp={"column": index, "status": "failed",
                                 "error": f"{type(error).__name__}: {error}"})
                result.setdefault("hvp_events", []).append({"column": index, "status": "failed",
                    "error": f"{type(error).__name__}: {error}"})
                _write(checkpoint_path, result)
                break
            hessian_columns.append(column)
            result["columns"] = [value.tolist() for value in hessian_columns]
            result["hvp_calls_completed"] = int(result.get("hvp_calls_completed", 0)) + 1
            result["hvp_columns_completed"] = len(hessian_columns)
            result["current_hvp"] = None
            result.setdefault("hvp_events", []).append({"column": index, "status": "returned",
                "call": result["hvp_calls_completed"]})
            _write(checkpoint_path, result)
        if len(hessian_columns) == CONTROL_SIZE:
            hessian = torch.stack(hessian_columns, dim=1)
            hnorm = torch.linalg.matrix_norm(hessian)
            symmetry = float(torch.linalg.matrix_norm(hessian-hessian.T) /
                             torch.clamp_min(hnorm, torch.finfo(hessian.dtype).tiny))
            eigenvalues = torch.linalg.eigvalsh(0.5 * (hessian+hessian.T))
            scale = float(eigenvalues.abs().max())
            floor = max(blocks.CURVATURE_REL_TOL * scale,
                        blocks.ROUNDING_MULT * torch.finfo(hessian.dtype).eps *
                        max(float(hnorm), torch.finfo(hessian.dtype).tiny))
            pd = math.isfinite(symmetry) and symmetry <= blocks.SYMMETRY_TOL and bool(eigenvalues[0] > floor)
            schur = blocks.schur_diagnostic(hessian, gradient)
            schur_eigenvalues = torch.as_tensor(schur.get("schur_eigenvalues") or [], dtype=hessian.dtype)
            schur_floor = max(blocks.CURVATURE_REL_TOL * float(schur_eigenvalues.abs().max())
                              if schur_eigenvalues.numel() else 0.0,
                              blocks.ROUNDING_MULT * torch.finfo(hessian.dtype).eps *
                              max(float(torch.linalg.vector_norm(schur_eigenvalues))
                                  if schur_eigenvalues.numel() else 0.0,
                                  torch.finfo(hessian.dtype).tiny))
            reduced_pd = (schur.get("status") == "schur_evaluated"
                          and schur_eigenvalues.numel() == CONTROL_SIZE-20
                          and bool(schur_eigenvalues.min() > schur_floor))
            branch_after = _analysis_branch(problem, control, parameters, deadline=deadline)
            branch_stable = (branch_after["classification"] == "smooth_regular"
                             and branch_after["actual_selector_signature_sha256"]
                             == branch["actual_selector_signature_sha256"])
            stationarity = gradient_inf < STATIONARITY_TOL
            minimum = stationarity and pd and reduced_pd and branch_stable
            result.update(phase="finished",
                status="smooth_local_minimum_qualified" if minimum else "smooth_hessian_complete_nonstationary_or_nonminimum",
                native_J=float(native_j_tensor), native_gradient=gradient.tolist(), gradient_inf=gradient_inf,
                hessian=hessian.tolist(), hessian_sha256=_tensor_sha(hessian),
                hessian_symmetry_relative=symmetry, hessian_eigenvalues=eigenvalues.tolist(),
                positive_curvature_floor=floor, full_hessian_positive_definite=pd,
                reduced_block_schur=schur, reduced_curvature_floor=schur_floor,
                reduced_curvature_positive_definite=reduced_pd,
                branch_after_hessian=branch_after, branch_stable=branch_stable,
                stationarity_passed=stationarity, minimum_claim=minimum,
                exact_hvp_columns=CONTROL_SIZE, exact_hvp_calls=CONTROL_SIZE,
                response_computed=False,
                scope="exact full-space native-J Hessian at the actual smooth 360-stage analysis branch; no global or forecast-minimum claim")
            _write(checkpoint_path, result)

    source_after = _execution_plan(
        continuation_plan_path=continuation_plan_path,
        expected_continuation_sha256=expected_continuation_sha256,
    )[1]
    runtime_after = cycle.shared._runtime()
    input_after = cycle.shared._input_identity(problem, original, control, parameters, truth)
    closure = {"authorized_source_and_archive_set_unchanged": source_before == source_after,
               "input_identity_unchanged": input_before == input_after,
               "runtime_identity_unchanged": runtime_before == runtime_after,
               "qualification_source_unchanged": qualification_source_sha == _sha(SELF)}
    result.update(closure=closure, input_before=input_before, input_after=input_after,
        runtime_before=runtime_before, runtime_after=runtime_after,
        signed_source_count=len(plan["source_files"]), signed_archive_count=len(plan["archive_files"]),
        qualification_source_sha256=qualification_source_sha,
        elapsed_seconds=time.monotonic()-started, score_computed=False,
        optimizer_step_applied=False,
        minimum_system={"F": "grad_c J(c,p)", "F_c": "exact native-J Hessian",
                        "B": "not computed; smooth implicit system is Hcc B + F_p = 0"},
        old_Q_and_mixed_G="not_applicable")
    if not all(closure.values()):
        result.update(status="provenance_closure_failed", minimum_claim=False)
    if full_hessian and checkpoint_path is not None:
        _write(Path(checkpoint_path), result)
    return result


def qualify_candidate_file(candidate_path: Path, output_path: Path, *, start: str | None = None,
                           remaining_seconds: float = INTERNAL_SECONDS,
                           continuation_plan_path: Path | None = None,
                           expected_continuation_sha256: str | None = None,
                           expected_parameters_sha256: str | None = None,
                           full_hessian: bool = True) -> dict[str, Any]:
    """Load an explicit search receipt and route it through the current qualifier."""
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise QualificationRefusal("candidate receipt must be a regular nonsymlink file")
    record = json.loads(candidate_path.read_text())
    if not isinstance(record, dict):
        raise QualificationRefusal("candidate receipt must be a JSON object")
    selected, control = _candidate_value(record, start)
    case = cycle.load_fixed_case()
    parameters = torch.as_tensor(selected.get("parameters", case["parameters"]), dtype=torch.float64)
    expected_p = (expected_parameters_sha256 or selected.get("parameters_sha256")
                  or _tensor_sha(parameters))
    plan_digest = selected.get("plan_sha256", record.get("plan_sha256"))
    if plan_digest != _sha(search.PLAN):
        raise QualificationRefusal("candidate receipt is not bound to the current frozen native-J plan")
    result = qualify(
        control, parameters, remaining_seconds=remaining_seconds,
        expected_parameters_sha256=expected_p,
        continuation_plan_path=continuation_plan_path,
        expected_continuation_sha256=expected_continuation_sha256,
        full_hessian=full_hessian,
        checkpoint_path=output_path if full_hessian else None,
    )
    if not full_hessian:
        _write(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", choices=("fb0a", "zero"), default=None,
                        help="required for a multi-start search receipt")
    parser.add_argument("--total-seconds", type=float, default=INTERNAL_SECONDS)
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--continuation-sha256", required=True)
    parser.add_argument("--parameters-sha256")
    args = parser.parse_args()
    qualify_candidate_file(args.candidate_json, args.output, start=args.start,
        remaining_seconds=args.total_seconds, continuation_plan_path=args.continuation_plan,
        expected_continuation_sha256=args.continuation_sha256,
        expected_parameters_sha256=args.parameters_sha256, full_hessian=True)


if __name__ == "__main__":
    main()
