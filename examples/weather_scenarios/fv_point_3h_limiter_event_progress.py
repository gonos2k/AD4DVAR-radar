"""Locate the observed limiter events, advance native J, and audit the endpoint.

Only an actually active positive-slope equality is used for constrained search.
Its Lagrangian model is a search model, not a nonsmooth minimum certificate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import resource
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012"
OUT = HERE / "event_progress_20261015/attempt1"
ENDPOINT = HERE / "two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json"
PRIOR = HERE / "limiter_ray_20261014"
STAGES = (47, 48)
FACES = (("y", 4, 3), ("y", 2, 1))
SECTORS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
EPS = torch.finfo(torch.float64).eps


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def positive_gap_active(left: float, right: float, scale: float) -> bool:
    return left > 0 and right > 0 and abs(right - left) <= 128 * EPS * scale


def bisect_gap(evaluate, stage: int, lower: float, upper: float):
    """Bracket a positive-operand equality using actual stage values."""
    a, b = [evaluate(alpha)["events"][str(stage)] for alpha in (lower, upper)]
    if min(a["left"], a["right"], b["left"], b["right"]) <= 0:
        raise ValueError("event bracket left its positive operand domain")
    left, right = a["gap"], b["gap"]
    if left <= 0 or right >= 0:
        raise ValueError("identified event lacks its positive-to-negative bracket")
    for _ in range(64):
        middle = (lower + upper) / 2
        row = evaluate(middle)
        event = row["events"][str(stage)]
        if positive_gap_active(event["left"], event["right"], event["q_scale"]):
            return middle, row, [lower, upper]
        if event["left"] <= 0 or event["right"] <= 0:
            raise ValueError("identified event left its positive slope domain")
        if event["gap"] > 0:
            lower = middle
        else:
            upper = middle
    raise ValueError("event equality not resolved after sixty-four bisections")


def event_newton_model(g: Tensor, hessian: Tensor, normal: Tensor | None,
                       event_hessian: Tensor | None) -> tuple[Tensor, dict[str, Any]]:
    """Form Qᵀ(H_J+λH_h)Q with λ fixed at the current point."""
    size = g.numel()
    multiplier = 0.0
    q = torch.eye(size, dtype=g.dtype)
    lagrangian = hessian
    if normal is not None:
        if not bool(torch.isfinite(normal).all()) or float(normal.norm()) <= torch.finfo(g.dtype).tiny:
            raise ValueError("event has no resolved chart normal")
        _, _, vh = torch.linalg.svd(normal[None], full_matrices=True)
        q = vh[1:].T
        multiplier = -float(normal @ g / (normal @ normal))
        if event_hessian is None:
            raise ValueError("event model requires the Hessian of its equality")
        lagrangian = hessian + multiplier * event_hessian
    if not bool(torch.isfinite(lagrangian).all()):
        raise ValueError("nonfinite current Lagrangian Hessian")
    symmetric = (lagrangian + lagrangian.T) / 2
    scale = max(float(symmetric.norm()), torch.finfo(g.dtype).tiny)
    symmetry = float((lagrangian - lagrangian.T).norm()) / scale
    if symmetry > 1e-8:
        raise ValueError("current Lagrangian Hessian is not numerically symmetric")
    reduced = q.T @ symmetric @ q
    eig = torch.linalg.eigvalsh(reduced)
    floor = max(128 * EPS * float(reduced.norm()), 1e-8 * float(eig.abs().max()))
    shift = max(0.0, floor - float(eig[0]))
    direction = q @ torch.linalg.solve(reduced + shift * torch.eye(q.shape[1], dtype=g.dtype), -(q.T @ g))
    length = float(direction.norm())
    if length > 0.01:
        direction *= 0.01 / length
    return direction, {"multiplier": multiplier, "projected_gradient_inf": float((q.T @ g).abs().max()),
        "lambda_min": float(eig[0]), "positive_floor": floor, "shift": shift,
        "symmetry_relative": symmetry, "tangent_dimension": q.shape[1],
        "normal_step_dot": None if normal is None else float(normal @ direction),
        "hessian": hessian.tolist(), "event_hessian": None if event_hessian is None else event_hessian.tolist(),
        "reduced_lagrangian_hessian": reduced.tolist(), "slope": float(g @ direction),
        "interpretation": "same-point selected AD branch and fixed multiplier; search model only"}


def supported_transition(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    a = bytes.fromhex(before["exact_minmod_selectors_hex"])
    b = bytes.fromhex(after["exact_minmod_selectors_hex"])
    if len(a) != 4320 or len(b) != 4320:
        raise ValueError("incomplete exact selector trace")
    changes = []
    for index, (old, new) in enumerate(zip(a, b)):
        if old == new:
            continue
        stage, interior = divmod(index, 12)
        axis, cell = divmod(interior, 6)
        row, col = divmod(cell, 3)
        change = {"stage": stage, "axis": "x" if axis == 0 else "y", "cell": [row+1, col+1],
                  "before": old, "after": new}
        changes.append(change)
        if not (stage in STAGES and axis == 0 and cell == 0 and old in (1, 2, 3) and new in (1, 2, 3)):
            raise ValueError(f"unsupported intervening limiter transition: {change}")
    for index, (old, new) in enumerate(zip(before["flux_signs"], after["flux_signs"])):
        if old != new and index not in (35, 47):
            raise ValueError(f"unsupported intervening flux-face transition at flattened index {index}")
    return changes


def close_claims(state: dict[str, Any], sources_ok: bool, inputs_ok: bool) -> None:
    state.update(source_unchanged=sources_ok, inputs_unchanged=inputs_ok)
    if not (sources_ok and inputs_ok):
        state.update(minimum_claim=False, status="source_or_input_closure_failed")
        if "terminal" in state:
            state["terminal"].update(minimum_claim=False, P2_complete=False,
                unresolved_reason="source/input closure failed; mathematical qualification is not publishable")


class Budget:
    def __init__(self, state: dict[str, Any], start: float):
        self.state, self.start = state, start

    def take(self, kind: str) -> None:
        budget = self.state["plan"]["budget"]
        reserve = 0 if self.state.get("in_terminal") else 120
        if time.monotonic() - self.start >= budget["additional_wall_seconds"] - reserve:
            raise TimeoutError("new phase wall budget exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss > budget["RSS_cap_bytes"]:
            raise MemoryError("new phase RSS budget exhausted")
        reserves = {"direct_J": 2, "gradient": 12, "Hessian_row": 144, "profile_forward": 12}
        allocation = budget[kind + "_cap"] - (0 if self.state.get("in_terminal") else reserves[kind])
        if self.state["counts"][kind + "_started"] >= allocation:
            raise RuntimeError(kind + " additional budget exhausted")
        self.state["counts"][kind + "_started"] += 1
        write(OUT / "RESULT.json", self.state)

    def done(self, kind: str) -> None:
        self.state["counts"][kind + "_completed"] += 1


class Execution:
    def __init__(self, case, q, geo, base, control, state, budget):
        self.problem, self.parameters = case["problem"], case["parameters"]
        self.q, self.geo = q, geo
        self.profile, identity = base._profile_problem(self.problem)
        self.state, self.budget = state, budget
        self.state["profile_identity"] = identity
        fv = self.problem.frozen.fv_transport
        matrix, pivots, free, metadata = geo._chart(fv.coefficient_limits, fv.psi_basis)
        self.lift, self.initial, projected, _ = geo._make_intersection_chart(control, matrix, pivots, free)
        if not torch.equal(projected, control):
            raise ValueError("current control does not roundtrip the two-face chart")
        self.state["chart"] = metadata

    def value(self, z: Tensor, purpose: str) -> dict[str, Any]:
        from advar import transport
        audit = self.q.AnalysisBranchAudit()
        events: dict[str, Any] = {}
        active = []
        exact_trace = bytearray()
        flux_signs = []
        counter = 0

        def capture(q, qx, qy):
            nonlocal counter
            audit(q, qx, qy)
            if q.shape != (4, 5):
                raise ValueError("event progress requires its fixed 4x5 grid")
            if counter == 0:
                flux_signs.extend(torch.cat((qx.flatten(), qy.flatten())).sign().to(torch.int8).tolist())
            scale = float(q.abs().max())
            tol = 128 * EPS * scale
            for axis, left, right in (
                ("x", q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]),
                ("y", q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]),
            ):
                from examples.weather_scenarios.fv_point_3h_limiter_ray_diagnostic import exact_selectors
                exact_trace.extend(exact_selectors(left, right).contiguous().numpy().tobytes())
                same = ((left > tol) & (right > tol)) | ((left < -tol) & (right < -tol))
                tied = (same & ((right - left).abs() <= tol)) | (left.abs() <= tol) | (right.abs() <= tol)
                for row, col in torch.nonzero(tied).tolist():
                    active.append({"stage": counter, "axis": axis, "cell": [row + 1, col + 1],
                        "left": float(left[row, col]), "right": float(right[row, col]), "q_scale": scale})
            if counter in STAGES:
                left, right = q[1, 1] - q[1, 0], q[1, 2] - q[1, 1]
                events[str(counter)] = {"left": float(left), "right": float(right),
                    "gap": float(right - left), "q_scale": scale,
                    "active": positive_gap_active(float(left), float(right), scale)}
            counter += 1

        self.budget.take("direct_J")
        with torch.no_grad(), transport.observe_minmod_stages(capture):
            value = float(self.problem.objective(self.lift(z), self.parameters))
        self.budget.done("direct_J")
        if counter != 360 or not math.isfinite(value):
            raise ValueError("native objective/360-stage audit incomplete")
        control = self.lift(z)
        row = {"index": len(self.state["evaluations"]), "purpose": purpose,
            "coordinates": z.tolist(), "control_sha256": tensor_sha(control), "native_J": value,
            "events": events, "active_limiter_events": active, "branch": audit.report(),
            "exact_minmod_selectors_hex": exact_trace.hex(), "flux_signs": flux_signs,
            "exact_selector_encoding": "360 stages; x then y; each interior2x3;0inactive/1left/2right/3tie"}
        self.state["evaluations"].append(row)
        write(OUT / "RESULT.json", self.state)
        return row

    def tape(self, point: Tensor, *, full_control: bool = False, sides=(1, 1)):
        from advar import transport
        self.budget.take("profile_forward")
        variable = point.detach().clone().requires_grad_(True)
        stages: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        counter = 0

        def capture(q, _qx, _qy):
            nonlocal counter
            if counter in STAGES:
                stages[counter] = (q[1, 1] - q[1, 0], q[1, 2] - q[1, 1], q.abs().max())
            counter += 1

        faces = tuple((*face, side) for face, side in zip(FACES, sides))
        with transport.selected_face_extensions(*faces), transport.observe_minmod_stages(capture, differentiable=True):
            value = self.profile.objective(variable if full_control else self.lift(variable), self.parameters)
        self.budget.done("profile_forward")
        if counter != 360:
            raise ValueError("differentiable profile did not capture exactly 360 forward stages")
        gaps = {stage: operands[1] - operands[0] for stage, operands in stages.items()}
        return variable, value, gaps

    def gradient(self, value: Tensor, variable: Tensor, *, graph: bool = False) -> Tensor:
        self.budget.take("gradient")
        result = torch.autograd.grad(value, variable, retain_graph=True, create_graph=graph)[0]
        self.budget.done("gradient")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite derivative")
        return result

    def hessian(self, gradient: Tensor, variable: Tensor, kind: str) -> Tensor:
        rows = []
        partial: dict[str, Any] = {"kind": kind, "coordinate_sha256": tensor_sha(variable), "rows": []}
        self.state["Hessian_in_progress"] = partial
        for index in range(variable.numel()):
            self.budget.take("Hessian_row")
            rows.append(torch.autograd.grad(gradient[index], variable, retain_graph=True)[0])
            self.budget.done("Hessian_row")
            partial["rows"] = [row.tolist() for row in rows]
            write(OUT / "RESULT.json", self.state)
        return torch.stack(rows)

    def normals(self, z: Tensor) -> dict[str, Any]:
        from advar import transport
        c, phi, gaps = self.tape(self.lift(z), full_control=True)
        normals = torch.stack([self.gradient(gaps[stage], c) for stage in STAGES])
        jac = torch.func.jacrev(self.lift)(z)
        fv = self.problem.frozen.fv_transport

        def fluxes(control):
            psi = torch.einsum("k,kij->ij", fv.coefficient_limits * torch.tanh(control[20:25]), fv.psi_basis)
            qx, qy = transport.face_volume_fluxes(psi)
            return torch.stack([qy[4, 3], qy[2, 1]])

        face_normals = torch.func.jacrev(fluxes)(c.detach())
        rows = torch.cat((face_normals, normals))
        normalized = rows / rows.norm(dim=1)[:, None]
        singular = torch.linalg.svdvals(normalized)
        triples = [torch.linalg.svdvals(normalized[[0, 1, i]]) for i in (2, 3)]
        return {"control_sha256": tensor_sha(c), "profile_J": float(phi),
            "gaps": {str(s): float(v) for s, v in gaps.items()},
            "full_event_normals": normals.tolist(), "chart_event_normals": (normals @ jac).tolist(),
            "full_flux_normals": face_normals.tolist(),
            "normalized_four_row_singular_values": singular.tolist(),
            "flux_plus_each_event_singular_values": [v.tolist() for v in triples],
            "each_event_independent_of_flux_faces": [float(v[-1]) > 128 * EPS * float(v[0]) for v in triples],
            "stage47_and48_joint_rank": int((singular > 128 * EPS * singular[0]).sum())}

    def model(self, z: Tensor, stage: int | None, native_j: float):
        variable, phi, gaps = self.tape(z)
        if abs(float(phi) - native_j) > 128 * EPS * abs(native_j):
            raise ValueError("profile/native current-point objective mismatch")
        g = self.gradient(phi, variable, graph=True)
        self.state["model_in_progress"] = {"control_sha256": tensor_sha(self.lift(z)),
            "native_J": native_j, "tangent_gradient": g.detach().tolist(), "active_stage": stage}
        hessian = self.hessian(g, variable, "current_objective")
        normal = event_hessian = None
        if stage is not None:
            normal = self.gradient(gaps[stage], variable, graph=True)
            event_hessian = self.hessian(normal, variable, "current_event_gap")
        direction, record = event_newton_model(g.detach(), hessian.detach(),
            None if normal is None else normal.detach(), None if event_hessian is None else event_hessian.detach())
        record.update(control_sha256=tensor_sha(self.lift(z)), tangent_gradient=g.detach().tolist(),
            native_J=native_j, active_stage=stage, profile_J=float(phi))
        return direction, None if normal is None else normal.detach(), record

    def qualify_two_faces(self, z: Tensor, terminal: dict[str, Any], gradients: list[Tensor]) -> None:
        """Close the existing four-sector gate only when no limiter is active."""
        branch = terminal["branch"]
        supported = (branch["unique_nearzero_face_entities"] == ["y:2:1", "y:4:3"]
            and not terminal["active_limiter_events"] and not branch["unique_near_tie_slope_entities"])
        if not supported:
            terminal["unresolved_reason"] = "terminal active family exceeds exactly two flux faces"
            return
        jac = torch.func.jacrev(self.lift)(z)
        full = torch.stack(gradients)
        tangents = full @ jac
        normals = torch.tensor(terminal["normal_audit"]["full_flux_normals"], dtype=z.dtype)
        eta = torch.linalg.solve(normals @ normals.T, normals @ full.T).T
        side = torch.tensor(SECTORS, dtype=z.dtype)
        eta_budget = 128 * EPS * full.norm(dim=1)[:, None] / normals.norm(dim=1)[None]
        tangent_error_budget = 128 * EPS * (full.abs() @ jac.abs()).max()
        full_hull = terminal["full_flux_sector_gradient_hull_proxy"]
        tangent_hull = self.geo.geometry.minimum_sampled_hull(tangents)
        full_inf = max(map(abs, full_hull["vector"]))
        tangent_inf = max(map(abs, tangent_hull["vector"]))
        tangent_agreement = float((tangents - tangents[:1]).abs().max())
        normal_pass = bool((side * eta > eta_budget).all())
        passed = full_inf < 1e-10 and tangent_inf < 1e-10 and normal_pass and tangent_agreement <= float(tangent_error_budget)
        terminal["two_face_first_order"] = {"full_hull_inf": full_inf, "tangent_hull_inf": tangent_inf,
            "tangent_hull": tangent_hull, "normal_coefficients": eta.tolist(), "strict_normal_signs": normal_pass,
            "tangent_agreement_inf": tangent_agreement, "component_product_error_budget": float(tangent_error_budget), "passed": passed}
        terminal["first_order_complete"] = True
        if not passed:
            terminal["unresolved_reason"] = "same-point two-face stationarity or strict normal sign gate failed"
            return
        curvature = []
        for sides, reference in zip(SECTORS, tangents):
            variable, phi, _ = self.tape(z, sides=sides)
            gradient = self.gradient(phi, variable, graph=True)
            matrix = self.hessian(gradient, variable, "terminal_four_flux_sectors")
            scale = float(matrix.norm())
            symmetry = float((matrix - matrix.T).norm()) / max(scale, torch.finfo(z.dtype).tiny)
            eig = torch.linalg.eigvalsh((matrix + matrix.T) / 2)
            floor = max(128 * EPS * scale, 1e-8 * float(eig.abs().max()))
            match = (abs(float(phi) - terminal["native_J"]) <= 128 * EPS * abs(terminal["native_J"])
                and float((gradient - reference).abs().max()) <= float(tangent_error_budget))
            curvature.append({"sides": list(sides), "control_sha256": terminal["control_sha256"],
                "hessian": matrix.tolist(), "lambda_min": float(eig[0]), "positive_floor": floor,
                "symmetry_relative": symmetry, "profile_match": match,
                "passed": symmetry < 1e-8 and float(eig[0]) > floor and match})
            terminal["four_sector_curvature"] = curvature
            write(OUT / "RESULT.json", self.state)
        terminal["P2_complete"] = True
        terminal["minimum_claim"] = all(row["passed"] for row in curvature)
        terminal["unresolved_reason"] = None if terminal["minimum_claim"] else "same-point four-sector P2 failed"
        self.state["minimum_claim"] = terminal["minimum_claim"]

    def retract(self, z: Tensor, stage: int | None, normal: Tensor | None,
                reference: dict[str, Any]):
        previous_gap = math.inf
        for index in range(6):
            row = self.value(z, "trial_retraction")
            row["selector_changes_from_reference"] = supported_transition(reference, row)
            reached = active_stage(row)
            if reached is not None and reached != stage:
                raise ValueError("retraction reached a different active event")
            if stage is None or row["events"][str(stage)]["active"]:
                return z, row
            event = row["events"][str(stage)]
            if event["left"] <= 0 or event["right"] <= 0:
                raise ValueError("target event left its positive-operand domain")
            if normal is None:
                raise ValueError("event retraction lacks a chart normal")
            correction = event["gap"] * normal / (normal @ normal)
            if float(correction.norm()) > 0.01:
                raise ValueError("event retraction exceeded local radius")
            if index and abs(event["gap"]) >= previous_gap:
                raise ValueError("event retraction residual failed to decrease")
            previous_gap = abs(event["gap"])
            z = z - correction
        raise ValueError("event retraction did not converge within six corrections")


def active_stage(row: dict[str, Any]) -> int | None:
    events = row["active_limiter_events"]
    if not events:
        return None
    if len(events) != 1:
        raise ValueError("unsupported simultaneous limiter events; one-event search only")
    event = events[0]
    if (event["stage"] not in STAGES or event["axis"] != "x" or event["cell"] != [1, 1]
            or not positive_gap_active(event["left"], event["right"], event["q_scale"])):
        raise ValueError("unsupported current active limiter identity/domain")
    return event["stage"]


def run() -> dict[str, Any]:
    start = time.monotonic()
    if OUT.exists():
        raise ValueError("attempt1 exists; never overwrite its plan/prefix")
    q = load("progress_qualifier", HERE / "qualify_candidate.py")
    geo = load("progress_geometry", HERE / "two_face_intersection_20261013/diagnose.py")
    base = load("progress_native24", HERE / "two_face_intersection_20261013/continue_native24.py")
    from examples.weather_scenarios.fv_point_3h_limiter_ray_diagnostic import local_module_sources
    case = q.cycle.load_fixed_case()
    prior = json.loads(ENDPOINT.read_text())
    control = torch.tensor(prior["current_control"], dtype=torch.float64)
    if tensor_sha(control) != prior["current_control_sha256"]:
        raise ValueError("endpoint control bytes differ")
    old_plan = json.loads((PRIOR / "PLAN.json").read_text())
    accounting = json.loads((PRIOR / "ACCOUNTING.json").read_text())
    sources = local_module_sources(tuple(sys.modules.values())) | {str(Path(__file__).relative_to(ROOT))}
    source_hashes = {name: sha(ROOT / name) for name in sorted(sources)}
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (
        ENDPOINT, PRIOR / "PLAN.json", PRIOR / "RESULT.json", PRIOR / "ACCOUNTING.json", q.cycle.CASE)}
    plan: dict[str, Any] = {"status": "frozen_event_progress_phase", "scope": "identified limiter events -> native-J progress -> same-endpoint qualification",
        "source_hashes": source_hashes, "input_hashes": inputs, "control_sha256": tensor_sha(control),
        "parameters_sha256": tensor_sha(case["parameters"]), "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "transport_transition": {"before": old_plan["source_hashes"]["src/advar/transport.py"], "after": sha(ROOT / "src/advar/transport.py"),
            "reason": "optional connected stage observation on replay=False; physical arithmetic/default API unchanged"},
        "budget": {"direct_J_before": 512, "gradient_before": 134,
            "prior_estimated_charged_seconds": accounting["inherited_estimated_charged_seconds_after"],
            "previous_wall_cap_seconds": 3600, "additional_wall_seconds": 900,
            "new_total_estimated_compute_cap_seconds": accounting["inherited_estimated_charged_seconds_after"] + 900,
            "direct_J_cap": 256, "new_cumulative_direct_J_cap": 768, "gradient_cap": 128,
            "new_cumulative_gradient_cap": 262, "Hessian_row_cap": 432, "profile_forward_cap": 48,
            "RSS_cap_bytes": 2 * 1024**3, "previous_Hessian_and_profile_totals_complete": False,
            "policy": "new authorized phase; previous caps are not reset or described as preserved; started calls consume allocation"},
        "max_steps": 6, "event_stages": list(STAGES), "event_axis_cell": ["x", 1, 1],
        "root_bracket": [2**-12, 2**-10], "selection_uses_forecast": False,
        "minimum_policy": "active limiter lacks complete sector P2 support; no minimum claim from event-constrained stationarity"}
    OUT.mkdir(parents=True)
    write(OUT / "PLAN.json", plan)
    (OUT / "PLAN_SHA256.txt").write_text(sha(OUT / "PLAN.json") + "\n")
    state: dict[str, Any] = {"status": "in_progress", "plan": plan, "plan_sha256": sha(OUT / "PLAN.json"),
        "counts": {f"{kind}_{status}": 0 for kind in ("direct_J", "gradient", "Hessian_row", "profile_forward") for status in ("started", "completed")},
        "evaluations": [], "event_roots": [], "accepted_steps": [], "models": [],
        "minimum_claim": False, "response_claim": False, "score_calls": 0}
    budget = Budget(state, start)
    write(OUT / "RESULT.json", state)
    execution = Execution(case, q, geo, base, control, state, budget)
    z = execution.initial
    try:
        with tarfile.open(OUT / "source.tar.gz", "w:gz") as archive:
            for name in source_hashes:
                archive.add(ROOT / name, arcname=name)
        baseline = execution.value(z, "native_input_requalification")
        if abs(baseline["native_J"] - prior["native_J"]) > 128 * EPS * abs(prior["native_J"]):
            raise ValueError("source transition changed current native J")
        initial_variable, initial_phi, _ = execution.tape(z)
        initial_g = execution.gradient(initial_phi, initial_variable)
        old_g = torch.tensor(prior["current_point"]["tangent_hull"]["vector"], dtype=z.dtype)
        gradient_budget = 128 * 360 * EPS * float(old_g.norm())
        if float((initial_g - old_g).abs().max()) > gradient_budget:
            raise ValueError("current gradient differs from pinned native endpoint")
        state["initial_profile_gradient_gap"] = float((initial_g - old_g).abs().max())
        state["initial_gradient_comparison_budget"] = gradient_budget
        direction = torch.tensor(old_plan["direction"], dtype=z.dtype)
        cache: dict[float, dict[str, Any]] = {}

        def ray(alpha):
            if alpha not in cache:
                cache[alpha] = execution.value(execution.initial + alpha * direction, "event_location")
            return cache[alpha]

        for stage in STAGES:
            lower, upper = plan["root_bracket"]
            root, row, bracket = bisect_gap(ray, stage, lower, upper)
            point = execution.initial + root * direction
            if active_stage(row) != stage:
                raise ValueError("located root contains unsupported/mixed active identity")
            root_changes = supported_transition(baseline, row)
            state["event_roots"].append({"stage": stage, "alpha": root, "native_J": row["native_J"],
                "last_sign_bracket": bracket,
                "selector_changes_from_base": root_changes,
                "events": row["events"], "active_limiter_events": row["active_limiter_events"],
                "coordinates": point.tolist(), "normal_audit": execution.normals(point)})
            write(OUT / "RESULT.json", state)
        first = min(state["event_roots"], key=lambda entry: entry["alpha"])
        z = torch.tensor(first["coordinates"], dtype=z.dtype)
        current = execution.value(z, "first_event_candidate")
        current["selector_changes_from_base"] = supported_transition(baseline, current)
        if current["native_J"] >= baseline["native_J"]:
            raise ValueError("first event candidate does not decrease native J")
        state["accepted_steps"].append({"kind": "earliest_located_of_two_bracketed_events", "native_J": current["native_J"],
            "control": execution.lift(z).tolist(), "control_sha256": current["control_sha256"], "coordinates": z.tolist()})
        state["current_point"] = current
        for index in range(plan["max_steps"]):
            stage = active_stage(current)
            if stage is not None:
                normal_audit = execution.normals(z)
                if not normal_audit["each_event_independent_of_flux_faces"][STAGES.index(stage)]:
                    raise ValueError("active limiter normal dependent on flux faces")
            step, normal, model = execution.model(z, stage, current["native_J"])
            state["models"].append(model)
            state["current_point"]["fresh_model"] = model
            write(OUT / "RESULT.json", state)
            if model["projected_gradient_inf"] < 1e-10:
                state["status"] = "event_constrained_stationary_requires_nonsmooth_qualification"
                break
            if model["slope"] >= 0:
                raise ValueError("current constrained Newton model is not descent")
            accepted = False
            for backtrack in range(12):
                alpha = 2.0**-backtrack
                try:
                    trial_z, trial = execution.retract(z + alpha * step, stage, normal, current)
                    active_stage(trial)
                    trial["selector_changes_from_previous"] = supported_transition(current, trial)
                except ValueError as error:
                    state.setdefault("rejected_trials", []).append({"iteration": index, "alpha": alpha, "error": str(error)})
                    continue
                threshold = current["native_J"] + 1e-4 * alpha * model["slope"]
                if trial["native_J"] < current["native_J"] and trial["native_J"] <= threshold:
                    state["accepted_steps"].append({"kind": "current_event_lagrangian_Newton", "iteration": index,
                        "alpha": alpha, "active_stage": stage, "from_control_sha256": current["control_sha256"],
                        "native_J": trial["native_J"], "control": execution.lift(trial_z).tolist(),
                        "control_sha256": trial["control_sha256"], "coordinates": trial_z.tolist(),
                        "armijo_threshold": threshold, "linear_slope": model["slope"]})
                    z, current = trial_z, trial
                    state["current_point"] = current
                    write(OUT / "RESULT.json", state)
                    accepted = True
                    break
            if not accepted:
                state["status"] = "no_native_Armijo_progress_at_current_event"
                break
        else:
            state["status"] = "step_cap_reached"
    except Exception as error:
        state.update(status="interrupted_or_unsupported", error=f"{type(error).__name__}: {error}")
    # Always audit the last accepted candidate; a budget failure preserves its prefix.
    state["in_terminal"] = True
    try:
        terminal = execution.value(z, "terminal_native_requalification")
        state["terminal"] = terminal
        terminal.update(first_order_complete=False, P2_complete=False, minimum_claim=False,
            unresolved_reason="terminal audit in progress")
        terminal["normal_audit"] = execution.normals(z)
        gradients = []
        for sides in SECTORS:
            c, phi, _ = execution.tape(execution.lift(z), full_control=True, sides=sides)
            g = execution.gradient(phi, c)
            gradients.append(g)
            terminal["exact_current_gradient_by_flux_sector"] = [value.tolist() for value in gradients]
            write(OUT / "RESULT.json", state)
        terminal["full_flux_sector_gradient_hull_proxy"] = geo.geometry.minimum_sampled_hull(torch.stack(gradients))
        terminal["exact_current_gradient_by_flux_sector"] = [g.tolist() for g in gradients]
        stage = active_stage(terminal)
        _, _, terminal_model = execution.model(z, stage, terminal["native_J"])
        terminal["same_point_selected_branch_curvature"] = terminal_model
        terminal["first_order_complete"] = False
        terminal["P2_complete"] = False
        terminal["minimum_claim"] = False
        terminal["unresolved_reason"] = ("active limiter needs limiting-sector stationarity and exact event-sector P2; two-flux-face Hessians cannot certify it"
            if stage is not None else "native stationarity/complete same-point four-sector P2 not established")
        if stage is None:
            execution.qualify_two_faces(z, terminal, gradients)
    except Exception as error:
        state["terminal_audit_error"] = f"{type(error).__name__}: {error}"
        if "terminal" in state:
            state["terminal"]["unresolved_reason"] = state["terminal_audit_error"]
    close_claims(state,
        all(sha(ROOT / name) == digest for name, digest in source_hashes.items()),
        all(sha(ROOT / name) == digest for name, digest in inputs.items()))
    state.update(elapsed_seconds=time.monotonic() - start,
        current_control=execution.lift(z).tolist(), current_control_sha256=tensor_sha(execution.lift(z)),
        cumulative_direct_J_started=512 + state["counts"]["direct_J_started"],
        cumulative_direct_J_completed=512 + state["counts"]["direct_J_completed"],
        cumulative_gradient_started=134 + state["counts"]["gradient_started"],
        cumulative_gradient_completed=134 + state["counts"]["gradient_completed"])
    write(OUT / "RESULT.json", state)
    return state


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "counts": result["counts"], "accepted_steps": len(result["accepted_steps"]),
        "terminal_J": result.get("terminal", {}).get("native_J"), "error": result.get("error"),
        "terminal_audit_error": result.get("terminal_audit_error")}))
