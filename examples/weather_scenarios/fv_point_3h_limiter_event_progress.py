"""Maintain/release observed minmod event forms and advance the native J.

Event identity comes from the actual trace, not a stage-number allowlist.
Side-specific derivatives propose a release; native J approves the move.
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
from contextlib import ExitStack
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.weather_scenarios.fv_minmod_events import MinmodEvent, classify_event, classify_change, event_action
HERE = ROOT / "graphify-out/fv-root-cause-20260919/native_reference_20261012"
OUT = HERE / "event_policy_20261015/attempt1"
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
        normals = normal[None] if normal.ndim == 1 else normal
        u, singular, vh = torch.linalg.svd(normals, full_matrices=True)
        if float(singular[-1]) <= 128 * EPS * float(singular[0]):
            raise ValueError("active event chart normals are dependent")
        q = vh[normals.shape[0]:].T
        multipliers = -u @ ((vh[:normals.shape[0]] @ g) / singular)
        multiplier = float(multipliers[0]) if normals.shape[0] == 1 else multipliers.tolist()
        if event_hessian is None:
            raise ValueError("event model requires the Hessian of its equality")
        constraints = event_hessian[None] if event_hessian.ndim == 2 else event_hessian
        lagrangian = hessian + torch.einsum("k,kij->ij", multipliers, constraints)
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
        "normal_step_dot": None if normal is None else (normal @ direction).tolist(),
        "hessian": hessian.tolist(), "event_hessian": None if event_hessian is None else event_hessian.tolist(),
        "reduced_lagrangian_hessian": reduced.tolist(), "slope": float(g @ direction),
        "interpretation": "same-point selected AD branch and fixed multiplier; search model only"}


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
        reserve = 0 if self.state.get("in_terminal") else 450
        if time.monotonic() - self.start >= budget["additional_wall_seconds"] - reserve:
            raise TimeoutError("new phase wall budget exhausted")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        if rss > budget["RSS_cap_bytes"]:
            raise MemoryError("new phase RSS budget exhausted")
        reserves = {"direct_J": 2, "gradient": 128, "Hessian_row": 1152, "profile_forward": 40}
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
            "active_limiter_events": active, "branch": audit.report(),
            "exact_minmod_selectors_hex": exact_trace.hex(), "flux_signs": flux_signs,
            "exact_selector_encoding": "360 stages; x then y; each interior2x3;0inactive/1left/2right/3tie"}
        self.state["evaluations"].append(row)
        write(OUT / "RESULT.json", self.state)
        return row

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



class EventExecution(Execution):
    """Reuse the native evaluator/Newton operations with two event forms."""

    def value(self, z: Tensor, purpose: str) -> dict[str, Any]:
        from advar import transport
        operands, scales, flux = [], [], []

        def capture(q, qx, qy):
            axes = []
            for left, right in (
                (q[1:-1,1:-1]-q[1:-1,:-2], q[1:-1,2:]-q[1:-1,1:-1]),
                (q[1:-1,1:-1]-q[:-2,1:-1], q[2:,1:-1]-q[1:-1,1:-1]),
            ):
                axes.append(torch.stack((left, right), dim=-1).numpy())
            operands.append(np.stack(axes))
            scales.append(float(q.abs().max()))
            if not flux:
                flux.extend(torch.cat((qx.flatten(), qy.flatten())).tolist())

        with transport.observe_minmod_stages(capture):
            row = super().value(z, purpose)
        path = OUT / f"trace_{row['index']:04d}.npz"
        with path.open("wb") as stream:
            np.savez_compressed(stream, operands=np.asarray(operands), scales=np.asarray(scales), flux=np.asarray(flux))
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=sha(path))
        write(OUT / "RESULT.json", self.state)
        return row

    def operands(self, row, event):
        with np.load(ROOT / row["trace_path"], allow_pickle=False) as trace:
            values = trace["operands"][event.stage, 0 if event.axis=="x" else 1, event.row-1, event.column-1]
            return float(values[0]), float(values[1]), float(trace["scales"][event.stage])

    def actual_events(self, row):
        if row["branch"]["unique_nearzero_face_entities"] != ["y:2:1","y:4:3"]:
            raise ValueError("additional current flux-face activation requires its own geometry")
        events = [classify_event(a["stage"], a["axis"], a["cell"][0], a["cell"][1], a["left"], a["right"], a["q_scale"])
                  for a in row["active_limiter_events"]]
        if len(events)>2:
            raise ValueError(f"more than two simultaneous active surfaces: {[asdict(e) for e in events]}")
        return events

    def changes(self, before, after, ignored=()):
        a, b = bytes.fromhex(before["exact_minmod_selectors_hex"]), bytes.fromhex(after["exact_minmod_selectors_hex"])
        result=[]
        for index,(old,new) in enumerate(zip(a,b)):
            if old==new: continue
            stage,cell=divmod(index,12);axis,interior=divmod(cell,6);row,col=divmod(interior,3)
            key=(stage,"x" if axis==0 else "y",row+1,col+1)
            if key in ignored: continue
            dummy=MinmodEvent(*key,"gap",1)
            left0,right0,_=self.operands(before,dummy);left1,right1,_=self.operands(after,dummy)
            result.append(classify_change(key,old,new,(left0,right0),(left1,right1)))
        for index,(old,new) in enumerate(zip(before["flux_signs"],after["flux_signs"])):
            if old!=new and index not in (35,47):
                raise ValueError(f"additional flux-face sign transition {index}")
        return result

    def changed_trace(self,before,after,ignored=()):
        for index,(old,new) in enumerate(zip(before["flux_signs"],after["flux_signs"])):
            if old!=new and index not in (35,47): raise ValueError(f"additional flux-face transition {index}")
        a,b=bytes.fromhex(before["exact_minmod_selectors_hex"]),bytes.fromhex(after["exact_minmod_selectors_hex"])
        for index,(old,new) in enumerate(zip(a,b)):
            stage,cell=divmod(index,12);axis,interior=divmod(cell,6);row,col=divmod(interior,3)
            key=(stage,"x" if axis==0 else "y",row+1,col+1)
            if old!=new and key not in ignored: return True
        return False

    def tape(self, point, *, events=(), signs=(), full_control=False, sides=(1,1)):
        from advar import transport
        self.budget.take("profile_forward")
        variable=point.detach().clone().requires_grad_(True)
        gaps={};counter=0

        def capture(q,_qx,_qy):
            nonlocal counter
            for event in events:
                if counter!=event.stage: continue
                r,c=event.row,event.column
                left=q[r,c]-(q[r,c-1] if event.axis=="x" else q[r-1,c])
                right=(q[r,c+1] if event.axis=="x" else q[r+1,c])-q[r,c]
                gaps[event.key]=event.value(left,right)
            counter+=1

        with ExitStack() as stack:
            stack.enter_context(transport.selected_face_extensions(*tuple((*face,side) for face,side in zip(FACES,sides))))
            for event,side in zip(events,signs):
                stack.enter_context(transport.selected_minmod_extension(event.stage,event.axis,event.row,event.column,event.selector(side)))
            stack.enter_context(transport.observe_minmod_stages(capture,differentiable=True))
            value=self.profile.objective(variable if full_control else self.lift(variable),self.parameters)
        self.budget.done("profile_forward")
        if counter!=360 or len(gaps)!=len(events): raise ValueError("incomplete connected stage trace")
        return variable,value,gaps

    def family(self,z,row,events):
        from advar import variational as v
        records=[]
        combinations=list(product((-1,1),repeat=len(events)))
        jac=torch.func.jacrev(self.lift)(z)
        prior_control=self.lift(z)
        prior_variable=prior_control.detach().clone().requires_grad_(True)
        contract=self.profile.contract(self.parameters)
        prior_residual=v._control_prior_residual(prior_variable,contract)
        prior_cost=(0.5*torch.dot(prior_residual,prior_residual)
            +v._field_smoothness_prior_cost(prior_variable,contract))
        prior_gradient=self.gradient(prior_cost,prior_variable)
        prior_chart_gradient=jac.T@prior_gradient
        for signs in combinations:
            for sides in SECTORS:
                variable,phi,gaps=self.tape(self.lift(z),events=events,signs=signs,sides=sides,full_control=True)
                full_g=self.gradient(phi,variable);g=jac.T@full_g
                full_n=torch.stack([self.gradient(gaps[e.key],variable) for e in events]) if events else torch.empty((0,26),dtype=z.dtype)
                normals=full_n@jac
                gap=abs(float(phi.detach())-row["native_J"])
                if gap>128*360*EPS*abs(row["native_J"]): raise ValueError("side extension value does not close with native J")
                records.append({"signs":list(signs),"flux_sides":list(sides),"gradient":g.tolist(),
                    "full_gradient":full_g.tolist(),"full_normals":full_n.tolist(),
                    "normals":normals.tolist(),"profile_J":float(phi.detach()),"native_J_gap":gap})
        groups=[]
        gradient_component_scale=0.
        for signs in combinations:
            group_rows=[a for a in records if a["signs"]==list(signs)]
            group=torch.tensor([a["gradient"] for a in group_rows],dtype=z.dtype)
            # Bound cancellation between the true prior and data gradient parts.
            component_scale=max(float(prior_chart_gradient.norm())+
                float((torch.tensor(a["gradient"],dtype=z.dtype)-prior_chart_gradient).norm())
                for a in group_rows)
            gradient_component_scale=max(gradient_component_scale,component_scale)
            error=128*360*EPS*component_scale
            if float((group-group[:1]).abs().max())>error: raise ValueError("flux-sector tangent gradient disagreement")
            groups.append(group.mean(dim=0))
        chart=torch.stack(groups)
        hull=self.geo.geometry.minimum_sampled_hull(chart)
        decisions=[]
        for index,event in enumerate(events):
            outgoing={-1:[],1:[]};directions={-1:[],1:[]}
            for rec in records:
                normals=torch.tensor(rec["normals"],dtype=z.dtype)
                if bool((normals.norm(dim=1)<=torch.finfo(z.dtype).tiny).any()):
                    raise ValueError("current active event has a zero normal")
                singular=torch.linalg.svdvals(normals/normals.norm(dim=1)[:,None])
                if float(singular[-1])<=128*EPS*float(singular[0]): raise ValueError("current active normals dependent")
                other=torch.cat((normals[:index],normals[index+1:]))
                if other.numel():
                    _,_,vh=torch.linalg.svd(other,full_matrices=True);basis=vh[other.shape[0]:].T
                    nu=basis@(basis.T@normals[index])
                else: nu=normals[index]
                nu=nu/nu.norm()
                side=rec["signs"][index];g=torch.tensor(rec["gradient"],dtype=z.dtype)
                outgoing[side].append(float(side*(g@nu)));directions[side].append(side*nu)
            error=128*360*EPS*gradient_component_scale
            means={s:sum(outgoing[s])/len(outgoing[s]) for s in (-1,1)}
            disagreement=max(max(v)-min(v) for v in outgoing.values())
            decision="release" if min(means.values()) < -error else "maintain" if min(means.values())>error else "unresolved"
            if disagreement>error: decision="unresolved"
            side=min(means,key=lambda s: means[s])
            direction=torch.stack(directions[side]).mean(dim=0);direction/=direction.norm()
            reference=next(rec for rec in records if rec["signs"][index]==side)
            reference_normals=torch.tensor(reference["normals"],dtype=z.dtype)
            other=torch.cat((reference_normals[:index],reference_normals[index+1:]))
            if other.numel():
                _,_,vh=torch.linalg.svd(other,full_matrices=True)
                basis=vh[other.shape[0]:].T
                direction=basis@(basis.T@direction);direction/=direction.norm()
            # Armijo must use the direction that is actually stepped, rather
            # than an average of derivatives along different sector normals.
            actual_slopes=[];direction_valid=True
            for rec in records:
                if rec["signs"][index]!=side: continue
                n=torch.tensor(rec["normals"],dtype=z.dtype)
                retained=torch.cat((n[:index],n[index+1:]))
                direction_valid=direction_valid and float(side*(n[index]@direction))>128*360*EPS*float(n[index].norm())
                if retained.numel():
                    direction_valid=direction_valid and bool(((retained@direction).abs()<=128*360*EPS*retained.norm(dim=1)).all())
                actual_slopes.append(float(torch.tensor(rec["gradient"],dtype=z.dtype)@direction))
            if decision=="release" and (not direction_valid or max(actual_slopes)>=-error or max(actual_slopes)-min(actual_slopes)>error):
                decision="unresolved"
            decisions.append({"event":asdict(event),"decision":decision,"side":side,"outgoing":outgoing,
                "error_budget":error,"direction":direction.tolist(),"derivative":max(actual_slopes),
                "stepped_direction_derivatives":actual_slopes,"stepped_direction_valid":direction_valid,
                "release_normals":reference["normals"]})
        family: dict[str,Any]={"control_sha256":row["control_sha256"],"events":[asdict(e) for e in events],
            "prior_cost":float(prior_cost.detach()),"prior_full_gradient":prior_gradient.tolist(),
            "prior_chart_gradient":prior_chart_gradient.tolist(),
            "gradient_component_scale":gradient_component_scale,
            "records":records,"chart_hull":hull,"chart_hull_inf":max(map(abs,hull["vector"])),"decisions":decisions}
        self.state.setdefault("side_families",[]).append(family);write(OUT/"RESULT.json",self.state)
        return family

    def search_model(self,z,events,row,family):
        signs=tuple(1 for _ in events)
        variable,phi,gaps=self.tape(z,events=events,signs=signs)
        g=self.gradient(phi,variable,graph=True);hj=self.hessian(g,variable,"current_objective")
        normals=[];hh=[]
        for event in events:
            n=self.gradient(gaps[event.key],variable,graph=True);normals.append(n.detach())
            hh.append(self.hessian(n,variable,"current_event"))
        n=torch.stack(normals) if normals else None
        step,model=event_newton_model(g.detach(),hj.detach(),n,torch.stack(hh).detach() if hh else None)
        model.update(control_sha256=row["control_sha256"],native_J=row["native_J"],events=[asdict(e) for e in events])
        self.state["models"].append(model);write(OUT/"RESULT.json",self.state)
        return step,n,model

    def restore_events(self,z,events,normals):
        previous=math.inf
        for _ in range(6):
            row=self.value(z,"retraction_intermediate")
            values=[]
            for event in events:
                left,right,scale=self.operands(row,event)
                values.append(float(event.value(torch.tensor(left,dtype=z.dtype),torch.tensor(right,dtype=z.dtype))))
            h=torch.tensor(values,dtype=z.dtype)
            if not events or all(abs(v)<=128*EPS*self.operands(row,e)[2] for e,v in zip(events,values)): return z,row
            if float(h.norm())>=previous: raise ValueError("retraction residual did not decrease")
            previous=float(h.norm())
            correction=torch.linalg.lstsq(normals,h,driver='gelsd').solution
            if float(correction.norm())>.01: raise ValueError("retraction exceeds local radius")
            z=z-correction
        raise ValueError("retraction did not resolve event equalities")

    def locate(self,z0,row0,z1,row1,ignored=(),keep=(),normals=None):
        if not self.changed_trace(row0,row1,ignored): return z1,row1,[],1.
        lower,upper=0.,1.;low_row,high=row0,row1
        for _ in range(28):
            mid=(lower+upper)/2;_,row=self.restore_events(z0+mid*(z1-z0),keep,normals)
            if self.changed_trace(row0,row,ignored): upper=mid;high=row
            else: lower=mid;low_row=row
        events=self.changes(low_row,high,ignored)
        event=events[0]
        a,b=self.operands(row0,event)[:2],self.operands(high,event)[:2]
        h0=float(event.value(torch.tensor(a[0],dtype=z0.dtype),torch.tensor(a[1],dtype=z0.dtype)))
        low=lower
        for _ in range(40):
            alpha=(low+upper)/2;point,row=self.restore_events(z0+alpha*(z1-z0),keep,normals)
            left,right,scale=self.operands(row,event)
            h=float(event.value(torch.tensor(left,dtype=z0.dtype),torch.tensor(right,dtype=z0.dtype)))
            if abs(h)<=128*EPS*scale:
                simultaneous=self.actual_events(row)
                located=[e for e in simultaneous if e.key not in ignored]
                if not any(e.key==event.key for e in located): raise ValueError("resolved root not present in actual active union")
                self.state.setdefault("transitions",[]).append({"event":asdict(event),"located_union":[asdict(e) for e in located],"control_sha256":row["control_sha256"],
                    "segment_fraction":alpha,"last_bracket":[low,upper],"native_J":row["native_J"],"h":h})
                write(OUT/"RESULT.json",self.state);return point,row,located,alpha
            if (h>0)==(h0>0): low=alpha
            else: upper=alpha
        raise ValueError("located transition root not resolved")

    def terminal_qualification(self,z,row,events,family):
        from advar import transport
        row.update(events=[asdict(e) for e in events],first_order_complete=False,P2_complete=False,minimum_claim=False,
            chart_hull_inf=family["chart_hull_inf"],unresolved_reason="current side-by-flux limiting family is nonstationary")
        if family["chart_hull_inf"]>=1e-10: return
        fv=self.problem.frozen.fv_transport
        def fluxes(c):
            qx,qy=transport.face_volume_fluxes(torch.einsum("k,kij->ij",fv.coefficient_limits*torch.tanh(c[20:25]),fv.psi_basis))
            return torch.stack((qy[4,3],qy[2,1]))
        nf=torch.func.jacrev(fluxes)(self.lift(z))
        normal_rows=[];passed=True
        for rec in family["records"]:
            normals=torch.cat((nf,torch.tensor(rec["full_normals"],dtype=z.dtype)))
            normalized=normals/normals.norm(dim=1)[:,None];u,s,vh=torch.linalg.svd(normals,full_matrices=True)
            resolved=float(torch.linalg.svdvals(normalized)[-1])>128*EPS*float(torch.linalg.svdvals(normalized)[0])
            g=torch.tensor(rec["full_gradient"],dtype=z.dtype)
            eta=u@((vh[:normals.shape[0]]@g)/s)
            signs=torch.tensor(rec["flux_sides"]+rec["signs"],dtype=z.dtype)
            budget=128*360*EPS*float(g.norm())/normals.norm(dim=1)
            strict=bool((eta*signs>budget).all())
            passed=passed and resolved and strict
            normal_rows.append({"flux_sides":rec["flux_sides"],"limiter_signs":rec["signs"],"eta":eta.tolist(),
                "strict_outward":strict,"normal_rank_resolved":resolved})
        groups=list(product((-1,1),repeat=len(events)));weights=family["chart_hull"]["weights"]
        mixed=[]
        for sides in SECTORS:
            vertices=torch.tensor([next(r["full_gradient"] for r in family["records"] if r["flux_sides"]==list(sides) and r["signs"]==list(signs)) for signs in groups],dtype=z.dtype)
            mixed.append(torch.tensor(weights,dtype=z.dtype)@vertices)
        full_hull=self.geo.geometry.minimum_sampled_hull(torch.stack(mixed))
        row.update(first_order_complete=True,normal_cones=normal_rows,full_product_hull=full_hull)
        if not passed or max(map(abs,full_hull["vector"]))>=1e-10:
            row["unresolved_reason"]="complete current normal-cone/product-mixture stationarity gate failed";return
        curvature=[]
        for signs in groups:
            for sides in SECTORS:
                variable,phi,gaps=self.tape(z,events=events,signs=signs,sides=sides)
                g=self.gradient(phi,variable,graph=True);hj=self.hessian(g,variable,"terminal_sector_objective")
                reference=next(r for r in family["records"] if r["signs"]==list(signs) and r["flux_sides"]==list(sides))
                gradient_budget=128*360*EPS*family["gradient_component_scale"]
                if (abs(float(phi)-row["native_J"])>128*360*EPS*abs(row["native_J"])
                    or float((g.detach()-torch.tensor(reference["gradient"],dtype=z.dtype)).abs().max())>gradient_budget):
                    raise ValueError("terminal curvature tape differs from current qualified side family")
                constraint_h=[];normals=[]
                for event in events:
                    normal=self.gradient(gaps[event.key],variable,graph=True);normals.append(normal.detach())
                    constraint_h.append(self.hessian(normal,variable,"terminal_sector_constraint"))
                n=torch.stack(normals) if events else None
                _,metric=event_newton_model(g.detach(),hj.detach(),n,torch.stack(constraint_h).detach() if events else None)
                curvature.append({"limiter_signs":list(signs),"flux_sides":list(sides),"control_sha256":row["control_sha256"],
                    "lambda_min":metric["lambda_min"],"floor":metric["positive_floor"],"passed":metric["lambda_min"]>metric["positive_floor"],
                    "reduced_lagrangian_hessian":metric["reduced_lagrangian_hessian"]})
                row["curvature"]=curvature;write(OUT/"RESULT.json",self.state)
        row["P2_complete"]=True;row["minimum_claim"]=all(c["passed"] for c in curvature)
        self.state["minimum_claim"]=row["minimum_claim"]
        row["unresolved_reason"]=None if row["minimum_claim"] else "current complete reduced-sector P2 failed"


def run() -> dict[str,Any]:
    start=time.monotonic()
    if OUT.exists(): raise ValueError("attempt path already exists")
    q=load("policy_q",HERE/"qualify_candidate.py");geo=load("policy_geo",HERE/"two_face_intersection_20261013/diagnose.py")
    base=load("policy_base",HERE/"two_face_intersection_20261013/continue_native24.py")
    from examples.weather_scenarios.fv_point_3h_limiter_ray_diagnostic import local_module_sources
    old=HERE/"event_progress_20261015";prior=json.loads((old/"attempt1/RESULT.json").read_text());account=json.loads((old/"ACCOUNTING.json").read_text())
    case=q.cycle.load_fixed_case();control=torch.tensor(prior["current_control"],dtype=torch.float64)
    if tensor_sha(control)!=prior["current_control_sha256"]: raise ValueError("endpoint control hash mismatch")
    sources=local_module_sources(tuple(sys.modules.values()))|{str(Path(__file__).relative_to(ROOT))}
    source_hashes={s:sha(ROOT/s) for s in sorted(sources)}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (old/"attempt1/RESULT.json",old/"ACCOUNTING.json",old/"TERMINAL_REPLAY_CHECK.json",q.cycle.CASE)}
    plan: dict[str,Any]={"status":"frozen_maintain_release_transition_phase","source_hashes":source_hashes,"input_hashes":inputs,
        "control_sha256":tensor_sha(control),"parameters_sha256":tensor_sha(case["parameters"]),
        "runtime":{"python":platform.python_version(),"torch":torch.__version__},
        "transport_transition":{"before":prior["plan"]["source_hashes"]["src/advar/transport.py"],"after":sha(ROOT/"src/advar/transport.py"),
            "reason":"opt-in stage-local derivative extension; native default arithmetic preserved"},
        "budget":{"direct_J_before":594,"gradient_before":163,"prior_estimated_charged_seconds":account["inherited_estimated_charged_seconds_after"],
            "additional_wall_seconds":900,"direct_J_cap":384,"gradient_cap":768,"Hessian_row_cap":2048,"profile_forward_cap":256,
            "RSS_cap_bytes":2*1024**3,"policy":"new phase; cumulative counts retained; complete historical Hessian/AD totals unknown"},
        "max_steps":12,"max_simultaneous_events":2,"supported_forms":["same-sign left/right","selected operand through zero"],
        "no_stage_whitelist":True,"forecast_selection":False}
    OUT.mkdir(parents=True);write(OUT/"PLAN.json",plan);(OUT/"PLAN_SHA256.txt").write_text(sha(OUT/"PLAN.json")+'\n')
    state: dict[str,Any]={"status":"running","plan":plan,"plan_sha256":sha(OUT/"PLAN.json"),
        "counts":{f"{k}_{s}":0 for k in ("direct_J","gradient","Hessian_row","profile_forward") for s in ("started","completed")},
        "evaluations":[],"accepted_steps":[],"models":[],"minimum_claim":False,"response_claim":False,"score_calls":0}
    budget=Budget(state,start);execution=EventExecution(case,q,geo,base,control,state,budget);z=execution.initial
    with tarfile.open(OUT/"source.tar.gz","w:gz") as archive:
        for s in sources: archive.add(ROOT/s,arcname=s)
    try:
        current=execution.value(z,"input_native_requalification")
        if abs(current["native_J"]-prior["terminal"]["native_J"])>128*EPS*current["native_J"]: raise ValueError("native input requalification failed")
        for iteration in range(plan["max_steps"]):
            events=execution.actual_events(current);family=execution.family(z,current,events)
            mode=event_action(family["decisions"],family["chart_hull_inf"])
            if mode=="stationary":
                state["status"]="stationary_requires_terminal_qualification";break
            releases=[d for d in family["decisions"] if d["decision"]=="release"]
            if mode=="unresolved":
                state["status"]="event_outgoing_sign_unresolved";break
            chosen: dict[str,Any] | None=None
            released: MinmodEvent | None=None
            if releases:
                proposal: dict[str,Any]=min(releases,key=lambda d:d["derivative"])
                chosen=proposal;released=MinmodEvent(**proposal["event"])
                step=torch.tensor(proposal["direction"],dtype=z.dtype)*1e-5;slope=proposal["derivative"]*1e-5
                keep=[e for e in events if e.key!=released.key]
                model={"mode":mode,"event":asdict(released),"slope":slope,"control_sha256":current["control_sha256"]}
                self_normals=torch.tensor(proposal["release_normals"],dtype=z.dtype)
                normals=self_normals[[events.index(e) for e in keep]] if keep else None
            else:
                keep=events;step,normals,model=execution.search_model(z,keep,current,family);slope=model["slope"]
            accepted=False
            for backtrack in range(16):
                alpha=2.**-backtrack
                try:
                    trial_z,trial=execution.restore_events(z+alpha*step,keep,normals)
                    ignored=[e.key for e in events]
                    # Detect also an interior sampled transition before accepting the endpoint.
                    sample_fraction=1.
                    for fraction in (.25,.5,.75):
                        mid_z,mid=execution.restore_events(z+alpha*fraction*step,keep,normals)
                        if execution.changed_trace(current,mid,ignored):
                            trial_z,trial=mid_z,mid;sample_fraction=fraction;break
                    trial_z,trial,located,root_fraction=execution.locate(z,current,trial_z,trial,ignored,keep,normals)
                    effective_alpha=alpha*sample_fraction*root_fraction
                    reached=execution.actual_events(trial)
                    release_confirmed=False
                    if chosen is not None and released is not None:
                        l,r,scale=execution.operands(trial,released)
                        h=float(released.value(torch.tensor(l,dtype=z.dtype),torch.tensor(r,dtype=z.dtype)))
                        if h*chosen["side"] < -128*EPS*scale: continue
                        release_confirmed=h*chosen["side"] > 128*EPS*scale
                        if not release_confirmed and not located: continue
                    if trial["native_J"]<current["native_J"] and trial["native_J"]<=current["native_J"]+1e-4*effective_alpha*slope:
                        state["accepted_steps"].append({"iteration":iteration,"mode":mode,"from_control_sha256":current["control_sha256"],
                            "control":execution.lift(trial_z).tolist(),"coordinates":trial_z.tolist(),"control_sha256":trial["control_sha256"],
                            "native_J":trial["native_J"],"alpha":alpha,"effective_alpha":effective_alpha,"slope":slope,
                            "armijo_threshold":current["native_J"]+1e-4*effective_alpha*slope,
                            "release_confirmed":release_confirmed,
                            "events":[asdict(e) for e in reached],"located":[asdict(e) for e in located]})
                        z,current=trial_z,trial;state["current_point"]=current;accepted=True;write(OUT/"RESULT.json",state);break
                except ValueError as error:
                    state.setdefault("rejected_trials",[]).append({"iteration":iteration,"alpha":alpha,"error":str(error)})
            if not accepted: state["status"]="no_supported_native_progress";break
        else: state["status"]="step_cap_reached"
    except Exception as error: state.update(status="interrupted_or_unsupported",error=f"{type(error).__name__}: {error}")
    state["in_terminal"]=True
    try:
        terminal=execution.value(z,"terminal_native_audit");state["terminal"]=terminal
        active=execution.actual_events(terminal)
        cache=next((f for f in reversed(state.get("side_families",[])) if f["control_sha256"]==terminal["control_sha256"]),None)
        family=cache if cache is not None else execution.family(z,terminal,active)
        terminal["first_order_cache_reused"]=cache is not None
        execution.terminal_qualification(z,terminal,active,family)
    except Exception as error: state["terminal_audit_error"]=f"{type(error).__name__}: {error}"
    state["parameters_unchanged"]=tensor_sha(execution.parameters)==plan["parameters_sha256"]
    close_claims(state,all(sha(ROOT/s)==d for s,d in source_hashes.items()),
        state["parameters_unchanged"] and all(sha(ROOT/p)==d for p,d in inputs.items()))
    state.update(elapsed_seconds=time.monotonic()-start,current_control=execution.lift(z).tolist(),current_control_sha256=tensor_sha(execution.lift(z)),
        cumulative_direct_J_started=594+state["counts"]["direct_J_started"],cumulative_direct_J_completed=594+state["counts"]["direct_J_completed"],
        cumulative_gradient_started=163+state["counts"]["gradient_started"],cumulative_gradient_completed=163+state["counts"]["gradient_completed"])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    state["peak_RSS_bytes"]=peak if sys.platform=="darwin" else peak*1024
    write(OUT/"RESULT.json",state);return state


if __name__=="__main__":
    result=run();print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],"counts":result["counts"],
        "accepted_steps":len(result["accepted_steps"]),"terminal_J":result.get("terminal",{}).get("native_J"),"error":result.get("error"),"terminal_error":result.get("terminal_audit_error")}))
