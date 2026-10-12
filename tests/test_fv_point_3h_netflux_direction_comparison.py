from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from examples.weather_scenarios import fv_point_3h_netflux_direction_comparison as compare


def test_efficiency_measures_angle_not_step_scale():
    G=torch.tensor([800.**.5,1.],dtype=torch.float64)*1e-3
    H=torch.tensor([1.,800.],dtype=torch.float64)
    grad=compare.direction_efficiency(SimpleNamespace(residual=G,residual_direction=-H*G))
    newton=compare.direction_efficiency(SimpleNamespace(residual=G,residual_direction=-G))
    assert grad["chi"]==pytest.approx(4*800/801**2)
    assert grad["unbounded_model_alpha"]==pytest.approx(2/801)
    assert newton["chi"]==pytest.approx(1.)
    scaled=compare.direction_efficiency(SimpleNamespace(residual=G,residual_direction=-3*H*G))
    assert scaled["chi"]==pytest.approx(grad["chi"])


def test_selection_uses_both_repeated_merits_and_never_a_single_arm():
    arms=[{"name":"baseline_tangent","status":"independent_P2_fullG_eligible",
           "repeat":{"F_squared":2.,"objective":1.},"candidate":{"F_squared":.1}},
          {"name":"robust_gn_coupled","status":"independent_P2_fullG_eligible",
           "repeat":{"F_squared":1.,"objective":1.},"candidate":{"F_squared":3.}}]
    assert compare.choose_repeated_arm(arms,torch.float64) is arms[1]
    arms[1]["status"]="model_refused"
    assert compare.choose_repeated_arm(arms,torch.float64) is None
    assert compare.choose_repeated_arm(arms[:1],torch.float64) is None
    arms[1]["status"]="independent_P2_fullG_eligible"
    arms[1]["repeat"]["F_squared"]=2.
    assert compare.choose_repeated_arm(arms,torch.float64) is arms[0]


def _run(monkeypatch,tmp_path, *, bad_gn_repeat=False, gn_timeout=False):
    dtype=torch.float64
    H=torch.ones(26,dtype=dtype);H[0]=0.;H[2]=800.
    c=torch.zeros(26,dtype=dtype);c[1]=800.**.5*1e-3;c[2]=1e-3/800.
    normal=torch.zeros_like(c);normal[0]=1.
    counts={"HVP":0,"row":0,"repeat":0}
    def observe(control):
        gm=H*control;gp=gm.clone();gm[0]=-1.;gp[0]=1.
        trace={"signature_sha256":compare.tangent._tensor_sha(control)}
        cost=.5*torch.dot(control,H*control)
        return {"native_j":cost,"q":torch.tensor(0.,dtype=dtype),
            "side":{-1:(cost,gm),1:(cost,gp)},"traces":{-1:trace,1:trace},
            "pair":{"passed":True},"face_ok":True,"side_objectives_match_native":True,"side_gradients_finite":True}
    def repeat(control):
        obs=observe(control)
        return {"control":control.tolist(),"theta":.5,"objective":float(obs["native_j"]),
            "F_squared":float(torch.dot(H*control,H*control)),
            "side_gradients":{str(s):obs["side"][s][1].tolist() for s in (-1,1)},
            "branch_trace":{str(s):obs["traces"][s] for s in (-1,1)}}
    base={"control":c,"theta":.5,"full_G":torch.cat((H*c,torch.zeros(1,dtype=dtype))),
        "repeat":repeat(c),"raw":{},"input_after":{"fixed":True},"runtime_after":{"fixed":True}}
    class Backend:
        def __init__(self,*_args):
            self.control=c.clone();self.weights=normal;self.parameters=torch.zeros(13,dtype=dtype)
            self.source_anchor={"source":"fixed"};self.input_anchor=base["input_after"];self.runtime_anchor=base["runtime_after"]
            self.problem=SimpleNamespace(objective=lambda x,p:.5*torch.dot(x,H*x))
            self.original=self.truth=object();self.timings={"pure_hvp_seconds":0.,"row_solve_seconds_inclusive":0.};self.row_count=self.solve_count=0
        def _check(self,*_args): pass
        def evaluate(self,bundle,candidate,alpha):
            r=repeat(candidate)
            return {**r,"alpha":alpha,"G":[* (H*candidate).tolist(),0.],
                "accepted":True,"mixing_minimum_valid":True,"J_armijo_passed":True,"F_squared_armijo_passed":True,
                "branch_pair_passed":True,"face_audit_passed":True,"side_objectives_match_native":True,
                "side_gradients_finite":True,"finite":True,"branch_trace":r["branch_trace"]}
        def chart(self,bundle,control,direction,alpha): return control+alpha*direction
        def repeat(self,bundle,selected):
            counts["repeat"]+=1
            x=torch.tensor(selected["control"],dtype=dtype);r=repeat(x)
            if bad_gn_repeat and counts["repeat"]==2:r["F_squared"]=1.
            return ({**selected,"control_sha256":compare.tangent._tensor_sha(x),
                "J_armijo_bound":float(base["repeat"]["objective"])*.999,
                "F_squared_armijo_bound":float(base["repeat"]["F_squared"])*.999,
                "conditional_Q0":True},r)
    def factory(context):
        def build(control,theta,gm,gp,n,z,pivot):
            counts["row"]+=24
            op=context["record"];op.update(jacobian_rows_started=24,jacobian_rows_completed=24,
                dense_solves_started=1,dense_solves_completed=1)
            if gn_timeout: raise TimeoutError("GN-only preparation hit budget")
            mixed=.5*(gm+gp)
            return [{"name":"baseline_tangent","direction":-mixed},
                {"name":"robust_gn_coupled","direction":-control}]
        return build
    def model(gm,gp,hm,hp,n,z,theta,q,pivot,scale,*,direction_override):
        G=torch.cat((.5*(gm+gp),torch.zeros(1,dtype=dtype)))
        r=torch.cat((.5*(hm+hp),torch.zeros(1,dtype=dtype)))
        gates={k:True for k in ("finite","both_side_gradients_descend","merit_descends","envelope_slope_matches_residual_dot")}
        return SimpleNamespace(direction=direction_override,residual=G,residual_direction=r,gates=gates,
            merit_product=float(torch.dot(G,r)),theta=theta,delta_theta=0.,side_products=(float(torch.dot(gm,direction_override)),float(torch.dot(gp,direction_override))))
    def full_g(r,expected,weights):
        x=torch.tensor(r["control"],dtype=dtype);G=torch.cat((H*x,torch.zeros(1,dtype=dtype)))
        merit=float(torch.dot(G,G))
        return {"G":G.tolist(),"passed":torch.equal(G,expected) and merit==r["F_squared"]}
    original_jvp=torch.func.jvp
    def jvp(*args,**kwargs):
        counts["HVP"]+=1
        return original_jvp(*args,**kwargs)
    monkeypatch.setattr(compare,"ROOT",tmp_path);monkeypatch.setattr(compare,"ATTEMPT",tmp_path/"attempt")
    monkeypatch.setattr(compare,"_sha",lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(compare,"load_plan",lambda *_a:{})
    monkeypatch.setattr(compare.endpoint,"load_endpoint",lambda:base)
    monkeypatch.setattr(compare,"CurrentBackend",Backend)
    monkeypatch.setattr(compare.tangent,"_observe",lambda _s,_p,x,*_a:observe(x))
    monkeypatch.setattr(compare.geometry,"_face_normal",lambda *_a:normal)
    monkeypatch.setattr(compare.geometry,"_pivot",lambda *_a:0)
    monkeypatch.setattr(compare.geometry,"_chart_jacobian",lambda *_a:(None,torch.eye(26,dtype=dtype)[:,1:],None))
    monkeypatch.setattr(compare.coupled,"_direction_model_factory",factory)
    monkeypatch.setattr(compare.torch.func,"jvp",jvp)
    monkeypatch.setattr(compare.prepared.mixing,"minimum_tangent_model",model)
    monkeypatch.setattr(compare.finite,"_base_model_closure",lambda *_a:{"passed":True})
    monkeypatch.setattr(compare.previous,"fresh_full_G",full_g)
    monkeypatch.setattr(compare.tangent,"fresh_final_closure",lambda *_a:True)
    monkeypatch.setattr(compare.shared,"_input_identity",lambda *_a:base["input_after"])
    monkeypatch.setattr(compare.shared,"_source_hashes",lambda *_a:{"source":"fixed"})
    monkeypatch.setattr(compare.shared,"_runtime",lambda:base["runtime_after"])
    monkeypatch.setattr(compare.geometry.model,"_fixed_input",lambda *_a:True)
    result=compare._run_child(tmp_path/"PLAN.json","synthetic",tmp_path/"step.json")
    return result,counts


def test_same_point_two_direction_hvps_and_repeats_commit_exact_curvature(monkeypatch,tmp_path):
    raw,counts=_run(monkeypatch,tmp_path)
    assert raw["execution_status"]=="completed",raw.get("termination_reason")
    assert raw["optimizer_steps_applied"]==1 and raw["selected_direction"]=="robust_gn_coupled"
    assert counts=={"HVP":4,"row":24,"repeat":2}
    assert all(a["status"]=="independent_P2_fullG_eligible" for a in raw["arms"])
    assert raw["GN_rows_completed"]==24 and raw["GN_solves_completed"]==1
    assert raw["arms"][1]["efficiency"]["chi"]==pytest.approx(1.)
    assert raw["terminal_E_recomputed"] is False


def test_failed_curvature_repeat_cannot_commit_baseline_fallback(monkeypatch,tmp_path):
    raw,counts=_run(monkeypatch,tmp_path,bad_gn_repeat=True)
    assert raw["execution_status"]=="failed"
    assert raw["optimizer_steps_applied"]==0 and not raw["candidate_committed"]
    assert raw["current_control"]==raw["last_confirmed_control"]
    assert counts["repeat"]==2


def test_GN_budget_retains_baseline_evidence_and_base_without_fallback(monkeypatch,tmp_path):
    raw,counts=_run(monkeypatch,tmp_path,gn_timeout=True)
    assert raw["numerical_status"]=="budget_exhausted_comparison_incomplete_no_commit"
    assert raw["arms"][0]["status"]=="independent_P2_fullG_eligible"
    assert raw["GN_rows_completed"]==24
    assert raw["optimizer_steps_applied"]==0 and not raw["candidate_committed"]
    assert counts["HVP"]==2
