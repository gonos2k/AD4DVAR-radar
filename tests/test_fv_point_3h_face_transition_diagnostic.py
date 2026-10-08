from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from advar import transport
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as diag


def profile():
    y, x = torch.meshgrid(torch.arange(5, dtype=torch.float64), torch.arange(6, dtype=torch.float64), indexing="ij")
    basis = torch.stack((y, x, x*y, (x.square()-y.square())/2, x.square()*y))
    spec = SimpleNamespace(psi_basis=basis,
        coefficient_limits=torch.tensor([.11,.08,.07,.04,.03],dtype=torch.float64),
        substeps_per_interval=900,spacing_yx=(10.,10.),reconstruction="minmod",max_courant=.5)
    return SimpleNamespace(frozen=SimpleNamespace(fv_transport=spec,
        nowcast_config=SimpleNamespace(interval_minutes=10,max_log_growth_per_step=.3)))


@pytest.mark.parametrize("face", [{"axis":"y","row":4,"column":3},{"axis":"x","row":1,"column":2}])
@pytest.mark.parametrize("eta", [-1e-6,0.,1e-6])
def test_generic_chart_keeps_retained_controls_and_matches_production(face,eta):
    problem=profile();base=torch.linspace(-.02,.03,26,dtype=torch.float64)
    weights=diag._face_weights(problem,**face);pivot=diag._pivot(base,weights)
    point=diag._chart(base,weights,pivot,eta);retained=[i for i in range(26) if i!=pivot]
    assert torch.equal(point[retained],base[retained])
    assert diag._verify_face_value(problem,point,weights,face,eta)==pytest.approx(eta,abs=2e-16)
    _,z,v=diag._chart_jacobian(base,weights,pivot,eta);normal=diag._face_normal(point,weights)
    assert float((normal@z).abs().max())<2e-16
    assert float(normal@v)==pytest.approx(1.,abs=2e-15)
    def gamma(values):
        control=base.clone();control[retained]=values
        return diag._chart(control,weights,pivot,eta)
    assert torch.allclose(torch.func.jacrev(gamma)(base[retained]),z,atol=2e-15,rtol=2e-15)


def test_external_weights_and_projection_metric():
    problem=profile();base=torch.zeros(26,dtype=torch.float64)
    w=diag._face_weights(problem,"y",4,3)
    assert torch.allclose(w,torch.tensor([0.,-.08,-.28,-.14,-.84],dtype=torch.float64),atol=2e-16,rtol=0.)
    g=torch.sin(torch.arange(26,dtype=torch.float64));point=diag._chart(base,w,24,1e-6)
    geo=diag._geometry(point,w,24,1e-6,g);n=diag._face_normal(point,w)
    gt=g-n*(n@g)/(n@n)
    assert geo["tangent_gradient_norm"]==pytest.approx(float(gt.norm()),abs=2e-14)
    assert geo["tangent_gradient_norm_projection"]==pytest.approx(float(gt.norm()),abs=2e-14)
    assert geo["normal_gradient_magnitude"]==pytest.approx(abs(float(n@g))/float(n.norm()))


@pytest.mark.parametrize("log", [.01,0.,-.01])
@pytest.mark.parametrize("stage", [0,1])
def test_effective_edge_growth_matches_euler_parity(log,stage):
    edge=torch.tensor([2.,3.],dtype=torch.float64)
    edges=(edge,edge.clone(),edge.clone(),edge.clone())
    growth=torch.tensor(log,dtype=torch.float64)
    effective=diag._effective_edges(edges,stage,growth)
    for raw,eff in zip(edges,effective,strict=True):
        expected=transport._scale_by_growth(raw,growth) if stage==0 and log>0 else raw
        assert torch.equal(eff,expected)


@pytest.mark.parametrize("axis,row,col,edge", [("y",4,3,3),("y",0,3,2),("x",1,0,0),("x",1,5,1)])
def test_boundary_donors_distinguish_raw_effective_and_internal(axis,row,col,edge):
    q=torch.arange(20,dtype=torch.float64).reshape(4,5)+10
    sx,sy=transport._muscl_slopes(q)
    edges=(torch.full((4,),2.,dtype=torch.float64),torch.full((4,),3.,dtype=torch.float64),
           torch.full((5,),4.,dtype=torch.float64),torch.full((5,),5.,dtype=torch.float64))
    effective=diag._effective_edges(edges,0,torch.tensor(.01,dtype=torch.float64))
    qx=torch.ones((4,6),dtype=torch.float64);qy=torch.ones((5,5),dtype=torch.float64)
    index=row if axis=="x" else col
    for sign in [-1.,1.]:
        flux=qx if axis=="x" else qy;flux[row,col]=sign
        donor=diag._donors(q,qx,qy,sx,sy,edges,effective,axis,row,col)
        assert donor["raw_boundary_edge"]==float(edges[edge][index])
        assert donor["selected_side"]==("minus" if sign>0 else "plus")
        outside_key="minus_donor" if edge in (0,2) else "plus_donor"
        assert donor[outside_key]==float(effective[edge][index])
        inside_key="plus_donor" if edge in (0,2) else "minus_donor"
        internal=q[0,col] if edge==2 else q[-1,col] if edge==3 else q[row,0] if edge==0 else q[row,-1]
        assert donor[inside_key]==float(internal)


@pytest.mark.parametrize("plus", [[-2.,1.],[2.,1.],[-1.,1.]])
def test_segment_minimizer_is_full_gradient_not_tangent_mean(plus):
    minus=torch.tensor([-1.,1.],dtype=torch.float64);gp=torch.tensor(plus,dtype=torch.float64)
    normal=torch.tensor([1.,0.],dtype=torch.float64)
    data=diag._gradient_pair(minus,gp,normal);hull=data["minimum_norm_convex_hull"]
    v=torch.tensor(hull["gradient"],dtype=torch.float64);d=-v
    assert 0<=hull["theta"]<=1
    assert float(minus@d)<=-float(v@v)+2e-15
    assert float(gp@d)<=-float(v@v)+2e-15
    assert data["phi_jump_identity"]==pytest.approx(float((gp@gp-minus@minus)/2))
    jump=gp-minus
    assert torch.allclose(torch.tensor(data["zero_chart_tangent_jump"],dtype=torch.float64),
                          jump-normal*(normal@jump)/(normal@normal))


def test_inner_and_outer_pairs_are_distinct_finite_diagnostics():
    control=torch.zeros(26,dtype=torch.float64);w=torch.tensor([0.,-.08,-.28,-.14,-.84],dtype=torch.float64)
    samples={}
    for eta in [-2e-6,-1e-6,1e-6,2e-6]:
        g=torch.ones(26,dtype=torch.float64);g[24]=eta*1e6
        samples[eta]={"gradient":g.tolist()}
    pairs=diag._gradient_jump(control,w,24,samples)
    assert pairs["zero_gradient_evaluated"] is False
    assert pairs["outer_pair"]["gradient_jump_l2"]==pytest.approx(2*pairs["inner_pair"]["gradient_jump_l2"])


def test_zero_cost_keeps_full_prior_without_gradient_or_branch(monkeypatch):
    p=profile();control=torch.linspace(-.02,.03,26,dtype=torch.float64)
    w=diag._face_weights(p,"y",4,3);point=diag._chart(control,w,24,0.)
    calls=[]
    def objective(c,params):
        calls.append(c.clone());return 1.+c.square().sum()/2
    p.objective=objective
    def forbidden(*args,**kwargs):raise AssertionError("zero-point differentiation/branch called")
    p.branch_check=forbidden;monkeypatch.setattr(torch.func,"grad",forbidden)
    result=diag._cost_only(p,point,torch.zeros(13,dtype=torch.float64),w,
        {"axis":"y","row":4,"column":3},time.monotonic()+10)
    assert result["objective"]==float(1.+point.square().sum()/2)
    assert result["gradient_computed"] is False and result["branch_checked"] is False
    assert "gradient" not in result and "phi" not in result
    assert torch.equal(calls[0],point)
