"""Event-location integration with scalar paths and full selector encodings."""
import pytest
import torch
from types import SimpleNamespace

from examples.weather_scenarios import fv_point_3h_limiter_event_progress as m


def fake_execution(monkeypatch, *, compound=False, simultaneous=False):
    execution=object.__new__(m.EventExecution)
    execution.state={}
    monkeypatch.setattr(m,"write",lambda *_: None)
    def value(z,purpose):
        t=float(z[0]);left=.4-t if compound else 2.;right=.6-t if compound else .4-t
        raw=bytearray(4320)
        choice=1 if left>0 and right>0 and left<right else 2 if left>0 and right>0 else 2 if left<0 and right<0 else 0
        raw[124*12+6+5]=choice
        if simultaneous: raw[125*12+6+5]=choice
        active=[]
        if min(abs(left),abs(right))<128*m.EPS*40:
            active.append({"stage":124,"axis":"y","cell":[2,3],"left":left,"right":right,"q_scale":40.})
            if simultaneous: active.append(dict(active[0],stage=125))
        return {"index":0,"coordinates":z.tolist(),"native_J":(1-t)**2,
            "control_sha256":str(t),"left":left,"right":right,
            "exact_minmod_selectors_hex":raw.hex(),"flux_signs":[1]*49,
            "active_limiter_events":active,"branch":{"unique_nearzero_face_entities":["y:2:1","y:4:3"]}}
    execution.value=value
    execution.operands=lambda row,event:(row["left"],row["right"],40.)
    execution.restore_events=lambda z,events,normals:(z,value(z,"path"))
    return execution


@pytest.mark.parametrize("compound",[False,True])
def test_actual_path_locates_operand_zero_at_arbitrary_stage_and_axis(monkeypatch,compound):
    execution=fake_execution(monkeypatch,compound=compound)
    a=torch.tensor([0.],dtype=torch.float64);b=torch.tensor([1.],dtype=torch.float64)
    point,row,events,fraction=execution.locate(a,execution.value(a,"start"),b,execution.value(b,"end"))
    assert abs(float(point[0])-.4)<1e-10
    assert fraction==float(point[0])
    assert events[0].key==(124,"y",2,3)
    assert events[0].kind==("left_zero" if compound else "right_zero")
    assert row["native_J"]<1


def test_location_preserves_simultaneous_current_event_union(monkeypatch):
    execution=fake_execution(monkeypatch,simultaneous=True)
    a=torch.tensor([0.],dtype=torch.float64);b=torch.tensor([1.],dtype=torch.float64)
    _,_,events,_=execution.locate(a,execution.value(a,"start"),b,execution.value(b,"end"))
    assert {e.stage for e in events}=={124,125}


def test_nonstationary_terminal_does_not_construct_certification_hessian():
    execution=object.__new__(m.EventExecution)
    execution.tape=lambda *_a,**_k:pytest.fail("nonstationary terminal must not build P2 tape")
    row={}
    execution.terminal_qualification(torch.zeros(24,dtype=torch.float64),row,[],{"chart_hull_inf":1e-3})
    assert row["minimum_claim"] is False and row["P2_complete"] is False


def test_concave_kink_zero_hull_does_not_override_actual_outgoing_descent():
    assert m.event_action([{"decision":"release"}],0.)=="release"
    assert m.event_action([{"decision":"maintain"}],0.)=="stationary"


def test_family_refuses_sector_average_release_direction_that_enters_one_sector(monkeypatch):
    from advar import variational as v

    monkeypatch.setattr(m, "write", lambda *_args: None)
    monkeypatch.setattr(v, "_control_prior_residual", lambda control, _contract: control * 0.0)
    monkeypatch.setattr(v, "_field_smoothness_prior_cost", lambda control, _contract: control.sum() * 0.0)
    execution = object.__new__(m.EventExecution)
    execution.lift = lambda z: torch.cat((z, z.new_zeros(2)))
    execution.parameters = torch.zeros(1, dtype=torch.float64)
    execution.profile = SimpleNamespace(contract=lambda _parameters: None)
    execution.state = {}
    execution.geo = SimpleNamespace(geometry=SimpleNamespace(
        minimum_sampled_hull=lambda _vertices: {"vector": [0.0] * 24, "weights": [0.5, 0.5]}))

    def gradient(value, variable, *, graph=False):
        return torch.autograd.grad(value, variable, retain_graph=True, create_graph=graph)[0]

    execution.gradient = gradient
    event = m.MinmodEvent(48, "x", 1, 1, "gap", 1)
    native_j = 1.0

    def tape(point, *, events=(), signs=(), sides=(1, 1), full_control=False):
        assert full_control is True
        variable = point.detach().clone().requires_grad_(True)
        objective_gradient = torch.zeros_like(variable)
        objective_gradient[0] = 1.0 if signs[0] == -1 else -1.0
        phi = native_j + torch.dot(objective_gradient, variable - variable.detach())
        # Three flux sectors have a steep positive event-normal y component;
        # the fourth has the opposite component. Their normal costs agree,
        # while their averaged normal points into one sector.
        normal = torch.zeros_like(variable)
        normal[0] = 1.0
        normal[1] = -10.0 if sides != (1, 1) else 10.0
        gap = torch.dot(normal, variable - variable.detach())
        return variable, phi, {events[0].key: gap}

    execution.tape = tape
    z = torch.zeros(24, dtype=torch.float64)
    family = execution.family(z, {"control_sha256": "fixture", "native_J": native_j}, [event])
    decision = family["decisions"][0]

    assert decision["outgoing"][ -1] == pytest.approx([-1.0 / (101.0**0.5)] * 4)
    assert decision["decision"] == "unresolved"
    assert decision["stepped_direction_valid"] is False
    assert all(value < 0 for value in decision["stepped_direction_derivatives"])
