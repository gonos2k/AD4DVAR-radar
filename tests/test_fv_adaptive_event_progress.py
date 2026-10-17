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
    execution.budget = SimpleNamespace(take=lambda _kind: None, done=lambda _kind: None)
    execution.geo = SimpleNamespace(geometry=SimpleNamespace(
        minimum_sampled_hull=lambda _vertices: {"vector": [0.0] * 24, "weights": [0.5, 0.5]}))
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


def family_fixture(monkeypatch, normal, gradient_by_side, gradient_by_flux=None):
    from advar import variational as v

    monkeypatch.setattr(m, "write", lambda *_args: None)
    monkeypatch.setattr(v, "_control_prior_residual", lambda control, _contract: control * 0.0)
    monkeypatch.setattr(v, "_field_smoothness_prior_cost", lambda control, _contract: control.sum() * 0.0)
    execution = object.__new__(m.EventExecution)
    execution.lift = lambda z: torch.cat((z, z.new_zeros(2)))
    execution.parameters = torch.zeros(1, dtype=torch.float64)
    execution.profile = SimpleNamespace(contract=lambda _parameters: None)
    execution.state = {}
    execution.budget = SimpleNamespace(take=lambda _kind: None, done=lambda _kind: None)
    execution.geo = SimpleNamespace(geometry=SimpleNamespace(
        minimum_sampled_hull=lambda _vertices: {"vector": [0.0] * 24, "weights": [1.0]}))

    def tape(point, *, events=(), signs=(), sides=(1, 1), full_control=False):
        assert full_control
        variable = point.detach().clone().requires_grad_(True)
        objective_gradient = torch.zeros_like(variable)
        objective_gradient[0] = gradient_by_side[signs[0]]
        if gradient_by_flux is not None:
            objective_gradient[1] = gradient_by_flux[sides]
        phi = 1.0 + torch.dot(objective_gradient, variable - variable.detach())
        gap_normal = torch.zeros_like(variable)
        gap_normal[0] = normal
        gap = torch.dot(gap_normal, variable - variable.detach())
        return variable, phi, {events[0].key: gap}

    execution.tape = tape
    event = m.MinmodEvent(48, "x", 1, 1, "gap", 1)
    family = execution.family(torch.zeros(24, dtype=torch.float64),
        {"control_sha256": "fixture", "native_J": 1.0}, [event])
    return family["decisions"][0]


@pytest.mark.parametrize(("normal", "gradient", "side"), [
    (1.0, 1.0, -1),
    (1.0, -1.0, 1),
])
def test_actual_family_proposes_descent_on_each_outgoing_side(monkeypatch, normal, gradient, side):
    decision = family_fixture(monkeypatch, normal, {-1: gradient, 1: gradient})
    assert decision["decision"] == "release"
    assert decision["side"] == side
    assert decision["stepped_direction_valid"]
    assert decision["derivative"] < -decision["error_budget"]


def test_actual_family_maintains_when_both_outgoing_derivatives_are_positive(monkeypatch):
    decision = family_fixture(monkeypatch, 1.0, {-1: -1.0, 1: 1.0})
    assert decision["decision"] == "maintain"
    assert min(map(min, decision["outgoing"].values())) > decision["error_budget"]


def test_actual_family_leaves_roundoff_sized_outgoing_sign_unresolved(monkeypatch):
    decision = family_fixture(monkeypatch, 1.0, {-1: 0.0, 1: 0.0})
    assert decision["decision"] == "unresolved"


def test_actual_family_rejects_flux_sector_tangent_gradient_disagreement(monkeypatch):
    perturbations = {(s0, s1): 0.0 for s0 in (-1, 1) for s1 in (-1, 1)}
    perturbations[(1, 1)] = 1.0e-6
    with pytest.raises(ValueError, match="flux-sector tangent gradient disagreement"):
        family_fixture(monkeypatch, 1.0, {-1: -1.0, 1: 1.0}, perturbations)


def test_actual_family_rejects_zero_event_normal(monkeypatch):
    with pytest.raises(ValueError, match="zero normal"):
        family_fixture(monkeypatch, 0.0, {-1: 1.0, 1: 1.0})


@pytest.mark.parametrize("normal", [float("nan"), float("inf")])
def test_actual_family_rejects_nonfinite_event_derivative(monkeypatch, normal):
    with pytest.raises(ValueError, match="nonfinite derivative"):
        family_fixture(monkeypatch, normal, {-1: 1.0, 1: 1.0})
