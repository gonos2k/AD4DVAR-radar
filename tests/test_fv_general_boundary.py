"""General known boundary traces for positive minmod FV, with independent oracles."""
from pathlib import Path
import sys

import pytest
import torch

from advar.transport import finite_volume_step, face_volume_fluxes
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/weather_scenarios'))
from muscl_experiment import muscl_step


def edges(q, value):
    h, w = q.shape
    return tuple(torch.full((n,), value, dtype=q.dtype) for n in (h, h, w, w))


def one_cell(left0=0., left1=1., growth=0.):
    q = torch.zeros((1, 1), dtype=torch.float64)
    zero = q.new_zeros(1)
    e0 = (q.new_tensor([left0]), zero, zero, zero)
    e1 = (q.new_tensor([left1]), zero, zero, zero)
    return finite_volume_step(q, torch.ones_like(q), q.new_ones((1, 2)), q.new_zeros((2, 1)),
        dt_seconds=.2, spacing_yx=(1., 1.), log_growth=growth,
        boundary_echo=(e0, e1), boundary_support=(edges(q, 1),)*2, reconstruction='minmod')


def test_minmod_stage_boundary_has_independent_growth_oracle():
    growth = .3
    result = one_cell(2., 3., growth)
    # r1=.2*b0; r2=r1+.2*(exp(-g)*b1-r1); qnew=exp(g)*r2/2.
    r1 = .2*2
    r2 = r1 + .2*(3*torch.exp(torch.tensor(-growth,dtype=torch.float64))-r1)
    expected = torch.exp(torch.tensor(growth,dtype=torch.float64))*r2/2
    torch.testing.assert_close(result.echo.squeeze(), expected)
    assert float(result.transport_inflow) == pytest.approx(.1*(2+3*torch.exp(torch.tensor(-growth,dtype=torch.float64)).item()))
    assert float(result.transport_outflow) == pytest.approx(.1*r1)


def test_minmod_boundary_jump_uses_separate_step_endpoints():
    assert float(one_cell(0, 0).echo) == 0
    assert float(one_cell(1, 1).echo) == pytest.approx(.18)


@pytest.mark.parametrize('support', [0., .5, .999])
def test_minmod_rejects_unknown_partial_initial_or_inflow(support):
    q = torch.ones((2, 2), dtype=torch.float64)
    args = dict(dt_seconds=.1, spacing_yx=(1.,1.), boundary_echo=(edges(q,0),)*2,
                boundary_support=(edges(q,1),)*2, reconstruction='minmod')
    qx, qy = q.new_ones((2,3)), q.new_zeros((3,2))
    with pytest.raises(ValueError, match='known|support'):
        finite_volume_step(q, torch.full_like(q,support), qx,qy,**args)
    args['boundary_support']=(edges(q,support),)*2
    with pytest.raises(ValueError, match='known|support'):
        finite_volume_step(q, torch.ones_like(q), qx,qy,**args)


def test_unused_outflow_boundary_support_need_not_be_known():
    q = torch.ones((1,1), dtype=torch.float64)
    known_left = (q.new_ones(1),q.new_zeros(1),q.new_zeros(1),q.new_zeros(1))
    result = finite_volume_step(q,q,q.new_ones((1,2)),q.new_zeros((2,1)),dt_seconds=.2,
        spacing_yx=(1.,1.),boundary_echo=(edges(q,1),)*2,boundary_support=(known_left,)*2,
        reconstruction='minmod')
    torch.testing.assert_close(result.echo,q)
    torch.testing.assert_close(result.support,q)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_known_zero_matches_frozen_minmod(dtype):
    q = torch.arange(30,dtype=dtype).reshape(5,6)/10
    y,x=torch.meshgrid(torch.arange(6,dtype=dtype),torch.arange(7,dtype=dtype),indexing='ij')
    qx,qy=face_volume_fluxes(.1*x*y+.3*y-.2*x)
    expected,out=muscl_step(q,qx,qy,dt_seconds=.1,spacing_yx=(2.,3.),log_growth=.03)
    result=finite_volume_step(q,torch.ones_like(q),qx,qy,dt_seconds=.1,spacing_yx=(2.,3.),
        log_growth=.03,boundary_echo=(edges(q,0),)*2,boundary_support=(edges(q,1),)*2,reconstruction='minmod')
    # Physical-coordinate staging changes rounding, not the discrete method.
    torch.testing.assert_close(result.echo,expected,rtol=8*torch.finfo(dtype).eps,atol=0)
    torch.testing.assert_close(result.transport_outflow,out)


@pytest.mark.parametrize('log_growth', [0.02, -0.02])
def test_boundary_budget_and_derivatives_on_nonsquare_grid(log_growth):
    q = torch.tensor([[.3,.5,.9,1.2],[.7,1.1,.8,1.3],[.4,.9,1.4,1.8]],dtype=torch.float64)
    y,x=torch.meshgrid(torch.arange(4,dtype=q.dtype),torch.arange(5,dtype=q.dtype),indexing='ij')
    qx,qy=face_volume_fluxes(.2*x*y-.3*x+.4*y)
    boundary=torch.linspace(.2,1.4,14,dtype=q.dtype)
    def advance(b):
        e=tuple(b.split((3,3,4,4)))
        result=finite_volume_step(q,torch.ones_like(q),qx,qy,dt_seconds=.1,spacing_yx=(2.,3.),
            log_growth=log_growth,boundary_echo=(e,tuple(1.1*v for v in e)),
            boundary_support=(edges(q,1),)*2,reconstruction='minmod')
        residual=torch.exp(q.new_tensor(-log_growth))*result.echo.sum()*6-q.sum()*6-result.transport_inflow+result.transport_outflow
        assert abs(float(residual.detach()))<1e-12
        return result.echo
    direction=torch.cos(torch.arange(14,dtype=q.dtype))*.1
    _,jvp=torch.func.jvp(advance,(boundary,),(direction,))
    _,vjp=torch.func.vjp(advance,boundary)
    cotangent=torch.sin(q)
    torch.testing.assert_close((jvp*cotangent).sum(),(direction*vjp(cotangent)[0]).sum(),rtol=1e-11,atol=1e-13)
    fd=(advance(boundary+1e-5*direction)-advance(boundary-1e-5*direction))/2e-5
    torch.testing.assert_close(jvp,fd,rtol=1e-7,atol=1e-10)


def test_minmod_uses_its_own_cfl_bound():
    q = torch.ones((1,1),dtype=torch.float64)
    with pytest.raises(ValueError,match='CFL|courant'):
        finite_volume_step(q,q,q.new_ones((1,2)),q.new_zeros((2,1)),dt_seconds=.6,
            spacing_yx=(1.,1.),boundary_echo=(edges(q,1),)*2,boundary_support=(edges(q,1),)*2,
            reconstruction='minmod')
