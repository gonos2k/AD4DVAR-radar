"""Common binding parity against immutable, pre-refactor adapter source."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from advar.fv_research_problem import FVResearchProblem
from examples.weather_scenarios import fv_minmod_matrix_free_probe as small
from examples.weather_scenarios import fv_scaled_research_case as large

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'graphify-out/fv-root-cause-20260919'


def reference_module(artifact, name, filename, expected_sha):
    source=(EVIDENCE/artifact).read_bytes()
    assert hashlib.sha256(source).hexdigest()==expected_sha
    module=types.ModuleType(name)
    module.__file__=str(ROOT/'examples/weather_scenarios'/filename)
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


@pytest.fixture(scope='module',params=['4x5','8x10'])
def bound_case(request):
    if request.param=='4x5':
        legacy=reference_module('legacy_fv4x5_adapter_0956709.py.txt','legacy4x5',
            'fv_minmod_matrix_free_probe.py','bdd42e0b5241ec4f40e205883391a3ca38838b061da68807d48083477fa66f42')
        obs,frozen,boundary,support=legacy._load('fv_minmod_inverse_probe').make_spatial_case()
        saved=json.loads((EVIDENCE/'minmod_middle_time_bias_final.json').read_text())
        c=torch.tensor(saved['nominal_control'],dtype=torch.float64)
        p=torch.cat((obs.dbz.flatten(),obs.dbz.new_tensor([.02])))
        pattern=torch.linspace(-.2,.3,20,dtype=p.dtype).reshape(4,5)
        contract=replace(frozen,initial_background_dbz=obs.dbz[0]+p[-1]*pattern)
        forecast=legacy.v.forecast_fv_analysis(c,contract,leads=1,boundary_start_interval=2,
                                              boundary_echo=boundary,boundary_support=support)
        verification=(legacy._load('fv_minmod_inverse_probe').echo_to_dbz(
            forecast.frames_linear[-1],min_dbz=-10.)+.1*pattern).detach()
        assert legacy._digest(p)==saved['input_identity']['parameters_sha256']
        response_reference=json.loads((EVIDENCE/'minmod_parameter_vjp.json').read_text())
        assert legacy._digest(verification)==response_reference['input_identity']['verification']
        args=(obs,frozen,boundary,support,pattern,verification,saved['nominal_branch'])
        problem=small.make_research_problem(*args)
        old=legacy.make_research_functions(*args)
    else:
        legacy=reference_module('fv86_measured_case.py.txt','legacy8x10','fv_scaled_research_case.py',
            '4622da4e7cd9f3aaffd35e52ad0eacebcf6479fb3fda9d69d9f74a67123f83d5')
        original=legacy.make_case()
        case=large.make_case()
        for first,second in ((case.observations.dbz,original.observations.dbz),
                             (case.parameters,original.parameters),(case.pattern,original.pattern),
                             (case.verification,original.verification)):
            torch.testing.assert_close(first,second,rtol=0,atol=0)
        saved=json.loads((EVIDENCE/'fv86_seed_a.json').read_text())
        c=torch.tensor(saved['workflow']['control'],dtype=torch.float64)
        p=case.parameters
        assert hashlib.sha256(p.numpy().tobytes()).hexdigest()==saved['input_identity']['parameters']
        assert hashlib.sha256(case.verification.numpy().tobytes()).hexdigest()==saved['input_identity']['verification']
        problem=large.make_problem(case,saved['nominal_branch'])
        old=legacy.functions(original,saved['nominal_branch'])
    return problem,old,c,p


def test_common_objective_score_and_derivatives_match_frozen_reference(bound_case):
    problem,(old_j,old_e,_),c,p=bound_case
    direction=torch.sin(torch.arange(c.numel(),dtype=c.dtype))
    torch.testing.assert_close(problem.objective(c,p),old_j(c,p),rtol=0,atol=0)
    torch.testing.assert_close(problem.score(c,p),old_e(c,p),rtol=0,atol=0)
    for new,old in ((problem.objective,old_j),(problem.score,old_e)):
        actual=torch.func.grad(new,argnums=(0,1))(c,p)
        expected=torch.func.grad(old,argnums=(0,1))(c,p)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    fresh_grad=torch.func.grad(problem.objective,argnums=0)
    reference_grad=torch.func.grad(old_j,argnums=0)
    actual=torch.func.jvp(lambda x:fresh_grad(x,p),(c,),(direction,))[1]
    expected=torch.func.jvp(lambda x:reference_grad(x,p),(c,),(direction,))[1]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_common_layout_identity_and_branch_match(bound_case):
    problem,(_,_,old_check),c,p=bound_case
    layout=problem.layout
    assert layout['controls']==c.numel()
    assert layout['parameters']==p.numel()
    assert layout['observation_values']==problem.observations.dbz.numel()
    assert layout['theta_index']==p.numel()-1
    assert layout['observation_times_seconds']==(0.,60.,120.)
    assert layout['forecast_time_seconds']==180.
    assert layout['euler_stages'] in (54,108)
    assert problem.branch_check(c,p)[0]==old_check(c,p)[0]
    assert replace(problem).identity==problem.identity
    changed=replace(problem,verification=problem.verification+.01)
    assert changed.identity!=problem.identity
    assert problem.support['general_minmod_response_eligible'] is False


@pytest.mark.parametrize('mutation',['missing','boundary','operator','stage_signature'])
def test_common_contract_refuses_unsupported_variants(bound_case,mutation):
    problem,_,c,p=bound_case
    with pytest.raises(ValueError):
        if mutation=='missing':
            replace(problem,observations=replace(problem.observations,
                valid_mask=torch.zeros_like(problem.observations.valid_mask)))
        elif mutation=='boundary':
            supports=tuple(tuple(tuple(torch.zeros_like(x) for x in edges) for edges in stages)
                           for stages in problem.future_boundary_support)
            replace(problem,future_boundary_support=supports)
        elif mutation=='operator':
            replace(problem,observation_operator='arbitrary_regridding')
        else:
            altered=replace(problem,expected_branch={'choices':[],'face_signs':[]})
            altered.branch_check(c,p)


def test_legacy_profile_wrappers_keep_fixed_control_count(bound_case):
    problem,_,_,p=bound_case
    frozen=replace(problem.frozen,active_field_index=problem.frozen.active_field_index[:-1])
    with pytest.raises(ValueError,match='control'):
        if problem.layout['state_shape']==(4,5):
            small.make_research_problem(problem.observations,frozen,problem.future_boundary_echo,
                problem.future_boundary_support,problem.pattern,problem.verification,problem.expected_branch)
        else:
            altered=replace(large.make_case(),frozen=frozen,parameters=p)
            large.make_problem(altered,problem.expected_branch)
