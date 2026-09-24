"""Serial, resource-bounded 86-control execution evidence; no general FV claim."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import torch
from advar import local_refinement, local_response, variational as v

ROOT = Path(__file__).resolve().parents[2]


def load(name) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load research module {name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().numpy().tobytes()
    else:
        data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def branch_summary(branch):
    return {'euler_stages': branch['euler_stages'],
            'minimum_scaled_slope_margin': branch['minimum_scaled_slope_margin'],
            'selectors_sha256': digest(branch['choices']),
            'face_signs_sha256': digest(branch['face_signs'])}


def failure_category(message, phase):
    if 'positive definite' in message:
        return 'nonpositive_curvature'
    if 'PCG did not converge' in message:
        return 'linear_budget_exhausted'
    if 'residual' in message and ('tolerance' in message or 'nonfinite' in message):
        return 'linear_residual_failed'
    if 'Armijo' in message:
        return 'line_search_exhausted'
    if 'iteration budget exhausted' in message or 'not stationary' in message:
        return 'stationarity_not_met'
    if 'CFL' in message:
        return 'cfl_domain'
    if 'branch' in message or 'strict positive growth' in message:
        return 'branch_uncertified'
    if phase in ('prepare', 'assessment') and 'finite' in message:
        return 'nonfinite_initial'
    return 'unclassified_refusal'


def run(output: Path, *, mode: str):
    if mode not in ('preflight', 'seed_a', 'seed_b'):
        raise ValueError('unknown predeclared case')
    started = time.monotonic()
    fixture = load('fv_scaled_research_case')
    workflow = load('fv_analysis_response')
    serialize = load('fv_gn_response_probe').serializable
    paths = [Path(__file__), Path(fixture.__file__), Path(workflow.__file__),
             Path(__file__).with_name('fv86_resource_runner.py'),
             Path(__file__).with_name('fv_gn_response_probe.py'),
             Path(__file__).with_name('fv_minmod_inverse_probe.py'),
             Path(__file__).with_name('fv_sensitivity_probe.py')]
    paths += [ROOT/'src/advar'/name for name in
              ('local_refinement.py', 'local_response.py', 'fv_research_problem.py', 'matrix_free.py', 'variational.py',
               'transport.py', 'physics.py', 'nowcast.py')]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report: dict[str, Any] = {'mode': mode, 'status': 'running', 'phase': 'prepare',
              'environment': {'python':platform.python_version(),'torch':torch.__version__,'device':'CPU FP64'},
              'scope': '86-control discrete execution, not grid convergence or general minmod FSOI',
              'source_sha256': sources, 'timings': {}, 'linear_solves': [],
              'branch_checks': [], 'general_minmod_response_eligible': False,
              'finite_path_certified': False, 'nonlinear_reanalyses': 0,
              'independent_response_validation': 'not performed'}

    def checkpoint():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)

    def phase(name):
        report['phase'] = name
        checkpoint()

    checkpoint()
    try:
        case = fixture.make_case()
        p = case.parameters
        objective, score, initial_check = fixture.functions(case)
        control = v.initial_control(case.frozen)
        scale = 0.5 if mode == 'seed_b' else 1.0
        control[-6:] = control.new_tensor([.1,-.08,.06,.04,-.03,.008])*scale
        report.update(problem=case.definition, seed=control.tolist(),
                      input_identity={'parameters': digest(p), 'verification': digest(case.verification),
                                      'initial_echo': digest(case.truth_initial_echo),
                                      'basis': digest(case.frozen.fv_transport.psi_basis),
                                      'problem': digest(case.definition)},
                      seed_actual_cfl=fixture.actual_cfl(case, control))
        report['timings']['prepare_seconds'] = time.monotonic()-started
        phase('preflight' if mode == 'preflight' else 'gn')
        if mode == 'preflight':
            gradient = torch.func.grad(objective, argnums=0)
            begin = time.monotonic()
            g = gradient(control, p)
            if not bool(torch.isfinite(g).all()):
                raise ValueError('initial preflight gradient is nonfinite')
            report['timings']['gradient_seconds'] = time.monotonic()-begin
            report['gradient_max'] = float(g.abs().max())
            report['hvp_samples'] = []
            for name, direction in [('sine', torch.sin(torch.arange(control.numel(), dtype=control.dtype))),
                                    ('ones', torch.ones_like(control))]:
                direction = direction/direction.norm()
                begin = time.monotonic()
                hv = torch.func.jvp(lambda c: gradient(c,p), (control,), (direction,))[1]
                report['hvp_samples'].append({'direction': name,
                    'seconds': time.monotonic()-begin, 'rayleigh': float(direction@hv),
                    'finite': bool(torch.isfinite(hv).all())})
                checkpoint()
            begin = time.monotonic()
            try:
                branch, _ = initial_check(control,p)
                report['seed_branch'] = branch_summary(branch)
                report['seed_branch_eligible'] = True
            except ValueError as error:
                report['seed_branch_refusal'] = str(error)
                report['seed_branch_eligible'] = False
            report['timings']['branch_seconds'] = time.monotonic()-begin
            report['status'] = 'cost_measured' if all(row['finite'] for row in report['hvp_samples']) else 'preflight_nonfinite'
        else:
            gn_started = time.monotonic()
            contract = replace(case.frozen, initial_background_dbz=case.observations.dbz[0]+p[-1]*case.pattern)
            gn = v.solve_analysis(case.observations, contract, control=control)
            report['timings']['gn_seconds'] = time.monotonic()-gn_started
            report['gn'] = {name: serialize(getattr(gn,name)) for name in
                            ('control','reason','outer_iterations','pcg_iterations','initial_objective','final_objective')}
            c = gn.control
            report['gn']['actual_cfl'] = fixture.actual_cfl(case,c)
            phase('assessment')
            g = torch.func.grad(objective, argnums=0)(c,p)
            if not bool(torch.isfinite(g).all()):
                raise ValueError('initial GN endpoint gradient is nonfinite')
            report['gn']['gradient_max'] = float(g.abs().max())
            try:
                nominal, _ = initial_check(c,p)
            except ValueError as error:
                report.update(status='ineligible', failure_category=failure_category(str(error),'assessment'),
                              error=str(error))
            else:
                report['nominal_branch'] = nominal
                objective, score, check = fixture.functions(case, expected_branch=nominal)

                def compact_check(trial, parameters):
                    branch, scope = check(trial, parameters)
                    summary = branch_summary(branch)
                    summary['actual_cfl'] = fixture.actual_cfl(case, trial)
                    report['branch_checks'].append({'control_sha256':digest(trial),
                        'correction_from_gn_l2':float((trial-c).norm()), 'phase':report['phase'], **summary})
                    checkpoint()
                    return summary, scope

                def monitored_pcg(original):
                    def solve(operator, rhs, **kwargs):
                        entry: dict[str, Any] = {'phase':report['phase'], 'rtol':kwargs.get('rtol'),
                                 'max_iterations':kwargs.get('max_iterations'), 'hvp_calls':0,
                                 'rayleigh_min':None, 'rayleigh_max':None,
                                 'rayleigh_scope':'sampled directions, not a condition number or SPD certificate'}
                        report['linear_solves'].append(entry)
                        tick = time.monotonic()
                        def measured(vv):
                            hh = operator(vv)
                            entry['hvp_calls'] += 1
                            norm = vv.norm()
                            if bool(torch.isfinite(norm)) and bool(norm > 0):
                                rayleigh = float((vv/norm)@(hh/norm))
                                if math.isfinite(rayleigh):
                                    entry['rayleigh_min'] = rayleigh if entry['rayleigh_min'] is None else min(entry['rayleigh_min'],rayleigh)
                                    entry['rayleigh_max'] = rayleigh if entry['rayleigh_max'] is None else max(entry['rayleigh_max'],rayleigh)
                            if entry['hvp_calls'] % 16 == 0:
                                checkpoint()
                            return hh
                        try:
                            result = original(measured, rhs, **kwargs)
                            entry.update(converged=result.converged, iterations=result.iterations,
                                         actual_relative_residual=result.relative_residual)
                            return result
                        except (ValueError, RuntimeError) as error:
                            entry['error'] = str(error)
                            raise
                        finally:
                            entry['seconds'] = time.monotonic()-tick
                            checkpoint()
                    return solve

                def refine(cc, pp):
                    phase('refinement')
                    result = local_refinement.refine_stationary(objective,cc,pp,branch_check=compact_check)
                    report['refinement'] = serialize(result)
                    report['refinement']['correction_l2'] = float((result.control-c).norm())
                    phase('response')
                    return result.control

                observation_count = case.observations.dbz.numel()
                frame_size = case.observations.dbz[0].numel()
                bias = torch.zeros_like(p)
                bias[frame_size:2*frame_size] = 1
                directions = {f'observation_sine{observation_count}':torch.cat((torch.sin(torch.arange(observation_count,dtype=p.dtype)),p.new_zeros(1))),
                              'theta':torch.cat((p.new_zeros(observation_count),p.new_ones(1))),
                              'middle_time_bias':bias}
                phase('response_eligibility')
                with patch.object(local_refinement,'pcg',monitored_pcg(local_refinement.pcg)), \
                     patch.object(local_response,'pcg',monitored_pcg(local_response.pcg)):
                    result = workflow.prepare_response(objective,score,c,p,directions,
                        branch_check=compact_check,input_identity=report['input_identity'],refine=refine)
                report['workflow'] = serialize(result)
                report['status'] = result['status']
                if result['response'] is None:
                    report['failure_category'] = failure_category(result['error'],report['phase'])
                    report['error'] = result['error']
                else:
                    response = result['response']
                    report['full_vjp_projection_absolute_errors'] = {
                        name:abs(float(response.total_gradient@d-response.total[name]))
                        for name,d in directions.items()}
                    report['final_score'] = float(score(result['control'],p))
        report['source_unchanged'] = all(hashlib.sha256(path.read_bytes()).hexdigest()==sources[str(path.relative_to(ROOT))] for path in paths)
        if not report['source_unchanged']:
            raise RuntimeError('source changed during execution')
        phase('finished')
    except Exception as error:
        report.update(status='execution_error',error_type=type(error).__name__,error=str(error))
        checkpoint()
        raise
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight','seed_a','seed_b'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(run(args.output,mode=args.mode)['status'])
