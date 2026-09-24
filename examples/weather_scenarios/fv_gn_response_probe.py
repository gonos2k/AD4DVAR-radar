"""Explicit refinement of saved or current GN output into a response.

Serial research workflow; --fresh-gn runs the current solver on the fixed case.
"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import torch

# Import by file location so the example is runnable without an examples package.
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'graphify-out/fv-root-cause-20260919'


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if is_dataclass(value):
        return {key: serializable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def run(output, *, fresh_gn=False, matrix_free_refine=False):
    started = time.monotonic()
    bridge = load('fv_minmod_matrix_free_probe')
    workflow = load('fv_analysis_response')
    fixture = bridge._load('fv_minmod_inverse_probe')
    oracle = bridge._load('fv_sensitivity_probe')
    source_paths = (Path(__file__), Path(workflow.__file__), Path(bridge.__file__),
                    Path(oracle.__file__), Path(fixture.__file__), ROOT/'src/advar/local_response.py', ROOT/'src/advar/fv_research_problem.py',
                    ROOT/'src/advar/variational.py', ROOT/'src/advar/transport.py', ROOT/'src/advar/matrix_free.py')
    if matrix_free_refine:
        source_paths += (ROOT/'src/advar/local_refinement.py',)
    source_hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    obs, frozen, boundary, support = fixture.make_spatial_case()
    saved_path = EVIDENCE/'minmod_middle_time_bias_final.json'
    gn_path = EVIDENCE/'minmod_spatial_inverse.json'
    saved = json.loads(saved_path.read_text())
    gn = None if fresh_gn else json.loads(gn_path.read_text())['product_gn']
    bridge.check_archived_sources(saved['cache_payload_identity']['core_source_sha256'])
    c = None if fresh_gn else torch.tensor(gn['control'], dtype=torch.float64)
    reference = torch.tensor(saved['nominal_control'], dtype=torch.float64)
    p = torch.cat((obs.dbz.flatten(),obs.dbz.new_tensor([.02])))
    if bridge._digest(p) != saved['input_identity']['parameters_sha256']:
        raise ValueError('archived parameter identity mismatch')
    pattern = torch.linspace(-.2,.3,20,dtype=torch.float64).reshape(4,5)
    contract = replace(frozen,initial_background_dbz=obs.dbz[0]+p[-1]*pattern)
    forecast = fixture.echo_to_dbz(bridge.v.forecast_fv_analysis(
        reference,contract,leads=1,boundary_start_interval=2,
        boundary_echo=boundary,boundary_support=support).frames_linear[-1],min_dbz=-10.)
    verification = (forecast+.1*pattern).detach()
    objective,score,check = bridge.make_research_functions(
        obs,frozen,boundary,support,pattern,verification,saved['nominal_branch'])
    directions = {name: torch.tensor(item['parameter_direction'],dtype=torch.float64)
                  for name,item in saved['tangents'].items()}
    candidates = []
    refinement_report = None

    def refine(control, parameters):
        nonlocal refinement_report
        if matrix_free_refine:
            from advar.local_refinement import refine_stationary

            def checked(trial, fixed_parameters):
                branch, scope = check(trial,fixed_parameters)
                candidates.append({'control':trial.detach().tolist(),'branch':branch})
                return branch, scope

            refinement_report = refine_stationary(
                objective,control,parameters,branch_check=checked)
            return refinement_report.control

        def check_step(previous, trial):
            branch, _ = check(trial,parameters)
            candidates.append({'control':trial.detach().tolist(),'branch':branch})
        return oracle.polish(objective,control,parameters,check_step=check_step)

    preparation_seconds = time.monotonic()-started
    gn_seconds = 0.0
    current_gn = None
    if fresh_gn:
        seed = bridge.v.initial_control(frozen)
        seed[-6:] = seed.new_tensor([.1,-.08,.06,.04,-.03,.008])
        gn_started = time.monotonic()
        current_gn = bridge.v.solve_analysis(obs,contract,control=seed)
        gn_seconds = time.monotonic()-gn_started
        c = current_gn.control
        gn = {'reason':current_gn.reason,'control':c.detach().tolist(),
              'outer_iterations':current_gn.outer_iterations,'pcg_iterations':current_gn.pcg_iterations,
              'seed':seed.detach().tolist()}

    identity = {'gn_report_sha256':hashlib.sha256(gn_path.read_bytes()).hexdigest(),
                'reference_report_sha256':hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                'parameters_sha256':bridge._digest(p),'verification_sha256':bridge._digest(verification),
                'analysis_control_sha256':bridge._digest(c),'analysis_origin':'current solve_analysis result' if fresh_gn else 'archived product GN'}
    result = workflow.prepare_response(objective,score,c,p,directions,
        branch_check=check,input_identity=identity,refine=refine)
    refinement_method = 'matrix-free Newton-PCG' if matrix_free_refine else 'dense small oracle'
    report = {'scope':('current' if fresh_gn else 'saved')+' GN -> explicit '+refinement_method+' -> matrix-free local response',
              'refinement_method':refinement_method,
              'refinement_diagnostics':serializable(refinement_report),
              'fresh_gn_runs':int(fresh_gn),'nonlinear_reanalyses':0,
              'gn_seconds':gn_seconds,'gn_result':gn,
              'control_from_current_result':current_gn is not None,
              'general_minmod_response_eligible':False,'finite_path_certified':False,
              'input_identity':identity,'gn_reason':gn['reason'],
              'preparation_seconds':preparation_seconds,'refinement_candidates':candidates,
              'archived_gn_source_sha256':json.loads(gn_path.read_text())['source_sha256'],
              'source_sha256':source_hashes,
              'workflow':serializable(result)}
    if result['response'] is not None:
        response = result['response']
        previous = json.loads((EVIDENCE/'minmod_parameter_vjp.json').read_text())
        g_ref = torch.tensor(previous['parameter_gradients']['total_gradient'],dtype=c.dtype)
        report['full_gradient_reference_relative_difference'] = float((response.total_gradient-g_ref).norm()/g_ref.norm())
        report['control_reference_distance'] = float((result['control']-reference).norm())
        report['direction_reference_relative_differences'] = {
            name: abs(float(value)-previous['directions'][name]['total'])/abs(previous['directions'][name]['total'])
            for name,value in response.total.items()}
    for path in source_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError('source changed during integration; result not published')
    report['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fresh-gn',action='store_true')
    parser.add_argument('--matrix-free-refine',action='store_true')
    args = parser.parse_args()
    print(run(args.output,fresh_gn=args.fresh_gn,matrix_free_refine=args.matrix_free_refine)['workflow']['status'])
