"""Diagnose one limiter/flux intersection; never promote an optimizer point."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advar import transport, variational
from examples.weather_scenarios import fv_point_3h_active_limiter_probe as event
from examples.weather_scenarios import fv_point_3h_gn_policy_experiment as predecessor
from examples.weather_scenarios import fv_point_3h_face_transition_diagnostic as geometry
from examples.weather_scenarios import fv_point_3h_nonsmooth_coupled_probe as shared
from examples.weather_scenarios import fv_point_3h_seed_linear_probe as seed
from examples.weather_scenarios import fv_point_3h_tangent_continuation as tangent
from examples.weather_scenarios.fv_point_3h_gn_local_window_probe import _scaled_error
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = predecessor.EVIDENCE
PLAN = EVIDENCE / 'ACTIVE_INTERSECTION_PLAN_20261011.json'
ATTEMPT = EVIDENCE / 'active_intersection_20261011_attempt1'
BASE_SHA = '68a6e8219e79e8367b7fdd049bc25f4c431db9abdda1ba2af7ed0b52411f9ed8'
PREDECESSOR_SHA = '8ab812c1bbff2d6546178c74010738515cbac10cd12582fe76f4753b42fa44ed'
PARENT = predecessor.OUTPUT_DIR
ARCHIVE = EVIDENCE / 'GN_POLICY_ARCHIVE_20261011.json'
OWN_SOURCES = {'examples/weather_scenarios/fv_point_3h_active_intersection_diagnostic.py',
    'examples/weather_scenarios/fv_point_3h_active_limiter_probe.py',
    'tests/test_fv_point_3h_active_limiter_probe.py',
    'tests/test_fv_point_3h_active_intersection_diagnostic.py'}
POLICY: dict[str, Any] = {'guarded_launches': 1, 'internal_seconds': 600.0, 'outer_seconds': 660.0,
    'rss_bytes': 2 * 1024**3, 'event_grad_cap': 4, 'root_evaluation_cap': 33,
    'bracket_expansions': 8, 'bisections': 24, 'sector_radii': [1e-9, 5e-10, 2.5e-10],
    'tangent_cost_radii':[1e-7,5e-8], 'native_sector_gradient_cap':12,
    'radius': 0.05, 'optimizer_steps': 0, 'jacobian_row_vjp_calls': 0,
    'dense_solves': 0, 'hvp_calls': 0, 'root_claim': False,
    'minimum_claim': False, 'response_claim': False, 'score_claim': False}


def _sha(path: Path) -> str:
    return predecessor._sha(path)


def load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError('intersection plan identity mismatch')
    previous = predecessor._load_plan(predecessor.PLAN, PREDECESSOR_SHA)
    plan = json.loads(path.read_text())
    extra = {str(p.relative_to(ROOT)) for p in (predecessor.PLAN, ARCHIVE,
        PARENT / 'step.json.gz', PARENT / 'step.run.json', PARENT / 'step.resource.json')}
    if (plan.get('experiment_kind') != 'same_event_reachable_sector_diagnostic'
            or plan.get('base_control_sha256') != BASE_SHA or plan.get('policy') != POLICY
            or set(plan.get('source_files', {})) != set(previous['source_files']) | OWN_SOURCES
            or set(plan.get('archive_files', {})) != set(previous['archive_files']) | extra):
        raise ValueError('intersection plan scope mismatch')
    for group in ('source_files', 'archive_files'):
        for name, value in previous[group].items():
            if plan[group].get(name) != value:
                raise ValueError('inherited evidence pin changed')
        for name, value in plan[group].items():
            if _sha(ROOT / name) != value:
                raise ValueError(f'intersection pin changed: {name}')
    return plan


def load_base() -> dict[str, Any]:
    import gzip
    manifest = json.loads(ARCHIVE.read_text())['manifest']
    raw_bytes = gzip.decompress((PARENT / 'step.json.gz').read_bytes())
    raw = json.loads(raw_bytes)
    for filename in ('step.json', 'step.run.json', 'step.resource.json'):
        path = PARENT / filename
        data = raw_bytes if filename == 'step.json' else path.read_bytes()
        pin = manifest[str(path.relative_to(ROOT))]
        if hashlib.sha256(data).hexdigest() != pin['sha256'] or len(data) != pin['bytes']:
            raise ValueError('closed policy receipt changed')
    resource = json.loads((PARENT / 'step.resource.json').read_text())
    run = json.loads((PARENT / 'step.run.json').read_text())
    if (resource['exit_code'] != 0 or resource['resource_termination'] is not None
            or run['execution_status'] != 'completed' or raw['canonical_promotions'] != 1
            or run['child_sha256'] != hashlib.sha256(raw_bytes).hexdigest()
            or raw['current_control_sha256'] != BASE_SHA or raw['selected_policy'] != 'seeded_JR'
            or not tangent.fresh_final_closure(raw['canonical_proposal'], raw['canonical_final_repeat'])):
        raise ValueError('canonical parent closure invalid')
    winner = next(arm for arm in raw['arms'] if arm.get('canonical_promoted') is True)
    predecessor.read_bound_endpoint(winner, raw['base_R'])
    if tangent._tensor_sha(torch.tensor(raw['current_control'], dtype=torch.float64)) != BASE_SHA:
        raise ValueError('canonical parent control hash mismatch')
    return raw


def minimum_sampled_hull(gradients: Tensor) -> dict[str, Any]:
    """Closest point of a finite sampled hull, not a stationarity certificate."""
    if (gradients.ndim != 2 or not 1 <= gradients.shape[0] <= 4
            or gradients.dtype != torch.float64 or not bool(torch.isfinite(gradients).all())):
        raise ValueError('finite double gradients, at most four, are required')
    best: tuple[float, Tensor, Tensor] | None = None
    eps = 512 * torch.finfo(gradients.dtype).eps
    for size in range(1, gradients.shape[0] + 1):
        for indices in itertools.combinations(range(gradients.shape[0]), size):
            vertices = gradients[list(indices)]
            if size == 1:
                weights = torch.ones(1, dtype=gradients.dtype)
            else:
                differences = (vertices[1:] - vertices[0]).T
                coordinates = torch.linalg.lstsq(differences, -vertices[0],
                    rcond=128 * torch.finfo(gradients.dtype).eps, driver='gelsd').solution
                weights = torch.cat(((1 - coordinates.sum()).reshape(1), coordinates))
            if bool((weights < -eps).any()):
                continue
            weights = weights.clamp_min(0); weights = weights / weights.sum()
            vector = weights @ vertices
            merit = float(torch.dot(vector, vector))
            if best is None or merit < best[0]:
                full = torch.zeros(gradients.shape[0], dtype=gradients.dtype)
                full[list(indices)] = weights
                best = merit, vector, full
    if best is None:
        raise ValueError('sampled hull had no feasible vertex')
    return {'squared_norm': best[0], 'vector': best[1].tolist(), 'weights': best[2].tolist(),
        'scope': 'finite gradients at nearby points; not limiting-gradient certification'}


def sector_directions(normal: Tensor, event_gradient: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Resolve signed first-order Q/zeta sectors from independent normals."""
    scales = torch.stack((torch.linalg.vector_norm(normal), torch.linalg.vector_norm(event_gradient)))
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError('unresolved event normals')
    rows = torch.stack((normal, event_gradient)) / scales[:, None]
    singular = torch.linalg.svdvals(rows)
    if float(singular[-1]) <= 128 * torch.finfo(rows.dtype).eps * float(singular[0]):
        raise ValueError('external and limiter normals are dependent')
    signs = torch.tensor(list(itertools.product((-1., 1.), repeat=2)), dtype=rows.dtype)
    directions = torch.linalg.solve(rows @ rows.T, signs.T).T @ rows
    directions = directions / torch.linalg.vector_norm(directions, dim=1)[:, None]
    basis = torch.linalg.qr(rows.T, mode='reduced').Q
    return directions, {'normal_singular_values': singular.tolist(),
        'normal_cosine': float(torch.dot(rows[0], rows[1])), 'normal_basis': basis.tolist()}


def branch_changes(trace: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Separate the declared limiter change from other active-region changes."""
    changed = []
    if len(trace['choices']) != 360 or len(reference['choices']) != 360:
        raise ValueError('sector trace count differs from declared trajectory')
    for stage, (actual, base) in enumerate(zip(trace['choices'],reference['choices'])):
        for axis, (a,b) in enumerate(zip(actual,base)):
            for field in ('choose_left','left_sign','right_sign','slope_sign'):
                aa=torch.tensor(a[field]); bb=torch.tensor(b[field])
                if aa.shape!=bb.shape:
                    raise ValueError('sector choice shape changed')
                for row,column in torch.nonzero(aa!=bb).tolist():
                    changed.append({'stage':stage,'axis':axis,'field':field,'row':row,'column':column})
    allowed={'stage':179,'axis':0,'field':'choose_left','row':1,'column':2}
    unexpected=[change for change in changed if change!=allowed]
    faces_match=trace['face_signs']==reference['face_signs']
    return {'changed_choices':changed,'unexpected_choice_changes':unexpected,
        'other_face_signs_match':faces_match,
        'only_declared_event_changed':not unexpected and faces_match}


def original_event_values(problem: Any, control: Tensor, parameters: Tensor, side: int) -> Tensor:
    """Read the original replay contract through its detached public observer."""
    captured=[]; count=0
    def observe(q: Tensor, _qx: Tensor, _qy: Tensor) -> None:
        nonlocal count
        if count==179:
            captured.append(torch.stack((q[2,3]-q[2,2],q[2,4]-q[2,3],q.abs().amax())))
        count+=1
    with torch.no_grad(),transport.selected_face_extension('y',4,3,side),transport.observe_minmod_stages(observe):
        problem.objective(control,parameters)
    if count!=360 or len(captured)!=1:
        raise ValueError('original event observer stage count changed')
    return captured[0]


def _run_child(path: Path, digest: str, output: Path) -> None:
    started = time.monotonic(); deadline = started + POLICY['internal_seconds']
    plan = load_plan(path, digest); base = load_base()
    c = torch.tensor(base['current_control'], dtype=torch.float64)
    record: dict[str, Any] = {'phase': 'running', 'policy': POLICY, 'plan_sha256': digest,
        'base_control_sha256': BASE_SHA, 'current_control': c.tolist(),
        'optimizer_steps_applied': 0, 'candidate_committed': False,
        'event_grad_started': 0, 'event_grad_completed': 0,
        'root_evaluations': [], 'sector_samples': [], 'native_gradient_started':0, 'native_gradient_completed':0, 'root_claim': False,
        'minimum_claim': False, 'response_claim': False, 'score_claim': False}
    write = lambda: tangent._write(output, record)
    source_before = shared._source_hashes(plan, path.resolve(), digest)
    problem, original, _, parameters, truth, _ = seed._prepare_fixed_seed()
    input_before = shared._input_identity(problem, original, c, parameters, truth)
    runtime_before = shared._runtime()
    if input_before != base['input_after'] or runtime_before != base['runtime_after']:
        raise ValueError('canonical seed input/runtime mismatch')
    weights = geometry._face_weights(problem, axis='y', row=4, column=3)
    pivot = geometry._pivot(c, weights)
    observed = tangent._observe(shared, problem, c, parameters, weights)
    repeat = base['canonical_final_repeat']
    theta = float(tangent.minimum_mixture_weight(observed['side'][-1][1], observed['side'][1][1]))
    G = torch.cat(((1-theta)*observed['side'][-1][1] + theta*observed['side'][1][1],
        (observed['q']/0.84).reshape(1)))
    from examples.weather_scenarios.fv_point_3h_gn_finite_continuation import _full_G_match
    if (not _full_G_match(G, torch.tensor(base['current_full_G'], dtype=c.dtype))['passed']
            or not shared._gradient_pair_match({s: observed['side'][s][1] for s in (-1,1)},
                {s: torch.tensor(repeat['side_gradients'][str(s)], dtype=c.dtype) for s in (-1,1)})
            or not observed['pair']['passed'] or not observed['face_ok']
            or not observed['side_objectives_match_native']
            or any(observed['traces'][s]['signature_sha256'] != repeat['branch_trace'][str(s)]['signature_sha256'] for s in (-1,1))
            or theta != repeat['theta'] or float(observed['native_j']) != repeat['objective']):
        raise ValueError('canonical J/G/theta/gradient/trace requalification failed')
    record.update(base_J=float(observed['native_j']), base_R=float(torch.dot(G,G)),
        base_full_G=G.tolist(), base_theta=theta, source_before=source_before,
        input_before=input_before, runtime_before=runtime_before, base_requalified=True)
    contract = problem.contract(parameters); fv = contract.fv_transport
    if fv is None:
        raise ValueError('FV contract missing')
    direct = replace(contract, fv_transport=replace(fv, replay=False))
    trajectory = lambda point: variational.analysis_trajectory(point, direct)

    def check() -> None:
        shared._deadline(deadline, POLICY['internal_seconds'])
        if shared._source_hashes(plan, path.resolve(), digest) != source_before or shared._runtime() != runtime_before:
            raise ValueError('source/runtime changed during intersection diagnosis')

    def values(point: Tensor, side: int) -> Tensor:
        check()
        result = event.event_values(trajectory, point, side)
        active_bound=128*torch.finfo(point.dtype).eps*result[2]
        if not bool(torch.isfinite(result).all() and (result[:2] < -active_bound).all()):
            raise ValueError('target event left/right negative regime changed')
        return result

    def derivatives(point: Tensor) -> tuple[Tensor, Tensor]:
        grads = []; vals = []
        for side in (-1, 1):
            if record['event_grad_started'] >= POLICY['event_grad_cap']:
                raise ValueError('event derivative cap exhausted')
            record['event_grad_started'] += 1; write(); check()
            function = lambda point, s=side: event.raw_zeta(event.event_values(trajectory, point, s))
            grad = torch.func.grad(function)(point)
            value = values(point, side)
            if not bool(torch.isfinite(grad).all()):
                raise ValueError('event gradient nonfinite')
            record['event_grad_completed'] += 1
            grads.append(grad); vals.append(value)
        if not torch.equal(vals[0], vals[1]):
            raise ValueError('external extensions give different Q=0 event values')
        return torch.stack(grads), torch.stack(vals)

    write(); grads, vals = derivatives(c)
    for index,side in enumerate((-1,1)):
        if not torch.equal(vals[index],original_event_values(problem,c,parameters,side)):
            raise ValueError('direct-tape/original event value parity failed')
    record['original_event_value_parity_exact']=True
    normal = geometry._face_normal(c, weights); unit = normal / torch.linalg.vector_norm(normal)
    projected = grads - (grads @ unit)[:,None] * unit
    projected_match=_scaled_error(projected[0],projected[1],grads.abs().amax(dim=0))
    record['base_projected_event_gradient_match']=projected_match
    if not projected_match['passed']:
        raise ValueError('event projected gradients differ across external extensions')
    w = projected.mean(dim=0); magnitude = torch.linalg.vector_norm(w)
    if float(magnitude) <= 128*torch.finfo(c.dtype).eps:
        raise ValueError('limiter normal unresolved in external-face tangent')
    direction = w / magnitude
    z0 = float(event.raw_zeta(vals[0])); estimate = -z0 / float(magnitude)
    record.update(base_event_values=vals.tolist(), base_event_gradients=grads.tolist(),
        event_tangent_gradient=w.tolist(), first_order_event_distance=estimate)

    def root_value(alpha: float) -> tuple[float, Tensor]:
        if len(record['root_evaluations']) >= POLICY['root_evaluation_cap']:
            raise ValueError('event root evaluation budget exhausted')
        point = geometry._chart(c + alpha*direction, weights, pivot, 0.0)
        movement = float(torch.linalg.vector_norm(point-c))
        if not 0 <= movement <= POLICY['radius']:
            raise ValueError('event root leaves declared chart radius')
        value = values(point, -1); zeta = float(event.raw_zeta(value))
        record['root_evaluations'].append({'alpha': alpha, 'zeta': zeta,
            'actual_movement': movement, 'control_sha256': tangent._tensor_sha(point)})
        write(); return zeta, point

    q_scale = float(vals[0,2]); event_bound = 256*torch.finfo(c.dtype).eps*q_scale
    # Both bracket endpoints evaluate exactly the same restored Q=0 chart.
    lower = 0.; z_lower, point_lower = root_value(lower)
    upper = estimate * 2
    if not torch.isfinite(torch.tensor(upper)):
        raise ValueError('invalid event distance estimate')
    if abs(z_lower) <= event_bound:
        upper=lower; z_upper=z_lower; point_upper=point_lower
    else:
        if upper == 0:
            raise ValueError('unresolved chart-zero event distance')
        z_upper, point_upper = root_value(upper)
        for _ in range(POLICY['bracket_expansions']-1):
            if z_lower*z_upper <= 0:
                break
            upper *= 2; z_upper, point_upper = root_value(upper)
        if z_lower*z_upper > 0:
            raise ValueError('same-event sign bracket not found')
    best_z, intersection = (z_lower, point_lower) if abs(z_lower) < abs(z_upper) else (z_upper, point_upper)
    for _ in range(POLICY['bisections']):
        if abs(best_z) <= event_bound:
            break
        middle = (lower+upper)/2; z_middle, point_middle = root_value(middle)
        if abs(z_middle) < abs(best_z):
            best_z, intersection = z_middle, point_middle
        if z_lower*z_middle <= 0:
            upper, z_upper = middle, z_middle
        else:
            lower, z_lower = middle, z_middle
    record.update(intersection_control=intersection.tolist(), intersection_control_sha256=tangent._tensor_sha(intersection),
        intersection_zeta=best_z, event_roundoff_bound=event_bound,
        event_root_closed=abs(best_z) <= event_bound, intersection_Q=float(shared._face_value(intersection,weights)),
        root_bracket=[lower,upper])
    write()
    if not record['event_root_closed']:
        raise ValueError('event root not resolved to declared state-scale bound')
    root_grads, root_vals = derivatives(intersection)
    root_normal = geometry._face_normal(intersection, weights)
    root_unit=root_normal/torch.linalg.vector_norm(root_normal)
    root_projected=root_grads-(root_grads @ root_unit)[:,None]*root_unit
    root_match=_scaled_error(root_projected[0],root_projected[1],root_grads.abs().amax(dim=0))
    root_bound=256*torch.finfo(c.dtype).eps*float(root_vals[:,2].min())
    record.update(root_projected_event_gradient_match=root_match,root_event_roundoff_bound=root_bound)
    if not root_match['passed'] or float((root_vals[:,0]-root_vals[:,1]).abs().max())>root_bound:
        raise ValueError('root-local event/gradient closure failed')
    root_J=float(problem.objective(intersection,parameters))
    record['intersection_native_J']=root_J
    production_Q,face_bound,face_ok=shared._face_audit(problem,intersection,weights,shared.FACE,0.0)
    record.update(intersection_production_Q=float(production_Q),intersection_face_bound=face_bound,intersection_face_passed=face_ok)
    if not face_ok:
        raise ValueError('intersection external face did not close')
    directions, audit = sector_directions(root_normal, root_grads.mean(dim=0))
    basis = torch.tensor(audit['normal_basis'], dtype=c.dtype)
    record.update(root_event_gradients=root_grads.tolist(), root_event_values=root_vals.tolist(), normal_audit=audit)
    sectors = list(itertools.product((-1,1),repeat=2))
    for radius in POLICY['sector_radii']:
        gradient_set = []
        for index, (qsign,zsign) in enumerate(sectors):
            check(); sample = intersection + radius*directions[index]
            if not bool(torch.isfinite(sample).all()):
                raise ValueError('sector control nonfinite')
            Q = float(shared._face_value(sample,weights)); ev = values(sample,qsign); zeta = float(event.raw_zeta(ev))
            production_Q,face_bound,face_ok=shared._face_audit(problem,sample,weights,shared.FACE,Q)
            # Only the matching external extension is used; native upwind is unchanged.
            J = problem.objective(sample,parameters)
            if record['native_gradient_started']>=POLICY['native_sector_gradient_cap']:
                raise ValueError('native sector gradient cap exhausted')
            record['native_gradient_started']+=1; write()
            gradient = torch.func.grad(problem.objective,argnums=0)(sample,parameters)
            if not bool(torch.isfinite(J) and torch.isfinite(gradient).all()):
                raise ValueError('native sector objective/gradient nonfinite')
            record['native_gradient_completed']+=1
            trace = shared._analysis_trace(problem,sample,parameters,qsign)
            choice = trace['choices'][179][0]
            reachable = (face_ok and qsign*float(production_Q)>face_bound and qsign*Q>0 and zsign*zeta>0 and choice['choose_left'][1][2] == (zsign>0)
                and choice['left_sign'][1][2] == -1 and choice['right_sign'][1][2] == -1)
            entry = {'radius': radius, 'Q_sign': qsign, 'zeta_sign': zsign, 'Q': Q, 'production_Q':float(production_Q),'face_parity_bound':face_bound,'zeta': zeta,
                'reachable_sector': reachable, 'native_J': float(J), 'native_J_minus_intersection':float(J)-root_J,
                'native_event_directional_quotient':(float(J)-root_J)/radius, 'native_gradient': gradient.tolist(),
                'branch_changes':branch_changes(trace,observed['traces'][qsign]),
                'control':sample.tolist(), 'control_sha256':tangent._tensor_sha(sample),
                'trace':trace, 'accepted':False}
            gradient_set.append(gradient); record['sector_samples'].append(entry); write()
        rows = torch.stack(gradient_set)
        scope_passed=all(entry['reachable_sector'] and entry['branch_changes']['only_declared_event_changed']
            and entry['trace']['nonfinite_or_tie'] is False
            for entry in record['sector_samples'][-4:])
        projected_rows = rows - (rows @ basis) @ basis.T
        common = projected_rows.mean(dim=0)
        direction_common = -common / torch.linalg.vector_norm(common) if float(torch.linalg.vector_norm(common))>0 else torch.zeros_like(common)
        record.setdefault('sector_summaries',[]).append({'radius':radius,
            'isolated_four_sector_scope_passed':scope_passed,
            'sampled_hull':minimum_sampled_hull(rows) if scope_passed else None,
            'common_tangent_gradient_norm':float(torch.linalg.vector_norm(common)),
            'common_tangent_projection_spread':float(torch.linalg.vector_norm(projected_rows-common,dim=1).max()),
            'common_tangent_descent_direction':direction_common.tolist(),
            'sampled_cost_slopes':(rows @ direction_common).tolist()})
        write()
    if record['sector_summaries'][-1]['isolated_four_sector_scope_passed']:
        record['tangent_cost_samples']=[]
        for radius in POLICY['tangent_cost_radii']:
            check(); tangent_direction=torch.tensor(record['sector_summaries'][-1]['common_tangent_descent_direction'],dtype=c.dtype)
            point=geometry._chart(intersection+radius*tangent_direction,weights,pivot,0.0)
            tangent_j=float(problem.objective(point,parameters)); value=values(point,-1)
            actual_Q,bound,face_ok=shared._face_audit(problem,point,weights,shared.FACE,0.0)
            record['tangent_cost_samples'].append({'radius':radius,'native_J':tangent_j,
                'native_J_minus_intersection':tangent_j-root_J,'native_directional_quotient':(tangent_j-root_J)/radius,
                'production_Q':float(actual_Q),'face_roundoff_bound':bound,'face_passed':face_ok,
                'zeta':float(event.raw_zeta(value)),'actual_movement':float(torch.linalg.vector_norm(point-intersection)),
                'control_sha256':tangent._tensor_sha(point),'accepted':False})
            write()
    check()
    input_after = shared._input_identity(problem,original,c,parameters,truth)
    if input_after != input_before:
        raise ValueError('fixed input changed')
    record.update(phase='finished',execution_status='completed',numerical_status='active_intersection_diagnostic_no_commit',
        input_after=input_after, source_after=shared._source_hashes(plan,path.resolve(),digest),
        runtime_after=shared._runtime(), fixed_input_unchanged=True, source_unchanged=True, runtime_unchanged=True,
        elapsed_seconds=time.monotonic()-started, deadline_passed=time.monotonic()<deadline)
    write()


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();output=ATTEMPT/'diagnostic.json'
    if args.child:
        _run_child(PLAN,args.plan_sha256,output);return
    load_plan(PLAN,args.plan_sha256)
    if ATTEMPT.exists():
        raise ValueError('intersection attempt must be fresh')
    ATTEMPT.mkdir()
    resource=run_guarded_diagnostic([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child','--plan-sha256',args.plan_sha256],
        wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes'],report_path=ATTEMPT/'diagnostic.resource.json',log_path=ATTEMPT/'diagnostic.log')
    result={'execution_status':shared._execution_status(resource,wall_seconds=POLICY['outer_seconds'],rss_bytes=POLICY['rss_bytes']),
        'resource':resource,'child_sha256':_sha(output) if output.exists() else None}
    tangent._write(ATTEMPT/'diagnostic.run.json',result);print(json.dumps(result))


if __name__ == '__main__':
    main()
