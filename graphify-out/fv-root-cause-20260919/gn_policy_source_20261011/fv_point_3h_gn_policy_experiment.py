"""One plan-driven comparison of search cost and bounded residual recovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.weather_scenarios import fv_point_3h_gn_finite_continuation as prior
from examples.weather_scenarios import fv_point_3h_policy_search as search
from examples.weather_scenarios.fv_point_3h_policy_backend import FVBackend
from examples.weather_scenarios.fv_diagnostic_guard import run_guarded_diagnostic

EVIDENCE = ROOT / 'graphify-out/fv-root-cause-20260919'
PLAN = EVIDENCE / 'GN_POLICY_EXPERIMENT_PLAN_20261011.json'
OUTPUT_DIR = EVIDENCE / 'gn_policy_experiment_20261011_attempt1'
PRIOR_PLAN_SHA = '9303b08062e68636a5da5dece5693db6d059615f2e4bcba585f466847b93c1ce'
OWN_SOURCES = {
    'examples/weather_scenarios/fv_point_3h_gn_policy_experiment.py',
    'examples/weather_scenarios/fv_point_3h_policy_backend.py',
    'examples/weather_scenarios/fv_point_3h_policy_search.py',
    'tests/test_fv_point_3h_gn_policy_experiment.py',
    'tests/test_fv_point_3h_policy_search.py',
    'tests/test_fv_independent_completion_oracles.py',
}
POLICY: dict[str, Any] = {
    'guarded_launches': 1, 'internal_seconds': 600.0, 'outer_seconds': 660.0,
    'rss_bytes': 2 * 1024**3, 'jacobian_row_vjp_calls': 72,
    'dense_solves': 3, 'hvp_calls': 6, 'candidate_count': 72,
    'face_scale': 0.84, 'radius': 0.05, 'nonmonotone_horizon': 3,
    'gamma': 4.0, 'expansion_factor': 2.0, 'max_expansions': 3,
    'max_backtracks': 24, 'max_evaluations_per_arm': 24,
    'Rcap_fraction': 0.01, 'recovery_relative_margin': 1e-8,
    'arms': ['dyadic_JR', 'seeded_JR', 'bounded_J_Rcap'],
    'root_claim': False, 'minimum_claim': False, 'response_claim': False,
    'score_claim': False,
}


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('pinned or recorded path escapes the repository')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path, digest: str) -> dict[str, Any]:
    if path.resolve() != PLAN.resolve() or _sha(path) != digest:
        raise ValueError('policy experiment plan identity mismatch')
    parent = prior._load_plan(prior.PLAN, PRIOR_PLAN_SHA)
    plan = json.loads(path.read_text())
    if (plan.get('experiment_kind') != 'bounded_GN_search_cost_and_R_policy_comparison'
            or plan.get('policy') != POLICY or plan.get('base_control_sha256') != prior.BASE_SHA
            or plan.get('predecessor_plan_sha256') != PRIOR_PLAN_SHA
            or set(plan.get('source_files', {})) != set(parent['source_files']) | OWN_SOURCES
            or set(plan.get('archive_files', {})) != set(parent['archive_files']) | {
                prior.PLAN.relative_to(ROOT).as_posix()}):
        raise ValueError('policy experiment scope changed')
    for group in ('source_files', 'archive_files'):
        if any(plan[group].get(name) != value for name, value in parent[group].items()):
            raise ValueError('inherited evidence pins changed')
        for name, value in plan[group].items():
            if _sha(ROOT / name) != value:
                raise ValueError(f'policy experiment pin mismatch: {name}')
    movement = plan.get('previous_actual_movement')
    if not isinstance(movement, (int, float)) or not math.isfinite(movement) or movement <= 0:
        raise ValueError('previous accepted actual movement is invalid')
    # Its value must come from the closed parent, not a post-run preferred seed.
    raw = json.loads(prior.gzip.decompress(prior.BASE_GZIP.read_bytes()))
    last = raw['closed_steps'][-1]
    chosen = next(t for t in last['candidate_trials'] if t.get('accepted') is True)
    if movement != chosen['actual_path_norm']:
        raise ValueError('previous movement differs from archived accepted trial')
    return plan


def endpoint_recovered(initial_J: float, initial_R: float, trial: dict[str, Any],
        repeat: dict[str, Any], *, relative_margin: float) -> bool:
    """Recovery is a finite endpoint test, never a root/minimum certificate."""
    return (prior.tangent.fresh_final_closure(trial, repeat)
        and math.isfinite(float(repeat['objective']))
        and math.isfinite(float(repeat['F_squared']))
        and float(repeat['objective']) < initial_J
        and float(repeat['F_squared']) < initial_R * (1.0 - relative_margin))


def promotable_arm(arm: dict[str, Any]) -> bool:
    return arm.get('status') == 'endpoint_recovered' and arm.get('endpoint_P2_passed') is True


def choose_endpoint(arms: list[dict[str, Any]]) -> dict[str, Any] | None:
    choices = [arm for arm in arms if promotable_arm(arm)]
    return min(choices, key=lambda arm: (arm['endpoint_R'], arm['endpoint_J'], arm['arm_order'])) if choices else None


def read_bound_endpoint(winner: dict[str, Any], initial_R: float) -> dict[str, Any]:
    """Bind every provisional P2 receipt to its recorded path before promotion."""
    closed: dict[str, Any] = {}
    for point in winner['points']:
        path = ROOT / point['closed_receipt']
        if _sha(path) != point['closed_sha256']:
            raise ValueError('provisional P2 path receipt changed')
        closed = json.loads(path.read_text())
        proposal, repeat = closed['proposal'], closed['final_repeat']
        if (closed['policy'] != winner['policy'] or proposal['acceptance_policy'] != winner['policy']
                or not prior.tangent.fresh_final_closure(proposal, repeat)
                or proposal['objective'] != point['endpoint_J'] or proposal['F_squared'] != point['endpoint_R']
                or proposal['J_armijo_passed'] is not True
                or repeat['objective'] > proposal['J_armijo_bound']):
            raise ValueError('recorded path does not satisfy its declared J/P2 policy')
        if winner['policy'] == 'bounded_J_Rcap':
            cap = initial_R * (1.0 + POLICY['Rcap_fraction'])
            if proposal['R_reference_initial'] != initial_R or proposal['R_cap'] != cap or repeat['F_squared'] > cap:
                raise ValueError('recorded path violates the fixed initial-R cap')
        elif proposal['F_squared_armijo_passed'] is not True or repeat['F_squared'] > proposal['F_squared_armijo_bound']:
            raise ValueError('recorded monotone path violates original R Armijo')
    if (not closed or winner['endpoint_receipt'] != winner['points'][-1]['closed_receipt']
            or winner['endpoint_receipt_sha256'] != winner['points'][-1]['closed_sha256']
            or closed['proposal']['control_sha256'] != winner['endpoint_control_sha256']):
        raise ValueError('winner does not bind its recorded endpoint')
    return closed


def persist_canonical(output: Path, record: dict[str, Any], winner: dict[str, Any],
        proposal: dict[str, Any], repeat: dict[str, Any], full_G: list[float]) -> dict[str, Any]:
    if not promotable_arm(winner) or not endpoint_recovered(record['base_J'], record['base_R'],
            proposal, repeat, relative_margin=POLICY['recovery_relative_margin']):
        raise ValueError('nonrecovered or unqualified experimental endpoint cannot be promoted')
    promoted = copy.deepcopy(record)
    control = torch.tensor(repeat['control'], dtype=torch.float64)
    G = torch.tensor(full_G, dtype=torch.float64)
    if control.shape != (26,) or G.shape != (27,) or not bool(torch.isfinite(control).all() and torch.isfinite(G).all()):
        raise ValueError('canonical control/full-G shape or finiteness failed')
    merit = float(torch.dot(G, G))
    budget = 128 * torch.finfo(G.dtype).eps * max(abs(merit), abs(repeat['F_squared']), torch.finfo(G.dtype).tiny)
    if abs(merit - repeat['F_squared']) > budget:
        raise ValueError('canonical full-G does not reproduce repeated merit')
    sha = prior.tangent._tensor_sha(control)
    promoted.update(current_control=repeat['control'], current_control_sha256=sha,
        current_theta=repeat['theta'], current_full_G=full_G,
        last_confirmed_control=repeat['control'], last_confirmed_control_sha256=sha,
        last_confirmed_theta=repeat['theta'], last_confirmed_full_G=full_G,
        candidate_committed=True, optimizer_steps_applied=len(winner['points']),
        canonical_promotions=1, selected_policy=winner['policy'],
        canonical_proposal=proposal, canonical_final_repeat=repeat)
    promoted['arms'][winner['arm_order']]['canonical_promoted'] = True
    prior.tangent._write(output, promoted)
    return promoted


def _run_child(plan_path: Path, plan_sha: str, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = time.monotonic() + POLICY['internal_seconds']
    plan = _load_plan(plan_path, plan_sha)
    folder = output.parent
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {'phase': 'running', 'execution_status': 'running',
        'numerical_status': 'comparison_initialization', 'plan_sha256': plan_sha,
        'policy': POLICY, 'base_control_sha256': prior.BASE_SHA,
        'candidate_committed': False, 'optimizer_steps_applied': 0, 'arms': [],
        'root_claim': False, 'minimum_claim': False, 'response_claim': False, 'score_claim': False}
    io_seconds = 0.0
    receipt_hash_seconds = 0.0

    def write(path: Path, payload: dict[str, Any]) -> None:
        nonlocal io_seconds
        then = time.perf_counter()
        prior.tangent._write(path, payload)
        io_seconds += time.perf_counter() - then

    def receipt_sha(path: Path) -> str:
        nonlocal receipt_hash_seconds
        then = time.perf_counter()
        digest = _sha(path)
        receipt_hash_seconds += time.perf_counter() - then
        return digest

    write(output, record)
    backend = FVBackend(plan, plan_path, plan_sha, folder / 'operators', deadline)
    parent = backend.parent
    initial_control = torch.tensor(parent['current_control'], dtype=torch.float64)
    initial_G = torch.tensor(parent['current_full_G'], dtype=torch.float64)
    initial_J, initial_R = parent['final_repeat']['objective'], parent['final_repeat']['F_squared']
    record.update(current_control=initial_control.tolist(), current_control_sha256=prior.BASE_SHA,
        current_theta=parent['current_theta'], current_full_G=initial_G.tolist(),
        last_confirmed_control=initial_control.tolist(), last_confirmed_control_sha256=prior.BASE_SHA,
        last_confirmed_theta=parent['current_theta'], last_confirmed_full_G=initial_G.tolist(),
        base_J=initial_J, base_R=initial_R)
    write(output, record)
    cache: dict[tuple[str, str, float], dict[str, Any]] = {}
    actual_evaluations = 0
    actual_evaluations_started = 0
    completed_arms: list[dict[str, Any]] = []
    endpoint_bundles: dict[str, dict[str, Any]] = {}
    initial_bundle: dict[str, Any] | None = None

    def finish(status: str, error: str = '', *, failed: bool = False) -> dict[str, Any]:
        source_after = prior.shared._source_hashes(plan, plan_path, plan_sha)
        canonical = torch.tensor(record['current_control'], dtype=torch.float64)
        input_after = prior.shared._input_identity(backend.problem, backend.original,
            canonical, backend.parameters, backend.truth)
        runtime_after = prior.shared._runtime()
        source_ok = source_after == backend.source_anchor
        input_ok = prior.geometry.model._fixed_input(input_after, backend.input_anchor,
            prior.tangent._tensor_sha(canonical))
        runtime_ok = runtime_after == backend.runtime_anchor
        operator_receipts = [json.loads(p.read_text()) for p in sorted((folder / 'operators').glob('*_operator.json'))]
        counted = {key: sum(r.get(key, 0) for r in operator_receipts) for key in (
            'jacobian_rows_started', 'jacobian_rows_completed', 'dense_solves_started',
            'dense_solves_completed', 'hvp_calls_started', 'hvp_calls_completed')}
        record.update(phase='finished', execution_status='failed' if failed else 'completed', numerical_status=status,
            termination_reason=error, actual_candidate_evaluations=actual_evaluations,
            actual_candidate_evaluations_started=actual_evaluations_started,
            backend_counts={'builds': backend.build_count, 'rows': backend.row_count,
                'solves': backend.solve_count, 'HVP': backend.hvp_count},
            backend_timing=backend.timings, runner_write_seconds=io_seconds,
            receipt_hash_seconds=receipt_hash_seconds,
            elapsed_seconds=time.perf_counter() - started,
            terminal_deadline_passed=time.monotonic() < deadline,
            counted_operator_receipts=counted, source_before=backend.source_anchor,
            source_after=source_after, source_unchanged=source_ok, input_after=input_after,
            fixed_input_unchanged=input_ok, runtime_after=runtime_after,
            runtime_unchanged=runtime_ok, postcommit_readiness_performed=False)
        write(output, record)
        if not all((source_ok, input_ok, runtime_ok)):
            raise ValueError('terminal comparison input/source/runtime changed')
        return record

    try:
        initial_bundle = backend.build(initial_control, parent['final_repeat'], initial_G, 'shared_initial')
        for arm_order, name in enumerate(POLICY['arms']):
            arm_started = time.perf_counter()
            arm: dict[str, Any] = {'policy': name, 'arm_order': arm_order,
                'status': 'provisional', 'points': [], 'reference_R': initial_R,
                'R_cap': initial_R * (1.0 + POLICY['Rcap_fraction']),
                'initial_J': initial_J, 'initial_R': initial_R,
                'logical_candidate_evaluations': 0, 'fresh_candidate_evaluations': 0,
                'cache_hits': 0, 'slots': 0, 'canonical_promoted': False}
            record['arms'].append(arm)
            write(output, record)
            current = initial_control.clone()
            closure = parent['final_repeat']
            full_G = initial_G.clone()
            movement = float(plan['previous_actual_movement'])
            horizon = POLICY['nonmonotone_horizon'] if name == 'bounded_J_Rcap' else 1
            bundle = initial_bundle
            for step in range(horizon):
                remaining = POLICY['max_evaluations_per_arm'] - arm['slots']
                if remaining <= 0:
                    arm['status'] = 'arm_candidate_budget_exhausted'
                    break
                if step:
                    bundle = backend.build(current, closure, full_G, f'{name}_{step}')
                point_folder = folder / name / f'point_{step}'
                point_folder.mkdir(parents=True, exist_ok=True)
                progress_path = point_folder / 'progress.json'
                point: dict[str, Any] = {'step': step, 'base_control_sha256': prior.tangent._tensor_sha(current),
                    'operator_record': bundle['operator_record'], 'base_J': bundle['J'],
                    'base_R': bundle['R'], 'direction_sha256': prior.tangent._tensor_sha(bundle['direction']),
                    'status': 'searching'}
                arm['points'].append(point)
                write(output, record)
                slot_before = arm['slots']

                def progress(trials: list[dict[str, Any]], meta: dict[str, Any]) -> None:
                    arm['slots'] = slot_before + meta['slots_started']
                    point['search'] = dict(meta)
                    write(progress_path, {'trials': trials, 'search': meta,
                        'canonical_control_sha256': record['current_control_sha256']})

                def evaluate(candidate: torch.Tensor, _theta: float, alpha: float) -> dict[str, Any]:
                    nonlocal actual_evaluations, actual_evaluations_started
                    key = (point['base_control_sha256'], point['direction_sha256'], alpha)
                    arm['logical_candidate_evaluations'] += 1
                    if key in cache:
                        backend._check(candidate)
                        full_facts = copy.deepcopy(cache[key])
                        arm['cache_hits'] += 1
                        cached = True
                    else:
                        actual_evaluations_started += 1
                        full_facts = backend.evaluate(bundle, candidate, alpha)
                        cache[key] = copy.deepcopy(full_facts)
                        actual_evaluations += 1
                        arm['fresh_candidate_evaluations'] += 1
                        cached = False
                    index = arm['logical_candidate_evaluations']
                    path = point_folder / f'candidate_{index:02d}.json'
                    write(path, {'control': candidate.tolist(), 'alpha': alpha,
                        'control_sha256': prior.tangent._tensor_sha(candidate),
                        'base_control_sha256': point['base_control_sha256'],
                        'direction_sha256': point['direction_sha256'],
                        'cache_hit_same_point_direction_alpha': cached, 'facts': full_facts})
                    facts = {k: v for k, v in full_facts.items() if k != 'branch_trace'}
                    facts.update(full_receipt_path=path.relative_to(ROOT).as_posix(),
                        full_receipt_sha256=receipt_sha(path), cache_hit=cached)
                    return facts

                config = {**search.DEFAULT_POLICY, 'name': name, 'gamma': POLICY['gamma'],
                    'expansion_factor': POLICY['expansion_factor'], 'max_expansions': POLICY['max_expansions'],
                    'max_evaluations': remaining, 'max_backtracks': min(remaining, POLICY['max_backtracks']),
                    'Rcap_fraction': POLICY['Rcap_fraction'], 'radius': POLICY['radius']}
                chosen, trials, meta = search.search(bundle['model'], current, evaluate,
                    chart_candidate=lambda c, d, a: backend.chart(bundle, c, d, a),
                    progress=progress, policy=config, previous_movement=movement,
                    reference_R=initial_R)
                point['search'] = meta
                if chosen is None:
                    point['status'] = 'no_eligible_candidate'
                    arm['status'] = 'no_eligible_candidate'
                    break
                receipt = ROOT / chosen['full_receipt_path']
                if receipt_sha(receipt) != chosen['full_receipt_sha256']:
                    raise ValueError('selected full candidate receipt changed')
                saved = json.loads(receipt.read_text())
                selected = {**chosen, **saved['facts']}
                if saved['control_sha256'] != chosen['control_sha256'] or saved['control'] != chosen['control']:
                    raise ValueError('candidate control does not bind its receipt')
                proposal, repeat = backend.repeat(bundle, selected)
                proposal['acceptance_policy'] = name
                proposal['acceptance_policy_version'] = search.POLICY_VERSION
                proposal['R_reference_initial'] = initial_R
                proposal['R_cap'] = arm['R_cap']
                proposal['R_armijo_required'] = name != 'bounded_J_Rcap'
                if not prior.tangent.fresh_final_closure(proposal, repeat):
                    raise ValueError('experimental candidate independent P2 failed')
                if (repeat['objective'] > proposal['J_armijo_bound']
                        or name != 'bounded_J_Rcap' and repeat['F_squared'] > proposal['F_squared_armijo_bound']
                        or name == 'bounded_J_Rcap' and repeat['F_squared'] > arm['R_cap']):
                    raise ValueError('repeated candidate violates its declared policy')
                closed_path = point_folder / 'closed_provisional.json'
                closed = {'policy': name, 'proposal': proposal, 'final_repeat': repeat,
                    'G': selected['G'], 'actual_path_norm': chosen['actual_path_norm'],
                    'canonical_promoted': False}
                write(closed_path, closed)
                point.update(status='P2_closed_provisional', closed_receipt=closed_path.relative_to(ROOT).as_posix(),
                    closed_sha256=receipt_sha(closed_path), alpha=chosen['alpha'], actual_path_norm=chosen['actual_path_norm'],
                    endpoint_J=repeat['objective'], endpoint_R=repeat['F_squared'],
                    F_squared_armijo_passed=proposal['F_squared_armijo_passed'], P2_passed=True)
                current = torch.tensor(repeat['control'], dtype=torch.float64)
                closure = repeat
                full_G = torch.tensor(selected['G'], dtype=torch.float64)
                movement = chosen['actual_path_norm']
                arm.update(endpoint_J=repeat['objective'], endpoint_R=repeat['F_squared'],
                    endpoint_control_sha256=chosen['control_sha256'], endpoint_P2_passed=True,
                    endpoint_receipt=closed_path.relative_to(ROOT).as_posix(),
                    endpoint_receipt_sha256=point['closed_sha256'])
                endpoint_bundles[name] = bundle
                if endpoint_recovered(initial_J, initial_R, proposal, repeat,
                        relative_margin=POLICY['recovery_relative_margin']):
                    arm['status'] = 'endpoint_recovered'
                    break
            if arm['status'] == 'provisional':
                arm['status'] = 'recovery_horizon_exhausted_noncanonical'
            arm['elapsed_seconds'] = time.perf_counter() - arm_started
            completed_arms.append(arm)
            write(output, record)
        winner = choose_endpoint(completed_arms)
        if winner is None:
            return finish('comparison_completed_no_recovered_endpoint')
        closed = read_bound_endpoint(winner, initial_R)
        # Recompute the chosen endpoint before changing the canonical record.
        selected = {**closed['proposal'], 'G': closed['G']}
        proposal, repeat = backend.repeat(endpoint_bundles[winner['policy']], selected)
        proposal.update({k: v for k, v in closed['proposal'].items()
            if k in ('acceptance_policy', 'acceptance_policy_version', 'R_reference_initial',
                'R_cap', 'R_armijo_required')})
        if (repeat['objective'] > proposal['J_armijo_bound']
                or proposal['R_armijo_required'] and repeat['F_squared'] > proposal['F_squared_armijo_bound']
                or not proposal['R_armijo_required'] and repeat['F_squared'] > proposal['R_cap']):
            raise ValueError('final canonical repeat violates the endpoint policy')
        if not endpoint_recovered(initial_J, initial_R, proposal, repeat,
                relative_margin=POLICY['recovery_relative_margin']):
            raise ValueError('chosen endpoint did not independently recover original J/R')
        backend._check(torch.tensor(repeat['control'], dtype=torch.float64))
        record = persist_canonical(output, record, winner, proposal, repeat, closed['G'])
        return finish('comparison_completed_recovered_endpoint_promoted')
    except TimeoutError as error:
        return finish('internal_time_budget_exhausted_canonical_prefix_preserved', str(error))
    except Exception as error:
        record['failure_type'] = type(error).__name__
        finish('comparison_failed_canonical_prefix_preserved', str(error), failed=True)
        raise


def run(plan_path: Path, digest: str, output: Path) -> dict[str, Any]:
    _load_plan(plan_path, digest)
    paths = (output, output.with_suffix('.run.json'), output.with_suffix('.resource.json'), output.with_suffix('.log'))
    if any(p.exists() or p.is_symlink() for p in paths) or output.parent.exists():
        raise ValueError('experiment output directory must be fresh')
    output.parent.mkdir(parents=True)
    result = run_guarded_diagnostic([str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()),
        '--child', '--plan', str(plan_path), '--plan-sha256', digest, '--output', str(output)],
        wall_seconds=POLICY['outer_seconds'], rss_bytes=POLICY['rss_bytes'],
        report_path=paths[2], log_path=paths[3])
    parent = {'execution_status': prior.shared._execution_status(result,
        wall_seconds=POLICY['outer_seconds'], rss_bytes=POLICY['rss_bytes']),
        'resource': result, 'child_sha256': _sha(output) if output.exists() else None}
    prior.tangent._write(paths[1], parent)
    return parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR / 'step.json')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        _run_child(args.plan, args.plan_sha256, args.output)
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.output), indent=2))


if __name__ == '__main__':
    main()
