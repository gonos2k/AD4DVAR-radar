"""Fixed three-launch ledger for one original-point resumable curvature experiment."""
from pathlib import Path
import argparse, hashlib, json, sys, time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.weather_scenarios import fv_point_3h_endpoint_diagnostics as diagnostic
from examples.weather_scenarios.fv_diagnostic_guard import atomic_write_text

DIRECTORY = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'graphify-out/fv-root-cause-20260919/coupled_original_j_continuation_attempt1/coupled_original_j_continuation.json'
PLAN = ROOT / 'graphify-out/fv-root-cause-20260919/RESUMABLE_HESSIAN_IMPLEMENTATION_PLAN_20261004.md'
CHECKPOINT = DIRECTORY / 'checkpoint'
MAX_LAUNCHES, TOTAL_SECONDS = 3, 900.0
PINS = {
    'examples/weather_scenarios/fv_hessian_checkpoint.py': '8544c312cd12d0ec91f1c4310b312a9ca229fd8bcfe4c2b584d0ac166cc9e2ee',
    'tests/test_fv_hessian_checkpoint.py': '7bcdff3890578b78aaff1aa3f4a9a0875ea40e218fe1fd6dd5a40782bcf825a5',
    'examples/weather_scenarios/fv_point_3h_endpoint_diagnostics.py': '6e48e23d2ad6be3a54ade759a35d720a9d928163dda70eb0906decc057047c8c',
    'tests/test_fv_point_3h_endpoint_diagnostics.py': 'b43ba53cd220ceb3d16c94fd8f269ed827a6a722145d1d55dc23f62b41dcb459',
    'examples/weather_scenarios/fv_diagnostic_guard.py': 'a1c6b71dcaa4a16952658c9057839cef405238390908476b04e793710f6d275c',
    'tests/test_fv_objective_score_dependency.py': 'de41a42612c22aa63c5eb60aac4fe3b36986657a74afc3c8c2530e1513b66e03',
}
ARCHIVE_SHA = '3e2e1e247be0de1c6b5e27d4c835dbbe6b456f7a0c0dc2e0e90ac18d71fd8385'
PARENT_SHA = '5cf9c94f61f8154ab895f3c7cd3d85cc45e8841774554629e35def59785f0790'
RESOURCE_SHA = '6e642669e49a776a928eaa9832e470bb3f27286902c4527f5c1910a8135ffc96'
CONTROL_SHA = 'cd6b5693e2bd32a8a41a3509b87b61e7cf647b03d864dfccc0a4cccce96e908d'
PARAMETERS_SHA = '8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    atomic_write_text(DIRECTORY / name, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def preflight():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('run the experiment from the repository root')
    for name, digest in PINS.items():
        if sha(ROOT / name) != digest:
            raise ValueError('reviewed source changed: ' + name)
    parent_path, resource_path = ARCHIVE.with_suffix('.run.json'), ARCHIVE.with_suffix('.resource.json')
    if (sha(ARCHIVE), sha(parent_path), sha(resource_path)) != (ARCHIVE_SHA, PARENT_SHA, RESOURCE_SHA):
        raise ValueError('original archive/sidecar changed')
    raw = json.loads(ARCHIVE.read_text())
    diagnostic._validate_parent_and_resource(raw, json.loads(parent_path.read_text()), json.loads(resource_path.read_text()))
    trial = diagnostic.extract_last_accepted_trial(raw)
    if raw['last_accepted_control_sha256'] != CONTROL_SHA or raw['parameters_sha256'] != PARAMETERS_SHA:
        raise ValueError('experiment is not the declared original point')
    paths = sorted(set(raw['source_before']) | set(PINS) | {
        'tests/test_fv_diagnostic_guard.py', str(PLAN.relative_to(ROOT)),
        str(ARCHIVE.relative_to(ROOT)), str(parent_path.relative_to(ROOT)), str(resource_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    })
    sources = {name: sha(ROOT / name) for name in paths}
    if any(sources[name] != digest for name, digest in raw['source_before'].items()):
        raise ValueError('original numerical dependencies changed')
    return {'scope': 'metadata-only; no FV/objective/branch/HVP evaluation', 'source_before': sources,
            'archive_sha256': ARCHIVE_SHA, 'control_sha256': CONTROL_SHA, 'parameters_sha256': PARAMETERS_SHA,
            'checkpoint_directory': str(CHECKPOINT.relative_to(ROOT)), 'saved_objective': trial['objective'],
            'max_guarded_launches': MAX_LAUNCHES, 'cumulative_outer_seconds': TOTAL_SECONDS,
            'per_launch_outer_seconds': 300, 'per_kernel_reserved_seconds': 240,
            'max_kernel_attempts': MAX_LAUNCHES, 'sampled_child_rss_limit_bytes': 1024**3}


def run():
    before = preflight()
    saved = json.loads((DIRECTORY / 'preflight.json').read_text())
    if saved != before:
        raise ValueError('experiment preflight/source receipt changed')
    with (DIRECTORY / '.experiment.claim').open('x') as claim:
        claim.write(json.dumps({'control_sha256': CONTROL_SHA, 'max_launches': MAX_LAUNCHES}) + '\n')
    if CHECKPOINT.exists() or any(DIRECTORY.glob('attempt_*')):
        raise ValueError('fresh experiment required; historical products cannot be recovered')
    started = time.monotonic()
    ledger = {'scope': 'one immutable source/point experiment; no optimization or response',
              'phase': 'running', 'source_before': before['source_before'], 'attempts': [],
              'max_guarded_launches': MAX_LAUNCHES, 'cumulative_outer_seconds': TOTAL_SECONDS,
              'control_sha256': CONTROL_SHA, 'parameters_sha256': PARAMETERS_SHA,
              'checkpoint_directory': before['checkpoint_directory'], 'optimizer_step_applied': False,
              'response_computed': False, 'physical_validated': False}
    write('experiment.json', ledger)
    for index in range(1, MAX_LAUNCHES + 1):
        elapsed = time.monotonic() - started
        if elapsed + 300 > TOTAL_SECONDS:
            ledger['stop_reason'] = 'insufficient cumulative allowance for another full 300-second guard'
            break
        # Reservation is durable before startup, so even a setup failure consumes a launch.
        row = {'attempt': index, 'reserved_outer_seconds': 300, 'status': 'reserved',
               'output_directory': f'attempt_{index}'}
        ledger['attempts'].append(row)
        write('experiment.json', ledger)
        output = DIRECTORY / f'attempt_{index}' / 'audit.json'
        try:
            record = diagnostic.guarded_run(archive=ARCHIVE, archive_sha256=ARCHIVE_SHA,
                parent_sha256=PARENT_SHA, resource_sha256=RESOURCE_SHA, plan=PLAN, output=output,
                checkpoint_directory=CHECKPOINT, checkpoint_max_attempts=MAX_LAUNCHES)
        except Exception as error:
            row.update(status='failed', failure=f'{type(error).__name__}: {error}')
            ledger['stop_reason'] = 'nonbudget exception; no retry'
            write('experiment.json', ledger)
            break
        resource = record.get('resource')
        row.update(status=record['execution_status'], numerical_status=record['numerical_status'],
                   parent_record=str(output.with_suffix('.run.json').relative_to(ROOT)),
                   resource_record=str(output.with_suffix('.resource.json').relative_to(ROOT)),
                   parent_record_sha256=sha(output.with_suffix('.run.json')),
                   resource_record_sha256=sha(output.with_suffix('.resource.json')),
                   elapsed_seconds=resource.get('elapsed_seconds') if isinstance(resource, dict) else None)
        child = json.loads(output.read_text()) if output.exists() else None
        row['checkpoint_progress'] = child.get('checkpoint') if isinstance(child, dict) else None
        write('experiment.json', ledger)
        if record['execution_status'] != 'completed':
            ledger['stop_reason'] = 'execution/resource/integrity failure; no retry'
            break
        if record['numerical_status'] == 'endpoint_diagnostic_completed':
            ledger['stop_reason'] = 'fresh curvature diagnostic completed'
            break
        if record['numerical_status'] != 'checkpoint_budget_refusal':
            ledger['stop_reason'] = 'nonbudget numerical qualification refusal; no retry'
            break
    else:
        ledger['stop_reason'] = 'declared three-launch limit exhausted'
    after = {name: sha(ROOT / name) for name in before['source_before']}
    ledger.update(phase='finished', source_after=after, source_unchanged=after == before['source_before'],
                  original_archive_unchanged=sha(ARCHIVE) == ARCHIVE_SHA,
                  elapsed_seconds=time.monotonic() - started,
                  reserved_outer_seconds=300 * len(ledger['attempts']))
    if not ledger['source_unchanged'] or not ledger['original_archive_unchanged']:
        ledger['stop_reason'] = 'source/archive changed; result publication refused'
    write('experiment.json', ledger)
    print(json.dumps({'stop_reason': ledger['stop_reason'], 'launches': len(ledger['attempts']),
                      'elapsed_seconds': ledger['elapsed_seconds'], 'source_unchanged': ledger['source_unchanged']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        if (DIRECTORY / 'preflight.json').exists():
            raise ValueError('preflight already exists; no silent replacement')
        write('preflight.json', preflight())
        print('Preflight recorded; no numerical execution.')
