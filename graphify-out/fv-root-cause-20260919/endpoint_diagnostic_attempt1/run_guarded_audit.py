"""One-shot guarded endpoint audit; preserves original numerical archives."""
from pathlib import Path
import hashlib, json, subprocess, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.weather_scenarios.fv86_resource_runner import run_guarded
from examples.weather_scenarios import fv_point_3h_endpoint_diagnostics as diagnostic

DIRECTORY = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'graphify-out/fv-root-cause-20260919/coupled_original_j_continuation_attempt1/coupled_original_j_continuation.json'
PLAN = ROOT / 'graphify-out/fv-root-cause-20260919/PR238_ENDPOINT_DIAGNOSTIC_PLAN_20261004.md'
PINS = {
    'examples/weather_scenarios/fv_point_3h_endpoint_diagnostics.py': 'd1e304ebd134a3ec0a18e55adcdc8f4064bb00675120a40fc9fe0eb6f9de455a',
    'tests/test_fv_point_3h_endpoint_diagnostics.py': '64af675be3274dd9e6c09c545604c29c8241a8b8a17b4568e167001c06ca2d4a',
    'graphify-out/fv-root-cause-20260919/PR238_ENDPOINT_DIAGNOSTIC_PLAN_20261004.md': '8de1ac8b29c383288c10476b333058f8d1c0746888670fb12e97a5c46b5e9858',
}
ARCHIVE_SHA = '3e2e1e247be0de1c6b5e27d4c835dbbe6b456f7a0c0dc2e0e90ac18d71fd8385'
PARENT_SHA = '5cf9c94f61f8154ab895f3c7cd3d85cc45e8841774554629e35def59785f0790'
RESOURCE_SHA = '6e642669e49a776a928eaa9832e470bb3f27286902c4527f5c1910a8135ffc96'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, data):
    path = DIRECTORY / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

if __name__ == '__main__':
    if any((DIRECTORY / name).exists() for name in ('preflight.json', 'audit.json', 'audit.resource.json', 'audit.run.json')):
        raise ValueError('one-shot audit directory already contains an attempt; no automatic retry')
    for name, digest in PINS.items():
        if sha(ROOT / name) != digest:
            raise ValueError('reviewed source/plan changed: ' + name)
    raw = json.loads(ARCHIVE.read_text())
    parent_path, resource_path = ARCHIVE.with_suffix('.run.json'), ARCHIVE.with_suffix('.resource.json')
    if (sha(ARCHIVE), sha(parent_path), sha(resource_path)) != (ARCHIVE_SHA, PARENT_SHA, RESOURCE_SHA):
        raise ValueError('historical numerical archive/sidecar changed')
    diagnostic._validate_parent_and_resource(raw, json.loads(parent_path.read_text()), json.loads(resource_path.read_text()))
    trial = diagnostic.extract_last_accepted_trial(raw)
    paths = sorted(set(raw['source_before']) | set(PINS) | {
        str(ARCHIVE.relative_to(ROOT)), str(parent_path.relative_to(ROOT)), str(resource_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)), 'examples/weather_scenarios/fv86_resource_runner.py',
        'examples/weather_scenarios/fv_chart_diagnostics.py', 'tests/test_fv_chart_diagnostics.py',
    })
    before = {name: sha(ROOT / name) for name in paths}
    if any(before[name] != digest for name, digest in raw['source_before'].items()):
        raise ValueError('historical original-problem dependency changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    command = [str(ROOT / '.venv/bin/python'), '-m', 'examples.weather_scenarios.fv_point_3h_endpoint_diagnostics',
               '--archive', str(ARCHIVE), '--archive-sha256', ARCHIVE_SHA,
               '--parent-sha256', PARENT_SHA, '--resource-sha256', RESOURCE_SHA,
               '--plan', str(PLAN), '--output', str(DIRECTORY / 'audit.json')]
    write('preflight.json', {'scope': 'metadata-only preflight; no FV evaluation', 'command': command,
                            'source_before': before, 'git_head': head, 'archive_sha256': ARCHIVE_SHA,
                            'endpoint_control_sha256': raw['last_accepted_control_sha256'],
                            'parameters_sha256': raw['parameters_sha256'], 'saved_objective': trial['objective'],
                            'budget': {'internal_seconds': 240, 'outer_seconds': 300, 'sampled_rss_bytes': 1024**3}})
    resource = run_guarded(command, wall_seconds=300, rss_bytes=1024**3,
                           report_path=DIRECTORY / 'audit.resource.json', log_path=DIRECTORY / 'audit.log')
    after = {name: sha(ROOT / name) for name in paths}
    child, read_error = None, None
    try:
        child = json.loads((DIRECTORY / 'audit.json').read_text())
    except (OSError, ValueError) as error:
        read_error = str(error)
    endpoint_matches = (child is not None
                        and child.get('endpoint_control_sha256') == raw['last_accepted_control_sha256']
                        and child.get('endpoint_control') == raw['last_accepted_control']
                        and child.get('parameters_sha256') == raw['parameters_sha256']
                        and child.get('parameters') == raw['parameters']
                        and child.get('input_before') == raw['input_after']
                        and child.get('input_after') == raw['input_after'])
    complete = (endpoint_matches and child.get('phase') in {'finished', 'diagnostic_refused'}
                and child.get('source_unchanged') is True and child.get('input_unchanged') is True
                and child.get('runtime') == child.get('runtime_after'))
    status = ('resource_limited' if resource['resource_termination'] else
              'completed' if resource['exit_code'] == 0 and not resource['monitor_error']
              and complete and before == after else 'failed')
    record = {'execution_status': status, 'numerical_status': child.get('numerical_status') if child else 'not_reached',
              'resource': resource, 'child_read_error': read_error,
              'source_before': before, 'source_after': after, 'source_unchanged': before == after,
              'endpoint_identity_matches_archive': endpoint_matches,
              'endpoint_control_sha256': child.get('endpoint_control_sha256') if child else None,
              'parameters_sha256': child.get('parameters_sha256') if child else None,
              'input_before': child.get('input_before') if child else None,
              'input_after': child.get('input_after') if child else None,
              'git_head_before': head, 'git_head_after': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'archive_unchanged': sha(ARCHIVE) == ARCHIVE_SHA, 'optimizer_step_applied': False,
              'response_computed': False, 'physical_validated': False}
    write('audit.run.json', record)
    print(json.dumps({'execution_status': status, 'numerical_status': record['numerical_status'],
                      'elapsed_seconds': resource['elapsed_seconds'], 'peak_sampled_rss_bytes': resource['sampled_peak_rss_bytes']}))
    sys.exit(0 if status == 'completed' else 1)
