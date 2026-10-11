"""Execute the frozen candidate loop with chart failure, without FV or HVP."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

root = Path(__file__).resolve().parents[2]
source = root / 'examples/weather_scenarios/fv_point_3h_gn_local_candidate_commit.py'
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_child')
loop = next(n for n in function.body if isinstance(n, ast.For) and ast.unparse(n.iter).startswith('enumerate(POLICY'))
module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
writes, calls = [], {'chart': 0, 'evaluate': 0}


def chart(*args):
    calls['chart'] += 1
    raise ValueError('synthetic chart failure')


def evaluate(*args):
    calls['evaluate'] += 1
    raise AssertionError('must not evaluate')


record = {}
environment = {'Any': Any, 'POLICY': {'fixed_candidate_alphas': [2e-5, 1e-5]},
    'check': lambda: None, 'trials': [], 'record': record, 'output': None,
    'tangent': SimpleNamespace(_write=lambda out, value: writes.append(json.loads(json.dumps(value)))),
    'prepared': SimpleNamespace(local=SimpleNamespace(_chart_candidate=chart)),
    'control': None, 'weights': None, 'direction': None, 'pivot': 0, 'evaluate': evaluate}
try:
    exec(compile(module, str(source), 'exec'), environment)
except ValueError as error:
    failure = str(error)
result = {'reproduction': 'AST exact frozen candidate loop; chart fails before evaluator; no FV/HVP',
    'chart_calls': calls['chart'], 'actual_evaluator_calls': calls['evaluate'],
    'saved_candidate_evaluations_started': writes[-1]['candidate_evaluations_started'],
    'saved_trial_status': writes[-1]['candidate_trials'][0]['status'], 'exception': failure,
    'candidate_committed': record.get('candidate_committed', False)}
assert result['actual_evaluator_calls'] == 0 and result['saved_candidate_evaluations_started'] == 1
print(json.dumps(result, indent=2))
