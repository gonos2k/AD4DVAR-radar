"""Reuse unchanged training/FD evidence; rerun only stricter held-out inference."""
import json
from pathlib import Path
import sys
from dataclasses import replace
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples/weather_scenarios'))
import fv_learning_probe as p

HERE = Path(__file__).resolve().parent
preliminary = HERE / 'fv_learning_preliminary.json'
if not preliminary.exists():
    preliminary.write_bytes((HERE / 'fv_learning.json').read_bytes())
report = json.loads(preliminary.read_text())
obs, frozen, future, support = p.oracle.make_case()
frozen = replace(frozen, analysis_config=p._analysis_config(frozen))
coordinate = torch.arange(obs.dbz.numel(), dtype=p.DTYPE).reshape_as(obs.dbz)
initial = obs.dbz[0] + 0.08 * torch.cos(torch.arange(obs.dbz[0].numel(), dtype=p.DTYPE).reshape_as(obs.dbz[0]))
trajectory = p._trajectory_dbz(frozen, future, support, initial)
obs, frozen = p._window(replace(obs, dbz=trajectory[:3]), frozen, 0.02 * torch.cos(0.61 * coordinate + 0.4))
verification = trajectory[3:]
weight = torch.ones_like(verification)
thetas = [torch.tensor(report[key], dtype=p.DTYPE) for key in ('initial_log_scale', 'updated_log_scale')]
controls, scores, diagnostics = [], [], []
for theta in thetas:
    result, control = p._run(obs, frozen, theta, polish=True)
    controls.append(control)
    scores.append(float(p._score(control, frozen, future, support, verification, weight)))
    diagnostics.append(p._local_score_error(obs, frozen, future, support, verification, weight, control, theta))
    assert diagnostics[-1]['gradient_max'] < 1e-10
checkpoint = HERE / 'fv_observation_scale.pt'
torch.save({'log_scale': thetas[1]}, checkpoint)
loaded = torch.load(checkpoint, weights_only=True)['log_scale']
_, replay = p._run(obs, frozen, loaded, polish=True)
replay_score = float(p._score(replay, frozen, future, support, verification, weight))
forecast_error = float((p._forecast(replay, frozen, future, support) - p._forecast(controls[1], frozen, future, support)).abs().max())
replay_error = abs(replay_score - scores[1])
roundoff = torch.finfo(p.DTYPE).eps * max(map(abs, scores))
error_estimate = 10 * (sum(d['local_score_error_scale'] for d in diagnostics) + replay_error + roundoff)
gain = scores[0] - scores[1]
for key in ('holdout_repeatability_floor', 'holdout_gain_above_repeatability'):
    report.pop(key, None)
report.update(
    reused_evidence='Training and three FD levels from fv_learning_preliminary.json; theta unchanged',
    holdout_inference='shared solve_analysis then exact local root polish for both parameters and reload',
    checkpoint_scope='parameter-only checkpoint; deterministic heldout fixture reconstructed, same prescribed contract retained for replay',
    numerical_error_scope='local linearized estimate with factor 10, not a rigorous neighborhood bound',
    holdout_score_before=scores[0], holdout_score_after=scores[1], holdout_gain=gain,
    holdout_base_diagnostics=diagnostics[0], holdout_updated_diagnostics=diagnostics[1],
    score_roundoff_scale=roundoff, holdout_error_budget=error_estimate,
    holdout_gain_above_error_budget=gain > error_estimate,
    reload_train_score_error=report.pop('reload_score_error'),
    reload_holdout_score_error=replay_error,
    reload_holdout_control_max_abs=float((replay-controls[1]).abs().max()),
    reload_holdout_forecast_max_abs=forecast_error,
    holdout_branch_margin=p.oracle.face_branch_margin(controls[0], controls[1], frozen),
)
assert gain > error_estimate
assert replay_error == 0 and forecast_error == 0
assert report['holdout_branch_margin'] > 0
(HERE / 'fv_learning.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
