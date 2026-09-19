"""Reproduce the rejected C5c endpoint, then verify a local gradient root."""
import json
import runpy
from dataclasses import replace
from pathlib import Path
import torch
from advar import variational as v

ROOT = Path(__file__).resolve().parents[2]
probe = runpy.run_path(str(ROOT / 'examples/weather_scenarios/fv_sensitivity_probe.py'))
old = runpy.run_path(str(ROOT / 'graphify-out/fv-root-cause-20260913/c5c_shared_reanalysis.py'))
obs, frozen, _, _ = probe['make_case']()
y = obs.dbz - 5e-4 * torch.sin(torch.arange(obs.dbz.numel(), dtype=obs.dbz.dtype)).reshape_as(obs.dbz)
result, frozen = old['_solve'](y, obs, frozen)
def objective(c, z):
    return v.robust_objective(c, replace(obs, dbz=z), replace(frozen, initial_background_dbz=z[0]))
margins = []
def check_step(a, b):
    margins.append(probe['face_branch_margin'](a, b, frozen))
c = probe['polish'](objective, result.control, y, check_step=check_step)
gradient = torch.func.grad(objective)
g = gradient(c, y)
H = probe['dense_hessian'](gradient, c, y)
eigenvalues = torch.linalg.eigvalsh(H)
report = {
    'scope': '4x5 CPU FP64 local exact-Hessian gradient-root oracle; production result remains unverified',
    'before': old['_diagnostics'](result),
    'after_gradient_max': float(g.abs().max()),
    'after_gradient_norm': float(g.norm()),
    'hessian_min_eigenvalue': float(eigenvalues.min()),
    'hessian_condition_number': float(eigenvalues.max()/eigenvalues.min()),
    'local_linearized_control_error_estimate': float(g.norm()/eigenvalues.min()),
    'control_step_norm': float((c-result.control).norm()),
    'minimum_checked_step_face_box_margin': min(margins),
    'sampled_local_evidence_not_interval_certificate': True,
}
assert report['after_gradient_max'] < 1e-10
assert report['hessian_min_eigenvalue'] > 0
assert report['minimum_checked_step_face_box_margin'] > 0
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
