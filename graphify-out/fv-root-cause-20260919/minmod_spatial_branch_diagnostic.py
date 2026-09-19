from __future__ import annotations
import importlib.util, json, time
from dataclasses import replace
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / 'examples/weather_scenarios/fv_minmod_inverse_probe.py'
REPORT_PATH = ROOT / 'graphify-out/fv-root-cause-20260919/minmod_spatial_inverse.json'
OUT = Path('/tmp/advar-minmod-negative-observation-exact')
OUT.mkdir(parents=True, exist_ok=True)
_spec = importlib.util.spec_from_file_location('minmod_probe_diag', PROBE_PATH)
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

nominal = json.loads(REPORT_PATH.read_text())
obs, frozen, boundary, support = probe.make_spatial_case()
shape = frozen.initial_background_dbz.shape
pattern = torch.linspace(-.2, .3, shape.numel(), dtype=obs.dbz.dtype).reshape(shape)
p = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([.02])))
control = torch.tensor(nominal['control'], dtype=torch.float64)
assert control.numel() == 26
h = 1e-3
direction = torch.cat((torch.sin(torch.arange(obs.dbz.numel(), dtype=p.dtype)), p.new_zeros(1)))
p_minus = p - h * direction

def contract(parameters):
    y, theta = parameters[:-1].reshape_as(obs.dbz), parameters[-1]
    return replace(frozen, initial_background_dbz=y[0] + theta * pattern)

def objective(c, parameters):
    return probe.v.robust_objective(c, replace(obs, dbz=parameters[:-1].reshape_as(obs.dbz)), contract(parameters))

def forecast(c, parameters):
    return probe.echo_to_dbz(probe.v.forecast_fv_analysis(
        c, contract(parameters), leads=1, boundary_start_interval=2,
        boundary_echo=boundary, boundary_support=support,
    ).frames_linear[-1], min_dbz=-10.)

gradient = torch.func.grad(objective)

def branch(c, parameters):
    return probe.inspect_branches(lambda: forecast(c, parameters))

def compact_branch(b):
    return {
        'euler_stages': b['euler_stages'],
        'minimum_scaled_slope_margin': b['minimum_scaled_slope_margin'],
        'choices': b['choices'],
        'face_signs': b['face_signs'],
    }

def dump(data):
    (OUT/'diagnostic.json').write_text(json.dumps(data, indent=2)+'\n')

result = {
    'status': 'running',
    'source': str(PROBE_PATH),
    'nominal_report': str(REPORT_PATH),
    'contract': 'current make_spatial_case; same 26-control parameterized objective as run_probe(spatial=True)',
    'perturbation': {'parameter': 'producer observation direction sin(arange(n)) with theta fixed', 'h': h, 'sign': -1},
    'nominal_saved_gradient_max': nominal.get('gradient_max'),
    'nominal_saved_objective': nominal.get('objective'),
    'start_control': control.tolist(),
    'iterations': [],
}
nominal_branch=branch(control,p)
records = json.loads((Path(__file__).with_name('minmod_spatial_diagnostic_partial.json')).read_text())
for d in records['iterations']:
    path = f"accepted_{d['iteration']:02d}"
    c=torch.tensor(d['accepted_control'],dtype=torch.float64)
    try:
        b=branch(c,p_minus)
        print(path, b['minimum_scaled_slope_margin'], b['choices']==nominal_branch['choices'], b['face_signs']==nominal_branch['face_signs'],flush=True)
    except ValueError as e: print(path,str(e),flush=True)
