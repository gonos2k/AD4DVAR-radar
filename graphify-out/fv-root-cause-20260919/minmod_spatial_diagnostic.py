from __future__ import annotations
import importlib.util, json, time
from dataclasses import replace
from pathlib import Path
import torch

diagnostic_started = time.monotonic()

ROOT = Path('/Users/yhlee/ADVAR')
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
try:
    nominal_branch = branch(control, p)
    result['nominal_branch'] = compact_branch(nominal_branch)
    result['perturbed_start'] = {
        'objective': float(objective(control, p_minus)),
        'gradient_max': float(gradient(control, p_minus).abs().max()),
        'gradient_l2': float(gradient(control, p_minus).norm()),
    }
    dump(result)

    c = control.detach().clone()
    g = None
    for iteration in range(12):
        started = time.monotonic()
        J = objective(c, p_minus)
        g = gradient(c, p_minus)
        rec = {
            'iteration': iteration,
            'objective': float(J),
            'gradient_max': float(g.abs().max()),
            'gradient_l2': float(g.norm()),
            'accepted_scales': [],
        }
        H = probe._oracle.dense_hessian(gradient, c, p_minus)
        rec['hessian_min_eigenvalue'] = float(torch.linalg.eigvalsh(H)[0])
        rec['hessian_condition_number'] = float(torch.linalg.cond(H))
        factor = torch.linalg.cholesky(H)
        if g.abs().max() < 1e-10:
            rec['converged_before_step'] = True
            rec['elapsed_seconds'] = time.monotonic() - started
            result['iterations'].append(rec)
            dump(result)
            break
        step = torch.cholesky_solve(-g[:, None], factor)[:, 0]
        rec['step_l2'] = float(step.norm())
        accepted = False
        for power in range(16):
            scale = 2.0 ** (-power)
            trial = c + scale * step
            finite = bool(torch.isfinite(objective(trial, p_minus)))
            trial_g = gradient(trial, p_minus)
            finite_grad = bool(torch.isfinite(trial_g).all())
            ratio_sq = float((trial_g.norm() / g.norm()).square()) if finite_grad else None
            improves = finite and finite_grad and ratio_sq <= 1 - 2e-4 * scale
            rec['accepted_scales'].append({
                'scale': scale, 'finite': finite, 'finite_gradient': finite_grad,
                'trial_gradient_max': float(trial_g.abs().max()) if finite_grad else None,
                'trial_gradient_l2': float(trial_g.norm()) if finite_grad else None,
                'gradient_ratio_squared': ratio_sq, 'accepted': bool(improves),
            })
            if improves:
                c = trial.detach()
                accepted = True
                rec['accepted_scale'] = scale
                rec['accepted_control'] = c.tolist()
                # Save every accepted point separately for post-mortem inspection.
                (OUT/f'accepted_{iteration:02d}.json').write_text(json.dumps({
                    'iteration': iteration, 'scale': scale,
                    'control': c.tolist(), 'gradient_max': float(trial_g.abs().max()),
                    'gradient_l2': float(trial_g.norm()), 'objective': float(objective(c, p_minus)),
                }, indent=2)+'\n')
                break
        rec['accepted'] = accepted
        rec['elapsed_seconds'] = time.monotonic() - started
        result['iterations'].append(rec)
        dump(result)
        if not accepted:
            raise RuntimeError('verification Newton refinement did not decrease')
        if float(gradient(c, p_minus).abs().max()) < 1e-10:
            result['converged'] = True
            break
    else:
        final_g = gradient(c, p_minus)
        if float(final_g.abs().max()) >= 1e-10:
            raise RuntimeError('verification point is not sufficiently stationary')
        result['converged'] = True

    final_branch = branch(c, p_minus)
    result['final_control'] = c.tolist()
    result['final'] = {
        'objective': float(objective(c, p_minus)),
        'gradient_max': float(gradient(c, p_minus).abs().max()),
        'gradient_l2': float(gradient(c, p_minus).norm()),
        'branch': compact_branch(final_branch),
        'branch_equal_nominal': (final_branch['choices'], final_branch['face_signs']) == (nominal_branch['choices'], nominal_branch['face_signs']),
    }
    result['status'] = 'complete'
except Exception as exc:
    result['status'] = 'failed'
    result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    result['elapsed_seconds'] = time.monotonic() - diagnostic_started
    dump(result)
    print(json.dumps({k: result[k] for k in ('status','failure','perturbed_start','final','elapsed_seconds') if k in result}, indent=2), flush=True)
