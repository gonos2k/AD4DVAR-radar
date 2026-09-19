"""Exact local FV response on the saved 240x240 rotation, without a dense H."""
from dataclasses import replace
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fv_rotation_demo as demo
from advar.fv_sensitivity import compute_fv_observation_response, refine_fv_stationarity, face_branch_margin
from advar.physics import echo_to_dbz
from advar.variational import forecast_fv_analysis, robust_objective

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'graphify-out/fv-root-cause-20260919'


def run(*, impacts: bool, leads: int, prefix: str = "rotation240", warm_start: Path | None = None):
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT/'src/advar/fv_sensitivity.py', ROOT/'src/advar/variational.py',
                               ROOT/'src/advar/transport.py', Path(__file__))}
    saved = torch.load(HERE/f'{prefix}_refined.pt', weights_only=False)
    control, observations, frozen = saved['control'], saved['observations'], saved['frozen']
    boundary = demo._boundary_schedule(0.0, leads)
    support = demo._support_schedule(leads)
    truth = echo_to_dbz(torch.stack([demo._q_field((i+1)*600.0) for i in range(leads)]), min_dbz=demo.MIN_DBZ)
    weights = torch.ones_like(truth)

    def contract(y):
        return replace(frozen, initial_background_dbz=y[0], input_frames_dbz=y)

    def score(c, y):
        with torch.no_grad():
            trajectory = forecast_fv_analysis(c, contract(y), leads=leads,
                boundary_start_interval=2, boundary_echo=boundary, boundary_support=support)
            prediction = echo_to_dbz(trajectory.frames_linear[1:], min_dbz=demo.MIN_DBZ)
            return float((weights * (prediction-truth).square()).sum()/weights.sum())

    cache = HERE/f'{prefix}_response_{leads}.pt'
    report_path = HERE/f'{prefix}_response_{leads}.json'
    if not impacts:
        response = compute_fv_observation_response(control, observations, frozen,
            verification_dbz=truth, metric_weight=weights, leads=leads,
            boundary_start_interval=2, boundary_echo=boundary, boundary_support=support,
            background_dependency='first_observation', curvature='exact_robust_hessian',
            maximum_normal_products=128)
        torch.save({'sensitivity': response.sensitivity_dbz.detach(), 'score': response.score,
                    'control': control}, cache)
        report = dict(scope='240x240 exact local response; fixed analytic boundaries and verification',
            leads=leads, curvature=response.curvature, gradient_max=response.gradient_max,
            adjoint_relative_residual=response.adjoint_relative_residual,
            normal_products=response.normal_products, face_margin=response.face_margin,
            score=response.score, sensitivity_norm=float(response.sensitivity_dbz.norm()),
            source_hashes=source_hashes,
            response_tensor_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            refined_checkpoint_sha256=hashlib.sha256((HERE/f'{prefix}_refined.pt').read_bytes()).hexdigest())
    else:
        adjoint = torch.load(cache, weights_only=True)
        if not torch.equal(control, adjoint['control']):
            raise ValueError('response cache belongs to a different analysis control')
        report = json.loads(report_path.read_text())
        report['impact_source_hashes'] = source_hashes
        height, width = observations.dbz.shape[-2:]
        yy = torch.arange(height, dtype=torch.float64)[:, None] / height
        xx = torch.arange(width, dtype=torch.float64)[None, :] / width
        spatial = 0.5 + 0.5 * torch.cos(2*torch.pi*xx) * torch.cos(2*torch.pi*yy)
        direction = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float64)[:, None, None] * spatial
        slope = float((adjoint['sensitivity'] * direction).sum())
        report['directional_slope'] = slope
        report['perturbation'] = 'fixed smooth nonuniform positive dBZ bias; temporal amplitudes 0.5, 0.75, 1; spatial 0.5+0.5*cos(2*pi*x/W)*cos(2*pi*y/H)'
        report['impacts'] = []
        nominal_score = score(control, observations.dbz)
        seed = torch.load(warm_start, weights_only=True) if warm_start is not None else None
        previous_solution = None
        previous_step = None
        for h in (0.001, 0.0005):
            y = observations.dbz + h * direction
            changed = replace(observations, dbz=y)
            checkpoint = HERE / f'{prefix}_impact_{h:g}.pt'
            start = control
            resumed = False
            warm_start_kind = 'nominal analysis'
            if seed is not None and torch.equal(seed['observations'], y):
                start = seed['control']
                warm_start_kind = 'saved candidate; stationarity rechecked under current operator'
            elif previous_solution is not None:
                start = control + (h / previous_step) * (previous_solution - control)
                warm_start_kind = 'interpolated previous reanalysis; stationarity rechecked'
            if checkpoint.exists():
                previous = torch.load(checkpoint, weights_only=True)
                if (previous['source_hashes'] != source_hashes
                        or not torch.equal(previous['base_control'], control)
                        or not torch.equal(previous['observations'], y)):
                    raise ValueError('impact checkpoint belongs to a different problem')
                start = previous['control']
                resumed = True

            def save_step(value, record):
                torch.save(dict(control=value, base_control=control,
                    observations=y, source_hashes=source_hashes, last_step=record), checkpoint)

            refined, records = refine_fv_stationarity(start, changed, contract(y),
                gradient_tolerance=1e-8, maximum_iterations=4, on_step=save_step)
            previous_solution, previous_step = refined, h
            actual = score(refined, y) - nominal_score
            g = torch.func.grad(robust_objective)(refined, changed, contract(y))
            report['impacts'].append(dict(step=h, actual_change=actual, linear_prediction=h*slope,
                taylor_error=abs(actual-h*slope), gradient_norm=float(g.norm()),
                gradient_max=float(g.abs().max()), resumed=resumed, warm_start=warm_start_kind,
                face_margin=face_branch_margin(control,refined,frozen), refinement=records))
            report_path.write_text(json.dumps(report,indent=2)+'\n')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leads', type=int, default=18)
    parser.add_argument('--impacts', action='store_true')
    parser.add_argument("--prefix", default="rotation240")
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    run(impacts=args.impacts, leads=args.leads, prefix=args.prefix, warm_start=args.warm_start)
