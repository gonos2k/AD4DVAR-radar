"""Compare a conditional research response API with the saved dense FV oracle.

No nonlinear solves and no new parameter directions. The dense matrix is only
an independent reference: the response API never receives it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from advar import variational as v
from advar.local_response import compute_local_response

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'graphify-out/fv-root-cause-20260919'


def _load(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


# Root recorded by the archived PR168 producer, not the current checkout.
ARCHIVED_ROOT = Path('/Users/yhlee/ADVAR')


def check_archived_sources(fingerprints, *, archived_root=ARCHIVED_ROOT, root=ROOT):
    """Preserve repository-relative identities when relocating a checkout."""
    for archived_path, fingerprint in fingerprints.items():
        relative = Path(archived_path).relative_to(archived_root)
        if '..' in relative.parts:
            raise ValueError('archived source escapes declared root')
        current = root / relative
        if hashlib.sha256(current.read_bytes()).hexdigest() != fingerprint:
            raise ValueError('archived core source identity mismatch')


def make_research_functions(obs, frozen, boundary, support, pattern, verification, expected_branch):
    """Fixed full-support mean-background contract; not a general FV API.

    Only y and a scalar mean-background coefficient are differentiated. All
    masks, precision, geometry and supplied boundary arrays remain fixed.
    The strict tracer is a serial research diagnostic (it patches transport).
    """
    probe = _load('fv_minmod_inverse_probe')
    spec = frozen.fv_transport
    if (spec is None or spec.reconstruction != 'minmod'
            or obs.dbz.dtype != torch.float64 or obs.dbz.device.type != 'cpu'
            or obs.dbz.shape != (3, 4, 5) or spec.substeps_per_interval != 9
            or not bool(obs.detected_mask.all()) or not bool(obs.valid_mask.all())
            or not bool(frozen.initial_support_mask.all())
            or frozen.neural_prior_dependency is not None
            or frozen.grid_time_contract is not None):
        raise ValueError('research minmod requires the bounded CPU FP64 full-support contract')
    for schedule in (spec.boundary_support, support):
        for stages in schedule:
            for edges in stages:
                if not all(bool(torch.all(edge == 1)) for edge in edges):
                    raise ValueError('research minmod requires fully known boundary support')
    for value in (pattern, verification):
        if value.shape != (4, 5) or value.dtype != obs.dbz.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('research minmod pattern/verification mismatch')

    def contract(p):
        y = p[:-1].reshape_as(obs.dbz)
        return replace(frozen, initial_background_dbz=y[0] + p[-1]*pattern)

    def objective(c, p):
        return v.robust_objective(c, replace(obs, dbz=p[:-1].reshape_as(obs.dbz)), contract(p))

    def forecast(c, p):
        return probe.echo_to_dbz(v.forecast_fv_analysis(
            c, contract(p), leads=1, boundary_start_interval=2,
            boundary_echo=boundary, boundary_support=support,
        ).frames_linear[-1], min_dbz=frozen.nowcast_config.min_dbz)

    def score(c, p):
        return (forecast(c, p)-verification).square().mean()

    def check_branch(c, p):
        if c.shape != (26,) or p.shape != (61,):
            raise ValueError('research minmod control/parameter shape mismatch')
        background = contract(p).initial_background_dbz
        cfg = frozen.nowcast_config
        # Keep the differentiable mean background away from both conversion
        # floors and the upper cap; this fixture is well inside the upper branch.
        ac = frozen.analysis_config
        offset = (background-cfg.min_dbz)/ac.echo_transform_scale_dbz
        margin = 64*torch.finfo(background.dtype).eps*((background.abs()+abs(cfg.min_dbz))/ac.echo_transform_scale_dbz+ac.transform_epsilon)
        if not bool(((offset-ac.transform_epsilon > margin) & (background < cfg.max_dbz)).all()):
            raise ValueError('research minmod background is outside its smooth branch')
        if not bool(((p[:-1] > ac.detection_limit_dbz) & (p[:-1] < cfg.max_dbz)).all()):
            raise ValueError('research minmod requires fixed detected observations')
        branch = probe.inspect_branches(lambda: forecast(c, p))
        if (branch['euler_stages'] != 54 or branch['choices'] != expected_branch['choices']
                or branch['face_signs'] != expected_branch['face_signs']):
            raise ValueError('research minmod nominal branch identity mismatch')
        return branch, "strict full-support 4x5 minmod; fixed masks/precision/boundaries; no finite-path or general FV eligibility"

    return objective, score, check_branch


def run(output: Path):
    started = time.monotonic()
    probe = _load('fv_minmod_inverse_probe')
    obs, frozen, boundary, support = probe.make_spatial_case()
    saved_path = EVIDENCE / 'minmod_middle_time_bias_final.json'
    saved = json.loads(saved_path.read_text())
    c = torch.tensor(saved['nominal_control'], dtype=torch.float64)
    p = torch.cat((obs.dbz.flatten(), obs.dbz.new_tensor([.02])))
    pattern = torch.linspace(-.2, .3, 20, dtype=torch.float64).reshape(4, 5)
    if _digest(c) != saved['input_identity']['control_sha256'] or _digest(p) != saved['input_identity']['parameters_sha256']:
        raise ValueError('archived nominal input identity mismatch')
    check_archived_sources(saved['cache_payload_identity']['core_source_sha256'])
    # Reproduce the archived conditional verification field, then hold it fixed.
    nominal_contract = replace(frozen, initial_background_dbz=obs.dbz[0]+p[-1]*pattern)
    nominal_forecast = probe.echo_to_dbz(v.forecast_fv_analysis(
        c, nominal_contract, leads=1, boundary_start_interval=2,
        boundary_echo=boundary, boundary_support=support,
    ).frames_linear[-1], min_dbz=-10.)
    verification = (nominal_forecast+.1*pattern).detach()
    objective, score, branch_check = make_research_functions(
        obs, frozen, boundary, support, pattern, verification, saved['nominal_branch'])
    directions = {name: torch.tensor(item['parameter_direction'], dtype=torch.float64)
                  for name, item in saved['tangents'].items()}
    identity = {'control': _digest(c), 'parameters': _digest(p),
                'verification': _digest(verification),
                'saved_report': hashlib.sha256(saved_path.read_bytes()).hexdigest()}
    preparation_seconds = time.monotonic()-started
    response_started = time.monotonic()
    response = compute_local_response(objective, score, c, p, directions,
                                      branch_check=branch_check, input_identity=identity)
    response_seconds = time.monotonic()-response_started
    comparison_started = time.monotonic()
    H = torch.tensor(saved['hessian']['matrix'], dtype=torch.float64)
    gradient = torch.func.grad(objective)
    products = torch.stack([torch.func.jvp(lambda x: gradient(x, p), (c,), (basis,))[1]
                            for basis in torch.eye(c.numel(), dtype=c.dtype)], dim=1)
    hvp_error = float((products-H).norm()/H.norm())
    rhs, direct = torch.func.grad(score, argnums=(0, 1))(c, p)
    dense_adjoint = torch.linalg.solve(H.T, rhs)
    report = {
        'status': 'pending_comparison', 'scope': 'conditional strict-branch full-support 4x5 minmod research response',
        'general_minmod_response_eligible': False, 'finite_path_certified': False,
        'dense_used_in_solver': False, 'nonlinear_reanalyses': 0,
        'input_identity': identity, 'hvp_basis_columns': c.numel(),
        'hvp_dense_relative_error': hvp_error,
        'declared_tolerances': {'hvp_dense': 1e-10, 'adjoint_dense': 1e-8, 'response_dense': 1e-6},
        'dense_min_eigenvalue': float(torch.linalg.eigvalsh(H)[0]),
        'preparation_seconds': preparation_seconds, 'response_seconds': response_seconds, 'source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT/'src/advar/local_response.py', ROOT/'src/advar/matrix_free.py',
                         Path(probe.__file__), ROOT/'src/advar/variational.py', ROOT/'src/advar/transport.py')},
    }
    adjoint_error = float((response.adjoint-dense_adjoint).norm()/dense_adjoint.norm())
    rows = {}
    for name, direction in directions.items():
        cross = torch.tensor(saved['tangents'][name]['cross_gradient'], dtype=c.dtype)
        expected = float(direct.dot(direction)-dense_adjoint.dot(cross))
        actual = float(response.total[name])
        rows[name] = {'direct': float(response.direct[name]), 'indirect': float(response.indirect[name]),
                      'total': actual, 'dense_response': expected,
                      'relative_difference': abs(actual-expected)/abs(expected),
                      'archived_response': saved['adjoint']['response_sensitivity'][name],
                      'full_vjp_projection': float(response.total_gradient.dot(direction)),
                      'projection_relative_difference': float((response.total_gradient.dot(direction)-response.total[name]).abs()/response.total[name].abs()),
                      'mixed_gradient_max_difference': float((response.mixed_gradients[name]-cross).abs().max())}
    report.update(
        adjoint_dense_relative_error=adjoint_error, adjoint=response.adjoint.tolist(),
        parameter_gradients={name: getattr(response, name).tolist() for name in
                             ("direct_gradient", "indirect_gradient", "total_gradient")},
        gradient_max=response.gradient_max,
        actual_transpose_residual=response.true_adjoint_relative_residual,
        pcg_reported_residual=response.pcg_relative_residual,
        pcg_iterations=response.pcg_iterations, hvp_count=response.hvp_count,
        branch=response.branch_signature, api_scope=response.scope, directions=rows,
        fresh_score_gradient_difference={
            'control': float((rhs-torch.tensor(saved['adjoint']['rhs'], dtype=c.dtype)).abs().max()),
            'parameters': float((direct-torch.tensor(saved['adjoint']['direct'], dtype=p.dtype)).abs().max()),
        },
    )
    report['status'] = 'complete' if (
        bool(torch.isfinite(torch.linalg.eigvalsh(H)).all()) and report['dense_min_eigenvalue'] > 0
        and hvp_error <= 1e-10 and adjoint_error <= 1e-8
        and response.true_adjoint_relative_residual <= 1e-10
        and all(row['relative_difference'] <= 1e-6 and row['projection_relative_difference'] <= 1e-6
                and row['mixed_gradient_max_difference'] == 0
                for row in rows.values())
        and all(x == 0 for x in report['fresh_score_gradient_difference'].values())
    ) else 'comparison_failed'
    report['comparison_seconds'] = time.monotonic()-comparison_started
    report['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(result['status'])
