"""Whole 18-lead transport costs, with bounded isolated workers (not full P1 D7)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WALL_LIMIT = 120.0
RSS_LIMIT = 4 * 1024**3

_WORKER_FIELDS = {
    'status', 'scheme', 'operation', 'seconds', 'peak_rss_bytes',
    'output_shape', 'product_bytes', 'output_sum', 'python', 'torch',
    'machine', 'dtype', 'threads', 'steps', 'flow_controls', 'replay',
    'directional_score',
}


def _failed_result(scheme, operation, reason, log):
    return dict(status='failed', scheme=scheme, operation=operation,
                reason=reason, log=str(log))


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _load_worker_result(path, scheme, operation, replay):
    """Load and validate one worker report, failing closed on bad evidence."""
    try:
        result = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f'worker result unavailable or invalid: {error}'
    if not isinstance(result, dict):
        return None, 'worker result is not an object'

    missing = _WORKER_FIELDS - result.keys()
    if missing:
        return None, f'worker result missing fields: {sorted(missing)}'
    if result['status'] != 'passed':
        return None, f"worker result status is {result['status']!r}"
    if result['scheme'] != scheme or result['operation'] != operation:
        return None, 'worker result scheme or operation does not match request'
    if result['replay'] is not replay:
        return None, 'worker result replay mode does not match request'

    for field in ('seconds', 'peak_rss_bytes', 'product_bytes', 'output_sum'):
        if not _finite_number(result[field]) or result[field] < 0:
            return None, f'worker result field {field!r} is not finite and nonnegative'
    if type(result['peak_rss_bytes']) is not int or type(result['product_bytes']) is not int:
        return None, 'worker result byte counts are not integers'
    if result['peak_rss_bytes'] <= 0 or result['product_bytes'] <= 0:
        return None, 'worker result byte counts must be positive'
    if result['seconds'] > WALL_LIMIT:
        return None, 'worker result seconds exceeds wall limit'
    if type(result['threads']) is not int or type(result['steps']) is not int \
            or type(result['flow_controls']) is not int:
        return None, 'worker result execution counts are not integers'
    if result['threads'] < 1 or result['steps'] < 1 or result['flow_controls'] < 1:
        return None, 'worker result execution counts are invalid'
    if result['output_shape'] != [18, 128, 128] or result['steps'] != 18 * 39 \
            or result['flow_controls'] != 24 or result['threads'] != 1:
        return None, 'worker result dimensions or execution contract does not match'
    if any(type(size) is not int or size < 0 for size in result['output_shape']):
        return None, 'worker result output shape is malformed'
    if not all(isinstance(result[field], str) and result[field] for field in
               ('python', 'torch', 'machine', 'dtype')):
        return None, 'worker result metadata is malformed'
    if result['dtype'] != 'float64':
        return None, 'worker result dtype does not match the FP64 workload'
    score = result['directional_score']
    if operation in ('jvp', 'vjp'):
        if not _finite_number(score):
            return None, f'worker result field {"directional_score"!r} is not finite'
    elif score is not None and not _finite_number(score):
        return None, f'worker result field {"directional_score"!r} is malformed'
    return result, None


def _monitor_worker(process, started):
    """Poll one worker and apply limits, including the post-exit wall check."""
    sampled_peak = 0
    exceeded = None
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                             capture_output=True, text=True, check=False).stdout.strip()
        sampled_peak = max(sampled_peak, int(rss or 0) * 1024)
        if sampled_peak > RSS_LIMIT or time.perf_counter() - started > WALL_LIMIT:
            exceeded = 'rss' if sampled_peak > RSS_LIMIT else 'wall_time'
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            break
        time.sleep(.25)
    process_wall_seconds = time.perf_counter() - started
    if exceeded is None and process_wall_seconds > WALL_LIMIT:
        exceeded = 'wall_time'
    return sampled_peak, exceeded, process_wall_seconds, returncode


def worker(scheme, operation, replay=False):
    import torch
    import torch.nn.functional as F
    from advar.transport import finite_volume_trajectory
    torch.set_num_threads(1)
    n, leads, substeps, spacing = 128, 18, 39, 375.0
    axis = torch.arange(n, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    q = .5 + .2 * torch.sin(x / 9) * torch.cos(y / 11)
    coarse = torch.linspace(-24000, 24000, 5, dtype=q.dtype)
    cy, cx = torch.meshgrid(coarse, coarse, indexing='ij')
    psi = -.5e-4 * (cx.square() + cy.square())
    controls = ((psi - psi[0, 0]) / 144000).flatten()[1:]
    growth = q.new_tensor(.002)
    directions = (.01*q, .001*torch.sin(torch.arange(24, dtype=q.dtype)), growth.new_tensor(.0001))

    coarse_basis = torch.eye(25, dtype=q.dtype)[1:].reshape(24, 1, 5, 5)
    basis = F.interpolate(coarse_basis, size=(n+1, n+1), mode='bilinear',
                          align_corners=True)[:, 0] * 144000
    zero = tuple(q.new_zeros(n) for _ in range(4))
    one = tuple(q.new_ones(n) for _ in range(4))
    boundary_echo = ((zero, zero),) * (leads * substeps)
    boundary_support = ((one, one),) * (leads * substeps)

    def forward(field, flow, log_growth):
        frames, _ = finite_volume_trajectory(
            field, torch.ones_like(field), flow, log_growth * substeps,
            psi_basis=basis, leads=leads, substeps_per_interval=substeps,
            interval_seconds=600., spacing_yx=(spacing, spacing),
            boundary_echo=boundary_echo, boundary_support=boundary_support,
            max_courant=.5, reconstruction=scheme, replay=replay,
        )
        return frames[1:]

    started = time.perf_counter()
    directional_score = None
    if operation == 'forward':
        with torch.no_grad():
            value = forward(q, controls, growth)
        products = (value,)
    elif operation == 'jvp':
        value, tangent = torch.func.jvp(forward, (q, controls, growth), directions)
        products = (value, tangent)
        directional_score = float(tangent.detach().mean())
    elif operation == 'vjp':
        value, pullback = torch.func.vjp(forward, q, controls, growth)
        products = (value,) + pullback(torch.ones_like(value) / value.numel())
        directional_score = float(sum((d*g.detach()).sum() for d,g in zip(directions, products[1:])))
    elif operation == 'gn':
        value, pullback = torch.func.vjp(forward, q, controls, growth)
        _, tangent = torch.func.jvp(forward, (q, controls, growth), directions)
        products = (value,) + pullback(tangent / value.numel())
    else:
        def score(*inputs):
            value = forward(*inputs)
            return .5 * value.square().mean(), value
        gradient = torch.func.grad(score, argnums=(0, 1, 2), has_aux=True)
        (gradients, value), (hessian_product, _) = torch.func.jvp(
            gradient, (q, controls, growth), directions)
        products = (value,) + gradients + hessian_product
    elapsed = time.perf_counter() - started
    assert all(bool(torch.isfinite(v).all()) for v in products)
    assert bool((value >= 0).all())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(status='passed', scheme=scheme, operation=operation, seconds=elapsed,
                peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024),
                output_shape=list(value.shape), product_bytes=sum(v.numel()*v.element_size() for v in products),
                output_sum=float(value.detach().sum()), python=platform.python_version(), torch=torch.__version__,
                machine=platform.platform(), dtype='float64', threads=1,
                steps=leads*substeps, flow_controls=24, replay=replay, directional_score=directional_score)


def run(output, replay=False, operations=('forward', 'jvp', 'vjp')):
    import os
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    report = dict(scope='18-lead transport derivative operators; public P1/retries/FSOI costs still unmeasured',
                  limits=dict(seconds=WALL_LIMIT, sampled_rss_bytes=RSS_LIMIT, polling_seconds=.25),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__).resolve(), ROOT/'src/advar/transport.py',
                                  ROOT/'src/advar/matrix_free.py')}, results=[], replay=replay,
                  operations=list(operations))
    output.parent.mkdir(parents=True, exist_ok=True)
    for scheme in ('donorcell', 'minmod'):
        for operation in operations:
            child_output = output.with_name(f'{output.stem}-{scheme}-{operation}.json')
            log = child_output.with_suffix('.log')
            started = time.perf_counter()
            try:
                child_output.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                result = _failed_result(scheme, operation,
                                        f'cannot clear prior worker result: {error}', log)
                result.update(sampled_peak_rss_bytes=0,
                              process_wall_seconds=time.perf_counter() - started)
                report['results'].append(result)
                output.write_text(json.dumps(report, indent=2)+'\n')
                print(scheme, operation, result['status'], flush=True)
                continue
            with log.open('w') as stream:
                command = [sys.executable, '-I', str(Path(__file__).resolve()),
                    '--worker', operation, '--scheme', scheme, '--output', str(child_output)]
                if replay:
                    command.append('--replay')
                process = subprocess.Popen(command,
                    env=environment, stdout=stream, stderr=subprocess.STDOUT)
                sampled_peak, exceeded, process_wall_seconds, returncode = \
                    _monitor_worker(process, started)
            if exceeded or returncode != 0:
                result = _failed_result(scheme, operation, exceeded or 'worker error', log)
            else:
                result, reason = _load_worker_result(child_output, scheme, operation, replay)
                if reason:
                    result = _failed_result(scheme, operation, reason, log)
                elif result['peak_rss_bytes'] > RSS_LIMIT:
                    result.update(status='failed', reason='reported peak exceeds RSS limit')
            result.update(sampled_peak_rss_bytes=sampled_peak,
                          process_wall_seconds=process_wall_seconds)
            report['results'].append(result)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(scheme, operation, result['status'], flush=True)
    report['adjoint_checks'] = []
    for scheme in ('donorcell', 'minmod'):
        results = {r['operation']:r for r in report['results'] if r['scheme']==scheme}
        if not {'jvp', 'vjp'} <= results.keys():
            continue
        if results['jvp']['status'] == results['vjp']['status'] == 'passed':
            lhs, rhs = (results[k]['directional_score'] for k in ('jvp', 'vjp'))
            error = abs(lhs-rhs) / max(abs(lhs), abs(rhs), 1e-30)
            report['adjoint_checks'].append(dict(scheme=scheme, relative_error=error,
                status='passed' if error < 1e-10 else 'failed'))
        else:
            report['adjoint_checks'].append(dict(scheme=scheme, status='unmeasured'))
    report['status'] = 'passed' if all(r['status']=='passed' for r in
                                     report['results']+report['adjoint_checks']) else 'failed'
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report['status'] == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('forward', 'jvp', 'vjp', 'gn', 'hvp'))
    parser.add_argument('--operations', nargs='+', choices=('forward', 'jvp', 'vjp', 'gn', 'hvp'),
                        default=('forward', 'jvp', 'vjp'))
    parser.add_argument('--scheme', choices=('donorcell', 'minmod'))
    parser.add_argument('--replay', action='store_true', help='Recompute each 39-step block during reverse AD')
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker(args.scheme, args.worker, args.replay), indent=2)+'\n')
    else:
        raise SystemExit(0 if run(args.output, args.replay, args.operations) else 1)
