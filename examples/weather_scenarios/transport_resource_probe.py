"""Bounded CPU forward/JVP/VJP measurement; not a complete D7 cost gate."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from muscl_experiment import muscl_step
from advar.transport import face_volume_fluxes


def run():
    torch.set_num_threads(1)
    size, steps, dt = 128, 3, 600 / 39
    spacing = 48000 / size
    index = torch.arange(size, dtype=torch.float64)
    y, x = torch.meshgrid(index, index, indexing='ij')
    q = .5 + .2 * torch.sin(x / 9) * torch.cos(y / 11)
    vertices = (torch.arange(size + 1, dtype=torch.float64) - size / 2) * spacing
    vy, vx = torch.meshgrid(vertices, vertices, indexing='ij')
    psi = -.5e-4 * (vx.square() + vy.square())
    growth = torch.tensor(.002, dtype=q.dtype)
    directions = (.01 * q, 10 * torch.sin(vx / 4000) * torch.cos(vy / 6000), growth.new_tensor(.01))
    cotangent = torch.sin(x / 7) * torch.cos(y / 8) / q.numel()

    def forward(echo, streamfunction, log_growth):
        qx, qy = face_volume_fluxes(streamfunction)
        for _ in range(steps):
            echo, _ = muscl_step(echo, qx, qy, dt_seconds=dt,
                                spacing_yx=(spacing, spacing), log_growth=log_growth)
        return echo

    start = time.perf_counter()
    output = forward(q, psi, growth)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, tangent = torch.func.jvp(forward, (q, psi, growth), directions)
    jvp_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, pullback = torch.func.vjp(forward, q, psi, growth)
    gradients = pullback(cotangent)
    vjp_seconds = time.perf_counter() - start
    lhs = (tangent * cotangent).sum()
    rhs = sum((d * g).sum() for d, g in zip(directions, gradients))
    relative_error = float((lhs-rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-30))
    assert bool(torch.isfinite(output).all()) and bool((output >= 0).all())
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    assert relative_error < 1e-10
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(size=size, steps=steps, dt_seconds=dt, forward_seconds=forward_seconds,
                jvp_seconds=jvp_seconds, vjp_seconds=vjp_seconds,
                peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024),
                adjoint_relative_error=relative_error, torch=torch.__version__, dtype='float64',
                python=platform.python_version(), machine=platform.machine(), system=platform.platform(),
                device='cpu', threads=1, timing='one cold call per operation; VJP includes forward tape and pullback',
                scope='Local three-step experiment only; full 18-lead P1/retry/adjoint D7 gate incomplete',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__).resolve(), HERE/'muscl_experiment.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + '\n')
