"""Refine a saved large analysis into a separate checkpoint."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
from advar.fv_sensitivity import refine_fv_stationarity
from advar.variational import robust_objective

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=HERE/'rotation240_refined.pt')
parser.add_argument('--prefix', default='rotation240_stable')
args = parser.parse_args()
output = HERE/f'{args.prefix}_refined.pt'
if output.resolve() == args.input.resolve():
    raise ValueError('use a separate output to preserve the input checkpoint')
logging.basicConfig(level=logging.INFO)
torch.set_num_threads(1)
saved = torch.load(args.input, weights_only=False)
c, records = refine_fv_stationarity(
    saved['control'], saved['observations'], saved['frozen'], gradient_tolerance=1e-9,
)
g = torch.func.grad(robust_objective)(c, saved['observations'], saved['frozen'])
torch.save({'control': c, 'observations': saved['observations'],
            'frozen': saved['frozen']}, output)
report = dict(
    scope='240x240 local matrix-free gradient-root refinement',
    gradient_max_tolerance=1e-9, gradient_norm=float(g.norm()),
    gradient_max=float(g.abs().max()),
    control_change_norm=float((c-saved['control']).norm()), steps=records,
    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__), ROOT/'src/advar/fv_sensitivity.py',
                             ROOT/'src/advar/transport.py', ROOT/'src/advar/variational.py')},
)
(HERE/f'{args.prefix}_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
