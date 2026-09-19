"""Compare existing FV response modes with the measured ce6e36a implementation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from advar.fv_sensitivity import compute_fv_observation_response

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = 'ce6e36a2dcfb2f823647e7cabb5369dca3c9f200'
torch.set_num_threads(1)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run():
    original = subprocess.check_output(['git', 'show', f'{BASELINE}:src/advar/fv_sensitivity.py'], cwd=ROOT)
    fixture = load('fv_response_fixture', ROOT/'tests/test_fv_observation_response.py')
    obs, frozen, control, boundary, support = fixture._refined_case()
    records = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'baseline.py'
        path.write_bytes(original)
        baseline = load('advar._fv_response_baseline', path)
        for dependency in ('frozen', 'first_observation'):
            for curvature in ('irls_gauss_newton', 'exact_robust_hessian'):
                kwargs = fixture._response_kwargs(obs, boundary, support)
                kwargs.update(background_dependency=dependency, curvature=curvature)
                before = baseline.compute_fv_observation_response(control, obs, frozen, **kwargs)
                after = compute_fv_observation_response(control, obs, frozen, **kwargs)
                assert after.sensitivity_theta is None
                assert torch.equal(before.sensitivity_dbz, after.sensitivity_dbz)
                for name in ('score', 'gradient_max', 'normal_products', 'adjoint_relative_residual', 'face_margin', 'curvature'):
                    assert getattr(before, name) == getattr(after, name), name
                records.append(dict(dependency=dependency, curvature=curvature,
                                    sensitivity_max_difference=0.0, all_existing_fields_equal=True))
    report = dict(scope='4x5 exact before/after comparison; not a repeated 240-grid run',
                  measured_revision=BASELINE, baseline_source_sha256=hashlib.sha256(original).hexdigest(),
                  current_source_sha256=hashlib.sha256((ROOT/'src/advar/fv_sensitivity.py').read_bytes()).hexdigest(),
                  records=records)
    (HERE/'parameterized_default_continuity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
