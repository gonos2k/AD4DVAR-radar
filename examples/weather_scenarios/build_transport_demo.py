"""Replay the frozen solver and embed its fields in a standalone teaching page."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affine_holdout as holdout
from evaluate_holdout import evaluate_files, digest
from build_demo import _atomic_write

ROOT = HERE.parents[1]
CHECKPOINT = ROOT / 'graphify-out/scenario-implementation-20260909/checkpoints'
TEXT = {
 'reverse_oblique_translation': ('비스듬한 병진', '방향과 세기는 그대로, 위치만 변하는 흐름입니다.',
    '비음수성과 보존성을 만족해도 반복 수송의 수치 확산은 에코를 넓힐 수 있습니다.'),
 'reverse_rotation': ('반대 방향 회전', '공간에 따라 다른 속도로 타원형 에코를 회전시킵니다.',
    '주어진 회전장을 수송할 수 있다는 사실은 관측에서 회전 속도를 추정했다는 뜻이 아닙니다.'),
 'shear': ('전단 흐름', '높이 좌표에 따라 수평 속도가 달라져 에코가 기울어집니다.',
    '전역 평행 이동만으로는 이런 모양 변화를 표현할 수 없습니다.'),
 'strain': ('늘어남과 압축', '한 방향으로 늘어나고 다른 방향으로 줄어드는 면적 보존 흐름입니다.',
    '셀별 값의 변화와 전체 에코 적분의 변화를 구분하세요. 성장항도 함께 작용합니다.'),
 'rotation_strain': ('회전과 변형', '회전과 신장이 동시에 작용하는 혼합 흐름입니다.',
    '회전 방향뿐 아니라 폭과 위치 오차를 함께 보아야 형상 보존을 판단할 수 있습니다.'),
}


def display_field(field: torch.Tensor) -> list[float]:
    # Only visualization is coarsened; metrics retain the original cell averages.
    n = field.shape[0]
    reduced = field.reshape(n // 2, 2, n // 2, 2).mean(dim=(1, 3))
    return [round(float(v), 6) for v in reduced.flatten()]


def boundary_examples():
    """One-cell SSPRK2 oracle: response to a constant inflow is (dt-dt²/2)b."""
    q = torch.zeros((1, 1), dtype=torch.float64)
    qx, qy = torch.ones((1, 2), dtype=q.dtype), torch.zeros((2, 1), dtype=q.dtype)
    rows = []
    for name, amplitude, known in [('미지 유입', 0., 0.), ('알려진 맑음', 0., 1.),
                                   ('부분적으로 알려진 유입', 3.5, .5)]:
        edges = tuple(torch.full((1,), amplitude, dtype=q.dtype) for _ in range(4))
        support = tuple(torch.full((1,), known, dtype=q.dtype) for _ in range(4))
        result = holdout.transport.finite_volume_step(
            q, q, qx, qy, dt_seconds=.1, spacing_yx=(1., 1.),
            boundary_echo=(edges, edges), boundary_support=(support, support))
        expected_echo, expected_support = .095 * amplitude, .095 * known
        torch.testing.assert_close(result.echo, torch.full_like(q, expected_echo), atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(result.support, torch.full_like(q, expected_support), atol=1e-14, rtol=1e-13)
        rows.append(dict(name=name, echo=float(result.echo), support=float(result.support),
                         expected_echo=expected_echo, expected_support=expected_support))
    return rows


def replay(case, manifest, scheme):
    module = holdout.transport if scheme == 'donorcell' else holdout.muscl_experiment
    name = 'finite_volume_step' if scheme == 'donorcell' else 'muscl_step'
    original = getattr(module, name)
    spacing = manifest['domain_side_m'] / manifest['grid_size']
    count = math.ceil(manifest['lead_interval_seconds'] * 4 * manifest['speed_bound_mps']
                      / spacing / manifest['max_courant'])
    frames, calls = [], 0

    def capture(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls % count == 0:
            frames.append((result.echo if scheme == 'donorcell' else result[0]).clone())
        return result

    setattr(module, name, capture)
    try:
        result = holdout._run_case(case, manifest, scheme)
    finally:
        setattr(module, name, original)
    if len(frames) != manifest['lead_count']:
        raise ValueError('replay missed a lead')
    return frames, result


def build_dataset():
    source = CHECKPOINT / 'CP3b_holdout.json'
    manifest_path = CHECKPOINT / 'CP3b_manifest.json'
    acceptance = evaluate_files(source, manifest_path)
    if acceptance['status'] != 'passed':
        raise ValueError('saved experiment has not passed candidate gates')
    manifest, saved = json.loads(manifest_path.read_text()), json.loads(source.read_text())
    cases, comparisons = [], []
    color_max, error_max = 0.0, 0.0
    with torch.no_grad():
        for case in manifest['cases']:
            fields, metrics = {}, {}
            initial = holdout.affine_cell_averages(case, 128, manifest['domain_side_m'], 0)
            for scheme in ('donorcell', 'minmod'):
                values, result = replay(case, manifest, scheme)
                previous = next(r for r in saved['results'] if r['case'] == case['name'] and r['scheme'] == scheme)
                # The captured solver must reproduce all saved scientific metrics.
                if result['leads'] != previous['leads']:
                    raise ValueError(f"replay differs from checkpoint: {case['name']}/{scheme}")
                comparisons.append(f"{case['name']}/{scheme}: all lead metrics identical")
                fields[scheme] = [initial] + values
                zero = dict(previous['leads'][0])
                angle, covered, anisotropy, _, _ = holdout._orientation_error(
                    initial, initial, manifest['thresholds']['truth_anisotropy_minimum'], manifest['domain_side_m'] / 128)
                zero.update(lead_minutes=0.0, relative_echo_l2=0.0, axis_width_relative_error=[0.0, 0.0],
                            principal_width_relative_error=[0.0, 0.0], orientation_error_degrees=angle,
                            orientation_covered=covered, truth_anisotropy=anisotropy,
                            center_error_pixels=0.0, center_error_m=0.0, minimum_echo=float(initial.min()),
                            transformed_outflow=0.0, transformed_budget_residual=0.0, lead_transformed_budget_residual=0.0)
                metrics[scheme] = [zero] + result['leads']
            frames = []
            for index in range(manifest['lead_count'] + 1):
                minute = index * manifest['lead_interval_seconds'] / 60
                truth = holdout.affine_cell_averages(case, 128, manifest['domain_side_m'], minute * 60)
                color_max = max(color_max, float(truth.max()), *(float(fields[s][index].max()) for s in fields))
                error_max = max(error_max, float((fields['minmod'][index] - truth).abs().max()))
                frames.append(dict(minute=minute, truth=display_field(truth),
                                   donorcell=display_field(fields['donorcell'][index]),
                                   minmod=display_field(fields['minmod'][index]),
                                   metrics={s: metrics[s][index] for s in metrics}))
            title, description, lesson = TEXT[case['name']]
            A, b = case['A_per_second'], case['b_mps']
            equation = f"u(x)=A x+b; A={A} s⁻¹, b={b} m s⁻¹; γ={case['growth_per_second']:.6g} s⁻¹"
            cases.append(dict(id=case['name'], title=title, description=description,
                              lesson=lesson, equation=equation, frames=frames))
            print(case['name'], '19 real solver frames captured', flush=True)
    sources = dict(saved['source_sha256'])
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    payload = dict(meta=dict(grid_size=64, computation_grid_size=128, domain_side_m=48000,
                 color_max=color_max, error_max=error_max, generated_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=sources, scope='Prescribed-flow transport only; no inferred motion/P1/FSO/learning',
                 acceptance_status=acceptance['status'], checkpoint_sha256=digest(source),
                 boundary_examples=boundary_examples(),
                 integration_status='새 수송 후보는 P1·FSO/FSOI와 미연결입니다. 기존 P1 구현의 존재와 후보 연결 완료는 다릅니다.'), cases=cases)
    return payload, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'transport.html')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--resource', type=Path)
    parser.add_argument('--learning', type=Path)
    args = parser.parse_args()
    template = (HERE / 'transport_template.html').read_text()
    if template.count('__TRANSPORT_DATA__') != 1:
        raise ValueError('expected exactly one data placeholder')
    torch.set_num_threads(1)
    payload, comparisons = build_dataset()
    if args.resource:
        resource_result = json.loads(args.resource.read_text())
        if any(digest(Path(path)) != sha for path, sha in resource_result['source_sha256'].items()):
            raise ValueError('resource probe source changed after measurement')
        payload['meta']['resource_probe'] = resource_result
    if args.learning:
        learning_result = json.loads(args.learning.read_text())
        if any(digest(Path(path)) != sha for path, sha in learning_result['source_sha256'].items()):
            raise ValueError('P1 probe source changed after execution')
        payload['meta']['existing_p1_learning'] = learning_result
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    _atomic_write(args.output, template.replace('__TRANSPORT_DATA__', encoded))
    evidence = dict(status='passed', cases=5, frames_per_case=19, comparisons=comparisons,
                    html_sha256=digest(args.output), template_sha256=digest(HERE / 'transport_template.html'),
                    builder_sha256=digest(Path(__file__)), meta=payload['meta'])
    _atomic_write(args.evidence, json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
