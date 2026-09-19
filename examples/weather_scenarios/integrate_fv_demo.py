"""Attach independently computed P0/FV cases to the original demo interface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from build_demo import _atomic_write, render_template
from motion_diagnostics import motion_diagnostics


def integrate(base_html: str, fv_data: dict) -> dict:
    match = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', base_html, re.S)
    if match is None:
        raise ValueError("base HTML must contain the original demo dataset")
    payload = json.loads(match.group(1))
    # Rebuilding replaces only this integration's entries; historical cases stay intact.
    payload['scenarios'] = [case for case in payload['scenarios'] if not case.get('fv_comparison_entry')]
    for case in fv_data['cases']:
        paired = []
        for p0, fv in zip(case['methods']['p0']['metrics'], case['methods']['fv']['metrics'], strict=True):
            if p0['domain_pixels'] != fv['domain_pixels'] or p0.get('lead_minutes') != fv.get('lead_minutes'):
                raise ValueError('P0 and FV must use the same domain and lead times')
            paired.append(dict(domain_pixels=p0['domain_pixels'], p0_mae=p0['mae'], fv_mae=fv['mae'],
                               p0_missing=p0['missing_pixels'], fv_missing=fv['missing_pixels']))
        common = {key: value for key, value in case.items() if key != 'methods'}
        fields = case.get('observations', []) + case.get('truth', [])
        for result in case['methods'].values():
            fields += result.get('forecast') or []
        values = [v for field in fields for row in field for v in row if v is not None and math.isfinite(v)]
        color_range = ({'color_min_dbz': math.floor(min(values)), 'color_max_dbz': max(math.floor(min(values)) + 1, math.ceil(max(values)))} if values else {})
        for method in ('p0', 'fv'):
            result = case['methods'][method]
            entry = {**common, **result}
            entry.update(id=f"{case['id']}_{method}", name=f"{case['name']} · {method.upper()}",
                         fv_comparison_entry=True, comparison=paired,
                         method_label='FV 공동 자료동화 · 조건부 합성' if method == 'fv' else 'P0 · 동일 합성 입력',
                         meta={**fv_data.get('meta', {}), **case.get('meta', {}), **color_range,
                               'model': '공유 FV 변분 분석 → FV 예측' if method == 'fv' else 'P0 nowcast · 동일 관측'})
            meta = entry['meta']
            if case.get('observations') and all(key in meta for key in ('spacing_m', 'background_echo', 'min_dbz')):
                def tensor(grid):
                    return torch.tensor([[float('nan') if v is None else v for v in row] for row in grid], dtype=torch.float64)
                reference = tensor(case['observations'][-1])
                parameters = dict(spacing_m=meta['spacing_m'], background_echo=meta['background_echo'], min_dbz=meta['min_dbz'])
                entry['field_motion'] = [
                    {'truth': motion_diagnostics(reference, tensor(truth), **parameters),
                     'forecast': motion_diagnostics(reference, tensor(forecast), **parameters)}
                    for truth, forecast in zip(case['truth'], result['forecast'], strict=True)
                ]
            payload['scenarios'].append(entry)
    if fv_data['cases']:
        payload['initial_scenario_id'] = fv_data['cases'][0]['id'] + '_fv'
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-html', type=Path, default=Path(__file__).with_name('index.html'))
    parser.add_argument('--fv-json', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('index.html'))
    args = parser.parse_args()
    cases = []
    for path in args.fv_json:
        data = json.loads(path.read_text())
        cases.extend({**case, 'meta': {**data.get('meta', {}), **case.get('meta', {})}} for case in data['cases'])
    payload = integrate(args.base_html.read_text(), {'cases': cases})
    _atomic_write(args.output, render_template(Path(__file__).with_name('template.html').read_text(), payload))


if __name__ == '__main__':
    main()
