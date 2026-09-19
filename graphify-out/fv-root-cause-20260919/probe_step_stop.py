"""Controlled local experiment; no production settings or gates are modified."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples/weather_scenarios'))
import fv_original_cases as demo

config = demo.NowcastConfig(interval_minutes=10, horizon_minutes=20, min_dbz=-10, max_dbz=70)
original_config = demo.AnalysisConfig
rows = []
for step_tolerance in (1e-5, 1e-10):
    def analysis_config(**kwargs):
        return original_config(**{**kwargs, 'step_tolerance': step_tolerance})
    demo.AnalysisConfig = analysis_config
    start = time.perf_counter()
    result = demo._case_payload(demo.CASE_SPECS[1], config, 12)['methods']['fv']
    rows.append({'step_tolerance': step_tolerance, 'seconds': time.perf_counter()-start,
                 'state': result['state'], 'solver': result['solver'], 'metrics': result['metrics']})
demo.AnalysisConfig = original_config
paths = [Path(demo.__file__), ROOT/'src/advar/variational.py', ROOT/'src/advar/transport.py', Path(__file__)]
payload = {'scope': 'same12x12 rotation, only step_tolerance changes; cap12, other tolerances fixed',
           'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'runs': rows}
Path(__file__).with_suffix('.json').write_text(json.dumps(payload, indent=2)+'\n')
for row in rows:
    print(row['step_tolerance'], row['state'], flush=True)
