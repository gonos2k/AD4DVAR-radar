"""Verify the locally produced 240x240 analysis without rerunning assimilation."""
import hashlib, json, time
from pathlib import Path
import torch
from advar.fv_sensitivity import verify_fv_stationarity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
start=time.monotonic()
# Trusted checkpoint produced by this session's builder; contains local dataclasses.
saved=torch.load(HERE/'rotation240_analysis.pt',weights_only=False)
checked,evidence=verify_fv_stationarity(saved['result'],saved['observations'],saved['frozen'])
evidence.update(elapsed_seconds=time.monotonic()-start,
                source_sha256=hashlib.sha256((ROOT/'src/advar/variational.py').read_bytes()).hexdigest(),
                verification_source_sha256=hashlib.sha256((ROOT/'src/advar/fv_sensitivity.py').read_bytes()).hexdigest(),
                stationarity_verified=checked.stationarity_verified,reason=checked.reason)
(HERE/'rotation240_stationarity.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
