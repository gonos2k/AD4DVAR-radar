"""Original P2 reproduction from the pinned PR301 source (no FV execution)."""
from pathlib import Path
import io
import json
import sys
import tarfile
import types
import hashlib
import torch
ROOT=Path(__file__).resolve().parents[3]
old=ROOT/'graphify-out/fv-root-cause-20260919/netflux_subspace_20261012'
name='examples/weather_scenarios/fv_point_3h_subspace_model.py'
plan=json.loads((old/'PLAN.json').read_text())
with tarfile.open(old/'source.tar.gz','r:gz') as ar:source=ar.extractfile(name).read()
assert hashlib.sha256(source).hexdigest()==plan['source_files'][name]
mod=types.ModuleType('original_pr301_solver');sys.modules[mod.__name__]=mod
exec(compile(source,name,'exec'),mod.__dict__)
try:
    mod.solve_two_direction_subspace(torch.tensor([-1.,-1.],dtype=torch.float64),torch.eye(2,dtype=torch.float64),torch.ones(2,2,dtype=torch.float64),.8)
except torch.linalg.LinAlgError as error:
    result={'reproduced':True,'error_type':type(error).__name__,'error':str(error),'torch_version':torch.__version__,'source_sha256':plan['source_files'][name],'scope':'pinned PR301 solver-only; no production FV/HVP/score calls'}
else:raise AssertionError('pinned PR301 defect did not reproduce')
p=Path(__file__).parent/'REPRODUCE_BEFORE.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
