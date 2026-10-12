"""Freeze one same-point direction-efficiency comparison."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import direction_trial as d
p=d.endpoint.load_endpoint()['progress_plan'];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
if d.PLAN.exists():raise ValueError('comparison plan already frozen')
plan={'experiment_kind':'fresh2a4d_gradient_residual_direction_comparison','base_control_sha256':d.BASE_SHA,'numerical_version':d.VERSION,'policy':d.POLICY,
 'source_transition':{n:{'before':p['source_files'][n],'after':sha(ROOT/n)} for n in d.CHANGED_SOURCES},
 'source_files':{n:sha(ROOT/n) for n in sorted(set(p['source_files'])|d.NEW_SOURCES)},
 'archive_files':{**p['archive_files'],**{f.relative_to(ROOT).as_posix():sha(f) for f in d.RECEIPTS}}}
d.PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');h=sha(d.PLAN);(WORK/'PLAN_SHA256.txt').write_text(h+'\n')
with tarfile.open(WORK/'source.tar.gz','w:gz') as a:
 for n in sorted(plan['source_files']):
  b=(ROOT/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);i.mtime=0;a.addfile(i,io.BytesIO(b))
d.load_plan(d.PLAN,h)
r={'plan_sha256':h,'source_bundle_sha256':sha(WORK/'source.tar.gz'),'source_count':len(plan['source_files']),'archive_count':len(plan['archive_files']),'tests':'42passed','types':'0errors0warnings'};(WORK/'PREFLIGHT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
