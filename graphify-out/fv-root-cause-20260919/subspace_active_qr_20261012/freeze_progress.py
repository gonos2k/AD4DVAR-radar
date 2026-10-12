"""Freeze one bounded original-J/R progress plan; no physical calls."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import cached_gradient_progress as p
parent=p.load_endpoint()['parent_plan'];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
if p.PLAN.exists():raise ValueError('progress plan already frozen')
plan={'experiment_kind':'cached7839_original_JR_bounded_continuation','numerical_version':p.VERSION,
 'base_control_sha256':p.BASE_SHA,'policy':p.POLICY,
 'source_transition':{n:{'before':parent['source_files'][n],'after':sha(ROOT/n)} for n in p.CHANGED_SOURCES},
 'source_files':{n:sha(ROOT/n) for n in sorted(set(parent['source_files'])|p.NEW_SOURCES)},
 'archive_files':{**parent['archive_files'],**{f.relative_to(ROOT).as_posix():sha(f) for f in p.RECEIPTS}}}
p.PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');digest=sha(p.PLAN)
(WORK/'PROGRESS_PLAN_SHA256.txt').write_text(digest+'\n')
with tarfile.open(WORK/'source.tar.gz','w:gz') as ar:
 for n in sorted(plan['source_files']):
  b=(ROOT/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(b);info.mtime=0;ar.addfile(info,io.BytesIO(b))
p.load_plan(p.PLAN,digest)
f={'plan_sha256':digest,'source_bundle_sha256':sha(WORK/'source.tar.gz'),'source_count':len(plan['source_files']),
 'archive_count':len(plan['archive_files']),'tests':'54pass2legacyimmutable-source ancestry refusals;27QR/rankpass','policy':p.POLICY}
(WORK/'PREFLIGHT.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps(f))
