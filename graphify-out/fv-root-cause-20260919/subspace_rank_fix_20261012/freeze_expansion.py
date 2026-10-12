"""Freeze the single bounded expansion after source/test review; no model calls."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[3];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK))
import radius_expansion_trial as expansion
parent=expansion.load_endpoint()['parent_plan'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if expansion.PLAN.exists():raise ValueError('expansion plan already frozen')
plan={'experiment_kind':'fresh_7839_KKT_bounded_radius_comparison','base_control_sha256':expansion.BASE_SHA,
    'numerical_version':expansion.VERSION,'policy':expansion.POLICY,
    'source_transition':{n:{'before':parent['source_files'][n],'after':sha(ROOT/n)} for n in expansion.CHANGED_SOURCES},
    'source_files':{n:sha(ROOT/n) for n in sorted(set(parent['source_files'])|expansion.NEW_SOURCES)},
    'archive_files':{**parent['archive_files'],**{p.relative_to(ROOT).as_posix():sha(p) for p in expansion.RECEIPTS}}}
expansion.PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');digest=sha(expansion.PLAN)
(WORK/'EXPANSION_PLAN_SHA256.txt').write_text(digest+'\n')
with tarfile.open(WORK/'source.tar.gz','w:gz') as archive:
    for name in sorted(plan['source_files']):
        data=(ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;archive.addfile(info,io.BytesIO(data))
expansion.load_plan(expansion.PLAN,digest)
result={'plan_sha256':digest,'source_bundle_sha256':sha(WORK/'source.tar.gz'),'source_member_count':len(plan['source_files']),
    'archive_pin_count':len(plan['archive_files']),'policy':expansion.POLICY}
(WORK/'EXPANSION_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
