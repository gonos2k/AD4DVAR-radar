"""Package exact run bytes, retain pinned source bundle, and verify all members."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
w=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
files={p.relative_to(w).as_posix():p.read_bytes() for dirname in ('attempt1','observation1') for p in (w/dirname).rglob('*') if p.is_file()}
with tarfile.open(w/'receipts.tar.gz','w:gz') as archive:
    for name,data in sorted(files.items()):
        info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;archive.addfile(info,io.BytesIO(data))
with tarfile.open(w/'receipts.tar.gz','r:gz') as archive:
    restored={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
assert restored==files
source=json.loads((w/'PLAN.json').read_text())['source_files']
with tarfile.open(w/'source.tar.gz','r:gz') as archive:
    members={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
assert set(members)==set(source) and all(sha(members[n])==h for n,h in source.items())
manifest={'bundle_sha256':sha((w/'receipts.tar.gz').read_bytes()),'source_bundle_sha256':sha((w/'source.tar.gz').read_bytes()),
    'files':{n:{'bytes':len(b),'sha256':sha(b)} for n,b in sorted(files.items())},
    'restore':'Extract receipts.tar.gz into this directory to restore exact loose attempt1 and observation1 receipts; never rerun to reconstruct them.'}
(w/'RECEIPT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'verified_receipt_members':len(files),'receipt_bundle_sha256':manifest['bundle_sha256'],'verified_source_members':len(source)},indent=2))
