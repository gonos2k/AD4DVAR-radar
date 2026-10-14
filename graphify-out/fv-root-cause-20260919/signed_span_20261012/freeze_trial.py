"""Freeze the bounded same-point signed comparison before any FV evaluation."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))
import signed_trial as trial

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if trial.PLAN.exists():
    raise ValueError("signed comparison plan already frozen")
parent = json.loads(trial.PARENT_PLAN.read_text())
plan = {
    "experiment_kind": trial.POLICY["experiment_kind"],
    "base_control_sha256": trial.BASE_SHA,
    "numerical_version": trial.VERSION,
    "policy": trial.POLICY,
    "parent_plan_sha256": trial.PARENT_PLAN_SHA,
    "parent_source_bundle_sha256": trial.PARENT_SOURCE_TAR_SHA,
    "parent_receipt_bundle_sha256": trial.PARENT_RECEIPTS_TAR_SHA,
    "stored_model_audit_sha256": sha(WORK / "AUDIT.json"),
    "source_transition": {name: {"before": parent["source_files"][name],
        "after": sha(ROOT / name)} for name in sorted(trial.CHANGED_SOURCES)},
    "source_files": {name: sha(ROOT / name)
        for name in sorted(set(parent["source_files"]) | trial.NEW_SOURCES)},
    "archive_files": {**parent["archive_files"],
        **{name: sha(ROOT / name) for name in sorted(trial.NEW_ARCHIVES)}},
}
trial.PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
digest = sha(trial.PLAN)
(WORK / "PLAN_SHA256.txt").write_text(digest + "\n")
with tarfile.open(WORK / "source.tar.gz", "w:gz") as archive:
    for name in sorted(plan["source_files"]):
        data = (ROOT / name).read_bytes()
        member = tarfile.TarInfo(name)
        member.size, member.mtime = len(data), 0
        archive.addfile(member, io.BytesIO(data))
trial.load_plan(trial.PLAN, digest)
preflight = {"plan_sha256": digest,
    "source_bundle_sha256": sha(WORK / "source.tar.gz"),
    "source_count": len(plan["source_files"]),
    "archive_count": len(plan["archive_files"]),
    "new_HVP_cap": 0, "cached_HVP_count": 4,
    "same_radius": trial.RADIUS, "source_receipt_admission": True,
    "prior_guarded_seconds_charged": trial.PRIOR_GUARDED_SECONDS}
(WORK / "PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n")
print(json.dumps(preflight))
