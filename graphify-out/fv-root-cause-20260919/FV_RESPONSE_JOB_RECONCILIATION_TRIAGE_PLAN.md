# R7-P fixed-case crash-artifact triage, without recovery actions

The PR #206 fixed `fv4x5` response coordinator atomically writes
`published.json` before it writes `lifecycle.json`. A parent crash can
therefore leave a publication without lifecycle, or complete raw
worker/resource files without publication. The PR #214 persistent
same-host job lock prevents simultaneous coordinators but does not
record a durable attempt or adopt an orphan worker. This increment
adds **classification only**, not recovery, retry or publication.

Expose `inspect_fixed_job_after_restart(job_dir, request_id)` under
the existing nonblocking per-job OS lock. It may create the persistent
sibling lock file but must not modify raw worker, resource,
publication, lifecycle or temporary job files. Never start a worker,
signal/reap a saved PID, or compare saved monotonic times to the
current clock (they may come from a different boot). Return one of:

- `no_job_artifacts`: no job directory or an empty one. This means
  only that no artifacts were observed, **not** proof that no orphan
  child ever existed.
- `publication_candidate`: an existing publication **and terminal
  completed lifecycle** with a complete, internally consistent
  fixed-case worker/raw/resource record and current archived-reference/
  input/worker-source checks. It remains a candidate, not an
  authenticated recovered result.
- `needs_reconciliation`: any artifacts without publication, or any
  incomplete, malformed, contradictory or unverifiable publication,
  **including a publication whose lifecycle is missing**.
  Never synthesize a publication or retry from this state.

The shared lock may raise `ResponseJobBusyError` when another caller
owns this job, or `ValueError` for a preexisting symlinked job
directory/lock file. These path precondition errors happen before
the three-state artifact classification and do not imply an empty
or valid job.

For a publication candidate, require regular non-symlink files for
`published.json`, `worker.raw.json`, `worker.resource.json` and
`lifecycle.json`.
Parse all four; require exact request id and fixed `fv4x5` case,
published raw-report SHA256, child and resource PIDs, worker command
pointing to the same resolved raw path and current worker executable,
the original full `_valid_worker` numerical/branch/adjoint/resource
gates, fixed input identity and archived reference. Cross-check
published direct/indirect/total response and true adjoint residual
against the validated child. Compare **saved** response-end,
child-completed and publication-decision monotonic values only with
one another. Require the lifecycle's exact completed/eligible/
published/archived-reference-matched statuses, identical request ID
and resource object, and ordered saved completion/publication times.
Require its `source_before == source_after ==`
the current coordinator `_sources()`, `archive_before == archive_after ==`
the current fixed `shared.ARCHIVED`, and `input_before == input_after ==`
the current `_identity()`. Missing or drifted snapshots remain
unresolved. Its absence or any contradiction is
`needs_reconciliation`.
Temporary files, missing/partial raw/resource, cancellation without
publication, worker/numerical source or current fixed-archive/input
drift, or contradictory lifecycle force `needs_reconciliation`.
When lifecycle is missing, the old publication does **not** contain
the coordinator's launch-time source snapshot or resource-report
digest, so those launch checks cannot be reconstructed. Do not infer publication authenticity
from these mutable files: they lack a signed/durable attempt manifest,
boot ID and resource-report digest.

Require the saved resource command's worker executable/script/case
and an **absolute** `--output` path resolving to this directory's
raw report. Relative paths without a persisted launch working
directory remain unresolved. A copied report in another directory
must not pass by resolving against the inspector's current cwd.

Regress with copied fixed archived fixtures and tiny fake artifacts,
no guarded FV worker: publication present/lifecycle missing must
stay unresolved, publication plus valid lifecycle is only a
candidate, raw/resource complete but no
publication, running/partial raw, cancelled/no publication,
missing/tampered raw/resource, mismatched SHA/PID/request/input/
source/value/time, contradictory lifecycle, parent-path aliases,
and two competing inspectors. The lock must make the second inspector
Busy, and repeated inspection must be idempotent and leave all job
artifacts byte-identical. Keep POSIX local-filesystem, fixed
synthetic case and preexisting-symlink limitations. R7-P durable
journal/restart/retry, generic requests, concurrent GN/refinement,
cross-host queue and whole-system resource bounds remain open.
