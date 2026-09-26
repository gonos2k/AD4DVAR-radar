# R7-P fixed response-job crash-artifact triage

The fixed `fv4x5` lifecycle writes `published.json` before its final
`lifecycle.json`. A parent crash can leave either an incomplete
published job or complete-looking worker files with no publication.
This follow-up adds `inspect_fixed_job_after_restart` to classify
those artifacts **without launching, adopting, signalling or retrying
a worker** and without creating or rewriting response, raw, resource
or lifecycle files. It uses the same nonblocking same-host job lock
as PR #214; acquiring that lock may create its persistent sibling
lock file.

| State | Meaning | Action |
|---|---|---|
| `no_job_artifacts` | Job directory missing or empty | No worker-absence claim |
| `publication_candidate` | Publication, worker, resource and terminal completed lifecycle are internally consistent with the current fixed case | Inspect only; not authenticated or served as recovered success |
| `needs_reconciliation` | Missing publication or lifecycle, partial/contradictory files, or any failed evidence gate | No auto publication or retry |

For a candidate, the inspector requires four regular non-symlink
JSON files; the existing full worker branch, stationarity, adjoint
and resource validator; current fixed archive/input/worker and
coordinator source identities; exact published request, fixed case,
scope, worker-report SHA256, PID and response scalars; and an absolute
worker output path resolving to this directory's raw report. The
terminal lifecycle must say completed, eligible, published and
archived-reference matched with no runner/read error or cancellation,
the same resource record, and current source/archive/input snapshots.
Saved monotonic values are compared **only with each other**, never
with the new process's clock or an inferred boot. A saved PID is
compared across records but is never signalled or used to adopt a
process.

The old PR #206 fixed-case publication has a relative worker output
path and an older coordinator source snapshot. A copy of those raw
files is correctly classified `needs_reconciliation` until a test
fixture explicitly supplies a current, absolute command and
self-consistent current lifecycle metadata. That fixture then yields
only a non-authoritative candidate. Publication without lifecycle,
raw/resource without publication, cancellation without publication,
partial JSON, unknown/temp files, nonregular/symlink records,
tampered report hashes/PID/request/case/scope/input/response/time,
resource-exit changes and current archive/source drift all remain
unresolved. The `False == 0.0` JSON/Python comparison case and NaN
response values are rejected before equality checks.

The affected classifier/lifecycle/process suite passed **69 tests**;
pinned basedpyright 1.39.9 reported zero diagnostics. A code-only
Graphify refresh recorded 7,807 nodes, 67,341 edges and 267
communities. The tests use copied archived/fake records and existing
small fixed-case checks; **no new guarded FV response worker** was
run. GREEN and RED reviewed the state contract and tampering
counterexamples. Full CPU/package regression was not run.

This is **triage, not durable recovery**. The existing publication
schema lacks an authenticated attempt manifest, resource-report
digest, boot/process-start identity and durable fsync protocol. A
coordinated rewrite of mutable files could still form an internally
consistent candidate, so `candidate_is_authoritative=false` is
mandatory. A parent crash may leave an orphan worker after the lock
is released; the inspector never assumes the worker is gone. Busy
jobs and preexisting symlinked job/lock paths raise precondition
errors before classification. POSIX local-filesystem locking and
preexisting-symlink checks retain their documented limits. R7-P
remains open for an actual durable journal, safe restart/cancellation
reconciliation, generic requests, concurrent GN/refinement and
whole-system resources.
