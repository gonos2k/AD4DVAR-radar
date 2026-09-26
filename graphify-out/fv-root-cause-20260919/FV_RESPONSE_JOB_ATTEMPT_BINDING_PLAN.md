# R7-P fixed-case attempt provenance, not automatic recovery

PR #214 excludes duplicate same-host launches and PR #215 classifies
crash artifacts conservatively. Their fixed `fv4x5` publication does
not contain a unique attempt identity or a resource-report digest,
and a crash before final lifecycle leaves no coordinator snapshot.
This increment binds **new** fixed-case attempts and their eventual
publication to an immutable pre-spawn manifest. It does not adopt an
orphan, retry, synthesize a response, reconcile cancellation after a
crash or provide a production queue.

Under the existing POSIX same-job lock and only for a fresh empty
job directory, require an already existing stable parent chain and
canonicalize the job directory and worker output path
to absolute paths. Create `attempt.json` schema 1 with UUID4
`attempt_id`, exact safe `request_id`, `case_id=fv4x5`, canonical job
path, UTC creation timestamp for audit only, exact worker command
including the attempt ID, current coordinator/worker source map,
archive hashes, fixed input identity, archived expected response,
plan hash and 300-second / sampled 768-MiB child budget. The manifest
is immutable; do not update its state in place. After creating the
fresh job directory, `fsync` its existing parent directory so the
new directory entry is persisted; failure prevents child launch.
Commit the manifest before
spawning: write a temporary file, flush and `fsync` the file,
`os.replace` to `attempt.json`, then `fsync` the job directory.
Read the resulting regular file through one stable no-symlink
descriptor, require its bytes to equal the exact canonical JSON
bytes intended for the manifest, and pin that digest before spawn.
Failure of any of these steps prevents child launch; a visible but
not fully committed manifest is an unresolved attempt, not proof of
a run.

Use stable field names: manifest `source_sha256`, `archive_sha256`,
`input_identity`, `expected_response_total`, and
`resource_budget={wall_seconds,rss_limit_bytes,sampling_seconds,scope}`;
resource report `attempt_id` and `attempt_manifest_sha256`;
publication `publication_schema_version=2`, `attempt_id`,
`attempt_manifest_sha256`, `worker_report_sha256`,
`resource_report_sha256`, canonical `job_dir`, `source_sha256`,
`archive_sha256`, plus the existing `input_identity` and response
fields. The final lifecycle record also stores `attempt_id`.

The worker gains an **optional** `--attempt-id` argument and echoes it
in running and final raw reports for new jobs. Legacy worker callers
without the option keep their response equations/output contract.
The attempt ID is provenance metadata only: it must not enter the
objective, model state, control, branch decisions or AD graph.
After `run_guarded` has reaped the child, add the same attempt ID and
manifest digest to the resource report, durably write it, flush/fsync
the completed worker raw and resource files and the directory, and
require the returned resource object to match the persisted report.
Parse and hash worker raw and enriched resource reports from the
same stable byte snapshots that were fsynced; do not validate one
read and later hash different bytes. Recheck the pinned manifest,
raw and resource digests under the cancellation/publication lock
immediately before publication. A mismatch suppresses publication
and marks numerical eligibility unverified.
Only then run the existing full source/input/archive/branch/adjoint/
resource gates. A missing/mismatched ID or durability error is not
numerical eligibility and cannot publish.

The new publication schema 2 retains all current response fields and
adds attempt ID, exact attempt-manifest SHA256, raw-worker SHA256,
resource-report SHA256, canonical job path, coordinator source and
archive/input snapshots. Under the existing cancellation token lock,
write publication by temporary-file fsync, atomic replace and
directory fsync **before** declaring the publication committed.
`lifecycle.json` follows and records the attempt ID and exact resource
object. A cancellation accepted before the publication lock decision
still wins; after a committed publication, cancellation cannot
retract it. If a durability operation fails, never report a completed
publication; any visible but uncertain file stays unresolved for
manual reconciliation. Do not use a saved PID or boot-relative
monotonic time to adopt or signal a process.

Extend the PR #215 read-only classifier without weakening legacy
handling. **Any** v2 marker in any artifact—`attempt.json`,
`attempt_id`, manifest/resource digest, or schema version `>=2`—requires the
manifest/worker/resource/publication/lifecycle ID and digest chain,
canonical job/command/budget, current source/archive/input checks,
and existing numerical gates. Missing or mismatched v2 proof means
`needs_reconciliation`, **never a fallback to legacy candidate**.
Coordinated removal of every marker cannot be detected without an
authenticated external record and remains outside this claim.
Legacy PR #206 artifacts remain unmodified and use the prior
conservative rules, including relative-output and coordinator-source
drift refusal. A fully bound v2 record can still be only an
unauthenticated `publication_candidate`: all files are mutable and
unsigned, and even file+directory fsync on POSIX local storage does
not prove cryptographic authenticity or guarantee every power-loss
scenario.

Regress without a guarded FV worker using fake guard/raw fixtures:
parent-directory fsync and manifest committed before spawn; failure
of parent fsync, temp write, file fsync, manifest byte mismatch,
replace or directory fsync blocks launch; worker and resource ID
echo; v2 publication digests and ordering; cancellation before and
during locked publication; failure after replace before fsync leaves
no completed status; missing/tampered/copied manifest/raw/resource/
publication/lifecycle and source/input/command drift fail closed;
v2 cannot downgrade to legacy; old v1 remains conservative. Preserve
PR #206/#214/#215 raw evidence, run affected tests and typecheck,
refresh Graphify, and record GREEN/RED review. R7-P durable restart,
orphan-worker handling, cross-host queue, generic inputs, concurrent
GN/refinement and whole-system resource bounds remain open.
