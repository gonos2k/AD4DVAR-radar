# Two process-isolated FV response jobs: one published, one cancelled

This is the bounded R7 lifecycle run declared in
`FV_RESPONSE_JOB_LIFECYCLE_PLAN.md`. It reuses the archived qualified
4×5 conditional minmod local response and its existing separate-process
worker. Two job directories were launched concurrently. One worker was
allowed to finish; the other was cancelled **after a complete, parseable
raw JSON report** said `status=running`, `phase=response` and gave a
positive PID. The work remains a fixed synthetic research case, not a
general radar service or in-process concurrent forward-AD test.

The finishing worker had PID `14273`, exit code 0 and no resource
termination. Its 54 observed minmod stages, fixed branch signatures,
control stationarity, PCG/HVP intervals and true-adjoint residual passed
the parent's existing source/archive/input and numerical gates. Its
directional total response was `-0.003038272528696361`, matching the
archived reference; the true transposed-adjoint relative residual was
`9.888163804832049e-11`, below the retained `1e-10` gate. The
worker response ended at monotonic time `579713.482664333`; the child
was reaped at `579713.970228208`, the token-locked publication decision
was `579714.078692`, and the atomic publication completed at
`579714.079100583`. Only after these checks did `complete/published.json`
appear. The job's axes are `execution_status=completed`,
`numerical_status=eligible`,
`response_validation=archived_reference_matched` and
`physical_validation=not_performed`. Archived parity is **not** signed
nonlinear reanalysis or independent meteorological truth.

The second worker had distinct PID `14275`. A cancellation request was
accepted at monotonic time `579692.477926208`, before its process was
reaped at `579692.74567825`. Its process-group termination was
`cancelled`, child exit code `-15`, and any raw `running` JSON was not
treated as a completed response. Its job axes are
`execution_status=cancelled`, `numerical_status=not_verified`,
`response_validation=not_performed`; it has **no `published.json`**.
The finishing worker's publication was unaffected by this distinct
token and child process.

The parent pair completed in `22.659842667053454` seconds. The
successful child took `22.3051692500012` seconds with 84 sampled RSS
measurements and a sampled peak of `340,033,536` bytes. The cancelled
child took `1.0699283750727773` seconds with four samples and a
`223,084,544` byte sampled peak. Each stayed under its 300-second and
sampled 768-MiB limit. These **per-child sampled peaks do not bound
combined system memory**; parent memory and between-sample spikes were
not included.

The cancellation/publication decision is linearized by one token lock.
A cancellation accepted before publication acquires that lock prevents
publication, including after a successful worker exit. Once
publication has acquired it, a concurrent cancellation waits and
cannot retract that result even if temporary serialization is still in
progress. The decision and completed-write timestamps are recorded
separately. A regression pauses temporary writing to verify this
ordering. The process guard also reaps a cancelled subprocess group,
and the pair coordinator waits for **parseable** `running` JSON rather
than reacting to the mere existence of a partially written file.

This closes a **fixed-case process response job lifecycle**: cancellation,
sampled resource accounting and publication ordering are exercised with
real local-response workers. It does not establish a production queue,
generic input validation, durable recovery after parent crash,
in-process simultaneous forward AD, concurrent GN/refinement, or
physical/finite-influence validation. Raw child/resource/lifecycle/pair
records and source snapshots are preserved in
`response_job_lifecycle_attempt1/`.
The affected guard/process/response/lifecycle suite passed 44 tests;
error-level basedpyright reported zero errors. GREEN and RED prelaunch
reviews approved the bounded run after cancellation waited for a
parseable `running` report and the token-lock publication decision was
specified precisely. The aggregate
`FV_RESPONSE_JOB_LIFECYCLE_EVIDENCE.json` binds the base commit, raw
reports, code/tests, checklist/KG and exact commands. This is focused
local evidence, not a full CPU/package CI run.
