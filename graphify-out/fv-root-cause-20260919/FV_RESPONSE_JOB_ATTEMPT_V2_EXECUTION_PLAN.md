# One fixed v2 response-job integration, after fake-guard checks

Run one **fresh** fixed `fv4x5` synthetic conditional local-response
job through `fv_response_job_lifecycle.run_job` in
`graphify-out/fv-root-cause-20260919/response_job_attempt_v2_integration1`
with request ID `v2_binding_1`. The job directory and its sibling
lock path must be absent before launch; its parent already exists.
Use the current code, unchanged fixed input, archived reference and
PR #216 attempt-binding plan. Do not rerun product GN or signed
nonlinear reanalysis, change the physical model, or substitute a
second case if this run fails.

The existing child guard is the resource contract: a **300-second
child wall-limit trigger** with a short termination/reap grace period,
and **sampled 768 MiB child RSS**. Successful publication still requires
reported child elapsed time at most 300 seconds. Record actual
command/PID/exit, sampled peak/count, source/archive/input before and
after, attempt ID, manifest/raw/resource/published SHA256s, branch
stage count, actual adjoint residual and total response. The parent
must require completed worker and eligible response with 54 observed
stages, maximum gradient `<1e-10`, true adjoint relative residual
`<=1e-10`, and total response within the existing `1e-6` relative
archived-reference gate. The v2 publication may appear only after
the durable manifest, child reap, raw/resource stabilization and
token-locked validation. If cancellation, resource limit, fsync,
source/input/hash or numerical gate fails, preserve the failure and
do not retry or publish.

After the job completes, call the read-only v2 inspector once on
the same path/request. It may return `publication_candidate` only
if the full attempt/raw/resource/pub/lifecycle chain matches and
will still set `candidate_is_authoritative=false`. A failed job or
missing publication remains `needs_reconciliation`. Rehash the raw
job files into a separate evidence manifest after the run; do not
modify historical PR #206 artifacts. This is **one same-operator
integration regression**, not durable crash recovery, physical
forecast skill, nonlinear observation impact, generic radar service
or a full CPU/package regression.
