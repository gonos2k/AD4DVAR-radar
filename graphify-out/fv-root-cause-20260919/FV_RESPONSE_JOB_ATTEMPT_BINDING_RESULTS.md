# Fixed FV response attempt binding: one v2 integration

The fixed `fv4x5` response job completed once through the new durable
attempt-binding path. Its process exit was 0, its numerical response
passed the existing fixed-case gates, and read-only post-run inspection
classified the complete record as an **unauthenticated publication
candidate**. This closes only the fixed-case attempt-binding subitem of
R7-P. Durable restart/recovery, generic requests and production
concurrency remain open.

## Scope and execution

The predeclared one-shot plan is
`FV_RESPONSE_JOB_ATTEMPT_V2_EXECUTION_PLAN.md`. GREEN and RED prelaunch
reviews found no blocker. RED clarified that 300 seconds is the child
wall-limit **trigger** plus termination/reap grace, not a hard elapsed
maximum; successful publication still requires a reported child elapsed
time no greater than 300 seconds. The RSS guard is sampled child RSS,
not a process-tree or hard allocation bound.

The only real FV launch used request `v2_binding_1` in the fresh
`response_job_attempt_v2_integration1` directory. Attempt UUID was
`7d503398-bd27-4a13-9140-bc3cf9fea981`; the worker PID was 51948.
The child exit code was 0, with 23.873 seconds child elapsed, 24.145
seconds coordinator elapsed, 89 RSS samples and 350,339,072 bytes
sampled peak. There was no resource termination, monitor error,
cancellation or runner/read error. The final lifecycle says
`completed`, `eligible`, `archived_reference_matched` and
`physical_validation=not_performed`.

The raw worker recorded 54 branch stages and 54 observer events,
maximum gradient `6.962040319247187e-12`, 25 adjoint PCG iterations
and 26 HVP calls. The independently evaluated transpose relative
residual was `9.888163804832049e-11`, below the existing `1e-10`
gate. Direct response was 0; indirect and total response were both
`-0.003038272528696361`, exactly the archived value in this run
(reported relative difference 0). No product GN or nonlinear signed
reanalysis was run.

## Record chain and read-only classification

`attempt.json` was durably installed before child launch. The worker
reported the same UUID. The persisted resource report binds the UUID
and attempt-manifest SHA256. The v2 publication binds the attempt,
manifest, raw worker and resource SHA256s, source/archive/input
identity and response. The terminal lifecycle records the same
attempt and resource result. The file hashes are in
`FV_RESPONSE_JOB_ATTEMPT_BINDING_EVIDENCE.json` alongside code, plan
and verification-log hashes.

After completion, exactly one read-only call to
`inspect_fixed_job_after_restart` on that directory/request returned
`state=publication_candidate`,
`reason_code=validated_publication_candidate`, and
`candidate_is_authoritative=false`. The files are mutable and unsigned;
this is an internally consistent local record, not authenticated
recovery or authorization to adopt/retry a worker.

## Verification and limits

Affected focused tests: **105 passed** in 13.24 seconds, recorded in
`fv_response_job_attempt_v2_affected_tests.log`. Error-level
basedpyright 1.39.9 for the three changed modules and their three test
files: **0 errors, 0 warnings, 0 notes**, recorded in
`fv_response_job_attempt_v2_typecheck.log`. `git diff --check` was clean.
Incremental Graphify code refresh was completed; no full semantic
rebuild was used. These checks and the single same-operator FV response
do not constitute full CPU/package CI, physical forecast validation,
finite observation impact, crash recovery, or a general radar service.

Remaining R7-P work includes an authenticated/durable recovery policy,
orphan-child and cancellation handling across parent restart, general
request/input contracts, concurrent GN/refinement and aggregate resource
limits. Coordinated changes to unsigned local artifacts can still form
an internally consistent candidate.
