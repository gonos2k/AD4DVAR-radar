# G9 bounded minmod-stage observer isolation

The current strict branch tracer and partial-observation face-margin tracer
previously replaced the process-global `transport._euler_minmod` function
during a call. A concurrent ordinary forecast could then run through a
diagnostic wrapper, be rejected by its stricter gates, or add stages to the
wrong report. Two overlapping traces could also restore wrappers in the
wrong order. This is a control-flow and provenance defect in prospective
in-process parallel use, even though the archived serial numerical results
remain valid at their original source versions.

`transport.observe_minmod_stages(observer)` now binds a diagnostic
callback to the current Python context with `ContextVar` and resets it with
the returned token. `_euler_minmod` calls the callback before its unchanged
tensor arithmetic only when an observer is present. Each callback receives
separate detached copies of stage state and fluxes, so even an accidental
in-place write cannot change transport or an enclosing observer's record.
Nested callbacks run
innermost then outermost, matching the previous serial wrapper order: the
strict branch gate can refuse before the outer face-margin recorder sees a
stage. The two research tracers now use this context rather than modifying
the transport module. An unrelated thread retains the unobserved forward
path, including valid zero-face states.

The new tests compare unobserved and observed stage output exactly, nested
callback order, mutation isolation and exception cleanup; two concurrent three-stage observers
match their separate serial input/flux records exactly. Two concurrent
strict traces with different face signs match their own full `choices` and
`face_signs` signatures, stage counts and minimum margin. An ordinary
zero-flux forward stage proceeds while another thread's observer is held
inside its callback. Existing 4×5/8×10 model, minmod derivative, replay,
preflight and finite-volume suites also pass: **110 affected tests**, with
18 existing TorchScript warnings. Pinned typecheck on the changed core
transport and new observer tests reports 0 errors/warnings/notes. The legacy
research probe modules are not claimed typecheck-clean by this record.

The observer is a **diagnostic**, called outside `torch.func` transforms and
checkpoint/replay differentiation. Callback writes to its own snapshots do
not alter the model, though diagnostics should remain observational. The
arithmetic path is unchanged when unobserved, and branch checking remains a
pointwise local eligibility check, not a finite-path certificate. Python
async tasks inherit context at creation; a task spawned inside an active
diagnostic context must explicitly clear or replace the inherited observer
if it should run independently after the parent scope exits.
This bounded test establishes thread-context isolation at the minmod stage
and serial model compatibility, not arbitrary async/service safety.

PCG monitoring in the partial reanalysis, FV86 execution and FV86 signed
reanalysis producers still temporarily patches imported module-level `pcg`
aliases, so **G9 is only partially closed**. Those
instrumented producers remain serial/process-isolated until an analogous
per-call solver observer replaces their patches and concurrent solver
results, residuals and report attribution are tested. Historical report
source hashes are not rewritten: the observer refactor changes current
source identity, while the old measurements retain their pinned source and
numeric evidence.

Exact commands, logs and current source hashes are in
`fv_stage_observer_manifest.json`.
