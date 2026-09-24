# G9 bounded PCG diagnostic isolation

The 86-control execution and signed reanalysis producers, and the partial
observation producer, previously replaced imported module-level `pcg`
functions temporarily to record Hessian–vector calls, Rayleigh diagnostics,
iterations, residuals and phase timing. An overlapping solve could then use
the other run's wrapper, lose its own counts, or restore a stale global
function. This is distinct from the minmod-stage patch already removed in
`FV_STAGE_OBSERVER_SCOPE.md`.

`matrix_free.observe_pcg_calls(factory)` now binds a trusted diagnostic
wrapper factory to the current Python context. The public `pcg` function
selects that wrapper and calls the unchanged `_pcg_impl` iteration; imported
aliases in variational analysis, local refinement and local response route
through the same context. Nested wrappers run innermost then outermost and
the context token resets after normal or exceptional exit. The three
research producers use this context rather than replacing module globals.
Their existing per-solve monitoring functions still wrap the operator and
record the same kinds of numerical diagnostics; no archived numerical solve
was rerun or relabeled.

On a small non-dyadic SPD problem, observed and unobserved PCG solutions,
iterations and recomputed relative residual agree exactly. The test checks
that the true residual is nonzero, so the comparison is not a vacuous
zero-residual case. Two overlapping threads each collect only their own
solve, while an ordinary call on the third thread remains unobserved.
Nested order, exception cleanup, a pre-imported `local_refinement.pcg`
alias and the partial reanalysis monitor's HVP count are also exercised.
Broader affected tests covering PCG, variational analysis, local response,
FV response and research wrappers reported **270 passed, 83 subtests passed,
18 existing TorchScript warnings**. Pinned basedpyright on the core and new
observer test reported 0 errors/warnings/notes. The broad run was observed
live; the compact verification record below transcribes its final summary,
and does not pretend to contain a per-test raw log.

The wrapper factory is a **trusted diagnostic contract**. It receives the
operator and can in principle alter inputs or results if misused; the
repository's monitoring wrappers call the supplied original solve and
return its result. Unlike the minmod-stage observer, this interface is not
an immutable event stream. The source-level check and tests show removal of
known production/research PCG global patches in the three producers, but
do not certify observer use while differentiating through PCG, async child
task inheritance, or full concurrent FV optimization/response workflows.
Each async task must bind or clear its own context if spawned inside an
active diagnostic. For now the actual large FV numerical producers remain
serial/process-isolated as measured; in-process parallel service is a
separate integration test.

Historical FV86 and partial-response reports retain their pinned source
hashes and numerical evidence. The current `matrix_free.py` and producer
sources differ from those historical runs, so a future numerical replay
must create fresh source-bound evidence; changing archived hashes would be
incorrect. No physical forecast skill or new response validation follows
from this instrumentation change.

The commands, compact observed test result and current source hashes are in
`fv_pcg_observer_manifest.json` and `fv_pcg_observer_verification.json`.
