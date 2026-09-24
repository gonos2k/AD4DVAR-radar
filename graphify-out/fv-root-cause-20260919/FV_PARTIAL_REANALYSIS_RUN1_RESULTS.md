# G3b partial-observation numerical attempt 1

The user's `go` authorized the predeclared single serial numerical attempt in
`FV_PARTIAL_REANALYSIS_PLAN.md` (1,200-second child wall cap, sampled 1-GiB
RSS cap). The guarded preflight completed, and the numerical child ran once.
The raw files are in `partial_reanalysis_approved_20260924/`; none was
overwritten or counted as a successful response validation.

| Axis | Measured outcome |
|---|---|
| Guarded preflight | exit 0; 1.813 s; sampled peak RSS 315,899,904 bytes |
| Numerical child | exit 1; 133.185 s; sampled peak RSS 343,343,104 bytes; no resource termination |
| GN analysis | 4 outer iterations, 83 PCG iterations; `maximum_outer_iterations`; objective 0.1712418172180851 → 0.02076475483611489 |
| Nominal Newton correction | 4 PCG solves, 29 iterations each; fourth correction refused |
| Refusal | `stationarity refinement failed to find an Armijo step: iteration=4; no finite candidate evaluation` |
| Adjoint and signed endpoints | Not reached; 0 nonlinear reanalyses and 0 endpoint pairs |
| Response validation | **not established** |

The exception comes from `refine_stationary` after its fourth Newton step
found no accepted backtrack. Its wording means none of the tested candidates
reached a finite objective/gradient evaluation; branch rejection and a
nonfinite candidate are both possible. The present checkpoint does not store
each rejected candidate's reason, so their relative contribution is unknown.
This is a local correction-policy refusal, not evidence that the partial
inverse problem has no stationary point or that its adjoint formula is wrong.

The raw report's `phase=partial_newton_refinement`,
`numerical_status=refused`, `execution_status=error` and
`response_validation=not_established` correctly prevent success accounting.
It also says `nominal_eligibility=not_attempted`, even though correction was
attempted. That status is a **bookkeeping defect** in the measured producer;
the phase/error/PCG records establish the attempt. The producer was corrected
for future runs without rewriting this raw report. No tolerance,
direction, target or branch margin was changed to turn this attempt into a
pass.

The numerical run's guarded preflight JSON pins all 14 relevant source hashes,
the plan SHA256, problem/input identities and warm-start report. All 14 source
hashes match the separately committed measured source tree `ab8aed4`.
The run report additionally records the preflight SHA256 and source/input
checks before the solve. The numerical child failed before its final
after-run identity check; the archived measured sources remain available for
external verification. The runner's exit code and resource record are kept
separate from the numerical checkpoint. Exact output hashes, the approved
command and the measured-source comparison are in
`fv_partial_reanalysis_run1_manifest.json`.
An independent read-only arithmetic/source check is in
`fv_partial_reanalysis_run1_audit.json`; it confirms the process failure,
resource use within the cap, four nominal-refinement PCG solves, zero
endpoints, and the measured-source hash match. After the raw attempt, a small
producer fix changed the future `nominal_eligibility` refusal status and added
an analytic regression; 53 current affected tests pass with 18 existing warnings and
the pinned typecheck reports 0 issues. This later source was **not** the
measured numerical source and was not used to claim a rerun.

This attempt used a one-lead 4×5/26-control problem with two genuine missing
observations and a synthetic conditional score. G3b remains open. Any new
numerical attempt needs a new, frozen policy and resource decision. The
`FV_PARTIAL_REFINEMENT_DIAGNOSTIC_PLAN.md` proposes a nominal-only run that
counts candidate refusal reasons; it is not approved or executed. This report
makes no two-lead, finite-path, physical skill, or
general minmod response claim.
