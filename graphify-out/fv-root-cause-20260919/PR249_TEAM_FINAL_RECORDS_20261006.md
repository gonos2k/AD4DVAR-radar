# PR249 final records/execution audit — 2026-10-06

**Disposition:** no actionable records/execution findings remain in the reviewed delta. The source fixes close the false-completed parent receipt, preserve HVP started/completed counts on operator exceptions, classify the finite PCG direction-update failure as a numerical refusal, and defer the optimizer-step record until post-candidate checks pass. The existing successful live-HVP artifact remains evidence for the original pinned source version; no live FV/HVP run was repeated after the changes.

## Change review

- The parent now rejects decoded non-object child JSON inside the caught parse path at [fv_point_3h_hvp_newton_step.py:337](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_hvp_newton_step.py:337). The toy `[]` child regression now requires a failed parent and a recorded read error. This addresses the reproduced `TypeError` that had left the preliminary parent receipt marked `completed`.
- On an unexpected exception, the run records `programming_error`, the exception type/message, and the in-memory HVP counters before re-raising at lines 280–287. A new mocked failure inside the HVP operator verifies the key partial-call case: one started and zero completed products are durable. The exact known `PCG direction update is not finite` RuntimeError is recorded as `step_refusal` and follows the normal integrity closeout.
- Candidate fields are staged in `candidate_update`; the common finalizer publishes them only after the fixed-input/source/runtime integrity check and a final cooperative deadline check (lines 256–310). Deadline expiry leaves zero optimizer steps, changes an accepted Armijo trial to `candidate_not_committed`, and restores the base control identity.
- An unexpected error during post-candidate diagnostics now relabels any durably accepted but uncommitted trial as `candidate_not_committed` before writing the programming-error record and re-raising; the model-error regression verifies this path.
- The diff is narrow: the one runner and its focused tests. `git diff --check` passed.

The two gaps noted in the earlier records review are now covered by focused toy regressions: operator-internal failure preserves 1 started / 0 completed, and post-candidate diagnostic failure records `candidate_not_committed` before propagating. No actionable residual finding remains in this records/execution scope.

## Existing attempt evidence

The archived successful attempt is internally consistent. Its raw `step.json` reports 24 PCG iterations, 26 started and 26 completed live HVPs, residual `8.214105073635838e-11`, and one optimizer step. The 26 live products decompose as 24 Krylov operator applications plus the fresh PCG residual product and the caller's true-residual product. The separate archived f82c preconditioner audit reports 26 Hessian columns and 27 HVP calls; the live Newton operator at a35 is original-J curvature, and f82c supplies only its block inverse preconditioner.

The raw child says `phase=finished` and `numerical_status=one_live_HVP_original_J_step_accepted`; the parent says `execution_status=completed`, exit code zero, no child-read error, 219.13 seconds elapsed, and sampled peak RSS 370,491,392 bytes. The candidate passes the strict point and Armijo gate, lowering J from 0.0716149827 to 0.0668157467. Phi rises from 1.1129736171 to 1.8573840876 and gradient infinity norm rises from 0.7980151516 to 0.9880409692. Stationarity, new-point curvature, response/score, physical validation, global SPD, and forecast skill remain unclaimed.

The original plan, accepted step, raw Newton attempt (`step.json`), parent/resource receipts, and all 7 archive files remain unchanged and match their pinned hashes. Before the caller/test edits, all 88 plan source hashes and all 19 manifest file hashes matched the workspace. The runner edits intentionally change the pinned caller/test source hashes; the old plan therefore refuses a rerun against the modified caller and continues to identify the historical source version used for the immutable raw result.

## Verification recorded for the fixes

The focused log [PR249_TEAM_TESTS_20261006.log](/Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/PR249_TEAM_TESTS_20261006.log) reports 51 passed in 1.66 seconds, with 18 `torch.jit.script` deprecation warnings. The type check [PR249_TEAM_TYPES_20261006.log](/Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/PR249_TEAM_TYPES_20261006.log) reports 0 errors, 0 warnings, and 0 notes. These checks use toy/mocked focused paths; they do not establish a rerun of the FV/HVP calculation or forecast validity.
