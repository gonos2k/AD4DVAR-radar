# PR249 final RED review — 2026-10-06

**Disposition: GO for the narrow recovery/report fixes.** Review was limited to the current diff in `fv_point_3h_hvp_newton_step.py` and its focused tests. I did not run the FV objective, a live HVP, or alter source, plan, or raw-run artifacts.

## Closure of prior findings

- The solver caller now classifies the exact shared-PCG message `PCG direction update is not finite` as a numerical `step_refusal`; other unexpected exceptions are written as `phase=numerical_status=programming_error` with their exception class/message and then re-raised. The HVP counter snapshot is already set by the solver `finally` block before this error handler. Toy regressions cover both a solver error after one complete HVP and an exception raised inside the operator: durable counters distinguish 1 started/1 completed from 1 started/0 completed, preserve the error class/status, and retain propagation for unexpected errors.
- Candidate details remain staged in `candidate_update` until post-accept diagnostics, source and input integrity checks, and another internal-deadline check complete. Expiry drops the pending update, resets `final_c` to the accepted base control, records a budget refusal, and marks the passing search row `candidate_not_committed`. The model-diagnostics and final-input-identity expiry mocks each assert zero committed steps and no accepted-control field. A further mock raises unexpectedly from model diagnostics and verifies the durable programming-error record marks the candidate uncommitted before re-raising.
- The parent now rejects valid non-object JSON child results inside the parse guard. The `[]` regression verifies the parent receipt becomes failed and records a child-read error.

## Verification reviewed

The saved focused-test record reports 51 passed in 1.66 seconds, with 18 TorchScript deprecation warnings. The saved type-check record reports zero errors, warnings, or notes. These checks exercise only bounded tests and mocks; they do not establish new FV/HVP numerical behavior.

The existing accepted-run receipt remains internally consistent for the pre-fix source snapshot: completed child, 26/26 live HVPs, 24 PCG iterations, true relative residual `8.2141050736e-11`, one original-J accepted trial, and 219.13 seconds under the external 300-second guard. It must remain attributed to its historical producer bytes.

## Evidence boundary

The frozen historical plan and raw receipts were left untouched as requested. Consequently, the plan’s producer and test source hashes still bind the pre-fix bytes (`SELF` expected `882f8358…` vs current `7103bbba…`; focused test expected `8d6ce8d2…` vs current `9ed19555…`). A fresh `load_base` run against that historical plan should refuse these edited sources. A future guarded attempt would require a separately versioned plan pinning the new producer and tests; no such attempt is authorized or claimed here.

No actionable defect remains in the requested error-record (including operator errors after a started but incomplete HVP), classified-PCG-failure, malformed-parent-child, post-diagnostic programming-error, or post-accept deadline paths. The only remaining limit is the intentional separation between the historical numerical result and current source fixes.
