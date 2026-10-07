# d31 inexact comparison — GREEN prelaunch review

Date: 2026-10-07
Scope: read-only review of the comparison/loader implementation and focused tests. No source edits, test runs, FV, HVP, PCG, forecast, adjoint, or reanalysis were performed in this review.

## Final disposition: GO for the bounded strict-first shared-budget trial

The earlier review findings were resolved in the code and frozen plan. The d31 continuation defaults to the original strict policy; the endpoint loader binds the requested committed point to pinned receipts; and the experiment shares one 720-second/90-HVP budget across strict-first then inexact cold starts. The plan explicitly gives no equal per-arm reserve and makes no comparison claim if either arm is unfinished. This is suitable to launch as the stated bounded trial. It does not provide equal compute to both methods and cannot support a general speedup claim.

## Confirmed contracts

- `fv_point_3h_dual_merit_continuation.py` derives the executed policy from the invocation's `linear_mode` and `max_iterations`; the default remains strict and three iterations. Inexact `eta` is computed from the current full gradient infinity norm, clipped to `[1e-10, 1e-3]`, and labeled a heuristic rather than Eisenstat–Walker.
- The relaxed solve is accepted only when PCG converges and a fresh current-point `Hs` gives `||Hs+g||₂/||g||₂ <= eta`. It retains finite-value checks, positive observed `sᵀHs` beyond the roundoff budget, `gᵀs < 0`, and `gᵀHs < 0`; relaxed mode also requires both slopes to clear their scale-aware roundoff budgets. The same current-point operator is used; archived f82c curvature remains only a preconditioner.
- Both modes use the same continuation path for candidate evaluation, endpoint branch/margin checks, actual `J` and `Phi` Armijo tests, candidate commit closure, and `||g||∞ <= 1e-10` root-pending threshold. The inexact residual setting does not alter root eligibility.
- `fv_point_3h_committed_base.py` requires a plan-pinned repository-relative base receipt and parent/resource receipts; rejects path escapes, symlink endpoints, hash mismatch, non-completed execution, source/input/runtime mismatch; closes the accepted control, metrics, branch, and margins against the last accepted trial/current state; and validates the producing plan and f82c preconditioner receipts. It uses the plan's `base_step` instead of a fixed 90fc start.
- `fv_point_3h_inexact_comparison.py` runs both policies from the pinned base with `max_iterations=1`, a common absolute deadline, and one shared HVP/PCG counter map. Per-arm deltas are recorded; child HVP counters remain cumulative and the continuation records `new_hvp_calls`. An uncompleted PCG solve is marked `not_recorded`, not inferred from HVP count. The selected control stays at d31 unless the inexact arm reports one committed step with completed execution and source/input integrity.
- The focused tests cover strict-default preservation, forcing clipping, a synthetic residual/work comparison, rejection of a perturbed true residual, deadline/HVP counter sharing, a pinned real endpoint, pin tampering, and no selection of an inexact result that fails integrity. These are source-level test cases; I did not rerun them.

## Resolved prelaunch findings

**Outcome selection is now distinct from the planned policy.** The parent receipt sets `planned_selected_arm="inexact"` and `selected_arm=null` initially, and sets `selected_arm="inexact"` only after the inexact arm reports one committed step with completed execution plus source/input integrity. A budget-only arm or expired preflight remains unselected; the base endpoint is retained as metadata. Focused regressions cover accepted selection, unverified endpoint, unstarted second arm, budget refusal, and deadline expiry before either arm.

**Shared-budget order is explicit and handled as an incomplete comparison.** Strict runs first; its deadline/HVP consumption is not reset before inexact. The plan and result label this as a strict-first shared-remaining-budget trial with no equal arm reserve or global speedup claim. A budget refusal marks `comparison_complete=false`; the run stops without extending the budget, and no paired cost/progress contrast is claimed when one arm is unstarted or incomplete. This is an accepted limitation of the chosen experiment design, not an implementation blocker.

## Launch interpretation and remaining limits

With the frozen policy, `numerical_status=budget_refusal`, `comparison_complete=false`, and `selected_arm=null` is a valid bounded stop. It is not evidence that inexact failed numerically. If both arms complete, compare their recorded per-arm deltas and progress while stating the strict-first shared allocation. An accepted inexact endpoint supports one correction from d31, not convergence or generalized efficiency.

The frozen plan exists with SHA-256 `d69ce3000f1605bed67ea67a4c6751bbbdf2ab142fa47bd24418c368d9220d41`; its scope declares strict-first shared remaining budget, no equal per-arm reserve, and no paired claim when an arm is unfinished. I recomputed the plan file hash and checked all 102 source pins and 24 archive pins; all matched. Persisted checks report 63 affected tests passing with 18 existing warnings, type check with zero errors/warnings/notes, and a six-file Graphify refresh (98 nodes, 311 edges) with shared graph/report hashes unchanged. I inspected these saved receipts but did not rerun tests or type checking. Metadata-only preflight expiry is classified as budget refusal when the pinned base receipts and current runtime/source closure validate.

After prelaunch, the single planned comparison completed with both arms accepted. See [INEXACT_GREEN_FINAL_20261007.md](INEXACT_GREEN_FINAL_20261007.md) for the saved strict/inexact result, independent saved-array arithmetic scope, and convergence/forecast limits. This addendum and the final review do not rerun FV/HVP/PCG.
