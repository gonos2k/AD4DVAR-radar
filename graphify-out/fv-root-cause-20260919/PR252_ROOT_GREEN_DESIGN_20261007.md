# PR #252 root-candidate cost-floor review (GREEN)

Date: 2026-10-07
Scope: read-only mathematical/design review of the reported P2 in the bounded dual-merit continuation. No source edits, real FV execution, HVP/PCG execution, or shared Graphify refresh were performed.

## Finding

The reported failure mode is valid. In `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`, `run_iterations` first receives an accepted trial that already contains a finite objective, full candidate gradient, `Phi`, and a strict-branch result from `dual_merit_search` (call at lines 274-277; candidate result extraction at 284). It checks displacement stagnation at 288-293, then checks absolute-scale decrease floors at 294-304, then performs candidate closure and deadline checks at 305-310, commits at 311-329, and only after commit checks candidate `gradient_inf` against `ROOT_GRADIENT_INF` at 330-332.

Consequently a candidate that satisfies the configured stationarity threshold can be discarded as `numerically_zero_decrease` before the status can become `root_pending_audit`. This does not invalidate the reported two FV accepts: their recorded candidate reductions exceeded the floors and their accepted gradients were not near the root threshold. It is a control-flow defect in the near-root regime, not an explanation for the third solve's budget refusal.

## Mathematical basis and scope

For `J_C(c) = C + 0.5 ||c||^2`, `g(c)=c` and `H(c)=I` for every constant `C`. The exact minimizer and all stationarity data are invariant to `C`; only the absolute scale of objective values changes. The test counterexample therefore isolates a numerical policy mismatch: `j_floor = 128 eps max(|J_k|, |J_trial|, tiny)` in lines 296-297 measures whether the scalar objective difference is resolvable relative to the objective's absolute magnitude, while `gradient_inf <= ROOT_GRADIENT_INF` (line 330) measures the desired first-order residual. One cannot stand in for the other. The reported `C=1`, initial `||g||=1e-7` case has exact Newton candidate `g_trial=0` and a true quadratic decrease near `5e-15`, below `128 eps * J ~= 2.84e-14`; adding a constant leaves the root unchanged but makes this absolute-floor gate fail.

The small-safe-change domain is limited to candidates that already passed the existing dual Armijo checks and complete strict branch gate. In `fv_point_3h_dual_merit_step.py`, `dual_merit_search` computes the candidate objective and gradient (lines 202-225), forms `Phi`, validates complete strict branch/margins (231-239), evaluates both Armijo predicates (240-248), and returns a candidate only if both predicates and strict branch pass (253-256). The proposed exception applies only when that already-evaluated candidate's `gradient_inf` meets `ROOT_GRADIENT_INF`. It does not authorize arbitrary cost increases, bypass a branch check, relax either Armijo condition, or establish final root eligibility.

## Minimal safe ordering

Recommended narrow control-flow adjustment in `run_iterations`:

1. Treat the gradient in `candidate_row` as a provisional root signal only. On the root-candidate path, re-evaluate candidate J and the full gradient at the exact accepted control, recompute Phi, check the deadline around these callbacks, and compare the fresh finite values with the trial receipt under an explicit deterministic/precision policy. Use the fresh gradient to decide whether the candidate meets `ROOT_GRADIENT_INF`; fail closed and leave it uncommitted if the evaluation is nonfinite or inconsistent.
2. Keep the existing displacement-stagnation rejection for non-root candidates. For root candidates only, a displacement below `128 * EPS * scale` may be consistent with a valid root step: for a quadratic with large positive curvature, a tiny control displacement can remove a gradient that was above threshold. Record the small displacement as a diagnostic; do not use this exception for a candidate whose gradient misses the declared threshold.
3. Apply the existing `j_floor` / `phi_floor` rejection to non-root candidates. A root candidate may bypass these two “numerically zero decrease” floors only after the existing dual Armijo and strict branch checks, and only if its actual J is not above the base by more than the existing J roundoff budget. Continue through ordinary candidate closure and deadline checks.
4. Keep the existing independent candidate branch and fixed-input/source integrity closure; additionally require the fresh root-candidate J/g/Phi check before commit, and check the deadline after closure. Preserve current provisional receipt behavior and `_mark_current_candidate_not_committed` handling (lines 305-329, 364-371).
5. Preserve `root_pending_audit=True`, `eligible_stationary_point=False`, and `full_root_claim=False`. This route records a pending audit; it does not auto-certify the point.

The trial point has passed actual-J and actual-Phi Armijo plus its strict branch check. If the scalar objective difference is below its reliable resolution, that fact should be retained in the receipt as a diagnostic if useful, rather than used to erase a candidate whose freshly re-evaluated gradient meets the declared threshold. The independent root-path J/g/Phi evaluation must agree with the trial receipt under an explicit precision policy; otherwise leave the point uncommitted. Acceptance remains conditional on all normal branch, integrity, closure, and deadline checks. A confirmed root candidate must still be refused for a J increase greater than the existing roundoff budget; the exception does not weaken either Armijo test.

Potential implementation detail: compare `candidate_row["gradient_inf"]` if added by the search record, or compute `max(abs(candidate_row["gradient"]))` from the gradient list already recorded at lines 223-225. Keep this as a detached diagnostic/control-flow decision, as the optimizer is making an acceptance decision; do not alter the differentiable objective/gradient/HVP calculations.

## Regression evidence expected

A lightweight callback-only regression should reproduce the reported constant-offset family (`C in {0, 1, 0.063}`) with exact quadratic HVP and a strict-branch fixture; a separate high-curvature fixture should exercise a sub-tolerance displacement that reaches a root threshold. It should show:

- an exact candidate root that previously fell below the absolute cost floor now reaches `root_pending_audit` after fresh candidate J/g/Phi re-evaluation and closure;
- a stateful callback whose trial gradient is root-sized but whose fresh candidate gradient is not must fail closed and remain uncommitted;
- a confirmed root candidate may pass a sub-tolerance displacement only when its candidate gradient meets the threshold; a non-root sub-tolerance move remains refused;
- the result still has `eligible_stationary_point=False` and `full_root_claim=False`;
- both Armijo predicates and strict candidate branch remain required;
- a non-root candidate below the same absolute floor still stops as `numerically_zero_decrease`;
- displacement stagnation, failed closure, and expired-deadline paths still leave the provisional candidate uncommitted.

These are behavior-level checks; no repeat FV run is needed to validate this ordering fix.

## Minimal bounded follow-up from the last accepted point

For a later scientific run, pin a new execution plan to the final committed control `90fc4555…` and its exact parameter/source/archive identities, use the same existing bounded live-HVP/PCG, dual-Armijo, strict-branch, closure, and receipt kernel, and start iteration numbering from that point. Do not rerun the earlier `e0b04a… -> 8c1285… -> 90fc4555…` steps or reuse their directions as current-point directions. Set an explicit new resource budget before launching; preserve the last committed point on any solve, branch, deadline, or closure refusal. Report subsequent HVP/PCG counts separately from the earlier 62 HVP record, and report stationarity, objective, block residuals, branch status, and numerical stop reason without extrapolating a convergence rate. A root-threshold hit must remain pending the existing separate curvature/branch/adjoint/reanalysis audit.

## Source locations reviewed

- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: `run_iterations`, especially current-point root gate 230-247, dual search 274-283, displacement/floors 284-304, closure/commit 305-332; provisional-candidate clearing 364-371.
- `examples/weather_scenarios/fv_point_3h_dual_merit_step.py`: `dual_merit_search`, actual objective/gradient/Phi evaluation 202-230, strict branch check 231-239, Armijo and acceptance 240-258.
- `tests/test_fv_point_3h_dual_merit_continuation.py`: current root-pending regression 65-76, commit-failure preservation 108-129, deadline refusal 132-147. The reported constant-offset case is not present in the checked-in test excerpt; add a focused case rather than altering these established invariants.

## Assumptions and limits

This review assumes the reported counterexample values and the stored PR #252 run summaries are accurate; I inspected the local control flow but did not execute the supplied review bundle or the FV model. The production trial path validates a finite FP64 scalar objective, finite 26-component gradient, finite `Phi`, and strict branch metadata before it can report acceptance. The candidate commit closure independently recaptures branch and fixed-input/source integrity; because the existing closure does not recompute candidate J/g, a narrow root path must add that fresh comparison before promoting a trial receipt to pending audit. The reasoning establishes the ordering flaw and a narrow remedy; it says nothing about existence of a strict-branch root, Hessian positive definiteness, adjoint correctness, forecast accuracy, or convergence of future iterations.

## Review of the implemented patch and saved evidence

Reviewed the shared diff after implementation. The final control flow in `run_iterations` now marks the trial gradient as provisional, then on a root-sized trial re-evaluates candidate J/g/Phi with `_true_merit`, re-evaluates the branch, checks deadline, compares fresh scalars and gradient with the trial receipt at a 128-epsilon scale, requires the same strict branch signature, and repeats both Armijo threshold comparisons before proceeding (current source lines 288-333). A failed recheck marks the candidate not committed and exits that iteration (315-328). The displacement floor remains for non-root candidates (334-339); a confirmed root candidate may continue with a recorded `displacement_below_roundoff` diagnostic (369-377). A significant J increase remains a refusal (348-355), non-root J/Phi floor behavior remains intact (356-362), and candidate closure/deadline checks still precede commit (363-392). Root status remains pending audit; the runner's initialized eligibility/full-root fields remain false.

The regression additions cover the constant-offset family and pending-only flags (test lines 81-105), exact root acceptance below the displacement floor (108-128), non-root refusal below that floor (130-157), and stateful objective/gradient/branch recheck mismatches failing closed before closure (207-250). Existing closure-failure coverage remains at 156-205. The implementation agent reported `.venv/bin/python -m pytest tests/test_fv_point_3h_dual_merit_continuation.py -q`: **28 passed in 0.98s**. I inspected the patch and reported command/result but did not independently launch pytest, run a type checker, or execute FV/HVP/PCG work.

GREEN disposition for the narrow P2 fix: **GO on the inspected source and reported focused test evidence**. This closes the reported cost-floor ordering failure and the associated sub-roundoff displacement/recheck failure modes. It does not certify a physical stationary point or alter the status of the prior bounded FV run; no new FV or long continuation execution is required for this fix.
