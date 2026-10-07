# PR #252 near-root audit path: RED design review

Date: 2026-10-07
Scope: read-only review of the reported P2 and failure modes around a candidate whose full gradient meets `ROOT_GRADIENT_INF`. No FV, HVP, PCG, forecast, adjoint, reanalysis, or repository test run was performed.

## Finding in the current implementation

In `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py::run_iterations`, a dual-merit trial is returned with status `accepted`, then the continuation checks displacement stagnation (around line 287) and the roundoff-scaled J/Phi reductions (around lines 296–302). The accepted candidate's `gradient_inf` is only checked after commit and state advancement (around lines 313–332). Thus both early guards can discard a candidate that the already-computed candidate gradient places below the root threshold. The reported offset-quadratic reproducer exercises the J floor; the displacement guard is a second ordering path and should be covered too.

The candidate gradient is produced by `dual_merit_search`, which evaluates the objective, full gradient, Phi, and strict branch gate at the candidate and enforces both Armijo inequalities. This is adequate for the existing deterministic production callbacks, but the continuation API accepts injectable callbacks. A pending-audit result must not rely on a stale or inconsistent trial row when callbacks vary across calls.

## Minimal safe ordering

After a trial passes both Armijo checks and its strict branch gate, inspect its recorded full-gradient infinity norm before classifying displacement or cost stagnation. If it meets the root threshold, treat it as a provisional audit candidate: run the same candidate closure and deadline checks as an ordinary commit, and only then advance the accepted point and mark `root_pending_audit`. Keep `eligible_stationary_point=False` and `full_root_claim=False`. For candidates above the threshold, retain the current displacement and numerical-decrease refusals unchanged.

The root path may bypass only the “numerically zero decrease” bookkeeping guard. It must not bypass Armijo acceptance, finite-value/callback contracts, the full strict branch/margin test, candidate closure, fixed-input/source integrity, or cooperative deadline checks. A candidate with significant positive J change should be refused; it must not reach pending audit merely because its gradient is small.

## RED cases the regression should distinguish

1. **Offset invariance:** for `J(c)=C+0.5||c||^2`, use multiple constants C and a step to the exact minimizer. Each otherwise-valid candidate should reach `root_pending_audit`; it must remain ineligible and make no full-root claim. This isolates the cost-floor defect without changing the objective or tolerance.
2. **Displacement floor:** choose a non-root base and a candidate whose control displacement is under the numerical displacement floor but whose freshly evaluated gradient is under the root threshold. It should take the pending-audit route. Keep a nearby case with candidate gradient above threshold and confirm that it still stops as displacement stagnation.
3. **Callback consistency / re-evaluation:** make the trial callback report a root gradient but have fresh candidate evaluation disagree. The candidate must not be certified as pending audit; fail closed with the candidate uncommitted. Production closure should re-evaluate J and g at the exact candidate control (and compare against the accepted trial receipt within its declared precision policy), or otherwise establish equivalent immutable callback identity.
4. **Branch failure:** a candidate with a root-sized gradient but failed or changed strict branch/margins must not enter pending audit. Preserve the prior accepted control and identify the branch refusal.
5. **Cost increase:** return a candidate objective above the base by a numerically significant amount. Armijo should reject it; verify no pending-audit state is written. Do not replace the original J, weaken its Armijo coefficient, or add a tolerance to conceal the increase.
6. **Closure and deadline:** if candidate closure fails or the deadline expires during/after closure, leave the candidate uncommitted and preserve the prior accepted point. Pending audit must be written only after closure and the final deadline check succeed.
7. **Prior commits:** make an earlier ordinary step commit, then have the next root candidate fail re-evaluation, branch, closure, or deadline. The earlier committed iteration/control must remain intact; only the current provisional candidate is discarded.
8. **No silent audit promotion:** every root-sized candidate must still have `eligible_stationary_point=False`, `full_root_claim=False`, and an explicit audit-pending status. This path does not perform curvature, adjoint, or forecast qualification.

## Review boundary

The observed P2 concerns ordering after an actually accepted trial; it does not invalidate the two recorded PR #252 FV steps. A minimal fixture regression is sufficient for this ordering fix. Follow-on continuation from the final accepted point should remain under its separately declared new plan and existing time/HVP limits; this review does not authorize extending a budget or restarting the long FV run.

## Current verdict

P2 was confirmed in the original inspected source. No implementation was edited here. The initial diff review below was provisional and is superseded by the stable implementation review in [PR252_ROOT_RED_FINAL_20261007.md](PR252_ROOT_RED_FINAL_20261007.md).

## Review of the first implementation draft

The draft diff moves the candidate-gradient check before both stagnation guards and preserves closure-before-commit. This addresses the direct cost-floor ordering and adds a guard against a reported J increase beyond the same roundoff budget. However, this draft is not yet sufficient for RED approval:

- The root decision still trusts the one gradient stored by `dual_merit_search`. The production `commit_candidate` rechecks candidate branch metadata and fixed-input/source identity, but it does not re-evaluate candidate J/g; its `_fresh_merit` comparison is for the starting point. A stateful callback can therefore provide a root-sized gradient for the trial receipt and a non-root gradient on re-evaluation without the current closure detecting the mismatch.
- The added `gradient` failure fixture makes the *trial* gradient non-root, so it does not simulate an accepted root-sized receipt that fails a fresh candidate gradient check.
- The `1e-7` and `1e-8` quadratic trial displacements exceed `128 * EPS * scale`; the draft has no focused case exercising the new root exception to displacement stagnation or confirming that a sub-floor non-root displacement still refuses.
- The existing commit-failure test protects an earlier ordinary accepted step, but there is no sequence where a prior commit is followed by a root candidate that fails closure/re-evaluation/deadline.

The Armijo test correctly prevents a deterministic objective callback from accepting a significant increase. The explicit `significant_j_increase` guard is thus mainly defensive against receipt/callback inconsistency; a fresh-J comparison is needed to make that defense meaningful. This assessment applied to the first implementation draft. The stable patch and final focused evidence were reviewed separately; see [PR252_ROOT_RED_FINAL_20261007.md](PR252_ROOT_RED_FINAL_20261007.md) for the final disposition.
