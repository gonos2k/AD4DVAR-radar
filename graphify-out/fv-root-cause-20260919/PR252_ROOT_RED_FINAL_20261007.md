# PR #252 near-root audit path — final RED review

Date: 2026-10-07
Scope: final read-only review of the P2 fix in `run_iterations`, its callback-only regressions, and saved verification receipts. No FV trajectory, new FV HVP/PCG solve, forecast, adjoint, reanalysis, or full repository test was launched for this review.

## Disposition

**GO for the bounded P2 correction and its focused integration scope.** The accepted candidate's root-sized gradient is now freshly checked at the exact candidate control before it can bypass the displacement and roundoff-decrease guards. The checks preserve the original objective, both Armijo thresholds, strict branch gate, closure, input/source integrity, and deadline behavior. The result remains `root_pending_audit`, with `eligible_stationary_point=False` and `full_root_claim=False`.

## What changed and why it closes the finding

In `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py::run_iterations`, after dual-merit search returns a trial with `accepted` status, the loop reads its full-gradient infinity norm. For a root-sized trial only, it freshly evaluates J, g, Phi, and the strict branch at the exact candidate. It checks finite callback contracts through `_true_merit`, scalar and gradient agreement against the search receipt at a 128-epsilon relative scale, branch strictness and signature agreement, both original Armijo inequalities, and the fresh gradient threshold.

Any mismatch produces `root_candidate_recheck_failed` and leaves the provisional candidate uncommitted. A fresh objective increase also fails the original Armijo check. A successful recheck may pass a roundoff-size control displacement or J/Phi difference to ordinary candidate closure; it cannot pass those guards if its gradient misses the declared threshold. Non-root candidates keep both original displacement and numerical-decrease refusals. Closure and a final deadline check still precede advancing the accepted point.

## RED failure paths reviewed

- **Offset-cost counterexample:** the original PR #252 source blob reproduced the issue for `C=1` and other nonzero offsets: exact-root trial receipt, zero commits, `numerically_zero_decrease`. The patched source reaches pending audit for all four reported consistent-quadratic cases.
- **Tiny displacement:** a steep quadratic with initial gradient above threshold and candidate displacement below `128 * EPS * scale` reaches pending audit only after the fresh root checks. A nearby candidate that remains above the gradient threshold is refused as `displacement_stagnation`.
- **Callback inconsistency:** stateful objective, gradient, and branch fixtures alter the second candidate evaluation. Each is refused as `root_candidate_recheck_failed`; no candidate commit callback runs.
- **Branch and closure failure:** strict branch failure blocks acceptance. Closure failure marks only the provisional trial `candidate_not_committed` and preserves any earlier accepted point.
- **Prior commit preservation:** the later-commit regression now covers both a subsequent non-root candidate and a subsequent root candidate whose closure fails; the first accepted control remains recorded.
- **Significant J increase:** fresh J must still satisfy the existing Armijo threshold. No cost-increase tolerance or non-root decrease floor was relaxed.
- **Eligibility boundary:** tests and the saved fixed-check receipt retain `eligible_stationary_point=False`; the correction only emits a pending audit state.

## Evidence inspected

- `PR252_ROOT_REPRO_20261007.json` records execution of the unchanged PR #252 source at Git commit `1758a6467132602a46b603fbb54758af13d41317` and confirms the four offset-quadratic outcomes. Its scope is a consistent FP64 quadratic, repository PCG, and explicit branch stub, not FV.
- `PR252_ROOT_FIXED_CHECKS_20261007.json` records the four corrected offset cases plus the high-curvature, sub-roundoff displacement root case. Each reaches `root_pending_audit`; none is eligible.
- `PR252_ROOT_TESTS_20261007.log` records **46 passed, 18 existing TorchScript deprecation warnings, 3.17 s**. `PR252_ROOT_TYPES_20261007.log` records **0 errors, warnings, or notes**. These are saved producer results; I did not rerun them.
- `PR252_ROOT_GRAPHIFY_20261007.json` records isolated changed-code AST refresh and unchanged shared `graph.json` and `GRAPH_REPORT.md` hashes. The current implementation and test SHA-256 values match that receipt.
- `git diff --check` returned clean during this review.

## Remaining limits

This closes the ordering and callback-consistency P2 at the focused-code level. It does not establish an eligible stationary point, Hessian definiteness, adjoint correctness, reanalysis validity, forecast accuracy, or convergence of a future continuation. The earlier PR #252 two-step FV result remains the only live continuation evidence; this P2 fix did not require rerunning it. A later run from the preserved accepted point should use its own pinned plan and already declared resource limits.
