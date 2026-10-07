# d31 inexact-Newton comparison — RED design

Date: 2026-10-07
Scope: adversarial design review only. No source edit, test run, FV/HVP/PCG execution, forecast, adjoint, or reanalysis was performed.

## Disposition

The proposed forcing rule is mathematically usable for a **bounded comparison**, provided its contract is stated as a relative true-residual target for the current Hessian:

\[
\eta_k=\min(10^{-3},\max(10^{-10},0.1\|g_k\|_\infty)),\qquad
\|H_k s_k+g_k\|_2\leq\eta_k\|g_k\|_2.
\]

At the recorded d31 endpoint, \(\|g\|_\infty=0.4267256181\), so this rule starts at \(\eta=10^{-3}\), ten million times looser than the current \(10^{-10}\) PCG threshold. It preserves the local gradient-merit descent bound only when the **fresh true residual** satisfies the inequality and \(\eta<1\):

\[
D\Phi[c;s]=g^THs=-\|g\|_2^2+g^Tr\leq-(1-\eta)\|g\|_2^2<0.
\]

That bound alone is insufficient for acceptance. Keep the independent checks below; do not label the rule a convergence improvement based on a single step.

## Required contract and failure modes

1. **Solve the current operator.** At every accepted endpoint rebuild HVPs from that endpoint and solve \(H_k s_k=-g_k\). The archived f82c matrix may remain a preconditioner only. Do not reuse the 90fc or d31 direction, partial PCG state, or cached `Hs`.
2. **Check the true residual independently.** After PCG stops, make a fresh current-point HVP for `Hs`, compute `r=Hs+g`, and enforce `||r||/||g|| <= eta_k`. Record the requested eta, PCG's internal residual, fresh residual, and HVP count separately. A recursive PCG residual alone can drift from the true residual.
3. **Require both directional slopes.** Retain the existing rounded-dot guard for `g·s < 0`; also require fresh `g·Hs < 0` beyond its own scale-aware rounding allowance. The latter is the direct first-order slope of Phi. A passing residual bound mathematically implies it in exact arithmetic, but explicit checks catch implementation, accounting, or finite-precision inconsistencies.
4. **Keep curvature refusal semantics.** PCG's positive `p·Hp` observations cover only directions visited before stopping. A loose target can stop before later Krylov directions expose nonpositive curvature. Refuse if any observed PCG curvature test fails; otherwise report “no nonpositive curvature observed on visited products,” never “H is SPD” or a curvature certificate.
5. **Keep actual nonlinear acceptance unchanged.** Preserve the existing J-Armijo and Phi-Armijo constants, candidate grid/radius, strict complete endpoint branch and margin checks, fresh candidate J/gradient/Phi, commit/closure, input/source identity, deadline, and HVP/resource caps. If the endpoint signature changes, record it as a branch change; pointwise endpoint checks do not prove a smooth path. No alpha should be accepted from Taylor prediction alone.
6. **Keep root eligibility fixed.** The full-gradient root threshold stays `1e-10`; `eta` controls only the inner linear solve. Neither a smaller residual target nor an accepted Armijo step qualifies a stationary point, final curvature, adjoint, reanalysis, or forecast score.

The principal numerical risk is not loss of the residual-based local Phi descent guarantee; it is that a much less accurate direction has a different nonlinear trajectory, changes which backtracking candidate is selected, or fails both-merit acceptance. This is plausible here: the previous 90fc→d31 accepted step had a 49.65% gradient linearization relative error and a changed endpoint branch signature despite a `1.31e-11` true linear residual. That observation motivates the comparison; it does **not** predict d31's new direction or justify assuming the inexact run will be cheaper or accepted.

The cost risk is also real: PCG may still require many products because the preconditioner is fixed from f82c and the d31 current Hessian/coupling changed. A looser residual can save iterations, save none, or return a direction rejected by the two merit tests. Count HVPs (including restarts and the independent true-residual HVP) and elapsed time; do not estimate time savings from PCG iteration count alone.

## Smallest interpretable experiment

Use two fresh, separately guarded one-step runs from the **same validated d31 checkpoint**:

| Arm | Inner target | Other controls |
|---|---|---|
| Strict baseline | Existing relative tolerance `1e-10` | Current code/policy |
| Inexact candidate | Frozen rule above, computed from that run's fresh base gradient | Same code path and all acceptance/resource policy |

Set a frozen `plan.base_step` reference to the completed PR254 run receipt that owns d31. Resolve the endpoint through a reusable completed-run loader: validate child/parent/resource linkage, plan and source pins, fixed-input/runtime identity, and that the chosen control/J/gradient/Phi/branch exactly match the final committed accepted trial. The loader must reject provisional trials and partial-solve state. Do not hardcode a new base profile into each adapter. Keep the older 90fc legacy parent with its preserved historical `failed` classification and its narrowly pinned offline classification exception as a separate predecessor case; do not generalize that exception to arbitrary failed parents.

For both arms, allow **at most one accepted correction**, with the same existing HVP ceiling, PCG iteration ceiling, 720-second cooperative deadline, outer guard, and memory guard. Start each from d31 and rebuild fresh HVPs; no warm-start or partial-solve reuse. This is enough to compare inner work and whether the first nonlinear correction survives the existing gates. It is not enough to claim better convergence. If either arm hits a budget/refusal or produces no accepted endpoint, report that outcome and costs; do not extend only that arm or compare endpoint reduction as though both completed.

Record per arm: exact start-control hash; eta and requested residual target; fresh true residual; PCG iterations and all HVP counts; elapsed time and peak sampled RSS; `g·s`, `g·Hs`; first accepted alpha or refusal; J and Phi before/after; full-gradient norms and block norms; base/candidate branch signatures and strict margins; and closure/source/input identity result. A successful inexact result supports only a one-step cost-versus-progress comparison at d31. It does not validate eta adaptation near the root or the next-point HVP budget.

## Evidence anchors

- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: shared continuation, current-point HVP and accepted endpoint revalidation.
- `examples/weather_scenarios/fv_point_3h_dual_merit_step.py`: PCG true-residual and rounded `g·s` gates; actual J/Phi Armijo and strict candidate checks.
- `examples/weather_scenarios/fv_point_3h_schur_newton_step.py`: current HVP PCG solve and independent true-residual recomputation.
- `examples/weather_scenarios/fv_point_3h_90fc_followup.py`: current hardcoded historical base loader and preserved legacy classification exception; this is the pattern to generalize through `plan.base_step`, not copy as another per-run profile.
- `graphify-out/fv-root-cause-20260919/90FC_DUAL_MERIT_FOLLOWUP_PLAN_20261007.json` and `90fc_dual_followup_20261007_attempt1/step.json`: frozen prior policy and d31 accepted endpoint.
- `graphify-out/fv-root-cause-20260919/PR254_TEAM_RECORDS_20261007.md`: accepted endpoint branch-signature change and Taylor-model limitations.

## RED decision

Proceed only as a new, predeclared one-step paired comparison with the residual, slope, observed-curvature, dual-merit, branch, and closure gates above. Keep the strict run as the control and preserve both receipts. Do not alter prior records or the final `1e-10` root threshold. No new mathematical blocker was found in the forcing rule itself; the risks are testable direction quality, nonlinear/branch behavior, and uncertain HVP savings.
