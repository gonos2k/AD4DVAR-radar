# Algorithm extension — approved 2026-09-19

User approved the computational/token expansion and then explicitly postponed
CI/deployment until algorithm completion. The current PR CI was cancelled;
local numerical experiments continue. PR remains open while required numerical checkpoints are unresolved.

## Governing and discrete contracts

For the echo proxy q, use q_t + div(u q) = g q, with prescribed inflow traces,
interior outflow traces, and shared face fluxes. The donorcell FV operator and
SSPRK2 retain nonnegative coefficients under their CFL bound. Growth uses its
existing integrating factor. q represents known whole-cell contributions;
support is a separate transported quantity, not a denominator or a second
multiplier of already weighted boundary echo. Missing data is not clear sky.

On a fixed differentiable contract, F(c,y,theta)=J_c=0 and
H=J_cc. The stationary response is E_y-J_cy^T H^{-T}E_c; learned statistics use
E_theta-J_c_theta^T H^{-T}E_c. Positive curvature on PCG directions alone does
not prove global positive definiteness. Actual linear residuals and nonlinear
reanalysis differences must be measured.

## Checkpoints and bounded work

1. **General FV local response (GREEN):** fixed missing/detection masks and
   partial-support domains. Restrict comparisons to valid known contributions;
   do not reinterpret unknown mass as zero total echo. Keep face-switch and
   domain-threshold ambiguity fail-closed. Compare exact response with small
   reanalysis finite differences. Existing transport general-boundary tests
   alone do not prove a valid general assimilation likelihood.
2. **240×240 FSOI (main):** reuse the saved analysis. Refine grad J using an
   exact-Hessian matrix-free Newton solve and actual gradient-norm decrease,
   avoiding unresolved cost subtraction. Test small negative-curvature rejection.
   Then calculate an exact adjoint on the same fixed verification function and
   compare it with finite observation perturbations and reanalyses. Each large
   child has an explicit time/RSS bound; no dense 57,604-square Hessian.
3. **Persistent neural learning (RED):** three fixed updates over distinct
   training windows; save/resume model and optimizer state mid-sequence; compare
   to uninterrupted execution. Held-out data is fixed and never used for update
   acceptance, stepsize selection or gradient evaluation. Confirm the actual
   held-out reanalysis change, not just an optimizer parameter change.
4. **Integration:** report failures without changing gates to make them pass;
   expose actual results and domain limits in the existing HTML. Reuse completed
   evidence and run only affected numerical regressions.

## Current evidence

- Matrix-free local root refinement: convergence, negative-curvature refusal and
  sensitive-face-tie refusal passed. Together with replay trajectory checks,
  15 tests passed in 18.14s. Replay comparisons include 9 substeps (8+1 chunks),
  forward, JVP, VJP and mixed second derivatives. A positive initial-observation
  diagonal only preconditions PCG; it does not replace the exact Hessian.
- First large refinement: failed its four-iteration budget, 805.16s / 9.649GB
  sampled peak RSS. Its old error did not preserve the final gradient; no claim
  is made about that unknown value. Preserved in *_unpreconditioned.{json,log}.
- Chunked replay bounds each reverse tape to 8 substeps, retaining the discrete
  equation, dt, stages and traces. A strict diagnostic run used 278.47s / 1.807GB
  peak. Norms: 6.72655e-7 -> 4.10009e-8 -> 2.08722e-9 -> 2.07178e-9;
  it failed the additional 1e-9 norm target on further backtracking. Face margins
  remained >0.2756. Preserved in *_strict_diagnostic.{json,log}.
- The existing response API gate is max(abs(grad J)) <= 1e-8. Save/reproduce with
  norm(grad J) <= 1e-8, which is stronger than that unchanged gate. This is not a
  claimed proof of a floating-point lower bound or a minimum. Actual adjoint and
  finite reanalysis checks remain required. The rerun is bounded at 600s/10GiB.
- Persistent neural learning: three updates, two distinct fixed training windows;
  heldout gain 1.1142961e-9, model/optimizer/forecast resume differences exactly 0.
  Actual bounded run: 538.80s / 333MB. This is small synthetic observation-error
  learning, not 240x240 learning or background-neural-prior evidence.
- General-support semantic closure and final GREEN/RED review remain active.

Background-neural-prior integration is a different chain from the current
observation-error neural model. It remains explicitly separate; neither an
observation-error update nor its replay removes the existing prior-support gate.

## Executed follow-up

- Nominal 240×240 refinement succeeded: gradient norm 3.89939e-9, maximum
  3.84035e-9; 121.08 s / 1.815 GB sampled peak RSS. This satisfies the existing
  response gate without changing its tolerance.
- Exact robust-Hessian response over all 18 leads succeeded: actual adjoint
  relative residual 4.28390e-11, 41 products; 461.12 s / 1.933 GB.
- First smooth 0.01 dBZ reanalysis failed: L2 residual reached 1.3584e-8 then
  stalled, 700.18 s / 1.875 GB. Maximum component and final control were not
  retained; do not classify it as stationary or as a successful FSOI check.
  Preserve failure evidence and save accepted iterates in subsequent probes.
- The root helper will use the existing API's maximum-component stopping rule,
  while retaining L2 residual decrease for line search. Smaller declared Taylor
  steps (0.001 and 0.0005 dBZ) test the local derivative without changing gates.
- Prescribed-flow grid convergence passed on translation, rotation and strain.
  Final 128-grid JVP/oracle relative errors remain about 10–13%; decreasing error
  supports consistency but is not high-accuracy or general P1 evidence.


## Historical source and regression continuity (before small-growth correction)

The measured nominal response is retained as historical evidence with its exact
source in `nominal_source/fv_sensitivity.py` (SHA-256 `69bbf85a...`). An AST
comparison against the current module differs only in `refine_fv_stationarity`;
the response definitions are identical, and transport/variational hashes match
those recorded by the run. `rotation240_source_continuity.json` also binds the
saved response and refined checkpoints. This avoids presenting old hashes as a
new run or repeating an unchanged 461-second adjoint solely for metadata.

Additional affected tests: 42 passed in 71.74 s; root-refinement tests: 4 passed
in 13.73 s. GREEN/RED found no new stencil or replay-chunk defect. RED's nominal
source-binding concern is resolved by the exact recovered source and explicit
historical label. Finite reanalysis remains an independent unfinished checkpoint.


## Second large perturbation and learned-background review

The 0.001 dBZ reanalysis used all four root corrections, reaching maximum
1.30578e-8 (L2 1.66988e-8), still above the unchanged 1e-8 gate. Run time was
612.48 s / 1.930 GB; the 0.0005 case was not reached. Its accepted controls
are now retained, so subsequent diagnostics do not need to repeat the solve.
The dominant component is log growth. A one-ULP probe rules out a simple
one-ULP control-spacing explanation; exponential scaling and gradient evaluation
roundoff are being investigated. This is not a declared numerical lower bound.

A nonzero-theta external learned background verifies a nonidentity cotangent
contribution and the complete fixed-support chain. Its heldout score worsened
by 2.16e-16 after the fixed update: no improvement claim is made. This does not
extend the typed neural-prior or legacy FSOI contract.


## Small-growth follow-up

The growth equation is unchanged. Physical products use q+q*expm1(g) via
addcmul for |g|<1/8, retaining the original exponential guards and fallback.
This resolves the independent sub-ULP zero-flux counterexample without relaxing
the stationary response gate. See GROWTH_SCALING_RESULTS.md and
GROWTH_SCALING_REVIEW.md for the candidate history and affected regressions.
Current nominal response: max gradient 5.73043e-10, actual adjoint residual
4.28392e-11, 41 products, 490.42 s / 1.947 GB. Finite reanalysis is checked
separately; an accepted root alone is not sufficient evidence of FSOI accuracy.

Next learned-background checkpoint: reuse the existing response solver for an
explicit differentiable B(y,theta), with fixed support and precision and both
direct and implicit y/theta terms. Do not duplicate the Hessian solver or remove
typed-prior gates. The typed path snapshots outputs and also carries support
and uncertainty semantics; these dependencies must be represented before
claiming that path is integrated. External composition evidence remains valid
only within its stated fixed-support/precision scope.
