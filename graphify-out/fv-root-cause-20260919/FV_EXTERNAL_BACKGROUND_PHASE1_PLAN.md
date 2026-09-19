# Bounded Phase 1 external-background proof

## Scope

This probe composes the existing 4x5 donor-cell FV objective/forecast with an
externally computed, three-parameter mean background

\[
 B_\theta(y)=y_0+0.1\tanh(\theta_0+\theta_1\tanh(y_0-Wy_0)
                         +\theta_2\tanh(y_0-Ny_0)),\quad\theta=(0.1,-0.2,0.15).
\]

The features use only spatial west/north differences of the first observation (the -20 minute background-valid time); no later observation is used, and the support mask,
observation standard deviation, transport, and branch policy remain fixed. The
composition injects `B_theta(y)` by replacing only
`FrozenOuterState.initial_background_dbz` inside each objective/score call.
There is no `NeuralPriorApplication`, no truth-seeded initialization, and no
legacy FSOI/learning-eligibility path.

## Chain-rule oracle

For a fixed stationary control `c*`, define `E` as the forecast score and `J`
as the robust analysis objective.  Solve
`H^T lambda = E_c`, with `H = J_cc`, then compute
`g_B = E_B - J_cB^T lambda`.  The predicted parameter and total-observation
responses are respectively `B_theta^T g_B` and
`E_y|B + B_y^T g_B`; direct finite reanalysis uses fresh local solves at each
perturbation.  The probe compares autodiff stationary responses, these explicit
chain terms, and centered finite reanalysis slopes.

## Learning/reload evidence

One fixed-size descent update is computed from the training response. A
separate deterministic heldout observation window is scored before/after its
own reanalysis. The updated three-parameter state is checkpointed, loaded into
a freshly constructed model, and reanalysed; scores/forecasts must match. The
report records local FP64 residuals and finite-difference errors, but makes no
global-minimum, typed-neural-prior, or FSOI claim.

## Acceptance

The bounded run must produce finite chain-rule residuals, fixed-support and
same-face checks, finite centered FD diagnostics for `y` and `theta`, and
checkpoint/reload equality. Heldout score change is reported without a directional assertion; a tiny or
negative change is retained as an inconclusive result.
