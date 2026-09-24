# PR174 review resolution checklist

Source: user's PR174 general-application review (2026-09-24). Preserve the
completed 86-control milestone. Resolve concrete defects first, then separately
validate extensions; future physics/application scope is not a defect in the
already accepted fixed cases. Updated for the user's 15:31 JST checklist request.

| ID | Finding / task | Completion evidence | State |
|---|---|---|---|
| R1 | Nonzero process exit could count an eligible checkpoint as success | exit/phase/source/resource regressions; independent execution/numerical/validation fields | Done in PR175; 15 focused status tests within 158 |
| R2 | Two duplicated fixed-case bindings | common FVResearchProblem over existing structures; immutable-reference J/E/full partial gradients/HVP/branch parity; old profile gates retained | Done in PR175; limited supported contract |
| R3 | 86-control response lacks nonlinear reanalysis comparison | one fixed middle-time bias direction; predictor/corrector ± endpoints; two consecutive h pass unchanged stationarity/branch/relative-error gates | Done for seed-A middle-time direction; two local sizes pass; FV86_REANALYSIS_RESULTS.md |
| R4 | Observation count/space, timing and support must be explicit | layout/regular times/collocated dBZ operator/full-support constraints returned, unsupported variants rejected | Current supported scope explicit in PR175; arbitrary H/irregular times not supported |
| R5 | Numerical policy versus physical problem | reuse exact HVP, existing PCG/refiner/adjoint; no damping substitution or tolerance weakening | Preserved; adaptive tolerances/preconditioning remain separate optional studies |
| R6 | Response residual alone can hide cancellation | preserve direct/indirect/total terms; report actual adjoint residual and tangent residual; add r_lambda^T c_dot directional diagnostic in R3 if available | Done as a diagnostic for the declared direction only; FV86_REANALYSIS_RESULTS.md |
| R7 | Forecast availability is not sensitivity or physical validity | preserve original control/forecast on refusal; execution, numerical and independent-validation status separate | Existing wrapper + PR175 separation; physical validation still unperformed |
| R8 | Full problem/source/measurement identity | common fixed-input hash; bind runtime p/c/code/branch separately; preserve raw reports and old source snapshots | Implemented for fixed research records; R3 must bind its own run |
| R9 | Different supplied seeds do not prove unique independent solver paths | disclose standard product initial candidate search and uncaptured first iterate; avoid uniqueness/identifiability claims | Reporting resolved; capturing internal first iterate belongs to next multistart experiment |
| R10 | Maintain/decay/flat/zero-flux states | distinguish forward support from classical response support; predeclared derivative/refusal tests | Pending extension; do not remove strict guards |
| R11 | Missing observations versus unknown state/boundaries | first extend observation masks with completely known state/boundary; preserve censored/invalid meanings | Pending extension |
| R12 | Non-collocated/regridded observations and covariance | explicit H and whitening; preserve original information; subset R before inversion; product-defined averaging units | Pending design/analytic tests and implementation |
| R13 | Same inverse problem across grids | consistent cell averages, boundary/verification, observation information/prior; separate interior/boundary error | Pending distinct milestone; 86 execution does not close it |
| R14 | Operational/general numerical applicability | report support/branch/curvature/linear-budget/stationarity/resource refusals and eligible fractions over predefined cases | Implemented accounting for two fixed runs; broad validation matrix remains pending |
| R15 | Physical additions and independent forecast/learning skill | separate model-change validation and independent event/holdout evidence | Pending Phase 2/model work; no claim from current synthetic score |
| R16 | Concurrent diagnostics | replace temporary transport/PCG patching with explicit observer/context before parallel service use | Pending; all current research runs remain serial |

## Current authorized execution (R3/R6)

Follow FV86_REANALYSIS_PLAN.md exactly: seed-A nominal point, middle-time bias,
h=.001*2^-j for j=0..5, two consecutive relative differences <=1e-4, endpoint
max gradient <1e-10, tangent/adjoint actual relative residual <=1e-10, same
108-stage branch. Overall 1200-second wall / sampled 2-GiB budget. No fresh GN,
new direction, large-grid sweep, tolerance relaxation or realistic finite-impact
certification. A capped/refused result remains evidence and does not close R3.

## Closure rule

A checklist item is marked done only for its stated scope and linked evidence.
After R3, review the result before moving to another axis. Unperformed tests,
future plans and source-only reasoning must never be counted as completed
experiments. The prior full CPU/package regression and general D7 remain open.

R3/R6 closure: 309.273 seconds, 351,141,888 sampled RSS bytes, four corrected endpoints, two larger-step predictor refusals preserved. GREEN/RED reviewed the signed parameter/predictor arrays and numerical/source gates. Remaining rows retain their stated pending scope.
