# Tangent precondition comparison — final GREEN preflight — 2026-10-09

## Disposition

The final source-bound plan and current base loader are internally consistent, and the direction/model equations match the fixed 33cb point contract. The prior tolerance concerns are resolved in source: the residual parity budget accounts for the standardized observation/whitening operations, and the projected-row parity budget includes ambient as well as projected row norms. The paired comparison remains a preconditioned-gradient experiment, not a Newton or curvature certificate. This review used only source, frozen plan, and saved base receipts; it did not run tests, seed preparation, FV, gradients, HVPs, or a guard.

## Frozen plan and base provenance

I loaded the final plan through the read-only adapter loader. Plan SHA-256 is `be2886f2686cd04e71d3d244967da378bae9c22e39168c5f22bd85e7801169a4`; all **131 source pins** and **116 archive pins** match local bytes. The read-only base loader closes at control `33cb86ca73a404e6ff9260acbf63ee97f248ac0c1f529427eb1af1ec85a43586`, theta `0.3295008210283633`, native `J=0.06124426330405451`, and `F²=0.005145254919599806` from the saved resume endpoint. It does not regenerate the FV output.

The new plan ties its method and policy to that exact base, with 12 residual rows, 24 scalar VJP rows, four fresh direction-specific HVPs, two arms, 16 backtracking slots per arm, one selected commit, 600-second internal / 660-second outer limits, and 1 GiB sampled RSS. Old PR267/PR268 HVPs are outside the new adapter's operator path.

## Mathematical and contract checks

The refactored `FVPointResearchProblem._observation_residual_groups` is the same source used by `objective`: echo frames are converted to dBZ, sampled at the fixed points, quality/std normalized, and correlation-whitened before pseudo-Huber cost. The comparison gates the fixed `(3,4)` all-detected point profile, identity control-prior residual, no neural prior, and zero field-smoothness weight. The 12 scalar VJP rows therefore match the exact dimensionless cost residual components used by `J`; pseudo-Huber uses the exact `ell''=(delta/hypot(delta,r))^3` outer curvature.

For `P=I−nu nuᵀ`, the data term `sum ell''(r_k)(P a_k)^2`, projected identity prior, and diagonal `−mu P Q_cc P` formula are correct for the stated projected robust-GN Jacobi metric. The floor is `max(1,D_raw)`, tied to the existing unit quadratic prior; it keeps `W` finite, positive, and at most one. Field-smoothness curvature is zero for this exact case. The source does not claim that the floored diagonal approximates every term of the true constrained Hessian or that the true Hessian is SPD.

The residual-pair roundoff helper now scales by the standardized observation/prediction construction and whitener, rather than by near-zero residual magnitude alone ([fv_point_3h_tangent_precondition_comparison.py](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_tangent_precondition_comparison.py:82)). The side-row comparison budget includes ambient row norms to cover projection cancellation ([same source](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_tangent_precondition_comparison.py:121)). Both retain FP64-relative tolerances. The current tangent jump check, direction/chart coherence check, actual nonlinear Q=0 chart candidate, actual path radius, theta bounds, dual J/F² Armijo, current pair, and individual-gradient final closure remain active.

The shared kernel computes both directions at the same base; each arm gets its own `-1/+1` current HVP pair tagged with its direction digest. Separate candidate evaluators bind each arm to its own HVP model. Both first-pass-or-grid-exhausted searches finish before selection by actual endpoint F², then J within the 128-epsilon tie rule, then baseline. Only the winner receives an independent final repeat and commit. A second-arm interruption preserves the original 33cb point.

## Verification record and remaining limit

Saved verification reports 124 passing focused tests, 18 existing TorchScript warnings, and 15.63 seconds; type checking reports 0 errors. I inspected these logs without rerunning them. Tests now cover the residual scale budget, ambient projection-cancellation budget, dense formula agreement, exact identity-prior baseline behavior, direction/HVP mismatch refusal, tie order, and paired-arm interruption/no-commit behavior.

The parent runner validates exactly two arms and at most 16 trials per arm, direction-tagged HVP pairs, and a total of 24 completed row VJPs. One small receipt-hardening opportunity remains: its final closure checks row-history length/status, but does not independently require 12 distinct row indices for each side at the fixed base/theta. The pinned row builder currently enumerates those rows deterministically, so this is not a flaw in the current computation; adding the explicit side×row coverage check would make the saved receipt self-validating if that contract is needed for launch review.

No production comparison was executed by this review. Root remains the sole launcher for any subsequent guarded run.
