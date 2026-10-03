# S4 accepted one-step endpoint audit

## Question and limit

At the single accepted endpoint of the archived one-step Newton experiment, do the original fixed problem's objective, full gradient, strict 3600-stage branch, and fresh exact Hessian reproduce? If so, what are the current 20-field/6-dynamics Hessian blocks and diagnostic Schur complement? This is a new-point relinearization. The prior Newton Hessian belongs to the old control and is not evidence about this endpoint.

The audit applies no step and does not optimize, compute a score/adjoint/response, certify stationarity, prove a root/minimum, or make physical claims. An Hff/Schur refusal is a valid terminal numerical outcome.

## Frozen inputs and validation

Read only `point_3h_schur_newton_step_attempt1/newton_step.json`, pinned at SHA-256 `b71411024191756d27bc30982800207a392e31d5b60862229ab24c181aa3fdbd`, and its preflight at `d757df0a311bca4f574a55492ee0fd38435e65310dd8bbeab0980bed88b8432a`. Require one-step accepted status, exactly one alpha-1 accepted trial, accepted control hash, accepted cached J/Phi/26-gradient, source/input/runtime before-after consistency, and unchanged original 13-parameter vector/hash. Rebuild the fixed original problem using the pinned S4 seed loader and verify its full input identity against the accepted record. Keep the original prior and time/model setup.

Check the strict branch and complete positive margins at the accepted control; bind its signature to the archived accepted branch. Recompute J, Phi, and the full gradient and compare each metric and gradient component to the archived accepted values using a declared FP64 component-scaled roundoff budget. Require a fresh branch replay after derivative work to retain the same strict signature and complete margins.

## Numerical procedure and limits

At the accepted control, compute 26 exact HVP columns and one independent HVP at the smallest-eigenvalue eigenvector. Require finite outputs, bounded Hessian symmetry, and a passing fresh eigenpair residual. Pass only this fresh Hessian and gradient into the frozen block/Schur diagnostic. Record the full matrix, eigenvalues, residual audit, and field/dynamics block results. Do not use the frozen-Hessian bytes or cached preconditioner from the one-step run as current curvature.

Allow up to 540 internal seconds, 600 seconds wall time, and 1 GiB sampled RSS, in one guarded child. A typed numerical qualification refusal is saved as a refusal with final source/input/runtime checks; programming or invariant errors fail the child. No retries or second numerical run are part of this plan.

## Provenance and verification

Pin this plan, producer, tests, accepted raw result, preflight, one-step plan, frozen S4 diagnostic, seed/tail helpers, and every original model/data source against before/after SHA maps. Pin runtime before and after. Preserve the original artifacts unchanged. Run only pure helper/wiring tests and offline error-level type checks before parent GREEN/RED review. A successful artifact supports only a fresh local block-curvature diagnosis at this one accepted point.
