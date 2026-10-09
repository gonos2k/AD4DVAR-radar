# Tangent preconditioner implementation — GREEN read-only review — 2026-10-09

## Scope

I reviewed the in-progress precondition comparison adapter, shared tangent-kernel changes, and residual/objective factoring. I did not run tests, source a fresh model residual, evaluate FV, compute HVPs, launch a guard, or edit source. The detailed mathematical proposal and source references are in `PRECOND_GREEN_DESIGN_20261009.md`.

## What matches the approved design

- `FVPointResearchProblem` now exposes residual groups used by the scalar objective; the all-valid branch preserves the exact quality-standardized, per-frame correlation-whitened dBZ residual used by `J`. The fixed point profile is 12 observations, all detected, no neural prior, and field-smoothness weight zero.
- `residual_jacobian_rows` uses one VJP of the residual vector per side and records twelve scalar pullback rows for each side. The preconditioner uses ℓ″=(δ/hypot(δ,r))³, projected side rows, unit identity-prior curvature, exact projected diagonal face curvature, and `W_i=1/max(1,D_raw_i)`. This creates a positive finite search metric and does not claim positive curvature of the true Hessian.
- Both baseline and robust-GN-Jacobi directions are checked against the current chart. They each get a separately tagged fresh `-1/+1` HVP pair, a separate bounded candidate search, and actual native-J/F²/branch/face checks. Winner selection uses actual candidate F², then J within the declared 128-epsilon tie tolerance, then baseline. Only the selected candidate gets the fresh independent final repeat and commit; an incomplete second arm leaves the start point uncommitted.
- The saved-base loader is source-bound to the accepted 33cb endpoint, and the new path passes the method context through the shared runner without changing the old resume plan or objective. The output records the per-arm diagnostics, 24 row VJPs, four current HVPs, and direction hashes.

## Actionable pre-freeze tolerance issue

In `examples/weather_scenarios/fv_point_3h_tangent_precondition_comparison.py:401-406`, residual-side equality tolerance scales only with the largest residual value. When a residual vector is close to zero, forward/whitening roundoff is governed by the magnitude of the standardized prediction/observation operations, so this can reject equal face-restricted residuals on an unnecessarily small relative tolerance. Use a documented component scale from the standardization and whitener operation, or a production-path equality comparison with an epsilon budget tied to those operands.

At `:414-422`, each projected-row SPI budget uses only `||P a_minus||` and `||P a_plus||`. The matrix projection can cancel a large normal row down to a small tangent row; roundoff in that cancellation scales with the full row norms. Scale the SPI budget with the ambient row norms as well as projected norms, preserving the FP64-relative bound rather than adding a fixed large tolerance. Otherwise an exactly face-supported row with near-zero tangent derivative can be rejected from roundoff.

The new unit tests cover the dense diagonal formula, exact residual factoring/whitening, VJP rows, two-arm HVP tags, selected commit, and interrupted second-arm no-commit. I did not find tests for those near-zero residual/SPI tolerance cases or for the 128-epsilon F²/J/baseline tie order. Add focused non-production cases before source freeze.

## Remaining verification boundary

The code checks both row groups, source/base closure, budgets, and endpoint acceptance, but no production row Jacobians or current-point HVPs were run in this review. Plan/resource completion and the real same-point F² winner remain for the root-owned guarded comparison and its saved receipts. The metric is a projected robust-GN Jacobi search direction, not a Newton step or minimum certificate.
