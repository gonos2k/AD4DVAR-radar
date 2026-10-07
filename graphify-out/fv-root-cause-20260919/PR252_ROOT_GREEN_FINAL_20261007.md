# PR #252 near-root cost-floor fix — final GREEN review

Date: 2026-10-07
Disposition: **GO for the narrow P2 correction**, subject to keeping the first commit limited to the continuation source and its focused regression file. No FV/HVP/PCG, forecast, adjoint, or reanalysis run was performed for this review.

## Finding and corrected behavior

The original P2 is reproduced in `PR252_ROOT_REPRO_20261007.json` against unmodified source commit `1758a6467132602a46b603fbb54758af13d41317`. For the consistent quadratic `J_C(c)=C+0.5||c||²`, the exact gradient, Hessian, and root do not depend on additive constant `C`, but the old absolute J floor did: `C=0` reached `root_pending_audit`; the three offset cases (`C=1` or `0.063`) with candidate gradient zero were discarded as `numerically_zero_decrease`.

The patched `run_iterations` in `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py` now treats a root-sized trial gradient as provisional. It freshly recomputes candidate J/g/Phi and the strict branch, checks the cooperative deadline, compares the fresh values to the trial receipt under the 128-epsilon consistency policy, requires the same branch signature and both original Armijo bounds, and fails closed if any check differs (current lines 288-338). Only a freshly confirmed root candidate can bypass the displacement and subtraction-scale decrease floors; a significant J increase remains refused. Non-root candidates retain both guards (334-362). The existing candidate closure and post-closure deadline check still precede state advancement (363-392). A successful path records `root_pending_audit`; it does not set `eligible_stationary_point` or `full_root_claim`.

The code-level regression covers the constant-offset family and pending-only status (test lines 81-105), a high-curvature exact root with sub-roundoff displacement (108-128), a non-root sub-roundoff move that remains `displacement_stagnation` (130-157), ordinary non-root cost-floor refusal (159-184), root re-evaluation/branch/closure failures (186-235), and stateful fresh-callback inconsistencies (237-280). The prior-commit rollback regression now runs from starting controls 0.10 and 0.15, confirming that a later provisional failure preserves the earlier accepted point (311-333).

## Verification evidence

- Pre-fix, independent quadratic replay: 4 cases in `PR252_ROOT_REPRO_20261007.json`; it reproduces the constant-offset defect using the actual repository PCG and an explicit strict-branch stub.
- Post-fix, independent quadratic replay: 5 cases in `PR252_ROOT_FIXED_CHECKS_20261007.json`; all five reach pending audit, the high-curvature case has `displacement_below_roundoff=true`, and every case retains `eligible_stationary_point=false`.
- Affected test log `PR252_ROOT_TESTS_20261007.log`: **46 passed, 18 warnings, 3.17s**. The warnings are existing TorchScript deprecations from `test_fv_point_3h_hvp_newton_step.py`.
- Type log `PR252_ROOT_TYPES_20261007.log`: **0 errors, 0 warnings, 0 notes**.
- Isolated Graphify record `PR252_ROOT_GRAPHIFY_20261007.json`: 46 nodes, 174 edges. The shared `graph.json` and `GRAPH_REPORT.md` hashes are identical before and after.

The source and focused test hashes checked against the isolated Graphify record are:

- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: `a9be7758a745f3b72423c2ab17d3af495cb39525440444c6a86d1e6b0c3abbdf`
- `tests/test_fv_point_3h_dual_merit_continuation.py`: `4051ea8ee8962ea139e013284ac398b708f61eee09b8d8c954b0b7cccd87dd9e`

## Review boundary

This closes the narrow root-candidate ordering defect with focused analytic evidence. It does not establish that the FV objective has a qualifying stationary point, does not alter the two accepted PR #252 FV steps, and does not establish convergence, curvature eligibility, adjoint correctness, reanalysis validity, or forecast skill. Keep the corrective commit limited to the two hashed code/test files before any separate connector or follow-up continuation changes.
