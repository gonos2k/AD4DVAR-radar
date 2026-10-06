# PR247 final GREEN delta review — 2026-10-06

**Disposition: GO.** The final delta closes the cached-curvature source-role gap, tightens resource-result classification, and improves physical-unit metadata. I found no blocking discrepancy in the reviewed changes or saved verification.

## Source roles and cached curvature

In `fv_point_3h_current_newton_step.py:28–37`, `_require_cached_objective_sources` binds the original-R9 objective manifest itself to `base["source_before"]`, then requires both the current plan and the curvature audit’s historical source map to equal each digest in that immutable manifest. This prevents a new plan from making changed objective code appear compatible with an old Hessian merely by repinning current files, and rejects missing proof from either map. I checked the saved R9 manifest’s 64 objective dependencies: all are present with matching hashes in the fresh-curvature audit’s historical source map. The plan still pins every current file for run integrity, so caller/test/resource-helper changes can receive current hashes without being mistaken for changes to the Hessian operator. The focused tests cover unchanged-objective reuse, rejection of a repinned changed objective, and rejection when an objective dependency is absent from both maps.

## Resource classification

`fv_point_3h_bounded_coupled_step.py:469–489` now requires declared wall/RSS limits and a present nonnegative integer sampled peak before classifying execution. A valid peak or elapsed duration above the declared cap is reported as `resource_limited`; malformed/missing limits or peak, monitor faults, and cleanup failures remain `failed`. This matches the guard’s sampled RSS record and avoids reporting over-limit success. The changes to fixtures cover these branches.

## Units and scientific interpretation

The added labels in `fv_point_3h_current_newton_step.py:197–203` are consistent with the code: dBZ is labeled as dBZ; `dbz_to_echo` computes `10^(dBZ/10) - 10^(min_dbz/10)`, so “linear reflectivity proxy Z minus Z_min” is accurate; face flux is the streamfunction difference and has configured-coordinate area/time units in the 2-D model with no depth coordinate; the coefficient label remains conditional on fixed coefficient-limit and basis normalization; normal speed is face flux divided by normal spacing; and growth is the configured log echo increment per 600-second interval. The existing scope continues to rule out claims about water mass, observed air wind, physical validation, and forecast skill.

The saved one-step JSON predates these metadata-only additions and was intentionally not regenerated in this review; it therefore does not itself contain the new unit fields. This limits the claim to the current code’s receipt schema and formula review, and does not change the archived numerical evidence.

## Verification reviewed

The current recorded focused suite reports `52 passed in 2.13s`; the recorded type check reports `0 errors, 0 warnings, 0 notes`. Current SHA-256 pins: Newton caller `b03c082d…fda666`, focused test `36d12fc9…e6172b`, R9 objective manifest `3e2e1e24…1fd8385`, fresh-curvature audit `3058fec5…9f2255`, test log `3a2e516f…1059fe`, and type log `56882e01…fbd91c`. I did not run tests, types, FV, or HVP during this read-only delta review. The old plan/raw run was not rewritten or rerun.

No actionable correctness issue remains in this delta.
