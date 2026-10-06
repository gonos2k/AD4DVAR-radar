# PR247 final RED fixes review

**Disposition: GO. No blocking issue found in the bounded fixes reviewed.** Review was read-only and did not run an FV trajectory or compute a new HVP.

## Cache-to-Hessian source binding

The new `_require_cached_objective_sources` call in `examples/weather_scenarios/fv_point_3h_current_newton_step.py` closes the reported gap. It requires the current plan to carry the historical R9 archive digest from the curvature receipt, reads the R9 `source_before` manifest, and requires both the current plan and historical curvature source map to equal each immutable R9 dependency digest before `_prepare_fixed_seed` or any objective/branch evaluation. Checking each against the manifest also rejects a dependency missing from both plan and historical map; `None == None` cannot pass. The refreshed Newton caller and its test can receive current plan hashes because they are not members of the R9 numerical-source manifest.

The regressions construct a refreshed plan from the current checkout, simulate a new `src/advar/variational.py` digest in both the checkout hash callback and plan, and remove that source from both maps. The changed-source and missing-proof cases are rejected; the companion verifies that a refreshed caller/test plan still loads when the historical objective sources are unchanged. These tests do not run FV work. The existing raw result and frozen old plan were left unmodified.

## Resource evidence and physical units

The reviewed plan fixes the outer wall limit at 300 seconds and RSS cap at 1 GiB. `main` passes those values to the guarded runner, and `search.execution_status` checks the recorded limits, termination state, sampled peak, and exit status. The archived execution used 18.66 seconds and a sampled peak of 358,252,544 bytes (about 342 MiB, 0.334 of the cap), with no termination, monitor error, or cleanup error. The record correctly calls this sampled monitoring rather than an OS hard allocation ceiling.

The added physical metadata is dimensionally consistent with the implementation: dBZ for `initial_dbz`; a linear reflectivity proxy with the `Z_min` offset removed for `initial_echo_proxy`; two-dimensional model-coordinate area per second for streamfunction face-flux differences when no depth is specified; coefficient units tied to fixed coefficient limits and basis normalization; configured model length per second for face-normal speeds; and dimensionless log echo growth per 600-second interval. These labels do not broaden the report's stated physical scope.

## Validation boundary

The recorded check logs show 52 tests passed in 2.13 seconds and type checking reported zero errors, warnings, or notes. I did not repeat those checks. Frozen plan SHA-256 remains `d7f639e82dcd9445eeef0e32472194b57f31088dc1247e22b52ab0acc37f4a8f`; it now has four stale working-tree source pins: `fv_point_3h_bounded_coupled_step.py`, `fv_point_3h_current_newton_step.py`, `test_f82c_curvature_experiment.py`, and `test_fv_point_3h_current_newton_step.py`. The raw attempt and old plan remain unchanged as historical evidence; rerunning the caller requires a new plan that pins the current source set and retains the same historical R9 dependency digests. The archived numerical status remains one accepted original-J step, with no stationarity, response, score, or new-curvature claim.

Evidence: `examples/weather_scenarios/fv_point_3h_current_newton_step.py:28–65,190–202`; `tests/test_fv_point_3h_current_newton_step.py:10–38`; `examples/weather_scenarios/fv_point_3h_bounded_coupled_step.py:469–489`; `examples/weather_scenarios/fv_diagnostic_guard.py:99–153`; archived run and resource records in `graphify-out/fv-root-cause-20260919/f82c_unshifted_newton_20261005_attempt1/`.
