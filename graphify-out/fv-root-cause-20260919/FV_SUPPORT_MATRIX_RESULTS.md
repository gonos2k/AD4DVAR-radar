# G8 archived support/refusal matrix

The read-only tool `examples/weather_scenarios/fv_support_matrix.py` used the
four profile strata and six case IDs fixed in `FV_SUPPORT_MATRIX_PLAN.md`.
It ran **no** FV trajectory, optimizer, tangent, adjoint or reanalysis. The
cases already existed, so this is a retrospective inventory, **not** a
general radar eligibility or success-rate estimate. No rates are pooled
across profiles with different objectives, observation types or gate
opportunities.

| Profile and declared cases | Execution evidence | Forecast | Stationarity / strict branch | Local response | Independent response / physical validation |
|---|---|---|---|---|---|
| 8×10 full-support, seed A/B (2) | certified completed 2 | available 2 | passed / pointwise passed 2 | computed 2 | neither measured in this execution cohort |
| 4×5 missing point observations (1) | result recorded, no separate child exit record | available 1 | not assessed / pointwise passed | not attempted | not performed |
| 4×5 regular 3-hour, zero/nonzero flow (2) | result recorded under local alarm, no separate child exit record | same-operator synthetic forward available 2 | not assessed / strict branch refused 2 | not attempted | not performed |
| 4×5 partial two-hole (1) | process exit 1 | final forecast not certified | stationarity not established because Newton refinement was refused / search-policy branch refusal | no adjoint or response | response not established; physical validation not performed |

The emitted JSON also gives **within-profile** fractions with distinct
denominators. For the FV86 two-start cohort, locally response-eligible and
response-computed evidence covers 2/2 declared starts. The two long-horizon
cases are 0/2 eligible **under the current strict branch policy**, while the
point-missing case has no stationary eligibility assessment. These values are
not pooled or interpreted as general radar success rates. Independent
validation coverage is 0/N because no row in these original execution
records contains a passing independent validation, but its
`pass_fraction_given_attempt` is `null` because the attempt count is zero.
Coverage describes available evidence, not a measured failure rate.

The two 86-control starts came from one **predeclared two-start cohort**. Its
original report has completed/locally eligible 2/2 but no signed reanalysis
in that run. The later seed-A signed reanalysis is a separate experiment and
does not turn the earlier 2/2 into an independent validation rate. The
partial two-hole case had three bounded attempts at the **same input**; they
count as one case here, not three independent failures. Its final run
returned a nonzero process exit and refused the nominal Newton correction
before tangent/adjoint/endpoints. That is a policy-local refusal, not proof
that the physical inverse problem lacks a stationary root.

Every row retains execution, forecast, stationarity, branch, response,
response-validation, physical-validation and resource states separately.
The output records each source artifact SHA-256 and its provenance level.
The FV86 summary's raw artifacts and every listed point/long source and
measurement manifest hash are verified before reading their classifications.
The partial run-3 output manifest, preflight hash and archived source-hash map
are also checked. Its historical final source check is incomplete, so the row is marked
`preflight_bound_nonzero_exit_no_final_source_check`. `recorded_only` is
deliberately weaker than a captured zero child exit plus completed source
check. A conditional validation pass fraction is JSON `null` when no
independent validation was attempted; it is not 0% or 100%.

The report is `fv_support_matrix.json`. The affected accounting tests report
**18 passed** and pinned typecheck on the new tool/test reports 0 errors,
warnings and notes. Exact commands and source hashes are in
`fv_support_matrix_manifest.json`.

This completes a **retrospective stratified status inventory**. G8 remains
open for a prospective radar-condition matrix with cases and resource caps
frozen before execution, explicit observation/support/domain identities,
captured exits for every producer, refusal reasons, and independent response
and physical validation on the same declared denominator. The inventory
does not license a cross-profile success fraction.
