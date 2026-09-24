# G8 archived-support accounting plan

This is a **retrospective, stratified inventory**, not a prospective sample
of radar conditions or a general success-rate estimate. The rows below are
fixed before generating this new summary. Repeated partial-refinement
attempts are history of one masked case, not three independent cases.

| Profile stratum | Case IDs | Source record | Why included |
|---|---|---|---|
| `fv86_full_support` | `seed_a`, `seed_b` | predeclared two-start FV86 execution summary and its raw reports | certified process/numerical/local-response reference |
| `point_missing_fixed_control` | `three_time_missing` | bounded point-missing metrics/manifest | valid-only observation and branch check without stationary analysis |
| `long_regular_forward` | `zero_flow`, `nonzero_flow` | bounded 18-lead metrics/manifest | forward available with strict branch refusal |
| `partial_two_hole` | `masked_case` | third bounded branch-gate attempt and resource record | keep a numerical refusal and nonzero process exit visible |

For each row, preserve distinct execution, forecast, stationarity, local
branch, local response, independent response validation, physical validation
and resource status. `completed` execution requires a captured zero process
exit and source-completion check; a successful JSON body without that evidence
is `recorded_only`. A process failure may still contain useful numerical
diagnostics but cannot count as a completed run. A local response is
`computed`, not independently validated. `not_performed` does not become
`failed`, `passed`, or a zero-percent validation rate.

Counts and fractions are **within each stratum only**. The declared row count
is the denominator for the corresponding stratum, including refusals. A
conditional pass fraction among attempts is null when no independent
validation was attempted. No cross-profile aggregate or between-profile
gradient comparison is allowed because units, objectives and opportunity to
reach each gate differ. The FV86 seed-A signed reanalysis is a later separate
experiment; it does not retroactively change the original two-start
execution summary's `response_validation=not_performed` fields.

The tool will verify the FV86 summary's pinned raw artifact hashes and every
listed point/long source and report hash from their manifests. The partial
run-3 output manifest, preflight hash and archived source-hash map are checked
before its saved artifact and resource hashes enter the new output; its historical
source-completion check is incomplete, so its provenance remains labeled as
preflight-bound rather than certified. No solver, optimizer or FV trajectory
will run in this accounting pass.
