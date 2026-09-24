# PR175/176 review resolution checklist

Source: user's PR175/176 generalization review (2026-09-24). This checklist
tracks each claim at its measured scope. The older PR174 items remain in
`PR174_REVIEW_RESOLUTION.md`; completion here does not imply arbitrary radar
inputs, a three-hour forecast, or physical forecast skill.

| ID | Finding / next task | Closure evidence | State |
|---|---|---|---|
| G1 | PR176 was merged into the PR175 development branch, not `main` | PR177 merged the unchanged PR176 tree into `main` at `e829a17`; PR176 head `03149e73` is an ancestor of `origin/main` | Done; integration only, no repeat FV measurement |
| G2 | Regular forecast leads were fixed to one in `FVResearchProblem` | Explicit positive lead count tied to configured horizon; forecast times, future boundary length, verification shape, RK stages, per-lead score and branch signature checked; old one-lead frozen-source parity retained | Two-lead 4x5 research profile implemented and focused tests passed; see below |
| G3 | Partly missing observations with fully known state/boundaries | Preserve missing versus censored meanings and observation-error transform; derivative and refusal tests | Pending separate observation-profile work; current all-valid/detected gate remains |
| G4 | Maintain/decay/flat/zero-face states | Separate forward availability from classical local-response eligibility; signed-growth and branch tests | Pending; strict branch tracer remains |
| G5 | Non-collocated observations | Explicit state-to-observation operator and compatible error covariance, including units/averaging order | Pending; `collocated_dbz` remains the only profile |
| G6 | Irregular observation times and three-hour horizon | Time/boundary/growth schedules with resource budget, rather than a hardcoded lead count | Pending; G2 tests only two one-minute forecast leads |
| G7 | Validation scope and coordinate/units invariance | Bind result to problem, point, code, branch, direction and step range; check physical response under control/parameter rescaling | Pending for new profiles; existing 86-control check remains one point/direction |
| G8 | Support, execution and refusal accounting | Separate forecast, stationarity, branch, response, independent-validation and resource states; report eligible and validated fractions over predefined inputs | Partly implemented for two fixed runs; broader case matrix pending |
| G9 | Concurrent diagnostics | Replace temporary shared-function patching with an explicit observer/context before parallel use | Pending; current research execution is serial |
| G10 | Independent physical or finite-impact validation | Distinct weather cases and observation truth; finite-amplitude response range evaluated separately | Pending; synthetic local score is not forecast skill |

## G2 evidence and interpretation

The retained three observations are at 0, 60 and 120 seconds. The bounded
two-lead case forecasts at 180 and 240 seconds, i.e. one and two minutes after
the final observation. It requires 18 future boundary pairs and traces
`2 * (2 analysis intervals + 2 forecast intervals) * 9 = 72` SSPRK Euler
stages. Each interval uses the existing log-growth subdivision; the
zero-flow analytic check reaches `q exp(.008)` and `q exp(.016)` at the two
endpoints. The first forecast is bit-identical to the one-lead binding.

The additional `forecast_times_seconds` and `forecast_leads` layout fields
change the fixed-input identity even for the numerically unchanged one-lead
case. Old cache records must fail identity matching and be rebound; no hash
continuity is claimed.

At the archived 26-control point, the two-lead per-frame MSEs are
`0.00024806534711284677` and `0.00026144151957140934`; their mean is
`0.00025475343334212806`. The control-direction score JVP is
`0.007624811135571487`; central slopes at `h=0.001` and `0.0005` are
`0.00762482276147136` and `0.007624814042039797`. Both signed endpoints
retain the nominal strict branch. These are local score/trajectory checks,
not a new GN/refinement/adjoint/reanalysis experiment or a 3-hour forecast.

The one-lead frozen-reference tests, new two-lead tests, and affected FV
forecast/trajectory tests report 57 passed, 18 existing TorchScript warnings
in `multilead_tests.log`. Pinned basedpyright on the changed module, case
builder and tests reports 0 errors, 0 warnings, 0 notes in
`multilead_typecheck.log`. Both logs were captured from the current source.
`multilead_run_manifest.json` records the exact commands, base commit,
interpreter versions, and SHA256 of the changed code and logs.
Larger FV optimization, broad CPU/package regression and a new numerical resource budget
were not part of this bounded check.
