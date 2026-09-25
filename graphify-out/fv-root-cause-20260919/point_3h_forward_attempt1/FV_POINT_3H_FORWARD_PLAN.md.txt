# R5 point observations with the existing 3-hour FV forward schedule

This is a **forward-only combination** of the established one-lead
off-grid point-observation contract and the existing regular 10-minute,
18-lead, 3-hour same-operator FV fixture. It does not certify GN
stationarity, local response, nonlinear reanalysis over 3 hours or
independent meteorological skill.

Generalize `FVPointResearchProblem` only where forecast length is
currently fixed to one interval. Preserve three analysis point frames
at 0, 10 and 20 minutes, their existing point sampler, observation
likelihood, error/correlation model, external state-grid background
`B_external+theta*P`, complete model support and fixed boundary
schedules. Require the future boundary stage pairs to have
`forecast_steps * substeps_per_interval` entries. Forecast from the
analysis state for `forecast_steps` regular intervals and return the
**terminal field only**; the fixed verification target and score remain
terminal-field MSE. The declared forecast time must be
`(2+forecast_steps)*interval_seconds`, so 18 leads after the final
20-minute observation end at 200 minutes absolute. For one lead,
preserve the existing numerical objective, forecast, score, layout,
branch and parameter semantics.

`forecast_fv_analysis` reconstructs the two analysis intervals from
the control before advancing 18 future intervals. Thus a single
terminal-forecast call observes **3,600 SSPRK stages** (360 analysis
+ 3,240 future), while the separately instrumented analysis-point
reproduction call observes 360 stages. The future boundary input still
contains only the 1,620 future stage pairs.

Construct one explicit 4×5 point profile from the already verified
nonzero-flow 3-hour FV fixture. Use its fixed control, 10-minute
interval, 90 substeps, 180 analysis-boundary stage pairs, 1,620 future
boundary stage pairs and same-operator terminal verification field.
Sample the three analysis fields at the existing four fixed interior
cell-center coordinates in dBZ; freeze those 12 values as point
observations, with fixed detected status, fixed standard deviations,
quality and same-time symmetric correlation. Keep the full state-grid
background from the fixture as an **external synthetic field**, never
reconstruct it from the 12 point values. Theta remains a separate
parameter. This test is an execution-scale profile; it does not assert
equal statistical weighting or inference equivalence to the earlier
full-grid 3-hour problem.

In a guarded forward run, verify exact layout and time values, stage
schedule lengths, complete state/boundary support, point-operator
observation reproduction, finite objective/forecast/terminal score,
and terminal forecast parity to the same-operator synthetic truth at
the predeclared FP64 tolerance `1e-9 dBZ`. Record whether strict
minmod branch certification passes or refuses, with its observed
stage count; **never emit a long-horizon response** from this run.
Record source/input/plan identities before and after, elapsed time,
sampled child RSS and explicit forward/response validation statuses.
One child is bounded to 180 seconds and sampled 1-GiB RSS. The parent
checks command/PID/exit/resource and current source/input/plan hashes.
Do not change the old one-lead profile's result or reinterpret this
same-operator parity as an independent 3-hour forecast skill score.
