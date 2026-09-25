# Point observations joined to the regular 3-hour FV forward schedule

This bounded run implements `FV_POINT_3H_FORWARD_PLAN.md` and closes a
**forward-only** R5 combination. The existing 4×5 nonzero-flow
same-operator 3-hour fixture supplied a fixed control, complete
state-grid background and boundaries, and terminal synthetic truth.
Three analysis fields at 0, 10 and 20 minutes were sampled at four
fixed off-grid interior point locations in dBZ and frozen as 12 detected
observations, with fixed standard deviations, quality and within-time
correlation. The state-grid background remains an external synthetic
field, not an interpolation of those 12 point values. Theta is a
separate parameter; the point problem has 26 controls and 13
parameters.

The point-problem forecast now uses the declared regular
`forecast_steps` count. For this case there are 18 ten-minute leads
after the last 20-minute observation, ending at absolute time 200
minutes (180 minutes after analysis). The FV transport uses 90
substeps per ten-minute interval, 180 analysis boundary stage pairs
and 1,620 future boundary stage pairs. The layout declares 3,600
SSPRK stages. A separate analysis-trajectory call observed 360 stages;
the terminal-forecast call observed **3,600** because it recomputes
those 360 analysis stages before 3,240 future stages. The code still
returns only the terminal 4×5 dBZ field and scores it against one
fixed terminal verification field.

The actual point-analysis samples differed from the frozen same-operator
point observations by at most `1.554312234475219e-11 dBZ`. The
terminal point-problem field differed from the source fixture's
same-operator synthetic truth by at most
`1.1688428003253648e-12 dBZ`, below the predeclared `1e-9 dBZ`
tolerance. The terminal MSE was `1.8176932978102554e-25`; the
fixed-control objective was finite at `0.9421520472268183`.
These are execution and same-discrete-operator parity checks, **not**
independent physical forecast accuracy or a new analysis optimum.

The separate strict minmod branch probe **refused** with
`minmod joint oracle left its strict smooth branch` before any stage
was admitted to its strict trace. Its observed strict-diagnostic
stage count is 0; this does not undo the independently observed 360
analysis and 3,600 terminal-forecast stages. No stationarity test,
adjoint, full VJP, nonlinear reanalysis or long-horizon local response
was performed or published.

The guarded child exited 0 after `7.464563124929555` seconds,
with 28 sampled child-RSS observations and a sampled peak of
`239,239,168` bytes under the 180-second and sampled 1-GiB limits.
The parent independently rebuilt the fixed input, reran the analysis
point sampling and terminal forecast, checked layout, timings,
boundary lengths, stage counts, actual field/score/objective and
source/input/plan/archive hashes, and returned
`execution_status=completed`, `forward_validation=passed`,
`response_validation=not_performed`,
`physical_validation=not_performed`. The sampled RSS bound is not a
hard allocation cap and excludes parent memory.

An original one-lead input-identity regression confirmed that its
54-stage, 180-second terminal layout remains unchanged. The affected
point and long-horizon suite passed 88 tests, and a separate constructed
point-response compatibility suite passed 34 tests, each with 18
existing TorchScript warnings. A read-only replay of PR #200's
archived child payload also passed the current `_valid_result`
numerical predicate; it did not replay full resource/provenance gates.
Error-level basedpyright reported zero errors.
GREEN and RED prelaunch reviews approved the bounded run after the
parent's JSON tuple/list comparison and total-stage gate were fixed.

This closes only **point observations + regular 18-lead FV terminal
forward execution**. A qualified 3-hour stationary response, a
strictly supported long-horizon minmod branch, irregular observation
times, changing future flow/growth/boundaries, finite observation
impact and independent forecast skill remain separate tasks. Raw
child/resource/parent reports and source snapshots are preserved in
`point_3h_forward_attempt1/`.
`FV_POINT_3H_FORWARD_EVIDENCE.json` binds the base commit, report,
source/tests, checklist/KG and exact verification commands. The focused
local checks are distinct from a full CPU/package CI run.
