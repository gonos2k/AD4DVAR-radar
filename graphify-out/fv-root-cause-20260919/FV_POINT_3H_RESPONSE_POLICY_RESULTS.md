# Fixed point-observation 3-hour input: forward yes, local response refused

This is the read-only support decision in
`FV_POINT_3H_RESPONSE_POLICY_PLAN.md`. It reuses PR #204's exact
source-/input-/plan-bound, resource-guarded 4×5 point-observation
18-lead report. No new 3,600-stage FV run, GN optimization, adjoint or
reanalysis was performed. The original raw manifest and PR #204
aggregate evidence are SHA256-pinned, and the exact input identity
has a separate canonical digest.

The archived child and parent completed successfully: child exit 0,
no wall/RSS termination or monitor error, `forward_validation=passed`.
Its three observation times are 0, 600 and 1,200 seconds; the 18th
ten-minute lead ends at 12,000 seconds absolute, or 180 minutes after
the last observation. There are 180 analysis and 1,620 future
boundary stage pairs. A separate point-analysis call observed 360
SSPRK stages, and the terminal forecast call replayed the analysis
plus future for 3,600 stages. Point-observation reproduction and
terminal same-operator synthetic field parity were each better than
`1e-9 dBZ`. These checks establish a bounded forward calculation;
they are not a physical forecast-skill evaluation.

The **separate strict branch diagnostic refused** with
`minmod joint oracle left its strict smooth branch` before admitting
any stage into that diagnostic. The raw report correctly kept
`stationarity_passed=not_tested`, `response_computed=false`,
`response_validation=not_performed` and
`physical_validation=not_performed`. The zero *strict-diagnostic*
stage count does not negate the independently observed 360 and 3,600
forward stages.

The support status for **this exact fixed input/control under the
current strict local-response gate** is
`unsupported_by_current_strict_branch_gate`; no 3-hour point
sensitivity is issued. This does not imply that no other long-horizon
branch or stationary point exists. A future response needs its own
strict final branch, stationary control, exact Hessian/PCG, full VJP
and signed nonlinear endpoints under the preserved time and score
contract. That research remains open.

The read-only classifier and ten focused tests cover false branch
success, response issuance, resource exit, forward failure, stage
count, wrong input, disagreement with the resource sidecar or PR #204
source map, and manifest path escape. Error-level basedpyright
reports zero errors. `FV_POINT_3H_RESPONSE_POLICY.json` separates
forward and response eligibility without relabeling either result.
The original archived forward artifacts are unchanged.
The aggregate `FV_POINT_3H_RESPONSE_POLICY_EVIDENCE.json` binds the
base commit, PR #204 raw and aggregate reports, classifier, tests,
checklist/KG and exact read-only commands. This focused accounting
does not replace a full CPU/package regression.
