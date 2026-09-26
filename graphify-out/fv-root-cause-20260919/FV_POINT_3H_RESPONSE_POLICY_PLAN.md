# R5-R support decision for the fixed point-observation 3-hour input

The fixed 4×5 off-grid point-observation 18-lead case in PR #204
completed a same-operator terminal **forward** check. Its strict
minmod branch diagnostic refused before any stage was admitted. This
decision is read-only: do not rerun the 3,600-stage trajectory, GN,
adjoint or nonlinear reanalysis, and do not edit the archived run.

Pin the original `point_3h_forward_attempt1/manifest.json` SHA256 and
verify every entry, the independent PR #204 aggregate evidence hash,
the exact child/parent/resource report, and the child source/input/plan
before/after flags. Require the parent's embedded resource record to
match its independently archived sidecar, and every child source hash
to match the pinned PR #204 aggregate evidence map. Bind this exact
input identity by a canonical
JSON SHA256 digest, independently of the raw-manifest seal. The child
reached `forward_validation=passed` with
12 point values reproduced within `1e-9 dBZ`, terminal field parity
within `1e-9 dBZ`, 360 separately observed analysis stages, and 3,600
stages in the replayed terminal forecast call. Its parent exited 0
within 180 seconds and sampled 1 GiB child RSS. These establish
forward execution only.

The separate `strict_branch_diagnostic` must have status `refused`,
reason `minmod joint oracle left its strict smooth branch`, and zero
admitted diagnostic stages. Keep `stationarity_passed=not_tested`,
`response_computed=false`, `response_validation=not_performed`, and
`physical_validation=not_performed`. Under the declared exact local
response gate, classify **this input at this control** as
`unsupported_by_current_strict_branch_gate` and issue no 3-hour
sensitivity. Do not call the FV forecast a numerical failure because
the branch-only response check refused, and do not infer that no
other long-horizon stationary point or branch could exist.

A future response would need a separately declared 3-hour case with
strict final branch, eligible stationary control, exact Hessian/PCG,
full parameter VJP and signed nonlinear endpoints while preserving
the 0/10/20-minute observation schedule, 18 ten-minute leads,
terminal score and boundary contract. That research remains R5-R-R.
This classification closes only the current-input response support
decision R5-R-D. Keep full process, resource, numerical eligibility,
response and physical-validation states separate in the emitted JSON.
