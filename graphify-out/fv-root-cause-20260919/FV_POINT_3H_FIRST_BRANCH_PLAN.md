# First strict-branch refusal in the fixed three-hour point case

Run one source/input-bound diagnostic on the **unchanged** PR #204
four-point 4×5 profile: observations at 0/10/20 minutes, 18 ten-minute
forecast leads, original terminal verification field, control,
parameters, boundaries and 3,600-stage layout. The archived forward
result already passed; its separate strict oracle refused before
admitting a stage. The question here is which strict predicate and
which stage/cell/face first refuses.

Use the existing read-only `observe_minmod_stages` hook on
`problem.forecast(control, parameters)` under `torch.no_grad()`.
At each observed stage apply the same ordered predicates as
`fv_minmod_inverse_probe.inspect_branches`: finite positive field
scale; left/right interior x and y slope magnitudes above
`128*eps*scale`; nonzero active-limiter operand difference above the
same tolerance; and nonzero face flux magnitude above
`128*eps*max(abs(qx),abs(qy))`. Record the first failing stage index,
predicate, x/y orientation where relevant, and array index. Stop the
forecast immediately after that failure. Label the zero-based global
callback index, its analysis-replay or future-lead phase (180 stages
per ten-minute interval; the first 360 replay two analysis intervals),
and distinguish full-field cell indices from
the interior-local slope slices and x/y face indices. If no predicate fails, record
all 3,600 observed stages and `passed_pointwise` only; do **not** infer
a stationary point or full local response from that outcome. The
diagnostic may use detached values but must not alter the forecast,
objective, score, branch oracle, tolerances or input tensors.

The runner must pin current relevant source hashes, the new plan hash,
the PR #204 archived input identity and R5 response-policy record, and
child before/after identities;
check the fixed 26-control/13-parameter layout, expected stage count,
child exit and resource report. Declare one fresh output directory,
`point_3h_first_branch_attempt1`, before launch. Use one guarded child
with a **180-second wall-limit trigger** (plus termination/reap grace)
and **sampled 768 MiB child RSS**. Do not retry or switch inputs if it
fails. No GN, Newton, HVP, adjoint, VJP or signed reanalysis.

Before launch, require focused classifier/runner regressions,
error-level typecheck, incremental Graphify refresh, and independent
GREEN/RED prelaunch review. Afterward preserve raw child, guard report,
source/input and cost evidence. The result can close only this
branch-cause diagnostic; R5-R-R remains open until a qualified
stationary long-horizon branch and independent response validation
exist. A refusal is not proof that no such branch or root exists.
