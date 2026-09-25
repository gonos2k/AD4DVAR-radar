# R2 point-objective exploratory basin search attempt 2: step budget

This separately planned run kept the same 4×5 correlated point objective,
observations, background, frozen score and single warm start as attempt 1.
Only exploratory candidate eligibility changed: it used the existing
roundoff-scaled strict minmod tracer and finite positive measured margins,
while the final `>1e-4` sensitivity margins remained mandatory before any
Hessian or Newton handoff. Attempt 1's report SHA was rechecked and its raw
results/source snapshots were left unchanged.

The guarded child exited 2 with `numerical_status=basin_incomplete` and
`basin_status=step_budget`, while the parent recorded a completed execution.
It took **58.475 seconds**, with 222 RSS samples and sampled peak
**475,561,984 B** under the 1200-second / sampled 1-GiB cap. Source,
input, both plans, Stage A preflight and attempt 1 identities remained
unchanged. This is a numerical budget outcome, not a resource termination.

The objective decreased from **0.07993508078523576** to
**0.001092566268397464** in the predeclared maximum **100 accepted
steps**. There were **331 trial evaluations**: 100 accepted and 231
finite actual-objective Armijo refusals. No strict branch or candidate
evaluation refusal occurred in this run; 22 accepted steps changed the
pointwise limiter/face-sign signature, resetting L-BFGS curvature history.
These endpoint records do not certify a smooth path between accepted steps.

The last accepted maximum control-gradient component was
**0.01080526397720594**, above the predeclared `1e-4` handoff gate. The
lowest recorded maximum gradient was **0.00424013333051669** at accepted
step 95, still above the gate. The last point's scaled slope and face-flux
margins were `0.00019559368949964172` and
`0.00035249320414798194`, respectively. They exceed the final
`1e-4` margin gate, but no terminal Hessian or exact Newton refinement ran
because the gradient handoff criterion failed. No stationary point,
adjoint or signed endpoint was produced;
`response_validation=not_performed`.

The distinct attempt-2 raw child/resource/runner/log records and exact
uncommitted search, producer, runner and plan source snapshots are preserved
in `point_basin_attempt2/`. The run demonstrates that lowering only the
**exploratory** margin to the tracer's existing strictness lets the same
objective cross 22 pointwise branch signatures and descend further. It
does not prove that a differentiable stationary minimum exists nearby or
that a finite perturbation remains on one branch. Increasing the 100-step
cap after seeing this plateau would be a new experiment, not a correction
to this measured result.
