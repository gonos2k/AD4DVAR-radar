# PR173 follow-up: recoverable nonfinite trial evaluations

Reviewed baseline: open PR173 head b38fdb7, not a merged revision. The original
26-control FV measurement and its source fingerprints are historical evidence
for that head and are preserved without rewriting or rerunning the experiment.

## Reproduction and mathematical policy

Executed the original `local_refinement.py` from `git show b38fdb7:...` against
current local PCG using `.venv/bin/python`. For J(x)=exp(x)-x, x=-2 converges in
5 Newton steps (max gradient 3.91258e-11); x=-7 and -8 abort with `ValueError:
objective must be finite`. Initial objective/gradient/Hessian are finite.

A nonfinite full-step trial does not imply that every shorter step is invalid.
Keep the initial evaluation outside the recovery handler. Within line search,
recover only numerical nonfiniteness explicitly detected by solver checks;
shape/type errors and exceptions raised by user callbacks must propagate.
Keep the objective, exact HVP, PCG tolerance, strict stationarity threshold,
Armijo formula and iteration/backtracking budgets unchanged.

No new FV execution was needed or performed for this candidate-evaluation
correction. The old FV results describe b38fdb7, not a newly measured run.


## Applied change and measured local results

Private `_NonFiniteEvaluation` is emitted by explicit numerical checks for
objective values, gradient components, or gradient norms. It is caught only
around candidate evaluation; initial evaluation remains outside the handler.
Nonfinite candidate controls are rejected before the branch callback. Recorded
rejection rows contain finite metadata without storing Inf/NaN values.
Structural errors and arbitrary objective callback exceptions propagate;
branch callback ValueError retains its separately declared rejection behavior.

| Initial x | Newton steps | HVP calls | Final max gradient |
|---|---:|---:|---:|
| -2 | 5 | 15 | 3.9125813700e-11 |
| -7 | 7 | 21 | 0 |
| -8 | 6 | 18 | 1.3322676296e-15 |

`nonfinite_candidate_checks.json` records the current local Python/PyTorch
versions, source hashes and step history. Zero is a floating-point result, not
an assertion of exact arithmetic. This is a local analytic diagnostic; it is
not an FV experiment or additional independent meteorological case.

## Verification

- 113 unique affected tests passed, 18 existing torch.jit.script warnings;
  `pr173_followup_tests.log`. This includes 26 refiner tests (the earlier 12
  are not added again), wrapper, producer, response, publisher and public-gate
  regression coverage.
- New tests check overflow recovery, gradient component/norm nonfiniteness,
  exhausted candidate budgets, malformed output and objective callback
  ValueError/RuntimeError/TypeError specifically at candidate evaluation.
  The candidate flag is set by the branch check so these exceptions are not
  accidentally injected only during HVP evaluation.
- Six 64-control cases use coupling kappa=0 or 4 and starts 0,+0.5,-0.5.
  The extra cost is kappa/2 * sum((delta[i+1]-delta[i])²), alongside the
  positive diagonal quadratic and quartic terms; the known root is delta=0.
- Pinned basedpyright: 0 errors, 0 warnings, 0 notes;
  `pr173_followup_typecheck.log`. No full CPU/package CI was launched.
- GREEN checked source hashes, numerical evidence and tests; RED checked
  exception boundaries, mathematical policy and scaled-experiment contract.
  Both final reviews found no remaining blocker.

The historical 26-control FV artifact and publisher digest are unchanged.
The demo explains the follow-up separately from those old measured costs.
Graphify code-only refresh: 271 files, 6725 nodes, 65149 edges, 251 communities;
no semantic extraction. User AGENTS.md model-preference edits are preserved.

## Next scope (not executed)

See `FV_86_CONTROL_EXECUTION_PLAN.md` for fixed geometry, CFL/boundary schedule,
objective normalization, two initializations, resource estimates and proposed
caps. That larger experiment requires the previously requested budget approval.
The plan is not an execution or convergence result.

Aside desktop DOM check confirmed the follow-up paragraph and record link.
No new screenshot/mobile verification was claimed. Publisher verification
confirmed unchanged original template and local-index frame-data SHA256 values.
