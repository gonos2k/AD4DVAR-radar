# R2 sector-adaptive point-root attempt 1: measured-decrease refusal

This was the one-start experiment declared in `FV_POINT_SECTOR_ROOT_PLAN.md`.
It reused the source- and input-bound 4×5 correlated point-observation
problem and the saved attempt-2 terminal control. The branch oracle admitted
finite candidates with positive roundoff-aware margins over all 54 minmod
stages. Same-signature candidates retained the existing normalized gradient
merit Armijo rule. Changed-signature candidates had to lower the **measured**
objective, half squared gradient norm and maximum gradient component; a
successful switch was relinearized at the next Newton iteration. The
original eight Newton, sixteen backtrack and 104 PCG iteration budgets,
`1e-10` linear residual, and final `1e-10` stationarity requirement stayed
fixed. No adjoint or nonlinear reanalysis was attempted.

The seed exact Hessian passed the 26-column pointwise curvature audit:
minimum eigenvalue `0.6942730154315004`, maximum `4453.220758903886`,
and relative antisymmetry `4.038200716034779e-16`. All seven PCG calls
converged. Two full-signature switches were accepted, at step scales `1`
and `0.0625`; four further same-signature corrections used scales
`0.03125`, `0.00390625`, `0.00048828125` and `0.000244140625`.
The accepted objective fell from the locked seed's prior value to
`0.0010782540255775742`, and maximum gradient fell from
`0.01080526397720594` to `0.006432750851827679`. These are improvement
measurements, **not** a stationary root.

At Newton iteration 7, all 16 tested candidates changed the full
signature and failed the declared triple-decrease policy. The final,
smallest trial had scale `3.0517578125e-05`. Its gradient merit decreased
from `6.742465958387455e-05` to `5.1210802935643544e-05`, but its
objective increased by `3.81511547613983e-11` and its maximum gradient
increased by `1.7041841533539659e-06`. The policy rejected that candidate
as declared; it was never made an accepted iterate. All 16 iteration-7
candidates were policy rejections, with no branch-oracle, nonfinite, or
legacy Armijo rejection. Across the full run there were 62 candidate
trials, six accepted, and 58 changed-signature policy comparisons: two
accepted and 56 rejected.

The child exited `2` with
`RefinementNumericalRefusal: stationarity refinement failed to find an
admissible step`. The independent parent classified execution as
`completed` and numerical status as `sector_refused`; it did **not**
classify an eligible response. The guarded child took `170.38231024995912`
seconds, with 641 RSS samples and sampled peak `362,954,752` bytes under
the 600-second / sampled 1-GiB limits. Sampling is not a hard memory cap.
The source, input, seed control, plan, preflight and prior-report identities
were unchanged before and after the run. Raw child, resource, parent,
source snapshots and a SHA256 manifest are in `point_sector_root_attempt1/`.
`FV_POINT_SECTOR_ROOT_EVIDENCE.json` additionally binds the base commit,
current checklist/KG records, test sources and exact verification commands.

The 48 focused tests passed; the affected refinement, response, matrix-free
and point-probe suite passed **132 tests and two subtests**, with 18 existing
`torch.jit.script` warnings. Error-level basedpyright reported zero errors.
These checks validate the
policy boundary and result classification, not an FV root or sensitivity.
GREEN and RED prelaunch reviews both allowed this bounded execution after
the callback-error and final-trace fail-open paths were closed.

This single refusal says the declared *triple-decrease sector-transition
policy* could not continue from this seed within its backtracking budget.
It does not establish that the point-observation inverse problem lacks a
stationary point, or that a different declared numerical policy cannot find
one. In particular, objective and maximum gradient need not decrease at
each step that lowers the squared-gradient merit. R2 remains open until a
fresh final root meets its stationarity, curvature and branch gates, then
the adjoint/full VJP and signed nonlinear endpoints pass independently.
