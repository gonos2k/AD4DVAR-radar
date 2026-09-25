# One wholly empty first point-observation time: qualified FV response

This is the separately declared R3-E profile in
`FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_PLAN.md`. All four first-time
point values are status 1 **genuinely missing** with canonical inactive
fills, and `empty_observation_time=0` is declared. The middle and last
times retain eight detected point observations. The first-time
observation cost and whitener are absent, while both analysis time
intervals, the full 4×5 FV state and boundaries, and the two later 4×4
correlation whiteners remain. The background is still the external
state-grid field plus `theta*pattern`; it does not use the missing first
point values. The fixed nonzero dynamics-prior mean and frozen
verification from the constructed PR #200 profile are unchanged.

At the constructed control, objective and all 26 control-gradient
components are zero. The exact Hessian audit gave minimum eigenvalue
`0.999999999999673`, maximum `4272.4195133621715`, and relative
antisymmetry `3.0862052805880227e-16`. All 54 minmod stages passed the
nominal strict branch, with scaled slope margin
`8.046479676988153e-4` and face margin
`7.752734889703382e-3`, above the retained `1e-4` response gates.

The matrix-free adjoint returned a complete 13-component parameter
sensitivity. First-time observation indices 0–3 each have **zero
direct, indirect and total component**. Shifting each stored inactive
value alone by `+0.25` changed nominal objective, score and control
gradient by exactly zero. The parent independently rechecked these
four component and payload-invariance results.

The external background dependency remains: theta's direct,
indirect and total components are respectively
`-0.004954104945433104`, `+0.0019543715132648506`, and
`-0.0029997334321682535`. These are derivatives of the constructed
conditional score, not evidence of physical background quality.

For the predeclared direction adding +1 dBZ to each of the four
**middle-time** points, direct effect is 0 and indirect/total response
is `0.024230937342476166`. The full-vector projection differs from the
separate directional computation by `5.273559366969494e-16`.
Independent transposed-adjoint relative residual was
`5.1811823419041186e-14` after 15 PCG iterations; the tangent true
relative residual was `2.403992927295551e-11` after 11 iterations.

All four actual signed endpoint analyses converged in one Newton
correction, retained the exact nominal 54-stage signature and `>1e-4`
slope/face margins, and had final maximum gradients below
`1.77e-12`.

| Middle-time offset h (dBZ) | Actual central difference | Adjoint relative error |
|---:|---:|---:|
| 0.001 | 0.024230854891313953 | 3.4027227691750685e-6 |
| 0.0005 | 0.02423091672957298 | 8.50685340607269e-7 |

Both pass the predeclared `1e-4` criterion; the absolute error ratio
is `3.9999781432062838` in this two-size local range. This validates
the effect of changing active middle-time **values**, not the finite
impact of removing a whole observation time.

The guarded child exited 0 in `117.53270274994429` seconds with 425
sampled child-RSS observations and a `342,294,528` byte sampled peak,
under the 600-second / sampled 1-GiB limits. Parent status is
`execution_status=completed`, `numerical_status=eligible`,
`response_validation=passed`. It independently recomputed the nominal
Hessian, adjoint/full VJP, tangent, all four inactive parameters,
nonzero theta dependence, endpoint objectives/scores/gradients/branches,
and signed central differences from its own endpoint scores. Source,
input, plan and archive hashes were stable before and after the child.
The RSS cap is sampled, not a hard allocation limit.

This closes **only the constructed exactly-one-empty-first-time point
profile**. More than one empty time, the original zero-centered-prior
input, 3-hour point sensitivity, finite observation-removal impact and
independent physical forecast skill remain open. Raw
child/resource/parent records and source snapshots are preserved in
`point_empty_centered_attempt1/`.
The focused empty/QC/missing/centered/response/refinement suite passed
68 tests with 18 existing TorchScript warnings; error-level basedpyright
reported zero errors. GREEN and RED prelaunch reviews found no empty-time,
theta-dependence or false-success blocker. The aggregate
`FV_POINT_EMPTY_TIME_CENTERED_RESPONSE_EVIDENCE.json` binds the base
commit, raw report, source/tests, checklists/KG and exact verification
commands. This is separate from a full CPU/package CI run.
