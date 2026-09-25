# Constructed centered-prior point FV response: one qualified integration case

This experiment implements `FV_POINT_CENTERED_PRIOR_RESPONSE_PLAN.md`. It is
a **new synthetic statistical problem**, not a successful rerun of the
original zero-centered-prior correlated point case. The state grid is 4×5,
with 26 controls, 12 full-valid off-grid dBZ point observations over three
times, one theta parameter, the same fixed symmetric 4×4 within-time
correlation and fully known model state and boundaries. The six dynamics
prior means are fixed to a nonzero control; observations are generated
once from that control and detached, and the verification field is fixed
from its forecast plus the predeclared pattern. The unchanged point FV
model, sampler, pseudo-Huber observation term, field prior and score are
reused. Only the dynamics Gaussian prior center changes via the explicit
objective in the plan. No prior learning or new physical truth is implied.

At the constructed control, the actual objective and all 26 gradient
components are zero. A separate exact 26-column Hessian audit found
`lambda_min=0.9999999999997873`, `lambda_max=4426.990015915346`, and
relative antisymmetry `2.930773598235177e-16`. All 54 minmod stages passed
the core branch oracle, with scaled slope margin
`8.046479676988153e-4` and face-flux margin
`7.752734889703382e-3`; both exceed the retained `1e-4` response gate.
The nominal synthetic score is `0.00025526315789473595`.

The matrix-free adjoint returned a full **13-component** direct,
indirect and total parameter gradient. For the fixed direction that adds
one dBZ to each of the four middle-time points, its direct effect is
exactly zero under the frozen score/background contract, indirect and
total response are both `0.0012340828462648612`. The independent
transposed-adjoint relative residual was `1.0487605606789624e-13` after
16 PCG iterations and 17 HVP calls. Projecting the full total VJP onto
the same direction differs from the separate directional JVP response by
`2.5153490401663703e-17`. The tangent PCG used 15 iterations and had
actual relative residual `1.0543583542611846e-11`.

Each of four signed endpoint analyses used its own changed 13-component
parameter vector and a tangent-predicted starting control. The existing
`refine_stationary` needed one Newton correction at every endpoint. All
four final maximum gradients were below `1.85e-12`, and all final
54-stage signatures matched the nominal signature with slope and face
margins above `1e-4`.

| Pointwise middle-time offset h (dBZ) | Actual reanalysis central difference | Adjoint relative error |
|---:|---:|---:|
| 0.001 | 0.0012340865951139958 | 3.0377613187776872e-6 |
| 0.0005 | 0.0012340837834979028 | 7.594571502605831e-7 |

Both errors pass the predeclared `1e-4` relative criterion. Their absolute
error ratio is `3.9999114074248667`, consistent with second-order central
truncation over these two local sizes. This verifies one point-observation
direction and local perturbation range, not every parameter component's
nonlinear effect or a finite-amplitude observation value.

The guarded child exited 0 after `120.32000154105481` seconds with 457
sampled RSS observations and a `343,736,320` byte sampled peak, within
the 600-second and sampled 1-GiB limits. The parent classified execution
as `completed`, numerical status as `eligible`, and response validation
as `passed`. It independently reconstructed the nominal Hessian, true
adjoint and tangent equations, full direct/indirect VJPs, and all four
endpoint objectives, scores, gradients and strict branches before forming
the signed differences from **recomputed** scores. Source, input, plan and
baseline archive hashes were unchanged through the child run. The sampled
memory guard is not a hard process allocation cap.

The original zero-centered-prior correlated point input remains unresolved:
its seven prior bounded attempts did not produce a qualified stationary
point. This constructed success shows that the existing point FV path can
connect a **qualified** smooth stationary point to a full matrix-free
response and actual signed reanalysis when its prior and observations
define such a point. It does not establish that ordinary GN finds that
point, that missing/QC/empty-time profiles have a stationary response,
that point observations support 3-hour response, or that real radar
forecast skill improves. Raw child/resource/parent reports and source
snapshots are in `point_centered_response_attempt1/`.
The focused stationarity, response and parent-publication suite passed
55 tests with 18 existing TorchScript warnings; error-level basedpyright
reported zero errors. GREEN and RED prelaunch review required and then
confirmed independent adjoint, tangent, Hessian and endpoint checks before
the bounded FV run. `FV_POINT_CENTERED_PRIOR_RESPONSE_EVIDENCE.json` binds
the base commit, raw record, source/tests, plan, checklists, KG and exact
verification commands. These local checks are distinct from a full
CPU/package CI run.
