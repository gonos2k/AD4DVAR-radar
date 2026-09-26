# Three-hour point seed: branch passes, objective is not stationary, PCG refuses

The one predeclared PR #218 alternate-control seed remains on a strict
3,600-stage minmod branch, but it is **not** a stationary analysis
control. At the unchanged 13-parameter point problem, the analysis
objective gradient has `||g||_inf=2.254729681485403`, far above the
existing `<1e-10` requirement. One exact-HVP PCG attempt for
`H s=-g` refused with `operator must be symmetric positive definite`
after 15 HVP calls. No Newton step, adjoint, VJP, signed reanalysis or
sensitivity was issued.

The objective is the two-interval analysis likelihood plus priors;
the separate terminal forecast score was not evaluated in this run.
The objective value at the seed was `0.9323736303384679`.
Field-control gradient norms were L2 `1.111840019565684` and
infinity `0.5736240663395087`; dynamics-control norms were L2
`2.614765018409301` and infinity `2.254729681485403`. The child
reported 360 analysis Euler stages **by fixed layout contract**, not
as a separately observed count.

The child first rebuilt the same fixed synthetic fixture inside the
guard, then collected normalized branch margins in a separate
no-grad forecast and ran the original strict branch checker. Both
forecast passes covered the 3,600-stage schedule. The original strict
checker returned 3,600 Euler, limiter-choice and face-sign stages with
the same signature SHA256 `80a1fac22b536bfedcedfda506e11d85f4b9f2a73c5610c73fa8dfdb210dfe51`
as PR #218. The margin record is complete for all 3,600 observed
stages. Its smallest normalized active x-limiter operand gap was
`9.968176299306184e-8` at future lead 4/global stage 1006; the
smallest active y gap was `8.928182994641934e-7` at lead 2/stage
594. The minimum absolute face flux remained
`0.0008000000000000229` at `q_y[3,1]`, or
`0.005025125628140845` of the trajectory-wide maximum absolute
face flux. These are pointwise diagnostic margins, not a finite
perturbation or global smoothness certificate.

PCG used `H v=JVP_c(grad_c J;v)` with the unchanged robust objective,
relative target `1e-10`, maximum 104 iterations and no
preconditioner. It raised the non-SPD curvature refusal during its
search; it returned no solution or true linear residual. The 15 HVP
calls include only this aborted PCG attempt. The raw `linear.iterations=0`
is an initialized placeholder: PCG raised before returning a result,
so its completed iteration count is **unavailable**, not measured as
zero. This visited-direction
event is **not** a computed full Hessian spectrum or proof of global
indefiniteness. A dense 26-column eigenvalue audit or another
independent curvature test would be needed before making that claim.
The existing SPD Newton–PCG refiner would encounter this same
unqualified starting point; this report does not attempt a modified
Newton/trust-region step or assume a nearby stationary branch.

The child exited 0 and the parent classified a completed numerical
refusal. The resource guard measured **117.199 seconds** elapsed,
438 sampled child-RSS readings and a peak of **324,206,592 bytes**
under the 600-second wall trigger and sampled 1 GiB limit; the child
reported **115.806 seconds** internal elapsed. Child phase times were
2.782 seconds source/fixture, 4.855 seconds full forecast and strict
branch, 1.519 seconds objective/gradient, and **106.645 seconds**
linear preflight. Source, archived input, shifted control, plan and
after-run identities matched. The parent performed no FV fixture
reconstruction outside the guarded child.

The final affected suite passed **75 tests** with 18 existing
TorchScript deprecation warnings; 10 focused tests passed. Error-level
basedpyright 1.39.9 reported 0 errors/warnings/notes. GREEN/RED
approved the one-shot plan and code before launch. Graphify's
incremental code refresh produced a valid graph with 8,106 nodes and
418,599 links. File hashes and exact checks are in
`FV_POINT_3H_SEED_LINEAR_EVIDENCE.json`.

This closes only the seed stationarity/one-RHS linear feasibility
preflight. R5-R-R remains open. The next numerical method, if pursued,
must be declared separately for a nonstationary seed whose current
PCG search refuses; relaxing the `<1e-10` final gradient or final
adjoint residual is not a valid workaround. The evidence does not
establish an absent root, a globally indefinite Hessian, an eligible
response, finite observation impact or physical forecast skill.

Known reporting edge: if a future margin diagnostic becomes nonfinite
only after all 3,600 callbacks, the parent currently treats that
incomplete diagnostic as an evidence failure rather than a completed
numerical refusal. It cannot create a false pass for this case, and
this run's margin record was finite and complete.
