# R2 gradient-merit sector-root attempt 1: declared refusal

This was one new experiment under `FV_POINT_MERIT_ROOT_PLAN.md`, starting
from the sixth accepted control in the previous sector-adaptive run. That
seed and the old report were pinned by SHA256, and the fixed correlated
4×5 point problem, 26 controls, 13 parameters and 54-stage core branch
oracle were unchanged. The prior run's triple-decrease policy was not
edited or retroactively reclassified. On a changed full signature, this
run required only an actual finite decrease in
`Phi=0.5*||grad_c J||_2^2`; same-signature candidates used the legacy
normalized gradient-merit Armijo test. The final stationarity, exact
curvature and `>1e-4` response-margin gates were unchanged. No adjoint or
nonlinear reanalysis ran.

The seed's maximum gradient was `0.006432750851827679`. All seven
Newton PCG solves converged to actual relative residuals below `1e-10`.
The first accepted step switched signature at scale `0.0078125` and
lowered the gradient 2-norm from `0.01161246395765124` to
`0.011075220544695369`; the objective and maximum gradient increased
at that switch, as this new root-finding rule permits. Five subsequent
same-signature corrections were accepted. The last accepted point had
`J=0.00107824897410049`, gradient 2-norm
`0.010552233324871483` and maximum gradient
`0.006602351499307973`. These values are far from the required
`1e-10` maximum-gradient stationarity threshold.

At Newton iteration 7, all 16 candidates changed signature and were
refused by the measured merit rule. The smallest candidate, at scale
`3.0517578125e-05`, raised merit from
`5.567481407126414e-05` to `7.19517485248372e-05`, despite being a
tiny step. Across the run there were 76 candidate records and six
accepted steps. The changed-signature policy made 71 comparisons:
one accepted and 70 refused. The final iteration had no branch-oracle,
nonfinite or same-signature Armijo rejection; its 16 refusals were all
policy decisions.

The child exited 2 with `numerical_status=merit_refused`. The independent
parent classified execution as `completed`, with
`response_validation=not_performed`: `170.295867958921` seconds,
638 sampled child-RSS observations and a `354,222,080` byte sampled peak
under the 600-second / sampled 1-GiB limits. The RSS guard is sampled,
not a hard allocation cap. Child source, input, seed, plan and prior
report identities stayed unchanged before and after the run. Raw child,
resource, runner, source snapshots and SHA256 manifest are preserved in
`point_merit_root_attempt1/`. The aggregate
`FV_POINT_MERIT_ROOT_EVIDENCE.json` binds the base commit, checklist/KG,
test sources and exact verification commands.

The 51 focused tests and 143 affected tests plus two subtests passed with
18 existing TorchScript deprecation warnings; error-level basedpyright
reported no errors. GREEN and RED
prelaunch review found no remaining fail-open status or root gate after a
callback-error classification fix. This evidence checks the declared
policy and failure classification, not a stationary response.

The result narrows this **particular** approach: starting at the locked
prior endpoint, a strict gradient-merit-only sector transition accepted
one switch, then could not continue within the declared 8/16/104 limits.
It does not prove that a smooth stationary point is absent for this
point-observation model, nor does it validate the full 13-component
sensitivity. R2 remains open. A different fixed observation profile
may be useful for a first qualified point-response integration, but must
be declared and validated as a **different** statistical problem; the
challenging original profile remains an unresolved refusal.
