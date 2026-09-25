# R2 locked point endpoint: SPD seed, fixed-branch Newton refusal

This is a **separate** diagnostic from basin attempt 2's deterministic last
accepted control. Attempt 2 itself ended `step_budget` with gradient
`0.01080526397720594`, above its planned `1e-4` handoff; it has not been
relabeled as a success. The new run pinned the exact attempt-2 report,
26-component seed gradient, control and full 54-stage branch signature,
same correlated point objective and both scaled margins. It did not run
an adjoint or signed reanalysis.

The fresh 26-column exact-HVP Hessian at the locked seed passed the
pointwise local curvature gate:

| Diagnostic | Recorded value |
|---|---:|
| Relative antisymmetry | `4.038200716034779e-16` |
| Smallest eigenvalue | `0.6942730154315004` |
| Largest eigenvalue | `4453.220758903886` |
| Smallest/largest ratio | `0.0001559035702515652` |

That check justified trying the unchanged exact-HVP Newton–PCG refiner
from this point. Its first four PCG solves all reported convergence after
20, 21, 21 and 22 iterations (observed HVP calls 21, 22, 22 and 23).
The refiner then refused **Newton iteration 4**: all 16 backtracking
candidates failed the branch callback before finite objective/gradient
merit evaluation. The error counted nine first-reported full signature
changes and seven combined stage/slope/face-margin refusals. Those seven
are not individually separated by the current callback message. No
qualified final stationary control was returned, so no final Hessian,
local response or endpoint validation followed.

The guarded child exited 2 with `numerical_status=newton_refused`; the
outer execution completed after **98.443 seconds**, with 375 RSS samples
and sampled peak **373,161,984 B** under the 600-second / sampled 1-GiB
cap. Child/parent source, input, seed, plan, Stage A and attempt-2 hashes
matched before and after. The raw child/resource/runner/log reports and
exact execution-time probe/runner/plan snapshots are preserved in
`point_terminal_newton_attempt1/` with a seven-file SHA manifest.

The result proves neither that a stationary point is absent nor that the
point-observation response is valid. It shows a fixed-branch Newton policy
could make several SPD linear solves but could not accept its fourth
correction while retaining the declared local branch and margins.
`response_validation` remains `not_performed`; physical forecast skill is
also untested. A new policy that moves between branches must evaluate the
actual nonlinear objective or stationarity residual after a branch change
and relinearize there. Carrying the old branch's Hessian merit slope across
a minmod kink would not be justified by this run.
