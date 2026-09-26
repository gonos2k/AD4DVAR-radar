# R4-R exploratory alternate-sector root-only attempt 1

This is the one fixed-seed experiment declared in
`FV_PARTIAL_ALTERNATE_ROOT_PLAN.md`. It started directly from the
post-hoc PR #212 locally SPD control `125e0200…7421`, previously
rejected on the PR #209 path. No product GN analysis was rerun, no
second seed was tried, and the objective, two missing observations,
prior, boundary and fixed verification field were unchanged. The
PR #209/#212 raw archives and source/input identities were checked
before and after.

The guarded child completed the bounded computation with exit code
**2** and `numerical_status=root_refused`; the parent classified the
*execution* as `completed`, not a qualified root. It ran for
250.520 s under the 600 s wall cap, with 926 sampled child-RSS
observations and peak **365,150,208 bytes** under the sampled 1-GiB
cap. The numerical child elapsed time was 247.960 s. There was no
resource termination or monitor error.

Fresh seed checks matched PR #212: maximum gradient
`0.006848117531724784`, slope margin `4.0606113e-4`, face margin
`1.1662714e-4` and the full 54-stage signature `50d3b1a4…`.
The seed's exact 26-column local SPD audit from PR #212 was reused
only after its child/parent/exit/resource/control/curvature/source/
input/runtime records and current scalar/branch checks matched.
This does **not** substitute for a final-root Hessian audit.

| Root-search measure | Observed result |
|---|---:|
| Newton systems solved | 8 |
| PCG iterations per system | 29 each |
| Accepted corrections | 8 |
| Rejected candidates | 67 of 75 trials, all by the measured switch policy |
| Accepted full-signature switches | 1 |
| Largest accepted-step true linear relative residual | `1.3605125544e-13` |
| Final accepted maximum gradient | **`0.006708378264014107`** |
| Final accepted scaled face margin | **`2.3865525984512733e-9`** |

The first accepted correction switched from `50d3b1a4…` back to
`695ecd60…`, the signature of the later accepted PR #209 path.
The next seven accepted corrections stayed in that signature. Their
backtrack counts rose to 15 on the eighth step, and the face margin
fell while the maximum gradient remained near `6.7e-3`. The refiner
then raised `stationarity refinement iteration budget exhausted`.
All eight PCG solves converged and their independent true relative
residuals were below `1e-10`; this refusal is not a PCG failure.

The last accepted gradient is about **67 million times** the strict
`1e-10` stationarity threshold. Its very small face margin is not a
`root_low_margin` result: that status requires a stationary point and
a fresh exact final Hessian audit, neither of which was obtained.
No final-root curvature certificate, adjoint, VJP, tangent, signed
nonlinear endpoint or physical validation was issued.

This single post-hoc restart reached a similar low-margin branch
region to PR #209 but did not find a smooth root under the same
predeclared eight-step merit policy. It does not prove root absence,
global convergence failure or that one face alone caused the stall.
R4-R remains open. Repeating nearby starts under the same policy
without a new mathematical hypothesis is lower priority than a
separately designed branch-event/globalization method or other
pending generalization axes.

The affected local selection passed **110 tests** with 18 existing
TorchScript warnings. Pinned basedpyright 1.39.9 reported zero
diagnostics for the new probe, parent runner and tests. A code-only
Graphify refresh recorded 7,762 nodes, 67,229 edges and 270
communities; the final rerun detected no topology change. GREEN and
RED independently checked the raw result, source/input identities,
accepted policy arithmetic and parent refusal classification.
These checks are not a full CPU/package regression.
