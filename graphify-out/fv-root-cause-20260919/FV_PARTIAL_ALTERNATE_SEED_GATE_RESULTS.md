# R4-R alternate-sector seed qualification, before root refinement

This is the single exploratory seed gate declared in
`FV_PARTIAL_ALTERNATE_SEED_GATE_PLAN.md`. The candidate was selected
**after examining PR #209's archived trial table**: among 29 finite
cross-signature candidates with both recorded margins strictly above
`1e-4`, exact stored-value ranking chose iteration 5/backtrack 3,
control SHA256 `125e0200…7421`. It was *rejected* on the prior path
because its gradient merit increased relative to the preceding
accepted iterate. This is a post-hoc alternate-basin initialization,
not an accepted continuation step or an independent observation case.

The guarded child and parent completed normally. The child exited 0
with `numerical_status=seed_locally_spd`; parent execution status is
`completed`. Child wall time was 23.507 s under the declared 120 s
limit, with 89 sampled RSS checks and peak 346,718,208 bytes under
the sampled 1-GiB ceiling. The inner numerical elapsed time was
21.801 s. No resource termination or monitor error occurred.

The current fixed two-hole objective and 54-stage branch were freshly
evaluated at that exact archived control. All scalar diagnostics and
the full branch signature match their stored PR #209 values:

| Seed diagnostic | Fresh value |
|---|---:|
| Objective (J) | `0.02076496753914496` |
| (\|\nabla_cJ\|_2) | `0.012208793948431365` |
| (\|\nabla_cJ\|_\infty) | `0.006848117531724784` |
| Scaled slope margin | `0.0004060611332810128` |
| Scaled face margin | `0.00011662714171735749` |

The face margin is only **1.166 times** the `1e-4` final-response
margin floor. The seed is inside the checked smooth sector at this
point, but has little margin for a subsequent step. Its maximum
gradient is roughly `6.85e7` times the `1e-10` stationarity gate;
the point is **not a qualified root**.

Only after that fresh branch check, the probe computed all 26 exact
gradient-JVP Hessian columns at the selected point. The matrix was
finite and symmetric to relative `1.62616e-16`; the minimum and
maximum eigenvalues were `1.0167708532836979` and
`4772.72431894504`, with ratio `2.130378e-4`. These pass the
predeclared local SPD/symmetry tests. They do not certify curvature
in other branches or guarantee that a Newton iteration will find a
root.

The PR #209 manifest/raw reports and the 16 objective/geometry
sources match their archived hashes; the two PR #210 status modules
match separately pinned current hashes. The plan, new probe and
guarded runner hashes were supplied at launch and checked before and
after. Fixed input, source and raw archive identity remained stable.
The child records **zero new product GN, Newton refinement, adjoint or
nonlinear reanalysis runs**. Response and physical validation remain
`not_performed`.

This result admits exactly one next research step: a separately
guarded **root-only** refinement starting directly from this fixed
candidate, with no seed substitution. It does not close R4-R; a
qualified classical response would still require fresh final
stationarity, exact final curvature, sufficient branch margins,
adjoint/VJP and signed nonlinear reanalysis.

The affected local selection passed **69 tests** with 18 existing
TorchScript warnings. Pinned basedpyright 1.39.9 reported zero
diagnostics on the new probe, runner and test. The code-only Graphify
refresh recorded 7,699 nodes, 67,065 edges and 271 communities; its
final rerun detected no topology change. GREEN and RED independently
checked the raw child, parent, resource and provenance records. This
is not a full CPU/package regression.
