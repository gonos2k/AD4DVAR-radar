# One joint-descent search step from the indefinite three-hour seed

The one predeclared `-Hg/||Hg||` exploration accepted **one strict
endpoint on a different minmod branch**. Both the unchanged analysis
objective `J` and stationarity merit
`Phi=||grad_c J||_2²/2` decreased, while the maximum control gradient
fell from `2.254729681485403` to `1.2605349625093007`. This is a
bounded root-search candidate, **not** a stationary analysis,
branch-preserving path, minimum, sensitivity or physical forecast
result.

The seed is the exact PR #218 alternate control, with unchanged
4×5 point observations, 13 parameters, background, verification
target, 0/10/20-minute observation times, 18 future ten-minute leads
and boundary schedule. The PR #220 exact-Hessian evidence is pinned:
its relative symmetry defect was `1.0949379174674463e-15` at this
same seed. The child computed one new exact
`Hg=JVP_c(grad_c J;g)` and used the **only declared direction**
`d=-Hg/||Hg||_2` in the existing latent control coordinates. It
recorded `||Hg||_2=8497.837241518195` and
`gᵀHg=21105.599973680168`, well above the FP64 scaled sign floor
`6.862479154494133e-10`. The seed slopes were
`D J[d]=-2.483643705314073` and
`D Phi[d]=-8497.837241518195`. These are local seed derivatives;
the latent Euclidean unit direction is not a physical trust radius.

Only the fixed grid `alpha=0.02*2^-k`, `k=0,...,7`, was available.
The first four tried values (`0.02`, `0.01`, `0.005`, `0.0025`)
reached individually strict endpoints on changed signatures but
increased both measured J and Phi, so were refused. The fifth,
`alpha=0.00125`, passed its own full 3,600-stage strict endpoint
check and the source-bound two-scalar decrease gate:

| Quantity | Seed | Accepted endpoint |
|---|---:|---:|
| `J` | `0.9323736303384679` | `0.9319504904669574` |
| `Phi` | `4.036592165302406` | `2.6187151084118883` |
| `||grad_c J||_inf` | `2.254729681485403` | `1.2605349625093007` |

The absolute reductions were `0.0004231398715104362` in J and
`1.417877056890518` in Phi, larger than their separate roundoff
floors (`2.6499652402050852e-14` and
`1.1472684961127033e-13`). The accepted control SHA256 is
`9f8d3ae0d77d8b24f1565ab622cf8fe7bbfd83766fbd67964a6a5bd52920a078`;
the accepted endpoint branch signature SHA256 is
`c1d60b87479fea5937ae44d4740cb0de3adf10e62fc50018cab69abb12ca47bb`,
different from the seed signature. Because the sector changed, the
old-sector Armijo slopes were **not** used as an acceptance proof;
only freshly measured J and Phi decreases were required. Neither
endpoint signatures nor this test certify smoothness of the segment.
The run stopped immediately after that single accepted endpoint.

The guarded child exited 0 with no wall/RSS termination or monitor
error. The guard measured **32.362 seconds** elapsed and 121 sampled
child-RSS readings with a peak of **395,788,288 bytes** under the
300-second wall trigger and sampled 1 GiB limit. The recorded
The raw child recorded **30.921 seconds** internal elapsed. Its
reported phase timings are a **partial** breakdown: their sum is
19.271 seconds, leaving **11.650 seconds** not separately attributed.
The `strict_seed_branch` phase (5.173 seconds) includes fixed-fixture
reconstruction as well as its branch check; the separate HVP took
7.227 seconds and cumulative candidate objective/gradient work took
4.963 seconds. Candidate branch checks likely account for much of
the unallocated interval, but were not timed separately. The parent
did not rebuild FV outside the guard. Source, archived input, PR #220
Hessian evidence, plan and
after-run identities matched.

The final affected suite passed **99 tests** with 18 existing
TorchScript deprecation warnings; 10 focused tests passed. Error-level
basedpyright 1.39.9 reported 0 errors/warnings/notes. GREEN/RED
reviewed the plan and code; a forged branch-change label that could
bypass same-branch Armijo was caught and fixed before launch. The
incremental Graphify code refresh produced a valid graph with 8,193
nodes and 1,057,103 links. Exact raw and source hashes are in
`FV_POINT_3H_MERIT_STEP_EVIDENCE.json`.

This closes only the **first exploratory joint-descent step**. The
accepted endpoint still has gradient maximum `1.2605`, far above the
`<1e-10` stationarity gate. Its own exact Hessian curvature,
adjoint true residual, full 13-parameter VJP, signed reanalysis and
physical skill were not tested. R5-R-R remains open. Any further
direction must be relinearized on this endpoint's own strict branch
under a new costed plan; the PR #220 seed Hessian cannot be reused
across the branch change.
