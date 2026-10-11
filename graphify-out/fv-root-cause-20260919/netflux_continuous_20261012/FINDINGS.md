# Fixed-version continuous gradient trial — 2026-10-12

PR298's accepted `0aafdf80…` point was continued under the identical `minmod_net_flux_fp64_v1` transport and fixed old-kernel synthetic observations/prior/parameters/boundaries/schedule/truth. Fixed case SHA is `1e810c23c73b93d89b2a10b46d9ddea44a8217b07dcd3ebfebb462858e3fa9b7`; transport SHA remains `b0335b7a72ecd22084d5ab8ea0e46707240be15ef0039f4b31f96adbc937ee1a`. No kernel, tolerance, objective, approval policy or solver direction changed. The new receipt-only endpoint loader reads the prior source and receipt tar bundles, validates their exact member sets/hashes, reconstructs the accepted full-G, and admits its new-version P2. Each iteration freshly observes the current point and matches that P2 before release, mixture/direction and two new directional HVPs. Old HVPs are not reused.

## One plan and actual result

Plan SHA `5fd879837c2e5898b16b01007aabae34d0eeea7da8b6d094d6d03b9dad366e07` freezes 205 source and 279 archive pins. The maximum is 20 commits / 40 HVP requests / 24 candidates per point, with 600 seconds internal deadline, 660 seconds outer guard and 2 GiB sampled RSS. Candidate normality `norm G <=1e-8` triggers exact qualification rather than a convergence claim. Postcommit residual qualification takes precedence over a rolling stop or cap. Three committed transitions with both R/maxmixed decreases <=0.3%, or any window-metric regression, stop for same-point direction comparison; strict stagnation and regression are separately recorded. A GREEN policy/implementation mismatch was corrected before launch, with a counterexample test. The unlaunched first plan/source/preflight is retained in `preflight_v1.tar.gz`.

The actual single guarded run closed **11 optimizer commits**, each with a **first-candidate** J/R pass and independent final P2/full-G approval. It ended **budget_exhausted**, not convergence, stagnation, support failure or commit cap. There are **23 completed /24 started HVP requests**: 22 products serve the 11 closed steps; one more side product was computed at the terminal attempted point before the other-side request hit the internal deadline. There are 11 completed candidates and 0 GN rows/solves or future-E evaluations. All confirmed J/R transitions decrease, and per-point full-G/previous-P2/native pair checks close.

| Metric | Fixed-version start | Final confirmed point | Decrease |
|---|---:|---:|---:|
| J | 0.061018648644175 | 0.061003289930848 | 0.02517052% |
| R | 0.0027434407274438 | 0.0025752047662763 | 6.13229801% |
| norm G | 0.05237786486145 | 0.05074647540742 | 3.11465436% |
| max mixed component | 0.02621282766019 | 0.02547686391175 | 2.80764730% |

Final control SHA: `b6470069c9b8992bf19fa85ab848f5ef424a1e2e8e33a4909bcb0d036ef88660`.

| Commit | J | R | norm G | max mixed | R decrease from previous | Candidates |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.06101685189866 | 0.0027287038470 | 0.05223699692 | 0.02610968351 | 0.537168% | 1 |
| 2 | 0.06101586360139 | 0.0027072098012 | 0.05203085432 | 0.02606921106 | 0.787702% | 1 |
| 3 | 0.06101407661013 | 0.0026930259390 | 0.05189437290 | 0.02598018025 | 0.523929% | 1 |
| 4 | 0.06101309844119 | 0.0026748467447 | 0.05171892057 | 0.02594088525 | 0.675047% | 1 |
| 5 | 0.06101133592137 | 0.0026610053153 | 0.05158493303 | 0.02585431480 | 0.517466% | 1 |
| 6 | 0.06101036633234 | 0.0026424682880 | 0.05140494420 | 0.02581545790 | 0.696617% | 1 |
| 7 | 0.06100863092193 | 0.0026288613466 | 0.05127242287 | 0.02572994516 | 0.514933% | 1 |
| 8 | 0.06100767019184 | 0.0026158336532 | 0.05114522122 | 0.02569130456 | 0.495564% | 1 |
| 9 | 0.06100594827905 | 0.0026029962908 | 0.05101956772 | 0.02560057788 | 0.490756% | 1 |
| 10 | 0.06100499619538 | 0.0025903289208 | 0.05089527405 | 0.02556431147 | 0.486646% | 1 |
| 11 | 0.06100328993085 | 0.0025752047663 | 0.05074647541 | 0.02547686391 | 0.583870% | 1 |

The most recent completed three-transition window has R decrease 1.553191% and maxmixed decrease 0.834682%. Neither rolling stagnation nor metric regression was detected. This supports continued progress in this bounded run; it supplies no fixed rate or extrapolation to the final threshold. A same-point curvature/GN comparison was not triggered and was not run.

## Execution, prefix and operator provenance

The outer guard completed with exit 0 in 605.900539 seconds, sampled maximum RSS 589889536 bytes (0.549377 GiB), no timeout/memory termination, no SIGTERM and no monitor error. The raw child reports 601.698216 seconds; `deadline_passed=false` correctly accompanies internal budget exhaustion. Stop/bookkeeping time is included in the guard total; this is not a claim that the complete run finished inside 600 seconds. Source/fixed non-control input/runtime closure passes. The full input identity advances with the intended committed control.

The final `current_point` belongs to this final confirmed control, unlike the previous one-cycle cap's stale start-point operators. Its new native/P2/support/direction checks were reached and **only one side HVP completed**. No paired HVP/model or postcommit readiness is certified; the partial product is not serialized or reused. The last confirmed control/theta/full-G and all 11 atomic prefix receipts remain intact. `verify_result.py` independently reconstructs each stored full-G/P2/Armijo prefix and metrics; it does not run FV/HVP/forecast again. `receipts.tar.gz` + its manifest restore exact original raw child, candidates, log and run/resource bytes.

## Verification and remaining limits

The final scoped test set has **55 passed /1 deselected, 9.34 seconds**, with 18 existing torch.jit.script deprecation warnings. The four new modules/tests have 0 basic-mode basedpyright errors/warnings. An initial widened set had 55 passing and one failing historical `test_archived_98ee_endpoint_load_is_receipt_only`: its untouched old live-source producer requires the old transport hash and correctly rejects the PR298 kernel. That source-domain-incompatible test is recorded, not renamed as a pass. The new endpoint and legacy-plan-refusal tests verify the intended new path; this is not a full-suite or remote FV regression claim. Earlier overlapping subset and historical/user test counts are not summed.

Graphify consulted prior structural evidence and refreshed four added source/test files with six cached prior files: 124 nodes /221 edges /17 communities, zero semantic tokens. GREEN/RED final audit records accompany the result. The read-only `FINAL_VALIDATION_INPUTS.md` maps current fixed-case metadata to exact response, actual signed reanalysis and future-score paths; no response or forecast was executed.

The final norm (~0.05075) and maximum component (~0.02548) remain far from the current normality candidate threshold. Q=0 remains a conditional active-face hypothesis, with no general face-exit solver. Limiter signatures changed at commits 2,4,6,11 (zero-based point_index 1,3,5,10). Their vector-model errors are 5.255e-4–1.081e-3, or 14.7707–28.2704% of actual full-G change; stable-signature steps have about 0.002624–0.008424% relative error. Side-pair and candidate-repeat traces still close, and actual J/R plus independent P2/full-G approval remains valid. These are candidate acceptance results, not uniform Taylor or HVP-accuracy certification. No eligible original minimum, exact reduced curvature/adjoint, actual p±h reoptimization, new-version E baseline/final score, or independent forecast validity is complete. Historical weighted research milestones are not advanced by these incomplete qualifications.

The next resume must freshly complete the two-sided operator/model qualification at `b6470069c9b8992bf19fa85ab848f5ef424a1e2e8e33a4909bcb0d036ef88660` under the same source/input version. Continued gradient progress is observed; exact response and independent future validation still require their eligible-point gates.
