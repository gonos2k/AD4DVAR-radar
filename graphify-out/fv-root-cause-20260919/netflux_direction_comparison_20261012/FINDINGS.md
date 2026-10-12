# Same-point gradient / robust GN comparison — 2026-10-12

The fixed minmod_net_flux_fp64_v1 evaluator and original old-kernel synthetic observations/prior/parameters/boundaries/schedule/truth were retained. Starting control is b6470069…ef88660. No partial HVP from the earlier budget stop was reused. A receipt-only new endpoint loader admits the closed 11-step prefix from source/receipt tar bundles. Both directions freshly qualify current P2 and use the same minimum mixture, selected Q0 chart, radius and original J/R policy. Q0 remains a conditional active-face hypothesis.

## Result

The curvature direction improves local residual alignment dramatically, but its large finite candidates do not realize that model. Gradient is selected by actual independently repeated R (then J within scale-aware ties); both arm candidates were fully P2/fullG approved before one commit. This is a completed same-point comparison, not a universal ranking of solvers.

| Item | Gradient | Existing robust GN |
|---|---:|---:|
| chi, local maximum removal | 0.485297% | 87.468153% |
| Model minimum alpha | 0.000369625235395 | 0.777448358389 |
| Accepted alpha | 0.000369625235395 | 0.00151845382498 |
| Actual candidates | 1 | 10 |
| Repeated J | 0.061002341038212 | 0.061000864456041 |
| Repeated R | 0.0025598661707892 | 0.0025677514835533 |
| J decrease | 0.00155548% | 0.00397597% |
| R decrease | 0.59562625% | 0.28942486% |
| Max mixed-component decrease | 0.13361712% | 0.13200482% |
| Movement norm | 1.8757177917e-05 | 5.8695204847e-05 |
| Arm time inclusive of preparation | 33.799126s | 89.304565s |
| GN-only rows/solve phase | 0 | 14.909892s |
| Vector model error / actual G change | 41.792103% | 100.073408% |

The first nine GN candidates pass J Armijo, minimum mixture, side pair, face, finite gradients and other native eligibility gates, but fail the R Armijo condition. The tenth succeeds, at 1/512 of the model-minimizing alpha. The largest candidate has J=0.06049526846046454 but R=0.041299682302486276 versus starting R=0.0025752047662763056. Every evaluated candidate changes the limiter signature relative to the base, including the accepted gradient candidate. Both accepted candidates close their own side pair and independent repeated traces. Branch changes and nonlinearity accompany large finite-model errors; this evidence does not identify one changed selector as the sole cause or prove a bad current-point HVP.

The selected point is d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e. Its J=0.061002341038212, R=0.0025598661707892, norm G=0.05059512002940, max mixed component=0.02544282245999. The GN candidate lowers J more but has higher R and slightly higher max component; those tradeoffs are recorded rather than ranked by model chi or future E. The frozen selection rule uses repeated actual R, with J tie-breaking, so gradient is selected.

## Cost and execution contract

One global 600s internal /660s outer/2GiB sampled RSS trial includes common admission/current-point work, both arms and the GN-only 24 VJPs/one 12x12 Woodbury Cholesky solve. There is no independent 600s per-arm allowance. Gradient runs first; a budget stop or unsupported arm would leave an incomplete comparison and no single-arm fallback. Common preparation is 6.394537s. These single-run timings include actual ordering/warm-up/bookkeeping and are not a statistical/general speedup benchmark.

The guard completed at 133.639551s, sampled peak RSS 1119698944 bytes (1.042801GiB), exit0, no resource termination/monitor error/SIGTERM. Four fresh direction HVPs, 24 completed row VJPs, one 12x12 solve, 11 actual candidate evaluations, two independent final repeats and one selected commit are recorded. Source/fixed non-control inputs/runtime/deadline close. Producer plan 36686ccf05c55ab9dff1d6c0c4658e14049a108eba402b76c86fc0b9ac904b35 pins209sources283archives; only four new caller/endpoint source/tests were added, all205 inherited numerical sources unchanged.

## Limits and next decision

The robust GN metric is identity prior curvature plus projected whitened pseudo-Huber row curvature. It omits residual curvature; positive definiteness is a search-direction property, not exact constrained curvature, minimum or adjoint eligibility. chi is measured from fresh direction-specific DGd at b647; it is neither an FV condition-number estimate nor a guaranteed finite gain. The final selected d19e point has no new direction/HVP/model qualification; archived operators belong to b647.

The empirical bottleneck has narrowed: GN yields a much better local residual direction, but its large J-descending candidates fail R amid changing finite branch structures and large vector errors. Repeating plain gradient solely because the historical stagnation detector stays off is not justified as an efficiency conclusion. Before further large-step curvature iterations, examine branch/local-model validity and the R-only refusal constraint at the next fresh point; do not relax tolerances, treat selectors as permanent physical constraints, or remove the R guard merely from this observation.

A separate frozen observation-only producer scores E_new at the earlier 0aaf start and b647 end of the 11-step run against identical fixed old truth. It does not score/select this new d19e candidate. No E is evaluated by the comparison. Eligible minimum, exact curvature/adjoint, actual p±h reoptimization and independent forecast validation remain open. Historical weighted milestones are not advanced by these uncompleted gates.

Verification:43focusedtests pass29.91s,18existing torch.jit.script deprecation warnings; six new files have0 basic-mode basedpyright errors/warnings. This is not a full repository or remote FV regression claim. Source/arithmetic/receipt audits by GREEN/RED accompany the result; verify_result.py reconstructs stored vectors/approvals only, not another FV/operator run. Scoped Graphify updates six new source/tests plus10cached prior files andone unchanged relevantGN file (219nodes417edges,zero semantic tokens).
