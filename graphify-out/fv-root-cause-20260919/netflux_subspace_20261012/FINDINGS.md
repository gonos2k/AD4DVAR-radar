# PR300 follow-up: finite two-direction correction and d19e observation

## Outcome

The bounded fresh-point experiment completed and confirmed one new conditional Q0 point `7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`. The nonnegative two-direction model selected the gradient axis at the frozen small radius, with GN coefficient zero. Both independently repeated arms produced identical controls, J and R; the tie rule chose gradient. This experiment provides **no incremental subspace benefit over its gradient comparator**, no multi-step efficiency result, and no fast-convergence evidence.

PR300's two requested comparisons remain closed. No production b647 HVP or pure gradient/GN comparison was repeated. The new point d19e was admitted from pinned PR300 receipts, its native pair/P2/outward/minimum-mixture state was checked freshly, and both new direction columns and four HVPs were recomputed at d19e.

## Cached finite-error diagnosis

The verified control/full-gradient layout is echo controls `0:20`, flow controls `20:25`, growth control `25`, with face residual at `26`. A proposed `flow[12:17]` grouping was rejected on inspection of the source contract. Those indices belong to the initial echo controls. `diagnostic.py` reads receipt-tar vectors only and reports no production regeneration or causal selector proof.

At the nearest rejected/accepted GN candidates #9/#10 from PR300, error norms in the echo block are about `0.00813264 / 0.00377652`; flow is `0.00012900 / 0.00005894`, growth `0.00002869 / 0.00001595`, face about `2e-18`. Largest error is echo component13. The d19e mixed-gradient theta correction is confined to flow, and does not explain the dominant echo error. Neighboring GN traces differ in nine choose_left flags per side, with no sign flips; the archived payload has no base selector array, so a causal selector attribution is unsupported.

## Frozen correction and finite results

The model solves `min(a>=0) 0.5||G+Va||²`, `||Da||<=Delta`, with matching D/V normalization and a two-variable active-set solve. Near positive dependence reduces the domain explicitly; antiparallel cancellation is refused. The general radius-active solve retains differentiable tensor arithmetic and implicit multiplier corrections, tested away from active-set transitions. These search coefficients are not an exact reduced Hessian or an adjoint.

Initial Delta was fixed before launch to PR300's accepted gradient displacement `1.8757177916958825e-5`. Rejected/nonfinite or rho_R<0.25 or eta_vec>0.5 halves the radius; otherwise it holds, with no expansion. Each arm stops at its first eligible candidate. Nonlinear chart displacement is capped by <=33 chart calls, with the same safeguarded alpha used in J/R bounds and residual prediction. All actual original J/R, candidate theta, branch pair, individual gradients, independent repeat and full-G approval checks remain in force; no base-selector invariant is imposed.

| Quantity | d19e start | Both repeated candidates |
|---|---:|---:|
| J | 0.061002341038211755 | 0.061001393469310765 |
| R=||G||² | 0.002559866170789153 | 0.002548715176361377 |
| J decrease | | 0.0015533320% |
| R decrease | | 0.4356084922% |
| maximum mixed-gradient component | 0.0254428 (PR300) | 0.025399639444261 |

The subspace active set is `axis_0`, with raw coefficients `[0.0003707309698259479, 0]`; there is no near-dependence fallback. Both actual path norms are `1.8757177916130278e-5`. The chart alpha is applied to the model and coefficients. Both candidates preserve the base branch signature **as an observed outcome**, not as an acceptance requirement. Vector model error is `9.6932323e-8`, eta_vec `4.5574584e-5`, rho_R `1.000031318`. The radius-update rule would hold; no second move was attempted.

Fresh chi remains direction-specific: gradient `0.5303957%`, GN `87.3952825%`. Despite this GN local alignment, at this one frozen radius the orthant-constrained full model optimum is the gradient axis. Larger radii and future points have not been established by this result. Do not recycle this experiment by adjusting caps/tolerances to get a preferred result.

One global budget included all preparation: 72.485s child / 74.814s guarded wall, sampled peak RSS `1,284,161,536` bytes (about1.196GiB). Counts: four fresh HVPs, 24 row VJPs, one12x12 GN solve, two tiny coefficient candidates, two actual FV candidates, two independent repeats, one commit. Common preparation was about51.927s; gradient search/repeat7.180s, subspace10.938s. Adding common preparation to each arm gives59.106s/62.865s; these overlapping inclusive costs must not be added as independent budgets.

## Separate observation at d19e

The separate producer made exactly two terminal score calls, zero optimizer/candidate calls, and never used E for selection. Same numerical transport version, old fixed parameters/observations/truth and score profile:

| Point | Terminal +180min dBZ MSE |
|---|---:|
| b6470069… | 7.386491165784028 |
| d19e882d… | 7.386414763452886 |

Delta E=`-0.0000764023311417 dBZ²`, a `0.0010343522%` decrease; RMSE decrease is about`0.0000140559 dBZ`. This is one matched fixed-truth observation at the final lead after the20min analysis window, not a lead average, statistical/meteorological skill evidence or independent validation. **7839625d… is not scored.** Observer guarded wall16.971s; sampled RSS1,403,125,760bytes. The prelaunch observation plan was superseded only to add explicit provenance fields, with zero evaluations under the superseded plan.

## Verification and remaining limits

44 affected tests passed in10.71s, including two active-radius gradchecks, a deterministic40-case dense-angle oracle, fresh paired-HVP/counter corruption/no-fallback tests, chart-radius contraction and good-scalar/poor-vector fidelity checks. Basic pyright on five new source modules/scripts:0 errors/warnings. Stored-vector/result audit and observer closure checks pass. The 216-member source archive and all287 inherited archive pins were checked by GREEN/RED; candidate and observer raw receipts are separately hashed. These are focused local tests and stored-vector audits; production HVPs were generated once by the experiment, not independently regenerated by reviewers. Full repository/FV/remote CPU/package regressions were not repeated.

Completion stays about70% under the historical milestones. R norm remains about0.05048, far above the unchanged1e-8 candidate qualification threshold. Exact minimum/stationarity/curvature qualification, exact adjoint/VJP, p±h reoptimization, actual reanalysis and independent future validation remain incomplete. No search-GN inverse substitutes for them. No source/kernel/prior/observations/limiter or candidate tolerance was relaxed.

Next work should use this closed current-point evidence to make an explicit path-policy decision; it should not infer that GN switching, this two-variable correction, or continued gradient steps will complete convergence quickly. The current observation path can be reused for later admitted points; exact adjoint/reoptimization remains gated on an eligible optimal condition.
