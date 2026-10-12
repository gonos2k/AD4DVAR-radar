# RED preflight — bounded 7839 radius comparison

## Scope and status

Static review of the rank-aware solver, the explicit experiment configuration
path, the KKT radius selector, and inherited candidate gates. No fresh FV,
HVP, optimizer, candidate, repeat, score, or guarded run was performed by RED.
The expansion plan is not present yet, so plan SHA, source/archive pin closure,
and the frozen candidate/resource caps remain pending final plan validation.

The receipt-only endpoint admission path in the new driver checks the archived
PR301 winner and both independently eligible arm closures against the prior
endpoint. GREEN also records successful receipt-only admission of the committed
7839 control `7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`
with 26 control values and 27 residual values. The radius callback receives
fresh `G`, `D`, and `V` after current P2 qualification, the 24 GN row VJPs,
the single dense solve, and four fresh HVPs. No d19e `D/V/G` or HVP products
enter that callback.

## RED findings

The rank-one residual path uses the control metric `Q = DᵀD` through a QR
factorization of normalized `D`; it does not form or invert `Q`. An FP64
synthetic probe with independent columns separated by `1e-8` made the Gram
determinant round to zero. The rank-one solution and its gradients with respect
to `G` and `D` remained finite. This was a local solver probe, not a test-suite
run or an FV/model evaluation. The rank-one formula selects the minimum-control
norm point on the affine residual-minimizer set, and the orthant/radius cases
are compared against the zero and axis candidates. Existing focused evidence
records 11 passing rank regressions, including fixed-rank gradient checks and
both sides of the residual-rank threshold; the separate stored 60-test suite
remains prior baseline evidence and was not repeated here.

The shared runner's explicit config supplies the new endpoint, paths, hashes,
policy, plan loader, and radius chooser while retaining its existing actual
J/R gates (`comparison.search._ALL_GATES`), independent repeated P2/full-G
closure, both-arm requirement, and repeated-R-then-J selection. The chart
safeguard still contracts the actual retracted control displacement to the
chosen radius. `rho_R` and `eta_vec` affect only the recorded next-radius
factor; rejected or poorly modeled trials halve it, and no post-accept
expansion is applied. The trial retains the inherited 600-second internal,
660-second outer, 2-GiB sampled-RSS, four-HVP, 24-row, one-solve, and six
candidate-slots-per-arm limits, subject to checking the frozen plan.

The current selector first requires full-rank fresh `V`, a valid axis-0 KKT
point at the original radius, resolved lambda and inactive-axis multiplier
signs, and a future activation radius strictly above the original radius. It
then limits the selected radius to at most twice the original radius and
requires both a roundoff-resolved positive second coefficient and positive
predicted `R` gain over the gradient axis before candidate evaluation. These
guards close the earlier RED finding that the stored base KKT result was
recorded but not enforced.

No source-level acceptance-gate bypass or mathematical blocker remains in the
current static review. Final RED clearance is conditional on validating the
frozen plan's source transition (only the solver and shared runner), inherited
archive pins, new source pins, exact resource/candidate caps, and output paths.
Do not launch until those checks close. Any resulting candidate remains a
single bounded comparison, not evidence of convergence, a minimum, response
improvement, or forecast skill.
