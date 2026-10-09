# GREEN design review: projected coupled-GN direction

**Decision:** the proposed comparison is mathematically coherent for the fixed all-detected 12-row profile at the saved c8 endpoint, provided the row-pair parity, exact face projection, and actual merit-HVP gates pass before acceptance. It is a local direction comparison, not a Newton or convergence claim. This review used saved predecessor arrays and source definitions only; it performed no FV, seed preparation, HVP, or guarded work.

## Saved starting point

The pinned PR272 predecessor is the three-step mixing-minimum resume, plan a3f24f93e84ec107c36a8888fbe2e644f9a5be3a71bff855ec219bf9be7c0bba, raw child 2a36ae30…, lossless archive 4d6d2eb1…. The terminal closure has control SHA c8fba1f93ee0b6c83edc5b56792de158921c0a64bab41eef5403c2a514b1fc33, carried theta 0.4818866600367756, native J 0.06123349298793274, fresh-minimum F² 0.004605971728402105, and target face Q 0. The two 26-component saved side gradients and identical branch-trace signatures close at this exact control. Recomputing the minimum from the saved side gradients gives theta-star 0.4818866600367756, jump norm 0.28491830770495113, and g(theta-star)·j ≈ 3.14e-18; this is a resolved, strict-interior mixing minimum.

The fixed profile used by the prior runner checks 26 controls, 13 parameters, a 3×4 all-detected observation mask, no neural prior, and zero field-smoothness weight. The data residual is the exact quality/std-standardized, fixed-correlation-whitened residual used by native J ([fv_point_research_problem.py](/Users/yhlee/ADVAR/src/advar/fv_point_research_problem.py:320)). The control-prior residual is identity on this profile ([variational.py](/Users/yhlee/ADVAR/src/advar/variational.py:5476)), and the selected face is the fixed tanh flow constraint ([fv_point_3h_face_transition_diagnostic.py](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_face_transition_diagnostic.py:84)).

## Direction and merit equations

Let g−, g+ be the two side gradients of native J at the common face, j=g+−g−, and theta-star the strict-interior minimizer of ||g−+theta*j||². For g=g(theta-star), let n be the face normal and P=I−nnᵀ/(nᵀn). The tangent gradient is t=P g.

The residual Jacobians A− and A+ have 12 rows by 26 controls, obtained as 12 scalar reverse-Jacobian rows per face side. Their projected row parity must pass the construction-scaled audit. Once it passes, construct B=sqrt(D)(A−P); on the common-face tangent this agrees with projecting the theta-star-weighted side Jacobian, within the recorded parity tolerance. Do not average unprojected side Jacobians before checking parity. With fixed standardized/whitened residual r and pseudo-Huber scale delta, the exact outer curvature is D_i=(delta/hypot(delta,r_i))³, the second derivative of the cost in [variational.py](/Users/yhlee/ADVAR/src/advar/variational.py:6552). Set S=I_12+B Bᵀ, and solve:

    d_GN = -t + Bᵀ solve(S, B t).

This is the Woodbury form of −(I+BᵀB)⁻¹t on the face tangent. Bᵀ maps back into the tangent, and S is symmetric positive definite with eigenvalues at least one because of the identity prior term. Use one 12×12 solve rather than an inverse. For nonzero t, tᵀd_GN<0, so the direction descends native J to first order under the common-face projection. This metric is a robust Gauss–Newton search preconditioner; it omits residual second derivatives and face-curvature multiplier terms, so it does not certify the true constrained Hessian.

That J-descent statement does not imply descent of the minimized gradient-mixture merit. For each arm direction d, obtain two fresh side HVPs h−=H−d and h+=H+d, then use:

    h_mix = (1−theta-star)h− + theta-star h+
    theta-star' = −[j·h_mix + g·(h+−h−)] / (j·j)
    DF = [h_mix + j theta-star', (n·d)/q_s]
    D Psi = g·h_mix + q (n·d)/q_s².

The saved base has q=0; each model still needs its envelope-slope and finite-derivative checks. The full theta derivative belongs in DF for the alpha cap, even though its contribution cancels from the scalar envelope slope at an interior theta-star. Actual candidate gates must freshly evaluate native J, both side gradients, candidate theta-star/minimum F², face, and branch pair. A transported linear theta is diagnostic only.

## Comparison and bounded run

The baseline and coupled-GN arm use the same c8 base, theta-star, current fixed residual, face, radius, and up-to-16-slot first-passing line search per arm. Compare each arm's first actually passing endpoint by actual minimum F², then native J, then the declared baseline tie rule. Only the selected endpoint receives an independent final repeat/commit. If either arm has no valid slope/candidate, record that refusal without weakening the other arm's gates. Verify the final repeat's per-side gradients and theta-star against the selected proposal.

The proposed allowance is one root-owned guard, 600-second inner /660-second outer limits, 1-GiB RSS, 24 row VJPs total, four current HVPs total (two per arm), one dense 12×12 solve for the coupled direction, and zero PCG/full-Hessian work. The row ledger must show 12 rows per side at the same c8 control and theta-star, and the HVP ledger must identify each arm's distinct direction digest. All progress remains provisional until the selected closure passes.

The saved PR272 diagnostic reports a 168.68° angle between the second and third prior step vectors, a net-displacement-to-summed-step norm ratio of 0.0987, and 99.92% / 99.98% of its two examined model errors concentrated in controls c12–c14. This flags strong step cancellation and localized model mismatch in that prior path; it does not establish the coupled-GN direction's success or a unique cause. The comparison should preserve per-arm actual residual/J outcomes and per-component model-error diagnostics and limit conclusions to this one same-point experiment.
