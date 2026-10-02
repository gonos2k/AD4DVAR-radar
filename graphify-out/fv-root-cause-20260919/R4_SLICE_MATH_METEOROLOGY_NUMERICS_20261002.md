# R4 interpretation and next verification contract

## Mathematics

Write the unchanged objective in face coordinates as j(t, eta, p)=J(C(t,eta),p), with 25 tangent controls t and eta=Qy[2,0]/0.08. The slice solver targets j_t=C_t^T J_c=0 at fixed eta. Its HVP includes both C_t^T J_cc C_t and the coordinate-curvature contraction with J_c. Tangent stationarity leaves sigma=j_eta=C_eta^T J_c uncontrolled; it does not establish the original 26-control stationarity condition. All four saved endpoints still fail that condition.

If a differentiable tangent-stationary branch t*(eta,p) exists with nonsingular j_tt, then d[j(t*(eta,p),eta,p)]/deta=j_eta. The experiment has not certified these hypotheses or tangent positive definiteness; therefore its sigma remains a partial coordinate slope. Opposite signs on finite positive/negative offsets with different limiter signatures do not prove a limiting cusp, generalized stationarity, uniqueness or absence of a smooth root.

An active Q=0 optimum and sensitivity would require a separate contract: a well-defined restricted discrete objective, tangent stationarity and nonsingularity, justified local normal optimality on both sides, and persistence under the requested p perturbations. A 25-dimensional restricted IFT alone would compute a constrained branch, not prove it solves the original unconstrained problem. Do not issue the existing classical full-root response for it.

## Numerical analysis

The saved second Newton solve for +2eta0 gives g^T s=-2.21618e-19 and s^T Hs=+2.21618e-19. Its local quadratic full-step objective prediction is -1.10809e-19, only 0.03194 ulp of the saved J=0.020765300308065577. The step norm is 1.27647e-10 and the audited relative linear residual is 1.35870e-11. This predicts a descent too small to resolve by subtracting two independently rounded scalar objective values, even though the tangent-gradient merit can decrease substantially.

The observed candidate increases (+4.60e-15 to +1.29e-14) are much larger than one final-scalar ulp. Their origin is not established by this arithmetic: forward evaluation, cancellation, component accumulation and higher-order effects have not been separated. Thus this is a scale diagnosis, not a roundoff error bound or permission to bypass the frozen actual-J policy. All 16 vetoes and the refused endpoint remain unchanged.

A next bounded numerical check should evaluate component differences on the same trial, without changing the objective. Prior difference can use (c_trial-c_base)^T(c_trial+c_base)/2. For pseudo-Huber rho_delta(r)=delta^2(sqrt(1+(r/delta)^2)-1), the exact real difference is

    rho_delta(a)-rho_delta(b)
      = (a-b)(a+b) / (sqrt(1+(a/delta)^2)+sqrt(1+(b/delta)^2)).

This removes loss from subtracting nearly equal loss scalars, but not error in the forward residual difference a-b. Any revised acceptance policy needs a component-scale error analysis or independent precision check; do not fit a larger allowance to this failed run. Initial nonfinite input, branch refusal, linear budget failure and objective-resolution uncertainty must remain distinct.

## Meteorological meaning

The prognostic echo proxy obeys discrete FV transport and multiplicative growth, corresponding to dq/dt+div(u q)=g q under the declared fixed boundary. Qy[2,0]=0 is a normal volume-flow event at one face: it is neither zero echo nor a missing radar observation. Its sign switches the upwind donor, which can alter the local derivative even when the forward transport remains defined. Changing only this shared face flux can break the streamfunction/divergence contract; a proposed Q=0 restriction must preserve the production face construction and mass budget.

The original two missing observations reduce observation information while complete initial/boundary support and priors remain. A prior-stabilized local Hessian would not by itself establish observation-based flow/growth identifiability. This experiment estimates no new radar state, forecast skill, physical growth law or missing-boundary contribution. The synthetic verification score and 4x5 domain remain unchanged.

## Checklist

- [x] Preserve original problem, input identities and accepted/rejected history.
- [x] Distinguish tangent stationarity from full stationarity; no response issued.
- [x] Recalculate saved Newton model, objective ulp and trial-policy scales.
- [ ] Establish stable objective-difference accuracy on the rejected trial with a bounded independent check.
- [ ] Justify a smooth-root search or an explicitly different active-face derivative contract before implementation.
- [ ] Verify any new contract against nonlinear parameter reanalysis; assess physical skill separately.

Evidence: R4_SLICE_NUMERICAL_SCALE_CHECK_20261002.py and its JSON reproduce only saved-vector arithmetic. The original 189.67-second FV run remains the integration evidence. No additional FV run, CI or deployment was performed for this interpretation.
