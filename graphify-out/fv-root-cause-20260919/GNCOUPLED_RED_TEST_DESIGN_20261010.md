# RED synthetic regression contract: coupled Woodbury direction — 2026-10-10

Design/test specification only; no source edit or execution. These tests should exercise the production helper on tiny analytic inputs rather than implement a second production solver in test code.

## Low-rank solve identity and tangent/SPD checks

Use `n=(1,0)`, `P=diag(0,1)`, `t=(0,3)`, `A=[[3,4],[2,-1]]`, and `D=diag(2,0.5)`. Then `B=sqrt(D) A P` has only a tangent column; the restricted 1-D curvature is `(4 sqrt(2))² + (1/sqrt(2))² = 32.5`. The exact restricted system is `M_T=33.5`, so `d_y=-3/33.5=-6/67`. Evaluate the implemented Woodbury expression `−t+Bᵀ solve(I+BBᵀ,Bt)` and require agreement with this value, `n·d=0`, and `t·d=−18/67<0`. This checks the sign, projection, coupling, and SPD descent while the reference solve is only scalar; it does not allocate or factor a synthetic 26×26 matrix.

Instrument the isolated production direction helper to confirm its only dense system is `12×12` and the solve/factorization count is one. The implementation must not form or solve `I_26+BᵀB`; it can compute `Bᵀ z` after the small solve. Include D=0 or all-zero rows as a boundary case: S remains identity and returns `d=−t`. Also reject nonfinite A/D/t and require nonnegative finite robust curvature before `sqrt(D)`.

## Direction-override/HVP/full-envelope consistency

Use the face `Q(x,y)=x=0` and the synthetic side extensions

`J_-(x,y)=y+0.5y²+(-0.2+0.4y)x`,
`J_+(x,y)=y+0.5y²+( 0.2+0.6y)x`.

At `(0,0)`, `j=(0.4,0)`, `theta*=0.5`, `g*=(0,1)`, and the face-supported jump is normal. Supply the distinct tangent override `d=(0,-6/67)` (rather than the canonical `−t=(0,-1)`). Obtain each HVP by differentiating the corresponding side gradient along that same override. Analytically,

`h_-=(-2.4/67,-6/67)`, `h_+=(-3.6/67,-6/67)`,
`h_mix=(-3/67,-6/67)`, `theta'=7.5/67`, and `DF=(0,-6/67)`.

A `torch.func.jvp` of the complete residual `[g_-(c)+theta*(c)(g_+(c)-g_-(c)), Q(c)/0.84]` along `d` must equal the stored full `DF`. Require the returned model direction to equal the override, the HVP directions/digests to equal the override, both side products to be `−6/67`, and the envelope slope/`F·DF` to equal `−6/67`. This catches a subtle integration bug where the GN direction generates HVPs but the minimum-mixture model builder silently restores canonical `−t`; direction, side products, theta-prime, residual derivative, predicted alpha, and candidate chart path must all refer to the same d.

The end-to-end synthetic step should keep the existing actual J/Psi Armijo and fresh candidate theta-star checks. The GN metric's SPD/native-J descent proof is not a surrogate merit gate: a valid direction can still have nonnegative `F·DF` and must cleanly refuse the Psi search. Do not assert that every GN step is accepted or better than baseline.

The saved production base for the later comparison is the c8fba1 endpoint; this regression establishes algebra and integration coherence only. It says nothing about that endpoint's candidate outcome, exact constrained curvature, convergence, response, or forecast skill.
