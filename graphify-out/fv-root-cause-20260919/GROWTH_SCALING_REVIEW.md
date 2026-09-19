# GREEN / RED review — 2026-09-19

Both reviewers used Luna high and existing source/test evidence.

GREEN: the small-growth product retains exp(g) mathematically, uses a strict
|g|<0.125 branch, and preserves overflow guards and the original fallback.
The Decimal zero-flux case distinguishes the old rounding loss. Replay JVP,
VJP and mixed second derivatives remain covered. Current transport hash matches
the newly measured nominal 240x240 response. No global-minimum or global-Hessian
positive-definiteness conclusion follows from the local result.

RED: positive and negative small growth require boundary-budget verification.
The existing nonsquare-grid boundary test now covers both +0.02 and -0.02;
both passed, including budget, JVP/VJP and finite differences. Inverse-budget
scaling remains the original exponential product. No universal fused-arithmetic
or all-backend rounding guarantee is claimed.

GREEN isolated the affected partial-support finite-difference regression:
endpoint optimization error, amplified by 1/h, dominated the slope comparison.
The existing exact root helper now polishes both endpoints at 1e-10. The
comparison tolerance is unchanged; the corrected regression passes.

Large finite-perturbation comparison remains under review separately from the
nominal stationarity/adjoint checks. Typed prior promotion is also separate:
fixed-support external mean composition does not justify removing gates for
missing support or precision dependencies.


Final RED: the matched central reanalysis agrees with the adjoint to 4.39743e-5
relative; both endpoints have maximum gradients below 3e-10 and positive face
margins. Symmetric quadratic remainders agree within 1e-10. This is local
one-direction evidence for the declared rotation, not general FSOI eligibility.

The map publisher coalesces equal-colour horizontal runs only. GREEN compared
all original pixel colours against reconstructed runs, with exact equality.
Maps retain all 240x240 cells; each SVG is now about 0.5–0.6 MB instead of 3.35 MB.
