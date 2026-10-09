# RED source preflight: interior minimum-norm mixture mode — 2026-10-10

## Disposition

After the status fix, no blocking source/math defect remains for the single planned bounded probe. The final frozen plan `e593b4516c06644190bdac7dd9ab295ba20152c4ce2730cf998cf90d6866a3f6` passes the read-only plan/base loader with 133 source pins and 124 archive pins. It resolves the expected `0da57dbc…` control, carried `theta=0.3298981720696071`, native `J=0.06124393564803218`, and carried `F²=0.005132555144362135`. No production computation, seed reconstruction, test run, or guard was performed for this review.

The shared runner now chooses `tangent_mixing_minimum_accepted/refused` based on the plan kind instead of reporting the precondition-comparison status. The recorded focused-test log is 75 passed with 18 existing warnings; the type log reports 0 errors, warnings, or notes. I read those artifacts without rerunning them.

## Math and state checks

For `j=g_+−g_-`, `theta_star=−g_-·j/(j·j)` is the unique minimizer of the side-gradient mixture norm when the jump is resolved. The helper enforces a strict interior margin and refuses boundary, out-of-domain, or unresolved weights without clipping. The face-gradient support check remains in the baseline `tangent_direction` path. The proposed direction is the existing `−P g(theta_star)` chart tangent.

The residual derivative includes

`theta_prime=−[j·h_mix + g(theta_star)·(h_+−h_-)]/(j·j)`,

so the vector `DF` includes `h_mix+j theta_prime` and the face derivative. The scalar envelope slope `F·DF=g(theta_star)·h_mix + (Q/0.84²)(n·d)` is checked against the full residual dot product. The existing norm-model alpha cap uses that full `DF`, while candidate evaluation recomputes a new interior theta-star from fresh side gradients at the actual candidate; the linear theta prediction is diagnostic only and cannot reject an otherwise-valid candidate. A synthetic AD/JVP regression verifies the full derivative and shows that omitting theta-prime preserves the first scalar slope but changes `||DF||²` and the model-optimal alpha (`1` versus `.8`).

The initial theta-star and `Psi(base)` are recorded under `base_mixing_minimum` with `optimizer_step=false`, separately from the carried input theta and old `F²` receipt. The solver keeps the carried control/theta until a nonzero control candidate passes actual J and Psi/F² Armijo, face/branch checks, and an independent final repeat that recomputes and matches its interior theta-star and each side gradient. A refusal at the base, an invalid candidate theta, an interrupted HVP, or failed final closure leaves the previous control/theta unchanged. The two side HVPs are fresh at the current control/direction; history labels preserve both carried theta and working theta-star. The parent closure checks the mixing-minimum result status, base/commit theta chain, and HVP labels.

The historical precondition result contains a saved side-gradient-only theta-min value near `0.32773766` at the `0da57dbc…` point, versus the carried theta `0.32989817`. It is useful as a static consistency hint only. The new run must reconstruct `g_-`, `g_+`, branch pair, J, source, and fixed input under its frozen live source before recomputing theta-star; this pre-run normalization must not be labeled a step. The old PR270 rejected candidates remain rejected under their original transported-theta policy.

## Limits

This mode uses the original native J, selected face, and baseline tangent control direction. Theta minimization can remove only the normal component when the side jump is face-normal; it cannot establish tangent stationarity by itself. Interior mixing at base/candidate is a supported domain restriction: boundary and unresolved cases refuse rather than switch to endpoint mixtures. If the tangent direction is zero, no theta-only control-free commit is made. Curvature/minimum qualification, response/reanalysis, forecast skill, and independent future validation remain outside this probe.

The raw base diagnostics deliberately retain both the old carried-theta `F²` and the new base `Psi`; subsequent result analysis must use the explicitly labeled `base_mixing_minimum` and candidate theta-star, not reinterpret the old `accepted_F_squared` as `Psi`. The two-HVP/240-second/300-second/1-GiB policy is frozen in the successor plan. This preflight establishes source and formula consistency only, not runtime or numerical outcome.
