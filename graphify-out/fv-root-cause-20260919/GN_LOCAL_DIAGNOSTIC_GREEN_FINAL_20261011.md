# GREEN saved-data review: same-point GN local window at 454396

The frozen plan `c98b16be54351c1f5711abc86e93bdc067a59578914e18f14f12d187181781f8` freshly requalifies point `454396e77e2f10492e80e0fd625b55e1a08c2be53fe9721b59a4a27d60be3b7a`, rebuilds 24 row VJPs, one 12D solve, and two same-point HVPs, then samples eight α values `model_alphas[8]/2^k` along that one current GN direction. The raw archive is lossless; source, input, runtime, and deadline flags are true.

| Sample | α | J Armijo | R Armijo | ||E||/h | ||E||/h² |
|---:|---:|---|---|---:|---:|
| 1 | 3.07730e-3 | pass | fail | 3.295 | 1,071 |
| 2 | 1.53865e-3 | pass | fail | 2.352 | 1,529 |
| 3 | 7.69324e-4 | pass | fail | 1.907 | 2,479 |
| 4 | 3.84662e-4 | pass | fail | 3.804 | 9,889 |
| 5 | 1.92331e-4 | pass | fail | 7.597 | 39,501 |
| 6 | 9.61655e-5 | pass | fail | 3.892 | 40,471 |
| 7 | 4.80828e-5 | pass | fail | 7.779 | 161,783 |
| 8 | 2.40414e-5 | pass | pass | 6.096 | 253,559 |

All eight actual G vectors and model predictions are finite; all branch-pair, face, and side-objective checks pass. Every sample is explicitly `accepted=false`; sample 8’s R-Armijo pass is diagnostic only. No optimizer step or candidate commit occurred. The residual remainder `E=G(c+αd)-G(c)-αDG[d]` does not show `||E||/h→0`; `||E||/h²` grows substantially as α shrinks, so these samples do not support uniform second-order scaling or prove a derivative defect.

Compared with the base trace, every sample changes only `choose_left` at x cell `[1,2]`; left/right/slope signs agree and no face signs change. Changed zero-based trace index ranges are: sample 1, 111–120 and 171–180; sample 2, 111–114 and 177–180; samples 3–5, 111–112 and 179–180; samples 6–7, 179–180; sample 8, 180. The targeted stage-124 selector remains `choose_left=false` on both sides throughout. The squared G-model error is 99.92–99.96% concentrated in zero-based components 12–14, with component 13 largest; these are residual slots, not assigned physical variables.

The run completed in `64.988 s`, sampled peak RSS `700,907,520` bytes under 2 GiB, with 24/24 rows, one solve, two HVPs, zero event gradient/JVP calls, zero optimizer steps, and no commit. The focused suite reports 7 passed; type checking reports zero errors or warnings. This is an eight-point same-direction local diagnostic, not a continuous path certificate, convergence result, or evidence of a derivative defect.
