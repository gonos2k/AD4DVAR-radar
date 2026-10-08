# Face-transition diagnostic RED result

Date: 2026-10-08
Scope: read-only audit of the single guarded diagnostic at the accepted PR #260 resume endpoint. I did not rerun FV, AD, HVP, PCG, an optimizer, a forecast score, an adjoint, or a reanalysis.

## Receipt and execution check

The frozen diagnostic plan is pinned at SHA-256 `6b79d07ce609ebe35879dcb3f1cec95649a786155c392e7ea041063b4e19283e`. The retained child `face_transition_attempt1/diagnostic.json` has SHA-256 `ffa9473e5c0852c8f525665d8e7ac4255ad19820aca9951ef6c137ea905798ce`; its parent receipt is `0e9c7a53d37f345bcbb690f9becc837d4355eb8cd02df53b1e0ca826273bbff8`, and resource receipt is `3b1ca80ca0c713b0b3fcd9c56f10bb6c5301f0cde1c9451990a72aa5e52e3c19`.

The parent and child report completed execution with the expected finite two-sided diagnostic status. All four nonzero probes passed their own strict endpoint branch and margin checks; each recorded 3,600 observer callbacks, of which 360 analysis stages have donor traces and 3,240 future stages appear only in the full branch signature. The η=0 record contains a primal J only and explicitly says no gradient and no branch check. Source, fixed-input and runtime closure flags all match. Counters remain zero for HVP, PCG and optimizer steps. The guarded attempt took 33.74 s wall time and sampled peak RSS was 824,360,960 bytes, under the 300 s / 1 GiB guards (RSS is sampled, not a hard allocation ceiling).

## Finite-sample findings

The selected production face is external `Qy[4,3]` at base value −2.37504e−6. The exact-face chart point has realized flux −6.94e−18 and original full-control (J=0.0613473890144), lower than all four nearby nonzero probes. The signed unit-normal gradient components at −2e−6, −1e−6, +1e−6 and +2e−6 are approximately −0.081706, −0.081386, +0.200349 and +0.200622. The corresponding chart-coordinate η slopes are −0.105244, −0.104907, +0.208152 and +0.208438. This sampled cost pattern is consistent with a local cost minimum near the face, but it is not an exact one-sided derivative or a proof of a kink minimum.

The measured Φ values are 0.01266937 at −2e−6, 0.01265944 at −1e−6, 0.02945016 at +1e−6, and 0.02952185 at +2e−6. No Φ value is assigned to η=0. For the inner pair, the full gradient jump has norm 0.28173883; its projection on the common zero-chart normal has norm 0.28173484 and the tangent residual norm is 0.00149925, so about 99.99717% of the squared jump is normal in this Euclidean raw-control metric. The outer pair gives 0.28234370 total, 0.28232778 normal, and 0.00299849 tangent. The recorded full-gradient identity matches the sampled ΔΦ for each pair to stored precision.

The inner sampled gradient segment's minimum-norm point has θ=0.288468 and norm 0.136798; its negative direction has pairing −0.0187136 with each endpoint gradient. The outer segment gives θ=0.288599, norm 0.136745 and pairings −0.0186991. No direction was applied. These are finite two-gradient segment diagnostics, not a Clarke certificate or a claim that the diagnostic direction is a valid model step.

## Branch and donor-trace interpretation

The full stored face-sign arrays show that the only face-sign slot changing between each negative/positive pair is the target `Qy[4,3]`, and it changes at all 3,600 stages. Same-side pairs have no face-sign changes. This localizes the sampled face-sign flip. However, the limiter-choice signatures also change at future stages within same-side pairs (two stages for each pair); across the inner cross-side pair two future stages differ, and across the outer pair six future stages differ. All four endpoints pass strict margins, but they do not share one identical complete limiter-choice branch. Future limiter changes are independent of the analysis-only J and are not a cause of its gradient difference. The analysis endpoint choices are identical and the target face is the sole sign change, strongly supporting the single analysis-interface hypothesis. Exact limits, intervening paths and adjoint-weighted trace contributions are still unverified, so causal certification is not claimed.

At each analysis stage, the difference between the reconstructed interior top state and the effective external boundary state ranges from about −0.9151 to +3.6877 across time; it changes sign. The raw/effective boundary growth adjustment also varies with the stage schedule. This supports a sampled donor-trace description, not an adjoint-weighted contribution to J or proof that this face alone caused the Φ change. Future-stage donor traces were intentionally not stored; future branch signature comparisons do not substitute for them.

## RED conclusion

The bounded diagnostic completed as designed and gives strong finite-sample evidence that crossing external `Qy[4,3]=0` coincides with a large, predominantly normal gradient change, while the original cost samples are lower near the face than on either side. Finite probe spacing, unverified intervening paths and missing adjoint-weighted trace attribution limit the J causal claim. Future donor traces were outside this analysis-only J diagnostic and do not establish or undermine the observed J jump. This run does not establish exact limiting gradients, Φ discontinuity at zero, a generalized-gradient or constrained stationary point, global branch regularity, forecast skill, or meteorological significance. No further execution is warranted to interpret this receipt; additional localization would be a separate offline analysis.
