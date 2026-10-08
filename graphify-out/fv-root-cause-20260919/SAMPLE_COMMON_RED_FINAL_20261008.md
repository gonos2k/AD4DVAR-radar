# Sample-common cost-step RED result

Date: 2026-10-08
Scope: read-only audit of the one guarded sample-common original-J search. I did not rerun FV, AD, HVP, PCG, score, adjoint or reanalysis.

## Receipt and contract check

The completed run used plan SHA-256 `617a81486352591f3e17068be78fc4047db5bd4e07333b84dba48297ba9dcc3e`, with 115 source pins, 59 archive pins and base control `2cdccade…`. The child raw SHA is `7df18a05130cd9e2d12ffde96657bc9217ac42b130095b72212bd317e21c028e`; the run receipt points to that exact hash, and the resource receipt reports exit code 0 with no termination or monitor error. The child completed in 23.14 s; outer elapsed time was 24.23 s and sampled peak RSS was 529,743,872 bytes against 300 s / 1 GiB limits.

The fresh base J, full gradient, Φ and strict branch passed comparison with the pinned accepted endpoint. The direction was reconstructed from the stored inner sample pair, with θ=0.288468 and norm 0.13679778. The fresh slope was (g^Td=-0.0186907663), resolved beyond its (6.19\times10^{-16}) rounding budget before the one HVP began. HVP started/completed/call counts are all 1; PCG is 0. The base HVP scalars were (d^THd=5.66634024) and diagnostic-only (g^THd=-4.29196669). The quadratic J scale was α=0.00329856053, giving actual control displacement 0.00045123575 (0.90% of the 0.05 radius).

The sole candidate was accepted by original-J Armijo and its own strict endpoint branch/margins. Candidate and final-recheck J, gradient components, Φ and branch signature match exactly as serialized. Source hashes, fixed inputs, and runtime close; `candidate_committed=true` records one finite step. The record also explicitly keeps `full_root_claim=false`, score/response false, and does not claim an adjoint or forecast result.

## What the accepted point shows

Original J fell from 0.0613476383231 to 0.0613171251401, a decrease of 3.05132e−5 (0.04974%). The J Armijo limit was 0.061347632158, so the trial passed by a wide margin. The base quadratic J model predicted a decrease of 3.08263e−5, close to the observed decrease.

Φ rose from 0.01267324650 to 0.02995458291 (+0.01728134, or 136.36%). The full gradient norm rose from 0.159206 to 0.244763, and its infinity norm rose from 0.0885109 to 0.138523. All three squared-gradient blocks increased: field Φ contribution 0.00762438→0.01094267, flow 0.00410089→0.00941761, growth 0.00094798→0.00959430. Thus this run achieved the predeclared **cost-only** outcome while worsening the reported stationarity measures. It is not a stationarity improvement or a convergence result.

The instantaneous base-branch Φ derivative is (g^THd=-4.29197); its linear term α(g^THd) is about −0.01416. That term alone is not the finite-change prediction: linearizing the gradient as (g+αHd) gives Φ≈0.0300648, close to the observed 0.0299546. Both are diagnostics only, and neither is an acceptance test. The branch signature changed, so the base-local HVP does not certify behavior across that branch change.

The direction did **not** cross the target face. Applying the archived smooth face formula to the recorded full controls gives `Qy[4,3]` −2.37504e−6 at the base and −2.03456e−6 at the accepted point: it moved toward zero but remained negative. Therefore this run is not a face-crossing experiment. Base and accepted full branch hashes differ, and both analysis and future partition hashes differ. The child preserves hashes/partitions, not the full branch arrays or changed indices, so the changed analysis limiter/face locations cannot be localized from this receipt alone. Future-signature changes do not themselves explain J, which is based on analysis observations; the analysis partition change does mean the accepted J decrease cannot be attributed to the target face from these records.

## Actionable limits for any continuation

- Describe the result as one accepted original-J decrease with a substantial Φ/gradient increase. Do not call it successful stationarity correction, a face crossing, or forecast improvement.
- Before claiming a face mechanism, localize the changed **analysis** branch choices/signs against the base and report the target-face sign separately. If a future diagnostic needs causal attribution, retain compact full signatures or per-category changed indices alongside hashes.
- This plan remains pinned to base `2cdccade…`; it is not a resume plan for the accepted control `26e830dc…`. Any subsequent step needs a new immutable plan and receipts whose parent is this completed run, with base control hash `26e830dc1cb1bce1240b9f0f88ddf616cb279414fa1eb46d01b94e0f52dde14`, accepted J/g/Φ, branch and closure pinned. Do not silently rerun or reinterpret the old plan at the new control.
- Keep J, Φ, gradient blocks, endpoint branch eligibility, execution/resource success and physical forecast evidence as separate outcomes. The present run provides only the first four numerical/branch records; it computes no forecast score.
