# RED final review: local GN direction window — 2026-10-10

## Disposition

The findings report, KG update, verification record, and saved-array analysis agree with the raw attempt-2 diagnostic and the requested scope. I found no blocking numerical, provenance, or claim error. The seven dual-Armijo passes are finite, uncommitted samples on one stored direction; they do not add an optimizer point. No additional guarded run is needed for this content review.

## Evidence checked

- Plan SHA-256 is `f0bce6629d3de1071276ce3cf93dcec70a2cba79f8d5661f847f7826670a92a9`, with 146 source pins and 185 archive pins. The attempt-2 raw child digest is `1c2139639661a9fb3f1c1937522b97f558010cb7cbcbd765e9ffdcbb3d777f0b`; its gzip round-trip is lossless, and the run child digest, resource receipt, and archive manifest agree.
- Attempt 2 completed in 45.361 seconds at sampled peak RSS 1,303,576,576 bytes under the 2-GiB cap. It records two fresh same-point HVPs, zero new row VJPs or dense solves, eight completed samples, zero optimizer steps, and no committed candidate. Source, fixed input, and runtime are unchanged.
- The attempt-1 path-admission failure remains separately pinned. It ended after 1.068 seconds with no diagnostic child and numerical status `not_reached`; its failure receipts and source/plan/test snapshots are included in the corrected plan's archive pins. The corrected plan and report do not count that failure as numerical work.
- Fresh-versus-stored HVP and residual/model component comparisons report zero maximum difference. Independent arithmetic on saved sample vectors reproduces squared residuals, remainders, and chart displacements to rounding precision.

## Numerical and branch interpretation

At sample 1, α=`1.1148976753e-5`, J Armijo passes while residual merit R (the squared norm of G) Armijo fails (`0.00448582625125` versus RHS `0.00447848921654`). Both side traces change one selector from the base at stored trace index 124, x `choose_left`, row 1, column 2; face signs remain unchanged. The normalized active-limiter gap is `3.12e-8`, well above the `128*eps` tie threshold, so the saved evidence identifies a resolved selector change rather than a recorded near-tie. The selector change and residual-merit failure co-occur; these samples do not isolate a unique causal effect.

Samples 2–8 retain the base selector and face-sign signatures at each evaluated point, and both J and residual-merit Armijo tests pass. Sample 2 is the largest tested dual-pass point, with R=`0.00447842981626`; it reduces R by `0.00132661%` and J by `0.00002205%`. These are pointwise results on the eight-sample grid. They do not certify every point between samples, an accepted line-search step, or a new optimizer endpoint.

For the remainder `E(h) = G(c(h)) - G(c0) - h DG[d]`, `E/h²` is about `0.593` and `0.597` at samples 2 and 3, then rises through `1.87`, `13.99`, `29.8`, `197.5`, and `593` as alpha shrinks. The absolute remainder fluctuates around a few `1e-12`. The report correctly avoids claiming uniform O(h²) behavior or assigning the small-step floor solely to roundoff.

## Required actions and limits

No correction to the reviewed findings, KG update, verification, or analysis is required. Keep the result described as seven sampled dual-Armijo passes and one sampled selector-changing residual-merit failure. Do not register a sample as an accepted or confirmed optimizer point, attribute the failure exclusively to the selector switch, claim a continuous selector-stable segment, or claim uniform second-order error. Any future step selection and final endpoint closure would be separate work.

The shared structural Graphify update is a separate workspace incident recorded in `GN_LOCAL_WINDOW_GRAPH_SCOPE_20261010.json`. Its generated graph/report files had no available before-bytes or backup and are not pinned numerical evidence. This RED memo does not treat that cache update as numerical validation or recommend rollback without recoverable originals.

Review was read-only against the saved receipts and arrays. No tests, producer, FV setup, derivatives, HVPs, or optimizer were rerun.
