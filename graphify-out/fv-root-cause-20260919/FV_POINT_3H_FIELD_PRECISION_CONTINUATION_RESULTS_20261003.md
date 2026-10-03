# S4 precision continuation from the latest accepted field iterate

One guarded field-precision continuation ran from the last accepted full control in `point_3h_field_continuation_attempt1/field_continuation.json` (SHA-256 `a5759c3eca7ca537e0038b3dbe95f3ff67abdabcc43f5f5a2b3aa3cc5982fb1b`; warm control SHA-256 `ecfbbbd4919b8b783c93dce4a8dbb27598eac8eb9ca698bcf7ecd510f737b38b`). It kept the same original 26-control objective and priors, 13 parameters, and six dynamics controls.

At the warm point, fresh (H_{ff}) qualification passed using 20 exact HVP columns and an independent 21st minimum-eigenpair HVP: eigenvalues `[0.8733561, 1656.4803]`, positive floor `1.65648e-5`, condition `1896.68`, relative symmetry error `1.0322e-15`, and eigenpair relative residual `6.9496e-17`. The selected point had objective `0.1337023635` and field-gradient maximum `0.3543717660` on strict branch `82dcbb7b…a3588`.

Five Newton updates were accepted with step sizes `0.0625, 0.125, 0.25, 0.0625, 0.125`. The objective decreased from `0.1337023635` to `0.1309895304`. The field-gradient maxima at accepted endpoints were `0.3422807540`, `0.3414442648`, `0.3654611642`, `0.3196750901`, and `0.3194945916`. Iteration 3’s maximum rose while the unchanged gradient-merit Armijo rule accepted its step; the acceptance rule tests the gradient merit, not monotonicity of the infinity norm. Each accepted point passed its strict pointwise branch check, with branch signatures recorded in the raw report.

The sixth PCG solve hit the declared 1,680-second internal budget after 10 operator calls and before returning a solution, so no sixth field step was accepted. Its terminal status is `ContinuationRefusal: 1680-second continuation budget exhausted`; field issuance is false. The first five saved PCG solves passed independent true-residual audits (`2.17484e-11`, `6.30305e-11`, `1.51720e-11`, `2.81590e-11`, `1.84882e-12`). The sixth solve is recorded as a solver refusal without a fabricated residual.

A fresh full 26-gradient audit completed on the last accepted point (control SHA-256 `a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d`). Objective is `0.1309895304`; field-gradient maximum is `0.3194945916`; flow-block maximum is `1.884308916`; growth/dynamics maximum is `3.779711223`; full-gradient maximum is `3.779711223`; full gradient merit is `9.654944139`. The fresh field-gradient gate remains false. Parameters, all six dynamics controls, the input/data/time/prior identity, runtime, and all 52 captured source paths match before and after.

The guarded child completed in `1725.364` seconds with exit code `0`, no resource termination or monitor error, and peak sampled RSS `377,978,880` bytes within the `1800` second / `1 GiB` outer limits. The parent run record reports execution completed separately from the numerical refusal. Full-root, full-minimum, response, score, and physical-validation claims remain false. This endpoint branch check is pointwise and provides no segment certificate.

Authoring verification was six focused pure tests, offline basedpyright with zero errors, and a clean diff check. No FV run occurred during authoring. The original a575 refusal artifact remains byte-for-byte unchanged.

## Scope and next step

This is a terminal field-only precision-stage refusal; five local corrections improved the objective, but the field stationarity gate failed and the full gradient remains dominated by the fixed dynamics block. No further field-only continuation is part of this stage. Any next diagnostic would need a separate math review at this terminal point; this result does not establish a full-control root, response, or physical forecast validity.
