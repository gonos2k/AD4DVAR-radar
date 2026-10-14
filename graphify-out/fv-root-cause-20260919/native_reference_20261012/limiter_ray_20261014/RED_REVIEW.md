# RED review: bounded limiter-ray diagnosis

The diagnostic and saved receipts support a narrow conclusion: the stored coupled-ray slope is consistent with the base selected-branch derivative, and the ray remains locally descending through the last sampled point without a minmod selector change. By α=2⁻¹⁰, exact selectors have changed and the selected AD derivative is positive; by α=2⁻⁷, native J is above baseline. This supports a nearby branch transition as part of why the extrapolated linear descent fails. It does not isolate minmod as the sole cause.

| α | ΔJ from base | secant ΔJ/α | AD branch gradient · d | minmod selector changes |
|---:|---:|---:|---:|---:|
| 0 | 0 | — | −6.52448×10⁻⁸ | 0 |
| 2⁻²⁰ | −5.4831×10⁻¹⁴ | −5.74946×10⁻⁸ | −6.52448×10⁻⁸ | 0 |
| 2⁻¹⁶ | −9.8992×10⁻¹³ | −6.48752×10⁻⁸ | −6.52438×10⁻⁸ | 0 |
| 2⁻¹² | −1.5923×10⁻¹¹ | −6.52221×10⁻⁸ | −6.52288×10⁻⁸ | 0 |
| 2⁻¹⁰ | −3.1633×10⁻¹¹ | −3.23917×10⁻⁸ | +6.74566×10⁻⁹ | 2 |
| 2⁻⁷ | +5.6898×10⁻¹¹ | +7.28300×10⁻⁹ | +1.57331×10⁻⁸ | 4 |

At α=2⁻¹⁰, x selectors at interior cell (1,1), stages 47–48, change left→right. At α=2⁻⁷ those remain changed and y selectors at cell (2,3), stages 123–124, change right→inactive. Flux-sign traces are separate: tiny α samples change only selected face y[2,1], and α=2⁻¹⁰ changes selected face y[4,3]. The derivative reversal at α=2⁻¹⁰ therefore coincides with both an x-limiter transition and a selected-face sign transition. The samples establish co-occurrence along this ray, not causality or a unique offending limiter. The interval between α=2⁻¹² and 2⁻¹⁰ is only bracketed; no crossing location is established.

The base AD contraction matches the stored slope, −6.5244839325259×10⁻⁸, to roundoff. At α=2⁻¹² the selector trace is unchanged and both the AD contraction and native-J secant remain close to that slope. This argues against a bad saved scalar contraction on the sampled branch. The α=0 AD value is explicitly a selected-branch derivative, not a certified right derivative at this multiple-face point. The small-step secants also approach native-J subtraction and forward-evaluation error; `scalar_roundoff_slope_scale` covers subtraction resolution only.

The code’s exact minmod labels match the strict sign and min selection in `transport._minmod`; exact selector traces do not alter the existing tolerance-based qualification audit. Unit tests cover positive/negative choices, active ties and inactive slopes, flux-versus-limiter separation, selector localization, and exclusion of synthetic module filenames. The preserved first source-inventory preflight failure started zero J/gradient calls and created no output directory.

The execution used 6 native-J calls and 6 fresh 24D chart-gradient calls, ending at 512 tracked direct-J calls and 134 tracked gradients. The prior chord plan had a cumulative gradient cap of 128; this diagnostic declared a separate six-call gradient allowance, so it must be reported as a cap transition and must not be described as compliant with the older cumulative cap. This is a ledger caveat, not a defect in the six-call diagnostic result. It completed in 13.786 s with peak RSS 336,527,360 bytes, no Hessian or score calls, and no accepted points. Plan and result hashes agree; all 132 source hashes and five input hashes match, the source archive contains exactly those pinned files with matching contents, and the result reports unchanged sources and inputs.

No minimum or full Clarke stationarity claim follows. The stale metric remains only a ray generator. The evidence diagnoses where its predicted decrease stops matching local sampled behavior; it neither determines how each switching surface contributes to ΔJ nor qualifies endpoint curvature or response.
