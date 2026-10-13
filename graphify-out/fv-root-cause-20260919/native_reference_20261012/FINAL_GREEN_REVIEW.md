# Final GREEN review

The saved evidence supports a focused native-reference and validation report. It does **not** support an original-problem minimum, an exact response, or a general forecast-skill claim. I reviewed the current report, accounting, checklist, one-chord result, growth/coupled ray receipts, observer metrics, and the generated HTML without model callbacks.

## Current candidate and scope

- The current candidate is `6ca56498…`, with native J `0.0601191808341518`. Its finite sampled full/tangent hull proxy has infinity norm about `0.00114502`, versus the `1e-10` stationarity threshold. The two-face branch and strict normal signs pass, but first-order stationarity fails. The terminal Hessian/P2 was not computed, so earlier Hessian results at `8eff6b1f…` or any other point do not qualify this endpoint.
- The one-chord extension is correctly reported as `one_chord_nonstationary`; it does not claim a minimum. The original two-face continuation also ended without stationarity. B’s exact response and perturbation reoptimization remain gated and unexecuted.
- The coupled ray shows all nonselected flux signs unchanged at all seven samples, while the limiter digest changes at every nonzero alpha. At alpha `1/128`, original native J rises by `5.6898e-11`; at alpha `1`, it rises by `2.3800e-7`. This uses a stale positive-definite metric with the current gradient, not an exact current Hessian. It supports rejecting that ray and continuing to investigate limiter transitions; it does not establish one limiter as the unique cause.
- The separate growth-only ray has five points with the same limiter signature and negative growth partials. Its smooth cost decrease is evidence about that coordinate ray only; it is not a full stationarity certificate and does not conflict with the coupled-ray branch changes.

## Forecast and HTML

The 6ca observer’s 18 lead values match the plot payload. Mean MAE is `1.163358` dBZ versus `1.144573` for analysis persistence; forecast MAE is lower through +100 minutes and higher from +110 through +180. The raw observer field called `first_persistence_mae_crossover_minutes=10` is mislabeled: +10 is the first better lead, while +110 is the first worse lead. The HTML states the latter. The historical `8eff6b1f…` curve remains separate from the current `6ca56498…` observer, and neither forecast was used to select a control.

The newer `HTML_UPDATE_VERIFICATION.json` reports unchanged replay DATA, byte-identical second build, unchanged signed-span SVG, 34 matching evidence-source hashes, no superseded/in-progress sources, 1,031 valid relative links, and a visually inspected PNG. Browser rendering remains unverified because local file rendering was blocked by browser security policy. The stored `HTML_VERIFICATION.json` hash map is stale: its digests for `fv_response.html`, `fv_response_template.html`, `update_fv_case_review.py`, and `CASE_REVIEW.json` differ from current bytes. Refresh or supersede that hash map after the final whitespace-only SVG edit before describing the HTML artifact pins as current.

## Accounting and readiness

The `2,506` direct-J figure is the audited 2,000-call reference search plus 506 later tracked calls. It is not a census of all earlier aborted qualification/debug calls, and direct-J counters do not count AD/Hessian internal forwards. Earlier partial HVP totals and one full-H runtime are unknown; the prior plan carried a conservative 600-second charge for the unknown full-H runtime. The final accounting is an estimate, not measured total session wall time and not a strict 3,600-second compliance certificate.

The evidence is ready for a focused native-reference and validation PR once the HTML verification hash map is refreshed or explicitly superseded. Keep the coupled/growth diagnostics in that evidence bundle rather than presenting them as separate optimizer results. State plainly that minimum qualification, P2, exact response, perturbation reoptimization, and held-out weather validation remain incomplete.
