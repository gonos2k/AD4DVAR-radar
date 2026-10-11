# GREEN saved-evidence review: PR287 local extension

Reviewed the frozen extension source and closed saved run. No actionable source or evidence-integrity issue found.

- Source plan SHA-256 `01704c452ad8f4c831296d7e94417d253250cba457b7eb2336d90b5996b66166`; archive records lossless raw/gzip round-trip. Raw step SHA-256 `098e688bfaa405d9c88f80da2ff09295993695b78215713608509ce1602448fe`.
- Run completed in 61.435 s, peak sampled RSS 752,648,192 bytes; child exited 0. Base control stayed at SHA `454396e77e2f10492e80e0fd625b55e1a08c2be53fe9721b59a4a27d60be3b7a`.
- Reuse/fresh-work accounting matches scope: 0 row VJPs, 0 dense solves, 2/2 fresh same-point HVPs; fresh residual-direction comparison passed with zero recorded difference. Eight of eight planned samples started and completed.
- All eight samples retained the base selector signatures; J and residual-squared Armijo checks passed. Every sample is explicitly `accepted: false`; no optimizer step or commit occurred, and readiness/root/minimum/response/score claims are false.
- With base (R=0.003127039112866745), sample (R) descends to `0.0031270384466969706` at the smallest (h=9.391166446070961e-8). The reported model-error norm falls from `3.52e-11` to the few-e-12 range, then fluctuates at that floor; (E/h) does not tend to zero over the final samples and (E/h^2) grows. This supports a tiny local descent under the unchanged selector, but not a uniform second-order error claim or a new optimizer step.
- Verification receipt records 6 focused tests passed, 0 type errors/warnings. This is a saved-evidence review; it makes no minimum, convergence, or forecast-validity claim.
