# GREEN final audit: saved coupled-GN comparison — 2026-10-10

## Disposition

GREEN clear for the bounded, single-base comparison recorded in the saved receipts. No actionable mismatch was found in the archive, saved-array reconstruction, selected-control closure, or scope statements. This supports the reported local comparison only; it does not establish general GN superiority, stationarity, exact constrained curvature, convergence, or forecast skill.

## Independent receipt and arithmetic audit

The raw child SHA-256 is `362e8e0aa7cf12202618c11873633201d23ddc2af835ac67a5d499612edb4599`; gzip is `bdf479a14b0ba0e3342a8dc630e9bce2c89ac83fd4fea4625d17b2a01e507b84`. Decompressing gzip and comparing it byte-for-byte with raw succeeds. Run and resource receipt hashes match the archive manifest: `627505212d26ca787e4979bcc25706c9ac68ff8fd4ed3fceb262a10611a2b653` and `7585c4dbc8a2fb9bfce1af96d691fb00f92ccefb17de8f4b412e35b8a4027f4a`.

I replayed `GNCOUPLED_ANALYZE_20261010.py` on saved arrays with `.venv/bin/python`; the reconstructed JSON is byte-identical to `GNCOUPLED_RESULT_20261010.json`. All 15 receipt checks pass, including pinned-source receipts, 24 complete rows, four correctly tagged HVPs, resource closure, winner selection, and selected-only final closure. The recomputed selected endpoint is control `4035904e46c11c67a6f937039fe3c88b0516330b05f3ba9d57c4cfa619423ab7`, `J=0.061202333947534084`, `F²=0.004490021941881948`, and `theta*=0.4829044955080727`; both side gradients match the selected proposal exactly in the saved comparison, and every final closure flag is true.

The saved-array formulas are internally consistent. For the robust-GN arm, `D=(2/hypot(2,r))³`, `B=√D A P`, `S=I+BBᵀ`, and `d=−t+BᵀS⁻¹Bt` match the reconstructed arrays to floating-point precision; the saved `S` has minimum eigenvalue `1.00162839`. The chart-repaired direction is tangent to the face normal to roundoff. Recomputing

`theta' = −[jump·h_mix + g_star·(h_plus−h_minus)]/(jump·jump)`

matches each stored `theta'`; reconstructed `DF` matches the saved direction derivative exactly at reported precision, and `F·DF` agrees with the envelope slope. This checks the recorded local model and direction bookkeeping, not production derivatives beyond the saved receipts.

## Documentation and limits

The findings and RED final review match the result: baseline first-passing `F²=0.004586433310117439`; coupled-GN first-passing and selected `F²=0.004490021941881948`. The comparison uses each arm’s first passing candidate, then selects by actual `F²` with the stated tie rule; it does not claim a sweep-wide optimum. The six larger GN candidates fail only actual minimum-`F²` Armijo according to the saved trials. The final face value is zero, and the docs preserve the branch-signature change and model-vs-actual limitations.

Resource receipts record one guarded execution, 92.535 seconds, 24/24 row products, 4/4 HVPs, one 12×12 solve, eight candidates, one committed optimizer step, and sampled peak RSS of 984,743,936 bytes under the sampled 1-GiB threshold. The receipt explicitly identifies RSS as sampled rather than an OS hard cap. Claim flags for full smooth root, response, and forecast score are false. The comparison remains one local step using a robust-GN search metric that omits residual-Hessian terms.
