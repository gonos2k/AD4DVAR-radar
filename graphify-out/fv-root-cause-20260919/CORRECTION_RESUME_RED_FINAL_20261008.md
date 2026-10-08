# RED final audit: PR263 correction resume

Scope: read-only audit of the completed correction-resume `step.json`, parent `step.run.json`, `step.resource.json`, fixed plan, and saved static analyzer result. No FV/HVP rerun was performed.

## Finding

**Receipts close and the accepted sequence is consistent.** The launch used plan SHA `9c29f4c0882272dccd68e2a35abf9a1fe9ff5b2fc5d5adc1502fe9a6ee8e9130` and records `resume_from` the PR263 plan and step. Step SHA `7bd91fa946b2c4f63380fb5df5dc2c47b4d2bfb2942ee5a5f579e703a419d83b` matches `step.run.json`; the parent resource object equals `step.resource.json`. Child and parent both report `execution_status=completed`, `numerical_status=max_iterations_completed`, exit code 0, and no child read, monitor, or resource termination error. Source, fixed-input, and runtime closure are true. The result analyzer records all nine closure/lineage checks true, including absence of the historical cycle gate.

The three committed states form a sequential chain from base `6c0fe485…` through `92352629…`, `cf44478f…`, and final `95a55578eea1d3b9600a210e7bdba37c21d7d7a1baf07244afc513a46df8c842`. `current_control` equals the third iteration’s accepted vector and hash; `optimizer_steps_applied=3`, three HVPs started and completed, PCG=0. No cycle reference or assessment is present in the resumed step. The plan and resource receipts preserve the single guarded launch limits: 240 s internal, 300 s external, and 1 GiB sampled RSS.

All three new points passed fresh direction, affine-control, J Armijo, Phi Armijo, strict-branch, and commit checks. Candidate counts were 2, 4, and 14. In the third correction, candidate 12 at alpha `1.6378415878e-7` passed strict branch and J Armijo but failed Phi Armijo (`Phi=0.02098658168`); the next half-sized candidate 13 at alpha `8.1892079391e-8` passed both Armijo tests and strict branch and was committed. Its J decrease from the third-correction base was only about `1.35e-9`; Phi decreased about `3.78e-7`. The run then reached its three-correction limit. This is ordinary cap completion after a valid small step, not a failed commit or old 2cd comparator stop.

Across the 17 rejected candidates, all passed the strict point check; 14 failed both Armijo tests and 3 passed J Armijo but failed Phi Armijo. The immediately larger trial at each stage was in the latter category. Saved candidate arithmetic also reconstructs each of those three larger trial controls from `base + alpha*d` with matching hashes. In the final correction the accepted alpha was `alpha_start/8192`; its displacement was `1.05161137e-8`. Static Qy43 geometry on each immediately larger trial was positive (`8.91e-6`, `6.11e-6`, `5.87e-9`), while the corresponding accepted endpoint values were negative (`−6.13e-6`, `−8.82e-9`, `−1.47e-9`). The endpoint remains near the Qy43 face; strict-branch checks passed, but this geometry is not stationarity evidence. The separate kernel parity record reports the guided numerical loop and math helpers identical to the archived PR263 producer.

From the resume base to final, J decreased from `0.0613117140406184` to `0.06130758760875752`, Phi from `0.010815707540872811` to `0.008244737124116911`, and `||g||inf` from `0.11651289508164255` to `0.10513700083904698`. The gradient remains far above `1e-10`; `full_root_claim=false` and `max_iterations_completed` are appropriate. The saved static analysis reports Qy43 geometry moving from `−2.1172343e−5` to `−1.4733279e−9`; this near-zero static flux value is not a root, adjoint, branch, or forecast certificate.

Resource use was 117.135 s and sampled peak RSS 377,700,352 bytes, with no guard termination. RSS is a periodic process sample rather than a hard OS allocation cap, as the receipt states.

## Limits

The numerical trend is favorable but the final correction was tiny and the stationarity gap remains substantial. Further convergence, eligible original 3-hour root, full curvature/original adjoint/reanalysis, and independent synthetic future verification remain open. The accepted strict-branch signature establishes endpoint branch eligibility under this kernel; it does not certify a smooth path or exact one-sided limit.
