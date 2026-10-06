# A35 dual-merit RED review — 2026-10-06

**Disposition: GO for one guarded, bounded dual-merit execution.** The earlier RED findings are resolved in the final source and frozen plan. This is a code/input review only; I did not execute the caller, tests, FV, or HVP work.

## Final code and input review

I independently recomputed the plan and source pins after the final branch-output fix. Plan SHA-256 is `62e291b8b683ef1480f60ae302e562c7313c2a545a3cb6517fd53f3b294220c0`; caller SHA-256 is `49b3ec88276c9176ff25f26f796a07999b6d3e55384b3ed7e0e084a6fb12a0b8`; focused test SHA-256 is `f4004f25d78f48f95f4c055039e1a091a6587e195d221eb0380aa88b5fb9b5e8`. All 94 source and 8 archive pins exist and match the current bytes. The plan fixes `cJ=cPhi=1e-4`, radius `0.05`, 16 dyadic candidates, zero new HVPs, one guarded launch, 240-second internal and 300-second outer limits, and a sampled 1-GiB RSS limit.

The source binds to the saved raw a35 base control (`a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c`) and parameter SHA `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`; the archived direction file is pinned by the plan. The separately preserved historical J-only accepted endpoint remains `38901e3a4c2ee97c8083b43e01e335f6328c3410982118b4ba891023468c471d`. The loader checks saved residual consistency and dual descent, producer/run/resource receipts, fixed-input identity, parameters, and source integrity. It uses the cached `s` and `Hs` and performs no HVP or PCG solve.

For `Phi=0.5||g||^2`, the policy uses `D J[s]=gᵀs` and `D Phi[s]=gᵀHs`. Candidate acceptance uses actual `J`, the full candidate gradient to form actual `Phi`, and the candidate’s own complete strict branch/margin check. Both Armijo inequalities are required. The saved a35 direction has `gᵀs=-0.0417182650` and `gᵀHs=-2.2259472341`, so both base slopes are negative. The previous 389 candidate decreased `J` but increased `Phi`; it remains a valid historical J-only result and may be rejected by this separate policy.

The final source defers the accepted-control update until diagnostics, the deadline, and source/input integrity closure pass. Failed closure or post-candidate budget refusal marks the trial as not committed. Candidate callback exceptions and callback contract errors persist the active trial row before propagating. Branch unpacking, strict-stage checks, and margin validation now run inside the branch evaluation wrapper; malformed branch results therefore persist the active candidate row as an evaluation error. Deadline checks cover objective, gradient, branch, and post-candidate physical diagnostic stages. Child output is required to be a JSON object, and the parent receipt records malformed or missing child output as failure.

The saved verification records report 24 focused tests passed in 1.56 seconds, with 18 existing TorchScript deprecation warnings, and a clean type check with zero errors, warnings, or notes. I reviewed these saved records and did not rerun them. They include mocked dual-gate, strict-branch, rollback, per-stage callback-error, and malformed-branch-result behavior. Source and test AST/Graphify evidence was refreshed for the final hashes.

## Supported scope

This GO covers the source, test evidence, and frozen inputs for one guarded same-direction a35 search. It is not evidence that any trial passed, that the direction yields a candidate under the dual policy, or that the run completes within its budget. The 16-point grid is not guaranteed to succeed. A resulting record would still establish no global SPD property, branch-path certificate, stationary point, root, response, score, or physical/weather validity. Sampled RSS remains a guard measurement rather than a hard OS allocation limit.
