# Qy[3,2] fixed-η tangent step — RED preflight

Date: 2026-10-07
Disposition: **GO for the single bounded attempt. No remaining must-fix issue found in the frozen plan/code/tests.** This is a preflight disposition only; no FV, HVP, PCG, line search, or test was run by this review.

## Frozen-plan and provenance check

The plan SHA-256 is `70f9560c0355fa6b65927ac70fb4de2c055f338d5dd276c8fa453fc30379ecf9`. I independently hashed its declared **109 source files and 39 archive files**; every hash matches. The pinned policy is one guarded launch, 240 s internal / 300 s external, sampled RSS at most 1 GiB, one live HVP, zero PCG solves, at most 16 line-search candidates, and a 0.05 full-control radius.

The new loader requires the e29-producing QY plan and the compressed prior diagnostic plus its archive metadata, parent and resource receipts. It decompresses in memory and matches the raw bytes to SHA `d05e5b8e…` and the prior parent's child SHA. It reconstructs the fixed input and fresh e29 state before forming the new direction/HVP. The prior raw diagnostic and receipts are pinned as evidence and remain outside the new attempt's output paths.

## Mathematical and numerical contract

The candidate path is `Gamma(t0 + alpha*dt, eta0)` with η₀ taken from e29's nonzero production face value. The pivot is restored from the analytic face equation for every candidate, then the production coefficient/streamfunction/face-flux path checks the realized Q before evaluating the original full-control objective. All 26 controls and prior contributions continue through the existing objective.

At the fresh base, the code computes one current-point JVP-of-full-gradient in the chart tangent direction. It uses the valid path derivatives (J'(0)=g^Td) and Φ′(0)=(g^THd) for the two Armijo limits. The candidate gradient model is explicitly labeled first order along the nonlinear chart path; no (H\delta\) or quadratic-cost prediction substitutes for the actual curved displacement. The actual 26-control norm is checked before candidate FV work. Chart-domain failures now shrink/refuse statically before FV, and resolved-slope failures are recorded as completed numerical `direction_refusal`, with closure and no candidate commit.

The final endpoint receives an independent J/Φ/gradient/branch check and source/input/runtime closure. Deadline checks now follow endpoint and final receipt closure, and timeout handling clears any tentative acceptance. A changed candidate branch remains a finite exploratory result; it does not certify a branch-fixed path, root, active-face minimum, response, or physical constraint.

## Verification boundary

The parent reports 27 focused tests passing in 1.55 s, 18 known warnings, zero type errors, and isolated AST refresh. This review did not rerun these checks. The only supported conclusion before launch is that the code/policy/source contract is internally coherent and resource bounded. The actual trial's J, Φ, HVP slope, branch outcome and numerical status remain unknown.
