# R2 one-start sector-adaptive point root feasibility plan

The previous core-strict fixed-signature root child report SHA256 is
`95e14c120d37c816aceed4f7ed0b98bb406549b5e14e1fd967aff133b8814a76`.
It made four same-signature corrections and then rejected all 16
iteration-5 candidates because their full 54-stage limiter/face-sign
signature changed. Those core-strict candidates were diagnostic only:
all 16 lowered actual objective and maximum gradient, while only one
lowered the Euclidean gradient norm. This is evidence to test a new
transition policy, not evidence of a root or a smooth path. Preserve the
old result and its final response-margin gate unchanged.

Use **one** saved attempt-2 terminal control, FP64-byte SHA256
`8f5110f2755217dece8ca9f6bbab23ceadb184b4f1a32b8e4654ddfde5f4b258`.
Keep the same correlated 4×5 point objective, 13 parameters, four fixed
interior observation positions, known state and boundaries, frozen score,
initial Hessian audit, exact gradient-JVP Newton operator, PCG true
residual `<=1e-10`, and 8 Newton /16 backtrack /104 PCG limits.

The only new numerical policy is an **optional trial-acceptance hook** in
`refine_stationary`. With no hook, its current behavior must remain
unchanged. For this run the branch callback admits any candidate passing
the existing roundoff-aware 54-stage minmod oracle, with finite positive
scaled slope/face margins; it records the complete observed signature and
actual objective/gradient diagnostics. A candidate on the current
signature uses the refiner's unchanged normalized-slope gradient-merit
Armijo criterion. A candidate with a changed full signature bypasses the
old-sector slope test and is accepted only when **all three actual endpoint
quantities** decrease:

\[
J(c,p),\qquad \Phi(c,p)=\tfrac12\|\nabla_cJ(c,p)\|_2^2,\qquad
\|\nabla_cJ(c,p)\|_\infty.
\]

For each scalar (m), require
`m_old-m_new > 128*eps64*max(|m_old|,|m_new|,tiny64)` and finite old/new
values. A failed metric gives an explicit refusal reason; no switch is
accepted solely on a predicted Hessian slope. After an accepted switch,
the next existing Newton outer iteration computes a fresh current
gradient/HVP and PCG direction on that new branch. It does not carry an
old-sector Hessian or L-BFGS memory. The current refiner's same-signature
Armijo remains a pointwise heuristic if a segment could leave and reenter
the same signature. Neither rule certifies an intervening smooth path or
global convergence.

Before the solve, recheck the pinned source/input/control/old-report hashes,
seed's exact 26-column Hessian finite/symmetric/SPD criterion, and strict
54-stage branch. For every trial record scale, candidate control hash and
full-signature digest, actual objective/gradient norm/maximum when evaluated,
and the existing Armijo ratio; link its core-trace margin record by the
candidate control hash. **For changed-signature trials**, additionally
record old/new signature digests, all three actual scalar changes/floors,
and the switch decision/reason. Same-signature trials delegate to legacy
Armijo and do not receive a switch-comparison record. Record PCG
HVP/iteration/residual evidence and flush candidate records atomically.
Unknown callback/invariant failures propagate as execution
errors, never a successful root.

Only a **fresh final** `||grad_cJ||_inf<1e-10`, a core-strict 54-stage
final trace and finite/symmetric/SPD exact Hessian can be classified as
a branch-local root. If either final scaled margin is `<=1e-4`, label it
`sector_root_low_margin`, not response-eligible. If both exceed `1e-4`,
label it `sector_root_margin_qualified`; this still does **not** validate
an adjoint or signed reanalysis. In every status,
`response_validation=not_performed` and no finite-path or physical-skill
claim is made. A switch may reach a different local minimum.

One serial child has a 600-second wall and sampled 1-GiB child RSS cap.
The parent checks elapsed time even if polling misses a just-over-limit
exit, child PID, sampled RSS count/peak, exit/termination/monitor state,
status-specific final evidence, and current source/input/plan/preflight/
attempt-2/prior-root hashes against before/after child records. RSS is
sampled every 250 ms, not a hard allocation cap, and does not include
parent or descendant memory. A refusal remains a refusal; no same-run
threshold, seed or objective change is permitted.
