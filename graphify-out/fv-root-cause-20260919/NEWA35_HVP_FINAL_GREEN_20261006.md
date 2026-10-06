# A35 live-HVP Newton final GREEN review — 2026-10-06

**Disposition: GO for the scoped one-step result.** The guarded child completed and accepted its first actual-original-J Armijo candidate. I reviewed saved run/resource records, identities, numerical gates, and current plan pins. I did not rerun the FV path or compute another HVP.

## Identity and execution record

The run used plan SHA-256 `610e9f1d9bacb8f235159dccf1e7e4b56a55eef088214a1c9fcea77437166090`. Its 88 source and 7 archive pins all match the current files. The child result digest matches `step.run.json`; the parent status and resource receipt agree. The run finished with unchanged source map, fixed input identity, and runtime.

The input was the accepted a35 control `a35e2f6ef000cca4b593a3e5ed49f1c3e94b951450f2f235902a50e1dde6647c` and fixed parameters `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`. Fresh base objective, all 26 gradient components, `Phi`, and the strict branch matched the saved a35 endpoint evidence. The next accepted control is `38901e3a4c2ee97c8083b43e01e335f6328c3410982118b4ba891023468c471d`.

The guarded child exited 0 and completed in 219.132 seconds, within the 240-second internal and 300-second outer limits. Sampled peak RSS was 370,491,392 bytes, below the 1-GiB cap. The guard recorded 836 RSS samples, no monitor error, no SIGTERM, and no resource termination. This is sampled process RSS evidence, not an OS hard memory limit.

## Operator and solve

The source computes the live operator at a35 as a JVP of the current original-J gradient. The archived f82c Hessian is used to build and audit the block-inverse preconditioner only. That cached matrix has a positive minimum eigenvalue at f82c; the run does not transfer that claim to a35 or report a global a35 SPD result.

PCG converged in 24 iterations. The independently recomputed true relative residual was `8.2141051e-11`, below `1e-10`, and `gᵀs=-0.0417182650`, safely below its roundoff budget `8.4392e-15`. The counted live-HVP total was 26 started and 26 completed, below the hard cap of 106; this consists of the Krylov products and residual checks. No dense Hessian or spectrum was assembled at a35. The saved last `pᵀHp` value is a single recorded live Krylov/residual action, not a global curvature certificate.

## Accepted candidate and scientific scope

The first candidate used `alpha=0.17287764` and a whole-control displacement norm of `0.05`. Actual original `J` fell from `0.0716149827` to `0.0668157467` (reduction `0.0047992360`), passing the endpoint Armijo inequality. The candidate passed its own complete 3,600-stage strict branch/margin gate. Both the analysis partition and future-forecast partition changed from a35; the evidence is endpoint-only and gives no path certificate.

`Phi` rose from `1.11297362` to `1.85738409`, and gradient infinity norm rose from `0.79801515` to `0.98804097`. The configured acceptance rule uses actual original `J`, so this is an accepted J-decreasing step with worse normality diagnostics. The current-H directional Taylor model predicted a `J` reduction of `0.0065887450`; actual reduction was `0.0047992360` (ratio `0.7284`). Gradient linearization relative error was `1.0216`, which shows that the local gradient prediction was poor over this step, consistent with the branch changes. These diagnostics do not invalidate the actual endpoint Armijo result, and they do not support stationarity or root progress.

The record explicitly leaves accepted-point curvature uncomputed and sets the global-SPD, root, response, score, and physical-validation claims false. No adjoint, forecast-skill result, or independent weather validation was performed. The supported conclusion is one bounded original-J step accepted at a strict endpoint; it does not establish a stationary point, local minimum, global Hessian definiteness, response, or physical validity.

## Verification reviewed

The saved focused test log reports 44 passed in 1.53 seconds, and the saved type-check log reports zero errors, warnings, or notes. The run itself used one guarded launch with no retry or budget increase. No new test, FV, or HVP execution was performed as part of this review.
