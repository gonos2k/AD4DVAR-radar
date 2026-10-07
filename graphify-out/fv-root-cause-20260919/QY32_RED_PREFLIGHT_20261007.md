# Qy[3,2] diagnostic — RED preflight review

Date: 2026-10-07  
Disposition: **GO for one planned guarded diagnostic. No prelaunch blocker found.**  
Review basis: read-only review of the pinned diagnostic plan, implementation, focused tests, PR #256 receipts, and arithmetic summary. I did not run tests, FV, gradients, HVP, PCG, forecast, adjoint, or reanalysis.

## Plan and provenance

The diagnostic plan SHA-256 is `bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f`. I independently hashed every declared file: all 107 source pins and 32 archive pins match. The plan binds the PR #256 plan, raw step, parent and resource receipts, and the diagnostic module/tests. The loader also delegates to the existing inexact-continuation loader and checks the expected accepted control `e29c348d51e7ec2de23f3f74d1522a34f58bf0bb72fd56e63d9b89cadea15e37` and fixed parameter/input identities (`fv_point_3h_qy32_diagnostic.py:69-159`).

The producer receipt distinguishes normal process/resource completion from its numerical `budget_refusal`; the diagnostic does not reuse its runtime/HVP budget. The new launch is fixed at one guard, 240 seconds internal, 300 seconds external, 1 GiB sampled RSS, zero optimizer steps/HVP/PCG (`:48-55, 492-545`). Child snapshots and rechecks source, input, and runtime. Atomic point-level writes preserve completed samples on an internal timeout; the parent receipt is written before child parsing and retains external resource-limited partial output. These states are separately recorded.

## Mathematical and implementation checks

The chart solves the pinned face equation by changing only `c[24]`, retaining the other 25 raw coordinates. The weights are derived from the loaded basis and coefficient limits and checked against the fixed profile. Production face flux is recomputed and compared to requested η, so a chart formula/reduction mismatch refuses the sample (`:162-217, 284-295`). The original problem objective receives the complete 26-vector; the pivot prior term is not removed. The focused toy regression explicitly exercises a full-vector objective and pivot contribution (`tests/test_fv_point_3h_qy32_diagnostic.py:97-133`).

At η=0, the child evaluates the primal objective with `with_gradient=False`; it does not ask AD for a branch-dependent derivative (`:427-430`). The four nonzero probes separately compute the full gradient and Φ. Chart derivatives are the correct chain-rule quantities: `j_t_fixed_eta = (D_t Γ)^T g` and `j_eta_fixed_t = (D_η Γ)^T g`. The Euclidean tangent projection uses (n=\nabla_c Q_\star), and the QR and direct projection norms are both retained. The raw chart covector norm is not presented as an invariant Euclidean tangent norm. Focused tests cover the chart Jacobian, projection identity and rescaling invariance, and distinguish pivot-coordinate (J_\eta) from intrinsic normal flux slope (`:220-266`; tests `:47-95`).

The full branch and limiter observations are retained at each sample, including the stage arrays; the offline analyzer computes same-side and cross-side indexed differences (`QY32_ANALYZE_20261007.py:11-26, 45-51`). This supports checking changes in the 360-stage analysis segment separately from the 3240-stage future segment. No endpoint branch status or finite sample comparison alone is treated as a root, minimum, or causal proof.

## Interpretation limit

The analyzer computes finite-interval (J) and Φ changes for available point pairs. When reporting them, interpret smooth-side gradient/secant evidence only if both points have `branch.status == "passed_strict_branch"` and complete valid margins. A branch-refused or margin-refused point still contains a framework gradient and Φ, but those values are not certified one-sided derivatives. The per-point status and margins remain in the raw receipt; this is a reporting guard for the eventual result, not a reason to block the bounded primal/gradient diagnostic.

The five levels are finite probes at a fixed tangent. They do not prove exact one-sided limits, Clarke stationarity, absence of nearby events, a constrained minimum, or that (Q_y[3,2]) alone caused the line-search shrinkage. A conclusion about a single-face merit conflict requires the pairwise branch/limiter differences and actual (J,Φ) results from this run. They do not establish physical wind/rain barriers or forecast skill.

## PR #256 review audit

The supplied PR #256 review’s reported arithmetic is consistent with the saved re-arithmetic record: both continuation steps have negative (g^Ts) and (g^THs), and the two nearest larger candidates pass their own strict endpoint checks and (J) Armijo but fail the Φ gate. The inexact forcing residuals meet the stated (10^{-3}) ceiling. Its local derivative bound for Φ is correct on a fixed smooth branch, and the report explicitly limits that argument at branch changes. The reported resource state also keeps process completion, numerical budget refusal, and sampled memory separate.

I found no material mathematical or arithmetic error in that review. Its main limit is already acknowledged: the local Φ descent bound does not guarantee globalization across a nonsmooth branch switch. The proposed Qy two-sided diagnostic addresses that remaining question without changing the original objective.

## Verification boundary

The parent reports **18 focused tests passed in 2.26 seconds, zero type errors**, and a successful isolated AST refresh; I did not independently rerun them. The current review validates the declared plan hashes and code contracts, not numerical behavior of the new FV evaluations. Proceed only with the single planned 300-second guarded attempt; do not interpret a resource-limited or partial receipt as a complete five-point comparison.
