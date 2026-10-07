# Qy[3,2] fixed-η tangent exploration — RED design review

Date: 2026-10-07
Disposition: **One narrowly bounded exploratory step is mathematically coherent with the gates below.** It is a coordinate-restricted trial on the original full objective, not a permanent equality constraint, an active-face result, or a root search. No FV, HVP, PCG, optimizer, score, or code was run or changed for this review.

## Mathematical contract

Start from the provenance-validated e29 control (c_0\in\mathbb R^{26}), with its actual nonzero production face value

\[
\eta_0=Q_\star(c_0)=Q_y[3,2]\simeq-5.9106127381\times10^{-7}.
\]

Keep this signed value fixed for the entire trial. Let (t) be the 25 nonpivot raw control coordinates, and let (c=\Gamma(t,\eta_0)) restore pivot `c[24]` with the pinned `atanh` inverse. For the pinned face formula,

\[
Q_\star(c)=-.08\tanh(c_{21})-.21\tanh(c_{22})-.10\tanh(c_{23})-.45\tanh(c_{24}).
\]

Let (Z=D_t\Gamma(t_0,\eta_0)\in\mathbb R^{26\times25}). One explicit, reproducible ambient-Euclidean choice is

\[
g=\nabla_cJ(c_0),\qquad n=\nabla_cQ_\star(c_0),\qquad
 g_T=g-n\frac{n^Tg}{n^Tn},\qquad d=-g_T,
\]

then solve (Z\,\dot t=d) by the pinned stable QR least-squares path and use the curved candidate

\[
c(\alpha)=\Gamma(t_0+\alpha\dot t,\eta_0).
\]

This makes (c'(0)=d), (n^Td=0), and (g^Td=-\|g_T\|_2^2<0) when the projected gradient is resolved. If the implementation instead chooses chart-coordinate steepest descent (\dot t=-Z^Tg), it must predeclare that coordinate metric; it must not describe that direction as the ambient Euclidean projection. Do not mix the two direction definitions after observing results.

At this current smooth point, the exact first derivative of the raw merit

\[
\Phi(c)=\tfrac12\|\nabla_cJ(c)\|_2^2
\]

along the chart path is

\[
\Phi'(0)=g^THd,
\]

where (H=J_{cc}(c_0)). Compute one fresh current-point live (Hd), e.g. JVP of the full gradient in direction (d), on the original objective and locally fixed current branch. Hessian symmetry for this scalar smooth branch gives the displayed slope. Do not reuse an archived Hessian as the operator or call Φ Armijo without this actual current-point slope.

### Why both merit slopes are needed

A tangent descent direction for (J) does not imply descent for Φ, even for a smooth strongly convex quadratic and an exactly maintained face. Let

\[
Q(x,y)=y,\qquad
J(z)=\tfrac12 z^T\begin{bmatrix}2&1\\1&2\end{bmatrix}z,
\qquad z_0=(-5/3,7/3),\qquad d=(1,0).
\]

The chart holds (y=7/3\) fixed, (g(z_0)=(-1,3)), and (d) is exactly tangent. Yet

\[
g^Td=-1<0,\qquad g^THd=1>0.
\]

For sufficiently small positive α, (J) falls while Φ rises. Therefore the actual nonlinear candidate gradient and actual Φ must be evaluated and gated. For this policy, a candidate passes both actual Armijo tests:

\[
J(c_\alpha)\le J_0+10^{-4}\alpha\,g^Td,
\qquad
\Phi(c_\alpha)\le\Phi_0+10^{-4}\alpha\,g^THd.
\]

If either slope is nonfinite, unresolved under the declared FP64 dot-product rounding budget, or not negative, refuse the trial. Do not silently fall back to a J-only search or to a different Φ rule. A separately named exploratory rule requiring only actual Φ decrease would be a different predeclared policy, not dual Armijo.

## Candidate and radius gates

The straight line (c_0+\alpha d) generally leaves (Q_\star=\eta_0) at second order. Generate candidates only through (\Gamma(t_0+\alpha\dot t,\eta_0)); never overwrite a `Qy` array. At every candidate, before any FV trajectory/objective/gradient evaluation:

1. Require finite retained controls and an atanh argument strictly inside ((-1,1)), with a reported domain margin.
2. Recompute the static production face from bounded coefficients, combined streamfunction, and production face-flux routine. Require its signed value to agree with η₀ within a predeclared FP64 sum-of-terms allowance; otherwise refuse that candidate. Do not clamp the argument or force one stored face entry to η₀.
3. Compute the actual full-control displacement (\Delta c_\alpha=c_\alpha-c_0), require (\|\Delta c_\alpha\|_2\le0.05), and record the full, field, flow, growth, and pivot displacement norms. This must be checked before expensive FV evaluations.

The chart is curved, so α∥d∥ is only the first-order displacement and cannot certify the radius. Concrete local counterexample: on (Q(x,y)=y-x^2=0), (\Gamma(t,0)=(t,t^2)), (d=(1,0)), and α=0.05, the linear estimate is 0.05 but the actual endpoint displacement is \(\sqrt{0.05^2+0.05^4}=0.05006246>0.05\). A bisection used to find the first radius crossing must maintain a valid bracket and verify the final actual norm; because chart displacement need not be globally monotone in α, retain a static dyadic backoff that rechecks the actual norm. The entire static radius preparation is capped and invokes no FV.

Use a radius no larger than 0.05, at most 16 dyadic objective candidates, one current-point HVP, no PCG, one guarded launch, 240-second internal / 300-second external limits and sampled RSS at most 1 GiB. Record whether alpha-zero radius setup or any candidate was refused. A budget refusal leaves e29 as the only committed point.

## Fresh-base, branch, and commit gates

Before constructing the direction, freshly evaluate at e29 with the original full-control objective: (J_0), full (g), Φ₀, production (Q_\star), complete strict branch/margins, and the frozen input/runtime identity. Require them to close against the endpoint receipt within declared scalar/hash tolerances. A stale saved gradient can be compared as an audit but must not drive (d), (Hd), or either slope.

At each candidate, evaluate original (J(c_\alpha,p)), a fresh full 26-component gradient and Φ, the candidate's own full strict branch/margins, and exact-static-η checks. Retain both base and candidate signatures and report whether any non-target face or limiter signature changes. Endpoint strict eligibility does not certify an event-free path; if the signature changes, classify the accepted point as a finite exploratory transition and do not use same-branch smooth convergence language. Never promote a candidate to normal-root or response eligibility.

The objective must remain the original full (J), including every existing prior contribution on all 26 controls. The fixed-η chart is only a local diagnostic parameterization for this one trial. It must not delete `c[24]` from (J), permanently constrain future controls to (Q_\star=\eta_0), change the prior/observations, or turn the face into a physical wind or rainfall boundary. A successful result is reported as one accepted restricted-chart exploratory correction, with the original unconstrained problem and all convergence, curvature, forecast, adjoint and response claims still open.

## Source, cache and gzip provenance

Pin a fresh plan with:

- the PR #256/e29 producing plan, current objective/transport/FV/observation/prior/branch/HVP sources, tangent probe and focused tests;
- the prior Qy diagnostic `diagnostic.json.gz`, its `archive.json`, parent and resource receipts, plus the analysis/check summaries used as comparison evidence;
- exact SHA values for every source and archive, with source/input/runtime snapshots before and after.

For the previous diagnostic archive, verify gzip SHA and sidecar metadata, stream-decompress without overwriting the retained local raw file, and require the decompressed byte SHA to equal both the recorded raw SHA and parent `child_sha256` (`d05e5b8e…`). Require its plan SHA `bea4c361…`, accepted control e29 SHA, parameters/input hashes, source closure and normal guarded-run receipt to match. The prior compressed archive is provenance/evidence only: recompute the e29 base (J,g,Φ,Q_\star), branch and margins freshly before the new HVP. Write all new child/parent/resource/log outputs to a fresh attempt directory; never rewrite the prior raw, gzip, parent/resource receipts, or QY result summary.

New output must bind its plan SHA, e29 base hash, fixed η, chosen tangent-metric convention, direction/control hashes, (g^Td), fresh (Hd), (g^THd), thresholds, every actual candidate displacement, production η residual, candidate J/Φ/gradient/branch results, HVP count (=1 if the slope was evaluated), PCG count (=0), source/input/runtime closure and separate process/numerical statuses. A completed process with no accepted candidate is not a root failure; it is a bounded exploratory refusal.

## RED conclusion

A single chart-preserving tangent exploration can directly answer whether the chosen local tangent correction lowers both (J) and Φ near the e29 face value. It needs one fresh (Hd) to use the existing dual-Armijo meaning for Φ, and the actual curved-chart endpoint norm and production face value must be checked before any trajectory evaluation. Keep the step short, retain all 26 prior terms, and report it only as an exploratory update on the original objective.
