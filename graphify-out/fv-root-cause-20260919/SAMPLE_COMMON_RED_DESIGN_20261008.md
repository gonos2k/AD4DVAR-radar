# Sample-common cost-step RED design review

Date: 2026-10-08
Scope: theory, execution contract, provenance and interpretation for one bounded original-cost exploration from the accepted `2cdccade…` endpoint. This is a design review only; no FV, HVP, PCG, optimizer, score, adjoint or reanalysis was run.

## Direction and descent gate

Use only the archived inner pair of full gradients at η=−1e−6 and +1e−6 from the face-transition diagnostic. Recompute

\[
\Delta g=g_+-g_-,\qquad
\theta=\operatorname{clip}\!\left(-\frac{g_-^T\Delta g}{\|\Delta g\|^2},0,1\right),\qquad
d=-\left(g_-+\theta\Delta g\right)
\]

from the pinned stored arrays, with a scale-aware degenerate-segment check. This is the minimum-norm point on that **sampled two-gradient segment**, not a Clarke selection or a limiting-gradient oracle. It is a full 26-control vector: the proposed path is \(c(\alpha)=c_0+\alpha d\), with all original prior terms retained. Do not restore a pivot, hold the face flux fixed, or label this a tangent/face-constrained step.

At the exact current control, reconstruct fresh original \(J_0\), \(g_0\), and its pointwise strict branch/margins; compare their control, parameters, sources and branch identity with the pinned accepted `2cdccade…` receipt. Before spending the one HVP, require a numerically resolved descent slope

\[
g_0^Td < -\tau_J,
\]

where \(\tau_J\) is an FP64 dot-product roundoff bound scaled by \(\sum_i|g_{0,i}d_i|\). If this fails, record direction refusal and stop without trying to turn the sampled direction into a search direction.

Compute exactly one fresh base-point product \(H_0d\) using JVP-of-gradient on the current smooth strict branch. Check its shape/finiteness and count start/completion. From this vector, keep the two distinct scalars separate:

- \(d^TH_0d\) is the local quadratic curvature for the original \(J\) along the straight line.
- \(g_0^TH_0d\) is the first derivative of \(\Phi=\|g\|^2/2\) on that base branch and is a diagnostic only.

Do not gate on the sign of \(g_0^TH_0d\), and do not reuse it as the J curvature. If \(d^TH_0d\) is finite and resolved positive, a nominal J-quadratic scale can be \(-g_0^Td/(d^TH_0d)\). Otherwise use the predeclared radius scale. In either case cap the initial scale by 1 and by \(0.05/\|d\|\). This HVP is a local model for selecting an initial trial only; it is not an acceptance test and it need not remain accurate after a limiter or face branch changes.

## Candidate and commit contract

Search at most 16 dyadic scales \(\alpha_i=\alpha_0 2^{-i}\), with one maximum accepted step. Check the actual full-space displacement \(\|\alpha_i d\|_2\le0.05\) before evaluating FV. For candidates within the radius, evaluate the unchanged original full-control objective and use only the predeclared J Armijo condition

\[
J(c_0+\alpha_i d)\le J_0+c_1\alpha_i g_0^Td,
\qquad c_1=10^{-4}.
\]

Require the candidate's own strict pointwise branch and margin pass. Record the full branch signature and every changed face/limiter category, but do not require equality to the base signature if the declared exploratory policy permits a branch crossing. Conversely, do not force a crossing, relax a strict endpoint check, smooth the upwind switch, change the prior, or change final tolerances.

The actual \(\Phi\), its change, gradient norms and component changes are outputs, not gates. A candidate that passes J Armijo may increase \(\Phi\), the max-gradient component, or another block residual. It is a **cost-descent exploratory step**, not a stationarity success, normality certificate, root, or increase in the completed research-milestone percentage.

Before committing one candidate, freshly recompute original \(J,g,\Phi\), strict branch/margins and the final input/source/runtime/deadline closure. Require the repeated values and branch signature to match the tentative candidate within pinned numerical checks. A timeout, stale receipt, closure mismatch or final branch refusal must leave the base as the committed point and the candidate explicitly uncommitted. Preserve prior records and expose execution completion, candidate acceptance, branch eligibility, cost decrease, \(\Phi\) change and closure as separate fields.

## Face crossing is an observed outcome, not a premise

The previous stored inner-pair direction has \(\|d\|_2\approx0.136798\). Recomputing its pairing against the PR #260 stored base gradient gives \(g_0^Td\approx-0.01869077\), a plausible starting descent direction to be rechecked from a fresh baseline. Its smooth target-face linear flux derivative is about +1.03228e−4 per unit \(\alpha\), so the direction points toward positive `Qy[4,3]`. This does **not** establish that the accepted candidate will cross: the J-only search may accept at any dyadic scale, and strict branch and cost checks determine the result. Store actual candidate face flux and sign for every trial. A direction selected from nearby samples is not evidence that a particular nonlinear trial crossed the face.

Even if an accepted trial crosses the target face, its cost change cannot be attributed solely to that face. The archived same-side samples already have future limiter-choice differences. Preserve complete signatures and describe the step as the combined effect of a full-space direction plus its actual finite displacement.

## Provenance and resource bounds

Pin the model-guided current accepted endpoint and receipts, face-diagnostic plan/child/run/resource, the exact archived gradients used to form \(d\), their source manifest, and every unchanged numerical-operator source. Verify hashes before and after execution. Write only to a fresh attempt directory. Preserve the earlier raw QY32 diagnostic, its gzip archive and archive metadata byte-for-byte; do not rewrite or re-compress them. Keep the old accepted point and the exploratory result as separate records.

Use one serial guarded launch, one fresh HVP maximum, no PCG, 16 candidate evaluations maximum, and at most one commit. Enforce the 240 s internal / 300 s outer / sampled 1 GiB RSS contract. Persist the attempt before numerical work and write each completed trial durably. Track HVP started and completed counts separately. A budget or provenance refusal must preserve all prior records and never imply that a candidate was committed.

## RED interpretation boundary

This policy is coherent as a narrow cost-only experiment because it admits only a freshly verified negative \(g_0^Td\), bounds the actual control displacement, tests the original J on each trial, and revalidates the tentative endpoint before commit. Its non-smooth risk is precisely why the HVP curvature is only a base-branch scale and why actual J plus endpoint branch checks govern acceptance. A passed step would establish one accepted finite cost decrease in this run. It would not establish monotone \(\Phi\), convergence, smooth crossing, an active-face solution, forecast improvement, or a meteorological mechanism.
