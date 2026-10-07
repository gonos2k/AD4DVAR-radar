# PR #258 follow-up — RED design for one full-space step

Date: 2026-10-07
Disposition: **One fresh full-space steepest-gradient trial is a coherent next experiment under the guards below.** It returns to the original 26-control problem at accepted point `05889584…`, so the temporary fixed-η chart is released. This is one bounded exploratory step, not a root search. No FV, HVP, PCG, old-candidate replay, or code run was performed for this review.

## Base and direction

Use the accepted control from the PR #258 tangent-step receipt:

- control SHA: `058895848fdeb348e8d6bb1f22ac13e3f2dadfa48bff46522ffbf94fedf72ffa`;
- fresh full-gradient norm: (\|g\|_2=0.6686406395), (\|g\|_\infty=0.5101284967);
- (J=0.0615557130112), Φ (=0.223540152425);
- (Q_y[3,2]=-5.9106127381\times10^{-7}), and Euclidean unit-normal gradient about (-0.15135847).

Reconstruct and freshly verify this base using the original full 26-control objective, prior, observations, parameters, boundaries and schedule. Require its J, full gradient, Φ, production face flux, runtime/input identities and own complete strict branch to match the accepted receipt. Use

\[
d=-g,
\]

in the original full control coordinates, with no pivot restoration. Then (g^Td=-\|g\|_2^2=-2\Phi\simeq-0.44708030485), resolved negative at this recorded point. The full prior continues to include all 26 controls. The normal coordinate is deliberately released; this is no permanent constraint and does not change the model.

The previous endpoint's face gradient gives the saved local flux slope (\nabla Q_\star^Td\simeq+0.07759839). Linearized from the current negative η, the face reaches zero at roughly

\[
\alpha_{Q=0}\simeq7.62\times10^{-6}.
\]

The full-space radius cap (0.05/\|d\|_2\simeq0.07478) is much larger. The fresh Φ-curvature cap may be smaller, but a trial can still cross the face. Record each candidate's production Q sign/value, full strict branch/margins and signature relation to the base. Own-endpoint strict checks allow testing a crossing; they do not certify an event-free path.

## Fresh HVP and initial Φ bound

At the fresh current point, compute exactly one live HVP

\[
Hd=J_{cc}(c)d
\]

by JVP of the full gradient on the current strict branch. Do not reuse PR #258's HVP, the e29 HVP, or any cached curvature. Define

\[
m=g^THd,\qquad q=\|Hd\|_2^2.
\]

Require finite HVP, (q>0), and (m<0) beyond a declared FP64 dot-product rounding budget. If this gate fails, close the run as a numerical direction refusal; do not fall back to J-only acceptance.

The reason for the proposed initial bound is the linearized-gradient merit model:

\[
\Phi_{lin}(\alpha)=\tfrac12\|g+\alpha Hd\|^2
=\Phi_0+\alpha m+\tfrac12\alpha^2q.
\]

For (m<0), setting (\alpha\le-m/q) gives

\[
\Phi_{lin}(\alpha)-\Phi_0\le-\tfrac12\alpha|m|<0.
\]

Thus a conservative proposed start is

\[
\alpha_0=\min\left(1,\frac{0.05}{\|d\|_2},\frac{-m}{\|Hd\|_2^2}\right),
\]

with finite/positive/representable denominator checks. This bound makes only the *linearized-gradient* Φ decrease; actual Φ still requires a fresh candidate gradient and the actual dual-Armijo gate. A smooth counterexample shows why: let (J'(x)=1+x+100x^2) at (x=0), so (g=1,H=1,d=-1,m=-1,q=1), and α₀ is radius-limited to 0.05. The linearized gradient is 0.95, but the actual candidate gradient is 1.2, so Φ rises from 0.5 to 0.72 even while J decreases. Actual nonlinear merit checks cannot be replaced by this bound.

Also, (m<0) is only a directional curvature result, not an SPD certificate. For example (H=\operatorname{diag}(1,-1)), (g=(1,0)), (d=-g) gives (g^Td=-1) and (g^THd=-1) even though H is indefinite. Report one HVP/Rayleigh observation only; make no global positive-definiteness claim.

## Candidate and acceptance contract

Use the exact straight full-control path

\[
c_\alpha=c+\alpha d.
\]

Here the candidate displacement is exactly α∥d∥; there is no curved-chart correction. Backtrack from α₀ on the predeclared grid, with at most 16 candidates. For each candidate, evaluate the original full J, fresh full gradient and actual Φ, the actual production face flux, and its own complete strict branch/margins. Accept only if both measured inequalities hold:

\[
J(c_\alpha)\le J_0+10^{-4}\alpha(g^Td),
\qquad
\Phi(c_\alpha)\le\Phi_0+10^{-4}\alpha(g^THd),
\]

and the candidate's own strict branch/margins pass. The Φ Armijo slope is from the fresh current-point HVP. It does not claim the same branch persists along the segment; if the endpoint signature changes, retain that fact and describe the result as a finite branch transition. Candidate branch crossings are permitted by this exploratory policy, but actual J/Φ and endpoint strict checks remain mandatory.

If the candidate is rejected or no alpha passes, report a completed numerical refusal with the last candidate and reasons; keep `05889584…` as the only accepted point. On success, independently re-evaluate final J/g/Φ/branch and close the full input, source, runtime and deadline receipts before marking the candidate committed. Keep one guarded launch at 240 s internal / 300 s external / sampled 1 GiB, exactly one current HVP maximum, and zero PCG solves.

## Provenance and scope

Create a new plan/attempt bound to the PR #258 accepted step and its parent/resource receipts, plan, sources, fixed input, and the QY32 compressed source archive. Verify the gzip bytes against `archive.json` and the prior parent child SHA; preserve old raw/gzip/summary files. Recompute the base gradient and fresh HVP at `05889584…`; no old HVP or old candidate alpha/2 is reusable at this new point. Snapshot all objective/transport/observation/prior/branch sources and runtime/input identities before and after.

The strongest supported outcome is one actual full-space candidate that passes both merit tests at the new starting point. A lower J/Φ, a changed face sign, or (m<0) does not establish a root, local minimum, full Hessian SPD, forecast skill, or adjoint/reanalysis eligibility. The preceding tangent step improved J and Φ but increased (\|g\|_\infty); the full-space step must stand on its own actual measurements.
