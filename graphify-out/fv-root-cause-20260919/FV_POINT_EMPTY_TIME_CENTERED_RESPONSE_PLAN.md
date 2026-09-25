# R3-E one entirely empty point-observation time: stationary response plan

This is a **new constructed incomplete-observation profile**, distinct
from PR #201's one missing point and PR #202's one external-QC exclusion.
Keep the same 4×5 FV state and complete initial/boundary support, fixed
nonzero dynamics prior mean, point sampler, within-time correlation,
verification and 13-parameter layout. At the **first observation time**
set all four point statuses to 1 (genuinely missing), assign each its
canonical inactive dBZ fill, and declare `empty_observation_time=0`.
The middle and last times retain all four detected observations each.
This removes the first-time observation likelihood term, **not** the
first model state or either analysis time interval. The state-grid
background remains external `B_external+theta*P`; it does not derive
from the missing first point values.

At the constructed control, the eight active observations equal model
samples and the centered prior residual is zero. Require fresh
`||grad_c J||_inf<1e-10`, finite/symmetric/SPD exact 26-column
Hessian, and a 54-stage strict minmod branch with both scaled slope and
face margins `>1e-4`. Keep the same fixed theta, point-error standard
deviations, quality weights, and per-time symmetric correlation rule.
The first-time whitener is absent; middle and last-time whitening use
their original full 4×4 matrices.

Compute a full 13-component matrix-free adjoint/VJP and exact-HVP
tangent. First-time parameter indices 0–3 must have zero direct,
indirect and total gradient to absolute `1e-12`; changing each stored
inactive value alone by `+0.25` must leave objective, score and control
gradient unchanged. The theta/background dependency must remain
nonzero in at least one of direct, indirect or total theta components;
this checks the external background path was not dropped with the empty
first observation time. The signed direction adds +1 dBZ to all four
active **middle-time** points, indices 4–7, and zero elsewhere.

At `h=0.001,0.0005` dBZ per active middle-time point, reanalyze
`p±h*d` from `c0±h*c_dot` using unchanged Newton/PCG/Armijo. Require
all four fresh endpoint gradients `<1e-10`, the exact nominal full
signature and `>1e-4` slope/face margins. Actual score central
differences must match the adjoint response to relative `<1e-4` when
`|s|>=1e-8` or absolute `<1e-10` otherwise, with smaller absolute
error for the smaller h. This is a local active-observation value
perturbation, not the finite influence of deleting a whole time.

Use one guarded serial child with 600-second wall and sampled 1-GiB
child-RSS limit. Pin plan, source, input/status, prior mean, baseline
archive and code identities before/after. The parent independently
recomputes nominal Hessian/adjoint/VJP/tangent, all four inactive-slot
effects, theta dependence, endpoint objective/score/gradient/branch,
and central scores. Distinguish execution, numerical eligibility and
response validation. Unknown callback/invariant errors remain execution
failures. Original zero-centered-prior input, more than one empty time,
long-horizon point sensitivity and physical skill remain open.
