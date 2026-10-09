# Candidate requalification: RED final audit

Date: 2026-10-09. Scope: read-only audit of the saved single guarded run for
plan SHA `5b7ebac50ebc38c0eaf34539c5bdf89e00c4d342ef96f2a96bc12a93fa30bfb4`.
No FV calculation was rerun for this audit.

## Result

The run completed in 50.699 s with exit code 0, no monitor error or resource
termination, and sampled peak RSS `656,474,112` bytes under the 300 s / 1 GiB
guard. It completed exactly two current-direction HVPs and zero dense solves.
The child receipt SHA is
`14a4b1c05363e316fddd3291946852e3334232388f975f1316030e85a4a0abfc`, matching
the parent run receipt. The plan keeps the original accepted control hash
`95a55578…6df8c842`; the accepted research candidate has control hash
`6b29dacd…b580a743`, is typed `one_nonsmooth_requalified_step`, and is not
activated (`active_candidate_committed=false`).

## Reuse and fresh operator evidence

The loader closed the archived PR265 refusal and its gzip/raw/run/resource
hashes. The fresh face-chart control hash exactly matches the archived same
point. Fixed inputs, runtime, and the current actual-J/H source map match the
archived record. Fresh one-sided gradients differ from their individually
reconstructed archived values by at most `1.39e-17` and `2.78e-17` (tolerance
`2.84e-14`). The pre-run selected-face producer snapshot remains the archived
d05 source; the live requalification module is separately pinned by the new
plan. The unchanged core transport source is pinned at
`93bacc2e…8e701738c`.

The archived 27-vector is reused only as a direction. Two fresh HVPs apply the
current selected-face operators along the chart tangent. With fresh side
gradients and current `theta`, the audit forms all 27 components of
`D F · delta + F`, including the face-constraint row. Its residual norm is
`4.3354e-14`, relative to the true RHS norm `0.0809658` by
`5.3546e-13`, below the declared `1e-10` limit. The separately reported
`3.1469e-18` is the normwise backward error for the **archived** matrix solve;
it is not the current-operator result. No current full Hessian was computed.
The serialized Hessian scope and curvature diagnostic both state that the
archived matrices are same-point references, not candidate curvature.

## Candidate gates and closure

The accepted slot was `alpha=0.01314167293`, with actual control displacement
`0.0007812502`. Native objective `0.06126370581` passed the recorded J Armijo
bound `0.06130758287`. The fresh-product scaled `F^2` slope was
`-0.00655546096`; the candidate value `0.00631438448` passed its bound
`0.00655544373`. The analytic and production face values were
`1.73e-18` and `-0.0`, within `5.20e-16`. Both endpoint side objectives match
the native objective, and both side gradients are finite.

Both endpoint traces contain 360/360 stages, report no ties/nonfinite values,
and share the same current non-target limiter and face-sign signature. Their
minimum normalized margins are `3.75e-5` for y slopes, `1.38e-6` for active
limiter gaps, and `8.21e-4` for non-target y-face fluxes (the x minima are
larger). Both current-side limiter-choice traces differ from the base trace;
this was allowed by the frozen requalification policy and is recorded in the
candidate receipt. It supports a strict local current branch pair, not a
certificate that the finite path stayed in one smooth region or that no other
switching surfaces matter.

Final closure repeats native and side objectives, both individual side
gradients, the merit, face roundoff, current branch-pair gate, source, fixed
input, runtime, and deadline; all recorded checks pass. The prior
mix-preserving side-gradient tampering regression is present in the tested
suite. The run therefore supports one accepted, endpoint-checked
requalification step under this declared policy. It does not establish a
smooth root, minimum, candidate Hessian/curvature, adjoint response,
reanalysis, forecast accuracy, or model validity.

## Remaining receipt issue

The child and parent `numerical_status` fields say
`one_nonsmooth_coupled_step_accepted`, while `candidate_type` says
`one_nonsmooth_requalified_step`. The plan, solve method, and Hessian scope
make the actual mode clear, but the inherited status label is imprecise and
should be corrected in a later receipt-format change. This is a reporting
issue; it does not change the measured gates or the accepted control.

Artifacts: `candidate_requalification_20261009_attempt1/step.json`,
`step.run.json`, `step.resource.json`, and `step.log`; fixed plan
`CANDIDATE_REQUALIFICATION_PLAN_20261009.json`.
