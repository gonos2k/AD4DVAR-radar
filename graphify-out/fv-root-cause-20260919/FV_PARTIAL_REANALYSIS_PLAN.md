# Proposed G3b: one-lead partial-observation response check

Status: **planned, not executed**. The bounded G3 input and fixed-control
derivative profile is merged; this plan asks whether its *new stationary
point* supports the exact implicit response. It does not rerun the old
all-detected problem and does not certify the two-lead partial response,
finite-amplitude impacts, or weather skill.

The numerical driver and resource wrapper are prepared but require explicit
`--execute` after separate budget approval. `--validate-only` performed the
source/input binding without GN, Newton, PCG or reanalysis; its output is
`fv_partial_reanalysis_validate_only.json`. The pinned cheap input report is
`fv_partial_reanalysis_preflight.json`. Both are distinct from an actual
numerical outcome.

## Frozen problem and direction

Use the existing 4×5/26-control minmod fixture. The three observation frames
are at 0/60/120 seconds and the one forecast verification frame is at
180 seconds. Set raw observation `(1,1,2)` and `(2,2,3)` to NaN before
`prepare_analysis`; preserve 58 detected, 2 genuine missing, 0 censored and
0 QC-rejected cells. Keep the first frame and all initial/boundary support
known. Use observation std 0.1 dBZ, common-bias std 0.25 dBZ, the existing
`B=y0+theta*pattern` with theta 0.02 and the fixed synthetic verification
field from the bounded G3 fixture. Specifically, the archived all-detected
control is forecast through the *new partial* frozen problem, transformed to
dBZ, and added to `0.1*pattern`, then detached. It is a conditional synthetic
target, not independent truth. Freeze and hash its value before optimizing;
never rebuild it from the refined control. The exact constructor is
`tests.test_fv_research_partial_observation._problem`, whose source hash is
bound by the read-only preflight.

The one predeclared direction is `middle_time_valid_bias`: set the 20 middle
frame parameter slots to that frame's fixed `valid_mask` (19 ones, one zero),
all other slots including theta to zero. Missing parameter coordinates remain
inactive. Use `p±h*d` with the same masks and whitener at both signs.

The saved full-observation 26-control point may be used **only as a warm
start**. It is not stationary for this objective: the small preflight found
`max|grad_c J| = 1.083842990538571`. Its Hessian, adjoint, branch or
directional response may not be reused. At that seed the strict tracer
visited 54 SSPRK Euler stages and the minimum scaled slope margin was
`0.0007615693500034068`. These values and the verification-field hash are
recorded in `fv_partial_reanalysis_preflight.json`; the seed branch does not
predict the refined point's branch.

## Preflight and numerical gates

Before a costly solve, pin SHA256 for the eventual numerical driver and code,
exact observations/masks, p,
theta/pattern, frozen whitener/config, boundary schedule, verification field,
warm start, direction and step sequence. The cheap preflight checks the 58/2
mask partition, zero whitener mode, finite objective/gradient and strict seed
branch. The pinned G3 regression separately checks that the fixed-control
mixed derivative is zero in the two missing slots. In addition to the
tracer's slope margin, observe every face flow and require
`min(abs(flux))/max(abs(flux)) > 1e-4` across all stages; the seed's recorded
minimum ratio is `0.007752734889703382`. Store the command,
source commit and environment. Recheck source/input identities at completion.

Set the GN frozen initial background to the same
`prepared.dbz[0] + theta*pattern` used by `FVResearchProblem.contract(p)`.
Use current `solve_analysis` for that *same partial* objective, then the existing
`local_refinement.refine_stationary` with unchanged 8 Newton steps, 16
backtracks and PCG relative tolerance `1e-10`. Accept the nominal point only
if `max|grad_c J| < 1e-10`, its own strict 54-stage branch passes, and its
minimum scaled slope and face-flow margins both exceed `1e-4`. A solver refusal is recorded as
such; it is not called a response mismatch or proof that no root exists.
The GN result's actual selector/face-sign signature is pinned through the
nominal Newton correction. If that conservative local correction refuses a
branch change, record the refusal and stop rather than relabeling another
branch's stationary point as the planned response.

At this newly qualified point, compute fresh `compute_local_response` for the
declared direction and a fresh tangent from `H c_dot = -J_cp d`. Require the
actual tangent and transposed-adjoint relative residuals to be `<=1e-10`;
keep the original Hessian operator. Report direct, indirect and total terms,
whole parameter gradient, branch signature and PCG/HVP costs. Do not infer
global positive definiteness from visited PCG curvatures.

For `h=0.001*2^-j`, `j=0,...,5`, attempt signed predictor/corrector endpoints
from `c*±h*c_dot` with `p±h*d`. Each endpoint must pass its own fixed-mask
contract, `max|grad_c J| < 1e-10`, and exact nominal selector and face-sign
signature at all 54 stages, and each predictor, correction candidate and
accepted endpoint must pass both the scaled slope and face-flow margin floors
of `1e-4`. Compare the central score difference to the
fresh nonzero adjoint response using `|s_adjoint|` as denominator. Before
attempting an endpoint pair, require its expected signed score separation
`2*h*|s_adjoint|` to exceed
`1024*eps64*max(|E_nominal|,tiny64)`. If this signal gate fails, stop with
response validation unresolved; a nonzero but roundoff-dominated slope is
not a successful comparison. This is a necessary signal check, not an error
certificate. Two
consecutive valid pairs with relative difference `<=1e-4` close **only this
point/direction/local-step scope**. A refused pair resets the consecutive
counter and is preserved; exhausting the sequence means unresolved, without
altering tolerances, direction or verification target.

## Proposed resource boundary and outputs

Use one serial preflight capped at 120 seconds and one serial numerical run
capped at **1200 seconds wall time and 1 GiB sampled child RSS**, including
the nominal GN/refinement, tangent/adjoint and attempted endpoints. The
existing `run_guarded` child-process monitor can enforce the cap and retain
exit status, sampled RSS and timing. The numerical child must atomically write
its own phase and signed-side checkpoints and raw refusal reasons, including
the last incomplete state if the outer monitor terminates it. The
previous 26-control local runs took several minutes and about 340–357 MB.
Four endpoint corrections took about 206 seconds in an earlier full-support
direction; twelve could exceed 600 seconds before the fresh partial nominal
solve. The 1200-second limit is a ceiling, not an expected runtime or success
guarantee. An unfinished sequence at the cap remains unresolved.
The existing read-only preflight JSON was produced in about two seconds
without a child resource monitor; it launched no nonlinear or linear solve.
Before any approved numerical run, repeat it under the proposed guard and
require the identities to match the pinned record.

Keep execution completion, nominal eligibility, endpoint eligibility and
independent response validation as separate statuses. A resource cap, branch
switch, zero directional signal, PCG failure, stationarity failure or
incomplete pair must leave response validation **not established**. Publish
raw report/log/manifest and GREEN/RED review before closing G3b. Full
CPU/package regression and physical prediction validation are outside this
single numerical experiment.

The approval-free preflight and existing partial/parity checks report
24 passed with 18 existing TorchScript warnings; pinned basedpyright reports
0 errors, 0 warnings and 0 notes. Exact commands, code/report SHA256, and
non-executed operations are in `fv_partial_reanalysis_preflight_manifest.json`.
The planned numerical driver, `fv_partial_reanalysis_probe.py`, and
`fv_partial_reanalysis_runner.py` are included in the pinned source hashes.
