# One alternate-control strict-branch feasibility attempt

PR #217 located one near-zero face flux at the **original** fixed
three-hour point control: `q_y[3,1]`, global callback 0. The source
streamfunction gives

`q_y[3,1] = -(a_X + 3 a_XY + 1.5 a_quadratic + 9 a_X2Y)`.

The original coefficient fractions `[0.7,-0.6,0.5,0.4,-0.3]` with
limits `[0.11,0.08,0.07,0.04,0.03]` cancel at this face. The basis
difference is nonzero, so this is not an all-control structural zero.
A read-only static face scan at the unchanged coefficients found one
face inside the strict near-zero threshold. Replacing only the second
fraction by **-0.59** (latent control `atanh(-0.59)`) gives static
`q_y[3,1]≈-0.0008` and no near-zero faces in that fixed coefficient
field. This is the **only candidate** for the following numerical run;
do not tune it after seeing the trajectory.

Keep the PR #204 point-observation problem, observation and background
parameters, terminal verification score, 0/10/20-minute times, 18
ten-minute forecast leads, boundary schedule and 3,600-stage layout
unchanged. Clone the original control and change only its second flow
latent component, zero-based `control[21]` (`control[-5]`). The score
definition and verification target stay fixed, although the evaluated
score may change. This is a branch-feasibility seed, not an analysis
normal point or a new observation case.

In one guarded child, first verify the archived input/source/policy and
the exact control change. Recompute the static face fluxes through the
actual `bounded_fv_coefficients` and `face_volume_fluxes` functions;
require no face within the unchanged strict `128*eps*max_abs_flux`
threshold before running the candidate diagnostic forecast. The
unchanged `make_case()` first rebuilds its synthetic truth through a
3,600-stage FV trajectory; include this fixture work in the measured
cost. Use the PR #217
first-failure observer to stop at any later strict predicate and
record the global callback, analysis/future phase and cell/face index.
If every callback passes, call the actual
`FVPointResearchProblem.branch_check` at this control to confirm the
full strict oracle and 3,600-stage signature; record only compact
counts/hashes. No GN/Newton, HVP, adjoint, VJP or reanalysis.

Use one fresh `point_3h_shifted_branch_attempt1` directory, one
guarded child, **180-second wall-limit trigger** plus reap grace and
**sampled 768 MiB child RSS**. Do not retry or use another fraction
if a guard, source/input, static face or strict trajectory check fails.
The parent checks archived identity and child provenance without
rebuilding the 3,600-stage fixture outside the child guard. Distinguish
child execution success from numerical branch refusal.
Run focused fake-guard/unit and affected tests, typecheck, Graphify
incremental code refresh, and GREEN/RED prelaunch review before launch.
Preserve raw/resource/parent artifacts and source hashes afterward.

Even a strict-branch pass would establish feasibility only at this
alternate control. It does not establish stationarity, a neighboring
stationary branch, local response, finite influence or physical skill.
R5-R-R remains open until those separate gates pass.
