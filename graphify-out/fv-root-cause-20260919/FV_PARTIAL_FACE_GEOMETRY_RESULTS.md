# R4-R two-hole root attempt: face-flux geometry diagnostic

This is the read-only diagnostic declared in
`FV_PARTIAL_FACE_GEOMETRY_PLAN.md`. It uses the **archived PR #209 GN
seed and all 52 Newton trial controls**, rather than starting another
GN or root search. The PR #209 manifest and four raw files match their
pinned SHA256 values. The old child source map matches that manifest;
all 16 geometry/input-critical current source files match their
archived SHA256 values. Only the two declared PR #210 status-only
probe/runner sources differ. The fixed two-hole input identity matches
the archive before and after this diagnostic. The reviewed plan and
diagnostic-probe hashes were passed on the command line and verified
before and after execution.

The production coefficient map, streamfunction sum and
`face_volume_fluxes` reconstructed the stationary (q_x,q_y) arrays
for each saved 26-control vector. Each vector matched its archived
control hash. For **all 53 vectors**, the recomputed
`min(abs(face flux))/max(abs(face flux))` matched the archived 54-stage
branch margin exactly at the recorded FP64 precision; maximum
absolute difference was **0**. All minima were unique at the
declared roundoff scale.

| Archived points | Unique minimum face | Count |
|---|---|---:|
| GN seed and trials | (q_y[2,0]) | 52 |
| Rejected iteration-2, backtrack-0 trial | (q_y[1,0]) | 1 |

At the GN seed, (q_y[2,0]=-2.2849761492\times10^{-5}). The first
accepted signature switch moved it to
(+2.6041837408\times10^{-6}); the second accepted switch returned it
to (-2.0221724001\times10^{-5}). Of the 44 rejected trials, 43
changed the sign of the **same unique minimizing face** relative to
their preceding accepted point. The remaining rejected trial changed
the minimum-face identity, so this narrow sign-crossing statistic is
undefined for it. These are endpoint comparisons, not a certificate
of every face sign along a trial segment. Since (q_y[2,0]) is a
continuous function of the control, each of the first two **accepted
straight step segments** contains at least one zero of that face
flux. This does not locate the crossing or certify other limiter/flux
events on the segment. The 43 rejected sign changes belong to trial
candidate chords, not accepted optimizer steps.

After the second accepted step, all six later accepted points retained
the same archived full signature and had (q_y[2,0]<0). Its absolute
flux fell from (2.0221724\times10^{-5}) to
(3.6893559\times10^{-8}), a factor of about **548.11**. Over those
points the largest absolute face flux stayed between `0.0666165072`
and `0.0666309199`, a relative span of only **0.0216%**. Thus the
recorded margin decline from `3.0355e-4` to `5.5370e-7` is due
primarily to this specific face's flux approaching zero, rather than
an increasing denominator or a moving minimum face.
The last accepted absolute flux is still well above the tracer's
roundoff zero threshold; it is a resolved, low-margin face, not an
unresolved numerical zero.

This localizes a **geometric upwind-switch boundary** approached by
the accepted sequence. It does not establish that this face is the
sole cause of merit-policy refusal, that its flux is dynamically
important at every stage, or that the inverse problem lacks a smooth
stationary root elsewhere. A low-margin endpoint is not an eligible
classical implicit-response point. The original PR #209 result remains
an eight-step `root_refused` execution with no adjoint, VJP or signed
nonlinear reanalysis.

The diagnostic took 3.188 seconds inside the process. It performed
two fixed-input preflights and one fixture construction, which include
FV forward/branch and warm objective/gradient evaluations, plus 53
small face-flux reconstructions. It did **not** rerun product GN,
Newton refinement, adjoint or nonlinear perturbed reanalysis. Peak
memory was not separately measured. The old raw archive was not
modified.

The affected local suite passed **53 tests** with 18 existing
TorchScript warnings; pinned basedpyright 1.39.9 reported zero
diagnostics for the new probe/test. A code-only Graphify update
reported 7,665 nodes, 67,006 edges and 251 communities. These checks
do not constitute a full CPU/package regression or independent
forecast-skill validation. GREEN and RED independently checked the
raw geometry, provenance and the causal limits above.
