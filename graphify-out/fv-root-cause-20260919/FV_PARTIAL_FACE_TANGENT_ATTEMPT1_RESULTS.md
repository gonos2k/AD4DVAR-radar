# R4-R finite-offset tangent-gradient diagnostic

The one declared guarded run completed. It recomputed the eight objective
gradients at PR #225's finite off-event controls and projected them onto
each recorded event's tangent hyperplane in the existing 26-control
Euclidean coordinates. Their tangent norms remain about `0.0096`, mostly
in the 20 initial-field controls. The sampled left/right gradient
segments also remain this distance from zero. These early archived
crossings therefore retain a substantial field-dominated residual;
the face sign switch alone does not remove it. **R4-R remains open.**

The production face scalar stayed a Tensor during differentiation.
Only `grad_c q_y[2,0]` was evaluated at an event; the objective was
differentiated at the eight fixed finite controls. Both event normals
have exact zero field/growth components and flow-only norm
`0.1729147487`. The recomputed gradient-2 and gradient-infinity scalars
match PR #225 within the predeclared FP64 tolerance. Original branch
signatures, scopes, 54-stage counts, slope/face margins and response
refusal flags accompany every sample.

| Chord and offset | Left tangent norm | Right tangent norm | Sampled gradient-segment minimum norm |
|---|---:|---:|---:|
| Seed → accepted step 1, `2^-8` | `0.0096031214` | `0.0095916579` | `0.0096020266` |
| Seed → accepted step 1, `2^-10` | `0.0095940632` | `0.0095911973` | `0.0095937914` |
| Step 1 → step 2, `2^-8` | `0.0095741750` | `0.0095832196` | `0.0095823573` |
| Step 1 → step 2, `2^-10` | `0.0095732954` | `0.0095755565` | `0.0095753422` |

At the near offset, opposite-side tangent vectors differ by
`2.9061270e-6` and `2.2992769e-6` for the two distinct events.
Same-side near/far changes range from `8.7958414e-7` to
`9.0581989e-6`. These are small compared with the reported tangent
norms, while the normal slopes have opposite signs: first chord
`−0.0007886421 / +0.0075445838`, second chord
`+0.0075443846 / −0.0007887418` (left/right at `2^-10`).
The segment-minimizing weights are near `0.0950` and `0.9050`,
respectively; mixing the normal slopes does not cancel the shared
field-dominated tangent residual.

This is a **finite-offset, coordinate-dependent** diagnostic. The
projections use the normal at each event and are not gradients of the
objective restricted to the curved face-zero surface. The two events
were kept separate. No limiting-gradient convergence, Clarke
subdifferential, constrained stationary point, smooth root, classical
implicit response or cause of the original optimizer refusal was
certified. All eight samples retain their archived face-margin failure
against `1e-4`. A later numerical method can use the measured residual
to motivate field/tangent corrections, but must still verify actual
objective behavior and find a qualified final point.

The plan SHA256 is
`1aac5f48f2dd02cf303131a7a42fbea1c8b69e87c0cc12943516b47434d9a900`;
reviewed probe SHA256 is
`1ad1e6b05b28252572e5b68d6feb64fb66b0ffef07506d0e2dfc991fff86e367`.
The caller supplied that hash before launch, and the shared resource
runner was separately pinned. PR #209 and #225 raw evidence, sources
and fixed inputs matched before and after. The child exited 0 and the
parent reports `completed`; numerical status is
`finite_offset_tangent_gradient_diagnostic_only`.

The guard measured **3.980 seconds**, 15 sampled child-RSS observations
and peak **327,598,080 bytes**, below the 120-second and sampled 1-GiB
triggers. Child numerical time was **2.448 seconds**; parent preparation
and postchecks are outside that child measurement. The affected local
selection passed **36 tests**, and error-level basedpyright 1.39.9
reported zero diagnostics. Graphify's two-file code-only AST refresh
preserved untouched graph records and produced 8,338 nodes and 210,532
links. GREEN/RED reviewed plan, code and actual records. No new GN/root,
adjoint, VJP, reanalysis, full CPU/package regression or real-radar
validation was performed.
