# R4-R exploratory alternate-sector seed gate

This is one **post-hoc exploratory seed qualification**, not a new
independent observation case or a continuation of the PR #209 accepted
path. Pin the PR #209 root-attempt manifest and raw child hashes; keep
the exact two-hole 4×5 input tensors, objective, prior, boundary and
verification unchanged. The PR #210 status-only source edits and
PR #211 face-geometry diagnostic are separate from the original
transport/objective, whose archived geometry-critical hashes must
still match. No product GN, Newton refinement, adjoint or signed
reanalysis occurs in this gate.

Deterministically inspect all 52 saved trial rows in order. Track
the *preceding accepted* full-signature SHA, beginning at the GN seed
and updating only after an accepted trial. A candidate qualifies for
this seed ranking only if its signature differs from that preceding
accepted signature, the stored maximum gradient is finite, both
stored scaled slope/face margins are **strictly greater than**
`1e-4`, and its 26 FP64 control values match the stored SHA256.
Sort qualifying rows by `(gradient_max ascending,
min(slope_margin, face_margin) descending, iteration ascending,
backtrack ascending)` using exact stored FP64 values. There must be
**29** qualifying rows and one unique selected row: iteration 5,
backtrack 3, control SHA256
`125e0200fdf85f996715ba1bc04aa3f631333614bf15cf2c16630b94c1897421`,
full signature SHA256
`50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d`.
It was previously `merit_switch_refused`; its merit increased from
the preceding accepted point, so do not call it an accepted step or
a policy improvement. No fallback seed is allowed if it fails.

Rebuild the fixed current problem and recompute the selected seed's
objective, gradient 2-norm, maximum gradient, full 54-stage strict
minmod/upwind branch and scaled face margin. Require the archived
values to agree within declared `rtol=1e-10, atol=1e-12` for scalar
diagnostics and exactly for the full branch signature; both fresh
margins must remain `>1e-4`. The archived measurements are
`J=0.02076496753914496`, `||g||_2=0.012208793948431365`,
`||g||_inf=0.006848117531724784`, slope margin
`0.0004060611332810128`, face margin
`0.00011662714171735749`. This point is only 1.166 times the face
qualification floor, not a robust interior seed.

Only after the branch and scalar checks, build all 26 exact
gradient-JVP Hessian columns at this selected seed. Require finite
entries, relative antisymmetry `<=1e-10`, minimum eigenvalue `>0`
and minimum/maximum eigenvalue ratio `>sqrt(eps64)` before declaring
`seed_locally_spd`. Distinguish `seed_branch_refused`,
`seed_curvature_refused`, resource termination and execution errors;
none proves that a root does not exist. Do not solve the Newton system
or emit sensitivity in this gate. If the seed passes, a later,
separately guarded root-only search may start **directly** from this
fixed candidate with unchanged 8/16/104 limits and final classical
stationarity/branch/curvature gates.

Run exactly one serial child with a 120-second wall and sampled
1-GiB child-RSS ceiling. Record command/PID/exit, elapsed/RSS,
archived and current source/input/plan identities before and after.
The current source may differ from PR #209 only in declared status
and read-only diagnostic files; never substitute a different
objective/transport. Classify an identity or malformed archive
disagreement as execution failure, not a scientific seed refusal.
The imported PR #210 status probe and runner must match their reviewed
current-source SHA256 values; the raw manifest hash, raw filenames,
and status-only source exceptions are literal constants in the new
gate, not imported from a mutable diagnostic helper. Require reviewed
plan, gate-probe and guarded-parent runner SHA256 inputs at launch and
verify them before and after the child.
