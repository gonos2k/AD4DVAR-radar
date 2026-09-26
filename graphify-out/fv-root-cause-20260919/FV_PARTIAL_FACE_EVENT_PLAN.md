# R4-R two-hole face-event diagnostic, attempt 1

This is a read-only, root-only diagnostic for the original 4×5 collocated
problem with two missing observations. It does not change the objective,
mask, prior, fixed verification field, minmod trace, stationarity threshold
or response margins. Its input is the PR #209 source-bound archive
`partial_sector_root_attempt1`, manifest SHA256
`d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd`.
Require the archive child/parent/resource/log hashes from that manifest,
both fixed-input preflight checks, and the archived control SHA256 for the
GN seed and the first two **accepted** trial endpoints. Neither rejected
trial controls nor a new GN/root run are used.

For each straight chord GN seed→accepted iteration 1 and accepted iteration
1→accepted iteration 2, reconstruct the production time-constant face
flux from the exact current transport coefficient map. The endpoint
`q_y[2,0]` signs must differ. Locate its zero by at most 64 bisections
on the chord, stopping only at FP64 representation or a bracket-width
at most `2^-48` in chord parameter. Record the bracket and face-flux
values. Use the midpoint of the final bracket as `t_event`, or the
exact representable root itself if one is encountered. Do **not**
differentiate at the zero or claim that this is the
only event along the chord.

At four off-event points per chord, use chord parameters
`t_event ± 2^-8` and `t_event ± 2^-10`, provided each is strictly within
`(0,1)`; record an out-of-segment point as unavailable without a
replacement offset. For each available point, inspect the complete
54-stage branch and scaled slope/
face margins. If strict tracing refuses, retain an explicit refusal and
do not treat its derivatives as qualified. At each admitted finite point,
measure `J`, `g=grad_c J`, `Phi=||g||_2^2/2`, `g·d` and the exact
gradient-JVP `H d` for chord direction `d=c_end-c_start`; record
`g·H d`, finite status and the full branch signature. The latter is a
one-sided **local** derivative of gradient merit along the given chord,
not a global descent or root certificate. Keep both sides and both
offsets separate. Compare the two signatures on each side: if they
differ, label attribution to the selected face **confounded by other
branch changes**. Report strict-branch admission separately from the
existing response-margin qualification requiring both scaled slope and
face margins `>1e-4`; admission alone is not response eligibility. No
interpolation across a branch switch is a classical Hessian claim.

Predeclare one serial child with a 300-second wall trigger and sampled
1-GiB child-RSS trigger, using the existing process resource guard.
Write only to a fresh attempt directory outside the PR #209 raw archive.
Pin the final reviewed probe and plan SHA256 before launch and check
source, archive, input and plan identity before/after the computation.
If a required identity or endpoint sign fails, classify the diagnostic
invalid; do not relax numerical gates or rerun with post-hoc offsets.

The only possible conclusion is which one-sided branch/merit behavior is
observed near a sign-changing face zero on these two historical chords.
This does not isolate the selected face as the cause of the archived
signature switch or refinement refusal. Even if one side has a negative
local `g·H d`, this does not prove an eligible stationary root exists.
R4-R remains open until a fresh strict stationary branch, exact final
curvature and margins, adjoint/VJP and signed endpoint reanalyses pass.
