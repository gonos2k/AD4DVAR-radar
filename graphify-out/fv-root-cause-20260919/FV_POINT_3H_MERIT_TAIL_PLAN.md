# One smaller-alpha continuation from the PR #223 refused epoch

PR #223 finished normally after two strict, branch-changing J/Phi
descent steps, then refused all eight endpoint trials in its third
epoch. Its final accepted 26-control tensor has SHA256
`baed75b17f6552cecf999be2a23db75725fef8d713a1dd360326ee82680159ba`,
strict signature
`2925e02c77e293436b8eccfeda1db44390bb4ca6f106f5a5b581818d9ab57dc5`,
J `0.9314590521034872`, Phi `0.9755199172601001` and gradient
maximum `0.8727746789931015`. Pin the raw report SHA256
`eb5b9d1ea712766607fc87b67d1053bd8b103dad1fdc06bcd8fa2916293b248c`
and aggregate evidence SHA256
`99c9fe27c2df7136a881e03d2f886709379ef8ded66d2b41af3dd9d224688b8f`.
The third epoch's smallest tested alpha `0.00015625` decreased J by
`3.6223e-6` but increased Phi by `0.0223059`; **it was correctly
refused**. All eight tested endpoints changed branch. The negative
local J/Phi slopes at the current point make a smaller-step tail
worth testing, not a prediction that the next halving succeeds or a
finite-segment smoothness certificate.

Reuse the existing
`examples/weather_scenarios/fv_point_3h_merit_continuation.py`
numerical loop by refactoring its hard-coded seed and alpha/budget
constants into a small immutable, explicitly selected run policy.
Do not copy the 1,000-line script or change its mathematical gates.
Preserve the original policy values and default CLI behavior with
focused regression tests; PR #223's exact executed source remains
available at Git commit `32ee38d72084735555e11ab4fbbad34afc321f45`
(source SHA256
`00bed148ffc7549eca3cd8446ac1fd1d5c4042f1278867808aafe78f51e848dc`).
Historical raw/resource/parent records are immutable; the refactored
current source has a new hash and must not be presented as the code
that executed PR #223.

For this **one new tail profile**, reconstruct the same PR #204 4×5
point-observation problem with unchanged 13 parameters, external
background, verification target, 0/10/20-minute observations, 18
future leads and boundary schedule. Set only the starting control to
the pinned PR #223 accepted endpoint. Use the fixed alpha sequence
`0.000078125*2^-k`, `k=0,...,11` in each epoch, at most **2 accepted
epochs** and **24 total endpoint trials**. Do not try the already
tested `0.00015625` or any larger alpha, change direction, or retune
the grid after seeing outcomes. Stop at gradient maximum `<1e-4`
as a **handoff candidate only**.

Retain all original gates: fresh complete 3,600-stage current branch
and normalized margin pass; finite J/g/Phi; fresh exact Hg and H^Tg
with relative transpose discrepancy at most `1e-8`; scaled positive
`g^THg` and finite negative true J and Phi slopes; full original
strict endpoint branch before J/g/Phi; separate J/Phi reductions
beyond `128eps` scaled floors; both Armijo tests (`c1=1e-4`) when
the endpoint signature equals the current signature, otherwise
actual dual decrease only. Recompute the branch-change flag from
signatures, checkpoint each trial, and relinearize after any accepted
endpoint. Neither matching nor switched endpoint signatures certify
the path. Every candidate status and numerical refusal must remain
separate from process/resource completion. No GN/Newton step, score,
adjoint, VJP, signed reanalysis or sensitivity publication.

Run exactly once in a fresh `point_3h_merit_tail_attempt1` directory.
The child has a **240-second internal search budget** inside a
**300-second wall-limit trigger** plus reap grace and **sampled
1 GiB child RSS**. Check the internal deadline after expensive
operations and preserve partial records on guard termination. The
parent performs only identity/source/plan/guard/report checks, no FV
fixture construction. No retry or unplanned fallback. Before launch,
run focused original-and-tail policy tests, affected tests, targeted
typecheck, incremental Graphify refresh, and independent GREEN/RED
prelaunch review; archive raw/resource/parent outputs and hashes.

Any accepted endpoint is an exploratory candidate, not a stationary
root. Final response work still requires that endpoint's own strict
branch/margins, `<1e-10` gradient, exact unshifted Hessian
suitability, original-H true adjoint residual, full 13-parameter VJP
and two signed nonlinear endpoint sizes. R5-R-R remains open until
those separate gates actually pass.
