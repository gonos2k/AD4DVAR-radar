# R2 point-objective exploratory basin search, attempt 2

This is a new, separately declared run on the **unchanged** 4×5 correlated
point problem and the same single warm control. Attempt 1's child report has
SHA256 `e0cfde720bd2002a7d8149eca3553c244a9f93f45d256e3324066c58e9a158dc`.
It reduced the objective from 0.07993508 to 0.00644675 but ended after 13
accepted steps with `||grad J||_inf=0.289`, when iteration 14 had 16
generic stage/slope/face candidate refusals. Its last accepted slope margin
was `0.0001000167`, just above the search gate `1e-4`. The failed
candidates' individual margins were not saved, so **attempt 1 does not
prove which check caused each refusal**. Its raw report and exact
search/producer/runner/plan source snapshots remain untouched.

This run separates **exploratory pointwise differentiability** from **final
response robustness**. During only the objective-descent L-BFGS search,
every accepted candidate must pass the existing 54-stage minmod strict
tracer, including its `128*eps64` scaled slope/tie and face-sign checks,
and have finite positive recorded scaled slope and face-flux margins. No
additional `1e-4` search floor is imposed. The actual unchanged point
objective, Armijo decrease, finite gradient-before-acceptance, positive
curvature-pair check, branch-change L-BFGS reset, max 100 accepted steps,
1600 trials and 600-second search phase cap remain as in attempt 1.
The search may jump across branch surfaces; equal endpoint signatures do
not certify an interior path. Curvature pairs are heuristic endpoint secants.

Record for every trial its control SHA256, step scale, status/refusal class,
branch signature digest and measured slope/face margins whenever a full
trace completes, plus L-BFGS fallback/reset and objective/gradient for
accepted iterates. A strict-trace failure keeps the tracer's specific
exception text and the candidate digest; unavailable margins stay null,
not invented. Checkpoint accepted controls/gradients and one full first
example per refusal class atomically.

If the search reaches `||grad J||_inf <=1e-4`, check its terminal point
with the **unchanged final** strict 54-stage signature and both margins
`>1e-4` *before* building a Hessian or calling Newton. A terminal point
below either final margin is `exploration_terminal_ineligible`, even if
its objective decreased or gradient is small. Only an eligible terminal
point proceeds to the same fresh exact 26-column finite/symmetric/SPD
Hessian gate, unchanged exact-HVP Newton–PCG refinement, final gradient
`<1e-10`, pinned signature, and repeated final SPD check. The final
response/adjoint and signed reanalysis are **not** run here.

One serial child has a 1200-second wall cap and sampled 1-GiB child RSS
cap, with outer elapsed/PID/status/source/input/plan checks. The sampled
memory bound can miss spikes. `response_validation` remains
`not_performed` on every status. A refusal or cap does not justify a
same-run threshold change, a second seed, or a claim that no stationary
point exists.
