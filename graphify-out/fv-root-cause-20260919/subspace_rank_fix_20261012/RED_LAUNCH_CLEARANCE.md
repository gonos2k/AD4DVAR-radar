# RED launch clearance — bounded 7839 radius comparison

## Preflight result

The frozen plan passed read-only RED validation. Its SHA-256 is
`466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03`, matching
`EXPANSION_PLAN_SHA256.txt`. `radius_expansion_trial.load_plan()` accepted the
plan and checked its source and archive pins. The source bundle hash is
`ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed`; its 222
tar members exactly match the 222 pinned source entries. The plan has 291
archive pins, all accepted by the plan loader. The only source transitions are
the shared runner and rank-aware subspace solver. The four additional archived
inputs are the frozen PR301 plan, receipt manifest, receipt bundle, and source
bundle.

The receipt-only endpoint loader accepted the PR301 closed winner and both
eligible arm closures against the archived endpoint. The admitted current base
is control
`7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`, with 26
control values, 27 full-residual values, and theta `0.6938343733708355`. This
validates endpoint lineage; current P2, face release, and minimum-mixture P2
remain runtime gates that must freshly pass before the radius callback.

## Frozen limits and runtime gates

The accepted plan declares one global 600-second internal budget, a 660-second
outer guard, and a 2-GiB sampled RSS limit. It caps work at four fresh HVPs,
24 Jacobian-row VJPs, one dense solve, 12 candidate evaluations (six per arm),
two arm repeats, one commit, and zero score evaluations. The radius cap is
`2 * 1.8757177916958825e-5 = 3.751435583391765e-5`. Both arms must independently
pass the inherited actual J/R gates and repeated P2/full-G closure before the
existing repeated-R-then-J selector can commit. Candidate chart displacement
remains bounded by the actual-retraction safeguard.

The fresh-point runner requires a supported native pair, a fresh match to the
admitted P2, no required native face release, and a fresh minimum-mixture P2
match before direction preparation. Radius selection then requires full-rank
fresh residual derivatives, resolved axis-0 KKT signs at the original radius,
a future activation radius above that radius and below the cap, and
roundoff-resolved positive second-direction coefficient and added model gain.
Failure at this gate returns without candidate evaluation. The radius policy
halves rejected or poorly modeled steps and never expands after selection.

## Clearance

No RED launch blocker remains for exactly one guarded run under this frozen
plan. RED ran no fresh FV observation, HVP, candidate, repeat, optimizer,
score, or guarded process. This is preflight clearance only: any result still
needs independent source/input/runtime closure, actual J/R and repeat checks,
and resource/deadline review. The run cannot establish convergence, a minimum,
response improvement, or forecast skill.
