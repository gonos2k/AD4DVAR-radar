# Approved local 86-control reanalysis check

Use the recorded seed-A refined point and fixed problem/verification from
PR174. Do not rerun GN or repeat the second essentially identical baseline.
Select the existing middle-time uniform dBZ direction: 80 ones at the middle
frame, zeros elsewhere, theta fixed. This does not directly alter B=y0+theta P.

1. Bind/check nominal inputs, actual gradient, branch and saved response against
   the common problem. Form b=J_cp d by a parameter JVP and solve H c_dot=-b
   with the existing exact-HVP PCG (rtol1e-10, cap344), checking actual residual.
2. Use the predeclared sequence h=.001*2^-j for j=0..5. Start each plus/minus
   correction from c0 +/- h*c_dot, preserving the baseline strict 108-stage
   branch. Refine using unchanged gmax<1e-10 / existing 8-by-16 budgets.
3. Record both endpoint controls, scores, gradients, branch identities and
   costs. Compare central reanalysis slope with the saved/fresh checked adjoint
   direction response; relative error <=1e-4 at two consecutive h values ends
   the experiment. Refused points remain records; do not alter direction,
   tolerances, objective or seed selection to obtain success.
4. Report validation only for this direction, nominal point and accepted local
   steps. No finite-path, realistic finite-impact, all-parameter or skill claim.

Measured HVP cost in PR174 was ~1.4–1.5 seconds; one 36-iteration linear solve
is roughly a minute. A tangent solve plus four corrected endpoints is estimated
at ~5–10 minutes when one/two Newton corrections suffice. More rejected or
poorly conditioned points can cost more; this is not a convergence guarantee.

Proposed overall cap: **20 minutes CPU wall time, sampled RSS 2 GiB**, serial,
including tangent and all attempted pairs; terminate with preserved checkpoint
at the cap. These are new computations outside the previous two-start execution
approval. The user approved this separate resource budget with “go” on 2026-09-24
15:30 JST. Execution outcomes are recorded separately from this plan. No other grid or observation direction is included.
