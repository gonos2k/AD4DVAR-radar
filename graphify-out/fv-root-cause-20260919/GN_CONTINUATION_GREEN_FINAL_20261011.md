# GREEN saved-data review: three GN commits from bef459

The frozen plan `319bb7798348643c0198fd09ae9bbe19ddfb88ae7bcfc94c50a80745b29c2d65` starts from the P2-closed bef459 point. The run made three consecutive fresh current-point GN builds and three independent P2 commits, ending at `454396e77e2f10492e80e0fd625b55e1a08c2be53fe9721b59a4a27d60be3b7a`. Each closed point becomes the next point’s base; all three row/HVP sets bind to their own control and direction. No prior-point products were reused.

Each point used 24 row VJPs, one 12D solve, two HVPs, and a 24-slot candidate cap. Actual accepted slots were 9, 10, and 11 (30 candidates total); run totals were 72 rows, 3 solves, and 6 HVPs. Event gradients/JVPs were zero. Each candidate passed actual-J and actual-R Armijo, current θ/branch/face checks, and an independent P2 repeat. The run stopped at its three-commit cap, with no post-commit readiness or additional candidate.

Saved NumPy arithmetic gives the following per-step R comparison:

| Point | α | Base R | Linearized model R | Actual R | Actual R decrease |
|---|---:|---:|---:|---:|---:|
| 1 | 0.00312562683 | 0.003157465742 | 0.003135464747 | 0.003133504035 | 0.75889% |
| 2 | 0.00154693748 | 0.003133504035 | 0.003122582060 | 0.003127755724 | 0.18345% |
| 3 | 0.000770519935 | 0.003127755724 | 0.003122301320 | 0.003127039113 | 0.02291% |

J decreased at each accepted point. Across the run, ||G||₂ fell from `0.05619133155` to `0.05591993484`, and G∞ fell monotonically from `0.02671637726` to `0.02658902354`. The model predicted too much R decrease on points 2 and 3. The saved vector errors were about 1.002, 1.004, and 1.004 times each actual vector change; the final step’s error was `0.01689×` the new residual norm. These are local model diagnostics alongside observed decreases, not a convergence or optimum claim.

The final record reports `gn_continuation_cap_reached`, 3/3 commits, P2-closed controls, source/input/runtime/deadline closure, zero event operators, and false root/minimum/score/response claims. The guard exited 0 in `331.855 s` at sampled RSS `476,413,952` bytes under 2 GiB. The lossless archive contains 118,723,420 raw bytes and 890,196 gzip bytes. The focused suite reports 14 passed; type checking reports zero errors, warnings, or notes.

This run establishes a three-step accepted prefix under the fixed budgets. It does not establish a stationary point, minimum, exact curvature, response skill, forecast accuracy, or future-point behavior.
