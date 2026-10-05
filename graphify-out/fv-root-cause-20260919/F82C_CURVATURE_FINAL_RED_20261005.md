# F82C accepted-point curvature: final RED review

**Disposition: GO; no blocking discrepancy found in the completed diagnostic.** This review checks the frozen plan, recorded launch, endpoint receipt, Hessian checkpoint, and reported scope. It performs no FV rerun and no new HVP.

## Evidence identity and execution

- Frozen plan SHA-256: `b9b92e69bbcae3d333169f381c8a2dd2a9de9efa105cd4985e0878915bb9ebb2`.
- All 73 plan source pins and all 4 archive pins match the current files. The child records 78 source identities before and after, equal; fixed input and runtime identities also match.
- The accepted step, parent, and resource artifacts are the pinned `bounded_coupled_original_j_20261005_attempt1` artifacts. The child records the expected control `f82cc341…` and parameter vector `8871db49…`.
- The guarded child completed in 235.878 s, sampled peak RSS was 373,686,272 bytes, exit code was 0, and no timeout, cancellation, monitor error, or process cleanup error was recorded. These are inside the declared 300 s and 1 GiB sampled-RSS limits; RSS is sampled, not an OS hard memory cap.
- One of three outer launches and one of three internal kernel reservations were used. The outer ledger stopped on completed curvature with 300 s reserved. No partial resume was needed.

## Numerical and scope review

- Fresh endpoint `J`, `Phi`, gradient infinity norm, and all 26 gradient components passed their pinned comparisons. The pointwise branch check passed all 3,600 stages; the independent complete-signature hash agrees. The branch partition records 360 analysis-replay stages and 3,240 future-forecast stages. This is a pointwise partition, not a path or branch-stability certificate.
- Checkpoint schema `advar.hessian-checkpoint.v2` is completed with all 26 basis columns and the independent 27th minimum-eigenvector HVP. Its receipt binds the same control, parameters, full gradient, objective/problem input, sources, runtime, branch signature/margins, and plan/step hashes; the stored final receipt agrees with the checkpoint header.
- The recorded original full-control `J(c,p)` Hessian is symmetric to relative error `1.21e-15`. Its eigenvalues range from `0.07362217` to `4202.87465`; the independent eigenpair audit passed with relative residual `1.39e-16`. The initial-field block is positive definite in the reported diagnostic (`min eigenvalue 0.9341`); the Schur diagnostic is positive definite (`min eigenvalue 0.38267`) and its scope is correctly labeled as a linearized Newton residual, not a profiled gradient.
- The accepted point remains nonstationary: `||g||_inf = 1.18153416`. The positive spectrum therefore describes curvature at this branch-fixed point; it does not establish a stationary point or local minimum. The report correctly leaves `full_root_claim`, `minimum_claim`, and `physical_validated` false, and records no step, response, or score.
- Strict branch margins are complete and finite, but the smallest normalized active-limiter gap is `5.18e-7` in the future forecast. This does not invalidate the strict pointwise result; it emphasizes that the Hessian is local to the recorded branch and should not be generalized across nearby branch changes.

## Cross-artifact checks

The child is `3058fec5…`; the parent names that exact child hash and reports completed execution with no child-read error. The run ledger repeats the parent/child/resource hashes, records source and archive integrity, and stops for the expected reason. Child, parent, resource, outer ledger, and checkpoint all agree on the same plan SHA and one-launch completion.

Reviewed artifacts: [child audit](f82c_fresh_curvature_20261005/attempt_1/audit.json), [guard resource record](f82c_fresh_curvature_20261005/attempt_1/audit.resource.json), [parent run record](f82c_fresh_curvature_20261005/attempt_1/audit.run.json), [outer experiment ledger](f82c_fresh_curvature_20261005/experiment.json), and [Hessian checkpoint](f82c_fresh_curvature_20261005/checkpoint/hessian_checkpoint.json).
