# Resumable curvature diagnostic — reviewed proposal, not implementation

Status: mathematical design reviewed by GREEN. No checkpoint code, actual FV run or new retry/resource allowance is created by this proposal. The original 240-second attempt did not save its columns; none can be reconstructed from its final report.

## Mathematical operation

At the same original full control c and fixed p, form the exact operator v -> J_cc(c,p)v using gradient JVP. For the declared original coordinate order collect H e_i, i=0..25. No Gauss–Newton, diagonal, shifted-H or preconditioner replacement is allowed. Do not optimize or issue a response.

## Binding and storage

One immutable checkpoint header must bind c/p vectors and their byte hashes, fixed problem/archive/objective/prior identity, observation/boundary/time/CFL/growth/terminal-target identity, original coordinate order and20+6 block split, fresh J/g/Phi and gradient hash, current strict branch signature/margins, operator/helper source map, CPU FP64 runtime, and declared resource/attempt policy. A receipt provider rechecks actual source/input/runtime before and after each product. Side-effecting callbacks remain unsupported.

Each completed column stores its contiguous index, canonical basis direction/hash, finite FP64 product vector/hash, and measured product/serialization cost. Commit through a unique exclusive temporary file plus fsync/atomic replacement. Serialize one checkpoint writer with an exclusive owner claim; a resume must authenticate every stored direction/product and all header identities exactly before computing only missing columns. Mismatch, corrupt hash, duplicate/reordered/out-of-range index or partial write is a refusal; do not repair by silently dropping columns.

## Partial and final results

Persist count and column immediately after validation. At a deadline or typed refusal retain completed columns with an explicit partial status and absent spectrum/SPD/Schur fields. Partial H cannot certify symmetry, positive definiteness, conditioning or a root. Never infer completed product count from the old failed attempt.

After all26 columns validate, reconstruct H and apply the unchanged full-matrix symmetry gate. Compute the minimum eigenpair and evaluate one independent27th HVP at the same receipt. Only a passing true eigenpair residual supports the fresh curvature diagnostic. Then reuse blocks.schur_diagnostic(H,g), including the nonzero field-gradient correction, and keep its refusal separate from execution success. A budget ending after26 but before27 leaves a complete matrix checkpoint with audit_pending, not qualified curvature.

## Resource policy required before execution

A future separately pinned plan must declare per-invocation internal/wall/RSS caps and cumulative allowance/maximum attempts. Every invocation is charged in an immutable run ledger before products; forced termination or incomplete cost must require reconciliation with the external guard record, not a fresh unnoticed budget. A source/identity change invalidates reuse. No automatic lock removal, retry, limit increase, new seed or replay of unrecorded historical work.

## Authoring and execution milestones

1. Implement a small checkpoint collector and synthetic tests: interrupted after k columns/resume only missing columns; exact identity/source changes; malformed product/hash/direction; atomic partial write; exclusive writer; cumulative-attempt policy; all26 before independent audit; refusal retention.
2. Add an explicit optional checkpoint path to the new guarded endpoint route, preserving the historical fresh_hessian source and historical evidence unchanged. Test wiring with a tiny quadratic and fake resources only.
3. Obtain final GREEN/RED reviews and a fixed preflight with exact new source/test/plan/input identities. Only then run the new declared original3h computation; record each actual guard outcome separately.

This proposal does not close fresh3h curvature, normal-point response, finite active-face impacts or meteorological validity. Its purpose is to avoid losing already validated products in the next scientific phase, rather than to lengthen the old run without accounting.
