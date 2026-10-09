# PR 267 follow-up — final GREEN review — 2026-10-09

## Disposition

The follow-up patches address the recorded progress-checkpoint, recovery-validation, and mathematical-test coverage findings. I found no new blocking code, test, or evidence issue. The archived PR 267 result remains supported by the original run source and immutable saved arrays; this review performed no FV, HVP, guard, weather, or test execution.

## Source identity and saved result

The frozen plan remains SHA-256 `cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69`. Its original runner and continuation-test pins match `tangent_actual_source_20261009/manifest.json` and the archived snapshot bytes exactly:

- Runner: `b184d15f8179290d273d117fb4b145e18b366a9f9ad777b0b9495095c2f228ca`.
- Test: `8cfe139f8a1ef76c682ac60bf6f9ae5d34140e61623cb0e4f7ffecb291cb5551`.

Those two live files now differ because of the reviewed follow-up patches. The other 124 source pins and all 94 archive pins still match. The original gzip remains SHA-256 `62d92dd38a6f97db4871d464201183db9b4bbda31ea224bb8c390646b917958b`; it still decompresses to raw SHA-256 `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`. Run/resource receipts still match their manifest pins. The frozen plan, raw result, and recorded J/F metrics have not changed.

The patched live runner correctly cannot use the old plan: its source pin no longer matches the current code. Reproduction of the historical run must use its archived runner/test, or future production work must use a newly reviewed, source-bound plan. No new production run was attempted.

## Code and regression review

The progress callback runs only after the candidate passes its independent final repeat and `bounded_continuation` appends the accepted iteration. It now atomically writes the iteration list, accepted count, current control/digest/theta, and last-confirmed control/digest/theta/count together. This closes the prior partial-receipt mismatch where a durable accepted count and last-confirmed point could coexist with stale current-point fields. The write uses a temporary file and `os.replace`.

The exception-recovery path rejects malformed durable checkpoints unless the 26-element control is finite and hashes to its receipt, theta is a finite non-boolean number in `[0,1]`, and count is a non-boolean integer in `[0,3]` consistent with whether the control differs from the base. Invalid checkpoints are marked `invalid_checkpoint`, report zero accepted commits, and do not promote candidate control/theta fields. The new parameterized regression covers theta `99`, count `-4`, and boolean count; the progress regression exercises the actual final-repeat/progress callbacks and verifies a simulated `SystemExit` leaves a self-consistent accepted checkpoint.

The revised curved-face AD test now uses a common smooth Hessian and linear term with differing coefficients of the face function `Q`. The side objectives therefore agree on `Q=0`, while retaining distinct normal-side curvature. It also differentiates the full `[g_mix, Q/0.84]` residual along the actual curved chart and `theta(alpha)` path and compares that derivative with the saved model direction. This tests the derivative identity on a valid common-face example.

The saved predecessor reproductions show the new checkpoint assertion failed against the archived original runner, and all three malformed-checkpoint cases failed against the original recovery logic. These are targeted mock reproductions, not a real guard kill, filesystem fault, or production transport test.

## Verification and remaining scope

The saved focused verification reports **49 passed**, 18 existing TorchScript deprecation warnings, 2.17 seconds; the saved type check reports **0 errors, warnings, or notes**. The focused suite adds four outcomes (the interruption test plus three parameterized recovery cases) to the prior 45-pass log. `git diff --check` is clean. I inspected these saved checks rather than rerunning them.

The original three-step numerical result remains execution-local continuation evidence only. The follow-up fixes add durable receipt consistency and strengthen unit-test mathematics; they do not certify a smooth root, nonsmooth minimum, tangent curvature, response, reanalysis, or forecast-skill gain. New PR/CI integration for these follow-up patches remains pending the root's planned GREEN/RED handoff and publication.
