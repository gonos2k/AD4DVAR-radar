# PR267 mathematical RED audit — 2026-10-09

## Disposition

I found no actionable mathematical defect that invalidates the archived three-step continuation. The saved endpoint arithmetic, selected-face chart direction, accepted line-search checks, and fresh endpoint closures are mutually consistent. I found one test qualification gap: the curved-face helper test uses side objectives that do not agree on the face, so it does not establish the same-face identity required by a true piecewise objective. This limits what the test proves; it does not undermine the production run, whose saved side-HVP differences are confined to the face-control block.

## Independent saved-vector checks

Using `.venv/bin/python` and the decompressed raw child only, with static face weights `[0, -0.08, -0.28, -0.14, -0.84]`, I checked the three stored directions against the Euclidean projected mixture gradient and the actual pivot-24 chart. The infinity error in `direction + tangent_gradient` is `0`, `0`, and `1.74e-18`; `normal·direction` is at most `8.7e-19`; and reconstructing `Z @ (-tangent_gradient[retained])` matches each stored direction exactly. Thus the retained-coordinate construction is valid for this particular implicit chart and pivot. This conclusion relies on the chart's identity rows for retained coordinates and `Z.T @ normal = 0`; it should not be generalized to arbitrary matrices passed as `chart_jacobian` without those properties.

The raw gzip decompresses to SHA-256 `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`. It records three accepted iterations, six completed HVPs, and final control hash `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7` at `theta=0.332026125873074`. Each move is below radius `0.05` (`1.10312e-4`, `2.59245e-5`, `3.39889e-5`); both Armijo predicates, the face audit, paired-branch gate, and stored fresh-repeat facts pass. Current-point side-gradient jump has tangent norm at most `5.8e-16`, within its recorded `~8.2e-15` support budget. No production FV/HVP/guard rerun was performed.

The saved two-side HVP vectors agree on non-face coordinates to roundoff at all three points; their material differences occur only in controls 20–24, where the selected face normal is supported. This is consistent with the expected equality of tangential derivatives for the production extensions. Receipt-level execution and endpoint facts were read from the raw gzip, rather than re-derived by evaluating production code.

## Test gap (P2, coverage/qualification)

In `tests/test_fv_point_3h_tangent_continuation.py:29-95`, `test_26d_curved_face_hvps_match_autodiff_along_true_chart_tangent` defines side objectives using diagonal matrices `a` and `b` that differ across all 26 coordinates. At a nonzero chart displacement with `Q=0`, the term `0.5 * dx @ (a-b) @ dx` generally remains nonzero, so these extensions do not represent one common objective restricted to the selected face. The test checks that each saved HVP equals AD of its own extension at the base point; it does not check that the side objectives agree along the face or compare the full chart-path derivative of `[g_mix, Q/0.84]` with `residual_direction`.

This is actionable for test coverage because an inconsistent pair of extensions can pass the current test while violating the mathematical premise for a nonsmooth objective defined by two smooth sides. A focused follow-up can use one shared smooth base term on both sides and distinct coefficients multiplying the face function `Q`; then compare the full AD derivative of the mixed-gradient-plus-face residual along the implicit chart and `theta(alpha)` path with the predicted `DF`. The production archive remains supported by its own paired native objectives, side gradients, branch-pair receipts, and fresh repeats; this test gap alone is not evidence against that result.

## Helpers and bounds reviewed

`tangent_direction` / `tangent_model` (`fv_point_3h_tangent_continuation.py:106-204`) use the chart pivot supplied by the production callback, project the mixed gradient onto the face tangent space, require the one-sided gradient jump to be face-normal within a scale-based roundoff budget, and refuse an unresolved normal denominator. The actual archived direction verifies the structured-chart relation above.

`candidate_alphas` and `search_candidates` (`:205-256`) cap by the actual direction norm and first-order squared-merit decrease, then halve at most 16 times. Search checks `theta` in `[0,1]`, measures the actual chart-point displacement, and refuses radius excess before evaluation. Production uses fixed positive finite defaults (`radius=0.05`, `face_scale=0.84`) and enforces the candidate limit. I found no zero-bound or overflow path exercised by these production values that defeats a gate. Direct callers of the lower-level `candidate_alphas` helper can supply unvalidated `radius`/`limit` values, but the production continuation calls it only through the bounded search, which validates `limit`; this is not a current-run defect.

Candidate evaluation checks actual native J Armijo using the larger one-sided slope, squared residual Armijo using fresh HVPs and the current `theta` direction, actual face audit, branch pair, side/native objective equality, and finite side gradients (`:648-685`). Fresh closure requires exact proposal control/theta identity plus both side-gradient matches, trace, objective/merit, face, source/input/runtime, and deadline facts (`:257-282`, `:687-720`). I found no mismatch between these predicates and the claims in the raw archive.

The repository's previously saved focused-test/type-check logs were read but not rerun. This was a read-only mathematical audit; no code or production evidence was changed.
