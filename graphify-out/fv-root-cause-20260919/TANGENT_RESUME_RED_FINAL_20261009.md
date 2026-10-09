# Tangent-resume final RED audit — 2026-10-09

## Disposition

The saved resumed run supports three bounded current-tangent corrections from the accepted PR267 endpoint. Receipt identities, the control/theta chain, saved residual arithmetic, accepted actual J/F² checks, selected-face constraints, current branch pairs, and fresh-repeat closures agree. The endpoint remains substantially nonstationary; this is not a root, minimum, response, or forecast result.

No second production run, FV/HVP regeneration, test run, or guard was performed for this RED audit. All arithmetic below uses the archived raw arrays and receipts.

## Provenance and execution

The resume plan SHA-256 is `d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10`; all 129 source pins and 105 archive pins still match local bytes. The raw child SHA is `15c780de8e6b55216407d399d7ce1984280024cfb7f53cbc0ce9e6e4a644db1d`. The gzip SHA `477f0272fc9ff021abb479afb3f7c7e789dc02fff5d948b1d66959ab5f1446d5` decompresses exactly to that raw SHA. Run/resource hashes match the archive manifest; the parent child digest matches the raw bytes and its nested resource object matches the standalone resource receipt.

The child and parent report `completed`, with `tangent_continuation_cap_reached`, 3 accepted steps, and 6/6 completed current HVPs. The one guarded launch took 79.278 seconds, sampled 297 RSS points, and peaked at 565,248,000 bytes under the 300-second and 1-GiB sampled limits. RSS is sampled child usage, not a hard operating-system allocation cap. `source_before==source_after` across 229 entries; fixed input, runtime, and deadline closures pass.

The continuation starts at SHA `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, `theta=0.332026125873074`, and original native `J=0.0612512070514904`. Parameters (`8871db49…`), fixed problem (`16fa95b5…`), and terminal truth (`b927c1a3…`) remain the same. The control advances through `ab27d917… → a579d422… → 33cb86ca…`; the final theta is `0.3295008210283633` and native J is `0.06124426330405451`. The objective definition is unchanged.

## Saved numerical checks

For each step, recomputing `F²=||[g_mix,Q/0.84]||²`, the linear model `||F+alpha DF||²`, `alpha* = -(F·DF)/||DF||²`, and the endpoint difference from saved arrays reproduces the result record. Each accepted alpha equals `alpha*` to FP64 precision. Both actual J and F² Armijo flags pass.

| Step | Alpha | Model `cos²(F,-DF)` | Model F² decrease | Actual F² decrease | Actual / model | J | F² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000413590482 | 0.00735061 | 0.735061% | 0.735040% | 0.99997 | 0.06124904864 | 0.00519988204 |
| 2 | 0.000501606621 | 0.00612887 | 0.612887% | 0.563970% | 0.92019 | 0.06124644853 | 0.00517055625 |
| 3 | 0.000423705418 | 0.00541132 | 0.541132% | 0.489335% | 0.90428 | 0.06124426330 | 0.00514525492 |

The model predicts each accepted local step well; actual relative F² decreases are modest and decline from 0.735% to 0.489%. The cosine measure is descriptive model geometry for the auxiliary residual, not a new acceptance gate or a convergence criterion. These values justify reporting measured slow progress; they do not alone trigger or qualify a curvature result.

Every move is far below radius `0.05` (about `3.0e-5` to `3.6e-5` in control norm). Theta remains inside `[0,1]`. At all three points, the selected face is zero to roundoff and production face audit passes. One-sided gradient jumps are supported by the face normal (`~0.9e-16` to `1.0e-16` tangent jump against `~8.1e-15` budgets); normal denominators remain resolved near `0.286`. Both side gradients have negative products with the current direction, merit products are negative, and all accepted trials pass side/native objective and finite-gradient checks.

The two 360-stage branch traces agree within each accepted endpoint, and each current branch-pair gate passes. The shared pair signature changes between successive points (`67524657…`, `9a22a185…`, `0351f7b2…`), so the evidence qualifies each paired endpoint and does not establish one unchanged branch signature over the whole continuation.

Across the resumed interval, J decreases `0.0113365%`, F² decreases `1.77786%`, and F norm decreases `0.892918%`. At the final point, `||F||=0.07173043`, tangent-gradient norm is `0.07172799`, mixed-gradient infinity norm is `0.02944709`, and the normal mixture component is `0.00059183`. The remaining residual is almost entirely tangent; stationarity has not been reached. No theta-only cleanup or curvature comparison was run.

## Closure limits

Each accepted trial has a separate fresh repeat that matches control, theta, objective, F², side gradients, traces, face audit, branch pair, source, input, runtime, and deadline. Six HVP-history entries give the correct base control hash, theta, side, phase, operator, scope, and completion status: two per resumed base, none borrowed from PR267. The saved iteration record contains the current direction and both HVP result vectors, and the pinned source computes them at that point/direction. HVP history does not include an independent direction digest; direction-to-product association therefore rests on the pinned runner and per-iteration arrays.

The run preserves `full_smooth_root=false`, `minimum_claim=false`, and `response_claim=false`. Curvature comparison remains a separate conditional follow-up if the research review requests it; a curvature observation at this nonstationary point would not establish a minimum. Theta-only cleanup, coupled response/reanalysis, and independent forecast verification remain outside this run.

## Final documentation cross-audit

I cross-checked `TANGENT_RESUME_FINDINGS_20261009.md`, `TANGENT_RESUME_KG_UPDATE_20261009.json`, `TANGENT_RESUME_RESULT_20261009.json`, the saved-array analyzer, archive manifest, and completed task checklist against the raw child and receipts. Their base/final control identities, J/F²/F norm changes, per-step predicted/actual reductions, alpha/angle values, 54-test and type-check results, one-guard resource data, and 129/105 source/archive pin counts agree with the evidence. The analyzer consumes saved arrays and static face geometry; it does not regenerate FV, gradients, AD, or HVPs.

The findings and checklist leave current curvature comparison, theta-only cleanup, final stationarity/minimum qualification, response/reanalysis, and forecast validation pending; no condition number or curvature result is claimed. The result/raw claims stay bounded to three accepted corrections, with `minimum_claim=false` and `response_claim=false`; the separate findings correctly preserve the historical 8/8 and external 70/100 estimate rather than attributing them to this run. No doc-scope contradiction was found. The remaining provenance limit is unchanged: HVP history has no independent per-call direction digest, so direction-to-product linkage relies on the source-pinned callback and saved per-iteration arrays.
