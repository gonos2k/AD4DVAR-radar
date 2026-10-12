# PR301 follow-up: rank-deficient residual fix and radius qualification

## P2 closed

The repository's pinned PR301 solver reproduced the review's singular-matrix error under local PyTorch2.13.0: independent D=I, V=ones, G=(-1,-1), Delta=.8. `REPRODUCE_BEFORE.json` identifies the original source bytes from the immutable PR301 source bundle, not a reconstruction from memory. The P2 reproduction and repair used solver-only calculations. No PR301 candidate search or final candidate repeat was rerun; fresh7839 native/P2 and derivative preparation was performed later only for the separately frozen conditional radius protocol.

The fix reveals rank in V with SVD before deciding radius activity. Full-rank lambda=0 uses QR rather than a normal-equation determinant; a rank-one model V=u w^T uses the least-squares solution set w^T beta=-u^T G and its minimum control-movement representative. Control QR applies Q=D^T D without squaring D's condition number. An interior nonnegative representative is accepted with lambda=0; an active rank-one solution is scaled to the radius; when the metric representative misses the positive orthant, the existing two axis candidates cover its constrained boundary. V=0 returns the zero candidate with an explicit diagnostic. No ridge, candidate tolerance, objective, prior, observations or transport change was made.

The rank decision has finite-precision SVD resolution, and coefficient derivatives are checked on fixed-rank/active-set regions. We make no derivative claim across rank or active-set transitions. The fix includes exact review cases, the smooth J=.5(x+y-1)^2 gradient/Hessian case, live-shaped26/27 padded inputs, control-metric vs Euclidean pseudoinverse, active rank1 and orthant boundary cases, V=0 finite gradients, near-rank samples, and fixed-rank active/inactive gradchecks.

## Cached d19e KKT result

The offline audit admits exact PR301 source and raw receipt bytes and never calls FV/HVP/score. Its D/V/G are explicitly at d19e, not the confirmed7839 endpoint. Archived V is full rank (singular values about113.3936,1.58558), so the P2 did not occur on this stored model. At the original radius1.8757177916958825e-5, the gradient axis has resolved KKT margins lambda=9417.50477, mu1=+0.06984030. This independently supports the saved zero-GN selection.

The inactive-axis multiplier crosses zero at2.538864637100271e-5 =1.353543x the base radius, before the gradient-axis radius multiplier reaches zero at3.249526221387056e-5. The offline sweep introduces a positive GN coefficient beyond that threshold; at1.5x, the predicted added R reduction is only about8.67e-8 (3.4e-5 relative to the gradient model). These are local model facts, not finite candidate or acceleration results. See `KKT_FINDINGS.md` and `d19e_axis_radius_audit.json` for margins, roundoff budgets and sweep.

## Fresh7839 preparation declined expansion within the cap

Before any physical candidate, a separate plan froze one600s internal/660s guard/2GiB budget, four fresh HVPs,24 GN rows/one solve, at most6 candidates per arm/12 total, two repeats and one commit. The candidate cap was not enlarged. The search-length rule allowed one current-KKT-selected radius10% past resolved GN entry, with a hard2x-base limit. It required current axis KKT sign resolution, future entry in its valid interval, a resolved positive GN coefficient and added model reduction. Original actual J/R, chart displacement, theta, side gradients, branch-pair, repeat and full-G gates remain inherited from the shared runner. The two allowed inherited source transitions are the repaired solver and the explicit experiment configuration that reuses this runner.

The archived7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a point was admitted from PR301 receipt/source bundles and both arm closures. Its native/P2/release/minimum-mixture checks were rerun, then both directions and all four HVPs were regenerated at7839. No d19e operator was reused.

Fresh7839 V is full rank. At the original radius its gradient-axis KKT is resolved: lambda=11661.79912, mu1=+0.10490553. Its GN entry is5.001297266163082e-5 =2.66634x the base radius, exceeding the frozen3.751435583391765e-5 cap. At that cap the independently audited model still chooses the gradient axis, with mu1 about+0.04194979 and zero added subspace gain.

The guard therefore completed with `no_resolved_subspace_entry_within_frozen_radius_cap`: **zero FV candidate evaluations, zero final repeats, zero optimizer commits and zero score calls**. No expanded-radius finite candidate was attempted. The previous7839 control/theta/full-G was preserved. This is a fresh preparation/KKT result, not a successful finite expansion or a new J/R improvement. The cap was not raised after seeing the outcome.

Guarded wall79.7018s; child77.8643s; sampled peak RSS1,096,499,200bytes (about1.021GiB). Fresh preparation counts: HVP4/4, row VJP24, one12x12 GN solve. Source, fixed input and runtime closed; all13 offline closure checks pass. E was not evaluated or used for selection.

## Verification and remaining limits

67 affected tests passed in24.43s with18 existing TorchScript deprecation warnings. Basic Pyright on the two changed core modules and three new scripts reports0 errors/warnings, using an explicit work-directory import path. GREEN/RED independently checked the source/policy seals, archived endpoint, rank mathematics and final stored-vector KKT/closure. Their final files are `GREEN_FINAL_EXPANSION.md` / `RED_FINAL_EXPANSION.md`; the earlier static RED preflight is superseded by `RED_LAUNCH_CLEARANCE.md`. The source bundle contains222 pinned files;291 inherited archives were verified. Graphify reused PR301's cached structure and refreshed the changed-source AST supplement.

The old confirmed J/R decrease and b647→d19e observation remain valid and unchanged. No new forecast score or independent forecast validation occurred. No entire repository/remote CPU/package FV baseline was repeated. Production HVPs were generated once by the guarded preparation; reviewers checked stored arithmetic without regenerating them.

Historical weighted completion stays about70%. The7839 residual norm0.0504848 and max mixed component0.0253996 still do not certify an eligible minimum; exact curvature, original-problem adjoint/VJP, p±h reoptimization, reanalysis and independent future validation remain incomplete. The corrected search-tool derivatives do not substitute for the assimilation adjoint.

Further work should use the archived current-point operators and decide an explicit path policy before allocating another comparison. Repeating a <=2x gradient/GN comparison at7839 would add preparation cost without a predicted subspace benefit. Raising the cap solely to pass this outcome is not supported by this closed experiment. The current KKT operators are qualified only at7839; any later point needs fresh qualification.
