# RED preflight review — paired terminal-E observation (2026-10-12)

## Result

The score computation contract is well-scoped and the earlier parent status gap is fixed: it admits two receipt-closed controls, loads the pinned old-truth fixed case under the new transport, calls the production terminal score exactly twice in start-then-final order, and has no optimizer/candidate/selection path. The parent `_run_summary` now requires full resource-guard success and a closed child receipt, and returns nonzero for a failed summary. This review ran no FV/score/HVP/tests.

## Verified scope and controls

- Consulted the saved Graphify snapshot first: the shared scoped graph has 212 nodes / 400 edges / 17 communities and includes the observer tests; no semantic rebuild is claimed here.
- Final frozen plan SHA `7c2b7f317e90e49a0c1bd9a289529816b8cef91e9d33048c8017b9b884279ae5` and source bundle SHA `2a5eb415af2b8f873d319b45fd443894cba8c4533bc5694b4013ea96f8caf833` match. I recomputed all 209 source and 283 archive pins: no mismatch. Its source tar contains exactly the 209 plan source members with matching hashes; all 205 inherited source and archive pins are unchanged, with four observer/continuous-endpoint additions.
- The plan pins numerical version `minmod_net_flux_fp64_v1`, transport SHA, fixed-case `.pt` and manifest hashes, start `0aafdf80…` and confirmed final `b6470069…` controls, parameters/truth/problem identities, and score profile. `load_endpoint()` is receipt-only for both controls. The observer verifies the fixed-case asset SHA before unpickling and does not call seed preparation or regenerate truth.
- The profile is 26 controls, 13 fixed parameters, 3x4 observations at 0/600/1200 seconds, 18 ten-minute forecast steps to 12000 seconds. The endpoint input identities are checked against the same fixed parameters/truth/problem and archived inputs. Runtime is checked against the old fixed-case manifest and both endpoint receipts.
- `observe_score_pair` validates CPU FP64 shapes, `problem.verification_dbz == truth`, profile, and fixed tensor hashes; it invokes `problem.score` once at start and once at final under `torch.no_grad`, checks finite nonnegative scalar values, and rechecks source/input/runtime/asset identities before/after each score. It writes started/completed progress and does not use scores for selection. No optimizer or candidate evaluations are permitted. The paired score is descriptive for this synthetic same-model truth, not independent forecast skill or response evidence.
- Final preflight evidence records 43 affected tests passing in 29.91 seconds, 18 existing deprecation warnings, six typechecked files with zero diagnostics, and a 17-file Graphify scope (219 nodes / 417 edges). The observer has 15 focused mocked tests covering two-call order, fixed inputs, profile/truth mismatch, finite/nonnegative results, timeout before the second score, and parent guard/child closure failures. This is not a full-suite claim.

## Launch status

The final plan is ready for its separate guarded score-pair launch under the 600-second internal / 660-second outer / 2-GiB policy. The completed score pair remains observation-only evidence. No source or plan was edited by RED.
