# FV synthetic-demonstration PR checkpoints

## Included and checked

- [x] Positive spatial FV transport, explicit stage boundaries/support and shared analysis/forecast path.
- [x] Recomputed transport differentiation and legacy-API rejection guards.
- [x] Exact robust-Hessian research response alongside the named approximate GN mode.
- [x] Declared 240×240 local first-order verification; no minimum/FSO eligibility claim.
- [x] Small actual neural observation-error learning, held-out reanalysis and model replay.
- [x] Isolated NN feature/statistics response checked against independent reanalysis FD.
- [x] General boundary/control/trajectory/response tests: 76 passed; demo/holdout tests: 24 passed.
- [x] Product source typecheck: pinned basedpyright 1.39.9, no errors/warnings.
- [x] Test collection excludes historical source archives. Initial 1,409 tests collected without errors; the additional isolated-path regression is subsequently included.
- [x] Preserve local generated HTML and large checkpoint data; ship generators, templates and small evidence.
- [x] GREEN/RED review of numerical scope, trajectory integrity and isolated-path attribution.

## Required before considering general completion

- Full Python 3.12 CI was subsequently requested. Run 35424239984 checks final PR head 0eb2a9d; record its actual conclusion separately from skipped automatic jobs. Deployment remains deferred.
- [x] Fixed missing/detection masks, conservative known-support stencil and sensitive-face refusal; partial-support reanalysis FD includes B=y[0] consistently.
- [x] Declared 240×240 rotation: unchanged stationarity gate, matrix-free adjoint actual residual, one-sided Taylor refinement and matched central reanalysis. Global Hessian conditioning and arbitrary perturbations are not certified.
- [ ] Integrate a background neural prior with its mean/support/residual derivatives; the existing small NN changes observation-error statistics instead.
- [x] Verify bounded persistent observation-error learning with checkpoint resume.
- [x] Prescribed translation, rotation and strain: 32/64/128 grid refinement decreases field and JVP errors; this does not establish general P1 completion.

These are substantive extensions, not permission to loosen gates or reinterpret a
local first-order flag. The user approved larger-token/compute expansion and then
deferred deployment. PR #160 was merged at 2026-09-19 14:33:15 KST/JST as
5d4666c. Merge is not evidence of general FV completion or full CPU success.
See ALGORITHM_EXTENSION_PLAN.md and POST_MERGE_REVIEW_CHECKLIST.md.

## Additional executed evidence

240×240 exact nominal adjoint: maximum gradient 3.84035e-9, actual relative
residual 4.28390e-11, 41 Hessian products, 461.12 s / 1.933 GB sampled RSS.
The first 0.01 dBZ reanalysis failed after reaching gradient L2 1.3584e-8;
its maximum component was not retained. This is not completed FSOI validation.
Evidence: rotation240_response_18.json and rotation240_impacts_large_step_failure.json.

External learned-mean composition (fixed nonzero theta) now has chain-rule and
finite-reanalysis evidence, with exact checkpoint reload. Its held-out score
worsened by 2.16e-16, so no improvement claim is made. Typed prior/legacy FSOI
integration remains separate and incomplete. See FV_EXTERNAL_BACKGROUND_PHASE1_RESULTS.md.


Current small-growth arithmetic supersedes the historical nominal numbers above:
max gradient 5.73043e-10, adjoint residual 4.28392e-11. Positive steps 0.001/0.0005
have remainder ratio 3.999751567; matched central slope at 0.0005 differs from
the adjoint by 4.39743e-5 relative. The positive/negative endpoints have maximum
gradients 1.42878e-10/2.93110e-10. Original HTML now displays actual maps and
reanalysis values; original embedded forecast arrays are unchanged. These are
case-specific local derivative results, not general typed-prior completion.

CI follow-up: attempt 2 of run 35411573920 was found running at 13:15 JST; cancellation was requested and GitHub confirmed completed/cancelled. No new full CI was started.

2026-09-19 CI policy: full CPU tests and packaging are manual-only during phase 1.
Manual runs use Python 3.12 only; the Python 3.10 option and version matrix were
removed at the user's request. Automatic runs retain only the lightweight UI check.
The workflow passed actionlint; no CI or numerical suite was launched for this
configuration change. The workflow changes were pushed and merged in PR #160;
the user subsequently requested the manual run recorded above.
