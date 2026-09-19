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

- Deferred by user: CI/deployment checks follow algorithm completion. Cancelled run 35411573920; local focused numerical checks do not establish full CI success.
- [x] Fixed missing/detection masks, conservative known-support stencil and sensitive-face refusal; partial-support reanalysis FD includes B=y[0] consistently.
- [ ] Tighten the 240×240 state to the exact-response gradient gate and verify large-grid adjoint conditioning, residual and perturbation response.
- [ ] Integrate a background neural prior with its mean/support/residual derivatives; the existing small NN changes observation-error statistics instead.
- [x] Verify bounded persistent observation-error learning with checkpoint resume.
- [x] Prescribed translation, rotation and strain: 32/64/128 grid refinement decreases field and JVP errors; this does not establish general P1 completion.

These are substantive extensions, not permission to loosen gates or reinterpret a
local first-order flag. The user approved larger-token/compute expansion and then
deferred CI/deployment until algorithms are finished. PR #160 is open (not draft);
preserve its existing state while completing numerical work. Merge/close remains
separate from local algorithm verification. See ALGORITHM_EXTENSION_PLAN.md.

## Additional executed evidence

240×240 exact nominal adjoint: maximum gradient 3.84035e-9, actual relative
residual 4.28390e-11, 41 Hessian products, 461.12 s / 1.933 GB sampled RSS.
The first 0.01 dBZ reanalysis failed after reaching gradient L2 1.3584e-8;
its maximum component was not retained. This is not completed FSOI validation.
Evidence: rotation240_response_18.json and rotation240_impacts_large_step_failure.json.
