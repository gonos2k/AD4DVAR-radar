# Tangent-resume final RED source preflight — 2026-10-09

## Disposition

The source-bound resume adapter is ready for the root's single planned guarded launch, subject to the already stated final runtime/result review. Static source and read-only loader checks found no math-kernel, control/theta identity, objective, or hard-policy defect. No child, FV, gradient, HVP, guard, test suite, or full/shared graph run was started here.

## Source and plan identity

The resume plan hashes to `d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10`. A read-only `.venv` call to the adapter's `_load_plan` and `_load_current_base` passes with 129 source pins and 105 archive pins, resolving the exact final raw child `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`, control SHA `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, `theta=0.332026125873074`, and `J=0.0612512070514904`.

The loader ties the run/resource receipts and PR267 source/raw lineage to immutable hashes. It checks the three accepted points as a control/theta chain; each accepted trial is unique and matches its committed point, each point has its own two completed HVP history rows labeled with the correct base hash, theta, side, phase, operator, and scope, and the final repeat agrees with the last trial and top-level last-confirmed closure. The exact PR267 plan remains pinned at `cc9c3a9b…`; it is not edited or reused as the new plan. The pre-resume shared-source snapshot is `e6a56e21…`; the new plan separately pins the current resume source and live corrected shared runner.

The adapter explicitly passes the new base SHA, plan loader, and base loader through the shared runner. The original module constants for the PR266 start remain unchanged. On the resumed path the core compares its loaded control SHA with the injected hash, records the actual base identity, and uses that identity in exception recovery and parent closure. Recovery still classifies a commit by `saved_control_sha != injected_base_sha`, then independently enforces finite 26-vector control, theta in `[0,1]`, integer non-boolean count in `[0,3]`, and consistency between base-versus-committed identity and zero-versus-positive count. A theta-only no-control-move commit remains outside this helper by design.

## Mathematical and policy review

The resume efficiency record now identifies its historical source control `747e5506…` and `theta=0.3288283299901934` as `preceding_step_efficiency_source_*`, separate from the new anchor at `ed106d7b…`. Its normalized model prediction includes the quadratic term:

`predicted_fraction = (-2 α (F·DF) - α² ||DF||²) / ||F||²`.

The saved alpha is the scalar model minimizer, so normalized prediction equals `cos²(F,-DF)=0.0111244865225056`; normalized actual reduction is `0.0111249485876895`. For an accepted alpha clipped below the scalar minimizer, the quadratic expression gives a correspondingly smaller prediction; this remains a descriptive field and is not consumed as a new acceptance or stopping gate.

The shared `tangent_direction`, `tangent_model`, `candidate_alphas`, `search_candidates`, `fresh_final_closure`, and `bounded_continuation` ASTs match the pre-resume source snapshot under `_load_plan`. Changes in the shared runner are base/plan/call-site plumbing, continuation status labels, source tags, and closure identity checks. The resumed plan keeps the same fixed native J, selected face/scale, two-current-side-HVP correction, actual J and F-squared Armijo gates, fresh closure, and resource caps: one guard, at most 3 accepted steps, 6 fresh HVPs, 16 candidates per step, radius `0.05`, 240 seconds internal, 300 seconds outer, and 1 GiB sampled RSS. It adds no curvature calculation, solver change, theta-only path, or `cos²` gate.

The remaining review after the root launch should check the actual resumed child/parent/resource receipt, current-control/theta commit chain, fresh per-point HVP associations, actual nonlinear objective and merit reductions, and the source/input/runtime closure. Curvature comparison remains conditional on the later slow-progress review and requires a separate scope/budget; theta-only cleanup remains a separate final-stage task. Neither can be claimed from this preflight.
