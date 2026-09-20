# PR170 follow-up: explicit refinement from saved GN to full response

Baseline: merged PR170 `b41057bb`. Scope is a serial, bounded research workflow.
The input is actual archived `product_gn.control`, not the polished top-level
control. No fresh GN optimization or perturbed observation reanalysis is run.

## Equations and decisions

The unchanged response condition is `max(abs(grad_c J)) < 1e-10` at a valid
54-stage minmod branch. A qualified input proceeds to `compute_local_response`.
An unqualified input either returns an ineligible status or, with an explicitly
provided refiner, calls the existing small dense Newton oracle. Its merit is
one half the squared gradient norm, with undamped positive-Hessian checks and
branch checks. The refined state is reassessed before the matrix-free adjoint.

The refiner is explicit: this does not silently convert the generic matrix-free
response into a dense solver. Its existing 32-control oracle limit and 8-step
limit remain. Dense refinement and matrix-free response costs are separate.

The score uses the old oracle forecast plus the old fixed pattern, frozen before
refinement; its verification digest is recorded. Refining does not redefine the
score target. Full gradients retain `E_p - J_cp.T @ lambda` and direct/indirect
terms separately. The fixed support, boundary, precision, and research-only
eligibility limits remain unchanged.

Failure preserves the original analysis control and emits no response. This is
not a claim that an unqualified sensitivity invalidates the original forecast.
Nor is it a guarantee that arbitrary new GN results can be refined successfully.

## Evidence scope

`fv_gn_response_probe.py` records both controls, before/after objective, gradient
and full branch signatures, accepted-signature Newton candidate checks, timings,
actual adjoint residual, full parameter gradients, and source/input identities.
Checks inspect discrete trial points; continuous line segments are not certified.

The archived inverse fixture and the current fixture have distinct source hashes;
previous changes include failure-recording around perturbed refinement. Both
identities are preserved. This execution uses the current fixture and does not
claim byte-identical replay of the historical producer.

General minmod FSOI, new-input GN convergence, typed-prior learning, finite impact,
finite-path certification, and whole-chain D7 remain open.

## Execution provenance and verification

The first integration result is preserved as `gn_refined_response_initial.json`.
It found the same qualified control and full gradient as the old oracle. However,
a failure-path edit to the wrapper overlapped the run; source hashes captured at
completion alone cannot certify its loaded source. It is preliminary evidence.
The final `gn_refined_response.json` is a separate run with frozen final source,
used for publishing. This justified repeat is not an additional independent case.
The measured-workflow snapshot matches the final source, not an attested snapshot
of the initial process's imported module.

76 distinct focused tests passed, including initial-assessment refusal, absent
refiner, refinement failure/bad output/branch rejection, response failure and
qualified success. Programming TypeErrors are not broadly swallowed; the explicit
non-tensor refiner-output contract is a ValueError refusal. Pinned basedpyright:
0 errors, 0 warnings, 0 notes. Full CPU/package CI was not requested.

Graphify code-only update: 6,679 nodes, 65,065 edges, 206 communities. No semantic
LLM extraction. General limitations above remain unchanged.

## Final frozen-source execution

- Status: eligible; refinement explicitly used: True.
- Gradient: 1.334876035182e-04 -> 6.962040319247e-12.
- Objective: 0.021720900392662709 -> 0.021720900386241804.
- Actual transpose residual: 9.888163804832e-11.
- Full gradient relative difference to old oracle: 0.0; control distance: 0.0.
- Assessment 1.207s, refinement 62.779s, response 24.110s; internal full producer 88.376s.
- Two successful candidate signature checks; all 54-stage choices/face signs retained.
- GREEN/RED final review: no blocker. Peak memory not measured in this run.

Aside desktop inspection confirmed the final integration panel in the original
demo HTML; animation arrays remain unchanged. Screenshot: `gn_response_desktop.png`.
No new mobile check is claimed.
