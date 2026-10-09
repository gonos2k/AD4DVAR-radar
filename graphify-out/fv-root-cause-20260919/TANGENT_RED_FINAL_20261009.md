# Tangent continuation RED final audit — 2026-10-09

Disposition: **the guarded tangent continuation completed its declared three-step cap with closed execution and per-step endpoint receipts.** This supports only the bounded correction result below. It does not establish a full smooth root, a nonsmooth local minimum, or an observation response.

## Evidence identity and execution

The frozen plan SHA-256 is `cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69`. The raw child SHA-256 is `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`; its gzip archive decompresses byte-for-byte to that raw, and the parent `child_sha256` and nested resource receipt match. The run is `phase=finished`, `execution_status=completed`, `numerical_status=tangent_continuation_cap_reached`, with exit code 0, no resource termination or monitor error, and no SIGTERM. It used 76.515 seconds and sampled peak RSS of 570,179,584 bytes, inside the declared 300-second and 1 GiB limits.

The plan starts from accepted requalification control hash `6b29dacd…` and `theta=0.3504554198817298`. The runner freshly reconstructs that producer endpoint and matches its saved input/runtime receipt before continuing (`fv_point_3h_tangent_continuation.py:440-507, 549-583`). The three committed control hashes form a continuous chain:

`6b29dacd… → 20ae0609… → 747e5506… → ed106d7b…`

The final raw control hash is `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, with `theta=0.332026125873074`. Parameters, terminal truth, and archived problem identity are unchanged; input control identity advances to the final control. The child reports source, fixed-input, runtime, and deadline closure true.

## Per-step closure and numerical record

All three iterations are accepted, with one candidate evaluated per iteration. Each trial passes actual-J Armijo, squared-residual Armijo, selected-face audit, paired branch gate, native/side objective parity, finite one-sided gradients, and the independent final repeat. The repeat checks the same proposal control and theta; both fresh side-gradient vectors match the proposal, and its branch trace matches the proposal trace (`fresh_final_closure`, `:257-282`; runner repeat, `:687-720`).

The selected face remains zero to roundoff at each point. Both one-sided 360-stage traces agree at each accepted endpoint. The first endpoint pair has signature `04e392ad…`; the second and third have signature `67524657…`. The trace therefore changes between accepted endpoints; this receipt establishes each endpoint’s paired trace, not one unchanged smooth branch across the whole path.

There are six completed HVP calls: exactly one minus-side and one plus-side product at each of the three base-control hashes, with the matching base theta and `selected_face_extension` operator. The source-pinned callback computes each JVP from the current control and the current iteration direction (`:633-647`); each iteration stores that direction and the two products. The HVP history does not separately store a hash of its input direction, so the direction association is supported by the pinned runner source and iteration records rather than an independent direction-hash field. No HVP is reused across committed points.

Saved-array arithmetic reports these net changes from the producer base to the final point:

- Native objective `J`: `0.0612637058103 → 0.0612512070515` (decrease `0.0204%`).
- Ambient residual norm `||F||`: `0.0794631 → 0.0723767` (decrease `8.92%`); squared norm decreases `17.04%`.
- Final mixed-gradient infinity norm: `0.02979`; final residual norm remains `0.07238`.

All three actual moves are below the `0.05` radius cap. The stored linearized-residual relative errors are about `1.42%`, `1.03%`, and `0.00235%`, respectively; these describe the local model fit at these steps, not a convergence bound. The final ambient residual norm is `0.07238` and mixed-gradient infinity norm is `0.02979`; neither establishes stationarity. The original smooth-root criterion is separate and was not evaluated here. This run defines no final nonsmooth stationarity tolerance or minimum claim.

## Verification scope and limits

The saved focused test log reports **45 passed** with 18 TorchScript deprecation warnings. The saved type-check log reports **0 errors, 0 warnings, 0 notes**. These logs were inspected, not rerun for this audit. The saved `TANGENT_ANALYZE_20261009.py` explicitly uses saved arrays and static face geometry only; its scope excludes objective, FV, gradient, and HVP regeneration. I independently checked the raw/gzip/parent/resource digest linkage and the three-point hash/HVP count chain from the saved receipts. No production rerun or source change was made for this audit.

This continuation is a bounded model-guided tangent-gradient correction along the selected `Q=0` face, using fresh two-sided HVPs and actual endpoint Armijo/branch/closure checks. It is neither a Newton solve nor a root certificate. The raw explicitly leaves `full_smooth_root=false`, `minimum_claim=false`, and `response_claim=false`. Historical phase-1 work remains 8/8 (100% on that historical scope); the current FV external weighted milestone remains 70/100. Curvature, unconstrained nonsmooth minimum conditions, adjoint/VJP, reanalysis response, and independent forecast verification remain outside this run.

## Documentation consistency review

The final findings, saved result, analyzer, KG update, checklist, and GREEN final agree on the plan/raw identity, three accepted steps, six HVPs, resource outcome, endpoint hashes, objective/residual reductions, and scope limitations. The result/analyzer are derived files rather than entries in the run archive; their claims were cross-checked against the pinned raw and producer base. The checklist correctly leaves final PR/CI/head integration pending. The wording above clarifies that `F` is the ambient mixed-gradient-plus-face residual, not the original smooth-root residual; the 70/100 figure is the current external FV milestone, while 8/8 is the historical phase-1 scope.

The older `TANGENT_GREEN_LOADER_SUPPLEMENT_20261009.md` records preflight plan SHA `559a840c…` and an initial/final deadline-check gap. It predates the final plan (`cc9c3a9b…`) and launch; the final source adds checks before those observations. Treat that supplement as historical preflight evidence, not the final loader disposition. With this distinction, no remaining actionable contradiction was found in the reviewed deliverables.
