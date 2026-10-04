# Receipt-bound resumable Hessian diagnostics

Status: implementation/authoring verification in progress. The previous proposal remains a historical proposal. This plan keeps the original full-control assimilation objective J, controls, parameters, prior and physical/time/boundary definitions; it does not optimize or issue a response.

## Implemented contract to verify

- Fixed26 standardized original controls,20 initial-field +6 dynamics; finite CPU FP64 p vector (13 for the actual profile), supplied fresh26-gradient.
- Deterministic basis columns H e_i, i=0..25, produced by the actual gradient-JVP of J. The caller's receipt provider must truthfully bind the passed callable, its closure/data/source, point, runtime, coordinates and branch. It is not an independent authenticator of arbitrary callables.
- Validate actualc/p/g records and current receipt before/after each product. Verify direction argument remains the canonical basis; shape/dtype/device/finiteness and byte hashes of every stored direction/product.
- Atomic per-product commit, contiguous column ledger, exclusive checkpoint writer. A clean Python exit releases only its own writer lock; an unclean stale claim is not automatically removed. Corruption or changed query identity refuses without rewriting prior valid data.
- Immutable maximum attempts and seconds per kernel attempt, full cap reserved before products. Resume cannot increase either. The kernel ledger counts kernel invocations; external guarded launches and initial case/branch/J/g work must be budgeted separately.
- Before26: partial columns only. At26: matrix_complete_audit_pending until full symmetry/eigensystem and independent27th minimum-eigenvector HVP agree. Record the latter as a minimum-eigenpair comparison, not independent verification of all matrix elements. Reuse the original block/Schur diagnostics with the actual nonzero field gradient. No root/minimum/forecast/response certification.
- New optional checkpoint CLI uses --guarded. Parent forwards an internal checkpoint-child marker, fresh attempt output plus shared separate checkpoint directory, and the unchanged whole-invocation deadline. Existing noncheckpoint behavior and historical fresh_hessian remain.

## Dependency scope

The operator is J_cc, not future-score E_cc. New dependency tests use real point objective/forecast/score methods and valid dataclasses with differentiable trajectory stubs: future-only verification/boundary/lead changes leave J/g/oneHVP equal while score changes. This is a wiring proof, not actual FV verification.

The first actual profile conservatively binds the full fixed case and its3600-stage branch, including future qualification evidence. It does not yet claim checkpoint reuse after score-only changes. Splitting objective-only and score/forecast receipts is a later optimization after dependency validation; no existing strict qualification is silently removed.

## Synthetic verification before execution

7/12/7 interrupted column assembly must exactly match uninterrupted data without recomputing saved columns; after26 but before27, retain audit_pending and resume one new product only. Test identity/source/p changes, product/direction mutation, checksum/row corruption, malformed or incomplete writes, stale writer lock, cached-final validation, wrong-query byte preservation and correct-query reuse, attempt/time policy increase refusal. Test guarded wiring/partial counts, source receipts, public/internal CLI use and distinct output/checkpoint folders. Final GREEN/RED review is required.

## Separate actual3h execution envelope

Only after frozen source/test reviews and preflight: same cd6b5693 endpoint in coupled_original_j_continuation_attempt1 raw SHA3e2e1e247be0de1c6b5e27d4c835dbbe6b456f7a0c0dc2e0e90ac18d71fd8385; fixed p SHA8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed. Original J=.08389718093130327, g_inf=1.0588168237438937.

At most3 serial guarded launches, each240s cooperative internal /300s outer wall /sampled1GiB at.25s. External cumulative allowance900s outer wall and kernel reservations at most720s; every guarded launch counts even if it fails before kernel entry. A fixed experiment coordinator must record all launch resource sidecars and never exceed3 or increase caps. Each attempt gets a new immutable output directory; one shared checkpoint directory. The first cooperative partial budget outcome may proceed to the next planned resume. Stop on completion, corruption/source/input/runtime failure, unsupported numerical qualification, monitor error or external termination; no ad-hoc retries/new seed/cap increase.

Reconstruct current J/g/branch per attempt under the original strict gates. Reuse only validated columns from this new experiment, never the unrecorded products from the old240s refusal. No code editing during numerical execution. Preserve raw inputs, old attempts and all64 original numerical source hashes. Record completed stored products and counts separately from attempted/unrecorded work; no full curvature until matrix/eigenpair/Schur qualify. A complete current-point curvature diagnostic does not close stationarity or meteorological performance.
