# PR #122 closure checklist

This checklist tracks implementation and executable evidence separately.  A
box is checked only when the production path and its focused regression test
both exist.

## A. Durable restart authorization

- [x] Add `OperationalDecisionActivationReceipt-v1` after terminal publication.
- [x] Bind `published`, `usable`, publication commit, activation commit, and
  the committed chain root in the signed receipt.
- [x] Require the activation receipt in current offline decision validation.
- [x] Reject pre-activation publication receipts during `forecast-run-v63`
  loading.
- [x] Make `forecast-run-v62` and deployment lineage v13 audit-only.

## B. Exact decision-to-run binding

- [x] Compare full-analysis input and analysis-derivation digests.
- [x] Compare promotion, classifier, policy, selection, and selected-prior
  digests.
- [x] Require the current typed input plan for deployed artifacts.
- [x] Add mixed-decision/mixed-run adversarial tests.

## C. General operational provenance

- [x] Add a provenance plan independent of promotion holdout families.
- [x] Add operational raw-volume resolution and durable provenance append APIs.
- [x] Exercise a post-promotion future cycle without holdout consumption.

## D. Recoverable ledger state machines

- [x] Resume retained publication payloads after the decision deadline when
  they were durably committed before it.
- [x] Enforce the publication-time upper bound.
- [x] Reconcile prepared raw-trust activations idempotently.
- [x] Verify bytes and canonical preimages before activation recovery.
- [x] Cover post-commit crash and retry paths.

## E. Learned-input and training semantics

- [x] Define canonical dBZ, QC mask, quality, normalized observation error, and
  source-availability feature channels.
- [x] Bind classifier and prior graphs to the five-channel feature contract.
- [x] Persist the learned-input feature digest in analysis provenance and
  `forecast-run-v63`.
- [x] Recompute routing evidence from all committed feature inputs.
- [x] Add member feature/target tensors, order, weights, split, augmentation,
  normalization, and archive digests to training dataset lineage.
- [x] Raise semantic replay, candidate, holdout, policy, and promotion evidence
  generations; retain prior generations for audit only.

## F. Observation availability and scope

- [x] Represent a missing radar acquisition with a signed typed receipt.
- [x] Allow complete mosaic outages without synthetic all-invalid raw volumes.
- [x] Keep native polar-volume independence explicitly outside the grid-product
  certification scope until an upstream acquisition identity exists.

## G. Release evidence

- [x] Run formatting and static type checking (`basedpyright`: 0 errors).
- [x] Run the full CPU test suite and subtests (747 tests and 393 subtests).
- [x] Build source and wheel distributions for v0.87.0.
- [x] Install the wheel and smoke-test `forecast-run-v63` / `nowcast-npz-v69`.
- [x] Audit the complete diff and legacy migrations.
- [ ] Open the PR, wait for every required check, and merge only on all-success.
