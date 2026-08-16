# PR #123 closure checklist

Implementation and executable evidence are tracked separately.  An item is
checked only when the production path and its adversarial regression exist.

## A. Terminal publication atomicity

- [x] Precompute one activation timestamp and activation receipt.
- [x] Commit `published`, `usable`, and the receipt in one SQLite transaction.
- [x] Prove a crash cannot leave a published decision without its receipt.

## B. Immutable learned-input execution context

- [x] Add `BoundNeuralPriorInput` with all five feature channels and digests.
- [x] Remove runner-global active input state.
- [x] Require the bound input for reproduce, JVP, VJP, and adjoint checks.
- [x] Persist the bound input in current P1 linearization artifacts.
- [x] Cover same-dBZ/different-QC, failed inference, concurrency, and fresh restart.

## C. Durable training feature/target archive

- [x] Replace parallel tuples with typed member records.
- [x] Validate actual feature and target archive bytes and member tensors.
- [x] Add signed target derivation lineage and cutoff chronology.
- [x] Reject byte mutation, member/order swaps, missing archives, and leakage.

## D. Operational raw history and ingress QC

- [x] Add signed original/correction/supersession raw-resolution entries.
- [x] Enforce cross-cycle predecessor and transition rules per raw slot.
- [x] Allow signed missing-to-resolved late-arrival correction.
- [x] Enforce finite quality weights in `[0,1]` at raw ingress.

## E. Contract generations and release evidence

- [x] Raise all contracts whose semantics changed and preserve audit loaders.
- [x] Update README, exports, CLI, schema migration, and CI expectations.
- [x] Run focused adversarial tests and static type checking.
- [x] Run the full CPU test suite and subtests.
- [x] Build source/wheel distributions and run installed CLI smoke.
- [x] Audit the complete diff and migrations.
- [x] Open the PR, require every CI check to pass, and merge.

## Verification evidence

- `python .github/scripts/check_basedpyright.py`: 0 errors.
- `python -m pytest -q`: 750 tests and 397 subtests passed.
- `python -m build`: v0.88.0 sdist and wheel built successfully.
- Fresh-venv wheel smoke: `forecast-run-v64` and `nowcast-npz-v70` validated.
- `git diff --check`: clean; schema upgrade tests validate index schema 38.
- PR #123 merged as `710e830e048d142ecafe04a87f4e002455cf8a7e` after all required CI checks passed.
