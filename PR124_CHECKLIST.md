# PR #124 closure checklist

Implementation and executable evidence are tracked separately. An item is
checked only when the production path, migration path, and adversarial
regression are all present.

## A. Recoverable analysis-input provenance

- [x] Replace the provenance `usable` handoff with explicit prepared/active/expired state.
- [x] Make holdout and operational provenance append idempotently resume prepared rows.
- [x] Revalidate NPZ, metadata, checksums, raw receipts, processor signature, and current trust before activation.
- [x] Reconcile final content-addressed directories left behind before their DB row.
- [x] Provide a public startup reconciliation API for prepared provenance.
- [x] Atomically bind each prepared row to a root-approved ledger preparation receipt over the exact ledger instance and registry/history side-effect set.
- [x] Derive the processor from the registered plan and the receipt signer from root-owned ledger trust; reject self-authorized rows even after a forged SQL activation.
- [x] Reject direct-SQL recovery that omits global-registry memberships or operational raw-history entries.
- [x] Cover crashes after rename, row/history commit, trust read, activation start, and activation commit.

## B. Hard publication chronology

- [x] Name the signed time as logical commit authorization; do not claim a cryptographic SQLite fsync timestamp.
- [x] Acquire the SQLite write lock before fixing the activation authorization timestamp.
- [x] Stage the activation receipt while unusable, then atomically bind a writer-lock-held authorization receipt to the terminal `published/usable` transition.
- [x] Enforce a signed publication guard interval and fail closed across delayed locks/signers/fsync.
- [x] Cover a competing write lock that crosses the hard publication boundary.

## C. Training target semantics and sharded archives

- [x] Add target v3 value/mask/quality/unit/shape/source-plan lineage.
- [x] Reject empty, non-finite, schema-incompatible, or out-of-window targets.
- [x] Bind target identity to training cases and reject holdout verification-target reuse.
- [x] Require a separately plan-pinned target-source authority receipt so an approved analysis processor cannot relabel holdout truth.
- [x] Replace inline base64 archives with immutable, content-addressed NPZ shards.
- [x] Single-open snapshot-check shard bytes, NPY headers, member counts, sizes, dtypes, shapes, and tensor digests.
- [x] Bind the validated tensor snapshot-set digest to signed training start/completion execution lineage.
- [x] Persist and validate actual normalization statistics bytes.
- [x] Cover byte mutation, member/order swaps, missing shards, cutoff leakage, and oversized inputs.

## D. Operational raw-resolution history

- [x] Bind correction authority to the provenance plan's approved processor authority.
- [x] Canonically reconstruct and reverify every current entry and composite receipt at append.
- [x] Add a schema-38 legacy anchor for pre-history operational resolutions.
- [x] Rebuild every legacy anchor from rehashed provenance bytes before record/load/predecessor use.
- [x] Cover post-construction mutation, alternate approved authority, and legacy migration.

## E. Bound-input hot path

- [x] Add a validated private snapshot handle for repeated derivative operations.
- [x] Perform full cryptographic validation at API/artifact boundaries only.
- [x] Isolate caller and private-field `.data` mutations with full rehash plus a per-call derivative clone.
- [x] Revalidate full digests at FSO/P1 operation completion.
- [x] Preserve fresh-process P1 save/load and same-dBZ/different-QC behavior.

## F. Contract generations and release evidence

- [x] Raise every contract whose semantics changed and preserve explicit audit loaders.
- [x] Update package version, README, exports, CLI, schema migration, and CI expectations.
- [x] Raise the vulnerable `cryptography` floor to `>=50.0.0` and audit the pinned runtime closure.
- [x] Commit Python 3.10/3.12 Linux CPU runtime and CI/build closures with exact versions and distribution hashes.
- [x] Require lock-only installs, lock synchronization, `pip check`, and strict vulnerability audits in every required CI path.
- [x] Run installer, auditor, build, type-check, and test modules in isolated mode; canary-check that repository modules cannot shadow `pip-audit`.
- [x] Require runtime/CI hash-set equality, cross-Python direct-pin equality, and a CUDA/`nvidia-*`/`triton`-free closure.
- [x] Run focused adversarial tests and static type checking on the final tree.
- [x] Run the full CPU test suite and subtests on the final tree.
- [x] Build source/wheel distributions and run installed CLI smoke on the final tree.
- [x] Audit the complete diff and migrations.
- [ ] Open the PR, require every CI check to pass, and merge.

## Verification evidence

- `python .github/scripts/check_basedpyright.py`: 0 errors.
- `PYTHONPATH=src python -m pytest -q -x`: 756 passed and 400 subtests
  passed under Python 3.10, Torch 2.13.0, and cryptography 50.0.0.
- Exact installed runtime closure audit with `pip-audit --strict --disable-pip
  --no-deps`: no known vulnerabilities.
- Four committed Python 3.10/3.12 Linux CPU runtime/CI locks pass
  `check_dependency_locks.py` and strict no-resolution `pip-audit`: no known
  vulnerabilities.
- Clean `linux/amd64` Python 3.10 and 3.12 containers installed the hashed CI
  closures from scratch; `pip check`, CPU-only validation, lock validation, and
  isolated strict `pip-audit` passed.
- `python -m build`: sdist and wheel built successfully.
- Installed-wheel CLI smoke: package `0.89.0`, output `nowcast-npz-v71`,
  forecast run `forecast-run-v65`, finite-mask and output-shape checks passed.
- `git diff --check`: clean.
