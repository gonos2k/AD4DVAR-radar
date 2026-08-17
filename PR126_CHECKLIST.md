# PR #126 closure checklist

Implementation and executable evidence are tracked separately. An item is
checked only after the production path, durable/restart path, and adversarial
regression all agree. CPU automatic promotion remains closed until section A
is complete.

## A. Holdout verification-target identity

- [x] Replace the three loose verification-target fields with a typed identity artifact covering source, UTC valid time, value, valid-mask, quality, QC/censor policy, and verification bundle.
- [x] Construct the identity from the exact `PriorUncertaintyTargetPlan` and target tensors used for scoring.
- [x] Require exact case/evaluation/scoring identity equality before any metric is computed or committed.
- [x] Use the typed identity for classifier and candidate training/holdout overlap checks.
- [x] Reject source, time, value, mask, and quality substitutions after all outer digests are recomputed.

## B. Event-group split and normalization lineage

- [x] Bind every training dataset member and target derivation to the physical event recorded for that case in the signed training catalog.
- [x] Require every physical event to belong to exactly one of train, validation, or test.
- [x] Add a typed normalization derivation over the ordered train-only members, channel definitions, algorithm, statistics tensors, and output shard.
- [x] Bind normalization derivation identity to dataset, training execution, start, and completion receipts.
- [x] Reject case-event relabeling, cross-split event leakage, and validation/test normalization inputs.

## C. Target execution semantics

- [x] Revalidate loaded target, valid-mask, and quality shard tensors independently of producer constructors.
- [x] Require finite nonempty targets, boolean masks, `quality[~mask] == 0`, and positive effective weight.
- [x] Canonicalize invalid target values or enforce mask-first access through the product-owned training loss.
- [x] Define and test the exact weighted-loss denominator and zero-weight failure behavior.

## D. Streaming and portable training datasets

- [x] Add dataset-level limits for shard count, archive bytes, expanded bytes, and member count.
- [x] Replace eager all-shard materialization with a validated shard/member iterator for training execution.
- [x] Bind execution to ordered shard snapshots without retaining the full dataset twice in memory.
- [x] Replace absolute paths in semantic identity with approved artifact-store ID plus content-addressed relative paths.
- [x] Preserve restart validation and reject symlinks, path traversal, path swaps, budget overflow, and shard-order changes.

## E. Target-source trust and derivative execution

- [x] Add a root-signed target-source trust store with key epoch, validity, revocation, source-contract, and radar/product scopes.
- [x] Revalidate the current trust-store digest at target receipt, dataset seal, training start/completion, promotion, deployment issuance, and durable forecast-load boundaries.
- [x] Preserve full bound-input cryptographic validation at external operation boundaries.
- [x] Use a sealed inner derivative handle between start/completion checks without changing the `.data` mutation threat model.
- [x] Benchmark or regression-check that repeated JVP/VJP no longer rebuilds and hashes the full five-channel input each primitive call.

## F. Deployment bundle and contract generations

- [x] Publish the wheel digest, platform/Python lock, SBOM, vulnerability-audit result, and installation attestation as one externally verifiable signed bundle; CI output is explicitly candidate-only.
- [x] Document and smoke-test `--require-hashes` dependency installation followed by `--no-deps` wheel installation.
- [x] Raise every changed contract generation and preserve explicit audit-only loaders for prior generations.
- [x] Update package version, README, exports, CLI, schema migration, and CI generation expectations.

## G. Verification and delivery

- [x] Run focused adversarial tests for every checklist invariant.
- [x] Run basedpyright with zero errors.
- [x] Run the full CPU test suite and subtests.
- [x] Build source/wheel distributions and run isolated installed-CLI smoke.
- [x] Run dependency lock synchronization, CPU-only checks, and strict vulnerability audits.
- [x] Complete an independent security review and resolve every critical/high blocker.
- [x] Audit the final diff and migrations.
- [ ] Open the PR, require every required CI check to pass, and merge.

## Verification evidence

- Full CPU suite: `757 passed`, `405 subtests passed`, `18 warnings`,
  `0 failed` in 34m25s.
- Focused deployment-bundle and current deployment-artifact tests passed,
  including signer relabeling, payload mutation, symlink-key, current target
  trust revocation, and trust-store change-during-load attacks.
- basedpyright: `0 errors`, `6561 warnings`, `0 notes`.
- Dependency locks: all four Python 3.10/3.12 Linux CPU runtime/CI closures
  are synchronized; strict `pip-audit==2.10.1` found no known
  vulnerabilities in any closure.
- Build/smoke: source distribution and `advar_radar_nowcast-0.90.0` wheel
  built with the exact build-tool pins; the wheel-loaded CLI produced a
  `nowcast-npz-v72` artifact carrying `forecast-run-v66`.
- Independent security review: no remaining critical/high code blocker;
  deployable bundles retain the documented protected external signer and
  root-owned staging trust boundary.
