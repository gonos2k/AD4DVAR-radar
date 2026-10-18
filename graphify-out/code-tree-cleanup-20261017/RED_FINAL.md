# RED final review — source tree cleanup

## Change scope and semantic risk

The cleanup is narrow. Four modules (`ensemble_sensitivity.py`, `intervention.py`, `promotion.py`, `sensitivity.py`) now import the existing `advar._digest.validate_sha256_digest` implementation under their prior private names. The old validators had the same predicate and `ValueError` text; `DIGEST_EQUIVALENCE.json` records 14 return/error cases per module with exact return and exception type/message equality. The private aliases remain available at their original module paths. This preserves call behavior, although Python function metadata such as `__module__` now points to the shared helper; I found no code path relying on those private function metadata fields.

The repeated `validate_resolved_source_coverage_artifact` import in `ledger.py` and repeated `Path` import in the model-guided test were removed. Neither changes runtime behavior or test setup. The historical digest validator is still the single implementation in `_digest.py`; no public package exports or stored fixtures were rewritten.

The event-controller cleanup removes the old stage-fixed event metadata, bisection, normal, retraction, and two-face-only terminal helpers from the same research runner after replacing their call sites with full-trace event handling. `LIVE_OPERATION_EQUIVALENCE.json` reports exact AST equality for retained budget, AD, event, Newton, retraction, qualification, and runner operations, except for the documented removal of legacy stage-47/48 gap metadata from `Execution.value`. The replacement tests exercise arbitrary event identities, event unions, release/maintain decisions, path truncation, and terminal gates. No broad code or artifact deletion was included.

## Verification evidence and limits

Focused checks recorded 36 event tests, 63 import tests (41 subtests), four digest-contract tests, and one post-revert pin test passing. Both type logs report zero errors. The digest equivalence probe covers behavior cases, not public byte-identical function metadata.

The broad core-test attempt is incomplete: it reports 131 passed and 114 subtests before `KeyboardInterrupt` at `sensitivity.py:2322` after 420.54 seconds. Do not describe this as a full-suite pass. `PROBE_TESTS.log` preserves two pin-sensitive failures from an exploratory pre-revert state. The paired-face refusal mismatch is explicitly recorded as inherited in `INHERITED_PIN_LIMIT.json`; the response-producer self-pin was rechecked after reverting its import edit, with `PIN_TEST.log` passing. No historical fixture or source pin was weakened to force a pass.

The source-byte changes intentionally invalidate any historical artifact that pins the old module bytes. Keep those receipts bound to their archived source/commit; do not update their hashes to authorize the cleaned source. The known paired-face failure predates this cleanup.

## RED disposition

I found no semantic blocker in the deduplicated digest validator or the two import removals. The cleanup is safe within its stated scope. The broad suite remains unverified, and historical pin failures must remain separate from the focused cleanup checks. Keep the large versioned ledger/promotion/sensitivity schemas and research scripts until a separate compatibility and artifact-migration plan supports moving them.
