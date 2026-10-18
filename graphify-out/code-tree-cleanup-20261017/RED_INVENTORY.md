# RED inventory review — `src/`, `examples/`, `tests/`

Read-only audit; generated `graphify-out` snapshots were excluded from the code inventory. The tracked tree contains 559 files: 33 Python files under `src`, 220 under `examples`, and 293 under `tests`. I found no byte-identical files or symlinks, and no backup-suffix copies. The two `index.html` files are in separate app roots and are distinct. `examples/weather_scenarios/transport.html` is a documented build output and is explicitly marked for preservation in its README.

## Safe cleanup candidates

- `src/.DS_Store` and `examples/.DS_Store` are untracked Finder metadata. They can be removed and ignored with a `.DS_Store` rule; neither is project source.
- `tests/fixtures/audit_generations/` is empty and has no repository references. Remove the empty directory if it was not reserved for future fixtures.
- The duplicate import in `src/advar/ledger.py` of `validate_resolved_source_coverage_artifact` (formerly repeated at lines 193 and 311) and duplicate `Path` import in `tests/test_fv_point_3h_model_guided_continuation.py` (formerly lines 6–7) are mechanical cleanup candidates. The root task is handling these two edits.
- `examples/weather_scenarios/fv_active_face_response_probe.py:27` imports `_problem` from a test module but never uses it; the actual problem is obtained from `slices._current_problem()` at line 292. This is a concrete example-to-test dependency to remove after its source pins are updated.
- Several probe/continuation files have ordinary unused imports. Current Ruff `F401` inventory reports 199 warnings across the three roots. Treat this as a review queue, not a bulk-autofix target: some experiment imports are unused as Python names but retained with immutable source/artifact pins, and many test imports intentionally smoke-test import compatibility.

## Preserve as compatibility or research evidence

- `src/advar_radar_nowcast.egg-info/` exists on disk but `git ls-files` returns no tracked entries; `.gitignore` already ignores `*.egg-info/`. It is generated packaging metadata, not a tracked duplicate tree. Runtime closure code uses `importlib.metadata`, so do not add it to source or prune it from an installed environment without a build/runtime-closure check.
- The versioned radar-domain evidence `src/advar/data/epsg5179_metric_domain_evidence_v1.json` through `v4.json` is not redundant: `nowcast.py` selects v4, while `tests/test_nowcast.py` covers legacy v1–v3 parsing and v4 use.
- The large `promotion.py`, `ledger.py`, and `sensitivity.py` modules contain versioned legacy receipt schemas, algorithm digests, and validators. Their size is a maintainability concern, but deleting generations or splitting modules without re-export shims risks breaking stored-artifact readers and public import paths. Preserve schema compatibility; consider staged internal module splits with stable facade imports.
- The 220 weather-scenario scripts are not exact copies. Many are bounded experiments, continuation/resume phases, and source-pinned scientific records. Their neighboring tests and archived receipts encode different inputs, branch states, and budgets. Filename similarity alone is not a deletion criterion. Unreferenced-looking imports such as the aliases in `fv_point_3h_gn_continuation.py` should be checked against each frozen source manifest before removal.

## RED disposition

There is little evidence for wholesale duplicate-code deletion in these roots. The safest cleanup is limited to untracked filesystem litter, an empty fixture directory, confirmed redundant imports, and carefully pin-aware unused-import pruning. Keep historical schemas, evidence generations, and research checkpoints until the project has an explicit archival/provenance relocation plan.
