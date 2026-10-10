> 역사적 설계/첫 preflight 검토 기록. 제어 의존 유동 진단을 누락한 anchor는 실제 첫 시도에서 거부됐고 후속 계획에서 수정됐다. 최신 판정과 근거는 STREAMGN_FINDINGS_20261010.md 및 GREEN/RED 최종 검토를 따른다. 아래 원문을 수정 후 인증으로 해석하지 않는다.

# GREEN design review: PR274 partial coupled-GN continuation — 2026-10-10

## Disposition

The proposed follow-up is sound as a checkpoint-lifetime change and a bounded continuation from the saved one-step endpoint, provided the resume loader treats the interrupted PR274 record as an unclosed prefix. The existing evidence establishes exactly one accepted step at control `7cec4c62bc2ebe3a96106086408adc0ecdbc959fe1b53beafa46902cd41fb7dd`, theta `0.4829895933920232`, objective `0.061195305062512445`, and `F²=0.004486899769141486`. The second point has only 25 completed of 26 started row VJPs, with no solve, HVP, candidate, or commit. The saved terminal state is usable as an endpoint anchor; the partial second-point work is not reusable numerical state.

## Resume provenance and closure

The archived raw/gzip/run/resource receipts and saved-array analysis should be pinned and checked before admitting the endpoint. Validate the one accepted iteration against its accepted trial and independent final repeat, including control hash, theta, J/F², finite two-side gradients, P2, face/branch checks, theta-minimum closure, and last-confirmed state. Preserve the raw interrupted child unchanged as evidence. Do not turn its absent global `input_before`, `input_after`, `runtime_before`, `runtime_after`, or `source_after` into historical closure claims.

For the new launch's expected input identity, derive the identity from the completed predecessor PR273 endpoint artifacts: copy the saved fixed-input identity and change only its control digest to the recorded PR274 committed control hash. Record that this is a derived expected anchor and identify both source receipt and transformation. Do not attach the predecessor's `fixed_input_unchanged` or `runtime_unchanged` flags to the PR274 interrupted child as if they were recorded there. The new execution must prepare its inputs, compare the fresh identity to the derived anchor and the pinned runtime identity before doing numerical work, and record fresh before/after identities and source receipts for its own closure. Its first committed endpoint still requires a new independent final repeat. A missing or mismatching anchor is a refusal that leaves the saved control/theta intact.

Only the exact committed prefix may be restored: its control, carried theta, endpoint J/F², and verified final-repeat closure needed to initialize the next current point. Recompute both side gradients/rows, projected-row parity, current minimum theta, GN matrix and 12-by-12 solve, and the paired HVPs at the resumed point. Never reuse the interrupted second-point rows or any prior row matrix, solve, direction, HVP, or partial candidate. The next fresh execution supplies its own one-arm GN accounting and final closure.

## Checkpoint writer and progress lifetime

Stream `json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False).iterencode(record)` chunks to a sibling temporary file, append one newline, close it, then atomically `os.replace` it over the destination. Keep the JSON options, UTF-8 bytes, indentation, sorted-key order, and newline identical to the legacy `json.dumps(...)+"\\n"` output. Serialization or replace failure must leave the previous checkpoint bytes intact; clean up an unfinished temporary file where possible. The writer must not construct a full serialized string alongside the live VJP pullback.

Keep `comparison_progress` while a point is uncommitted, since it is the recoverable progress record during row/HVP/model/candidate work. Once `committed_progress` has copied the fully closed iteration into `iterations` and updated the confirmed control/theta, remove `comparison_progress` before writing the committed checkpoint. Do not remove it at candidate acceptance or before independent final-repeat closure. This removes the observed duplicate `model_comparisons` payload only after its durable counterpart exists.

Tiny synthetic tests should compare streamed output byte-for-byte with the legacy encoder for representative nested values, exercise `allow_nan=False`, and prove a write/replace failure preserves the prior file. A progress-lifecycle regression should prove the in-flight checkpoint retains progress and the committed checkpoint has the complete iteration and no duplicate progress field. These checks do not need FV or derivative work.

## Guard and claims

Retain the one-run, 600-second internal / 660-second outer, 1-GiB RSS, at-most-three-commit policy and the existing ceilings of 72 row VJPs, six HVPs, three dense solves, and 16 candidate slots per modeled point. Streaming and duplicate removal may reduce transient allocation, but the saved evidence does not show that serialization alone caused the RSS crossing or guarantee any RSS savings. The previous run stopped at 1,074,937,856 bytes, 1,196,032 bytes above the sampled cap, after 84.95 seconds; the future run's guard remains the evidence for feasibility.

The saved endpoint supports a local confirmed prefix only. A completed follow-up, even if all bounded steps close, supports only that recorded path and its local receipts; it does not establish solver convergence, a minimum/root, exact curvature, response, adjoint, reanalysis, or forecast skill. No actual FV, seed, production AD, HVP, or guard rerun belongs to this design review.

## Evidence consulted

- `GNRESUME_GREEN_FINAL_20261010.md` and `GNRESUME_MEMORY_20261010.md` — one confirmed commit, interrupt accounting, source and input/runtime gaps, and measured serialization duplication.
- `AUDIT274_SOURCE_SCOPE_20261010.json` and `AUDIT274_FINDINGS_20261010.md` — exact producer snapshots and the bounded PR274 correction scope.
- `GNRESUME_RESULT_20261010.json` / `GNRESUME_ARCHIVE_20261010.json` — saved endpoint, accounting, resource termination, and receipt hashes.
- `examples/weather_scenarios/fv_point_3h_tangent_continuation.py` — `_write`, `direction_progress`, and `committed_progress` lifecycle.

This is a design review of saved artifacts and code structure only; it does not assert new numerical verification.
