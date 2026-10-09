# PR 267 evidence audit — GREEN — 2026-10-09

## Disposition

The saved run, frozen plan, source and archive pins, producer, numerical result, and PR integration receipt are internally consistent. Independent saved-array recomputation confirms the three-step chain and its reported native objective and auxiliary merit changes. This audit did not launch FV, HVP, guard, test, or weather computations. It made no code or evidence changes beyond this report.

## Verified provenance and receipts

- `TANGENT_CONTINUATION_PLAN_20261009.json` hashes to `cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69`. Its 126 `source_files` and 94 `archive_files` match the corresponding maps in the decompressed raw run; every pinned path exists locally and its bytes match.
- The raw child is 23,638,675 bytes with SHA-256 `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`. The tracked gzip has SHA-256 `62d92dd38a6f97db4871d464201183db9b4bbda31ea224bb8c390646b917958b` and decompresses byte-for-byte to the raw child. The run and resource files match their archive pins (`7bd3dbbe…` and `0712e9ea…`); the nested resource receipt matches the standalone resource file.
- The execution receipt records exit 0, 76.515 seconds, 287 RSS samples, and sampled peak RSS 570,179,584 bytes against 300 seconds and 1 GiB. The record correctly identifies RSS as sampled `ps` data, not an OS-enforced memory cap. The child and parent report `completed`; numerical status is `tangent_continuation_cap_reached`.
- `source_before` and `source_after` are equal across 215 entries, with `source_unchanged=true`. Fixed problem, parameters (`8871db49…`), and terminal truth (`b927c1a3…`) remain fixed; the control changes from the selected continuation base to the accepted final control as intended. Runtime remains CPU FP64, Python 3.12.13, Torch 2.13.0. The raw also retains the producer's archived input identity separately from the new continuation base.
- The source-pinned producer and focused regression tests are present in merge commit `391156a6853682077d836ed1dfecd0a35fa8f481`. `TANGENT_ANALYZE_20261009.py` and `TANGENT_RESULT_20261009.json` are committed in that same PR and accessible; they are derived artifacts but are not individually pinned by `TANGENT_ARCHIVE_20261009.json`.

## Independent saved-array check

Starting from the accepted point in the pinned candidate-requalification archive, I recomputed the mixture, face residual, and `F²` from the saved control, theta, and side gradients. For each iteration I checked the saved base hash, 26-component direction, stored HVP vectors, accepted control hash, theta update, native objective, `F²`, and Armijo flags. No FV or derivative was regenerated.

The committed controls chain exactly:

`6b29dacd01fc… → 20ae0609804b… → 747e5506c4e2… → ed106d7bc277…`

The corresponding theta values are `0.3504554198817298 → 0.33591983873372505 → 0.3288283299901934 → 0.332026125873074`. Each iteration has two completed HVP-history entries, sides `-1` and `+1`, tied to that iteration's base control hash and theta. The raw stores the per-iteration current tangent direction plus both 26-element HVP result arrays; all are finite. The pinned producer computes both products using that current direction. HVP-history entries do not contain a separate direction hash, so call-to-direction binding is supported by the ordered per-iteration records and pinned producer implementation rather than an independent per-call direction digest.

The native objective `J` moves from `0.06126370581028981` to `0.0612512070514904` across the full chain (0.0204016% decrease). The distinct continuation merit is `F²=||[g_mix,Q/0.84]||²`, which moves from `0.006314384483614977` to `0.005238386295728529` (17.0404% decrease). Recomputed intermediate objective/merit values match the saved records to floating-point roundoff. These numbers support three accepted tangent-gradient corrections, not a minimum, root, response, or forecast-skill claim.

## PR, CI, test scope, and checklist

GitHub reports PR 267 merged at `391156a` into merge commit `63610e9540809804d536814f59fcfed3769c2867`. CI run `37897929978` completed successfully for the exact PR head SHA. The only executed job was “Initial-field lab UI”; “Python 3.12 CPU”, the Python CPU shards, and “Wheel and CLI smoke” were skipped. The receipt's distinction between successful UI CI and skipped Python/wheel CI is accurate.

The committed focused-test log records 45 passes, 18 TorchScript deprecation warnings, and 2.16 seconds; the type log records 0 errors, warnings, or notes. The focused test log does not include its exact pytest command, so this audit verifies the saved result and the committed test files, not the invocation scope from the log alone. No Python test job ran in the PR CI workflow.

**Actionable record inconsistency:** `TANGENT_TASK_CHECKLIST_20261009.md` still leaves “PR/자동 CI/정확 head 통합” unchecked, while `TANGENT_PR267_INTEGRATION_20261009.json` marks exact-head integration complete and GitHub confirms the merge. The checklist is stale after integration and should be reconciled in the final handoff.

**Traceability limitation:** the archive manifest pins raw/gzip/run/resource, not the derived analyzer/result. The analyzer and result are nevertheless committed and available in the PR. The HVP call history likewise lacks per-call direction digests, though the direction arrays, product arrays, ordered HVP records, and pinned implementation are all present. These are provenance-granularity limitations; no numerical contradiction was found.
