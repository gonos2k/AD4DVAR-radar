# d31 strict/inexact comparison — final GREEN review

Date: 2026-10-07
Disposition: **GO for the single bounded comparison as recorded.** This review checked the frozen plan and saved run/arm receipts, persisted test/type/Graphify records, and the saved-array scientific-check script/result. It did not rerun FV, HVP, PCG, or the scientific-check script.

## Result

The comparison started both cold-start arms from the pinned d31 control `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`. Each arm independently accepted one step after evaluating four candidates, with displacement norm `0.00625`. Both passed actual J-Armijo and Phi-Armijo checks and their strict endpoint branch checks. The parent receipt reports `comparison_complete=true`, `selected_arm=inexact`, `fixed_input_unchanged=true`, and `source_unchanged=true`.

| Arm | Inner tolerance / true relative residual | PCG / HVP | J at accepted point | Phi at accepted point | max gradient component |
|---|---:|---:|---:|---:|---:|
| Strict | `1e-10` / `8.10355e-11` | 29 / 31 | `0.06177956399461902` | `0.2837064060538132` | `0.41661933405` |
| Inexact | `1e-3` / `3.77048e-4` | 20 / 22 | `0.06177954756643603` | `0.28369125567712217` | `0.41661899700` |

At this d31 point the inexact arm used **9 fewer HVPs (29.0% fewer than strict)** and 9 fewer completed PCG iterations. Its recorded J and Phi were slightly lower, and its maximum gradient component was slightly smaller. This supports a one-pair cost/progress result for this start point. The observed per-arm elapsed times were 389.556 s strict and 304.147 s inexact; since strict ran first and both arms shared one process and remaining budget, timing can include order/cache effects and is not a general speedup estimate.

The actual residual met the declared inexact target, and the saved arrays pass the reported recomputation of residual, both directional slopes, accepted control reconstruction/hash, Armijo/branch flags, and Phi from the saved gradient. Inexact directional values include `gᵀs=-0.00747206233`, `gᵀHs=-0.65066942839`, and observed `sᵀHs=0.00747206176`; the strict solve has corresponding strict descent and positive observed direction curvature. The independent check is saved-array arithmetic, not another model/HVP execution.

Both endpoints share the same pointwise branch signature, which differs from the d31 base signature. The endpoint gates passed; the records do not prove branch smoothness along the segment. The top-level selected control hash is `f462a4961634efa253e3757f7de44304386135eb3a229f3729a61eb911e6eed9`, matching the accepted inexact endpoint.

## Budget, plan, and implementation checks

The comparison used 53 total HVPs under the shared cap of 90. The guarded parent completed in `695.530 s` under the 780-second wall limit, sampled peak RSS was `373,604,352` bytes under the 1 GiB guard, child exit code was 0, and the resource receipt records no signal, monitor error, or resource termination. The plan hash is `d69ce3000f1605bed67ea67a4c6751bbbdf2ab142fa47bd24418c368d9220d41`. I rechecked all **102 source pins and 24 archive pins** against the frozen plan; none are missing or mismatched. The raw child hash in the parent receipt matches the saved result.

The persisted affected suite reports **63 passed, 18 existing deprecation warnings**; persisted type checking reports **0 errors, 0 warnings, 0 notes**. The incremental Graphify receipt covers six files (98 nodes, 311 edges) and shows the shared graph/report hashes unchanged. These are recorded checks; they were not rerun during this review.

The earlier prelaunch concerns remain addressed: the outcome field starts unselected and is set to inexact only after a verified accepted step; a budget stop leaves the comparison incomplete and the base selected. The frozen plan explicitly defines strict-first shared remaining budget, no equal per-arm reserve, and no paired cost contrast if an arm is unfinished. Here both arms completed, so their one-step results are comparable at the same start. No budget was extended after observing an arm.

## Limits

The selected endpoint is still far from the final stationarity criterion: `||g||∞=0.416618997` versus `1e-10`, roughly `4.17e9` times the threshold. The inexact true residual is correctly looser than the strict residual; the result does not establish convergence, a root, or the efficacy of the forcing rule near a root. The two-arm comparison is one ordered pair, so it does not establish general speedup.

No final current-point curvature, eligible stationary point, adjoint, reanalysis, or independent forecast score was produced. The optimizer's J/Phi decrease is not evidence of improved 3-hour forecast skill.

## Evidence locations

- Plan: `D31_COMMITTED_BASE_COMPARISON_PLAN_20261007.json`
- Parent result and strict/inexact arms: `d31_inexact_comparison_20261007_attempt1/step.json`, `strict.json`, `inexact.json`, `step.run.json`, `step.resource.json`
- Saved-array check code/result: `D31_INEXACT_SCIENTIFIC_CHECKS_20261007.py`, `D31_INEXACT_RESULT_20261007.json`
- Focused test/type/Graphify receipts: `D31_INEXACT_TESTS_20261007.log`, `D31_INEXACT_TYPES_20261007.log`, `D31_INEXACT_GRAPHIFY_20261007.json`
