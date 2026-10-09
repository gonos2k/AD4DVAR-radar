# Same-point tangent preconditioner comparison — final GREEN audit — 2026-10-09

## Disposition

The saved paired comparison completed from the frozen 33cb endpoint. The raw execution state, parent receipt, resource receipt, source map, row/HVP ledger, both candidate arms, selected endpoint, and final-repeat closure are consistent. Independent saved-array recomputation confirms both eligible arm endpoints and the baseline winner. The comparison shows a one-point result only; it does not establish systematic baseline superiority, a root, a minimum, a response, or forecast improvement.

No FV, gradient, HVP, test, seed reconstruction, or guard was rerun for this audit. No source or run artifact was edited.

## Plan, source, and execution receipts

The frozen comparison plan SHA-256 is `be2886f2686cd04e71d3d244967da378bae9c22e39168c5f22bd85e7801169a4`; all 131 source pins and 116 archive pins match local bytes. The raw child is 102,618,544 bytes with SHA-256 `c64480cfdc8b011b63b8bec4fd0e704187b28657d47e5e51b63f606406e0c4cf`. Its parent `child_sha256` matches; the parent’s nested resource receipt equals the standalone resource receipt. `source_before` equals `source_after` across 242 entries and equals the plan maps plus the plan digest. Fixed input identities and CPU FP64/Python 3.12.13/Torch 2.13.0 runtime are unchanged.

The root-owned command completed with exit code 0 and `tangent_precondition_comparison_accepted`. Elapsed time was 91.368 seconds. The monitor recorded 342 RSS samples and a peak of 951,271,424 bytes, about 88.6% of the 1-GiB sampled limit, with no timeout, SIGTERM, or resource termination. This is sampled child RSS, not a hard OS allocation cap.

The archive manifest is now present. It pins the 102,618,544-byte raw child (`c64480c…`), the lossless 744,512-byte gzip (`e7b28a86…`), and the run/resource receipts. I independently decompressed the gzip and confirmed byte-exact equality with the raw; both receipt hashes and the nested resource object match.

## Base, side products, and receipts

The comparison starts exactly at control `33cb86ca73a404e6ff9260acbf63ee97f248ac0c1f529427eb1af1ec85a43586`, theta `0.3295008210283633`, native `J=0.06124426330405451`, and `F²=0.005145254919599806`. `input_before` matches the prior resume endpoint and the fixed parameter, truth, and problem identities remain unchanged. The row receipt contains 24 completed gradients: one unique row set `0..11` for each side `-1/+1`, all tied to the same base hash and theta. The current row ledger is complete and has no partial row.

The comparison records four completed HVPs: one `-1/+1` pair for each arm, both at the same base and theta, with direction hashes matching their respective saved 26-vectors. The baseline and robust-GN-Jacobi models each evaluated their own bounded grid. The baseline used four candidates and the Jacobi arm five. All seven rejected candidates passed native-J Armijo, current branch-pair, face, side-objective, and finite-gradient checks; each failed only the actual F² Armijo test. One candidate per arm passed those tests. Only the baseline winner received the independent final repeat; that repeat matched both individual side gradients and passed objective, merit, face, branch, trace, source, fixed-input, runtime, and deadline closure. The selected current control/theta and atomic last-confirmed state match that committed baseline endpoint.

The preconditioned arm used a finite positive diagonal ranging from 1 to 2,840.2344; no entries hit the unit floor. Its direction norm was 0.01930 versus 0.07173 for the baseline tangent direction. The recorded data diagonal, projected prior, projected face-curvature diagonal, and direction reproduce the frozen method; this is a floored search metric, not a full-Hessian or SPD-curvature certificate.

## Independent candidate and endpoint arithmetic

I recomputed each evaluated candidate `F²` from its saved 26-control vector, candidate theta, one-sided gradients, and static face value. The values match the saved result to FP64 rounding. Both eligible endpoint reductions are relative to the same 33cb base:

| Arm | Eligible endpoint | Theta | Native J | Actual F² | Actual F² decrease from base |
|---|---|---:|---:|---:|---:|
| Baseline tangent | `0da57dbc…` | 0.3298981721 | 0.0612439356480 | 0.00513255514436 | 0.246825% |
| Robust-GN Jacobi | `04ce1a76…` | 0.3298070938 | 0.0612440158174 | 0.00514288184747 | 0.046122% |

Both candidates passed actual native-J and F² Armijo. The baseline endpoint has lower actual F² and slightly lower J, so the declared selection rule chooses `baseline_tangent`. Its final current control hash is `0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1` and final theta is `0.3298981720696071`. J falls by 0.0005350% from the shared base; the selected endpoint F² falls by 0.246825%. This single comparison does not show that the baseline method is generally better.

The seven rejected candidates had actual F² above the starting F² despite passing J Armijo and the strict current-pair gate. Their non-target `Q_y(3,2)` static-basis sign changed at every one of the 360 analysis stages relative to the base; the selected target `Q_y(4,3)` flux stayed zero to roundoff (largest observed magnitude `6.94e-18` against approximately `5.21e-16` face-audit bounds). The exact pinned pure-streamfunction basis is the fixed `y, x, xy, 0.5(x²−y²), x²y` basis with coefficient limits `[0.11, 0.08, 0.07, 0.04, 0.03]` in `fv_minmod_inverse_probe.py`; static flux recomputation agrees with the saved `Q_y(3,2)` values. The current minus/plus pair passes at every candidate. This association does not establish that the non-target sign change caused the F² rejection. Among the two eligible endpoints, the baseline changes one limiter choice at one stage but no other-face sign; the Jacobi endpoint preserves the base trace. Both endpoints have their own valid paired traces.

At baseline rejected candidate 1, the saved candidate gradients can be recombined post hoc at `theta_min=0.4824901` to give an auxiliary gradient-mixture `F²≈0.00469956`. This is an unapplied scalar reweighting of saved gradients, not a generated point, not an accepted theta-path step, and not evidence of a new J/face/trace closure. It is not used in the arm winner or any claim.

## Result limits and remaining work

The selected final tangent-gradient norm remains about `0.07164`, and both arm endpoints are nonstationary. The comparison performs no Newton solve, dense solve, PCG, full Hessian construction, curvature certification, response, reanalysis, or forecast score. It compares one same-point candidate from each direction family under the unchanged J/F²/branch/face/radius gates. The fixed objective is unchanged and the external FV 70/100 estimate is not updated by this experiment.

The saved focused test log reports 124 passed with 18 existing warnings; the saved type-check log reports 0 errors. I inspected these logs without rerunning them. `PRECOND_RESULT_20261009.json` and the saved-array `PRECOND_ANALYZE_20261009.py` are now present; the result pins the same plan/raw hashes and its base, arm endpoints, branch diagnostics, and theta-min values agree with independent saved-array recomputation. The archive manifest pins raw/gzip/run/resource but not the derived analyzer/result bytes; those files should remain clearly identified as derived evidence.

An earlier audit pass preceded the final findings/KG/checklist files. Those files are now present and their final status is cross-checked below; the theta-min value remains unapplied and the Qy32 sign association remains noncausal.

## Directional surrogate-action and final-document cross-check

Using only the saved row gradients, two HVP vectors per arm, static face normal, and multiplier `mu=n·g_mix/(n·n)`, I independently reconstructed the action `P[(1−theta)H_mix d−mu Q_cc d]`, the full robust-GN-row surrogate action, and the floored diagonal surrogate action for the two stored directions. The action norms/errors reproduce the result record:

| Direction | Actual projected HVP action norm | Full-GN surrogate relative action error | Floored-diagonal surrogate relative action error |
|---|---:|---:|---:|
| Baseline tangent | 9.93713 | 0.578683% | 259.5802% |
| Robust-GN Jacobi | 0.830261 | 0.905159% | 99.1977% |

The full-GN comparison is a row-built surrogate action in these two tested directions only. It does not provide a full Hessian, condition number, global curvature bound, or SPD certificate. The large diagonal-action errors are consistent with information lost by diagonalization; they do not establish which omitted coupling causes the baseline arm to win.

The final `PRECOND_FINDINGS`, `PRECOND_KG_UPDATE`, `PRECOND_RESULT`, analyzer, archive manifest, RED review, checklist, and saved test/type/Graphify logs agree on the plan hash, source/archive counts, raw/run/resource receipts, selected baseline endpoint, arm candidate counts, actual J/F² outcomes, row/HVP counts, and remaining limits. The plan has 131 source and 116 archive pins; result plan/raw hashes match the frozen plan and raw child; gzip unpacks to the raw byte-for-byte. The task checklist records the single comparison, saved-array audit, and final review complete. The saved logs report 124 passing tests with 18 existing warnings, 0 type errors, and isolated Graphify 125 nodes/436 edges; I inspected these logs and records without rerunning them.

One wording clarification is useful in `PRECOND_ARCHIVE_20261009.json`: its scope says “no recomputed candidates,” while the saved-array analyzer intentionally recomputes candidate metrics from stored arrays. The evidence shows no new production candidate evaluations. Rewording this as “no newly evaluated production candidates” would distinguish model execution from offline arithmetic. No numerical claim depends on the wording.

Root clarified the archive scope to state that compression/storage performed no additional production candidate evaluation; saved-array recomputation remains explicitly allowed and labeled.
