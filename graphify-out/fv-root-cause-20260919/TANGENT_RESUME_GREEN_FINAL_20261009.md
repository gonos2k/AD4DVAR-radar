# Tangent resume — final GREEN saved-run audit — 2026-10-09

## Disposition

The guarded resume from the accepted PR 267 endpoint completed its declared three-step cap. The saved plan, source pins, raw/gzip data, parent run receipt, resource receipt, accepted-point chain, and final-repeat closures agree. Independent saved-array recomputation matches every saved point field and per-step `F²` model/actual decrease. This supports a limited current-HVP tangent-gradient continuation only; it does not establish stationarity, a minimum, a response, or forecast improvement.

No FV/HVP/guard/test/weather computation was rerun for this audit, and no code or evidence file was edited beyond this memo.

## Provenance and resources

The frozen resume plan is SHA-256 `d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10`. All **129 source pins** and **105 archive pins** match their current file bytes. The run's `source_before` equals `source_after` over 229 entries and equals the plan source/archive receipt plus the frozen plan digest.

The retained raw child is SHA-256 `15c780de8e6b55216407d399d7ce1984280024cfb7f53cbc0ce9e6e4a644db1d` (23,621,880 bytes). The gzip SHA-256 is `477f0272fc9ff021abb479afb3f7c7e789dc02fff5d948b1d66959ab5f1446d5`; decompressing it reproduces the raw bytes exactly. The run receipt (`63530283…`) and resource receipt (`01966d84…`) match their archive manifest hashes; nested and standalone resource receipts are equal, and the parent child hash matches the raw hash.

The guarded command used the resume adapter and exact frozen plan. It exited 0 with `execution_status=completed`, `numerical_status=tangent_continuation_cap_reached`, 79.278 seconds elapsed, and 297 sampled RSS observations with peak 565,248,000 bytes against the 300-second and 1 GiB limits. RSS is sampled `ps` data, not an OS hard allocation limit.

## Resume base, fixed inputs, and closure

The plan loader starts from the preceding accepted endpoint: control SHA `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`, theta `0.332026125873074`, native `J=0.0612512070514904`, and saved `F²=0.005238386295728529`. The new run's `input_before` exactly equals PR 267's `input_after`; the accepted control then advances to `33cb86ca73a404e6ff9260acbf63ee97f248ac0c1f529427eb1af1ec85a43586`. Parameters, terminal truth, and archived fixed-problem identity remain equal to the resume base. Runtime remains CPU FP64, Python 3.12.13, Torch 2.13.0.

The child closed the fresh current-source base before continuation: native J, `F²`, both side gradients, side traces, face value, and strict current branch pair match the saved endpoint. Each of the three new iterations committed its first evaluated candidate. Their six HVP history entries form one completed `-1/+1` pair at each of the three new base controls and corresponding theta, with `current_tangent` / `selected_face_extension` metadata. The raw stores each new direction and both HVP result vectors. No preceding-run HVP was used as the resumed operator.

At every accepted point the independent final repeat passed native objective and merit agreement, side-objective parity, face and branch-pair checks, trace match, both individual side-gradient comparisons, fixed-input/source/runtime equality, and deadline closure. Final `current_control` and `current_theta` match the third accepted commit; the last-confirmed digest/theta/count agree, and the top-level last-confirmed closure equals the last iteration's final repeat.

## Saved-array arithmetic

I recomputed the four point records from the prior final trial and the new accepted trials using the saved controls, theta values, side gradients, and static selected-face geometry. Hashes, J, theta, `F²`, `||F||`, mixed-gradient infinity norm, tangent and normal components, one-sided normal derivatives, and block norms match `TANGENT_RESUME_RESULT_20261009.json` to FP64 rounding.

For each new step I independently reconstructed `F`, `DF`, and `F·DF` from the stored current direction, two current HVP vectors, side-gradient jump, and theta correction. The accepted alpha matches the scalar quadratic model optimum `-F·DF/||DF||²`; actual and predicted `F²` reductions reproduce the saved result:

| Step | Accepted control SHA prefix | Alpha | Actual `F²` decrease | Linear-model decrease | Angle `F` vs `-DF` |
|---|---|---:|---:|---:|---:|
| 1 | `ab27d917…` | 0.000413590482 | 0.735040% | 0.735061% | 85.0817° |
| 2 | `a579d422…` | 0.000501606621 | 0.563970% | 0.612887% | 85.5099° |
| 3 | `33cb86ca…` | 0.000423705418 | 0.489335% | 0.541132% | 85.7814° |

Across the resume, native `J` falls from `0.0612512070514904` to `0.06124426330405451` (0.0113365%); `F²` falls from `0.005238386295728529` to `0.005145254919599806` (1.777864%); `||F||` falls by 0.892918%. The angle and model decrease describe the auxiliary local linearization, while the actual reduction comes from separately saved endpoint evaluations.

The final tangent-gradient norm is still `0.0717279907` and final `||F||` is `0.0717304323`. The run makes bounded progress and remains far from stationarity. It performed no Newton solve, dense solve, PCG solve, response calculation, reanalysis, or forecast validation; the raw root/minimum/response claims remain false and the external 70/100 FV estimate is unchanged.

## Final document consistency cross-check

The final findings, KG update, result, saved-array analyzer, archive manifest, checklist, RED review, and focused verification logs agree on the plan/raw identity, base and final control/theta/J/F², three accepted first-candidate steps, six current HVPs, resource receipt, stepwise model/actual decreases, and remaining claims. The 54-pass/18-warning/3.19-second test log, zero-error type log, and isolated Graphify counts (74 nodes/265 edges) match the checklist and findings; this audit did not rerun them.

The accepted-endpoint branch-signature sequence is `67524657… → 9a22a185… → 0351f7b2…`. The first resumed accepted point retains the starting endpoint's `67524657…` signature; the signature changes on resumed steps two and three. Both side traces agree within each accepted endpoint. Thus the findings' stepwise-change statement describes the accepted-endpoint sequence and does not indicate a branch transition on the first resumed step. No actionable documentation inconsistency remains.
