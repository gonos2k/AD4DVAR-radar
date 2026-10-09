# RED final audit: three-step interior mixing-minimum resume — 2026-10-10

## Disposition

The saved run closes as three sequential bounded control commits from PR271 endpoint `e801cf…` to `c8fba1…`. All three candidate steps passed fresh theta-minimum, actual J/Psi Armijo, face/current paired-branch, and independent final-repeat checks. No actionable numerical inconsistency was found in the saved run or findings/KG metrics. This is a three-step local decrease record, not a stationarity or convergence result.

RED used the frozen raw/gzip/run/resource receipts, result, and saved-array analyzer only. I did not rerun FV, seed preparation, gradients, HVPs, tests, or the guard.

## State and merit arithmetic

The initial committed state is control SHA `e801cf4651233ec11b2ec98ac9569e411b8e62e68511440e39db1cb9c3872567`, theta `0.48316874590279574`, native `J=0.061240252230254005`, and freshly minimized `F²=0.004689202366398875`. At each subsequent point the carried theta equals the freshly recomputed working theta-star. The saved analyzer independently recomputes each theta from its pair of side gradients and each actual candidate merit from that candidate's side gradients.

| Step | Accepted control SHA prefix | theta-star at candidate | Native J | Fresh-minimum F² | F² decrease | J decrease |
|---|---|---:|---:|---:|---:|---:|
| 1 | `69e382be4f` | 0.480999440896 | 0.061238692758 | 0.004662950029 | 0.559847% | 0.002546% |
| 2 | `94e8559254` | 0.484030665586 | 0.061235045290 | 0.004643267711 | 0.422100% | 0.005956% |
| 3 | `c8fba1f93e` | 0.481886660037 | 0.061233492988 | 0.004605971728 | 0.803227% | 0.002535% |

The terminal state hash is `c8fba1f93ee0b6c83edc5b56792de158921c0a64bab41eef5403c2a514b1fc33`. From the initial state, J decreases `0.0110373%`, F² decreases `1.77494%`, and the minimum-gradient L2 norm falls from `0.068477751` to `0.067867310` (`0.89145%`). The terminal L-infinity norm remains `0.029043684`; the result is far from zero.

The local model predicts stepwise F² decreases of `0.559863%`, `0.537017%`, and `0.518281%`; actual values are `0.559847%`, `0.422100%`, and `0.803227%`. For clarity, the analyzer reports two different relative errors: predicted-versus-actual **change-vector** error is `1.926996e-5`, `0.0600543`, `0.211443`, each divided by the actual change-vector norm; **full-residual** error is `1.445887e-6`, `0.00442590`, `0.0155383`, each divided by the actual endpoint residual norm. These denominators differ, so the latter errors are smaller and are not interchangeable with the former. This is local model agreement for these three steps only.

## Chain, branch and resource closure

Controls and committed theta values chain in order; every step used its first candidate slot, with path norms `2.2837e-5`, `5.3535e-5`, and `2.2842e-5`, each within the per-step `0.05` chart radius. All three endpoint theta-stars are strictly interior and resolved. Target face Q is zero. Paired current-branch endpoint gates pass each step; signatures remain the same across step 1 and change at steps 2 and 3, so this does not certify a smooth path. Static non-target Qy[3,2] stays negative and changes from `-1.84411e-6` to `-4.98393e-6`; that descriptive value does not establish causal contribution.

The raw SHA `2a36ae30ec9749af1b212ef3395c8125fd46a18ca77956807ddc73ca5e9cb0ff` matches the result and parent receipt. Gzip SHA `4d6d2eb1b81562ab50dc98b0540e0480155ea40a5c9783bcf87d7f4ff4d8e844` decompresses byte-for-byte to the 38,511,526-byte raw file. Run SHA is `6b99e9e93d60dab10b7377d9c27fa6ab18e9ca1a4e3a35bc8e6e609ca52dbfbf`; resource SHA is `cbca6ceebda5ac70206429c1522e34ae396ba8ca3cf459806c9e5c48237f1972`. The result is `tangent_mixing_minimum_resume_cap_reached`: 3 commits, 6 current HVPs (3 per side), 0 row VJPs/PCG/dense solves, 89.046 seconds wall elapsed, 86.993 seconds internal, and sampled peak RSS 749,273,088 bytes under the 300/360-second and 1-GiB sampled limits. The cap means the planned three steps completed, not convergence; sampled RSS is not an OS hard cap.

The findings correctly distinguish model predictions from actual merit, note branch changes, preserve the single-sequence/no-minimum-response-forecast limits, and defer curvature, coupled response/reanalysis, and independent future validation. The parent validator checks saved chain/closure flags; it does not independently regenerate theta formulas, which the saved-array analyzer does. The task checklist now marks the run, saved-array audit, and GREEN/RED/result/KG records complete; PR/CI/head integration remains a separate pending receipt.

No root, minimum, curvature qualification, smooth-path, coupled response, reanalysis, or forecast-skill claim follows from this run.
