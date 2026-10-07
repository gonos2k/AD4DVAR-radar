# Model-guided resume from 69c4 — RED final audit

Date: 2026-10-08
Disposition: **The one bounded resume completed three additional full-space steps and preserved the original producer history.** Actual J and Φ both decreased at every committed step. The final point remains far from stationarity, and the observations near the external face cannot be attributed to that face alone.

This is a read-only audit of saved records and derived arithmetic. I did not rerun FV, gradient, HVP, PCG, forecast, adjoint, reanalysis or tests.

## Resume and resource closure

The new child raw is `model_guided_resume_20261008_attempt1/step.json` (SHA `66aa57c3…`); the parent child hash matches it and the embedded resource receipt matches the sidecar. The child, parent and resource report `completed` / `max_iterations_completed`, exit 0, **3 new commits, 3 new HVPs and 0 PCG**. Guard time was **188.261 s**, sampled peak RSS **385,761,280 bytes**, within the 300 s / 1 GiB guard. Source, fixed-input and runtime snapshots close across 156 paths.

The resume plan SHA is `9fc5e046…`; all 111 source and 50 archive pins match. The resume base is exactly the prior committed `69c4a794…` control, not the earlier 058 or e29 points. The old producer's 3 commits/HVPs remain immutable provenance; this invocation starts its own iteration/HVP counters at zero. The legitimate `input_before` control hash 69c4 and `input_after` final hash 2cdc differ, while parameters, terminal truth and archived fixed inputs remain equal. The previous producer files and compressed/source snapshots are not rewritten.

## Actual progress

| Point | (J) | Φ | ∥g∥₂ | ∥g∥∞ | (Q_y[3,2]) |
|---|---:|---:|---:|---:|---:|
| Resume base `69c4a794…` | 0.061369252443 | 0.018131732819 | 0.190429687 | 0.097245034 | (+1.37948\times10^{-5}) |
| Commit 1 `03a083d7…` | 0.061356445952 | 0.014639238839 | 0.171109549 | 0.092597766 | (+3.02440\times10^{-5}) |
| Commit 2 `4a60bfe6…` | 0.061348741146 | 0.012878690002 | 0.160491059 | 0.088835566 | (+4.14794\times10^{-5}) |
| Commit 3 `2cdccade…` | 0.061347638323 | 0.012673246503 | 0.159205820 | 0.088510893 | (+4.31613\times10^{-5}) |

From resume base to final, J fell **0.03522%**, Φ **30.10460%**, ∥g∥₂ **16.39653%**, and ∥g∥∞ **8.98158%**. These are measured full-objective values, not forecasts or response scores. The final maximum gradient is still about (8.85\times10^8) times the (10^{-10}) pending-root threshold.

Every accepted point was the iteration’s last accepted trial after actual J/Phi Armijo and its own complete strict-branch check. Candidate counts were 3, 4 and 7; accepted alphas were (3.73720\times10^{-4}), (2.71691\times10^{-4}), and (4.29927\times10^{-5}). Actual moves were (7.11673\times10^{-5}), (4.64890\times10^{-5}), and (6.89994\times10^{-6}), all below the 0.05 radius. Fresh current-point slopes were resolved negative at each base. No old HVP or direction entered the resumed run.

## External-face event and causal limit

The original monitored face (Q_y[3,2]) remains positive through this resume. A separate external face, zero-based (Q_y[4,3]), approaches zero from the negative side at the accepted controls:

\[
-5.75220\times10^{-5}\ \to\ -2.66999\times10^{-5}\ \to\ -5.55503\times10^{-6}\ \to\ -2.37504\times10^{-6}.
\]

The saved static-face arithmetic shows larger candidates crossed this external face to positive flux while retaining their own strict branch and passing J Armijo, but failing actual Φ Armijo. This recurs at each iteration’s larger candidates. Accepted points stay on the negative side and move closer to zero. This is a useful search observation, not proof that the external face alone caused rejection: every direction is the full 26-vector (-g), and both analysis and future branch-partition hashes change at every accepted iteration. No face-zero gradient, conditional face minimum, or event-free path was evaluated.

At the final negative-(Q_y[4,3]) point, the saved Euclidean tangent projection norm is about **0.13657** against full ∥g∥₂ **0.15921**; the unit-normal gradient is about **−0.08183**. This is a projection at a nearby nonzero-flux point, not a gradient evaluated on the face. The substantial tangent residual rules out claiming conditional face stationarity from this diagnostic.

## Research scope

The result is a successful bounded resume of the same full-space numerical kernel. It establishes three additional accepted points from the last committed 69 endpoint, with fresh HVPs and independent endpoint/source/input/runtime closure. It does not establish full-Hessian SPD, a full 3-hour stationary root, adjoint/VJP/reanalysis, an independent future score, or physical meaning for either face. Preserve the original three producer commits and report this continuation as a separate bounded invocation.
