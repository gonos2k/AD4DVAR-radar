# Tangent continuation GREEN final audit

## Outcome

The saved run supports three accepted tangent-gradient corrections, each accepted on its first evaluated candidate. Receipt hashes, source/input/runtime closure, per-step side-gradient and trace repeats, static face geometry, and saved-array model arithmetic are consistent. This is local numerical continuation evidence only; it does not establish a smooth root, minimum, response, or forecast-skill improvement.

Keep the scalar objective and tangent merit distinct: J is the original objective over 26 controls; F is the 27-component auxiliary residual `[g_mix, Q/0.84]` and F² is its continuation merit. These F values are not the native tie-autodiff gradient or the legacy raw Φ from the earlier nonzero-face point.

## Provenance and resource record

- Frozen plan SHA-256: `cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69` (126 source pins, 94 archive pins). The raw report's plan hash and source/archive maps match. All pinned source/archive bytes still match the plan; raw `source_before` and `source_after` are equal (215 entries).
- Raw step: `tangent_continuation_20261009_attempt1/step.json`, SHA-256 `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`, 23,638,675 bytes. Gzip SHA-256 `62d92dd38a6f97db4871d464201183db9b4bbda31ea224bb8c390646b917958b`; decompressing it reproduces the raw bytes exactly. `TANGENT_ARCHIVE_20261009.json` pins both.
- The saved run and resource receipts match their archive hashes (`7bd3dbbea7735413e6f3ea0fe34ebd4c46081694443b7d3f4eb8501d62d36854` and `0712e9ea77b8253502dc7ae290bd1968d84fd6b123ebea279dac8665f7ad609a`). Both report exit 0, no monitor error or resource termination, 76.515 s elapsed, and sampled peak RSS 570,179,584 bytes from 287 samples against the 1 GiB / 300 s limits. RSS is sampled by `ps`, not a hard allocation cap.
- The parent and child report `completed`; numerical status is `tangent_continuation_cap_reached`. HVP counters and history are all 6, with two completed current-tangent calls (side -1 and +1) for each accepted base point. All saved gradient, HVP, direction, F, and DF arrays are finite.

## Per-step saved-array audit

Using the previous accepted pair of side gradients and theta as each step's base, I recomputed the mixture, implicit face normal, 25-coordinate chart direction, residual `F = [g_mix, Q/0.84]`, `DF`, theta update, merit product, Armijo bounds, candidate face value, and accepted F² from the saved arrays. No model or production derivative was evaluated. The three steps have zero maximum component error for the saved mixture and F, maximum DF component error below `2.0e-18`, and exact saved theta-update arithmetic. The direction matches the reconstructed 26-by-25 implicit chart direction exactly; its face-normal component is at most `8.7e-19`.

| Step | α | Actual control move | J | F² | θ | J/F² Armijo | HVP sides |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0.00138825994 | 1.10311e-4 | 0.06125555372 | 0.00544789242 | 0.33591983873 | pass / pass | -1, +1 |
| 2 | 0.000351246737 | 2.59245e-5 | 0.06125366699 | 0.00529731869 | 0.32882832999 | pass / pass | -1, +1 |
| 3 | 0.000467008981 | 3.39889e-5 | 0.06125120705 | 0.00523838630 | 0.33202612587 | pass / pass | -1, +1 |

All actual control moves are inside the 0.05 radius. Each step used one candidate slot and one evaluated candidate. Recomputed candidate F² matches the raw value exactly at all three steps. The saved J slope, J Armijo bound, and F² Armijo bound agree with the base gradients, saved direction, α, and `F·DF` arithmetic.

For each accepted point, the two branch traces have 360 observed stages, no nonfinite/tie flag, and a passing current branch-pair gate. Final-repeat side gradients equal the proposal arrays; side trace signatures match; native J, merit, face audit, source, fixed input, runtime, and deadline flags pass. The final current control SHA `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7` equals the last confirmed control and last accepted trial.

Relative to the immediately preceding point, the branch-pair signature changed on steps 1 and 2, then stayed the same on step 3. Both side traces agree within each current pair at every accepted point. The final mixed-gradient infinity norm is `0.029792523644067155`; the tangent-gradient norm is `0.07237388241205849`, and the mixed normal component is `0.0006383105310589993`. Thus almost all of the final F norm is still tangent residual (about 99.9922% of F²); the correction did not reach stationarity. The saved policy and run record show 6 current HVPs, 0 dense solves, and 0 PCG solves.

## Endpoint and interpretation

Saved-array recomputation matches all four point records and all three step records in `TANGENT_RESULT_20261009.json`. From the accepted PR266 base to the final control, J decreased from `0.06126370581028981` to `0.0612512070514904` (0.0204016%); F² decreased from `0.006314384483614977` to `0.005238386295728529` (17.0404%); F norm decreased from `0.07946310139690609` to `0.07237669718720612` (8.91786%).

`TANGENT_ARCHIVE_20261009.json` pins the raw/gzip/run/resource evidence, but does not pin the derived `TANGENT_RESULT_20261009.json` or `TANGENT_ANALYZE_20261009.py`. I independently recomputed their point and step claims from the pinned raw and prior accepted arrays; the claims match. The retained 70% FV-completeness value remains an external estimate from earlier work, not a result of this continuation. No root, minimum, response, or new forecast-validation claim is supported here.

Audit was static and arithmetic-only: no FV, HVP, weather computation, production rerun, or guard was launched.
