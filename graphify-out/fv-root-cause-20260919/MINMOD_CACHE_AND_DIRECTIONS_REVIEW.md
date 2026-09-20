# PR167 follow-up: cache integrity and two further directions

Baseline: merged PR167 (`0407b8e3`). The archived sine experiment remains valid
and immutable. This follow-up does not repeat its smaller steps or the old
180-minute/240-grid experiments.

## Numerical contract and predeclared experiment

The governing/discrete operators are unchanged: positive minmod FV with SSPRK2,
fixed known boundary/support, jointly controlled initial echo, five flow
coefficients and growth. The same robust objective and forecast define
`H c_dot = -J_cp d` and `s = E_p d + E_c c_dot` at the saved stationary point.

A small linear-system residual cannot validate the direct term `E_p`. Cache
reuse must bind the numerical contract and payload, compare stored directions,
and compare cached `E_c,E_p` with fresh derivatives of the current fixed score.
This does not require rebuilding the Hessian.

Two directions are fixed before their experiments:

1. `theta`: only the final mean-background parameter changes, with all 60
   observations fixed. This is a parameter derivative, not a learning update.
2. `middle_time_bias`: the 20 middle-time observations change equally, while
   first/last observations and theta are fixed. Unlike the sine direction, it
   leaves `B=y[0]+theta*pattern` directly unchanged.

Both use `h=.001*2^-j`, unchanged maximum gradient `<1e-10`, full 54-stage
selector/face-sign matching, and relative central-derivative error `<=1e-4`
for two consecutive step sizes. The verification field remains the fixed
nominal forecast plus `0.1*pattern`; it is not independent forecast truth.

## Completion record

Cache P2 closed: literal direction checks, componentwise fresh `E_c,E_p`
comparison, numerical AST/core/input identity and full linearization digest.
Trusted legacy artifacts are anchored by archive/producer hashes. Altering
validation/display code alone does not change the calculation AST identity.

Both predeclared directions passed at `h=.001,.0005`:

| Direction | h | Adjoint | Central reanalysis | Relative difference | Max endpoint gradient |
|---|---:|---:|---:|---:|---:|
| theta | 0.001 | 6.409908768178e-08 | 6.409909671730e-08 | 1.409618201e-07 | 4.402750733e-11 |
| theta | 0.0005 | 6.409908768178e-08 | 6.409910243647e-08 | 2.301856557e-07 | 1.227560154e-11 |
| middle_time_bias | 0.001 | -3.038272528697e-03 | -3.038252063594e-03 | 6.735769382e-06 | 1.057987964e-11 |
| middle_time_bias | 0.0005 | -3.038272528697e-03 | -3.038267412452e-03 | 1.683932240e-06 | 1.387557777e-11 |

General minmod API, product GN stationarity,
finite-path certification, finite impact at the original sine `h=.001`, typed
prior learning and whole-chain D7 remain separate open tasks.


## Evidence accounting

- Main measurements: `minmod_theta.json` (83.74 s, sampled peak RSS 353,583,104
  bytes) and `minmod_middle_time_bias.json` (205.75 s, 348,880,896 bytes).
  Both producer hashes match `minmod_additional_directions_measured.py`.
- Each `*_final.json` is a validated resume with the same measured pairs and no
  repeated reanalysis. Cache-only rechecks took about 3 seconds each; separate
  `*_cache_recheck.run.json` files record them. These are not extra successful
  reanalysis experiments. Fresh score derivative differences were exactly zero.
- The original reports inherited historical sine `metadata_correction`. The
  final reports move it under `resume_provenance.historical_metadata_correction`;
  the original files remain unchanged and are linked by resume path/file hash.
  Inherited `timings` describe the ancestor's setup/Hessian calculation, not the
  cost of these new runs. Use the separate wrapper records for measured costs.
- Regressions cover finite direct corruption, dtype/shape/nonfinite values, tiny
  components, changed verification/score, direction mismatch, payload mutation,
  numerical AST changes, actual new-direction wiring, and completed-cache resume.
- Focused suite: 67 pass/1 fail initially; the failure was the publisher's older
  fixture without optional new reports. After preserving that optional behavior,
  all 41 publisher tests passed. Together with 18 cache, 6 spatial-branch and
  3 joint/replay tests, **68 distinct tests passed**, without counting reruns.
  PyTorch emitted existing `torch.jit.script` deprecation warnings. Full CPU CI
  and package checks were not run.
- HTML uses separate measured rows; original `demo-data` hashes unchanged:
  template `88bb5eb25dfb62cd519d98912d421407678f9957dc7b3a7503444d78a113c8e1`,
  index `5fe7d42540b66e83ca40eb4943a070696402d332f91e4ecea54270ebfb28c31e`.
- GREEN/RED found no numerical blocker; the historical metadata placement and
  explicit h-schedule validation were corrected. Their reviews do not certify
  the general minmod API or all parameter directions.
- Aside browser: inspected the live original demo's new table and limits on
  desktop; screenshot `minmod_directions_desktop.png`. Tiny theta differences
  use scientific percent notation rather than rounding to zero. No new mobile
  viewport check was performed. The table retains horizontal overflow handling.
- Graphify code-only cached refresh: 6607 nodes, 64944 edges, 226 communities,
  zero semantic LLM tokens. No semantic corpus rebuild.
- RED's remaining scope notes: the inherited `direction` description names the
  original sine/theta setup; the additional direction is unambiguously stored
  in `selected_direction` and its literal tangent `parameter_direction` vector.
  The linearization digest binds solver-cache data, not completed pair records;
  publication binds the full archived report. This is not a general certificate
  for arbitrary edited completion records.
