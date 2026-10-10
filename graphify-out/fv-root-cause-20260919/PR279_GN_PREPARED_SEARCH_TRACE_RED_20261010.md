# RED trace review: PR279 prepared GN candidate — 2026-10-10

## Finding

The saved 311→027 endpoint traces confirm one changed selector at the closest rejected endpoint, candidate 20, and show it reverted at the accepted candidate 21 and its independent P2 endpoint.

In `gn_prepared_search_20261010_attempt1/step.json.gz`, compare `candidate_trials[19].branch_trace` with the prior PR278 `final_repeat.branch_trace` in `gn_local_commit_20261010_attempt1/step.json.gz`. For both sides, there is exactly one choices-array difference: `choices[124][0].choose_left[1][2]`, false at 311 and true at candidate 20. The saved choice values at that cell are `left_sign=-1`, `right_sign=-1`, and `slope_sign=-1` at both endpoints. Thus the observed change is selection of the left versus right negative increment; it is not a sign flip. The `face_signs` arrays have no differences, and the complete saved 360-stage endpoint traces contain no other selector or face-sign differences.

Candidate 21 (`candidate_trials[20]`) and the independent P2 endpoint (`final_repeat.branch_trace`, control `027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769`) match 311's choices and face signs exactly. Both sides have the same saved signature at these three endpoints. Candidate 20 has `nonfinite_or_tie=false`, 360 observed/expected stages, and global `active_limiter_gap_over_q=9.995662442482622e-10`; the corresponding values are `9.075476315907308e-9` at 311 and `4.0379611279147925e-9` at candidate 21/027.

The old PR277 gzip `gn_local_window_20261010_attempt2/diagnostic.json.gz`, sample α=`1.1148976753303202e-5`, reports the same first changed-selector location on both sides: stage 124, x, row 1, column 2, `choose_left`; it reports `choices_changed=true` and `face_signs_changed=false`. This confirms the event location recurred in a different saved search. The old trace delta stores only the first changed selector and not its before/after values or full candidate trace, so it cannot establish the old toggle orientation or whether that sample had downstream selector changes.

## Diagnostic scope

The recorded row/column are indices in the interior choice arrays. Source construction uses `q[1:-1, 1:-1]`, so choice-array row 1/column 2 maps to state cell `q[2, 3]`; orientation index 0 maps to x. The selector compares two negative x increments and chooses the smaller-magnitude side under the minmod rule.

The normalized margin is a global minimum across captured stages using a per-stage q scale. It does not identify the stage attaining the minimum, give a signed distance to this specific selector boundary, or locate a crossing along the α path. Endpoint signatures show the discrete state at those endpoints; they do not certify path smoothness, path crossing count, derivatives at the event, or later trajectory behavior. In particular, the accepted endpoint matching 311 does not establish that every intermediate point shares that branch.

## Source and saved evidence

- `examples/weather_scenarios/fv_point_3h_nonsmooth_coupled_probe.py:177-204` builds the x/y selector arrays and records signs; `:187-194` computes normalized limiter-gap minima; `:245-246` hashes choices and face signs.
- `examples/weather_scenarios/fv_point_3h_gn_local_window_probe.py:328-339` maps a first difference to stage, axis, row, and column.
- `gn_local_commit_20261010_attempt1/step.json.gz` → `final_repeat.branch_trace` is the 311 endpoint.
- `gn_prepared_search_20261010_attempt1/step.json.gz` → `candidate_trials[19]`, `candidate_trials[20]`, and `final_repeat.branch_trace` are candidate 20, candidate 21, and the 027 P2 endpoint.
- `gn_local_window_20261010_attempt2/diagnostic.json.gz` → `samples[0].trace_delta` is the prior PR277 location-only evidence.
