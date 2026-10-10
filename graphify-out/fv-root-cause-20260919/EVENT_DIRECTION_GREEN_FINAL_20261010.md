# GREEN saved-data review: event-orthogonal GN comparison

Frozen plan `14773f392f41eaf9cf0fdba2b59bbf3427f8d3fdf0d0e7cc92b266704928ea29`; pin, run, operator, and P2 checks: **True** (11/11).
This audit reads saved JSON and uses standard-library arithmetic only; it ran no FV, seed, AD, or HVP code.

## Direction comparison

The Euclidean event projection removed a component of length 0.00363844934891 (7.833472% of the baseline direction norm) without rescaling. The projected event dot is 0 (budget 1.02052586758e-13); the selected-face dot is 4.33680868994e-19 (budget 1.18424130847e-15).

| Arm | first pass | α | actual move | actual J change | actual R change | max local model χ | actual/model R decrease |
|---|---:|---:|---:|---:|---:|---:|---:|
| prepared_gn | 22 / 24 | 3.48406303677e-07 | 1.61825895107e-08 | -8.43286461494e-10 | -3.71326998923e-09 | 86.941291% | 1.00001246 |
| event_orthogonal | 1 / 24 | 0.00344146052506 | 0.000159355926612 | -8.3602860407e-06 | -0.000105202636644 | 2.343802% | 1.00226059 |

Selection chose **event_orthogonal** by actual R (0.00437321975313 vs 0.00447841867651); the R tie rule was not invoked. The event-orthogonal arm passed on its first grid slot; the prepared GN arm first passed on slot 22.
The selected endpoint reduced R by 2.349100% and J by 0.01366228%. G∞ moved from 0.0285818984029 to 0.0285885679533, so this does not establish improvement in every gradient norm.
Its local maximum model χ was 2.343802%, below the prepared GN arm's 86.941291%. The evidence favors a longer admissible first-passing step at this point, not a better maximum linear-model reduction or a general convergence claim.

Four fresh HVPs were computed at the same 027 base point (two sides per arm); there were no Jacobian rows, dense solves, or new-point readiness operators. One selected candidate passed one independent P2 repeat and was committed; no further candidates ran.

## Checks

- `plan_preflight_and_all_pins_close`: **True**
- `attempt_manifest_raw_gzip_run_resource_close`: **True**
- `guard_completed_zero_exit_within_resource_caps`: **True**
- `parent_event_and_GN_receipts_match`: **True**
- `fresh_base_J_G_theta_event_and_input_requalification`: **True**
- `event_projection_math_and_Q_tangency_close`: **True**
- `fresh_four_HVPs_bind_to_both_same_point_directions`: **True**
- `both_arms_supported_and_each_stopped_at_first_passing_slot`: **True**
- `selected_winner_matches_R_then_J_then_baseline_tie_rule`: **True**
- `one_P2_closed_commit_no_new_point_operators`: **True**
- `P2_source_input_runtime_deadline_and_control_hash_close`: **True**
