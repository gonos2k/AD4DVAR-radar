# Graph Report - graphify-out/fv-root-cause-20260919/netflux_subspace_20261012/graph_delta  (2026-10-12)

## Corpus Check
- 7 files · ~5,189 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 62 nodes · 107 edges · 7 communities (6 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f466c9a8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]

## God Nodes (most connected - your core abstractions)
1. `solve_two_direction_subspace()` - 18 edges
2. `_run_child()` - 10 edges
3. `_validate()` - 8 edges
4. `load_endpoint()` - 8 edges
5. `residual_model_fidelity()` - 7 edges
6. `load_plan()` - 5 edges
7. `_run()` - 5 edges
8. `_json()` - 4 edges
9. `composed_model()` - 4 edges
10. `chart_within_radius()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_column_normalization_preserves_physical_step_and_raw_coefficients()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_subspace_model.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_deterministic_40_case_dense_angle_oracle_and_finite_gradients()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_subspace_model.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_invalid_shapes_zero_columns_and_radius_are_refused()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_subspace_model.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_near_antiparallel_directions_are_refused_before_cancellation()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_subspace_model.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_radius_active_coefficients_have_implicit_kkt_gradient()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_subspace_model.py → examples/weather_scenarios/fv_point_3h_subspace_model.py

## Import Cycles
- None detected.

## Communities (7 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.27
Nodes (13): Any, Path, Tensor, chart_within_radius(), composed_model(), load_plan(), main(), model_fidelity() (+5 more)

### Community 1 - "Community 1"
Cohesion: 0.33
Nodes (11): Any, Path, Tensor, _json(), load_endpoint(), Receipt-only admission of the d19e net-flux direction-comparison endpoint., Load d19e from pinned receipts and arithmetic checks; never call the FV model., _read() (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.30
Nodes (11): test_column_normalization_preserves_physical_step_and_raw_coefficients(), test_deterministic_40_case_dense_angle_oracle_and_finite_gradients(), test_invalid_shapes_zero_columns_and_radius_are_refused(), test_near_antiparallel_directions_are_refused_before_cancellation(), test_radius_active_coefficients_have_implicit_kkt_gradient(), test_radius_active_kkt_is_feasible_and_beats_axes(), test_same_ray_dependence_falls_back_to_best_axis_and_flags_reduced_domain(), test_skew_control_and_residual_columns_have_radius_active_gradients() (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.31
Nodes (8): Tensor, test_actual_merit_gate_must_reject_a_model_that_predicts_gain(), test_fidelity_keeps_full_residual_and_step_change_errors_distinct(), _candidate_R(), _finite_vector(), Small differentiable model for a nonnegative two-direction residual step., Compare a predicted residual vector with the actual vector at a trial.      Repo, residual_model_fidelity()

### Community 4 - "Community 4"
Cohesion: 0.36
Nodes (5): MonkeyPatch, _run(), test_finite_candidate_with_corrupt_subspace_repeat_cannot_fallback_to_gradient(), test_GN_timeout_keeps_d19e_base_without_single_arm_commit(), test_two_direction_trial_shares_hvps_rows_repeats_and_DG_linearity()

### Community 5 - "Community 5"
Cohesion: 0.67
Nodes (3): MonkeyPatch, test_endpoint_rejects_a_changed_receipt_manifest(), test_receipt_endpoint_admits_d19e_without_model_evaluation()

## Knowledge Gaps
- **2 isolated node(s):** `Tensor`, `MonkeyPatch`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `solve_two_direction_subspace()` connect `Community 2` to `Community 0`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.243) - this node is a cross-community bridge._
- **Why does `_run_child()` connect `Community 0` to `Community 2`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **What connects `Tensor`, `Receipt-only admission of the d19e net-flux direction-comparison endpoint.`, `Load d19e from pinned receipts and arithmetic checks; never call the FV model.` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._