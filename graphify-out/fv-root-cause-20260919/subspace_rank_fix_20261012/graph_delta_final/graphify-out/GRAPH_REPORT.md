# Graph Report - graphify-out/fv-root-cause-20260919/subspace_rank_fix_20261012/graph_delta_final  (2026-10-12)

## Corpus Check
- 6 files · ~4,777 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 61 nodes · 109 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5e23be99`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 5|Community 5]]

## God Nodes (most connected - your core abstractions)
1. `solve_two_direction_subspace()` - 33 edges
2. `_run_child()` - 10 edges
3. `residual_model_fidelity()` - 7 edges
4. `load_plan()` - 5 edges
5. `choose_radius()` - 5 edges
6. `Tensor` - 5 edges
7. `composed_model()` - 4 edges
8. `Tensor` - 4 edges
9. `chart_within_radius()` - 4 edges
10. `main()` - 4 edges

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

## Communities (6 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.36
Nodes (12): Any, Tensor, Path, chart_within_radius(), choose_radius(), composed_model(), load_plan(), main() (+4 more)

### Community 1 - "Community 1"
Cohesion: 0.28
Nodes (12): test_near_rank_transition_keeps_solution_and_inactive_branch_gradients_finite(), test_rank_one_active_radius_has_finite_gradients_on_fixed_rank_manifold(), test_rank_one_exact_fit_inside_radius_is_not_sent_to_singular_lambda_zero(), test_rank_one_exact_fit_padded_to_live_26_control_27_residual_shapes(), test_rank_one_exact_fit_uses_control_metric_not_euclidean_pseudoinverse(), test_rank_one_metric_candidate_has_finite_gradients_on_fixed_rank_manifold(), test_rank_one_metric_minimum_outside_positive_orthant_uses_axis_boundary(), test_rank_one_radius_active_minimum_norm_solution_hits_radius() (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.23
Nodes (11): Tensor, test_actual_merit_gate_must_reject_a_model_that_predicts_gain(), test_fidelity_keeps_full_residual_and_step_change_errors_distinct(), _candidate_R(), _finite_vector(), _rank_one_candidate(), Small differentiable model for a nonnegative two-direction residual step., Result of minimizing a two-column linear residual model. (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.20
Nodes (9): test_column_normalization_preserves_physical_step_and_raw_coefficients(), test_deterministic_40_case_dense_angle_oracle_and_finite_gradients(), test_invalid_shapes_zero_columns_and_radius_are_refused(), test_near_antiparallel_directions_are_refused_before_cancellation(), test_radius_active_coefficients_have_implicit_kkt_gradient(), test_radius_active_kkt_is_feasible_and_beats_axes(), test_same_ray_dependence_falls_back_to_best_axis_and_flags_reduced_domain(), test_skew_control_and_residual_columns_have_radius_active_gradients() (+1 more)

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `solve_two_direction_subspace()` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.637) - this node is a cross-community bridge._
- **Why does `_run_child()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `residual_model_fidelity()` connect `Community 2` to `Community 3`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `One fresh-point gradient versus two-direction trust-radius trial.  All candidate`, `The original fixed-radius experiment keeps its frozen search length.`, `Small differentiable model for a nonnegative two-direction residual step.` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._