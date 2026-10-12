# Graph Report - /Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/netflux_direction_comparison_20261012/graph_delta  (2026-10-12)

## Corpus Check
- Corpus is ~18,615 words - fits in a single context window. You may not need a graph.

## Summary
- 219 nodes · 417 edges · 17 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY__boundary_budget()  bounded_fv_coefficients()|_boundary_budget() / bounded_fv_coefficients()]]
- [[_COMMUNITY_test_fv_point_3h_netflux_future_observation.py  _inputs()|test_fv_point_3h_netflux_future_observation.py / _inputs()]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_Any  dtype|Any / dtype]]
- [[_COMMUNITY_test_fv_point_3h_netflux_continuation.py  _steps()|test_fv_point_3h_netflux_continuation.py / _steps()]]
- [[_COMMUNITY__plan()  Any|_plan() / Any]]
- [[_COMMUNITY__base()  _install_mocks()|_base() / _install_mocks()]]
- [[_COMMUNITY_test_fv_point_3h_netflux_continuous_endpoint.py  _records()|test_fv_point_3h_netflux_continuous_endpoint.py / _records()]]
- [[_COMMUNITY_test_fv_point_3h_netflux_direction_comparison.py  _run()|test_fv_point_3h_netflux_direction_comparison.py / _run()]]
- [[_COMMUNITY_test_fv_point_3h_netflux_endpoint.py  _receipt_fixture()|test_fv_point_3h_netflux_endpoint.py / _receipt_fixture()]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY__constant_step()  Tensor|_constant_step() / Tensor]]
- [[_COMMUNITY_observe_minmod_stages()  Observe this call context's minmod stages without replacing model code.      Dia|observe_minmod_stages() / Observe this call context's minmod stages without replacing model code.      Dia]]

## God Nodes (most connected - your core abstractions)
1. `Tensor` - 19 edges
2. `finite_volume_step()` - 15 edges
3. `_run_child()` - 14 edges
4. `_sha()` - 14 edges
5. `finite_volume_trajectory()` - 11 edges
6. `_run_child()` - 9 edges
7. `_check_tensor()` - 9 edges
8. `_run_child()` - 8 edges
9. `load_endpoint()` - 8 edges
10. `load_endpoint()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `_run_child()` --calls--> `CurrentBackend`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_direction_comparison.py → examples/weather_scenarios/fv_point_3h_netflux_current_cycle.py
- `load_plan()` --calls--> `_sha()`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_direction_comparison.py → examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py
- `_run_child()` --calls--> `_sha()`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_direction_comparison.py → examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py
- `main()` --calls--> `_sha()`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_direction_comparison.py → examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py
- `load_plan()` --calls--> `_sha()`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_continuation.py → examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py

## Import Cycles
- None detected.

## Communities (17 total, 0 thin omitted)

### Community 0 - "_boundary_budget() / bounded_fv_coefficients()"
Cohesion: 0.10
Nodes (44): _boundary_budget(), bounded_fv_coefficients(), _check_minmod_support(), _check_nonnegative(), _check_support(), _check_tensor(), _euler_donorcell(), _euler_minmod() (+36 more)

### Community 1 - "test_fv_point_3h_netflux_future_observation.py / _inputs()"
Cohesion: 0.16
Nodes (15): _inputs(), Tensor, ScoreOnlyProblem, test_observer_calls_only_two_terminal_scores_with_same_fixed_inputs(), test_observer_checks_deadline_and_records_order_around_each_score(), test_observer_rejects_a_truth_that_is_not_the_problem_verification(), test_observer_rejects_nonfinite_or_negative_terminal_scores(), test_observer_rejects_profile_mismatch_before_scoring() (+7 more)

### Community 2 - "Any / Path"
Cohesion: 0.29
Nodes (20): Any, Path, Tensor, _endpoint_control(), _endpoint_input(), _fixed_case_tensors(), _load_case(), _load_endpoint() (+12 more)

### Community 3 - "Any / Path"
Cohesion: 0.23
Nodes (17): Any, Path, Any, Path, convergence_progress(), load_plan(), main(), Fixed-version continuous current-gradient trial from the admitted net-flux endpo (+9 more)

### Community 4 - "Any / Path"
Cohesion: 0.24
Nodes (14): Any, Path, Tensor, coupled_woodbury_direction(), _direction_model_factory(), _load_current_base(), _load_plan(), main() (+6 more)

### Community 5 - "Any / Path"
Cohesion: 0.30
Nodes (11): Any, Path, Tensor, _json(), load_endpoint(), Receipt-only admission of the closed continuous net-flux prefix., Load the last confirmed continuous point from committed receipt bytes only., _read() (+3 more)

### Community 6 - "Any / Path"
Cohesion: 0.30
Nodes (11): Any, Path, Tensor, _bundle_members(), load_endpoint(), _load_json(), Receipt-only endpoint admission for the net-flux numerical evaluator., Admit the prior net-flux cycle from pinned bytes only; no FV or seed calls. (+3 more)

### Community 7 - "Any / dtype"
Cohesion: 0.42
Nodes (9): Any, dtype, Path, choose_repeated_arm(), direction_efficiency(), load_plan(), main(), Same-point gradient/robust-GN comparison under fixed inputs and one budget. (+1 more)

### Community 8 - "test_fv_point_3h_netflux_continuation.py / _steps()"
Cohesion: 0.33
Nodes (6): _steps(), _synthetic_loop(), test_later_point_mismatch_preserves_first_commit(), test_postcommit_residual_candidate_precedes_cap(), test_rolling_stagnation_uses_three_commits_and_both_metrics(), test_two_steps_refresh_each_point_and_keep_new_version()

### Community 9 - "_plan() / Any"
Cohesion: 0.25
Nodes (3): _plan(), Any, test_plan_only_allows_declared_transport_transition()

### Community 10 - "_base() / _install_mocks()"
Cohesion: 0.50
Nodes (8): _base(), _install_mocks(), Any, test_release_gate_precedes_unresolved_mixture(), test_requalification_closes_fresh_repeat_without_gating_on_old_p2(), test_requalification_rejects_changed_trusted_input_identity(), test_requalification_rejects_invalid_new_evaluator_closures(), test_requalification_rejects_reused_transport_identity()

### Community 11 - "test_fv_point_3h_netflux_continuous_endpoint.py / _records()"
Cohesion: 0.48
Nodes (5): _records(), test_endpoint_rejects_a_damaged_committed_step_p2(), test_endpoint_rejects_any_committed_step_armijo_or_repeat_mismatch(), test_endpoint_rejects_treating_the_partial_hvp_as_complete(), test_internal_budget_stop_is_admitted_only_with_all_11_closed_prefixes()

### Community 12 - "test_fv_point_3h_netflux_direction_comparison.py / _run()"
Cohesion: 0.43
Nodes (4): _run(), test_failed_curvature_repeat_cannot_commit_baseline_fallback(), test_GN_budget_retains_baseline_evidence_and_base_without_fallback(), test_same_point_two_direction_hvps_and_repeats_commit_exact_curvature()

### Community 13 - "test_fv_point_3h_netflux_endpoint.py / _receipt_fixture()"
Cohesion: 0.38
Nodes (3): _receipt_fixture(), test_endpoint_rejects_a_broken_p2_prefix_without_fv(), test_endpoint_requires_final_repeat_to_satisfy_recorded_armijo_bound()

### Community 14 - "Any / Path"
Cohesion: 0.33
Nodes (5): Any, Path, Fresh same-control P2 closure for the proposed net-flux evaluator., Recompute a fresh current P2 closure without admitting the archived P2., requalify_base()

### Community 15 - "_constant_step() / Tensor"
Cohesion: 0.60
Nodes (4): _constant_step(), Tensor, test_balanced_update_preserves_constant_amplitude_derivative(), test_constant_field_is_stationary_at_balanced_nonzero_flux()

### Community 16 - "observe_minmod_stages() / Observe this call context's minmod stages without replacing model code.      Dia"
Cohesion: 0.67
Nodes (3): observe_minmod_stages(), Observe this call context's minmod stages without replacing model code.      Dia, _MinmodStageObserver

## Knowledge Gaps
- **12 isolated node(s):** `Path`, `Tensor`, `dtype`, `Path`, `Tensor` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_sha()` connect `Any / Path` to `Any / Path`, `Any / dtype`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `_run_child()` connect `Any / dtype` to `Any / Path`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `_sha()` (e.g. with `load_plan()` and `main()`) actually correct?**
  _`_sha()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Fixed-version continuous current-gradient trial from the admitted net-flux endpo`, `A small three-step decrease is a comparison trigger, never a root claim.`, `Path` to the rest of the system?**
  _43 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_boundary_budget() / bounded_fv_coefficients()` be split into smaller, more focused modules?**
  _Cohesion score 0.10404040404040404 - nodes in this community are weakly interconnected._