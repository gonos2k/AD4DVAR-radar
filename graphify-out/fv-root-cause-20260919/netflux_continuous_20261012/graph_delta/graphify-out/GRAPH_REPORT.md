# Graph Report - /Users/yhlee/ADVAR/graphify-out/fv-root-cause-20260919/netflux_continuous_20261012/graph_delta  (2026-10-12)

## Corpus Check
- Corpus is ~11,306 words - fits in a single context window. You may not need a graph.

## Summary
- 124 nodes · 221 edges · 17 communities (16 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY__boundary_budget()  _check_minmod_support()|_boundary_budget() / _check_minmod_support()]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY__check_nonnegative()  _check_support()|_check_nonnegative() / _check_support()]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY__euler_minmod()  FVStepResult|_euler_minmod() / FVStepResult]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY_test_fv_point_3h_netflux_continuation.py  _steps()|test_fv_point_3h_netflux_continuation.py / _steps()]]
- [[_COMMUNITY__plan()  Any|_plan() / Any]]
- [[_COMMUNITY__base()  _install_mocks()|_base() / _install_mocks()]]
- [[_COMMUNITY_test_fv_point_3h_netflux_endpoint.py  _receipt_fixture()|test_fv_point_3h_netflux_endpoint.py / _receipt_fixture()]]
- [[_COMMUNITY_Any  Path|Any / Path]]
- [[_COMMUNITY__constant_step()  Tensor|_constant_step() / Tensor]]
- [[_COMMUNITY_bounded_fv_coefficients()  Validate the fixed, gauge-anchored basis outside the AD path.|bounded_fv_coefficients() / Validate the fixed, gauge-anchored basis outside the AD path.]]
- [[_COMMUNITY__check_tensor()  _validate_boundary_schedule()|_check_tensor() / _validate_boundary_schedule()]]
- [[_COMMUNITY_observe_minmod_stages()  Observe this call context's minmod stages without replacing model code.      Dia|observe_minmod_stages() / Observe this call context's minmod stages without replacing model code.      Dia]]
- [[_COMMUNITY_Extend one fixed signed-volume branch for local kink derivatives.      ``side``  selected_face_extension()|Extend one fixed signed-volume branch for local kink derivatives.      ``side`` / selected_face_extension()]]
- [[_COMMUNITY_Scale a transported value without rounding a small integrating factor.  _scale_by_growth()|Scale a transported value without rounding a small integrating factor. / _scale_by_growth()]]

## God Nodes (most connected - your core abstractions)
1. `Tensor` - 19 edges
2. `finite_volume_step()` - 15 edges
3. `finite_volume_trajectory()` - 11 edges
4. `_check_tensor()` - 9 edges
5. `load_endpoint()` - 8 edges
6. `_run_child()` - 7 edges
7. `_base()` - 7 edges
8. `_volume_parts()` - 7 edges
9. `_euler_minmod()` - 7 edges
10. `_run_child()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `_run_child()` --calls--> `CurrentBackend`  [INFERRED]
  examples/weather_scenarios/fv_point_3h_netflux_continuation.py → examples/weather_scenarios/fv_point_3h_netflux_current_cycle.py

## Import Cycles
- None detected.

## Communities (17 total, 1 thin omitted)

### Community 0 - "_boundary_budget() / _check_minmod_support()"
Cohesion: 0.22
Nodes (13): _boundary_budget(), _check_minmod_support(), _euler_donorcell(), finite_volume_step(), Positive-coefficient Euler update for one stage., Require full-known support wherever an exterior value can enter., Return incoming/outgoing boundary mass rates for one stage.      Directional pos, Advance one constant-growth Cartesian FV SSPRK2 substep.      ``log_growth`` is (+5 more)

### Community 1 - "Any / Path"
Cohesion: 0.30
Nodes (11): Any, Path, Tensor, _bundle_members(), load_endpoint(), _load_json(), Receipt-only endpoint admission for the net-flux numerical evaluator., Admit the prior net-flux cycle from pinned bytes only; no FV or seed calls. (+3 more)

### Community 2 - "_check_nonnegative() / _check_support()"
Cohesion: 0.33
Nodes (10): _check_nonnegative(), _check_support(), face_volume_fluxes(), finite_volume_trajectory(), _flatten_step_boundaries(), Construct shared Cartesian face volume fluxes from vertex ``psi``.      ``psi``, Pack one interval's stage edges in the replay block's tensor order., Advance a stationary-ψ FV trajectory and return lead-boundary frames.      ψ and (+2 more)

### Community 3 - "Any / Path"
Cohesion: 0.44
Nodes (8): Any, Path, CurrentBackend, load_fixed_case(), load_plan(), main(), One current-gradient cycle after explicit net-flux numerical requalification., _run_child()

### Community 4 - "_euler_minmod() / FVStepResult"
Cohesion: 0.28
Nodes (8): _euler_minmod(), FVStepResult, _minmod(), _muscl_slopes(), Small finite-volume transport baseline for the spatial-flow candidate.  The modu, Result and mass-budget terms for one SSPRK2 finite-volume step., Return minmod slopes with zero reconstruction on the perimeter., Advance one minmod-MUSCL Euler stage with supplied exterior traces.

### Community 5 - "Any / Path"
Cohesion: 0.44
Nodes (8): Any, Path, convergence_progress(), load_plan(), main(), Fixed-version continuous current-gradient trial from the admitted net-flux endpo, A small three-step decrease is a comparison trigger, never a root claim., _run_child()

### Community 6 - "test_fv_point_3h_netflux_continuation.py / _steps()"
Cohesion: 0.33
Nodes (6): _steps(), _synthetic_loop(), test_later_point_mismatch_preserves_first_commit(), test_postcommit_residual_candidate_precedes_cap(), test_rolling_stagnation_uses_three_commits_and_both_metrics(), test_two_steps_refresh_each_point_and_keep_new_version()

### Community 7 - "_plan() / Any"
Cohesion: 0.25
Nodes (3): _plan(), Any, test_plan_only_allows_declared_transport_transition()

### Community 8 - "_base() / _install_mocks()"
Cohesion: 0.50
Nodes (8): _base(), _install_mocks(), Any, test_release_gate_precedes_unresolved_mixture(), test_requalification_closes_fresh_repeat_without_gating_on_old_p2(), test_requalification_rejects_changed_trusted_input_identity(), test_requalification_rejects_invalid_new_evaluator_closures(), test_requalification_rejects_reused_transport_identity()

### Community 9 - "test_fv_point_3h_netflux_endpoint.py / _receipt_fixture()"
Cohesion: 0.38
Nodes (3): _receipt_fixture(), test_endpoint_rejects_a_broken_p2_prefix_without_fv(), test_endpoint_requires_final_repeat_to_satisfy_recorded_armijo_bound()

### Community 10 - "Any / Path"
Cohesion: 0.33
Nodes (5): Any, Path, Fresh same-control P2 closure for the proposed net-flux evaluator., Recompute a fresh current P2 closure without admitting the archived P2., requalify_base()

### Community 11 - "_constant_step() / Tensor"
Cohesion: 0.60
Nodes (4): _constant_step(), Tensor, test_balanced_update_preserves_constant_amplitude_derivative(), test_constant_field_is_stationary_at_balanced_nonzero_flux()

### Community 12 - "bounded_fv_coefficients() / Validate the fixed, gauge-anchored basis outside the AD path."
Cohesion: 0.50
Nodes (4): bounded_fv_coefficients(), Validate the fixed, gauge-anchored basis outside the AD path., Map unconstrained controls to a fixed CFL-safe coefficient box.      For diverge, _validate_psi_basis()

### Community 13 - "_check_tensor() / _validate_boundary_schedule()"
Cohesion: 0.83
Nodes (4): _check_tensor(), _validate_boundary_schedule(), _validate_edges(), dtype

### Community 14 - "observe_minmod_stages() / Observe this call context's minmod stages without replacing model code.      Dia"
Cohesion: 0.67
Nodes (3): observe_minmod_stages(), Observe this call context's minmod stages without replacing model code.      Dia, _MinmodStageObserver

### Community 15 - "Extend one fixed signed-volume branch for local kink derivatives.      ``side`` / selected_face_extension()"
Cohesion: 0.67
Nodes (3): Extend one fixed signed-volume branch for local kink derivatives.      ``side``, selected_face_extension(), _SelectedFace

## Knowledge Gaps
- **8 isolated node(s):** `Path`, `Tensor`, `Any`, `Path`, `Tensor` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tensor` connect `_check_nonnegative() / _check_support()` to `_boundary_budget() / _check_minmod_support()`, `_euler_minmod() / FVStepResult`, `bounded_fv_coefficients() / Validate the fixed, gauge-anchored basis outside the AD path.`, `_check_tensor() / _validate_boundary_schedule()`, `Scale a transported value without rounding a small integrating factor. / _scale_by_growth()`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `finite_volume_step()` connect `_boundary_budget() / _check_minmod_support()` to `Scale a transported value without rounding a small integrating factor. / _scale_by_growth()`, `_check_nonnegative() / _check_support()`, `_euler_minmod() / FVStepResult`, `_check_tensor() / _validate_boundary_schedule()`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `observe_minmod_stages()` connect `observe_minmod_stages() / Observe this call context's minmod stages without replacing model code.      Dia` to `_euler_minmod() / FVStepResult`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `Fixed-version continuous current-gradient trial from the admitted net-flux endpo`, `A small three-step decrease is a comparison trigger, never a root claim.`, `One current-gradient cycle after explicit net-flux numerical requalification.` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._