# Graph Report - graphify-out/fv-root-cause-20260919/netflux_current_cycle_20261012/graph_delta  (2026-10-12)

## Corpus Check
- 6 files · ~7,889 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 87 nodes · 164 edges · 13 communities (11 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `37bc51b9`
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
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `Tensor` - 19 edges
2. `finite_volume_step()` - 15 edges
3. `finite_volume_trajectory()` - 11 edges
4. `_check_tensor()` - 9 edges
5. `_volume_parts()` - 7 edges
6. `_euler_minmod()` - 7 edges
7. `_base()` - 7 edges
8. `_run_child()` - 6 edges
9. `_validate_edges()` - 6 edges
10. `_euler_donorcell()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `finite_volume_step()` --references--> `FVStepResult`  [EXTRACTED]
  src/advar/transport.py → src/advar/transport.py  _Bridges community 12 → community 0_
- `bounded_fv_coefficients()` --calls--> `_check_tensor()`  [EXTRACTED]
  src/advar/transport.py → src/advar/transport.py  _Bridges community 9 → community 8_
- `_check_tensor()` --references--> `Tensor`  [EXTRACTED]
  src/advar/transport.py → src/advar/transport.py  _Bridges community 9 → community 0_
- `face_volume_fluxes()` --calls--> `_check_tensor()`  [EXTRACTED]
  src/advar/transport.py → src/advar/transport.py  _Bridges community 9 → community 11_
- `finite_volume_trajectory()` --calls--> `_check_tensor()`  [EXTRACTED]
  src/advar/transport.py → src/advar/transport.py  _Bridges community 9 → community 4_

## Import Cycles
- None detected.

## Communities (13 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (21): _boundary_budget(), _check_minmod_support(), _euler_donorcell(), _euler_minmod(), finite_volume_step(), _minmod(), _muscl_slopes(), Scale a transported value without rounding a small integrating factor. (+13 more)

### Community 1 - "Community 1"
Cohesion: 0.44
Nodes (8): Any, Path, CurrentBackend, load_fixed_case(), load_plan(), main(), One current-gradient cycle after explicit net-flux numerical requalification., _run_child()

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (3): _plan(), Any, test_plan_only_allows_declared_transport_transition()

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (8): _base(), _install_mocks(), Any, test_release_gate_precedes_unresolved_mixture(), test_requalification_closes_fresh_repeat_without_gating_on_old_p2(), test_requalification_rejects_changed_trusted_input_identity(), test_requalification_rejects_invalid_new_evaluator_closures(), test_requalification_rejects_reused_transport_identity()

### Community 4 - "Community 4"
Cohesion: 0.38
Nodes (7): _check_nonnegative(), _check_support(), finite_volume_trajectory(), _flatten_step_boundaries(), Pack one interval's stage edges in the replay block's tensor order., Advance a stationary-ψ FV trajectory and return lead-boundary frames.      ψ and, BoundarySchedule

### Community 5 - "Community 5"
Cohesion: 0.33
Nodes (5): Any, Path, Fresh same-control P2 closure for the proposed net-flux evaluator., Recompute a fresh current P2 closure without admitting the archived P2., requalify_base()

### Community 6 - "Community 6"
Cohesion: 0.50
Nodes (4): Small finite-volume transport baseline for the spatial-flow candidate.  The modu, Extend one fixed signed-volume branch for local kink derivatives.      ``side``, selected_face_extension(), _SelectedFace

### Community 7 - "Community 7"
Cohesion: 0.60
Nodes (4): _constant_step(), Tensor, test_balanced_update_preserves_constant_amplitude_derivative(), test_constant_field_is_stationary_at_balanced_nonzero_flux()

### Community 8 - "Community 8"
Cohesion: 0.50
Nodes (4): bounded_fv_coefficients(), Validate the fixed, gauge-anchored basis outside the AD path., Map unconstrained controls to a fixed CFL-safe coefficient box.      For diverge, _validate_psi_basis()

### Community 9 - "Community 9"
Cohesion: 0.83
Nodes (4): _check_tensor(), _validate_boundary_schedule(), _validate_edges(), dtype

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (3): observe_minmod_stages(), Observe this call context's minmod stages without replacing model code.      Dia, _MinmodStageObserver

## Knowledge Gaps
- **6 isolated node(s):** `Any`, `Path`, `_MinmodStageObserver`, `Tensor`, `Any` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tensor` connect `Community 0` to `Community 8`, `Community 9`, `Community 11`, `Community 4`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `finite_volume_step()` connect `Community 0` to `Community 9`, `Community 4`, `Community 12`, `Community 6`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `observe_minmod_stages()` connect `Community 10` to `Community 6`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **What connects `One current-gradient cycle after explicit net-flux numerical requalification.`, `Any`, `Path` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._