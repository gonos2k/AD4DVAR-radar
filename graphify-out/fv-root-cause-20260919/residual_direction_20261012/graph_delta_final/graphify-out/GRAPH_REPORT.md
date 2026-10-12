# Graph Report - graphify-out/fv-root-cause-20260919/residual_direction_20261012/graph_delta_final  (2026-10-12)

## Corpus Check
- 6 files · ~3,709 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 50 nodes · 82 edges · 8 communities (6 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c145f69f`
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

## God Nodes (most connected - your core abstractions)
1. `_run_child()` - 11 edges
2. `Tensor` - 6 edges
3. `build_residual_direction()` - 6 edges
4. `load_plan()` - 5 edges
5. `choose_radius()` - 5 edges
6. `_restore_pinned_tree()` - 5 edges
7. `_run_residual_trial()` - 5 edges
8. `composed_model()` - 4 edges
9. `chart_within_radius()` - 4 edges
10. `direction_specs()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `load_plan()` --references--> `Any`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py → examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py  _Bridges community 5 → community 2_
- `choose_radius()` --references--> `Any`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py → examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py  _Bridges community 2 → community 7_
- `_run_child()` --calls--> `choose_radius()`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py → examples/weather_scenarios/fv_point_3h_netflux_subspace_trial.py  _Bridges community 7 → community 5_

## Import Cycles
- None detected.

## Communities (8 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.38
Nodes (9): TempPathFactory, historical_root(), Path, Restore PR301 source and archived inputs from their actual pinned bytes., _restore_pinned_tree(), _run_historical_admission(), _sha(), test_endpoint_rejects_a_changed_receipt_manifest() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.33
Nodes (8): dtype, Any, Tensor, build_residual_direction(), _finite_vector(), Build a face-tangent residual direction from a supplied derivative action.  The, Return normalized ``s=P DG[d]`` and diagnostics for a smooth common face.      T, _roundoff()

### Community 2 - "Community 2"
Cohesion: 0.42
Nodes (8): Any, Tensor, chart_within_radius(), composed_model(), direction_specs(), model_fidelity(), next_direction(), One fresh-point gradient versus two-direction trust-radius trial.  All candidate

### Community 4 - "Community 4"
Cohesion: 0.47
Nodes (5): MonkeyPatch, Path, _run_residual_trial(), test_configured_trial_orders_seed_pair_residual_action_and_second_pair(), test_failed_residual_arm_repeat_cannot_fall_back_to_single_commit()

### Community 5 - "Community 5"
Cohesion: 1.00
Nodes (4): Path, load_plan(), main(), _run_child()

### Community 6 - "Community 6"
Cohesion: 0.67
Nodes (3): MonkeyPatch, test_endpoint_rejects_a_changed_progress_manifest(), test_receipt_endpoint_closes_nine_progress_steps_without_forward_evaluation()

## Knowledge Gaps
- **5 isolated node(s):** `dtype`, `Any`, `TempPathFactory`, `MonkeyPatch`, `Path`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_run_child()` connect `Community 5` to `Community 2`, `Community 7`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `choose_radius()` connect `Community 7` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `One fresh-point gradient versus two-direction trust-radius trial.  All candidate`, `The original fixed-radius experiment keeps its frozen search length.`, `dtype` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._