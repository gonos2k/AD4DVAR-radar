# Graph Report - graphify-out/fv-root-cause-20260919/gradient_resume_graph_20261011  (2026-10-11)

## Corpus Check
- 4 files · ~2,921 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 32 nodes · 58 edges · 6 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9bf27a5f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]

## God Nodes (most connected - your core abstractions)
1. `validate_endpoint()` - 7 edges
2. `load_endpoint()` - 7 edges
3. `_case()` - 7 edges
4. `_run()` - 7 edges
5. `_read_bytes()` - 5 edges
6. `_read_json()` - 5 edges
7. `load_plan()` - 5 edges
8. `_run_child()` - 5 edges
9. `_close_step()` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `validate_endpoint()` --calls--> `_sha_bytes()`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py → examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py  _Bridges community 3 → community 2_
- `load_endpoint()` --calls--> `_read_bytes()`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py → examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py  _Bridges community 4 → community 3_
- `validate_endpoint()` --calls--> `_read_bytes()`  [EXTRACTED]
  examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py → examples/weather_scenarios/fv_point_3h_current_gradient_endpoint.py  _Bridges community 4 → community 2_

## Import Cycles
- None detected.

## Communities (6 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.56
Nodes (8): _case(), _run(), test_child_classifies_release_before_unsupported_mixture(), test_child_invalid_native_pair_cannot_become_release(), test_child_p2_mismatch_cannot_become_face_release(), test_child_reminimized_theta_mismatch_still_fails(), test_child_retained_interior_reaches_chart_only_after_p2_and_release(), test_child_unresolved_jump_without_release_is_supported_unresolved_state()

### Community 1 - "Community 1"
Cohesion: 0.57
Nodes (6): Any, Path, load_plan(), main(), Fresh gradient continuation from 556a with P2-first face-release decisions.  The, _run_child()

### Community 2 - "Community 2"
Cohesion: 0.50
Nodes (5): Any, Tensor, _close_step(), Validate all three persisted commits without preparing or evaluating FV., validate_endpoint()

### Community 3 - "Community 3"
Cohesion: 0.50
Nodes (4): load_endpoint(), Receipt-only admission of the closed 556a current-gradient endpoint., Load the 556a endpoint from fixed receipts only; this function makes no FV calls, _sha_bytes()

### Community 4 - "Community 4"
Cohesion: 1.00
Nodes (3): Path, _read_bytes(), _read_json()

## Knowledge Gaps
- **1 isolated node(s):** `Tensor`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `validate_endpoint()` connect `Community 2` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `load_endpoint()` connect `Community 3` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `_close_step()` connect `Community 2` to `Community 3`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **What connects `Tensor`, `Receipt-only admission of the closed 556a current-gradient endpoint.`, `Validate all three persisted commits without preparing or evaluating FV.` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._