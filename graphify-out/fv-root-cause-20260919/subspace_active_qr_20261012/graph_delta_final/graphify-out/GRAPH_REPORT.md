# Graph Report - graphify-out/fv-root-cause-20260919/subspace_active_qr_20261012/graph_delta_final  (2026-10-12)

## Corpus Check
- 5 files · ~4,250 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 49 nodes · 90 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54f8e245`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `solve_two_direction_subspace()` - 14 edges
2. `_run_child()` - 9 edges
3. `Tensor` - 7 edges
4. `_ill_conditioned_spd_case()` - 6 edges
5. `load_plan()` - 5 edges
6. `_decimal_reference()` - 5 edges
7. `_decimal_active_reference()` - 5 edges
8. `test_active_rank_two_spd_uses_stable_augmented_qr_solution_when_padded()` - 5 edges
9. `convergence_progress()` - 4 edges
10. `initial_products()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_active_rank_two_spd_coefficients_have_finite_implicit_derivatives()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_active_qr_regression.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_active_rank_two_spd_uses_stable_augmented_qr_solution_when_padded()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_active_qr_regression.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_near_singular_rank_two_inactive_solution_keeps_finite_gradients()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_active_qr_regression.py → examples/weather_scenarios/fv_point_3h_subspace_model.py
- `test_original_scaled_gradient_active_case_matches_decimal_kkt_reference()` --calls--> `solve_two_direction_subspace()`  [EXTRACTED]
  tests/test_fv_point_3h_active_qr_regression.py → examples/weather_scenarios/fv_point_3h_subspace_model.py

## Import Cycles
- 1-file cycle: `tests/test_fv_point_3h_active_qr_regression.py -> tests/test_fv_point_3h_active_qr_regression.py`

## Communities (10 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.33
Nodes (11): Decimal, _decimal_active_reference(), _decimal_reference(), _ill_conditioned_spd_case(), Tensor, Return small eigenvector, eigenvalue, and multiplier at ||beta||=.8.      Decima, Solve the represented 2D active KKT equations at 80 digit precision., test_active_rank_two_spd_coefficients_have_finite_implicit_derivatives() (+3 more)

### Community 1 - "Community 1"
Cohesion: 0.47
Nodes (6): Any, Tensor, candidate_chart(), evaluate_candidate(), initial_products(), qualify_model()

### Community 2 - "Community 2"
Cohesion: 0.53
Nodes (6): Tensor, _radius_coefficients(), _radius_inverse_product(), _rank_one_candidate(), Solve ``min_{a>=0} 0.5||G + V a||²`` subject to ``||D a||<=radius``.      ``D``, solve_two_direction_subspace()

### Community 4 - "Community 4"
Cohesion: 1.00
Nodes (4): Path, load_plan(), main(), _run_child()

### Community 5 - "Community 5"
Cohesion: 0.50
Nodes (3): Path, Regression for PR302's pinned offline current-model operator cache., test_cached_current_model_closes_pins_and_rejects_unresolved_gn_gain()

### Community 6 - "Community 6"
Cohesion: 0.50
Nodes (3): convergence_progress(), Fixed-version continuous current-gradient trial from the admitted net-flux endpo, A small three-step decrease is a comparison trigger, never a root claim.

### Community 8 - "Community 8"
Cohesion: 0.67
Nodes (3): _finite_vector(), Compare a predicted residual vector with the actual vector at a trial.      Repo, residual_model_fidelity()

## Knowledge Gaps
- **1 isolated node(s):** `Path`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `solve_two_direction_subspace()` connect `Community 2` to `Community 0`, `Community 8`, `Community 9`, `Community 7`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **What connects `Fixed-version continuous current-gradient trial from the admitted net-flux endpo`, `A small three-step decrease is a comparison trigger, never a root claim.`, `Small differentiable model for a nonnegative two-direction residual step.` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._