# PR168 follow-up: reusable conditional response and matrix-free bridge

Baseline: merged PR168 `8b306871`. Its three local directions and nonlinear
reanalysis evidence are reused; no new direction or nonlinear solve is added.

## Callable research boundary

`advar.local_response.compute_local_response(objective, score, control,
parameters, directions, branch_check=..., input_identity=...)` is a reusable
conditional algebraic utility. It accepts CPU FP64 vectors, checks the fixed
maximum-gradient gate `<1e-10`, solves the exact gradient-JVP operator using the
existing PCG, then checks `H^T lambda - E_c` independently with a gradient VJP.
It returns direct/indirect/total directional responses, the adjoint, actual
residual, HVP/iteration counts, mixed gradients, score control gradient, and
caller-supplied branch signature/scope/identity. It does not claim to verify an
arbitrary callback's physical assumptions or to certify higher derivatives
through the iterative solve. Callbacks must be deterministic with fixed inputs.

The actual minmod adapter is `make_research_functions` in
`examples/weather_scenarios/fv_minmod_matrix_free_probe.py`. It preserves the
validated 4x5, 26-control, full-support CPU FP64 setting: fixed masks/error
statistics/boundaries, B=y0+theta*pattern, fixed conditional MSE verification,
9 substeps per interval and all 54 RK stages. Unsupported reconstruction,
missing support, background/detection boundaries or changed nominal branch are
rejected. It reuses the existing serial branch tracer rather than copying it
into the product package. The tracer patches transport during inspection;
concurrent use of this example adapter is not supported.

Example use after preparing this bounded contract:

```python
from advar.local_response import compute_local_response

objective, score, check_branch = make_research_functions(
    observations, frozen, boundary, support, pattern, verification, expected_branch
)
response = compute_local_response(
    objective, score, control, parameters, directions,
    branch_check=check_branch, input_identity=input_identity,
)
# response.total[name] = response.direct[name] + response.indirect[name]
```

The public `compute_fv_observation_response` remains donorcell-only, with a
regression exercising that rejection. The new utility never receives the dense
Hessian or uses it as a preconditioner. PCG curvature checks cover visited
vectors, not a proof of global SPD. For this measured point, all 26 fresh HVP
columns are compared with the saved full SPD Hessian; positive finite minimum
eigenvalue is an explicit completion condition. This is pointwise evidence,
not a finite-path or global minimum certificate.

## Declared comparison criteria and measured result

Before execution: relative HVP Frobenius error <=1e-10, adjoint difference
<=1e-8, each directional response difference <=1e-6, independent transpose
residual <=1e-10. Stationarity remained `<1e-10`; no gate was relaxed.

Final `minmod_matrix_free.json` records:

- 26 basis HVP columns versus saved dense Hessian: relative difference **0**.
- Minimum dense eigenvalue **1.0151942952**; pointwise SPD reference.
- PCG **25 iterations**, **26 HVP calls** (includes its true residual check).
  The independent VJP and 26 post-solve comparison HVPs are separate operations.
- Actual transpose relative residual **9.8881638e-11**; dense/matrix-free
  adjoint relative difference **3.3854477e-12**.
- Nominal maximum gradient **6.9620403e-12**; 54-stage signature preserved.
- Current score gradients and all three mixed-gradient vectors match archived
  values exactly in this run.

| Direction | Direct | Indirect | Total | Dense-relative difference |
|---|---:|---:|---:|---:|
| sine | 3.8976559076e-3 | -4.6534484195e-3 | -7.5579251186e-4 | 8.37989e-10 |
| theta | -4.9540210142e-3 | 4.9540851133e-3 | 6.4099082931e-8 | 7.41124e-8 |
| middle-time bias | 0 | -3.0382725287e-3 | -3.0382725287e-3 | 3.07032e-13 |

Theta exhibits strong cancellation; separately returning its components and
mixed gradient prevents the small total from hiding different large terms.
The middle-time direct term is zero under this declared B/score contract.
Directions have different units/scales and are not a physical importance ranking.

## Execution and scope accounting

- Initial comparison: 44.37 seconds, sampled peak RSS 344,637,440 bytes.
  Preserved as `minmod_matrix_free_initial.json` and its wrapper records, with
  measured producer/API snapshots. This already passed aggregate comparisons.
- RED requested explicit dense-SPD gating and per-direction mixed-vector
  comparison. Final augmented run: **43.38 seconds**, sampled peak RSS
  **342,704,128 bytes**. This repeat adds those diagnostics; it is not an
  additional nonlinear reanalysis or independent meteorological case.
- The final source hashes point to the current probe/API files. Initial source
  snapshots are `minmod_matrix_free_measured.py` and `local_response_measured.py`.
- Initial focused suite: 59 passed. Final suite adds two publisher mutation cases:
  **61 distinct focused tests passed**; reruns are not summed. Native/full CPU/package CI is not requested.
- Pinned basedpyright for the new product module: 0 errors, 0 warnings.
- General minmod response eligibility, finite-path certification, product GN
  stationarity, typed mean/precision/support learning, finite impact and D7
  remain open. Matrix-free algebra alone does not certify grid scalability or
  practical forecast/learning improvement.

## Final review and demo

- GREEN and RED final reviews found no blocking issue after the SPD and mixed-vector checks.
- Aside desktop inspection confirmed the new three-direction table, actual residual,
  and scope text in the original demo HTML. Screenshot: `minmod_matrix_free_desktop.png`.
  Existing animation data is unchanged; no new mobile check is claimed.
- Graphify code-only refresh: 6,645 nodes, 65,010 edges, 194 communities; no semantic LLM extraction.
