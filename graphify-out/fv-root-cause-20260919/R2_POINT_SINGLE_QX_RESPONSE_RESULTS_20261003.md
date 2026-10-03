# Original point single-qx conditional response: attempt 2

The guarded response completed for the original full-valid, zero-centered-prior point problem, conditioned on the structural `qx[3,4]=0` face and a tested approximate tangent root. The run issued a 13-parameter local VJP and passed the four declared signed nonlinear endpoint checks at two step sizes. This is conditional selected-face evidence, not a classical full-26-control root or a physical-skill result.

## Response and endpoint checks

The nominal 25D tangent gradient maximum is `1.96876e-12`; its restricted-Hessian eigenvalues range from `0.920347` to `4457.145` and the 25D numerical curvature audit qualifies. The four endpoint gradients range from `4.81e-13` to `3.85e-12`; each endpoint has its own qualified 25D curvature audit and independent actual-parameter qx normal. All endpoint traces retain the same 54-stage branch signature `f72bbd2a…74283820`; the smallest recorded other-face/slope margins are `0.00592760` / `0.000840557`.

For the declared middle-four-observation direction, the direct score term is zero and the total response is `-0.004483496197405623`. The independently checked adjoint relative residual is `2.39750e-11` (20 PCG iterations); the tangent predictor true relative residual is `1.62926e-11` (16 iterations). Both are below `1e-10`.

| Per-slot step (dBZ) | Signed central score derivative | Relative error vs. conditional response | Absolute error |
|---:|---:|---:|---:|
| `0.00025` | `-0.004483496417546397` | `4.9100e-8` | `2.2014e-10` |
| `0.000125` | `-0.004483496252050308` | `1.2188e-8` | `5.4645e-11` |

Both pass the declared `1e-4` relative gate, and the smaller step has lower absolute error. Each signed endpoint uses its own actual parameter vector for refinement, objective, score, branch, 25D curvature, and two-sided 80-digit normal; the captured symmetric whitener and other fixed inputs are retained.

## Execution and provenance

Parent and child report `completed` / `restricted_active_face_response_numerically_supported_at_tested_points`, exit 0, with no resource termination or monitor error. The child used 214.321 seconds and sampled peak RSS `358,612,992` bytes under the 600-second / 1-GiB guard. Parent preflight, child before/after source maps, and fixed input identity are unchanged. Release2 SHA256 is `e3f4f083…edd2ad24`; normal input SHA256 is `dd851944…0d6313e4`; independent normal SHA256 is `ae6e65aa…7974fb38`; plan SHA256 is `1aae37e9…b24408`. The response raw SHA256 is `cb3283e0ee70518700a6ee5e2ade3f118683a42bbede26433e9e96aed231a59f`.

Attempt 1 is preserved as a setup failure before response calculation: its producer referenced nonexistent `FVPointResearchProblem._observation_values`. The amendment corrected the fixture from the real observation layout and added a real fixed-problem regression; its eight-test result is recorded there. Attempt 2 used the corrected producer and the same numerical policy.

The original zero-centered 26-term prior, original 13 parameters, point observations, correlation whitening, background dependency, boundaries, time, growth, and score remain in scope. `full_root_claim`, `original_full26_classical_root_supported`, `uniform_parameter_neighborhood_proved`, and `physical_validated` are all false. No global or unique optimizer, full-26 smooth root, finite-impact accuracy, or physical-weather claim follows.
