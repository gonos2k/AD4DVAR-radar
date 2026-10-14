# Native reoptimized parameter response

`perturbation_response.py` exposes `run_reoptimized_response`. It is inert on
import and returns `gate_pending_base` without invoking the qualification
callback or SciPy unless the caller supplies a full-control smooth-minimum
qualification for the exact base control and parameter hashes. The qualification
must establish native-gradient infinity norm below `1e-10`, positive-definite
exact full Hessian, and a stable smooth selector signature.

The caller supplies the already-loaded fixed `problem`, base tensors, the
qualification record, a `qualify(control, parameters)` callback that performs
the same exact native qualification at each requested point, the same mutable
cumulative budget shared with the parent run, and an output path. The callback
must return the final qualified control (in `qualified_control`, or `control`
for the normalized RED record), its hash, the matching parameter hash, native
gradient norm, native objective, and unchanged branch signature. If local
qualification refines a Powell result, the runner uses that returned control
for score and receipt fields while retaining the optimizer proposal hash.
RED's current path-based qualification CLI needs a wrapper that can qualify
arbitrary parameter vectors; its raw output is not a direct callback.

For each eligible base, the runner forms `F_z = ∂²J/∂z²`,
`F_zp = ∂²J/(∂z∂p)`, `E_z = ∂E/∂z`, and `E_p = ∂E/∂p` from the original native
objective and score, solves `F_zᵀ λ = E_z`, and reports `E_p - λᵀF_zp` for
parameter indices 0 and 12. It then performs four independent native Powell
reoptimizations per parameter at `p±h` and `p±h/2`, checking stationarity and
the unchanged branch before scoring each point. The observation step is its
fixed noise standard deviation divided by 100; the background step is `1e-3`
in coefficient units. Output records the fixed background-pattern RMS and
maximum absolute field change for that coefficient step.

The original A plan remains pinned. This legacy smooth-only path validates its
approved continuation source/archive scope and fixed problem/runtime if its
budget gate permits entry. The current active-intersection path uses the
separate `active_two_face_response.py` validator because transport now has the
new tuple selector API pinned by the final A source plan. Objective evaluations
and wall time use the caller's shared counters; this smooth runner creates no
new budget. The original 2,000-evaluation native-J cap is already closed, so it
returns `budget_exhausted_no_response` before source loading when that shared
counter is at or above its limit.

`_qualified_singleface_base` validates the narrow RED `qy[4,3]` event receipt:
the exact face chart, real `Q=0` event, 25-dimensional tangent stationarity,
both strict one-sided normal slopes, mixture weight, and both reduced-curvature
checks. It is a gate contract only. The smooth runner rejects active-face
qualifications; active-face reoptimization and its separate phase plan are not
implemented here. No response has been executed by this module.

Contract tests: `tests/test_fv_native_perturbation_response.py`.
