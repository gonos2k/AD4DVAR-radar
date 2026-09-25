# R2 correlated point response: preflight and direct warm-root refusal

The fixed 4×5, 26-control, 13-parameter all-detected point problem with
four off-grid dBZ observations at each of three times and fixed nonidentity
same-time correlation passed the no-solver preflight. The middle-time
direction is +1 dBZ in parameter slots 4:8; the verification target was
frozen before any solve. The guarded child exited 0 after 2.625 seconds,
with 10 RSS samples and sampled peak 313,507,840 B under the 120-second /
sampled 1-GiB cap. Its full warm minmod signature contains 54 choices and
54 face-sign records. Minimum scaled slope and face-flux margins were
`0.0007615693500034068` and `0.007752734889703382`, both above the
predeclared `1e-4` gate.

The correlated point objective at that warm control was
`0.07993508078523576`, the fixed synthetic score was
`0.00025526315789473595`, and the maximum control-gradient component was
`15.150450664433208`. The earlier diagonal point result of `14.327043`
belongs to a different observation-error model. Neither warm point is a
qualified stationary analysis for this correlated objective.

One exact-HVP Newton–PCG refinement attempt started from the same warm
control and pinned the full preflight branch signature and both margins.
The guarded child exited 2 with a **numerical refusal**, while the runner
completed normally after 7.614 seconds with 29 RSS samples and sampled peak
333,250,560 B under its 240-second / sampled 1-GiB cap. The first linear
solve called the HVP operator seven times and raised:

```text
RuntimeError: operator must be symmetric positive definite
```

Only the initial branch check ran; no Newton candidate, Armijo test,
stationary point, adjoint or signed endpoint was produced. This is evidence
that the **current SPD Newton correction policy is unsuitable at this warm
point**, not evidence that the point objective lacks a stationary minimum.
The resource, source, input and both plan/preflight identity checks stayed
stable. `response_validation` remains `not_performed`.

The first direct-file preflight launch failed at import before a child was
created; the runner's repository-root module path was then fixed and the
guarded preflight was launched once. This import correction is separate
from the numerical refusal. The focused runner/preflight regressions passed
17 tests and selected basedpyright reported 0 errors/warnings/notes before
the nominal attempt. The raw JSON, resource records and logs are retained
in `point_response_preflight_attempt1/` and `point_nominal_attempt1/`.

The next numerical policy must be defined separately: search for a smooth
local basin by decreasing the **point objective** and inspecting actual
candidate branches, then apply the unchanged exact-HVP stationarity and
response gates only at a qualified terminal point. Repeating the same warm
PCG with a higher iteration count would not address this negative-curvature
refusal. No weather forecast skill or 3-hour point-observation response is
inferred from either run.
