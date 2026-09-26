# Exact-HVP curvature audit of the fixed three-hour point seed

The one guarded 26-column audit found a **negative curvature
direction in the measured analysis-objective Hessian at the fixed,
nonstationary PR #218 seed**. This confirms the SPD premise is false
at this seed and is consistent with the PR #219 PCG refusal; it does
not identify the Krylov direction that triggered that refusal. The
audit does not establish that no stationary point exists, that another
branch is indefinite, or
that an adjoint response is eligible.

The child reconstructed the unchanged 4×5 point-observation problem,
13 parameters and shifted `control[21]` seed, and rechecked the
original strict branch: 3,600 Euler, limiter-choice and face-sign
stages with the PR #218 signature SHA256
`80a1fac22b536bfedcedfda506e11d85f4b9f2a73c5610c73fa8dfdb210dfe51`.
The analysis objective value was `0.9323736303384679` and its
gradient maximum `2.254729681485403`; the seed remains far from the
`<1e-10` stationarity gate. The terminal score was not evaluated.

For each of the 26 control basis vectors, the child computed
`H e_j=JVP_c(grad_c J;e_j)` using the exact robust objective. It
formed a 26×26 matrix without Gauss–Newton or finite differences,
checked its symmetric part, then applied a **27th fresh HVP** to the
unit minimum-eigenvector. The recorded results are:

| Diagnostic | Value |
|---|---:|
| Minimum eigenvalue of symmetric part | `-0.746077701362521` |
| Maximum eigenvalue | `3433.313750146871` |
| Negative eigenvalue count | `1` |
| Relative symmetry defect | `1.0949379174674463e-15` |
| Fresh minimum-eigenpair relative residual | `1.5316195634377663e-16` |
| Predeclared sign uncertainty `delta` | `3.433313750146871e-5` |
| Exact HVP calls | `27` |

The symmetry and fresh eigenpair checks pass their CPU FP64 `1e-8`
relative gates. The negative eigenvalue lies well outside the
predeclared `[-delta,+delta]` inconclusive band. The Hessian SHA256 is
`647115b7dc0b56acae4a34d45122798e39dddde36d3973f4d972b85b0763406c`;
the sign-canonical minimum-eigenvector SHA256 is
`621cd1cdcdf2db098a7435835cf4b1bb975b8d7f16174d10aa867cc78a77067d`.
This is a reproducible finite-dimensional numerical audit, not a
rigorous interval-arithmetic eigenvalue certificate.

The child exited 0; the parent classified `curvature_verified` only
after validating the reported eigenvalues, scale, uncertainty formula
and verdict. The resource guard measured **286.918 seconds** elapsed
and 1,058 sampled child-RSS readings with peak **343,703,552 bytes**
under the 600-second wall trigger and sampled 1 GiB limit. Child
internal elapsed was **285.035 seconds**, including 4.255 seconds
source/fixture, 7.549 seconds strict branch, 2.296 seconds
objective/gradient and **270.932 seconds** exact Hessian audit.
The parent did not build the FV fixture. Source, archived input,
control, plan and after-run identities matched.

No Newton step, PCG retry, terminal score, adjoint, VJP, signed
reanalysis or physical validation was run. The test suite passed
**89 affected tests**, including **14 focused tests**, with 18
existing TorchScript deprecation warnings. Error-level basedpyright
1.39.9 reported 0 errors/warnings/notes. GREEN/RED reviewed the
predeclared gates, fixed a parent false-curvature-label acceptance
path before launch, and audited the raw result. The incremental
Graphify code refresh produced a valid graph with 8,150 nodes and
773,039 links. Exact artifact hashes are in
`FV_POINT_3H_SEED_HESSIAN_EVIDENCE.json`.

This closes the **seed curvature ambiguity** left by PR #219; it does
not close R5-R-R. The current SPD Newton–PCG policy is inappropriate
at this particular nonstationary starting point. A future search
would need a separately declared policy for negative curvature and
branch changes, followed by strict final stationarity, curvature,
adjoint, full 13-vector VJP and signed endpoint checks. The result
does not justify relaxing any final numerical gate.

Optional remaining hardening: for a future eigensolver-refusal report,
the parent could additionally reject a contradictory
`curvature_verdict` field while the child currently emits no such
field. This does not affect the verified curvature result here.
