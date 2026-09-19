# PR165 follow-up: report integrity and spatial branches

Baseline: merged `aee38b77`. Phase1 synthetic research only.

## Closed review items

The archived 3x3 report is checked for finite stationarity/adjoint residuals,
positive Hessian eigenvalue and branch margin, and nonnegative arithmetic-consistent
FD errors before rendering. Its canonical JSON SHA256 is bound to the measured
PR165 artifact, including its source fingerprints. Mutation regressions reject
corrupted numerical evidence and missing/changed provenance. The archived
experiment and existing demo arrays are unchanged.

The RK tracer now retains each interior cell's selected slope, both slope signs,
and each face-flow sign. Two 4x5 layouts that formerly shared the `.all()` signature
are distinguished. Opposite-sign, strictly nonzero slopes select a locally constant
zero limiter and are permitted; active ties and nominal zero controlled fluxes
remain rejected. This is a strict fixture-specific local check, not a general
control-space relevance classifier or finite-path certificate.

## 26-control experiment

The new fixture adapts the existing 4x5 shape and five spatial streamfunction bases.
It uses positive detected echo (20 times the original derivative fixture), fixed
edge-echo boundaries, 60-second intervals and nine substeps per interval. These
are explicitly different from the earlier one-second joint derivative fixture.
There are 20 initial-field controls, five flow coefficients and one growth control.
The same actual minmod robust objective and forecast operator are used throughout.
Background is B(y,theta)=y[0]+theta*pattern. Verification is the frozen nominal
forecast plus .1*pattern: a conditional derivative test, not independent forecast
skill or neural-network learning.

The default product GN solve was followed by the existing dense Newton oracle.
No solver tolerance, PDE, limiter, or public donorcell-only response gate changed.

| Quantity | Measured value |
|---|---:|
| Product GN time | 24.0119 s |
| Product GN exit | step_tolerance_unverified |
| Product GN maximum gradient | 1.3348760e-4 |
| Oracle maximum gradient | 6.9620403e-12 |
| GN objective minus oracle objective | 6.4209055e-12 |
| GN-to-oracle control distance | 5.3311039e-7 |
| Dense Hessian minimum eigenvalue | 1.0151943 |
| Dense Hessian condition number | 19585.3754 |
| True adjoint relative residual | 2.6529274e-16 |
| Minimum normalized slope margin | 7.6156935e-4 |
| Euler stages inspected | 54 |

Small objective disagreement does not imply that GN meets the oracle's
stationarity criterion. The positive Hessian is a local regularized curvature
result, not global optimality or observational identifiability.

The bounded full experiment stopped at the first negative observation perturbation
(h=.001): the existing oracle exhausted eight Newton iterations without satisfying
its unchanged 1e-10 maximum-gradient gate. Thus **no 4x5 reanalysis response is
certified by this run**. `minmod_spatial_inverse.json` is the last successful
checkpoint (its `running` status is historical); the `.run.json` and `.run.log`
record exit code1 after271.27s, peak sampled RSS343048192bytes. The exact producer
is preserved as `minmod_spatial_inverse_measured.py`, matching its recorded SHA256.
Future Newton-polish RuntimeErrors now persist a failed status, direction,
step size and sign. Other branch-check failures remain visible in the run log.

## Verification

37 distinct focused tests passed (28 publisher, 6 inverse tracer, 3 joint
derivative/replay), pytest6.22s. No full CPU CI, unchanged 180-minute comparison,
or 240-grid experiment was repeated. Graphify used a cached code-only update.

The two review P2 items are closed. Full spatial reanalysis, product optimizer
stationarity, general minmod response, typed mean/precision/support learning,
finite-impact validity and whole-chain D7 remain separate open conditions.

## Failed-endpoint diagnosis

An instrumented replay of the exact failed direction, `-0.001*sin(arange(60))`
with theta unchanged, reused the saved nominal control. Accepted Newton scales
began 1, 1/2, 1/4, 1/8, 1/32. A post-hoc trace of saved accepted controls00–05
found unchanged cellwise limiter choices but changed face-flow signs versus the
nominal root in every case. Their slope margins remained .000834–.000899.
Thus the failed trajectory already left the nominal flow branch; calling this
only an insufficient iteration budget would be unsupported. This does not prove
that no perturbed stationary point exists, nor certify any intervening path.
A future local-response experiment must choose perturbations within an explicitly
measured branch neighborhood and independently verify both endpoints. No successful
4x5 response is added to the HTML. GREEN/RED review preserved this distinction.

The diagnostic was stopped after six accepted steps once the branch mismatch
was established. Its partial JSON preserves the last `running` checkpoint and
is not a completed solve; the diagnostic script and branch audit are archived
alongside it. An initial diagnostic mistakenly used only the first observation
scalar, was discarded, and contributes no evidence to the sine-direction finding.
