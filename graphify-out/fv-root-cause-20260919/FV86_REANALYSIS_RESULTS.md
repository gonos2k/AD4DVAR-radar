# 86-control local reanalysis check

User authorization: 2026-09-24 15:30 JST, total 1200-second wall time and
sampled RSS 2 GiB. Checklist: PR174_REVIEW_RESOLUTION.md (R3/R6). The baseline
is PR174 seed A; the current shared problem definition is from PR175.

The raw seed-A report is pinned by SHA256. Numerical core source hashes must
match; three refactored adapter paths are explicitly reported rather than
pretended unchanged. PR175 frozen-source parity and fresh nominal J/E, Ec/Ep,
J_cp d, branch and actual H^T lambda residual bind the reused adjoint to the
current problem. No fresh GN or second baseline optimization is performed.

Direction: +1 dBZ at each of the 80 middle-time pixels, zero elsewhere and at
theta. Baseline B=y0+theta P is unchanged by that direction. The tangent solves
H c_dot=-J_cp d, and each endpoint passes its own p +/- h*d to objective,
refinement, branch check and score. The score/verification remain fixed.

Step sequence is .001*2^-j, j=0..5. Two consecutive central differences with
relative error <=1e-4, fresh endpoint max gradient <1e-10 and unchanged 108-stage
selectors/face signs are required. The denominator is |s_adjoint|, not a larger
component scale. Both signs/refusals are retained. No finite-segment or
realistic finite-impact accuracy is inferred from endpoint checks.

## Prelaunch verification

GREEN implementation plus RED review caught a draft endpoint path that changed
the control predictor but still used baseline p. It was corrected before any
FV execution. A full 86-control/241-parameter analytic toy test now exercises
the same orchestration and requires endpoint c*=p_endpoint[:86], four distinct
parameter-endpoint calls and the correct nonzero central response. Cache
mutation tests refuse bad full direct gradients or adjoints before the tangent.
Four focused tests and the producer/runner typecheck passed before source freeze.

The old archived-source equality check and initial GN-branch-row selection were
also corrected before launch: refactored adapters are declared, core sources
remain pinned, and the branch row is selected by the refined-control hash.
No numerical tolerance, direction, operator or step schedule was changed.

## Measurements

Execution results are recorded in fv86_reanalysis.json and its separate
.resource.json/.log. An incomplete child checkpoint is never an execution
success: the outer exit code/resource state and final phase/source checks must
also pass. Final outcomes will be appended after the guarded process finishes.

## Final measured result

Process exit 0, no resource termination; 309.273 seconds, sampled maximum RSS 351,141,888 bytes. The run stayed within its 1200-second / 2-GiB budget. All input/source/baseline/plan stability checks passed.

| h (dBZ per middle-frame pixel) | Plus / minus | Central score slope | Relative difference |
|---|---|---:|---:|
| 0.001 | predictor branch refused / refused | not evaluated | not evaluated |
| 0.0005 | predictor branch refused / refused | not evaluated | not evaluated |
| 0.00025 | eligible / eligible | -2.997838136970334e-03 | 3.266414027e-07 |
| 0.000125 | eligible / eligible | -2.997838871426342e-03 | 8.164626491e-08 |

The first two step sizes were refused **before Newton correction** because the
predicted point had a different selector/face signature. This is neither a
failed stationary solve nor evidence that no stationary point exists there.
The refusals remain part of the record. Four actual endpoint refinements were
performed at the accepted two sizes; both signs were assessed independently.

Freshly checked adjoint response is -2.997839116188708e-03. Direct term is 0 and the response is indirect for this middle-time-only direction. The tangent took 34 PCG iterations with actual relative residual 7.343475183e-12. The fresh VJP residual of the reused adjoint is 8.243789281e-12.

| h | Sign | Fresh max gradient | Endpoint seconds |
|---|---|---:|---:|
| 0.00025 | plus | 3.168108484e-11 | 63.744 |
| 0.00025 | minus | 1.497927477e-11 | 62.385 |
| 0.000125 | plus | 2.527477533e-11 | 62.744 |
| 0.000125 | minus | 1.316442197e-11 | 63.174 |

All four endpoints retained the same 108-stage selectors/face signs and max gradient <1e-10. Relative-error reduction on halving h was 4.0006901, consistent with central-difference second-order truncation in this range. This ratio reuses the two measurements, not an additional experiment.

The signed adjoint-residual projection r_lambda^T c_dot is -5.252749121e-15. The tangent is itself computed iteratively; this is an interest-specific linear-solve diagnostic, not an endpoint-normality error bound or a total finite-difference error certificate.

R3 and R6 are closed **only for seed A, this middle-time direction and these
local step sizes**. The full 241-dimensional response, other initial points,
realistic finite impacts, intervening path, general minmod FSOI and physical
forecast skill remain unvalidated. No tolerance, direction, support or objective
was changed to obtain the result. No new GN solve was performed.


## Final verification and checklist

167 unique affected tests passed, with 18 existing TorchScript warnings
(`reanalysis_tests.log`); producer/runner pinned typecheck reports 0 errors,
0 warnings and 0 notes (`reanalysis_typecheck.log`). The eight new analytic
orchestration checks include signed-parameter use, corrupted cached Ec/Ep/
mixed/adjoint inputs and resetting the consecutive-pair counter after refusal.
No full CPU/package suite was run. GREEN/RED independently checked the saved
parameter/predictor arrays and the experiment's numerical/source/resource gates.

Checklist R3/R6 are done only in the scope above. Other rows remain explicitly
pending or limited. Graphify was refreshed code-only: 283 files, 6835 nodes,
65390 edges, 249 communities; no semantic LLM extraction. Publication pins both
the raw experiment and resource record and checks process completion separately
from local mathematical validation.

Aside desktop DOM and screenshot (`fv86_reanalysis_desktop.png`) show both
predictor refusals and the two passing local pairs, with the checklist linked.
The earlier run table is labeled as its then-current validation state. Original
animation frame-data hashes are unchanged; no new mobile validation is claimed.
