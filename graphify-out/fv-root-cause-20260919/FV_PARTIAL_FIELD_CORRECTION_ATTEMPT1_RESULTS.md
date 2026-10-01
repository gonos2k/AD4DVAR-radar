# R4-R fixed-dynamics field correction

The one predeclared field-block Newton–PCG run completed and reached
the original `<1e-10` field-gradient threshold. Starting from the
hash-pinned PR #212 alternate seed, it changed only the twenty field
controls; all six flow/growth controls remained byte-identical.
The full gradient still has a dynamic residual, so **R4-R remains open**.
This is a margin-supported block-stationary handoff, not a full root
or an issued sensitivity.

| Measure | Seed | Final |
|---|---:|---:|
| Field gradient infinity norm | `0.0068481175317` | `7.0329883410e-12` |
| Full gradient infinity norm | `0.0068481175317` | `0.0061284025566` |
| Dynamic gradient 2-norm | `0.0076234298311` | `0.0075788864409` |
| Objective J | `0.020764967539144960` | `0.020764957942635602` |
| Scaled face margin | `0.00011662714171735749` | same exactly |
| Final scaled slope margin | — | `0.00040608748624064856` |

Three reduced Newton systems each used ten PCG iterations. All true
linear relative residuals passed `1e-10`: `2.18838e-11`,
`1.58465e-11`, `1.36432e-11`. There were four endpoint trials and
three accepted corrections, with step scales `1`, `0.5`, `1`.
Accepted field gradient maxima were `2.03845e-10`, `1.01594e-10`,
then `7.03299e-12`; the first two were not passed as stationary.
All complete 54-stage traces retained the frozen seed signature and
all six recorded face margins were exactly equal, confirming the
fixed-flow invariant. The original observations, mask, prior, objective,
parameters, boundary and verification field were preserved.

The actual objective gate rejected iteration 2's first candidate:
its increase `6.31439e-16` exceeded the predeclared roundoff allowance
`5.90176e-16`, despite field-merit decrease. Later accepted increases
were `2.42861e-17` and `5.72459e-16`, within that same allowance.
The total objective decreased by about `9.59651e-9`; this is numerical
field correction and does not establish physical forecast improvement.
The existing normalized field-merit Armijo gate remained active.

The recorded 39 HVPs comprise 30 PCG iteration products, three PCG
true-residual products, three observer audit products and three refiner
residual/slope products. PCG plus observer products total 36; the
returned refiner count matches the 39-product total. No full-control
Newton solve, final 26-column Hessian audit, score, adjoint, VJP or
perturbed reanalysis was run. Seed SPD evidence was reused only after
its existing source/input/runtime/raw checks; it is not a final-Hessian
certificate.

The final control SHA256 is
`a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26`.
Its branch SHA256 remains
`50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d`.
A separate parent audit freshly recomputed final J, reduced/full
gradients, fixed controls and the full branch; all checks passed in
`0.621` seconds without optimizer or PCG calls. The saved
`response_eligibility=margin_supported_handoff` refers only to field
stationarity and branch margins. `full_stationarity=false`, and
`response_validation=not_performed`.

The guarded child exited 0; parent execution is `completed`, numerical
status `block_stationary_candidate`. Guard elapsed time was **38.025
seconds**, with 142 sampled child-RSS observations and peak
**347,389,952 bytes** under the 300-second / sampled 1-GiB triggers.
Parent preparation and final audit are outside the child measurement.
Caller-reviewed probe SHA256 was
`30c6c1177e0a001a37751aa4f0396cb616f2f2e1d0c76f5b47472246f0373c18`;
plan SHA256 was
`391bfdbeb4f97af617dc00840700cadba36c36be8ed797e78026ba8cc3aed1f8`.
Source, PR #209/#212 raw records, fixed inputs and the actual parameter
tensor matched before and after.

The affected local suite passed **54 tests** with 18 existing TorchScript
warnings; targeted error-level basedpyright 1.39.9 reported zero
diagnostics. Graphify's code-only refresh preserved untouched records
and produced 8,381 nodes / 210,666 links. GREEN/RED reviewed the frozen
plan, final code and raw run. Full CPU/package regression and real-data
validation were not performed. A further separately declared method
must reduce the dynamic residual and pass fresh full stationarity,
curvature, branch and signed response gates to close R4-R.
