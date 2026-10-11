---
type: "query"
date: "2026-10-11T16:41:39.688881+00:00"
question: "Eight-team diagnosis of PR297 common-face support mismatch"
contributor: "graphify"
source_nodes: ["minimum_mixture_weight()", "tangent_direction()", "_euler_minmod()"]
---

# Q: Eight-team diagnosis of PR297 common-face support mismatch

## Answer

At pinned556a, original gradients reproduce exactly and forward/reverse checks agree. Highprecision projector and replay toggle do not remove the gap. VJP decomposition attributes tangent3.84e-13 mainly to losscotangent variation from t600 residualdelta1.64e-14; Jacobian term tangent7.55e-16. A diagnostic-only one-line Euler sum reassociation keepsQ/input unchanged, gives equal side residuals and Pjump7.76e-18, and67affectedtests pass. NumericalJ/gradients change, so this screenedcandidate needs distinctsource identity and newP2/HVP/candidate qualification; no productionpatch oroptimizercommit. All8 Luna/high teams reports at graphify-out/fv-root-cause-20260919/common_face_diagnosis_20261012/FINDINGS.md. Existingepsilon/frozenraw/source preserved; historicalFVmilestone70percent.

## Source Nodes

- minimum_mixture_weight()
- tangent_direction()
- _euler_minmod()