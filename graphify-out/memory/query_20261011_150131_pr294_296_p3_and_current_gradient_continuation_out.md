---
type: "query"
date: "2026-10-11T15:01:31.008618+00:00"
question: "PR294-296 P3 and current gradient continuation outcome"
contributor: "graphify"
source_nodes: ["minimum_mixture_weight()", "tangent_direction()"]
---

# Q: PR294-296 P3 and current gradient continuation outcome

## Answer

Local successor preserves frozen producers and checks previous P2 before native release and strict-interior mixture. 65 local tests pass; production module type checks show zero errors or warnings. One guarded run at closed 556a failed before HVPs, candidates or commits at common-face gradient-jump support: 3.837774472618545e-13 versus budget 7.975354596702747e-15. Fresh previous P2 matched exactly; outward slopes positive; confirmed 556a prefix preserved. Independent after-failure source/input/runtime checks passed. Cause unresolved, tolerances unchanged, no relaunch; raw numerical_status preflight reporting gap recorded. Details: graphify-out/fv-root-cause-20260919/GRADIENT_RESUME_FINDINGS_20261011.md and GRADIENT_RESUME_VERIFICATION_20261011.json. Exact adjoint/reanalysis/independent future verification remain incomplete; historical milestone 70 percent unchanged.

## Source Nodes

- minimum_mixture_weight()
- tangent_direction()