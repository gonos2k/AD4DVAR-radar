# R4 precision-aware same-problem correction

One new correction from the archived last-accepted +2eta0 tangent. Preserve c/p layout, observations/masks, boundaries, objective/prior/score, 54-stage strict signature, max backtracks16, PCG80, true relative residual1e-10 and tangent stationarity1e-10. Max Newton iterations1 is a smaller declared budget, not a relaxed threshold. No GN, adjoint, response or physical validation.

New trial objective policy: compute 80-digit directed interval endpoint costs from the fixed captured-input equations. Require all 36 analysis-stage choices to match the fixed nominal choices. Keep the original one-sided allowance tau=128*eps64*max(abs(Jcurrent_FP64),abs(Jcandidate_FP64),tiny). Pass the objective gate only when upper(DeltaJ)<=tau, veto a definite increase when lower(DeltaJ)>tau, otherwise refuse as precision uncertainty. Preserve the refiner's normalized gradient Armijo predicate unchanged. This changes evaluation precision, not the objective or allowed increase. Reference uncertainty and branch mismatch must refuse, not fall back to native scalar acceptance.

Runtime and original25source/input/control/parameter hashes must match before calculation. Pin new producer/reference/adapter/precision-helper and this plan before/after. Recompute final original control-space gradient and independently audit the Newton solve. A successful 25-direction endpoint is not an original26control root; keep historical refusal immutable and any new endpoint distinctly named.

One child, CPU FP64, wall120s and sampled RSS1GiB at .25s. Record numerical refusal separately from exit/resource error. No automatic repeat after a numerical refusal. No additional CI or packaging.
