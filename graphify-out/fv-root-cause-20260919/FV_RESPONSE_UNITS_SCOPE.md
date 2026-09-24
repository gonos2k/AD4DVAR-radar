# G7 bounded response identity and coordinate-unit check

The exact conditional response at a smooth stationary point is

```text
H^T λ = E_c
g_p = E_p − J_cp^T λ
s_d = g_p^T d
```

For nonsingular diagonal coordinate changes `c = S z`, `p = L η`, and a
positive constant objective scale `J_new = a J`, the same physical
parameter perturbation has `dη = L^-1 d`. The parameter gradient transforms
as `g_η = L^T g_p`; therefore `g_η^T dη = g_p^T d`. The direct and indirect
directional components are separately invariant. The new analytic regression
checks these three scalar components and the gradient transformation on the
existing three-control/two-parameter quadratic test, using nonuniform `S`
and `L` and `a=1.7`. It does not re-run FV or prove solver iteration counts
are invariant under rescaling.

The 86-control reanalysis producer already records the problem/input and
current source identifiers, nominal control, branch selector/face hashes,
middle-time direction hash and dBZ units, and each pair's `h` plus its signed
endpoints' parameter/control hashes and branches. The expanded toy orchestration test now
asserts these fields, source stability, the applied `p ± h d` and
`c ± h c_dot` predictors, and the two accepted step sizes,
`0.001` and `0.0005`, against an analytic 86/241 substitute. This protects
the **recording contract** without repeating the 309-second FV run.
The archived FV result itself remains limited to seed A, one middle-time
common-bias direction and its two valid local step sizes (0.00025 and
0.000125); the toy's accepted sizes are different because its branch does
not refuse the first two.

The two affected test modules report **16 passed**, with 18 existing
TorchScript warnings. Pinned basedpyright reports 0 errors/warnings/notes;
the existing heterogeneous toy dictionaries were explicitly annotated for
that check without changing test behavior. No numerical thresholds,
objective scaling or response publication rules were changed. A fixed absolute `1e-10` control-gradient
gate is coordinate and objective-scale dependent; passing that number alone
does not bound state error in weak-curvature directions. Future generalized
profiles need a declared control/objective unit policy and condition
diagnostics before comparing stationarity across coordinates. Existing legacy
tensor digests hash bytes under fixed CPU FP64 layouts; a general identity
format should version dtype, shape and units without rewriting archived
hashes. No new-profile FV stationary response or signed reanalysis was
performed here, so G7 remains open beyond this algebraic and record-contract
check.
