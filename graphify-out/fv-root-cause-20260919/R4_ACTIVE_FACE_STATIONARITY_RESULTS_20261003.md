# R4 structural-zero family: restricted stationarity/curvature only

One frozen-source native CPU FP64 run completed in70.7729s with sampled peak child RSS348,372,992bytes (270 samples), exit0, no monitor error or resource termination. It preserves the original26 latent prior and the original problem/input/parameter/runtime identity. The4-mode projected streamfunction gives the named qy[2,0] identically zero while all other faces/limiter inputs remain strict; this is a coordinate family within the original problem, not a physically verified minimum.

| Measured qualification | Result |
|---|---:|
| Newton corrections |2|
| PCG iterations |25,26|
| Fresh tangent gradient max |7.947620860038196e-12|
| Independent actual linear relative residuals |3.148851060284958e-11,1.4930680013821634e-11|
| Tangent Hessian minimum/maximum eigenvalue |1.031278100528556 /4772.621686532755|
| Tangent Hessian condition number |4627.870682104726|
| Hessian relative antisymmetry |2.1787461091266194e-16|
| Other-face scaled margin |0.016027678482693927|
| Interior slope scaled margin |0.00040656738986663495|
| Native ambient26-gradient diagnostic max |0.0027110269875353574|

All25 Hessian columns were freshly built using exact objective-gradient JVPs only after the matrix-free refinement; no dense Hessian was used for the Newton solves. Both Newton equations were independently recomputed. Positive curvature is numerical restricted-family evidence under a dtype-scaled criterion, not an interval certificate over a parameter neighborhood. The ambient26 derivative may choose a rounded donor side at the lifted event; it is not a classical stationary-gradient certificate.

The structural face is the only permitted zero. The54-stage checker locks every other face sign and minmod choice, maintains the fixed1e-4 margins and checks background/observed-value branches and the original pivot-latent domain. The original full26 scalar loss includes the recovered pivot prior; dropping it changes the statistical problem. Six focused foundation/observer tests passed with18 existing torch.jit warnings; error-level type checks had0errors. No full CI/package/deployment workflow was requested.

## Status

- [x] Preserve full prior/domain/data/time/boundary dependencies in the structural family.
- [x] Numerically check objective, forecast, parameter JVP, gradient and HVP against the original lifted representation.
- [x] Obtain fresh restricted tangent stationarity and full25-column numerical positive-curvature evidence with strict other branches.
- [ ] Compute both one-sided normal derivatives at eta=0 with resolved signs; currently not performed.
- [ ] Establish active-optimizer selection/persistence for declared parameter perturbations.
- [ ] Only then issue the corresponding tangent-adjoint/full-parameter response and signed nonlinear checks.

R4-R remains open. Existing finite-offset opposing slopes do not prove the normal conditions at the zero face. No original unconstrained minimum, full smooth root, uniqueness, forecast skill, identifiability, physical influence or learning result is claimed. The normal input fixture is a prospective artifact bound to this approximate restricted root, not a completed normal certificate.
