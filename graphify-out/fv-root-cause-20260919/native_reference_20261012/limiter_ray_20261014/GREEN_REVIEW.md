# GREEN review — limiter ray diagnostic

The completed run is internally consistent for its stated diagnostic scope. The saved baseline cost is `0.0601191808341518`; the six requested cost evaluations and six native gradients completed. The run used 13.7856 seconds and 336,527,360 bytes peak RSS, ending at cumulative counts of 512 direct-J evaluations and 134 gradients. It reports no accepted points, Hessian calls, score calls, or minimum claim.

The saved `PLAN_SHA256.txt` matches `PLAN.json`, and `RESULT.json` points to the same plan hash. All 132 planned source files are present in `source.tar.gz`; each archived digest matches the plan, and the current files still match. All five planned input digests also match current bytes. The run records unchanged sources, inputs, and parameters. The separate synthetic source-inventory preflight failure records zero cost or gradient calls and no output directory.

The new diagnostic declared six additional gradients from a baseline of 128 and completed at 134. This satisfies its own six-gradient allocation, but exceeds the earlier chord plan's cumulative gradient cap of 128; inherited gradient-cap compliance is therefore not established. The direct-J cap is exhausted at 512. The final records report 29 affected tests passed, zero type errors, and 18 existing TorchScript warnings; no further model calls are authorized by this accounting.

The baseline native gradient contracted with the saved direction is `-6.524483932525922e-08`, matching the stored slope `-6.524483932525888e-08` to roundoff. At `alpha=2^-12`, no limiter selector changes occur; the secant slope `-6.522205353576284e-08` agrees with the selected-branch AD directional derivative `-6.522884370198173e-08` and the baseline slope.

At `alpha=2^-10`, the exact limiter selector changes at x cell `[1, 1]`, stages 47 and 48, from left to right. The selected-branch AD directional derivative is positive (`6.745659196986029e-09`) while the secant from the baseline remains negative (`-3.239171775248906e-08`). This supports a local branch-transition interpretation; the stored AD value is not an independently certified one-sided derivative. At `alpha=2^-7`, two additional changes occur at y cell `[2, 3]`, stages 123 and 124, from right to inactive, and native J is `5.689843041167819e-11` above baseline.

Exact flux-sign changes are recorded separately from limiter selectors. Some occur on the two constrained y faces (`[2, 1]` at the two smallest positive alphas and `[4, 3]` at `2^-10`); the receipt records signs, not flux magnitudes. Those sign records do not by themselves establish a materially different flux or identify a unique cause.

These samples support the reported local slope agreement and observed selector transitions along this fixed stale-metric ray. They do not certify a minimum, establish a unique cause for nonstationarity, or qualify an exact response.
