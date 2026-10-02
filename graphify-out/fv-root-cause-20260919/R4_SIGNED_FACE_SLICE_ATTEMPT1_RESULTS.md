# R4 original two-hole signed-face slice experiment

One frozen-source run completed under its declared guard: 189.6712 seconds, sampled peak child RSS 352,174,080 bytes in 721 samples, exit code 0, no resource termination or monitor error. The parent independently rechecked four endpoints and seven transformed Newton solves; all audits passed. Plan/probe/runner, all 25 source paths, fixed parameters and original input, seed archives, and field-correction archive hashes matched before/after. The four copied code files are byte-identical to the measured producer; after copying them, all 25 source hashes also match this checkout based on main `24c7c69`. No CI or deployment validation was run.

| Fixed eta | Result | Tangent gradient max | Original full gradient max | Partial eta slope |
|---|---|---:|---:|---:|
| -eta0 | tangent-stationary candidate | 8.8387e-12 | 7.0113e-4 | -4.01345e-4 |
| -2eta0 | tangent-stationary candidate | 6.3513e-12 | 7.6395e-4 | -4.37307e-4 |
| +eta0 | tangent-stationary candidate | 5.0659e-11 | 6.1251e-3 | +3.50615e-3 |
| +2eta0 | refused; last accepted point | 5.4135e-9 | 6.1898e-3 | +3.54319e-3 |

Seven PCG solves converged; independently recomputed relative residuals span 1.2625e-11 to 9.0843e-11. The global scaled face margins at saved endpoints remain above 1e-4. The negative-side signature differs from the positive-side signature; endpoints within each sign share their own signature.

The +2eta0 iteration-2 refusal is an actual-objective policy veto: all 16 candidates increased J by 4.60e-15 to 1.29e-14, above the approximately 5.90e-16 allowance. Thirteen candidates satisfied gradient-merit Armijo before the veto; the last three also failed Armijo. One rejected trial had tangent gradient below 1e-10, but its control was not retained and it is NOT an accepted stationary endpoint. This result preserves the frozen policy; it does not justify retrospectively relaxing it.

The recorded sigma is the partial coordinate slope g^T C_eta. It is not an established minimum-envelope derivative: tangent-Hessian nonsingularity/positive curvature and a differentiable stationary branch were not independently certified. No same-sign qualified pair brackets zero; across-sign signatures differ. Finite samples do not establish a cusp, Clarke stationarity, a limiting derivative, or root absence.

Every stored endpoint fails the original 26-control full-gradient threshold. No fresh full Hessian qualification, adjoint/VJP, nonlinear response reanalysis or physical skill test was performed. R4-R remains open; the completed contribution is a reusable face-coordinate chart plus an audited branch-local slice calculation on the unchanged original problem.

Next research decision: use this normal-direction diagnosis to design a justified search or derivative contract. Adding more finite points or lowering a tolerance is not, by itself, full-root evidence. Current refusal and supported calculation scopes stay separate. GREEN and RED reviewed the original run without rerunning it. The existing complete-Linux CI failure is unrelated and is not relabeled as passed.
