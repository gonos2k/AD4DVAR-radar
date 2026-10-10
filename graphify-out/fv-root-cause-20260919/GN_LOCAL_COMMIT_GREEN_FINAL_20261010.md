# GREEN final review: one GN local-window commit

## Disposition

The saved evidence closes one freshly requalified candidate and its independent P2 repeat, then records fresh coupled-GN readiness at that committed point. The run made exactly one optimizer commit and selected no second candidate. The result supports this local accepted step and the recorded next-point model readiness; it does not establish a minimum, root, forecast response, or future convergence.

## Receipt and commit closure

- Frozen plan SHA-256: `11e4eff337084ba9a9602b477854349969af8306926911fa49c4ab1e8e869b6e`; its 148 source and 190 archive pins matched during preflight.
- Raw child SHA-256: `0f2892badb2eeebb9ae8d11ff3f3898a7166d68e13e0279e3a3c9625c8c1ff00`. The lossless gzip SHA-256 is `4e59a33c63b16f0b0c8d62c1bccdd9f46b554427bcf8ec524bb36c802d579a94`.
- The one committed control matches the approved sample hash `311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652` at α=`5.574488376651601e-6`, θ=`0.483079578890406`, J=`0.06119247898831076`, and R=`0.004478429816264897`. Relative to the closed base, J fell by `1.34926e-8` (about `0.00002205%`) and R fell by `5.94122e-8` (about `0.00132661%`).
- The candidate and independent final repeat agree on control, θ, objective, merit, both side gradients, branch signatures, face and branch-pair checks, and unchanged source/input/runtime. P2 closure passed before the control was persisted as the confirmed commit.
- The guard completed in `56.535638915840536 s`, sampled peak RSS `1,328,414,720 bytes` under the 2 GiB cap, with exit code 0 and no resource termination. The raw receipt reports source, fixed-input, runtime, and deadline flags true.

## Independent saved-array arithmetic

The saved-only analyzer passed the child, run, and resource receipts. A separate JSON/NumPy recomputation, without importing ADVAR or running FV/derivative code, verified the coupled-GN Woodbury algebra:

- `B` is 12×26 and `S` is 12×12. Recomputing the symmetrized `I + B Bᵀ` gives maximum absolute difference 0 from saved `S`; recomputing `B t` gives difference 0 from the saved right-hand side.
- Solving saved `S v = B t` independently gives maximum component difference `1.08415e-15` from the stored solve vector, with residual norm `2.16558e-15` versus budget `2.15420e-12`. The independent Cholesky diagonal minimum is `2.15931`, consistent with the recorded positive-definite factorization.
- Recomputed `Bᵀv` differs from the saved Woodbury correction by at most `2.01228e-16`; `-t + Bᵀv` differs from the saved uncorrected direction by at most the same amount. The dense-solve direction equals the published next-point GN direction; both paired HVP histories bind to that direction and the committed control.
- Mixing the independently saved final-repeat side gradients at the committed θ reproduces the first 26 components of model residual exactly. The squared norm of the saved 27-component model residual equals the closed R exactly in the stored float64 values.
- Both final branch traces have the same signature, 360 stages, and no nonfinite/tie flag. Their normalized active-limiter gap is `9.075476315907308e-9`, about `319,315 × 128ε`; this is a pointwise margin at the committed control.

## Post-commit readiness and limits

The committed point has 24 of 24 row VJPs, one of one 12D Cholesky solve, and two of two fresh side HVPs. The row histories and started/completed HVP labels bind to the committed control and the same robust-GN direction. The saved accounting distinguishes two reused same-point HVP vectors at the original base, zero fresh HVP calls at that base, and two fresh HVP calls at the committed point. Readiness records zero candidate evaluations and zero optimizer steps after the commit.

The focused suite reports 18 passed with 18 existing PyTorch deprecation warnings; basedpyright reports zero errors, warnings, or notes. This is execution, closure, and local GN readiness evidence for one point. It does not claim global or local optimality, a smooth root, response skill, forecast accuracy, long-run convergence, or future-point behavior.
