# R4 face-flux coordinate chart — local method results

This artifact records a bounded coordinate-chart foundation for the pinned
PR227 control. It does not change the physical problem, original objective, or
source control coordinates.

The chart replaces one flow latent with the selected oriented face flux
normalized by `abs(w_p) * L_p`. It preserves field, other flow, and growth
slots. Its inverse solves the pivot fraction from the normalized fixed face
row and applies `atanh` only inside the strict open interval. It rejects
unrepresentable scales/rows and finite source controls whose pivot contribution
can be erased by FP64 accumulation. That last guard is deliberately
conservative: opposing nonpivot terms can cancel in this chart's reduction
while a production combined-streamfunction reduction loses the pivot.

The metric is the normalized signed physical face flux in real arithmetic.
FP64 reduction order may differ from the production combined-streamfunction
face calculation, and the chart provides no event or branch certificate. The
transformed Hessian is `J.T @ H @ J + sum_i g_i * Hess(c_i)`; its second term can
make it indefinite away from a stationary point even when the original
Hessian is positive definite. Any original physical full-gradient `1e-10`
gate remains authoritative; a small chart-coordinate gradient is not a root
or stationarity certificate.

The fixture is face-only: it uses the explicit 5x6 production polynomial basis,
the production coefficient limiter and face-flux operator, and the
hash-pinned 26-control PR227 vector (`a78d1b8e...f8484e26`). It makes no FV
trajectory, forecast, FV/research objective, refiner, PCG, optimization, root,
or response call. The AD chain checks use a small analytical toy objective
solely to verify JVP, VJP, and Hessian composition through the chart. The
exercised profile is 4x5; no 8x10 chart-domain coverage is claimed.

Verification completed in the isolated `agent/r4-face-coordinate` worktree:

- Focused tests: 6 passed. The 18 warnings are existing `torch.jit.script`
  deprecation warnings emitted by the production coefficient limiter.
- Basedpyright 1.39.9, error level, using
  `/Users/yhlee/ADVAR/.venv/bin/python`: 0 errors, 0 warnings, 0 notes.
- `git diff --check`: clean.
- Raw command output is preserved in
  `R4_FACE_FLUX_COORDINATE_PYTEST_20261001.log` and
  `R4_FACE_FLUX_COORDINATE_TYPECHECK_20261001.log`; their SHA-256 values are
  bound in the JSON manifest.
- GREEN review: no blocker. RED review: no remaining correctness blocker;
  retained the documented conservative FP64-domain restriction above.
- Graphify AST-only extraction of the two changed code files: 34 nodes, 78
  edges, no semantic extraction or clustering. The isolated AST artifact is
  `R4_FACE_FLUX_COORDINATE_AST_20261001.json`; the main cached graph was not
  modified.
