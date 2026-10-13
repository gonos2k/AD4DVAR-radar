# Stored signed-span audit — PR304 at 2a4d

## Scope and frozen inputs

This is a receipt-only linear-model audit. It reads the PR304 2a4d base,
freshly stored `D/V/G`, side gradients, and the PR303 final repeat; it does not
run FV, HVP, score, or actual-candidate calls. Plan SHA-256:
`cd9db66869652c8807264dcdb9172f94e4c65118fc3982465e516e706502fa64`;
source-tar SHA-256:
`da936a87c96bcb5970afbed650ba00760949763652f77febfebdd9b44d6ac2b8`;
PR304 receipt-tar SHA-256:
`5fae12a5f5a8a39c1556a6e62180c399d654f837882026f650bd2c2cb7b1aa11`.
All 234 source members, 302 inherited archive pins, 10 PR304 receipt members,
and the PR303 parent receipt members match their pinned hashes and lengths.

The base is the closed `2a4dba…` control. `G` has 27 components; `D` is
26×2 and `V` is 27×2. `gminus/gplus` are the two side gradients from the final
2a4d repeat in the PR303 progress receipt. The same frozen radius
`3.751435583391765e-5` is used for the nonnegative cone and all four signed
coefficient quadrants.

## Stored-model results

The nonnegative cone selects axis 0 with coefficients
`[0.0004843149107912605, 0]`, control norm `2.3929614848884738e-5`, and
predicted `R=0.0024305357397502642`. Its original coefficient gradient
`V.T @ (G + V @ a)` is `[-7.81e-18, 0.3481196663]`: the selected positive
coefficient is stationary, while the inactive nonnegative coefficient has a
positive lower-bound derivative. Both stored side products along `D @ a` are
`-1.18234325e-6`.

The best signed quadrant is `(+,-)`, with coefficients
`[0.0007601936136195862, -1.6941850777339738e-5]`. Its movement is exactly on
the same radius; predicted `R=0.0024215440768215043`, about `0.3683%` below
the nonnegative model's predicted R, and both stored side products are
`-1.85309504e-6`. The active-radius multiplier estimate is `6068.25497`; the
quadrant KKT positive-coefficient stationarity residual is `4.93e-16`, with
zero lower-bound violation. This is a signed linear-model result, not an
actual accepted control or demonstrated nonlinear gain.

The other signed quadrants were also evaluated with the same radius. Their
best predicted R was the `(+,-)` quadrant above; the complete coefficients,
`V.T @ (G + V @ a)`, control norms, side products, predicted R, and KKT
diagnostics are in `AUDIT.json`.

Unconstrained signed QR least squares gives coefficients
`[0.07617289788980232, -0.0024583533952371563]`, predicted
`R=0.001574734576275045`, and an almost-zero coefficient-gradient residual
(`9.82e-16` L-infinity). Its control norm is `0.003757543266742791`, about
100.15 times the frozen radius, so this result is infeasible under the trial's
movement cap. It must not be presented as a candidate outcome.

The nonnegative cone's predicted reduction agrees with the stored PR304
selected model. The signed audit shows a modest additional model reduction
that may motivate—but does not establish—an actual same-radius candidate
evaluation. Nonlinear J/R acceptance, chart closure, branch consistency, and
independent full-G repeat remain untested for this signed direction.
