# RED review — external-background Phase 1

The saved nonzero-theta run uses the fixed initial state `(0.1, -0.2, 0.15)`,
so the total-observation chain exercises the learned spatial-feature path.
The reported `background_y_nonidentity_norm` is
`||(B_y^T-I^T) g_B|| = 1.46e-9`, a nonzero score cotangent contribution; it
is not a norm or certification of the full Jacobian `B_y-I`. The total-y chain
residual is `3.43e-19`, and centered finite-reanalysis errors decrease from
`7.1e-10` to `1.8e-10`.

The fixed training update lowers training score by `4.46e-15`, while the
independent heldout score worsens by `2.16e-16`, below a meaningful learning
claim. The result therefore makes no robust heldout-improvement claim.
Checkpoint reload is exact, and targeted tests pass (`2 passed`). The proof
remains an external research composition; it does not extend typed neural-prior
application, general FV/D7, legacy FSOI, or production-learning eligibility.
