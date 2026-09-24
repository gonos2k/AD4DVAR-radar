# G4 bounded growth and flat-state classification

The FV transport core already accepts negative, zero and positive **interval
log growth** and divides it among substeps. The research `inspect_branches`
callback formerly rejected nonpositive substep growth even when all actual
minmod/upwind choices were smooth. The callback now delegates growth
representability, nonnegativity and CFL to the unchanged transport core,
while retaining its strict stencil-slope and nonzero-face-flux checks.

On the fixed 4×5/26-control, fully known case, growth-control values
`−0.023187493904503132`, `0`, and `+0.023187493904503132` each traversed
54 RK Euler stages. Both signed `h=1e-4` endpoints retained the nominal
cellwise limiter and face-sign signatures. Fixed-control score JVP versus
central differences had relative differences `8.90e-9`, `4.68e-9` and
`1.58e-7`, respectively. At zero growth, the objective HVP central-gradient
relative error decreased from `3.997e-7` at `h=1e-4` to `9.992e-8` at
`h=5e-5`, matching second-order truncation in this range. Exact values and
source SHA256 are in `fv_growth_branch_scope_metrics.json`.

A separate full-support **zero-face** trajectory of a uniform positive echo
field remains finite, nonnegative and unchanged at zero growth. Its strict
research branch tracer refuses the zero stencil differences. This is the
intended separation: the forward model can return a field without certifying
the full joint control derivative. The analytic two-lead zero-flow test now
checks interval multipliers `exp(g)` and `exp(2g)` for `g=-0.008, 0, +0.008`;
`g` is dimensionless log change per interval, not a per-second rate.

Existing structural-versus-sensitive zero-face and minmod tie tests were
retained. The full-control research tracer still refuses zero/near-zero face
flux, zero interior stencil differences and active same-sign minmod ties.
Strict opposite-sign slopes already have a locally constant zero limiter and
remain accepted. We did not open a generic zero-face derivative or finite
perturbation path certificate.

After an old 8×10 test expectation for zero-growth refusal failed, it was
updated to the new declared behavior. The final disjoint test groups passed:
42 focused branch/one-lead/8×10/joint derivative tests and 91 existing
transport/numerical-range/zero-face tests, with 18 existing TorchScript
warnings in each run. The three edited tests typechecked with 0 issues.
The legacy standalone `fv_minmod_inverse_probe.py` contains pre-existing
static type errors outside this small change; no whole-script type-clean
claim is made. No GN, nominal refinement, adjoint, signed reanalysis, physical
forecast validation, or long-horizon case was run for G4.
The exact commands, observed exit results, source SHA256 and unsaved-raw-log
limitation are in `fv_growth_branch_scope_manifest.json`; the numerical rows
are preserved separately in `fv_growth_branch_scope_metrics.json`.
