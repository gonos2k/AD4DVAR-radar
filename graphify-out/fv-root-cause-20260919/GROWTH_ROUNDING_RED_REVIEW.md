# RED review — growth rounding hypothesis

The dominant impact control (index 57603) is the scalar log-growth control.
At the saved c4 checkpoint its value is `-2.766764778170254e-4`; with
`max_log_growth_per_step=0.30010459245033816` and 64 substeps this gives
`log_growth_step=-1.2973731171434402e-6`.

`transport.py:478-489` evaluates both `exp(g)` and `exp(-g)`, and
`transport.py:487-536` uses those factors at the physical SSPRK stages. In
FP64, `exp(g)=0.9999987026277244`, while `expm1(g)=-1.2973722755553016e-6`
versus `exp(g)-1=-1.2973722756104422e-6`; the baseline exponential is not
near the unit-rounding threshold, but a one-control-ULP perturbation maps to
about `2.5e-22` in `g`, below the `exp(g)` ULP (`1.11e-16`) and near one
`expm1(g)` ULP (`2.12e-22`). This is consistent with the observed one-ULP
finite-gradient change (`5.42e-20`) versus one-column HVP prediction
(`1.04e-13`), but does not prove growth is the sole stall cause.

A blanket `expm1` rewrite is unsafe. Replacing `growth*value` with
`value + expm1(growth_log)*value` at `transport.py:487`, `507-510`, `523`,
`529`, or the inverse-scaled budget at `556-570` can lose nonnegativity by
cancellation for strong decay, overflow large traces, and alter the transformed
transport-budget identity. Existing tests explicitly protect tiny/subnormal
traces and extreme growth (`tests/test_fv_numerical_range.py:47-133`) and
analytic growth derivatives (`:102-113`). A safe experiment would need a
small-|g| branch with positivity/overflow proofs and budget/AD regressions; no
product change or gate adjustment is justified from this diagnostic alone.
