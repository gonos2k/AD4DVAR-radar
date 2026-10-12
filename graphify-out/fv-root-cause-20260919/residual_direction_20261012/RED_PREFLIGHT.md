# RED preflight — residual-direction helper

## Mathematical checks before integration

- The face projector must use the Euclidean control metric explicitly:
  `P=I-nn.T/(n.Tn)`. The normal may be scaled; the result must be invariant.
- The side-gradient jump must be resolved and parallel to the common-face
  normal. The strict-interior mixture must be tangent within dtype-scaled
  roundoff before defining `d=-P g*`.
- The curved-face action must be the derivative of the full current map along
  `d`, including any mixture-weight and changing-normal terms. A cached product
  from another direction is invalid.
- `s_raw=P DG[d]` and the equal-norm trial scaling must remain separately
  visible. Actual side and merit products must be measured; no descent conclusion
  follows from the construction alone.
- The helper must delegate directional-action evaluation to its caller and must
  not run FV, score, HVP, or candidate evaluation itself.

## Integration boundary

The helper has no base-point argument, so `model` is defined as a directional
action callback `model(direction) -> DG[direction]`, bound by the caller to the
current control. If the PR's intended caller contract gives `model` a different
meaning, update the API before integrating. This preflight covers helper
mathematics only; it does not clear launch or acceptance policy.

## Integrated seed override review

The new `seed_direction` path checks that the first model's direction agrees
with `-P g*` within a scale-aware 128-epsilon budget, then retains that exact
tensor for the callback. The configured callback also checks exact tensor-value
equality with `model.direction`, so no HVP result is silently reused at an
ULP-different recomputed seed. On a 26-control curved `tanh` face, direct AD
confirmed exact callback equality and the scalar identity below to a
`5.6e-17` absolute error.

For `G` a smooth face gradient and `L=P DG P` its tangent derivative,
self-adjointness of the restricted Hessian gives

`d=-P G`, `r_d=Ld`, `s=c r_d`,
`G · DG[s] = -c ||r_d||^2`, with `c=||d||/||r_d||`.

The curved-face probe measured `G·DG[s] = -0.42661106856487907` and
`-c||r_d||² = -0.42661106856487900`; the two side products along `s` were both
`-0.2875842563154102`. The production second model must still gate its actual
fresh `DG[s]` and merit product before comparison. This identity is conditional
on a smooth common face and the symmetric restricted derivative; it does not
extend across a face/branch transition.
