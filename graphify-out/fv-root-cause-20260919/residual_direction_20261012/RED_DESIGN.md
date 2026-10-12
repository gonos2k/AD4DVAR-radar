# RED design — projected residual direction on a curved common face

## Domain and equations

At one control point, let `n` be a nonzero Euclidean face normal and let
`g-`, `g+` be gradients from the two selected sides. This helper supports the
regular, smooth common-face case only: the side-gradient jump must be resolved
and parallel to `n`, and its strict-interior minimum-norm mixture

`theta* = -g- · (g+ - g-) / ||g+ - g-||^2`,
`g* = (1-theta*) g- + theta* g+`

must be tangent to the face within a dtype-scaled roundoff budget. With the
Euclidean control metric, the orthogonal face projector is

`P = I - n n^T / (n^T n)`.

The seed descent is `d = -P g*`. The supplied directional-action callback
returns the current raw derivative `DG[d]`; the helper forms
`s_raw = P DG[d]`. It returns `s = s_raw ||d||/||s_raw||` by default, retaining
the raw derivative and scale in diagnostics. This equal-norm scaling only makes
the two directions comparable as bounded trial directions; it does not change
the derivative `DG[d]` or establish a descent property for `s`.

For a curved face, `DG[d]` must differentiate the full current control map
`G(c)` along the ambient tangent `d`. In particular, if `G` includes a
control-dependent mixture or a face projection, that derivative must include
the mixture-weight and changing-normal terms. Projection of the result uses
the current normal at the base point. The retained-coordinate chart and its
actual pivot are checked using the existing `_validate_direction_override`
contract.

## Evidence gates and limits

The helper measures, rather than assumes, `g- · s`, `g+ · s`, and
`g* · s_raw`; these are reported with scale-aware roundoff budgets. A favorable
seed derivative or common-face jump does not prove that either side descends
along the returned residual direction, or that the residual merit decreases.
An adverse finite curvature result remains a conditional direction and is
reported through false descent gates. The consumer must recompute fresh actions
after a point or direction changes.

The helper does not evaluate FV objectives, forecasts, scores, branches, HVPs,
or acceptance. It makes no minimum, root, response, or forecast claim. It does
not model face transitions or nonsmooth/non-common jumps; those inputs are
rejected. Near unresolved normals, jumps, mixtures, or residual directions,
normalization is refused instead of hiding the scale with an arbitrary floor.

## RED counterexamples and regression evidence

The focused tests use a 26-variable tanh face and compare the callback action
against independent direct autodiff of a projected smooth gradient, including
the chart-path derivative. A planar scaling test checks invariance to normal
rescaling under the Euclidean metric. Separate tests reject a jump with a
tangential component, an unresolved jump, and a chart mismatch. A concave
curvature action checks that finite adverse side and merit products are marked
as failed gates rather than treated as descent.
