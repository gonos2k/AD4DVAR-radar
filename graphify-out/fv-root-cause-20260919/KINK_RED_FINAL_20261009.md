# PR265 kink runner: RED final audit

Date: 2026-10-09
Scope: read-only audit of the one guarded run using plan SHA
`83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299`. No FV
calculation was rerun and no source code was edited for this audit.

## Result

The guarded execution completed in 553.126 s with 535,576,576 sampled peak RSS
bytes, exit code 0, no monitor error, no resource termination, and all 56 HVPs
completed. The 27-variable dense solve passed with relative scaled residual
`3.1469e-18`. The probe refused the step and kept the original PR26495a control
(`95a55578…`); it did not commit or activate a candidate.

There were eight line-search slots: the first was refused because its curved
chart displacement was `0.050000515` (over the 0.05 radius), and seven trials
were evaluated. All seven passed the actual-objective Armijo check and the
production-face roundoff check. The last two also passed the scaled residual
square Armijo check (`0.00631438 <= 0.00655544` and
`0.00640408 <= 0.00655545`). Every evaluated trial failed the frozen
branch-support comparison, so no candidate was accepted. At the first
evaluated step, residual-square merit was `0.470176` against a bound of
`0.00655491`; the intermediate steps also failed that gate.

## Base-point evidence

The fresh native objective at the accepted nonzero-face point was
`0.0613075876088`; its gradient matched the accepted receipt exactly, with
infinity norm `0.105137000839`. No native AD gradient was taken at the exact
zero face. The analytically retracted zero-face chart had objective
`0.0613075874243`, formula face value zero, and production face flux
`-6.94e-18` within the recorded `5.19e-16` roundoff bound. The base analysis
traces each observed exactly 360 stages; both selected sides had the same
other-face/minmod signature and no detected tie. The non-target minimum
normalized margins were `3.07e-5` for y slopes, `1.26e-3` for x slopes,
`4.09e-6` for active limiter gaps, and `2.95e-3` / `2.68e-1` for non-target
y/x face fluxes.

Successful parity-gate details were not written into the step record. The
record contains four completed parity HVP labels and then the complete Hessian
calculation, so control flow shows the parity gate passed, but the actual
relative `J`, gradient, and HVP errors cannot be recovered from this receipt.

The step record does not store raw side-gradient vectors. They are
reconstructible from the saved coupled matrix and residual: with
`theta = 0.34623726456466775`, `mix = residual[:26]`, and
`jump = matrix[:26, 26]`,
`g_minus = mix - theta * jump` and
`g_plus = mix + (1 - theta) * jump`. These give reconstructed norms:

| Quantity | L2 norm | Infinity norm |
| --- | ---: | ---: |
| `g_minus` | 0.1284109163 | 0.1051365838 |
| `g_plus` | 0.2048719809 | 0.1635846563 |
| mixed gradient | 0.0809658012 | 0.0338955459 |
| gradient jump | 0.2878631928 | 0.2687212401 |

Both raw 26x26 one-sided Hessian matrices are saved. Their relative symmetry
errors are `1.10e-15` and `1.17e-15`. The recorded projected base-point
curvature diagnostic has minimum tangent eigenvalue `0.2206` and maximum
`4235.67`; it remains only a base-point diagnostic, with no candidate
curvature or minimum claim.

## Why the candidate was refused

At the two smallest evaluated steps, alpha `0.01314167` and `0.00657084`,
non-target face-sign patterns matched the base trace and all 360 stages had
`nonfinite_or_tie = false`. Their minmod selector traces still differed from
the base: respectively 60 and 28 stage records differed. For the smallest
step, the trace comparison found 26 `choose_left` element changes and two
y-slope/right-sign changes across the stage history. These are observed
selector changes relative to the base trace; they do not establish a tie or a
claim about simultaneous active constraints. Both candidate-side traces had
matching signatures with each other, but not with the qualified base trace.

The final closure repeated the base chart because there was no proposal. It
confirmed finite side gradients, matching side objectives, face roundoff,
and unchanged source, fixed input, runtime, and deadline. The final control
hash stayed at the original accepted point; this is refusal closure, not a
candidate closure.

## Evidence limits and follow-up

The saved run artifacts are `nonsmooth_coupled_20261009_attempt1/step.json`,
`step.run.json`, and `step.resource.json`; plan hash and source map are
recorded in the step. The exact producer and test bytes are preserved under
`kink_actual_source_20261009/` (runner SHA
`d05c2c745e88f38cd839c3816347ae636d1c9dc8e70c74df551671c9e22235a2`, test SHA
`c480f2b1194a7d5771256a0263a98846fe0587add156cf2219bec6d3b357218e`). The
archived runner hash matches the live pre-run source; its candidate gate
already checked strict stage count and tie status, and its J directional
bound used the worse of the two one-sided slopes. No post-run source change or
repeat occurred. The successful parity error values remain absent from the
receipt.

The result supports one completed, resource-bounded diagnostic with a linear
solve residual below its declared threshold and no accepted correction under the
declared objective, radius, merit, and branch-support gates. It does not
establish a full smooth root, a local minimum, a response, forecast accuracy,
or model validity.
