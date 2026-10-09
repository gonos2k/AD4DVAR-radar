# PR265 GREEN final audit

Date: 2026-10-09
Scope: read-only audit of the completed guarded attempt. No follow-up FV/model run was made, and neither the frozen source, plan, nor raw result was changed.

## Outcome and receipts

The attempt completed within its guard: child elapsed time was 551.346 s (outer monitor 553.126 s), sampled peak RSS was 535,576,576 bytes against 1 GiB, and all 56/56 HVPs completed. The dense 27-variable solve reported relative residual `3.1469141e−18`. The plan SHA is `83ea077a6d7ba615a84729a3588a92b156687dd02a8fc7320cf9b90ecf5ce299`; the `step.json` SHA is `8938215a1d553d25e86b7e4befb1c45a2b90bfef52c9a2b805648b4b89a80c69`, matching the parent receipt’s child hash. All 122 source pins and 78 archive pins match before and after.

The result is `coupled_step_refused`: no optimizer step was committed, and the final control SHA remains the accepted PR264 SHA `95a55578eea1d3b9600a210e7bdba37c21d7d7a1baf07244afc513a46df8c842`. Final closure passed for the fixed inputs, runtime, deadline, native objective, side-gradient finiteness, face value, source hashes, and unchanged control. This is an execution-complete refusal, not a stationary point or minimum result.

## Numerical audit

The exact-face chart retained the other 25 raw controls and set `Qy[4,3]` to zero. Its production reconstruction was `−6.9388939e−18`, inside the recorded `5.1931384e−16` roundoff bound. Native J at the exact-face chart was `0.0613075874243448`; the raw PR264 endpoint J was `0.0613075876087575`, a chart-lift change of `−1.8441269e−10`.

Successful parity gates are implicit in the run reaching its Hessian columns, but their individual J/g/HVP comparison values were not saved in `step.json`. The exact-face side gradients are recoverable algebraically from the stored coupled residual `F_mix`, Jacobian jump column `d = g⁺−g⁻`, and `θ=0.34623726456466775`:

```text
g⁻ = F_mix − θ d       ||g⁻||₂ = 0.1284109163, ||g⁻||∞ = 0.1051365838
g⁺ = F_mix +(1−θ)d     ||g⁺||₂ = 0.2048719809, ||g⁺||∞ = 0.1635846563
```

These reconstructed norms exactly match the stored endpoint norms; `||F_mix||₂=0.08096580118`, `||d||₂=0.28786319279`. They are recoverable evidence, but the full side vectors were not separately archived as parity outputs. The minimum tangent eigenvalue `0.22063566` is explicitly a base-point diagnostic, not a candidate curvature certificate.

Of eight trial slots, the first was refused because its actual path norm `0.0500005151` exceeded the `0.05` radius. The remaining seven had J Armijo pass and exact-face audit pass. Trials 7 and 8 also passed scaled `F²` Armijo:

| Trial | J | J change from face base | F² | F² Armijo bound | Other-branch support |
|---|---:|---:|---:|---:|---|
| 7 | 0.06126370581 | −4.38816e−5 | 0.00631438448 | 0.00655544373 | refused |
| 8 | 0.06128518866 | −2.23988e−5 | 0.00640408261 | 0.00655545235 | refused |

Both endpoint traces recorded 360/360 analysis stages and `nonfinite_or_tie=false`, with positive normalized margins. They nevertheless differed from the base selector signatures: trial 7 changed 58 x `choose_left` entries and 2 y slope/right-sign entries at stages 119–120; trial 8 changed 26 x `choose_left` entries and the same 2 y entries. Non-target face signs did not change in those two trials. Thus both candidates were rejected for changed limiter selectors despite lower J and F²; their endpoint strictness does not establish a branch-preserving path.

The actual frozen probe SHA is `d05c2c745e88f38cd839c3816347ae636d1c9dc8e70c74df551671c9e22235a2`. Its candidate predicate explicitly requires `nonfinite_or_tie=false`, exactly 360 stages, and equality of non-target choices and face-sign signatures to the exact-face base; final closure again checks no-tie and signature equality. The last two traces satisfy the count/no-tie checks but fail signature equality because their non-target limiter selectors changed. There was no post-run source patch, and the frozen plan and raw receipt remain intact. Successful parity comparison values remain an archive limit as noted above. No forecast score, reanalysis, response, or model-validity claim follows from this solve or refusal.
