# RED final — bounded 7839 radius comparison

## Run and receipt closure

The frozen expansion run completed successfully under plan
`466c4ed8c8eb594e6c4f3293a204a9692dd145448eb0c85fc79c110e108f0c03` and source
bundle `ac3dd6ece38bd23425adbca6d5c436529cc4054e4b5e8fedb709ef4f7723bbed`.
The child/run hash link and all 13 offline verification checks pass, including
source pin closure, fixed-input and runtime identity, current-base identity,
fresh-HVP control identity, execution caps, resource closure, and preservation
of the last-confirmed control. The archived base remains
`7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`.

The child completed in 77.864 seconds; the outer guard recorded 79.702 seconds
and exit code 0. Sampled peak RSS was 1,096,499,200 bytes, below 2 GiB. It used
four of four fresh HVPs, 24 GN row VJPs, and one dense solve. It evaluated zero
candidates, performed zero commits, and evaluated no score. The source,
fixed-input, and runtime closure flags are true. Current and last-confirmed
control hashes are unchanged from the admitted base.

## Fresh KKT and radius result

At the freshly rebuilt full-rank residual derivative (`rank=2`), the original
gradient-axis point passed its KKT and sign-resolution checks. Its inactive
axis multiplier was `0.10490553098501357` with a sign budget of
`5.101632733755189e-15`. The fresh inactive-axis activation radius is
`5.001297266163082e-5`, which is `2.6663378` times the original radius
`1.8757177916958825e-5`. The frozen cap is twice the original radius,
`3.751435583391765e-5`, so the fresh model did not justify GN entry within the
allowed search radius. At that cap, the inactive-axis multiplier remains
positive at approximately `0.04194979`.

As an independent stored-vector check, solving the current two-direction local
model at the cap still selects `axis_0`, with normalized coefficients
`[3.751435583391765e-5, 0]`. Its predicted residual-merit gain over the
gradient axis is zero. No candidate existed, so actual J/R gain, vector-model
fidelity, and repeat closure are not applicable. The numerical status
`no_resolved_subspace_entry_within_frozen_radius_cap` is consistent with the
fresh KKT geometry and the frozen cap; it is not an optimizer failure.

## Limits

This was one bounded current-point diagnostic. It establishes no benefit from
radius expansion at 7839 because the resolved second-direction entry lies
outside the unchanged `2x` cap; no finite expanded-radius candidate was
attempted. It provides no evidence about convergence, a normality root,
an eligible minimum, exact constrained curvature or adjoint validity, response
improvement, or forecast skill. RED performed receipt and stored-vector
arithmetic only; it ran no FV, HVP, candidate, optimizer, repeat, or score
evaluation.
