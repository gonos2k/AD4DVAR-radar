# GREEN final — residual-direction comparison

## Frozen inputs

The direction-trial plan hash is
`cd9db66869652c8807264dcdb9172f94e4c65118fc3982465e516e706502fa64` and the
source bundle hash is
`da936a87c96bcb5970afbed650ba00760949763652f77febfebdd9b44d6ac2b8`. I
independently read the plan and tar, confirmed all 234 tar members match their
source hashes, and checked all 234 current source files and 302 current archive
files against the frozen plan. There were no mismatches.

## Guarded comparison

The completed child hash is
`d8d36a361834439968a188ba669d5ea6352a0cc0f0e39288ec4a290b016dd014`. The run
started and completed exactly four HVPs, in order: both sides for
`baseline_tangent`, then both sides for `residual_merit`. GN row starts and
completions were zero; dense solve starts and completions were also zero. Both
arms independently produced accepted candidates and passed repeat, P2, and
full-G closure. The single committed step closes from the admitted 2a4d control
to `fb0a2efaeefe3a8b07a9385a81260757e4923b181a9430c384cfb96b8d8ea7b4`.

Both arms returned the same candidate control and the same repeated values:
`J=0.060988878373729814` and `R=0.002430534626706718`, from base
`J=0.06099005811764152` and `R=0.002441269566693863`. The two-direction arm's
coefficients were `[0.0004843149107912605, 0.0]`; the baseline arm's were
`[0.00048431491079126057, 0.0]`. The actual J/R tie selected the baseline arm
under the configured ordering, so this run demonstrates no additional gain
from the residual direction. It still completed the required two-arm
comparison and did not fall back after a missing or unqualified second arm.

The residual direction retained the seed norm: `||d||=0.04940920528296183`
and `||s||=0.04940920528296184`. Its stored control-space derivative action
was the first model's `residual_direction[:26]`. Reconstructing the implied
projected vector from that action and `s` gives a normal component supported
only on flow coordinates to `2.3e-16` outside the flow block and a projection
orthogonality residual of `3.0e-16`. The frozen source and staged-trial test
confirm that `s` is built from the baseline model's same-direction action,
then receives its own second paired HVP.

For coordinate interpretation, controls `[0:20]` are field controls,
`[20:25]` are flow coefficients, and `[25]` is the growth control. The
candidate displacement norms for these groups were respectively
`2.41093e-4`, `6.84511e-5`, and `1.20558e-5`; these control-space values do not
identify observed reflectivity, water mass, or air wind. The separate terminal
E-only observation evaluated `E(7839)=7.386235570851804` and
`E(2a4d)=7.384565484459929`, with `selection_use=false`; it did not participate
in this J/R comparison.

The guarded run completed in 54.00 seconds with 929,120,256 bytes sampled peak
RSS, below the 660-second and 2-GiB caps. Source, fixed input, and runtime
identity checks passed. Root, minimum, response, score, terminal-E recompute,
and post-commit-readiness claims were all false.

## Handoff artifacts

The packed receipt SHA is
`5fae12a5f5a8a39c1556a6e62180c399d654f837882026f650bd2c2cb7b1aa11`; it has
exactly 10 manifest members. Every packed member's byte length and SHA match
the manifest, and every packed byte string matches its loose run file. The
manifest's source-bundle SHA matches the frozen source tar. The prepared PR
body reports no residual-direction gain, keeps the two E calls separate from
the J/R selection, and states that neither the new point nor the matched score
establishes an original minimum, exact response, adjoint, or independent
forecast skill.
