# GREEN saved-data comparison: PR285 step 2 candidate boundary

This audit reads only the saved PR285 arrays at base `1825954e…`. Candidate 10 (α=`0.0015410398692810362`) is the adjacent rejected point; candidate 11 (α=`0.0007705199346405181`) is the accepted point. No FV or derivative operator was rerun.

Candidate 10 passes J Armijo, finiteness, mixing minimum, paired-branch, and face checks, but fails R Armijo: actual R=`0.003135800529928406` exceeds its bound `0.0031277546326086017` by `8.04590e-6`. Candidate 11 passes both original J/R Armijo checks; actual R=`0.003127039112866745` is below its bound `0.003127755178315469`. The selected candidate’s independent P2 repeat is closed.

Comparing base, candidate 10, and candidate 11 branch traces on both sides finds only `choose_left` changes, at x cell `[1,2]`. Relative to base, candidate 10 switches at stages 109–112 and 178–181; candidate 11 switches at 109, 110, and 181. Candidate 10 versus 11 differs at 111, 112, 178, 179, and 180. All `left_sign`, `right_sign`, and `slope_sign` values agree, and there are zero face-sign differences over the saved traces. The targeted stage-124 selector remains right-selected (`choose_left=false`) for base and both candidates. This is pointwise saved-trace evidence, not a path certificate.

Both candidates’ actual 27-component G and model-predicted G are finite. The largest componentwise model errors are at zero-based G components 13, 12, and 14: candidate 10 errors are `+2.42198e-3`, `−1.38051e-3`, and `−9.36240e-4`; candidate 11 errors are `+7.77849e-4`, `−4.30454e-4`, and `−3.18915e-4`. These identify where the saved model mismatch concentrates without assigning unsupported physical names to residual slots.

No issue found in this saved comparison. It does not revise the committed-point result or justify widening/replacing the frozen search gates.
