# GREEN saved-data review: GN continuation from 29b52

The frozen plan `e9d8c9b356c9bb97ebc1b74bfd69ee3c2bbd6aea7cfb8020440b656cfb4a422d` requalifies current J, both side gradients, minimum θ, full G/R, trace, input, and runtime at `29b52a1377caa7fb52431bdbea25124fa827ac548eeefb9b39ce0e5f240d97c4`. The trace-124 target selector remains `choose_left=false` on both sides. Event gradients, JVPs, and projections are zero; the prior direction/HVPs are not reused.

The run completed 24 row VJPs, one 12D solve, and two fresh HVPs at 29b52. Candidate 8 of the 24-slot search was the first pass, at α=`0.006481251126157063`. One independent P2 repeat passed and committed control `bef45912ae6253673c6bd743670b374bdde76a02ba6199c464d29f6a1c863e5c`, θ=`0.6863690764427828`.

J decreased from `0.06104894133302345` to `0.061035034667252686` (`0.02278%`); R decreased from `0.0032814674811390797` to `0.003157465741517131` (`3.77885%`). The residual norm fell from `0.0572840945` to `0.0561913316`, and G∞ from `0.0268768518` to `0.0267163773`.

Independent NumPy arithmetic from saved base G, model residual direction, α, and accepted trial G gives model R=`0.0032355904777380583` and actual R=`0.003157465741517131`. The actual decrease is `2.70292×` the modeled decrease. The vector error norm is `0.00795247`, equal to `0.99670×` the actual vector change and `0.14152×` the new residual norm: the actual merit improved while the linearized vector prediction remained poor.

The target event snapshot remains right-selected (`choose_left=false`) at both the base and committed point. P2 branch, face, source/input/runtime, and deadline checks pass. This pointwise fact does not certify the full path. The run made one commit, ran no additional candidate or post-commit readiness, and carries no optimum, convergence, response, or forecast claim.

Receipts report exit 0 in `62.809 s`, sampled RSS `417,497,088` bytes under 2 GiB. The focused suite reports 13 passed; type checking reports zero errors or warnings. The earlier PR283 report displacement was corrected to the saved-control delta norm `0.001159532966076956`; this was a documentation-only correction and did not alter the plan or rerun its producer.
