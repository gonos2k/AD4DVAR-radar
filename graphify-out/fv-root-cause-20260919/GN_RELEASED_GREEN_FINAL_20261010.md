# GREEN saved-data review: released GN resume from 15516

The saved run starts from the P2-closed `15516fbae…` control and requalifies current J, both side gradients, minimum θ, full 27-component G, paired branch, face, trace, input, and runtime. At trace 124 / x / cell `[1,2]`, the base snapshot has `choose_left=true` on both sides; the earlier negative/right-selector probe is marked unsupported at this point. The run records this state without using an event gradient, JVP, or projection.

The fresh current-point work completed 24 row VJPs, one positive-definite 12D solve, and two HVPs. Candidate 6 of the 24-slot dyadic search was the first passing GN candidate: α=`0.026139762146698584`, actual chart movement within the `0.05` radius. One independent P2 repeat passed for the actual J/R, minimum θ, both side gradients, trace, face/branch pair, and source/input/runtime/deadline receipts. The run committed once to `29b52a1377caa7fb52431bdbea25124fa827ac548eeefb9b39ce0e5f240d97c4`, θ=`0.6865600659804322`.

From base J=`0.06109417561387711`, R=`0.0042980213317858635`, the committed point has J=`0.06104894133302345` and R=`0.0032814674811390797`: decreases of about `0.07404%` and `23.65167%`. The saved residual norm falls from `0.0655592963` to `0.0572840945`; G∞ falls from `0.0322529450` to `0.0268768518`.

Independent NumPy arithmetic on the saved base G, model residual direction, α, and accepted trial G gives model R=`0.004054322690748495` versus actual R=`0.0032814674811390797`. The actual R decrease is `4.17136×` the model-predicted decrease. The vector error norm is `0.04548387`, or `0.98117×` the actual vector change and `0.79401×` the new residual norm. The candidate improved the actual merit despite a large linearized vector error.

The target selector returns to `choose_left=false` at the committed candidate; both endpoint branch-pair and P2 trace checks pass. This is a pointwise transition record, not a selector constraint or continuous-path certificate. No event-specific derivative was used. The run applies one new optimizer commit, makes no further candidate, and deliberately performs no post-commit readiness. It establishes neither an optimum nor future convergence, response skill, forecast accuracy, or model validity.

Receipts report exit 0 in `57.8854 s`, sampled RSS `428,621,824` bytes under 2 GiB, 24/24 rows, 1/1 solve, 2/2 HVPs, and zero event gradients/JVPs. The focused suite reports 13 passed; basedpyright reports zero errors, warnings, and notes. No FV or derivative work was rerun for this saved-data review.
