# GREEN final audit: 2-GiB GN continuation — 2026-10-10

## Disposition

The new attempt closed its resource envelope and source/input/runtime receipts. It committed **one new step** from the saved f2ca endpoint, then stopped because all 16 candidates at the next point failed the actual F² Armijo test. This is a numerical stop with completed execution, not a resource interruption or a convergence result. The previous 1-GiB run remains a two-of-three partial run; this 2-GiB plan is a separate experiment with at most three new commits.

## Closed new step and rejected point

The new plan starts at f2ca control `f2ca572936a8bf646c920c542b3780ff65a0f94c0d715aa61ba00a5cec57313d`, θ `0.4830637745604483`, J `0.06119270839098437`, and F² `0.004479439946247095`. Its one accepted endpoint is control `2ec34eeeef531b94e21b5b4372ced2e07257a5407541072d300f5a3f513e1fcf`, θ `0.4830786491678902`, J `0.06119249248090723`, and F² `0.00447848922842147`. Its final repeat passes the saved endpoint/P2 checks. The accepted alpha is `8.918897372755977e-5`, or 1/8192 of the initial model alpha. The local dimensionless model diagnostic χ=`cos²(F,DF)` is `0.8694144`; full-residual relative error to the endpoint is `5.18e-8` at `2.70e-6°`, while residual-change relative error is `0.0455%` at `0.0258°`.

At the next point, control `2ec34eee…`, the current theta is `0.4830786491678902`. Its 16 candidates all passed J Armijo, face, paired branch, finite-gradient, side-objective, and fresh-minimum gates, but all failed F² Armijo. The smallest candidate F² was `0.004485470694473677` at α `4.459590701321281e-5`; the smallest alpha, `2.2297953506606405e-5`, had F² `0.004485707729662102`. Both are above the base F² `0.00447848922842147`. The progress receipt reports 16 rejected trials, no accepted endpoint, and the expected “no direction model produced a candidate passing the tangent search gates” refusal. This finite search does not establish that the underlying problem has no solution.

Both modeled points have 24 fresh row VJPs, two paired HVPs, and one saved 12×12 solve receipt. Totals are 48/48 rows, 4/4 HVPs, 2/2 dense solves, and 30 candidate slots (14 at the accepted point and 16 at the refused point). The one-commit result is within the plan's limit of three new commits.

## Resource and provenance

Execution completed with exit 0 and no resource termination after `287.313` seconds. Sampled peak RSS was `1,061,421,056` bytes under the new `2,147,483,648`-byte cap, and `12,320,768` bytes below the old 1-GiB threshold. The sample is close to 1 GiB; this single run does not establish that a future run is safe under a lower limit. A 1.5-GiB plan remains a later comparison proposal, not a tested cap.

All 144 source and 173 archive pins match the frozen plan SHA `7c93aa28219c73e7a437feaac7341887da8fc6fc0b9f0b8e4633ad4424224c67`. The archive round-trip, parent/resource receipts, and confirmed prefix reconcile. The fresh base input matches the derived f2ca anchor. Between input-before and input-after, only the control SHA and `shifted_flow_fractions` change, as expected from the committed control; the fixed-input flag is true. Runtime and runtime-after match, and source-before/source-after match. The original preflight failure remains preserved as a separate zero-row/zero-HVP non-numerical receipt.

The saved analysis artifact is [STREAMGN_MEMORY_RESULT_20261010.json](STREAMGN_MEMORY_RESULT_20261010.json). Its compatibility guard only handles an absent accepted-alpha field in the inherited NumPy collector for refused points; the frozen collector source and all production sources remain unchanged. The analyzer keeps the full child bytes, hashes, counters, and refusal trials, and reports the unaccepted point separately from the one committed endpoint.

Resource closure establishes neither optimizer convergence nor a root/minimum. It does not certify exact curvature, response, adjoint, reanalysis, or forecast skill. Lowering the RSS cap requires a separate bounded run.
