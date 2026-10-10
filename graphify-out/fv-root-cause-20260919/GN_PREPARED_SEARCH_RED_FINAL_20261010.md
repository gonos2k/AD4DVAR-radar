# RED final review: prepared-point GN search — 2026-10-10

## Disposition

The frozen plan and completed saved run close one first-passing candidate and one optimizer commit, followed by fresh GN readiness at the committed point. I found no actionable state, receipt, or numerical-contract inconsistency. This is a local decrease and a ready next-point model; it is not a convergence, minimum, response, or forecast-skill result.

## Candidate and commit

The frozen plan SHA is `1392032f3f0b859f74f03551a98fa5465216cb7fd981250984b603a495526a1a`; all 151 source pins and 195 archive pins were present and matched. It retains the prior plan SHA and sets a 24-slot cap, actual chart radius 0.05, one commit, and no root/minimum/score/response claims.

The saved base was freshly requalified at control `311f27d095be4976ca6131b83b6c4f3d46bc404d4f58921800b5600e38ec0652` with its native J, side-gradient pair, minimum-mixture theta, branch traces, face, fixed input, and runtime. Its same-point HVP pair and direction were reused only after the saved audit and model reconstruction passed. The model cap was α₀=`0.7306607947800585`; 21 dyadic endpoints were evaluated and the first 20 failed the actual F² Armijo check. All 21 stayed within the 0.05 actual displacement radius. Selector signatures differed at candidates 1–20, while each endpoint's paired-branch and face checks passed; the search did not require path-wide selector invariance.

Candidate 21 passed both original Armijo checks at α=`6.968124339867196e-7`, actual displacement `3.236519866409528e-8`, native J=`0.061192477301754936`, and minimum mixed-gradient F²=`0.004478422389776647`, from base J=`0.06119247898831076` and F²=`0.004478429816264897`. Its fresh independent P2 repeat matched objective, merit, theta, and branch trace, and passed paired-branch, face, input, runtime, source, and deadline closure. The committed control hash is `027e83cf77f7f9cf97c0c72da9f737b5627d312d6375b698bb5b7f7ae94c6769`; this run applied exactly one commit.

## New-point readiness and resources

At the committed control, the record has all 24 distinct side/row VJPs, one Cholesky-positive 12-dimensional dense solve, and two completed selected-face HVPs. The row, solve direction, and HVP labels bind to the new control, direction, theta, operator, and `robust_gn_coupled` model. The solve residual is `4.354005256868789e-15`, below its `2.154198039753652e-12` budget. Readiness reports zero further candidates and zero optimizer steps after commit; no second candidate was run.

The child completed in `181.7425 s`; the guarded parent completed in `185.9167 s` with exit 0, no resource termination or monitor error, and sampled peak RSS `439,566,336` bytes under the 2-GiB sampled limit. Source, fixed-input, runtime, and deadline closure flags are true. The archive records a lossless raw/gzip round trip: raw SHA `a2fca18bb519de98536f9e902860f7df035a58a1b60d9b67cd89d4628ab3bef1`, gzip SHA `322606baa43778ea093f2fa4ac8bb656ab5803cce35850c072fb77eaf8a9d782`, and matching parent child SHA. The affected suite reports 14 passed and type checking reports zero errors and warnings; these checks do not independently replay production FV derivatives.

## Evidence locations

- Frozen plan: `GN_PREPARED_SEARCH_PLAN_20261010.json`
- Producer: `examples/weather_scenarios/fv_point_3h_gn_prepared_search.py`
- Raw saved record, run receipt, and resource receipt: `gn_prepared_search_20261010_attempt1/step.json`, `step.run.json`, and `step.resource.json`
- Archive manifest: `GN_PREPARED_SEARCH_ARCHIVE_20261010.json`
- Verification summary: `GN_PREPARED_SEARCH_VERIFICATION_20261010.json`

No convergence, optimum, response, or forecast-value claim is supported by this run.
