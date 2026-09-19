# PR162 regression completion and low-diffusion comparison

Baseline: merged main c40a1203. No product module or exact-response gate changed.

## Regression completion

- The changed censored input now uses its own freshly prepared state throughout solve, refinement, forecast, gradient and response. Its numerical outputs remain bitwise invariant; raw provenance need not match.
- At nonzero control c=0.3, the initial transform is checked against an independent closed-form background derivative and mixed background/control derivative. Cases include both open branches, floors 0/-10 and scales 1/4. The mixed derivative is also checked by central gradient differences; control sensitivity remains positive below the B-clamp.
- Focused tests: 6 transform cases passed (3.03 s), 1 changed-state integration case passed (83.62 s), and 5 probe/default-convergence/shape cases passed (5.09 s). Total 12 distinct cases; development reruns are not added.
- Full CPU CI, exact minmod response and unchanged 240-grid reanalysis were not run.

## Same-duration transport measurement

Both schemes use the same 48 km domain, 32/64/128 grids, 18 leads, fixed CFL schedule, initial cell averages, known-zero exterior and full support. The area threshold is fixed at echo=0.1 across grids and times (all analytic initial profiles have unit pointwise peak). Moments describe the piecewise-constant cell averages, including dx^2/12 within-cell variance. They are shape diagnostics, not precipitation totals.

The 128-grid final (180 minute) results are:

| Flow | Scheme | Echo L2 error | Selected-AD JVP error | Peak/reference | Width/reference y,x | Centroid error m | Area/reference |
|---|---|---:|---:|---:|---|---:|---:|
| translation | donorcell | 15.25% | 31.11% | 83.76% | 1.0000, 1.2159 | 8.283e-09 | 1.1092 |
| rotation | donorcell | 9.61% | 20.05% | 89.29% | 1.0982, 1.0644 | 0.0008824 | 1.0803 |
| area_preserving_strain | donorcell | 3.37% | 13.45% | 96.79% | 1.0196, 1.0340 | 37.15 | 1.0308 |
| translation | minmod | 3.53% | 17.48% | 94.59% | 1.0000, 1.0266 | 0.006652 | 1.0044 |
| rotation | minmod | 1.51% | 8.26% | 96.25% | 1.0083, 1.0058 | 0.1008 | 1.0058 |
| area_preserving_strain | minmod | 0.72% | 4.25% | 98.27% | 1.0024, 1.0031 | 3.665 | 1.0000 |

All sampled states remain nonnegative. Maximum transformed budget residuals are donorcell: 7.052e-16, minmod: 5.818e-16.

The donor-cell L2 and JVP values exactly reproduce every saved 32/64/128 result from the earlier 180-minute probe. The new run was needed for shape metrics, not a replacement of prior evidence. Its translation added x-variance is approximately 2.025e6 m^2, consistent with |u| T dx.

For minmod these are outputs of the AD-selected limiter branches versus the continuous reference. Branch stability is explicitly unverified. The long-duration comparison is not a classical differentiability certificate, a JVP/VJP inner-product test, or an exact Hessian/FSOI result. Subsequent focused tie tests are recorded in MINMOD_BRANCH_AND_PUBLISHER_REVIEW.md: their inner-product check passes but does not establish two-sided differentiability. Reduced field error alone does not justify lifting donorcell-only inverse-response gates.

## Cost and remaining checkpoints

- Final donorcell comparison: 38.44 s, sampled peak RSS 332808192 bytes; bounds 180 s/2 GiB, exit 0.
- Final minmod comparison: 44.48 s, sampled peak RSS 336707584 bytes; bounds 180 s/2 GiB, exit 0.
- An exploratory pair used a resolution-dependent initial-peak area threshold (donorcell 41.45 s, minmod 49.47 s). Those local exploratory files are retained separately; the final tables use the corrected fixed physical threshold. They are not additional independent accuracy evidence.
- Next: limiter branch/tie diagnostics over the RK trajectory, discrete JVP/VJP and mixed derivatives, then fixed-support inverse reanalysis. No scheme promotion yet.
- Typed mean/precision/support learning, finite-impact signal/error tolerances, general input integration and full D7 remain open.
- The original HTML animation arrays are preserved. The comparison is a separate evidence table, not a replacement forecast or learned flow.

Aside desktop DOM and screenshot confirm the six comparison rows and the
limiter/FSOI limitation text in the original demo evidence panel. The template
and local index demo-data hashes remain respectively
88bb5eb25dfb62cd519d98912d421407678f9957dc7b3a7503444d78a113c8e1 and
5fe7d42540b66e83ca40eb4943a070696402d332f91e4ecea54270ebfb28c31e.
The new table has its own horizontal overflow container; a new mobile viewport
run was not performed in this follow-up. Graphify used cached code-only
extraction (6508 nodes/64786 edges), with zero semantic LLM tokens.

Final GREEN/RED review found no blocker within this stated scope. The review
confirmed reconstruction forwarding, fixed support and threshold, within-cell
variance, derivative-oracle formulas and the unverified limiter/inverse boundary.
