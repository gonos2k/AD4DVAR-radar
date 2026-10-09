# RED final audit: interior minimum-norm mixture tangent step — 2026-10-10

## Disposition

The saved run completed one bounded baseline-tangent control move under the new interior mixture-merit mode. The candidate control and freshly recomputed interior theta-star passed native-J and actual Psi/F² Armijo, selected-face and current side-pair checks, individual side-gradient closure, and final theta-star closure. The final tangent residual remains `0.06847775`, so this is a bounded diagnostic step, not a root/minimum or response result.

No second production run, seed reconstruction, FV/HVP regeneration, test, or guard was performed for this RED audit. Checks below use the saved raw step, parent/resource receipts, prior comparison endpoint, and saved side arrays.

## Provenance and resource closure

Frozen mixing-minimum plan SHA-256: `9611b13d3392c198f0a92436c1e9536a6286363a2c64a70a48e5b43f6c480576`, with 133 source pins and 124 archive pins. The read-only plan/base loader passes and resolves the expected baseline control `0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1`, carried theta `0.3298981720696071`, and original J `0.06124393564803218`.

The raw step SHA-256 is `8581c6ae0d430c5c9c36c71209bfa03bd53a8627271f6cb900f60790a67d79aa`; the parent child digest matches and nested/standalone resource receipts agree. The root-owned launch completed in 32.103 seconds, sampled peak RSS `766,574,592` bytes under the 300-second / 1-GiB sampled limits, with 2/2 fresh HVPs and 0/0 row VJPs. Source hashes stayed equal across 252 paths; input, runtime, source, and deadline closures pass. RSS is sampled child usage, not an OS hard allocation cap.

## Saved base and candidate theta-star arithmetic

At the old baseline endpoint, the prior archive has carried theta `0.3298981720696071` and carried-theta `F²=0.005132555144362135`. The new run freshly recomputed an interior base theta-star `0.3277376621747043`, giving base `Psi=0.0025660872704794586` (`F²=0.005132174540958917`). The raw marks this normalization `optimizer_step=false`; it does not advance the current control or carried theta.

For the one accepted candidate, the raw line search's linear theta prediction is `0.3314459594337958`, but fresh candidate side gradients give `theta_star=0.48316874590279574`. The difference `0.15172278646899995` is expected evidence that the candidate theta must be recomputed from its actual gradients; the linear prediction is diagnostic, not a rejection test. The candidate remains strictly interior without clipping.

The accepted control hash is `e801cf4651233ec11b2ec98ac9569e411b8e62e68511440e39db1cb9c3872567`, with movement norm `5.380344666148523e-05` inside radius `0.05`. It has native `J=0.061240252230254005`, `F²=0.004689202366398875`, and `Psi=0.0023446011831994374`. Relative to the fresh base Psi/F² normalization, actual F² decreases `8.6313%`; J decreases `0.0060143%`. The final mixed gradient is normal-free to roundoff and face Q is zero, but the tangent norm remains `0.0684777509` and F norm `0.0684777509`. The residual is wholly tangent and far from zero; no stationarity claim follows.

## Full envelope derivative and local model

The saved model stores `theta_prime=4.937591432647229`; its numerator/denominator formula agrees with the raw arrays. Recomputing `F·DF` gives the envelope slope `-0.0396956279639186`, matching the scalar envelope formula within `1.4e-17`. The full residual-direction formula has zero componentwise error against saved arrays. The model-optimal alpha is `0.00075103363850081`, exactly the accepted alpha under the frozen cap.

The local quadratic model predicts `F²=0.005102361789056601` at that alpha (about `0.581%` decrease from the fresh base normalization), while the actual candidate at freshly minimized theta-star has `F²=0.004689202366398875` (about `8.631%` decrease). This large favorable mismatch reflects the endpoint response, not a more accurate prediction. The accepted candidate passed the actual Psi/F² Armijo bound `0.005132168578408537`, actual J Armijo, face, branch, side-objective, finite-gradient, and final repeat gates; endpoint checks, not the local-model prediction, support acceptance.

The base branch signature is `8c730e55…`; the candidate's two side traces both have signature `7245bd3a…`, so the strict paired endpoint gate passes while the path changes branch. Static face arithmetic shows non-target `Q_y[3,2]` flips from positive `1.08465e-7` at the base to negative `-1.84411e-6` at the candidate, while target `Q_y[4,3]` remains zero. This is consistent with the trace change and does not establish a smooth path. The candidate theta-star, both fresh side gradients, branch traces, native J, Psi, face, source, fixed input, runtime, and deadline were independently repeated and matched the proposal.

## Limits and interpretation

The result is one nonzero-control step using the existing tangent baseline direction and the new interior theta-minimized residual. It does not retroactively accept any old PR270 rejected candidate or its post-hoc theta recombination. The base theta-star is a reported normalization, not an optimizer step; the actual commit is the selected candidate control plus its independently repeated candidate theta-star.

Raw fields report `full_smooth_root=false`, `minimum_claim=false`, and `response_claim=false`. Curvature qualification, theta-only finishing, coupled response/reanalysis, and forecast validation remain separate work. This run makes no global convergence, minimum, response, or forecast claim.


## Final archive and document cross-check

After archive finalization, I verified `step.json.gz` SHA-256 `77ab19d185fb7fafa9caba0395590e7efe78d12b403f02c7a6e9e649a417af3c`; decompression is byte-identical to the 18,381,365-byte raw step with SHA-256 `8581c6ae0d430c5c9c36c71209bfa03bd53a8627271f6cb900f60790a67d79aa`. The archive manifest's raw, gzip, run, and resource hashes match the files. The result's raw digest agrees as well.

I cross-checked the finalized Korean findings, result JSON, saved-array analyzer, task checklist, and KG update against these receipts. The base and accepted control/theta hashes, J and minimized F² values, one-step/2-HVP/0-row accounting, and one-run limits agree. The analyzer is explicitly limited to saved JSON arrays and static flux arithmetic; it does not import or execute the model. The findings correctly report the local-model prediction of about 0.581% F² decrease versus the actual 8.631% decrease and identify that as a favorable model mismatch, with actual-value and final-repeat gates supporting acceptance. They also label base theta normalization as not an optimizer step, treat linear theta as diagnostic, note the branch-signature change and non-target Qy sign flip, and preserve the single-step/no-minimum-response-forecast limits. The KG update leaves repeat merit reduction, curvature, theta-only finishing, coupled response, and independent future validation outstanding.

No actionable discrepancy was found in the final archive or cross-audited documents. This remains one accepted local step; the favorable model mismatch does not establish model accuracy, convergence, or a broader performance gain. No production computation or guard was run for this final cross-check.
