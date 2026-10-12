# GREEN final — cached d19e KKT/radius audit

The offline audit is complete. It admits the PR301 plan/source bundle/receipt bundle from pinned bytes, checks all 11 receipt members and the raw child/run/resource/candidate links, and takes `G`, `D`, and `V` only from bundled d19e step bytes. It does not call FV, HVP, or score code.

At d19e, normalized V has rank 2 (`sigma=[113.3936229, 1.5855801]`, threshold `4.3508e-11`). The frozen axis0 candidate satisfies KKT with `lambda=9417.50477`, `mu1=+0.06984030` and negligible stationarity/radius residuals. Axis1 reaches `mu1=0` at `Delta=2.5388646371e-5`, bracketed to about `1.48e-14`, before the axis0 `lambda=0` radius `3.2495262214e-5`. The separate rank-one fixture exercises the main solver fix and its `lambda=0` interior branch.

Five tests pass in 2.28 seconds; expansion-config Pyright on the audit script and tests reports zero errors/warnings. The scoped Graphify delta has 30 nodes and 55 edges. The full matrices, budgets, radius sweep, solver hash and recommendation are in `KKT_FINDINGS.md` and `d19e_axis_radius_audit.json`.

One bounded fresh-point test at no more than `1.5x` the prior radius is worth considering only after requalifying the new base and rebuilding its directions. The d19e model predicts the second coefficient becomes positive above the computed critical radius, but its modeled R gain at `1.5x` is only `3.4053e-5` relative to the gradient-axis R. No above-radius finite candidate was evaluated; the result is a local diagnostic, not a production radius decision.
