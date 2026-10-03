# QY release provenance replay amendment (2026-10-03)

The completed attempt-1 release is preserved at `point_qy_release_attempt1/release.json` (SHA-256 896b38c34fdc87fbaaa4ec51d271522942d96e1c4cddfcdf2901d087b35fd462). A later capture audit found one source-map difference: `examples/weather_scenarios/fv_point_qy_release_probe.py` was 24d784ff006cafefa99eaffa39cfaa53efd025248b0e5c9b532c9504f6a09335 at release time and is 1e9445f07de26536ea90af4d0a187309302443bae20ff75ca09a3b8f2d4d5a4f now. Every other attempt-1 producer source matched the current bytes. The stored attempt has no source preimage, so the precise change cannot be reconstructed from the artifacts.

This amendment authorizes one provenance replay under the current frozen driver, using the same paired tangent seed, positive qy coordinate eta=1e-4, 4 Newton iterations, 16 backtracks, 104 PCG iterations, and existing 1e-10 residual/gradient tolerances. It changes no physics, point input, or numerical gate. Attempt 1 remains untouched. The replay is a provenance resolution, not a response or full-root certification.

Pre-amendment source/test/plan snapshot: `point_qy_release_attempt2/pre_amendment_source_snapshot.json` (SHA-256 ff75a222199bd367726ec9fdc933f1f67b2df7ee0a9f6eb71a9b0989d57bceef).
