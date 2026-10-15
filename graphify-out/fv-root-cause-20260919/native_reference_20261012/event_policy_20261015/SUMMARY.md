# PR306 review closure: event maintain/release and continuation

Input main926aff02/current30bed → endpoint`9c106af45be7141fb70727e0993842c3408ba6c13a83f47d66c1c5398e16be5d`.

The existing Newton/retraction progression now uses event forms instead of stage IDs. Event identities, operands, normal rank, all limiter-side × flux gradients and actual native decisions are archived under attempt1. No Powell, previous ray, future benchmark or response callbacks were rerun.

Outcome: 12 native Armijo decreases; J0.0601191807868374 → 0.060119179243380519 (decrease1.5434568831e-9, relative2.5673285346e-6%). 5 release, 3 smooth and 4 maintained-face moves; 6 bracketed transition roots. Ended at the frozen12-step cap with no rejected unsupported-stage trials.

Stage48 is released using all8 side/flux extensions: a−≈−0.0003493946191572, a+≈+0.0005269887984897. It reaches stage47, releases it, reaches y124/y123 operand-zero events, releases those, then revisits stage47 with both outgoing signs positive. At the final point stage47 gap and y123 right-zero are simultaneously maintained; past events are not retained.

Terminal native audit is independent; terminal16 limiter-side×flux gradients are recomputed, no same-point family cache reused. Hull maximum component=1.1755242589376992e-07, above1e-10. Final growth chart component is the maximum. first_order_complete=false, P2_complete=false, minimum_claim=false, response_claim=false. No certification Hessian is calculated after this failed first-order gate. Search Hessians apply only to their recorded current points.

81 affected tests passed with18 existing TorchScript deprecation warnings; targeted basedpyright reports0 errors. Expanded preflight found3 inherited test failures: old test assumed a singleton _selected_face object; main926aff02 uses a tuple. The assertion now checks exactly one tuple entry and the same required face; final expanded suite passed. Saved provenance/trace/control/Armijo/derivative arithmetic verification passed13 checks, and is not an independent FV/P2/adjoint run.

Measured phase wall=349.107523s; peakRSS=784859136bytes. Counts J311/gradient327/Hessian rows336/profile131; started=completed. Cumulative tracked J905/gradient490. Prior incomplete historical cost ledger stays explicitly incomplete.

Limits: root detection is bracketed using endpoint plus three interior trace samples, not a global no-cross/revert or earliest-event proof. Supported mathematical forms are same-sign slope exchange and one operand crossing zero; at most two independent simultaneous events, exactly the existing two flux faces. Ambiguous coincident operand zeros/additional flux faces/normal dependence stop explicitly; no blanket stage admission. Error budgets are dtype/operation/component scale diagnostics, not interval arithmetic certificates.

Remaining: current endpoint is nonstationary despite resolved active outgoing signs. The last search model predicts g·d≈−1.1916e-17 before Armijo backtracking; native costs are near floating-point resolution, but this observation does not certify a minimum. Close the same active KKT residual before terminalP2 or p±h response work. Existing6ca independent-time-integrator18-lead forecast (late failure) remains complete; the new endpoint has not been forecast-scored.
