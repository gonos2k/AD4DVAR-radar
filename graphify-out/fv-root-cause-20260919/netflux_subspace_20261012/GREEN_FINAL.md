# GREEN final audit — two-direction trial and paired observation

**Disposition: the frozen trial and separate observer both closed successfully.** This audit used the stored child records, candidate payloads, and independent vector arithmetic. It did not call the FV model, HVP operator, or score function.

## Candidate trial

The child record SHA256 `0802769b3dd6624c11e67ff9a222594f2ec25c728ea704021903419e64c6a4cb` matches `step.run.json`. The guard reports exit code 0, no monitor error or resource termination, 74.81 seconds elapsed, and sampled peak RSS of 1,284,161,536 bytes (about 1.196 GiB of the 2 GiB limit). The child closed with four fresh HVPs, 24 GN row VJPs, one dense solve, two candidate evaluations, two independent repeats, and one commit. Source, runtime, and fixed-input closure flags pass; the fixed non-control parameter, truth, and archived problem identities remain pinned while the control advances from d19e to `7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`.

Both arms passed their candidate gates and independent repeated P2/full-G checks. The raw receipt hashes match the arm links: baseline candidate `d258ec3c558c533a3b314984b7024daad22487e934b71cde7303bbfda51d238e`; subspace candidate `4bb6f2025db348e72e5cadbe0bd18d9587060f77751e73ac2c28150dda3afb9`. Both repeats return identical control, J, and R. The frozen repeated-R-then-J rule therefore selects the gradient arm.

The two-direction solver selected its `axis_0` active set with coefficients `[0.0003707309698259479, 0.0]`. It reproduces the gradient ray at this point, so the trial shows no incremental gain from the added direction. The accepted candidate changes J from `0.061002341038211755` to `0.061001393469310765` (0.00155333% decrease) and R from `0.002559866170789153` to `0.002548715176361377` (0.43560849% decrease). Both candidate branch signatures match the base.

I independently recomputed each stored `G(candidate) - (G_base + DGd)` vector from the step and candidate receipts. Both recomputed errors equal the recorded norm, `9.693232300420351e-8`; both have `eta_vec = 4.557458384961589e-5` and `rho_R = 1.00003131806914` (rounding differs only in the last digits from the recorded ratio). The error norms across echo, flow, growth, and face components are `9.3478e-8`, `5.6473e-9`, `2.5018e-8`, and approximately `1.3e-22`. The candidate receipts contain matching actual controls and payload vectors, and their SHA256 values match the frozen step links.

`VERIFICATION.json` reports all stored receipt, linearity, radius, fidelity, repeated-candidate, selection, resource, source, and fixed-input checks as true. The run remains a local finite step; it does not establish convergence or an exact constrained minimum.

## Separate fixed-truth observation

The observer record SHA256 `7248c681645ee37cb2f2900618bb777da095eed8107323409e2f3b99a7d6fe1d` matches its run receipt. Its guard reports exit code 0, no monitor error or resource termination, 16.97 seconds elapsed, and sampled peak RSS 1,403,125,760 bytes. It records exactly two score evaluations, zero optimizer steps, zero candidate evaluations, and `selection_use: false`; source, input, runtime, and fixed-case closures pass.

Under the separately pinned observer plan, the fixed old-truth score decreases from `7.386491165784028` at b647 to `7.386414763452886` at d19e, a change of `-7.640233114170059e-5` (`-0.00103435%`). The new committed `7839625…` control was not scored. `OBSERVATION_VERIFICATION.json` reports all checks true. This is a paired score on the frozen old truth, not independent forecast validation and not part of candidate selection.

## Verification record

The frozen candidate and observer plan hashes and preflight limits are recorded in [GREEN_PREFLIGHT.md](GREEN_PREFLIGHT.md). The combined targeted suite reports 44 passed in 10.71 seconds, with 18 existing TorchScript deprecation warnings; the final basic Pyright run reports zero errors and warnings. The scoped Graphify refresh contains 62 nodes and 107 edges. RED's preflight clearance is recorded separately in [RED_PREFLIGHT.md](RED_PREFLIGHT.md).
