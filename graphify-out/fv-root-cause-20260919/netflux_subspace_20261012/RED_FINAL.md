# RED final audit — d19e fixed-radius direction comparison

## Finding

The one guarded trial completed and committed one candidate after both arms
passed independent actual J/R and repeated P2/full-G closure. Both arms produced
the same control byte-for-byte. The subspace model selected only axis 0, the
baseline tangent direction; its second coefficient was zero. The repeated J and
R values tied exactly, so the declared tie rule selected `baseline_tangent`.
This run establishes one small conditional step from d19e, not an improvement
from combining two directions, a converged optimization, or a minimum.

## Independent receipt and arithmetic checks

The frozen trial plan is SHA-256
`a0fef5718772032b4567d24ca75fbaf39cff8b58e30f7e4b249dbabb4a5c7d26`; its
source bundle is `b332268a74f0c5ed9f9e05b3d890288d2e953215b1422253eec2426cb183cab3`.
These match the preflight record. The plan carries 216 source and 287 archive
pins. Preflight verified all pins and all source-bundle members; the run records
source, fixed non-control input, runtime, and internal-deadline closure as true.

The raw child SHA matches `step.run.json`. The outer guard exited 0 after
74.814 seconds, with no monitor error, signal, or resource termination. It
sampled peak RSS at 1,284,161,536 bytes under the 2-GiB limit; this is a sampled
guard, not a hard allocation ceiling. The child recorded 72.485 seconds,
four of four fresh HVPs at d19e, 24 of 24 GN row VJPs, one dense solve, two
candidate evaluations, two final repeats, and one commit. Both candidate receipt
SHA values match their raw candidate files, whose stored controls, J, and R agree
between arms.

At the freshly requalified d19e point, base `J=0.061002341038211755` and
`R=0.002559866170789153`. Each arm’s first candidate passed, at radius
`1.8757177916958825e-5`, chart alpha 1, and actual displacement
`1.8757177916130278e-5`. Each candidate and independent repeat report
`J=0.061001393469310765` and `R=0.002548715176361377`. Directly summing squares
of each stored 27-vector G reproduces R exactly. The decreases are
`9.47568900989959e-7` in J (0.0015533%) and `1.1150994427775937e-5` in R
(0.4356085%). Both repeated J/R Armijo bounds pass, both full-G checks pass,
and both arms are marked independently eligible.

The subspace solver reports `selected_active_set=axis_0`,
`full_kkt_status=full_2d_orthant_boundary_axis`, and `reduced_domain=false`.
Its coefficients `[0.0003707309698259479, 0]` equal the baseline ray
coefficients. Both candidates have control SHA
`7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`; repeated
J and R are exactly equal. The outcome therefore does not demonstrate a gain
from the second direction at this base and radius.

The recorded local residual prediction is close for this single accepted step:
predicted `R=0.002548715525578055`, actual `R=0.002548715176361377`,
`rho_R=1.000031318069182`, vector error norm `9.693232300420351e-8`, and
`eta_vec=4.557458384961589e-5`. These describe model fit at this candidate; the
actual repeated J/R and P2/full-G checks provide acceptance evidence. They do
not establish accuracy at other radii, branch regions, or controls. Both arms
record matching candidate/base branch signatures in this run; no branch-path
smoothness claim follows.

## Provenance, score separation, and limits

The trial starts from the admitted d19e endpoint SHA
`d19e882d31b634ef0be755e6419b57ecf3691734efc736d5f40ea18c3ac4145e`, which was
the selected endpoint of the prior b647 comparison. The current run freshly
reobserved its side pair, previous-P2 and minimum-mixture P2, native release,
and both direction/HVP models at d19e. The committed endpoint is
`7839625d5ff2ab8f3d482f5fab368f19a6a3d60405ed373267c0fcacc81c7c1a`. No b647
HVP was reused. If this comparison had stopped before both eligible repeats,
the commit rule would leave d19e as the last confirmed control; here both arms
closed, so the one selected candidate advanced it. Postcommit readiness and
terminal E recomputation remain false in the trial receipt.

A separate observation-only run used the frozen score plan
`4d467799150c88af4ebdef79cda7034aedbbb2e9bbb99f8321c8e10f4809b667` to score
only b647 and d19e against the same archived truth. It completed two score
evaluations, performed zero optimizer steps, and marks `selection_use=false`.
It reports `E=7.386491165784028` at b647 and `E=7.386414763452886` at d19e,
a difference of `-7.640233114170059e-5` (`-0.00103435%` from the b647 value).
This score pair does not include the
newly committed 7839 endpoint and did not select the trial candidate. It is one
observation under a fixed synthetic truth, not independent forecast validation.

The completed trial is one fixed-radius, one-commit comparison. It does not
establish convergence, a normality root, an eligible minimum, exact constrained
curvature/adjoint validity, response improvement, or forecast skill. Historical
research milestones and observational evidence are unchanged. RED read saved
JSON, receipt hashes, and vector arithmetic only; no additional FV, HVP,
optimizer, or score evaluation was run for this audit.
