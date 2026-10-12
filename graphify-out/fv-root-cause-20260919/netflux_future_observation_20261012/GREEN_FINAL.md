# GREEN final audit: paired terminal observation

The paired observer completed with a closed child and resource receipt. This
is a read-only audit of those saved bytes and their arithmetic; I did not rerun
the forecast or score.

The frozen plan SHA-256 is
`7c2b7f317e90e49a0c1bd9a289529816b8cef91e9d33048c8017b9b884279ae5`; its
209-member source bundle SHA-256 is
`2a5eb415af2b8f873d319b45fd443894cba8c4533bc5694b4013ea96f8caf833`.
The bundle members match all source hashes, and the plan retains 283 archive
pins. The raw child SHA-256,
`2003ceee1aa058fc1aa866ea36f000fbf2d9210b321c0c461711171494fefb89`, matches
both `score_pair.run.json` and the saved metrics. The run receipt links to the
resource receipt; its child status is `closed` and its execution status is
`completed`.

The child reports exactly two started and two completed scores under
`minmod_net_flux_fp64_v1`: start control `0aaf…655b` scored `7.388754298552231`,
and final continuous control `b647…88660` scored `7.386491165784028`. Their
recomputed difference is `-0.0022631327682036684`, a 0.0306294% decrease from
the start score. Both controls share the pinned parameters, fixed truth,
problem identity, and terminal profile. Source, input, runtime, and fixed-case
closures are true; all root/minimum/response/score claims, optimizer steps,
candidate evaluations, and selection use are false/zero.

The resource receipt reports exit code 0, no monitor error or resource
termination, 12.638 seconds elapsed, and 944,701,440 sampled peak RSS bytes
under the 660-second / 2-GiB guard. This is a paired terminal-E observation at
the two previously closed controls. It is separate from the comparison's
selected `d19e…145e` point: no score was computed for that point, and this
measurement does not select a control or establish independent forecast
skill.

Final packaging check: the receipt manifest lists exactly four members; every
archived hash and byte length matches both the manifest and retained
`attempt1` files. The receipt tar hash matches its manifest and is 49,741
compressed bytes (155,978 uncompressed member bytes). Findings, metrics, and
verification stay scoped to the two saved terminal scores and closure checks;
the bundle contains no claim or score for the selected `d19e…` point.
