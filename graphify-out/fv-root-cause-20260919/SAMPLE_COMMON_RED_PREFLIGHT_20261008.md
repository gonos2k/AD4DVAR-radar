# Sample-common search RED preflight

Date: 2026-10-08
Decision: **GO for the single planned guarded original-J exploration.** This is a static receipt/code review only; this reviewer did not launch the child and ran no FV, HVP, PCG, optimizer, score, adjoint or reanalysis.

## Pin and scope

The frozen plan `SAMPLE_COMMON_COST_SEARCH_PLAN_20261008.json` has SHA-256 `617a81486352591f3e17068be78fc4047db5bd4e07333b84dba48297ba9dcc3e`. I independently recomputed this hash and read its scope: 115 source pins and 59 archive pins, producer face-diagnostic plan SHA `6b79d07c…`, base control `2cdccade…`, one guarded launch, 240 s internal / 300 s outer / sampled 1 GiB RSS, at most 16 candidates, at most one accepted cost step, one HVP, no PCG, no forecast score or response. The plan records J Armijo (c_1=10^{-4}) and states Φ is diagnostic-only.

The root reports that real loaders passed, the direction norm reconstructed as 0.13679778, 11 focused tests passed and type checking had zero errors. These are producer-reported checks, not executions by this reviewer.

## Numerical gates

The child reconstructs the direction from the pinned inner ±1e−6 full gradients, rather than trusting a copied direction vector. It rechecks current original J, full gradient, Φ and strict base branch against the accepted resume receipt, then tests a scale-resolved (g^Td<0) before starting its only HVP. The new report records both the HVP start/completion counts and (g^THd) as a diagnostic. The initial scale uses positive, resolved (d^THd) only for the quadratic J scale; otherwise it falls back to the trust-radius scale. The helper's Armijo constant is now exposed in the pinned policy as (10^{-4}).

The search sends the scaled full 26-control vector to the existing bounded original-J helper. The helper applies the actual whole-control radius before each FV evaluation and tests only original-J Armijo plus the candidate's own strict endpoint branch/margins. It records branch-signature change against the base without making equality to the base signature an acceptance rule. Φ decrease is not an acceptance requirement; same-point Φ reproduction in the base/final checks is an integrity check. A nonfinite gradient/Φ or nonfinite HVP-derived diagnostic is a numerical refusal, not a Φ-decrease gate.

The helper's line-search multiplier is applied to the already scaled vector. The result records α as `initial_alpha × helper_alpha` and the actual control displacement from the candidate vector; this matches the accepted control path. The base-branch HVP is used to choose an initial J scale only. It is not an acceptance model after a branch change.

The final CLI wiring passes `--resource` and `--log` to the child invocation, and both are required parser arguments. I inspected the subprocess command and parser path: the parent invokes the actual command-line parser with those values, while the outer guard owns resource/log receipt creation. The added regression exercises this parent-child command through the real parser with the guarded process mocked; it does not start an FV run.

## Commit, receipts and interpretation

Final original J/full gradient/Φ and strict branch are freshly recomputed at a tentative accepted endpoint; source, fixed-input, runtime and deadline closure precede the commit record. Failed final checks or budget refusal mark the tentative trial uncommitted and retain the base as the committed state. Parent/resource status and child numerical status remain separately visible. The importer verifies the prior face diagnostic's raw and gzip hashes, lossless decompression, archive metadata, parent/resource agreement and zero optimizer/HVP/PCG/score/response counters. It reads the old QY32 raw/gzip records and writes only to a fresh attempt location.

No prelaunch blocker remains. The result, if a step is accepted, may support only one finite original-cost decrease along this sampled full-space direction at the fixed problem. It does not establish face crossing, monotone Φ, convergence, a root, forecast improvement or causal attribution to one face. The accepted trial's stored control and face/branch evidence must determine whether it actually crossed; do not infer crossing from the direction alone.
