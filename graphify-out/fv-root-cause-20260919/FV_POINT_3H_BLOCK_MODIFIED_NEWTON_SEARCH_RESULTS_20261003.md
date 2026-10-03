# One shifted block modified-Newton search at the precision terminal

One bounded original-objective search ran from the refused R6 terminal control (SHA-256 `a890f344503cb3da02919de45f1c5b244beac300b1d4d2ca36f403c653e42b6d`). The R7 terminal audit reported two negative eigenvalues in the original full Hessian and Schur complement, so the ordinary full-Hessian inverse is not an SPD PCG preconditioner.

The search used the shift

\[
\mu=\max(0,1-\lambda_{\min}(S))=8.0612614799,
\]

on only the six dynamics coordinates. The shifted block factorization was SPD: the shifted full matrix minimum eigenvalue was `0.0862395`, Hff minimum eigenvalue `0.973310`, and shifted Schur minimum eigenvalue approximately `1.0`. PCG applied the actual modified operator (A_\mu v=H_{\rm true}v+(0,\mu v_d)), using the shifted block inverse only as its preconditioner. It converged in one iteration with modified-system relative residual `5.7208e-13` and resolved negative slope `g^T s=-0.534002`. The separate original-H Newton residual relative norm is `1.12800`, so this is not an original Newton solve.

The line search evaluated the unchanged original (J), all original priors, and the strict 3,600-stage branch at each finite candidate. Scales `1`, `0.5`, `0.25`, and `0.125` passed their strict endpoint checks but failed Armijo. One candidate was accepted at `alpha=0.0625`, reducing J from `0.1309895304` to `0.1077570519` (Armijo limit `0.1309861929`). The accepted branch was recorded at the endpoint; this does not certify a connecting path.

At the accepted endpoint, the full-gradient maximum is `3.472450616` and Φ is `18.59197598`, up from `9.65494414` at the start. Thus the step reduced the original objective but did not establish stationarity. The six dynamics controls were allowed to change as search variables; all 13 parameters and the underlying data/time/prior identity remained fixed. No full-root, adjoint, response, score, or physical-validity claim was produced.

The guarded child completed in `66.385` seconds with exit code `0`, no external termination or monitor error, and peak sampled RSS `374,226,944` bytes under the 300-second / 1-GiB limits. Git HEAD remained `480f1cebdb485f451498f99893442b096a9259be`; all 60 captured source paths and the pinned terminal raw/input identities matched before and after. Authoring verification used six synthetic/pure tests, offline basedpyright with zero errors, and a clean diff check; no FV/search execution occurred during authoring.

## Scope

This is one block-modified original-J search step. It is not a root solve: Φ rose, and the original-H residual remains large. No further phase is authorized by this result; the next step requires a separate root decision.
