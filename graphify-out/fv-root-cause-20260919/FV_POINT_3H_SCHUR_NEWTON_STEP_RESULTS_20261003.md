# S4 one-step original-H Newton attempt

The guarded run accepted one `alpha=1` Newton candidate under the original-J Armijo test. The PCG operator was a fresh original-objective HVP; the frozen 20+6 Hessian supplied only the block-Schur preconditioner. The solve converged in one PCG iteration with fresh true relative residual `2.207e-13` and descent product `gᵀs=-1.797633` (roundoff budget `5.16e-14`).

| Measurement | Pinned endpoint | Accepted candidate |
|---|---:|---:|
| Objective J | `0.9314353875` | `0.4254527702` |
| Gradient merit Phi | `0.9522514199` | `485.7293098` |
| Full gradient infinity norm | `0.8719793801` | `22.08953658` |
| Full gradient L2 | `1.3800372603` | `31.16823093` |

The objective fell by `0.5059826173`, but gradient merit and gradient norm rose sharply. This is one accepted objective step, not stationarity or root progress. The candidate passed its own strict 3,600-stage branch/margin oracle; its signature changed from `18a36b01…c5d0eed8` to `16ef21e8…9a52e975`. No segment/path certificate follows from those endpoint checks. The previous endpoint Hessian is not evidence of curvature at this new candidate; no candidate Hessian, response, score, adjoint, or physical validation was computed.

Parent status is completed/exit 0, one step accepted, 37.055 seconds, sampled peak RSS `367,050,752` bytes under the 360-second / 1-GiB guard. Source and fixed-input identities remained unchanged. The pinned starting Hessian raw SHA256 is `ca04e5f9712667cee95de69121d889dd8f2d33f81192b70e3724ee6fa52f224b`; this one-step result SHA256 is `b71411024191756d27bc30982800207a392e31d5b60862229ab24c181aa3fdbd`.

No full root, three-hour stationary branch, response, minimum, uniform neighborhood, or physical-weather claim is established. Further root/response work remains open and requires fresh evidence at any new accepted point.
