# G3b branch-gate detail, nominal-only attempt 3

Under the user's autonomous checklist instruction, the separately frozen
`FV_PARTIAL_BRANCH_GATE_DETAIL_PLAN.md` ran once with the same partial
problem/input identities as attempt 2. This run only added raw callback
reason counts to the existing fixed policy. Its six unmodified child/guard
outputs are in `partial_branch_gate_detail_attempt3/`; the manifest pins
their SHA256 and all 14 measured source files to commit `d7fab34`.

| Axis | Measured outcome |
|---|---|
| Guarded preflight | exit 0; 2.061 s; sampled peak RSS 315,883,520 bytes |
| Numerical child | exit 1; 136.746 s; sampled peak RSS 344,702,976 bytes; no resource termination |
| GN / Newton | same GN 4 outer / 83 PCG iterations; fourth Newton line search refused after its fourth 29-iteration PCG solve |
| Final 16 backtracks | **3 first failed at exact branch signature; 13 first failed at face-flux margin ≤ 1e-4** |
| Other final refusals | 0 nonfinite candidates and 0 finite Armijo evaluations/refusals |
| Response | No nominal stationary point, tangent, adjoint or signed endpoints; validation **not established** |

The current fixed GN-branch correction policy cannot reach a qualifying
nominal point from this seed within 16 backtracks. Three candidates reached
the strict tracer and quantitative margin checks but differed in the
selector/face-sign signature; thirteen reached a face-flow margin at or below
the predeclared `1e-4` floor **before their signatures were compared**. The
counts are first-failing-predicate counts; some of those thirteen could also
have changed signature. Their actual margins and limiting faces were not
saved. The raw report's single
`failure_category=face_flux_margin_refusal` is an over-specific summary of
the **mixed** 3+13 first-failure reasons; the full `error` field and manifest
preserve the correct counts. That bookkeeping label was broadened for future runs
without changing this source-bound raw record.

The evidence does **not** establish that the partial inverse problem lacks
a stationary point. It does not justify deleting the margin or GN-signature
guard. The existing Newton–Armijo prediction is built from the Hessian of the
current smooth minmod sector; crossing a limiter/upwind kink would require a
separately justified branch-aware globalization method. A strict final
branch, if found, would support only a conditional *local* derivative.

G3b remains open. A fourth repeat of the same GN/Newton policy would be
unlikely to add evidence. This **exact two-hole case is refused by the
declared nominal correction policy**, and therefore currently ineligible for
its exact response. A separately declared partial-observation case matrix can
measure how often the policy admits other inputs. If nominal search across
limiter/upwind sectors is needed, design a branch-aware globalization and
validate it as a new numerical method. Do not weaken
normality, linear residual, or endpoint response criteria to force a pass.
