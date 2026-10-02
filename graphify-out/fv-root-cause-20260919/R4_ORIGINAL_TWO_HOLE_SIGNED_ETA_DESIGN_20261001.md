# R4 original two-hole problem: signed-η slice experiment

**Design only; not executed.** This proposes four fixed-coordinate slice solves on the unchanged original objective. It makes no cusp, Clarke, root, or response claim.

## Frozen control and chart

Use the pinned 26-control vector `a78d1b8e0cd8f997572f5e2a913f7023dcd6333c124ac4b7c9414358f8484e26`, branch signature `50d3b1a4bad6761b14806c62708400d87d16e506ab08ef545f7ba9d870bece6d`, and original parameter tensor SHA `e3a45fac86e622332c6afd3472b46d58ef97cd37db1445260c86923a3c423bf9`. Preserve the original 4×5 observations (58 valid, two missing), masks, prior, verification field, boundaries, times, parameters, objective, and source/runtime/archive identities byte-for-byte.

For the production polynomial basis at `q_y[2,0]`, `c=(z,u,γ)` has field `z∈R²⁰`, five bounded-flow latents `u`, and growth `γ`. The oriented face weights are `w=(0,-1,-2,-1/2,-2)` and limits `L=(0.11,0.08,0.07,0.04,0.03)`. With pivot `p=1` (full slot 21),

```text
Q(c) = q_y[2,0] = Σⱼ wⱼLⱼ tanh(uⱼ),  s=|wₚ|Lₚ=0.08
aⱼ = wⱼLⱼ/s = (0,-1,-1.75,-0.25,-0.75),  η=Q/s.
```

The pinned face-only artifact records `Q=+7.771623999008881e-6`, so `η₀=+9.714529998761101e-5`. Its hash is `a198b66f84b979632e26c34095761b59094335f3822cce10da8e416f461e65bf`. The current field-corrected evidence (hash `8ad5b5e3d56a074ad79a28fa0f23c0cbdae745c26efed5101cd826fe2d9442cf`) records `||g||∞=0.0061284025566`, field `||g||∞=7.03e-12`, dynamic `||g||₂=0.00757888644`, and global scaled face/slope margins `1.1662714171735749e-4` / `4.0608748624064856e-4`. These baseline values do not predict slice results.

Let `x∈R²⁵` be all original slots except slot 21, with `η` inserted there. `C(x,η)` copies the other 25 values and sets

```text
tanh(uₚ)=(η-Σⱼ≠ₚaⱼtanh(uⱼ))/aₚ,  uₚ=atanh(tanh(uₚ)),  -1<tanh(uₚ)<1.
```

No clipping. The four predeclared signed slices are `η∈{-2η₀,-η₀,+η₀,+2η₀}` (approximately `{-1.942906e-4,-9.71453e-5,+9.71453e-5,+1.942906e-4}`). Do not evaluate `J` or its derivative at `η=0`. Start `+η₀` from the pinned control; initialize `-η₀` by changing only the pivot through the inverse chart; continue outward within each sign. Freshly establish each slice’s start branch. Do not require signatures to match across endpoints or signs.

## Slice solve and normal derivative

For fixed original parameters `p₀`, solve `rₓ=∇ₓF=Cₓᵀg=0` for `F(x,η)=J(C(x,η),p₀)` using the existing `refine_stationary` Newton–PCG path and unchanged gradient-merit Armijo/refusal behavior. Its exact Hessian action must include

```text
Hₓ = CₓᵀH꜀Cₓ + Σᵢ gᵢ∇²ₓCᵢ.
```

The `gᵢ∇²Cᵢ` term is required away from a full stationary point; the transformed Hessian may be indefinite. Retain the fixed-field method’s actual-`J` roundoff guard unchanged (`128*eps64*max(|J_old|,|J_new|,tiny64)`). For every slice, save the full 25-vector `rₓ`, its norms, PCG convergence and independently recomputed true linear residuals, iteration/HVP counts, and refusal reason. Use the pinned refiner’s stationarity threshold `||rₓ||∞<1e-10` and true linear-relative-residual threshold `<=1e-10`; save the values, not only pass flags.

Freshly compute the full original 26-gradient `g=∇꜀J` at each endpoint: save its vector, L2/infinity norms, and field/flow/growth block norms. Save each start, accepted, and final full 26-control vector/hash, original `J`, and unchanged parameter hash as well. The original physical `||g||∞<1e-10` gate remains authoritative; small `||rₓ||` only means tangent stationarity. Also save

```text
σ = ∂ηF = gᵀCη,       λ = σ/s  (per unit physical Q),
g·(∇Q/||∇Q||₂),       ||∇Q||₂.
```

At an exact tangent-stationary point on a differentiable, nonsingular stationary branch `x*(η)`, `ψ(η)=F(x*(η),η)` has `ψ′=σ`. Without that branch regularity, report `σ` only as the partial normal slope at the endpoint. It is a local minimum-envelope derivative only if tangent-Hessian positive curvature is independently established; Newton–PCG convergence alone does not prove that. Call `σ` resolved only when `|σ|>128*eps64*||g||₂*||Cη||₂`; otherwise report it as roundoff-indeterminate. This is a dot-product roundoff floor, not an optimizer-error bound. The solver’s Euclidean merit is on chart coordinates `x`, distinct from projecting an ambient gradient onto an event hyperplane.

## Admission, budget, and interpretation

At every start, accepted trial, and final endpoint, run the full branch checker fresh. Trial admission keeps the existing strict branch rule: complete finite trace, all scaled slope margins positive, global minimum scaled face-flux margin `>0`, and the within-slice starting signature unchanged. Do not apply the response threshold as a substitute branch test. Response eligibility remains a separate stricter gate: the existing global minimum scaled slope and face margins must each be `>1e-4`. Record both; a tangent-stationary endpoint failing either response margin remains response-ineligible. Endpoint agreement does not certify the connecting path.

Run one serial guarded child: at most 8 Newton corrections, 16 backtracks, and 80 PCG iterations per slice; 300-second wall trigger and sampled 1-GiB RSS trigger. Pin design/probe/runner hashes and retain all existing input, source, archive, parameter, and runtime checks. No retries, budget increases, cache shortcuts, or gate relaxations. Preserve the last accepted endpoint and exact refusal status.

Interpretation is deliberately narrow. A same-sign pair with opposite resolved `σ` values is only a finite indication of a possible smooth-side stationary-branch zero, and only when both slices meet the `1e-10` tangent gate, their fresh signatures match, and strict branch checks pass. Do not bracket across `η=0` or claim a root without path and endpoint qualification. Small `rₓ` with nonzero full `g`/`σ` at these tested slices indicates slice stationarity with remaining normal residual, not absence of a closer root. A smooth full-root candidate requires an endpoint passing the original full-gradient gate plus a fresh full 26-coordinate final-curvature and branch audit. Response still requires every original global-margin and signed-response gate. This experiment neither certifies a full root nor evaluates an event derivative, cusp, Clarke set, limiting-gradient convergence, or response; `R4-R` remains open otherwise.
