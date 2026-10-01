# Parameterized FV adjoint tolerance diagnosis

## Governing derivative

At a stationary control `c*(theta)`, let `F(c, theta) = grad_c J(c, theta) = 0`. The score derivative is

`dS/dtheta = partial_theta S - lambda^T partial_theta F`,

where `H^T lambda = grad_c S` and `H = partial_c F`. In this fixture the final scalar subtracts two large terms, so the adjoint residual can be amplified in the reported total even when PCG's requested relative residual is small.

## Controlled fixed-point comparison

The same parameterized refined case, control, objective, exact robust Hessian operator, diagonal preconditioner, score, and parameter were used for both solves. Only PCG `rtol` changed through the existing diagnostic observer. The independently formed dense 24-by-24 Hessian oracle and already measured centered FD at `h=1e-4` were reused; no endpoint reanalysis was rerun.

| PCG rtol | iterations | fresh relative residual | direct theta | indirect theta | total theta | error vs dense oracle | error vs centered FD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-10` | 20 | `3.5941e-11` | `87.00032062946823` | `-87.05010317862690` | `-0.04978254915866387` | `2.2708e-9` | `2.4690e-9` |
| `1e-12` | 25 | `3.3695e-13` | `87.00032062946823` | `-87.05010318086545` | `-0.04978255139721455` | `3.2230e-11` | `2.3047e-10` |

The score stayed `436.41272550968233` and the stationarity max-gradient stayed `1.3283e-11`. Tightening the solve added five normal products and reduced the response-to-FD discrepancy by about 10.7 times and the response-to-dense discrepancy by about 70 times. The centered finite-difference step and its `rtol=1e-7`, `atol=2e-9` assertion are unchanged.

The guarded diagnostic completed in 29.50 seconds with sampled peak RSS 338,493,440 bytes under 170-second and 1-GiB sampled limits. The earlier centered FD sweep found `h=2e-4` versus `1e-4` differed by only `1.42e-10`; shrinking to `2.5e-5` worsened the FD difference to `6.45e-9`. Endpoint polished max-gradients were between `7.65e-12` and `2.31e-11`. These measurements support adjoint stopping error amplified by cancellation, not endpoint stationarity or leading `h^2` truncation, as the CI mismatch source.

## Precision policy and validation

`compute_fv_observation_response` now accepts `adjoint_relative_tolerance` in `(0, 1e-10]`, defaulting to the historical `1e-10`; booleans, huge integers, and other invalid values are rejected before further validation. PCG uses this request and the routine separately recomputes and gates the true residual at the same threshold. The independent residual product remains inside `maximum_normal_products`; for exactly zero RHS, the zero solution's residual is known without another operator product. The polished theta finite-difference regression requests `1e-12`; its perturbation step and assertion tolerance are unchanged.

Focused CPU FP64 validation used the existing 165-second / 1-GiB sampled resource guard: 14 passed, including invalid tolerance inputs, zero-score response, dense exact-oracle agreement, polished centered reanalysis, and the tighter-residual gate. Elapsed time was 96.66 seconds; sampled peak RSS was 339,787,776 bytes. No GPU, broad suite, or official CI rerun was performed for this patch.

## Source identity boundary

The failing official CI run is immutable evidence for HEAD `a9e321d15e4f7fc80b4ea494e1c732897fa0ccdf` and its original source hashes; it is not rebound to this patch. For comparison, pre-change SHA256 values from that HEAD were:

- `src/advar/fv_sensitivity.py`: `3ed5f840b7316eb77e2e5f403b484036c9a9f7ba32523198dc208e5684d9bdcb`
- `tests/test_fv_observation_response.py`: `4ce59e37007fc04ce0daf4a4e16547611645bebe4349c9f810de2308cf3ee1d5`

Post-change SHA256 values are:

- `src/advar/fv_sensitivity.py`: `2fdc5d4683fa57ebad9f987e28f92d46ead82f07f751592c98ea98a66dc7905d`
- `tests/test_fv_observation_response.py`: `7b7be8f9b73de82b9f2767fa7aec0051102a7747b512af8bbc524d328c719123`

A new official CI result must bind to the eventual tested source tree separately.
