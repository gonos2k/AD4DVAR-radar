# PR 267 continuation resume — GREEN preflight — 2026-10-09

## Disposition

The saved PR 267 run supports a well-defined continuation base at the third accepted endpoint. Its final accepted trial and independent final repeat agree on control, theta, native objective, residual merit, both one-sided gradients, and current branch pair. The archived evidence is hash-consistent. The old run's direction/HVP arrays are at the preceding base, so they cannot supply a current HVP at this endpoint. No FV, gradient, HVP, guard, or weather calculation was launched for this preflight.

## Archived run and receipt identity

The frozen plan is SHA-256 `cc9c3a9baa6fdcbf6216d8a5d1613ebfaf63e708c279c5341aaba77cc35e3c69`. The archive manifest's raw hash is `77726dcdf533bb5b8949b97d7bebf39e28820c00240cdeb16401170a01f9ebd1`, and its gzip hash is `62d92dd38a6f97db4871d464201183db9b4bbda31ea224bb8c390646b917958b`; decompressing the gzip exactly reproduces the retained raw bytes. The run receipt hash is `7bd3dbbea7735413e6f3ea0fe34ebd4c46081694443b7d3f4eb8501d62d36854`, resource receipt hash is `0712e9ea77b8253502dc7ae290bd1968d84fd6b123ebea279dac8665f7ad609a`; both match the archive manifest, and the nested resource receipt matches the separate resource file. The run completed with exit code 0, 76.515 seconds elapsed, and sampled peak RSS 570,179,584 bytes. These receipts describe the original PR 267 source/run only.

The old PR 267 plan was source-bound to the runner and test snapshots in `tangent_actual_source_20261009/manifest.json` (runner `b184d15f…`, test `8cfe139f…`). Those exact archived bytes match the original plan pins. PR 268 follow-up patches changed those two live files; therefore, the old plan must refuse the current live source. A new plan must freeze the current live source and pin the old plan/archive/raw/run/resource as immutable input evidence. Do not edit the old plan pins or treat the old run's `source_unchanged` as a closure claim for new source.

## Exact continuation base from the final fresh repeat

Use the final accepted endpoint, not the preceding HVP base:

- Point: `iterations[2].final_repeat` in `tangent_continuation_20261009_attempt1/step.json.gz` (also recorded as `last_confirmed_closure`).
- Control: 26-vector whose little-endian FP64 SHA-256 is `ed106d7bc277906141e9d4a52aea529c575c9ac17df334fcbfb8d9c5013d44f7`.
- Theta: `0.332026125873074`.
- Native objective `J`: `0.0612512070514904`.
- Selected face: `Q=0`, axis `y`, row 4, column 3, weights `[0, -0.08, -0.28, -0.14, -0.84]`, residual scale `0.84`.
- Final-repeat one-sided gradients, in the raw array order and encoded as FP64 values:
  - `g_minus`: digest `a83403829ac039d0a94f88dd174d8a39b377baa5a3de0578d28498294bcdef37`.
  - `g_plus`: digest `fd2f91fa2813a9e1ec50a6301da8db6c3cd46ed227a6c136db8bba0b2719cc7a`.
  - Digests above are SHA-256 over `np.asarray(values, dtype='<f8').tobytes()`; the full vectors are retained at the raw path above.
- The final trial control/theta/J and both side-gradient arrays exactly equal the final-repeat values. The final repeat reports native-J and `F²` proposal matches, face and branch-pair pass, trace match, fixed-input/source/runtime/deadline closure pass. Both side trace signatures are `67524657bd4168588075f4ee0285dd7bbf9f95f33d9bf33b9c0836497a652c69` (360 observed stages).

Recomputing only static face geometry and arithmetic from this saved endpoint gives `F=[g_mix,Q/0.84]`, `F²=0.005238386295728529`, `||g_mix||=0.07237669718720612`, tangent-gradient norm `0.07237388241205849`, and mixed normal component `0.0006383105310589993`. The one-sided gradient jump's tangential norm is `8.78e-17`; it is supported by the selected face normal. This is a continuation seed with a substantial remaining tangent residual, not a stationary point.

The fixed input identities retained by the endpoint are parameter SHA `8871db49c227c03f9c155c3250da5acc9c30de3be18334921c464010cd4ad6ed`, fixed problem SHA `16fa95b534c3b61cf5d1506770cbfc77edb527c40d1f4c536f5130cba24e1fc3`, and terminal truth SHA `b927c1a39d16f05285cccb7a9ff48d182eea539bfba746f7aff446aee0974610`. The archived input control SHA `adc5bfd0…` is a distinct historical identity; the accepted continuation control is `ed106d7b…`. Keep those roles separate. The endpoint runtime receipt is CPU FP64, Python 3.12.13, Torch 2.13.0.

A new execution should bind its start to this exact control and theta, freshly establish current-source input/runtime/branch/side-gradient closure, and then compute a new tangent direction and two new side HVPs at `ed106d7b…`. The last historical HVP pair belongs to base `747e5506c4e2…` at theta `0.3288283299901934`; neither that pair nor any previous direction is reusable at the new point. Keep the prior objective/prior/observations/boundaries/time and policy fixed, with no more than three accepted steps, six fresh HVPs, and 16 candidates per step.

## Scalar `F²` line-model geometry from the three saved iterations

For each saved base, let `DF` be its recorded residual direction and consider the one-dimensional linearized merit `m(a)=||F+a DF||²`. Its minimizer is `a*=-F·DF/||DF||²`. The maximum fractional decrease in this scalar model is `cos²(angle(F,-DF))`, where the angle is computed from the saved vectors. Saved accepted alphas equal this scalar minimizer; the calculations below do not evaluate the nonlinear FV model.

| Step | `F·DF` | `||DF||` | angle(`F`, `-DF`) | `cos²` = max model decrease | `a*` = accepted alpha |
|---|---:|---:|---:|---:|---:|
| 1 | -0.6364752577179492 | 21.411907900375 | 68.0327418119° | 13.9933370616% | 0.00138825994168 |
| 2 | -0.4040759366983856 | 33.917618027298 | 80.7114051068° | 2.60523415746% | 0.000351246737024 |
| 3 | -0.12618590394801418 | 16.437766438955 | 83.9455995669° | 1.11244865225% | 0.000467008980967 |

These decreasing `cos²` values show that each selected scalar step becomes less effective for the saved local linearized merit. They are model geometry, not convergence bounds. The saved actual nonlinear endpoint `F²` reductions and native `J` checks remain separate; candidate acceptance must continue to require fresh actual J/F² Armijo and endpoint closure.

## New-base contract for a limited follow-up

A new source-bound plan may continue from `ed106d7b…` only if it records the old immutable evidence hashes above, selects `iterations[2].final_repeat` as the base receipt, and binds its current execution to control hash `ed106d7b…` and theta `0.332026125873074`. It must carry the unchanged fixed problem/parameter/truth identities, selected face and face scale, native `J`, saved `g_minus/g_plus`, and side trace signatures as comparison evidence. At start it must validate the current live source and freshly re-establish the point under that source before constructing a current direction; the saved old `source_unchanged` flag is historical evidence only. Then it may execute at most three accepted current-tangent corrections with two freshly computed side HVPs per accepted base (six total), checking each actual candidate and independent repeat. Do not call the old live `_load_plan` with the PR 267 plan, reuse the old base's HVPs, or claim a root/minimum/response/forecast result.
