# PR #259 model-guided resume — GREEN 최종 사전 실행 검토

날짜: 2026-10-08
판정: **GO — 고정된 resume 계획으로 단일 bounded launch 가능.** 실제 HVP 부호와 actual candidate merit는 실행 결과에서 판정한다.
범위: 고정 resume plan, 기존 full-space kernel의 변경·snapshot, accepted 69c4 producer raw/parent/resource 계보를 읽기 전용으로 검토. 이 검토자는 FV, objective/gradient/HVP/PCG/forecast/adjoint/reanalysis를 실행하지 않았다.

## Plan과 고정 source/archive 계보

Plan `MODEL_GUIDED_FULL_SPACE_PLAN_20261008.json`의 SHA-256은 `9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c`다. 111 source pin과 50 archive pin을 각각 현재 바이트와 대조했고 불일치가 없었다. Resume base는 69c4 control `69c4a794730750c09c1bcddf542b66131f64f5bfec362e02ac269413a75d4992`, parameters `8871db49…`다. Accepted producer step/run/resource raw, child SHA, control SHA와 plan lineage를 별도로 닫는다.

PR259은 중복 wrapper를 추가하지 않고 model-guided full-space kernel을 재사용한다. 편집 전 producer module/test bytes를 별도 immutable snapshot으로 보존하며 plan은 변경 허용 경로를 self와 focused test 두 개로 한정한다. Old/new AST proof에는 policy·model alpha·first-order model·merit acceptance·current HVP·static faces·commit helpers·imports·full iteration loop의 mathematical code가 모두 동일하다고 기록되어 있다. Original problem/objective/transport/HVP operator source는 기존 pin에 고정되어 있다.

이전에 작성한 rejected preflight는 metadata-only loader refusal이었다. Correction receipt는 guarded launch 0회, HVP 0회, 기존 수치 자료 미변경이라고 기록한다. 원인은 `accepted_branch`를 raw top-level에 요구한 schema 오판이었고, 수정된 loader는 raw의 마지막 accepted trial과 `current_state.branch`를 비교한다. 수정 뒤 real `load_plan`/`load_base`가 통과했다.

집중 결합 시험 기록은 19개 통과, 기존 경고 18개, 2.69초이며 basedpyright error-level 오류 0이다. 이 검토자가 해당 명령들을 재실행하지는 않았다. 저장 AST/source proofs와 test log는 preflight 근거이며, 연구 계산 완료 증거가 아니다.

## 시작점 identity와 fresh derivative

PR258 accepted raw에는 실제 PR259 base가 끝 state로 들어 있다. 독립 읽기 전용 대조에서 `current_control_sha256`, 마지막 iteration의 `accepted_control_sha256`, summary final control은 모두 69c4다. 마지막 accepted gradient/control은 raw `current_state`와 이어지고 parameters, archived observation/truth identity, runtime이 보존되어 있다. Optimization control은 이전 base에서 달라졌으므로 `input_before==input_after` 전체를 요구하지 않고, adapter는 immutable `archived_input`·parameter/truth hashes를 보존하면서 current control SHA를 새 base에 결합한다.

새 실행은 fixed problem을 재구성한 뒤 69c4에서 원래 full-26 `J`와 gradient를 다시 평가해 producer accepted state 및 strict branch digest와 비교한다. `d=−g_fresh`를 만든 뒤 같은 current point에서만 새 HVP `Hd`를 계산한다. 이전 tangent HVP, old direction, e29/058 alpha, 평균 tangent covector는 재사용하지 않는다. 이후 committed iteration은 그 endpoint의 final fresh-recheck gradient가 next `g`가 되고, 다음 HVP도 그 새 current point에서 계산한다.

각 current point에서

```text
gᵀd = −||g||₂²
alphaPhi = −gᵀHd / ||Hd||₂²
alpha0 = min(1, 0.05/||d||₂, alphaPhi)
```

를 사용한다. `gᵀd`와 `gᵀHd`는 scale-aware roundoff budget를 넘어 음수여야 한다. alphaPhi는 선형화 gradient merit의 step cap이며 actual decrease를 보장하지 않는다. Full raw-control path는 straight여서 `gamma''=0`; HVP가 양의 `dᵀHd=−gᵀHd`를 줄 때 optional local quadratic J diagnostic은 `alphaJ=−gᵀd/(dᵀHd)`다. 이전 QY32 curved-chart `alphaPhi`/`alphaJ` numeric values는 다른 point/direction/path 값이므로 PR259에 복사하면 안 된다.

## Current 69c4의 archive-only 상태

69c4 summary/raw로 확인한 시작량은

| 항목 | 보관값 |
|---|---:|
| `J`, `Phi` | 0.0613692524426, 0.0181317328195 |
| `||g||₂`, `||g||∞` | 0.190429686864, 0.0972450338224 |
| 최대 gradient control | `c[24]` |
| tangent projection norm | 0.1699788452 |
| 단위 Euclidean normal gradient | −0.0858525354 |
| tangent/normal norm ratio | 1.979893 |
| production `Qy[3,2]` | +1.3794752636×10⁻⁵ |

Static face signs were reconstructed from the saved 26-vector and pinned integer-coordinate streamfunction basis, without integrating FV. From PR258 start `05889584…` to the PR258 final `69c4…`, exactly one static face sign changes: zero-based `Qy[3,2]`, one-based **`Qy[4,3]`**, from `−5.9106127381×10⁻⁷` to `+1.3794752636×10⁻⁵`. All 24 `Qx` face signs and the other 24 `Qy` signs remain unchanged; neither endpoint has an exact-zero static face. The new resume therefore begins on the positive side, with no persistent face constraint. These endpoint static signs do not certify the intervening SSPRK/minmod path.

The final max-gradient index is flow control `c[24]`; growth is `c[25]`. The previous e29-to-058 angle (91.09°) is not the PR259 starting history. The actual 058-to-69 gradient angle is 63.133623°. At 69c4, the normal and tangent components are both nonzero; this supports use of the full-space negative gradient and does not imply a face stationary point.

## Actual candidate and bounded-loop gates

For every candidate, the kernel uses the unchanged original 26-control objective and all priors, evaluates actual `J`, fresh full-gradient `Phi`, and the endpoint’s own strict branch, and requires both actual Armijo inequalities (`c1_J=c1_Phi=1e−4`) plus a resolved actual Phi decrease. It performs at most 16 dyadic trials per iteration. Branch signature changes may occur and must be recorded; endpoint strictness does not certify a smooth between-point path.

The plan allows one guarded launch, at most 3 accepted steps and 3 current HVPs total, PCG 0, internal 240 seconds, external 300 seconds, sampled RSS 1 GiB. On direction/HVP/merit/grid/budget refusal it preserves all earlier committed points and does not commit an unaccepted candidate. Each successful endpoint must pass fresh objective/gradient/branch recheck plus source/input/runtime/deadline closure before it becomes the next start.

No extra experiment cap, permanent face constraint or new stationarity threshold is introduced. Reaching the existing `1e−10` max-gradient gate is only `root_pending_audit`; it is not a root certificate without the separate existing final audit. Three steps do not complete full curvature, adjoint/VJP, reanalysis or independent future-score validation.

## Recommendation and limitation

**No must-fix prelaunch issue found.** The loader’s accepted-control schema matches the actual producer raw; frozen numerical operator sources and old self/test bytes are retained; new state starts at exactly 69c4 and receives fresh derivatives. Fresh HVP sign, `alphaPhi`, line-search acceptance, candidate branch changes and the max-three-step outcome remain execution-time findings. No alpha-only causal, speed-up, root, same-branch-path or forecasting-skill claim follows from this preflight.
