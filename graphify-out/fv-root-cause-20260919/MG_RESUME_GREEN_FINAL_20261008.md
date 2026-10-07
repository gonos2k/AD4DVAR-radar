# PR #259 resume — GREEN 실제 결과 검토

날짜: 2026-10-08
판정: **선언한 세-step continuation 실행과 receipts를 수용한다.** 이 결과는 root나 response certificate가 아니다.
범위: resume raw/parent/resource, result/boundary/independent summaries 및 PR259 base artifact를 읽기 전용으로 대조했다. FV, objective/gradient/HVP/PCG/forecast/adjoint/reanalysis를 다시 실행하지 않았다.

## Lineage·receipt closure

재개 base는 직전 model-guided 3-step의 마지막 accepted point `69c4a794…`다. 새 raw의 `base_control_sha256` 및 start current control은 정확히 `69c4a794…`; 최종 accepted control은 `2cdccade…`다. parameters `8871db49…` 및 archived-input/truth/problem identity는 유지됐다. 이전 raw의 last accepted state·resource receipt·source manifest를 통해 시작점을 고정한 계보는 일관된다.

Resume plan SHA는 `9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c`. 이 계획은 111 source와 50 archive를 pin한다. Producer source snapshot은 변경된 self/test bytes만 허용하고, numerical AST evidence는 optimizer/helper/import/iteration loop가 이전 kernel과 동일하다고 기록한다. 원래 수치 operator source를 바꾸지 않았다.

Raw SHA `66aa57c3…`는 parent child SHA 및 independent-check summary의 raw SHA와 일치한다. 상태는 `completed / max_iterations_completed`, optimizer commits 3, HVP started/completed 3/3, PCG 0이다. Iteration 1 base gradient/control은 69c4 producer state와 일치하고 iteration 2·3 base gradient/control은 각각 직전 accepted state와 배열 단위로 일치한다. Guard 188.2606546초, sampled peak RSS 385,761,280 bytes, exit 0, SIGTERM/resource/monitor error 없음이다. Raw source map 156 paths는 before/after 동일했고 현재 파일의 SHA mismatch는 0개였다. Input 및 runtime closure도 true다. RSS는 0.25초 표본 최대이지 OS hard limit 증명은 아니다.

## 방향, step cap, actual merit

세 반복 모두 base gradient 배열에 대해 `d=−g`가 성분별 정확히 맞았다. Saved fresh `Hd`를 이용한 독립 산술은 다음을 확인했다.

| Iteration | `g∞` | `gᵀd` (`−||g||²`) | `gᵀHd` | `alphaPhi` = `alphaStart` | 채택 alpha | displacement |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.097245034 | −0.036263466 | −10.6153374 | 0.001494878663 | 0.000373719666 | 7.11673×10⁻⁵ |
| 2 | 0.092597766 | −0.029278478 | −6.7249120 | 0.002173529215 | 0.000271691152 | 4.64890×10⁻⁵ |
| 3 | 0.088835566 | −0.025757380 | −5.2581202 | 0.002751530066 | 0.000042992657 | 6.89994×10⁻⁶ |

반경 alpha `0.05/||d||₂`는 각각 0.26256, 0.29221, 0.31154로 `alphaPhi`보다 컸다. 각 반복의 accepted candidate는 실제 full-26 `J`, fresh-gradient `Phi`, own strict endpoint branch를 모두 통과했다. Acceptance threshold와 independent NumPy arithmetic의 residual은 0 또는 FP64 roundoff 수준이다. AlphaPhi는 linearized-gradient 모델의 initial cap이고 actual Phi decrease 보장은 actual Armijo 재평가가 담당했다.

기준점→최종점의 전체 변화는 다음과 같다.

| 양 | 69c4 시작 | 최종 2cdccade | 변화 |
|---|---:|---:|---:|
| `J` | 0.061369252443 | 0.061347638323 | −0.0352198% |
| `Phi` | 0.018131732819 | 0.012673246503 | −30.1046037% |
| `||g||₂` | 0.190429687 | 0.159205820 | −16.3965334% |
| `||g||∞` | 0.097245034 | 0.088510893 | −8.9815808% |

세 단계는 모두 한 번의 fresh current HVP로 시작했다. 각 다음 base gradient는 이전 endpoint의 final fresh recheck gradient 배열과 일치한다. 이전 HVP나 tangent direction을 새 point에 재사용하지 않았다.

## 외부 `Qy[4,3]`와 internal face

Static streamfunction geometry를 보관된 full-26 controls에 대입해 재계산했다. 여기서 `Qy[4,3]`는 storage의 zero-based index다.

```text
external Qy[4,3]:  −5.75220e−5 (69c4)
                  → −2.66999e−5 → −5.55503e−6 → −2.37504e−6
internal Qy[3,2]:  +1.37948e−5 (69c4)
                  → +3.02440e−5 → +4.14794e−5 → +4.31613e−5
```

Accepted points에서 두 값 모두 0이 아니며, external face는 음수로 0에 가까워지고 internal face는 계속 양수다. Static face sign은 매 accepted endpoint에서 변하지 않은 채다. 별도 저장 boundary diagnostic은 external face를 양수로 만드는 더 큰 주변 후보들이 strict endpoint branch 및 `J` Armijo를 통과해도 actual `Phi` Armijo에서 거부됐다고 기록한다. 3번째 iteration에서 제일 큰 후보는 `J`도 거부됐다. 그러므로 strict branch checker가 부호 횡단 자체를 금지한 것이 아니라, 현재 dual-merit policy가 해당 후보들을 수용하지 않았다.

이것은 고정된 full-gradient 선분/line-search 후보의 실제 merit 충돌이지 `Qy[4,3]`만의 isolated causal test가 아니다. 방향은 모든 field/flow/growth 제어를 함께 움직이며, analysis와 future branch hashes는 세 단계마다 모두 바뀌었다. Endpoint strictness는 smooth continuous path 증명이 아니다. `Qy[4,3]`는 모델 격자 경계의 static normal flux로, 실제 바람 장벽이나 강수 경계의 증거가 아니다.

최종 external-face normal projection 진단은 현재 **음의 비영 유량점**에서만 계산됐다: full gradient norm 0.15920582, tangent norm 0.13656910, unit control-normal derivative −0.08182526. 이는 해당 면의 0점 gradient나 conditional minimum이 아니다. 접선 잔차도 남아 있다.

## Model residual 및 수렴 해석

Actual candidates는 기준점의 전체 `J/Phi`를 내렸지만, 세 단계마다 분석 360/future 3240 branch partitions가 모두 달라졌다. Independent gradient prediction relative residual은 `0.00337`, `0.00635`, `0.00191`로 작았고 quadratic-J predicted-vs-actual reduction ratios는 `0.99964`, `0.99978`, `1.00028`이었다. 이는 이 보관 경로에서의 model-vs-actual 산술이지 same-branch path·일반화 정확도 증명이 아니다.

최종 max-gradient `0.0885109`은 기존 root pending gate `1e−10`의 약 **8.85×10⁸배**다. 종료는 `max_iterations_completed`; root audit pending, full Hessian/curvature certificate, normal/active-face minimum, original adjoint/VJP/reanalysis, independent future-score 검증을 만들지 않았다. PR258의 tangent direction/temporary eta 정책에서 PR259는 full-space fresh-gradient policy로 바뀌었고, PR259 resume도 line-search candidates와 accepted update를 함께 반영한다. 따라서 세 step의 감소를 `alphaPhi`나 경계 해제 하나의 효과로 귀속하거나 baseline 대비 속도 향상으로 말하지 않는다.

## Evidence hashes

- Resume plan: `9fc5e046ca484737ebd040469990f10cc918fcd9dc245d0762e65f297151440c`
- Raw `step.json`: `66aa57c3056220a7057a5b3c3cca2a09c06d44ed34279b055174e6930ab1838e`
- Parent `step.run.json`: `604d447997c76f358fda4c8f0c895c563c1f2772171de511cf1015d19f2cf6bd`
- Resource `step.resource.json`: `67b7294e85b19022b205be03f8cb2f4198be5cd8d50c21935d92354d682344e5`
- Summary `MG_RESUME_RESULT_20261008.json`: `6de20e5a614c547c3132cc8081ed8b4039dfaf1542395d73eeee26bb28cee00b`
- Boundary diagnostic: `MG_RESUME_BOUNDARY_DIAGNOSTIC_20261008.json`, SHA `6486d3cd3e3203840cbadb6534460c834d92f4c8f6604293fefdae79eda3a941`
- Independent arithmetic: `MG_RESUME_INDEPENDENT_CHECKS_20261008.json`, SHA `eef161152f725195a8c4aeea6f11acf171bd17b60d39926826d7f6c3284f5946`
