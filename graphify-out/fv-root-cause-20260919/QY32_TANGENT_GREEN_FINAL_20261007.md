# PR #257 Qy32 접선 보정 — GREEN 실제 결과 검토

날짜: 2026-10-07
판정: **선언한 단일 tangent-step 목표를 수용한다.** 이 한 걸음은 원래 3h 문제의 root·normal stationarity·response를 완료하지 않았다.
검토 범위: step raw/parent/resource, 계획된 summary와 현재 source/input 폐쇄의 읽기 전용 재산술. 새 FV, gradient, HVP, PCG, 예측, 수반, 재분석을 재실행하지 않았다.

## 실행 및 provenance

계획 SHA는 `70f9560c0355fa6b65927ac70fb4de2c055f338d5dd276c8fa453fc30379ecf9`다. raw `step.json` SHA `061c51d2…`는 parent child SHA와 일치한다. Parent/resource receipt는 `completed`, exit 0, monitor/resource error 없음, SIGTERM 없음이라고 기록한다. elapsed는 **59.2936179초**, sampled peak RSS는 **521,453,568 bytes**이며 300초/1 GiB 표본 한도 이내다. RSS는 0.25초 표본의 최대값이지 OS hard limit 보증은 아니다.

Step raw의 143 source/archive 경로는 before/after 동일하며, 현재 파일을 저장된 after map에 다시 대조했을 때 불일치가 없었다. `fixed_input_unchanged=true`, runtime before/after 동일, 기준 e29·파라미터 identity 폐쇄도 통과했다. 계획의 109 source 및 39 archive pin 역시 이 실행 전 계획 확인에서 일치했다.

## 현재점 방향과 실제 HVP

저장된 QY32 쪽 기울기로 만들고 e29의 실제 비영 `eta`에 lift한 방향은 보관 산술과 일치한다. 현재점 baseline의 fresh objective/`Phi`/gradient 26성분은 e29 saved state와 반올림 기준 안에서 일치했고, branch digest도 old e29 branch digest와 같다.

한 번 계산한 현재 full-objective HVP는 유한 26성분이며 raw의 내적을 독립 재산술하면

```text
gᵀd   = −0.5587502088011976
gᵀHd  = −1087.1234794291438
HVP count = 1; PCG solves = 0
```

저장 `H_direction`에서 별도로 재산술한 `dᵀHd=+1087.0354619`, directional Rayleigh quotient `dᵀHd/(dᵀd)=+1944.1663463`도 요약값과 일치한다. 양의 `dᵀHd`와 음의 `gᵀHd`는 모순이 아니다. `d`는 Newton 방향이나 Hessian eigenvector가 아니며, `g`와 `Hd`의 교차곱이 음수이기 때문이다.

이다. 둘 다 코드의 scale-aware roundoff budget를 넘어 음수다. 따라서 이 기준점에서는 원래 `J`와 smooth-base `Phi`의 local directional derivative를 각각 해상했다. (g^	op Hd)는 저장된 HVP와 gradient의 산술 검산이며 이 리뷰에서 새 HVP를 수행한 결과가 아니다.

## Backtracking 및 실제 채택 후보

8개 dyadic chart 후보를 평가했다. 처음 7개는 둘 다 `J` Armijo와 `Phi` Armijo를 통과하지 못했다. 이 7개는 모두 자신의 complete strict branch 검사에는 통과했다. 8번째의 `alpha=0.0005224020700718299` 후보는 다음을 만족했다.

| 항목 | e29 기준 | 채택점 |
|---|---:|---:|
| `J` | 0.06169926870755131 | **0.06155571301117497** |
| `Phi` | 0.27919820600857015 | **0.22354015242492317** |
| `||g||₂` | 0.7472592669 | **0.6686406395** |
| `||g||∞` | 0.4135580586 | **0.5101284967** |
| `||g_T||₂` | 0.7472444235 | **0.6512840542** |
| 단위 Euclidean normal gradient | +0.0047099507 | **−0.1513584674** |
| 실제 제어 displacement | — | **0.0003906250** |

독립 재산술한 비용 감소는 `J` **0.2326700%**, `Phi` **19.9349610%**, gradient 2-norm **10.5209304%**다. 최대 gradient 성분은 **23.3511199% 증가**했다. 원래 26 control 전체의 `J`, 새 full gradient로 계산한 실제 `Phi`, 후보 고유의 strict branch/margins를 사용했다. 채택점 `J/Phi/gradient/branch hash`는 마지막 Armijo 통과 trial과 fresh final recheck 저장값에 정확히 일치하고, 최종 source/input/runtime closure 뒤 `candidate_committed=true`로 기록됐다.

제어 경로는 곡선 chart에서 `eta`를 고정했고 실제 생산 `Qy[3,2]`는 시작과 끝 모두 `−5.9106127381e−7`이다. 즉 target 면을 넘지 않은 등면 tangent step이다. 후보의 자체 branch는 strict pass지만 branch signature hash는 시작점과 다르고, partition hash도 analysis/future 모두 바뀌었다. 이 raw에는 후보별 full per-stage signature array가 없으므로 정확히 어떤 stage/cell 선택이 바뀌었는지는 이 receipt만으로 위치 특정할 수 없다. 따라서 한 branch 위의 매끄러운 유한 경로였다고 말하지 않고, 실제 full-value Armijo 수용이라고만 기록한다.

Gradient 블록을 보면 field norm은 `0.7022406→0.3921142`로 낮아졌고, flow norm은 `0.0491297→0.1819222`, growth 성분 절대값은 `0.2506806→0.5101285`로 커졌다. 전체 `Phi` 감소는 field 블록의 큰 감소가 flow/growth 증가를 상쇄한 결과다. 접선 잔차 `0.651284`가 여전히 크므로 tangent stationary point 또는 root로 볼 수 없다. 단위 법선 기울기의 부호가 변했어도 이동은 고정-eta tangent path이므로 이를 normal step이나 경계의 법선 최소 조건으로 해석하지 않는다.

## HVP 선형모형의 해석 한계

채택 alpha에서 저장된 1차 gradient 모델은 `g+alpha*Hd`이고, raw와 별도 재산술에서 모델과 actual endpoint gradient의 차이 norm은 약 **6.2012e−4** (actual gradient norm의 **0.09274%**)였다. 하지만 같은 alpha의 scalar 선형 Taylor 식 `Phi0 + alpha*(gᵀHd)`는 **−0.28871735**다. `Phi`는 0 이상인 제곱노름 merit이므로 이 값은 실제 후보 예측값으로 사용할 수 없다. 큰 `alpha*Hd`에 대해 scalar 1차 Taylor 항만 외삽해 생긴 값이며,

```text
Phi(g + alpha Hd) = Phi(g) + alpha gᵀHd + 0.5 alpha² ||Hd||²
```

의 이차 gradient 항이 같은 크기 차수를 갖는다. 실제 수용은 이 선형 예측치나 음수 merit을 근거로 하지 않고, 실제 fresh candidate `Phi=0.22354015`가 실제 Armijo limit `0.27914141` 아래인 점으로 판정됐다. `J`도 실제값 `0.06155571`이 threshold `0.06169924`보다 낮았다. 따라서 1차 기울기는 Armijo threshold를 정하는 local slope이며, 유한 변화량이나 `Phi` 전체 모델로 확대할 수 없다.

post-audit summary 필드도 raw와 독립 계산으로 대조했다. `Phi(g+alpha Hd)=0.2236298922` 및 gradient 경로 상대 잔차 `9.27435e−4`는 saved HVP·gradient로 재현된다. 반면 scalar 1차 `Phi0+alpha gᵀHd=−0.28871735`는 actual merit값이 아니다. 새 gradient와 old direction의 내적 `+0.0091206892`, old/new gradient angle `91.091746°`, `dᵀHd=1087.0354619`, Rayleigh 값 `1944.1663463`, analysis/future branch changed flag 둘 다 true 또한 각각 raw/partition arithmetic와 일치한다.

## 연구 범위 및 남은 일

이 한 걸음은 작은 radius 안에서 비용과 raw stationarity merit를 동시에 낮춘 실제 접선 보정이다. 동시에 최대 gradient와 flow/growth gradient가 증가했고, branch digest는 endpoint에서 바뀌었다. 이후 방향은 새점에서 gradient·HVP를 다시 구성하고 branch 정보를 확인해야 한다. 이번 결과는 고정 tangent surface 밖으로 이동하거나 양쪽 면 원인을 분리하는 시험이 아니다.

원래 unconstrained 3h 문제의 적격 정상점, full curvature, 수반/VJP/재분석, 독립 합성 미래점수는 여전히 미완료다. 기존 P1 `8/8=100%`와 현재 FV 확장 연구 외부 가중 추정 `약 70%`는 이 접선 step으로 갱신하지 않는다.

## Evidence hashes

- Plan: `70f9560c0355fa6b65927ac70fb4de2c055f338d5dd276c8fa453fc30379ecf9`
- Raw `step.json`: `061c51d2b016f024e24ef7e1afb30be1c1169836305c92e97d806301d3040c53`
- Parent `step.run.json`: `e84cd4b87a22ebceeb5d625d15eb6043d5434b5a855e66703e0513ddda9487d7`
- Resource `step.resource.json`: `ab0d550cf94d75df79bf3017cc1b3583736afe5f665e0c7f4d879557b5e1c316`
- Summary `QY32_TANGENT_RESULT_20261007.json` (post-audit analysis fields; underlying step raw unchanged): `695e1a17a4f3d4f7926b9e7d73167ed671f10ecaf059f5e057351068fd6d20c3`
