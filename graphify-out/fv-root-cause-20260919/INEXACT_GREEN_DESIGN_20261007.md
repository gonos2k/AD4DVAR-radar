# d31 이후 inexact Newton 비교 — GREEN 설계 검토

날짜: 2026-10-07
범위: 기록·현재 구현·수치 조건을 읽은 설계 검토. 실제 FV/HVP/PCG, 시험, 예보, 수반은 실행하지 않았고 코드도 변경하지 않았다.

## 판정

제안한 forcing 정책은 **실제 잔차를 새 허용치로 검사하고 기존 하강·곡률·분기·두 Armijo 게이트를 모두 유지한다면**, d31에서 비용과 비선형 진행을 비교할 수 있는 별도 실험안이다. 현재 evidence만으로 반복 계산을 더 빨리 하거나 수렴률을 높인다고 결론 내릴 수는 없다. 기존 엄격 정책의 실행 기록과 새 정책의 결과를 섞거나 기존 영수증을 소급 변경하지 않는다.

## 근거와 수학적 조건

고정 시작점은 `d31ecd393002a1e098c939838ce22e9a7526c1b7f0f426effbb3f3a408ccdeeb`이다. 저장된 시작점에서 `||g||∞ = 0.4267256181`, `||g||₂ = 0.8065689289`이다. 바로 전 90fc 채택 방향은 true relative residual `1.3093e-11`을 얻는 데 PCG 29회, 최종 true-residual 검사를 포함해 HVP 31회가 들었다. 이어 같은 실행 안에서 d31의 다음 strict 풀이가 시작됐고, 23 HVP 뒤 720초 제한에 중단됐다.

직전 strict 실행에서는 d31에서 다음 풀이가 시작되어 23 HVP 뒤 전체 720초 제한에 도달했다. 앞선 90fc solve도 같은 실행 예산을 소비했으므로 이것은 **독립 strict 재실행 비용의 측정치가 아니며**, 그 23 HVP를 PCG 반복 수로 읽을 수도 없다. 다만 strict d31 baseline이 한 번의 제한 실행에서 완료되지 않을 위험을 보여준다.

후보 forcing 식

```text
eta(g) = min(1e-3, max(1e-10, 0.1 * ||g||∞))
```

을 d31에 대입하면 `eta = 1e-3`이다. 이 값은 `||g||∞`의 단위와 제어 좌표 스케일에 의존한다. 따라서 이를 범용·불변 forcing 법칙 또는 검증된 Eisenstat–Walker 정책이라 부르지 말고, **이 고정 문제에서 시험할 사전 선언 휴리스틱**으로 기록한다.

현재점에서 `H s = -g + r`이고 `||r||₂ <= eta ||g||₂`, `eta < 1`이면

```text
D Phi[c; s] = gᵀ Hs = -||g||₂² + gᵀr
             <= -(1 - eta) ||g||₂² < 0.
```

이는 현재점의 미소 방향에 대한 보장이다. 유한 trial의 `J`/`Phi` 감소, branch 연속성, 전역 수렴은 보장하지 않는다. 따라서 `gᵀs < 0`, `gᵀHs < 0`, 양의 판정 가능한 `sᵀHs`, complete endpoint branch 검사, 실제 두 Armijo 조건은 그대로 필수다. 실제 잔차는 PCG의 재귀 잔차만 믿지 않고 현재 코드처럼 새 `Hs`로 계산해야 한다.

## 최소 비교 설계

두 정책을 **같은 고정 d31 체크포인트에서 시작하는 별도의 단일-step bounded 실행**으로 비교한다. 각 실행은 같은 objective, 관측/prior, parameter `p`, 소스·runtime, preconditioner, branch 규칙, 후보 grid, radius, 두 Armijo 상수와 시간·메모리·HVP 상한을 쓴다. 한 실행 안에서 strict가 먼저 720초를 소비한 뒤 inexact를 돌리는 순서는 피한다. 그 구조는 두 번째 정책에 동일한 계산 기회를 주지 못한다. strict가 제한 안에서 방향을 끝내지 못하면 기준군은 `budget_refusal`로 기록하고, 비교 가능한 accepted-step 비용/진행은 산출하지 않는다.

1. Strict 기준군은 기존 `rtol=1e-10`을 그대로 쓴다. 별도 inexact 군은 매 solve 시작점의 전체 gradient로 위 `eta(g)`를 한 번 계산하고, PCG 상대 허용치와 별도 true-residual gate 모두 `eta`를 사용한다. `max_iterations=40`, 총 HVP 상한, 내부 720초/외부 780초, RSS 1 GiB를 사전에 고정한다. 각 결과와 계획에 정책별 해시·시작 제어 해시를 기록한다.
2. 정책별로 최대 한 방향과 기존 line search만 수행한다. 새 현재점 HVP를 사용하고 cached `f82c` 자료는 preconditioner에만 쓴다. 완료되지 않은 방향, 참조 trial, 다른 정책의 방향을 재사용하지 않는다.
3. `J`, `Phi=||g||₂²/2`, 전체 gradient, 기존 분기 서명·마진, 입력 무결성, 동일 후보 계열의 두 실제 Armijo 조건, 최종 시간 한도를 현재 구현과 같이 다시 평가한다. 한 정책이 선형계 잔차·곡률·예산·분기에서 거부되면 거부로 보존한다. 실패 후 허용치를 키우거나 후보 범위를 넓히지 않는다.
4. 비교표에는 각 정책의 완료/거부 상태, 실제 residual과 threshold, PCG 반복 수, HVP 수(모든 재계산 포함), 벽시계/RSS, `gᵀs`, `gᵀHs`, `sᵀHs`, 채택 alpha/이동량, 실제 `ΔJ`, `ΔPhi`, 최종 `||g||∞`, 선형화 예측 오차를 함께 둔다. 정책 간 비용은 HVP 수와 시간 모두로 비교하며 HVP 수를 정확한 시간 절감률로 환산하지 않는다.

최소 비교의 목적은 **동일한 시작점에서 한 번의 보정에 대한 비용–진행 tradeoff**를 측정하는 것이다. 한 쌍의 실행만으로 전체 수렴 우위나 일반 성능을 주장하지 않는다. 계산 예산을 아끼려면 먼저 둘 다 한-step bounded로 계획하고, 결과가 나온 뒤 반복 적용의 별도 계획을 정한다.

## 유지할 완료 조건과 제한

각 accepted endpoint는 이전과 같은 actual `J` 및 `Phi` Armijo 감소를 만족해야 한다. endpoint stationarity/eligibility 기준은 계속 `||g||∞ <= 1e-10`이며, 이 연구 비교의 느슨한 선형 잔차를 이유로 최종 기준·곡률 인증·수반 잔차·재분석 기준을 바꾸지 않는다. 실제 d31의 최대 gradient는 root threshold보다 약 `4.27e9`배 크다.

이 실험은 최소 한 스텝의 운용 비교일 뿐 최종 정상점 판정이 아니다. d31에서 새 HVP/PCG를 만들지 않았고, branch 변화, 앞서 관측한 49.65% 유한-step gradient 선형화 오차, 동역학 블록 잔차 반등은 비선형 진행이 어려울 수 있다는 이유다. 선형 허용치를 느슨하게 했을 때 실제 `Phi`가 얼마나 감소하고 후보가 채택되는지는 관측 전까지 미확인이다.

## 확인한 근거 위치

- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: strict `PCG_RTOL=1e-10`, PCG 이후 현재점 `Hs`로 계산한 true residual, `sᵀHs` roundoff-aware 곡률 게이트, `gᵀs`·`gᵀHs` 하강 게이트, 양쪽 actual Armijo 및 root 후보 재감사.
- `src/advar/matrix_free.py`: PCG는 허용치 후보 시 true residual를 `rhs - operator(x)`로 재계산하고, drift 시 recurrence를 restart한다.
- `graphify-out/fv-root-cause-20260919/90FC_FOLLOWUP_RESULT_20261007.json`: d31 수치, 직전 완료 방향의 residual/HVP 및 두 번째 풀이의 budget stop.
- `graphify-out/fv-root-cause-20260919/90FC_FOLLOWUP_GREEN_FINAL_20261007.md`, `90FC_FOLLOWUP_RED_FINAL_20261007.md`: 기록 범위·불확실성의 독립 점검.

검토 산출물은 설계 제안이다. 구현 승인, 실행 완료, 성능 결과를 의미하지 않는다.
