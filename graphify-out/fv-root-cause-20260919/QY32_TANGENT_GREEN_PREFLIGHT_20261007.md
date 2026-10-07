# PR #257 Qy32 접선 보정 — GREEN 최종 사전 실행 검토

날짜: 2026-10-07
판정: **GO — 고정 계획 아래 단일 bounded 실행 가능.** HVP 부호와 후보 수용은 실제 실행이 확인해야 한다.
범위: plan `QY32_TANGENT_STEP_PLAN_20261007.json`, 현재 tangent-step 코드·집중 시험, 앞서 보관된 QY32 원시 결과를 읽기 전용으로 검토했다. 이 검토자는 FV/objective/gradient/HVP/PCG/예측을 실행하지 않았다.

## 계획·pin 및 선행 자료

고정 계획 SHA-256은 `70f9560c0355fa6b65927ac70fb4de2c055f338d5dd276c8fa453fc30379ecf9`다. 계획은 앞선 QY32 양측 진단 계획 `bea4c361…`, 기준 제어 `e29c348d…`, 파라미터 `8871db49…` 및 압축 raw evidence에 고정되어 있다. 계획의 109 source pin과 39 archive pin을 독립 SHA-256 대조했으며 불일치는 없었다.

보관 QY32 raw와 요약으로부터 계산한 차원·접선 metric 방향은 앞선 검토 기록과 일치한다. `QY32_TANGENT_DIRECTION_20261007.json`에는 `cond(Z0ᵀZ0)=1.29798`, `||d||₂=0.7477478`, cached `g_e29ᵀd=−0.5587502`와 네 표본의 `q_tᵀdt<0`이 기록되어 있다. 이들은 방향을 정의하는 과거 증거다. 새 current-point HVP나 후보 목적함수 평가를 대신하지 않는다.

검토 시 저장소 집중 결과는 tangent module 시험 27개 통과, 기존 경고 18개, 1.55초; basedpyright error-level 오류 0으로 기록되어 있었다. 이 검토자는 시험·타입 검사·AST 추출을 독립 재실행하지 않았다. 실제 FV/HVP 실행도 수행하지 않았다.

## 방향과 fresh HVP

구현은 양측의 저장된 `q_t=Z_etaᵀg` 중 `eta=±1e−6` 평균과 `G0=Z0ᵀZ0`를 써서 `dt=−G0⁻¹qbar`를 구성한다. `d=Z_eta_b dt`는 actual e29 `eta_b=Qy[3,2]`에서 lift하며, objective/history의 cached Hessian은 사용하지 않는다. 새 문제를 바꾸거나 `eta=0`으로 고정하지 않는다.

현재점 방향 확인은 다음 HVP를 한 번 계산한다.

```text
H_e29 d = J_cc(e29,p) d
J 방향 기울기:       g_e29ᵀ d
Phi 방향 기울기:     g_e29ᵀ H_e29 d
```

코드는 `torch.func.jvp(grad(problem.objective))`에 제어 방향 `d`와 영 파라미터 방향을 전달하므로 원래 full-control objective의 현재점 `H d`를 계산한다. `resolved_direction_checks`는 둘의 내적이 반올림 예산을 넘어 엄격히 음수일 때만 line search를 연다. 따라서 두 번째 부호는 미리 추정하거나 QY32 chart 표본에서 가져오지 않는다. `gᵀHd`가 비음수·미해상 상태면 명시적 direction refusal로 닫아야 한다.

기준점에서는 저장 e29의 `J`, `Phi`, max-gradient, gradient 26성분을 새 baseline `qy._measure` 결과와 roundoff-scaled 기준으로 비교한다. branch hash도 비교한다. 생산 branch 서명의 compact serialization은 양측 진단·seed continuation 양쪽 모두 정렬 JSON `{choices, face_signs}`와 같은 separators/NaN 금지를 쓰므로, e29 branch identity 대조는 호환된다. strict branch 및 complete margins를 통과하지 못하면 방향 계산 결과로 진행하지 않는다.

## 곡선 후보와 dual Armijo

후보는 ambient 직선 `e29 + alpha*d`가 아니라

```text
c(alpha) = Gamma(t_e29 + alpha*dt, eta_e29)
```

이다. chart는 원래 26개 좌표에서 pivot `c[24]`만 복원하고 target face flux를 e29의 실제 비영 `eta_e29`에 유지한다. 이 실험은 면을 가로지르는 탐색이 아니라 그 비영 유량 등면에서의 한 번의 접선 보정이다. 초기 radius는 실제 곡선 control displacement로 제한하고, chart가 정의역을 벗어난 후보는 FV/objective 평가 전에 reject/backtrack한다. 나머지 후보는 최대 16개 dyadic 규모를 쓴다.

각 실제 후보에서 원래 full-26 `J`, 새 full gradient, `Phi=0.5||g_candidate||₂²`, 실제 endpoint branch와 margin을 재계산한다. accept에는 기존 상수 `c1_J=c1_Phi=1e−4`로 다음 둘이 모두 필요하다.

```text
J(candidate)   <= J(e29)   + c1_J   alpha (g_e29ᵀ d)
Phi(candidate) <= Phi(e29) + c1_Phi alpha (g_e29ᵀ H_e29 d)
```

여기서 `gᵀHd`는 base point의 smooth branch에서 계산한 1차 `Phi` 기울기이고, 후보 `Phi`는 새 전체 gradient로 실제 평가한다. 문서/결과는 이를 곡선 경로의 1차 모델로만 기록해야 한다. `alpha H d`는 곡선 displacement와 같지 않고 exact quadratic model도 아니다. 새 HVP 수는 1, PCG는 0이다.

Armijo와 strict branch를 통과한 점도 tentative 상태다. 별도 fresh endpoint recheck와 source/input/runtime/deadline 폐쇄를 통과한 뒤에만 최종 commit으로 기록한다. 실패하거나 예산이 끝나면 수치상 accept 후보와 committed endpoint를 분리해 기록한다.

## 남은 해석 한계

실행 전에는 `g_e29ᵀH_e29d`의 부호를 알 수 없으므로 현재 점에서
`Phi`가 실제로 내려갈 방향인지 아직 인증되지 않았다. 부호가 음수여도 finite endpoint에서 dual Armijo가 통과한다는 보장은 없다. 예산/branch/domain/Armijo 거부는 모두 유효한 실험 결과이며 강제로 후보를 채택하면 안 된다.

만약 한 점이 accept·commit되더라도 이는 동일한 비영 `Qy[3,2]` chart 안의 한 tangent correction일 뿐이다. 영유량 경계 횡단, 해당 면의 법선 최소 조건, 전체 3h 문제의 정상점, 곡률·수반·재분석 또는 미래 예측 검증을 뜻하지 않는다. 단일 HVP를 통과해도 research-completion 마일스톤을 완료한 것으로 환산하지 않는다.

따라서 사전 실행은 **조건부 GO**다. 계획된 단일 bounded launch에서 현재 HVP 부호·actual J/Phi 조건·endpoint closure를 모두 기록하면 된다. 이 검토는 실행 승인 범위의 수치 설계 판단이며 실행 결과 판정은 아니다.
