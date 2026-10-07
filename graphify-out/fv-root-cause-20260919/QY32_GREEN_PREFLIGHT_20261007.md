# Qy[3,2] 양측 진단 — GREEN 최종 사전 실행 검토

날짜: 2026-10-07  
판정: **GO — 고정된 단일 진단 계획으로 실행 가능**  
검토 범위: 최종 진단 모듈·집중 시험·명시 실행 계획·PR #256 보관 산술의 읽기 전용 검토. 이 검토자는 새 FV/목적함수/gradient/HVP/PCG/예측/수반/재분석을 실행하지 않았다.

## 판정 근거

필수 실행 전 조건에 차단 결함은 보이지 않았다. plan `QY32_TWO_SIDED_DIAGNOSTIC_PLAN_20261007.json`의 SHA-256은 `bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f`다. 계획은 F462 producer plan `499e8986…`와 e29 종료점 `e29c348d…`에 고정되어 있다. 선언 범위는 고정한 나머지 25개 제어 좌표에서 `eta=Qy[3,2]`의 네 비영 표본과 `eta=0`의 primal 비용 표본이다. 원래 목적함수와 prior를 유지하며 optimizer/HVP/PCG/score/수반/재분석은 실행하지 않고, 단일 guarded launch에 내부 240초·외부 300초·RSS 1 GiB 한도를 둔다.

계획의 107개 source pin과 32개 archive pin을 각각 SHA-256으로 독립 대조했다. 불일치는 없었다. 여기에는 진단 module/test, analyzer, F462 producer plan과 raw/run/resource, 상속 source/curvature 자료가 포함된다. 별도로 producer raw/run/resource/plan의 실제 SHA도 module 상수와 계획 pin에 맞았다. 재산정은 바이트 pin 확인이며 선행 FV 계산을 재실행한 것이 아니다.

## 원래 목적함수와 입력 계보

`fv_point_3h_qy32_diagnostic.py:69–159`의 loader는 plan·producer plan·raw/run/resource 및 상속 source/archive pin을 검사한다. e29 raw에서 완료 상태, 두 accepted iteration, 최종 제어 SHA, 파라미터 SHA, branch 상태, parent child hash와 완료 횟수, resource 종료·시간·RSS 제한을 닫는다. `_run_child`는 `_prepare_fixed_seed()`의 고정 문제를 다시 구성하고 전체 26제어·13 파라미터의 SHA와 `input_after` identity를 producer receipt와 비교한다 (`:397–423`).

각 점의 비용은 full 26-vector로 `problem.objective(control, parameters)`를 호출한다 (`:284–300`). 구현된 원래 목적함수는 data cost, 전체 control prior residual의 제곱합, field smoothness prior를 포함한다 (`src/advar/fv_point_research_problem.py:320–351`). 따라서 pivot `c[24]`를 바꾸면서도 해당 prior 항을 유지한다. 새 objective나 축소 목적함수로 대체하지 않았다. focused test는 pivot을 바꾼 full-vector 목적함수 호출이 `_measure`로 전달되는지 별도 synthetic objective에서 확인한다 (`tests/test_fv_point_3h_qy32_diagnostic.py:92–120`). 이 시험은 생산 FV 결과를 검증하지 않는다.

## 면 좌표·Jacobian 및 접선/법선 보고

생산 유량 경로는 `bounded_fv_coefficients`와 `face_volume_fluxes`를 사용하고, 현재 FV profile에서 면 가중치를 기저·계수 한계로 다시 계산해 선언 상수와 비교한다 (`fv_point_3h_qy32_diagnostic.py:162–206`). `Qy[3,2]`는

```text
Qstar(c) = -0.08 tanh(c[21]) - 0.21 tanh(c[22])
           -0.10 tanh(c[23]) -0.45 tanh(c[24])
```

이다. chart는 지정 `eta`에서 `c[24]`만 역산하고, 매 점에서 생산 연산자로 얻은 면유량이 `eta`와 맞는지 확인한다 (`:208–237`, `:284–291`). Jacobian의 25개 retained 열은 `Qstar`의 접선공간을 만들며 `Gamma_eta`는 pivot 좌표만 바꾸는 transverse chart 벡터다. focused tests는 chart 값, production flux 일치, autograd Jacobian 일치, pivot 정의역을 점검한다 (`tests/test_fv_point_3h_qy32_diagnostic.py:28–69`).

최종 `geometry()`는 QR tangent projection과 `g - n(nᵀg)/(nᵀn)` projection의 norm을 별도 출력하고, retained-coordinate covector를 `j_t_fixed_eta`로 명시한다. 다음 세 양은 코드에서 별도 필드로 기록한다 (`fv_point_3h_qy32_diagnostic.py:239–266`).

```text
j_eta_fixed_t               = gᵀ D_eta Gamma
euclidean_unit_normal_gradient = nᵀg / ||n||₂
intrinsic_normal_flux_slope = nᵀg / (nᵀn)
```

이는 서로 다른 양이다. e29에서 보관 gradient를 사용한 독립 기하 산술은 다음처럼 실제 부호도 다르다.

| 지표 | e29 재산술 |
|---|---:|
| QR 및 직교투영 접선 잔차 `||g_T||₂` | 0.7472444234531769 |
| 단위 유클리드 법선 방향 기울기 | +0.00470995066527895 |
| `Qstar` 단위당 최소노름 법선 기울기 | +0.009186929094707799 |
| 고정 `t` pivot chart `j_eta_fixed_t` | −0.0330987474886055 |

큰 접선 성분이 남으므로 chart의 transverse 선은 유클리드 순수 법선선이 아니다. 따라서 `j_eta_fixed_t`의 음수를 법선방향 비용 감소로 해석하면 안 된다. tests는 QR projection과 명시 투영의 norm 일치 및 retained chart 열을 재척도해도 tangent span projection이 유지되는지 확인한다 (`tests/test_fv_point_3h_qy32_diagnostic.py:72–90`).

## 비용·branch 진단의 범위

`eta=0`은 전체 primal 목적함수와 정적 유량을 계산하고 곧바로 반환한다. gradient, `Phi`, branch eligibility를 계산하지 않는다 (`fv_point_3h_qy32_diagnostic.py:284–302`, plan `chart.zero_scope`). 네 비영 표본은 전체 gradient/`Phi`를 계산하고 branch collector를 `problem.branch_check` 호출 주위에 둔다. 이 oracle은 고정 off-grid 관측과 알려진 상태·경계에서의 pointwise branch 검사를 사용한다 (`src/advar/fv_point_research_problem.py:366–394`). collector는 완전한 3600 stage 서명과 margin completeness를 검사한다 (`fv_point_3h_qy32_diagnostic.py:313–353`).

최종 결과는 full choice/face signature 배열과 analysis/future partition hash를 보존한다. 분석 360 stage 및 미래 3240 stage 변화는 이후 재산술로 같은 쪽·양쪽 표본 간 비교할 수 있다. 또한 정적 `Qx/Qy`의 목표면 외 sign 변화를 별도 집계한다 (`fv_point_3h_qy32_diagnostic.py:438–458`). 이 기록은 이 고정 합성 문제의 점별 분기 정보이며, 모든 기상 입력에서의 분기 연속성이나 단일면 원인을 보증하지 않는다.

## PR #256 재산술의 독립 해석

보관 수치에서 시작점 `f462a496…`부터 `e29c348…`까지 `J`는 약 **0.1299441%**, raw `Phi`는 약 **1.5837815%** 감소한다. 마지막 gradient도 여전히 `||g||₂≈0.7472593`이며, 이는 정상점이 아니다.

가까운 더 큰 후보 두 개는 `J` Armijo와 endpoint 분기 검사를 통과했지만 raw `Phi` Armijo에서 거부됐다. 보관 반올림 수치의 accepted 대비 차이는 다음과 같다.

| 단계 | `Delta J` (큰 후보 − 채택점) | `Delta Phi` (큰 후보 − 채택점) |
|---|---:|---:|
| 1 | −5.820611×10⁻⁵ | +0.0477496551 |
| 2 | −8.095170×10⁻⁶ | +0.00097494797 |

이것은 solver 방향을 따라 관찰한 실질적인 `J`/`Phi` 충돌이며, 작은 선형잔차를 원인으로 지목할 근거는 아니다. 그러나 큰 후보는 `Qy[3,2]` pivot만 바꾼 같은-retained-coordinate 점이 아니라 다수 제어를 따라 이동한 Newton 후보이고, 다른 면/limiter branch도 바뀔 수 있다. 따라서 이 두 raw merit 상승은 **고정 `t`의 경계 양쪽 비용이나 정확한 한쪽 극한을 대신하지 않으며, `Qy[3,2]`만의 불연속/인과를 증명하지 않는다.** 본 진단의 역할은 동일한 접선 좌표에서 이 가설을 더 직접적으로 검사하는 것이다.

## 실행 근거와 남은 한계

작성자 실행 기록에 따르면 집중 시험 18개 통과, basedpyright 오류 0, 격리된 incremental AST 추출을 완료했다. 이 검토자는 그 명령을 독립 재실행하지 않았다. 저장 코드에는 optimizer step/HVP/PCG counter를 0으로 고정하고, fixed input·source·runtime 종료 후 폐쇄, budget refusal과 diagnostic error 분리 기록이 있다. 실제 실행은 계획된 단일 guarded launch 한 번으로 제한해야 한다.

이 사전 검토는 실행 성공·수치 결과·분기 원인 판정·비선형 정상점·적격 수반·재분석을 보증하지 않는다. 네 비영 표본과 하나의 primal 영점은 유한 고정 좌표 표본이다. 양쪽 극한의 존재, 경계에서의 미분 가능성, 단일면 인과, Clarke 정상성, global minimum, 물리적 바람 장벽, 미래 예보 skill을 인증하지 않는다. 예산 종료나 branch refusal이 나와도 부분 receipt와 거부 이유를 결과로 보존해야 한다.
