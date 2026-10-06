# e0b 반복 dual-merit 실행 — RED 최종 검토

## 판정

저장된 실제 실행은 두 개의 dual-merit 반복을 커밋했고, 세 번째 solve 도중 내부 예산 거부로 종료했다. 첫 두 채택에서 원래 `J`와 `Phi`가 모두 줄었고 candidate strict endpoint gate도 통과했다. 저장된 child는 실행·입력·source closure가 완료됐으며, **정상점에는 도달하지 않았다.** 최종 `||g||∞=0.54803068`은 기준 `1e-10`의 약 `5.48×10^9`배다.

검토 대상은 고정 계획 `E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json` SHA-256 `ee75ca45b03495b055023a984457c4aab73eb76c646c1c17bf8760d2e53f80d9`과 `e0b_repeat_dual_20261006_attempt1`의 저장 raw, parent, resource, producer snapshot이다. FV/HVP/PCG 재실행은 하지 않았다.

## 반복과 수치 결과

| 지점 | 이동량 | `J` | `Phi` | `||g||∞` | PCG | HVP |
|---|---:|---:|---:|---:|---:|---:|
| 시작 `e0b04a…` | — | 0.06835683126 | 0.87567774445 | 0.78815557616 | — | — |
| 반복 1 `8c1285…` | 0.0250 | 0.06457612277 | 0.50556623587 | 0.62478453429 | 26 | 28 |
| 반복 2 `90fc45…` | 0.0125 | 0.06301158390 | 0.40098104405 | 0.54803068272 | 27 | 29 |

첫 반복은 두 후보를 평가했고 첫 후보는 `J` 및 분기 검사를 통과했으나 `Phi` Armijo를 통과하지 못했다. 둘째 반복은 첫 두 후보가 같은 이유로 `Phi` 조건에서 거부되고 세 번째를 채택했다. 채택점 두 곳 모두 strict branch 기록에서 Euler, choice, face-sign stage가 각각 3,600이며, 채택 때마다 전체 서명이 시작점과 달라졌다. 이는 endpoint 통과 기록이지 두 점 사이 경로의 매끄러움 인증은 아니다.

동일한 시작점과 비교하면 종료점까지 `J`는 7.81962%, `Phi`는 54.20906%, 최대 gradient 성분은 30.46669% 감소했다. 두 개별 solve의 참 상대잔차는 각각 `1.28018e-11`, `6.80879e-11`로 PCG 기준 `1e-10` 안이다. `Phi` 감소는 정상성 개선을 보이지만 최종 정상성 판정은 위 최대 성분 기준에서 명확히 미달한다.

## 예산과 종료 영수증

두 solve는 각각 26/28과 27/29의 PCG 반복/HVP를 기록했다. 세 번째 solve는 HVP 누계 57에서 시작해 5개 HVP가 더 완료된 뒤 내부 예산 거부를 기록했다. 저장값은 `pcg_solves_started=3`, `pcg_iterations_completed=53`, `hvp_calls=62`, `hvp_calls_completed=62`이며, 세 번째 solve의 완료 반복 수는 `not_recorded`로 남겼다. 따라서 이 실행이 세 번째 방향이나 후보를 완료했다고 해석하지 않는다.

child 경과시간은 723.516초로 내부 협력 예산 720초보다 약 3.516초 길다. 진행 중인 HVP callback이 끝난 뒤 deadline 검사가 예산 만료를 발견했기 때문이다. 외부 guard는 724.979초에 종료했으며 780초 한도 안이고, child exit code는 0, SIGTERM/monitor/cleanup 오류 및 자원 종료는 없었다. 표본 RSS 최대는 372,686,848 byte로 1 GiB 한도 미만이다. 내부 720초는 callback 도중 강제 선점하는 하드 제한이 아니라 작업 사이에서 확인하는 협력 제한이다.

child는 `phase=finished`, `execution_status=completed`, `numerical_status=budget_refusal`, 두 커밋, source unchanged 및 fixed-input closure 통과를 기록했다. 그러나 당시 `step.run.json`은 과거 300초 runner 분류기를 재사용해 `execution_status=failed`였다. 동일한 저장 resource에 새 780초 전용 `execution_status`를 적용한 오프라인 분류는 `completed`다. `REPEAT_DUAL_CLASSIFICATION_FIX_20261006.json`은 이 차이를 기록하며 raw SHA `6be9cf6e…`, 원 parent SHA `e70abf09…`, resource SHA `de454a96…`를 보존한다. **원 parent/raw는 수정되지 않았으므로 저장된 parent 값을 `completed`로 인용하면 안 된다.**

분류기 변경 후 보고된 30개 영향 시험 통과, 18개 기존 warning, typecheck 오류 0건은 통합자가 전달한 검증값이며 이번 RED 감사에서 다시 실행하지 않았다. 현재 producer SHA는 `c4bfa13b…`이고 실제 실행 producer snapshot SHA는 계획과 일치하는 `b8c02f5c…`다. post-run 분류기 변경이 실제 수치 실행에서 사용된 것으로 취급하지 않는다. 이 결과를 재실행하려면 새 소스 pin을 가진 새 계획이 필요하지만, 본 검토는 재실행을 요구하지 않는다.

## 범위와 남은 상태

같은 목적함수·parameter와 전처리기 전용 f82c block inverse를 유지하고, 매 반복 새 현재점 HVP/PCG를 생성한 뒤 그 방향/Hs를 같은 반복의 두 Armijo search에서 재사용했다. 두 방향 solve 및 두 후보 closure는 기록과 일치한다. 실행은 허용된 누적 HVP 90보다 앞서 내부 시간 한도에서 끝났다. 이 terminal 상태는 부분 성공이며, 앞선 두 커밋은 보존된다.

이 실행은 새점의 최종 Hessian/전체 SPD, 수반, 비선형 재분석, 독립 기상 예측 점수 또는 물리적 예측 적격성을 계산하지 않았다. 따라서 결론은 **정해진 예산 안에서 두 번의 비용·정상성 동시 감소를 보인 제한 보정**까지다. root/minimum/forecast 성공으로 올리지 않는다.
