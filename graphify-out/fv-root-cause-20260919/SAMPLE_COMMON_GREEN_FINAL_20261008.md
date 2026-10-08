# Sample-common bounded cost search — GREEN final audit

**결론: 단일 predeclared J-only 비용 탐색이 정상 완료됐다.** 원래 목적함수는 감소하고 strict endpoint 및 최종 재검사를 통과했지만, Phi는 크게 증가했다. 이는 정상성 개선이나 기상학적 성능 향상이 아니라 제한된 비용 탐색 성과다. 이번 검토는 실행 보관물의 해시·수치 재산술이며 FV/HVP/PCG를 재실행하지 않았다.

## 계보와 실행 종료

계획 `SAMPLE_COMMON_COST_SEARCH_PLAN_20261008.json`의 SHA-256은 `617a81486352591f3e17068be78fc4047db5bd4e07333b84dba48297ba9dcc3e`다. 기준 control은 PR260 endpoint `2cdccade81a8…`, sample 방향의 parent는 핀된 face diagnostic raw SHA `ffa9473e…`다. 실행 raw `step.json` SHA-256 `7df18a05…`가 `step.run.json`의 child digest와 일치한다. 실행은 exit 0, 완료 상태, HVP 1회(시작/완료 모두 1), PCG 0회, 후보 1개, 확정 step 1개다. Score, response, full-root claim은 모두 없다.

실행 source map은 계획 source/archive와 plan 자체의 정확한 169-path union이며, `source_before==source_after`; runtime도 전후 같다. Fixed-input 검사는 통과했다. Input identity 전체는 control 변경 때문에 같지 않지만, 독립 비교에서 바뀐 항목은 `control_sha256`와 그 파생량 `shifted_flow_fractions`뿐이다. parameters, original flow fractions, archived input, terminal truth는 유지됐다. Base/accepted control FP64 해시도 각각 `2cdccade…` 및 `26e830dc…`로 재산정 결과와 일치한다.

Guard receipt는 24.2306초, 표본 최대 RSS 529,743,872 bytes/1 GiB 제한, exit 0이며 resource termination·SIGTERM·monitor error가 없다. Child 내부 elapsed는 23.1353초다.

## 방향과 스케일 재산술

핀된 inner gradient pair에서 재계산한 `theta=0.288468024322864`, `||d||₂=0.136797778407174`가 저장 방향과 일치한다. 새 기준점 gradient에 대해

```text
g_base·d        = −0.0186907662532645   (resolution budget 6.19e−16)
dᵀHd            =  5.66634023845905
g_baseᵀHd        = −4.29196669306327     (Phi 방향미분 진단값)
alpha_curv       = −(g_base·d)/(dᵀHd) = 0.00329856052878805
radius / ||d||₂  = 0.365503011687638
```

따라서 곡률 cap이 실제 초기 scale을 정했다. Helper alpha는 1.0이므로 실제 `d` 계수는 `0.00329856052878805`; 실제 control 변위는 `0.000451235752280`으로 반경 0.05 안이다. `c₁=1e−4`인 원래-J Armijo 우변은 `0.0613476321578230`; 실제 후보 J는 그보다 `3.05070e−5` 낮다.

`d`의 공통 zero-face 접선 이탈각은 저장 geometry 기준 `0.04805°`로 앞서 고정한 `0.305°` 한계보다 작다. 방향은 현재 base에서 scale-resolved 하강이다. 이는 finite sampled-gradient direction의 성질이며 steepest descent, Newton 방향, Clarke subgradient 또는 Qy face 제약을 뜻하지 않는다.

## 실제 후보 결과

| 값 | Base `2cdccade…` | 채택 후보 `26e830dc…` | 변화 |
|---|---:|---:|---:|
| 원래 전체 J | 0.0613476383231 | 0.0613171251401 | **−3.0513183e−5 (−0.04974%)** |
| Phi | 0.0126732465030 | 0.0299545829052 | **+0.0172813364022 (+136.36%)** |
| `||g||₂` | 0.159205820 | 0.244763490 | 증가 |
| `||g||∞` | 0.088510893 | 0.138522954 | 증가 |

Base와 후보 모두 자체 strict branch check에서 3,600 Euler stage를 통과했다. 후보의 fresh J, full gradient, Phi 및 branch 재검사는 trial 기록과 허용오차 안에서 일치했고 final recheck가 통과했다. Acceptance는 계획대로 original-J Armijo와 strict endpoint만 사용했다. Phi 감소는 요구하지 않았으므로 큰 Phi 증가는 정책 위반이 아니다. 다만 이 결과를 정상성 향상으로 설명하면 안 된다.

HVP가 준 국소 Phi 1차 변화 `alpha_init * g_baseᵀHd`는 약 `−0.0141573`이지만 실제 finite 후보 Phi는 `+0.0172813` 증가했다. 이 차이는 현재 HVP의 base-local 성격과 finite-step/branch 변화를 분명히 보여준다. Branch signature는 base `f58e2d09…`에서 후보 `aecc7f2e…`로 바뀌었고, analysis와 future partition hash 모두 달라졌다. 두 endpoint는 각각 strict하지만 경로 전체의 branch 인증은 없다.

저장 streamfunction 기하식으로 정적 target face `Qy[4,3]`를 재산술하면 약 `−2.37504e−6`에서 `−2.03456e−6`으로 이동해 0에 가까워졌지만 **부호는 바뀌지 않았다**. 따라서 이 step은 문제로 삼은 face를 횡단하지 않았다. Endpoint signature가 바뀐 구체 stage/face/limiter를 현재 step receipt의 hash만으로 특정할 수 없으므로, 그 변화를 target face에 귀속하지 않는다.

## 판정과 범위

이 실행에서 새로 확인한 성과는 현재 base에서 하강하는 sampled common direction을 한 번 적용해, 기존 원래 J를 낮추고 strict endpoint에서 재검증했다는 점이다. 새 필수 코드 결함은 확인되지 않았다. 동시에 Phi 및 gradient norms가 크게 증가했고 branch signature가 달라졌으며 목표 face는 건너지 않았다. 따라서 결과는 **J-only 제한 비용 탐색 1회 수용**으로만 분류한다. 정상점 탐색, 수반, 재분석, score, branch 경로 보장, 독립 예측 검증은 완료하지 않았다.
