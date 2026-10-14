# Limiter 사건 위치 → 실제 후보 진행 → 끝점 자격검사

PR #305는 `d13d9a0c`로 병합됐다(2026-10-14 13:40:12 UTC). 이번 작업은 그 main에서 시작했으며, 기존 Powell·같은 ray 진단·기존 18-lead 예측을 반복하지 않았다. **두 limiter 사건을 실제 제어점으로 찾고 세 번의 native-J 감소를 채택한 뒤 같은 끝점의 미분·곡률을 확인했다. 끝점 최소점 자격은 얻지 못했다.**

## 사건의 위치와 활성조건

기존 24D 두 유동면 차트와 저장된 결합 방향에서 x 셀 `(1,1)`의 positive minmod gap `h=right-left`를 실제 native forward로 이분했다. 사건 tolerance는 `128·eps(FP64)·max|q|`다. 고정된 실제 stage의 q 차분을 미분했으며, minmod 연산·원래 J·관측·prior는 바꾸지 않았다.

| 사건 | 찾은 배율 α | 해당 gap | 다른 사건 gap | 실제 활성 limiter |
|---|---:|---:|---:|---|
| stage 48 | 0.000444346049334854 | +8.882e−13 | stage47 +2.44385e−6 | stage48만 |
| stage 47 | 0.000617820012848824 | +4.867e−13 | stage48 −2.44687e−6 | stage47만 |

두 조건은 서로 다른 점에서 활성화된다. 각 사건점에서 두 유동면 법선과 두 gap 법선의 normalized 행렬은 수치 rank 4였고, 최소 singular value는 약 1.647e−6이다. 거의 평행하여 conditioning이 나쁘지만 동일 조건이라는 결과는 아니다. 각 gap 법선은 두 유동면 법선과 독립이었다. 이는 (+,+) 유동 branch extension에서 계산한 국소 rank이며 모든 sector의 법선 자격은 아니다. **법선 둘의 독립성과 둘의 동시 활성은 다르므로, 실제 현재점에서 활성인 조건 하나만 검색 등식으로 썼다.**

두 식별된 gap의 sign bracket을 좁힌 결과이며, 구간 전체에서 모든 다른 가능한 사건의 순서·유일성·전역 최초 사건을 증명하지 않았다. 실행에 사용한 마지막 bracket과 실제 제어/법선은 `attempt1/RESULT.json:event_roots`에 보존했다.

## 원래 비용의 실제 진행

| 단계 | 제어 SHA 접두사 | native J | 승인 |
|---|---|---:|---|
| 입력 재자격 | `6ca56498…` | 0.0601191808341518 | 기존값 일치 |
| 먼저 찾은 stage48 사건점 | `39ad956e…` | 0.06011918080517632 | 실제 비용 감소 |
| 사건 접선 Newton 보정 1 | `312603da…` | 0.060119180790486296 | native Armijo, α=1/512 |
| 사건 접선 Newton 보정 2 | `30bed163…` | 0.0601191807868374 | native Armijo, α=1/2048 |

마지막 채택 제어는 `30bed163abb770fb4cb708cabd168852d895892da05d0655bc9ec604430044f2`다. 추가 J 감소는 **4.73144e−11**이며 매우 작다. 이를 유의미한 수렴·예측 개선으로 과장하지 않는다. 최초 사건점은 단순 비용 감소로 채택했고, 두 후속점은 기록된 실제 native J와 Armijo threshold를 모두 통과했다.

새점마다 gradient·사건 법선을 갱신하고, λ를 그 점에서 고정해 `Qᵀ(H_J+λH_h)Q`를 구성했다. 검색에는 양의 정부호 현재점 reduced Lagrangian model을 썼다. retraction의 각 중간점에서도 실제 selector와 비선택 유동면 sign을 확인했다. 과거 PD Hessian을 현재 곡률로 승계하지 않았다.

끝점의 검색 곡률 최소 고윳값은 **0.859089**, projected gradient 최대 성분은 **0.0011416883**다. **정상성이 미달하므로 positive search curvature를 최소점 인증으로 사용할 수 없다.** 여기의 residual은 stage48 등식을 추가한 23차원 접선 좌표의 값이다. 이전 단일 유동면 G나 다른 차트 proxy와 감소율을 비교하지 않았다.

## 정확히 무엇이 진행을 막았는가

큰 trial부터 y 셀 `(2,3)`의 stage123·124가 right→inactive로 바뀌었다. 세 번째 Newton 반복의 12개 trial은 모두 이 전환 검사에서 거부됐다. 이는 이번에 지원한 x-stage47/48 family 밖의 전환이다.

raw status `no_native_Armijo_progress_at_current_event`는 **지원·승인된 추가 진행이 없었다**는 실행 상태다. 원래 J가 더 내려갈 수 없다는 판정이 아니다. 실제로 거부한 trial 중에는 native J가 끝점보다 더 낮은 것도 있다(관찰된 최저 raw J=0.06011918073978765). 그 값을 확정점이나 적격 최소점으로 승격하지 않았다. 따라서 새 후보는 전체 평가들 가운데 최소 비용점이라는 주장도 하지 않는다.

최종 실제 활성 limiter는 stage48 x(1,1) 하나이고, stage47은 아직 gap 약 +2.44e−6으로 비활성이다. y-stage123/124는 거부된 trial에서 만난 전환이며 최종점의 영구 등식으로 추가하지 않았다.

같은 끝점에서 original `replay=True` native J를 재확인했다. 별도 종료 검사에서는 네 유동 sector의 original replay gradient와 retained-graph gradient를 직접 비교했다. 최대 차이는 **4.34375e−15**, 네 비교 모두 선언한 component/연산수 기준을 통과했다. 이는 동일점 미분 closure이며 완결된 limiter limiting-gradient 인증이 아니다.

끝점에서 두 유동면만의 네 gradient hull은 여전히 finite proxy다. **추가 활성 minmod의 양측 limiting sector 정상성·원래 활성구조의 exact P2가 없고, 접선 gradient도 기준 1e−10을 넘는다.** stage48이 현재 수치적으로 활성이라는 사실은 원래 최소점에서도 그 등식을 유지해야 한다는 증명이 아니다. 그 양측 법선 부호·혼합계수의 자격도 미확인이다. `first_order_complete=false`, `P2_complete=false`, `minimum_claim=false`가 유지됐다. 현재의 미해결 항목은 구체적으로 이 활성 limiter 자격과, 진행 중 실제 만나는 y123/124 전환의 처리다.

## 예산·검증·기록

새 phase 계획은 기존 이후 누계 J512/gradient134를 기초로 추가 J256/gradient128/Hessian-row432/retained forward48/900초/RSS2GiB를 선언했다. 이전 장부를 0으로 돌리지 않았고 이전 cap 준수로 표현하지 않았다. 끝점 시간·호출은 탐색이 사용할 수 없도록 예약했다.

- 주 실행: **98.85294초**, direct J82, gradient25, Hessian row192, retained forward15. gradient에는 사건 gap 미분16회와 목적함수 미분9회가 포함된다. Hessian row는 J96행과 gap96행이다.
- 종료 original-replay 비교: gradient4회, **3.99945초**, J/Hessian/score 호출0. 이전 PLAN의 추가 gradient allocation 안에서 소비했다. 원래 attempt 결과를 덮어쓰지 않고 별도 PLAN/receipt를 남겼다.
- 누계: 이후 direct J **594**, gradient **163**. 원래 reference J2000까지 포함한 추적 J는 **2594**다. 전체 과거 AD/중단 실행의 census가 아니다.
- 측정 계산시간 합102.85239초, 이전 추정 장부와 합3540.62313초. 실제 전체 세션 wall/완전 비용 계측 인증이 아니다. RSS는 호출 전2GiB 검사를 했지만 process peak를 serialize하지 않아 측정 peak를 주장하지 않는다.
- source/archive **117개**, 입력5개, plan sidecar, actual accepted control/J, root normal SVD 및 두 Armijo threshold를 독립 재검산했다. 소스·입력은 주 실행 중 바뀌지 않았다.
- 영향 시험 **48개 통과**, 기존 TorchScript warning18개. 새 API·진행 함수·추가 검산 스크립트 타입 error0. GREEN/RED 검토는 별도 파일에 유지한다.

종료 검산 스크립트에는 실행 후 dict 타입 annotation만 추가했다. 실제 실행한 bytes는 `terminal_replay_executed_source.tar.gz`와 `TERMINAL_REPLAY_SOURCE_TRANSITION.json`에 보존했다. 이 수정 때문에 비용·미분을 다시 계산하지 않았다. 주 실행 소스는 수정하지 않았다.

연결된 [장부](ACCOUNTING.json), [검산](VERIFICATION.json), [GREEN](GREEN_REVIEW.md), [RED](parentRED_REVIEW.md), [기존 응답 마지막 연결](RESPONSE_READINESS.md), [원시 실행](attempt1/RESULT.json)을 참조한다. Graphify는 변경 코드만 cache 기반 AST로 증분 반영했다.

## 예측과 응답의 상태

현재 합성 사례의 독립 시간적분·18-lead 검증은 **6ca 후보에서 완료**된 결과로 유지한다. 단기 이득, 후반·평균 실패를 되돌리지 않았다. 평균 MAE1.163358 대 persistence1.144573, 평균 MSE2.434426 대1.821550(+33.65%), +180분 MSE 비율약2.5724다. 같은 공간 이산화의 합성 수치 기준이며 보류 기상사례·연속 PDE 검증은 아니다. 새 30bed 후보의 미래점수는 평가하지 않았고 예측 개선을 주장하지 않는다.

적격 기준점이 없으므로 실제 p±h 재최적화·정확 수반은 실행하지 않았다. 기존 두 유동면 응답의 자격검사와 관측0/배경β12의 ±h, ±h/2 경로 및 새 transport hash 재자격 필요를 마지막 연결 노트에 정리했다. 추가 limiter 활성점에 두 면 전용 응답을 그대로 적용하지 않는다.
