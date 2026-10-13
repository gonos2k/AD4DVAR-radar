# Native 기준계산·최소점 자격·미래검증 결과

현재 연구는 **적격 최소점과 정확 응답을 완료하지 못했다.** 원래 26개 제어의 native J 참고계산, 실제 활성면의 국소 보정, 최신 후보의 상태·미래검증을 수행했다. 예측 검증은 성능 실패를 포함한 실제 결과이며, optimizer 종료나 비용 감소를 최소점 인증으로 사용하지 않았다.

## 실제 결과와 상태

| 항목 | 결과 | 판정 |
|---|---|---|
| 원래 J 전체 공간 참고계산 | Powell, fb0a·제어 0 두 시작점, Q=0/R 단조감소/E 선택 제외 | 실행 완료, 최적점 인증 아님 |
| fb0a 시작 비용 | 0.060988878373729814 | 고정 입력에서 확인 |
| 현재 후보 비용 | 0.0601191808341518 | 유효한 비용 감소 |
| 현재 계산점 | `6ca56498f82002634a5103fecd41b1ec9d67e8ff745ecbe9b554906043c0c4d5` | 참고후보, 정식 최소점 아님 |
| 측정한 전체/접선 gradient hull proxy 최대값 | 약 0.00114502 | 기준 1e-10 미달 |
| 실제 활성 구조 | qy[4,3]=0, qy[2,1]=0, 정상성 미달 | 두 면·정칙 normals·sector 부호 확인 |
| 최종점 정확 Hessian/P2 | 수행하지 않음 | 이전점의 PD Hessian을 승계하지 않음 |
| 실제 p±h 재최적화·수반 대조 | 미실행 | 미인증 기준점 gate 유지 |
| 현재점 18개 미래 lead 관찰 | 실행 완료 | 평균·후반 성능 실패 |

위 hull proxy는 구현된 branch gradient들로 계산한 수치다. 모든 비매끄러운 구조의 limiting-gradient 자격을 인증한 값으로 확대하지 않는다. 같은 작업 안에서 2-face Newton·반복 보정과 후속 chord까지 실행했지만, 최종 자격을 얻지 못했다.

## 수학·수치·공학적으로 확인한 것

**존재와 복원은 다르다.** 관측손실이 비음수이고 identity prior가 1/2‖c‖²이므로 J≥1/2‖c‖²이다. 목적함수의 연속성·허용집합 닫힘 등 조건 아래 최소해 존재를 논증할 수 있다. 12관측·26제어는 관측만으로 참상태를 유일하게 구별하지 못한다. 참 생성제어의 prior만 약 0.94215로 현재 전체 J보다 크므로, 그 참제어는 이 고정 J의 최소점이 될 수 없다. 이는 모든 최소점의 예측 실패나 문제의 해 부재를 증명하지 않는다.

**참고 경로가 실제 장애를 드러냈다.** 제어 0 경로에서 알려진 support가 roundoff 누적으로 1에서 벗어나 거부됐다. divergence-free 영역에서 동등한 material-difference 식으로 상수를 보존했다. 원래 echo 수송·손실·prior·승인 기준은 유지했다. 실패 기록과 원래 source는 보존했고, 새 source 전이를 명시했다.

**정확 미분의 비용도 줄였다.** 이미 있던 replay=False 옵션으로 같은 FV substep을 수행하며 그래프를 보관했다. 실제 `16793291a9848e432ee9c567fa39828848dbce9fee2eb2ab1d4ffd57cf07bbce` 점에서 J는 동일했고, gradient 및 첫 Hessian 행은 기존 replay 방식과 roundoff 수준에서 일치했다. 해당 행 계산은 8.80초에서 0.724초로 줄었고, 성공한 probe 최대 RSS는 약 478MB였다. 이 probe는 현재 6ca 점의 곡률 검증이 아니다. 전체 optimizer Hessian은 별도로 계산했다. 초기 probe 실패·중복 baseline 비용도 기록했다.

**추가 실제 활성면을 처리했다.** 한 면에서의 보정이 qy[2,1]에 접근함을 실제 제어 벡터로 확인했다. 두 면의 교차점 native J를 재평가했고, 24D 차트의 rank·실제 360-stage branch·네 sector 기울기를 연결했다. 추가 derivative-extension API의 기본/단일면 물리 동작은 유지했다. 내부 limiter 활성조건까지 포함하는 최종 최소점 자격은 확보하지 못했다.

## 정체 원인을 구분한 마지막 실제 검사

`growth_ray_20261013`은 성장률만 ±2e-8, ±2e-7 변경했다. 다섯 점의 limiter 서명은 같았고, 기울기와 실제 비용 변화가 매끄러웠다. +2e-7의 실제 J 감소는 1.72e-10이었다. 이 관찰은 최종 후보가 최소점이 아니라는 추가 근거이며, gradient 전체의 자격을 대신하지 않는다.

`coupled_ray_20261013`은 현재 gradient와 이전 PD Hessian을 근사 검색 행렬로 만든 방향을 실제 평가했다. 하강 곱은 −6.52e-8이지만, 배율 1/128에서 실제 J는 5.69e-11 증가했고 배율 1에서 2.38e-7 증가했다. 모든 비영 평가점에서 limiter 서명이 바뀌었고, 다른 유동면의 부호는 그대로였다. 비용 검사가 이 후보들을 거부한 것은 맞다. Executor와 일부 helper hash는 실행 후 `POST_RUN_SOURCE_PROVENANCE.json`에 수록했으며 원래 pre-execution PLAN을 사후 변경하지 않았다. 따라서 이 ray의 source-level pre-execution 고정은 불완전하다. 특정 limiter 하나가 유일 원인이라는 인증은 아니지만, 같은 매끄러운 모형·작은 chord 반복을 더 수행할 근거는 약하다.

중간 retained-graph 실행은 `7748a8733f714d650a2e378f06e9d43a773f1e29d526fdedca37a5d6d2390806`에서 J=0.06012109894172103, 8개 채택, 정상성 미달로 끝났다. 최종점 P2는 없으며 metadata ledger 오류는 별도 read-only closure로 기록했다.

현재 우선 차단 항목은 **이 후보 근방 limiter 전환을 포함한 원래 비매끄러운 정상성/활성조건의 자격**이다. 허용오차를 완화하거나 비용·잔차 감소를 수반 자격으로 승격하지 않는다.

## 최신 후보의 상태와 미래검증

원래 replay=True·기본 selector에서 native J를 독립 재계산하고, 18개 미래 lead를 한 번 예측했다. 예측점수는 선택에 사용하지 않았다. 기준은 원래 생성제어로 만든 별도 NumPy RHS/DOP853 시간적분이다. 같은 공간 이산화의 독립 수치 기준이며, 연속 PDE 해·보류된 기상사례·실제 레이더 검증은 아니다.

| lead | 예측 MAE dBZ | persistence MAE dBZ |
|---|---:|---:|
| +10분 | 0.122949 | 0.277197 |
| +30분 | 0.231955 | 0.646865 |
| +60분 | 0.610460 | 1.043451 |
| +180분 | 2.370457 | 1.477708 |
| 18개 lead 평균 | 1.163358 | 1.144573 |

+10~100분은 예측 MAE가 더 낮고 +110~180분은 더 높다. 최종 lead MSE는 7.341130 dBZ²이며, 18-lead 평균 MSE 2.434426과 구분한다. raw observer의 `first_persistence_mae_crossover_minutes=10`은 최초 개선 lead를 뜻하는 잘못된 명명이다. 최초 악화 lead는 저장 배열에서 계산한 +110분이며 raw 기록을 덮어쓰지 않았다.

초기 dBZ MAE는 0.06636이지만 유동 flux 상대 L2 오차는 1.1384, 성장량은 참값 0.08/10분 대신 0.05496/10분이었다. 초기장 근접성과 유동·성장·장기 예측 복원은 별개다. 비용 감소를 같은 비율의 예측 개선으로 해석하지 않는다.

## 실행과 재현 범위

원래 Powell 참고계산의 2000 direct-J cap을 닫은 뒤 자격·관찰 phase를 별도로 기록했다. 이후 phase의 direct-J 합계는 506이며, 원래 참고계산까지 합치면 추적된 호출은 2506이다. 초기 자격검사·중단 실행의 모든 호출을 재구성한 전체 합계는 아니다. AD/Hessian의 내부 forward를 이 숫자로 대체하지 않는다. 최종 시간 장부는 `FINAL_ACCOUNTING.json`이며, 계측된 phase 시간과 과거 미계측 구간의 예약을 합친 **추정 계산 장부**이다. 실제 세션 전체 경과시간이나 엄격한 3600초 wall 준수 인증이 아니다.

과거 중단 PCG의 부분 HVP 총량과 한 full-H 실행시간은 미계측이다. 이전 source closure 실패, launcher 오류, mutable prefix overwrite와 metadata-finalization 오류를 감추지 않는다. `QUALIFICATION_CONTINUATION_PLAN_v6.json`이 고정한 초기 prefix SHA `6191484c29ac6ff2079a2d9bf58d4ddc42d709c31439b2efc67bfe8da04f23d8`의 원시 bytes는 덮어써져 복원할 수 없다. 그 hash만 v6 plan에 남아 있고, 이후 최종 bytes와 snapshot을 구분한다. 새 실행의 snapshot/derived closure를 구분하며, 동일 계산을 ledger 오류만으로 다시 수행하지 않는다. 최종점의 P2·정확 응답은 끝까지 미인증이다.

검증 기록은 해당 phase·suite별로 유지한다. selected-face/FV/trajectory 영향 시험 82개, dual-response 계약 시험 11개 등의 결과를 합산해 연구 완성도로 사용하지 않는다. HTML 정적 검사는 근거 hash·링크·idempotency와 DATA 보존을 확인했으며 browser 렌더 검사는 수행하지 않았다.

## 주요 산출물

- `two_face_intersection_20261013/continuation2_20261013/native24_chord_extension_result.json`: 현재 후보와 실제 정상성 미달.
- `current_endpoint_observer_20261013/final_native_forecast_20261013.json`: 최신 상태·lead별 예측·persistence 비교.
- `growth_ray_20261013/RESULT.json`, `coupled_ray_20261013/RESULT.json`: 실제 국소/유한 변화 구분.
- `active_face_response.py`, `active_two_face_response.py`: 실제 재최적화·정확 응답의 실행 경로. 기준점 미인증으로 실행 gate 유지.
- 기존 `examples/weather_scenarios/fv_response.html`: 성공·실패 사례와 최신 곡선.

이번 작업은 원래 문제의 참고계산과 실제 종단 관찰을 연결했다. **원래 적격 최소점·정확 수반·섭동 재최적화·독립 기상사례 검증 완료를 주장하지 않는다.**
