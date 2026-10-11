# PR #289 팀 재감사

검토 기준: main `36e85740621e7ff2997017d6e76dff07ad03eed3`, PR head `44d4c51e7e2afeff043b6b20c45e041b052d115e`. GREEN은 원시 저장 근거와 개별 gradient·제어·해시·수치를 검산했고 RED는 실패 경로를 조사했다. 별도 수학 팀은 최소 혼합 미분·Armijo·선택과 모형 오차를 검토했다. 생산용 FV/HVP를 다시 실행하지 않았다. frozen producer·계획·보관 자료는 변경하지 않았다.

## 발견 사항

### P2: 실패 시 실제 평가 시작 횟수 오기록 — 완료 실행 비차단, 후속 수정 필요

`examples/weather_scenarios/fv_point_3h_gn_local_candidate_commit.py:325`–334는 차트 생성·이동량 검사 전에 `evaluation_started`와 `candidate_evaluations_started`를 저장한다. 차트 생성 또는 이동량 검사가 실패하면 실제 비용·gradient evaluator는 호출하지 않았는데 평가 시작으로 기록된다. PR #286에서 구분한 후보 slot과 실제 평가가 이 새 고정 후보 경로에는 분리되지 않았다.

소스 AST에서 해당 실제 for-loop를 추출해 차트만 실패시키는 대역으로 실행했다. 차트 호출 1회, evaluator 호출 0회인데 저장된 평가 시작 횟수 1, 상태 `evaluation_started`가 남았다. 재현 값은 `AUDIT289_COUNTER_REPRO_20261011.json`에 있다. 이 경로는 선택/P2/확정에 도달하지 않는다. 실제 #289는 두 차트와 두 평가를 정상 완료했으므로 기록된 실제 평가 2회와 확정점은 유효하다.

직전 RED 문서는 명시적 interrupted 상태의 부재를 기록했지만, 평가 이전의 차트 실패도 평가 시작으로 세는 별도 정확성 문제를 명시하지 않았다. 직전 최종 보고의 “중단 상태 제한”보다 구체적인 후속 기록 수정 항목이다. frozen 소스를 소급 수정하지 않고 새 후속 경로에서 slot 시작과 evaluator 시작을 구분하며, 차트 실패·반경/영 이동·평가 timeout을 구분하는 회귀시험이 필요하다.

### 현재 producer의 실패 경로 시험 공백

신규 시험은 부모 자격, changed-selector 선택, 동률 순서, P2 runtime 거부의 4개다. 기존 prepared-search 7개와 함께 11개가 통과했지만, 현재 `_run_child`의 후보 loop에서 차트 생성 실패 또는 평가 중단을 직접 실행한 회귀시험은 아니다. 시험 통과가 그 경로의 정확성을 보장한다고 확대해서는 안 된다.

## 재확인한 정상 근거

- 현재·마지막 확정·proposal·final_repeat의 FP64 제어와 SHA256가 모두 `a3d2ad7b…ea30cc5a`로 일치한다. proposal→repeat 양측 gradient의 최대 차이는 각각 0이다.
- 개별 양측 gradient에서 재계산한 theta는 0.6860964916049087, R는 0.0031263066786563877이며 저장값과 일치한다. J 감소 0.0000690623%, R 감소 0.0234226111%, norm 감소 0.0117119914%다.
- 최소 혼합계수 방향미분 식과 R Armijo 계수 2는 맞다. 새 잔차 방향미분 DG[d]와 같은 점의 보관 벡터는 성분별 차이 0이며 전체 Hessian 일치를 뜻하지 않는다.
- 원래 후보 자격을 통과한 두 점 중 실제 R가 낮은 changed-selector 후보를 선택했다. selected→repeat의 trace는 같고, 기준 selector 불변을 승인 조건으로 추가하지 않았다.
- 선택 후보의 벡터 모형 오차/실제 변화 비율은 약 99.8809%다. 보고서가 이를 공개했으며 실제 감소를 모형 정확성의 증거로 주장하지 않았다.
- 소스 171개·보관 파일 243개 핀, raw/gzip/run/resource 해시와 무손실 압축을 확인했다. 실제 연산량은 HVP 2회, 새 row/풀이 0회, 후보 2개, 확정 1개다.

## 별도로 남는 정책·검증 한계

공통 P2 gate는 기록한 theta 상대 허용오차와 별개로 proposal/repeat theta의 exact FP64 동등성도 요구한다. 따라서 허용오차 안의 재평가 차이를 거부할 수 있는 엄격한 정책이다. 이번 theta 차이는 0이며 잘못 승인하는 결함은 아니다. 이 정책을 바꾸려면 기존 P2 계약을 별도로 검토해야 한다.

후보 선택의 동률은 정확한 FP64 값의 동등성이다. 정상 producer는 유한성 gate로 NaN을 제외하며 두 실제 R의 차이는 약 6.47e-7로 동률 처리와 무관하다.

새점의 GN·HVP·사건 미분, 최종 적격점·정확 수반·재분석·미래점수는 미완료다. PR #286 수정 3단계 producer의 실제 FV 완주, OS 강제 종료의 기록 보존도 이번 검사로 완료되지 않았다. CI는 UI 성공이며 CPU·wheel 작업은 opt-in 정책에 따라 건너뛴 결과다. 기존 이론 100%, FV 약 70%를 유지한다.

판정: 성공 실행의 확정 결과를 무효화할 계산 결함은 확인하지 못했다. 실패 경로의 평가 시작 횟수 오기록과 그 경로의 시험 누락은 구체적인 후속 항목으로 남긴다. 이번 감사는 소스 수정 또는 추가 최적화 실행을 수행하지 않았다.

후속 기록: 이후 별도 `fv_point_3h_gn_committed_continuation.py`에서 기존 progressive 검색 helper를 연결해 slot/실제 평가 카운터를 분리하고 직접 callback 실패 회귀를 추가했다. 이 수정과 후속 실제 실행은 `GN_COMMITTED_CONTINUATION_FINDINGS_20261011.md`에 기록한다. PR289의 frozen 코드·계획·완료 실행을 소급 변경한 것은 아니다.
