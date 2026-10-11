# PR290 비유한 차트 기록 P2 후속 수정

실제 strict JSON 원자 writer와 검색 progress callback을 연결해 세 경우를 재현했다. frozen helper는 +Inf 좌표, NaN 좌표, 유한 좌표의 norm overflow 모두에서 `actual_path_norm`의 비유한 float 저장이 실패했다. evaluator 호출은 0, 디스크는 slot_started, 이전 확정 prefix는 보존됐다. `FINITE_CHART_FROZEN_REPRO_20261011.json`에 결과를 기록했다. 실제 FV를 재실행한 재현은 아니다.

새 `fv_point_3h_search_progress_finite.py`는 frozen 검색 함수를 그대로 호출하되 progress를 전달하기 전에 현재 trial의 비유한 actual_path_norm만 JSON null로 바꾼다. 원래 local movement 값·유한성 검사·거부 상태·모델·배율·반경·승인 수학은 변경하지 않는다. 계산 불능 값을 0으로 바꾸거나 후보를 통과시키지 않는다. evaluator 반환 facts의 기존 비유한 정리는 별도의 경로로 유지한다.

| 실제 writer 연결 대역 시험 | frozen helper | 후속 helper |
|---|---|---|
| +Inf 좌표 | JSON ValueError, 다음 후보 미평가 | null/비유한 slot 거부, 정상 두 번째 후보 평가 |
| NaN 좌표 | 동일 | 동일 |
| 유한 좌표지만 norm overflow | 동일 | 동일 |

후속 경로는 슬롯 시작·완료 2/2, 실제 evaluator 시작·완료 1/1, 첫 후보 accepted=false를 저장하고 두 번째 적격 후보를 반환한다. callback 저장 전후의 마지막 확정점·횟수·닫힌 prefix는 유지된다. 유한 검색의 반환값과 최종 JSON 바이트는 frozen 경로와 동일했다. 이 시험의 evaluator는 명시적인 승인 facts 대역이며 실제 J/gradient/FV의 인증은 아니다.

새 `fv_point_3h_gn_finite_continuation.py`가 수정 helper를 사용한다. prospective 시작점은 `fec9f3062339a0a24d1934fe9ad35fbe9c9cbea4951ca2a5967bde933d6a841c`다. 이전 실행의 내부 budget 종료와 종료 deadline=false는 그대로 인정하면서, 마지막 닫힌 P2의 deadline 통과·제어·theta·전체 G·원래 입력·runtime·source를 검증한다. 이전 미완료 row 19개와 이전점의 풀이·HVP·방향은 새점 연산으로 가져오지 않는다. 원래 3회 확정 상한·24개 후보·600/660초·2GiB 정책은 그대로다.

P3도 후속 코드에서 정리했다. 새점 진입과 확정 checkpoint에서 최상위 dense_solve_audit를 제거하고, 닫힌 단계의 gn_direction 내부에 해당 점의 완료 풀이 근거를 보존한다. 임시 파일의 저장 계약 시험으로 제거를 확인했고, 새점에서 실제 GN 연산을 실행한 것은 아니다.

영향 시험 24개 통과(3.73초), 타입 오류·경고 0개, GREEN/RED 교차검토, source 176개/archive 253개 핀 및 정확 source snapshot을 확인했다. 24개에는 frozen 결함을 기대 ValueError로 재현한 3개가 포함된다. 기존 소스·계획·원시 실행·gzip·run/resource는 변경하지 않았다. 실제 결과가 무효화된 것이 아니므로 PR290의 두 FV 확정점은 재실행하지 않았다.

Graphify는 기존 GN_COMMITTED_CONTINUATION_GRAPHIFY를 확인하고 소유한 세 파일의 격리 AST만 갱신했다(34노드·124엣지, 모델 토큰 0). 공유 graph.json을 변경하거나 전체 semantic rebuild를 수행하지 않았다.

계획은 prospective이며 이번 변경의 실제 FV·HVP·GN 풀이·새 후보·새 확정은 모두 0이다. 처음부터 실행한 후속 producer의 FV 완주, 후보 timeout 또는 OS 강제 종료를 새로 인증한 결과가 아니다. 후보 초기 길이를 이전 유효 이동과 연결하는 제안도 구현하지 않았다. 이후 현재점의 연산을 재자격하고 가까운 거부·채택 후보의 gate와 활성구조를 구분하는 연구가 남는다.

기존 1차 이론 실증 100%, FV 확장 약 70%를 유지한다. 현재 norm 약 0.0559124와 적격점·정확 곡률·수반·재분석·미래점수의 미완료 상태는 그대로다. 이번 성과는 비유한 차트의 거부·저장 경로 수정이며 수렴 또는 예측성능 개선이 아니다.
