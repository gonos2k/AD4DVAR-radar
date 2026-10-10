# PR #274 팀 재조사

GREEN(승인·근거), RED(상태·예외), RED(수학·보고)의 세 독립 역할로 조사했다. 실제 FV·seed·생산용 gradient/HVP·guard·최적화는 다시 실행하지 않았다.

## 발견과 수정

**초기 row-only 거부를 실행 실패로 분류하는 P2를 재현했고, 두 누락 조건을 수정했다.** 원래 시작점의 양측 gradient가 기존 P2 재현 허용오차 안에서 달라져도 F² 재현은 통과할 수 있고, 새 최소 θ는 carried θ와 그 θ 자체의128epsilon 범위보다 조금 더 달라질 수 있다. 실제 child 경로의 합성 시험에서 이를 재현했다. 제어·비용을 움직인 시험이나 새로운 FV 실행은 아니다.

기존 부모 검사는24row 계산 후 거부될 때 row θ를 이전 carried θ와 비교했다. 수정은 그 계산점의 `base_mixing_minimum.working_theta_star`에 연결하며, reference carried θ·내부값·유한성·24개(side,row)·현재 제어 계보를 그대로 유지한다. 허용오차를 낮추거나 새 θ solver를 넣지 않았다.

또한 첫 계산점에서 row parity가 실패하면 dense counter가 아직 초기화되지 않을 수 있었다. 기존 조건은 None을0과 다르다고 판단했다. 풀이 history가0개일 때 없는 counter를0으로 읽도록 수정했다. 실제 풀이 history가 있으면 누락 counter는 계속 거부한다. 두 조건은 정상 지원 거부를 실행 실패로 잘못 표시하던 문제이며, 잘못된 후보 확정을 재현한 것은 아니다.

회귀시험은 공통 gradient drift를 각 side P2 budget 이내로 구성하고 carried merit의1차 변화도 상쇄한다. 실제 admission을 통과한 후 현재 최소 θ·24row·0HVP·0풀이·0스텝 거부를 닫으며, 현재 최소 θ 기록을 carried 값으로 바꾸면 거부되는지도 확인한다. 이 사례는 초기점에 한정된다. 정상 확정 이후에는 현재점 cache와 최종 최소 θ 재현 조건이 같은 이탈을 제외하므로, 그 경로에 같은 결함이 있다고 주장하지 않는다.

## 기존 실행의 유효 범위

PR #274의 첫 확정점과 보고 수치는 재검산에서 일치했다. raw/gzip·run/resource hash, 제어·θ·개별 gradient P2, J/F², Woodbury·DF, row26시작/25완료·HVP2·풀이1·확정1을 확인했다. 실제 실행은 row-only 거부가 아니라 RSS 중단이므로 이번 수정이 보관 결과를 바꾸지 않는다. source_after와 전역 입력/runtime 전후 객체가 없는 점도 그대로 유지한다. 실행 전체가 종료 검증됐거나 세 반복을 완료했다는 주장은 없다.

NaN row를 직접 넣는 변조 시험에서 부모가 row 값 자체를 다시 검사하지 않는 범위도 확인했지만, 생산자는 finite 검사와 allow_nan=False 직렬화로 이를 만들지 않는다. 현재 고정 source에서 도달하는 계산 결함으로 분류하지 않았고 새로운 중복 검증 계층을 추가하지 않았다.

메모리 직렬화 비용은 실제저장 크기와 소스 구조에서 확인되지만 RSS 초과의 유일 원인이라는 근거는 없다. Streaming writer·중복 진행 기록 제거는 미적용 후속 제안이다. 수반·재분석·독립 미래검증과 FV완성도70% 미완료 범위도 유지한다.

## 검증·보존

수정 후 영향 시험89개 통과·기존 경고18개·41.18초, 타입 오류0이다. GREEN·RED 최종 patch 검토가 통과했다. 수정은 승인 메타데이터 분류와 합성 회귀이며 원래 목적함수·최소 θ·GN 방향·미분·실제 후보 채택 기준은 바꾸지 않았다.

과거 frozen plan은 그대로 두고, 수정 전 core/test의 정확한 bytes를 `audit274_source_20261010`에 보존했다. 이두개 snapshots를 통해 과거 실제실행의 source139/archive148 pins를 모두 확인했다. 현재코드가 바뀌었으므로 과거계획의 live-source 검증을 그대로 통과한다고 주장하지 않는다. 과거실행·기존원시자료·계획·제어값을 덮어쓰지 않았다. Graphify는소유두파일만 AST갱신했다.

근거: [GREEN](AUDIT274_GREEN_20261010.md), [RED 상태](AUDIT274_RED_STATE_20261010.md), [RED 수학](AUDIT274_RED_MATH_20261010.md), [원본 source 보존](AUDIT274_SOURCE_SCOPE_20261010.json), [시험](AUDIT274_TESTS_20261010.log), [타입](AUDIT274_TYPES_20261010.log).
