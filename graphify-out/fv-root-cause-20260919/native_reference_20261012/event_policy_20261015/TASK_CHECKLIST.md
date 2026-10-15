# 유지·해제·다음 전환 실행

- [x] PR306/main926aff02 병합 상태와30bed 입력/기존 장부594J·163gradient 확인.
- [x] 기본 수송을 유지하는 minmod 피연산자별 smooth extension과 replayFalse guard.
- [x] gap교환/피연산자zero통과의 동적 식별. 번호 허용 목록을 실제 진행에서 제거.
- [x] 8/16 side×flux 연장, 양측 나가는 미분과 실제 native side/Armijo로 유지·해제.
- [x] 실제 활성집합만 재구축, 최대2동시사건. 선형독립/정상성 검사 후에만P2.
- [x] 끝점 같은연산 캐시는 표시해 재사용. 비정상 끝점의 불필요한Hessian 반복을 생략.
- [x] 새phase 추가900초/J384/gradient768/Hessian행2048/profile256/RSS2GiB 계획 동결. 기존 누계 유지, 전체과거AD·wall완전인증 아님.
- [x] 실제 기존stage48 유지·해제와 그에따른 후보 승인.
- [x] 관찰된 새전환 형식 위치·연산 갱신·후보 진행을 같은실행으로 연결.
- [x] 동일끝점 정상성/현재활성구조/필요P2 또는정확미해결항목.
- [x] GREEN/RED 최종검토·독립장부/입력/제어/연장/Armijo 검산·Graphify증분·PR.

전환 위치는 세 내부표본과 binary changed-trace 구간에서 실제root를 찾는다. 표본 사이의 모든 교차/재교차를 배제한 전역최초사건 인증은 아니다. 중간retraction비용을최종후보로직접채택하지 않는다. 기존Powell/ray/6ca예측검증은 반복하지 않는다.

실제 결과:12회 채택, 끝점9c106af4…/J0.06011917924338052. 현재47gap+123right-zero 유지 필요조건 통과; hullInf1.1755e-7>1e-10으로 정상성 미달/P2·수반 미실행. 추가단계 반복 상한으로 종료했고 모든phase평가를 누계에 반영했다.

Draft PR307: https://github.com/gonos2k/AD4DVAR-radar/pull/307 (base926aff02). 정상성·P2·정확수반 미완료 항목은 그대로이며 병합하지 않는다. 자동 CPU/Wheel 검사는 workflow_dispatch 전용으로 skipped; UI 검사만 자동 실행된다.
