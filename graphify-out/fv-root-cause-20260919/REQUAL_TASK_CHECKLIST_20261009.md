# 후보의 현재 분기 재자격 검증

- [x] P2 승인 누락 재현·gradient_match 필수 gate·상쇄 gradient 회귀.
- [x] 과거 문서의 면 부호 범위 및 선형잔차 분모 구분 수정.
- [x] 기존 Graphify/KINK 구조·부모 실행·같은 점 방향 재사용 범위 검토.
- [x] 별도 current-candidate qualification 정책·현재 방향 HVP 감사·공통 확정 kernel 연결.
- [x] 원본/고정계획 보존·회귀·타입·Graphify·GREEN/RED preflight.
- [x] 단일240/300초·1GiB·2현재HVP·최대8후보·최대1확정점 실행.
- [x] 실제J/F·양측gradient·현재 selector·변경 기록·최종팀검토·KG.
- [ ] PR/CI/병합: 별도 통합 영수증으로 종료 기록.

과거0채택 실험은 보존한다. 같은 원래95a/Q0 기준점의 보관 방향은 새양측 gradient/두현재HVP/실제27잔차로 확인한 뒤 후보생성에만 사용한다. 새점에서 과거행렬을 현재Hessian으로 표시하지 않는다. 후보의 양쪽은 현재 다른selectors가 서로 같고 strict여야 한다. 이전점과의selector불변은 별도진단으로남기며 새정책에서필수조건이아니다. 원래J/prior/관측/경계/시간/정상점기준과 최종개별gradient재현조건은유지한다.

실행은단일guard1회로한재자격점채택. 기존56HVP/0채택기록보존,최종개별gradientgate통과,최적점/응답미완료.
