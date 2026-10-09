# 현재점 양측 분기와 결합 최적조건 검토

- [x] PR264 현재95a 확정점·큰 접선 잔차·과거 다른점 증거의 한계 확인.
- [x] GREEN/RED 설계: 같은 원래26변수 J의 단일 upwind분기 extension, 기존 smoothroot와 별도.
- [x] Opt-in core extension·기본 연산 parity·checkpoint replay·y/minmod 회귀.
- [x] 새점/Q0 chart/분기 gate·scaled27 F/Jacobian·한 스텝 구현 및 수학시험.
- [x] 원래 source 보존·plan 고정·타입·격리Graphify·GREEN/RED preflight.
- [x] 단일 제한 실행: 지원 gate 먼저, 통과 시만 현재 양측Hessian/결합선형계와 후보 평가.
- [x] 실제 수치·normal/tangent 잔차·곡률/분기 한계·최종팀검토·KG 기록.
- [ ] PR/CI/병합: 별도 통합 영수증에서 완료 기록.

원래26개prior/관측/경계/시간/J를 유지한다. targetflux 배열을 0으로 덮어쓰지 않는다. 현재점에서 같은 retained25좌표를 Q0로 복원하고 고정 upwind 분기의 미분을 별도 타입으로 기록한다. 다른 minmod/유량 selector가 strict하지 않으면 비지원으로 종료한다. 원래 smooth gradient1e-10 통과나 최소점/수반을 사후 선언하지 않는다.

단일 실행 완료: 양측지원/56HVP/27선형계는 성공, 고정 branch 정책의8후보 안에서 확정점은 없음. 원래95a 보존.
