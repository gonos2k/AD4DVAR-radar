# PR289 새 확정점에서 제한된 GN 연속 실행

- [x] 현재 a3d2ad7b 부모 확정 근거와 사용자 검토 대조.
- [x] cached GN_LOCAL_CANDIDATE_GRAPHIFY 구조 consulted; 기존 연구 식·지원 범위 유지.
- [x] 새 prospective producer: 현재점별 양측 gradient/최소 theta/fullG, row24/GN/HVP2 갱신.
- [x] counter P2: slots/actual evaluation 시작·완료 분리, 실패 상태와 마지막 P2 prefix 보존.
- [x] 영향 시험·타입·GREEN/RED 사전 검토·정확 소스 snapshot 및 173 source/248 archive 계획 동결.
- [x] ROOT만 600/660초·2GiB guard 한 번, 최대 3확정·점별 24후보.
- [x] 실제 J/R/max 성분/이동량/후보 수/모형 오차/종료 이유·원시 해시 검산과 archive.
- [ ] GREEN/RED 사후 검토·KG 기록·PR/CI·정확 head 통합.

기존 frozen producer/계획/수치 archive는 변경하지 않는다. 동일 selector는 진단이며 새 승인 조건이 아니다. 원래 prior/관측/수송/minmod/반경·dual Armijo/P2를 유지한다. 종료 원인을 확정 상한·후보 소진·지원 거부·시간·실패·잔차 후보로 구분한다. 잔차 후보는 최종 자격검사 미완료 상태다. 미래점수/수반/재분석은 이번 실행 범위 밖이며 FV 완성도 약 70%를 승계한다.

실제 종료: 두 확정 후 세 번째 점 row19/24에서 내부600초 예산 종료. 3스텝 완주 아님. 최종 deadline_passed=false지만 두 P2 시점은 통과했고 마지막 확정점 보존. P3 top dense_solve_audit의 점1 범위는 문서에 명시했으며 다음 코드의 label/drop 개선은 미적용.
