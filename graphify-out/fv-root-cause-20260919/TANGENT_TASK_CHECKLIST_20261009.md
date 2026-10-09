# 새점의 현재 접선 미분과 제한된 반복

- [x] GREEN/RED 수학 검토: 현재 gT 방향·두현재HVP·결합normal보정·현재point재자격.
- [x] Graphify 공유 구조와 최근 REQUAL AST 재사용.
- [x] 기존 차트/분기/P2 helper 재사용·보관 producer 보존·중간 실패와 확정 횟수 회귀.
- [x] source/plan 고정·집중시험·타입·격리Graphify·팀preflight.
- [x] root 단일 최대3보정/6HVP/16후보/240초 내부/300초 외부/1GiB 실행.
- [x] 접선/법선·Fnorm/inf·theta·각분기·진전·정체/거부/예산 상태 점검.
- [x] 최종 GREEN/RED·KG 검산: 새 필수 계산 결함 없음.
- [ ] PR/자동 CI/정확 head 통합.

원래J/prior/관측/경계/시간 유지. 새6b29에서매번현재미분으로갱신한다. 과거행렬은현재Hessian이아니며이번반복에는사용하지않는다. 정상점·최소점·곡률·수반·재분석과예측성능은별도미완료. Root만실제guard를시작하고팀은중복실행하지않는다.
