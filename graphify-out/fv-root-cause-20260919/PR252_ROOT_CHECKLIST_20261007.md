# PR252 정상점 후보 중단 순서 P2

- [x] 원본 Git blob에서 C=0,1,0.063 이차문제 반례 재현. 소형 실제 repository PCG, FV 분기는 대역.
- [x] 정상점 후보의 J/g/Phi/엄격한 분기 재평가와 trial 일치·두 Armijo 조건 재확인.
- [x] 확인된 정상점 후보만 비용 감소량·이동 정체 하한을 우회. 비정상 후보는 기존 거부 유지.
- [x] 최종 callback·입력/소스 무결성·deadline 이후에만 확정. pending-audit이며 eligible 아님.
- [x] 큰 곡률·작은 이동, stale gradient/objective/branch, 이전 확정점 뒤 root후처리 실패 회귀.
- [x] 집중46시험 통과(기존18경고), 타입오류0, 격리Graphify46nodes174edges.
- [x] 최종 GREEN/RED GO.
- [ ] PR통합.

이 수정에서 새 FV 계산을 수행하지 않았으며 PR252 원시 계획·실행자료·소스 사본은 보존했다.
최종90fc의 정상성·전체 곡률·수반·재분석·독립 기상예측 성능은 별도 미완료이다.
