# PR275–276 국소 유효구간 조사

- [x] 최신2ec 확정점·거부 방향·양측HVP·후보16개 보관 근거 GREEN/RED 대조.
- [x] 원래J와 최소혼합잔차G를 유지; 현재양측분기 일치와 기준selector불변을 구분.
- [x] 독립 진단 구현·회귀시험13개·타입0·격리 Graphify AST 갱신.
- [x] 2GiB/600–660초/HVP2/표본8/optimizer0 계획 고정. 첫 상대경로 preflight 실패를 보존하고 새 계획으로 수정.
- [x] 현재점 J·양측 gradient·분기·방향·실제 chart·fresh DG 재자격.
- [x] 실제 G(h)와 선형모형·selector·margin·두 Armijo 대조. 7개 표본에서 감소 회복, 확정0.
- [x] 저장 배열 재산술과 GREEN/RED 최종 감사, KG·보고서 기록.
- [ ] PR/CI 통합은 별도 integration receipt로 확인.

이번은 관찰 전용이며 감소조건 통과도 optimizer확정으로 세지 않는다. 과거 실패후보를 성공으로 바꾸지 않는다. 같은방향의 더작은유한구간만 검사하며 적격점·응답·미래예측을 완료로 표시하지 않는다.
