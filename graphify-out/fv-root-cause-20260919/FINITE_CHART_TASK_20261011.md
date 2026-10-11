# PR290 비유한 차트 기록 수정

- [x] frozen 검색+실제 strict writer로 inf/NaN/norm overflow 실패 재현.
- [x] 동결 소스를 유지한 후속 helper에서 비유한 이동량만 null 기록.
- [x] 첫 slot 거부·다음 정상 후보·slot/evaluator 카운터·P2 prefix 회귀.
- [x] 유한 검색 반환값·JSON 바이트 동일성 확인.
- [x] 수정 helper를 사용하는 새 producer·fec9 닫힌 부모 자격·P3 audit 정리.
- [x] 영향 시험 24개·타입0·GREEN/RED·Graphify 격리 AST·176/253 핀·snapshot.
- [ ] PR/CI 및 정확 head 통합 확인(사전 체크리스트; 통합 receipt로 종료 기록).
- [ ] 후속 계획의 실제 FV 실행·현재점 수렴 및 최종 자격/수반/예측. 이번 수정 범위 밖.

이번 production 호출 및 새 optimizer 확정 0. 과거 frozen helper/producer/계획/실행은 보존한다.
