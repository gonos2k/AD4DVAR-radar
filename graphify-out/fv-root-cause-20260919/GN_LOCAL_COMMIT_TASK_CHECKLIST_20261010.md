# 국소 통과 표본의 실제 확정과 새점 연산 갱신

- [x] GREEN/RED 설계: 원래 J/G·반경·최소theta·분기·P2 유지.
- [x] 고정 alpha 한 후보·독립 최종 재평가·원자적 한 확정 구현.
- [x] 다음점의 새 관측 row24·GN풀이1·HVP2 준비 구조, 두 번째 후보 없음.
- [x] 쓰기/재평가/확정후연산 실패 시 마지막 확정 보존 회귀.
- [x] 관련18시험/타입0, 격리AST·source148/archive190 계획 고정.
- [x] root guard 1회 완료: 실제 후보 1회 확정과 새점 readiness 완료를 구분.
- [x] raw/gzip 해시·무손실 왕복 및 RED 최종 P2/row·solve/HVP/source closure 감사.
- [x] GREEN/RED final memo와 교차 내용 검토 완료.
- [x] GREEN/RED 저장 배열 최종 감사와 KG·보고서 기록.
- [ ] PR/CI·정확 head 통합은 별도 integration receipt로 확인.

기존16검색상한과 과거생산 helper는 그대로 둔다. 과거2ec의 HVP벡터2개는 같은점 재자격 후만 사용하며 새점에 승계하지 않는다. 새점의2HVP를 별도집계한다. 적격점·수반·재분석·독립미래점수는 미완료로 유지한다.
