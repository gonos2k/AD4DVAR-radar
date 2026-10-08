# 표본공통 방향 실제 비용 탐색

- [x] Inner 전체-gradient 최소norm방향과 currentbase 보관gTd/법선각도 재산술.
- [x] GREEN/RED 설계: J-only 비용탐색/Phi진단/final기준 유지.
- [x] 기존bounded-original-J search 재사용 adapter·스케일/계보/확정 회귀시험.
- [x] Source/plan 고정·타입·격리Graphify·GREEN/RED preflight.
- [x] 단일240초 내부/300초 외부/1GiB/1HVP/최대16후보/1확정점 실행.
- [x] 실제J/Phi/분기/이동량/방향 후속유효성/마지막GREEN/RED 기록.

원래26변수J·prior·관측·파라미터·경계·시간을 유지한다. 새로운 평활화/면제약/정상성 기준완화 없이 실제원래J로채택한다. Phi증가가능성을실행전에명시하며 비용탐색성공을정상성성공으로 승격하지않는다. 공통방향은거의접선이며 경계횡단을보장하지않는다.

실제 결과: J −0.04974%, Phi +136.36%. 확정점 26e830dc…는 비용 탐색 결과이며 정상점이 아니다. 대상 면유량은 음수를 유지했다. GREEN/RED는 새 필수 코드 결함 없이 범위 제한을 확인했다.
