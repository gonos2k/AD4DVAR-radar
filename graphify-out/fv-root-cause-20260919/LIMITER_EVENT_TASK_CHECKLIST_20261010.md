# PR279 후속: 특정 limiter 사건의 현재 민감도

- [x] 기존 구조 그래프 GN_PREPARED_SEARCH_GRAPHIFY 확인; 생산 transport observer 구조 확인.
- [x] 저장된 거부/확정 trace에서 실제 사건을 특정한다(GREEN/RED).
- [x] 이벤트 좌표는 trace 내부 좌표와 전체 q 셀 좌표를 구분한다.
- [x] 공개 diagnostic observer는 detach하므로 AD 경로에 사용하지 않는다. 읽기 전용 private ContextVar scope를 쓰고 기존 모델은 변경하지 않는다.
- [x] 양측 현재 ζ·gradζ·GN JVP·grad-dot-direction을 확인한다.
- [x] 제한된 차트 표본에서 사건 예측/실제/selector/원래 J와 최소R를 비교한다.
- [x] source/archive 계획 고정, 회귀/타입/격리AST, ROOT 실제guard.
- [x] 최적화 확정0·탐색제약0·새solver0. 지원범위를 벗어나면 진단 거부로 분류한다.
- [x] GREEN/RED 저장 감사, 결과/KG/PR 근거와 남은 한계를 기록한다.

사용자 승인 RSS2GiB 유지, 실제 자원한도 필요시에만 확대. 현재점027e의 보관GN방향/HVP는 같은점재자격 확인 후 재사용한다. 원래J·관측·prior·경계·limiter 및 최종기준은 변경하지 않는다.

실제 attempt2 정상완료:26.032초·표본RSS596,967,424bytes·eventgrad2/JVP2/sample3·확정0. 첫실행 소스폐쇄실패 근거는 보존했지만 당시helper/test정확바이트는 미보존임을 결과에 명시했다.
