# Qy32 양측 진단 작업 기록

- [x] PR256 검토와 원래 문제/계산점 계보 확인; 104 source +28 archive pins 일치.
- [x] 기존 구조 graph의 transport face_volume_fluxes AST와 기존 반복/수송 코드 확인.
- [x] GREEN/RED 설계 검토: full26 prior 유지, fixed retained25 chart, eta0 cost only.
- [x] 점별 결과 내구성 및 실행 전후 입력/소스/환경 확인.
- [x] 집중 시험·타입·격리된 incremental AST 추출.
- [x] 명시적 단일 실행 계획 고정 및 GREEN/RED 사전 검토.
- [x] 최대240초 내부/300초 외부/1GiB 단일 진단 실행; 추가 실행 없음.
- [x] 결과 산술/접선·법선·구간별 분기 비교와 최종 GREEN/RED 검토.

완료 범위: e29 고정 접선에서 Qy[3,2]의 네 비영 표본 및 eta0 primal 비용 진단. 최적화/HVP/PCG/score/수반/재분석은 수행하지 않는다. 유한 표본은 양쪽 극한, 단일면 원인, 정상점/최소점, 기상예측 skill을 인증하지 않는다. FV 70%는 외부 가중평가이며 이번 진단으로 상위 정상점/응답 마일스톤을 닫지 않는다.

실행 시작: 계획 SHA bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f. GREEN·RED 사전 GO; raw qy32_diagnostic_attempt1/diagnostic.json 점별 원자적 보존.

완료: guard34.12265554초; four strictpass, eta0 costonly; HVP/PCG/optimizer0. 최종 GREEN/RED 기록을 함께 보존한다. 원래e29 확정점은 이동하지 않았다.
