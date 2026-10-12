# HTML 수행 사례 검토 · 2026-10-12

- [x] 기존 구조 graph에서 FV replay builder → template 연결 확인.
- [x] 원수치에서 10개 사례 목록 생성, source SHA-256 기록.
- [x] 기존 4×5 소형 데모와 26제어·12점관측 연구 구분.
- [x] PR302 후보 미평가와 실제 실패 구분; PR301/304의 동일 후보 추가이득 미입증 표시.
- [x] J/R/E 감소와 수렴·참상태 복원·정확 수반·독립 예측검증 구분.
- [x] 원래 최소해의 조건부 존재, Q=0 전체 면의 제한해 존재, 관측 국소 nullspace와 유일성 한계 검토.
- [x] 현재 비용 기반 norm bound와 tanh 도함수 bound 검토; 포화를 입증된 원인으로 쓰지 않음.
- [x] 기존 replay DATA SHA 불변, 10 details/unique ID/근거 링크/세 E pairing 확인.
- [x] 기존 builder의 저장 DATA 재생성 일치, 갱신 idempotency, JavaScript syntax 확인.
- [x] 표시 helper만 Graphify AST 갱신; 수치 source 변경과 새 FV/HVP/점수 호출 없음.
- [ ] 화면 렌더링 검증: Browser 보안 정책이 file: protocol을 거부해 수행하지 않음. 우회 안 함.
- [ ] 원래 문제 적격 최소점·정확 곡률·수반·섭동 재최적화·독립 미래검증: 기존 미완료 유지.

갱신 명령: repository root에서 `.venv/bin/python examples/weather_scenarios/update_fv_case_review.py`.
이 명령은 저장 근거만 읽고 template과 기존 standalone `fv_response.html`의 검토 섹션만 갱신한다. 기존 numerical builder는 변경하지 않았고 이후 재생성에도 섹션이 유지된다.
