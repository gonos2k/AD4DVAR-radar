# HTML 수행 사례 검토 · 2026-10-13

- [x] 기존 구조 graph에서 FV replay builder → template 연결 확인.
- [x] 저장 수치에서 18개 사례 목록 생성, 모든 근거 파일의 SHA-256 기록.
- [x] 기존 4×5 소형 데모와 26제어·12점관측 연구 구분.
- [x] PR302 후보 미평가와 실제 실패 구분; PR301/304의 동일 후보 추가이득 미입증 표시.
- [x] J/R/E 감소와 수렴·참상태 복원·정확 수반·독립 예측검증 구분.
- [x] 원래 최소해의 조건부 존재, Q=0 전체 면의 제한해 존재, 관측 국소 nullspace와 유일성 한계 검토.
- [x] 현재 비용 기반 norm bound와 tanh 도함수 bound 검토; 포화를 입증된 원인으로 쓰지 않음.
- [x] 기존 replay DATA SHA 불변, 18 details/unique ID/근거 링크/세 E pairing 확인.
- [x] 기존 builder의 저장 DATA 재생성 일치, 갱신 idempotency, JavaScript syntax 확인.
- [x] 표시 helper만 Graphify AST 갱신; 수치 source 변경과 새 FV/HVP/점수 호출 없음.
- [x] A: native Powell 두 start의 budget/maxfev stop, support-roundoff 원인/fix, 비정상점의 PD Hessian과 다른 face point의 음의 곡률을 구분. 6ca native24 fixed-H chord는 J 0.0601191808390→0.0601191808342까지 낮췄지만 measured finite-gradient hull proxy ∞=0.00114502>1e−10, terminal Hessian/P2 미실행으로 qualification 실패; receipt는 이를 limiting-gradient certification으로 주장하지 않음.
- [x] A: growth-coordinate partial은 h=±2e−7에서 같은 selector signature와 유한한 음의 partial을 보였고, 별도 stale-metric coupled ray는 첫 nonzero sample에서 branch를 바꿈. 둘 다 최소점 증거로 해석하지 않음.
- [x] B: 최신 6ca full-gradient/curvature/branch gate를 열지 못해 실제 perturbed reoptimization, adjoint response, score가 실행되지 않았음을 명시.
- [x] C: GREEN 검증 observer receipt·fields hash·독립 native J를 확인한 최신 6ca native forecast를 observer only로 표시; 18-lead mean MAE 1.163358 대 persistence 1.144573 dBZ, +110분부터 MAE가 더 나쁨. Minimum 및 score-selection 주장은 없음.
- [x] C의 누적 budget erratum 표시: correctly indexed trajectory 2개, superseded indexing attempt 1개, 닫힌 2,000-call search cap 뒤 native-J 3회 초과. Reference는 true generator; 96108은 이전 candidate metadata임.
- [x] Saved 6ca fields와 fixed-case true-generator reference에서 18-lead MAE curve 생성; lead1–18/+10…+180min 정렬, first worse +110min. 이전 8eff curve도 별도 보존; 두 PNG 시각 점검 완료, 기존 signed SVG는 그대로 둠.
- [x] `DATA` literal SHA-256 `8e5e9772…29c4a498` 불변, 두 번째 builder 실행의 HTML/template/JSON bytes 동일, 34 evidence hashes 일치. Embedded historical signed-span SVG has a direct source link and its own hash pin.
- [x] In-progress·superseded evidence 파일은 source pin에서 제외; GREEN check의 budget/lead erratum은 보존.
- [x] Current and historical MAE chart PNGs visually inspected. Browser page rendering remains unchecked.
- [ ] 적격 최소점·정확 응답/수반·섭동 재최적화·held-out weather forecast: 기존 미완료 유지.

갱신 명령: repository root에서 `.venv/bin/python examples/weather_scenarios/update_fv_case_review.py`.
이 명령은 저장 근거만 읽고 template과 기존 standalone `fv_response.html`의 검토 섹션만 갱신한다. 기존 numerical builder는 변경하지 않았고 이후 재생성에도 섹션이 유지된다.
