# 초기장 실험·수치 코어 검토 후속 조치 — 2026-09-05

사용자 제공 검토의 기준은 `06a161c7fc59fd72c2309ab8dde63137af189e5d`이다.
이번 작업은 PR #155의 수치 수정 `ca8f18c` 위에서 진행한다.
제공된 독립 시험 수치는 이번 저장소 실행 결과와 구분한다.

## 처리 순서

- [x] S01: 배경 나이가 장을 노화시키지 않는 메타데이터임을 화면·문서에 설명하고 전달 경로를 시험한다.
- [x] S02: 동일 입력 persistence 대비 개선량의 의미를 명시하고 실제 평가 화소 수·비율을 표시한다.
- [x] S03: A의 설정·기준예측·평가영역을 고정한 A/B 비교를 제공한다. B의 누락으로 영역을 축소하지 않는다.
- [x] S04: `shift_zero()`의 중간 저장공간을 출력 크기로 제한하고 순전파·JVP·VJP·역전파를 확인한다.
- [x] S05: 조건부 불확실성, 비확률적 confidence, 전역 이동·성장 모형의 적용 범위를 문서화한다.
- [x] S06: 입력 출처 검증의 한 책임을 `nowcast.py`에서 분리한다. 타입·registry·검증 규칙을 중복하거나 삭제하지 않는다.
- [x] S07: 영향받는 P0/P1·미분·출처·화면 제어 시험과 HTTP 동작을 확인하고 독립 리뷰 및 검증 한계를 기록한다.

## 유지할 설계

비음수·국소 보존 수송, 직접 선행시간 예측, 고정 선형화, PCG의 실제 잔차 확인,
결측·미탐지·맑음의 구분, 비직교 격자 평활화 거부, 물리·격자·시각·출처 검증을 유지한다.
보존량은 정의된 에코 적분이며 수분 총량이나 강수량 자체가 아니다.
구조 정리는 입력 출처 검증 한 책임으로 제한하며 전체 계약 계층을 재설계하지 않는다.

## 검증 기록

### 결과

- S01–S03: 나이 0/60분의 배경값 동일성과 `nowcast()` 전달값, 평가영역 분모, 고정 A/B 점수, 빈 영역, 누락 화소와 선행시간 불일치를 시험했다.
- S04: 배치·비연속 입력의 81개 정수 이동에서 기존 식과 값·JVP·VJP가 정확히 일치했다. 4×4 FP32 장의 `(1000, 1000)` 이동은 0을 반환하며 저장공간은 64바이트다. 영역 밖에서도 일반 `backward()`가 0 기울기를 반환한다.
- S05: README와 화면에 에코 보존량, 조건부 불확실성, 비확률적 신뢰 지표, 전역 이동·성장 모형의 표현 한계를 명시했다.
- S06: 출처 payload 형식·시각·서명 검증을 `_input_derivation.py`로 옮겼다. 기존 검증과 AST가 동일함을 독립 리뷰에서 확인했다. 실행 입력과의 일치 검사는 기존 경계에 남겼다. 왕복 저장·복원, 서명 변조·비정규 시각·필드 누락 거부가 통과했다.
- S07: 독립 리뷰에서 찾은 A 재고정 실패 시 화면/내부 기준 불일치를 수정했다. 서버 성공 응답 후에만 A를 갱신한다. 수정 전 실패·수정 후 통과를 모의 DOM/네트워크로 확인했다.

### 실행 명령

```sh
.venv/bin/python -m pytest -q tests/test_initial_field_lab.py tests/test_shift_zero.py tests/test_matrix_free.py --tb=short
# 22 passed, 88 subtests passed.

.venv/bin/python -m pytest -q tests/test_nowcast.py tests/test_variational.py tests/test_run_artifact.py tests/test_numerical_review.py -k 'full_forecast_remaps or batched_remap or transport or long_lead_uses or stale_frozen or roundoff_tolerance_cannot or p1_run_preserves_grid_time_contract or final_linearization_tracks_remap_branch_changes or frozen_irls_matches_true_robust_gradient or analysis_can_cross_zero or public_trajectory_refreezes or v62_derivation_round_trip' --tb=short
# 15 passed, 352 deselected, 5 subtests passed.

.venv/bin/python -m pytest -q tests/test_initial_field_lab.py --tb=short
# 최종 서버 코드 정리 후 9 passed.

node --test tests/test_initial_field_lab_ui.cjs
# 2 passed: 재고정 실패 후 기존 A 유지, 계산 완료 결과로 A 지정 및 해제.
node --check examples/initial_field_lab/app.js
# PASS.

uvx --from basedpyright==1.39.9 basedpyright --level error --pythonpath .venv/bin/python src/advar/physics.py src/advar/nowcast.py src/advar/_digest.py src/advar/_input_derivation.py examples/initial_field_lab/server.py
# 0 errors, 0 warnings, 0 notes.
git diff --check
# PASS.
```

`server.py --port 8766`의 실제 HTTP 응답도 확인했다. 기본 실행, 동일 A/B,
강도 +6 dBZ의 B, 배경 없는 B, 잘못된 A/B 선행시간·reference 요청과 세 정적 파일을 확인했다.
기본 A의 고정 영역은 1,980화소, persistence MAE는 2.5873 dBZ였다.
강도 +6 dBZ B의 개선량은 −0.6691 dBZ였고, 배경 없는 B는 누락 240화소로 비교 불가였다.
이는 이번 구현의 합성 사례 실행값이며 사용자 제공 보고서의 전체 2,304화소 재현값과 다른 평가다.

### 한계

- Python 3.12.13, PyTorch 2.13.0, CPU 및 Node.js 24.13.1에서 검증했다. MPS/CUDA는 실행하지 않았다.
- 전체 baseline은 상위 `AGENTS.md`에 따라 반복하지 않았다. 위 37개 Python 시험과 2개 Node 시험은 영향 범위 검증이다.
- 브라우저 연결 도구가 `No browser is available`, `native pipe startup failed`로 실패했다. 실제 브라우저 렌더링·반응형 배치는 확인하지 못했으며, HTTP·모의 DOM 시험을 시각 검증으로 간주하지 않는다.
- 기존 TorchScript 폐기 예정 경고가 남아 있다.
- A를 새로 지정하거나 선행시간을 바꾸면 별도 비교다. 고정 A의 영역 밖에서 B가 추가로 덮은 화소는 A/B 점수에 포함하지 않는다.
- 구조 정리는 입력 출처 검증 한 책임에 한정했다. `nowcast.py` 전체의 계약·감사 코드 분리가 끝났다는 뜻은 아니다.
