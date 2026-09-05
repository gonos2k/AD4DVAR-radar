# 전수 읽기 범위

기준 커밋: `5097dbfd1ddf51483578bef5889ebc809ff024f0`.

GREEN / RED 리드 8개와 범위별 Luna 검토자 25개의 실제 읽기 기록을 합산했다.
아래 수치는 서로 겹치는 범위를 한 번만 센 합집합이다. 각 행을 두 팀 모두 또는
네 분야 모두에서 독립적으로 읽었다는 뜻은 아니다. 소스·테스트·도구·UI·설정
60개 파일, 162,670행에 미검토 구간이 없다. 읽기는 시험 실행이나 정확성 증명과 구분한다.
문서와 잠금 파일·내장 JSON 자료는 관련 의미 및 파서·해시·계약을 추가 확인했으며,
그 전체 내용의 수작업 검토를 이 표에 포함하지 않는다.

개별 읽기 범위와 보고서 매핑은 [재현 자료 묶음](review_artifacts/green_red_5097dbf.zip)의
`coverage_union.json`에 보존했다. 최종 판정은 [통합 체크리스트](FULL_CODE_REVIEW_CHECKLIST.md)를 따른다.

| 파일 | 전체 행 | 읽기 기록 |
| --- | ---: | ---: |
| `.github/scripts/build_deployment_bundle.py` | 2,089 | 2,089 |
| `.github/scripts/check_basedpyright.py` | 92 | 92 |
| `.github/scripts/check_dependency_locks.py` | 168 | 168 |
| `.github/scripts/generate_metric_domain_evidence.py` | 961 | 961 |
| `.github/scripts/run_real_case_acceptance.py` | 38 | 38 |
| `.github/workflows/ci.yml` | 282 | 282 |
| `.github/workflows/mps-certification.yml` | 35 | 35 |
| `benchmarks/benchmark_variational_fso.py` | 204 | 204 |
| `examples/initial_field_lab/app.js` | 253 | 253 |
| `examples/initial_field_lab/index.html` | 198 | 198 |
| `examples/initial_field_lab/server.py` | 528 | 528 |
| `examples/initial_field_lab/styles.css` | 549 | 549 |
| `pyproject.toml` | 28 | 28 |
| `src/advar/__init__.py` | 17 | 17 |
| `src/advar/_contract_registry.py` | 191 | 191 |
| `src/advar/_digest.py` | 46 | 46 |
| `src/advar/_input_derivation.py` | 188 | 188 |
| `src/advar/_learned_input.py` | 72 | 72 |
| `src/advar/_runtime.py` | 275 | 275 |
| `src/advar/acceptance.py` | 431 | 431 |
| `src/advar/action_artifacts.py` | 72 | 72 |
| `src/advar/action_contracts.py` | 54 | 54 |
| `src/advar/calibration.py` | 804 | 804 |
| `src/advar/cli.py` | 1,531 | 1,531 |
| `src/advar/diagnostics.py` | 135 | 135 |
| `src/advar/ensemble_sensitivity.py` | 954 | 954 |
| `src/advar/intervention.py` | 2,754 | 2,754 |
| `src/advar/ledger.py` | 22,296 | 22,296 |
| `src/advar/linearization_artifact.py` | 817 | 817 |
| `src/advar/matrix_free.py` | 388 | 388 |
| `src/advar/metrics.py` | 50 | 50 |
| `src/advar/nowcast.py` | 10,518 | 10,518 |
| `src/advar/physics.py` | 199 | 199 |
| `src/advar/promotion.py` | 27,989 | 27,989 |
| `src/advar/range_geometry.py` | 607 | 607 |
| `src/advar/run_artifact.py` | 2,557 | 2,557 |
| `src/advar/runtime_closure.py` | 518 | 518 |
| `src/advar/sensitivity.py` | 18,029 | 18,029 |
| `src/advar/variational.py` | 9,756 | 9,756 |
| `tests/test_acceptance.py` | 207 | 207 |
| `tests/test_calibration.py` | 435 | 435 |
| `tests/test_cli.py` | 1,521 | 1,521 |
| `tests/test_contract_registry.py` | 102 | 102 |
| `tests/test_deployment_bundle.py` | 999 | 999 |
| `tests/test_ensemble_sensitivity.py` | 335 | 335 |
| `tests/test_initial_field_lab.py` | 148 | 148 |
| `tests/test_initial_field_lab_ui.cjs` | 124 | 124 |
| `tests/test_ledger.py` | 5,909 | 5,909 |
| `tests/test_matrix_free.py` | 290 | 290 |
| `tests/test_metrics.py` | 35 | 35 |
| `tests/test_nowcast.py` | 6,352 | 6,352 |
| `tests/test_numerical_review.py` | 287 | 287 |
| `tests/test_pcg.py` | 308 | 308 |
| `tests/test_promotion.py` | 20,794 | 20,794 |
| `tests/test_run_artifact.py` | 3,025 | 3,025 |
| `tests/test_runtime.py` | 42 | 42 |
| `tests/test_runtime_closure.py` | 203 | 203 |
| `tests/test_sensitivity.py` | 8,206 | 8,206 |
| `tests/test_shift_zero.py` | 96 | 96 |
| `tests/test_variational.py` | 6,579 | 6,579 |
