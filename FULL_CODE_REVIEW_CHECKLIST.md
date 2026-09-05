# GREEN / RED 전수 코드 검토

## 판정과 범위

**CPU 수치 코어의 구조를 유지하면서, 확정한 R02–R11을 수정하고 영향 범위의 회귀시험을 통과했다.** R01인 full MPS P1은 기존 NO-GO를 유지한다. 아래 원인·재현은 최초 조사 시점의 기록이며, 이번 수정 결과는 별도 표에 연결했다. 추가 근거가 필요한 후보까지 해결했다고 주장하지 않는다.

- 기준: `5097dbfd1ddf51483578bef5889ebc809ff024f0`, 2026-09-05. 최초 전수 검토의 기준이며, 이번 후속 수정 이후의 전수 재검토를 뜻하지 않는다.
- 모델: `gpt-5.6-luna` 서브에이전트 33개. 수학·공학·수치해석·기상학의 GREEN / RED 리드 8개와 미검토 구간 담당 25개. 주 에이전트가 중복·반례·호출 경로를 대조하고 주요 재현을 다시 실행했다.
- 전수 읽기: 추적 소스·테스트·도구·UI·설정 **60개 파일, 162,670행**의 실제 읽기 기록 합집합에 미검토 구간이 없다. 각 행을 두 팀 모두 또는 네 분야 모두가 독립 검토했다는 뜻은 아니다.
- 문서·잠금 파일·내장 JSON 자료는 관련 의미와 파서·해시·계약을 확인했다. 그 전체 내용을 수작업으로 읽었다는 주장은 위 집계에 포함하지 않는다.
- 환경: 주 재현은 Python 3.12.13 / PyTorch 2.13.0, CPU 및 명시한 MPS 사례. 다른 인터프리터 결과는 개별 보고서에서 구분한다. CUDA, 전체 baseline, 실제 레이더 종단 간 성능, 실제 브라우저 렌더링은 검증하지 않았다.
- 범위: [파일별 읽기 기록](FULL_CODE_REVIEW_COVERAGE.md). 근거: [보고서·프로브·출력·해시 묶음](review_artifacts/green_red_5097dbf.zip). 원보고서와 판정이 다르면 **이 통합 판정이 우선**한다.

## 조사 완료 체크리스트

- [x] 기준 커밋과 파일 목록 고정.
- [x] 네 분야의 GREEN / RED 검토자 가동 및 실제 읽기 범위 합산.
- [x] 수학: 가정, 수식, 제약, 불변량, 고정 선형화와 미분 경로 확인.
- [x] 공학: 입력·출력 계약, 저장·복원, 자원 사용, 장치·자료형, 모듈 경계 확인.
- [x] 수치해석: 원점, 꼬리, 큰 값·작은 값, 정밀도, 잔차와 수렴 조건 확인.
- [x] 기상학: 보존량, 단위·시각, 탐지·결측, 이동 모형과 평가 해석 확인.
- [x] 후보를 호출 경로와 대조하고, 실제 결함·조건부 결함·모형 한계·오탐 구분.
- [x] 주요 결함의 작은 재현, 수정 방향 및 완료 조건 기록.
- [x] 검토 자료 보존 및 세션 말 KG 코드 그래프 갱신.

위 완료 표시는 **조사 완료**를 뜻한다. 아래 R02–R11의 완료 표시는 이번 패치와 명시한 회귀시험에 근거한다. 위치 링크와 원래 재현 수치는 최초 기준 커밋에 고정했다.

## 유지할 GREEN 근거

| 관점 | 확인한 구조 | 성립 조건과 해석의 경계 |
| --- | --- | --- |
| 수학 | 비음수 가중치의 정수 이동 결합, 경계 유출을 제외한 에코 적분 보존 | 보존량은 정의된 에코량이다. 수분 질량·강수량 보존을 뜻하지 않는다. |
| 수학·수치해석 | 고정 잔차 선형화의 `JᵀJ + λI`, 양의 damping, 실제 잔차로 PCG 수렴 재확인 | 고정 연산자의 SPD와 비선형 전역 최적성은 별개다. |
| 수치해석 | 원점 Taylor 분기, 작은 잔차 pseudo-Huber, Gaussian 꼬리 반사, 경계 띠 유출 합산 | R09의 극단 척도 계산을 보강했다. 장치별 AD 지원까지 자동 보장하지 않으며 R01은 별도 지원 경계다. |
| 공학 | 결측·탐지·미탐지 분리, 입력 출처 기록, 불변 snapshot과 현재 계약 검증 | 개별 해시의 형식 검사는 필드 사이의 정합성 검사를 대신하지 않는다. |
| 기상학 | 각 선행시간을 현재장에서 직접 수송하고 성장 누적에 감쇠 적용 | 전역 이동·성장 모형이다. 회전·변형·독립적인 새 대류 발생을 일반적으로 표현하지 않는다. |
| 실험 | 배경 age 메타데이터 설명, 고정 A 기준·영역, 평가 화소 수 표시 | 해당 비교의 질문에 맞는 점수다. 독립 실제 사례에서의 예측 skill을 대신하지 않는다. |

관측에서 만든 정규화 중심은 독립 사전정보가 아니며, 조건부 곡률 기반 불확실성을 무조건 베이지안 posterior로 읽으면 안 된다. `confidence` 역시 보정된 정답 확률이 아니다. 강수 외삽의 범위를 설명하는 외부 맥락은 [PySTEPS 원 논문](https://gmd.copernicus.org/articles/12/4185/2019/)과 일치하지만, 그 논문이 이 저장소의 정확성을 검증하는 것은 아니다.

## 확인된 후속 수정과 지원 경계 체크리스트

P2는 구체적인 계산·검증 결함, P3는 극단 입력·진단·문서 경계다. 같은 P2라도 일반 예측 오류와 비정상 artifact 수용은 구분한다. R01은 이미 문서화된 미지원 범위의 구체적 원인으로 따로 분류한다.

### R01 · 기존 지원 경계 · MPS P1 NO-GO의 구체적인 실패 경로

- 위치: [variational.py:8749](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/variational.py#L8749), 호출 [variational.py:8861](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/variational.py#L8861).
- 재현: 16×16 이동 Gaussian 에코의 3시각 입력, `maximum_outer_iterations=1`. CPU는 잔차 JVP가 유한하고 fallback 없이 목적함수 `80.623207 → 13.899658`로 감소한다. 같은 MPS 입력은 순전파가 유한하지만 `_bounded_update()`의 `scale * control`에서 `TypeError`가 발생하고 `solve_analysis()`도 같은 경로에서 실패한다.
- 국소 대조: MPS 0차원 dual과 Python 실수의 곱셈에서도 재현된다. 배경을 미분 대상으로 삼는 별도 helper 시험에서는 `background.new_tensor(1.0)`의 장치 dispatch 오류도 발생했다. 벡터 속도 제한 함수에 대한 이전 MPS 미분 통과와 구분해야 한다.
- 영향: 현재 환경의 P1 MPS 분석 경로가 실행되지 않는 구체적 호환성 문제다. [README:1717](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/README.md#L1717)은 이미 full MPS P1을 NO-GO로 명시한다. 따라서 지원 기능의 신규 회귀나 긴급 수치 코어 결함으로 세지 않는다. MPS P0와 Gauss–Newton 수식 자체의 오류도 아니다.
- 근거: `root_mps_p1.py`, `root_mps_p1.json`, `root_mps_scalar.json`.
- [ ] **향후 MPS P1 지원을 확대할 경우:** 스칼라 상수·연산의 dtype/device와 실제 P1 JVP 호출 경로를 함께 보강한다. 원점 극한과 도함수를 유지하고 위 CPU/MPS 분석 사례를 회귀시험으로 통과시킨다. 현재는 NO-GO 경계를 유지하며, 상수 한 개만 바꾼 것으로 전체 경로 해결을 선언하지 않는다.

### R02 · P2 · QC 개입에서 승인하지 않은 활성 화소 표준편차 변경 수용

- 위치: [intervention.py:2535](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/intervention.py#L2535)의 QC 전이 검증.
- 재현: 마스크·quality 변경으로 선언한 QC action에서 계속 유효한 화소의 관측 표준편차를 `2 → 9`로 바꾸어도 `RealizedInterventionReceipt.from_decision()`이 수용한다.
- 중요한 반례: 새로 QC 제외된 화소는 표준편차가 내부 값 `1`로 정규화될 수 있다. 이 정당한 전이도 전체 std digest를 바꾸므로, **무조건 digest 동일 조건을 추가하면 잘못된 패치**가 된다.
- 근거: `gap_20_probes.py`, `gap_20_probes.stdout`. 실제 action·run·receipt 전이에서 활성 std 변경과 정상 마스킹 정규화를 각각 재현했다.
- [x] QC 후 활성 영역의 std가 승인한 입력에서 유지되는지, 제외된 영역은 기존 정규화 규칙과 일치하는지 확인한다. 두 사례를 함께 회귀시험으로 고정한다.

### R03 · P2 · 자동 bounded UCB에 무관한 bootstrap 해상도 조건 적용

- 위치: [promotion.py:22261](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L22261), [promotion.py:22586](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L22586).
- 원인: 자동 metric-cell 경로는 bounded empirical-Bernstein UCB를 계산하지만, 그 전에 `metric_cell_tail_ok`를 요구한다. 이 UCB는 bootstrap 반복 수를 사용하지 않는다.
- 재현: 현재 typed policy/plan fixture, family size 16, 최소 꼬리 반복 20에서 bootstrap 1,024회는 꼬리 반복 1.6으로 거부된다. 동일한 1,000개 독립 event 값 `−0.5`에 대해 1,024회와 16,384회가 **같은 음의 UCB**를 계산하지만 꼬리 gate만 실패/통과로 바뀐다.
- 영향: 통계 방법과 무관한 보수적 거부 조건이다. 전체 서명·plan·holdout을 갖춘 promotion의 최종 false negative를 종단 간 재실행한 것은 아니다. 안전하지 않은 후보 승격을 입증한 것도 아니다.
- 근거: `gap_10_probes.py`, `gap_10_probes.stdout.json`, 독립 호출 경로 대조 `cross_stats.md`.
- [x] 선택된 추론 방법에 필요한 해상도 조건만 적용한다. 자동 bounded 경로에서 bootstrap 횟수에 대한 판정 불변성을 확인하고, 실제 bootstrap 경로의 해상도 조건은 보존한다.

### R04 · P2 · 1초 미만 간격의 물리적 가속도를 과소 계산

- 위치: [promotion.py:7406](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L7406), 중복 검증 [promotion.py:7507](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L7507).
- 원인: 인접 구간 속도의 차이를 실제 중점 시간차 대신 `max(1.0, midpoint_dt)`로 나눈다. 현재 계약은 시각 증가를 확인하지만 최소 1초 간격을 요구하지 않는다.
- 재현: 시각 `(0, 0.1, 0.2) s`, x 위치 `(0, 0, 0.01) m`. 속도 차이 `0.1 m/s`, 실제 가속도 `1 m/s²`지만 코드 값은 `0.1 m/s²`여서 상한 `0.25`를 통과한다.
- 영향: 허용된 짧은 간격에서 물리 제약이 느슨해진다. 통상 10분 간격 레이더 예측의 실패를 보여준 사례는 아니다.
- 근거: `gap_07_probes.py`, `gap_07_probes_stdout.json`; 별도 `gap_16.md` 대조.
- [x] 실제 구간 중점의 시간차로 나누거나 최소 시간 해상도를 입력 계약으로 명시한다. 두 검증 경로에 같은 물리식을 적용하고 단위 기반 반례를 시험한다.

### R05 · P2 · ledger가 범위 밖 trust와 무한 evidence를 저장·복원

- 위치: [ledger.py:22004](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/ledger.py#L22004), [ledger.py:22160](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/ledger.py#L22160).
- 재현: caller snapshot의 trust 성분과 곱을 `2.0`으로 설정하면 append → verify → load 후에도 `trust_score=2.0`이 남는다. `path_evidence_by_metric[0,0]=inf` 역시 NPZ 저장·검증·복원을 통과한다.
- 원인: 정상 생산자는 trust 성분을 `[0,1]`에 두지만 저장 경계는 유한성과 곱의 일치만 확인한다. evidence 범위 검사는 유한값만 추려 검사하므로 Inf가 검사 대상에서 빠진다.
- 범위: 비정상 caller snapshot 수용을 실제 왕복 재현했다. 정상 생산자가 이 값을 만든다는 주장은 아니다. evidence의 의도된 unavailable `NaN`까지 일괄 거부해서는 안 된다.
- 근거: `gap_05_probes.py`, `gap_05_probes.stdout.json`; 중복 후보 `gap_13.md`.
- [x] 기존 trust 성분·결과의 codomain과 evidence의 availability/유한성 관계를 저장 경계에서 확인한다. 정상 unavailable NaN은 유지하면서 Inf와 범위 밖 trust를 거부하는 왕복 시험을 추가한다.

### R06 · P2 · 실행 입력 identity의 필드 간 정합성 누락

- 위치: [nowcast.py:5154](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/nowcast.py#L5154), canonical bundle 계산 [nowcast.py:5508](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/nowcast.py#L5508).
- 재현 A: 정상 `ForecastRunContract`의 `input_bundle_digest`만 64개의 `0`으로 바꾸어도 `validate_integrity()`가 수용한다. 이미 저장된 구성 digest들로 canonical bundle을 다시 계산할 수 있다.
- 재현 B: 배경이 실제 사용된 현재 결과에서 `background_frames_digest`를 제거하고 관련 실행 digest를 재계산해도 `ForecastResult.validate_issuance()`가 수용한다. 재현 결과는 `background_used=True`, 배경 사용 비율 `0.5`다.
- 경계: A는 개별 run 검증 결함이며, 이전 result identity를 그대로 둔 전체 결과 검증은 별도로 오류를 잡을 수 있다. B의 수정도 현재 full-context digest가 있는 실행에 적용해야 한다. 전체 배경 digest가 없던 legacy v42 audit 복원은 의도된 호환성이다.
- 근거: `root_numeric_edges.py`, `gap_21.md`, `gap_24_probes.py`, `gap_24_probes.stdout`.
- [x] 현재 run에서 canonical bundle을 재구성해 비교한다. 현재 full-context 실행의 배경 필드 공존 조건을 검증하고 legacy audit 복원을 별도 회귀시험으로 보존한다. 새 registry나 중복 identity 타입을 만들 필요는 없다.

### R07 · P2 · 현재 입력 plan이 잘못된 schema·시각·source kind를 수용

- 위치: [promotion.py:2818](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L2818), [promotion.py:3255](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L3255).
- 재현: `NeuralPriorInputPlan`은 임의 또는 v1 contract와 1개·역순·중복 시각을 수용한다. v1 payload는 generic lineage 검증도 통과한 뒤 현재 resolver에서 `KeyError('issue_time')`가 난다.
- 별도 재현: `radar_source_kind='bogus'`가 plan을 통과하고 `OperationalIssuanceDomainArtifact.from_masks()`에서 non-mosaic 분기로 처리된다.
- 영향: 현재 typed 생성자/해석 경계의 결함이다. 정상 v2 plan으로 잘못된 예측을 생성하거나 전체 승격을 우회했다는 주장은 아니다.
- 근거: `gap_06_probes.py`, `gap_06_probes.stdout.txt`, 독립 `gap_18.md`.
- [x] 현재 plan의 schema, 지원 시각 개수·엄격한 시간 순서, 마지막 관측시각, 허용 source kind를 경계에서 검증한다. 역사적 payload 해석은 현재 입력과 명확히 구분한다.

### R08 · P2 · range 평가 artifact의 화소 수·면적 모순 수용

- 위치: [promotion.py:12806](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L12806)의 `RangeBandEvaluation`.
- 재현: 발행 가능 영역 1화소/면적 1보다 parent·candidate 발행 영역 2화소/면적 2가 큰 자료, 또는 metric 면적 2가 전체 공통 면적 1보다 큰 자료를 생성자가 수용한다.
- 범위: 정상 producer의 부분집합 관계는 확인했다. 모순된 typed/serialized 필드 수용이 입증됐으며, 완전한 서명·replay 검증의 우회가 입증된 것은 아니다.
- 근거: `gap_08_probes.py`, `gap_08_probes.stdout.json`.
- [x] 기존 필드 사이의 부분집합 수·면적 관계를 검증한다. 정상 producer와 모순 artifact를 각각 시험하며 새로운 평가 계층은 추가하지 않는다.

### R09 · P3 · 극단 척도에서 표현 가능한 목적함수·오차·확률 계산 실패

| 경로 | 현재 결과와 수학적 기대 | 근거 |
| --- | --- | --- |
| [matrix_free.py:182](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/matrix_free.py#L182) | FP32 잔차 4개가 각각 `1e19`이면 `0.5 * sum(r²)`가 Inf. FP64 대조값 약 `2.0e38`은 FP32 범위 안이고 gradient도 유한하다. | `root_numeric_edges.py` |
| [metrics.py:13](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/metrics.py#L13), [metrics.py:20](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/metrics.py#L20) | `±1e20`의 RMSE와 `±finfo.max`의 MAE가 Inf. 각각 참값은 `1e20`, `finfo.max`다. | `root_numeric_edges.py` |
| [promotion.py:16449](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L16449), [promotion.py:16817](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L16817) | location 0, scale `1e16`, reference 5, threshold 0, 분해능 0.5에서 양의 구간 확률이 로그 CDF 차의 반올림 때문에 0으로 계산되어 score가 거부된다. | `gap_09.md`, `root_numeric_edges.py` |

모두 수학적 경계 결함이지만 정상 레이더 값에서의 실패를 확인한 사례는 아니다. Gaussian scale은 현재 finite-positive 계약이 수용하므로 입력 허용 범위와 계산 가능 범위가 어긋난다. 이전의 location 45 / scale 1 / reference 5 꼬리 문제는 기준 커밋에서 NLL `794.6334302644476`으로 정상이며 새 결함으로 세지 않는다.

- [x] 합산·제곱 순서의 중간 overflow를 피하고, 확률 head에는 정당한 수치/물리 범위 또는 안정적인 좁은 구간 계산을 적용한다. 유한 출력 확인에 더해 높은 정밀도의 독립값과 비교한다. NaN/Inf를 0으로 바꾸거나 NLL을 임의 clamp하는 패치로 숨기지 않는다.

PyTorch도 유한한 최종값과 중간 연산의 overflow를 구분한다. [수치 정확도 문서](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#extremal-values)는 원리 설명에 사용했으며, 위 수치는 설치된 2.13.0에서 직접 확인한 결과다.

### R10 · P3 · 입력 경계와 문서의 작은 불일치

| 항목 | 확인 결과 | 완료 조건 |
| --- | --- | --- |
| CSI threshold | NaN/±Inf threshold에서 예측 40, 정답 0의 CSI가 1. finite threshold 35에서는 0. no-event CSI=1 자체는 기존 명시적 관례다. | 유한한 threshold를 경계에서 요구한다. |
| precision dtype | FP16/BF16 `PrecisionOperatorArtifact`가 CPU Cholesky `NotImplementedError`에 도달한다. FP32/64는 통과한다. | 지원 dtype을 조기에 명시하고 검사한다. |
| nowcast dtype/device | FP16 이동 에코는 CPU FFT unsupported dtype 오류. CPU 관측에 MPS quality/std를 붙이면 device mismatch 오류. | 허용 precision/device를 입력 경계에서 설명·검사한다. 배경장의 의도된 정규화는 보존한다. |
| legacy affine displacement | 정수 `[1,1]`, 간격 `1000.5/2000.5 m`가 `[1000,-2000]`으로 절삭된다. | 정수 입력을 거부하거나 명시한 실수 dtype으로 변환한다. 일반 forecast state의 실수 입력과 구분한다. |
| archive 설명 | README의 기본 member 상한 160과 코드의 256이 다르다. 161개 archive는 기본값으로 수용, 명시적 160에서는 거부된다. | 의도한 현재 상한으로 문서·구현을 일치시킨다. |

근거: `root_numeric_edges.py`, `green_numerics_probes.py`, `red_engineering.md`, `gap_22_fp16_probe.py`, `gap_21.md`, `gap_25.md`.

- [x] 위 입력 경계와 문서 항목을 각각 작은 변경으로 처리하고, 정상 지원 입력의 동작을 유지한다.

### R11 · P2/P3 · 비권위적 legacy 배포 도구의 오류

- **P2, 두 파일 loader:** [promotion.py:25088](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L25088), [promotion.py:25467](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L25467). 읽은 `block`을 `retained`에 추가하지 않아 비어 있지 않은 파일도 크기 변경으로 거부한다. 안정적인 descriptor metadata와 비어 있지 않은 read block을 사용하는 제한된 I/O 모의 재현 및 실제 코드 누락을 확인했다. 현재 수치 예측 경로의 장애를 뜻하지 않는다.
- **P3, sequence:** [promotion.py:25365](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L25365)의 독립 validator는 서명된 activation sequence `1.5`를 수용한다. 정상 ledger의 정수 head 비교는 별도 방어다.
- **P3, fallback 이유:** [promotion.py:26763](https://github.com/gonos2k/AD4DVAR-radar/blob/5097dbfd1ddf51483578bef5889ebc809ff024f0/src/advar/promotion.py#L26763)의 선행 branch 때문에 낮은 confidence가 `ambiguous_classifier_branch`로 기록된다. 선택 결과는 여전히 parent이므로 위험한 candidate 선택을 입증한 것은 아니다.
- **P3, 독립 signed reference 형식:** malformed digest/빈 label을 decoder helper와 standalone validator가 수용한다. 완전한 typed promotion 우회는 재현하지 않았다.
- 근거: `gap_11_probes.py`, `gap_11_probes.stdout.txt`, `gap_07_probes.py`.
- [x] 비권위적 legacy 범위를 유지하면서 read 누락, 정수 sequence, fallback 설명과 입력 형식을 보강한다. 이 작업을 수치 코어의 새 인증 계층으로 확대하지 않는다.

## 반영한 개선과 회귀시험

2026-09-05 최초 후속 수정(`965d230`)에서 **새 시험 48개와 영향받는 기존 시험 55개, 고유 시험 합계 103개가 통과했다.** 중복 재실행과 subtest는 합산하지 않는다. 소스 전체 basedpyright는 오류 0이며, 마지막 Boolean 검사 수정 후 해당 ledger 모듈도 오류 0을 확인했다. 정확한 선택자·환경·소스 해시는 [검증 기록](review_artifacts/followup_validation.json)에 보존했다.

| 항목 | 실제 반영 | 직접 검증 |
| --- | --- | --- |
| R02 | QC 후 활성 화소의 std는 유지하고 제외 화소의 기존 값 1 정규화는 허용 | [QC 회귀시험](tests/test_review_intervention.py), 3개 |
| R03 | analytic bounded metric-cell UCB에서만 bootstrap 해상도 gate 제거; 진단 metadata와 실제 bootstrap 경로 유지 | [통계 회귀시험](tests/test_review_promotion_statistics.py), 1개. 두 독립 manifest case로 실제 promotion 계산을 호출해 1,024/16,384회에서 동일한 bound 확인 |
| R04 | 생성자와 독립 validator 모두 실제 구간 중점 시간차로 가속도 계산 | [promotion 계약 시험](tests/test_review_promotion_contracts.py)에 정상 속도와 재해시한 짧은 간격 반례 포함 |
| R05 | trust 성분·점수의 [0,1] 범위와 Boolean 거부; evidence Inf 거부, unavailable NaN 유지 | [ledger 회귀시험](tests/test_review_ledger.py), 4개. NaN 왕복, 네 evidence 채널, 작은 범위 초과와 Boolean 포함 |
| R06 | 기존 canonical payload를 공유하는 digest helper로 bundle 재검증; current full-context의 배경 digest 공존 확인 | [입력 identity 시험](tests/test_review_run_identity.py), 6개 중 identity·v42 복원 사례와 기존 artifact 4개 선택 시험 |
| R07 | 현재 v2 plan, 3개 엄격 증가 시각, 마지막 관측시각과 source kind 검증 | R04/R08과 함께 promotion 계약 시험 19개 |
| R08 | 발행·철회·신규 화소의 집합 관계, domain 대비 면적, 공통 영역 대비 metric 면적 확인 | 정상 집합 차와 모순 화소 수·면적을 함께 검증. fallback은 입증된 domain 상한만 적용 |
| R09 | GN의 1/2을 합산 전에 적용; MAE/RMSE 오차를 안전하게 정규화; Gaussian 좁은 구간 적분 급수와 `expm1` 적용 | [수치 회귀시험](tests/test_review_numerics.py), 11개 중 높은 정밀도 독립값·1/2차 미분·영점·극단 척도, 기존 수치 시험 41개 |
| R10 | CSI threshold 유한성, precision/관측 FP32·64 및 관측 companion device 검사, 정수 affine 변위 거부; archive 기본 상한 256 문서화 | 수치·입력 identity 시험의 정상 입력 및 거부 사례, 기존 nowcast/precision 5개 |
| R11 | 두 legacy loader의 읽은 block 보존, 양의 정수 sequence, 낮은 confidence 이유와 signed reference 형식 검증 | [legacy 회귀시험](tests/test_review_legacy_loaders.py), 4개. 실제 signed payload 파일 왕복; 소유권·trust-store load·runtime closure만 모의 |

수치 수정 후 Luna 검토자가 R09의 값·미분과 R06의 현재/legacy 경계를 독립 대조했다. RED 재검토가 발견한 Boolean trust 수용도 마지막 패치·회귀시험에 포함했다. 좁은 구간 AD 시험은 구간 폭에 맞는 유한차분 간격과 cotangent 척도로 검증하며, artifact 변조 시험은 더 일찍 발생하는 `input bundle digest mismatch`를 기대하도록 갱신했다.

이 검증은 CPU 중심의 선택 시험이다. 관측 companion device 경계 시험은 MPS가 있는 환경에서 실행했지만, **전체 MPS P1·CUDA·전체 baseline·실제 레이더 성능·실제 브라우저 렌더링 통과를 의미하지 않는다.** MPS P1 NO-GO, 기존 결측 의미, 구형 audit 복원 및 핵심 수송·PCG 구조를 유지한다. 원보고서 ZIP은 최초 조사 자료로 보존했으며 수정된 소스의 결과로 바꿔 쓰지 않았다.

## 수정 후 재검토: CI 격리 실행과 evidence 공동 결측

- **확정·수정:** `tests/test_review_intervention.py`의 `from tests.test_ledger import ...`가 CI의 `python -I`에서 `ModuleNotFoundError`를 일으켰다. 마지막 로컬 103개 시험은 `PYTHONPATH=src:tests`를 사용했으므로 이 수집 오류를 잡지 못했다. [실패한 CI 실행](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/33948411069)의 Python 3.10·3.12에서 같은 원인을 확인했다.
- 기존 시험들의 방식인 `from test_ledger import ...`로 수정했다. 제품 수치 코드는 이 수정으로 바뀌지 않는다.
- 수정 전 로컬 `-I` 수집 실패를 재현했고, 수정 후 같은 격리 모드에서 **901개 시험 수집 성공**, QC 영향 시험 **3개 통과**를 확인했다. 수집은 전체 baseline 실행이 아니며 이 3개는 앞선 103개와 중복이다.

- **확정·수정:** 실제 producer의 유한 evidence 네 채널 중 하나만 NaN으로 바꾼 mixed 상태가 저장·검증·복원을 통과했다. `_metric_evidence_ratios()`는 같은 분모로 네 값을 함께 계산하므로 네 채널의 per-cell 유한값 마스크를 동일하게 검증하고, 그 마스크로 source evidence 합도 확인하도록 보강했다.
- **오탐 철회:** `metric_available=True`면 evidence가 무조건 유한해야 한다는 지적은 잘못이다. 예측과 정답이 같은 실제 producer는 점수가 0이고 민감도 가중치의 분모가 0이므로 네 채널 모두 NaN이 정상이다. 처음 NaN을 다시 대입한 probe는 no-op이었다. 실제 `truth=20.5 dBZ`로 유한한 네 채널을 생성한 뒤 일부만 NaN으로 바꿔 위 mixed 결함을 별도로 재현했다.
- 추가한 두 회귀시험까지 포함한 최종 격리 수집은 **903개 성공**이다. 추가한 두 회귀시험과 기존 ledger 시험을 `python -I`에서 실행해 **6개 통과(13 subtests)**를 확인했다. 정상 finite 왕복, available/all-NaN 왕복, unavailable/all-NaN 왕복과 네 mixed 채널 거부를 함께 확인했다. 현재 typed append 검증의 보강이며 legacy loader의 해석은 변경하지 않았다. 해당 모듈 basedpyright는 오류 0이다.
- Luna 재검토 결과, R03/R04/R07/R08과 GN·MAE/RMSE·좁은 Gaussian 구간의 수정에서는 새 확정 결함이나 정상 입력 과잉 거부를 찾지 못했다. 수치 시험 11개와 계약 시험 19개를 재확인했다. R03의 97.68초 결과는 앞선 실행 근거를 재사용했으며 이번 신규 실행으로 세지 않는다.
- 이번 검증과 최종 소스 해시는 [최종 재검토 기록](review_artifacts/final_audit_validation.json)에 보존한다. 앞선 103개와 이번 중복 재실행을 합산해 새로운 전체 baseline 통과로 보고하지 않는다.

## 추가 근거가 필요한 후보

다음 항목은 일부 서브에이전트 원보고서의 심각도보다 낮춰 보류했다. 격리 재현을 현재 전체 경로의 실패로 확대하지 않는다.

| 후보 | 현재 근거와 부족한 확인 | 다음 확인 |
| --- | --- | --- |
| coverage deadline | DB insert 이전에 시간을 읽는 코드와 지연 insert 격리 재현. artifact validator를 대체한 probe이며, 정상 서명 입력의 최종 commit 시각까지 대조하지 않았다. | 유효 artifact로 deadline의 의미가 호출/확정 중 무엇인지 확인하고 commit 시각을 계측. `gap_02.md` |
| 동시 resume | SELECT 후 무조건 INSERT인 legacy 경로와 두 SQLite connection의 unique 충돌 재현. 전체 공개 API의 서명·트랜잭션 경로를 동시에 실행하지 않았다. | 실제 두 요청에서 idempotent resume 계약과 외부 직렬화 확인. `gap_04.md` |
| ledger index/manifest identity | immutable trigger를 제거하고 checksum까지 바꾼 저장소 변조에서 메타데이터 모순을 수용. receipt key 사례는 signature/artifact 검사를 stub했다. | 현재의 완전한 서명 fixture로 별도 저장소 손상 검증. 일반 append 또는 정상 예측 오류로 분류하지 않음. `gap_12.md` |
| native closure·만료 receipt·audit coverage | legacy 도구의 mock/Linux 분기와 독립 validator 위주 재현. 실제 Linux bundle 전체·현재 권위 경로 확인은 없음. | 실제 dependency closure와 current/audit 시간 계약 확인. `gap_19.md` |
| whitened norm | caller가 만든 잘못된 진단값의 validator-only 수용. | 이미 선택한 기존 direct map과의 관계를 비교할 수 있는지 확인. 전체 map 복제·새 registry는 피함. `gap_05.md` |
| no-op QC decision | changed pixel 0인 decision 수용 확인. realized receipt와의 의미적 모순은 추가 대조 필요. | 명시적 no-op 허용 여부를 정하고 양쪽 계약을 맞춤. `gap_20.md` |
| 독립 시험 oracle | 일부 semantic replay 시험이 동일 함수를 기대값 생성에도 사용. 다른 수학 시험은 존재함. | metric 식의 독립 oracle이 필요한 핵심 사례만 보강. 전면 테스트 재작성 근거 아님. `gap_14.md`, `gap_15.md`, `gap_17.md`, `gap_23.md` |

## 철회하거나 결함으로 세지 않은 지적

- Python 3.10 annotation import 실패는 3.9 pytest launcher 혼용이었다. 3.10/3.12의 직접 import는 정상이다.
- low-rank whitener의 반복 고유값에서 helper 미분이 NaN인 사례는 현재 caller가 해당 factor를 고정·detach한다. 도달 가능한 현재 미분 결함으로 세지 않는다. 작은 계수의 상대오차도 전체 whitening operator 영향은 정밀도 수준이었다.
- publication의 `abs(value-threshold)`는 고정 active set에서 포함·제외 양쪽의 경계 거리다. signed feasibility로 바꾸면 안정적인 제외 화소를 잘못 거부할 수 있다.
- source 0만 도달한 `BLENDED`는 이전/최근 두 이동 구간을 모두 거친 경로를 나타낸다. 단순 대칭 패치로 `EARLIER`로 바꾸면 경로 의미가 훼손된다.
- 배경 NaN·dtype·device 수용은 결측과 내부 변환을 지원하는 의도된 경로다. 원본 digest와 정규화된 계산장의 차이도 구분한다.
- zero-weight calibration count 반례는 private helper 직접 호출이었다. 현재 producer의 valid mask는 `quality > 0`을 포함하므로 현재 점수 경로의 표본 수 부풀림으로 세지 않는다.
- shadow의 0분산 점 bound는 비배포 진단 모드이고 자동 경로는 bounded inference를 쓴다. 현재 배포 승격 우회를 재현한 것이 아니다.
- P1 탐지 경계와 검증기의 strict threshold, no-event CSI=1은 서로의 계약을 먼저 확인해야 한다. 기호 차이나 관례만으로 기상학적 버그를 단정하지 않는다.
- scoring input의 plan binding 의심은 start receipt 및 promotion/load의 재검증을 확인한 뒤 철회했다.

## 최초 조사 검증 기록과 후속 사용 순서

1. 현재 코어의 선택 시험은 `pytest_core.stdout.txt`의 41개, 계약 44개, variational subset 27개, sensitivity subset 19개 등이 통과했다. 다른 검토자 실행과 중복되므로 합산하여 고유 테스트 총수로 보고하지 않는다.
2. 일부 초반 module 실행은 범위 제한 전에 시작되었다. promotion 실행 하나는 20개 통과 후 696초에서 중단했으며, 별도 fixture 대기 실행도 중단했다. 완료되지 않은 실행은 성공으로 세지 않는다. 전체 baseline은 완료 실행하지 않았다.
3. MPS는 R01의 작은 실제 P1 경로에서 실패가 확인됐다. CPU 통과를 MPS/CUDA 통과로 바꾸어 보고하지 않는다. UI 근거는 Node 제어 시험·HTTP·정적 검토이며 실제 브라우저 렌더링이 아니다.
4. 재현 묶음의 `README.md`, `SHA256SUMS`, `manifest.json`, `coverage_union.json`으로 환경·파일·읽기 범위를 확인한다. 프로브는 기준 커밋의 소스를 `PYTHONPATH=src:tests`로 실행한다. 전체 probe를 CI 시험처럼 일괄 실행하는 묶음은 아니다.
5. R02–R11은 위 개선 표의 패치와 선택 시험으로 해소했다. 다음 조사는 추가 근거가 필요한 후보의 실제 호출 경로를 확인하는 데서 이어간다. R01은 향후 MPS P1 지원 확대 시 별도로 처리한다. 새 registry나 인증 계층을 추가하지 않았다.

KG는 기존 코드 그래프를 갱신한다. graph-only 저장소이므로 이 문서의 의미가 자동으로 지식 노드에 추출됐다고 주장하지 않는다. 세션의 판단·반례는 이 추적 문서와 재현 묶음에 보존하며, 다음 검토는 지원 경계와 보류 후보를 구분해 이어간다.
