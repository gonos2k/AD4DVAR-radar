# 두 번째 전체 코드베이스 추가 조사

기준 커밋: `94c75d3323c986b6fb2963336b906acc85384e12`.

**조사는 완료했으며, 아래 새 제품 결함은 미수정 상태다.** 이번 변경은 팀 모델 설정, 조사 기록, 기존 수정에서 빠뜨린 CI fixture 보정이다. 수치 알고리즘을 다시 설계하거나 새 인증 계층을 추가하지 않았다.

- [x] 팀 34개를 `gpt-5.6-luna`(`lunar`) / `xhigh`로 가동하고 AGENTS.md에 기록.
- [x] 코드·시험·UI·CI·설정·lock 71개 파일, 166,618행의 Git blob SHA-256 고정.
- [x] 각 담당 구간의 실제 읽기와 전체 합집합 검증. 미검토 행 0.
- [x] 후보의 공개 호출·입력·기대값·상위 검사 경로를 대조하고 주요 항목 교차 검증.
- [x] 확정 결함, 제한된 입력 경계 문제, 보류·철회 후보 구분.
- [x] CI에서 확인한 fixture 실패 8개 보정 및 해당 시험 통과 확인.
- [x] 원보고서·재현 코드·출력·해시 보존.
- [ ] 아래 새 제품 결함의 수정 및 영향 시험. 조사 완료를 수정 완료로 세지 않는다.

[파일별 읽기 범위](SECOND_CODE_REVIEW_COVERAGE.md) · [원보고서·프로브·출력 묶음](review_artifacts/team_audit_94c75d3.zip) · [수정 후 검증 기록](review_artifacts/second_audit_validation.json) · [Draft PR #156](https://github.com/gonos2k/AD4DVAR-radar/pull/156).

각 행을 GREEN·RED 양쪽이나 네 분야 모두가 독립 읽었다는 뜻은 아니다. 문서 전체와 내장 JSON 자료 전체의 수작업 읽기는 집계에 포함하지 않는다. 내장 지리 자료는 생성기 소스 바인딩·격자 시험으로 확인했다. 원보고서의 EOF 초과 끝 행은 실제 마지막 행에서 잘랐고, 중복 읽기는 한 번만 셌다.

## 우선 해결할 수학·수치·기상학 문제

모든 항목은 **OPEN / 미수정**이다. P1은 과학 계산의 주요 오류, P2는 특정 지원 경로의 잘못된 결과·실패, P3는 제한된 자료 경계·진단·도구 문제를 뜻한다. 심각도와 확정 범위는 이 통합 판정을 우선한다.

| ID | 판정 | 재현한 문제와 영향 범위 | 가장 작은 수정 방향 |
| --- | --- | --- | --- |
| A2-01 | P1 | [M0](src/advar/sensitivity.py#L12610), [P1 FSO/FSOI](src/advar/sensitivity.py#L15296), [learning step](src/advar/sensitivity.py#L14827)이 typed verification의 FSO 가중치를 누락한다. v22에서 제외 화소가 점수·미분·지지영역에 포함된다. | 예측 평가영역 × 검증 가중치를 한 번 구성해 support, 점수, 미분, digest, 학습 재계산에 일관되게 사용. |
| A2-02 | P2 | [이벤트 연관 판정](src/advar/promotion.py#L8192)이 겹치는 시간 구간에서 인자 순서에 의존한다. 같은 signed event 쌍을 A/B로 넣으면 연관, B/A면 비연관이다. 공개 catalog에서 한 순서만 분할이 통과한다. | 같은 시각의 궤적 비교와 대칭인 연관 규칙을 적용. 어느 순서가 참이어야 하는지는 명시한 기상학적 기준으로 결정. |
| A2-03 | P2 | [분기 검사](src/advar/sensitivity.py#L16869)가 0.5·1.0에서만 같은 FFT peak를 확인하고 `certified`를 반환한다. 실제 0.25에서 peak가 바뀌었다가 돌아오는 반례가 있다. | 유한 표본 검사를 전 구간 보장으로 표현하지 말 것. 근거 없는 `certified`를 제한하고 기존 full/half 재해석 검사는 유지. |
| A2-04 | P2 | [동시 불확실성 UCB](src/advar/promotion.py#L21140)의 automatic bounded 방법에도 [사용하지 않은 bootstrap tail gate](src/advar/promotion.py#L22895)가 남아 있다. 표본 횟수만 바꾸면 같은 상한에서 추가 거부 사유가 바뀐다. | 실제 선택한 추론 방법에만 tail gate 적용. 기존 R03의 metric-cell 수정과 별도로 이 경로 정리. |
| A2-05 | P2 | [현재 run 생성](src/advar/nowcast.py#L4852)·[artifact load](src/advar/run_artifact.py#L1434)가 증거 없는 v5 격자를 받아 current v72로 왕복시킨다. registry는 v5를 audit-only로 명시한다. | 역사적 v5 decoding은 유지하고 현재 issuance/save의 과학적 격자 허용 경계를 명확히 강제. |

A2-01의 current v22 공개 재현은 다음과 같다. M0는 가중치 0인 미탐지 한 셀 때문에 기대 점수 0 대신 `2.9823176871`, 해당 셀 미분 `0.0086346941`을 반환했다. P1은 평가 가중치 합 48 대신 49, 기대 점수 약 `5.10e-31` 대신 `0.0821876262`를 반환했고 자체 artifact 검증도 통과했다. 별도의 공간 나이 조건에서는 선언된 지지가 0인데도 점수를 계산했다. observation-removal의 `_resolved_forecast_scores()`는 가중치를 적용한다. 처음 나온 “P1은 정상” 판정은 이 다른 호출 경로를 혼동한 것이어서 정정했다. 공개 M0/P1 재현에는 수치 함수나 validator mock이 없다. 학습 단계의 별도 재현은 private helper와 최소 정책 shim을 사용했으며, 자동학습 전체 승인을 실행했다고 주장하지 않는다. ZIP의 `cross_fso.md`와 관련 probe를 참조한다.

A2-03은 raw Tensor 검증자료를 쓰는 exploratory FSOI이며 비기본 perturbation 면적 한도를 사용했다. 30개 화소, 실제 최대 변화 `0.3136172199 dBZ`다. 한 lead·한 metric의 같은 49셀 평가영역에서 1차 impact 약 `-3056.3544`, 재해석 변화 약 `+8.35e-05`였다. 이 차이를 내부 peak 변경 하나만의 인과 효과라고 단정하지 않는다. 기본 면적 한도는 이 사례를 거부하고, 별도 full/half 1차 검증도 학습을 거부하며, raw Tensor는 자동학습 lineage gate를 통과하지 못한다. **자동학습 승인 우회가 아니라 분기 상태의 과도한 보장**이다.

A2-04 공개 fixture에서 bootstrap 64/16,384 모두 상한 1.0과 `support_bounded_hybrid`였고, 작은 횟수에만 `insufficient_bootstrap_tail_resolution`이 추가됐다. 다른 거부 이유 때문에 최종 eligible은 두 경우 모두 false다. 최종 승인 여부가 뒤집혔다고 주장하지 않는다.

A2-05는 v5 helper의 역사적 projected 의미 자체를 결함으로 보지 않는다. 1,000 m 셀, 600초, 10 m/s에서 v5는 6 px를 허용하지만 v6의 지면 척도 상한을 적용한 한도는 약 5.964 px다. current verification geometry는 v5를 올바르게 거부한다. 현재 run/artifact 허용과의 차이가 문제다.

## 저장·실행 경로의 확정 문제

아래도 모두 **OPEN / 미수정**이다. 합성 자료라도 공개 생성·호출 경로로 재현한 경우와 malformed typed 입력을 구분했다.

| ID | 판정 | 위치·재현 | 수정 방향 / 한계 |
| --- | --- | --- | --- |
| A2-06 | P2 | [run age 직렬화](src/advar/run_artifact.py#L520): `background_age_minutes=10`은 nowcast/save 성공 뒤 load에서 정수 dtype 때문에 거부된다. | 허용한 분 단위 수치를 float로 정규화. 정상적인 공개 입력 왕복 문제. |
| A2-07 | P2 | [legacy migration](src/advar/run_artifact.py#L1957): 같은 NaN 위치를 가진 두 예측에 `torch.equal`을 써 v42 복원이 실패한다. | 유효값과 NaN 위치를 함께 비교. 현재 생성물을 v42로 재봉인한 fixture이며 실제 과거 binary 표본은 없다. |
| A2-08 | P2 | [prospective decision](src/advar/intervention.py#L1824)은 no-op dBZ/QC/override를 채택하지만 [receipt](src/advar/intervention.py#L2579)는 변화가 없다고 거부한다. | decision과 receipt의 변화 조건 일치. QC 공개 ledger probe는 trust-store loader만 fixture로 대체. 물리적으로 위험한 조작을 재현한 것은 아니다. |
| A2-09 | P2 | [실현 receipt 저장](src/advar/ledger.py#L6380)이 디렉터리를 먼저 공개하고 중복 decision의 SQLite INSERT가 실패하면 orphan 디렉터리를 남긴다. | 중복 확인과 새 디렉터리 rollback 정리. 공개 signed receipt fixture에서 2개 디렉터리/1개 row 재현; loader가 orphan을 승인한 것은 아니다. |
| A2-10 | P2 | [source coverage 등록](src/advar/ledger.py#L7827)이 preflight에서만 시간을 확인한다. 연결 지연으로 deadline 뒤 INSERT해도 성공한다. | write transaction에서 시각 재확인. 주입 clock으로 지연을 모델링했으며 실제 지연 시간을 측정한 시험은 아니다. |
| A2-11 | P2 | [idempotent provenance recovery](src/advar/ledger.py#L10411)를 두 worker가 동시에 실행하면 하나는 성공, 하나는 `activation raced` 실패다. | 조건부 갱신 경쟁 후 동일 artifact가 이미 active면 성공 반환. 최종 row는 active/usable이며 손상·유실은 없었다. |
| A2-12 | P2 | [M0 append 검사](src/advar/ledger.py#L21867)는 임의 context 이름을 허용하지만 자체 [verify](src/advar/ledger.py#L21732)는 거부한다. | 저장 전 현재 context schema 일치 확인. 정상 producer가 아닌 외부 typed snapshot의 입력 경계. |
| A2-13 | P3 | [tile impact 검사](src/advar/ledger.py#L22153)가 unavailable metric의 finite tile impact `17.0`을 저장·복원한다. | unavailable whole-field/tile missingness 일치. 현재 직접 소비자는 ledger 외에 찾지 못했다. |
| A2-14 | P3 | [whitened tile norm 검사](src/advar/ledger.py#L22056)가 보존한 direct map=3, std=1인데 norm=999를 허용한다. | 실제 보존한 map이 있는 lead만 재계산 대조. 외부 malformed 진단값이며 정상 계산 오류로 확대하지 않는다. |
| A2-15 | P3 | [M0 공간 검사](src/advar/ledger.py#L21853)가 모두 unavailable인 0×0 snapshot을 저장한다. | 두 공간 차원의 양수 조건. 정상 producer는 빈 입력을 거부한다. |
| A2-16 | P3 | [P1 내용 검증](src/advar/variational.py#L9694)이 control과 맞지 않는 재봉인 remap cell을 허용한다. 공개 save/load에서 값은 같지만 dynamics Jacobian 차이 약 199.8. | 기존 `_analysis_remap_cells_match` 관계 검사. digest까지 다시 만든 malformed artifact이며 정상 solver 출력 결함은 아니다. |
| A2-17 | P3 | [learning 승인 대조](src/advar/sensitivity.py#L12210)가 v4 `nominal_full_analysis_input_digest`를 validation과 비교하지 않는다. | 기존 expected 필드 집합에 같은 lineage 관계 반영. 실제 FSO/FSOI 위에 합성 validation/evidence를 재구성한 validator 경계 probe다. |
| A2-18 | P3 | [canonical observations 검사](src/advar/variational.py#L8972)가 직접 구성한 NaN dbz를 residual까지 허용한다. | typed canonical 관측의 유한값 검사. raw 입력 정리는 정상이고 solve는 nonfinite objective로 fallback한다. |
| A2-19 | P3 | [remap cell 변환](src/advar/physics.py#L136)이 유한한 `1e20` px에서 Python int 변환 overflow를 낸다. | 영역 밖 이동의 명시적 처리 또는 지원 범위 거부. configured nowcast에서 도달한 사례는 아니다. |
| A2-20 | P3 | [range evidence](src/advar/range_geometry.py#L353)가 중복 label을 받아 `mask(label)`이 첫 mask만 반환한다. | label의 비어 있지 않음·유일성 확인. 생성기와 authoritative replay는 추가 검사를 한다. |
| A2-21 | P2 | [research CLI](src/advar/cli.py#L1039)가 격자 없이 m/s pair threshold를 받아 실제로는 px 기준을 사용한다. | 이 물리 단위 옵션에도 격자 요구. 같은 입력·0.001 m/s에서 격자 유무에 따라 conflict false/true 재현. |
| A2-22 | P3 | [CLI preflight](src/advar/cli.py#L1038)에 물리 pair 옵션 3개가 빠져 argparse 오류 대신 uncaught ValueError가 발생한다. | 기존 physical-option 집합 완성. 실행은 fail-closed이며 잘못된 예측을 쓰지 않는다. |
| A2-23 | P3 | [legacy bundle build](.github/scripts/build_deployment_bundle.py#L829)는 Linux ARM을 받아 만들지만 [verify](.github/scripts/build_deployment_bundle.py#L1009)는 x86만 허용한다. | 지원 architecture를 build 시작에서 맞춤. platform mock을 이용한 legacy 도구 시험이며 ARM 실기기 배포는 하지 않았다. |

## 유지할 GREEN

- 비음수 수송과 국소 경계 유출은 독립 index oracle과 보존 예산으로 일치했다. 정수 이동의 crop-first 메모리 개선도 유지한다.
- JVP–VJP 내적, 명시 Jacobian/HVP, 실제 잔차를 확인하는 PCG와 극단 척도 시험이 양호했다. 선택 수치 시험 83개/131 subtest가 통과했다. 이 수를 다른 담당자의 중복 시험과 합산하지 않는다.
- bounded velocity의 원점 1·2차 미분, 큰 유한 제어, 작은 pseudo-Huber 비용, Gaussian 꼬리·좁은 구간은 선택 시험과 독립 고정밀 값으로 확인했다.
- event 단위 가중 평균과 bounded UCB의 보수성은 조건부로 타당하다. Maurer–Pontil의 empirical-Bernstein 식과 현재 반경을 수학적으로 대조한 근거는 ZIP의 `root_statistical_bound.md`에 있다. 실제 기상 이벤트의 독립성을 증명했다는 뜻은 아니다.
- 초기장 age는 메타데이터, A/B는 고정 기준예측·영역, 실제 평가 화소 수는 별도 표시라는 수정이 유지된다. Python 10개/10 subtest, Node 2개와 별도 probe가 통과했다.
- 현재 geometry 생성기 source binding, 4개 hashed CPU dependency closure, 설치 CLI·wheel 경로가 양호했다. full metric report의 환경 closure 재현은 이 호스트와 달라 전체 재생 성공으로 세지 않는다.

## 보류·철회와 시험의 한계

| 후보 | 통합 판정 |
| --- | --- |
| 관측과 frozen state의 digest를 반드시 같게 묶어야 한다 | **철회.** fixed-background perturb-and-resolve가 의도적으로 관측을 바꾸어 같은 frozen state와 함께 푼다. 무조건 identity를 묶으면 올바른 민감도 경로를 깨뜨린다. |
| P1 envelope의 no-device digest와 state의 device digest가 다르다 | **명명/설명 후보로 하향.** 공개 CPU save/load는 성공하고 두 scope의 검사는 별도다. 실제 replay 모순을 찾지 못했다. |
| 모든 raw receipt가 Missing이면 replay 저장이 실패한다 | **확정에서 제외.** current source coverage 생성자가 empty 상태를 먼저 거부한다. zero-row helper 직접 주입은 공개 archive 재현이 아니다. |
| scoring replay 공개 뒤 crash가 orphan을 남긴다 | **후보.** 실제 private durable publish의 child-process crash와 소스 순서는 확인했지만, 완전한 typed public append/retry 재현은 하지 않았다. A2-09와 구분한다. |
| 같은 input digest가 두 이벤트에 있으므로 반드시 통계 중복이다 | **현재 scorer 결함 주장은 철회.** metadata 생성은 허용하지만 distinct input plans와 실제 ForecastRunContract 재계산이 valid scorer 경로의 중복을 막는다. 한 분석장에 복수 강수계가 있다는 사실도 독립 표본 수 오류의 직접 증거는 아니다. |
| uncertainty/state target plan 중복 | **미확정.** target 시간·정체성·서명과 실제 scorer 도달을 추가 확인해야 한다. |
| `_interval_square` 음수 구간에서 1 ULP inward | **helper 후보.** 전체 2차원 outward range 계산 1,020개 반례 탐색에서는 enclosure 실패가 없었다. |
| sample-size preflight의 minimum_cases 생략 | **제한된 진단 후보.** 해당 함수는 event 수 입력만 받고 실제 promotion은 qualifying cases를 별도로 거부한다. 배포 우회로 세지 않는다. |
| control NaN/Inf, 변경 std의 stale whitener, threshold 동등점 `>=`/`>` | **입력·명세 후보.** 정상 생성자·고정 가중치·solver gate와 구분하여 적용 범위를 결정해야 한다. |
| 위조 ranking, mutable range geometry, index/manifest 중복 메타데이터 | **변조/재구성 후보.** ordinary producer 또는 상위 replay를 통한 실패는 추가 입증이 필요하다. |
| complex CLI 입력, empty EFSO observation ID, raw verification 파일 자원 상한 | **경계 보강 후보.** 실제 사용 범위를 확인하고 작은 validation으로 해결할지 결정한다. |
| trust-store issuer의 입력 정렬, module action bounds, source reservation helper, metric support shape, 비정렬 lead | **호출 경계 후보.** 상위 정규화·검사·의도한 유연성을 대조해야 하며 새 확정 forecast 결함으로 세지 않는다. |

일부 operational geometry 시험은 현재의 unconditional deployment-unsupported 오류만 확인하므로 geometry 세부 검증의 증거로 사용할 수 없다. prior runner의 mutable-state 시험도 “거부”보다 frozen export 유지에 관한 시험이다. 각각 앞으로 기능이 바뀔 때 oracle·명칭을 정리할 항목이다. 기본 불변량·외부 계약을 검증하는 시험과 구현 세부를 그대로 반복하는 시험을 구분한다.

## 이번에 수정한 누락과 검증

- [x] `AGENTS.md`: 팀 모델 `gpt-5.6-luna` / `xhigh` 기록.
- [x] evidence closure fixture: source fraction도 finite로 만들어 원래의 합계 오류까지 검사. 정상 all-NaN evidence와 부분 결측 거부 유지.
- [x] 7개 시간 관련 fixture: 마지막 관측 시각을 유지한 세 입력 시각. holdout-clock fixture의 raw slots·sampling unit·서명 예약도 같은 세 시각으로 맞추고 최초 clock은 첫 raw slot보다 앞에 둠.
- [x] CI에서 실패한 8개 시험이 모두 수정 후 로컬 `python -I`에서 통과. 인접 evidence 시험 6개도 통과해 **고유 14개** 확인. 중간 실패와 최종 성공 로그를 함께 보존.
- [x] `git diff --check` 및 기준 운영 소스 불변 확인. 제품 코드는 `94c75d3`와 같다.

기준 커밋의 CI run `33949634139` Python 3.10·3.12는 각각 **887 passed, 8 failed, 8 skipped, 638 subtests passed**로 완료됐다. 두 버전 모두 같은 8개 fixture가 실패했다. Python 3.12 source type 검사와 wheel/CLI·UI job은 성공했다. 수정 커밋의 원격 전체 CPU CI 결과는 별도로 확인해야 하며, 로컬 선택 시험을 전체 CI 성공으로 표현하지 않는다.

전수 읽기와 선택 시험을 수행했지만 로컬 전체 baseline은 반복 실행하지 않았다. 실제 레이더 예측 성능, 전체 P0/P1 실자료 종단 간 평가, CUDA, 전체 MPS P1, 실제 브라우저 렌더링은 이번 검증 범위가 아니다. full MPS P1의 기존 NO-GO는 유지한다. 앞선 `FULL_CODE_REVIEW_*`와 ZIP은 역사적 기록 그대로 보존한다.

세션 KG는 기존 graph-only 저장소의 구조 그래프와 `graphify-out/REVIEW_INDEX.md`를 갱신한다. 위 의미 판정은 이 체크리스트와 원보고서에 보존하며, AST 그래프가 수학적 타당성을 자동 인증한다고 주장하지 않는다.
