# A5 조사 해소 기록

조사 기준은 `286c0c9416a645ec4350b24ab9e9faca6584bb56`이다. 원래의
[조사 체크리스트](FIFTH_CODE_REVIEW_CHECKLIST.md)와
[조사 자료](review_artifacts/a5_full_review.json)는 수정 전 증거를 보존한다.
이 문서는 후속 수정과 검증을 기록한다. A5-01–12와 확정된 추가 입력·저장 경계를
해소했다. 원 조사 의견 70개에는 중복·시험 공백·철회한 주장도 포함된다. 이들을
각각 분류한 [해소 목록](review_artifacts/a5_resolution.json)을 함께 보존한다.

## 수식과 실제 계산 경로의 일치

| ID | 조치 | 검증 근거 |
| --- | --- | --- |
| A5-01 | 현재 v15 source 입력에도 기존 공간·시간 품질 및 오차 구성을 적용한다. | 현재 v22 생성 결과를 별도 수식으로 계산한 quality/std/역분산 가중치와 비교한다. |
| A5-02 | 관측 제거 시 같은 prior 모델을 변경 QC에 재추론한다. 비자명한 source mask는 명목 digest와 대조한다. | 공개 제거 함수의 점수를 같은 모델로 독립 재분석한 결과와 비교한다. |
| A5-03 | 부분 prior의 미제공 영역에서 관측 초기장으로 돌아가는 경로를 JVP·VJP에 포함한다. | 실제 fallback 셀의 초기장 섭동과 공개 FSOI를 확인한다. |
| A5-04 | P0 경향 미분은 실제 P0가 소비한 valid/observed 셀을 사용한다. | 탐지 하한보다 낮은 echo threshold 사례에서 검열 관측의 미분을 확인한다. 분기 상태는 `unknown`을 유지한다. |
| A5-05 | 품질 생략 시 관측과 source availability의 교집합을 사용한다. | 공개 `variational_nowcast`에서 명시적 유효 품질 입력과 결과·run digest가 같다. |
| A5-06 | 가중 학습 손실은 마스크를 먼저 적용하고 가중치를 정규화한다. FP16은 FP32로 누산한다. | 정상 FP16 격자, 큰 유한 가중 오차, 제외 셀, FP64 및 예측값 도함수 oracle을 확인한다. |
| A5-07 | 정확한 속도 상한에서는 닫힌 원판 투영의 내부 쪽 일반화 Jacobian을 선택한다. | 안쪽 radial JVP와 실제 바깥쪽 후보의 속도 제한을 함께 확인한다. |
| A5-08 | pseudo-Huber 비용과 IRLS를 같은 함수의 안정적인 척도로 계산한다. | 극단 delta의 이차 극한, 일반 범위 미분 및 지원 범위를 확인한다. |
| A5-09 | holdout의 선행시간을 실제 발행된 시각 목록에 대조한다. | 30분 간격에서 45분 요청을 거부한다. |
| A5-10 | 신규 발행·철회는 Boolean 발행 마스크의 집합 차이로 계산한다. | 검증 가중치가 0인 셀도 실제 발행 변경에 포함되는지 확인한다. |
| A5-11 | joint-min Brier는 진단으로 유지하고 proper score 조건으로 판정한다. | 판정·사전 표본 수 계산이 진단용 임계값에 좌우되지 않는다. |
| A5-12 | acceptance 보고서와 입력 파일의 경로·inode 충돌을 거부한다. | 직접 경로·symlink·hardlink 반례에서 입력 바이트를 보존한다. |

속도 제한의 정확한 경계에는 보통의 미분이 존재하지 않는다. 선택한 일반화
Jacobian은 안쪽 개선 방향을 지우지 않도록 하는 계산 규칙이며, 비선형 목적함수의
전역 최적성이나 KKT 조건 충족을 인증하지 않는다. 실제 후보 평가와 수용 검사는 유지한다.

검열된 반사도는 점 관측과 구분한다. 검열 구간의 상한보다 support threshold가
엄격하게 높을 때만 알려진 비사건으로 support/clear-sky 평가에 포함한다. 구간이
사건 경계를 가로지르면 평가 근거로 쓰지 않는다. 검열값을 정확한 반사도 관측으로
사용하지 않는다. 별도의 `CONFIRMED_CLEAR` 상태는 저장된 dBZ가 임계값보다
높더라도 비사건이다. 두 target 생성자는 `OBSERVED_ECHO`와 임계값을 모두 만족하는
경우에만 echo label을 부여하여 intensity 평가에 clear를 섞지 않는다.

관측 제거는 입력이 달라진 반사실적 실험이다. 연구 prior는 같은 runner로 다시
계산하고, 원래 입력에 묶인 운영 deployment의 재추론은 명시적으로 거부한다.
source mask를 digest만으로 복원하지 않는다. 기존 durable intervention 형식에는
source tensor가 없으므로 비자명한 source mask가 필요한 append/replay는 명시적으로
거부한다. 기존 `None`·전체 유효 source 기록은 호환한다.

pseudo-Huber의 작은 잔차는 상쇄를 피하는 이차형, 큰 잔차는 제곱·역전파 중간 곱의
overflow를 피하는 인수분해형으로 계산한다. 사용하지 않는 분기도 유한하게 유지한다.
state dtype의 정상 유한 범위 밖에 있는 물리 속도 scale·limit·비율과 subnormal
Huber delta는 준비 단계에서 거부한다. 모든 유한 입력 조합의 모든 AD 차수를
지원한다는 계약을 새로 만들지 않는다.

## 입력·저장·도구

- verification replay는 dtype·device·shape·값을 모두 대조한다. PSR은 부동소수점
  스칼라이어야 하고, Taylor 신뢰도 실험의 방향은 0일 수 없다.
- raw 공개키의 대소문자는 바이트로 비교한다. 서명 payload의 원래 표기는 유지한다.
  episode는 UTC 시각으로 정렬하고 저장 문자열은 바꾸지 않는다.
- source coverage를 해당 계획·입력·전체 분석 식별자와 조기에 결합한다. 중복
  site/time slot과 catalog의 잘못된 availability를 일찍 거부한다.
- 같은 provenance의 동시 재시도는 동일한 history로 수렴한다. scoring replay의
  rename 후 DB 등록 전 중단으로 생긴 미등록 디렉터리는 writer lock 안에서 재구성한다.
- M0의 mask·direct map·tile norm·innovation impact를 대조한다. migration은 기존
  row 보존을, deadline 실패는 관계 테이블 전체 rollback을 확인한다.
- P1 artifact의 중복 수용 진단, 단독 evaluation 목록, 단일 site geometry digest,
  오프라인 서명 artifact의 trust/genesis/expiry 관계를 확인한다.
- intervention에 source availability를 반영하고 잘못된 common-bias 메타데이터를
  거부한다. 픽셀 footprint의 무효 offset을 제거한다. common-bias tile은 두 축을
  각각 제한하여 동일 공분산 block과 JVP를 유지하고 padding 면적을 입력의 4배 미만으로
  제한한다. 임의의 거대 격자가 저렴해지는 것은 아니다.
- acceptance 파일은 디렉터리 descriptor를 따라 읽는다. runtime closure는 FIFO 등
  특수 파일과 미해결 native 의존성을 거부한다. MPS workflow는 Torch 2.13.0을 고정한다.

이 변경은 현재 미지원인 운영 inference나 과거 배포 참고 도구를 운영 가능한 것으로
바꾸지 않는다. 정상 producer를 통한 수치 결함과 외부 변조 artifact 경계의 검사를
구분해 해소 목록에 기록한다.

## 코드 변경 없이 종결한 주장

- 세 skill 조건의 동시 통과 판정은 intersection-union test이다. 무조건 계수 3을
  더하는 Bonferroni 수정은 하지 않는다.
- 투영 미터 입력에 경위도를 넣는 것은 caller 단위 계약 위반이다. learned raw의
  고정 결측값과 물리장의 설정 가능한 clamp는 서로 다른 입력 계약이다.
- 이류로 좌표를 맞춘 재검토에서는 전부 미정의 상태에서 온 발행 셀을 찾지 못했다.
  configured support cutoff의 의미 차이를 신규 수송 결함으로 판정하지 않는다.
- 같은 family의 rolling window 원자료 공유는 event clustering과 함께 의도된
  동작이다. acceptance artifact의 여러 case 공유도 금지 계약이 확인되지 않았다.
- 마지막 검사와 반환 사이의 trust 변경은 구체적인 선형화 위반 반례가 없었다.
  같은 검사를 끝없이 덧붙이지 않는다. 별개의 replay crash/retry 문제는 시험·수정한다.
- benchmark의 RSS/time 보고를 강제 입력 메모리 예산으로 해석하지 않는다.
  Boolean 숫자 수용만으로 새로운 수치 결함을 주장하지 않는다.

## 검증 범위

Luna xhigh 구현 담당과 독립 교차 검토가 변경을 분담했다. 공유 Python 환경과
사용자 파일을 보존했다. 원 조사에서 읽은 행 수는 결함 부재의 증명이 아니다.

최종 통합 시험 결과는 아래에 기록한다. 수정 전 반례와 수정 후 통과 로그는 별도
[재현 묶음](review_artifacts/a5_resolution.zip)에 보존한다.

- Python 3.12.13 / Torch 2.13.0 CPU: 변경·관련 통합 시험 **203 passed, 83 subtests passed** (177.92초). TorchScript deprecation 경고 18개.
- basedpyright: **0 errors, 0 warnings**. 의존성 lock 정합성 검사 통과. 환경 package 목록 전후 동일.
- FP32·FP64 Huber 16개 독립 사례에서 비용·1차/2차 미분·JVP가 해석값과 일치한다.
- 새 회귀시험 일부를 수정 전 `286c0c9` 소스에 적용하면 27개 실패(하위 시험 포함), 9개 통과. 기존 오류를 검출하는 증거이며 현재 코드의 실패가 아니다.
- 전체 CPU baseline은 [PR #158 CI](https://github.com/gonos2k/AD4DVAR-radar/pull/158/checks)에서 실행한다. 이 기록을 커밋할 때 최신 A5 CI는 아직 시작 전이다.
  이전 `286c0c9` CI `33996929420`의 네 작업 성공을 A5 변경의 검증으로 사용하지 않는다.

실제 레이더 hindcast와 CUDA 종단 간 검증은 이번 작업 범위에서 실행하지 않았다.
MPS의 작은 가중 손실 probe는 P0/P1 전체 검증을 대신하지 않는다.

## 격리 모드 CI 보정

첫 A5 커밋 `72bab53`의 [CI 34004612074](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/34004612074)는
Wheel/CLI·UI가 성공했지만 CPU 두 작업은 새 시험 두 개의 import 오류로 수집 단계에서
중단됐다. `python -I`에서 존재하지 않는 `tests` 패키지를 참조한 것이 원인이다.
기존 시험 모듈을 직접 가져오도록 두 줄을 고쳤고 제품 소스는 변경하지 않았다.

같은 격리 모드에서 해당 **13 passed, 6 subtests passed**, 추적 `tests/` 전체
**1055개 수집 성공**을 확인했다. 이것은 전체 시험 실행의 성공을 뜻하지 않는다.
[보정 증거](review_artifacts/a5_ci_compatibility_validation.zip)와
[범위 기록](review_artifacts/a5_ci_compatibility_validation.json)을 보존했다.
최종 커밋의 전체 CI는 PR의 최신 실행을 확인한다.
