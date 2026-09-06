# 다섯 번째 전체 코드베이스 조사

기준: `286c0c9416a645ec4350b24ab9e9faca6584bb56` (PR #158).
조사일: 2026-09-06 KST/JST. **조사는 완료했으며, 이상 없음으로 판정하지 않는다.**
아래 반례·줄 번호·CI 상태는 수정 전 조사 기록이다. 후속 수정은 완료했으며,
최종 코드와 검증은 [해소 기록](FIFTH_CODE_REVIEW_RESOLUTION.md)에 따로 남긴다.

## 조사 완료

- [x] Git 추적 코드·시험·UI·CI·설정 78개 파일, 167,392행과 SHA-256 고정.
- [x] Luna (`gpt-5.6-luna`) xhigh 담당 34개 배정 및 최종 보고서 34개 수집.
- [x] 소스 읽기 로그 합집합이 고정된 전체 범위를 덮는지 확인.
- [x] 수학·수치해석·공학·기상학적 주장과 상위 호출부·입력 검사를 대조.
- [x] 핵심 반례의 루트 독립 실행 및 교차 검토로 과장·오탐 정정.
- [x] 원보고서·재현 코드·출력·소스 해시·최종 판정 보존.
- [x] 공유 Python 환경·사용자 파일 보존, CI 상태 구분, KG 인계.

행 읽기 로그는 의미 이해나 결함 부재의 증명이 아니다. 각 행을 네 분야 또는
GREEN·RED가 각각 독립 검토했다는 뜻도 아니다. 내장 JSON과 의존성 lock은 생성기·
정합성 검사로 확인했고, README는 관련 설명만 선택 확인했다. 담당별 시험은 중복을
포함하므로 통과 수를 합산하지 않는다. 전체 로컬 baseline, 실제 레이더 hindcast,
MPS/CUDA 종단 간 검증은 이번에 실행하지 않았다.

## 우선 항목의 해소 상태

A5-01–12는 모두 수정했다. 아래 설명은 원래 결함과 작업 요청을 보존한 것이다.
P1/P2/P3는 당시의 우선·보통·낮은 수정 우선순위다. 상세 변경·시험은 해소 기록을 참조한다.

| 상태 | ID | 우선순위 | 확인 내용과 최소 후속 작업 |
| --- | --- | --- | --- |
| [x] | A5-01 | P1 | `sensitivity.py:6747`의 source 구성 분기에서 현재 v15 입력이 빠진다. v22 검증자료의 공간 품질·표준편차 구성을 복원하고 독립 수치 oracle을 추가한다. |
| [x] | A5-02 | P1 | `sensitivity.py:13080`의 관측 제거 재분석이 nominal neural prior를 전달하지 않는다. 같은 prior를 변경 입력에 재결합·재추론하거나 지원하지 않는 조합을 명시적으로 거부한다. |
| [x] | A5-03 | P2 | `sensitivity.py:9315,16173`은 부분적으로 유효한 prior의 radar fallback 미분을 누락한다. 실제 초기장 구성과 물리 섭동 채널의 chain rule을 일치시킨다. |
| [x] | A5-04 | P2 | `sensitivity.py:16275`의 P0 민감도 마스크가 관측 허용영역보다 좁다. echo threshold가 detection limit보다 낮을 때 검열 관측의 실제 미분을 제거한다. 고정 분기 미분의 지지영역을 맞춘다. |
| [x] | A5-05 | P2 | `nowcast.py:4785`의 기본 품질 생성이 source availability를 반영하지 않는다. 부분 source mask만 제공하면 공개 `variational_nowcast`가 내부 기본값을 거부한다. |
| [x] | A5-06 | P2 | `promotion.py:4710`의 학습 손실이 입력 dtype으로 합산한다. FP16 정상 크기 배열의 분모 overflow, FP32 가중 손실의 중간 overflow 및 마스크 밖 `0*inf`를 방지한다. |
| [x] | A5-07 | P2 | `variational.py:8804`의 정확한 물리 속도 상한에서 AD radial JVP가 0이다. 안쪽 개선 방향의 일방향 미분·제약 정상성 조건을 다뤄야 한다. |
| [x] | A5-08 | P3 | `variational.py:6108`의 극단적 유한 pseudo-Huber delta에서 참값이 표현 가능해도 비용·gradient가 0이 된다. 안정적인 스케일 정규화 또는 명시적인 지원 범위가 필요하다. |
| [x] | A5-09 | P2 | `promotion.py:14451` 이하의 holdout helper가 lead를 정수 나눗셈으로 선택한다. 발행하지 않은 lead를 기존 발행 시각 검사와 같은 기준으로 거부한다. |
| [x] | A5-10 | P2 | `promotion.py:14538`의 전역 신규 발행·철회 비율이 점수 가중치의 양수 영역을 센다. 발행 마스크의 집합 차이와 평가 가능 영역을 구분한다. |
| [x] | A5-11 | P2 | `promotion.py:22078` 이하에서 README의 diagnostic-only joint-min Brier가 여전히 필수 gate다. 의도한 과학 계약을 정한 뒤 문서와 판정을 맞춘다. |
| [x] | A5-12 | P2 | `.github/scripts/run_real_case_acceptance.py:28`의 `--report`가 입력 artifact 경로와 같으면 검증 성공 직후 그 파일을 덮어쓴다. 입력·출력 경로 충돌을 먼저 거부한다. |

### 반례와 적용 범위

**A5-01.** 공개 v22 생성·재검증 경로에서 quality/std가 모든 셀에 `1/2`로 남았다.
같은 입력의 공간 항만 적용해도 quality는 약 `0.9815–0.9832`, std는
`2.1671–2.1741`이다. 도입 커밋 `ccdc9b2`는 내부 v15 ground-range 분기를 추가했지만
바깥 v10–v14 목록에 v15를 추가하지 않았다. age 60초의 기존 시간 식까지 적용하면
약 `0.7644–0.7657 / 2.2486–2.2554`다. 공간 항 누락은 확정이며, 시간 항은 source
선택의 transcendental 인증 제한과 런타임 오차 구성의 역할을 구분해 처리해야 한다.
같은 함수를 재실행하는 digest/replay 시험만으로는 누락을 검출하지 못한다.

**A5-02.** 8×8 exogenous-prior 사례에서 명목 P1과 두 재분석이 모두 적격이었다.
공개 함수의 removed score는 `0.04431874046394474`, 같은 runner를 변경 QC에
재결합·추론한 score는 `0.015242213929648809`, 차이는 `0.029076526534295932`다.
현재 공개 함수의 점수 정의로 비교한 값이며 실제 레이더 성능 개선량이 아니다.
source availability의 별도 전달 누락은 prior 반례와 구분한다.

**A5-03/04.** 부분 prior fallback에서 입력을 `+0.01 dBZ` 바꾸면 실제 frozen 초기장은
`30 → 30.01`로 바뀌지만 물리 initial-background 채널은 0이다. 이 FSOI 프로브는
curvature 허용치를 완화한 연구 설정이다. 별도 검열 관측 사례에서는 실제 동일
adjoint를 사용해 마스크만 accepted observed 영역으로 바꾸면 28셀의 최대 절대
미분이 `195673.1732`인 반면 공개 FSO 맵은 0이다. 큰 값의 실용성이나 전체 비선형
구간 안정성을 주장하지 않는다. 두 경로의 P0 branch status는 `unknown`이며 엄격한
자동학습 제한을 완화하면 안 된다.

**A5-05/06.** source mask가 일부 false인 3×4×4 입력에서 품질을 생략하면 오류,
해당 셀 품질을 명시적으로 0으로 주면 수용한다. FP16 256×256 단위 오차의 참 손실은
1이지만 품질 합 65,536이 overflow해 예외가 발생한다. 같은 FP32 사례는 1이다.
FP32의 작은 가중치·큰 유한 오차에서는 참값 약 `1.00000005e30`이 표현 가능하지만
현재 결과는 `inf`; 평가 제외 셀의 큰 유한 예측은 참 손실 0을 `NaN`으로 바꾼다.
후자의 두 사례는 극단 입력이다.

**A5-07/08.** 정확한 속도 상한의 공개 준비 경로에서 안쪽 robust-objective AD
방향미분은 거의 0이고 유한차분은 약 `-22763.7`이다. 전체 solver의 잘못된 최종
수용까지 재현한 것은 아니다. 별도 pseudo-Huber 반례는 원시 입력 `20→21→20 dBZ`,
FP32 delta=`2e38`, FP64 delta=`1e308`에서 이차 극한 비용 약 `1.125` 대신
비용·gradient 0이다. 기본 delta의 결함은 아니다.

**A5-09/10/11.** 30분 간격 helper에 45분을 요청하면 30분 frame을 선택한다.
전역 발행 비율의 작은 helper 반례는 신규 발행 1셀을 0셀로 센다. 관련 상위 검사를
정적으로 추적했으나, 이 세 항목에서 모든 서명·ledger를 갖춘 완전한 promotion
실행을 독립 재현하지 않았다. 현재 운영 inference의 명시적 미지원 제한을 유지한다.
연구 평가 오류와 운영 배포 우회를 혼동하지 않는다.

## 추가 경계와 시험 보강

전체 담당 의견과 루트 판정은 [JSON](review_artifacts/a5_full_review.json) 및
[재현 묶음](review_artifacts/a5_full_review.zip)에 있다. 담당자의 최초 severity보다
루트의 최종 판정과 적용 범위를 우선한다.

- [x] 검열 관측 support/clear-sky 평가: point metric weight 재사용으로 검열 셀을
  모두 제외한다. censor interval과 support threshold에 따라 알려진 event와 미확정
  event를 구분해야 한다. 모든 검열 셀을 clear로 넣는 수정도 잘못이다. 과학적 평가영역
  보완 후보이며 완전한 공개 promotion 반례는 아직 없다.
- [x] 입력·저장: 대소문자 hex 공개키 비교, UTC offset이 다른 episode 문자열 정렬,
  외부 snapshot의 map/tile/impact 일관성, source mask intervention, source coverage
  교차 계획 결합·동시 재시도, 단독 artifact 검증 관계. 정상 producer와 최종 holdout
  검사가 이미 막는 경우를 JSON에 구분했다.
- [x] 자원·도구: 극단 pixel footprint/tile 설정, 특수 파일 runtime snapshot의
  blocking open, 누락된 `ldd` 의존성, acceptance 경로 교체 경쟁, 선택 MPS workflow의
  비고정 PyTorch 설치. 과거 운영 참고 도구·변조 입력·극단 설정의 범위를 명시한다.
- [x] 시험: 비중앙 quantized/conditional PIT의 수치 oracle, 실제 posterior 물리 단위
  변환, 최종 stationarity 재계산, 전체 동시 bound 키 대응을 보강한다. Hessian/NLL/score
  독립 시험과 scorer를 패치하지 않는 종단 간 replay 시험은 이미 있다.
- [x] README 2448–2452의 과거 “neural prior 미구현·pair 불일치만으로 불확실성 계산”
  설명을 현재 구현과 맞춘다.

## 철회·보류

- 세 skill 조건을 모두 통과해야 하는 판정에 무조건 Bonferroni 계수 3이 필요하다는
  주장은 철회했다. union null에 대한 intersection-union 판정이다.
- 투영 미터 좌표 API에 경위도를 넣는 것은 caller 단위 계약 위반이다. 기존 DEP-002
  genesis 불일치도 신규 P0가 아니라 알려진 비권위적 운영 참고 도구 문제다.
- learned raw 입력의 고정 `-10 dBZ`와 물리장 설정 clamp는 다른 계약이다.
  반드시 같아야 한다는 근거가 없어 결함 주장을 철회했다.
- source-time 상태와 lead-time 발행 마스크를 같은 좌표에서 비교한 반례는 부적절했다.
  이류로 정렬한 후 전부 미정의 상태에서 온 발행 셀은 찾지 못했다. support cutoff의
  의미 차이만 낮은 우선순위 후보로 남긴다.
- 같은 family rolling-window의 원자료 공유는 event clustering과 함께 의도된 동작이다.
  Boolean 숫자 수용·타입 주석 미강제만으로 수치 결함을 확정하지 않았다.
- 마지막 trust 검사 이후 회전 경쟁과 scoring-replay 디렉터리 process-crash 복구는
  조사 당시 공개 재현이 없는 정적 후보였다. 후속 작업에서는 전자는 추가 코드 없이
  종결하고, 후자는 중단 후 재시도 시험과 writer-lock 복구로 보강했다.

## 조사 당시의 검증 상태

기본 수송의 비음수성·경계 유출, 고정 선형화·실제 PCG 잔차 확인을 유지한다.
이번 조사에서 이를 다시 설계할 근거를 찾지 못했다.

- 로컬: Python 3.12.13 / PyTorch 2.13.0 CPU. 공유 환경 package 목록 전후 동일.
- main `2398e5f` CI [33993713947](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/33993713947): 네 작업 성공.
- PR #158 `286c0c9` CI [33996929420](https://github.com/gonos2k/AD4DVAR-radar/actions/runs/33996929420): 기록 시 CPU 두 작업 진행 중; Wheel/CLI·UI 성공.
  실제 PR checkout `a559617…`의 tree `7b1e261…`는 기준 HEAD와 같다.
  이전 main 성공을 현재 PR 전체 성공으로 대신하지 않는다.
- `AGENTS.md`의 Luna xhigh·이론 우선·단순성·환경 보존 원칙은 이미 반영되어 있다.
- KG는 graph-only로 갱신했다. 그래프는 사용자 로컬 자료·문서도 포함하므로
  그래프 추출 수와 이번 Git 추적 78파일 조사 범위는 같지 않다.
