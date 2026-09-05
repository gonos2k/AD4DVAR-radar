# 두 번째 코드 조사 후속 수정

조사 기준은 `94c75d3`, 구현 시작점은 `fc0dc6d`다. 원래 반례와 적용 범위는
[조사 체크리스트](SECOND_CODE_REVIEW_CHECKLIST.md), 당시 파일별 읽기 증거는
[범위 기록](SECOND_CODE_REVIEW_COVERAGE.md)에 보존한다. 아래는 그 이후의 수정이다.

## 수정과 회귀시험

| 항목 | 수정한 관계 | 직접 검증하는 시험 |
| --- | --- | --- |
| A2-01 | typed 검증 가중치를 평가영역·점수·미분·digest·학습 재해석에 한 번 적용 | [FSO](tests/test_a2_fso.py) |
| A2-02 | 시간이 겹치는 이벤트는 같은 시각의 궤적 거리와 공간 IoU로 대칭 비교 | [이벤트](tests/test_a2_promotion.py) |
| A2-03 | 비영 P0 입력 변화는 유한 표본이 같아도 `unknown`; strict/학습은 거부 | [분기 반례](tests/test_a2_branch.py) |
| A2-04 | analytic UCB 경로에서 사용하지 않는 bootstrap 생성·tail 거부 제거 | [UCB](tests/test_a2_promotion.py) |
| A2-05 | 현재 생성·저장·복원에서 audit-only v5 과학 격자 거부; 역사적 복원 유지 | [artifact](tests/test_a2_artifacts.py) |
| A2-06 | 허용한 정수 age를 float로 정규화하고 float64로 직렬화 | [age 왕복](tests/test_a2_artifacts.py) |
| A2-07 | legacy 예측 비교에서 NaN 위치와 나머지 정확한 값을 함께 확인 | [legacy 왕복](tests/test_a2_artifacts.py) |
| A2-08 | decision 단계에서도 canonical dBZ·mask·quality가 변하지 않는 개입 거부 | [개입](tests/test_a2_intervention.py) |
| A2-09 | 중복 사전 검사와 실패한 호출 소유 디렉터리 정리; 기존 파일 보존 | [저장 실패](tests/test_a2_ledger_io.py) |
| A2-10 | SQLite 쓰기 잠금을 얻은 후와 commit 직전에 publication 시각 재확인 | [마감 시각](tests/test_a2_ledger_io.py) |
| A2-11 | 경쟁에서 먼저 완료된 동일 provenance·receipt·상태를 확인하고 복구 성공 반환 | [동시 복구](tests/test_a2_ledger_io.py) |
| A2-12 | M0 저장 전에 현재 context 이름과 순서 확인 | [M0 저장](tests/test_a2_ledger_snapshot.py) |
| A2-13 | unavailable whole-field와 tile impact 모두 NaN 요구 | [M0 결측](tests/test_a2_ledger_snapshot.py) |
| A2-14 | 실제 보존된 lead의 direct map × std로 whitened tile norm 대조 | [M0 norm](tests/test_a2_ledger_snapshot.py) |
| A2-15 | M0 공간 차원은 둘 다 양수 요구 | [M0 차원](tests/test_a2_ledger_snapshot.py) |
| A2-16 | 보존한 control과 remap cell 관계 확인 | [선형화](tests/test_a2_numerics.py) |
| A2-17 | v4 승인 evidence의 full analysis input digest를 validation과 대조 | [승인 lineage](tests/test_a2_fso.py) |
| A2-18 | canonical 관측은 finite 요구; raw 결측 정리와 고정 배경 재해석 유지 | [관측 계약](tests/test_a2_numerics.py) |
| A2-19 | 큰 유한 이동의 정수 좌표 변환과 영역 밖 수송의 0·AD 경로 보강 | [수송 경계](tests/test_a2_numerics.py) |
| A2-20 | range label의 비어 있지 않음과 유일성 요구 | [range](tests/test_a2_numerics.py) |
| A2-21·22 | 물리 단위 CLI 옵션 모두 격자·시각을 요구하고 argparse 오류로 보고 | [CLI](tests/test_a2_cli_tools.py) |
| A2-23 | legacy bundle 생성 전에 Linux x86_64 지원 범위 확인 | [bundle](tests/test_a2_cli_tools.py) |
| A2-24 (교차 검토 후속) | holdout·operational provenance 정리도 쓰기 잠금 안에서 수행; 새 rename 직후 소유권 기록 | [동기화 실패 주입](tests/test_a2_ledger_io.py) |

A2-16·19의 교차 검토에서는 control 길이·dtype 검사와 큰 변위의 교차 dtype
처리를 추가했다. [교차 자료형 시험](tests/test_a2_numerics_cross.py)은 CPU FP64
`1e308` 이동의 0 출력·미분과 CPU↔MPS 양방향 수송의 JVP/VJP를 확인한다.
MPS가 FP64를 만들지 않도록 변환 순서를 정하며, MPS 함수 변환에서 지원하지
않는 Python sequence `new_tensor` 대신 scalar 연산을 유지한다.

## 이론적 범위와 호환성

P0 FFT peak와 pair 선택은 중간에 바뀌었다가 돌아올 수 있다. 0.5·1.0 배율의
일치로 전 구간을 인증할 수 없으므로, 관측 기반 P0 변화가 0인 경로만 현재
`certified`로 판정한다. 표본에서 분기가 다르면 `invalid`, 그 외 비영 변화는
`unknown`이다. 탐색용 FSOI는 남기되 trusted total은 제공하지 않는다.
외부·배경 P0처럼 관측 기반 경로가 없으면 기존 `not_applicable`을 유지한다.
관측 기반 P0를 바꾸는 물리적 dBZ 개입은 현재 자동학습 승인을 받지 못한다.
full/half 비선형 재해석을 통한 Taylor·부호·수렴 검사는 계속 별도로 수행한다.

기존 learning 승인 시험의 branch certificate는 이제 명시적인 fixture다.
그 시험을 실제 물리 분기 증명으로 세지 않는다. 새 current v22 시험은 policy
trust loader만 대체하고, 실제 분기·격자·검증·수치 경로가 비영 개입을 거부함을
확인한다. 원래 0.25 내부 peak 반례도 별도 시험으로 남긴다.

v5는 과학적 격자의 audit-only 세대다. 이 수정은 연구용 v1–v4 격자를 v6로
재해석하지 않으며, 역사적 artifact 읽기와 현재 발행의 경계를 구분한다.
legacy NaN 회귀시험은 현재 자료를 과거 형식으로 재봉인한 fixture이고 실제
과거 운영 binary 모음에 대한 시험은 아니다.

M0 whitened norm은 저장된 지도에 대해서만 재계산한다. 저장하지 않은 lead의
지도를 만들어 검증했다고 주장하지 않는다. 관측과 frozen state를 무조건 같은
digest로 묶지 않으며, 고정 배경에 새 관측을 넣는 재해석을 유지한다.

## 검증 기록

구현 커밋은 `95c0ccdc56a997b4770a93008b4b8058f6884fff`다.

- 분기 반례와 기존 민감도: 95 passed, 111 subtests.
- 통합 A2·ledger·artifact·수송·선택 P1/복구: 148 passed, 173 subtests.
- 마지막 수송·FSO·provenance 영향 재검증: 28 passed, 91 subtests.
- 세 실행의 **중복을 제거한 고유 시험은 245개**다. 팀별 시험 수를 더하지 않았다.
- 제품 소스 basedpyright 1.39.9: 0 errors, 0 warnings, 0 notes.
- 새 A2 시험 Ruff 및 `git diff --check` 통과. 전체 소스 Ruff의 기존 38개 지적은
  이번 수정에 섞어 정리하지 않았다.

[시험·환경·파일 SHA-256 JSON](review_artifacts/a2_implementation_validation.json)과
[팀 결과·중간 실패·최종 로그·JUnit·manifest ZIP](review_artifacts/a2_implementation_validation.zip)에
재현 기록을 보존한다. ZIP은 29개 파일, 39,894바이트,
SHA-256 `c0a3831c4d41d257ef8d8f0439bdf21fd6627eec4110bb83d30fb03d0ee956f8`다.
원래 전수 조사 ZIP과 coverage는 수정하지 않았다.

MPS 중간 실패는 최종 성공과 구분해 남겼다. provenance fsync 실패는 공개
fixture에 직접 주입했으며, 별도의 시간 경쟁 기반 cleanup/retry 시험을 실행한
것은 아니다. 초기 통합 명령의 잘못된 시험 파일명은 수집 전 오류였고 실제
관측한 파일명으로 고쳐 실행했다.

원격 CPU pytest도 로컬과 같은 `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`을
사용한다. 전체 시험 목록과 timeout은 유지한다. 정확한 runtime digest는 이
설정을 반영하며, Linux 성능 개선을 계측했다고 주장하지 않는다.
[PR #156 checks](https://github.com/gonos2k/AD4DVAR-radar/pull/156/checks)에서 최종
커밋의 전체 원격 CI 결과를 확인한다. 이 문서의 로컬 통과를 원격 성공으로
표현하지 않는다.

로컬 전체 baseline은 반복하지 않았다. 실자료 예측 성능, CUDA, 전체 MPS P1,
운영 배포는 별도 범위다. 확인 범위 밖의 결함이 없다는 보장은 하지 않는다.
