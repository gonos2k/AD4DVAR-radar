# A6 prior 공간 기울기 전달 수정

기준: `main 1d0ce7d6b2a50238d0188b8bdcc968084907be2b` (PR #158 병합).

- [x] `_frozen_initial_background_observation_sensitivity()`의 두 반환값과 호출부 차원 확인.
- [x] `[H,W]`인 `background_field_sensitivity`의 `[0]` 두 곳만 제거.
- [x] `[3,H,W]`인 `background_sensitivity[0]`의 첫 관측시각 선택은 유지.
- [x] 실제 P1 identity prior의 제공 화소와 관측 fallback을 함께 검증.
- [x] 비균일·첫 행 0·행 반전 기울기로 합성함수의 해석값·중앙차분 검증.
- [x] Luna xhigh 독립 검토와 관련 시험·타입 검사 완료.
- [x] 수정 전 호출부를 복원하면 최종 회귀시험이 실패하는지 확인.
- [ ] 후속 PR의 전체 CI 성공 확인 — 로컬 검증과 구분한다.

## 원인과 수정

두 번째 반환값은 배경장에 대한 직접·암시적 미분을 더한 2차원 장이다. 여기에
`[0]`을 적용하면 첫 공간 행만 남고, `torch.where`가 이를 모든 행으로 broadcast한다.
prior의 VJP에는 `M * G` 대신 `M * G[0]`가 전달된다. 반환 shape와 유한성 또는
prior 내부의 JVP–VJP 내적 검사만으로는 호출자의 잘못된 cotangent를 검출할 수 없다.

전체 장을 전달하도록 두 인덱스를 제거하고 반환 차원을 주석에 명시했다. 반환 타입,
prior 모델, Gauss–Newton/PCG 수식, 분기 신뢰 조건은 변경하지 않았다.

## 회귀시험의 의미

`tests/test_prior_spatial_cotangent.py`는 두 수준을 구분한다.

1. **실제 P1 경로:** 기존 부분 identity prior `P(Y)=Y[0]`와 일정한 prior 표준편차를
   사용한다. 관측 fallback도 항등 경로이므로 결합 민감도는 관측 직접 민감도,
   P0 경향 민감도, 전체 배경장 민감도의 합이어야 한다. `(3,3)` prior 화소와
   `(2,3)` fallback 화소를 함께 섭동하고 FSO map·FSOI 성분·총 영향값을 확인한다.
2. **해석적 합성함수:** 같은 공개 P1 호출부와 실제 prior runner를 사용하되 배경
   부분문제만 `J(c,B)=||c-B||²/2`, `E(c,B)=.25<G,B>+.75<G,c>`로 교체한다.
   실제 배경 민감도 helper는 정확히 `G`를 반환해야 한다. 이후 전달·마스킹·VJP를
   통과한 배경 영향값을 `G*delta` 및 `c*=B`를 사용한 전체 합성함수 중앙차분과
   대조한다. 첫 행이 0인 장과 행 반전도 포함한다.

두 번째 시험은 실제 비선형 레이더 재분석의 유한차분 검증이 아니다. 첫 번째 시험도
prior와 fallback 연결의 chain rule을 검증하며, 독립 Hessian 해나 실제 레이더 예측
성능을 인증하지 않는다. 기존 연구 fixture의 완화된 Gauss–Newton curvature 허용치를
사용하고, 제품의 기본 신뢰 조건이나 `unknown` 분기 제한을 완화하지 않는다.

## 검증 기록

- Python 3.12.13 / Torch 2.13.0 CPU, 격리 모드(`-I`), 단일 thread.
- 최종 관련 시험: **93 passed, 99 subtests passed** (65.09초), TorchScript deprecation 경고18개.
- basedpyright: **0 errors, 0 warnings**. 새 시험 Ruff 검사와 Python3.10 문법 검사 통과.
- 최종 새 시험에 기준 커밋의 호출부를 메모리에서 복원하면 일반 실패1개·subtest 실패3개가
  prior 제공 화소에서 재현된다. 출력의 `4 failed, 1 passed`는 하위 시험 실패와 부모
  시험 집계 방식에 따른 것이며 통과 판정이 아니다.
- 실제 P1 시험과 합성 부분문제 시험의 범위는 위 설명대로 구분한다. 새 시험의 P1
  계산은 FP64이며, 이번 실행을 FP32/MPS/CUDA의 동일 경로 검증으로 확대하지 않는다.
- 기준 main CI34015464185와 PR158 CI34009581126은 조회 시 네 작업 모두 성공했다.
  이들은 이번 수정 전 실행이며 후속 PR의 검증을 대신하지 않는다.
- 공유 환경의 package 목록은 작업 전후 동일하다.
- [검증 JSON](review_artifacts/a6_prior_cotangent_validation.json) ·
  [소스 변경·시험·교차 검토·로그 묶음](review_artifacts/a6_prior_cotangent_validation.zip).

기존 radar-dependent prior FSO/FSOI 산출물은 수정한 버전에서 재계산해야 한다.

전체 로컬 baseline, 실제 레이더 hindcast, MPS/CUDA 종단 간 시험은 실행하지 않았다.
검증한 수정과 전체 코드베이스의 결함 부재를 구분한다.
