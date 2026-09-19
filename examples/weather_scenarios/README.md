# 기상 시나리오 교육 데모

생성한 `index.html`을 브라우저에서 직접 연다. 서버나 Python 없이 저장된 결과를
재생한다. 모바일·태블릿·데스크톱에서 같은 메뉴를 사용한다.
대형 생성 HTML·240격자 중복 원시장은 Git에 포함하지 않으며 아래 명령으로 만든다.

다음 명령으로 현재 Python 구현의 P0 예측을 다시 계산하고 HTML을 생성한다.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/build_demo.py
```

240×240 회전과 국소 1차 정상성 검사를 포함하려면 기본 HTML 생성 후 실행한다.
이 단계는 CPU에서 수분과 약 5GiB 메모리가 필요하다. 전체 FSO 적격성 검사가 아니다.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_rotation_demo.py \
  --maximum-outer-iterations 12 --step-tolerance 1e-10 --verify-stationarity \
  --output /tmp/advar-rotation240.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/integrate_fv_demo.py \
  --fv-json /tmp/advar-rotation240.json
```

작은 신경망 실험은 `fv_neural_learning_probe.py`, 전체 관측 민감도는
`fv_neural_observation_probe.py`, 신경망 특징 경로만 분리한 재동화는
`fv_neural_observation_path_probe.py`로 재현한다. 저장된 작은 모델과 수치 결과는
`graphify-out/fv-root-cause-20260919/`에 있다. 결과·한계는
`LOCAL_VERIFICATION_RESULTS.md`를 참조한다.

이동, 감쇠, 회전, 분리·서로 다른 이동, 새로운 셀 발생, 경계 유출, 결측의
7개 합성 상황을 10–180분에 걸쳐 비교한다. 정답은 독립적인 해석적 장으로
생성하고, 예측에는 분석 시각 이전의 세 관측만 제공한다. 회전·분리·새 셀은
전역 이동·성장 모형의 표현 한계를 학습하는 사례다. 실자료 성능 검증은 아니다.

화면은 반사도(dBZ)를 표시한다. 코어는 비음수 에코량으로 수송한다.
MAE와 persistence MAE는 같은 유효 화소에서 계산하며, 결측을 정답으로 채우지 않는다.
리드타임별 평가영역은 달라질 수 있다. 점수는 원래 텐서에서 계산하고 표시 격자만
소수 둘째 자리로 반올림하므로 화면 값을 이용한 재계산에는 작은 차이가 있다.

`template.html`은 `/Users/yhlee/MetSim/seabreeze.html`을 복사한 뒤 교육 내용에
맞게 수정했다. 원본은 보존했다. 원본 SHA-256:
`314bcf972751733bb76ba8a34d14a4068af985cc247d0d8fe4219296540bdcff`.

`scenarios.py`는 데이터 생성, `build_demo.py`는 JSON 삽입만 담당한다.
브라우저에서 예측 수식을 다시 구현하지 않는다. 자세한 검토 범위는 저장소의
`PHASE1_SCENARIO_REVIEW.md`에 기록한다.

## 공간 수송 수치 기준선

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/finite_volume_probe.py --output /tmp/advar-fv-probe.json
```

이 도구는 `advar.transport`의 공유 면유량·상류 flux·SSPRK2·적분인자를 실제로
실행하고, 독립 특성곡선의 셀 평균과 18개 선행시간을 비교한다. 속도장은 주어진
수치 시험 입력이다. 관측에서 회전을 추정하거나 P1·FSO/FSOI·학습을 실행한
결과가 아니며, 기존 HTML의 예측 결과도 바꾸지 않는다.

출력은 에코 L2·중심·축별 폭·적분·support와 계산 시간, 프로세스 peak RSS를
담는다. 1차 상류 수송의 확산 오차가 커 현재는 제품 승격 대상이 아니다.
검증 및 재개 기록은 `graphify-out/scenario-implementation-20260909/checkpoints/`에 둔다.

## 저확산 MUSCL 후보 비교

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/compare_transport.py --output /tmp/advar-muscl-comparison.json
```

48/64/128/256격자에서 donor-cell과 독립 `muscl_experiment.py`를 같은 시간
분할·물리 영역·정답으로 비교한다. MUSCL은 minmod 기울기, SSPRK2, CFL≤0.5를
사용하며 전체 장과 영 외부 경계가 알려진 연구 사례만 지원한다. limiter가
비선형이므로 support의 기여 분해나 FSO 학습 적격성을 제공하지 않는다.

`--sizes 64 --time-multiplier 2`로 시간 간격만 절반으로 줄인 결과도 확인할 수
있다. 비용은 진단과 서로 다른 부가 기능을 포함하므로 제품 성능 비교가 아니다.
기존 교육 HTML과 P0/P1 호출은 그대로 유지된다.

## 저장된 CP3b 후보 판정

```sh
.venv/bin/python -I examples/weather_scenarios/evaluate_holdout.py \
  --report graphify-out/scenario-implementation-20260909/checkpoints/CP3b_holdout.json \
  --manifest graphify-out/scenario-implementation-20260909/checkpoints/CP3b_manifest.json \
  --output /tmp/advar-cp3b-acceptance.json
```

시뮬레이션을 재실행하지 않고 실행 해시와 후보의 모든 선행시간 기준을 확인한다.
폭은 절댓값으로 판정한다. 방향 평가가 없으면 `incomplete`, 기준 미달·증거 누락은
`failed`이며 둘 다 종료 코드 1을 반환한다. 비교군 donorcell의 기준 미달은 후보의
판정에 섞지 않는다. `passed`는 저장된 CP3b 적용 범위의 수치 기준 통과만 뜻한다.

## 수송 교육 실험실 · 실제 구현 실증

`transport.html`을 브라우저에서 직접 열면 된다. 외부 서버나 인터넷 없이 동작한다.
기존 `index.html`, `template.html`, MetSim의 `seabreeze.html`은 보존했다.

- 5개 affine 상황의 0–180분 실제 계산 프레임(각 19개)을 재생한다.
- 계산은 128², 화면은 64² 셀 평균이다. 숫자 지표는 원래 계산장에서 구한다.
- 독립 특성선 정답·donorcell·minmod 및 signed 오차를 고정 색 범위로 비교한다.
- 일반 경계 표는 기존 FV 기준선의 1셀 실증이다. minmod 일반 경계가 아니다.
- D7 표는 128²·3단계 CPU forward/JVP/VJP의 초기화 포함 단일 측정이다.
- 기존 P1 표는 평균 파라미터 1개를 FSO로 1회 학습하고 FSOI/재분석을 비교한 별도 실증이다.
  작은 비용 개선을 전체 기상 상황의 성능 개선으로 해석하지 않는다. 새 FV-P1 연결은 남아 있다.

재현(공유 환경 재생성 불필요):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/transport_resource_probe.py --output /tmp/transport-resource.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/existing_p1_probe.py --output /tmp/existing-p1-learning.json --checkpoint-destination /tmp/mean-only-prior.pt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/build_transport_demo.py --output examples/weather_scenarios/transport.html --evidence /tmp/transport-build.json --resource /tmp/transport-resource.json --learning /tmp/existing-p1-learning.json
```

P1 실행 도구는 격리된 임시 체크포인트로 실험하며, 전체 실증 성공 후에만 고유한
이름으로 사본을 보존한다. 저장 실패는 실패 상태와 종료 코드 1로 보고한다.
실증 기록: `graphify-out/transport-education-20260910/`.

## FV 관측 민감도와 재동화 검증

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I \
  examples/weather_scenarios/fv_sensitivity_probe.py \
  --output /tmp/advar-fv-sensitivity.json
```

실제 robust 분석 목적함수와 FV 예측을 4×5 CPU FP64 합성 문제에서 실행한다.
정답 유속을 분석 시작값으로 넣지 않는다. 작은 dense Newton 계산은 반환점을
정상점 근처로 정밀화하는 **검증용 oracle**이며 제품 solver가 아니다.

관측에서 만든 초기 배경 `B=y[0]`을 포함하여
`dE/dy = E_y - (D_y grad_c J)^T H^{-T} E_c`를 계산한다. 기존 HVP·PCG를
사용한 정확한 robust Hessian 수반을 dense 풀이 및 관측 섭동 후 재동화와
비교한다. 점수의 실수 가중치·평가영역·미래 경계는 고정한다. 출력은 정상성
잔차, 수반 잔차, Hessian과 IRLS GN의 차이, 관측만의 조건부 유동/성장 랭크,
계수 구간의 면유량 부호 여유, h/h2/h4 Taylor 오차를 담는다.

전부 탐지된 관측, 고정 마스크·불확실성·support, 처방된 미래 경계와 donorcell에
한정한다. 계수 상자 내부의 부호 보존은 미지의 정상점 경로가 그 상자 안에
있다는 보장이 아니다. 이 실증은 legacy FSO/FSOI 적격성, minmod의 모든 분기,
실자료 성능, 신경망 학습, 전체 D7를 통과시킨 결과가 아니다.

## 현재 FV 분석·예측·관측 응답의 화면 확인

`fv_response.html`은 현재 shared solver의 분석→FV 예측→IRLS–GN 관측 응답을
실행한 뒤 저장한 장을 재생한다. 시간 슬라이더, 관측별 signed 민감도, 고정
평가영역의 MAE, 정상성·수반 잔차와 통과/미검증 카드를 확인할 수 있다.
이 데모의 4×5 같은모형 합성 성공을 일반 기상 상황·전체 D7·학습 완료로
해석하지 않는다. 기존 `index.html`과 `transport.html`은 보존한다.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I \
  examples/weather_scenarios/build_fv_response_demo.py \
  --data-output /tmp/advar-fv-response-data.json
```

브라우저는 Python 결과를 재생할 뿐 예측을 다시 만들거나 화소를 보간하여
움직임을 꾸미지 않는다. 미래 정답은 평가에만 사용한다. `--solver-only`로
HTML 없이 같은 solver→GN 응답 경로도 실행할 수 있다.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I \
  examples/weather_scenarios/fv_sensitivity_probe.py --solver-only \
  --output /tmp/advar-fv-shared-solver.json
```

### Original-page P0/FV comparison

`index.html` retains the original seven P0 cases and adds two clearly named
PDE-consistent cases, each with P0 and shared-FV analysis/forecast outputs.
The new cases use 12×12 cells, three observed times and +10/+20-minute leads.
FV receives prescribed analytic stage boundaries; P0 does not accept those
traces, so the displayed comparison is conditional, not a fair operational
skill ranking. Both FV solves currently stop at the outer-iteration limit;
the page keeps this stationarity limitation visible. FSO/learning is not included.

Reproduce calculation and update the original page (separate steps allow
rendering without rerunning the solver):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_original_cases.py --output graphify-out/fv-root-cause-20260913/fv_original_cases.json
.venv/bin/python -I examples/weather_scenarios/integrate_fv_demo.py --fv-json graphify-out/fv-root-cause-20260913/fv_original_cases.json
```

The original `build_demo.py` builds the historical P0-only dataset; run the
integration command afterwards to attach the FV comparison. Integration is
idempotent and preserves the seven historical case payloads.

The current original-page dataset uses the bounded convergence reproduction:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_original_cases.py --maximum-outer-iterations 12 --output graphify-out/fv-root-cause-20260913/fv_original_cases_convergence.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/integrate_fv_demo.py --fv-json graphify-out/fv-root-cause-20260913/fv_original_cases_convergence.json
```

Output-field centroid and covariance-axis changes are calculated from positive
excess echo over the declared background, independently of inferred coefficients.
The page compares the same finite-window truth and forecast. Clipping changes
these moments; centroid motion alone is not proof of rotation. Isotropic shape
orientation and incomplete fields are unavailable. Both analyses still end at
`step_tolerance_unverified`; the improved diagnostics are not a convergence claim.

The current default is the actual240×240 rotation run (48km domain,200m cells,180min forecast). Rebuild it and preserve the prior small cases with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_rotation_demo.py --output graphify-out/fv-root-cause-20260913/fv_rotation240.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/integrate_fv_demo.py --fv-json graphify-out/fv-root-cause-20260913/fv_rotation240.json graphify-out/fv-root-cause-20260913/fv_original_cases_convergence.json
```

The measured run took331s and4.69GiB peak sampled RSS. It ended at the outer-iteration limit, so the page preserves unverified stationarity. The white dashed line is the initial observed principal axis; the black line is the current field axis.

The September19 default uses the stricter stopping run. To reproduce it, add `--maximum-outer-iterations 12 --step-tolerance 1e-10` to `fv_rotation_demo.py` and write `graphify-out/fv-root-cause-20260919/fv_rotation240_strict.json`. Pass that JSON first to the same integration command, followed by the existing small-case JSON. The run met the gradient threshold after6outer iterations (477s/4.76GiB); stationarity certification remains unavailable. Previous240² artifacts are retained as the baseline.


### September 19 algorithm extensions

The current PR keeps nominal sensitivity, finite reanalysis, and learning evidence
separate. `PR_CHECKLIST.md` in `graphify-out/fv-root-cause-20260919/` is the current
status; earlier completion statements apply to their recorded scope and commit.

Reproduce the small, independently prescribed-flow grid study:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_grid_convergence_probe.py --output /tmp/advar-fv-grid.json
```

The final 128-grid directional derivative errors are still about 10–13%; decreasing
errors support consistency rather than high accuracy at that resolution.
Persistent observation-error learning uses a fixed three-update schedule and keeps
held-out data out of training and step selection:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I examples/weather_scenarios/fv_persistent_learning_probe.py --checkpoint /tmp/advar-fv-learning.pt --output /tmp/advar-fv-learning.json
```

The saved 240×240 exact adjoint covers all 18 forecast leads, with actual relative
residual 4.28e-11. Its finite-reanalysis check is separate and still in progress.
The large response probe requires the locally retained `rotation240_refined.pt`;
large field checkpoints and generated HTML are deliberately not Git artifacts.
CI and deployment checks are deferred until algorithm completion. Real-data and
operational validation belong to Phase 2.
