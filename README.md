# ADVAR 3-frame radar nowcast v0.21

`main`과 pull request는 GitHub Actions에서 Python 3.10·3.12 CPU 전체
시험을 실행하고, Python 3.12 환경에서 product source basedpyright를
검사한다. 별도 package job은 sdist와 wheel을 빌드한 뒤 격리 환경에 wheel을
설치하여 `advar-nowcast` CLI와 NPZ 출력계약을 smoke-test한다.

10분 간격 레이더 dBZ 3장으로 다음 3시간을 10분 간격으로 예측하는
작고 해석 가능한 matrix-free 변분 구현이다. 기존 FFT 기준예측은 항상
수치 분석 실패 시 fallback으로 유지한다. 단, 관측과 이전 주기 배경이
모두 없으면 맑음장을 만들지 않고 `UNAVAILABLE`을 반환한다.

## 모델

입력 시각은 `[-20, -10, 0]분`, 출력 시각은 `[+10, ..., +180]분`이다.

1. 실제로 가용한 프레임 쌍에서 phase correlation으로 전역 이동량을 추정한다.
   peak 주변을 제외한 sidelobe 대비 peak의 분리도(PSR)가 연구 기본값
   `8.0`보다 낮으면 그 pair는 경향 추정에서 fail-close한다.
2. 이전 에코를 이동시킨 뒤 겹치는 영역의 에코 적분비로 로그 성장률을 추정한다.
3. 각 선행시간을 최신 에코에서 국지적 양성 보존 remap으로 직접 이류한다.
4. 성장률은 60분 시간규모로 감쇠시켜 장시간 폭주를 막는다.

각 선행시간에 `h × 이동량`을 한 번만 적용한다. 18번 재귀 보간하지
않으므로 긴 선행시간에서 생기는 불필요한 수치 확산을 줄인다.
FFT는 phase-correlation 이동량 추정에만 사용하며, 분석·예측 전이에는
사용하지 않는다.

내부 에코량은 다음처럼 양수 선형 공간에서 계산한다.

```text
q = 10 ** (min_dBZ / 10) * expm1(ln(10) / 10 * (dBZ - min_dBZ))
```

`q >= 0`은 수송·반응·변분분석 전체의 물리 상태 불변조건이다. 의미 있는
음수나 NaN/Inf는 `EchoPositivityError`로 fail-close하며, dtype 반올림
크기의 미세 음수만 0으로 보정한다. 검사는 입력, LM trial 수용, 최종
분석·예측 경계에서만 수행하며 JVP·VJP·HVP 내부에는 들어가지 않는다.
물리 상태에만 적용하며 JVP·VJP·HVP와 상태증분의 부호는 제한하지 않는다.
고정 remap 셀과 실제 이동량이 맞지 않으면
`FrozenCellMismatchError`로 해당 연산을 거부한다.

전이는 선형 에코 `q`를 네 개의 인접 목적 셀에 비음수 가중치로 보낸다.
영역 안에서는 가중치 합이 1이므로 에코 적분을 보존하고, 영역 밖 성분은
경계 유출로 버린다. 따라서 fractional 이동에서도 먼 위치에 양의
유령에코가 생기지 않는다.

정수 셀 경계에서 remap은 연속이지만 미분은 구간별로 정의된다. 변분분석은
각 LM 외부 반복에서 두 분석시각의 정수 remap 셀을 고정하여 PCG가 같은
JVP/VJP/GN-HVP 연산자를 사용하게 한다. trial은 실제 셀로 다시 평가하고,
목적함수가 감소하는 경계 횡단만 수용한다. 최종 dBZ 상한의 활성집합도
외부 반복에서 고정한다.

## 설치와 실행

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Python API:

```python
import numpy as np
import torch

from advar import NowcastConfig, nowcast

frames = torch.from_numpy(np.load("three_frames.npy")).float()  # [3, H, W]
result = nowcast(frames)

print(result.forecast_dbz.shape)          # [18, H, W]
print(result.state.displacement_yx)       # (dy, dx), pixel / 10 min
print(result.state.log_growth_per_step)   # log growth / 10 min
print(result.metadata.data_status)
```

실제 자료에서는 시각·격자 계약을 함께 전달한다.

```python
from advar import RadarGridTimeContract

grid_time = RadarGridTimeContract(
    valid_times=(
        "2026-07-31T00:00:00Z",
        "2026-07-31T00:10:00Z",
        "2026-07-31T00:20:00Z",
    ),
    dx_m=1000.0,
    dy_m=1000.0,
    projection="EPSG:5179",
    grid_hash="<lowercase SHA-256 grid identity>",
    pixel_to_projected_matrix_m=(
        (1000.0, 0.0),
        (0.0, -1000.0),
    ),
)
result = nowcast(
    frames,
    NowcastConfig(maximum_motion_speed_mps=40.0),
    grid_time_contract=grid_time,
)
```

시각은 UTC로 canonicalize되며 세 간격이 `interval_minutes`와 정확히
일치해야 한다. 배경을 사용하면 같은 세 시각의 `background_valid_times`와
최신 `background_age_minutes`가 일치해야 한다. affine matrix는 배열
`(column, row)` 증분을 투영좌표 `(x, y)` metre 증분으로 변환하며, 생략하면
north-up `((dx, 0), (0, -dy))`를 사용한다. `maximum_motion_speed_mps`를
설정하면 P0 raw motion과 P1 후보를 이 투영좌표 속도에서 fail-close한다.
affine의 정규화 determinant가 `0.01`보다 작거나 2-norm condition number가
`1000`보다 크면 거의 평행하거나 지나치게 왜곡된 격자로 보고 거부한다.
`maximum_background_age_minutes`를 넘는 배경은 거부한다. 계약을 생략한
호출은 합성·연구용 index-time/pixel-grid 경로로 유지되며 실제 레이더
운영자료의 물리 provenance를 주장하지 않는다.

발행 프로세스가 종료된 뒤에도 동일한 실행계약으로 M0를 계산하려면
forecast-run artifact를 저장한다.

```python
from advar import (
    compute_sensitivity_snapshot_from_run,
    load_forecast_run,
    save_forecast_run,
)

save_forecast_run(result, "forecast-run.npz")

# 미래 검증자료가 도착한 별도 프로세스
result = load_forecast_run("forecast-run.npz")
result.validate_issuance()
snapshot = compute_sensitivity_snapshot_from_run(result, verification_dbz)
```

artifact는 발행장·유효영역, 현재 에코상태, source support, 실제
`NowcastConfig`, 최신 관측장·수용 mask·배경장과 각 digest를 함께 저장한다.
세 관측장·세 실제 수용 mask·배경 전체·배경 age는 하나의
`input_bundle_digest`로 묶고, 상태와 발행 결과까지 포함한
`forecast_run_digest`로 정확한 실행 identity를 고정한다.
P1 실행은 실제 `AnalysisConfig` JSON과 observation-std·quality-weight의
analysis input digest도 같은 identity에 포함한다.
재적재 전 ZIP member 수·개별/전체 압축해제 크기·이름, NPY header의
dtype·shape·선언 payload를 검사하고 알 수 없는 member와 object dtype을
거부한다. 기본 한도는 160개, member당 1 GiB, 전체 2 GiB이며 API 인자로
더 낮출 수 있다. 재적재 시 모든 member를 묶는 artifact digest와
tensor/config/state/metadata digest를 독립적으로 재계산한다. 각 member는
archive에서 한 번만 materialize하며 같은 NumPy storage를 digest
검증과 반환 Tensor 구성에 재사용한다. 연속 수치배열은 복사 없는 buffer로,
비연속 수치배열은 1 MiB 이하 C-order chunk로 digest한다.
선택적인 positivity/transport audit 객체는 M0 재현에
필요하지 않으므로 `load_forecast_run()` 결과에서는 `None`이다.

배경 사용 provenance는 현재 상태 support와 경향 초기화를 분리한다.
`background_state_support_fraction`은 현재 에코상태에서 배경 support가
차지한 비율이고, `background_tendency_used`는 이동·성장률 추정이 배경
pair에 의존했는지를 나타낸다. `background_used`는 두 경로의 논리합이며,
어느 경로에서든 배경을 사용하면 `background_age_minutes`를 보존한다.

세 관측시각을 함께 분석하려면 다음처럼 사용한다.

```python
from advar import variational_nowcast

forecast, analysis = variational_nowcast(
    frames,
    observation_std_dbz=2.0,
)

print(analysis.used_fallback)
print(analysis.initial_objective, analysis.final_objective)
print(analysis.state.echo_linear)  # 세 장으로 분석된 현재 q(0)
```

실제 격자에서는 `AnalysisConfig(causal_support_uncertainty_m=...,
amplitude_displacement_tolerance_m=...)`로 causal envelope와 진폭 위치허용을
metre 단위로 지정할 수 있다. 두 값은 `RadarGridTimeContract`의 축 간격을
사용해 실제 투영거리 안에 있는 integer `(row, column)` offset만 선택한다.
따라서 1 km 정방격자의 1 km tolerance는 축방향 이웃을 포함하지만 1.414 km
대각 이웃은 제외한다. 필요한 pixel bound가 분석격자보다 크면 거부하며,
격자계약 없이 물리거리 설정을 사용해도 거부한다.

P1 제어벡터는 다음 하나뿐이다.

```text
[a_q(-20분, |S_control|), a_motion_1, a_motion_2, a_log_growth]
```

`a_q`는 고정 control support의 활성 격자만 포함하며
`analysis.active_field_index`가 각 값을 원래 `H×W` 격자의 flat index로
연결한다. dBZ latent의 softplus 좌표에서 양의 선형 에코로 변환하고,
support 밖 제어변수는 PCG 벡터에 만들지 않는다. 무에코 영역은 고정
support mask로 잠가 레이더 세 장만으로 신규 에코를 만들지 않는다.
연구용 기본 경로의 운동성분은 `(row, column)` pixel 증분이지만,
`motion_increment_scale_mps`와 격자계약을 제공하면 두 성분은 projected
`(x, y)` m/s 증분이다. 두 성분을 하나의 radial `tanh` speed-ball로 decode하여
모든 finite control이 원형 물리속도 상한 안에 매끄럽게 머문다. 운용모드는 이
물리 제어를 강제하고 affine 역변환으로 수송코어의 `(row, column)`
displacement를 만든다. 실제 좌표계와 제약형태는
`analysis_motion_control_coordinate_system`에 기록한다.
baseline 운동 또는 성장률이 hard bound에 정확히 닿으면 zero control은 그 값을
정확히 보존하고 outward update는 투영으로 막되, 관측이 지지하는 inward update는
허용한다. saturation margin은 baseline에서 0으로 기록한다.
다만 분석창 후반에 탐지된 에코는 baseline 운동으로
초기시각에 역수송하고, 초기 관측 또는 배경 anchor가 있는 위치만 2 pixel
범위에서 precursor control로 연다. 이 확장영역은 운동오차를 허용하는
control envelope일 뿐 precursor 초기 추정은 아니다. 탐지한계 바로 아래의
warm start는 역이류 core와 초기 anchor가 직접 겹치는 causal seed에만
적용하므로, 주변 envelope는 zero control에서 시작한다. seed는 배경을
바꾸지 않으며 제어 prior 비용도 그대로 부담한다. 모든 분석시각의 탐지
에코에는 최소 0.25의 control
reachability를 요구하고, 그 최소 여유를
`analysis.minimum_reachability_margin`에 기록한다. 초기 탐지 에코의 정상
수송으로 설명되지 않는 -10분 및 최신 에코는 3×3 근린의 분석 최댓값과
비교한다. `quality_weight`는 관측 precision multiplier로 사용한다.
quality-weighted 표준화 deficit이 3을 넘거나 에코가 precursor floor
아래인 정보가중 비율을 -10분과 0분에 각각 계산하고,
`analysis.unresolved_amplitude_fraction_by_time`에 기록한다. 기존 scalar
`analysis.unresolved_amplitude_fraction`은 두 시각 값의 최댓값이다.
같은 시각 안에서 작은 대류셀 실패가 큰 정상영역에 희석되지 않도록
precursor-required mask를 8-연결 객체로 나누고, 객체 수·정보부족 객체 수,
객체별 unresolved fraction의 최댓값과 integrated-echo·soft-area 비율의
최솟값/최댓값도 기록한다. 객체 라벨링과 공간적 비율 계산은 최종 결과
경계에서만 수행하므로 LM 후보의 JVP/VJP/HVP 경로에는 들어가지 않는다.
단, precursor 관측의 총 quality weight가 기본 0.01 미만이거나 유효
정보화소 수가 기본 1.0 미만이면 해당 시각은 hard amplitude gate에서
제외하고 `analysis.amplitude_information_sufficient_by_time`과
`analysis.insufficient_amplitude_information`에 명시한다. raw 시간별
fraction과 quality weight는 감사용으로 그대로 보존하며, 이 경우 수용된
분석도 기본 연구정책 `amplitude_information_policy="research_degraded"`
에서는 `degraded=True`로 표시한다. 운용정책
`"operational_fallback"`에서는 precursor 진폭정보가 부족한 P1을 발행하지
않고 `insufficient_amplitude_information`으로 P0에 복귀한다. CLI에서는
`--amplitude-information-policy`로 이 정책을 선택한다. 정보가 충분한
시각의 최댓값이 연구용 fail-close 기본값 1%를 넘으면
`unresolved_growth_or_emergence`로 기준예측에 복귀한다. 이 1%는 운용
임계값이 아니며 정보량 하한과 함께 실제 hindcast에서 보정해야 한다.
에코 적분비와 soft echo area 비는 각각 설정된 상·하한을 모두 검사한다.
급격한 established-echo 성장과 함께 하나라도 벗어나면
`amplitude_confidence_failed=True`가 된다. 연구정책
`amplitude_confidence_policy="research_degraded"`는 분석을 degraded로
보존하고, 운용정책 `"operational_fallback"`은
`amplitude_confidence_failure`로 P0에 복귀한다. 기존
`maximum_latest_detected_error_std` 생성자 인자는 0.4 API 호환을 위해
유지되며, 현재는 두 후속 분석시각 모두에 적용된다.

초기 탐지 에코에서 기하학적으로 도달 가능한 후속 에코도 별도 성장능력
진단을 받는다. 초기 관측의 quality-aware 3-sigma 상한을 분석 이동량과
`max_log_growth_per_step`으로 수송한 뒤, 후속 관측이 그 envelope를 넘는
정보가중 비율과 최대 선형에코 비를 각각
`established_echo_excess_growth_fraction_by_time`과
`maximum_growth_envelope_ratio_by_time`에 기록한다. 이 값은 현재 모델의
전역 성장률 상한으로 설명하기 어려운 established echo를 찾는 진단이며,
실제 hindcast 보정 전에는 hard fallback 조건으로 사용하지 않는다.

warm start는 solver의 출발점일 뿐 수용 기준은 아니다. P1은 zero-control
목적함수를 `initial_objective`로 고정하고, 최종 제어가 이를 수치
허용오차보다 명확히 낮출 때만 분석을 발행한다. fallback이면 반환 제어는
zero이고 `final_objective == initial_objective`다. amplitude 조건을
위반한 warm start에서는 quality로 한 번 백색화한 연속 초과량 제곱 점수의
`(시간별 최댓값, 시간별 합)`을 사전식으로 줄이는 LM 후보를 허용하고,
최종 안전판정에는 시간별 discrete fraction의 최댓값을 사용한다. 작은
float32 violation에도 절대 1 기준 허용오차를 적용하지 않는다. 일단
feasible해지면 다시 infeasible 영역으로 나갈 수 없다. 시간별 선형 에코
적분비, displacement-tolerant soft echo area 비, quality scale에 불변인
유효 정보화소 수, bad/total quality weight는 분석 진단으로 기록한다.
적분비나 면적비가 연구 기본 하한 0.5보다 작거나 established echo의
초과성장 비율이 1%보다 크면 분석은 `degraded=True`가 되지만 hard fallback은
하지 않는다. 이 신뢰도 임계값은 운용값이 아니며 실제 hindcast에서 반드시
보정해야 한다. 진단이 반환된 분석을 평가했는지, 폐기된 후보를
평가했는지는 `analysis.amplitude_diagnostics_source`로 구분한다. causal
envelope의 제어 셀 수, 실제 seed 셀 수, seed prior 비용도 각각
`causal_control_cell_count`, `causal_seed_cell_count`,
`causal_seed_prior_cost`로 기록한다.

active initial-field control에는 기본 `field_smoothness_weight=0.01`의
1차 차분 prior를 추가한다. 이 residual은 active support 안에서 서로 맞닿은
상하·좌우 셀 사이에만 존재하므로 결측 또는 비활성 경계를 가로질러 제어값을
연결하지 않는다. 해당 비용은 `analysis_field_smoothness_prior_cost`에
기록되며 control의 독립 unit prior와 함께 목적함수·JVP·VJP·GN-HVP에 같은
형태로 포함된다. active edge와 local control index는 분석 준비단계에서 한 번
계산하므로 hot path에서 전체 `H×W` field를 다시 만들지 않는다. 격자계약이
있으면 affine cell area와 축 길이에서 얻은 graph metric 가중치를 적용하여
직교 비등방 격자에서 수평 `dy/dx`, 수직 `dx/dy`가 된다. 이는 dBZ장 자체가
아니라 standardized field-control graph prior이며 좌표계는
`analysis_field_smoothness_coordinate_system`에 기록한다. 현재 물리 graph
smoothness는 직교 projected grid에만 정의되며, 비직교 affine에서는 이를
끄지 않으면 분석을 fail-close한다.

세 관측잔차는 다음 순서로 정확히 한 번 처리한다.

```text
detected 또는 censored residual
→ sqrt(quality_weight) / observation_std_dbz
→ 외부 반복에서 고정한 pseudo-Huber IRLS weight
```

잔차벡터에는 표준화된 제어 prior도 그대로 포함한다. 따라서
Gauss–Newton HVP는 `J.T @ (J @ v)`로 계산되고, LM 증분은 PCG로 푼다.
각 외부 반복은 VJP pullback을 한 번만 만들고 PCG의 각 `J.T @ (J @ v)`에
재사용한다. PCG의 수렴 여부와 `relative_residual`은 반환 전에 다시 계산한
실제 `b - A @ x`로 확인하며, 재귀잔차가 먼저 수렴했지만 실제잔차가 남으면
Krylov recurrence를 재시작한다. 행렬이나 Jacobian은 생성하지 않는다.
수용된 분석에는 최종 IRLS 선형화점의 관측 Jacobian만 사용하여
표준화된 dynamics 3변수의 자료 Gram `G=J_dyn.T @ J_dyn`,
regularized Hessian `G+I`의 고유값·조건수와, 초기장–성장 및
초기장–이동 Jacobian 절대 cosine을 진단으로 기록한다. 이 값은 아직
hindcast로 보정된 발행 gate가 아니며 분석 수용 여부를 바꾸지 않는다.
자료 고유값별 posterior precision 기여율 `lambda/(1+lambda)`와 그 합인
`dynamics_data_effective_dimension`도 기록한다. 기존 effective-rank 이름은
수치 rank 호환필드이며 새 `dynamics_data_numerical_rank`가 정확한 명칭이다.
첫 분석 증분 전 PCG 실패,
비유한값, 수용할 수 없는 증분은 FFT 기준예측으로 복귀한다. 한 번이라도
목적함수를 낮춘 분석이 수용된 뒤 후속 반복이 실패하면 그 최선 분석을
`degraded=True`로 보존한다.

결측, QC 탈락, 관측된 무에코는 서로 다른 mask로 보존한다. 결측 또는
QC 탈락 화소는 관측잔차에 들어가지 않는다. 운동과 성장률은 실제로
가용한 시각쌍에서만 추정한다. 가운데 영상이 없으면 `-20→0분` 추정량을
20분 간격으로 정규화하고, 한 시각만 가용하면 zero-motion,
zero-growth persistence를 사용한다. 활성 에코 주변에 결측 또는 QC
탈락이 하나라도 있는 시각쌍은 사용하지 않는다. 부분 관측장을 현재시각으로
옮길 때는 에코와 support를 함께 수송하고 support로 정규화하므로,
fractional 이동한 결측 경계가 인공적인 에코 소멸로 바뀌지 않는다.
관측 pair를 먼저 사용하고 관측 pair가 없을 때만 배경 pair를 사용한다.
현재 상태도 직접·전파 관측을 우선하며, 관측 support가 없는 위치만 배경으로
채운다. 선택적으로 이전 주기의 시간 정렬된 3장 배경을 제공할 수 있으며,
이때 배경 나이는 필수이다.

```python
forecast, analysis = variational_nowcast(
    frames,
    qc_mask=qc_mask,
    background_frames_dbz=previous_cycle_background,
    background_age_minutes=10.0,
)
```

관측이 전혀 없으면 이전 주기 배경을 `STALE_BACKGROUND`으로 사용한다.
배경도 없으면 결과 상태는 `UNAVAILABLE`이고 예측장은 `NaN`이다.
Python API의 `ForecastResult.valid_mask`와
`ForecastMetadata.source_support`는 항상 Tensor이다. 외부 유입 support는
0으로 두므로 이동 후 지원되지 않는 경계는 `forecast_dbz`에서 `NaN`이며,
그 finite 영역은 `valid_mask`와 정확히 같다. 선형 예측은 공개 결과에
중복 저장하지 않고 필요할 때 상태와 순수 물리 코어에서 계산한다.

현재 P1에는 이전 분석주기의 독립 배경장이 없다. 따라서 `Y(-20)`을 초기장
anchor이자 첫 관측으로 함께 사용한다. 이는 완전한 Bayesian 4D-Var가 아니라
세 장만으로 분석 경로와 미분 계약을 검증하는 관측기반 P1이다. 분석창의
`-10/0분` 상태는 `q(-20)`에서 각각 한 번의 직접 warp로 계산하며, 미래
18개 시점은 분석된 `q(0)`에서 시작한다.

CLI:

```bash
advar-nowcast three_frames.npy forecast.npz
advar-nowcast three_frames.npy forecast.npz --variational
advar-nowcast three_frames.npy forecast.npz \
  --qc-mask qc.npy \
  --background previous_cycle.npy \
  --background-age-minutes 10
advar-nowcast three_frames.npy forecast.npz \
  --valid-times 2026-07-31T00:00:00Z 2026-07-31T00:10:00Z 2026-07-31T00:20:00Z \
  --dx-m 1000 --dy-m 1000 --projection EPSG:5179 \
  --grid-hash <lowercase-SHA-256> \
  --pixel-to-projected-matrix-m 1000 0 0 -1000 \
  --maximum-motion-speed-mps 40
advar-nowcast three_frames.npy forecast.npz --audit
```

CLI 기본 `--mode research`는 합성·hindcast 진단용이다. `--mode operational`
은 `--variational`, 완전한 시각·격자 계약, 물리속도 상한, 물리 causal 및
amplitude 거리, projected m/s 운동증분 scale, 명시적인
PSR·pair 운동/성장 불일치·pair 신뢰도 우위·성장 overlap
support·물리면적·관측오차·amplitude
정보량·적분량·면적·성장
임계값, 검증된 상태경로 support 발행 임계값과
`--operational-calibration-id`를 모두 요구한다. 누락된 보정값이
있으면 실행 전에 거부하며, 두
amplitude 정책을 모두 `operational_fallback`으로 고정한다. 보정값은 실제
레이더 hindcast에서 얻어야 하며 저장된 `nowcast_config_json`,
`analysis_config_json`과 각 digest에 포함된다.

출력 `forecast.npz`에는 다음 항목이 들어간다.

- `output_contract_version`: 현재 `nowcast-npz-v30`
- `forecast_run_artifact_version`: 현재 `forecast-run-v22`
- `forecast_run_digest`, `input_bundle_digest`
- `grid_time_contract_json`, `grid_time_contract_digest`
- `run_background_age_minutes`: 실제 입력계약의 배경 age
- `displacement_yx`: `(row, column)` pixel/step
- `grid_velocity_mps_yx`, `displacement_mps_yx`: 호환용 grid-axis
  `(row, column)` m/s
- `projected_velocity_mps_xy`: affine 계약을 적용한 projected `(x, y)` m/s
- `analysis_config_json`, `analysis_config_digest`, `analysis_input_digest`
- `forecast_dbz`: `[18, H, W]`
- `valid_mask`, `state_echo_linear`, `source_support`,
  `path_verified_source_support`, `verified_source_support`,
  `observation_verified_source_support`,
  `background_verified_source_support`, `forecast_path_verified_support`,
  `forecast_verified_support`
- `source_support`는 상태가 정의됐는지를,
  `path_verified_source_support`는 국지 에코 위치경로가 검증됐는지를,
  `verified_source_support`는 성장증거까지 있는 상태경로인지를 나타낸다.
  관측과 배경의 엄격한 evidence support도 별도로 보존한다.
  연구모드에서는 예전 persistence를 유지하지만, 운영모드는
  `minimum_publish_verified_support`로 검증되지 않은 경로를 발행에서
  제외한다. `forecast_path_verified_support`와
  `forecast_verified_support`는 각 support를 선행시간별로 수송한
  진단이다.
- `nowcast_config_json`, `nowcast_config_digest`
- `latest_frame_dbz`, `latest_observation_mask`, `latest_background_dbz`와
  최신 입력·배경 digest
- `forecast_dbz_digest`, `valid_mask_digest`, `state_metadata_digest`,
  `forecast_run_artifact_digest`
- `lead_minutes`: `[10, 20, ..., 180]`
- `displacement_yx`: 10분당 픽셀 이동량
- `log_growth_per_step`: 10분당 로그 성장률
- `motion_disagreement_px`: 두 인접 가용쌍의 이동 추정 불일치. 가용쌍이 하나 이하면 `0`
- `growth_disagreement`: 두 인접 가용쌍의 성장 추정 불일치. 가용쌍이 하나 이하면 `0`
- `minimum_phase_correlation_psr`: 실제 경향 추정에 사용한 pair들의 최저
  peak-to-sidelobe ratio. 가용 pair가 없으면 `NaN`. 기본 임계값 `8.0`과
  peak 제외반경 2 pixel은 합성·연구 설정이며 실제 hindcast로 보정해야 한다.
- `tendency_pair_count`: 운동 또는 성장에 실제 사용한 독립 pair의 합집합 크기
- `motion_pair_count`, `growth_pair_count`: 각 성분에 실제 사용한 pair 수
- `motion_pair_selection`, `growth_pair_selection`: 각 성분의 독립 결합 결과.
  `NONE`, `SINGLE`, `LONG`, `BLENDED`, `EARLIER`, `RECENT`,
  `PERSISTENCE` 중 하나
- `motion_pair_conflict`, `growth_pair_conflict`: 두 pair 추정이 각 성분에서
  서로 모순됐는지 나타내는 관측 가능한 provenance. 셀 분열·병합, 가속·회전,
  다중 객체 충돌 같은 기상학적 원인을 확정하는 라벨은 아니며, 실제 처리 결과는
  대응하는 `*_pair_selection`과 함께 해석한다.
- `tendency_source`: 경향 추정 출처. `OBSERVATION`, `BACKGROUND`,
  `NONE` 중 하나
- `dynamics_source`: 최종 발행상태 동역학의 출처.
  `P0_RECONSTRUCTION`, `P1_VARIATIONAL`, `P0_FALLBACK` 중 하나
- `state_path_source`, `state_path_mode`, `state_path_pair_count`,
  `state_path_minimum_psr`, `state_path_conflict`,
  `state_path_extrapolated`, `state_path_age_minutes`: 현재상태 재구성
  경로의 출처·선택·신뢰도·최대 source age. 관측과 배경이
  함께 기여하면 `state_path_source` 는 우선순위가 높은 관측을
  기록하는 호환용 요약이다.
- `observation_state_support_fraction`, `background_state_support_fraction`,
  `observation_path_*`, `background_path_*`: 혼합 상태의 실제 기여비와
  source별 mode·pair count·PSR·conflict·extrapolation·age를 동시에
  보존한다. 관측 1%+배경 99%도 관측 경로 하나로 축약되지 않는다.
  accepted P1에서는 이 P0 재구성 증거를 최종상태
  provenance로 재사용하지 않고 unavailable로 기록한다.
- `minimum_growth_overlap_support`, `minimum_growth_overlap_area_km2`:
  선택된 운동에서 실제 성장률 결합에 사용된 pair들의 최소
  에코 관련 fractional-overlap support와 물리면적. 동일한 support
  weight를 이전·현재 에코 적분에도 적용하며, 사용 가능한 성장 증거가
  없으면 `NaN`이다.

직접 발행과 `forecast-run` 재적재는 같은 중앙 의미검증을 사용한다. P0와
P0 fallback에서 `growth_pair_count=0`이면 성장 overlap은 반드시 `NaN`이고,
성장 pair를 사용했다면 support와 사용 가능한 물리면적 증거가 설정 임계값을
만족해야 한다. `background_used`도 상태 기여 또는 배경 tendency 사용 여부와
일치해야 하며, 사용 시 run contract의 배경 age와 같은 값을 기록한다.

Phase-correlation의 raw peak가 `max_displacement_px` 범위 밖이거나 허용
search boundary bin에 있으면 높은 PSR이어도 사용하지 않는다. pair 일관성은
검색상한과 분리된 `maximum_pair_motion_disagreement_px`, 격자계약이 있을 때의
`maximum_pair_velocity_disagreement_mps`, 그리고
`maximum_pair_growth_disagreement`로 검사한다. 운동과 성장은 독립적으로
결합한다. 일관된 pair는 PSR과 recency를 함께 가중해 평균하고, 충돌할 때 한
pair의 PSR이 `minimum_pair_psr_advantage` 이상 우세하면 그 pair만 사용한다.
우위가 없으면 해당 성분만 persistence로 fail-close한다. 이 네 기본 임계값은
합성·연구 설정이며 실제 레이더 hindcast로 보정해야 한다.

인접 pair가 하나만 유효하고 20분 long pair도 유효하면 두 후보를 독립적으로
검사한다. near-echo completeness가 이미 에코 주변 자료 가용성을 검사하므로
에코와 무관한 전체 도메인 coverage는 confidence에 곱하지 않는다. 신뢰도는
PSR에 long pair의 시간간격·형태변형 위험을 나타내는
`long_pair_confidence_penalty`만 곱한다. long 또는 adjacent 후보가
`minimum_pair_confidence_ratio` 이상 우세할 때만 교체하고, 충돌하면서 우위가
없으면 해당 성분은 persistence로 fail-close한다. long pair는 adjacent pair와
관측을 공유하므로 두 값을 독립 표본처럼 평균하지 않는다. `operational`
profile에서는 span penalty와 confidence ratio를 모두 hindcast로 보정해 CLI에
명시해야 한다.
- `min_publish_support`: 유한한 예측값을 발행하는 최소 source support
- `data_status`: `OBSERVED`, `PARTIAL`, `STALE_BACKGROUND`,
  `UNAVAILABLE` 중 하나
- `coverage_by_frame`: 입력 세 시각의 관측 coverage
- `background_used`, `background_state_support_fraction`,
  `background_tendency_used`, `background_age_minutes`
- `background_contribution_fraction`: 호환성을 위해 유지되는
  `background_state_support_fraction`의 기존 이름
- `analysis_converged`, `analysis_degraded`, `analysis_used_fallback`
- `analysis_unresolved_amplitude_fraction_by_time`,
  `analysis_amplitude_violation_score_by_time`: `[-10분, 0분]`
- `analysis_integrated_echo_ratio_by_time`,
  `analysis_displacement_tolerant_soft_echo_area_ratio_by_time`:
  precursor 영역의 공간 폐합 진단
- `analysis_effective_precursor_pixel_count_by_time`,
  `analysis_bad_quality_weight_by_time`,
  `analysis_total_quality_weight_by_time`
- `analysis_amplitude_information_sufficient_by_time`,
  `analysis_insufficient_amplitude_information`
- `analysis_precursor_object_count_by_time`,
  `analysis_insufficient_amplitude_object_count_by_time`: 8-연결 precursor
  객체 수와 amplitude 정보량 기준에 미달한 객체 수
- `analysis_maximum_object_unresolved_fraction_by_time`,
  `analysis_minimum_object_integrated_echo_ratio_by_time`,
  `analysis_maximum_object_integrated_echo_ratio_by_time`,
  `analysis_minimum_object_soft_echo_area_ratio_by_time`,
  `analysis_maximum_object_soft_echo_area_ratio_by_time`: 작은 객체 실패가
  전체 시간별 비율에 희석되지 않도록 보존한 최악값. unresolved fraction은
  원 객체별로 계산하고, 에코 적분·면적은 물리 tolerance footprint가 겹치는
  객체를 하나의 matching group으로 묶어 같은 예측 에코를 중복 귀속하지 않는다.
  초기 established echo에서 도달 가능한 예측량도 precursor group 분자에서
  제외하므로 인접한 기존 에코가 신규 객체를 대신 설명할 수 없다.
- `analysis_amplitude_confidence_failed`: 적분량·soft area의 양방향 한계 또는
  established-echo 성장·객체별 신뢰도 한계를 벗어났는지 여부
- `analysis_established_echo_excess_growth_fraction_by_time`,
  `analysis_maximum_growth_envelope_ratio_by_time`: 초기 established echo의
  최대 성장능력 envelope 진단
- `analysis_amplitude_diagnostics_source`: `returned_analysis`,
  `rejected_candidate`, `unavailable` 중 하나
- `analysis_relative_objective_reduction`: zero-control P1 목적함수 대비 감소율
- `analysis_dynamics_data_gram_eigenvalues`,
  `analysis_dynamics_data_information_trace`,
  `analysis_dynamics_data_numerical_rank`: 최종 IRLS 선형화점에서 prior를
  제외한 관측 Jacobian Gram의 정보량과 수치 rank
- `analysis_dynamics_data_to_prior_ratio_by_mode`,
  `analysis_dynamics_data_effective_dimension`: mode별
  `lambda/(1+lambda)`와 그 합
- `analysis_regularized_dynamics_hessian_eigenvalues`,
  `analysis_regularized_dynamics_hessian_condition_number`: unit prior를
  포함한 dynamics Hessian의 solver 조건성
- `analysis_field_smoothness_prior_cost`,
  `analysis_motion_control_coordinate_system`,
  `analysis_field_smoothness_coordinate_system`,
  `analysis_motion_saturation_margin_yx`,
  `analysis_motion_speed_saturation_margin_mps`,
  `analysis_growth_saturation_margin`: 공간 prior 비용과 이동·성장 상한까지의
  남은 control margin. 물리속도 상한이 없으면 speed margin은 `NaN`
- `analysis_field_growth_jacobian_cosine`,
  `analysis_field_motion_jacobian_cosine_by_control`: 관측공간에서 초기장
  증분이 성장·현재 motion-control 좌표의 이동증분과 얼마나 유사한지 나타내는
  절대 cosine

`--audit`를 지정할 때만 최종 양성 보정량과 선행시간별
`echo_integral_before_transport`, `echo_integral_after_transport`,
`boundary_outflow_integral`, `echo_budget_error`를 추가한다. audit는 이미
계산한 18개 예측 remap을 재사용하며 예측을 다시 수행하지 않는다.

NPZ는 같은 디렉터리의 임시 파일을 `fsync`하고 원자적으로 교체한 다음
parent directory를 `fsync`한다. 원자교체 이전 기록 실패 시 기존 출력은
그대로 유지되며, directory `fsync` 실패는 내구성을 보장할 수 없으므로
호출자에게 전달한다.

## 의도적으로 제한한 부분

기본 CLI 경로는 동작과 미분 계약을 검증한 P0 기준시스템이다.
`--variational` 또는 `variational_nowcast()`가 P1 분석 경로다.

- 전역 이동장 하나만 사용하므로 회전·변형·서로 다른 세포 이동을 표현하지 못한다.
- 전역 성장률 하나만 사용하므로 국지적 발생·소멸을 예측하지 못한다.
- 경계 밖 에코 유입 정보가 없으므로 경계는 0으로 둔다.
- 결정론적 예측이며 불확실성은 가용쌍의 추정 불일치만 진단한다.
- 3시간 동안 새로 발생하는 대류는 외삽만으로 예측할 수 없다.

현재 P1도 전역 이동·성장만 사용한다. 저해상도 운동장, 성장률장,
보존형 flux 적분기, 약제약 모델오차, 뉴럴 prior는 아직 추가하지 않았다.

## M0 민감도 사례 원장

미래 검증 레이더 18장이 도착하면 다음 값을 계산해 하나의 불변 사례로
저장할 수 있다.

- 선행시간·지표별 제어 민감도
  `dE / d[dy, dx, log_growth]`: `[18, metric, 3]`
- 30·60·120·180분의 예측장 민감도
- 최신 입력 영상의 고정 제어 직접 민감도 `dE / d(dBZ)`
- 16×16 타일별 직접 민감도 크기와 innovation 영향
- 결측·자료출처·물리격자·상태 재구성 경로·성장 overlap을
  분리한 상황 특징 70개, 선형성 신뢰도, 계약 해시
- 민감도를 생성한 정확한 발행 실행의 `forecast_run_digest`

민감도 점수는 검증 유효영역과 실제 발행 유효영역의 교집합에서만
계산하며, 발행된 dBZ 상한과 같은 고정 활성집합을 사용한다.
검증장이 없거나 FSS·객체중심을 정의할 에코가 없으면
`metric_available=False`와 `NaN`으로 기록하며, 0 오차로 해석하지 않는다.
background의 비유한값과 QC 탈락 화소도 innovation 영향에서 제외한다.
innovation·whitening·impact·reward가 없으면 해당 배열은 저장하지 않는다.
기본 지표는 `log_echo_mse`, `soft_fss_error_35`(`1-softFSS`),
`centroid_error`이며 모두 작을수록 좋다.

```python
from datetime import datetime, timezone

from advar import (
    EpisodeLedger,
    ModelContract,
    SensitivityEpisode,
    compute_sensitivity_snapshot,
    nowcast,
)
from advar.physics import FORECAST_INTEGRATOR_VERSION

result = nowcast(
    frames,
    background_frames_dbz=background_frames,
    background_age_minutes=10.0,
)

# 이 값들은 +180분까지 미래 관측이 도착한 뒤에만 사용한다.
snapshot = compute_sensitivity_snapshot(
    frames[-1],
    result,
    verification_frames_dbz,       # [18, H, W]
    latest_background_dbz=background_frames[-1],
    observation_std_dbz=2.0,
)

contract = ModelContract(
    model_commit="working-tree",
    residual_contract_version="none-v1",
    forecast_metric_version="issued-domain-metrics-v2",
    observation_contract_version="direct-latest-dbz-active-set-v2",
    forecast_integrator_version=FORECAST_INTEGRATOR_VERSION,
    grid_geometry_version="my-grid-v1",
    radar_qc_version="my-qc-v1",
    nowcast_config_digest=snapshot.nowcast_config_digest,
    sensitivity_config_digest=snapshot.sensitivity_config_digest,
    grid_time_contract_digest=snapshot.grid_time_contract_digest,
)
episode = SensitivityEpisode(
    episode_id="20260726T120000Z-radar-a",
    issue_time=datetime.now(timezone.utc).isoformat(),
    radar_id="radar-a",
    contract=contract,
    snapshot=snapshot,
)
EpisodeLedger("memory").append(episode)
```

원장은 다음처럼 구성된다.

```text
memory/
├── index.sqlite
└── episodes/
    └── <episode_id>/
        ├── manifest.json
        ├── sensitivity_arrays.npz
        └── checksums.json
```

SQLite 커밋이 완료된 사례만 조회할 수 있다. manifest와 NPZ의 SHA-256은
SQLite에도 고정되며, 배열 이름·shape·dtype까지 검증한다. Pickle은
사용하지 않는다. manifest의 `forecast_run_digest`는 정확히 어느 발행
실행에서 민감도 episode가 생성됐는지를 보존한다. 이 값은 episode
identity이며, 서로 결합 가능한 경험을 정하는 `ModelContract.digest`에는
포함하지 않는다.
M0 context에는 `motion_pair_conflict`, `growth_pair_conflict`가 각각 0/1로
포함된다. 두 `*_pair_selection`도 `NONE`, `SINGLE`, `LONG`, `BLENDED`,
`EARLIER`, `RECENT`, `PERSISTENCE`를 성분별 7-way one-hot으로 저장하여 pair
불일치 발생과 실제 처리결과를 분리해 hindcast·학습할 수 있다. 그 뒤에는
`phase_correlation_psr_available`과
`log1p_minimum_phase_correlation_psr`를 저장한다. 가용 pair가 없을 때는
availability=0과 finite value 0을 함께 기록하여 결측 PSR을 실제 0과 구분한다.
격자계약이 있으면 `projected_velocity_available=1`과 투영좌표
`projected_velocity_x_mps`, `projected_velocity_y_mps`, `projected_speed_mps`도
저장한다. 격자계약이 없는 연구사례는 availability=0과 finite 0을 함께 기록해
실제 정지사례와 구분한다. 두 pair를 실제로 비교하고 격자계약이 있을 때는
`motion_disagreement_mps_available=1`과 `motion_disagreement_mps`를 append한다.
비교할 pair나 물리격자 계약이 없으면 availability=0과 finite 0을 기록한다.
격자계약이 있으면 affine determinant로 cell 면적을 계산해
`area_weighted_echo_available=1`과
`log1p_linear_reflectivity_integral_km2`도 저장한다. 동일한 물리영역을 다른
해상도로 표현해도 이 값은 유지된다. 이어서 `grid_spacing_available=1`과
`grid_column_spacing_m`, `grid_row_spacing_m`를 저장하므로 `tile_size`와 함께
각 sensitivity tile의 물리 축 길이를 별도 격자 조회 없이 해석할 수 있다.
격자계약이 없는 연구사례는 availability=0과 finite 0을 저장한다.
sensitivity snapshot은 실제
`grid_time_contract_digest`도 보존하고 `ModelContract`의 같은 필드와 정확히
일치해야 하므로 서로 다른 affine/grid의 spatial sensitivity가 같은 계약으로
섞이지 않는다. aggregate state path 뒤에는 관측·배경 각각의 pair 수, conflict,
extrapolation, age, minimum PSR도 별도 context feature로 저장한다. 혼합 상태가
하나의 path provenance로 축약되지 않으므로 source별 신뢰도를 나중에 다시
평가할 수 있다. 현재 episode schema는 v14, model-contract hash schema는
v11이며 기존 schema 1–13을 그대로 검증한다. 과거 episode에
존재하지 않던 conflict나 selection 값을 임의로 보간하지 않으므로 서로 다른
context 계약이 같은 학습집합으로 섞이지 않는다.
M0 `trust_components`의 `pair_consistency`는 기본적으로 충돌한 성분 하나당
`SensitivityConfig.pair_conflict_trust_penalty=0.5`를 곱한다. 두 성분이 모두
충돌하면 0.25가 되며, 이 값은 sensitivity config digest에 포함된다. 기본값은
연구용 보수적 prior이고 검색·운용 임계값은 실제 hindcast로 보정해야 한다.
선행시간별 `forecast_confidence`는 검증된 현재 상태 support를 수송한 뒤
`exp(-0.5 * (sigma_v * lead_seconds / length_scale_m)^2)`로 감쇠한다.
`sigma_v`와 길이척도는 각각 m/s와 m 단위이며 sensitivity config digest에
포함된다. 이 값은 보정된 확률이 아니라 명시적인 연구용 evidence score다.
`path_evidence_by_metric`은 각 metric의 `abs(dJ/dforecast)`로 이 confidence를
가중한 값이다. `observation_evidence_by_metric`도 같은 민감도 가중치로 실제
관측 source support의 기여를 계산한다. 따라서 넓은 맑은 영역이 대류코어의
path·observation trust를 면적만으로 압도하지 않는다. 두 배열과 최종 집계값은
episode schema v14에 보존된다.

### M0의 엄밀한 경계

현재 이동·성장 추정은 FFT peak의 이산 선택을 포함한다. 따라서 M0가
계산하는 관측 민감도는 최신 영상이 예측 초기장으로 들어가는
`partial_direct_latest_dbz_fixed_control` 경로뿐이다.
발행되지 않은 예측과 유효한 최신 직접관측이 없는 사례는 거부한다.

- `-20분`, `-10분`: 직접 예측 경로 없음
- `0분`: 고정된 `(dy, dx, log_growth)`에서 직접 dBZ 민감도 제공
- 분석을 통한 간접 민감도: P1 implicit FSO가 아직 연결되지 않아 계산 불가
- 전체 관측 민감도와 FSOI: 계산 불가
- 자동 일반화 기억 승격: 비활성

이 구분은 manifest와 SQLite에 명시된다. 간접 민감도를 0으로 저장해
“효과 없음”으로 오해하게 만들지 않는다. P1은 미분 가능한 잔차와 HVP를
제공하지만 mixed observation VJP와 implicit 수반계는 아직 구현하지 않았다.
또한 P1 분석상태는 M0 직접민감도 API에서 provenance 검사로 거부된다.
