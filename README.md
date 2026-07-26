# ADVAR 3-frame radar nowcast v0.3

10분 간격 레이더 dBZ 3장으로 다음 3시간을 10분 간격으로 예측하는
작고 해석 가능한 matrix-free 변분 구현이다. 기존 FFT 기준예측은 항상
fallback으로 유지한다.

## 모델

입력 시각은 `[-20, -10, 0]분`, 출력 시각은 `[+10, ..., +180]분`이다.

1. 두 프레임 쌍에서 phase correlation으로 전역 이동량을 추정한다.
2. 이전 에코를 이동시킨 뒤 겹치는 영역의 질량비로 로그 성장률을 추정한다.
3. 각 선행시간을 최신 에코에서 zero-padded Fourier shift로 직접 이류한다.
4. 성장률은 60분 시간규모로 감쇠시켜 장시간 폭주를 막는다.

각 선행시간에 `h × 이동량`을 한 번만 적용한다. 18번 재귀 보간하지
않으므로 긴 선행시간에서 생기는 불필요한 수치 확산을 줄인다.

내부 에코량은 다음처럼 양수 선형 공간에서 계산한다.

```text
q = max(10 ** (dBZ / 10) - 10 ** (min_dBZ / 10), 0)
```

예측 제어변수는 진폭 `a = sqrt(q)`로 둔다. 진폭을 매끄럽게 이동한 뒤
`q = a ** 2`로 복원하므로 음의 에코를 만들거나 0에서 잘라낼 필요가 없다.
이동 추정은 이산 peak 선택을 사용하지만, 분석된 상태에서 예측으로 가는
`forecast_from_state()`는 정수 픽셀 이동에서도 PyTorch JVP/VJP/GN-HVP가
가능하다. 최종 dBZ 상한이 활성화되는 지점만 외부 반복에서 고정된
활성집합으로 취급한다.

## 설치와 실행

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Python API:

```python
import numpy as np
import torch

from advar import nowcast

frames = torch.from_numpy(np.load("three_frames.npy")).float()  # [3, H, W]
forecast_dbz, state = nowcast(frames)

print(forecast_dbz.shape)          # [18, H, W]
print(state.displacement_yx)       # (dy, dx), pixel / 10 min
print(state.log_growth_per_step)   # log growth / 10 min
```

세 관측시각을 함께 분석하려면 다음처럼 사용한다.

```python
from advar import variational_nowcast

forecast_dbz, analysis = variational_nowcast(
    frames,
    observation_std_dbz=2.0,
)

print(analysis.used_fallback)
print(analysis.initial_objective, analysis.final_objective)
print(analysis.state.echo_linear)  # 세 장으로 분석된 현재 q(0)
```

P1 제어벡터는 다음 하나뿐이다.

```text
[a_q(-20분, H×W), a_dy, a_dx, a_log_growth]
```

`a_q`는 dBZ latent의 softplus 좌표에서 양의 선형 에코로 변환한다.
무에코 영역은 고정 support mask로 잠가 레이더 세 장만으로 신규 에코를
만들지 않는다. 세 관측잔차는 다음 순서로 정확히 한 번 처리한다.

```text
detected 또는 censored residual
→ sqrt(quality_weight) / observation_std_dbz
→ 외부 반복에서 고정한 pseudo-Huber IRLS weight
```

잔차벡터에는 표준화된 제어 prior도 그대로 포함한다. 따라서
Gauss–Newton HVP는 `J.T @ (J @ v)`로 계산되고, LM 증분은 PCG로 푼다.
행렬이나 Jacobian은 생성하지 않는다. PCG 실패, 비유한값, 수용할 수 없는
증분이 반복되면 QC가 적용된 FFT 기준예측으로 자동 복귀한다.

현재 P1에는 이전 분석주기의 독립 배경장이 없다. 따라서 `Y(-20)`을 초기장
anchor이자 첫 관측으로 함께 사용한다. 이는 완전한 Bayesian 4D-Var가 아니라
세 장만으로 분석 경로와 미분 계약을 검증하는 관측기반 P1이다. 분석창의
`-10/0분` 상태는 `q(-20)`에서 각각 한 번의 직접 warp로 계산하며, 미래
18개 시점은 분석된 `q(0)`에서 시작한다.

CLI:

```bash
advar-nowcast three_frames.npy forecast.npz
advar-nowcast three_frames.npy forecast.npz --variational
```

출력 `forecast.npz`에는 다음 항목이 들어간다.

- `forecast_dbz`: `[18, H, W]`
- `lead_minutes`: `[10, 20, ..., 180]`
- `displacement_yx`: 10분당 픽셀 이동량
- `log_growth_per_step`: 10분당 로그 성장률
- `motion_disagreement_px`: 두 프레임 쌍의 이동 추정 불일치
- `growth_disagreement`: 두 프레임 쌍의 성장 추정 불일치

## 의도적으로 제한한 부분

기본 CLI 경로는 동작과 미분 계약을 검증한 P0 기준시스템이다.
`--variational` 또는 `variational_nowcast()`가 P1 분석 경로다.

- 전역 이동장 하나만 사용하므로 회전·변형·서로 다른 세포 이동을 표현하지 못한다.
- 전역 성장률 하나만 사용하므로 국지적 발생·소멸을 예측하지 못한다.
- 경계 밖 에코 유입 정보가 없으므로 경계는 0으로 둔다.
- 결정론적 예측이며 불확실성은 두 쌍의 추정 불일치만 진단한다.
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
- 상황 특징 15개, 선형성 신뢰도, 모델·관측·지표 계약 해시

민감도 점수는 실제 발행된 dBZ 상한과 같은 고정 활성집합을 사용한다.
검증장이 없거나 FSS·객체중심을 정의할 에코가 없으면
`metric_available=False`와 `NaN`으로 기록하며, 0 오차로 해석하지 않는다.
background의 비유한값과 QC 탈락 화소도 innovation 영향에서 제외한다.
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

forecast_dbz, state = nowcast(frames)

# 이 값들은 +180분까지 미래 관측이 도착한 뒤에만 사용한다.
snapshot = compute_sensitivity_snapshot(
    frames,
    state,
    verification_frames_dbz,       # [18, H, W]
    background_frames_dbz=background_frames,
    observation_std_dbz=2.0,
)

contract = ModelContract(
    model_commit="working-tree",
    residual_contract_version="none-v1",
    forecast_metric_version="metrics-v1",
    observation_contract_version="direct-latest-dbz-active-set-v1",
    forecast_integrator_version="fourier-direct-v1",
    grid_geometry_version="my-grid-v1",
    radar_qc_version="my-qc-v1",
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
사용하지 않는다.

### M0의 엄밀한 경계

현재 이동·성장 추정은 FFT peak의 이산 선택을 포함한다. 따라서 M0가
계산하는 관측 민감도는 최신 영상이 예측 초기장으로 들어가는
`partial_direct_latest_dbz_fixed_control` 경로뿐이다.

- `-20분`, `-10분`: 직접 예측 경로 없음
- `0분`: 고정된 `(dy, dx, log_growth)`에서 직접 dBZ 민감도 제공
- 분석을 통한 간접 민감도: P1 implicit FSO가 아직 연결되지 않아 계산 불가
- 전체 관측 민감도와 FSOI: 계산 불가
- 자동 일반화 기억 승격: 비활성

이 구분은 manifest와 SQLite에 명시된다. 간접 민감도를 0으로 저장해
“효과 없음”으로 오해하게 만들지 않는다. P1은 미분 가능한 잔차와 HVP를
제공하지만 mixed observation VJP와 implicit 수반계는 아직 구현하지 않았다.
또한 P1 분석상태는 M0 직접민감도 API에서 provenance 검사로 거부된다.
