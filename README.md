# ADVAR 3-frame radar nowcast

ADVAR는 운영 배포 시스템이 아니라 레이더 기반 변분 nowcast의 **과학적 실증과
재현 가능한 offline 연구**를 위한 구현이다. 핵심 산출물은 수치 안정성, 입력·target
provenance, 독립 physical-event 평가와 사전등록된 통계 계약이다. 저장소에 남아 있는
bundle 및 activation 도구는 과거 artifact를 읽기 위한 비권위적 engineering
scaffold이며 required PR CI나 과학적 claim의 근거가 아니다. Shadow·canary·
state-advancing LIVE는 승인하지 않는다.

### 현재 계약 capability

아래 표는 `src/advar/_contract_registry.py`에서 생성·검증된다. Runtime registry는
현재 세대와 직전 migration/read 세대만 다루며 장기 archive 전체를 대표하지 않는다.
`∅`는 해당 세대가 없다는 명시적 계약이며, 특히 scientific promotion evidence는
운영 권한으로 해석되지 않는다.
`Code-supported scientific path`는 현재 코드가 해당 타입·세대를 구성하고 읽을 수
있다는 뜻만 가지며 confirmatory, publication 또는 외부 과학 승인 권한을 뜻하지 않는다.

<!-- CONTRACT_CAPABILITY_TABLE:START -->
| Contract family | Current | Predecessor | Issuable | Audit-readable | Code-supported scientific path | Operational |
|---|---|---|---|---|---|---|
| radar_metric_domain_evidence | radar-metric-domain-evidence-v4 | radar-metric-domain-evidence-v3 | radar-metric-domain-evidence-v4 | radar-metric-domain-evidence-v3, radar-metric-domain-evidence-v4 | radar-metric-domain-evidence-v4 | ∅ |
| neural_prior_promotion_evidence | neural-prior-promotion-evidence-v32 | neural-prior-promotion-evidence-v31 | neural-prior-promotion-evidence-v32 | neural-prior-promotion-evidence-v31, neural-prior-promotion-evidence-v32 | neural-prior-promotion-evidence-v32 | ∅ |
| deployed_neural_prior_policy | deployed-neural-prior-policy-v17 | — | deployed-neural-prior-policy-v17 | deployed-neural-prior-policy-v17 | ∅ | ∅ |
| neural_prior_deployment_lineage | neural-prior-deployment-lineage-v19 | neural-prior-deployment-lineage-v18-audit | neural-prior-deployment-lineage-v19 | neural-prior-deployment-lineage-v18-audit, neural-prior-deployment-lineage-v19 | ∅ | ∅ |
| radar_spatial_grid_identity | radar-spatial-grid-identity-v6 | radar-spatial-grid-identity-v5 | radar-spatial-grid-identity-v6 | radar-spatial-grid-identity-v5, radar-spatial-grid-identity-v6 | radar-spatial-grid-identity-v6 | ∅ |
| mosaic_observation_source_registry | mosaic-observation-source-registry-v7 | mosaic-observation-source-registry-v6 | mosaic-observation-source-registry-v7 | mosaic-observation-source-registry-v6, mosaic-observation-source-registry-v7 | mosaic-observation-source-registry-v7 | ∅ |
| radar_observation_geometry | radar-observation-geometry-v7 | radar-observation-geometry-v6 | radar-observation-geometry-v7 | radar-observation-geometry-v6, radar-observation-geometry-v7 | radar-observation-geometry-v7 | ∅ |
| verification_observation_error_plan | verification-observation-error-plan-v16 | verification-observation-error-plan-v15 | verification-observation-error-plan-v16 | verification-observation-error-plan-v15, verification-observation-error-plan-v16 | verification-observation-error-plan-v16 | ∅ |
| verification_bundle | radar-verification-bundle-v22 | radar-verification-bundle-v21 | radar-verification-bundle-v22 | radar-verification-bundle-v21, radar-verification-bundle-v22 | radar-verification-bundle-v22 | ∅ |
| variational_fso | p1-variational-fso-v28 | p1-variational-fso-v27 | p1-variational-fso-v28 | p1-variational-fso-v27, p1-variational-fso-v28 | p1-variational-fso-v28 | ∅ |
| variational_fsoi | p1-linearized-observation-impact-v24 | p1-linearized-observation-impact-v23 | p1-linearized-observation-impact-v24 | p1-linearized-observation-impact-v23, p1-linearized-observation-impact-v24 | p1-linearized-observation-impact-v24 | ∅ |
| semantic_scoring_replay | neural-prior-scoring-replay-bundle-v27 | neural-prior-scoring-replay-bundle-v26 | neural-prior-scoring-replay-bundle-v27 | neural-prior-scoring-replay-bundle-v26, neural-prior-scoring-replay-bundle-v27 | neural-prior-scoring-replay-bundle-v27 | ∅ |
| forecast_run_artifact | forecast-run-v72 | forecast-run-v71 | forecast-run-v72 | forecast-run-v71, forecast-run-v72 | forecast-run-v72 | ∅ |
| neural_prior_holdout_plan | neural-prior-holdout-plan-v37 | neural-prior-holdout-plan-v36 | neural-prior-holdout-plan-v37 | neural-prior-holdout-plan-v36, neural-prior-holdout-plan-v37 | neural-prior-holdout-plan-v37 | ∅ |
<!-- CONTRACT_CAPABILITY_TABLE:END -->

`main`과 pull request는 GitHub Actions에서 Python 3.10·3.12 CPU 전체
시험을 실행하고, Python 3.12 환경에서 product source basedpyright를
검사한다. 별도 package job은 sdist와 wheel을 빌드한 뒤 격리 환경에 wheel을
설치하여 `advar-nowcast` CLI와 NPZ 출력계약을 smoke-test한다. Required package
job은 hash-locked Linux CPU dependency wheelhouse의 offline 설치와 CLI 결과 확인에서
끝난다. Signed deployment bundle, release approval, host runtime activation과
deployment artifact upload는 과학적 실증에 필요하지 않으므로 required PR CI에서
실행하지 않는다.

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

### 초기장 실험 홈페이지

고정된 합성 사례에서 관측과 정답은 그대로 두고, 배경 초기장의 나이·위치·강도·
공간 범위만 바꿔 P0 예측 변화를 비교할 수 있다. 화면은 실제 `advar.nowcast()`를
호출하며 실시간 자료 연결이나 운영 판정을 수행하지 않는다.

```bash
python examples/initial_field_lab/server.py
```

브라우저에서 `http://127.0.0.1:8765`를 연다. 결과 MAE와 persistence MAE는
두 예측이 모두 유효한 같은 화소에서 계산하며, 유효영역 비율은 별도로 표시한다.

### 0.114 import migration

Package root는 P0/P1 핵심 API만 공개한다. `SensitivityConfig`와
`VerificationBundle` 같은 sensitivity 타입, replay·ledger·promotion·deployment
타입은 각 `advar.<owning_module>`에서 직접 import해야 한다.

```python
from advar.sensitivity import SensitivityConfig, VerificationBundle
from advar.run_artifact import load_forecast_run
```

재현 가능한 연구 패키지와 필수 CI는 일반 resolver 설치를 사용하지 않는다. Linux x86-64 CPU용
Python 3.10/3.12 runtime closure와 test/build closure를 각각
`requirements/*-linux.lock`에 exact version과 distribution SHA-256으로 보존한다.
검증된 direct runtime version은 `requirements/runtime.in`에 고정한다.
CI는 PyTorch CPU index를 명시하고 `--require-hashes --only-binary=:all:`로만
설치한 뒤 project 자체는 `--no-build-isolation --no-deps`로 연결한다. 따라서
같은 source commit의 required test, wheel build와 installed-wheel smoke가 새로
게시된 transitive dependency나 build backend를 조용히 선택하지 않는다.
`check_dependency_locks.py`, `pip check`, `pip-audit --strict --disable-pip
--no-deps`가 lock 동기화·설치 closure·알려진 취약점을 required job에서
검사한다. PyTorch의 public version을 lock에 기록하되 SHA-256은 CPU index에서
resolve한 artifact만 허용하므로 vulnerability database identity와 실제 wheel
identity를 함께 유지한다.
감사·installer·build·test module은 isolated Python mode에서 실행하고 감사기는
checkout 밖의 runner 임시 directory에서 먼저 실행한다. 저장소의 동명
`pip_audit.py`, `pip.py`, `build.py`, `pytest.py`가 required tool을 shadow할 수
없으며, 실제 `pip_audit` module이 interpreter site-packages 아래에 있는지도
canary로 확인한다. Lock checker는 runtime과 CI closure의 version뿐 아니라 hash
set이 같은지, direct numerical/security pin이 Python 세대 사이에서 같은지,
`nvidia-*`·`triton`이 없는 CPU-only closure인지도 검증한다.

### 비필수 legacy deployment engineering reference

아래 bundle·activation 절차는 과거 artifact audit와 별도 engineering 실험을 위한
참고자료다. Required PR CI에서 실행하지 않으며, scientific review eligibility나
confirmatory skill claim을 만들지 않는다. 현재 프로젝트는 이 절차의 운영 인증이나
LIVE 사용을 목표로 하지 않는다.

과거 Linux CPU deployment scaffold는 bundle의 detached Ed25519 signature를 host에
out-of-band로 고정한 공개키로 검증하고, `mode=deployable`, repository/ref/commit,
workflow SHA와 signer identity가 승인값과 exact 일치하는지 확인한다. Private key는
protected release environment에서만 공급하며 bundle이나 저장소에 넣지 않는다.
Deploy host는 bundle directory와 모든 payload 및 public-key ancestry를 root-owned,
group/world-non-writable 상태로 staging한 뒤 검증한다. Verifier는 symlink key/file을
거부하고 public key를 `O_NOFOLLOW` 단일 file descriptor로 읽으므로 검증 후 다른
wheel로 바꾸는 unprivileged path-swap은 허용되지 않는다.
CI의 `candidate-smoke` public key는 일회성이므로 그 artifact는 설치 smoke evidence일
뿐 release authority가 아니다. Signature 검증 뒤 manifest와 각 file SHA-256을
검증하고 다음 순서만 사용한다. Wheel metadata의 범위 dependency를 다시 resolve하는 일반
`pip install <wheel>`은 동일한 배포 closure로 간주하지 않는다.

```bash
python -I .github/scripts/build_deployment_bundle.py verify \
  --bundle /srv/advar/bundles/<digest> \
  --trusted-public-key /etc/advar/release-bundle-ed25519.pub \
  --expected-mode deployable \
  --expected-repository gonos2k/AD4DVAR-radar \
  --expected-source-ref refs/tags/v0.101.0 \
  --expected-source-commit <signed-release-commit> \
  --expected-workflow-sha <protected-workflow-sha> \
  --expected-signer-id advar-release
```

```bash
python -I -m pip install --no-index --find-links wheelhouse \
  --require-hashes --only-binary=:all: --no-compile \
  --requirement runtime-py312-linux.lock
python -I -m pip install --no-index --no-deps --no-compile \
  advar_radar_nowcast-0.101.0-*.whl
find <deployment-venv> -type f \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pth' \) -delete
```

설치 뒤에는 verifier가 활성 site-packages root를 전수조사해 lock에 없는
distribution, 소유자가 없는 shadow module, `.pth`, `sitecustomize.py`,
`usercustomize.py`, `.pyc`와 `__pycache__`를 거부한다. Python executable, stdlib,
extension module이 연결한 native library의 bytes와 ABI도
`advar-import-runtime-tree-v2`에 결합하며 deployable runtime은 root-owned이고
group/world-non-writable여야 한다. Release authority는 검증한 bundle manifest
전체와 bundle digest를 detached `DeploymentBundleReleaseApproval-v1`으로 먼저
승인한다. 그 approval digest, runtime-tree digest, interpreter closure,
deployment instance, host identity, activation predecessor와 expiry는 별도의
`deployment-runtime-activation-receipt-v3`에 host activation key로 서명한다.
Runtime snapshot·verify·launch process는 반드시 `python -I -B` 또는 동등한
격리/bytecode-write 금지 launcher로 시작해야 하며 interpreter closure가 이 상태를
봉인한다.
Protected release signer와 activation signer의 private key는 둘 다 bundle 밖에
보관하며, production에서는 root-owned immutable staging에서만 다음 명령을 실행한다.

```bash
python -I .github/scripts/build_deployment_bundle.py approve-release \
  --bundle /srv/advar/bundles/<digest> \
  --trusted-bundle-public-key /etc/advar/release-bundle-ed25519.pub \
  --expected-mode deployable \
  --expected-repository gonos2k/AD4DVAR-radar \
  --expected-source-ref refs/tags/v0.101.0 \
  --expected-source-commit <signed-release-commit> \
  --expected-workflow-sha <protected-workflow-sha> \
  --expected-bundle-signer-id advar-release \
  --expires-at <canonical-utc-expiry> \
  --release-authority-id advar-release-approval \
  --release-authority-trust-store-digest <sha256-current-trust-store> \
  --release-signing-private-key /etc/advar/release-approval.key \
  --approval /var/lib/advar/bundle-release-approval.json

python -I .github/scripts/build_deployment_bundle.py activate-runtime \
  --bundle /srv/advar/bundles/<digest> \
  --trusted-bundle-public-key /etc/advar/release-bundle-ed25519.pub \
  --expected-mode deployable \
  --expected-repository gonos2k/AD4DVAR-radar \
  --expected-source-ref refs/tags/v0.101.0 \
  --expected-source-commit <signed-release-commit> \
  --expected-workflow-sha <protected-workflow-sha> \
  --expected-bundle-signer-id advar-release \
  --deployment-instance-digest <sha256-host-slot-identity> \
  --host-identity-digest <sha256-host-identity> \
  --activation-sequence-number <monotonic-sequence> \
  --previous-activation-receipt-digest <current-head-or-genesis-digest> \
  --expires-at <canonical-utc-expiry> \
  --release-approval /var/lib/advar/bundle-release-approval.json \
  --trusted-release-public-key /etc/advar/release-approval-ed25519.pub \
  --expected-release-authority-id advar-release-approval \
  --expected-release-authority-trust-store-digest <sha256-current-trust-store> \
  --activation-authority-id advar-runtime-activation \
  --activation-authority-trust-store-digest <sha256-current-trust-store> \
  --activation-signing-private-key /etc/advar/runtime-activation.key \
  --receipt /var/lib/advar/runtime-activation-receipt.json
```

제품은 receipt 파일을 수동 dictionary로 변환하지 않는다.
`load_deployment_bundle_release_approval()`과
`load_deployment_runtime_activation_receipt()`가 root-owned/non-writable,
`O_NOFOLLOW` 단일-FD snapshot과 current authority trust를 검증한 typed receipt만
반환한다. `EpisodeLedger.issue_operational_deployment_decision()`은 이 receipt를
release approval과 함께 필수로 받아 approval table, activation head, issuance state,
operational certificate와 decision row에 같은 digest를 기록한다. Committed client와 durable forecast loader는 현재
trust, expiry, bundle/runtime/interpreter/host identity 및 receipt signature를 다시
검증한다. 이때 `candidate-smoke` receipt는 operational 경로에서 거부되고,
`snapshot_current_runtime()`이 실행 중인 process의 import roots, interpreter, stdlib와
native library를 다시 해시해 receipt의 deployable closure와 exact 비교한다. 따라서
다른 venv/process에서 유효한 receipt만 재사용하거나 activation 뒤 runtime bytes를
바꾼 경우 scientific decision issuance와 `forecast-run-v72` restart가 모두 fail-close한다.

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
affine은 scale-normalized determinant와 singular value로 검증하며 모든 복원된
물리 metric이 finite여야 한다. 정규화 determinant가 `0.01`보다 작거나 2-norm
condition number가 `1000`보다 크면 거의 평행하거나 지나치게 왜곡된 격자로 보고
거부한다.
`maximum_background_age_minutes`를 넘는 배경은 거부한다. 계약을 생략한
호출은 합성·연구용 index-time/pixel-grid 경로로 유지되며 실제 레이더
운영자료의 물리 provenance를 주장하지 않는다.

발행 프로세스가 종료된 뒤에도 동일한 실행계약으로 M0를 계산하려면
forecast-run artifact를 저장한다.

```python
from advar.run_artifact import (
    load_forecast_run,
    save_forecast_run,
)
from advar.sensitivity import compute_sensitivity_snapshot_from_run

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
analysis input digest도 같은 identity에 포함한다. `input_bundle_digest`는
candidate/parent의 동일 외생 입력을 비교할 수 있도록 radar·mask·background·grid
계보만 나타내며, 분석 설정과 실제 neural-prior application digest는 별도 run
identity에 결합된다. Neural prior는 metadata나 임의 Tensor로 선언할 수 없다.
eval-mode PyTorch model을 감싼 `NeuralPriorInferenceRunner`가 feature transform과
model을 `torch.export` graph로 고정하고, model state·실제 numerical runtime·input
bundle·공간별 state/probability output을 하나의 inference evidence로 묶는다.
`NeuralPriorStateContract`는 P1 background/std/support를 분석 radar product, QC·mask·
censor policy, detection threshold, support/validity decision probability와 물리 dBZ·std
범위에 결합한다. Active state가 이 범위의 경계에 닿거나 벗어나면 inference를 거부하며,
P1에서 사후 clamp하지 않으므로 exported JVP/VJP와 실제 P1 입력이 같은 함수를 나타낸다.
별도의 `NeuralPriorProbabilityContract`는 event-probability channel을
특정 검증 radar product와 QC pipeline의 `P(Z >= threshold_dbz)` 사건에 결합하고,
truncated-location/scale channel을 그 support 사건에서 절단되는 Gaussian의
pre-truncation parameter로 정의한다. Prior runner, holdout target plan,
completed case와 inference evidence의 probability/support-event digest가 모두 같지
않으면 paired 평가가 시작되기 전에 거부된다.
Full probabilistic model의 exported output 순서는
`(state_background_dbz, state_std_dbz, state_valid_probability,
state_support_probability, event_probability, truncated_location_dbz,
truncated_scale_dbz)`이다. 첫 네 channel만 P1이 소비하고 마지막 세 channel만 hurdle
calibration이 소비한다.
승인 후에는 원래 Python callable이 아니라 export된 graph 자체를 실행하며, 그
factory가 만든 `NeuralPriorApplication`만 실제 P1 초기배경으로 소비된다.
Radar-dependent prior는 deterministic Rademacher JVP/VJP·finite-difference 검사를
통과해야 하고 FSOI의 실제 adjoint cotangent와 full/half 재분석에서도 다시 실행된다.
Prior가 새 echo support를 만들 때는 기존 causal support로 clip하거나, 물리면적·
echo-integral budget 안에서 control domain을 명시적으로 확장해야 한다.
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
from advar import AnalysisConfig, variational_nowcast

forecast, analysis = variational_nowcast(
    frames,
    observation_std_dbz=2.0,
    analysis_config=AnalysisConfig(
        # 0.0이면 기존 diagonal-R 계약과 정확히 같다.
        observation_common_bias_std_dbz=0.5,
        observation_common_bias_scope="per_frame",
        # 0은 domain-wide, 양수는 독립 square-tile bias mode다.
        observation_common_bias_tile_size_px=32,
    ),
)

print(analysis.used_fallback)
print(analysis.initial_objective, analysis.final_objective)
print(analysis.state.echo_linear)  # 세 장으로 분석된 현재 q(0)
```

수용·수렴한 P1 분석은 최종 control에서 IRLS weight와 remap cell을 다시
고정한 뒤 제한된 GN polish를 수행한다. 각 수용 step 뒤에는 IRLS weight를
다시 계산하며, frozen stationarity·실제 pseudo-Huber gradient·retained weight
일관성을 모두 만족해야 P1 forecast, posterior와 FSO를 제공한다. polish 뒤에는
기존 amplitude·causal·saturation·objective gate도 다시 검사한다. 보존되는
`p1-final-frozen-irls-gn-v14` 선형화에는 frozen/robust gradient 진단, IRLS
weight 변화, polish 횟수와 최종 hard feasibility margin이 함께 들어간다.
stationarity는 격자크기에 따라 작아지는 objective 정규화 대신 active field
gradient RMS와 표준화된 3개 dynamics gradient 최대값의 큰 값을 사용하며,
희소한 국지 오차를 숨기지 않도록 field gradient 최대값도 별도로 제한한다.
관측·분류 mask·최종 IRLS weight·remap cell·prior graph를 포함한 모든 frozen
입력은 `linearization_digest`에 결합되며, 보존 Tensor는 caller storage와
분리된 clone이다. 미래 검증장이 도착하면 이 고정 선형화에서 세 입력시각
전체의 observation 민감도를 matrix-free adjoint로 계산할 수 있다.
P1 결과는 `outer_converged`, `final_linearization_stationary`,
`final_robust_stationary`, `final_irls_fixed_point`, `p1_forecast_eligible`,
`posterior_eligible`, `fso_eligible`을 분리해 기록한다. 운용모드에서 최종
robust fixed-point 계약을 만족하지 못하면 P1을 발행하지 않고 P0로 fallback하며,
연구모드에서도 posterior와 FSO용 선형화는 제공하지 않는다.
운용모드는 분석 평균의 saturation margin뿐 아니라
`safe_margin + p1_posterior_saturation_sigma_multiplier × posterior_sigma`를
요구한다. posterior가 비유한하거나 이 경계를 넘으면 P1을 발행하지 않는다.

탐지한계 아래의 censored 값은 기본적으로 `min_dbz` clear-sky floor로
canonicalize한다. 따라서 같은 below-detection 사건을 -10 dBZ 또는 4.9 dBZ로
저장해도 P1 초기배경은 같고, 넓은 clear 영역에 탐지한계 직하의 약한 에코를
인위적으로 만들지 않는다. `censored_background_policy`를 `detection_limit`
또는 `external_background`로 명시할 수도 있다. 외부배경 정책은 모든 censored
화소의 배경 coverage를 요구하며 탐지한계 바로 아래로 상한을 둔다.

```python
from advar.sensitivity import (
    SensitivityConfig,
    VerificationBundle,
    compute_variational_fso,
)

verification = VerificationBundle(
    frames_dbz=verification_frames_dbz,
    valid_mask=verification_qc_mask,
    valid_times=verification_valid_times,
    grid_contract_digest=forecast.run.grid_time_contract_digest,
    radar_product_digest=radar_product_digest,
    qc_pipeline_digest=qc_pipeline_digest,
)

fso = compute_variational_fso(
    forecast,
    analysis,
    verification,
    sensitivity_config=SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(30, 60, 120, 180),
        require_verification_lineage=True,
    ),
)

# [selected_lead, metric, observation_time, H, W]
print(fso.observation.detected_dbz.maps.shape)
# censored event의 detection-threshold 민감도
print(fso.observation.censor_threshold_dbz.maps.shape)
# detected/censored를 모두 포함한 objective-weight 민감도
print(fso.observation.observation_weight.maps.shape)
# 첫 관측이 P1 초기배경으로 들어가는 direct+implicit 민감도
print(fso.observation.initial_background_dbz.maps.shape)
# 세 관측이 P0 motion/growth를 만드는 연속 경로(pair/peak 선택 고정)
print(fso.observation.baseline_dynamics_dbz.maps.shape)
# residual dBZ, 초기배경, baseline dynamics 경로의 합
print(fso.observation.frozen_structure_input_dbz.maps.shape)
# 모든 lead·metric에서 -20/-10/0분별 L2 norm
print(fso.observation.detected_dbz.norm_by_time.shape)
# 선형화·검증장·출력 Tensor까지 묶은 결과 identity
print(fso.linearization_digest, fso.verification_bundle_digest)
print(fso.variational_fso_digest)
```

각 metric의 수반계는 P1 최종 frozen IRLS/GN normal operator와 동일한
matrix-free JVP/VJP로 PCG 풀이한다. PCG 비수렴, P0/degraded 분석, 실행 digest와
다른 관측오차·quality 입력, 최종 분석상태를 재현하지 못하는 선형화, 또는
`||J^T r|| / (1 + ||r||)`가
`AnalysisConfig.final_linearization_relative_stationarity_tolerance`를 넘는
선형화는 fail-close한다. `maximum_final_linearization_polish_iterations=0`으로
polish를 끌 수 있지만 이 경우 stationarity를 만족하지 않는 분석에는 수반
민감도를 제공하지 않는다. 계산하도록 선택한 lead에는 시각별 norm과 tile
norm을 제공하고, 그중 `SensitivityConfig.full_map_lead_minutes`에는 전체 지도를
제공한다.

`VerificationBundle`은 검증 dBZ와 QC mask뿐 아니라 각 lead의 정확한 UTC
valid time, 발행격자 digest, 레이더 product와 QC pipeline identity를 하나의
content digest로 묶는다. `require_verification_lineage=True`이면 shape만 있는
legacy Tensor를 거부한다. raw Tensor 입력은 연구 호환용으로 계속 허용하지만
결과와 M0 원장에 `verification_lineage_complete=False`로 기록되므로 지연 자동
학습의 완전한 검증자료로 승격할 수 없다.

`compute_variational_fso()`의 current `p1-variational-fso-v28` 결과는 영향값이 아니라
다음 관측 parameter와 frozen 초기배경 경로에 대한 미분이다.

Radar-dependent neural prior는 mean과 spatial `log(std_dbz)` JVP/VJP를 모두
포함한다. Export artifact의 영점 probe와 별도로 실제 radar 입력에서 finite-
difference probe를 다시 수행하고, 실제 FSO cotangent 방향의 adjoint defect도 결과
digest에 남긴다. Validity는 연속 확률이어야 하며 validity·support 확률의 0.5 hard
branch margin을 모두 active-set 진단에 기록한다.

```text
detected_dbz                 관측잔차를 통한 d(metric) / d(observed dBZ)
censor_threshold_dbz         d(metric) / d(censor threshold)
observation_weight           d(metric) / d(objective multiplier alpha), alpha=1
initial_background_dbz       첫 관측→P1 초기배경의 direct+implicit 경로
baseline_dynamics_dbz        세 관측→P0 motion/growth의 direct+implicit 경로
frozen_structure_input_dbz   위 세 dBZ 경로의 합
```

`initial_background_dbz`는 첫 시각에서 실제 관측이 초기배경을 제공한 화소에만
존재한다. 이 경로는 metric의 초기배경 직접미분과 frozen GN stationarity를 통한
제어해의 간접미분을 모두 포함한다. `baseline_dynamics_dbz`는 관측에서 P0
phase-correlation subpixel displacement와 growth를 거쳐 P1 baseline dynamics로
들어가는 직접·간접 경로를 포함한다. 이때 observation/background source, pair
조합, integer FFT peak, search-boundary 상태, support/classification, active
control과 remap cell은 nominal 실행에서 선택된 그대로 고정한다. 명시적 FSOI는
nominal·half-step·full-step에서 pair span, pair availability, integer peak와
selection/conflict signature가 같은지도 검사한다. 따라서
`frozen_structure_input_dbz`는 전체
preprocessing 총미분이 아니라
`residual_plus_input_dependent_initial_state_and_baseline_with_frozen_selection` 범위의
piecewise-smooth 민감도다.

`observation_weight` 채널은 detected와 censored 관측을 모두 포함한다. FSOI는
작은 국지 perturbation만 허용하며 기본 한계는 dBZ 계열 0.5 dBZ, weight 0.1이다.
또한 실제 frozen observation whitener로 계산한 전역·tile별 norm, 변경 화소 수와
면적비를 함께 제한한다. detected/censored 분류나 P0 pair/FFT 선택 branch를 넘는
방향도 fail-close한다.
`from_radar_dbz_delta()`는 정량값이 있는 detected 화소만 허용한다. censored
사건은 저장된 임의 dBZ가 아니라 `from_censor_threshold_delta()` 또는
`from_censored_event_weight_delta()`로 별도 표현한다.
`delta_alpha=-1`인 완전 관측 제거는 active structure와 공분산을 바꿀 수 있으므로
이 1차 계약에서 거부하며, 실제 제거 영향을 구하려면 관측을 제외하고 다시 풀어야
한다. 명시적인 perturbation과 곱한 signed first-order impact가 필요할 때만 FSOI API를
사용한다.

forecast-error metric의 공간영역은 `SensitivityConfig.metric_domain`으로
명시한다.

```text
issued                    finite verification ∩ issued valid mask
radar_dynamics_anchored   finite verification ∩ radar-dynamics mask
confidence_weighted       issued domain에 발행 confidence를 frozen weight로 적용
```

기존 연구결과를 보존하기 위한 기본값은 `issued`다. 자동 observation ranking에는
`radar_dynamics_anchored`를 명시해야 하며, 이 정책의 강도는 발행 결과의 local
motion/growth evidence 계약보다 강해질 수 없다. `confidence_weighted`는 미분 중
confidence를 재계산하지 않고 발행된 값을 상수로 고정한다. 실제 사용 weight의
digest, lead별 합과 격자면적 대비 유효분율은 FSO 결과에 보존된다.

```python
from advar.sensitivity import (
    VariationalObservationPerturbation,
    compute_variational_fsoi,
)

perturbation = VariationalObservationPerturbation(
    detected_dbz=detected_delta_dbz,
    censor_threshold_dbz=censor_threshold_delta_dbz,
    observation_weight=observation_weight_delta,
    # 첫 관측값을 초기배경에서도 함께 바꿀 때만 지정한다.
    initial_background_dbz=initial_background_delta_dbz,
    # 같은 입력이 P0 motion/growth에도 들어가는 효과를 포함할 때 지정한다.
    baseline_dynamics_dbz=baseline_dynamics_delta_dbz,
)
# 하나의 실제 radar dBZ 변경은 다음 factory가 retained input 경로에 자동 배분한다.
physical = VariationalObservationPerturbation.from_radar_dbz_delta(
    radar_delta_dbz,
    analysis.linearization,
)
fsoi = compute_variational_fsoi(
    forecast,
    analysis,
    verification_frames_dbz,
    perturbation,
    sensitivity_config=SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(30, 60, 120, 180),
    ),
)

# component별 signed impact와 합계
print(fsoi.observation.detected_dbz.maps.shape)
print(fsoi.observation.initial_background_dbz.maps.shape)
print(fsoi.observation.baseline_dynamics_dbz.maps.shape)
print(fsoi.observation.total.sum_by_time.shape)
print(fsoi.perturbation_diagnostics.whitened_l2)
```

각 impact는 `sensitivity * perturbation`이며 양수는 지정 perturbation에 의해
forecast-error metric이 1차적으로 증가함을 뜻한다. perturbation Tensor는
원 관측과 dtype·device·shape가 같고 각 채널의 active mask 밖에서는 정확히
0이어야 한다. `initial_background_dbz`는 선택사항이며 첫 시각의 accepted
observation 밖에서는 0이어야 한다. `baseline_dynamics_dbz`도 선택사항이며 accepted
관측 밖에서는 0이어야 한다. 같은 입력 dBZ perturbation의 모든 frozen-selection
경로를 평가하려면 해당 위치의 `detected_dbz`, `initial_background_dbz`(첫 시각),
`baseline_dynamics_dbz`에 같은 delta를 넣는다. `observation_weight`는 새 multiplier가
음수가 되지 않도록
`delta_alpha >= -1`을 요구하며, 모든 Tensor와 contract는 perturbation digest에
결합된다. 이 값은 frozen-GN 국지근사이며 EFSO는 아니다.

완전 관측제거는 국지 FSOI로 근사하지 않는다. 원래 세 입력장과 외부배경을
linearization에 보존하고, 지정 관측을 QC에서 제외한 뒤 P0 baseline, active
support, common-bias whitener, robust P1, posterior와 발행 forecast를 모두 다시
계산한다.

```python
from advar.sensitivity import (
    ObservationRemovalConfig,
    compute_variational_observation_removal_impact,
)

denial = compute_variational_observation_removal_impact(
    forecast,
    analysis,
    verification_bundle,
    removal_mask,
    sensitivity_config=sensitivity_config,
    removal_config=ObservationRemovalConfig(
        maximum_removed_observation_count=256,
        maximum_removed_fraction=0.01,
    ),
)
```

`denial.metric_change`는 제거 후 score에서 nominal score를 뺀 비선형 영향이다.
양수는 제거가 오차를 늘렸음을 뜻한다. count·fraction·union-area와 전체 whitener
연산 budget을 넘거나 제거 후 eligible P1이 나오지 않으면 부분결과 없이 거부한다.

EFSO는 deterministic P1 FSOI와 분리된 ensemble API다. 실제 analysis ensemble의
observation-space perturbation과 forecast-error projection, innovation, 관측오차
통계를 모두 요구하며 단일 분석에서 가짜 ensemble을 만들지 않는다. 일반 관측오차
모델은 caller가 임의의 `R⁻¹d`를 넘기는 대신, content-addressed dense precision과
covariance artifact를 제공해 `R(R⁻¹d)=d` 잔차와 observation ordering을 검증한다.
Root-owned trust store는 재사용 가능한 승인 토큰이 아니라 정확한
`operator_digest`를 `approved_operator_digests`에 기록한다.

```python
from advar.ensemble_sensitivity import (
    EnsembleFSOStatistics,
    PrecisionOperatorArtifact,
    compute_ensemble_fso,
)

precision_artifact = PrecisionOperatorArtifact(
    precision=precision_matrix,
    covariance=covariance_matrix,
    observation_ids=observation_ids,
    forecast_run_digest=forecast.forecast_run_digest,
    observation_error_model_digest=observation_error_model_digest,
    calibration_manifest_digest=calibration_manifest_digest,
)

efso = compute_ensemble_fso(
    EnsembleFSOStatistics.from_full_r(
        innovation=innovation,
        precision_operator=precision_artifact,
        maximum_relative_residual=1e-6,
        trust_store_path="/etc/advar/precision-operators.json",
        analysis_observation_perturbations=analysis_y_perturbations,
        forecast_error_projection_by_member=forecast_error_projection,
        lead_minutes=(60, 120),
        metric_names=("log_echo_mse",),
        verification_reference_digest=verification_reference_digest,
        ensemble_member_ids=ensemble_member_ids,
    )
)
```

대각 관측오차만 사용하는 경우에는
`EnsembleFSOStatistics.from_diagonal_r(...)` factory를 사용한다. 일반 경로는
실제 관측오차모델과 결합되고 잔차가 검증된 `R⁻¹d`를 요구한다. EFSO impact가
음수이면 해당 관측이 지정 forecast-error metric을 줄이는 방향이다. 결과에는
ensemble member jackknife 표준오차도 함께 기록된다.
Dense full-R artifact는 covariance와 precision 양쪽의 Cholesky SPD, 상호 역행렬,
condition number, observation-count와 byte budget을 통과해야 한다. 사용 직전에도
operator digest를 다시 계산하므로 승인 뒤 Tensor 변조는 precision evidence로
전환되지 않는다. 대규모 radar pixel은 dense artifact 대신 block/low-rank
operator가 필요하며 현재 dense 경로는 예산 안의 연구규모에만 허용된다.
입력 ensemble perturbation과 forecast projection은 member 축에서 중심화되어야 한다.
구현식은 Kalnay et al.의 ensemble observation-impact formulation을 따르며 입력
통계의 의미는 [Tellus A 원문](https://doi.org/10.3402/tellusa.v64i0.18462)에
명시된 계약으로 고정한다.

대격자 delayed adjoint는 분석 solver 설정과 분리된 실행계약으로 제한한다.

```python
from advar.sensitivity import VariationalAdjointConfig

fso = compute_variational_fso(
    forecast,
    analysis,
    verification_frames_dbz,
    sensitivity_config=SensitivityConfig(
        metric_names=("log_echo_mse",),
        full_map_lead_minutes=(60, 180),
    ),
    adjoint_config=VariationalAdjointConfig(
        lead_minutes=(60, 180),
        maximum_normal_products=400,
        maximum_whitener_total_operations=10_000_000_000,
        maximum_materialized_output_bytes=512 * 1024**2,
        warm_start_by_metric=True,
        gauss_newton_probe_count=4,
        maximum_gauss_newton_relative_curvature_defect=0.25,
    ),
)
```

자동학습 경로는 두 strict config를 하나의 외부 승인 policy로 묶는다.

```python
policy = AutomatedLearningPolicy(
    sensitivity_config=SensitivityConfig.for_automated_learning(
        radar_product_digest=approved_radar_product_digest,
        qc_pipeline_digest=approved_qc_pipeline_digest,
    ),
    adjoint_config=VariationalAdjointConfig.for_automated_learning(),
    algorithm_bundle_digest=approved_algorithm_bundle_digest,
    numerical_runtime_digest=approved_numerical_runtime_digest,
    metric_taylor_thresholds=(
        MetricTaylorThreshold("log_echo_mse", 1e-6, 1e-5),
        MetricTaylorThreshold("soft_fss_error_35", 1e-6, 1e-5),
        MetricTaylorThreshold("centroid_error_m2", 1.0, 100.0),
    ),
)
learning = compute_variational_fsoi_for_learning(
    forecast,
    analysis,
    verification_bundle,
    physical,
    policy=policy,
    policy_trust_store_path="/etc/advar/learning-policies.json",
)
if learning.eligibility.eligible:
    validate_variational_learning_impact(learning)
    ledger.append_variational_learning_approval(learning)
```

이 결과는 승인된 counterfactual이며 실제 행동의 결과는 아니다. 같은 사례의 미래
verification으로 만든 결과를 과거 행동처럼 기록할 수 없으며, historical 분석은
`RetrospectiveCounterfactualReplay`로만 감사 보존한다. 실제 행동은 publication
deadline 전에 결정과 실행 receipt가 각각 append-only ledger에 기록돼야 한다.

```python
from advar.intervention import (
    InterventionActionGenerator,
    InterventionInputContext,
    OperatorActionApproval,
    ProspectiveInterventionDecision,
    RealizedInterventionReceipt,
    ReusableInterventionPolicyEvidence,
)

input_before_context = InterventionInputContext.from_inputs(
    frames_dbz=input_before_frames,
    observation_masks=input_before_masks,
    quality_weight=input_before_quality,
    observation_std_dbz=input_observation_std,
    background_frames_dbz=input_background_frames,
    radar_id=radar_id,
    applicability_mask=approved_applicability_mask,
    run=input_before_run,
)
action_generator = InterventionActionGenerator.from_model(
    action_model.eval(),
    input_before_context,
    intervention_type="realized_qc_intervention",
    action_reason="clutter",
)
action_policy = ReusableInterventionPolicyEvidence(
    policy_id="radar-qc-v1",
    action_generator_digest=action_generator.generator_digest,
    context_schema_digest=input_before_context.context_schema_digest,
    applicability_region_digest=(
        input_before_context.applicability_region_digest
    ),
    execution_policy_digest=execution_policy_digest,
    allowed_intervention_types=("realized_qc_intervention",),
    maximum_absolute_delta_dbz=0.5,
    validation_evidence_digests=validation_evidence_digests,
)
decision = ProspectiveInterventionDecision.from_policy(
    action_policy,
    action_generator=action_generator,
    decision_id=decision_id,
    case_id=case_id,
    radar_id=radar_id,
    intervention_type="realized_qc_intervention",
    actual_input_context=input_before_context,
    actual_input_before_run=input_before_run,
    input_plan_digest=input_plan_digest,
    decision_basis_digest=validated_policy_evidence_digest,
    decision_policy_digest=action_policy.execution_policy_digest,
    decision_trust_store_digest=execution_trust_store_digest,
    decided_at=decided_at,
    observation_valid_time=observation_valid_time,
    input_available_time=input_available_time,
    decision_deadline=decision_deadline,
    publication_time=publication_time,
)
operator_approval = OperatorActionApproval.from_decision(
    decision,
    operator_key_id=operator_key_id,
    operator_role="duty-meteorologist",
    operator_trust_store_digest=operator_trust_store_digest,
    operator_private_key=operator_private_key,
    reviewed_at=reviewed_at,
    expires_at=decision_deadline,
    operator_comment_digest=operator_comment_digest,
)
ledger.append_prospective_intervention_decision(
    decision,
    operator_approval=operator_approval,
    action_policy=action_policy,
    action_generator=action_generator,
    actual_input_before_context=input_before_context,
    actual_input_before_run=input_before_run,
    trust_store_path="/etc/advar/intervention-policies.json",
    operator_trust_store_path="/etc/advar/intervention-operators.json",
)
receipt = RealizedInterventionReceipt.from_decision(
    decision,
    actual_input_before_context=input_before_context,
    actual_input_before_run=input_before_run,
    actual_input_after_context=input_after_context,
    actual_input_after_run=input_after_run,
    action_policy=action_policy,
    action_generator=action_generator,
    executor_key_id="radar-qc-executor",
    executor_trust_store_digest=executor_trust_store_digest,
    executor_private_key=executor_private_key,
    executor_sequence_number=executor_sequence_number,
    applied_time=applied_time,
    receipt_time=receipt_time,
)
ledger.append_realized_intervention_receipt(
    decision,
    receipt,
    action_policy=action_policy,
    action_generator=action_generator,
    actual_input_before_context=input_before_context,
    actual_input_before_run=input_before_run,
    actual_input_after_context=input_after_context,
    actual_input_after_run=input_after_run,
    trust_store_path="/etc/advar/intervention-policies.json",
    executor_trust_store_path="/etc/advar/intervention-executors.json",
    operator_trust_store_path="/etc/advar/intervention-operators.json",
)
```

QC action model은 `(valid_mask_after, quality_weight_after, applicable)`을 반환한다.
Sensor correction은 additive dBZ, operator override는 명시적 replacement/mask 계약을
각각 사용한다. `InterventionInputContext`는 radar ID, 승인 적용영역, 전체 mask·quality,
관측오차와 background를 묶으므로 generator가 승인영역 밖을 변경할 수 없다. 시간계약은
observation valid, input available, decision deadline, publication을 분리한다. ledger는
root-approved action policy를 다시 실행해 decision을 재현하고, receipt에서는 typed
before/action/after 전이를 다시 검사한다. Executor trust store에는 Ed25519 public key만
들어가며 private key는 executor 밖으로 나오지 않는다. 이전 v1/v2 intervention
계약은 read-only 감사용으로 보존된다.

`operator_reviewed_only`는 정책 문자열만으로 충족되지 않는다. 각 decision에는
decision/action/full-input/safety digest와 검토 역할·만료시각을 묶은
`OperatorActionApproval`의 Ed25519 서명이 필요하다. Operator trust store는 검토자
public key와 허용 role만 보존하며, executor 서명과 operator 서명은 각각 사전 승인과
사후 실행이라는 서로 다른 사실을 증명한다. 서명이 없거나 다른 input/action에 대한
서명이면 decision append가 거부된다.

Decision은 전체 `InterventionInputContext`, fixed-input context와 applicability mask
digest를 직접 보존한다. Receipt 생성·재적재 시 같은 context에서 action 안전진단을
다시 계산해 decision의 진단 digest와 비교한다. `AnalysisInputIdentity.full_data_digest`
(run의 `full_analysis_input_digest`)는 radar
frame과 mask·quality·observation std·background·calibration을 포함하는 fixed context를
하나로 묶으며, receipt와 candidate/parent holdout의 입력 동일성은 이 digest를 기준으로
판정한다. Quality-only QC는 input bundle이 그대로여도 full-input identity의 변경으로
정상 기록된다. `forecast-input-plan-resolution-v2`도 plan digest와 이 full-data
digest를 결합하므로 quality/std만 달라져도 resolved plan identity가 달라진다. v1
resolution은 기존 artifact의 read-only 무결성 검증에만 허용한다. QC 크기는 raw quality 차이가 아니라
`(sqrt(Q_after) - sqrt(Q_before)) / observation_std_dbz`의 전역·tile L2 norm으로
제한하고, prospective QC는 reject/deweight만 허용한다. 현재 prospective 실행은
`operator_reviewed_only`이며 current-case benefit 계약이 없는 automatic policy는
생성 단계에서 fail-close한다. dBZ action norm은 명시적인 diagonal-R 표준화 거리이고,
`observation_common_bias_std_dbz > 0`인 run은 상관 관측오차용 Mahalanobis action
계약이 추가될 때까지 거부한다. NaN/Inf와
invalid 관측은 `min_dbz`로 canonicalize되고 finite valid 값만 물리 clamp에 들어간다.
이 canonicalization도 digest에 포함된다. 장기 action artifact는 파일·member·expanded
byte 상한을 allocation 전에 검사한 뒤 서명과 before/action/after 전이를 재생한다.

Prior artifact 승격은 intervention 실행 여부와 무관하게 사전등록된 holdout 전체를
candidate/parent paired forecast로 평가한다. 서로 단위가 다른 forecast-error metric은
정책의 물리 scale로 정규화하며, 최소 표본수, 개선·악화 비율, 평균개선과 최악
단일악화를 모두 통과해야 한다.

```python
from advar.promotion import (
    NeuralPriorHoldoutPlanPolicy,
    NeuralPriorPromotionPolicy,
    PromotionMetricScale,
    RangeMetricRequirement,
    compute_neural_prior_promotion,
)

ledger.append_neural_prior_holdout_plan(
    holdout_plan,
    policy=holdout_plan_policy,
    policy_trust_store_path="/etc/advar/learning-policies.json",
)

promotion_policy = NeuralPriorPromotionPolicy(
    metric_scales=(
        PromotionMetricScale("log_echo_mse", 0.1, 0.01),
        PromotionMetricScale("soft_fss_error_35", 0.05, 0.005),
    ),
    approved_candidate_manifest_digests=(candidate_manifest.manifest_digest,),
    approved_holdout_plan_digests=(holdout_plan.plan_digest,),
    approved_metric_contract_digests=(metric_config.digest,),
    deployment_regime_classifier_digest=regime_classifier.classifier_digest,
    deployment_regime_classifier_manifest_digest=(
        regime_classifier_manifest.manifest_digest
    ),
    minimum_holdout_cases=20,
    minimum_material_cases=20,
    required_range_metrics=(
        RangeMetricRequirement(
            weather_regime="convective",
            range_regime="near_range",
            metric_name="soft_fss_error_35",
            lead_minutes=60,
            minimum_cases=5,
            minimum_valid_area_km2=100.0,
        ),
    ),
    maximum_prior_conditional_pit_residual_mean_abs=2.0,
    maximum_prior_conditional_underdispersion_fraction=0.1,
    maximum_prior_echo_support_miss_score=0.25,
)
promotion = compute_neural_prior_promotion(
    candidate_manifest,
    holdout_plan,
    prior_holdout_evaluations,
    policy=promotion_policy,
    policy_trust_store_path="/etc/advar/learning-policies.json",
)
if promotion.eligible:
    ledger.append_neural_prior_promotion(
        promotion,
        candidate_manifest,
        holdout_plan,
        prior_holdout_evaluations,
        policy=promotion_policy,
        policy_trust_store_path="/etc/advar/learning-policies.json",
        raw_ingestor_trust_store_path="/etc/advar/raw-ingestors.json",
        training_target_source_trust_store_path=(
            "/etc/advar/training-target-sources.json"
        ),
    )
```

holdout plan은 candidate/parent forecast보다 먼저 원장에 등록하며 미래 input
content digest 대신 input valid-time/source/QC/grid/background/mask 선택규칙의
`input_plan_digest`를 고정한다. 미래 frame 내용이 아닌 verification
source/QC/grid/valid-time identity, metric contract와 issue time도 함께 고정한다.
Prior uncertainty target도 `PriorUncertaintyTargetPlan`으로 source 종류(독립 sensor,
withheld radar/time/mask), QC·mask·censor·floor measurement contract,
feature-exclusion 및 independence evidence를
사전등록하며 plan payload 자체가 holdout digest에 포함된다. 실제 target은 임의
Tensor로 만들 수 없고, plan에 고정된 radar product·QC·grid·valid time과 일치하는
content-addressed `radar-verification-bundle-v22`에서만 생성한다.
P1 state head에는 별도의 `NeuralPriorStateCalibrationPlan`을 사전등록한다. State target은
state product·QC·mask·censor·floor policy, dBZ resolution·quantization origin과 prior output
valid time에 결합되고 feature에서 withhold됐음을 검증한다. Target은 이 측정계보를 실제
자료와 함께 observation-error contract를 attestation한
`radar-verification-bundle-v22`에서만 생성된다. Candidate와
parent의 state interval-Gaussian NLL·PIT,
support Brier·pixel/object miss·false-support 및 validity Brier를 같은 target에서 paired
평가한다. 절대 calibration과 cluster max-statistic 비열화 상한을 모두 통과하지 못하면
forecast skill이 좋아도 `state_calibration_eligible=False`이며 posterior/confidence용
confirmatory scientific claim에는 사용할 수 없다.
실제 input·verification content digest는 자료가 도착한 뒤 completed holdout case에
결합한다. 이미 알려진 historical holdout은 sealed dataset을 candidate training 전에
등록하는 별도 mode를 사용한다. 두 forecast run은 동일 input
bundle을 사용하고 candidate/parent prior·model/schema/training manifest identity를
각자 직접 포함해야 하며 retained output은 같은 model/runtime에서 재추론되어야 한다.
정확한 candidate/parent forecast, prior application, inference evidence, verification,
metric contract와 operational issuance-domain digest의 ordered set은
`HoldoutScoringInputArtifact`로 scoring start receipt보다 먼저 append-only ledger에
기록한다. 현재 자동승격 경로는 candidate family를 정확히 하나로 제한하므로 하나의
start/completion receipt와 하나의 canonical scoring output 사이에 cardinality 모호성이
없다. Forecast realization 하나라도 바뀌면 scoring-input digest가 달라져 기존 start
receipt와 output artifact를 재사용할 수 없다. Scoring에 사용되는 사후 weather-regime
reference, physical-event identity, uncertainty/state target과 range 계약도 ordered
completed-holdout-case digest로 함께 봉인되므로 수치 forecast를 유지한 채 truth grouping만
바꿀 수 없다.
통계 판정 임계값·metric scale/support·필수 metric/issuance cell·confidence와
finite-sample 설정은 candidate authorization과 분리된 `PromotionDecisionRule`로
정규화한다. Exact root-approved rule digest는 prospective issue와 forecast·verification
생성 전에 holdout plan에 결합되고, ledger는 plan과 재사용 가능한 canonical rule 정의의
binding을 같은 transaction에 기록한다. Scoring input과 최종 authorization은 이
plan-bound digest를 그대로 참조해야 한다. 따라서 결과를 본 뒤 rule family에서 유리한
threshold나 classifier 선택을 고르거나 policy를 사후 승인해도 canonical scoring
artifact와 결합할 수 없다. Ledger는 scoring-input
row의 trusted `created_at`이 receipt의 시작시각보다 늦으면 backdated start를 거부한다.
평가는 parent의 고정 domain에서 paired skill을 계산하고 candidate native domain과의
차이를 issuance effect로 분리한다. end-to-end candidate-minus-parent metric도 모든
lead·metric의 non-inferiority guard로 사용하므로 작은 신규발행 면적의 큰 error도
숨길 수 없다. ledger append는
root-owned trust store를 다시 읽고 promotion을 재계산하며, plan의 모든 case가
manifest와 evaluation에 정확히 한 번씩 존재해야 한다. 결측·forecast 실패는 유리한
subset 선택으로 제거할 수 없고 promotion이 fail-close한다. Prior promotion은 전체
preregistered forecast population의 `PriorHoldoutEvaluation`을 사용하며, intervention을
선택·실행한 사례만 모은 action-effect population과 분리된다. Material 사례의
case/day/radar/regime 다양성뿐 아니라 signed physical-event catalog의 event
membership을 검사한다. 같은 event의 radar·day·cycle은 반복측정으로 유지하고 outer
cluster는 event digest 하나만 사용한다. Candidate와 classifier training event가 holdout
event와 겹치면 exact input digest가 달라도 거부한다.
Event association은 공통 CRS의 timestamped object track을 보존하며 case source-object
evidence를 정확한 track sample·mask에 결합한다. 각 piecewise-linear segment의 속도와
가속도를 fail-close하고, 서로 겹치는 두 track은 knot만 비교하지 않고 각 segment pair의
analytic closest approach를 계산한다. Knot 사이에서 교차하거나 radar source가 handoff된
동일 storm을 별도 independent event로 세는 경로를 차단한다.
Neural prior는 P1이 소비하는 deterministic Gaussian state head와 holdout calibration용
hurdle-probability head를 별도 계약으로 출력한다. State product와 support threshold는
분석 radar product와 detection limit에 정확히 일치해야 하며, probability head의
truncated location/scale은 P1 초기장으로 재사용되지 않는다. Probability head는 모델
입력과 분리된 withheld target의 positive-echo 화소에서 float64 `log_ndtr` 기반 conditional
truncated-Gaussian intensity NLL로 검증한다. Radar dBZ의 해상도·quantization origin·첫
threshold bin 규칙을 계약에 포함한다. Nearest-rounding+censor 계약에서 첫 echo bin은
`[L,L+Δ/2)`, 다음 bin은 `[L+Δ/2,L+3Δ/2)`이므로 인접 code가 확률질량을 중복하지 않는다.
관측값은 origin 기반 dBZ lattice에 있어야 하며,
off-lattice 값은 다른 측정계약으로 간주해 거부한다. 점 likelihood 대신 관측 bin의
interval mass를 사용하고, 양·음 tail은 각각 survival/CDF log-mass로 계산해 극단값도
유한한 score로 유지한다. Conditional PIT도 interval midpoint를 사용하므로 threshold와 정확히 같은
양자화 값이 모델과 무관하게 극단 residual이 되지 않는다. Event probability는
고정 target mask 전체 Brier뿐 아니라 pixel-level echo miss, connected echo-object miss와
clear false-echo score로 각각 검증한다.
Clear-sky에서는 support probability의 false-echo score를 별도로 계산하므로 동일한
no-echo를 -10/0/4.9 dBZ 중 어떤 floor로 저장해도 intensity score가 달라지지 않는다.
Candidate와 parent를 동일한 사전등록 target mask에서 paired 평가하고,
physical-event outer cluster inference로 구한 intensity-NLL·support-Brier·echo-miss·
clear-sky·conditional-underdispersion 증가의 전역 및 regime별 최악 상한도 정책 한계를
넘어서는 안 된다.
순수 clear case의 intensity component와 순수 echo case의 clear component는 실패가
아니라 `NOT_APPLICABLE`로 기록한다. 대신 전체 및 regime별 echo/clear 사례·독립
cluster 최소수뿐 아니라 case별 component 화소수·물리면적과 echo object 수도 요구한다.
Candidate family 크기는 Bonferroni로, 현재 candidate의 component와 regime 비교는 동일
cluster sign replicate의 studentized max statistic으로 함께 보정한다. 작은 cluster 집합은
모든 sign pattern을 정확히 열거하고, 큰 집합은 요구 tail replicate 수를 충족해야 한다.
검정 방법·유효 replicate·critical quantile·Monte Carlo 오차와 실제 simultaneous test 수를
promotion evidence에 보존한다. Echo와 clear 증거가 모두 충족된 classifier-output
regime/range만 deployment applicability에 포함되며 그 밖에서는 parent prior로
fail-close한다. 한 group의 표본이 부족하면 그 group만 인증에서 제외하며, 정책이
`require_all_registered_regimes_certified`를 명시한 경우에만 candidate 전체를 거부한다.
Range applicability는 classifier label에 whole-domain score를 복제하지 않는다. Holdout
plan에 사전등록된 `RangeBandContract`의 물리 mask 안에서 forecast metric과
state/probability score를 다시 계산한다. Geometry mask 면적과 별도로 lead별 metric-valid,
probability-valid, state-valid 면적과 echo/clear pixel·connected-object 수를 보존하며,
component별 최소 유효 표본을 통과해야 한다. Range mask는 서로 겹치지 않을 뿐 아니라
전체 분석격자를 정확히 한 번 덮어야 하므로 미인증 영역을 숨긴 채 candidate를 전역 선택할
수 없다. Band-local skill의 평균·beneficial bound는 signed physical event 단위로 계산하고,
harmful fraction은 event 안의 cycle 빈도를 보존한다. Binary event gate는 exact/Wilson
bound를 사용하고, event 내부 cycle 비율은 family-adjusted empirical-Bernstein bound를
사용한다. 따라서 5/5 성공이나 0/5 실패를 확률 1 또는 0으로 오해하지 않으면서
fractional rate에 binomial 공식을 적용하지 않는다. Candidate의 절대 band score와
parent 대비 비열화 상한을 함께 검사하므로 두 prior가 모두 나쁜 band도 인증되지 않는다.
Band별 family-adjusted tail replicate 수와 Monte Carlo 오차는 promotion evidence에 보존되고,
해상도가 부족하면 fail-close한다. Reference active set에 대한 set precision·recall,
exact-set accuracy와 false-active-band rate까지 통과한 band만 인증하므로 모든 range
logit을 높여 near/mid/far를 동시에 활성화하는 classifier는 인증을 얻지 못한다. 각
weather×range group은 정책이 요구한 metric×lead별 최소 case와 실제 valid 면적뿐 아니라
그 cell 자체의 common-domain 및 end-to-end mean-degradation UCB와
harmful-fraction UCB, issuance withdrawal/new-issuance 한계도 통과해야 한다. 자동배포용
metric cell은 최소 5개 physical event를 요구한다. 따라서 required FSS의 일관된 악화나
발행영역 악화를 다른 metric 개선으로 상쇄할 수 없다. `promotion_sample_size_preflight()`는
family-adjusted finite-sample radius의 best-case event 요구량을 score 계산 전에 산출하며,
holdout event가 그 수보다 적으면 결과와 무관하게 infeasible로 표시한다.
Band issuance는 verification mask나 metric availability를 사용하지 않는다. Holdout
plan이 publication eligibility, source coverage, permanent exclusion mask digest를
사전고정하고, `OperationalIssuanceDomainArtifact`가 그 교집합을 lead별 분모로 보존한다.
따라서 영구차폐·no-source cell로 신규발행이나 철회 비율을 희석할 수 없다. Continuous
metric mean bound의 support도 정책 숫자가 아니라 `MetricSupportContract`가 FSS의
unit interval, bounded dBZ log-echo MSE, grid-diagonal centroid error에서 해석적으로
도출한 범위만 사용한다. Support contract는 실제 nowcast-config와 time-independent
spatial-grid digest, 설치된 ADVAR source bundle에서 계산한 scoring metric-engine
identity 및 derivation parameter를 함께 보존하고, evaluation의
config/spatial-grid/engine lineage와 exact 일치해야 한다. 같은 공간격자의 서로 다른
valid time 사례는 동일 support를 공유하고, 여러 공간격자는
config×spatial-grid×engine별 support로 분리된다. Affine grid의 centroid
support는 한 대각선만 쓰지 않고 projected 네 corner의 모든 pairwise 거리 중 최댓값으로
계산한다. 자동승격의 global·band aggregate mean은 physical-event 수와 무관하게 같은
support-derived empirical-Bernstein inference를 사용하므로 동일한 5--9개 소표본의
percentile bootstrap이 불확실성 0으로 퇴화하지 않는다. 작은 표본 bootstrap은 명시적인
shadow 진단에서만 허용되고 sample-size preflight가 자동배포를 fail-close한다. 유한
support가 증명된 uncertainty component도 bounded event UCB를 사용한다. NLL과 PIT는
고정 `UncertaintyScoreSupportContract`에 따라 화소별 score를 event 집계 전에 clip하고,
그 finite support로 같은 bounded UCB를 계산한다. 따라서 zero 또는 near-zero
표본분산을 모집단 불확실성 0으로 해석하지 않는다.
실제 배포에서는 caller가 regime 문자열이나 operational role을 넘기지 않는다. Exported
`NeuralPriorRegimeClassifier`가 현재 `full_analysis_input_digest`에 결합된
`RegimeClassificationEvidence`를 만들고, 모든 holdout case에서도 동일 classifier를
실행한다. Reference-label accuracy·regime recall·confidence calibration·false routing과
unknown/OOD abstention을 통과해야 deployment evidence가 유효하다. Classifier family와
각 `RegimeClassifierManifest`의 training dataset/case/storm/day/time-window뿐 아니라
input-bundle/full-analysis-input/radar/grid와 signed member-manifest lineage는
forecast issue 전에 holdout plan에 사전등록되고 holdout과 겹치면 거부된다. 서로 겹치는
plan·candidate·rule·classifier trial은 하나의 root-approved
`PromotionExperimentFamily`에 사전등록되며 전체 trial multiplicity가 모든 skill,
uncertainty, classifier, metric-cell bound와 sample-size preflight에 적용된다. 동일한
input/verification cohort는 ledger에서 다른 family로 다시 등록하거나 promotion 후 재사용할
수 없다. Candidate와 classifier의 `TrainingDatasetDerivationArtifact-v1`은 각 training member의
완전한 signed `AnalysisInputDerivationArtifact-v5` preimage를 보존하고, 그 member들의
raw identity/input-bundle/full-analysis/grid 집합과 decoder/QC/regrid digest가 선언된
training dataset lineage와 정확히 일치해야 한다. Feature algorithm/schema digest도
dataset identity에 결합되지만, 이 계약은 임의 native decoder나 feature implementation을
실행하는 증명으로 해석하지 않는다. Classifier accuracy·recall·range-set precision과
false-routing은 point estimate뿐 아니라
physical-event cluster 신뢰하한·상한도 통과해야 한다. Prospective plan에는 미래 weather
정답 대신 `pending`과 `RegimeReferencePlan`의 labeler/source/adjudication rule만 기록한다.
검증시각 뒤의 observed label은 exact input·verification·storm에 묶인 Ed25519
`RegimeReferenceEvidence`로 완료된다. Classifier
runtime·dtype·device도 evidence에 결합되고 weather top-two와 deployment confidence hard
branch margin이 부족하면 `ambiguous_classifier_branch`로 parent에 fallback한다. Range
logits는 holdout의 shadow classifier 진단에만 남는다. Operational range applicability는
radar-site identity, grid coordinates, radial distance edges와 resolver algorithm을 묶은
root-approved `RangeGeometryContract`에서 결정한다. 현재 격자의 완전분할에서 비어 있지
않은 모든 band와 exact geometry contract가 인증돼야 candidate를 전 격자에 선택하며,
하나라도 미인증이면 전체 parent로 fallback한다. Single-site는 projected horizontal
single-radar geometry를 사용하고, mosaic는 `SourceRadarRegistry`와
`RadarSiteLocationRegistry`의 동일한 site ordering에 결합된
`MosaicRangeGeometryContract-v3`를 사용한다. Input-time `[H,W]` source-index map으로
cell별 effective horizontal range를 계산하며 source가 없는 `-1` 셀은 어떤 range band에도
포함하지 않는다. Geometry grid digest, coordinate shape, source-map digest와 effective-range
digest는 operational run의 source identity와 직접 비교된다. Beam-height는 아직 이
horizontal range 계약의 범위 밖이다.
`infer_deployed_neural_prior()`는 root-owned trust store가 승인한
`DeployedNeuralPriorPolicy`의 confidence rule까지 확인한 뒤 candidate 또는 parent를
선택한다. 연구용 `NeuralPriorInferenceRunner.infer()`는 operational input을 항상 거부한다.

Holdout scoring 완료 전에는 direct constructor가 닫힌
`ScoringReplayCaseArtifact.from_products()`가 exact shipped product type만 받아 case별
radar/QC/quality/background, candidate/parent의 7개 prior channel·최종 state·forecast·
publication/fallback/confidence mask, verification·calibration target, classifier logits,
range 좌표와 operational-domain mask를 typed product object에서 직접 추출해 NPZ로
보존한다. Append, completion, promotion은 caller callback을 받지 않으며 제품 코드의
`PriorHoldoutEvaluation.from_forecasts()`를 반드시 다시 실행한다. 재계산된 모든
evaluation digest와 archive tensor digest가 봉인된 manifest와 같아야 하므로, 기존
evaluation JSON에 무관한 임의 tensor를 붙인 snapshot은 자동승격 증거가 될 수 없다.
Forecast, prior application, inference runner, verification, metric config, calibration
target, classifier와 operational-domain artifact는 각각 제품 validator와 runner
reproduction을 통과해야 하며, factory가 한 번 동결한 tensor snapshot과 completion 시점의
live product bytes가 다르면 거부된다. Replay v27 contract와 v27 method, typed
verification-target identity와 source-composition algorithms를 포함한 CPU-only
generation v25 digest는 `HoldoutScoringArtifact-v15`와 scientific promotion evidence
v32까지 직접 전파된다. Current package에는 운영 selector의 양성 경로가 없으며
`infer_deployed_neural_prior()`는 전용 `OperationalDeploymentUnsupportedError`를
발생시킨다. Deployment authorization은 프로젝트 범위 밖이다.
각 holdout case의 `VerificationTargetIdentityArtifact-v2`는 실제 scoring에 사용되는
target plan에서 source identity와 UTC valid time을 가져오고, exact target value,
valid-mask, quality, QC/censor policy, observation-error contract 및 verification
bundle digest를 함께 봉인한다.
Evaluation factory는 metric 계산 전에 이 typed identity를 실제 target tensors에서
다시 만들어 case 선언과 byte-for-byte 비교한다. Training/holdout overlap gate도 같은
typed identity를 사용하므로 outer manifest digest를 다시 계산해도 scoring target를
다른 source/time/value/mask/quality로 재라벨링할 수 없다.
따라서 이전 replay 세대의 promotion evidence는 audit-only이며 배포 selector가
소비할 수 없다.
`VerificationObservationErrorPlan-v16`은 forecast scoring 전에 source/calibration
registry, range/elevation·beam-blockage·QC·censoring·mosaic source-assignment
algorithm, quality/std 생성규칙, spatial-block 생성규칙과 reference observation
standard deviation, product-owned mask/error derivation identity와 독립 verification
source authority key를 사전등록한다.
`MosaicObservationSourceRegistry-v6`는 source-map index를 ordered radar-site,
calibration epoch, source-specific quality/std와 사전등록된 range/elevation detection-limit
함수에 결합한다. Registry와 `RadarObservationGeometryContract-v6`는 같은
`projected_crs_digest`를 가져야 한다. Current geometry는 독립 좌표 배열을 authority로
받지 않는다. Forecast의 `RadarSpatialGridIdentity-v5`가 shape, canonical projection에서
유도한 CRS digest, 첫 cell-center 원점, column/row affine matrix, float64 dtype과
cell-center convention을 하나의 content-addressed preimage로 보존하고, verification
x/y 좌표를 다음 식으로 제품 코드가 생성한다.

```text
[x(i,j), y(i,j)] = origin + pixel_to_projected_matrix @ [j, i]
```

따라서 forecast와 verification의 y축 방향, 원점, 회전, anisotropic spacing, CRS 또는
cell-center convention이 다르면 current FSO/replay에서 거부된다. 좌표는 처음부터
float64로 생성해야 하며 float32 좌표를 사후 변환해 current scientific geometry로
사용할 수 없다. Current physical-distance research는 bounded Korean study domain의
`EPSG:5179`만 허용한다. `EPSG:3857`의 projected metre는 위도에 따라 ground scale이
달라 current metric CRS로 사용할 수 없으며 historical artifact의 byte audit에만 남는다.
`RadarMetricDomainEvidence-v3`는 같은 PROJ/EPSG database에서 재생한 17×17 factor
lattice, PROJ·`projinfo` binary 및 `proj.db` SHA-256, PROJJSON digest, 선형·면적
scale error와 report SHA를 보존한다. 생성 스크립트 SHA-256과 canonical output contract,
강제된 `LC_ALL=C`·`LANG=C`·`TZ=UTC`·`PROJ_NETWORK=OFF`, platform/ABI/libc/SQLite 및
dependency inspector, Python executable, `_sqlite3`·`_decimal` native extension과
실제로 해석된 dynamic-library closure도 report에 포함한다. Loader override가 있으면
생성을 거부하며 `--check`는 sealed host에서 generator source와 committed bytes를 함께
검증한다. Darwin shared cache처럼 파일 byte identity를 얻지 못한 closure는 보고서에
명시하며 독립 sealed environment가 여전히 필요하다고 fail-close한다. 이전 v1/v2 report는
raw JSON bytes와 원래 SHA-256만 검증하는 audit-only 타입이다.
Report가 실제로 표본화한 projected coverage
(`592664≤E≤1576674`, `976711≤N≤2251910` metre) 밖의 grid 또는 radar site는 broad
historical bbox 안에 있더라도 current scientific evidence를 만들 수 없다. 이 표본 report는
등록된 scale budget을 재현하는 repository evidence이며 ground-distance 정확성의 독립적인
geodetic 인증은 아니다. Current
grid/run은 이 evidence digest를 정방향으로 포함하며 `forecast-run-v72`와 durable
intervention action v7은 cold replay에서도 같은 면적 불확실성 정책을 다시 적용한다.
최대 면적 cap은 ground-area interval 상한, 최소 growth evidence는 interval 하한으로
판정하며 threshold를 가로지르면 과학적 증거를 fail-close한다. 거리·반경·속도도 같은
원칙을 사용한다. 물리반경 footprint는 확실한 내부, 불확실 annulus, 가능한 내부를
별도로 보존한다. Positive support는 확실한 내부만 사용하지만 neighborhood completeness는
가능한 내부 전체를 검사한다. PSR과 metre-based FSS처럼 집합 변화에 비단조인 통계는
annulus가 비어 있을 때만 current scientific metric으로 제공한다. Motion speed·pair disagreement·
saturation·posterior eligibility는 ground-speed interval 상한으로 판정한다. 과거 evidence 없는
`forecast-run-v69`는 audit-only이다. `radar-spatial-grid-identity-v5` 자체는 역사적
evidence-absent payload도 byte audit을 위해 표현할 수 있으므로 contract 문자열만으로
current capability를 판단하지 않는다. Current run/verification/replay의 outer validator가
exact evidence digest의 존재와 sampled coverage를 반드시 다시 검사한다.
2-D grid의 spatial-age spacing은 affine minimum singular value, 1×N/N×1 grid는 실제로
존재하는 column/row 축 norm을 사용하고 1×1 grid는 spatial metric을 지원하지 않는다.
현재 geometry model은
`projected-horizontal-representative-tilt-v1`이고 radar altitude는
`provenance_only`이다. 따라서 이 계약은 beam height, Earth curvature 또는 refractivity를
재현하는 full 3-D beam model이라고 주장하지 않는다.
`VerificationObservationMaskEvidence-v14`는 source별 nominal acquisition
time과 cell-local time offset, grid, radar product, native source identity 및 radar별
`[source,time,y,x]` spatial/QC/value bytes를 source authority signature로 봉인한다.
Cell 관측 나이는 `verification valid time - source nominal acquisition time - local offset`으로
제품 코드가 재계산한다. Source가 서명한 report kind는 detected echo, confirmed clear,
below-detection censored를 구분하며 값·threshold와 불가능한 조합은 거부한다.
Detection limit field는 ordered registry의 source별 base/range/elevation 계수에서 제품 코드가
재계산하고 attenuation 불확실성은 threshold에 중복 가산하지 않는다. Source dimension은
ordered registry의 exact source digest 순서에 결합된다.
Source 선택의 시간 품질 `exp(-((age/scale)**power))`은 current v5에서 backend
`pow`/`exp` 결과를 승자 인증에 사용하지 않는다. 정확히 age 0인 경우만 `[1,1]`, 그 밖은
해석적으로 보장되는 `[0,1]`을 score interval에 넣는다. 이 넓은 구간에서도 한 source의
lower score가 모든 경쟁 source의 upper score를 엄격히 이길 때만 선택하고, 시간
transcendental 값이 우열을 결정할 수 있는 경우에는 source를 미할당한다.
`VerificationObservationMaskDerivationArtifact-v14`는 선택된 source index에서 value,
detection limit, local offset, absolute age와 네 spatial field를 gather한 뒤 source-present,
range/elevation-valid, blockage, acquisition-time-valid, attenuation-QC, confirmed-clear,
censoring mask와 source index map을 다시 계산한다. Caller가 mask를 직접 선택하는 legacy
input은 confirmatory 경로에서 소비하지 않는다.
`derive_verification_observation_error()`는 derivation-input v15와 ordered registry에서
valid/quality/std/state tensor를 계산한다. 같은 radar 안에서도 range, elevation,
blockage와 attenuation evidence에 따라 quality/std가 공간적으로 변한다. 사전등록된
maximum acquisition age를 넘은 cell은 `STALE_ACQUISITION`으로 분리되며, 나이가 증가하면
temporal quality가 단조 감소하고 temporal representativeness variance가 standard
deviation에 추가된다. `ObservationErrorDerivationArtifact-v15`는 동일 입력으로 그 결과를
다시 생성해 `torch.equal`과 content digest를 모두 확인한다.
`VerificationObservationErrorContract-v18`는 plan, signed raw input, mask derivation,
ordered registry와 exact
valid/quality/observation-std/state/source-map 및 absolute acquisition-age tensor digest를
derivation artifact에 결합한다. 이 상태 tensor는 clear, echo, source missing, QC invalid,
beam blockage, stale acquisition, below-detection censoring과 mosaic source 미할당을 서로
다른 과학적 의미로 보존한다.
직접 `from_tensors()`로 만든 v3 contract와 `radar-verification-bundle-v6`는
`exploratory_only`이며 proper-score diagnostic에는 사용할 수 있지만 confirmatory
target 또는 scientific-review eligibility를 만들 수 없다.
Output tensor만 재생하는 v4 contract와 `radar-verification-bundle-v7`은 audit
compatibility 세대이며 current confirmatory target을 만들 수 없다.
Holdout plan v37은 모든
uncertainty/state target이 참조하는 observation-error plan payload의 정확한 집합을
보존하고, current target는 deterministic replay를 포함한
`radar-verification-bundle-v22`만 허용한다. v21 이하는 audit-only이며 bundle valid time, shared projected
grid, radar product를
signed source identity와 exact 비교한다. 따라서 결과를 본 뒤 mask, source time,
source index ordering, calibration mapping, selected-source value/time/detection limit
또는 realized tensor를 바꾸면 source
signature, derivation replay와 holdout plan 중 적어도 하나가 실패한다. 현재 scoring
measure는
`quality * min(1, (reference_std / observation_std)^2)`이다. Invalid·unobserved
cell은 weight 0이어야 하고 low-quality 영역의 apparent gain은 promotion score에
그 비율만큼만 기여한다. Below-detection censored cell은 lineage에는 남지만 point-value
metric weight는 0으로 둔다. 이는 proper censored likelihood가 도입되기 전까지 해당
관측을 임의의 dBZ point truth로 취급하지 않기 위한 보수적 연구 계약이다.
`compute_observation_error_gaussian_diagnostic()`은 promotion gate와 독립된 report-only
artifact로 `sqrt(forecast_std**2 + observation_std**2)` predictive scale의 quantized
Gaussian NLL/PIT를 계산한다. `BELOW_DETECTION_CENSORED` cell은 point value 대신
cell별 selected-source detection limit의 left-censored likelihood로 평가한다. 관측오차는
이미 predictive variance에 포함되므로 이 진단의 aggregation에는 quality만 사용하고
inverse-variance를 다시 곱하지 않는다. 결과는 항상 `diagnostic_only=True`이며,
사전등록된 과학 protocol 없이 promotion을 승인하지 않는다.
Current scientific replay는 `neural-prior-scoring-replay-bundle-v27`이며 source-specific
report kind, absolute acquisition age, temporal-valid mask, spatial-metric age support와
confirmed-clear mask, 그리고 product-derived verification geometry의 float64 x/y 좌표를
content-addressed shard에 보존한다. 직전 v26은 byte audit만 가능하고 current semantic
replay나 confirmatory claim으로 승격할 수 없다.
Confirmed clear는 quantitative dBZ point가 아니라 categorical no-echo evidence로
평가한다. 따라서 intensity Gaussian/FSO point metric에서는 제외하지만 echo-support
Brier와 false-support score에는 남아 representation floor와 무관하게 false echo를
벌점화한다. Threshold equality는 `echo: Z > L`, `censored: Z <= L`로 단일화한다.
Spatial-correlation block identity도 observation-error plan과 realized contract에
`spatial_correlation_role="diagnostic_only"`로 고정한다. Source assignment는
사전등록된 product-owned eligibility/score 함수로 range, elevation, acquisition age,
beam blockage와 attenuation QC를 결합한다. Current scientific source selection은
backend `pow`/`exp` 결과를 radar 우열의 권위값으로 사용하지 않는다. Acquisition age가
정확히 0인 source의 시간 품질 구간은 `[1, 1]`, 그 밖의 source는 해석적 codomain
`[0, 1]`로 두며, 한 source의 score 하한이 모든 경쟁 source의 score 상한보다 엄격히
클 때만 선택한다. Invalid source는 먼저 제외하지만, score 구간이 겹치거나 동점이면
ordered-registry 순서로 임의 결정하지 않고 cell을 unassigned로 남긴다. 이 정책은
false winner를 막지만 multi-radar overlap coverage를 보장하지 않는다. 실제 radar
cohort에서 assigned/eligible 비율과 differential loss를 측정하기 전에는 이 결과를
confirmatory source selection으로 승격하지 않는다. 그 미측정 한계를 별도의 100%
coverage wrapper로 감추거나 전체 실행을 막지는 않는다. Current typed verification
geometry와 source-selection certification은 CPU binary64-only이며 MPS에서
전용 비지원 오류로 fail-close한다. Current confirmatory confidence bound는 pixel
block이 아니라 independent physical-event
cluster를 사용하며, spatial block metadata가 추론에 사용됐다고 주장하지 않는다.
Audit load는 저장된 typed evaluation을 볼 수 있지만, 자동 completion과 promotion은
동일한 typed replay case를 다시 제공해 semantic replay까지 통과해야 한다. 배포
certificate 발급 시에도 typed case에서 제품 scorer를 다시 실행하며, checksum-only
archive는 certificate를 받을 수 없다. Ledger는
current replay bundle/method v27, semantic generation v25와 scoring case v26만
재실행하며, 이전 replay bundle v26과 그 이전 세대는 typed audit-only
object로만 decode한다. Current bundle은 `verification_provenance.json`과 source-specific
verification tensors를 보존한다. 원래 Python case object가 없어도 source signature,
ordered registry, product-owned radar geometry/source selection, mask derivation,
observation-error derivation과 v22 verification bundle digest를
cold-start 재검증하고 `verification_semantic_replay_verified=True`를 보고한다. Model
runner와 forecast products까지 다시 실행하는 full scoring replay는 exact typed case가
제공될 때만 `semantic_replay_verified=True`이며 두 주장을 혼동하지 않는다.
Source tensor는 단일 8 GiB archive가 아니라 content-addressed NPZ shard로 보존한다.
동일 tensor bytes는 deduplicate되고 cold load는 shard를 하나씩 검증·해제하므로 전체
cohort tensor를 동시에 메모리에 적재하지 않는다. Scoring start가 선행했는지 확인하고 archive/file/Tensor/evaluation digest를 append,
completion, promotion load 시마다 다시 계산한다.
입력 QC mask와 quality weight, optional background는 모두 실제
`ForecastRunContract`와 동일한 `[T,H,W]` shape·dtype·device 및 Tensor digest를 가져야
한다. Replay bundle의 numerical runtime도 caller 문자열이 아니라 typed case의 실제
input Tensor device에서 계산한다. 따라서 시간별 quality를 2차원 mask로 축약하거나
background의 시간축 일부만 보존한 archive는 semantic replay 전에 거부된다.

## 실제사례 acceptance (REPORT_ONLY)

`advar-real-case-acceptance-manifest-v1`은 single-site, 지연 frame, complete outage,
mosaic handoff, duplicate acquisition, target 지연/손상, key revocation, provenance
crash, activation crash와 offline restart의 열 가지 시나리오를 exact physical-event와
13단계 product artifact chain으로 묶는다. 이 harness는 ledger를 수정하거나 deployment를
승인하지 않으며 항상 `REPORT_ONLY` report를 낸다.

```bash
python -I .github/scripts/run_real_case_acceptance.py \
  --manifest /srv/advar/acceptance/manifest.json \
  --artifact-root /srv/advar/acceptance/artifacts \
  --report /srv/advar/acceptance/report.json
```

Manifest는 canonical `neural-prior-promotion-sample-size-preflight-v5` 파일의 SHA-256과
product digest를 직접 포함한다. Verifier는 그 bytes를 typed preflight로 다시 만들고
`required_physical_events`와 manifest count, 모든 case의 preflight digest가 정확히 같은지
확인한다. 저장소의 synthetic unit fixture는 harness integrity만 검사하며 현업 radar
acceptance evidence로 간주하지 않는다. Current `advar-real-case-acceptance-report-v2`는
이를 content-addressed artifact index로만 취급한다. Generic stage JSON은
`semantic_e2e_validated=False`, `sample_size_satisfied=False`,
`eligible_for_scientific_review=False`이며 caller가 제공한 event digest 수는
`declared_event_label_count`로만 보고한다. 독립 event 수는 typed stage replay와
제품이 재계산한 event identity가 도입되기 전까지 `None`이다. 실증 승격에는 legally usable native radar bytes,
independent verification observations와 보호된 deployable signer가 별도로 필요하다.

동적 mosaic/outage coverage는 nominal source domain을 미래에 고정하는 대신 input
availability 이후 decision deadline 이전에 data-ingestor가 서명한
`ResolvedSourceCoverageArtifact`로 해석한다. Mosaic plan은 canonical
`SourceRadarRegistry`와 data-ingestor public key를 사전등록하고, source index는 `-1`
또는 registry 범위 안의 값만 허용한다. Input-time source/QC/outage map은 `[H,W]`,
publication eligibility는 `[lead,H,W]`로 분리되며, exact source-map·selection-policy·
input/full-analysis digest에 결합된다. Ledger에 deadline 전에 append되지 않은 resolved
coverage나 resolved coverage가 없는 mosaic scoring input은 fail-close한다.
`MosaicRangeGeometryContract-v3`는 radar registry, site-location ordering, projection과
range resolver만 사전 고정하고 issue-time source handoff map 자체는 고정하지 않는다.
`RangeBandContract-v3`는 허용 가능한 active-range set을 outcome-blind하게 등록하며,
실제 source/outage/dynamic-QC/nominal·resolved coverage bytes와 계산된 effective range는
case별 replay archive에 저장된다. 따라서 outage로 source radar가 바뀌어도 같은 resolver
계약 아래 재현할 수 있고, source가 없거나 QC-invalid인 셀은 어떤 range band에도
참여하지 않는다.
Holdout의 meteorological sampling identity는 processing grid나 아직 관측되지 않은 content
hash가 아니라 issue 전에 예약된 radar-site/acquisition/scan observation slot으로 구성한다.
Ingest 뒤에는 signer-independent canonical raw-volume identity와 별도 ingestor attestation을
slot에 결합하고, 중앙 sampling registry가 연속 sequence/root로 그 resolution을 봉인한다.
각 case의 canonical binary32 gridded source bytes와 resolution은 scoring보다 늦게 붙일 수
없으며 `append_analysis_input_provenance()`가 deadline 전에 durable payload를 commit하고
exact ledger instance와 registry/history side-effect set에 결합된
`analysis-input-provenance-preparation-receipt-v3`를 함께 commit한 exact row만
scoring replay가 소비한다. Process crash로 receipt 발급만 늦어진 경우에는 pre-deadline
payload commit과 전체 raw-to-derived replay 및 ledger side effect를 다시 검증한 뒤에만
root-approved ledger authority가 receipt를 발급할 수 있다. Receipt 검증 authority는 row
내부의 자기 선언 key가 아니라 root-owned ledger trust와 exact ledger-instance binding에서
도출하므로, self-signed derivation/receipt나 registry/history를 건너뛴 SQL row는 소비자가
거부한다.
제품 decoder는 그
source bytes를 float32로 다시 해석하고 native flag·finite·positive-weight·observation-std
QC를 재실행한다. 현재 지원하는 regrid 계약은 source와 target grid가 같은 exact-identity
경로이며 임의 polar decoder나 공간 resampler를 인증한다고 주장하지 않는다. 서명된
`AnalysisInputDerivationArtifact-v5`는 이 decode/QC/source selection/identity-regrid 결과가
실제 input tensor digest와 같음을 forecast run과 semantic replay에 결합한다. Background는
관측 시각을 재사용하지 않고 실제 `background_valid_times`, background-model identity,
등록된 cycle-rule, operational-data identity와 frame bytes를 함께 봉인한다. Canonical raw
identity v2는 content뿐 아니라 radar-product와 source-grid identity도 포함하므로 같은
bytes를 다른 product/grid metadata로 재라벨링할 수 없다.
서로 다른 regrid 또는 겹치는 3-frame window가 canonical raw volume
하나라도 공유하면 다른 family에서 재사용할 수 없고, candidate/classifier training raw
identity와 holdout raw identity도 겹칠 수 없다. Candidate와 classifier는 plan 등록 전에
global registry에 commit된 하나의 family-wide signed training-raw receipt를 공유하며,
ledger는 그 canonical payload preimage를 저장하고 scoring에서 다시 검증한다.
과거 deployment 실험에서 생성된 선택 digest, classifier probability, 활성 band, policy,
certified group, horizontal range-geometry payload, radar source와 trust-store snapshot은
historical forecast artifact에 audit bytes로만 남는다. Current `forecast-run-v72`은 이러한
nonempty deployment lineage를 받아들이지 않는다. Legacy 적재는 저장된 canonical payload와
digest의 내부 일관성만 감사하며 current selector나 operational action을 실행하지 않는다.
v54 run은 source-aware selection 이전
계약을 `neural-prior-deployment-lineage-v5-audit`로만 읽고, v53 run은 current-grid binding과 durable geometry
payload 이전 계약을 `neural-prior-deployment-lineage-v4-audit`로만 읽고, v52 run은 physical range deployment 이전 계약을
`neural-prior-deployment-lineage-v3-audit`로만 읽는다. v51 run은 durable decision 이전 계약을 audit-only로 읽고, v50 run은 policy 승인
이전 계약을 `neural-prior-deployment-lineage-v1-audit`로, v49 이하는
`neural-prior-deployment-lineage-v0-audit`로만 읽는다.
따라서 clear-sky 개선으로 echo intensity 악화를 상쇄하거나 절대 calibration 상한만
가까스로 통과하면서 parent보다 불확실성이 크게 악화된 candidate는 승격되지 않는다. Initial-state prior의 target valid time은 prior output
valid time과 같아야 하며, 같은 source/time을 쓰는 withheld target은 실제 feature-
exclusion mask가 target mask를 덮었는지 계산해 확인한다. 불확실성 score는 candidate가
스스로 선택한 valid 영역이 아니라 사전등록 target mask에서 계산하고, 최소 valid
fraction·면적과 parent 대비 abstention 증가 및 NLL abstention penalty를 함께 적용한다.
따라서 caller가 `eligible=True` 객체만 직접 만들어 prior를 승격할 수 없다.

Classifier calibration gate는 weather softmax의 multiclass Brier, conditionally-independent
Bernoulli range heads의 multilabel Brier, weather unknown probability와 no-active-range
probability의 binary Brier를 physical event 동일가중 UCB로 판정한다. 기존 joint-min
surrogate와 ECE는 diagnostic-only다. Sample-size preflight도 known weather/range,
weather/range OOD와 Brier-valid event subset을 각각 확인한다.

현재 scientific promotion evidence는 v32, candidate manifest는 v19, holdout plan은 v37,
holdout scoring artifact는 v15, holdout evaluation은 v24, promotion policy는 v30,
metric support는 v3이다. 각 predecessor generation은 typed audit-only이며 current source와
metric-engine identity를 다시 확인하는 자동 과학 경로에 들어갈 수 없다. 운영 selector는
current package에 노출되지 않고 v32의 operational capability는 명시적으로 비어 있다.
Deployment는 이 프로젝트의 범위 밖이다.
Training lineage는 case별 feature/target/mask/quality member, split, weight,
augmentation seed와 normalization bytes를 포함하는
`TrainingFeatureDatasetArtifact-v5`와 `TrainingDatasetDerivationArtifact-v7`을
요구한다. 같은 physical event의 모든 member는 하나의 train/validation/test split에만
속하며, `NormalizationDerivationArtifact-v1`은 정확한 train member ordering과 실제
statistics shard를 봉인한다. Production archive는 inline base64가 아니라 approved
artifact-store ID와 `<sha256>.npz` 상대경로로 식별되는 content-addressed read-only NPZ
shard로 저장된다. 절대 host path는 semantic digest에 들어가지 않는다. Product-owned loader는 한 번 연
file descriptor에서 byte snapshot을 고정하고, SHA-256과 ZIP/NPY preflight 뒤 실제 tensor
digest를 다시 계산한다. Dataset 전체 shard/archive/expanded/member budget을 먼저
확인하고 trainer는 ordered streaming snapshot iterator만 소비하므로 전체 dataset을 두 벌
메모리에 보존하지 않는다. Validation 뒤 path를 swap/restore해도 이미 검증된 snapshot과
execution subject는 변하지 않는다. 이 계약은 canonical grid-product
feature/target archive를 인증하지만 native polar acquisition decoder까지 인증하지는
않는다.
`sealed_historical` plan은 결과 비공개 escrow를 증명하지 않으므로 연구·shadow audit에만
사용되고 deployment eligibility는 prospective plan에만 부여된다. Candidate-neutral
`PhysicalEventCatalogPlan`은 association·spatial-membership rule, spatial reference,
event merge threshold와 완료시한을 사전등록하고, append-only result 하나만 허용한다.
각 case의 observed envelope와 trusted input-availability가 event evidence와 맞는지 검증하고,
서로 연관되는 event component를 분할하면 거부한다. Catalog ledger append 뒤에만
scheduler-signed training/scoring start receipt를 기록할 수 있으며 evaluation은 그 receipt를
참조한다. Candidate와 classifier training도 같은 association contract에서 signed physical-
event identity와 holdout association component 모두에 대해 분리한다. 이전 promotion evidence·candidate manifest·holdout plan·
evaluation은 원래 payload와 digest를 그대로 검증하는
read-only audit 타입으로만 적재된다. Migration 때 추가된 UCB 컬럼의 기본값 0을 과거
증거의 계산값으로 해석하거나 과거 row를 현재 승격판정에 재사용하지 않는다.

자동학습은 의도적으로 nominal metric weight를 고정한
`frozen_metric_domain`만 승인한다. perturbation 뒤의 confidence·local evidence·
valid mask까지 다시 만든 발행영역 변화는 별도 연구 진단으로 확인한다.

```python
from advar.sensitivity import validate_variational_fsoi_issuance_impact

issuance = validate_variational_fsoi_issuance_impact(
    forecast,
    analysis,
    verification_bundle,
    learning.fsoi,
    policy=policy,
)
assert issuance.metric_domain_contract == "resolved_issuance_domain"
assert torch.allclose(
    issuance.frozen_domain_state_effect
    + issuance.issuance_policy_effect,
    issuance.end_to_end_issuance_effect,
    equal_nan=True,
)
```

이 결과는 full/half P1을 다시 풀고 `ForecastResult`도 다시 생성한다. 자동 개입은
이 resolved-issuance 검증을 필수로 사용하지만 frozen-domain 학습값과는 구분한다.
검증 digest는 source FSOI, nominal forecast와 nominal/full-step input bundle에도
결합되므로 다른 perturbation이나 다른 사례의 유리한 검증을 재사용할 수 없다.
state effect, issuance-policy effect,
end-to-end effect와 coverage before/after, newly-issued/withdrawn fraction을 따로
보존하므로 발행영역 축소로 생긴 겉보기 개선을 state 개선과 혼합하지 않는다.

후보가 많을 때는 FSO를 후보마다 다시 풀지 않는다. 하나의 공통 FSO에서 모든
physical-radar perturbation을 점수화한 뒤 정책이 허용한 상위 K개만
full/half robust P1 re-solve로 검증한다.

```python
ranking_fso = compute_variational_fso(
    forecast,
    analysis,
    verification_bundle,
    sensitivity_config=policy.sensitivity_config,
    adjoint_config=policy.ranking_adjoint_config,
)
ranking = score_candidate_perturbations(
    ranking_fso,
    analysis,
    (("tile-17", SparseRadarPerturbation.from_dense(radar_delta_dbz)),),
    policy=policy,
)
validated = validate_top_k_learning_impacts(
    forecast,
    analysis,
    verification_bundle,
    ranking,
    policy=policy,
    policy_trust_store_path="/etc/advar/learning-policies.json",
)
```

후보는 iterator로 받아 한 번씩 선검사하고 희소 index에서만 sensitivity dot
product를 계산한다. 점수는 metric별 `ranking_scale`과 `ranking_weight`, lead별
`ranking_lead_weights`로 무차원화한다. 기본 `expected_error_reduction`은 error가
감소하는 음의 metric change만 순위에 반영한다. 선검사에 실패한 후보는 top-K를
점유하지 않으며, ranking에는 후보 ID와 거부 이유가 content-addressed된다.
`maximum_candidate_count`는 입력 수를,
`maximum_learning_candidates_to_validate`는 실제 검증 후보 수를,
`maximum_total_robust_resolves`는 full/half solve의 합을 제한한다. 승인 결과는
`RankedLearningOutcome`으로 candidate ID·rank·score·ranking digest를 보존한다.

이 policy는 verification lineage, radar-dynamics domain, active-set·feasibility·
GN 신뢰도, physical-radar perturbation과 baseline branch 인증을 모두 요구한다.
strict factory는 16 km tile, 9 km soft-FSS window, projected-metre centroid와
256 km² perturbation-area 상한을 사용한다. metre FSS는 full 2×2 grid affine의
projected-distance footprint를 직접 사용하며, tile은 행·열의 물리 간격을 따로
해석한다. 실제 `tile_shape_yx`는 FSO 결과에 기록된다. 픽셀 단위 설정은 기존
연구계약과의 호환을 위해 유지된다.
정책 저장소는 root 소유의 group/world 비쓰기 JSON 파일이어야 하며 symlink와
상대경로는 거부된다. 따라서 일반 caller가 policy와 allowlist를 함께 만들어
self-approval할 수 없다. 일반 결과의
`baseline_branch_trusted_total`은 baseline branch만 인증한다. 전체 학습 승인은
`LearningEligibility`만 사용한다.
Taylor 절대오차와 materiality는 단위가 다른 metric마다 따로 지정한다. full/half
step 중 material metric이 하나도 없으면 `no_material_learning_signal`로 거부한다.
Physical radar perturbation은 원 입력의 `min_dbz`/`max_dbz` clamp 안에 남아야 한다.
승인된 `LearningApprovalEvidence`는 policy와 trust-store, FSOI, full/half
분석·forecast, Taylor validation, 후보선정 계보, 실제 common-bias whitener
apply 횟수와 최종 impact digest를 하나로 묶는다.
`validate_variational_learning_impact()`는 저장 직전에 이 결합을 다시 검사하고,
`EpisodeLedger.append_variational_learning_approval()`은 대형 P1 Tensor 대신 이
작은 승인증거만 append-only index에 보존한다.

```json
{
  "contract": "advar-learning-policy-trust-store-v1",
  "approved_policy_digests": ["<64-character-lowercase-sha256>"]
}
```

`lead_minutes`는 원래 forecast 축에서 실제 수반계를 풀 lead만 고른다. metric별
직전 lead 해는 다음 PCG의 초기값으로만 사용하며 true residual을 다시 계산한다.
unit control prior와 field-smoothness graph diagonal preconditioner를 기본으로
사용한다. normal-product 또는 materialized-output byte 예산을 넘으면 부분 결과를
반환하지 않고 fail-close한다. `adjoint_iterations`, `adjoint_normal_products`,
`adjoint_warm_started`, `total_normal_products`로 실제 비용을 감사할 수 있다.
`adjoint_true_residual_norm`은 PCG 종료 후 다시 계산한 수반계 residual의 L2 norm을
기록한다. 이 값은 수반해의 수치 잔차이며, `B H^-1` 연산자 norm을 포함하지 않으므로
민감도 지도의 L2 오차상한으로 해석하지 않는다.
Overlapping common-bias mode는 1회 적용량뿐 아니라
`maximum_whitener_total_operations`도 hot path에서 누적 검사한다. 따라서 PCG가
끝난 뒤 사후 거부하는 대신 설정된 총 연산량을 넘기기 직전에 중단한다. 기본
1회 적용 한도는 2048²·64 spatial mode를 분석 준비 단계에서 거부한다.

`baseline_dynamics_dbz`는 FFT peak와 pair 선택을 고정한 연구용 채널이다. 결과의
`baseline_dynamics_branch_status`는 경로 없음, 미인증, 인증, 무효를 구분한다.
미인증·무효 경로가 있으면 trusted-total 채널을 생성하지 않으며,
`require_baseline_dynamics_branch_validity=True`인 계산은 fail-close한다.

동일한 frozen residual objective에 대해 고정-seed unit Rademacher probe마다 exact
Hessian-vector product와 GN product를 비교한다. `gauss_newton_diagnostics`에는
probe별 상대 curvature defect, 최댓값, exact/GN product 수와 reliability 판정이
저장된다. `require_gauss_newton_reliability=True`이면 설정된 최대 defect를 넘는
분석은 FSO를 발행하지 않는다. probe의 GN product도 전역 normal-product 예산을
소비한다.

`active_set_margins`는 detection classification, 분석·예측 remap cell, output cap,
발행 support/confidence 경계까지의 거리를 보존한다. 기본값은
`low_local_validity` 진단만 발행하고, `require_active_set_margin=True`이면 경계에
가까운 frozen derivative를 거부한다. 이 margin은 민감도 크기를 보정하는 값이
아니라 동일 piecewise-smooth 계약이 유지될 국지 유효성 진단이다.

`feasibility_margins`는 최종 분석의 reachability support, 허용 unresolved-amplitude
fraction까지의 여유, amplitude-confidence gate, motion bound의 정규화 여유와
물리속도 여유, growth bound까지의 여유를 별도로 보존한다. 이 값은 최종
linearization digest와 지연 artifact에 결합된다. 기본 연구 설정은
`low_interior_validity`를 진단으로 남기며,
`require_feasibility_margin=True`이면 수반계를 풀기 전에 경계에 가까운 분석을
fail-close한다. 이는 경계해의 KKT 수반계를 근사하지 않는다. 실제 active
constraint가 있는 분석은 향후 bordered KKT 계약이 구현되기 전까지 자동
관측순위나 학습 승인에 사용할 수 없다.

실제 artifact의 wall time과 process peak RSS는 다음 도구로 측정한다.

```bash
python benchmarks/benchmark_variational_fso.py \
  forecast-run.npz p1-linearization.npz verification.npy \
  --lead-minutes 60,180 --metrics log_echo_mse \
  --metric-domain radar_dynamics_anchored --gauss-newton-probes 4 \
  --max-normal-products 400 --max-output-bytes 536870912 \
  --max-whitener-total-operations 10000000000 \
  --max-wall-seconds 900 --max-peak-rss-bytes 17179869184
```

보고되는 `materialized_output_bytes`는 결과 Tensor의 사전 산정량이다. AD tape와
JVP/VJP workspace까지 포함한 실제 peak memory는 benchmark의 `peak_rss_*`로
별도 확인해야 한다. benchmark v4는 같은 whitener 총 연산 budget을 계산 중에도
강제하고 실제 apply 횟수와 총 연산량을 함께 기록한다. wall time·peak RSS·normal
product·결과 byte budget도 같은 실행에서 검사하며, grid shape와 모든 결과를
`benchmark_digest`로 내용주소화한다.

검증장이 도착하기 전에 프로세스가 재시작될 수 있으므로 수용·수렴한 P1의
최종 선형화는 안전한 NPZ artifact로 보존할 수 있다.

```python
from advar.linearization_artifact import (
    load_p1_linearization,
    save_p1_linearization,
)

save_p1_linearization(analysis, "p1-linearization.npz")
restarted_analysis = load_p1_linearization("p1-linearization.npz")
fso = compute_variational_fso(
    forecast,
    restarted_analysis,
    verification_frames_dbz,
)
```

Current `p1-linearization-v15` loader는 pickle을 사용하지 않고 archive 크기·member
allowlist·각 Tensor digest·전체 artifact digest를 먼저 검사한다. 그 뒤 저장된
control에서 state와 `J^T r`를 다시 계산한다. algorithm bundle 또는 Python,
NumPy, PyTorch, backend capability, deterministic-policy로 구성된 numerical
runtime identity가 달라져도 fail-close한다. 따라서 이 artifact는 임의 환경
사이의 이식 포맷이 아니라 동일 수치계약에서 3시간 지연 FSO를 재개하는 감사
포맷이다. `p1-linearization-v14`는 five-channel bound input이 없는 audit-only
세대로 보존된다.
반복 JVP/VJP는 `ValidatedBoundNeuralPriorInputHandle-v1`에서 경계 시점의 전체
cryptographic validation과 tensor version snapshot을 한 번 수행한다. Matrix-free
hot path에서는 object identity와 in-place version만 확인하고, FSO/P1 연산 종료 시
전체 digest를 다시 검증한다.

`NumericalRuntimeManifest-v2`는 운용상 함께 취급할 수 있는 compatibility digest와
정확한 replay를 위한 exact digest를 분리한다. Exact identity에는 실제 device
model/index, CUDA driver·compute capability, cuDNN/TF32/matmul 설정, thread 수,
PyTorch·NumPy build configuration, OS release 및 설치 distribution/source manifest가
들어간다. Algorithm source manifest도 package 최상위만이 아니라 모든 하위 Python
module의 canonical relative path와 bytes를 재귀적으로 주소화한다.

`.github/workflows/mps-certification.yml`의 `MPS P0 regression`은
`PYTORCH_ENABLE_MPS_FALLBACK=0`인 self-hosted Apple Silicon runner에서 native MPS
availability와 다섯 regression oracle을 실행한다. 여기에는 비대각 SPD PCG,
current projected-grid v6의 3-frame P0 end-to-end nowcast, CPU binary64 speed
authority 및 current source-selection 경계 검사가 포함된다. 이 workflow는 P1
인증·서명·artifact를 생성하지 않는다. 설치 package의 console command는
`advar-nowcast` 하나다.
MPS P0 regression은 지원하지만 full MPS P1은 NO-GO이고 운영 deployment는 범위 밖이다.
Current verification/FSO scoring은 tensor-level float64 evidence contract 때문에 계속
CPU-only다. MPS workflow는 이 경계를 숨기지 않고 current source-selection 요청이
전용 CPU-only 오류로 거부되는지도 검사한다.

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
→ 선택적 additive common-bias covariance whitening
→ 외부 반복에서 고정한 pseudo-Huber IRLS weight
```

`observation_common_bias_std_dbz=σ_b`가 양수이면 대각 표준화 뒤의
additive radar calibration bias를

```text
C = I + σ_b² a aᵀ,  a = sqrt(quality_weight) / observation_std_dbz
```

로 모델링하고, `C^{-1/2}`를 rank-one 공식으로 적용한다. dense covariance는
만들지 않는다. `per_frame`은 세 입력시각마다 독립된 bias mode를,
`all_times`는 분석창 전체에 지속되는 bias mode를 뜻한다.
`observation_common_bias_tile_size_px=0`이면 기존처럼 공간 domain 전체가 한
mode이고, 양수이면 각 square tile이 독립적인 mode가 된다. 격자 크기가 tile
크기로 나누어지지 않아도 오른쪽·아래쪽 ragged tile을 그대로 사용한다.
tile과 temporal scope의 조합은 block별 rank-one inverse square root로 계산되어
dense covariance를 만들지 않는다. 기본 `σ_b=0`, tile size `0`은 기존 residual
Tensor를 그대로 반환하므로 이전 diagonal-R 결과가 변하지 않는다. 세
common-bias 설정은 analysis config·forecast run·최종 linearization digest에
포함된다. P1 FSO의 detected/censored/weight 채널도 같은 비대각 whitening을
통과한 frozen stationarity의 혼합미분을 사용한다. 운용 P1에서는 bias 크기,
시간 scope, tile 크기를 calibration profile에 모두 명시해야 한다.
overlapping mode의 basis와 작은 correction matrix는 분석 준비 시 한 번만
고유분해하여 frozen linearization에 저장한다. 이후 residual·JVP·VJP hot path는
저장된 선형연산만 적용한다.

regular tile 대신 실제 레이더·사이트·모자이크 영역을 사용하려면 정수 group
map을 전달한다. `-1`은 common-bias mode에서 제외되고 0 이상의 같은 label은
하나의 mode가 된다. label 숫자는 canonical compact partition으로 변환되므로
임의의 label 재번호화는 같은 digest를 만든다.

```python
from advar.variational import (
    AnalysisConfig,
    observation_common_bias_group_map_digest,
    variational_nowcast,
)

group_digest = observation_common_bias_group_map_digest(
    radar_site_index,
    temporal_scope="all_times",
)
forecast, analysis = variational_nowcast(
    frames,
    analysis_config=AnalysisConfig(
        observation_common_bias_std_dbz=0.5,
        observation_common_bias_scope="all_times",
        observation_common_bias_group_map_digest=group_digest,
    ),
    observation_common_bias_group_index=radar_site_index,
)
```

group map은 `[H,W]` 또는 `[3,H,W]` 정수 Tensor이며 tile mode와 동시에 사용할
수 없다. 연구모드는 digest를 생략하면 canonical map digest를 자동 결합하지만,
운용모드는 사전 승인된 `AnalysisConfig`에 정확한 digest가 있어야 한다. map은
analysis-input·forecast-run·linearization digest와 `p1-linearization-v15`
artifact에 포함된다. CLI에서는
`--observation-common-bias-group-map groups.npy`를 사용한다.

레이더 footprint가 겹치거나 seam을 부드럽게 연결해야 하면 `[K,H,W]` 또는
`[3,K,H,W]` mode-weight Tensor를 사용할 수 있다. 각 weight는 `[0,1]`이고
화소마다 `sum_k(weight_k**2) <= 1`이어야 하므로 설정된 common-bias 표준편차가
국지 marginal 상한으로 유지된다. 최대 mode 수는 64다.

```python
from advar.variational import observation_common_bias_mode_weights_digest

mode_digest = observation_common_bias_mode_weights_digest(mode_weights)
forecast, analysis = variational_nowcast(
    frames,
    analysis_config=AnalysisConfig(
        observation_common_bias_std_dbz=0.5,
        observation_common_bias_scope="all_times",
        observation_common_bias_mode_weights_digest=mode_digest,
    ),
    observation_common_bias_mode_weights=mode_weights,
)
```

이 경로는 표준화된 mode basis의 작은 `K×K` Gram만 고유분해하여
`(I + A A.T)^(-1/2)`를 정확히 적용한다. `HW×HW` covariance는 만들지 않는다.
최종 선형화에는 대형 weighted basis를 한 벌 더 저장하지 않고 원래 mode
weights와 작은 correction matrix만 보존한다. `[K,H,W]` 입력도 세 시각으로
복제하지 않는다. `maximum_common_bias_mode_weight_bytes`,
`maximum_common_bias_whitener_apply_operations`,
`maximum_frozen_whitener_bytes`, `maximum_linearization_bytes`는 큰 allocation이나
비현실적인 `K×T×H×W` 적용 전에 fail-close한다.

대형 `.npy`를 만들거나 읽기 전에 shape와 dtype만으로 같은 예산을 확인할 수 있다.

```python
import torch
from advar.variational import estimate_common_bias_resources

estimate = estimate_common_bias_resources(
    (64, 2048, 2048),
    (3, 2048, 2048),
    dtype=torch.float32,
    temporal_scope="all_times",
)
assert not estimate.within_budget
```

CLI도 mode 파일을 read-only mmap으로 열어 이 검사를 먼저 수행하고, 통과한 경우에만
한 번 복사해 Tensor로 만든다. 이 정적 추정은 불가능한 입력의 조기 차단용이며,
허용된 입력의 실제 wall time과 peak RSS는 benchmark로 별도 검증해야 한다.
regular tile·group map·overlapping mode는 서로 배타적이며 mode Tensor와 digest는
지연 FSO artifact에 보존된다. CLI에서는
`--observation-common-bias-mode-weights modes.npy`를 사용한다.

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
초기장 field가 같은 관측을 대신 설명하는 효과는 field normal equation을
dynamics basis별로 세 번 matrix-free PCG로 풀어 Schur complement
`G_dyn|field`로 별도 기록한다. 이 field-conditioned 정보도 보정 전에는
진단일 뿐 분석 수용 gate가 아니다.
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
은 P0와 `--variational` P1 profile을 각각 지원한다. 두 profile 모두 완전한
시각·격자 계약, 물리속도 상한과 명시적인
PSR·pair 운동/성장 불일치·pair 신뢰도 우위·국지 성장 residual·성장 overlap
support·물리면적,
임계값, 검증된 상태경로·레이더 관측 support 발행 임계값, 선행시간
confidence, 배경 기여율, 속도불확실성·위치오차 길이척도와
`--operational-calibration-manifest`를 요구한다. P1은 여기에 대각 관측오차,
additive common-bias 크기·시간범위, amplitude
정보량·적분량·면적·성장, 물리 causal/amplitude 거리, projected m/s 운동증분,
국지 dBZ·precision, posterior saturation 보정을 추가한다.

manifest v2는 P0/P1 종류, runtime profile, 설치된 `advar` Python module 전체의
algorithm bundle, calibration/validation dataset, QC·관측오차·배경모델 identity,
서로 겹치지 않는 기간, validation 사례·regime 분포, metric 정의·방향·수용임계값을
canonical JSON으로 묶는다. 모든 metric이 임계값을 통과해야 한다. 배포 allowlist의
manifest digest를 `--approved-operational-calibration-manifest-digest`로 별도
제공해야 하며, 누락·불일치·algorithm 변경은 실행 전에 거부한다. P1의 두
amplitude 정책은 `operational_fallback`으로 고정한다.
실행 시 `--radar-class`, `--qc-pipeline-digest`,
`--observation-error-model-digest`, `--background-model-digest`도 명시하며,
manifest에 보정된 data identity와 다르면 fail-close한다.

출력 `forecast.npz`에는 다음 항목이 들어간다.

- `output_contract_version`: 현재 `nowcast-npz-v76`
- `forecast_run_artifact_version`: 현재 `forecast-run-v72`
- `forecast_run_digest`, `input_bundle_digest`
- `grid_time_contract_json`, `grid_time_contract_digest`
- `run_background_age_minutes`: 실제 입력계약의 배경 age

`forecast-run-v72`은 typed verification-target identity와 target-source current trust를
결합한 five-channel CPU-only scoring generation v10, two-phase raw observation slot과
canonical raw-volume identity 단위의 전역
sampling reservation, 같은 family의 rolling-window membership, source-registry와
location-registry가 결합된 mosaic range domain,
역할 분리된 ledger issuance
receipt/promotion certificate/operational decision signature, 그리고 input-plan의
sub-second decision deadline chronology, final decision-row publication receipt와
terminal activation receipt와 writer-lock-bound commit authorization receipt를 함께
검증한다. QC-invalid 관측은 registered finite fill/zero/sentinel로 canonicalize되어
classifier와 learned prior에 유입되지 않으며, signed
`AnalysisInputDerivationArtifact-v5`의 canonical JSON과 digest도 NPZ에 보존된다.
`forecast-run-v71` 이하는 audit-only다.

Contract registry는 current generation 상수와 선언된 capability 표만 제공한다.
test 이름, 최소 metadata envelope 또는 caller 제공 action flag를 historical replay의
대리물로 사용하지 않는다. 장기 cold replay는 실제 보존된 artifact bytes를 해당
family loader가 직접 복원하고 원래 digest를 재현하는 시험이 있는 세대에만 주장한다.
원본 archive가 없는 세대는 decoder 코드가 남아 있더라도 cold-replay-certified로
간주하지 않는다.

Current `forecast-run-v72`은 nonempty
`neural-prior-deployment-lineage-v19`를 전용 비지원 예외로 거부한다. 과거
deployment certificate, decision artifact와 activation receipt는 typed audit
lineage로만 적재되며 current scientific run 생성, 모델 선택 또는 운영 action에 사용할
수 없다. 내부적으로 canonical JSON, digest와 caller-generated signature가 모두 일치해도
운영 capability 집합이 비어 있는 동안 이 경계는 열리지 않는다.

Semantic holdout scoring은 모든 case와 tensor role이 하나의 execution
device/runtime을 사용해야 하며 automatic promotion scoring은 CPU-only다. Current
archive의 과거 certification digest 필드는 항상 `null`이고, 이를 생성하거나 현재
scoring authority로 전환하는 policy/evidence 실행 경로는 없다. 아래
certificate·receipt·state-machine 설명은 과거 deployment engineering의
audit scaffold이며 current scientific package에서 발급·선택·활성화할 수 없다. Historical
scaffold는 EpisodeLedger가 발급한 root-signed promotion deployment certificate와 외부
authority, learning-policy 및 당시 raw-ingestor trust store를 함께 요구했다. Authority
trust-store v4는 승인된 ledger-instance digest마다
유일한 canonical `index.sqlite` 절대경로도 root-owned 설정으로 고정하므로,
Python 객체를 위조해 공격자 DB로 verifier를 재지정할 수 없다. 저장되는 operational
selection 전체도 별도의 authority-signed
`OperationalDeploymentDecisionCertificate-v8`와 먼저 durable commit된 ledger decision
receipt, final certificate row의 commit 뒤 ledger signer가 발급한
`OperationalDecisionPublicationReceipt-v3`, 최종 `published/usable` 상태와
publication payload commit 및 activation authorization 시각을 봉인한
`OperationalDecisionActivationReceipt-v3`와
`OperationalDecisionCommitAuthorizationReceipt-v2`에
결합되므로, 재시작 시 regime evidence,
policy threshold 또는 selected prior를 재해시해 바꾸는 경로가 차단된다.
이 authorization은 SQLite의 물리 fsync 완료시각을 주장하지 않는다. Finalizer는
non-waiting `BEGIN IMMEDIATE`로 writer lock을 먼저 획득하고 signed guard interval
안에서 authorization receipt, `published`, `usable`을 한 transaction으로 기록한다.
아래 issuance state machine은 historical audit scaffold이며 current package에서
operational action으로 도달할 수 없다. Historical issuance는 cycle-id 기반 state machine으로 재개되며 terminal
`published` row만 선택 가능하다. Activation receipt는 먼저 `usable=0` 상태로 durable하게
staging되고, commit authorization receipt의 INSERT와 `published/usable` 전환이 하나의
transaction에서 수행된다. Certificate signer, decision-row transaction, publication
signer 또는 final activation이 실패하면 동일 cycle은 이미 durable한 certificate/receipt를
재사용해 재개되고, commit authorization receipt가 없는 publication은 계속 `usable=0`이다.
Raw ingestor trust-store v2는 plan에 고정된 snapshot과 provenance commit, scoring
replay/completion, promotion, certificate 발급 및 operational decision 시점의 current
root-owned store 양쪽에서 attestation 시각의 key validity/revocation을 대조한다.
Current store의 content digest는 replay-v15 manifest, scoring-v14 artifact, scheduler가
봉인한 completion output, promotion evidence v31, promotion deployment certificate v6와
operational decision certificate에 연속 결합된다. Certificate/publication 서명 전후와
activation 직전·직후에도 store를 다시 읽고, historical `forecast-run-v69` audit load에서도 외부
store와 대조하므로 이후 revocation view가 달라지면 기존 certificate를 automatic
deployment에 재사용할 수 없다.
Index schema 42는 release approval, monotonic runtime activation head, replay, scoring completion, promotion evidence와 promotion
certificate를 immutable payload와 별도의 raw-trust activation row로 기록한다. 각
payload는 처음에는 `usable=0`이며 current store 재검증을 거친 activation만
`usable=1`로 소비된다. Activation 직후 store 변경은 `usable=0`으로 fail-close된다.
Analysis-input provenance는 별도의 `prepared/active/expired` 상태와 payload/activation
시각을 보존한다. Exact retry와 startup reconciliation은 content-addressed directory를
single-open file-descriptor snapshot으로 읽고 NPZ·metadata checksum, raw-ingestor receipt,
slot binding, source coverage, raw-to-derived tensor replay, processor signature와 current
trust를 모두 다시 검증해 prepared row 또는 rename 뒤 orphan directory를 idempotently
활성화한다.
training/raw-slot/raw-resolution은 동일한 global registry chain을 사용한다.
일반 미래 운영 cycle은 별도의 `OperationalAnalysisInputProvenancePlan-v1`과
`OperationalRawVolumeResolutionReceipt-v2`를 사용하므로 promotion holdout family를
소비하지 않는다. 두 경로 모두 동일한 signed raw receipt, processor authority와
raw-to-grid replay를 사용한다.
운영 raw slot의 해석은 `OperationalRawResolutionHistoryEntry-v1` sequence로 별도
보존한다. 동일 identity 재사용은 허용하지만 missing→resolved correction,
resolved→resolved supersession, resolved→missing cancellation은 직전 entry digest와
plan에 고정된 processor signature가 없으면 거부된다. Schema-38 이전 operational
resolution은 immutable legacy anchor로 이력 genesis에 결합되며, 새 entry와 composite
receipt는 append 시 canonical decode와 signature 검증을 다시 수행한다. Raw ingress의
quality weight는 finite `[0,1]`
범위를 벗어나면 즉시 fail-close한다.

Learned prior 실행은 runner 내부의 마지막 입력 cache를 사용하지 않는다.
`BoundNeuralPriorInput-v1`이 canonical dBZ, QC mask, quality, observation standard
deviation, source availability와 완성된 five-channel tensor를 불변 context로 묶으며,
reproduce/JVP/VJP와 P1 linearization restart는 이 context를 명시적으로 전달한다.
반복 미분용 validated handle은 caller tensor와 분리된 private clone을 봉인하고 public
access에는 defensive copy만 반환하므로 PyTorch `.data` alias도 평가점을 바꾸지 못한다.
따라서 같은 dBZ라도 QC context가 다른 동시 실행과 delayed FSO가 서로 오염되지 않는다.

`TrainingFeatureDatasetArtifact-v5`는 `TrainingDatasetMember` record와 외부 immutable
feature/target/normalization NPZ shard를 content-address한다. Loader는 한 번 연 file
descriptor의 byte snapshot에서 archive SHA-256, ZIP member, dtype/shape와 tensor digest를
재계산하며, training execution v9의 start/completion subject는 이 validated snapshot-set
digest와 exact equality로 결합된다. Product 밖의 trainer도 검증 뒤 원래 path를 다시 읽는
대신 이 snapshot handle만 소비해야 한다. 각 target는 value,
valid-mask, quality, unit, shape, source plan/receipt, physical event, valid time,
training cutoff와 승인 processor signature를 가진
`TrainingTargetDerivationArtifact-v4`에 결합된다. `quality[~valid_mask]`는 0이며 positive
effective weight가 없는 target는 거부되고 product-owned loss는 정확히
`sum(mask*quality*error)/sum(mask*quality)`를 사용한다. Target source
identity/time/event/object는 plan에 별도로 고정된 `TrainingTargetSourceReceipt-v2`와
root-signed `TrainingTargetSourceTrustStore-v1`의 유효 key epoch, source-contract 및
radar/product scope가 없으면 사용할 수 없다. Current trust는 dataset seal, training
start/completion과 scientific promotion에서 다시 대조된다. Historical deployment
issuance 및 `forecast-run-v69` audit load도 저장된 trust snapshot을 검증하지만 current
operational action에는 도달하지 않는다. 각 검증의
시작/종료에서 다시 대조된다. Target는 nonempty·finite이어야 하고
valid time이 등록된 training-window union 안에 있어야 한다. Tensor/QC 표현이 달라도
holdout verification target와 source identity/valid time이 같으면 재사용으로 거부된다.
Mosaic source가 실제로 도착하지 않은 경우에는 synthetic all-invalid raw volume을
만들지 않고 `MissingRawObservationReceipt-v1`로 slot/site/time, outage·late·corrupt·
unavailable 사유와 ingestor authority signature를 봉인한다. Source-history map이 이
missing receipt를 선택하면 fail-close하고, 모든 source가 missing인 input time은
명시적인 source-unavailable mask와 canonical invalid learned-input channels로 재생된다.
- `displacement_yx`: `(row, column)` pixel/step
- `grid_velocity_mps_yx`, `displacement_mps_yx`: 호환용 grid-axis
  `(row, column)` m/s
- `projected_velocity_mps_xy`: affine 계약을 적용한 projected `(x, y)` m/s
- `analysis_config_json`, `analysis_config_digest`, `analysis_input_digest`
- `operational_runtime_profile_digest`: P0/P1 운용 설정과 격자 정체성의
  content address; 연구모드에서는 빈 문자열
- `operational_calibration_manifest_json`,
  `operational_calibration_manifest_digest`: 운용 hindcast 보정 근거와 그
  content address; 연구모드에서는 빈 문자열
- `operational_calibration_approval_digest`: 배포 allowlist가 승인한 manifest
  digest; 연구모드에서는 빈 문자열
- `operational_data_identity_json`, `operational_data_identity_digest`: 실행이
  선언한 radar·QC·관측오차·배경모델 identity와 그 content address
- `forecast_dbz`: `[18, H, W]`
- `valid_mask`, `state_echo_linear`, `source_support`,
  `path_verified_source_support`, `verified_source_support`,
  `local_motion_verified_support`, `local_growth_verified_support`,
  `local_dynamics_verified_support`,
  `observation_verified_source_support`,
  `background_verified_source_support`, `forecast_path_verified_support`,
  `forecast_verified_support`, `forecast_local_motion_verified_support`,
  `forecast_local_growth_verified_support`,
  `forecast_local_dynamics_verified_support`,
  `forecast_observation_verified_support`,
  `forecast_background_verified_support`
- `forecast_velocity_uncertainty_mps`,
  `motion_evidence_uncertainty_multiplier`,
  `growth_evidence_uncertainty_multiplier`,
  `forecast_position_uncertainty_m`, `forecast_log_growth_uncertainty`,
  `maximum_growth_saturation_excess`,
  `posterior_velocity_uncertainty_mps`,
  `posterior_log_growth_uncertainty_per_step`,
  `p1_velocity_saturation_uncertainty_mps`,
  `p1_log_growth_saturation_uncertainty_per_step`, `forecast_confidence`: pair
  disagreement, pair 수·PSR, background tendency age, 성장모델 상한 초과량과
  P1 field-conditioned posterior에서 계산한 선행시간별 불확실성과 confidence.
  P1 총 불확실성은 posterior가 기존 model-error floor를 대체하지 않으며
  `sqrt(posterior² + model_error² + saturation²)`로 계산한다. bounded decoder의
  safe margin 안에서는 saturation 항이 0이고, 경계에 가까워질수록 보정된
  최대 배수까지 증가한다. 운용 P1 candidate가 safe margin을 침범하면
  confidence를 발행하지 않고 P0로 fallback한다.
- `radar_state_anchored_valid_mask`,
  `radar_dynamics_anchored_valid_mask`, `background_dynamics_mask`:
  현재상태의 레이더 evidence와 미래 tendency의 레이더·배경 출처를 분리한 mask.
  `radar_anchored_valid_mask`는 state-anchored 호환 alias다.
- `source_support`는 상태가 정의됐는지를,
  `path_verified_source_support`는 국지 에코 위치경로가 검증됐는지를,
  `verified_source_support`는 현재시각의 직접 source evidence를 나타낸다.
  `local_motion_verified_support`는 실제로 선택된 미래 motion pair를 그 pair의
  관측구간 `(previous_index, current_index)`에서 먼저 검사하고, pair 종료시각의
  일치 evidence만 선택 운동으로 현재시각까지 수송한다. 따라서 중간시각에서
  소멸한 에코가 최종시각의 다른 에코와 우연히 만나는 endpoint 일치는 경로를
  인증하지 않는다. `local_growth_verified_support`도 선택된 growth pair의 실제
  관측구간에서 계산한 국지 log-growth residual이
  `maximum_local_growth_log_error_per_step` 이내인지를 추가로 요구한다.
  BLENDED 선택은 선택된 모든 pair의 교집합을 사용하고,
  `local_dynamics_verified_support`는 motion·growth 두 support의 교집합이다.
  한 과거 candidate가 여러 현재 에코를 동시에 인증하는 모호한 claim은 모두
  fail-close한다. clear-sky source evidence는 그대로 보존하며 이 필드들은
  국지 확률이나 국지 속도분산이 아닌 보수적인 dynamics evidence mask다.
  과거 source는 물리 footprint 안의 최신 에코와 일치할 때 path evidence만
  제공하며, 최신 결측부를 채운 과거 상태를 state-verified로 승격하지 않는다.
  관측과 배경의 엄격한 evidence support도 별도로 보존한다.
  수용된 P1은 전체 support를 자동 승격하지 않고, 최신 detected 관측의
  precision 하한과 표준화·절대 dBZ 오차 또는 censored 관측의
  precision·detection-limit 조건을 국지적으로 만족한 화소만
  observation/state verified로 기록한다. 이 최신 상태 적합도는 P1의
  motion/growth evidence로 재사용하지 않는다. P1 분석 운동·성장으로 두 인접
  관측구간을 각각 정렬한 국지 motion/growth evidence를 별도로 계산하며,
  `radar_dynamics_anchored_valid_mask`는 finite posterior와 이 국지 교집합을
  모두 요구한다.
  연구모드에서는 예전 persistence를 유지하지만, 운영모드는
  `minimum_publish_verified_support`로 검증되지 않은 경로를 발행에서
  제외한다. `forecast_path_verified_support`,
  `forecast_verified_support`, `forecast_local_motion_verified_support`,
  `forecast_local_growth_verified_support`,
  `forecast_local_dynamics_verified_support`는 각 support를 선행시간별로
  수송한 진단이다. 이 선행시간 evidence는 에코와 같은 remap cell에서
  모든 채널을 한 번에 수송한 immutable `ForecastEvidenceFields`로 계산하여,
  valid-mask·artifact·M0가 동일한 값을 공유한다. live confidence와
  dynamics-anchored mask는 국지 dynamics evidence도 함께 사용한다.
  운용모드는 `minimum_publish_confidence`,
  `minimum_publish_observation_verified_support`,
  `maximum_publish_background_fraction`도 명시해야 한다.
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
결합한다. 운동은 PSR과 recency를 함께 사용한다. 성장 confidence는 PSR뿐
아니라 echo overlap support 또는 물리면적, 이전·현재 평균 echo, 선택 운동에서
전역 성장률을 제거한 국지 log-ratio 정렬오차를 함께 사용한다. 따라서 작은
echo 표본의 높은 motion PSR가 넓은 성장증거를 자동으로 압도하지 않는다.
성장 충돌에서는 이 confidence가 `minimum_pair_confidence_ratio` 이상 우세한
pair만 사용하고, 우위가 없으면 growth persistence로 fail-close한다. 기본
임계값과 confidence 식은 합성·연구 설정이며 실제 레이더 hindcast로 보정해야
한다.

인접 pair가 하나만 유효하고 20분 long pair도 유효하면 두 후보를 독립적으로
검사한다. near-echo completeness가 이미 에코 주변 자료 가용성을 검사하므로
에코와 무관한 전체 도메인 coverage는 confidence에 곱하지 않는다. 운동
신뢰도는 PSR에, 성장 신뢰도는 위의 growth evidence에 시간간격·형태변형
위험을 나타내는 `long_pair_confidence_penalty`를 곱한다. long 또는 adjacent 후보가
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
- `analysis_converged`, `analysis_outer_converged`
- `analysis_final_linearization_stationary`, `analysis_final_robust_stationary`
- `analysis_final_irls_fixed_point`, `analysis_p1_forecast_eligible`
- `analysis_posterior_eligible`, `analysis_fso_eligible`
- `analysis_degraded`, `analysis_used_fallback`
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
  `analysis_maximum_object_soft_echo_area_ratio_by_time`,
  `analysis_minimum_object_count_ratio_by_time`: 작은 객체 실패가
  전체 시간별 비율에 희석되지 않도록 보존한 최악값. unresolved fraction은
  원 객체별로 계산하고, 에코 적분·면적은 물리 tolerance footprint가 겹치는
  객체를 하나의 matching group으로 묶어 같은 예측 에코를 중복 귀속하지 않는다.
  초기 established echo에서 도달 가능한 예측량도 precursor group 분자에서
  제외하므로 인접한 기존 에코가 신규 객체를 대신 설명할 수 없다. object-count
  ratio는 각 matching group의 예측 threshold component 수를 관측 component
  수로 나눈 값의 시간별 최솟값으로, 여러 셀이 하나로 붕괴하는 topology 실패를
  별도로 기록한다. 운용 profile은
  `--minimum-object-count-ratio-for-confidence`를 명시적으로 보정해야 한다.
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
- `analysis_field_conditioned_dynamics_data_gram_eigenvalues`,
  `analysis_field_conditioned_dynamics_data_information_trace`,
  `analysis_field_conditioned_dynamics_data_effective_dimension`: 초기장 field
  nuisance를 prior와 공간 smoothness 아래에서 제거한 dynamics Schur-complement
  정보. `analysis_field_conditioning_maximum_relative_residual`은 세 field PCG
  solve의 최대 실제 상대잔차이며, solve가 수렴하지 않으면 조건부 정보는 `NaN`
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

현재 schema 15 episode는 lineage 없는 `baseline_scores`를 거부하고
`direct_observation_impact`만 저장한다. 동일 issue time, verification mask,
metric, grid와 baseline model/run digest를 묶는 계약이 추가되기 전에는
정규화 reward를 생성하지 않는다. schema 1–14의 기존 episode는 계속
검증·열람할 수 있지만 새 episode로 자동 승격되지는 않는다.

```python
from datetime import datetime, timezone

from advar.ledger import (
    EpisodeLedger,
    ModelContract,
    SensitivityEpisode,
)
from advar.nowcast import nowcast
from advar.sensitivity import compute_sensitivity_snapshot
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
평가할 수 있다. 현재 episode schema는 v17, model-contract hash schema는
v11이며 기존 schema 1–15를 그대로 검증한다. 과거 episode에
존재하지 않던 conflict나 selection 값을 임의로 보간하지 않으므로 서로 다른
context 계약이 같은 학습집합으로 섞이지 않는다.
M0 `trust_components`의 `pair_consistency`는 기본적으로 충돌한 성분 하나당
`SensitivityConfig.pair_conflict_trust_penalty=0.5`를 곱한다. 두 성분이 모두
충돌하면 0.25가 되며, 이 값은 sensitivity config digest에 포함된다. 기본값은
연구용 보수적 prior이고 검색·운용 임계값은 실제 hindcast로 보정해야 한다.
M0의 선행시간별 `forecast_confidence`는 발행 artifact의 동일 필드를 그대로
사용한다. 이 값은 검증된 현재 상태 support에 P0 dynamics evidence 또는 P1
field-conditioned posterior 기반 위치·로그성장 불확실성 감쇠를 적용하며
nowcast config digest에 포함된다. 보정된 확률이 아니라 명시적인 연구용
evidence score다.
`path_evidence_by_metric`은 각 metric의 `abs(dJ/dforecast)`로 이 confidence를
가중한 값이다. `observation_source_fraction_by_metric`은 같은 가중치에서 관측
origin state의 비율을 진단한다. 자동 trust는 이 주변비율을 path evidence와
따로 곱하지 않고, 관측 source이면서 실제 검증된 교집합을 직접 가중한
`observation_verified_evidence_by_metric`만 사용한다. 배경 교집합은
`background_verified_evidence_by_metric`에 별도로 저장하며, 두 검증 채널의
합은 path evidence와 일치해야 한다. 따라서 공간적으로 분리된 관측 origin과
검증 evidence가 거짓 nonzero trust를 만들지 않는다. 이 배열과 최종 집계값은
episode schema v17에 보존된다. 같은 manifest에는 verification contract와
content digest가 항상 저장되며, 완전한 `VerificationBundle`을 사용한 경우
valid time·grid·radar product·QC pipeline digest도 함께 보존된다.

### M0와 P1 FSO·FSOI의 엄밀한 경계

현재 이동·성장 추정은 FFT peak의 이산 선택을 포함한다. 따라서 M0가
계산하는 관측 민감도는 최신 영상이 예측 초기장으로 들어가는
`partial_direct_latest_dbz_fixed_control` 경로뿐이다.
발행되지 않은 예측과 유효한 최신 직접관측이 없는 사례는 거부한다.

- `-20분`, `-10분`: 직접 예측 경로 없음
- `0분`: 고정된 `(dy, dx, log_growth)`에서 직접 dBZ 민감도 제공
- 분석을 통한 간접 민감도: P0 M0에는 포함하지 않음
- P1 frozen-final 관측 민감도 FSO: `VariationalFSO`로 세 시각 모두 제공
- P1 signed 관측영향 FSOI: 명시적인 `VariationalObservationPerturbation`이
  있을 때만 제공
- 자동 일반화 기억 승격: counterfactual approval과 realized intervention을
  분리해 저장하며, 모델 갱신기는 별도 정책으로 유지

이 구분은 M0 manifest와 SQLite에 명시된다. 간접 민감도를 0으로 저장해
“효과 없음”으로 오해하게 만들지 않는다. P1 `VariationalFSO`는 detected dBZ,
censor threshold와 observation objective weight의 frozen-final 국지 implicit
sensitivity를 분리한다. `VariationalFSOI`는 이 민감도에 digest-bound 명시적
perturbation을 곱한 signed first-order impact다. 두 계산 모두 최종 IRLS
weight, active set, remap cell, observation classification과 baseline을 고정하며,
재계산한 frozen·robust stationarity와 IRLS fixed-point 오차가 설정 임계값
이하여야 한다. outer-loop 선택
자체의 변화와 검증된 baseline-normalized reward는 포함하지 않는다. EFSO는 실제
ensemble 통계를 요구하는 별도 API이며 deterministic FSOI와 혼합하지 않는다.
P1 FSO·FSOI Tensor 자체는 M0 episode ledger에 저장하지 않지만,
자동학습 wrapper는 동일 frozen 준비구조에 실제 physical-dBZ perturbation의
full step과 half step을 적용해 robust P1 분석을 각각 다시 풀고, 선택된
lead·metric의 실제 변화와 1차 예측을 비교한다. 절대+상대 Taylor 오차, 물질적
영향의 부호, remap/output-cap branch 중 하나라도 실패하면
`first_order_valid=False`로 거부한다. 이 값은 명목 metric domain을 고정한
조건부 영향이므로 결과명도 `frozen_domain_learning_impact`로 제한한다.
material metric이 없는 수치잡음은 학습으로 승격하지 않으며, 승인된 결과는
content-addressed learning evidence만 원장에 별도로 기록할 수 있다.
3시간 지연 재계산에 필요한 frozen linearization은 content-addressed
`p1-linearization-v15` artifact로 안전하게 보존·재적재할 수 있다.
P1 분석상태는 기존 M0 직접민감도 API에서 계속 provenance 검사로 거부된다.
