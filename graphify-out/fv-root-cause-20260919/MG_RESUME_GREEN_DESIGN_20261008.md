# PR #259 model-guided resume at 69c4 — GREEN 설계 검토

날짜: 2026-10-08
판정: **69c4a794에서 같은 bounded full-gradient 정책을 다시 시작하는 수치 계획은 타당하다.** 새 시작점의 `alphaPhi`·`alphaJ`는 fresh HVP 전까지 숫자로 확정되지 않는다.
범위: 저장된 model-guided 3-step 결과와 tangent/분기 archive를 정적 산술·provenance 관점에서 읽기 전용 확인했다. 새 FV, J, gradient, HVP, PCG, forecast, response/reanalysis는 실행하지 않았다.

## 올바른 PR259 시작점

새 resume base는 이전 tangent-loop 초기점 `05889584…`도, e29 `f462…`도 아니라 PR258 3-step continuation이 최종 확정한 **`69c4a794730750c09c1bcddf542b66131f64f5bfec362e02ac269413a75d4992`**다. `MODEL_GUIDED` raw의 `current_control_sha256`, 마지막 iteration의 `accepted_control_sha256`, result summary의 `final_control_sha256`가 일치한다. 전체 제어·파라미터·archived observation/truth identity는 계속 동일한 fixed original problem 계보 안에 있다.

시작점의 raw state는

```text
J      = 0.06136925244260338
Phi    = 0.018131732819468303
||g||2 = 0.1904296868635156
||g||∞ = 0.09724503382238599, 최대 index c[24]
```

다. c[24]는 26-vector에서 0-based flow latent이며 growth latent는 c[25]다. 이 점의 tangent projection norm은 `0.1699788452`, 단위 Euclidean normal gradient는 `−0.0858525354`, `||g_T||/|g_N|≈1.97989`다. 즉 flow-face 정상성이나 normal minimum으로 간주할 수 없다. full-26 unconstrained `d=−g`는 `||d||₂=0.1904296869`; radius `.05`만 적용했을 때 step cap은 `0.05/||d||₂=0.2625641034`다. 최종 initial alpha는 새 fresh HVP의 `alphaPhi`도 함께 계산한 뒤 정한다.

## Archived alpha 및 곡률 공식

이전 e29 tangent path의 저장 HVP/gradient로 old-path 값을 재산술하면

```text
alphaPhi(old tangent) = -gᵀHd / ||Hd||² = 0.00028953045674
alphaJ(old curved chart) = -gᵀd / (dᵀHd + gᵀGamma'') = 0.00051401286187
Gamma'' pivot c24     = 8.28349418e-6
gᵀGamma''             = 1.23377967e-7
```

가 된다. 이들은 e29의 **이전 tangent 방향** 전용 값이며 69c4의 방향·operator·step cap으로 재사용할 수 없다. PR259에서는 매 현재점 fresh `d=−g`, fresh `Hd=H(c)d`를 계산한다. Raw 26-control 경로 `c(alpha)=c+alpha d`는 affine여서 `Gamma''=0`; J quadratic estimate가 필요하면 `J''=dᵀHd`이고 `alphaJ=−gᵀd/(dᵀHd)` (분모가 양수일 때)다.

새 방향에서

```text
alphaPhi = -gᵀHd / ||Hd||²
```

는 `g+alpha Hd` 선형화 gradient의 제곱 norm을 최소화하는 양의 model scale이다. `d=−g`이고 smooth branch의 대칭 Hessian에서 `gᵀHd<0`이면 `dᵀHd=−gᵀHd>0`; Cauchy–Schwarz로 `alphaPhi ≤ alphaJ`다. 그러나 69c4의 fresh HVP가 archive에 없으므로 이 비교의 새 수치는 알 수 없다. `alphaPhi`는 initial cap일 뿐 actual `Phi` decrease guarantee가 아니다.

## 정적 면 sign·branch context

정수 vertex basis `psi_k=(i,j,ij,(j²−i²)/2,ij²)`와 fixed coefficient limits로 정적 face flux를 재계산했다. 이전 tangent accepted `05889584…`에서 현재 69c4로 가며 sign이 바뀐 정적 face는 하나다.

```text
zero-based Qy[3,2]   : −5.9106127381e−7 → +1.3794752636e−5
one-based Qy[4,3]    : same selected face
all Qx 24 faces      : sign unchanged
other Qy 24 faces    : sign unchanged
exact-zero faces     : 0 at the compared endpoints
```

따라서 새 base에서는 선택 face가 이미 양의 side에 있다. `Qy[4,3]`을 0으로 고정하거나 tangent chart를 이어 쓸 근거가 없다. 정적 sign은 endpoint geometry일 뿐 전체 3600-stage branch path 인증이 아니다. 전 단계는 analysis/future branch hash가 바뀌었으므로 69c4의 **자기** strict branch를 fresh 재구성하고 그 새점에서 시작해야 한다.

## Resume iteration math/policy

각 iteration은 current committed point의 fresh full objective/gradient 및 strict own branch를 기준으로 한다. Fresh gradient를 raw current-control ordering (field c[0:20], flow c[20:25], growth c[25])에 둔 뒤 전체 26-vector `d=−g`를 만든다. 새 full-objective HVP `Hd=Jcc(c,p)d`는 current point에서 한 번 계산한다. `gᵀd`와 `gᵀHd`를 scale-aware roundoff guard로 해상하고 둘 다 음수일 때만 line search를 진행한다. 기존 `Hd`는 어떤 점에서도 재사용하지 않는다.

```text
alpha0 = min(1, 0.05/||d||₂, -gᵀHd/||Hd||₂²)
```

로 초기 candidate를 제한하고, 최대 16 dyadic backtrack 동안 straight full-space controls를 검사한다. Candidate accept에는 actual original-26 `J` Armijo와 fresh full-gradient `Phi` Armijo가 모두 필요하다. Φ는 (0.5||g_mathrm{candidate}||²)를 각 후보에서 다시 계산하며, roundoff보다 큰 actual decrease도 확인한다. Candidate마다 static exact-zero diagnostic와 full strict own branch/margin을 다시 검사한다. Branch signature가 바뀌면 기록하되 같은-branch 경로로 해석하지 않는다. `alphaPhi`나 quadratic `J` model 값만으로 후보를 채택하지 않는다.

Loop cap은 한 guarded launch, 최대 3 accepted steps/HVPs, PCG 0회, iteration당 최대 16 candidates, 내부 240초·외부 300초·sampled RSS 1 GiB다. 각 accepted candidate는 fresh endpoint recheck와 source/input/runtime/deadline closure 뒤 commit한다. later iteration refusal이면 앞선 commit과 HVP/iteration 수를 보존한다. Root threshold는 기존 `1e−10` gate 그대로이며 root indication은 `root_pending_audit`이지 root certificate가 아니다. 배포·예보·response step으로 승격하지 않는다.

## Provenance adapter 권고

Resume plan은 이전 model-guided **최종 raw/parent/resource**를 exact pin하고 raw accepted/current control SHA `69c4a794…`, parameters SHA `8871db49…`, tangent-plan ancestor와 inherited operator/archive pins를 닫는다. Raw가 `max_iterations_completed`, 세 iteration 모두 committed, HVP 시작/완료 3/3, PCG 0, source/input/runtime closed인지 확인한다. 마지막 iteration의 `accepted_control`, raw `current_control`, summary final control가 동일한 26-vector인지 검증한다.

실행 adapter는 fixed case를 재구성해 `input_after.archived_input`, parameters/truth identity와 runtime을 직전 raw에 대조한다. 최적화 control SHA만 새 starting point `69c4`로 명시적으로 바뀌며, 원래 archived observations, boundaries, prior, objective와 physics-source pins는 같아야 한다. 현재 `J`, gradient, branch/margins를 다시 평가해 final raw의 J/Phi/full-gradient/branch와 roundoff-scale로 일치시킨 다음 그 fresh gradient로 첫 `d=−g`를 만든다. 이전 tangent direction, e29 gradient/HVP, e29/tangent-step alpha, old branch signature는 start operator, direction 또는 alpha로 승계하지 않는다.

동일 kernel을 수정하는 방식이라면 편의를 위한 중복 optimizer wrapper를 만들지 않는다. 편집 직전 기존 module/test bytes를 lossless archive로 보존하고, source manifest에서 **self와 its focused test 두 파일만** 이전 snapshot→새 digest 버전 전환을 허용한다. `src/advar` 원래 J/transport/problem/HVP와 upstream source hashes는 기존 111/43 pin에 맞게 고정한다. 새 plan은 이전 raw/parent/resource/archive, 전·후 self/test snapshots, new source manifest, policy와 output 경로를 명시적으로 pin한다. Mismatch 또는 HVP slope refusal이면 기존 69c4 committed point를 보존하고 실행 receipt로 닫는다.

## 판단과 한계

저장 archive는 PR259를 시작할 적절한 nonstationary full-space state를 보여준다. 하지만 새 HVP의 `gᵀHd` 부호와 `alphaPhi`는 69c4에서만 정해진다. 실제 candidate acceptance, branch event 변화, 세 step 이후 gradient 개선은 새 guarded run 이전에는 알 수 없다. 목적함수·방향·line-search 정책이 PR257의 tangent chart 정책에서 바뀌었으므로 기존 tangent step과 속도나 alpha별 인과 비교를 하지 않는다. 이후 실행이 three-step cap으로 끝나도 full root, Hessian curvature certificate, adjoint/VJP/reanalysis, independent synthetic future skill은 미완료다.
