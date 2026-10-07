# PR #258 model-guided full-space continuation — GREEN preflight

날짜: 2026-10-07
판정: **GO — 고정된 bounded-loop 계획으로 실행 가능.** 각 step의 `gᵀHd` 부호와 actual candidate dual Armijo는 새 실행이 별도로 판정한다.
범위: frozen plan, model-guided module/tests, PR258 accepted endpoint raw/parent/resource와 보관 계산을 읽기 전용으로 검토. 이 검토자는 FV, objective/gradient/HVP, PCG, forecast, adjoint, reanalysis를 실행하지 않았다.

## 계획·출처

계획 `MODEL_GUIDED_FULL_SPACE_PLAN_20261007.json`의 SHA-256은 `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`이다. 계획에 선언된 111 source pin과 43 archive pin을 독립 SHA-256으로 대조했으며 불일치는 없었다. Parent plan은 QY32 tangent-step plan SHA `70f9560c…`; 기준 제어는 committed accepted tangent point `058895848fdeb348e8d6bb1f22ac13e3f2dadfa48bff46522ffbf94fedf72ffa`와 파라미터 SHA `8871db49…`다. Adapter는 step raw·parent·resource를 pin하고 accepted status, child hash, accepted trial의 dual Armijo, candidate commit, source/input/runtime closure를 검증한다. 실행 때 고정 point-problem·input identity·runtime·accepted branch를 다시 구성해 receipt와 비교하고, 그 endpoint의 full-control gradient를 재평가한다.

실행 작성자 기록은 새 model-guided/tangent 결합 시험 18개 통과(기존 경고 18개), basedpyright error-level 오류 0, isolated incremental AST 추출 완료다. 나는 시험과 타입 검사를 다시 실행하지 않았다. 이 기록은 실제 연구 step의 수치 결과가 아니다.

## 방향, 모델 alpha, merits

매 accepted current point의 방향은 `d=−g`, 여기서 `g`는 원래 full-26 objective의 fresh gradient다. 코드는 한 번의 fresh matrix-free HVP `Hd=H(c)d`를 구하고 roundoff-resolved `gᵀd<0`, `gᵀHd<0`를 요구한다. 첫 부호는 `gᵀd=−||g||₂²`; 둘째는 smooth current branch에서

```text
D Phi(c)[d] = gᵀ H d,   Phi(c)=0.5 ||g(c)||₂².
```

와 맞다. 수송 profile의 실제 branch oracle은 `fv_minmod_inverse_probe.inspect_branches`를 통해 complete trajectory를 확인하며, 각 stage에서 interior 좌/우 minmod slope가 `128 eps q_scale` 안에 있거나 active limiter gap이 미해상인 경우, 또는 face flux가 `128 eps global_flux_scale` 안에 있는 경우를 strict branch failure로 거부한다 (`examples/weather_scenarios/fv_minmod_inverse_probe.py:137–200`). 따라서 current HVP의 국소 branch 미분을 승인하는 base gate가 있으며, static exact-zero face precheck도 추가돼 있다. 후보 endpoint는 매번 실제 원래 문제에서 자기 branch와 margins를 재평가한다.

선형화 gradient `g+alpha Hd`의 제곱노름 모델을 최소화하는 양의 step은

```text
alphaPhi = -gᵀHd / ||Hd||₂²
```

다. 구현은 overflow를 피하는 normalized dot으로 계산하고 `alpha0=min(1, radius/||d||₂, alphaPhi)`를 사용한다. 이는 model-based initial cap이며 actual `Phi` minimum 보장은 아니다. 원래 full-control 후보에서는 `c(alpha)=c+alpha d`라 `gamma''=0`; 따라서 선택적 local quadratic `J` model의 curvature는 `dᵀHd`다. 이전 curved `eta` chart의 `gᵀgamma''` correction은 새 full-space path에 적용하지 않는다.

각 후보에서 원래 full-26 `J`, fresh full gradient로 계산한 actual `Phi`, strict own branch/margins를 검사한다. Accept에는 actual dual Armijo가 모두 필요하며 두 상수는 `c1_J=c1_Phi=1e−4`다. 추가로 actual `Phi` 감소가 roundoff budget보다 커야 한다. 최근 code fix는 후보 하나의 actual `Phi` 감소가 unresolved이거나 증가한 경우 그 후보를 거부하고 다음 작은 dyadic alpha로 진행한다. Grid 전체가 실패하면 refusal로 끝낸다. 시험 fixture는 큰 step의 `Phi` 증가 뒤 절반 step이 통과하는 경우를 구별한다.

## iteration·commit·resource 처리

한 guarded launch 안에서 최대 3 accepted steps/3 fresh current-HVPs, PCG 0회, 최대 16 candidates per iteration, 내부 240초·외부 300초·sampled RSS 1 GiB를 사용한다. 다음 iteration은 이전 accepted candidate의 **fresh final recheck gradient/state**에서 새 `d=−g`를 만든다. 이전 tangent 방향이나 cached `Hd`는 다음 점에 재사용하지 않는다. Stale/static zero-face refusal, unresolved descent, no accepted grid candidate, budget stop은 별개 상태로 남긴다.

Iteration commit은 fresh endpoint 재검사와 source/input/runtime/deadline closure 뒤에만 이뤄진다. 중간 iteration에서 거부가 생기면 앞서 commit한 제어·gradient와 iteration/HVP 수를 보존하고 tentative candidate는 commit하지 않는다. Parent receipt는 accepted iteration 수와 committed current control hash를 닫는다. Gradient가 기존 `ROOT_GINF=1e−10`에 들어와도 상태는 `root_pending_audit`로만 기록하며 full curvature/root/response 인증을 주장하지 않는다.

## 재사용할 PR258 진단과 해석 경계

PR258 tangent point의 archive-only 값은 다음 step 선택의 배경일 뿐 새 current HVP 대용은 아니다. tangent accepted point에서 `||g||₂=.66864064`, `||g||∞=.51012850`; tangent norm `.65128405`; unit normal gradient `−.15135847`; `||g_T||/|g_N|≈4.3029`였다. 이전 tangent 방향으로 `g_newᵀd_old=+0.0091206892`였으므로 그대로 이어 쓰면 안 됐고, full-space fresh negative gradient로 복귀하는 선택은 수치상 타당하다. 새 plan의 alphaPhi, 새점 `gᵀHd`, curvature는 실행 전에는 알 수 없다.

이 bounded loop는 원래 목적함수에서 최대 세 번의 local correction이다. Branch changes는 허용된 actual endpoint 정책 안에서 기록하되, branch hashes만으로 전체 경로 smoothness를 주장하지 않는다. 예보 score, full unconstrained stationary root, current full curvature certificate, original adjoint/VJP/reanalysis, independent synthetic future validation은 남아 있다. 실행 전 GO는 candidate acceptance나 연구 완료 보증이 아니다.
