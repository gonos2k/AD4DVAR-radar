# PR #258 full-gradient continuation — GREEN 수치 설계 검토

날짜: 2026-10-07
범위: PR258 tangent-step raw `061c51d2…`, accepted control `05889584…`, QY32 post-audit summary `695e1a17…`의 읽기 전용 재산술과 다음 bounded-loop 제안 검토. 새 FV, objective, gradient, HVP, PCG 또는 후보 실행은 수행하지 않았다.

## 저장 산술 검산

PR258 accepted point에서 전체 26-gradient는 `||g||₂=0.6686406395`, `||g||∞=0.5101284967`이다. 이 점의 접선 projection norm은 `0.6512840542`, 단위 유클리드 법선 gradient는 `−0.1513584674`이며 `||g_T||/|g_N|=4.30292448`이다. 아직 큰 접선 잔차와 법선 성분이 함께 남았다. 원래 tangent step에서 단위 법선 기울기 sign은 `+0.00470995→−0.15135847`로 바뀌었지만, 그 이동은 `Qy[3,2]`의 비영값 `−5.9106127381e−7`을 고정한 등면 경로였다. 이 sign 변화는 face 위 stationarity나 정상점을 뜻하지 않는다. 따라서 PR258의 전체 제어 방향 `d=−g`는 임시 eta 제약을 푸는 합리적인 다음 탐색 방향이다.

보관 tangent HVP는 이전 e29 점과 다른 tangent 방향에 대한 값이다. 그 자료로 이전 경로 수치는 다음처럼 재현된다.

| 이전 e29 tangent path 진단 | 재산술 |
|---|---:|
| `alphaPhi=−gᵀHd/||Hd||²` | `0.00028953045674` |
| `||Hd||₂` | `1937.72574798` |
| curved chart `c24''` | `8.28349418e−6` |
| `gᵀgamma''` | `+1.23377967e−7` |
| `dᵀHd` | `1087.03546193` |
| `J''(0)=dᵀHd+gᵀgamma''` | `1087.03546205` |
| curved-path quadratic `alphaJ=−gᵀd/J''(0)` | `0.00051401286187` |

이전 chart는 `c24=atanh(z(t,eta))`여서 pivot만 `gamma''`가 0이 아니며 `J''(0)=dᵀHd+gᵀgamma''(0)`을 쓴다. 저장 chart 식의 analytic `c24''`는 중앙 유한차분과 FP64 정밀도에서 일치한다. 그 경로의 곡률 보정 `gᵀgamma''`는 이 자료에서 매우 작다. 이 수치는 새 accepted point의 `alphaPhi`나 `alphaJ`가 아니다. PR258은 원래 26 raw-control 공간에서 직선 `c(alpha)=c+alpha d`를 쓰므로 그 path의 `gamma''=0`; 이전 chart의 pivot 보정을 가져오지 않는다.

추가 예측량도 구분한다. 이전 tangent move에서 실제 old/new gradient 각도 `91.091746°`는 **선형 예측과 actual gradient 사이의 각도**가 아니다. latter는 약 `0.051874°`이고, `g_old+alpha Hd_old`의 actual-gradient 상대 잔차는 `9.27435e−4`다. `Phi(g_old+alpha Hd_old)=0.22362989`는 actual `0.22354015`에 가까웠지만, scalar 1차 식 `Phi0+alpha*gᵀHd=−0.28871735`는 merit 값의 예측으로 쓸 수 없었다. 실제 accept에는 candidate full gradient에서 새로 구한 `Phi`만 사용했다. 새 점에서는 `g_newᵀd_old=+0.0091206892`이므로 구 방향을 재사용하지 않는다.

## 다음 방향의 수식과 실제 조건

각 committed current point에서 원래 full-26 objective를 fresh 평가하여 `g=∇J(c)`를 구하고 `d=−g`로 둔다. 원래 full-26 data/prior/smoothness objective·파라미터·관측·경계·시간 계약을 보존한다. 현재점의 fresh HVP `Hd=H(c)d`는 한 번만 계산한다. `gᵀd=−||g||²`가 유한하고 roundoff budget을 넘어 음수인지, `gᵀHd`가 역시 scale-aware budget을 넘어 음수인지 검사한다. 후자는 smooth current branch에서 `D Phi[d]=gᵀHd`를 주므로 actual `Phi` 감소 방향을 통과시키는 조건이다. 새 점의 old HVP나 old tangent average는 대입하지 않는다.

선형화한 gradient 경로에서

```text
g(alpha) ≈ g + alpha H d
Phi_linear_gradient(alpha) = 0.5 ||g + alpha H d||²
alphaPhi = -gᵀ H d / ||H d||²
```

이고, `alphaPhi`는 이 제곱노름 선형화 모델의 최소점이다. 실제 Φ 최소점 보장은 아니며, denominator가 양수·유한하고 HVP descent가 해상되어야 유효하다. 후보 acceptance는 각 점의 fresh full gradient로 계산한 actual Φ에 맡긴다.

현재 제어 경로는 raw control에서 affine이므로 γ″=0이다. J의 선택적 국소 quadratic 진단을 보고한다면 αJ=`−gᵀd/(dᵀHd)`이며 `dᵀHd=−gᵀHd>0`일 때만 양의 추정이 된다. 이 값을 추가 line-search gate로 쓸 필요는 없다. 제안된 초기 alpha는

```text
alpha0 = min(1, 0.05 / ||d||₂, alphaPhi)
```

로 둔다. 각 후보는 최대 16개 dyadic backtrack 안에서 실제 원래 `J` 및 fresh-gradient `Phi`의 Armijo 조건 (`c1=1e−4` 각각)을 모두 통과하고 자체 strict branch/margins를 통과해야 한다. `alphaPhi`는 초기 크기 cap이고 actual merit 검사를 대체하지 않는다. step들이 다른 limiter/upwind 분기를 바꾸면 그 endpoint를 그대로 기록하되 경로의 smoothness나 branch-preserving derivative를 주장하지 않는다.

## bounded-loop·receipt 계약

한 guarded launch 안에서 최대 **3 accepted steps / 3 fresh HVPs**, PCG 0회, 최대 16 후보/step, 내부 240초·외부 300초·RSS sample 1 GiB를 쓴다. 매 반복마다 baseline objective, full gradient, branch/margins를 먼저 재평가하고 이후에 그 점의 HVP를 계산한다. 각 후보는 실제 `J`/`Phi`와 own endpoint branch를 검사한다. 통과 후보도 독립 recheck 및 source/input/runtime/deadline closure 뒤에만 commit한다. 후속 HVP direction-refusal, line-search grid 고갈, timeout이 나면 기존 committed point와 회계 수를 보존하고 새 후보를 commit하지 않는다. 새 root/stationarity threshold, normal constraint, additional experiment quota를 도입하지 않는다.

## source/provenance adapter 게이트

시작점은 tangent-step raw/parent/resource와 계획 SHA에 pin된 committed `058895848fdeb348e8d6bb1f22ac13e3f2dadfa48bff46522ffbf94fedf72ffa`다. Adapter는 정상적으로 완료된 `candidate_committed=true`와 actual dual-Armijo·independent recheck·source/input/runtime closure를 요구하고, plan의 parent/child/raw hash를 닫아야 한다. 같은 원래 문제를 재구성한 뒤 accepted full-control 26-vector, 13-parameter, archived-input identity, runtime, accepted strict-branch digest를 대조한다.

방향을 만들기 전에 현재점의 fresh full gradient를 계산하여 raw accepted gradient와 roundoff-scale 기준으로 일치하는지 검사하고, 그 fresh `g`만으로 `d=−g`를 구성한다. 이전 tangent `H_direction`, saved alpha, QY32 two-sided covectors는 새 점의 HVP·방향·step cap으로 재사용하지 않는다. Fresh current HVP를 기록하고 `alphaPhi`·`alphaJ` 입력과 resolved descent 조건을 함께 보존한다. Input/source/runtime/pin mismatch면 objective/HVP를 더 진행하지 않고 refusal receipt를 닫는다.

## 판단과 미완료 범위

archive 산술은 tangent move가 full gradient를 충분히 줄이지 못했고 정상 법선 성분도 남겼음을 보여준다. 따라서 원래 full-gradient 방향으로 돌아가는 한 제한 탐색이 수치적으로 일관된다. 다만 다음점 HVP 부호와 positive curvature, `alphaPhi`, candidate 실제 Armijo는 새로운 계산 전에는 알 수 없다. Negative/ambiguous `gᵀHd`면 direction refusal이 올바른 결과다.

성공적인 bounded-loop도 원래 3h root, 최종 curvature, adjoint/VJP/reanalysis, independent synthetic future score를 완료하지 않는다. Full-gradient step을 채택해도 maximum gradient 증가·분기 변경·비선형 merit 거부가 있으면 이를 그대로 보고하고 연구 단계 완료로 확대하지 않는다. 여기의 archive-only 수치는 공식 proposal의 형식 검산이며 새로운 FV/HVP나 적용 허가는 아니다.
