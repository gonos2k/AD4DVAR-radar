# PR #257 공통 접선 방향 — GREEN 읽기 전용 검토

날짜: 2026-10-07
판정: **저장된 기하 산술은 일관되고, 제한된 접선 시험은 조건부 GO.** 실제 후보 수용은 현재점 fresh HVP와 실제 `J`/`Phi` Armijo 평가를 거친 뒤에만 판단한다.
범위: QY32 raw/요약/고정 3h 제어에서 chart 방향·정적 면 유량 Jacobian을 독립 NumPy 산술로 확인했다. FV, 목적함수, gradient, HVP, PCG, 예측은 다시 실행하지 않았다.

## 출처와 방향 정의

사용한 전체 raw는 `qy32_diagnostic_attempt1/diagnostic.json` (SHA `d05e5b8e…`)이며, `QY32_RESULT_20261007.json` (SHA `348dd001…`)의 `raw_sha256`와 같다. 압축 보관본은 압축 해제 시 동일 raw SHA를 보존한다. 원래 기준점은 F462 continuation의 e29 제어(`e29c348d…`)와 그 `current_state.gradient`다. QY32의 네 gradient는 저장된 전체 26제어 gradient이고, 어느 쪽에서도 새 gradient를 계산하지 않았다.

각 비영 표본에서 `Z_eta = D_t Gamma(t, eta)`를 계산하고

```text
q_t(eta) = Z_etaᵀ g(eta)
G0       = Z_0ᵀ Z_0
dt       = -G0⁻¹ [q_t(-1e-6) + q_t(+1e-6)] / 2
```

로 공통 retained-coordinate 방향을 구성했다. 이것은 두 안쪽 표본의 chart 접선 기울기를 `eta=0` chart의 유클리드 metric으로 평균한 방향이다. `cond(G0)=1.29798`이라 저장된 25차원 계산은 작은 조건수에서 유한하다. 네 표본 각각의 `q_t(eta_i)ᵀdt`는 다음과 같이 모두 음수다.

| `eta_i` | `q_t(eta_i)ᵀdt` |
|---:|---:|
| `−2e−6` | `−0.5579511898` |
| `−1e−6` | `−0.5585182949` |
| `+1e−6` | `−0.5597352579` |
| `+2e−6` | `−0.5603851153` |

이는 기록된 네 gradient의 chart 접선 방향미분이 감소방향이라는 확인이다. 이 부호만으로 원래 비선형 `J` 감소나 `Phi` 감소를 보증하지 않는다.

## e29 lift와 곡선/직선 구분

e29의 현재 면 좌표는 생산 면식에 따른 `eta_b=Qstar(c_e29)=−5.91061273804e−7`이다. `dt`를 현재 chart로 lift한 방향은

```text
d = Z_eta_b dt
||dt||₂ = 0.7475056294
||d||₂  = 0.7477478027
g_e29ᵀ d = −0.5587502088
n_e29ᵀ d = 0                 (FP64 산술)
```

이다. 여기서 `n=∇_c Qstar`. 현재 cached gradient의 `gᵀd`는 원래 `J`의 국소 chart 접선 하강을 지지한다. `d`는 선형 접선 벡터이고, finite candidate 경로는 `c(alpha)=Gamma(t_e29+alpha dt, eta_b)`여야 한다. 직선 `c_e29+alpha d`는 2차부터 곡선 chart와 다르며 `Qstar`를 보존하지 않는다.

저장된 chart만으로 정적 bisection을 하면 곡선 경로 displacement norm이 `0.05`가 되는 상한은 `alpha0=0.0668674656179`이다. 이때 `||c(alpha0)-c_e29||₂=0.05`, 최대 성분 변화는 유한하다. 선형 근사 `0.05/||d||₂`와의 차이는 약 `6.5e−10`이다. 이 정적 계산은 비용 평가 없이 수행했다. 방향 생성·lift와 경로를 서로 바꾸어 해석하지 않아야 한다.

## chain decomposition 검산

고정-`t` transverse chart 벡터 `v=D_eta Gamma`는 pivot `c[24]`만 바꾸며 `nᵀv=1`이다. 이를 유클리드 법선·접선으로

```text
v = n/(nᵀn) + v_T,       nᵀv_T = 0
g = g_T + n(nᵀg)/(nᵀn)
gᵀv = nᵀg/(nᵀn) + g_Tᵀv_T
```

분해하면 QY32 네 표본 및 e29에서 저장된 `j_eta_fixed_t`와 재결합 값이 표시 정밀도까지 같다. e29 값은

```text
intrinsic normal flux slope  nᵀg/(nᵀn) = +0.009186929095
tangent contribution          g_Tᵀv_T       = −0.04228567658
chart derivative               gᵀv           = −0.03309874749
```

이다. 따라서 pivot chart derivative를 pure normal derivative로 부를 수 없다. QY32에서 보았듯 `j_eta_fixed_t`의 부호만으로 법선 방향 최적성을 판단하지 않는다.

common direction의 full chain도 대조했다. `g_e29ᵀ(Z_eta_b dt)=(Z_eta_bᵀg_e29)ᵀdt`; 원래 control 좌표로 나눈 값은 field `−0.4933936942`, flow `−0.0023942732`, growth `−0.0629622414`로 합계 `−0.5587502088`이다. 이는 저장된 현재 gradient와 `dt`의 곱이며 새 함수 평가가 아니다.

## 정적 다중 면 pivot 미분의 20/20 산술

고정 grid는 정수 vertex 좌표 `i=0,…,4`, `j=0,…,5`를 쓰며 마지막 streamfunction basis는 `psi4(i,j)=i j²`다. 이 basis의 real coefficient를 `a4`라 하면

```text
Qx[i,j] = Δ_i psi = j² a4 + terms independent of a4,  i=0..3, j=0..5
Qy[i,j] = -Δ_j psi = -i(2j+1) a4 + terms independent of a4, i=0..4, j=0..4
Qstar   = C(other coefficients) - 15 a4,   at Qy[3,2]
da4/deta = -1/15,   eta=Qstar with other coefficients fixed
```

이로부터

```text
dQx[i,j]/deta = -j²/15
dQy[i,j]/deta =  i(2j+1)/15
```

를 얻는다. `Qstar=C−15a4`의 `C`는 다른 네 기저 계수 중 `Qy`에 기여하는 항을 포함하며, 일반적으로 0이 아니다. 그러므로 `a4=−eta/15`라는 절대 계수 등식으로 쓰면 안 되지만 도함수 `da4/deta=−1/15`는 맞다.

| 정적 배열 | 모양 | 0이 아닌 `dQ/deta` 수 | 최대 절대 미분 |
|---|---:|---:|---:|
| `Qx` | 4×6 | 20/24 (`j=1,…,5`; `j=0` 네 항은 0) | `25/15=5/3` |
| `Qy` | 5×5 | 20/25 (`i=1,…,4`; `i=0` 다섯 항은 0) | `36/15=2.4` |

관심 face `Qy[3,2]`의 미분은 `3(2·2+1)/15=1`이다. 이 검산은 선택된 고정 profile의 정적 basis 연산이다. 20개 면이 모두 독립 제약이라는 rank 주장은 아니며, 시간 적분이나 미래 상태의 면 유량 민감도도 아니다.

## 최소 사전 수용 정책 권고

현재점 HVP가 아직 없으므로 다음은 적용 조건을 사전에 고정할 연구 제안이다.

1. e29에서 원래 full Hessian operator로 **한 번만** `H_e29 d`를 계산한다. scale-aware roundoff margin을 적용해 `g_e29ᵀH_e29d<0`인지 확인한다. 음의 값이 분해되지 않으면 이 direction을 `Phi` 하강 방향으로 승인하지 않고 종료한다. HVP 방향값은 양수·0일 수 있으며 QY32의 chart `Phi` 자료에서 추론하지 않는다.
2. 조건을 통과하면 `c(alpha)=Gamma(t_e29+alpha dt, eta_b)`를 쓰고, 정적 `alpha0=0.0668674656179`, 이후 `alpha0·2^-m`, `m=0,…,15`를 사전 고정한다. 매 점에서 실제 26제어 displacement가 `≤0.05`인지 확인한다. `eta_b`를 유지하므로 이 경로는 경계 횡단이 아니라 e29의 음의 유량 등면에서의 접선 보정이다.
3. 각 후보에서 원래 full-26 `J`와 새 full gradient로 실제 `Phi=0.5||g_candidate||₂²`를 평가하고, 원래 관측/prior/경계/시간과 strict branch/margin 게이트를 유지한다. 기존 상수 `c1_J=c1_Phi=1e−4`를 사용하여

   ```text
   J(candidate)   <= J(e29)   + c1_J   alpha (g_e29ᵀd)
   Phi(candidate) <= Phi(e29) + c1_Phi alpha (g_e29ᵀH_e29 d)
   ```

   두 실제 조건을 함께 통과한 첫 후보만 채택한다. 후보에서 `Phi`를 실제 gradient로 재계산한다. `q_tᵀdt<0` 또는 chart chain 식만으로 `Phi` 감소를 대신하지 않는다. branch signature가 바뀌면 서명과 변화 위치를 보존하고 base-branch HVP 해석을 그 endpoint의 매끄러운 미분 증명으로 확대하지 않는다.
4. HVP budget은 1, PCG solve는 0, optimizer/root/score/adjoint/reanalysis 주장은 0으로 기록한다. 미수용·HVP slope refusal도 정상 종료 결과로 남긴다.

이 정책은 현재 방향의 한 번의 국소 접선 보정을 판정한다. 영유량면을 넘어가는 탐색, 비매끄러운 경계의 일방 최적성, `Phi`의 branch-crossing 하강, 원래 3h 문제 정상점을 증명하지 않는다. 실제 비선형 후보가 실행되기 전까지는 여기서의 방향·면 민감도는 보관 배열에 대한 읽기 전용 재산술이다.
