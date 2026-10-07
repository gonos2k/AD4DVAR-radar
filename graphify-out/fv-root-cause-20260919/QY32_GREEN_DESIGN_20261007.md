# Qy[3,2] 양측 면 진단 — GREEN 설계 검토

날짜: 2026-10-07
범위: PR #256의 마지막 확정점 `e29c348d…` 및 현재 제어·면유량 정의를 읽기 전용으로 확인하고, `Qy[3,2]=0` 주변의 제한된 양측 진단을 설계했다. 새 FV, gradient, HVP, PCG, 예측, 수반 또는 재분석은 실행하지 않았다. 코드와 공유 graph는 수정하지 않았다.

## 판정

제안된 실험은 원래 문제에서 영 유량 경계 통과가 원래 비용 `J`와 정상성 merit `Phi`에 미치는 영향을 분리하는 적절한 다음 진단이다. `Qy[3,2]`를 직접 덮어쓰지 않고, 같은 나머지 25개 제어 좌표를 유지한 채 pivot 제어 하나를 역산한다. 각 비영점 양측 표본에서 원래 26개 제어 전체와 기존 26개 prior를 포함해 `J`, 전체 gradient, branch 서명을 기록한다. `eta=0`은 primal 비용·상태·branch 관찰로만 취급하고 gradient나 영유량의 strict branch 인증을 주장하지 않는다.

이 실험은 경계의 양쪽에서 관찰한 유한 표본 비교다. 정확한 일방 극한, 미분의 존재 여부, 다른 branch 변화가 없다는 사실을 증명하지 않는다. 여러 유량면·limiter 선택이 동시에 바뀌면 `Qy[3,2]` 단독 원인으로 결론 내리지 않는다.

## 저장된 점과 기하 확인

입력은 `f462_inexact_continuation_20261007_attempt1/step.json`의 `accepted_control`이며 해시는 `e29c348d51e7ec2de23f3f74d1522a34f58bf0bb72fd56e63d9b89cadea15e37`이다. 이 점의 전체 제어 차원은 26이며, 최초 20개는 초기장, 인덱스 20–24는 유동 latent 제어, 인덱스 25는 성장 제어다. 비교에 사용할 고정 파라미터와 관측·경계·prior 등 입력은 해당 continuation의 기존 pin을 그대로 사용한다.

검토 대상 면의 제어식은

```text
Qstar(c) = -0.08 tanh(c[21]) - 0.21 tanh(c[22])
           - 0.10 tanh(c[23]) - 0.45 tanh(c[24])
```

이다. 저장된 최종점의 `c[21:25]`에 대입한 값은 `-5.9106127380e-7`로 보관된 `Qy[3,2]`와 일치한다. 면 연산의 부호 규약은 `src/advar/transport.py`의 `face_volume_fluxes`에서 `Qy = -(psi[:, 1:] - psi[:, :-1])`로 정의되고, 제어에서 계수와 면유량을 구성하는 경로는 `examples/weather_scenarios/fv_point_3h_current_newton_step.py`의 `physical_state`와 맞는다.

비영점 좌표는

```text
t = (c[0:24], c[25])   # pivot c[24]를 제외한 나머지 25개 좌표
eta = Qstar(c)
c[24] = atanh((-eta - 0.08 tanh(c[21]) - 0.21 tanh(c[22])
                - 0.10 tanh(c[23])) / 0.45)
```

로 잡는다. 각 `eta`에서 `t`는 e29 값으로 동일하게 둔다. 저장점의 고정된 `c[21:24]`로 역산하면 다음과 같다.

| `eta` | 역산한 `c[24]` | `atanh` 입력 |
|---:|---:|---:|
| `-2e-6` | `-2.87176079914805e-4` | `-2.87176072020325e-4` |
| `-1e-6` | `-2.89398302321716e-4` | `-2.89398294242548e-4` |
| `0` | `-2.91620524731486e-4` | `-2.91620516464770e-4` |
| `+1e-6` | `-2.93842747144136e-4` | `-2.93842738686993e-4` |
| `+2e-6` | `-2.96064969559687e-4` | `-2.96064960909215e-4` |

모든 입력은 `atanh`의 열린 정의역 안에 충분히 있다. 이 표는 정적 제어 기하만 확인한 결과이며, 표본의 상태 적격성·J·gradient·branch에 대한 FV 결과가 아니다. 설정에서 면유량 단위는 모델 좌표의 2-D 면적/시간 단위다. 이 면을 실제 바람의 장벽이나 무강수 경계로 해석하지 않는다.

## 권장 표본과 기록

필수 다섯 점은 `eta = -2e-6, -1e-6, 0, +1e-6, +2e-6`이다. 각 점에서 저장점의 나머지 25개 좌표, 파라미터, 관측, prior, 시간·경계 및 목적함수 정의를 유지한다. pivot 이외의 좌표가 변하지 않았는지 해시와 배열 비교로 기록한다. 모든 점에서 제어 범위, 유한성, 물리·CFL 입력 계약을 확인한다.

- 네 비영점 표본: 원래 전체 `J`, 전체 26차원 gradient, `Phi=0.5 ||g||^2`, 전체 분기 서명과 유량면 부호·margin, 활성 limiter 선택, 제어 해시를 기록한다. 각 점의 branch oracle이 거부되면 비용/gradient 결과와 구분해 `branch unavailable/refused`로 남긴다.
- `eta=0`: 전체 원래 제어에 대한 primal `J`, 유량 및 분기 관찰만 수행한다. AD gradient, `Phi`, strict nonzero-face 적격성 또는 exact one-sided derivative를 보고하지 않는다. 기존 primal 경로가 정확 영유량 분기에서 거부되면 거부 사실을 남기고 임의의 flux 덮어쓰기나 epsilon 이동으로 대체하지 않는다.
- 진단 목적은 동일 `t`에서 양쪽의 `J`가 어떤 방향으로 변하는지, 전체 gradient와 접선/법선 성분이 어떻게 달라지는지, branch 서명이 어떤 요소 때문에 달라지는지 비교하는 것이다. `J` 감소와 `Phi` 변화의 방향이 서로 다르면 경계 주변의 dual-merit 충돌을 지지하는 근거로 기록할 수 있다. 이는 전역 탐색 정책이 잘못됐다는 증명은 아니다.

## 전체 gradient의 접선·법선 분해

비영점 각 점에서 원래 좌표의 전체 gradient `g=∇c J`를 보존한다. 좌표 사상 `c=Gamma(t, eta)`는 `c[24]`만 위 식으로 복원하고 나머지 25개는 고정한다. 같은 표본의 좌표 미분은 체인 규칙으로

```text
j_t   = (D_t Gamma)^T g       # eta 고정 시 25개 접선좌표의 J 기울기
j_eta = (D_eta Gamma)^T g     # t 고정 시 면 법선 좌표의 J 기울기
```

로 계산한다. 이 chart에서는 `j_eta_fixed_t = g[24] / (d Qstar / d c[24])`, `d Qstar / d c[24] = -0.45 sech²(c[24])`이다. 구현 결과는 직접 Jacobian 곱과 이 독립 scalar 식으로 대조한다. 이 `j_eta_fixed_t`는 pivot을 통해 `eta`를 바꿀 때의 chart 방향 미분이며, 아래의 두 기하학적 법선 진단과 같다고 가정하면 안 된다.

기하학적 접선 잔차는 `n=∇c Qstar`, `T=ker(n^T)`에서 평가한다. `D_t Gamma`의 열 공간에 QR을 적용해 정규직교 기저 `Q_T`를 만들고 `||Q_T^T g||₂`를 보고한다. 동시에 `g - n (n^T g)/(n^T n)`로 얻은 유클리드 접선 투영과 일치하는지 수치적으로 확인한다. QR 기반 값은 접선 좌표의 재척도·가역 재매개화에 불변인 25차원 기하량이다. `||j_t||₂` 자체를 불변 접선 norm으로 부르지 않는다.

법선 관련 세 양은 별도 이름으로 보고한다.

```text
j_eta_fixed_t                  = g^T (D_eta Gamma) = g[24] / Qstar_c[24]
normal_gradient_unit           = n^T g / ||n||₂
normal_flux_slope_intrinsic    = n^T g / (n^T n)
```

`normal_gradient_unit`은 제어공간에서 단위 유클리드 법선 방향으로의 기울기다. `normal_flux_slope_intrinsic`은 최소 노름의 법선 이동으로 `Qstar`를 한 단위 바꿀 때의 기울기이며, `normal_gradient_unit / ||n||₂`다. `j_eta_fixed_t`는 pivot만 움직이는 좌표선의 기울기라서 두 값 중 어느 것과도 일반적으로 같지 않다. `g_N = n (n^T g)/(n^T n)`와 그 크기도 기록한다. 현재 e29 점에서 앞선 재산술로 확인한 `||g||₂≈0.747259267`, `||g_T||₂≈0.747244423`은 현재점의 국소 투영에 한정된다. 새 양측 gradient가 없으므로 경계에서 접선 정상성이나 법선 최소 조건이 성립한다고 미리 가정하지 않는다.

## 구현 결과 필드와 구별 시험

후속 probe의 점별 JSON은 최소한 다음 명시 필드를 둔다. `eta_target`, `eta_reconstructed`, `control_sha256`, `J_full`, `gradient_full`(비영점 표본만), `Phi_full`(비영점 표본만), `j_t_fixed_eta`, `j_eta_fixed_t`, `tangent_gradient_norm_qr`, `tangent_gradient_norm_projection`, `normal_gradient_unit`, `normal_flux_slope_intrinsic`, `normal_gradient_projection_norm`, `face_branch_signature`, `limiter_branch_signature`, `strict_margin_summary`, `evaluation_status`, `refusal_reason`이다. zero 표본의 gradient 기반 필드는 생략/null로 남기고 값 0을 채워 넣지 않는다. 양측 구조 비교 요약에는 `plus_minus_J_delta`, 각 측의 `Phi_full`, `j_eta_fixed_t`, `normal_gradient_unit`, `normal_flux_slope_intrinsic`, tangent norm 및 바뀐 branch 항목을 둔다. 이는 설계 제안 필드이며 저장소의 기존 schema를 이미 바꿨다는 뜻이 아니다.

읽기 전용 시험/preflight는 아래 서로 다른 오류를 구분해 잡아야 한다.

1. 지정 다섯 `eta_target`에서 pivot 역산 뒤 `eta_reconstructed`와 target의 오차를 FP64 스케일에 맞춰 제한하고, 나머지 25 좌표의 배열 해시/동등성을 확인한다.
2. analytic `D_eta Gamma` 및 `D_t Gamma`가 `Qstar(Gamma(t, eta))=eta`의 미분 관계를 만족하는지 검사한다. `j_eta_fixed_t`는 `g @ D_eta Gamma`와 `g[24] / Qstar_c[24]`를 대조한다.
3. `normal_gradient_unit`과 `normal_flux_slope_intrinsic`이 각각 `n @ g / ||n||`와 `(n @ g)/(n @ n)`에 맞으며, `normal_flux_slope_intrinsic = normal_gradient_unit / ||n||`를 확인한다. 일반적으로 둘과 `j_eta_fixed_t`가 같지 않은 비대칭 예제를 사용해 이름 혼동 회귀를 잡는다.
4. QR 기저에서 계산한 `tangent_gradient_norm_qr`와 명시적 직교 투영 `tangent_gradient_norm_projection`이 허용오차 내에서 일치한다. invertible 열 재척도 후 QR tangent norm은 유지되며, raw `||j_t_fixed_eta||₂`는 보편적으로 불변이라고 가정하지 않는다.
5. `eta=0` 결과에 gradient, `Phi`, strict nonzero-face certification 필드가 생성되지 않으며, zero-flux 주변 면 배열을 덮어쓰지 않는다.
6. fixture branch signature에서 관심 유량면 이외의 face/limiter/stage 변경이 발생하면 요약에 이를 드러내고, `Qy[3,2]` 단독 원인 label을 붙이지 않는다.

## branch와 해석의 한계

모든 비영점에서 branch oracle의 전체 서명과 strict margin을 비교하고, 특히 `Qy[3,2]` 외의 유량면 부호, 모든 donor/minmod limiter 분기, 단계별 선택을 함께 기록한다. 양측에서 다른 분기가 하나라도 더 바뀌면 `J` 변화의 원인을 단일 면으로 귀속하지 않는다. 영점에서 oracle이 정의하지 않는 strict sign이나 margin은 인증하지 않는다.

네 비영점 점은 단지 지정된 `eta` 간격에서의 점별 계산이다. 표본 간 연속 경로의 분기 고정, 정확한 양쪽 극한, Clarke 정상성, 해당 면을 넘는 적격 목적함수의 미분성, global minimum 또는 실제 기상학적 장벽을 입증하지 않는다. 같은 `t`에서 `J`가 내려가고 `Phi`가 올라가는 양측 결과가 관찰돼도, 다른 접선 변화를 배제하거나 특정 solver 변경을 정당화하지 않는다. 이는 다음 방법 선택에 필요한 원인 진단 근거다.

## 실행 전 게이트

실제 계산을 시작하기 전에 읽기 전용 preflight/test에서 다음을 확인한다.

1. e29 raw, parent/resource 영수증, 계획, 전체 제어 및 고정 입력/source identity가 예상 SHA로 닫힌다.
2. 좌표 변환은 원래 26차원 입력을 만들고 `t` 외의 pivot만 바꾸며, 지정 각 표본의 `Qstar` 재구성 오차가 FP64 허용범위 안이다.
3. 다섯 점의 `atanh` 정의역·제어 유한성 및 원래 입력 계약을 만족한다. strict zero-flux 인증은 요구하지 않는다.
4. 전체 원래 목적함수 경로와 26개 prior 항이 유지된다. 손실 정의를 축약하거나 유량 배열을 직접 바꾸는 경로가 없다.
5. gradient/체인 미분은 네 비영점에서만 계산되고, `j_eta_fixed_t`, 단위 법선 기울기 및 flux 단위 intrinsic 법선 기울기를 별도 필드로 대조한다. QR 접선 norm과 투영식 결과도 허용범위에서 일치한다.
6. branch 기록은 전체 분기 요소를 포함하고, 양측 변화가 겹칠 때 단일 면 원인 주장을 자동으로 하지 않는다.
7. 자원·wall 제한과 결과/실패 상태 저장이 계획에 고정된다. 비용이 과도하거나 어느 한쪽 표본을 평가할 수 없을 경우 부분 결과와 거부 이유를 보존한다.

본 GREEN 설계 검토는 preflight 통과, 실제 FV 계산, 양측 원인 판정, 알고리즘 변경 또는 PR 수용을 뜻하지 않는다.

## 확인한 소스·기록

- `graphify-out/fv-root-cause-20260919/f462_inexact_continuation_20261007_attempt1/step.json`: 두 반복 및 마지막 확정 제어.
- `examples/weather_scenarios/fv_point_3h_current_newton_step.py`: 전체 제어의 물리 진단 및 생산 coefficient/face 연산 연결.
- `src/advar/transport.py`: `face_volume_fluxes`의 유량 방향 규약.
- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: 원래 `J`/`Phi` 후보 채택 정책의 연결 지점.
- `graphify-out/fv-root-cause-20260919/F462_GREEN_DESIGN_20261007.md`: 앞선 inexact 반복의 출처 및 receipt 검증 배경.
