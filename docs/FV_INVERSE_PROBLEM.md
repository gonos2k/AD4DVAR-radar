# FV 역문제의 목표·탐색 범위·완료 조건

이 문서는 현재 12관측·26제어 FV 합성 실증의 기준 문제를 정의한다. 주 목표는 **20분 분석창의 원래 강건 정규화 비용 J의 적격 국소최소점**을 구하는 것이다. 특정 영유량면에서의 보정은 이 목표에 조건부인 탐색이다. 기본 README의 전체 3-frame nowcast와 이 작은 FV 실증을 구분한다. 전역 최소점 보장은 완료 조건에 포함하지 않는다.

## 1. 원래 문제와 시간

고정된 관측·배경·경계·격자·시간 및 원래 FV/minmod 모형에서

\[
\mathcal P_{\mathrm{original}}:\quad
\min_{c\in\mathcal D_{\mathrm{phys}}}J(c;p)
\]

의 국소해를 찾는다. D_phys는 제어 변환과 수송 상태가 물리적·수치적으로 유효한 영역이다. minmod/upwind 전환점은 함수값이 유효하면 이 영역에 포함될 수 있다. strict 미분 검사의 지원 거부를 물리적 정의역 삭제로 해석하지 않는다. 현재 구현은 이 고정 profile의 입력·상태 불변조건과 국소 미분 지원을 검사하며, D_phys의 전역 membership이나 전체 허용 근방을 판정하는 인증기는 아니다.

| 항목 | 현재 고정 profile |
|---|---|
| 상태 격자·제어 | 4×5 에코 상태, 26개 정규화 제어 |
| 관측 | 고정 비격자점 4개 × 시각 3개 = dBZ 잔차 12개 |
| 관측 시각·분석창 | 0/600/1200초, 20분 |
| 분석 적분 | 관측간격당 FV 부스텝 90개, dt=600/90초, SSPRK2 Euler 단계 360개 |
| 미래 예측 | 마지막 관측 이후 +10,…,+180분, 18개 lead |
| 마지막 검증 시각 | 최초 시각 기준 12000초; 마지막 관측 이후 10800초 |
| 분석+미래 적분 | Euler 단계 3600개 |

분석비용의 gradient/HVP는 분석창의 연산이다. 그 검증을 3시간 미래궤적 전체의 미분 인증으로 확장하지 않는다. 근거: [시간 계약](../examples/weather_scenarios/fv_point_3h_seed_linear_probe.py), [점 관측 문제](../src/advar/fv_point_research_problem.py).

## 2. 제어·외부 파라미터·보조변수

c=(c_field,c_psi,c_growth)는 20+5+1개의 제어다. 초기장 제어는 배경 dBZ의 softplus 역변환 공간에 증분을 더한다. 유동은 정적 스트림함수의 다섯 기저 계수이고, 성장도 시간·공간에 공통인 한 계수다.

\[
\mathrm{dBZ}_{0,i}=Z_{\min}+s_Z\operatorname{softplus}\left[
\operatorname{softplus}^{-1}\left(\max\left(\frac{b_i(\beta)-Z_{\min}}{s_Z},\epsilon_{\mathrm{transform}}\right)\right)
+\frac{s_c}{s_Z}c_i\right],
\quad a_k=L_k\tanh c_{\psi,k},\quad
\ell=\ell_{\max}\tanh c_{\mathrm{growth}}.
\]

현재 s_Z=1 dBZ, s_c=4 dBZ이며 지원 분기를 따로 확인한다. ell은 관측간격당 로그성장이고, 연속 성장률로 표현할 때는 s=ell/600초다. 계수와 성장은 분석·예측 중 고정된다. 에코 대리량 q의 이류·곱셈 성장 모형이며 수분량·강수량 전체 모형은 아니다. 모델오차를 시간별 독립 제어로 추정하지 않는다. strong-constraint형의 제한된 에코 역문제다. weak-constraint에서 모델오차를 별도 제어로 추정하는 구성과 구분한다. [ECMWF 설명](https://www.ecmwf.int/en/elibrary/76747-model-error-estimation-4d-var)

\[
p=(y_{0,1:4},y_{1,1:4},y_{2,1:4},\beta)\in\mathbb R^{13},
\qquad b(\beta)=b_0+\beta B_{\mathrm{pattern}}.
\]

inner 최적화에서 p는 고정한다. 현재 사례의 beta는 0이다. 코드 layout의 `theta_index=12`와 support의 background `theta`는 **외부 beta의 기존 API 명칭**이다. 문서·수식에서는 beta로 표기한다. 양측 gradient 혼합 보조변수는 **vartheta**로 표기하며 c나 p의 성분이 아니다. 이 문서는 동결 API 이름을 바꾸지 않는다.

근거: [제어 변환](../src/advar/variational.py), [외부 파라미터 contract](../src/advar/fv_point_research_problem.py), [현재 사례 생성](../examples/weather_scenarios/fv_point_3h_forward_case.py).

## 3. J와 E의 정확한 역할

예측 관측은 에코를 dBZ로 변환한 다음 고정 점 좌표에서 bilinear interpolation한 값이다. 선형 반사도 평균 후 dBZ 변환이나 빔 체적 평균과 같지 않다. 현재 네 좌표는 row/column cell-center 인덱스 기준 (0.35,0.65), (1.25,2.40), (2.50,3.10), (1.70,1.45)다.

\[
\widehat y_k=\mathcal H_{\mathrm{point}}[\mathcal Z(q_k(c,p))],\quad
r_k=C^{-1/2}D_k(\widehat y_k-y_k),\quad
D_k=\operatorname{diag}(\sqrt{w_{ki}}/\sigma_{ki}),
\]
\[
J(c;p)=\sum_{k=0}^{2}\sum_{i=1}^{4}\rho_2(r_{ki})+\tfrac12\|c\|_2^2,
\qquad \rho_\delta(r)=\delta^2\left(\sqrt{1+(r/\delta)^2}-1\right).
\]

현재 표준편차는 0.1 dBZ, quality는 0.5∼1.0이다. C의 대각은 1이고 C02=C20=0.3, C13=C31=-0.2, 나머지 비대각은 0이다. 같은 시각에서 표준화 후 symmetric inverse-square-root whitening을 한다. 시각 간 상관은 없다. 관측은 모두 detected, prior는 정규화 제어의 identity prior, field smoothness weight=0, neural prior 없음, pseudo-Huber delta=2다. 서로 다른 profile로 확장하면 별도 계약을 기록한다.

\[
E(c,p)=\frac1{20}\sum_{i=1}^{20}
[Z_i(12000;c,p)-Z_i^{\mathrm{ver}}]^2.
\]

J는 표준화 잔차와 정규화 제어의 무차원 비용이고, E의 단위는 dBZ²이다. E는 **최종 +180분 lead의 격자 dBZ MSE**다. 18개 lead 평균이 아니며 J에 포함되지 않는다. 여러 lead의 예측은 생성하지만 현재 `forecast()` 반환은 최종 프레임, `score()`는 그 MSE다. 진단용 E를 관찰하는 것은 가능하나, 후보 선택에 사용하면 탐색 계약을 변경한 것이다.

근거: [점 sampler](../src/advar/fv_point_sampler.py), [잔차·objective·forecast·score](../src/advar/fv_point_research_problem.py), [현재 GN profile 검사](../examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py).

## 4. Q=0 보정은 활성면 가설이다

최근 탐색은 `M={c:Q_y[4,3](c)=0}`의 차트와 양측 국소 연장을 사용한다. Q는 설정된 모형 좌표의 면 체적유량(좌표 면적/초, 깊이 제외)이며 계수의 단위는 psi 기저에 종속된다. 0.104와 정규화 척도 q_s=0.84는 같은 유량 convention을 사용한다. 관측 풍속이나 무차원 에코량으로 해석하지 않는다. 원래 J의 비용식에는 이 등식 제약이 없고, D_phys에도 영구 물리조건으로 추가하지 않는다. `selected_face_extension`은 선택 부호 반대편에서 물리적 upwind 규칙이 아닌 연구용 연장이다. 그 값을 원래 native 비용으로 대신하지 않는다.

단순히 같은 J를 사용한다고 원래 문제와 탐색 영역이 동치인 것은 아니다.

\[
\inf_{\mathcal D_{\mathrm{phys}}}J
\le\inf_{\mathcal D_{\mathrm{phys}}\cap M}J.
\]

최근 혼합 조건

\[
F(c,\vartheta;p)=\begin{pmatrix}(1-\vartheta)\nabla J_-+\vartheta\nabla J_+\\Q(c)/0.84\end{pmatrix}=0
\]

이며, 최근 기록의 G(c)는 후보 양측 gradient에서 norm이 최소인 vartheta를 다시 구해 평가한 F(c,vartheta;p), R=‖G‖₂²이다. Q/0.84는 정규화된 면 잔차다. F는 단순한 면 위 제한 최소화보다 강한, **선택한 단일 활성면에서 원래 비매끄러운 정상성을 찾으려는 가설**이다. 정칙 면·양측 연장 일치·실제 도달 가능성·다른 활성 사건의 처리가 유효해야 한다. F=0도 최소점의 충분조건은 아니다. 현재 `minimum_mixture_weight()`의 jump 해상도와 strict-interior vartheta gate는 이 보정 경로의 지원 조건이다. 가중치가 0/1 끝점이거나 jump가 해상되지 않아 지원 거부되더라도 수학적 국소최소가 불가능하다는 뜻은 아니다.

생성 기저 (y,x,xy,(x²−y²)/2,x²y), 한계 L=(0.11,0.08,0.07,0.04,0.03), 참값 비율 f=(0.7,-0.6,0.5,0.4,-0.3)을 코드의 Q_y=-Delta_x psi에 대입한 **정적 재산술**은 Q_y[4,3]=0.104다. Q=0은 합성 참값에 주어진 제약도 아니다. 이 값은 J의 국소최소점 위치를 증명하지 않는다.

후속 실행은 다음 결정을 명시한다. 현재 점의 native 양측 법선 비용 변화와 다른 사건을 갱신한다. 원래 허용 방향에서 하강이 확인되면 면을 떠나는 탐색을 허용하거나 면 가설을 보류한다. 면 위 정상성 후보를 예산 내 얻지 못하면 해당 가설의 미완료로 기록한다. 원래 문제에 해가 없다고 결론내리지 않는다. 과거 모든 selector를 제약으로 추가하지 않는다. **현재 보정기는 면을 자동 해제하는 일반 solver가 아니며, 이 결정 규칙의 실제 구현·실행은 후속 계획으로 검증해야 한다.**

근거: [국소 연장의 범위](../src/advar/transport.py), [혼합 잔차](../examples/weather_scenarios/fv_point_3h_tangent_continuation.py), [생성 기저](../examples/weather_scenarios/fv_minmod_inverse_probe.py), [생성 계수](../examples/weather_scenarios/fv_long_horizon_case.py).

## 5. 정상성 후보와 최소점 자격

주 목표의 정확한 해는 어떤 물리적 근방에서 모든 허용 c에 대해 J(c;p)>=J(c_star;p)를 만족하는 국소최소점이다. 아래 수치 gate는 그 자격을 검사하는 절차이며 이 부등식의 무조건적인 보증은 아니다.

- **매끄러운 내부점:** 원래 26성분 gradient와 정확 곡률을 검사한다. 기존 정규화 제어의 FP64 기준 `max(abs(gradient)) < 1e-10`을 정상성 **후보** 기준으로 유지한다. 정확 Hessian의 수치적으로 분해 가능한 양의 곡률과 branch 안정성을 함께 확인한다. 이는 정칙 비퇴화한 strict 국소최소의 충분조건을 검사하는 범위이며, 평평하거나 고차로 퇴화한 국소최소는 이 인증 범위에서 미해결로 남긴다. 검색용 GN의 SPD로 대신하지 않는다.
- **단일 정칙 전환면:** 같은 사건점의 실제 양측 limiting gradient가 유효할 때 최소혼합 residual, Q의 geometry-scaled closure, 양측 바깥 법선 방향의 비용 변화와 정확 접선/축약 곡률을 검사한다. 후보 기준의 제어 gradient 성분 척도는 위와 같고, Q는 코드의 유량·FP64 closure bound로 별도 판정한다. 면의 곡률에 따른 항을 생략한 단순 projected GN을 정확 축약 Hessian으로 쓰지 않는다.
- **복수 전환면:** 같은 사건점에 도달 가능한 모든 관련 영역을 먼저 특정한다. 이론적 selector 조합이나 떨어진 후보 gradient를 무조건 혼합하지 않는다. limiting gradient의 최소 유클리드 norm 볼록혼합을 정상성 진단으로 쓰려면 접근 오차와 활성구조의 지원 범위를 제시해야 한다. 유한 표본 hull은 후보 조사 근거다. 단일면의 숫자 기준을 이 결과에 기계적으로 적용해 종료하지 않는다. 허용 방향의 일차 비용 변화와 임계 접선/영역의 정확 이차 조건을 포함한 별도 최종 자격 계획이 없으면 `minimum_qualification_unresolved`로 남긴다.

국소 Lipschitz 가정 아래 Clarke 정상성은 필요조건으로 사용할 수 있지만 국소최소의 충분조건이 아니다. -|x|도 0이 양측 gradient hull에 들어간다. [비매끄러운 최적화 연구](https://optimization-online.org/2003/10/762/)

정확 곡률의 symmetry·eigenpair·solve·양의 곡률 floor는 [기존 scale-aware 계약](../examples/weather_scenarios/fv_point_3h_terminal_block_schur.py)을 기준으로, 실제 인증할 연산자와 좌표를 최종 계획에 고정한다. smooth/단일면 밖의 일반 최소점 인증기는 아직 구현되지 않았다. 본 문서는 그 공백을 완료 처리하지 않는다.

| 실행·수치 조건 | 인정하는 의미 | 최종 최소점 선언 |
|---|---|---|
| native J/R 충분 감소 | 한 후보의 해당 정책 승인 | 불가 |
| R_end<R_start(1-1e-8) | 제한된 경로의 상대 회복 | 불가 |
| norm(G)<=1e-8 | 최근 코드의 추가 자격검사 필요 후보 | 불가 |
| P2 재평가 | 같은 제어의 비용·개별 gradient·trace·입력 등의 재현 | 불가 |
| 예산 내 정상 종료 | 제어된 실행 완료 | 불가 |
| 정상성 후보 + 정확 최소조건·활성 안정성 | 지원 범위 안의 적격 국소해 근거 | 범위를 명시해 판정 |

매 스텝 J/R 동시 감소는 현재 탐색 정책이며 원래 J 최소화의 수학적 필요조건은 아니다. 회복 상대값 1e-8, norm 후보값 1e-8, gradient 최대값 1e-10은 서로 다른 양이다. 정상성 수치·오차·norm·scale, 최소조건과 지원 범위, 최종 재평가를 같은 종료 기록에 제시한다. 참 gradient 오차를 P2의 동일 구현 재현만으로 인증하지 않는다.

## 6. 분석해의 민감도

평가 대상은 E(c_star(p),p)이며 p의 어느 성분을 바꾸는지 기록한다. p[0:12] 변경은 관측만 바꾸고, beta 변경은 고정 b0/pattern에서 초기 배경을 바꾼다. 경계 echo/support, 격자·시간·관측 위치·std/quality/correlation·제어 척도·검증장은 고정한다. p를 바꿀 때 합성 사례를 다시 생성하거나 관측을 다시 detached truth로 덮지 않는다. 여러 해가 있으면 자격 있는 기준점에서 이어지는 국소 가지를 선택하고, 초기값·재최적화·활성구조 변화를 함께 보고한다. 가지의 존재·유일성·미분 가능성을 가정으로 숨기지 않는다.

적격하고 비특이한 정확 최적조건 F(z,p)=0에서만

\[
F_z^T\lambda=E_z,\qquad dE/dp=E_p-\lambda^T F_p
\]

를 사용한다. smooth 해의 z는 c, 단일 정칙 활성면의 유효한 결합계에서는 z=(c,vartheta)로 정의할 수 있다. 여러 사건·활성 변경에서는 해당 계와 방향 응답의 정의를 다시 제시한다. 고정 Q=0 **제한해**의 수반을 원래 전체 문제 분석해의 수반으로 자동 승계하지 않는다. GN 역행렬은 정확 최적조건 Jacobian·분석오차 공분산이 아니다.

최종 검증은 정확 전치 내적/선형 잔차, 실제 p±h 섭동 후 같은 문제를 재최적화한 해의 E 중심차분 또는 적절한 일방 차분과 수반을 대조한다. 해 공식을 직접 대입한 합성 차분은 실제 FV 재분석을 대신하지 않는다.

## 7. 연구 완료와 현재 상태

| 검증 축 | 필요한 완료 근거 |
|---|---|
| 역문제 solver | 원래 전체 문제의 적격 국소해; 선택면 가설 유지·해제와 정상성/최소조건 기록 |
| 합성 상태 복원 | 알려진 상태 오차와 prior 의존성; 비용 최소화와 참값 복원을 구분 |
| 수송·미래예측 | 독립 해·정확 셀 평균·경계·시간 대조, 현재 대상 최종 E 및 lead별 관찰 범위 |
| 분석 민감도 | 선택한 해 가지의 정확 최적조건, 수반/VJP 및 실제 섭동 재최적화 |

관측·검증을 같은 FV 연산자로 생성한 현재 사례는 matched twin이다. 고정 배경을 통계적으로 독립인 오차 표본이라고 부르지 않는다. 정확 관측 적합점에서도 비영 c_truth라면 identity prior 때문에 gradient J=c_truth가 될 수 있다. 관측 Jacobian rank<=12라는 사실은 상태 식별의 한계이며 전체 정규화 해의 비유일성 증명은 아니다.

현재 P2-closed endpoint는 `68a6e821…11f9ed8`이고 적격 최소점·수반·현재점 미래점수는 미완료다. [PR293 진단](../graphify-out/fv-root-cause-20260919/ACTIVE_INTERSECTION_FINDINGS_20261011.md)에서는 네 유한 gradient hull norm 약0.055913, 공통 접선 gradient norm 약0.055882와 native Q=0 차트 비용 하강을 관측했다. 현 상태를 이미 교차점 최소점으로 취급하지 않는다.

기존 1차 100%와 FV 약70%는 과거 가중 마일스톤의 역사적 진행표다. 원래 문제를 정확히 그 비율만큼 해결했다는 의미가 아니다. 완료 정의가 바뀌면 대응표를 공개하고 새 근거 없이 점수를 올리지 않는다. 배포·운영·실자료 확대는 현재 연구 분모에 추가하지 않는다.

## 8. 설명 정정과 다음 실행

과거 기록의 “원래 J 유지”는 **비용 functional·관측·prior·수송을 유지했다**는 뜻으로 한정한다. 탐색 영역·정상성 모델·원래 문제의 해·정확 응답까지 동치라는 뜻으로 읽지 않는다. 과거 frozen 기록과 수치는 수정하지 않고 이 문서로 해석을 정정한다.

다음 실행은 새 solver 도입 전에 search domain과 활성면 가설을 명시하고, PR293의 남은 접선 하강을 native J 탐색에 연결한다. 원래 전체 잔차와 후보 자격을 평가하되, 한 스텝 승인 정책과 최종 국소최소 조건을 구분한다. 정상성·곡률·수반·미래검증의 미완료를 한 단계의 성공으로 닫지 않는다.
