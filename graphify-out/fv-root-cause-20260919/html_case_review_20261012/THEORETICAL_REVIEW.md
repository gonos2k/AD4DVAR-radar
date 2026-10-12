# 수행 예시의 해결 가능성과 현재 기법의 한계

2026-10-12. 저장된 PR300–304 실행과 코드 계약을 검토했다. 이번 HTML 갱신에는 새 FV·HVP·최적화·미래점수 계산이 없다. 기존 4×5·2분 데모와 26제어·12점관측·3시간 연구는 서로 다른 문제다.

## 1. 원래 최소화 문제: 존재는 뒷받침되지만 유일성은 미증명

고정 입력 p에서 연구 profile의 원래 비용은

\[
J(c,p)=\sum_i\rho(r_i(c,p))+\tfrac12\|c\|^2+J_s(c),\quad c\in\mathbb R^{26}.
\]

`FVPointResearchProblem.objective`는 pseudo-Huber 관측비용, 제어 prior, smoothness prior를 합한다 ([코드](../../../src/advar/fv_point_research_problem.py#L360)). 이 profile은 neural prior 없이 identity 제어 prior와 smoothness weight=0을 요구한다 ([계약](../../../examples/weather_scenarios/fv_point_3h_tangent_coupled_gn_comparison.py#L297)); `_control_prior_residual`은 해당 경우 c를 반환한다 ([코드](../../../src/advar/variational.py#L5476)). 각 비용은 비음수이므로 **J(c,p) ≥ ½‖c‖²**이다.

고정 geometry·boundary·시간단계의 정확한 실수 연산에서 연속·유한한 forward/objective를 전제로 한다. `_minmod`는 분기에서 미분 가능성이 바뀌지만 함수값은 연속이다 ([코드](../../../src/advar/transport.py#L251)). 유동계수는 고정 CFL-safe box에서 limits·tanh(control)로 정해지며 시간단계를 control마다 바꾸는 적응형 문제는 아니다 ([코드](../../../src/advar/transport.py#L767)). softplus, tanh, 연속 수송·관측 조합은 이 존재 논증에 필요한 연속성을 제공한다. **비용이 무한대에서 발산(coercive)하므로 하위수준집합이 닫히고 유계이며, 유한차원에서 적어도 한 전역 최소해를 달성한다.** FP64 overflow·검증 코드가 거부하는 입력영역까지 포함한 전역 실행 가능성 인증은 아니다.

identity prior가 있다고 전체 목적함수가 강볼록해지지는 않는다. 매끄러운 구간에서 data Hessian에는 `AᵀDA` 외에 `Σρ′(rᵢ)∇²rᵢ`가 있다. 뒤 항의 부호를 제어하지 못하면 `I` prior만으로 전역 볼록성·최소해 유일성·정확 Hessian의 양의 정부호성을 보장할 수 없다. GN 검색계의 양의 정부호성과 구분한다.

독립적인 1변수 반례도 있다. 이 profile과 같은 δ=2 pseudo-Huber를 사용한 `j(x)=ρ₂(x²−1)+½x²`는 `j≥½x²`이지만 `j″(0)=1−2/√1.25<0`이다. `x=±√(1−2/√15)`에 두 동일 최소해가 있다. 따라서 coercivity와 identity prior는 볼록성·유일성을 함의하지 않는다. 이 반례는 실제 FV가 두 최소해를 가진다는 증명이 아니다. 독립 재산술은 [THEORY_CHECKS.json](THEORY_CHECKS.json)에 기록했다.

## 2. 상태 복원: 관측만의 국소 식별 가능성은 부족

현재 all-detected profile은 관측 (3,4), 제어 26, 파라미터 13이다 ([layout](../../../src/advar/fv_point_research_problem.py#L208)). 매끄러운 분기의 관측 Jacobian A는 12×26이므로 rank(A) ≤12, dim ker(A) ≥14이다. 이는 **관측만으로 미소 제어 변화 전부를 국소적으로 구별할 수 없다는 뜻**이다. 정확한 비선형 해집합이 반드시 14차원이라는 주장은 하지 않는다. prior가 포함된 전체 Hessian이 특이라는 뜻도 아니다.

prior가 관측 null 방향을 정규화해 분석점을 선택할 수 있지만, 실제 참상태나 생성계 제어를 선택한다는 보장은 없다. 심지어 생성계와 같은 forward이고 관측잔차가 0인 제어 c_true에서도, 매끄러운 경우 data gradient는 0이고 identity-prior gradient는 c_true이다. c_true ≠0이면 그 점이 정규화 J의 정상점일 필요가 없다. **정규화 문제의 최소점, 무오차 관측 적합, 유일한 참상태 복원, 미래점수 최적화는 각각 다른 목표**다.

## 3. Q=0 면: 제한해의 존재와 원래 해의 자격은 다름

Q(c)=wᵀtanh(c[20:25])=0은 연속 함수의 0-level set이다. 0 유동을 포함하므로 비공집합이고 닫혀 있다. 원래 비용의 coercivity를 유지하므로 위 전제 아래 이 전체 면에도 제한 최소해가 존재한다.

그러나 **현재 차트의 정칙·단일 공통면·limiter 양측 자격을 만족하는 해가 존재한다는 결론은 나오지 않는다.** 지원되는 차트/활성분기 부분집합은 전체 닫힌 면과 다르다. 면 제한 최소해는 원래 unconstrained/nonsmooth 최소점이 아닐 수 있다. 접선 정상성, 유효한 양측 gradient 혼합, 원래 활성구조 및 정확 축약 곡률을 확인해야 한다. 현재 producer는 native face release가 필요하면 확정을 거부한다 (`fv_point_3h_netflux_subspace_trial.py`, candidate qualification). 두 분기만으로 복수 limiter 전환의 전체 Clarke 조건을 인증하지 않는다.

최소해가 존재하더라도 국소 해 응답 c*(p)의 유일성·매끄러움은 보장되지 않는다. 정확 implicit 수반에는 원래 전체 정상성/KKT와 필요한 활성 제약·승수를 포함한 적격 F(z,p)=0, 비특이 F_z, 필요한 활성구조 안정성이 있어야 한다. 검색 GN·계수 solver 미분·한 방향 HVP는 이를 대체하지 않는다.

## 4. 실제로 잘된 것과 안 된 것

| 사례 | 확인한 성과 | 확인하지 못한 것 또는 제한 |
|---|---|---|
| 기존 4×5 FV 응답 | 저장된 비용 감소, gradient 기준, GN 선형 수반 잔차, 두 짧은 리드 MAE의 persistence 대비 감소 | 정확 FSOI·일반 기상사례 성능; 26변수 문제와 별개 |
| PR300 gradient/GN | 같은 점에서 두 후보 독립 재평가; gradient R 우선 선택, GN은 J를 더 낮춤 | GN χ≈87.47%가 큰 유한 이동에서 실현되지 않음. 앞선 9개 후보 R 증가; 모든 후보 selector 변경. 특정 selector 인과효과/HVP 오류 증명은 아님 |
| PR301 gradient+GN | 원래 J/R 한 점 감소 | GN 계수 0, 동일 후보. 추가 보정 성능 없음 |
| PR302 KKT 준비 | 현재점 미분 갱신과 사전 반경 진입 조건 판단 | 후보 0개. 물리 실패·수렴이 아니라 GN 추가이득 조건의 미충족 |
| PR303 연속 gradient | 9점 확정, R 4.215677% 감소 | 예산 종료, ‖G‖≈0.0494092. 2–9단계는 방향별 모형 최소배율, χ≈0.44%. 내부 예산 7.160초 초과 |
| PR304 gradient+s | 원래 J/R 한 점 감소; 새 방향의 실제 HVP/gate 완료 | s 계수 0, 두 경로 동일 fb0a. 추가 가속 미입증; 새 fb0a 미래점수 없음 |
| 별도 미래 관찰 3쌍 | 고정 합성 참값의 최종 +180분 MSE가 조금 감소 | 후보 선택에 사용 안 함. 독립 forecast skill·통계적 유의성·전체 lead 평균이 아님 |

원수치와 점 pairing은 [CASE_REVIEW.json](CASE_REVIEW.json)의 SHA-256 원자료 목록에 연결된다. PR304 mapping은 [현재 draft PR304](https://github.com/gonos2k/AD4DVAR-radar/pull/304)의 branch `codex/residual-direction-efficiency`, 당시 head `b481fb12d12e01baa31273920fa1cf057290ae0d`에서 확인했다. 이전 검토의 각 별도 합성 solver 반례 수정은 실제 FV 성능으로 집계하지 않는다.

## 5. 문제 성격 때문에 추적하기 어려운가

관측의 미소 null 방향, 비선형 수송/관측, minmod의 비매끄러운 분기, 조건부 활성면 가설은 어려움을 만들 수 있다. **실제 근거로 확인된 것은 국소 방향의 제한된 잔차 제거율과 유한 이동에서의 모형 불일치**이다. PR300의 분기 변경/큰 벡터오차와 PR303의 대체로 정확하지만 χ가 작은 구간을 구분한다. PR303의 8번째는 eta_vec≈0.31046인 예외이며 나머지 8개는 eta_vec≤1.59×10⁻⁴이다. 아홉 후보 모두 실제 J/R 승인을 통과했다. 모형이 정확한 후반을 반경 부족이나 RAM 부족만으로 설명할 수 없다.

J와 R의 매 스텝 동시 감소, 비음수 두 계수, 고정 반경, 지원되는 분기만의 후보 승인은 실행 정책이다. 원래 J 최소화의 필수조건과 같지 않으며 경로를 제한할 가능성이 있다. 그 가능성이 전체 정체의 원인이라는 증명은 아직 없다. 큰 GN 후보의 J 감소/R 증가가 상충을 실제로 보여주지만, 그 후보를 승인할 자격이나 이후 R 회복을 증명한 것은 아니다.

포화도 과장하지 않는다. 현재 identity-prior J≈0.06099보다 비용이 낮은 해에는 ‖c‖≤√(2J)≈0.3493이다. 유동·성장 control 성분도 그 범위 안에 있으므로 tanh′(c_i)=sech²(c_i) ≥약 0.887이다. **강한 tanh 포화가 이 경로의 주된 병목이라고 단정할 수 없다.** 이 bound는 field softplus의 conditioning, physical coefficient scales, 전체 Hessian의 conditioning까지 보장하지 않는다.

따라서 판단은 다음과 같다: **예시는 수학적으로 최소해가 존재할 근거가 있는 정규화 문제다. 현재 방식이 빠르게 그 해를 찾고 참상태·정확 응답까지 복원한다는 근거는 부족하다.** 기존 합성 한 사례의 서로 다른 계산점/방법을 기상 유형별 성공·실패로 일반화할 수 없다. 다음에는 적격 원래 최소점/활성구조를 찾는 능력과 관측 식별·prior 의존성을 분리해 확인해야 한다. 이미 추가이득이 없는 동일점 비교를 그대로 반복하는 것은 근거가 부족하다.
