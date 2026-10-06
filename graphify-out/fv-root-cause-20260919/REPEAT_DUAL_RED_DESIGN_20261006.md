# e0b 반복 dual-merit 보정 — RED 설계 검토

검토 범위: `e0b04a…`에서 시작하는 최대 3회 제한 반복 설계와 기존 커널. 반복 실행 모듈이 아직 없어서 소스 변경분 검토는 하지 않았다. FV/HVP/PCG/예측 재실행, 테스트 실행, 공유 그래프 변경도 하지 않았다.

## 판정

설계 방향은 진행 가능하다. 반복 커밋 의미와 두 독립 자원 상한의 상호작용을 명시해야 한다. 아래 항목은 구현 계약이지, 새 자원 한도를 요구하는 finding은 아니다.

## 필수 계약

1. **PCG 40회/solve와 누적 HVP 90회는 독립적인 거부 상한이다.** 두 제한은 세 번의 solve나 세 번의 채택을 보장하지 않는다. `src/advar/matrix_free.py:438-489`의 수렴·한도 시 참 잔차 재계산과 `fv_point_3h_schur_newton_step.py:114-119`의 방향 참 잔차 감사는 통상 PCG 반복 수 외 HVP를 더 사용한다. PCG 잔차 드리프트로 재시작하면 추가 연산자 호출도 생길 수 있어 `iterations + 2`는 완전한 solve의 상한이 아니다. 그러므로 40/90 정책은 그대로 두고, 공유 누적 HVP 계수를 매 product 시작 전에 검사해 90회에서 초과 호출 없이 중단한다. 이 중단은 이전 커밋을 보존한 유효한 `hvp_budget_refusal` 종료다. 기록에는 solve별 PCG 반복, 누적 started/completed HVP, 아직 시작하지 못한 solve/iteration을 구분해 세 번의 완전한 solve가 보장된 듯 표현하지 않는다.

2. **각 반복의 선형계는 그 반복의 현재점에서 다시 정의해야 한다.** `fv_point_3h_hvp_newton_step.py:202-224` 패턴대로 연산자 `H_k v = J_cc(c_k,p)v`를 현재 `c_k`에 닫고, 현재 `g_k`에 PCG를 적용한다. f82c Hessian은 고정 SPD block inverse 전처리기로만 재사용 가능하다. 이는 `H_k` 연산자를 대체하지 않으며 새 점의 Hessian/SPD 증거가 아니다. PCG 내부 `pᵀH_kp≤0` 또는 선형 잔차 실패는 현재 방향 생성 거부로 기록하고, 자동 J-only fallback이나 이동된 과거 곡률 사용 없이 종료한다.

3. **dual-merit 기울기는 현재점 HVP/참 잔차 receipt와 연결해야 한다.** 현재 방향의 `H_k s_k`가 필요하다. Newton solver의 참 잔차 `r_k=H_k s_k+g_k`로부터 `g_kᵀH_k s_k = -||g_k||²+g_kᵀr_k`를 계산할 수 있으므로 추가 HVP를 중복하지 않아도 된다. 역추적 후보마다 `c_k,g_k,J_k,Phi_k,s_k,H_k s_k`를 고정해 같은 두 Armijo 기준(`1e-4`)과 후보의 완전 strict endpoint branch gate를 검사한다. 후보 평가 중 `c_k` 또는 기준값을 갱신하면 안 된다.

## 반복 상태와 종료 영수증

- 한 반복의 채택 후보는 잠정 상태로 기록하고, branch partition·필수 endpoint 진단·입력/source/runtime 무결성 closure가 성공한 뒤에만 `committed_iterations`를 증가시킨다. 커밋 receipt에는 `c_{k+1}` 해시와 실제 `J/Phi/g`, branch hash, 방향 및 PCG 참 잔차·반복수·HVP 누적계수를 함께 저장한다.
- 이후 방향 생성/라인서치가 거부되거나 내부 예산에 닿아도 마지막 커밋된 점과 이전 커밋 receipt는 보존한다. 최종 상태는 `partial_convergence`가 아니라 구체적인 종료 원인(예: `pcg_curvature_refusal`, `dual_merit_grid_exhausted`, `branch_gate_exhausted`, `hvp_budget_refusal`, `internal_budget_refusal`)과 함께 `committed_iterations`로 표현한다. 끝까지 무결성 closure가 되지 않으면 단계 성공으로 세지 않는다.
- 외부 종료/강제 종료 시 디스크에 남은 `accepted_candidate` 기록만으로 커밋을 주장하지 않는다. 부모 실행 receipt와 child의 마지막 커밋 receipt가 같은 control hash를 가리키는지 구분한다. 정상성 목표 `||g||∞ ≤ 1e-10` 도달은 `root_pending_audit`로 기록하고, 최종 곡률·분기·수반 확인 전 root/최소점 완료로 올리지 않는다. 목표 미도달 뒤의 보정은 실패가 아니라 최대 반복 내 미도달 상태다.
- `dual_merit_search`는 후보 tensor를 objective, gradient, branch callback에 차례로 전달한다 (`fv_point_3h_dual_merit_step.py:176-240`). 알려진 ADVAR 목적함수는 순수 계산이지만, 반복 상태 소유권은 callback 계약으로 고정해야 한다. 간단한 보장은 각 callback에 별도 clone을 전달하고 기준 후보/해시를 보존하는 것이다. 또한 반경 후보가 FP64 반올림 때문에 현재 control과 완전히 같으면 Armijo 비교가 동률로 통과할 수 있으므로 `candidate == c_k`는 평가 전 `stagnated_step`으로 종료/다음 축소 없이 기록한다.

## 기존 근거와 한계

- `examples/weather_scenarios/fv_point_3h_hvp_newton_step.py:33-35, 202-224`는 기존 HVP/PCG 계수, 현재점 JVP operator, f82c 전처리기 전용 사용의 기준 구현이다.
- `examples/weather_scenarios/fv_point_3h_schur_newton_step.py:95-124`와 `src/advar/matrix_free.py:438-489`는 PCG SPD 가정, 비양의 곡률 거부, 참 잔차 재계산을 정의한다.
- `examples/weather_scenarios/fv_point_3h_dual_merit_step.py:160-240, 270-280`는 한 기준점에서 후보 두 함수를 검사하고 종료 전에 후보를 커밋하는 기존 패턴을 보여준다. 그 one-step transaction은 반복 receipt 소유권을 아직 정의하지 않는다.
- `graphify-out/fv-root-cause-20260919/E0B_REPEAT_DUAL_CHECKLIST_20261006.md`는 최대 3반복·HVP90·PCG40, gradient∞ 기준 `1e-10`, `root_pending_audit` 의도를 기록한다. 하지만 HVP/PCG 예산 모순과 반복 후 부분 성공 표현은 아직 체크되지 않았다.

### 이 메모에서 권고하는 가장 작은 계획 명료화

누적 HVP90과 solve별 PCG40을 별도 hard cap으로 유지하고 3회 완전 solve 보장이 없다고 계획에 쓴다. 반복은 마지막 완전히 closure된 endpoint까지만 커밋하고, 비양의 곡률/후보격자/분기/예산 종료를 서로 다른 terminal status로 저장한다. 채택 후보 callback에는 immutable basepoint/candidate 소유권을 보장한다. 이 계약을 반영한 구현이 생기면 해당 모듈과 합성 시험을 다시 RED 검토한다.
