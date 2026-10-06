# e0b 반복 dual-merit 보정 — RED prelaunch 검토

## 판정

**제한된 단일 실행을 시작해도 된다.** 이번 판정은 `e0b04a…`에서 최대 3회까지 반복하는 보정 진단의 설계·영수증 범위에 한정한다. 새점에서 정상점, 국소 최소점, 적격 발행점을 얻는다는 뜻은 아니다.

기준 계획은 `E0B_DUAL_MERIT_CONTINUATION_PLAN_20261006.json`, SHA-256 `ee75ca45b03495b055023a984457c4aab73eb76c646c1c17bf8760d2e53f80d9`다. 계획은 96개 source와 12개 archive를 고정하고, 최대 3회·누적 HVP 90회·solve별 PCG 40회·각 역추적 16후보·반경 0.05·내부 720초·외부 780초·표본 RSS 1 GiB를 정한다. HVP 및 PCG 상한은 독립 거부 한도다. 세 번의 solve 또는 채택을 보장하지 않고, HVP 호출 전에 누적 카운터를 확인한다.

## 확인한 연결과 제어 흐름

- 시작 control은 기록된 `e0b04a…`의 control hash와 이어진다. loader는 보관된 dual-merit 실행의 child/parent/resource 상태, 단일 통과 trial, 두 Armijo 조건, strict endpoint gate, fixed-input closure를 검사한다. 시작 직후 source/archive map을 다시 해시하고 새 계획의 SHA 및 각 pin과 비교하므로 loader와 계산 시작 사이의 plan/source 변경을 차단한다.
- 각 반복은 현재 control에서 `J`, 전체 gradient, `Phi`, strict branch를 다시 계산한다. 그 반복의 HVP closure는 현재 control과 고정 parameter를 복사해 사용하고, f82c 행렬은 block inverse preconditioner에만 쓴다. PCG 완료 뒤 현재 방향의 `H s`를 별도 계산해 참 잔차와 두 번째 Armijo 방향미분을 만든다. 같은 방향과 `Hs`를 해당 반복의 역추적에서 재사용한다.
- 각 채택 후보는 endpoint branch partition, 물리 상태 진단, fixed-input/source/runtime closure, deadline check를 마친 뒤에만 optimizer state와 완료 반복 목록에 커밋된다. 후보 실패나 후속 solve 실패는 앞선 커밋 control과 반복 receipt를 보존한다. 현재 반복 receipt는 solve 전에 기록되고 HVP 시작/완료, solve 완료 수, failed-solve의 반복 미기록 상태를 구분한다.
- 수렴 수치 기준은 `||g||∞ ≤ 1e-10`이다. 도달 결과는 `root_pending_audit`로만 남고 `eligible_stationary_point=false`, `full_root_claim=false`가 유지된다. iteration limit, curvature/linear solve 거부, branch/refusal, dual-merit grid exhaustion, budget refusal을 계산 완료·물리 검증과 혼동하지 않는다. 최종 input/source/runtime closure가 깨지면 integrity failure로 표기한다.
- 원래 목적함수, 관측, prior, parameter와 입력을 바꾸지 않는다. 새 HVP/PCG 반복 실행의 기록 검토는 하지 않았고, 모듈 단위에 한정했다. 구현자가 보고한 영향 시험 24개 통과와 typecheck 오류 0건은 재실행하지 않았다.

## 남은 관찰성 한계

한 가지 비차단 항목은 candidate branch gate가 terminal status에 별도 반영되지 않는 점이다. 모든 후보가 branch gate에서 탈락해도 terminal status는 `dual_merit_grid_exhausted`이며, branch 거부는 각 trial의 `strict_point_passed`와 `reasons`에서 구분한다. trial 영수증만으로 원인은 복구 가능하므로 현재 실행의 수치 타당성을 막지는 않는다. 후속 분석에서 단일 terminal 분류가 필요하면 trial 결과를 집계해 `branch_gate_exhausted`와 순수 dual-merit grid exhaustion을 구분하면 된다.

f82c 전처리기 출처는 고정 계획의 archive SHA와 저장된 curvature/checkpoint/parent/resource를 통해 보호되고, 코드에 그 행렬은 preconditioner-only라고 명시돼 있다. 코드가 curvature receipt의 `control_sha256`/`parameters_sha256`를 상수 기준점에 직접 대조하지는 않는다. 현재 frozen plan의 검토 범위에서는 실행 차단 사유가 아니지만, lineage를 코드 자체에서 완결하려면 이 두 identity 대조 또는 기존 `live_step.load_base()` 검증을 재사용할 수 있다.

## 실행 범위

이번 검토는 실제 FV, 새 HVP/PCG, 기상 예측, 수반, 비선형 재분석을 실행하지 않았다. 따라서 한 번의 제한된 실행이 성공해도 여러 반복 수렴, 최종 곡률·수반 적격성, 강수 예측 성능은 별도로 미검증 상태로 남는다.
