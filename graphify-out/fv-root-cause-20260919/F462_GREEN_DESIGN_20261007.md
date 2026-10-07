# f462 inexact 반복 — GREEN 사전 설계 검토

날짜: 2026-10-07
범위: 현재 반복기·committed-base loader와 PR #255 보관 실행 기록의 읽기 전용 검토. 새 FV/HVP/PCG, 테스트, 예보, 수반, 재분석은 실행하지 않았다. 공유 소스나 그래프는 수정하지 않았다.

## 판정

f462에서 inexact 반복을 이어가는 것은 수치적으로 타당하다. PR #255의 선택 결과는 top-level 비교 영수증에 보존되어 있고, 새 반복은 그 제어값에서 **새 gradient와 현재점 HVP로 시작**하면 된다. 다만 기존 loader의 입력 계약은 일반적인 완료된 endpoint를 읽는 형태다. strict/inexact 비교 영수증을 일반 반복 영수증으로 암묵적으로 받아들이지 말고, 계획에 고정한 `comparison_selected_inexact` 출처 모드에서 선택 arm·비교 완료·부모/자원 영수증을 추가 확인해야 한다.

선택 기준은 strict 비교점 `e986…`가 아니라 `f462a4961634efa253e3757f7de44304386135eb3a229f3729a61eb911e6eed9`다. 실제 inexact arm raw는 두 arm의 공유 비용으로 누적된 `hvp_calls=53`을 담지만, 선택 arm의 비용은 `new_hvp_calls=22`다. 새 실행의 90 HVP 상한은 새 실행에만 적용하고, 이전 비교의 누적 카운터를 새 반복 카운터로 넘기면 안 된다.

## 저장 기록에서 확인한 시작점

비교 top raw는 `d31_inexact_comparison_20261007_attempt1/step.json`, 부모는 `step.run.json`, 자원 기록은 `step.resource.json`이다. top raw는 `phase=finished`, `execution_status=completed`, `numerical_status=paired_comparison_completed`, `comparison_complete=true`, `selected_arm=inexact`, `planned_selected_arm=inexact`를 기록한다. 부모는 child SHA를 top raw의 SHA와 일치시키며, `completed_iterations=1`, `hvp_calls=53`이다. resource 기록은 exit 0, 695.530초, sampled RSS 373,604,352 bytes, 제한 780초/1 GiB, 종료 신호·monitor 오류·resource termination 없음이다.

top raw의 `base_control_sha256`는 d31을, `accepted_control_sha256`는 f462를 가리킨다. `optimizer_steps_applied=1`, accepted trial 1개, iteration 1개, `current_state` 및 `accepted_control`은 f462의 상태로 닫힌다. 파라미터 해시는 계속 `8871db49…`; runtime은 CPU FP64 / Python 3.12.13 / Torch 2.13.0이며 before/after가 같다. source map은 121개 항목이고 before/after가 동일하다. 저장된 inexact arm 요약은 1 step accepted, `rtol=1e-3`, 실제 상대잔차 `3.77048408e-4`, 20 PCG 반복, 22 arm HVP를 기록한다. 이는 저장 영수증 확인이며 이번 리뷰가 모델 연산을 재실행한 결과가 아니다.

끝점은 아직 정상점이 아니다. `||g||∞=0.416618997`이고 최종 root 기준은 `1e-10`이다. PR #255의 한 스텝은 다음 반복을 시작할 적격 출발점이지 수렴·곡률·수반 인증이 아니다.

## Loader 계약: 비교 선택점과 일반 반복점

기존 `fv_point_3h_committed_base.py`의 `_accepted_endpoint`는 마지막 accepted trial, 마지막 iteration, `current_state`, 제어 해시, J/Phi Armijo와 strict endpoint branch flag를 닫는다. `_validate_completed_receipt`는 child hash, 완료 반복/HVP 수, resource 완료, source/input/runtime 일관성을 검사한다. 하지만 이 일반 계약만으로는 비교 영수증에서 **어느 arm이 선택됐는지**를 증명하지 않는다. 현재 코어 `dual_merit_continuation.load_base`는 d31/e0b 계보와 특정 BASE 경로에 묶여 있으므로, f462 반복에는 비교 계획을 명시적으로 받는 adapter가 필요하다.

비교 출처 모드에서는 다음을 모두 요구한다.

1. 새 계획에 정확한 top raw 경로와 SHA, 해당 비교 계획 경로와 SHA, top parent/resource 경로와 SHA를 고정한다. 비교 계획의 `source_files`가 실행 raw의 `source_before/source_after` map과 일치하며, 비교 계획·raw·parent의 child hash가 서로 닫히는지 확인한다. 별도 신규 반복기/loader 소스는 새 계획에서 추가 pin한다.
2. top-level raw에서 `comparison_complete is True`, `selected_arm == planned_selected_arm == "inexact"`를 확인한다. `arms.inexact`가 한 accepted step과 정상 실행·입력/source 무결성을 가졌고, 그 endpoint 해시·값·J·Phi·gradient·branch가 top-level selected endpoint와 일치하는지 닫는다. `arms.strict`는 대조 기록으로 남기되 시작점 선택이나 fallback에 사용하지 않는다. `selected_arm=null`, 미완료 비교, 예산 거부, 무결성 오류면 loader가 거부한다.
3. top raw, parent, resource에서 `phase=finished`, 실행 완료, child SHA, completed iteration 수, 누적 parent HVP 수가 일치하는지 확인한다. resource는 exit 0, 제한 안 elapsed/RSS, signal/monitor/termination 오류 없음을 요구한다. numerical status는 비교 자체의 `paired_comparison_completed`와 선택 arm의 한-step 상태를 각각 검사한다. accepted endpoint가 있다고 해서 budget-refused/미완료 comparison을 정상 비교로 승격하지 않는다.
4. `accepted_control`의 FP64 SHA, 선택 arm accepted trial, top `iterations[-1]`, `current_state`, input-after control SHA를 동일한 f462 값으로 닫는다. 파라미터 해시, fixed input identity와 허용 runtime도 비교 계획에 고정한다. 저장된 제어값에서 목적함수·gradient를 새로 계산하고 다음 반복은 그 재구성된 현재점으로 시작한다.
5. 비교 부모의 `hvp_calls=53`은 두 arm 공유 총계다. inexact 방향 비용은 별도 `new_hvp_calls=22` 및 선택 arm delta로 보존한다. 둘 중 어느 수치도 다음 실행의 시작 카운터가 아니다. 다음 guarded launch는 HVP 카운터 0에서 시작하고 새 실행에서 사용한 모든 PCG HVP와 독립 true-residual HVP를 90 상한에 포함한다.

일반 반복 raw와 비교 top raw는 같은 endpoint 필드를 공유해도 의미가 다르다. selector를 raw 형태나 `accepted_control` 존재만 보고 추론하지 않는다. plan-pinned enum/source mode를 요구하고, mode별 필수 receipt 검증을 분리하는 것이 안전하다.

## 다음 반복 수치 정책과 고정 게이트

새 반복 계획은 비교 실행과 같은 원래 목적함수·관측·prior·제어/파라미터를 유지하고, 시작 제어 해시만 f462로 바꾼다. f82c Hessian은 전처리기로만 남긴다. 각 반복에서 전체 gradient와 현재점 연산자 `H_k v`를 새로 계산하고, 이전 strict/inexact 방향이나 과거 HVP를 재사용하지 않는다.

모든 반복에 inexact 모드를 명시한다. 각 solve의 forcing 값은 **그 solve 현재점의 전체** `||g_k||∞`에서

`eta_k = clip(0.1 * ||g_k||∞, [1e-10, 1e-3])`

로 산출한다. PCG 요청과 현재점의 fresh `Hs`로 다시 계산한 true residual가 같은 eta 기준을 만족해야 한다. 이번 시작점은 상한 구간이라 `eta=1e-3`이다. 실제 gradient가 감소하면 매 solve eta도 다시 계산되며, 정상점에 가까워진 정책 동작은 다음 실행에서 별도로 확인한다.

true residual가 `||r||₂ <= eta ||g||₂`, `eta<1`이면 `Phi=||g||₂²/2`의 미소 하강을 뒷받침한다. 원래 J의 하강은 별도 조건이므로 현재 코드의 `gᵀs<0`, `gᵀHs<0`, 부동소수점 여유가 반영된 양의 관측 곡률 판정을 유지해야 한다. 모든 후보에서 실제 J Armijo와 Phi Armijo (`c1=1e-4` 각각), 현재점에서 계산한 endpoint 전체 branch/margin gate, 입력·source·runtime 무결성을 유지한다. root 후보의 재평가 순서도 보존하고 최종 정상성 기준은 `||g||∞ <= 1e-10` 그대로 둔다. eta 완화는 root·최종 곡률·수반·재분석 허용치를 바꾸지 않는다.

반복 범위는 최대 3 iterations, 총 90 HVP, 각 solve PCG 최대 40회, 내부 deadline 720초, 외부 wall 780초, 단일 guarded launch, sampled RSS 상한 1 GiB다. 수치는 새 plan/result/parent/resource에 선언하고 기록한다. 각 iteration마다 실제 eta와 true relative residual, PCG 및 모든 HVP 비용, 선택 step alpha/displacement, 실제 `ΔJ`/`ΔPhi`, max-gradient, 분기·margin 및 거부 이유를 남긴다. 실행 완료·예산 거부·수치 거부는 서로 구분한다. 3회 반복 한도 종료는 정상점 인증이 아니다.

## Phase 1 완료도와 현재 확장 연구

`PHASE1_COMPLETION_CHECKLIST.md`는 8개 P1 조건의 범위를 정의하며, 마지막 CI/병합/인계는 PR #159 기록으로 확인하도록 명시한다. 저장소 `graphify-out/REVIEW_INDEX.md`에는 해당 **정의된 조건 8/8, 100%**, PR #159 병합 완료를 별도로 기록한다. 그러므로 기존 Phase 1 완료 상태를 유지하는 근거는 있으며, 이는 정해진 전역 이동/성장 기반 이론 실증의 완료이지 최신 코드 무결함이나 현재 FV 확장 완료율의 의미가 아니다.

현재 FV/minmod 확장의 `약 70%`는 이번 외부 검토에 제시된 가중 마일스톤 합산이다. 저장소의 Phase 1 공식 체크리스트 완료율과 같은 공식 수치가 아니며, 남은 원래 3시간 문제 정상점·수반·재분석과 독립 예측 검증을 포함한 별도 추정으로 표기해야 한다. 실제 FV·미래 점수의 새 근거 없이 숫자를 갱신하지 않는다.

## 확인 근거

- `examples/weather_scenarios/fv_point_3h_committed_base.py`: 일반 committed endpoint pinning 및 종료 영수증 검증.
- `examples/weather_scenarios/fv_point_3h_dual_merit_continuation.py`: 현재 반복기, forcing 정책, HVP/시간 상한, dual Armijo/root audit.
- `examples/weather_scenarios/fv_point_3h_inexact_comparison.py`: comparison selector 및 shared counter 동작.
- `graphify-out/fv-root-cause-20260919/INEXACT_GREEN_FINAL_20261007.md`, `INEXACT_RED_FINAL_20261007.md`, `D31_INEXACT_RESULT_20261007.json`, `D31_INEXACT_PR255_INTEGRATION_20261007.json`: PR #255 독립 검토/통합 요약.
- `graphify-out/fv-root-cause-20260919/d31_inexact_comparison_20261007_attempt1/{step.json,strict.json,inexact.json,step.run.json,step.resource.json}`: 비교 raw 및 부모/자원 기록.
- `PHASE1_COMPLETION_CHECKLIST.md`, `graphify-out/REVIEW_INDEX.md`: 공식 Phase 1 조건 및 완료 상태의 근거.

## 제한

이 문서는 다음 반복의 설계 검토이며 loader 변경 승인·시험 통과·새 계산 완료를 의미하지 않는다. PR #255의 저장 기록은 기존 비교 실행을 재현성 있게 설명하지만, 독립 FV/HVP 재실행·매끄러운 분기 경로·전역 수렴·정상점·최종 수반/재분석·미래 예측 성능은 입증하지 않는다.
