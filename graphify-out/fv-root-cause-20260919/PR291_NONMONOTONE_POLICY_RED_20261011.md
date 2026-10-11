# PR291 저장 후보 감사와 bounded-nonmonotone R 정책 — RED

읽기 전용 분석. `gn_committed_continuation_20261011_attempt1/step.json.gz`만 읽었고 FV/HVP를 실행하지 않았다.

## PR290의 실제 거부 후보

- 두 닫힌 점의 후보는 17+19개이며, 34개가 거부되고 2개가 첫 통과 후 P2 확정됐다. 34개 모두 실제 이동량이 양수·유한·반경 0.05 이하였고, 최소 θ, J Armijo, strict pair/face, side-gradient 유한성 및 native 비용 정합성은 통과했다. 거부 34개 전부 R Armijo 실패; point_index 1의 후보 8 하나는 R 실패에 더해 side-objective/native 불일치도 있었다. θ/pair/face/finite/radius 거부는 없었다.
- 각 point의 마지막 거부와 바로 다음 첫 채택은 각각 후보 16→17 및 18→19다. 후보 16/18은 J Armijo 통과·R Armijo 실패, 후보 17/19는 J/R 모두 통과한다. point 0의 16→17은 R `0.0031283297091940165→0.0031262214393219237`; point 1의 18→19는 `0.0031283713653851263→0.00312620012952993`이다.
- 기준점과 가까운 거부의 selector 변화는 양측 모두 zero-based `choices[179][0].choose_left[1][2]`(x-axis mask) 한 항목뿐이다. 거부에서는 true, 다음 채택에서 false이며, producer의 `choices[124]` 이벤트 스냅샷은 양측 모두 `(-1,-1,-1,false)`로 유지된다. 점별 base·첫 통과 인덱스는 raw `closed_steps[0/1]`; reducer/계산식은 frozen search loop `fv_point_3h_gn_committed_continuation.py:414-444`, 정책은 `:23-30`을 참조한다.

## 제안된 bounded nonmonotone 정책의 수학적 한계

새 시작 R을 고정 기준 `R0`로 두고 매 candidate의 실제 R에 `R ≤ 1.01 R0` 상한을 적용한다면, PR290의 저장된 마지막 거부 R들은 사후 산술상 각각 현재 fec9 기준보다 약 0.06812%, 0.06945% 높아 이 상한 안에 든다. 두 값은 이전점/방향에서 계산된 역사적 진단 후보일 뿐, fec9에서 재평가된 신규 후보로 간주하면 안 된다. 후보 8은 이 기준보다 24.38% 높고 native-side gate도 실패한다.

1% 상한은 R의 허용 증가며 Armijo가 아니다. R이 제곱 norm이므로 1% 증가는 residual norm 약 0.4988% 증가까지 허용한다. 또한 horizon 3에서 `R_final < R0`만 요구하면 `1.009 R0 → 1.005 R0 → 0.999 R0` 경로도 최종 회복으로 분류된다. 그 경로는 끝에서만 R이 개선되고 중간 하강이나 수렴을 보장하지 않는다. 상한은 매 단계의 직전 R에 곱해 누적하지 말고 고정 R0에 비교해야 한다. 최종 R이 R0 이상이면 상태를 provisional/noncanonical로 끝내고 표준 하강이나 수렴 주장에 승격하지 않아야 한다.

R Armijo를 면제하더라도 원래 native J Armijo, valid minimum θ, 양측 strict pair/face, 유한 gradient·full-G, current input/source/runtime 확인, 점별 독립 P2·atomic last-confirmed prefix는 계속 필수다. R Armijo 결과는 false로 그대로 기록하고 “bounded-R accepted”를 별도 기준으로 표시해야 한다. 후보 timeout 또는 P2 실패 뒤 마지막 확인 prefix만 유지하며, 세 점의 net recovery만으로 최적점·root·예측 성능을 주장할 수 없다. 1%·horizon3 정책은 제안이며 아직 승인된 변경으로 취급하지 않는다.
