# PR #258 model-guided continuation — GREEN 실제 결과 검토

날짜: 2026-10-07
판정: **선언된 bounded-loop 목표의 3개 iteration을 수용한다.** 이는 Newton 단계도, 정상점 인증도 아니다.
범위: step raw, parent/resource receipt, pinned summary와 independent arithmetic checks를 읽기 전용으로 대조했다. FV/gradient/HVP/PCG/예보/수반을 재실행하지 않았다.

## 실행·입력 폐쇄

계획 SHA `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`에 111 source와 43 archive pin이 있고, 독립 SHA 대조에서 불일치는 없었다. Raw `step.json` SHA `92497bab…`는 parent child SHA와 summary `raw_sha256`에 일치한다. Parent/resource는 completed, exit 0, monitor/resource error·SIGTERM 없음이라고 기록한다. Guarded elapsed **82.1991821초**, sampled peak RSS **381,239,296 bytes**로 240/300초·1 GiB 계획 이내다. RSS는 0.25초 표본 최대값이지 OS hard limit 보증은 아니다.

Raw의 149-path source/archive before/after map은 같고, 현재 파일을 after map과 대조한 불일치는 없었다. 고정 input identity와 runtime before/after도 일치한다. 3 accepted steps, `optimizer_steps_applied=3`, `hvp_calls_started=completed=3`, PCG 0, no full Hessian/root/score/response/reanalysis claim으로 닫혔다.

## Fresh full-gradient/HVP와 alpha arithmetic

Iteration 0의 base control/gradient는 committed tangent endpoint `05889584…` 및 그 accepted fresh gradient와 배열 단위로 같다. Iteration 1·2의 base control hash와 full gradient 배열은 각각 바로 전 iteration의 accepted endpoint/control gradient와 같다. 각 점에서 `d=−g`; raw 방향의 최대 성분 잔차는 0이다. 코드 경로는 그 점의 full original-J gradient에 JVP를 한 번 적용해 `Hd`를 만들며, 각 iteration에 completed HVP 하나가 기록돼 있다.

저장된 `Hd`, `g`를 NumPy로 재산술해 `gᵀd=−||g||₂²`, `gᵀHd`를 확인했다. 각 `alphaPhi=−gᵀHd/||Hd||₂²`가 반경 alpha보다 작아 `alpha_start`와 표시 반올림까지 일치했다.

| 반복 | `g∞` 기준 | `gᵀd` | `gᵀHd` | `alphaPhi` / `alpha_start` | 실제 step norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.510128497 | −0.447080305 | −1303.428862 | 0.000243428500 / 0.000243428500 | 0.0001627662 |
| 2 | 0.165249192 | −0.125034388 | −55.0057825 | 0.001182945973 / 0.001182945973 | 0.0004182921 |
| 3 | 0.171123583 | −0.083827876 | −52.1683270 | 0.000359772446 / 0.000359772446 | 0.0001041651 |

각 actual endpoint는 두 Armijo 조건, strict own branch, 실제 `Phi` 감소 roundoff gate를 통과했다. 매 반복 첫 후보가 수용됐고 backtracking은 없었다. 이는 `alphaPhi` cap과 실제 merit acceptance가 각각 적용됐다는 receipt다. `alphaPhi`는 linearized-gradient 모델의 시작 길이 cap이지 actual `Phi` 최소점이나 수렴 보장은 아니다.

## 세 점의 실제 변화와 경계 해석

전체 시작점부터 종료점까지

| 양 | 시작 `05889584…` | 종료 `69c4a794…` | 변화 |
|---|---:|---:|---:|
| `J` | 0.061555713011 | 0.061369252443 | −0.3029135% |
| `Phi` | 0.223540152425 | 0.018131732819 | −91.8888251% |
| `||g||₂` | 0.668640640 | 0.190429687 | −71.5198755% |
| `||g||∞` | 0.510128497 | 0.097245034 | −80.9371493% |

중간 `g∞`는 `0.51013→0.16525→0.17112→0.09725`로 두 번째 step에서 잠시 상승했다. 세 endpoint 모두 각자 strict branch를 통과했으나 **분석 360-stage와 미래 3240-stage partition hash가 매 step 바뀌었다**. 요약 flag는 매 step analysis/future changed 모두 true다. Raw는 branch signature 전체 배열이 아닌 per-partition hashes를 보존하므로 변화의 구체 stage/cell은 이 receipt만으로 특정할 수 없다. 실제 endpoint merits는 그대로 검사해 수용했지만 이 경로를 한 smooth branch라고 해석하지 않는다.

Target face `Qy[3,2]`는 `−5.910612738e−7→+1.829860184e−5→+4.793660946e−5→+1.379475264e−5`로 첫 step에서 부호를 바꿨고, 이후 양수에 머물렀다. 이를 유지하거나 0으로 고정하지 않은 full-26 탐색이며, 매 endpoint에서 실제 objective/gradient/Phi와 strict branch를 다시 평가했다.

이번 model-guided loop의 실제 시작점은 PR258 tangent step이 확정한 `05889584…`다. 그 점의 gradient block norm은 field `0.39211420`, flow `0.18192217`, growth `0.51012850`이다. e29에서 PR258 tangent 보정을 시작할 당시의 `.70224/.04913/.25068`은 이전 기준점이며 이번 표의 시작값이 아니다. PR258 tangent 보정 뒤에 이미 flow와 growth gradient가 커진 상태에서 이번 full-space loop가 시작됐다. 이후 field/flow/growth gradient norm은 다음과 같다.

| 시점 | field | flow | growth |
|---|---:|---:|---:|
| PR258 accepted start `05889584…` | 0.39211420 | 0.18192217 | 0.51012850 |
| 1단계 | 0.34085 | 0.06810 | 0.06495 |
| 2단계 | 0.14119 | 0.18604 | 0.17112 |
| 3단계 | 0.15553 | 0.09875 | 0.04819 |

마지막 `g∞=0.0972450`은 기존 `1e−10` root gate보다 약 **9.72×10⁸배** 크다. 반복이 `max_iterations_completed`로 끝났으므로 root pending, stationarity certificate 또는 normal minimum으로 보고하지 않는다.

## Linearized model과 인과 범위

Independent checks는 각 step의 affine full-control displacement/hash, `alphaPhi`, `J/Phi` thresholds와 actual Armijo를 원시 값에서 재산술했다. HVP linearized-gradient relative errors는 **0.09877, 0.95663, 1.49917**이고, predicted-vs-actual gradient angles는 **5.506°, 61.767°, 78.228°**였다. 두 번째·세 번째 step에서 HVP 기반 gradient model은 actual gradient 방향을 잘 예측하지 못한다. Linearized-gradient `Phi` 예측도 actual과 각각 `.0648943/.0625172`, `.0299828/.0419139`, `.0325296/.0181317`로 벌어진다. Actual J reduction을 local quadratic J model reduction으로 나눈 비율은 **0.98265, 0.99467, 0.32209**라 세 번째 step에서 quadratic model이 실제 J 감소를 크게 과대예측했다. 이 차이는 보관 점 사이의 model-vs-actual 오차이며 AD 오류 추정은 아니다.

Actual candidate `J`와 actual full-gradient `Phi`는 모든 단계에서 thresholds 아래라 수용 자체는 명시된 merit policy에 따른다. 두 번째·세 번째의 큰 gradient prediction residual 및 세 번째 J model residual은 branch hashes가 step마다 바뀐 사실과 함께 봐야 한다. 따라서 세 actual decrease를 HVP linear model이 정확히 예측했다고 보거나 이를 한 smooth branch의 curvature 보증으로 확대하지 않는다.

이 실행은 PR258 tangent correction 이후 full-gradient `d=−g`, temporary face chart 해제, alphaPhi cap, dyadic dual-merit loop를 한 정책 안에서 함께 바꿨다. 따라서 세 accepted step의 이득을 alpha cap, face 해제, 방향 변경 중 하나의 단독 인과로 귀속하거나 tangent-step 정책 대비 속도 향상률로 말할 수 없다. 비교 baseline도 없다.

## 보존할 남은 범위

원래 unconstrained 3h eligible stationary point, current full-curvature certificate, original adjoint/VJP/reanalysis, independent synthetic future/finite-range verification은 여전히 남아 있다. 이번 3단계 continuation은 수치 탐색의 일부이며 FV 연구 완료율을 자동으로 올리지 않는다. Full-model physics의 기상학적 예보 skill도 검증하지 않았다.

## Evidence hashes

- Plan: `a75b5e474e68fb74475c3fd7fb2f31e85c891edcb06c40e83b4287b16a7b3136`
- Raw `step.json`: `92497babfe8eae3c55e2dcd4d244e533740b395f050e1a207612fcb78680cb0d`
- Parent `step.run.json`: `686a43f22088ca44025a29e7b52689848fe4ba37b1f3f14fae631bcf87a98f8a`
- Resource `step.resource.json`: `f1e1ca5e7329a76308dc4e6b58db6129ec5e157324f89fa64c8dee25359f8149`
- Result summary `AMODEL_RESULT_20261007.json`: `285c2f005865c1ae9e81cbb83a664a03f58cf20facfc52beb84b57ac3592fb16`
- Independent arithmetic `AMODEL_INDEPENDENT_CHECKS_20261007.json`: `2e3dfdf54675721611c23097fbde40d45158cd9985c7628cd4a8b779247d8d04`
- Archived model-vs-actual diagnostics `AMODEL_MODEL_DIAGNOSTICS_20261007.json`: `7502e22ee5424e780075d6c224e7c66b1040ac7abf1ff793e0a137b96cd17173`
