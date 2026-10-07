# Qy[3,2] 양측 진단 — GREEN 실제 결과 검토

날짜: 2026-10-07  
판정: **계획 범위의 진단 실행과 결과 폐쇄를 수용한다.**  
범위: 이미 보관된 단일 실행 raw, parent/resource receipt, pinned 분석 요약에 대한 읽기 전용 재산술 및 해석 점검. FV/gradient/HVP/PCG/예측/수반/재분석을 다시 실행하지 않았다.

## 실행·출처 폐쇄

계획 `QY32_TWO_SIDED_DIAGNOSTIC_PLAN_20261007.json`의 SHA-256은 `bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f`다. 단일 parent receipt는 child exit 0, `completed`, `finite_chart_samples`, child SHA `d05e5b8e…`를 기록하고, 이 SHA는 `diagnostic.json`의 실제 SHA와 일치한다. 자원 receipt는 34.1226555초, sampled peak RSS 768,950,272 bytes, 300초/1 GiB 한도 이내, 종료 신호·monitor error·resource termination이 없다고 기록한다. RSS는 0.25초 간격 표본의 최대값이며 OS의 hard allocation limit 증명은 아니다.

Raw 결과는 e29 제어 SHA `e29c348d…`, 계획 SHA와 파라미터 SHA `8871db49…`를 보존한다. 134개 source/archive 경로에 대해 raw의 before/after 해시가 같고, 현재 파일을 raw의 after map과 대조해도 불일치가 없었다. 고정 입력 identity와 runtime before/after도 각각 같으며 `fixed_input_unchanged=true`다. parent child hash, raw hash, `QY32_RESULT_20261007.json`의 raw hash가 모두 일치한다. 저장된 optimizer/HVP/PCG 수는 각각 0이다.

이 실행은 목표에 맞게 한 번만 수행됐다. 비용이 약 31초였다는 child elapsed 기록과 약 34초인 guarded wall 기록은 서로 다른 구간을 측정한다.

## 실제 표본

| `eta=Qy[3,2]` | 생산 유량 재계산 | 원래 full `J` | `Phi=||g||²/2` | `||g||₂` | `||g||∞` | branch/margin |
|---:|---:|---:|---:|---:|---:|---|
| `−2e−6` | `−1.999999999995e−6` | 0.06169931635639 | 0.27839846477788 | 0.746188267 | 0.413477111 | strict pass, 3600/3600 margin stages |
| `−1e−6` | `−1.000000000001e−6` | 0.06169928232841 | 0.27896576582137 | 0.746948145 | 0.413534565 | strict pass, 3600/3600 margin stages |
| `+1e−6` | `+1.000000000001e−6` | 0.06169929522253 | 0.28120835277499 | 0.749944468 | 0.413649900 | strict pass, 3600/3600 margin stages |
| `+2e−6` | `+2.000000000002e−6` | 0.06169934229707 | 0.28188838444171 | 0.750850697 | 0.413707781 | strict pass, 3600/3600 margin stages |

`eta=0`은 생산 면 연산 결과 `Qy[3,2]=6.94e−18`, 해석식 결과 `8.67e−19`로 FP64 반올림 수준에 있다. full objective는 **0.06169924932275067**이며 `gradient_computed=false`; 그 점에는 `Phi`나 branch strict 자격을 기록하지 않았다. 영유량의 strict 부호 인증은 수행하지 않았다. `J`의 영점 표본은 네 비영 표본 모두보다 낮지만, 이는 고정 좌표에서 뽑은 다섯 개 점의 비교일 뿐 정확한 국소 최소점 증명은 아니다.

## 비용, merit, 그리고 원인 범위

독립적으로 raw에서 재산술한 `J` 유한구간 변화는 다음과 같다.

| 구간 | `ΔJ` (오른쪽 − 왼쪽) | `ΔPhi` | `J` 구간 secant |
|---|---:|---:|---:|
| `−2e−6 → −1e−6` | `−3.40279834e−8` | `+5.67301043e−4` | `−0.03402798` |
| `−1e−6 → +1e−6` | `+1.28941211e−8` | `+2.24258695e−3` | `+0.00644706` |
| `+1e−6 → +2e−6` | `+4.70745438e−8` | `+6.80031667e−4` | `+0.04707454` |

따라서 raw `Phi`는 네 비영 표본에서 `eta`가 커질수록 증가했다. 그러나 `Phi(eta=0)`는 평가하지 않았다. 이 표본들만으로 `Phi`가 정확히 영유량 경계에서 불연속이라고 말할 수 없다. `J`는 영점 표본까지는 왼쪽에서 감소하고 영점에서 오른쪽 첫 점보다 낮으며, 두 표본 오른쪽에서는 증가한다. 이 모양은 고정 retained-coordinate chart에서 영점 부근 비용 골짜기를 시사하지만, 한쪽 도함수나 경계 최소를 증명하지 않는다.

분기 기록은 관찰된 비용 경쟁의 범위를 좁힌다.

- 모든 표본이 pointwise strict branch 검사와 완전한 3600-stage margin 수집을 통과했다.
- 저장된 분석 360 stage에서 limiter `choices` 변화는 같은 쪽/양쪽의 세 쌍 모두 0이다. 미래 3240 stage에서는 `−2e−6→−1e−6`에 2개 stage(2849, 2850), `−1e−6→+1e−6`에 2개(1841, 1842), `+1e−6→+2e−6`에 4개(1839, 1840, 3143, 3144)가 바뀌었다. 따라서 유량 부호가 고정된 한쪽 안에서도 미래 limiter 선택이 일부 달라진다.
- 양쪽을 가로지르는 비교에서 `face_signs`는 3600개 stage 모두 달랐다. 전체 배열의 원소별 대조에서는 바뀐 면이 매 stage의 `Qy[3,2]` 하나뿐이었다. 정적 면유량 sign 대조에서도 `Qy[3,2]` 이외의 `Qx/Qy` 면 변화는 0개였다.

이 근거는 지정 좌표에서 target face의 부호가 바뀌고 전체 수송 경로가 반응했음을 보여준다. 동시에 미래 limiter의 추가 변화도 있으므로 `J`·gradient·`Phi` 차이를 오직 그 한 면의 효과로 분리하지는 못한다. 요약의 무변화 analysis limiter, 같은 쪽의 미래 choice 변화, 모든 stage의 target face sign 변화를 함께 보존해야 한다.

## 좌표 방향과 기하학적 법선

`geometry` 필드를 raw에서 다시 읽어 보면 접선 gradient norm은 QR과 명시 직교투영 두 방식에서 점별로 정확히 일치하며 약 **0.74618–0.74943**이다. 접선 잔차가 여전히 크므로 face 위의 조건부 정상점으로 해석할 근거가 없다.

고정 `t` pivot chart 미분 `j_eta_fixed_t`와 단위 유클리드 법선 미분은 다음처럼 다르다.

| `eta` | `j_eta_fixed_t = gᵀD_eta Gamma` | `nᵀg/||n||₂` | `nᵀg/(nᵀn)` |
|---:|---:|---:|---:|
| `−2e−6` | −0.03453916 | +0.00397214 | +0.00774781 |
| `−1e−6` | −0.03351682 | +0.00449580 | +0.00876923 |
| `+1e−6` | +0.04648716 | +0.04551330 | +0.08877533 |
| `+2e−6` | +0.04766194 | +0.04611714 | +0.08995314 |

`j_eta_fixed_t`는 pivot만 움직이는 transverse chart 경로에서의 변화율이다. 유클리드 unit-normal 값은 양쪽 모든 표본에서 양수다. 따라서 chart secant/미분의 부호가 왼쪽과 오른쪽에서 바뀌는 현상을 **기하학적 법선방향의 최소 조건**으로 설명할 수 없다. chart 경로에는 큰 접선 성분도 포함된다. `intrinsic_normal_flux_slope` 역시 unit-normal 미분과 다른 `Qstar` 단위당 최소노름 법선 기울기이며 별도 필드로 유지해야 한다.

## 종합 해석과 남은 한계

진단은 “dual merit 충돌이 실제 Newton 후보에서 관찰됐다”는 점을 뒷받침한다. PR #256의 두 큰 거부 후보는 비용 `J`를 낮추면서 `Phi`를 크게 올렸지만, 그 후보들은 많은 좌표를 함께 움직였고 단일 면 양측 시험이 아니었다. 이번 고정 `t` 진단은 `J`가 영점 표본에서 가장 낮고 raw `Phi`가 비영점 표본에서 양의 방향으로 증가함을 보였으며, strict branch 조건에서 선택한 양측 점들을 얻었다. 그러나 이 유한 간격 결과는 정확한 한쪽 극한, 영점에서의 gradient, `Phi` 연속성, 단일면 인과, Clarke 정상성, global minimum을 증명하지 않는다. 다른 미래 limiter branch 변경과 큰 접선 잔차도 남는다.

이 실행은 원래 문제의 적격 정상점, 현재점 full curvature, 원래 수반/재분석 또는 독립 미래 예측 점수를 종료하지 않았다. 따라서 기존 연구 완성도 평가는 바뀌지 않으며, 이 결과를 실제 바람 장벽·무강수 면·예보 skill의 증거로 해석하지 않는다.

## 증거 식별자

- Plan SHA: `bea4c3611e39d070a0234e9ed80b1a345e132926f97f63e325ba2c9cd53aa08f`
- Raw `diagnostic.json` SHA: `d05e5b8ed86f9067b4d29d5f9b9bf7ccd4ea7c23d8157a94410c0d59914d15c3`
- Lossless `diagnostic.json.gz`: 298,604 bytes; decompressed 49,600,039 bytes match the locally retained raw byte-for-byte; archive SHA `1b81004e56e31e051221ddebd6af6bbab8f8809f1eb888796cc1cb37476fea3e`.
- Parent `diagnostic.run.json` SHA: `f2ca48b68d3dd22b3bf124473b733b9564de9cc405870bd1a6b1f709021e3e32`
- Resource `diagnostic.resource.json` SHA: `32f39d1f0954d2bb3e535d975fad90714c238a04890029cb91daba354a03ea13`
- Pinned post-run arithmetic summary `QY32_RESULT_20261007.json` SHA: `348dd001f3ae1ef11876b1b7d5882edeba5918e8dd86fbe06404d9a62a340d1c`
- 분석 코드 `QY32_ANALYZE_20261007.py`는 실행 계획 source pin `9b79d426…`과 일치한다. 본 검토자의 raw 산술은 해당 요약 스크립트와 별도로 수행했다.

Lossless compression was independently checked after this review: decompressing the stored gzip bytes yields the same raw SHA, and the original local JSON remains unchanged. The evidence can therefore be retained compactly without losing the full per-stage signatures.
