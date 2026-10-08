# PR #260 face-transition diagnostic — GREEN final audit

**판정: 제한된 양측 진단은 정상 완료했고 보관 기록과 수치가 일치한다.** 이것은 선택한 `Qy[4,3]` 주변의 유한 표본 증거이며, face 단독 원인이나 양측 극한의 증명은 아니다. 이번 감사는 보관 결과를 읽고 독립 산술을 대조했으며 FV·HVP·PCG를 재실행하지 않았다.

## 계보·실행 종료

기준 제어는 PR260 모델-guided resume endpoint `2cdccade81a8…`이고, 이전 PR259 endpoint `69c4a794…`가 아니다. 계획 SHA-256은 `6b79d07c…`; 실행 raw JSON SHA-256 `ffa9473e…`가 `.run.json`의 child digest와 일치한다. Raw 기록은 완료 상태, 네 비영점 표본 모두 strict branch 통과, eta=0은 `J`만 평가했으며 gradient/branch 판정은 하지 않았다. HVP, PCG, optimizer step, score, response는 모두 0/미계산이다. 보관 `source_before`는 계획 source/archive와 계획 자체의 162개 경로 전체를 포함하고 pin digest가 맞으며 `source_before==source_after`, input identity 및 runtime도 전후 동일하다.

Guard 기록은 exit 0, 33.7396초, 300초 벽시계 한도 내 완료, 표본 최대 RSS 824,360,960 bytes/1 GiB 제한이다. RSS 값은 주기적으로 읽은 자식 프로세스 표본 최대치다. 내부 elapsed 30.8796초와 guard elapsed의 차이는 실행 경계 측정 차이로, 실행 결과를 바꾸는 불일치가 아니다.

## 재산술

| eta | 원래 전체 J | Phi=||g||²/2 | ||g||₂ | unit-normal g 성분 |
|---:|---:|---:|---:|---:|
| −2e−6 | 0.061347598829 | 0.012669366047 | 0.159181444 | −0.08170551 |
| −1e−6 | 0.061347493753 | 0.012659437405 | 0.159119059 | −0.08138621 |
| 0 (J only) | 0.061347389014 | — | — | — |
| +1e−6 | 0.061347597024 | 0.029450162098 | 0.242693890 | +0.20034863 |
| +2e−6 | 0.061347805319 | 0.029521848586 | 0.242989089 | +0.20062227 |

보관된 FP64 gradients에서 재계산한 inner `||g⁺−g⁻||₂=0.281738830542`, 공통 zero-face 법선에 수직인 성분 norm `0.001499246905`; 따라서 이 유한 jump의 제곱 norm 중 약 **99.9972%가 법선 성분**이다. Outer pair는 각각 `0.282343698592`, `0.002998494686`이고 약 **99.9887%가 법선 성분**이다. Outer tangent 성분이 inner의 약 두 배인 것은 더 넓은 표본 구간에서의 유한 차이이며, 극한 추정이나 오차 경계로 해석하지 않는다.

각 pair의 `DeltaPhi = 0.5 (g⁺−g⁻)·(g⁺+g⁻)`를 gradients로 독립 계산한 값은 저장된 실제 Phi 차 및 보고 identity와 일치한다: inner `0.0167907246923`, outer `0.0168524825383`. Full-gradient 선분 최소점은 inner `theta=0.288468024323`, `||g(theta)||₂=0.136797778407`; 그 음의 gradient 방향과 두 inner endpoint의 pairing은 각각 `−0.0187136321771`이다. Outer pair의 finite-segment 값은 `theta=0.288598683855`, norm `0.136744603587`, pairings `−0.0186990866102`다. 이는 조사 방향 후보일 뿐, 실행한 최적화 스텝이 아니다.

eta=0의 J와 양측 J 표본은 모두 `0.061347389…` 부근에서 차이가 O(|eta|)이고, chart eta에 대한 유한 secant는 음측 약 `−0.10474`에서 `−0.10491`, 양측 약 `+0.20801`에서 `+0.20815`다. 이는 J의 연속적인 경계 접근과 서로 다른 방향 기울기와 양립하지만, 유한 데이터만으로 경계 연속성/한쪽 미분을 증명하지 않는다. 반면 Phi는 inner cross-side에서 `+0.0167907`, outer에서 `+0.0168525` 차이가 있어 J가 가까워진다는 이유로 Phi도 연속이라 말할 수 없다.

## branch·donor 해석

모든 비영점 표본에서 3,600개 observer Euler stage를 기록했고, 상세 donor/raw/effective boundary trace 360개는 두 analysis interval에 한정된다. 양측 cross 비교에서 face sign 배열은 3,600 stage 각각 달라졌지만, 공간 face 좌표를 독립 비교하면 바뀐 면은 target `Qy[4,3]` 하나뿐이다. 각 same-side 비교의 face-sign 배열은 바뀌지 않았다. 따라서 전 시공간 face-sign 차이 개수 3,600은 **3,600개의 서로 다른 면 변화**를 뜻하지 않는다.

Limiter choice 서명은 inner cross pair에서 future stage 두 개(2473–2474), outer cross pair에서 여섯 개(1805–1806, 2097–2098, 2473–2474)가 바뀌었다. Analysis limiter choices는 비교 쌍 전부에서 바뀌지 않았다. 같은 부호 안에서도 미래 limiter 선택은 달라졌다. 다만 원래 J는 분석구간에만 의존하므로 이 미래 전이를 J gradient 차이의 원인으로 해석하지 않는다. 분석구간의 네 표본에서는 limiter 선택이 같고 target face만 부호가 달라져 단일 분석 전환면 가설을 강하게 지지한다. 유한 표본의 중간 경로와 정확한 일방 극한, 수반 가중 trace 기여는 확인하지 않았으므로 인과 인증은 아니다.

Negative `eta=−2e−6` 상세 analysis trace에서는 선택 face flux가 모든 360 stage에서 음수이고, 따라서 외부 positive-y face의 plus-side boundary donor가 선택된다. Interior reconstructed minus donor와 growth-adjusted boundary 사이 값은 `−0.9151228`에서 `+3.6875979`까지 변한다. 첫 stage raw top edge `36.7020`은 positive growth 뒤 effective edge `36.72444`가 되었고 boundary growth gap은 `0.02244185`; odd Euler stage에는 raw/effective가 같다. 이는 donor 방향, 경계 trace와 작은 양의 growth scale이 기록된 값과 일관됨을 보여준다. 내부 `q` 전체 격자를 JSON에 저장한 것은 아니므로 이 감사는 선택 면 donor 추적과 서명 자료 범위 안의 확인이다.

## 제한과 다음 판단

이 결과는 해당 고정 합성 사례의 원래 목적함수에서 비용·gradient merit·branch trace가 면 양쪽에서 어떻게 달라지는지 보여준다. Inner/outer gradient jump가 공통 법선에 거의 정렬된다는 것은 단일 연속 면 가설과 양립하지만, future limiter 선택 변화와 유한 표본 간격 때문에 selected face 단독 원인 또는 매끄러운 한쪽 극한의 증명은 아니다. Phi는 이 경계에서 크게 달라질 수 있으므로, face 횡단 정책을 설계할 때 비용 감소와 Phi 감소를 별도로 검사해야 한다. 물리적 무풍/강수 경계나 예측 정확도에 대한 결론은 이 실행 범위에 없다.
