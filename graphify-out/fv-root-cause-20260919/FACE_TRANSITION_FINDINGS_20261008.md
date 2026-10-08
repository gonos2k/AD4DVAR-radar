# PR260 후속 공통 face-transition 진단

원래26변수 문제와 모든prior/관측/파라미터/경계/시간을 유지한 공통 face chart와 trace 진단을 구현했다. 특정 face를 영구 제약으로 추가하지 않았다. 기본실험은 PR260의 마지막2cdccade…에서 외부 Qy[4,3]의 유량 eta를 ±1e−6/±2e−6으로 복원하며, 다른25개 제어를 그대로 유지한다. Pivot은 basis/limits에서 계산한 가장 큰 유량 미분으로 선택한다. 원래 수치 연산 소스는 변경하지 않았다.

## 실제 진단 결과

| eta | J | Phi | 단위제어법선 gradient | 접선gradient norm |
|---|---:|---:|---:|---:|
| −2e−6 | .061347598829 | .012669366047 | −.08170551 | .13661238 |
| −1e−6 | .061347493753 | .012659437405 | −.08138621 | .13673024 |
| 0 비용만 | .061347389014 | 미계산 | 미계산 | 미계산 |
| +1e−6 | .061347597024 | .029450162098 | +.20034863 | .13696989 |
| +2e−6 | .061347805319 | .029521848586 | +.20062227 | .13709268 |

네 비영점 모두 자체 strict branch를 통과했다. 영점에서는 full J만 평가했고 AD/gradient/Phi/strictzero-face 인증은 하지 않았다. 실제생산flux의 영점 잔차는 반올림 수준이다. 비용은 고정 retained 좌표선의 영점에서 가장 낮게 표본화됐고 양쪽 제어법선 기울기는 면을 향한다. 그러나 접선잔차~.137이 크게 남아 있어 조건부면 정상점/최소점이 아니다.

Inner finite pair의 전체gradient차이 norm은 .28173883, 공통영점기하 법선에 직교하는 차이 norm은 .00149925다. 차이 제곱합의 **99.99717%**가 법선에 있다. Outer pair의 접선차이는 .00299849로 약두배이며, 표본간격을절반으로줄일때접선차이도약절반이된다. 이는 단일 연속전환면의gradient차이가법선방향이라는가설과 일치하는유한표본증거다. 정확일방극한이나 중간경로의분기를 인증한것은아니다.

Inner Phi차이 .01679072469는 .5(gplus−gminus)ᵀ(gplus+gminus)와 일치한다. 법선부분 .01675792902와접선부분 .00003279567의합이다. Raw gradient norm은 연속인 비용과 달리 양측에서 크게달라질수있다. 이사실만으로 원래문제에 최소점이없다거나 Hessian이부정부호라고 판단하지 않는다.

## 분석구간의 trace와 분기

분석360stage에서 실제Euler의 내부 재구성 donor와 유효 외부경계 donor를 기록했다. 양의growth 첫Euler는 생산 `_scale_by_growth`의addcmul/expm1 경계변환을사용하고, 두번째Euler는원래공급된경계값을사용한다. Observer는 torch.no_grad branch call에서만 연결했고 objective/gradient AD에는 연결하지 않았다. 공급원본 경계, 유효경계, 선택donor와상태차이를구분한다. 큰경계값 overflow fallback이 추가Euler호출을만드는경우는확인후거부한다.

내부minus donor−유효외부plus donor 차이는분석시간에따라약−.9151부터+3.6877까지변하며부호도바뀐다. Trace불일치는실제로관측됐지만 전체J의gradient점프에대한수반가중기여를계산한것은아니다.

네표본의 분석limiterchoices는같았다. Cross-side face sign이바뀐공간면은 Qy[4,3] 하나이고각3600stage에서그slot이바뀌었다. 미래limiterchoices는inner/outer cross pair에서2/6stage, 같은측pair에서각2stage 달라졌다. **미래전이는 분석-only J의gradient차이 원인이 아니다.** 분석자료는단일전환면 가설을강하게지지하지만 유한η간격/중간경로/일방극한/수반가중인과를확인하지않았으므로인증으로확대하지않는다. 미래donor detail은이번범위에포함하지않았고분기서명만별도로기록했다.

## 표본공통 비용하강 후보 — 미적용

두전체gradient의선분에서최소norm 점을계산했다. Inner pair theta=.28846802, 최소norm=.13679778이며 두표본의directional pairing은모두−.01871363218이다. Outer pair도비슷하다. 이는 두유한표본이함께지지하는원래비용의하강후보다. Newton방향이나 Phi최급강하, 정확Clarke정상성 검사가아니다. **해당방향을실제FV후보에적용하지않았다.** 양측계량접선 평균방향과도구분해 full26 convex-segment 공식을사용했다.

## 출처·자원·검증

단일guard33.739610초, sampledpeakRSS824,360,960bytes, exit0. 내부240초/외부300초/1GiB 한도이내이며 HVP/PCG/optimizer/score/response는0/미계산이다. 실제제어확정점2cd는이동하지않았다. 출력raw를손실없는gzip으로보존했고압축해제바이트SHA ffa9473e5c0852c8f525665d8e7ac4255ad19820aca9951ef6c137ea905798ce는부모childSHA와같다. 원시JSON이없는checkout에서gzip -dk face_transition_attempt1/diagnostic.json.gz로복원할수있다.

고정planSHA6b79d07ce609ebe35879dcb3f1cec95649a786155c392e7ea041063b4e19283e. 소스113개/보관54개와입력·환경·소스폐쇄가일치했다. 집중회귀22개통과/2.15초, type error-level0. Generic x/y face chart의생산flux/Jacobian, raw/effectiveboundary parity, 전체-gradient segmentminimizer와Phi항등식, zeroJ-only/fullprior경로를시험했다. 새FV이동/수반/재분석의통합시험과구분한다. Graphify는변경2파일의격리복사에서AST추출하고공유graph를보존했다.

Source단계오류와유한pair 방향의잘못된명칭은사전검토에서수정했다. 실제최종검토는GREEN/RED가원시벡터/서명/영수증/trace와대조했다. FACE_TRANSITION_ANALYZE_20261008.py는저장값산술만하고새FV/J/g/HVP를계산하지않는다.

완료범위는공통양측trace/gradient 진단이다. 원래3h정상점·전체곡률·수반/VJP/재분석·독립합성미래점수는열려있다. 기존1차8/8과FV외부가중70%판정은유지한다. 다음탐색은원래문제/최종기준을보존하면서표본공통방향과분기횡단·매끄러운정상성보정의역할을사전에구분한실제후보검증으로판단해야한다.
