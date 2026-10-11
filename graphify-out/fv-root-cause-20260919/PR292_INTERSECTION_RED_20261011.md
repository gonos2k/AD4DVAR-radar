# PR292 교차점/정상성 RED 검토

`gn_policy_experiment_20261011_attempt1`의 저장된 raw 후보 영수증을 읽기 전용으로 검토했다. FV/HVP/seed 재실행은 하지 않았다.

- 기준점은 fec9이며 저장된 canonical 점은 `68a6e821…11f9ed8`이다. 가장 가까운 dyadic 쌍은 reject 21 → accept 22, seeded 쌍은 reject 5 → accept 6이다. 각 쌍은 같은 base와 방향을 쓰며 accept α는 reject의 절반이다.
- 0-based trace index 179, x축 choice 0, 내부 마스크 `[1][2]`(전체 배열 `q[2,3]`)에서 양쪽 side trace 모두 `choose_left: true → false`로 바뀐다. `left_sign=right_sign=slope_sign=−1`은 유지된다. 음수 slope 규칙은 `fv_point_3h_nonsmooth_coupled_probe.py:177–203`에서 `left > right`를 비교하므로, 이 selector 전환은 두 slope가 음수인 채 `left=right` 경계를 지나는 것과 일치한다. 저장 후보쌍에서 달라진 `choose_left` 비트는 이 selector 하나뿐이다. 각 후보의 ± trace 해시는 서로 같다. accept trace는 base signature와 일치하고 reject trace는 일치하지 않는다.
- 두 reject 모두 J-Armijo, finite, branch-pair, face gate는 통과하고 `F_squared_armijo_passed`만 실패한다. 두 accept에서는 해당 R gate도 통과한다. index 179의 저장 target-face flux는 dyadic `−0.0 → 6.938893903907228e−18`, seeded `−6.938893903907228e−18 → −0.0`이다. 이는 거의 0인 target flux 값이지 minmod의 `left`/`right` 입력은 아니다.
- 영수증에는 부호, `slope_sign`, `choose_left`, trace hash, trajectory 전체의 최소 margin만 저장되어 있다. 각 event에서의 scalar minmod `left`/`right`나 `ζ=left−right`는 없다. 따라서 selector 전환과 R gate 통과의 동시 관측은 뒷받침하지만, 정확한 `ζ=0`, R 실패의 단일 원인, event normal 또는 gradient jump를 입증하지 않는다. `nonfinite_or_tie`는 false다.

두 slope가 음수일 때 minmod는 `max(left,right)`를 고른다. 고립된 `ζ=0` tie에서 값은 연속이고, 한쪽 미분은 `∇right`, 다른 쪽은 `∇left`가 되어 차 `∇ζ`는 limiter switching surface의 법선이다. 후속 진단은 동일 제어점 `c_event`에서 `Q(c_event)=0`과 `ζ(c_event)=0`을 모두 확인해야 한다. `n=∇Q(c_event)`, `w=P∇ζ(c_event)`를 외부면 위 두 사건 법선으로 계산하고, 두 법선의 독립성과 네 영역의 실제 도달 가능성을 검증한다. 외부면의 두 연장과 limiter 양측 조합에서 얻는 네 지역 gradient 및 실제 목적함수/잔차 비용은 모두 같은 `c_event`로 접근해야 한다. 서로 다른 저장 후보점의 gradient를 섞어서는 안 된다. 법선 독립성이나 네 영역 도달성이 확인되지 않으면 네 영역 Clarke 검사를 주장하지 않는다.

국소 Lipschitz piecewise-smooth 목적함수에서 도달 가능한 gradient들의 convex hull에 0이 포함된다는 것은 Clarke 정지성 인증이다(제약이 있으면 해당 tangent/normal 조건 포함). 이는 국소 최소 인증과 다르며, 현재 영수증은 그 정지성이나 최소점을 입증하지 않는다.
