# GN 정책 실험 RED 최종 대조 (2026-10-11)

읽기 전용 대조 대상: `GN_POLICY_EXPERIMENT_PLAN_20261011.json`, raw/gzip 실행 기록, 3개 arm의 후보·P2 영수증, 실행·자원 영수증, 메트릭·검증·아카이브 기록.

- 고정 계획 SHA `8ab812c1…fa44ed`; 182개 소스와 254개 아카이브 핀이 모두 일치한다. tar.gz의 SHA는 `6df4d4d3…f503`이며 61개 manifest 항목의 파일 해시·크기와 tar 멤버가 모두 일치한다. step gzip을 풀면 raw JSON과 바이트 단위로 같고, raw step SHA는 run 기록의 child SHA(`55c5876a…aaad3`)와 일치한다.
- 실행은 exit 0, 303.28초, 표본 peak RSS 809,205,760 bytes였고 외부 종료 없이 제한(660초/2GiB) 안에서 끝났다. 연산 수는 row 72, solve 3, HVP 6이다. 총 45개 논리 슬롯은 실제 평가 시작·완료 44회와 동일점/방향/α 캐시 재사용 1회로 맞는다. canonical 승격은 1회, provisional P2 점은 5개이며 각 P2 영수증 해시가 검증된다.
- 단조 `dyadic_JR`는 22회, `seeded_JR`는 6회의 한 점 검색으로 P2 회복했다. 승자는 endpoint `F_squared`, J, arm 순서 기준으로 `seeded_JR`다. 이는 같은 초기 연산·한 점 검색에서 후보 평가가 72.73% 줄었다는 관측이지, 전체 solver의 wall-time 가속률이나 수렴 비교가 아니다. arm 시간은 실행 순서와 캐시에 좌우된다.
- bounded arm은 시작 `F_squared=0.00312620012952993`의 고정 1% 상한 `0.0031574621308252293` 아래에서 3개 provisional P2 점을 닫았다. 0·1점에서 `F_squared_armijo_passed=false`, `R_armijo_required=false`가 그대로 기록됐다. 마지막 `F_squared=0.003151863429253204`는 상한 안이지만 시작값보다 높아 원래 R 회복에는 실패했으며 canonical으로 승격되지 않았다.
- 승격점의 J 개선은 `6.59e−10`, `F_squared` 개선은 `2.66e−9`, 실제 이동량은 `1.51085563e−8`로 매우 작다. 최종 자격·미래점수 재분석은 없고 postcommit readiness도 수행되지 않아, 이 기록은 수렴·최적성·예측성능 근거가 아니다.
- 이 실험은 고정 계획·현재 단일 profile 및 기존 producer/backend 경로에서 제한된 세 검색 규칙을 비교한 것이다. 1%/3점 정책은 실험 가정이며 일반 solver나 보편적 R 허용 규칙으로 확장하지 않는다. 승리 경로는 코드에서 P2 영수증을 재해시하고 정책을 재검사한 뒤 최종 재평가했다. 나머지 전체 기록은 별도 사후 번들 대조로 검증했다.

구현 대조 지점은 `examples/weather_scenarios/fv_point_3h_gn_policy_experiment.py`의 `endpoint_recovered`, `choose_endpoint`, `read_bound_endpoint`, `persist_canonical`이다.
