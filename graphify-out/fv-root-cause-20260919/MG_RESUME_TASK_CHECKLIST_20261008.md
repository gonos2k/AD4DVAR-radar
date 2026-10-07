# 같은 전체 공간 kernel 재개

- [x] PR259 수치·외부 경계 부호 변화 재산술 및 마지막69c4 확정점 확인.
- [x] 이전 producer/test 바이트를 이전 plan/raw hash와 대조하여 불변 스냅샷 저장.
- [x] 입력 adapter만 최소 확장; 수학 helper와 전체 반복 loop AST 동일성 확인.
- [x] 재개 계보·스냅샷 변조·마지막점 일치 회귀시험, 타입, 격리 AST.
- [x] 소스·계획 고정 및 GREEN/RED 사전 검토.
- [x] 240초 내부/300초 외부/1GiB/최대3스텝·3HVP 단일 실행.
- [x] 잔차·배율·블록·분기·모형 정확도와 최종 GREEN/RED 기록.

새 실행은69c4에서 새g/Hd를 구하며 이전 방향/HVP/누적횟수를 재사용하지 않는다. 원래J와모든prior/관측/파라미터/경계/시간 및 최종1e−10 기준을 유지한다. 이전 실행을 새소스로 재인증하지 않는다. source exception은 원본바이트가 보존된 producer/test 두경로뿐이며 실제수치operator sources는 동일해야 한다.
