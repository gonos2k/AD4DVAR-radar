# 비용 탐색 이후의 제한된 보정 주기

- [x] GREEN/RED 설계: 기존 dual-merit 전체 공간 kernel 재사용, PR262 출처 분리.
- [x] 공유 구조 graph와 최근 변경 AST 캐시 확인.
- [x] 기존 producer source/test 보존 및 별도 mode loader.
- [x] 회귀·타입·격리 Graphify·GREEN/RED preflight.
- [x] 새 계획 아래 단일 240초 내부/300초 외부/1GiB 실행, 최대 3보정/3HVP.
- [x] 탐색 전 2cd와 보정 시작 26e 대비 J/Phi/잔차 평가.
- [x] 최종 GREEN/RED 검토와 KG 기록.
- [ ] PR·CI·정확 head 병합 (별도 통합 영수증에서 완료 기록).

하나의 주기는 이미 확정한 PR262 J-only 탐색과 새점 26e에서 최대 3회의 dual-merit 보정을 연결한다. 탐색 재실행은 없다. 새점에서 -g 방향과 현재 HVP로 alphaPhi를 계산한다. 실제 J/Phi/후보 strict branch와 fresh endpoint 및 input/source/runtime/deadline gate를 유지한다. 회복은 원래 2cd J/Phi보다 수치적으로 유의하게 낮은 경우이며, 최종 정상성이나 수반 인증은 아니다. 회복/정상성 도달(감사 대기)/예산/정체/방향·분기 거부/반복 한도를 구분한다.

결과: 2보정 후 회복 gate 조기 종료. 탐색 전 J −0.058559%, Phi −14.657167%. 최대gradient는0.11651로 탐색 전보다 커 정상점은 미완료. GREEN/RED raw audit에서 불일치 없음.
