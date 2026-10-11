# 원래 FV 역문제와 선택면 보정의 구분

- [x] 주 목표를 20분 원래 J의 적격 국소최소점으로 문서화하고 README 연결.
- [x] Q=0은 영구 물리 제약이 아닌 활성면 가설; 제한 탐색·해제 판단과 현재 구현 공백 명시.
- [x] beta/gradient-mixture vartheta, J/E, fixed boundary와 선택한 해 가지의 민감도 범위 고정.
- [x] P2/Armijo/회복/예산/정상성 후보/최소조건을 분리하고 strict 지원거부와 물리적 정의역 구분.
- [x] 생성 면유량0.104를 AST소스 결합 독립 정적 재산술로 확인; 새 FV/AD/최적화0.
- [x] 이전186 source/259 archive 핀 불변, 문서링크16개·static checker type0, 범위별 Graphify.
- [x] GREEN/RED 사실·수학 최종 대조.
- [ ] 후속 PR/CI 및 reviewed-head 통합은 별도 integration receipt로 종료 기록.

문서 고정은 solver 완성이 아니다. 다중면·퇴화점의 일반 자격, 자동 면 해제, 적격 원래 해·정확 FV응답·실제 섭동 재분석·현재점 미래점수는 미완료다. 70%는 기존 가중 마일스톤 진행표이며 원래 문제를 푼 비율이 아니다. 현재 canonical68 유지.
