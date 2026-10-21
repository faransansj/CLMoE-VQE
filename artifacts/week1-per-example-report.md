# Week-1 1-C 예별(per-example) 재채점 리포트

- 실행(UTC): 2026-10-16T11:49:04Z · 스크립트: `scripts/week1_per_example_rescore.py` · sha256 baf73c570bcc2e89…
- 총 런타임: 328.946초 (계측 단계: archive_preload_seconds=2.579, build_scorers_seconds=3.49, scoring_loop_seconds=312.062, rows_write_seconds=6.41)
- 상태: pass · 실패 레코드 0건
- 재채점 corpus 416개 · occurrence rows 212992개 (artifacts/week1-per-example-scores.jsonl.gz)

## 검증 (strict tolerance)

- per-image 평균 == fresh TaskScorer.score == 저장값: 최대 오차 max |fresh−stored| = 0.000e+00, 저장 복사본 간 최대 = 0.000e+00 (허용 1e-12/1e-9)
- PTB 토큰 캐시: 요청된 모든 caption corpus에서 배치 토큰과 라인 단위 일치, 불일치 0건
- G5 native 파일 ↔ G4 native 파일: 192/192개 파일 행 단위 동일
- scorer lock: 구현 SHA·문서빈도 SHA 포함 필드 전부 원 lock/런 구성과 일치: ['C', 'F', 'G', 'V']
- 메모: occurrence는 4 orders × 8 seeds × 512 eval images의 재평가 건수이며 서로 다른 이미지 수가 아님; 통계 검정·방법 선택·인과 주장 없음

## 불일치(own-route text mismatch) 정합

- VQA diagonal 21/21, caption final 41/41 (review 기록과 정합; 배치-1 스코프 62 = 21 + 41)
- caption diagonal 총 20 rows: 그중 16 rows는 F@stage3 셀로 caption final corpus와 동일 파일·동일 occurrence(역할 중복, 계산 1회 재사용), 나머지 4 rows는 62 범위 밖의 diagonal-stage 추가 불일치
- 고유 occurrence 합계(역할 중복 제외): 66

## 주요 집단 절대값 (occurrence pooled, 원 단위)

- V [accuracy_points]: diag native mean 32.764282, oracle 33.181152, independent 33.457031; own_rows 14137, mis_rows 2247; N−O own 0.0, mis -3.0396083667111706
- C [native_cider_d]: diag native mean 0.481254, oracle 0.516005, independent 0.516708; own_rows 14464, mis_rows 1920; N−O own 0.00011829618595297811, mis -0.2974348630066567
- G [accuracy_points]: diag native mean 36.169434, oracle 37.182617, independent 38.427734; own_rows 13989, mis_rows 2395; N−O own 0.02144542140253056, mis -7.056367432150314
- F [native_cider_d]: diag native mean 0.114033, oracle 0.101394, independent 0.099927; own_rows 14949, mis_rows 1435; N−O own 3.261241665951238e-05, mis 0.14396735992442086

## caption final (native vs task-ID oracle, 자기/타집단)

- C [native_cider_d]: own 12715 rows, other 3669 + cross 0; own-route text mismatch 14/41.셀별 own/other 절대값·paired Δ는 envelope의 caption_final_full_subgroups 참조
- F [native_cider_d]: own 13561 rows, other 2823 + cross 0; own-route text mismatch 27/41.셀별 own/other 절대값·paired Δ는 envelope의 caption_final_full_subgroups 참조

## Flickr later-task 16 cells (before vs after)

- per-image: wins 2173, losses 4584, ties 1435 (post−pre, occurrence pooled); corpus mean post−pre -0.028751
- task-ID oracle의 before-stage 예측은 저장 산출물에 없음(by construction) — 명시적 결측; family-restricted 사전행이 유일하게 저장된 사전-상태 caption-family 출력 (셀별 family_restricted_pre 전사/사후 전이·Δ는 envelope 참조)

## 한계

- rows are evaluation occurrences (4 orders x 8 seeds x 512 eval images per stage); the same image recurs per order and seed, so rows are not independent images
- the G5 task-ID oracle has no before-stage prediction files by construction, so the pre-Flickr paired oracle evidence is reported missing and never imputed; the saved caption-family restricted pre rows are the only stored pre-state caption-family outputs
- no significance test, no method selection, no causal claim; all deltas are descriptive
- original source hashes and scorer locks must match; the artifacts record the strict tolerances and maxima
- caption scoring uses the frozen train references IDF only; no evaluation-reference IDF enters the scorer path

