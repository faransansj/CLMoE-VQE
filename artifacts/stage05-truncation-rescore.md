# Stage 0.5 task F — 저장 Flickr 예측 사후 잘라내기(truncation) 재채점 리포트

- 실행(UTC): 2026-10-20T03:09:41Z · 스크립트: `scripts/stage05_truncation_rescore.py` · sha256 ea35410170d1cf09… · python 3.12.14 · pycocoevalcap 1.2 · numpy 1.26.4
- 실행 환경: Linux-7.2.8-x86_64-with-glibc2.42 · java: openjdk version "21.0.12.1" 2026-08-18 · local OpenJDK 21 (nixos) vs remote 22.0.2; PTBTokenizer -preserveLines -lowerCase assumed version-stable; the parity gates against stored values verify this empirically
- 총 런타임: 450.495초 (계측: load_rows_seconds=1.921, build_scorers_seconds=1.567)
- 상태: ok · 실패 레코드 0건

## 라벨 (바인딩)

- **모든 잘라내기 규칙은 eval 분할의 '저장된' 예측에 사후(post hoc) 적용됨 — 진단 목적 전용.**
- 방법/설정/스코어 선택·변경 아님; 디코딩 변경 아님; 저장 산출물 수정 없음(읽기 전용 입력).
- 규칙은 비-EOS 출력에만 적용(generation.eos == true 행은 어떤 규칙에서도 원문 그대로).
- 규칙은 텍스트를 삭제만 하고 문자를 삽입하지 않음(특히 '.'은 어떤 규칙 출력에도 등장 불가 — 검증 포함).

## 코호트 정의 (행 = 저장 예측 1건/eval 이미지 1장)

- g4_independent_F: kind=independent, task=F, stage=0 (8 seeds x 512 eval rows); no same-image old C exists (independent C is saved on the COCO eval split), so paired statistics are None for this cohort · 저장 corpus 근거: audit_g4 transfer_records independent_score (order-invariant per seed/task)
- flickr_last_cells: 16 cells = orders {V_C_G_F, V_G_C_F} x 8 seeds; F = saved post-stage task-ID-oracle F-expert outputs; old C = saved pre-stage native outputs (expert C), paired per (order, seed, id) · 저장 corpus 근거: oracle: routing_audit task_id_oracle matrix[3][3]; pre native: g4 results.json pre_task_scores[3]

## 사전선언 잘라내기 규칙 (공백 토큰 기준)

- none: untruncated baseline (parity anchor)
- R1: cut at start of first repeated token trigram (tokens[pos:pos+3] == tokens[pos+3:pos+6], the exact repeated_trigram definition of sema_multimodal/model.py); keep text before the repeat; unchanged if none
- R2: cut at earliest ', and' or ' and a ' marker with >= 8 words before it; keep text before the marker start, rstripped; unchanged if none
- R3_12: keep first 12 words (unchanged if <= 12)
- R3_16: keep first 16 words (unchanged if <= 16)
- R3_20: keep first 20 words (unchanged if <= 20)
- R4: R1 then R3 with N=20
- scope: applied ONLY to non-EOS outputs (generation.eos == true stays byte-identical); rules only delete text, never insert characters

## 검증 (parity)

- 비잘라내기(baseline) per-image 재계산 vs 저장 행 점수: 최대 오차 4.441e-16 (허용 1e-12)
- old C(pre-stage native C) per-image 재계산 vs 저장 행 점수: 최대 오차 4.441e-16 (허용 1e-12)
- 비잘라내기 corpus 재계산(TaskScorer.score) vs 저장 scalar: 최대 오차 2.776e-17 (허용 1e-09)
- g4_independent_F 셀 수 8 · 행 수 4096 · EOS 2536 / 비-EOS 1560 (EOS share 0.6191)
- flickr_last_cells 셀 수 16 · 행 수 8192 · EOS 5104 / 비-EOS 3088 (EOS share 0.6230) · both-EOS 행 5100

## P('.') 진단 — 저장 데이터 감사 (추론 없음)

- ind_F (F): 행 4096 · '.' 포함 0건 · ',' 포함 1494건 · 비-EOS 중 '.' 포함 0건
- pre_C (C): 행 8192 · '.' 포함 0건 · ',' 포함 10건 · 비-EOS 중 '.' 포함 0건
- orac_F (F): 행 8192 · '.' 포함 0건 · ',' 포함 2964건 · 비-EOS 중 '.' 포함 0건
- 잘라내기 규칙 출력 전부( 27888개 ) '.' 포함 0건 — 규칙은 삭제만 하므로 '.'은 도입될 수 없음(검증 완료)
- week1 per-example 파일 전체( 212992행 ): '.' 포함 총 0건 (C 0 / F 0 / V·G 0); 아카이브 전체 합계( G1 포함 )는 artifacts/stage05-target-format.json 참조(C·F 각 625,664행, '.' 0건)
- 해석 경계: 저장 데이터는 greedy 디코딩 산출뿐이므로 '디코딩이 '.'을 낼 수 있는가'는 저장 산출물상 P('.')=0으로만 말할 수 있음(샘플링 등 변형 디코딩 실험은 본 산출물 범위 밖; 추론 금지 레인)

## 코호트1 G4 independent F (8 seeds) — 규칙별 결과

- paired 비교(N/A): independent C는 COCO eval 분할로 이미지 집합이 disjoint → paired Δ/wins/both-EOS는 본 코호트에서 정의되지 않음(구조적 한계, 명시)

| 규칙 | pooled corpus | 셀 평균 | 잘라낸 행 | 빈 출력 | 비-EOS 평균 길이(전→후) |
|---|---|---|---|---|---|
| none | 0.099927 | 0.099927 | 0 | 0 | 56.44→56.44 |
| R1 | 0.101713 | 0.101713 | 418 | 0 | 56.44→51.84 |
| R2 | 0.151306 | 0.151306 | 1447 | 0 | 56.44→18.80 |
| R3_12 | 0.174647 | 0.174647 | 1560 | 0 | 56.44→12.00 |
| R3_16 | 0.164773 | 0.164773 | 1560 | 0 | 56.44→16.00 |
| R3_20 | 0.138910 | 0.138910 | 1560 | 0 | 56.44→20.00 |
| R4 | 0.138910 | 0.138910 | 1560 | 0 | 56.44→20.00 |


## 코호트2 Flickr-last 16 cells — 규칙별 결과


| 규칙 | pooled corpus | 셀 평균 | 잘라낸 행 | 빈 출력 | 비-EOS 평균 길이(전→후) |
|---|---|---|---|---|---|
| none | 0.102862 | 0.102862 | 0 | 0 | 56.26→56.26 |
| R1 | 0.104568 | 0.104568 | 838 | 0 | 56.26→51.39 |
| R2 | 0.152265 | 0.152265 | 2880 | 0 | 56.26→18.57 |
| R3_12 | 0.176309 | 0.176309 | 3088 | 0 | 56.26→12.00 |
| R3_16 | 0.166551 | 0.166551 | 3088 | 0 | 56.26→16.00 |
| R3_20 | 0.141632 | 0.141632 | 3088 | 0 | 56.26→20.00 |
| R4 | 0.141633 | 0.141633 | 3088 | 0 | 56.26→20.00 |

paired 비교(F − old C, per-image, occurrence pooled)

| 규칙 | Δ(all) | Δ(비-EOS) | Δ(EOS) | Δ(both-EOS) | 비-EOS wins/losses/ties | all wins/losses/ties |
|---|---|---|---|---|---|---|
| none | -0.054029 (8192) | -0.137582 (3088) | -0.003478 (5104) | -0.003463 (5100) | 59/3029/0 | 2530/5660/2 |
| R1 | -0.052323 (8192) | -0.133056 (3088) | -0.003478 (5104) | -0.003463 (5100) | 142/2946/0 | 2613/5577/2 |
| R2 | -0.004626 (8192) | -0.006523 (3088) | -0.003478 (5104) | -0.003463 (5100) | 1596/1492/0 | 4067/4123/2 |
| R3_12 | 0.019417 (8192) | 0.057261 (3088) | -0.003478 (5104) | -0.003463 (5100) | 2017/1071/0 | 4488/3702/2 |
| R3_16 | 0.009660 (8192) | 0.031376 (3088) | -0.003478 (5104) | -0.003463 (5100) | 1891/1197/0 | 4362/3828/2 |
| R3_20 | -0.015259 (8192) | -0.034730 (3088) | -0.003478 (5104) | -0.003463 (5100) | 1502/1586/0 | 3973/4217/2 |
| R4 | -0.015258 (8192) | -0.034728 (3088) | -0.003478 (5104) | -0.003463 (5100) | 1502/1586/0 | 3973/4217/2 |

- both-EOS 전용 비교는 '둘 다 정상 종료'인 행만으로 F와 old C를 직접 비교(잘라내기 무관).

## 사전선언 해석 가이드 (서술형)

- 잘라낸 F의 점수가 old C에 가까워지면(≈) 'F 전문가가 내용은 학습했으나 종료(stopping)만 깨진 상태'라는 판독이 뒷받침됨. 반대로 잘라내기 후에도 격차가 남으면 내용 손실 가능성이 남음. 본 산출물은 상관 서술만 하고 인과 주장/방법 선택을 하지 않음.
- 규칙은 eval 데이터에 사후 적용된 진단 도구이며, 실제 디코딩 정책 후보로 사용될 수 없음.

## 한계

- occurrence는 셀(512행)의 재평가 건수이며 서로 다른 이미지 수가 아님; 코호트2는 512 이미지 × 16 셀.
- R1은 model.py의 adjacent-trigram 정의를 그대로 텍스트 토큰에 적용한 것으로, 생성 토큰 id 기준 저장 repeated_trigram 플래그와 부분적으로만 일치할 수 있음(두 수치 모두 보고).
- 스크립트/입력 sha256과 실측 런타임은 envelope(JSON)에 기록됨.
