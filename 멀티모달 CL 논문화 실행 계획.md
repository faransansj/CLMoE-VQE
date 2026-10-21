# 멀티모달 CL 논문화 실행 계획

Oct 16, 2026 · @Midori

## 개요

1번 분석 논문(약 3\~4주)을 먼저 확보하고, 진입 gate를 통과하면 2번 방법 논문(약 2.5\~3.5개월)으로 간다. 2번이 실패해도 1번 논문은 남는 구조다.

- **기준 문서:** `MULTIMODAL_RESEARCH_SUMMARY.md` (커밋 `c75c2ba` 기준 G1\~G5 완료)
- **불변 원칙:** 기존 confirmation eval은 방법 선택에 다시 쓰지 않는다. 학습·추론·재채점은 항목별 승인 후에만 실행한다.
- **최우선 리스크:** Flickr CIDEr-D normalizer ≈ 0.153(×1)로 COCO ≈ 0.680 대비 4배 이상 낮다. 데이터 버그라면 Flickr 관련 결론과 G4 전체가 영향을 받는다.

## 1번 — 분석 논문

목표는 확장·동결 CL의 보존 효과와 한계를 보고하는 분석 논문 초고다. Week 1은 검증, Week 2\~3은 작성이다.

### 핵심 주장 (초안)

1. 확장·동결은 capacity-matched shared보다 망각이 적고 최종 성능이 높다.
2. 그러나 task당 64장 replay를 이기지 못한다(caption 유의 열세, VQA 비유의 열세).
3. 현 구조에는 task 간 파라미터 전이 경로가 없고, 이전 지식 활용은 routing 재사용뿐이다.
4. Routing 병목은 family 구분이 아니라 같은 family 안의 expert 선택이며, task-ID oracle이 최선이 아닌 경우가 있다.

### Week 1 — 검증

| ID | 작업 | 실행 성격 | 승인 |
| --- | --- | --- | --- |
| 1-A | Flickr sanity check | 기존 데이터·예측 조회, 소규모 CPU 채점 | 조회는 불필요, 채점은 필요 |
| 1-B | index-0 O=I 확인 | 기존 artifact 비교만 | 불필요 |
| 1-C | Phase 0 per-example 재채점 | CPU scorer 재실행 | 필요 |
| 1-D | 문장 불일치 62행 batch size 1 재생성 | GPU 소규모 추론 | 필요(선택) |

**1-A. Flickr sanity check (최우선)**

- Reference 파싱: 이미지당 ref 수(정상 약 5), 길이 분포, 리스트-문자열 파싱 오류 여부
- CSV 내부 split이 의도한 split·이미지 수와 일치하는지
- 랜덤 50개 예측 vs reference 육안 비교
- Frozen base와 independent의 Flickr/COCO CIDEr 비율이 BLIP zero-shot 경향과 크게 어긋나는지
- Train-IDF 구성 방식이 COCO와 동일한지
- **Gate:** 버그 발견 시 수정 후 재실행 범위를 별도 결정. 모든 order에 Flickr가 있으므로 최악의 경우 G4 전체(약 120h GPU) 재실행.

**1-B. index-0 O=I 확인**

- Stream index 0 task는 init seed와 shuffle seed가 independent와 같으므로 oracle 점수 = independent 점수여야 한다.
- Caption oracle이 stream position별로 정확히 같은 값으로 묶이는 관찰(COCO index 1·2, Flickr index 0·3)과 함께 보고.
- 불일치 시 GPU 비결정성으로 기록하고 K 해석에 반영.

**1-C. Phase 0 per-example 재채점**

- Own/other expert 그룹의 절대 점수와 paired Δ (현재 미식별 항목)
- Flickr 학습 전후 expert별 승패 샘플
- Caption 길이·EOS·반복과 점수의 관계
- 결과는 기존 결론을 덮어쓰지 않고 별도 artifact로 저장

**1-D. (선택) 문장 불일치 재생성**

- 같은 expert인데 native/oracle 문장이 다른 VQA 21 + caption 41행을 batch size 1로 재생성해 batched padding 원인 여부 확인

### Week 2\~3 — 작성

- 구성: Setup → 주 결과 → 망각·gain·transfer 분해 → routing 진단 → 비용 → 한계
- 핵심 figure: 방법별 normalized final, T=R+K 분해, COCO/Flickr native vs oracle 부호 반전, routing 혼동 행렬
- 투고처: CL·PEFT 워크숍 또는 CoLLAs급 (마감은 별도 조회)

### 2번 진입 gate

- [ ] 1-A에서 Flickr 데이터 버그 없음 (또는 수정·재검증 완료)
- [ ] 1-C에서 기존 expert가 새 Flickr expert보다 일관되게 나은 샘플군 확인

* 미충족 시: 재사용(A) 방향을 접고 Flickr expert 학습 문제(과적합, LR, epoch)를 먼저 조사

## 2번 — 방법 논문

목표는 선택적 재사용·확장이 replay와의 격차를 줄이는지(또는 replay-free 조건에서 경쟁력 있는지) 새 held-out split에서 확인하는 것이다. 주 방향은 A(재사용·확장 선택), C(expansion + replay)는 필수 비교군이다.

### 2-0. 사전 계획서 (1주)

- 주장 고정. SEMA(분포 변화 기반 확장)와의 차별점을 문장으로 명시. 후보: task identity가 아닌 생성 품질 기반 재사용/확장 선택, heterogeneous 생성 task 환경.
- 주 비교 축소: 제안 vs common replay, 제안 vs expansion × 2 family = 4개. 10 seeds면 최소 raw p = 2/1024, Holm 4개 최소 보정 p ≈ 0.0078.
- Replay-free 포지셔닝 시 non-inferiority margin과 검정력 사전 지정.

### 2-1. Phase 1 개발 (1\~2주, 약 12h GPU)

Seed 17, validation split만 사용한다.

| 조건 | 목적 |
| --- | --- |
| 현재 expansion | 기준 |
| Validation으로 add vs reuse 선택 | 가장 단순한 재사용 |
| Best 기존 expert로 새 expert 초기화 | 파라미터 전이 경로 추가 |
| Family shared + task private | 공유·보존 균형 |
| Common replay | 넘어야 할 기준 |
| Expansion + replay | Hybrid 비교 |

통과 기준:

- [ ] Flickr 후속 gain ≥ 0
- [ ] Caption·balanced score가 현 expansion보다 높음
- [ ] Common replay와의 격차 감소
- [ ] 평균 AF 증가가 사전 한도 이내

### 2-2. Baseline 구현 (Phase 1과 병렬, 3\~4주)

- O-LoRA, InfLoRA, SEMA를 생성형 BLIP + Q/V LoRA 구조에 이식
- 각각 소규모 run으로 정상 동작 확인 후 lock

### 2-3. 새 held-out split (2\~3일)

- 기존 eval 이미지 제외, identity/byte-hash 중복 재검사
- 기존 confirmation eval은 방법 선택에 사용 금지

### 2-4. Phase 2 confirmation (2\~3주, 약 200h GPU)

- 방법: 제안, expansion, common replay, hybrid, shared, capacity-matched, O-LoRA, InfLoRA, SEMA, independent
- 규모: 10 seeds × 4 orders × 약 9개 방법 ≈ 360 runs
- 처음부터 기록: GPU 시간, 추론 latency, routing overhead, per-example score
- 데이터·코드·비교 목록을 실행 전 lock

### 2-5. 작성 (2\~3주)

- 1번 논문의 확장판으로 갈지 별도 논문으로 갈지는 Phase 2 결과로 결정

## 일정·compute 요약

총 약 3.5\~4.5개월, GPU 약 210h(재실행 제외)다. GPU 시간은 기존 4-task run 평균 약 2,000초 기준 추정이며 SLURM 대기는 제외했다.

| 단계 | 기간 | GPU 추정 | 승인 필요 |
| --- | --- | --- | --- |
| 1-A, 1-B 검증 | 2\~3일 | 0 | 1-A 채점만 |
| 1-C 재채점, 1-D 재생성 | 2\~3일 | 1-D 소량 | 예 |
| 1번 작성 | 2주 | 0 | — |
| 2-0 계획서 | 1주 | 0 | 계획 확정 |
| 2-1 Phase 1 | 1\~2주 | \~12h | 예 |
| 2-2 Baseline 구현 | 3\~4주 (병렬) | 소량 | 예 |
| 2-3 Held-out split | 2\~3일 | 0 | 예 |
| 2-4 Phase 2 | 2\~3주 | \~200h | 예 |
| 2-5 작성 | 2\~3주 | 0 | — |

- Flickr 버그로 G4를 재실행하게 되면 약 120h GPU와 1\~2주가 추가된다.
- Main venue로 확장(더 긴 stream, LLaVA급 backbone)은 이 계획 범위 밖이며, DC 할당량 확인 후 별도로 결정한다.
