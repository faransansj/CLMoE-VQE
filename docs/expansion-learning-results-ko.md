# Expansion 기반 멀티모달 Continual Learning 실험 결과 보고서

## 1. 요약

**Task별 expert 확장·동결은 기존 지식 보존에 효과적이었고, shared 및 동일 용량의 단일 adapter보다 최종 성능이 높았다. 그러나 replay 대비 우위는 확보하지 못했으며, Flickr30k에서 신규 task 학습 실패가 나타났다.**

후속 진단에서는 Flickr 학습 타깃의 마지막 마침표를 제거했을 때 종료(EOS) 실패와 caption 점수가 개선됐다. 다만 이는 3개 seed의 단일 task validation 실험이다. 수정된 조건에서 전체 continual-learning stream을 재학습한 결과는 본 보고서에 포함되지 않으므로, replay 대비 결론이 뒤집혔다고 해석할 수 없다.

본 문서는 저장소에 기록된 **G4/G5 및 Flickr 후속 진단 결과**를 정리한다. 최근 수행한 VQAv2 평가 코드 속도 최적화는 별개의 보조 작업이므로 제외한다. 원격 서버에서 현재 실행 중인 작업을 확인한 보고서는 아니다.

## 2. 연구 질문과 방법

### 연구 질문

1. Expert를 확장하고 이전 expert를 동결하면 망각이 줄고 최종 성능이 개선되는가?
2. 그 이점이 동일 용량의 단일 adapter 및 작은 replay buffer와 비교해도 유지되는가?
3. 기존 지식 보존뿐 아니라 새 task 학습과 긍정적 전이도 나타나는가?
4. 이러한 결과가 VQA와 captioning을 함께 다루는 이질적인 task stream에서도 성립하는가?

### 평가한 expansion 방식

- Backbone: 고정된 BLIP.
- 적용 위치: 질문 encoder와 답변 decoder의 Q/V LoRA.
- 새 task마다 rank-8 LoRA expert를 하나 추가하고, 이전 expert와 backbone은 동결한다.
- 새 expert는 zero-delta(`B=0`)로 시작한다. 이전 expert의 파라미터를 새 expert로 전달하는 경로는 없다.
- 추론 시 정답 task ID를 제공하지 않는다. Frozen image/text 특징에 기반한 ridge router가 expert 하나를 선택한다.
- Router는 task당 128개 특징 벡터를 저장하므로 memory-free 방법은 아니다.

**범위 구분:** 여기서 평가한 것은 *task마다 expert를 추가하는 방식*이다. 생성 품질을 기준으로 기존 expert 재사용과 신규 확장 중 하나를 자동 선택하는 방식은 후속 연구 후보이며, 아래 결과로 검증된 방법이 아니다.

## 3. 실험 구성

| 항목 | 설정 |
|---|---|
| Task | VQAv2, GQA, COCO2014 caption, Flickr30k |
| Task family | VQA 2개, caption 2개 |
| Task별 이미지 수 | Train 2,048 / validation 256 / eval 512 |
| 순서 | V→C→G→F, F→G→C→V, V→G→C→F, F→C→G→V |
| Confirmation seed 수 | 8개 |
| 공통 학습 조건 | LR 3e-4, 5 epochs, effective batch 16 |
| 생성 조건 | Greedy decoding; VQA 최대 32, caption 최대 64 new tokens |
| G4 | 비교군 전체 224 runs, 예측 1,409,024개 재채점 |
| G5 | Expert bank를 고정한 routing 진단 160조건, 예측 1,081,344개 재채점 |

V/C/G/F는 각각 VQAv2/COCO/GQA/Flickr를 의미한다. 위 규모는 각 데이터셋의 전체 benchmark 크기가 아니라 실험에 사용한 부분집합이다.

### 주요 비교군

- **Shared LoRA:** 하나의 rank-8 adapter를 순차 학습.
- **Capacity-matched shared:** Expansion의 총 adapter 용량에 맞춘 rank-32 단일 adapter.
- **Common replay:** 과거 task당 64장의 이미지를 사용하는 replay.
- **Expansion:** Task별 expert 추가·동결과 router 선택.
- **Tuned replay:** 개발 단계에서 선택한 replay 설정. 10 epochs로, 공통 비교군보다 학습 예산이 두 배다.

## 4. 최종 성능

### 지표 해석

각 task 점수를 G1의 독립 학습 validation 점수로 나눈 뒤, 같은 family 안에서 평균한다. 아래 값은 **정규화 점수이며 정확도 퍼센트가 아니다.** 기준점 대비 비율이므로 1을 넘을 수 있다. Balanced는 두 family의 정규화 점수 평균이다.

| 방법 | VQA | Caption | Balanced |
|---|---:|---:|---:|
| Frozen-first | 0.414 | 0.191 | 0.303 |
| Shared LoRA | 0.701 | 0.365 | 0.533 |
| Capacity-matched shared | 0.775 | 0.650 | 0.713 |
| **Expansion** | **0.973** | **0.743** | **0.858** |
| Common replay | 0.993 | 0.943 | 0.968 |
| Tuned replay — 10 epochs | 1.069 | 1.062 | 1.066 |

### 비교 결과

| 비교 | VQA 차이 | Caption 차이 | 판정 |
|---|---:|---:|---|
| Expansion − Shared | +0.272 | +0.378 | 두 family 모두 유의하게 우세 |
| Expansion − Capacity-matched | +0.198 | +0.093 | 두 family 모두 유의하게 우세 |
| Expansion − Common replay | −0.020 | −0.200 | VQA는 유의하지 않음; caption은 유의하게 열세 |

차이는 표에 표시한 반올림 평균을 기준으로 제시했다.

- 통계 단위는 **8개 seed**이며, 각 seed에서 4개 task 순서를 먼저 평균했다.
- 주 비교 6개에는 exact two-sided sign-flip 검정과 Holm 보정을 적용했다.
- Shared·capacity 대비 네 비교의 Holm 보정 p는 모두 0.046875였다.
- Common replay 대비 VQA의 exact-test p는 0.125였다. 유의하지 않다는 것은 동등함을 입증했다는 뜻이 아니다.
- Tuned replay 대비로는 두 family 모두 열세였지만, 학습 예산이 다르므로 공통 예산 비교와 구분해야 한다.

**판정:** Expansion의 이점은 단순한 총 adapter 용량 증가만으로 설명되지 않는다. 그러나 replay를 포함한 사전 선언 종합 성공 기준은 충족하지 못했다. 확장·동결·routing 각각의 독립적인 인과 효과를 분리해 입증한 결과는 아니다.

## 5. 망각과 신규 학습

### 5.1 이전 지식 보존

| 평균 망각 — 정규화 단위 | Shared | Expansion |
|---|---:|---:|
| VQA | 0.437 | **0.037** |
| Caption | 0.639 | **0.009** |

Expansion의 평균 망각은 shared보다 크게 낮았다. 이전 expert를 보존하는 구조가 낮은 망각과 연결된 결과다. 다만 낮은 망각만으로 전체 방법의 우수성을 판단할 수는 없다. 새 task 학습과 최종 성능을 함께 봐야 한다.

### 5.2 새 task 학습

- **VQAv2·GQA·COCO:** 측정한 후속 task 조건 모두에서 학습 후 성능이 개선됐다.
- **Flickr30k:** 후속 task로 학습한 **16개 조건 모두에서 pre→post gain이 음수**였다.

따라서 모든 task에서 신규 학습이 성공했다고 주장할 수 없다. 특히 family 평균은 Flickr의 실패를 가릴 수 있으므로 task별 결과가 중요하다.

### 5.3 긍정적 전이

학습 직후 성능을 같은 seed의 독립 학습과 비교한 정규화 차이는 VQA −0.043, caption +0.020이었다. 이 결과는 **일반적인 긍정적 전이를 지지하지 않는다.** 이전 expert가 보존된다는 사실과 새 expert가 이전 지식을 학습에 활용한다는 주장은 구분해야 한다.

## 6. Routing 진단

Expert bank를 고정한 G5 결과는 다음과 같다.

- 다른 family로 보내는 비율은 약 **0.02%**로 매우 낮았다.
- 오routing 약 **23.9%**는 대부분 같은 family 내부의 다른 task expert 선택이었다.
- Family mask를 적용해도 성능 변화가 없어, 단순 VQA/caption 구분은 주요 병목이 아니었다.
- 정답 task expert를 강제로 선택하는 oracle은 VQA 점수를 **0.973 → 1.015**로 높였지만, caption은 **0.743 → 0.711**로 낮췄다.
- Caption의 역전은 Flickr에서 자기 expert보다 COCO expert를 사용하는 편이 나았던 현상과 관련됐다.

이 결과만으로 품질 기반 expert 재사용의 우수성을 주장할 수는 없다. 후속 진단에서 Flickr expert의 종료 실패가 드러났기 때문에, 먼저 해당 문제를 수정한 조건에서 다시 확인해야 한다.

## 7. Flickr 실패 원인과 후속 진단

### 관찰

Flickr로 학습한 expert는 캡션을 적절히 끝내지 못하고 생성 길이 한도에 도달하는 경우가 많았다. 학습 타깃의 마지막 마침표와 EOS 생성 실패 사이의 연관성을 조사했다.

### 단일 요인 대조

기존 경로를 재현하는 fidelity 검증 후, 학습 타깃 끝의 공백과 **마지막 마침표 하나만 제거**했다. Scoring reference는 변경하지 않았다. 3개 seed의 독립 Flickr 학습을 validation에서 비교한 결과다.

| Seed | 기존 CIDEr-D | 마침표 제거 CIDEr-D | 개선량 |
|---|---:|---:|---:|
| 137 | 0.112 | 0.274 | +0.162 |
| 149 | 0.105 | 0.260 | +0.155 |
| 163 | 0.107 | 0.306 | +0.199 |
| **평균 — 반올림** | **0.108** | **0.280** | **약 +0.172** |

- 평균 EOS 도달률: 약 **0.56 → 1.00**.
- 수정 후 평균 생성 길이: 약 11.4 tokens.
- 수정 후에는 Flickr expert가 비교한 COCO expert보다 Flickr 점수가 높았다.

### 해석과 한계

종료 실패가 Flickr 성능 저하에 기여했다는 근거다. 그러나 마침표와 EOS 사이의 세부 메커니즘까지 완전히 입증한 것은 아니다.

또한 이는 **단일 task·3 seed·validation 조건**의 진단이다. 전체 CL stream에서 다음이 어떻게 달라지는지는 아직 이 결과로 알 수 없다.

- Flickr의 신규 학습 gain.
- Flickr normalizer와 caption family 정규화 점수.
- Native router와 task-ID oracle의 상대 성능.
- Expansion과 replay 사이의 성능 격차.

특히 기존 Flickr normalizer가 종료 실패의 영향을 받았을 가능성이 있어, 수정 전후의 정규화 점수를 기준 변경 없이 직접 비교해서는 안 된다.

## 8. 연구 질문별 최종 판정

| 연구 질문 | 현재 판정 | 근거 |
|---|---|---|
| Expansion이 망각을 줄이는가? | **지지** | Shared 대비 매우 낮은 평균 망각 |
| Shared보다 최종 성능이 높은가? | **지지** | VQA·caption 모두 유의하게 우세 |
| 동일 용량 단일 adapter보다 좋은가? | **지지** | 두 family에서 capacity-matched 대비 우세 |
| 작은 replay보다 좋은가? | **미지지** | VQA 우위 없음, caption은 유의하게 열세 |
| 모든 새 task를 잘 학습하는가? | **미지지** | Flickr 후속 학습 16/16 조건에서 음의 gain |
| 일반적 긍정적 전이가 있는가? | **미입증** | 독립 학습 대비 일관된 개선 없음 |
| 종료 문제를 수정하면 replay를 이기는가? | **미검증** | 전체 stream 재검증 결과 필요 |
| 품질 기반 자동 재사용/확장이 유효한가? | **미검증** | 현재 결과는 task별 고정 확장 방식의 결과 |

**현재 주장 가능한 결론:** Task별 expert 확장·동결은 shared 및 동일 용량의 단일 adapter보다 보존과 최종 성능에 이점을 보였다. 하지만 replay 대비 일반적 우위와 모든 task에서의 신규 학습 성공은 입증하지 못했다. Flickr 종료 문제 수정 후의 전체 stream 재검증이 필요하다.

## 9. 재검증 시 확정할 항목

저장소의 P0 사전등록 초안에는 주 비교군, 1차 지표, paired 단위, 검정·임계값, 평가 split, L40S 실행 범위, 결과 노출 및 manifest 확인 등이 미해결로 기록돼 있다.

특히 기존 G4는 `eval`, P0 runner는 `validation`을 사용하므로 개발용 진단과 confirmatory 평가를 구분해야 한다. 결과를 본 뒤 비교 조건을 선택하지 않도록 평가 계약을 먼저 확정해야 한다. 본 문서는 추가 학습의 실행 승인이나 완료 보고가 아니다.

## 10. 근거 자료

- [연구 개요 및 G4/G5 결과](../RESEARCH_OVERVIEW.md): 실험 구성, 최종 성능, 통계, 망각, 전이, routing, Flickr 진단.
- [멀티모달 연구 상세 정리](../MULTIMODAL_RESEARCH_SUMMARY.md).
- [멀티모달 실험 기록](../MULTIMODAL_EXPERIMENTS.md).
- [Stage 1 period-strip 결과](../artifacts/stage1-period-strip.json).
- [Stage 1 fidelity gate](../artifacts/stage1-fidelity.json).
- [P0 사전등록 초안](../preregistration-p0-strip.md).

본 보고서는 기존 기록의 요약이며 새로운 학습·통계 분석 결과를 추가하지 않는다.
