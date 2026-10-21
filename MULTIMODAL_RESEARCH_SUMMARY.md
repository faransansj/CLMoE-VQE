# 멀티모달 Continual Learning 연구 정리

작성일: 2026-10-16  
대상: `research/multimodal-cl`, 실험 완료 커밋 `c75c2ba`  
상태: G1~G5 완료. 향후 계획은 제안이며 추가 실험·방법 변경은 실행하지 않음.

> **핵심 결론:** Expert 확장·동결은 shared adapter보다 망각을 줄이고 최종 성능을 높였지만, replay보다 우수한 CL 방법이라는 주장은 성립하지 않았다.

이 문서는 처음 검증하려던 주장, 실제 관찰 결과, 아직 검증되지 않은 원인 가설을 구분한다.

## 1. 연구의 목적과 주장

### 1.1 연구 목적

상위 목표는 **MoE식 expert 확장·동결이 continual learning에서 의미 있는가**를 검증하는 것이다. 먼저 [VQA 연구](VQA_CLOSURE.md)에서 확장·보존·routing 결합의 효과와 한계를 확인했고, 이번에는 그 검증을 **heterogeneous 멀티모달 환경(VQA + captioning)**으로 확장했다. 이는 모든 MoE의 우수성이나 각 구성 요소의 독립적인 인과효과를 검증했다는 뜻은 아니다.

그 상위 목표 아래 **이번 단계의 사전 선언 성공 기준**은 새 task 학습과 기존 성능 보존을 유지하면서 shared·capacity-matched·replay 대비 최종 성능 우위를 보이는 것이었다. Replay보다 좋은 최종 성능은 상위 연구 목적 전체가 아니라 이번 confirmation에서 통과 여부를 판정한 기준으로 남긴다.

이번 연구에서 멀티모달 확장은 이미지·텍스트에 audio/video를 추가한다는 뜻이 아니다. **VQA와 captioning이라는 서로 다른 이미지–텍스트 task family를 하나의 모델에서 연속 학습하는 것**이다. 기존 VQA 연구의 readout expert 구조를 heterogeneous task 환경으로 확장했다.

### 1.2 검증한 방법

공유 BLIP backbone 위에 LoRA expert를 두고 다음 방식으로 학습한다.

1. 새 task가 도입되면 rank-8 expert를 추가한다.
2. Backbone과 기존 expert는 동결한다.
3. 새 expert만 현재 task 데이터로 학습한다.
4. 추론 시 frozen 이미지·텍스트 특징을 사용하는 joint ridge router가 expert 하나를 선택한다.

이는 **task마다 무조건 expert를 추가하는 구조**다. Adaptive growth는 이번 실험에 포함하지 않았다. 새 expert는 zero-delta로 초기화하며, 기존 expert를 복사해 시작하지 않는다.

학습 시 task 경계는 제공하지만, 일반 추론 시 정답 task ID나 dataset ID는 제공하지 않는다. 다만 caption instruction과 질문 형태가 입력에 있으므로 **task family 자체를 완전히 모르는 추론 환경은 아니다.**

### 1.3 연구 가설

| 가설 | 검증할 내용 |
|---|---|
| H1. 보존·최종 성능 | Expansion이 shared adapter, capacity-matched adapter, replay보다 최종 성능을 높이는가? |
| H2. 신규 학습 | 낮은 망각이 단순히 새 task를 배우지 않은 결과는 아닌가? |
| H3. Positive transfer | 사전 선언 지표는 CL 학습 직후 점수와 independent 점수의 차이다. 현 구조에서는 파라미터 전이의 직접 검정이 아니라 routing을 통한 이전 expert 재사용과 학습 난수 경로 차이가 섞인 진단으로 해석한다. |
| H4. Routing 효과 | 성능이 단순한 VQA/caption family 식별 때문인가, 같은 family 안의 task 구분도 중요한가? |

**코드 확인에 따른 H3 해석:** 새 expert는 `A=Kaiming`, `B=0`이므로 초기 출력 delta가 0이고, 이전 expert 복사 없이 추가된다. Backbone과 이전 expert는 동결되며 현재 expert만 현재 task 데이터로 학습한다. Optimizer는 phase마다 새로 만들고 BLIP dropout은 eval mode로 비활성화한다. Router는 phase 학습 뒤 적합하며 expert loss/gradient에 들어가지 않는다. 따라서 **이전 CL task의 학습된 파라미터가 새 expert로 전달되는 경로는 없고, 이전 task 지식을 활용하는 경로는 추론 시 router가 이전 expert를 선택하는 것**이다. 공통 pretrained backbone의 지식은 CL 이전부터 양 방법에 동일하게 존재하므로 task 간 학습 전이와 구분한다.

Independent와 데이터·학습 예산·목적함수·rank는 같지만 **학습 실행 자체가 bitwise 동일하지는 않다.** Stream 위치 `index`에서 expansion의 초기화 seed는 `seed + index×10007`, independent는 `seed`다. Batch shuffle도 `seed + index×1000 + epoch`여서 단일-task independent의 `index=0`과 다르다. Caption reference cycling은 seed/image/epoch 기반으로 동일하다. 따라서 independent 점수와의 차이 전체를 routing 또는 parameter transfer로 귀속하면 안 된다. 근거: `sema_vqa/readout_model.py:26–36,115–163`, `sema_multimodal/run.py:61–78,389–425,471–477`; 실행 당시 source hash와의 일치도 재집계 스크립트에서 확인했다.

다음 개념을 구분한다.

- 기존 지식 보존 ≠ 새로운 task로의 positive transfer
- 망각 감소 ≠ 최종 성능 우위
- Expert/task 식별 정확도 ≠ 생성 답변의 품질

## 2. 관련 연구와 현재 연구의 위치

### 2.1 직접 관련된 연구

| 연구 | 핵심 내용 | 현재 연구와의 관계 |
|---|---|---|
| [SEMA: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning](https://arxiv.org/abs/2403.18886) | 분포 변화를 감지해 adapter를 재사용하거나 추가하고 mixture routing으로 조합 | 확장·보존이라는 출발점은 유사하지만, 현재 방법은 task마다 무조건 확장하며 SEMA의 adaptive expansion을 그대로 구현한 것이 아님 |
| [SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://arxiv.org/abs/2411.13949) | 시각 이해와 instruction following의 망각을 구분하고 separable LoRA routing 적용 | 멀티모달 CL에서 LoRA mixture와 routing을 사용한다는 점에서 가까운 선행 연구. 현재 joint ridge router와는 구조가 다름 |
| [CoLoR: Continual Learning with Low Rank Adaptation](https://arxiv.org/abs/2311.17601) | LoRA를 continual learning에 적용해 domain-incremental 환경의 성능과 효율 검증 | LoRA 기반 CL 자체가 이미 연구된 방향임을 보여줌 |
| [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486) | 작은 과거 데이터 memory를 현재 데이터와 함께 학습하는 단순 replay의 강력함을 검증 | Shared baseline만 이기는 것으로 충분하지 않으며 replay가 중요한 비교군인 이유 |

추가로 검증한 직접 관련 연구는 다음과 같다. 저자·연도·발표처 및 링크는 원문/공식 proceedings에서 확인했으며, 아래 비교는 성능 재현이 아닌 **구조 비교**다.

| 연구·서지 | 현재 방법과의 구조적 차이 |
|---|---|
| Aljundi, Chakravarty, Tuytelaars. **[Expert Gate: Lifelong Learning with a Network of Experts](https://openaccess.thecvf.com/content_cvpr_2017/papers/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.pdf)**. CVPR 2017. | Task별 expert + task ID 없는 gate 선택이라는 점에서 가장 가까운 고전이다. Expert Gate는 task별 전체 네트워크와 autoencoder 재구성 오차 gate를 사용하고, 관련 이전 expert의 fine-tuning/LwF로 새 expert를 만든다. 현재 방법은 공유·동결 BLIP의 zero-delta LoRA만 추가하고, 저장 train feature의 판별적 ridge gate로 top-1 선택하며 이전 expert 가중치를 이식하지 않는다. |
| Yu et al. **[Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters_CVPR_2024_paper.html)**. CVPR 2024. | `MoE-Adapters4CL`은 코드 저장소 이름이다. CLIP의 layer 내부 expert pool을 task별 router가 top-2 혼합하고 activate/freeze 및 DDAS 배정을 사용한다. 현재 방법은 task별 독립 LoRA bank가 증가하며 layer별 혼합 없이 하나를 hard top-1로 선택한다. |
| Rusu et al. **[Progressive Neural Networks](https://arxiv.org/abs/1606.04671)**. arXiv:1606.04671, 2016. | 이전 column을 동결하고 새 전체 column에 lateral connection으로 이전 표현을 전달한다. 현재 방법에는 expert 간 lateral connection이 없고, 새 LoRA 학습으로 이전 expert의 표현을 주입하지 않는다. 확인되지 않은 학회 발표처는 기재하지 않는다. |
| Xiao Wang et al. **[Orthogonal Subspace Learning for Language Model Continual Learning](https://aclanthology.org/2023.findings-emnlp.715/)** (O-LoRA). Findings of EMNLP 2023. | Task별 LoRA 부분공간에 직교 제약을 적용하고 업데이트를 병합해 task ID 없는 단일 모델로 추론한다. 현재 방법은 직교 제약/병합 없이 expert를 분리 보존하고 명시적 router로 선택한다. |
| Yan-Shuo Liang, Wu-Jun Li. **[InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html)**. CVPR 2024. | 간섭을 줄이는 학습 부분공간을 설계해 LoRA의 일부 행렬만 학습하고 추론 가중치를 병합한다. 현재 방법은 Q/V의 LoRA 양 행렬을 학습하고 별도 expert 선택을 사용한다. InfLoRA의 분류 CL과 현재 생성형 VQA/caption 평가도 다르다. |
| Cheng Chen et al. **[CoIN: A Benchmark of Continual Instruction Tuning for Multimodel Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6a45500d9eda640deed90d8a62742be5-Abstract-Datasets_and_Benchmarks_Track.html)**. NeurIPS 2024 Datasets and Benchmarks; [arXiv:2403.08350](https://arxiv.org/abs/2403.08350). | CoIN은 멀티모달 instruction CL benchmark이며 instruction following과 지식 보존을 구분한다. 동반 MoELoRA는 layer 내부의 고정 expert 혼합이고, 현재 방법은 task별 expert 추가·동결과 별도 ridge top-1 routing이다. 제목은 공식 proceedings 표기를 유지한다. |

### 2.2 기반 기술

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): 사전학습 가중치를 동결하고 저랭크 업데이트만 학습하는 parameter-efficient adaptation.
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086): 이미지–텍스트 이해와 생성 task에 대응하는 vision-language pretraining 모델. 이번 연구의 고정 backbone.

### 2.3 현재 연구의 차별성과 한계

현재 연구에서 의미 있는 부분은 새로운 LoRA/MoE 구조 자체보다는 다음의 통제된 검증이다.

- VQA와 captioning을 동일 backbone·학습 경로에서 비교.
- 총 용량을 맞춘 shared baseline과 replay를 함께 평가.
- 보존, 신규 학습, 독립 학습 대비 전이를 분리.
- 고정된 expert bank에서 routing 효과만 따로 진단.
- 개발 결과와 별도의 새 seed confirmation을 분리.

**SEMA·SMoLoRA·CoLoR와 동일 조건의 직접 성능 비교는 수행하지 않았다.** 따라서 현재 결과로 이들보다 우수하다거나 LoRA expert 확장 자체가 새로운 방법이라고 주장할 수 없다. 위 관련 연구 정리는 현재 설계와 직접 관련된 연구를 중심으로 한 것이며, 전체 문헌의 포괄적 조사나 novelty 검증을 대신하지 않는다.

## 3. 연구 설계와 수행 과정

### 3.1 데이터와 task 구성

| 기호 | Dataset | Task family | 평가 지표 |
|---|---|---|---|
| V | VQAv2 | VQA | Official consensus accuracy |
| C | COCO2014 captions | Captioning | CIDEr-D |
| G | GQA balanced | VQA | Official normalized accuracy |
| F | Flickr30k captions | Captioning | CIDEr-D |

각 task에 train 2,048 images, validation 256 images, confirmation eval 512 images를 사용했다. VQA는 이미지당 최대 한 QA, caption은 이미지당 모든 reference를 보존한다.

총 **11,264개 이미지**를 구성하고 task·split 사이의 이미지 identity/byte-hash 중복과 과거 실험 데이터 중복을 검사했다. 다만 이 검사는 **near-duplicate나 BLIP 사전학습 데이터와의 중복까지 제거했다는 의미는 아니다.**

Flickr30k는 사용자가 제시한 공개 [HF mirror](https://huggingface.co/datasets/nlphuji/flickr30k)의 고정 revision `2b239befc81b6e3f035ce6bd52f5f4d60f5625f7`을 사용했다. HF 상위 split 명칭이 아니라 CSV 내부의 실제 split 정보를 따랐다. Mirror 확보가 원본 이미지 이용 조건을 없애는 것은 아니다.

### 3.2 학습 순서

다음 네 순서를 사전에 고정했다.

1. `V → C → G → F`
2. `F → G → C → V`
3. `V → G → C → F`
4. `F → C → G → V`

교차 학습과 같은 family를 연속 학습하는 상황을 포함한다. 가능한 모든 task 순서를 평가한 것은 아니다. 같은 데이터의 순서만 바꾼 것이므로 네 order를 독립 dataset처럼 세지 않는다.

### 3.3 비교 방법과 공정성

| 방법 | 역할 |
|---|---|
| Shared LoRA | Rank-8 단일 adapter를 계속 업데이트하는 기본 순차학습 |
| Capacity-matched shared | Rank-32 adapter로 마지막 expansion bank와 총 expert 파라미터 용량을 맞춘 비교군 |
| Common replay | Rank-8 adapter, 과거 task당 64개 이미지를 저장해 재학습 |
| Expansion | Task마다 rank-8 expert 추가, 기존 expert 동결, joint routing |
| Frozen-first | 첫 task만 학습하고 이후 업데이트하지 않는 진단 |
| Independent | Task별로 처음부터 독립 학습하는 전이 기준선 |
| Tuned replay | Validation에서 선택한 설정을 사용하는 별도 replay 비교 |

공통 조건은 LR `3e-4`, task당 5 epochs, effective batch 16, microbatch 8, image size 224다. 기본 LoRA rank/alpha는 8/16, capacity-matched는 32/64다. 모든 방법에 같은 초기화, 예제별 target-token 평균 loss, greedy generation을 적용했다. 생성 한도는 VQA 32, caption 64 new tokens다.

공통 방법의 optimizer steps와 총 image exposures를 맞췄다. 단, replay는 batch 절반을 과거 이미지로 대체하므로 **새 task 이미지 노출은 더 적다.** 같은 image exposures가 동일 FLOPs·학습 token 수·wall time을 뜻하지는 않는다.

Expansion도 memory-free가 아니다. Expert bank가 증가하며 router가 task당 128개의 train feature를 보존한다. 저장·학습·추론 비용은 성능과 별도로 비교해야 한다.

### 3.4 단계별 수행

| 단계 | 목적 | 수행 규모·결과 |
|---|---|---|
| G0 | 데이터·scorer·학습·생성·resume 구현 검증 | 실제 GPU smoke 및 기존 VQA parity 검증 |
| G1 | 네 task가 각각 학습 가능한지 확인 | 독립 학습 16개, gate 통과 |
| G2 | VQA↔caption 양방향 pilot | 8개 학습, gate 통과 |
| G3 | 4-task 개발 및 replay 설정 선택 | 68개 학습, gate 통과 |
| G4 | 새 seed의 confirmation | 8 seeds, 224개 학습 |
| G5 | 학습된 expert를 고정한 routing 진단 | 32 banks × 5 variants = 160조건 |

G1~G3 개발은 총 **92개 학습**, G4까지 합하면 **316개 학습 조건**이다. G0 기술 smoke는 이 합계에 포함하지 않는다. G5는 expert를 추가 학습하지 않은 평가다.

개발 seed는 17이며 confirmation seeds는 `137, 149, 163, 179, 191, 211, 227, 241`이다. G4의 224개 조건은 공통 네 방법 128개, frozen-first 32개, independent 32개, tuned replay 32개로 구성했다.

G4 전에 데이터·코드·scorer·정규화·설정·비교 목록을 lock했다. Eval 결과로 expansion 설정이나 router를 다시 선택하지 않았다.

### 3.5 평가·통계 설계

Accuracy와 CIDEr-D를 그대로 평균하지 않았다. Task별 점수를 G1에서 고정한 independent validation reference score로 나눈 뒤 family 내부 평균을 계산했다. 따라서 normalized score는 정확도 백분율이 아니며 **1보다 클 수 있다.**

- **Family final score:** 마지막 stage에서 해당 family의 두 task normalized score 평균.
- **Balanced score:** VQA와 caption family score의 1:1 평균.
- **AF:** 해당 task 학습 직후 점수 − 마지막 stage 점수. 마지막 task는 제외.
- **Pre-to-post gain:** 새 task 학습 직후 점수 − 학습 직전 점수.
- **Transfer:** CL에서 task 학습 직후 점수 − 같은 seed·공통예산 independent 점수.

Confirmation 통계는 다음과 같이 고정했다.

- Seed마다 네 order의 결과를 먼저 평균.
- **8개 seed**를 통계 단위로 사용; 32개 독립 관측으로 취급하지 않음.
- Expansion 대 shared/capacity/replay × 두 family = **6개 주 비교**.
- Exact two-sided sign-flip 검정 및 Holm 보정.
- Seed bootstrap 20,000회로 95% percentile CI 산출.
- Tuned replay 비교는 별도 2개 Holm 비교군이며 탐색적 비교로 구분.
- AF, transfer, 개별 task/order 및 routing 진단도 탐색적 분석으로 구분.

CI는 개별 구간이며 동시 CI가 아니다. 새로운 dataset이나 평가하지 않은 task order에 대한 불확실성까지 포함하지 않는다.

**8-seed 검정의 해상도:** 부호 조합은 `2^8 = 256`개이고 exact two-sided sign-flip의 최소 raw p는 `2/256 = 0.0078125`다. 따라서 Holm 6개 비교의 최소 보정 p는 `6 × 0.0078125 = 0.046875`, 별도 tuned replay 2개 비교는 `2 × 0.0078125 = 0.015625`다. 이산 검정과 다중비교 때문에 0.05 기준의 여유가 매우 작다. 이는 더 큰 효과일수록 현재 p가 계속 작아질 수 있는 검정이 아니다.

G4 **1,409,024개**, G5 **1,081,344개** 원 예측을 독립 재채점했다.

## 4. 실험 결과 및 분석

### 4.1 개발 단계에서 확인한 것

G1에서는 네 task 모두 frozen base보다 개선되어 이 경로에서 학습 자체가 가능함을 확인했다. G2도 양방향 pilot 진입 기준을 통과했다.

그러나 G3부터 경고 신호가 있었다.

- Expansion은 shared·capacity보다 높았다.
- **Common replay와 tuned replay보다 낮았다.**
- Flickr가 후속 task일 때 음의 pre-to-post gain이 발생했다.

G3 gate는 family 평균의 학습 가능성 등을 확인하는 기준이므로 **gate 통과 자체가 replay 우월성을 의미하지 않았다.**

### 4.2 Confirmation 최종 성능

고정 네 order·8 seed의 평균이다. Balanced는 두 family의 normalized final score를 1:1로 평균한 기술통계다.

| 방법 | VQA | Caption | Balanced |
|---|---:|---:|---:|
| Shared | 0.7006 | 0.3653 | 0.5329 |
| Capacity-matched | 0.7754 | 0.6497 | 0.7126 |
| Common replay | 0.9926 | 0.9429 | 0.9678 |
| Expansion | 0.9725 | 0.7434 | 0.8580 |
| Frozen-first | 0.4140 | 0.1913 | 0.3026 |
| Tuned replay | 1.0689 | 1.0621 | 1.0655 |

Tuned replay는 개발 단계에서 선택된 **10 epochs** 조건이므로 5-epoch 공통 방법들과 동일 학습 예산 비교는 아니다.

### 4.3 주 비교의 통계 결과

차이는 모두 **Expansion − baseline**이다.

| Family | Baseline | 평균 차이 | 95% CI | Raw p | Holm p |
|---|---|---:|---|---:|---:|
| VQA | Shared | +0.271990 | [0.249411, 0.295666] | 0.0078125 | 0.046875 |
| VQA | Capacity | +0.197128 | [0.169414, 0.225909] | 0.0078125 | 0.046875 |
| VQA | Common replay | −0.020090 | [−0.041010, −0.001266] | 0.125 | 0.125 |
| Caption | Shared | +0.378081 | [0.353165, 0.403818] | 0.0078125 | 0.046875 |
| Caption | Capacity | +0.093677 | [0.070897, 0.115995] | 0.0078125 | 0.046875 |
| Caption | Common replay | −0.199517 | [−0.246175, −0.152715] | 0.0078125 | 0.046875 |

해석:

1. **Shared·capacity 대비 최종 성능 우위는 두 family 모두 확인했다.**
2. **Caption은 common replay보다 유의하게 낮다.**
3. VQA도 평균은 replay보다 낮지만 사전 지정 exact-test 기준에서는 유의하지 않다.
4. 따라서 **replay를 포함한 종합 성공 기준은 두 family 모두 미충족**이다.

VQA replay 비교에서 bootstrap CI가 음수인데 exact-test p는 유의하지 않은 것은 서로 다른 추론 방식의 결과다. 유리한 쪽으로 판정 기준을 바꾸지 않았다. **8 seed의 낮은 해상도와 검정력 한계는 이 비유의 결과를 해석할 때 큰 제약이며, 비유의는 동등함을 뜻하지 않는다.** 다만 관측된 비유의의 원인 중 검정력 부족이 차지하는 비율을 이 자료만으로 정량화하지는 않았다. 동등성/비열등성 검정을 수행한 것도 아니다.

위 raw p는 기존 G4 통계 artifact에서 재확인했다. 유의한 주 비교 5개는 모두 raw p와 Holm p가 각각의 바닥값이다. 따라서 서로 같은 p를 가진 비교들의 효과 크기까지 같다고 읽으면 안 된다.

Tuned replay 대비 별도 비교는 다음과 같다.

| Family | 평균 차이 | 95% CI | Raw p | 별도 Holm p |
|---|---:|---|---:|---:|
| VQA | −0.096337 | [−0.112906, −0.080001] | 0.0078125 | 0.015625 |
| Caption | −0.318700 | [−0.370600, −0.270844] | 0.0078125 | 0.015625 |

두 family 모두 tuned replay에 열세였다. 학습 예산 차이를 감안해 공통 예산 비교와 구분한다.
두 tuned 비교도 모두 해당 raw/보정 p의 바닥값이다.

### 4.4 망각, 신규 학습, positive transfer

#### 망각은 줄었다

Family-normalized 평균 AF는 다음과 같다. 낮을수록 기존 task 성능 감소가 작다.

| Family | Shared AF | Expansion AF |
|---|---:|---:|
| VQA | 0.4367 | 0.0370 |
| Caption | 0.6391 | 0.0089 |

이는 **망각 감소라는 관찰**이다. 최종 성능의 유의성 검정을 AF 자체의 유의성으로 대신하지 않는다. 기존 expert가 동결되어도 router 갱신으로 이전 task 성능은 바뀔 수 있다.

#### 새 task 학습은 task별로 다르다

Expansion의 후속 task pre-to-post gain을 **같은 transition 가중치**로 raw와 normalized 양쪽에 제시한다. 첫 task는 제외하며, task별 normalized gain은 raw gain을 고정 G1 task reference로 나눈 값이다.

| 집계 | 후속 조건 수 | Raw gain 평균 | Raw 단위 | Normalized gain 평균 | 음의 gain 조건 수 |
|---|---:|---:|---|---:|---:|
| VQA family | 48 | +17.840169 | accuracy points | +0.527395 | 0 |
| Caption family | 48 | +0.286678 | CIDEr-D, ×1 | +0.372718 | 16 |
| VQAv2 | 16 | +5.571289 | accuracy points | +0.154859 | 0 |
| GQA | 32 | +23.974609 | accuracy points | +0.713663 | 0 |
| COCO | 32 | +0.444392 | CIDEr-D, ×1 | +0.653260 | 0 |
| Flickr30k | 16 | −0.028751 | CIDEr-D, ×1 | −0.188366 | 16 |

**Flickr 후속 16개 조건은 모두 음의 gain**이다. Family 평균만 보면 이 실패가 가려진다.

**가중치 주의:** 위 family 표는 48개 후속 transition을 동등 가중한 기존 기술통계다. Gain 추론 통계는 각 order/seed 안의 해당 family 후속 task를 먼저 평균한다. 이 order·seed 동등 가중치로 맞춘 raw/normalized 평균은 VQA **+16.071777 / +0.475670**, caption **+0.308423 / +0.416859**다. Order마다 후속 task 수가 달라 두 평균은 다르며 혼용하지 않는다.

**CIDEr-D 스케일:** 표는 `pycocoevalcap`의 native 출력 **×1**이며 논문 표기 관행의 추가 **×100**을 적용하지 않았다. Scorer 내부 CIDEr 배율 `10.0`은 알고리즘 자체의 값이지 여기서 추가로 곱한 표시 배율이 아니다. IDF는 task train references로 고정했다.

즉, 새 task를 전혀 배우지 않는 모델은 아니지만 **모든 task를 안정적으로 개선하는 방법도 아니다.**

#### Positive transfer도 일반적으로 성립하지 않는다

같은 seed·공통 학습 예산의 independent 기준선 대비 학습 직후 normalized score 차이는:

- VQA: **−0.04324**
- Caption: **+0.02015**

Caption에서는 작은 양의 평균 차이가 있지만 VQA에서는 독립 학습보다 낮다. **보존 효과를 일반적인 positive transfer로 해석할 수 없다.** 이 비교는 탐색적 분석이다.

#### H3 재분해: 파라미터 전이와 routing을 구분

기존 G4/G5에서 **각 task 학습 직후의 diagonal 시점**을 맞춰 `N=native`, `O=그 task expert를 강제한 oracle`, `I=같은 seed independent`로 두었다. 각 task의 고정 normalizer로 나눈 후 task→order→seed 순서로 기존 G4 transfer와 동일하게 평균했다. 첫 task도 포함하는 원 transfer 정의를 유지한다.

`T=N−I = R(N−O) + K(O−I)`는 저장된 corpus 점수의 항등 분해다.

| 집계 | T: 원 independent 대비 차이 | R: native−oracle | K: oracle−independent |
|---|---:|---:|---:|
| VQAv2 | −0.019256 | −0.011587 | −0.007668 |
| GQA | −0.067224 | −0.030160 | −0.037064 |
| VQA family | −0.043240 | −0.020874 | −0.022366 |
| Caption family | +0.020153 | +0.015862 | +0.004291 |

VQA의 oracle 강제 선택으로 제거되는 aggregate gap `R`의 크기는 원 deficit의 **48.27%**다. 그러나 이것을 곧바로 정확한 오routing 인과 기여율로 부르면 안 된다. 같은 expert를 선택한 샘플에서도 native/oracle 문장이 다른 경우가 VQAv2 1개, GQA 20개 존재한다. 그 원인은 현재 산출물로 확정하지 않았다. `K` 역시 초기화·batch seed 차이 등의 실행 차이가 남는 잔차이며, 이전 task가 파라미터를 악화시켰다는 증거가 아니다.

**오routing 기여의 보수적 범위:** VQA per-image accuracy가 0–100 points이고 own-route 문장이 같은 샘플의 paired Δ는 0이라는 사실만 사용하면, 다른 문장 21개의 own-route 기여 절댓값은 `B=(100×1/R_V + 100×20/R_G)/32768 = 0.001901687` 이하이다. 따라서 **오routing subset이 `N−O`에 기여한 값은 [−0.022775, −0.018972]** 범위다. 이는 기존 점수·개수에서 계산한 최악경계이지 CI나 인과 추정치가 아니다. `N−I`의 오routing subset 기여와는 다르며, `K`의 그룹별 배분은 여전히 식별되지 않는다.

| Task | 학습 직후 own-expert 샘플 수 | 오routing 샘플 수 | Own 그룹 평균 accuracy | 오routing 그룹 평균 accuracy |
|---|---:|---:|---|---|
| VQAv2 | 14,137 | 2,247 | 미기록·현 범위에서 식별 불가 | 미기록·현 범위에서 식별 불가 |
| GQA | 13,989 | 2,395 | 미기록·현 범위에서 식별 불가 | 미기록·현 범위에서 식별 불가 |

샘플 수는 4 orders × 8 seeds × 512 images의 **평가 occurrence**이며 서로 다른 이미지 수가 아니다. JSONL에는 ID·예측 문자열·expert·생성 진단만 있고 per-example VQA accuracy는 저장되지 않았다. 따라서 own/오routing의 절대 점수나 `N−I`를 두 그룹에 정확히 배분한 값은 scorer를 다시 실행하지 않는 이번 범위에서 복원할 수 없다. 위 corpus 단위 `R/K`는 저장 점수만으로 가능한 분해이지 누락된 per-example 점수를 추정해 채운 결과가 아니다.

### 4.5 Routing 진단

동일한 32개 expert bank를 고정한 결과다.

| Router | VQA | Caption | Balanced |
|---|---:|---:|---:|
| Native joint | 0.9725 | 0.7434 | 0.8580 |
| Image-only | 0.5016 | 0.4192 | 0.4604 |
| Question-only | 0.9751 | 0.6531 | 0.8141 |
| Family-restricted joint | 0.9731 | 0.7434 | 0.8583 |
| Task-ID oracle | 1.0146 | 0.7114 | 0.8630 |

모든 bank에서 native 14개 파일의 부모 byte parity와 expert 재학습 0회를 검증했다. Native 재채점 점수는 32개 G4 expansion 결과와 모두 일치한다.

주요 관찰:

- Native가 다른 family로 보내는 비율은 **0.01984%**로 매우 작다.
- 다른 task expert로 보내는 비율은 **23.88%**다.
- Family mask를 적용해도 성능 변화가 작다.
- Image-only는 다른 family로 보내는 비율이 **39.69%**이고 성능도 크게 낮다.
- Question-only는 caption의 50%를 다른 task expert로 보내며 caption 평균이 native보다 낮다.
- Task-ID oracle은 VQA를 개선하지만 caption은 native보다 낮다.

따라서 **family 구분 실패가 전체 성능 저하의 주원인이라는 근거는 약하다.** 정답 task expert를 선택하면 성능도 최대가 된다는 가정도 성립하지 않는다. Task-ID oracle은 expert/task 식별의 진단이지 task 성능의 수학적 상한이 아니다.

Validation 단일 bank에서 보였던 question-only 이득은 confirmation 전체 평균으로 일반화되지 않았다. 이 탐색 결과로 주 router를 다시 선택하지 않았다.

#### 별도 관찰: Caption native > task-ID oracle

Caption normalized final은 **native 0.743379 > task-ID oracle 0.711416**이다. 이를 task·order별로 분해하면 **COCO에서는 oracle이 높고, Flickr에서는 native가 높다.** 따라서 family 평균만으로 모든 caption task에서 다른 expert 선택이 유리하다고 결론낼 수 없다.

아래 점수는 raw CIDEr-D **×1**의 8-seed 평균이며, 표의 점수는 그룹 점수가 아닌 **전체 512-image corpus 점수**다. Own/other 수는 order당 8 seeds의 4,096 occurrences다. Other는 모두 같은 caption family의 다른 task expert이며 cross-family 선택은 0이다.

| Task | Order | Own / other 수 | Native 전체 점수 | Oracle 전체 점수 | 전체 Δ(N−O) | Own 선택인데 문장이 다른 수 |
|---|---|---:|---:|---:|---:|---:|
| COCO | V→C→G→F | 3,143 / 953 | 0.446388 | 0.520057 | −0.073668 | 3 |
| COCO | F→G→C→V | 3,216 / 880 | 0.453848 | 0.511954 | −0.058106 | 2 |
| COCO | V→G→C→F | 3,199 / 897 | 0.449333 | 0.511954 | −0.062622 | 4 |
| COCO | F→C→G→V | 3,157 / 939 | 0.448407 | 0.520057 | −0.071649 | 5 |
| Flickr | V→C→G→F | 3,391 / 705 | 0.127488 | 0.102862 | +0.024626 | 8 |
| Flickr | F→G→C→V | 3,388 / 708 | 0.124296 | 0.099927 | +0.024370 | 6 |
| Flickr | V→G→C→F | 3,366 / 730 | 0.128793 | 0.102862 | +0.025931 | 8 |
| Flickr | F→C→G→V | 3,416 / 680 | 0.123722 | 0.099927 | +0.023795 | 5 |

**전체 subgroup 점수는 식별할 수 없다:** per-image CIDEr-D contribution은 저장되지 않았다. 또한 own 선택 샘플 26,276개 중 41개에서 native/oracle 예측 문장이 다르므로, own 그룹의 차이를 무조건 0으로 놓고 전체 Δ를 other 그룹에 귀속하는 계산은 유효하지 않다. 이 문장 차이의 원인도 미확정이다. 전체 cohort에서 own/other 각각의 절대 점수와 paired difference는 **미기록·식별 불가**로 남긴다.

다만 **own 선택 샘플의 문장이 전부 동일한 bank/task cell**은 예외다. 고정 train-IDF CIDEr-D의 per-image 가산성 때문에 그 cell에서는 own 그룹 Δ가 정확히 0이고, other 그룹의 평균 paired difference를 `전체 Δ × 512 / other 수`로 복원할 수 있다. 이는 새 채점이 아니라 기존 corpus 점수의 대수적 분해다. 아래 값은 식별 가능한 seed cell 안에서 먼저 계산한 뒤 그 cell들을 동등 평균했으며, 제외된 seed를 숨기지 않는다.

| Task | Order | 식별 가능 seed / 전체 | Other 그룹 paired Δ, 식별 가능 cell만 |
|---|---|---:|---:|
| COCO | V→C→G→F | 5 / 8 | −0.309108 |
| COCO | F→G→C→V | 6 / 8 | −0.257097 |
| COCO | V→G→C→F | 4 / 8 | −0.292184 |
| COCO | F→C→G→V | 4 / 8 | −0.303277 |
| Flickr | V→C→G→F | 3 / 8 | +0.125975 |
| Flickr | F→G→C→V | 4 / 8 | +0.131622 |
| Flickr | V→G→C→F | 2 / 8 | +0.214403 |
| Flickr | F→C→G→V | 5 / 8 | +0.149400 |

이 식별 가능한 부분집합에서는 Flickr 샘플을 COCO expert로 보냈을 때 평균 paired Δ가 양수이고 COCO→Flickr는 음수다. 그러나 router가 선택한 비무작위 샘플이며 식별 가능 seed만의 결과다. 전체 cohort의 인과효과, 모든 샘플에서의 expert 우열, 초기화 개선 효과로 일반화하지 않는다. **품질 기반 expert 선택과 이전 expert 재사용을 조사할 근거는 되지만, 기존 expert에서 초기화하면 좋아진다는 실험 증거는 아니다.**

### 4.6 결과의 신뢰성과 남은 제한

- 원 pre-eval lock과 source snapshot을 보존했다.
- 학습 완료 후 감사 도구의 parity·자원 출력·lock binding 오류를 수정했고 전후 SHA와 수정 범위를 기록했다.
- 모델 학습, 원 예측, scorer, 통계 기준은 변경하지 않았다.
- Repository-owned 테스트는 **308개 통과**했다.
- 범위 없는 pytest는 third-party SMoLoRA의 Transformers `llava` config 충돌로 collection에 실패했다. 이를 테스트 통과로 계산하지 않았다.
- 원 예측·로그·checkpoint와 독립 로컬 백업을 보존했다.

결과의 범위는 **고정 BLIP, task당 2,048 train/512 eval 이미지, 네 순서, 8 seeds**다. 전체 dataset, 다른 backbone, 다양한 instruction, 더 긴 task stream에 대한 일반화는 확인하지 않았다. Caption CIDEr-D는 train-IDF를 고정한 평가이므로 외부 SOTA 수치와 직접 비교하면 안 된다.

### 4.7 기존 기록에 근거한 저장·시간 비용

**범위:** G4 `resources.csv`의 224 runs/800 stage 행과 보관 checkpoint 256개의 ZIP storage metadata를 사용했다. 모델을 로드하거나 성능을 새로 측정하지 않았다. 아래 bytes는 공통 BLIP backbone을 제외한 적응 상태이며, tensor payload와 직렬화 파일 크기를 구분한다. LoRA/router float32는 실제 checkpoint storage type으로 확인했다.

#### 학습 후 저장 크기

각 방법 32개 run 기준이다. Independent만 단일 task run이고 나머지는 4-task 종료 상태다. `latest.pt`는 부가 metadata/RNG/ZIP overhead도 포함한 평균 파일 크기다.

| 방법 | Adapter/expert tensor bytes | Router feature tensor bytes | Router parameter tensor bytes | Replay 원본 JPEG 참조 bytes 평균 | Replay row JSON bytes 평균 | `latest.pt` bytes 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Shared | 4,718,592 | 0 | 0 | 0 | 0 | 5,015,634 |
| Capacity-matched | 18,874,368 | 0 | 0 | 0 | 0 | 19,184,216 |
| Common replay | 4,718,592 | 0 | 0 | 38,758,830.94 | 143,338.19 | 5,168,182 |
| Expansion | 18,874,368 | 3,145,728 | 30,736 | 0 | 0 | 22,560,498 |
| Frozen-first | 4,718,592 | 0 | 0 | 0 | 0 | 5,159,138 |
| Independent, 1 task | 4,718,592 | 0 | 0 | 0 | 0 | 4,825,932 |
| Tuned replay | 4,718,592 | 0 | 0 | 38,758,830.94 | 143,338.19 | 5,182,010 |

- **Replay 이미지 기준:** 기존 원본 JPEG 파일의 실제 크기를 참조한 논리적 보존량이다. Buffer는 이미지 경로·질문·정답/reference row를 저장하며 이미지 파일을 별도 복제하지 않는다. 따라서 위 JPEG bytes를 추가 디스크 복사량으로 해석하면 안 된다. 전처리 이미지 tensor 저장 크기는 **미기록**이며 현재 구현은 이를 buffer에 영속 저장하지 않는다.
- Replay 원본 JPEG 범위는 36,678,985–42,029,040 bytes, row JSON 범위는 142,127–145,290 bytes다. JSON counter와 checkpoint pickle 내부 row 직렬화 크기는 같은 표현이 아니므로 서로 그대로 더하거나 치환하지 않는다.
- **Expansion router:** feature는 `4 × 128 × 1536 × 4 = 3,145,728` bytes, 파라미터는 weight 24,576 + center 6,144 + prior 16 = 30,736 bytes다. 기존 `router_memory_bytes=3,176,586` counter는 여기에 label의 논리적 32 bytes와 JSON metadata 90 bytes를 더한 값이다. Python 객체의 실제 RAM 크기나 ZIP 파일 크기가 아니다.
- Expansion `latest.pt` 범위는 22,416,186–22,715,706 bytes다. 모든 256개 보관 checkpoint의 optimizer/scheduler state는 실제로 `None`이다. 이는 학습 중 optimizer 메모리가 0이라는 뜻이 아니며, 그 상태의 실제 byte 측정값은 **미기록**이다.
- Independent를 4개 task 모델로 보존하면 adapter tensor payload는 총 18,874,368 bytes다. 공통 backbone은 공유 가능한 별도 자산이며 각 adapter checkpoint에 중복 저장되지 않는다. 별도 `independent_*.pt` archive도 존재하므로 실험 백업 전체 크기와 배포 모델 하나의 크기를 혼동하지 않는다.

#### Task 수에 따른 증가

아래는 기록된 stage별 값이다. Replay JPEG/row는 seed·order 평균으로 소수 둘째 자리까지 표시했다.

| 학습 task 수 T | Expansion bank bytes | Expansion router counter bytes | Replay images | Replay JPEG 참조 bytes 평균 | Replay row JSON bytes 평균 |
|---|---:|---:|---:|---:|---:|
| 1 | 4,718,592 | 798,822 | 64 | 9,654,931.81 | 38,590.19 |
| 2 | 9,437,184 | 1,591,410 | 128 | 19,154,615.19 | 71,658.38 |
| 3 | 14,155,776 | 2,383,998 | 192 | 29,169,863.19 | 104,714.88 |
| 4 | 18,874,368 | 3,176,586 | 256 | 38,758,830.94 | 143,338.19 |

Expansion bank는 task당 +4,718,592 bytes, router counter는 +792,588 bytes다. Feature tensor만의 증분은 +786,432 bytes이며, vector 수는 `128 × T`다. Shared/replay/frozen의 adapter는 고정 4,718,592 bytes, 이번 capacity 대조는 모든 stage에서 고정 rank-32/18,874,368 bytes다. Replay는 task당 64개 이미지가 늘지만 JPEG/row byte 증분은 선택된 파일에 따라 달라 고정 byte 증가율로 외삽하지 않는다. Independent의 전체 task 모델 보존량은 task당 adapter 한 개씩 증가한다.

#### 기록된 시간과 미기록 항목

단위는 run당 평균 초다. 학습 시간은 기존 `training_seconds`, feature/fit은 `feature_seconds`, 평가는 `evaluation_seconds`, 전체 wall은 `wall_seconds`의 합을 run 수로 나눈 값이다. Independent는 1-task run이라 4-task run과 그대로 속도를 비교하지 않는다.

| 방법 | 학습 구간 wall s | Feature 추출·router fit s | 평가 구간 wall s | 전체 run wall s |
|---|---:|---:|---:|---:|
| Shared | 1,097.654 | 0.000 | 556.179 | 1,855.198 |
| Capacity-matched | 1,095.132 | 0.000 | 558.026 | 1,992.772 |
| Common replay | 1,082.367 | 0.000 | 637.532 | 1,926.293 |
| Expansion | 1,093.118 | 98.155 | 681.617 | 2,140.811 |
| Frozen-first | 277.106 | 0.000 | 828.734 | 1,207.749 |
| Independent, 1 task | 269.968 | 0.000 | 84.930 | 449.580 |
| Tuned replay | 2,192.979 | 0.000 | 641.171 | 3,186.659 |

Feature/fit의 0.000은 소수 셋째 자리 반올림이며 해당 방법에 router 학습이 없다는 코드 계약과 일치한다. 전체 wall에는 표의 세 구간 외 I/O 등도 들어 있어 합과 일치하지 않는다. Expansion의 98.155초는 **학습 단계 특징 추출·router fit**이며 추론 routing overhead가 아니다.

| 요청 비용 | 기록 여부 |
|---|---|
| GPU 시간/GPU-hours | **미기록** — 위 wall time이나 SLURM 대기 포함 시간을 GPU compute 시간으로 환산하지 않음 |
| 추론 routing overhead | **미기록** — 별도 timing 구간 없음 |
| 샘플당 추론 latency | **미기록** — 전체 평가 wall time을 샘플 수로 나눈 값을 latency로 대신 제시하지 않음 |

저장 측면에서는 raw-image 보존 의무와 feature/model 보존 의무가 다르다. 비용 표만으로 expansion의 일반적인 효율 우위를 주장하지 않으며, 기존 5.1절 성능 결론과 충돌하지 않는다.

## 5. 결론 및 향후 계획

### 5.1 현재 주장할 수 있는 결론

> 고정된 BLIP 기반 heterogeneous VQA/captioning CL 환경에서 expert 확장·동결은 shared 및 capacity-matched adapter보다 높은 최종 성능과 낮은 평균 망각을 보였다. 그러나 common replay 대비 우월성은 확인하지 못했고 caption에서는 열세였다. 신규 task 학습과 positive transfer에도 task별 실패가 남았다.

현재 **주장할 수 없는 것**:

- Replay보다 우수한 일반적 CL 방법.
- 모든 신규 task에 대한 안정적인 학습.
- 일반적인 positive transfer.
- Routing 문제의 해결.
- SEMA·SMoLoRA 등 선행 방법 대비 우월성이나 새로운 SOTA.

현재까지는 새로운 방법의 성공 증명보다는 **보존에는 효과가 있지만 재사용·신규 학습·최종 성능에는 한계가 있는 구조라는 검증 결과**로 보는 것이 타당하다.

### 5.2 향후 계획 제안

**권장 순서: 실패 원인 분해 → 최소한의 방법 변경 → 새 confirmation.** 추가 대규모 실험부터 시작하지 않는다.

#### 1단계 — 기존 산출물로 원인부터 분해

이번 리뷰로 diagonal transfer의 `R/K` 분해, caption의 task/order별 차이와 식별 한계, 기록된 저장·시간 비용을 4.4–4.7절에 추가했다. 아래 질문 중 아직 해결되지 않은 인과 원인·예측 품질 분석만 후속 검토 대상으로 남긴다.

- Flickr의 음의 gain이 어떤 순서와 예측 유형에서 발생하는가?
- Pre-task에서 사용하던 expert와 학습 후 선택되는 expert의 답변이 어떻게 달라지는가?
- Task-ID oracle에서도 Flickr 문제가 남는가?
- Caption 길이, EOS, 반복, 이미지–문장 정합성과 성능 저하가 연결되는가?
- 기록된 비용 차이가 실제 배포 조건에서도 이점인지 판단할 수 있는가? 현재 기록에 없는 GPU 시간·추론 latency는 이 리뷰로 채우지 않았다.

목적은 **routing 문제와 expert 학습 문제를 구분**하는 것이다. 현재 Flickr 실패 원인이 routing이라고 확정할 수 없다.

#### 2단계 — 원인에 맞는 최소 변경만 검증

| 방향 | 검증할 가설 |
|---|---|
| 기존 expert 재사용/초기화 | 4.5절의 식별 가능한 Flickr subset에서 COCO expert 선택의 paired Δ가 양수이며, COCO가 먼저 학습된 두 order에서도 관찰된다. 이는 추론 시 재사용을 조사할 근거이며, **[INFERENCE]** 이를 초기화로 이식하면 신규 학습이 좋아질지는 미검증이다. 현재는 이전 expert에서 새 expert로 파라미터 전이 경로가 없다. |
| Shared + private 구성 | **[INFERENCE]** 공통 지식은 공유하고 task별 지식만 분리하면 보존·전이 균형이 개선될 수 있음 |
| 선택적 확장 | 새 expert가 항상 필요한지, 기존 expert를 유지하는 편이 나은 경우가 있는지 검증 |
| 품질 기반 expert 선택 | Caption native > task-ID oracle과 COCO/Flickr의 반대 부호는 task identity와 생성 품질이 일치하지 않을 수 있음을 지지한다. 다만 부분집합 선택 편향과 미식별 cell 때문에 전체 cohort의 최적 gate 또는 개선량을 주장하지 않는다. |

위 효과는 아직 검증되지 않았다. 선행 연구와 겹칠 수 있으므로 구조를 추가하기 전에 **SEMA·SMoLoRA와의 차별성과 검증 가능한 새 주장**부터 명확히 해야 한다. 현재 G5 결과상 family routing만 개선하는 방향의 우선순위는 낮다.

#### 3단계 — 개발 검증을 통과한 방법만 재확인

- Common-budget replay를 필수 비교군으로 유지.
- 용량뿐 아니라 token·계산·저장 비용을 함께 비교.
- Flickr 신규 학습 실패를 별도 수용 기준으로 설정.
- 현재 eval을 이용해 방법을 선택했다면 이를 다시 독립 confirmation으로 부르지 않고 새로운 held-out 확인 계획 수립.
- 방법 수준의 기여를 주장하려면 관련 방법의 직접 비교 추가.

**현재 권장안은 1단계 원인 분석부터 진행하는 것이다.** 이를 통해 구조 변경이 필요한지, 연구 주장을 replay 우월성에서 특정 비용·보존 조건의 이점으로 조정할 근거가 있는지 판단한다. 비용상 이점도 현재 확정된 결론은 아니다.

향후 계획은 사용자 검토용 제안이며 **추가 실험이나 방법 변경을 실행한 상태가 아니다.**

## 6. 상세 근거와 산출물

- [연구 설계](MULTIMODAL_DESIGN.md)
- [실험계획·단계별 실행·최종 결과](MULTIMODAL_EXPERIMENTS.md)
- [기계 판독 설정 및 완료 상태](configs/multimodal_research.json)
- [G4 통계 원문](artifacts/multimodal-g4-statistics.json)
- [G5 routing 집계](artifacts/multimodal-g5-summary.json)
- [최종 실행·산출물 무결성 기록](artifacts/multimodal-confirmation-execution.json)
- [학습 후 감사 코드 교정 이력](artifacts/multimodal-confirmation-audit-revision.json)
- [원 pre-eval lock](artifacts/multimodal-confirmation-lock.json)
- [G4 전체 audit·CSV archive](artifacts/multimodal-g4-report.tar.gz)
- [G5 전체 audit·CSV archive](artifacts/multimodal-g5-report.tar.gz)

### 6.1 이번 리뷰의 범위와 재현

이번 리뷰에서는 **기존 저장 점수·예측·자원 기록만 재집계**했다. 새 학습, 모델 추론, scorer 재실행, 방법 변경 및 새 유의성 검정은 수행하지 않았다. 앞서 기재한 학습·재채점·308개 테스트는 기존 실험 완료 당시 기록이며 이번 리뷰에서 다시 실행한 수치가 아니다. 기존 5.1절 결론은 변경하지 않는다.

통계/gain 집계:

```bash
python scripts/aggregate_multimodal_review_statistics.py
```

- 스크립트: [aggregate_multimodal_review_statistics.py](scripts/aggregate_multimodal_review_statistics.py)
- 출력: [multimodal-review-statistics.json](artifacts/multimodal-review-statistics.json)
- 입력: `artifacts/multimodal-g4-report.tar.gz`의 `g4/audit.json`, `artifacts/multimodal-g4-statistics.json`, `artifacts/multimodal-g1-normalizers.json`. 출력 JSON에 입력 SHA256을 기록했다.
- 기존 가중치 확인: `scripts/evaluate_multimodal_confirmation.py:887–917`의 `subsequent_gain_statistics`. CIDEr 스케일 확인: `sema_multimodal/metrics.py:162,218,253–264` 및 audit의 `scorer_locks`.
- 문헌 검증: 2026-10-16에 2절에 연결한 원 논문과 공식 CVF/ACL/NeurIPS·arXiv 기록을 확인했다. 문헌 수치로 본 실험 수치를 대체하지 않았고, 관련 방법을 새로 실행하지 않았다.

Routing/transfer 재집계:

```bash
python scripts/routing_review_items27.py
```

- 스크립트: [routing_review_items27.py](scripts/routing_review_items27.py)
- 출력: [코드·학습 조건 확인](artifacts/routing-review-code-conditions.json), [routing 분해·식별 범위](artifacts/routing-review-routing-partition.json). 후자에 per-seed cell, task/order counts, 예측 문자열 일치 검사, 식별 가능 seed coverage, 입력 SHA256을 남겼다.
- 점수 입력: `artifacts/multimodal-g4-report/audit.json`, `artifacts/multimodal-g4-report/statistics.json`, `artifacts/multimodal-g5-report/routing_audit.json`, `artifacts/multimodal-g5-summary.json`, `artifacts/multimodal-g1-normalizers.json`.
- 예측 입력: `artifacts/multimodal-g4-raw.tar.gz`, `artifacts/multimodal-g5-raw.tar.gz`의 task별 `predictions_after_*.jsonl` 및 config/results. Diagonal은 task 학습 직후, caption final은 마지막 stage를 사용한다. 같은 ID끼리 비교하며 새 scorer는 호출하지 않는다.
- 위 unpacked report 파일은 각각 `multimodal-g4-report.tar.gz`의 `g4/`, `multimodal-g5-report.tar.gz`의 `g5/`와 같은 내용이다. 재현 시 이 경로로 풀어 둔 사본을 사용한다.

Checkpoint 및 비용 집계:

```bash
python scripts/inspect_multimodal_checkpoint_zip.py
python scripts/aggregate_multimodal_resource_review.py
```

- 스크립트: [inspect_multimodal_checkpoint_zip.py](scripts/inspect_multimodal_checkpoint_zip.py), [aggregate_multimodal_resource_review.py](scripts/aggregate_multimodal_resource_review.py).
- 출력: [checkpoint 실제 storage 구성](artifacts/multimodal-checkpoint-zip-inspection.json), [자원 재집계](artifacts/multimodal-resource-review.json).
- 주요 입력: `artifacts/multimodal-g4-report/resources.csv`, `artifacts/multimodal-confirmation-checkpoints.tar`, `artifacts/multimodal-confirmation-checkpoints.json`. Tensor storage dtype/shape/bytes 및 ZIP member 크기를 읽으며 torch 모델 로드나 forward를 수행하지 않는다.
- 보조 inventory 입력: `artifacts/multimodal-development-checkpoints.json`, `artifacts/multimodal-backups.json`, `artifacts/multimodal-confirmation-execution.json`, `artifacts/multimodal-g5-summary.json`. 파일별 출처와 counter 코드 위치는 출력의 `references`/`code_citations`에 기록했다. 보조 백업 총량을 방법별 학습 후 크기로 사용하지 않았다.
- 시간 표는 `methods[*].sums`를 `n_runs`로 나눈 기존 기록 평균이다. 저장 표는 `checkpoint_probe_exact`와 replay final counters, 증가율 표는 `stage_growth`를 사용했다.

### 6.2 기존 결론과의 관계 및 수정 diff

**2·5·7번 재분석에서 기존 5.1절의 성능 결론을 뒤집는 결과는 없었으며, 5.1절 본문을 그대로 보존했다.** 다만 다음은 별도 해석 보강 사항이다.

1. H3의 차이를 파라미터 전이로 읽을 수 없다. VQA deficit 전체가 routing 때문인 것도 아니며, own-expert 강제 조건에서도 independent와의 잔차가 남는다.
2. Caption native 우위는 COCO/Flickr에서 부호가 반대다. 전체 subgroup 점수는 미기록이고 같은 expert의 예측도 일부 달라, 모든 차이를 other-expert 효과로 귀속할 수 없다.
3. 비용은 정확한 tensor/file bytes와 기록된 wall time까지만 확인했다. Replay 이미지 참조량은 추가 복사량이 아니며, GPU 시간·latency·추론 routing overhead는 미기록이다.

수정 전후 문서 diff는 [multimodal-summary-review.diff](artifacts/multimodal-summary-review.diff), 표기·링크·5.1절 보존 확인 및 계산 입력/스크립트 해시는 [리뷰 검증 기록](artifacts/multimodal-summary-review-verification.json)에 보존한다. 숫자 앞 공백 누락은 `[가-힣][0-9]`, `(?:native|first|independent|replay)[0-9]`, `,[0-9]+ seeds` 패턴으로 찾고 문서 diff에서 확인했다. 논문명 `MoE-Adapters4CL`과 식별자 `G4` 등은 공백 누락으로 처리하지 않았다. 이번 리뷰는 **커밋하지 않는다.**
