---
status: draft
kind: lightweight
created_at: 2026-10-21
locked_code_commit: TBD
---

# P0 경량 사전 등록 초안 — multimodal CL에서의 CLMoE expansion

정식 사전 등록이 아닌 경량 기록이다. 아직 lock하지 않았다. 이 문서가 학습을 승인하지는 않는다. 각 항목의 근거는 `근거:`에 적은 repo 파일이다. 확인하지 못한 항목은 추측하지 않고 **TBD**로 남긴다.

## 1. 배경 (재검증 대상 아님)

VQA 단독 설정에서 연구자가 이미 확인한 expansion 우위 결과가 있다. 이 결과는 인용만 하며 confirmatory 대상이 아니다.

- 인용할 결과 파일과 수치 — **TBD**. 작성 과정에서 점수 파일을 열지 않았다. 후보 설정은 `configs/readout_confirmation_jobs.json`과 `configs/readout_confirmation_comparisons.json`이다(해당 설정이 인용 대상인지 연구자 확인 필요).
- 기존 사전 등록 문서는 로컬 Git 이력(`git log --all -- '*prereg*'`, `git log --all -S preregistration`)과 확인한 DC 디렉터리에서 찾지 못했다.

## 2. 핵심 claim (P0, confirmatory)

> VQA에서 관찰된 expansion 우위가 VQA+captioning task sequence에서도 유지된다.

## 3. 실험 구성 (repo에서 확인)

### 3.1 Task와 순서

| ID | source | family | metric |
|---|---|---|---|
| V | vqav2 | vqa | vqav2 |
| C | coco_captions | caption | cider_d |
| G | gqa | vqa | gqa |
| F | flickr30k | caption | cider_d |

근거: `sema_multimodal/data.py` (`SOURCE_SPECS`). 실제 manifest의 task 정의와 일치하는지는 서버 manifest(SHA-256 `c501786957b9aa7260feabd780d561160abbf53e4bcb722a2375a76179f90b39`)에서 확인해야 한다.

Task 순서 4종: `V C G F`, `V G C F`, `F C G V`, `F G C V`. 근거: `configs/multimodal_g4_jobs.json` (`--task-ids`).

### 3.2 Cell 구성 (224 job, 8 seed)

seeds: 137, 149, 163, 179, 191, 211, 227, 241. 공통 설정은 lr 3e-4, batch 16, micro-batch 8이다. 근거: `configs/multimodal_g4_jobs.json`.

| method | epoch | 순서 | job 수 |
|---|---|---|---|
| `expand` | 5 | 4 | 32 |
| `shared_lora` | 5 | 4 | 32 |
| `capacity_matched_shared` | 5 | 4 | 32 |
| `replay_lora` | 5 | 4 | 32 |
| `replay_lora` | 10 | 4 | 32 |
| `frozen_first` | 5 | 4 | 32 |
| `independent` | 5 | 단일 task V/C/G/F | 32 |

argv에 없는 값은 기본값을 따른다: rank 8, alpha 16, image 224, router `joint`, router memory 128, ridge 1.0, replay-per-task 64. 근거: `sema_multimodal/run.py` argparse 기본값.

### 3.3 구현된 비교군

근거: `sema_multimodal/run.py` (`METHODS`, 학습 loop).

- `expand`: task마다 expert를 추가하고, `FeatureRouter`로 입력을 route한다. 검증 대상 방법이다.
- `shared_lora`: 모든 task가 단일 LoRA를 공유한다.
- `capacity_matched_shared`: 단일 공유 LoRA이며 rank·alpha에 task 수를 곱해 용량을 맞춘다.
- `replay_lora`: 공유 LoRA에 task당 64개 replay를 더한다. ep5와 ep10 두 조건이 있다.
- `frozen_first`: 첫 task만 학습하고 이후 task는 학습하지 않는다.
- `independent`: task별 독립 expert다. sequence 비교군이 아닌 참고 상한이다.
- **"expansion 없는 MoE"(고정 expert 수 + router)는 구현되어 있지 않다.** 가장 가까운 대조군은 `capacity_matched_shared`(용량 통제)와 `shared_lora`다.
- primary 비교군 — **TBD**. 후보: `capacity_matched_shared`, `shared_lora`, 또는 G3에서 선택된 `replay_lora` 조건. G3 선택 결과(`artifacts/multimodal-g3-lock.json`)는 열지 않았다.

### 3.4 측정 가능한 지표

근거: `sema_multimodal/metrics.py`, `sema_multimodal/run.py`.

| 지표 | 구현 | 비고 |
|---|---|---|
| VQAv2 accuracy | 있음 | 공식 leave-one-out consensus |
| GQA accuracy | 있음 | exact match |
| CIDEr-D | 있음 | train-IDF 고정, PTB tokenizer(Java) |
| BLEU-4 | 있음 | 보조 |
| task별 final / post_train | 있음 | `continual_metrics` |
| forgetting (AF = post_train − final) | 있음 | 마지막 task는 None |
| pre_to_post_gain | 있음 | |
| family native final / AF | 있음 | VQA(V,G)와 caption(C,F)이 각각 같은 단위 안에서만 평균 |
| normalized final / AF, family-balanced | 있음 | G1 validation normalizer artifact 필요, `scripts/summarize_multimodal.py` |
| 단위가 다른 metric의 raw 평균 | 없음, 의도적으로 금지 | `run.py` docstring |
| oracle(forced-expert) matrix | 조건부 | `--oracle-diagnostic` 필요. G4 argv에 없음 |
| routing 할당, router memory | 있음 | `task_expert_assignments`, stage resources |

## 4. 판정 기준 초안 (미확정)

- 단위: 같은 seed, 같은 task 순서에서 expand와 비교군의 paired 차이를 계산한다. 8 seed × 4 순서다. paired 단위를 seed(순서 평균 8쌍)로 할지, (seed, 순서) 32쌍으로 할지는 **TBD**.
- 효과 방향이 VQA family와 caption family에서 모두 expand 우위로 일치하는지 본다.
- 1차 지표 — **TBD**. 후보: family native final, family AF, family-balanced normalized final.
- 검정 후보(확정 아님): paired sign-flip permutation test, Wilcoxon signed-rank, seed bootstrap CI, family 2개에 대한 다중 비교 보정(Holm 등).
- 임계값(유의수준, 최소 효과 크기) — **TBD**.

### 실패 조건 초안

| 결과 | 판정 |
|---|---|
| 두 family 모두 expand 우위 | 지지 |
| 한 family에서만 우위 | 부분 지지 |
| 차이 없음 또는 비교군 우위 | 미지지 |

## 5. Exploratory (판정 기준 없음)

- F1 probe
- dose-response
- expert 분화
- routing 분석
- period-strip target 효과 (`scripts/stage1_period_strip_train.py`, P0 runner `cell --strip`)

## 6. 하드웨어와 결과 분리

- confirmatory는 L40S 결과를 쓴다. 기존 4090 결과는 교차 하드웨어 재현성 보조로만 쓴다. 두 결과를 같은 표에 섞지 않는다.
- L40S 카드 간 재현: 같은 cell을 다른 UUID의 L40S 두 장에서 실행하고 bitwise 비교한다(`scripts/compare_p0_gate.py l40s-card`).
- L40S vs 4090: 같은 원본 cell의 차이를 측정만 하고 bitwise를 기대하지 않는다(`scripts/compare_p0_gate.py l40s-4090`). 자동 허용치는 없다.
- 물리 GPU 0(L40)은 기본 제외한다. 같은 seed의 expand와 비교군은 같은 GPU 모델에서 실행한다.

## 7. 결과 노출 (연구자 확인 필요)

> 연구자가 작성한다.

작성 중 확인한 사실(점수는 열지 않음): DC에 이 claim과 같은 grid의 4090 G4 결과가 이미 있다. `artifacts/multimodal_runs/g4/` 224 run, `artifacts/multimodal-g4-report*`, `artifacts/multimodal-g4-statistics.json`이다. G5 routing 결과와 진행 중인 `multimodal_runs/p0_strip`도 있다. 로컬에는 `MULTIMODAL_RESEARCH_SUMMARY.md`(미추적)가 있다.

## 8. 미해결 TBD 목록

| 항목 | 이유 |
|---|---|
| VQA 배경 인용 | 점수 파일을 열지 않음 |
| primary 비교군 | 연구 결정. G3 lock 미열람 |
| 1차 지표, paired 단위, 검정, 임계값 | 사용자 지시에 따라 후보만 제시 |
| confirmatory 평가 split | G4는 `eval`, P0 runner는 `validation` 고정. validation은 G1–G3 개발에 사용됨(`scripts/evaluate_multimodal_g3.py`) |
| L40S 실행 범위(224 전체 / 부분) | 연구 결정. 용량·시간 추정은 이전 보고 참조 |
| 결과 노출 | 연구자만 확인 가능 |
| manifest task 정의 일치 | 서버 manifest 내용 미열람 |

## 9. 실행 provenance

`scripts/p0_strip_runner.py`는 매 실행 report에 Git HEAD commit, prereg 파일 SHA-256, 실행 파일별 SHA-256, GPU 모델·UUID, 노드, 환경 hash를 기록한다. Git이 없는 사본에서는 commit을 `null`로 남기고 사유를 함께 기록한다. lock 강제는 `--require-prereg-lock`으로 켠다(기본 꺼짐).
