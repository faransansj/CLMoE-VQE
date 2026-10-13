# Heterogeneous vision-language continual learning 설계

작성: 2026-10-13. 상태: **G0 실행 경로 구현 및 실제 GPU 기술 smoke 검증 완료; 전체 source gate는 Flickr30k 접근 권한으로 차단**. 브랜치: `research/multimodal-cl`. VQA 종료 기준선: `main`의 `47efefe`. 실험 조건·진행 gate와 실행 기록은 [MULTIMODAL_EXPERIMENTS.md](MULTIMODAL_EXPERIMENTS.md), 기존 근거는 [VQA_CLOSURE.md](VQA_CLOSURE.md)를 따른다.

## 1. 연구 질문과 범위

VQA 자체가 multimodal이다. 이번 확장은 modality 수를 늘리는 것이 아니라 **서로 다른 이미지–텍스트 task family를 연속 학습하는 CL**이다. VQA와 captioning을 먼저 다루며 audio/video, retrieval, 모델 크기 확대, adaptive growth는 이번 범위 밖이다.

질문: 동일 초기화·데이터·학습 노출에서 readout expert 확장/동결/routing이 공유 adapter와 replay보다 task 간 망각을 줄이고 최종 성능 및 신규 task 학습을 유지하는가? 보존과 positive transfer는 별도 가설이다. Task별 독립 학습보다 새 task를 더 잘 배우지 못하면 positive transfer를 주장하지 않는다. 단순한 LoRA/MoE 조합 자체의 새로움도 주장하지 않는다.

## 2. 하나의 모델 경로를 유지한다

기존 `ReadoutMoEVQA`가 사용하는 **동일 pure-pretraining BLIP checkpoint**와 vision encoder → text encoder → answer decoder 경로를 유지한다. Caption 전용 pretrained decoder를 교체하거나 두 개의 backbone을 병렬로 붙이지 않는다.

- VQA 입력: image + 원 question/context/options. 출력: answer.
- Caption 입력: image + 고정 instruction `Describe the image.`. 출력: caption.
- 이미지 224px, tokenizer 및 normalization은 두 family와 모든 방법에서 동일하다.
- Rank8/alpha16 LoRA를 질문 encoder·답변 decoder의 self/cross-attention query/value에 적용한다. 공유 backbone과 기존 expert는 고정하며 새 expert만 학습한다.
- 새 task마다 zero-delta expert를 추가하는 unconditional expansion을 주 방법으로 한다. Old expert 복사 초기화, shared+private expert, adaptive growth는 추가하지 않는다.
- 기존 학습 task 경계는 제공한다. 평가 시 stream task index, dataset ID, 정답을 모델/router에 주지 않는다. 요청에 포함된 question/caption instruction은 합법적 입력이다. 따라서 task-family-blind 추론이라고 부르지 않는다.

**미검증 가정:** BLIP의 VQA 경로에 고정 질문을 넣는 방식이 captioning을 충분히 학습할 수 있다. 원래 caption 모델과 conditioning 경로가 같다는 주장은 하지 않는다. 독립 caption 학습 gate가 실패하면 CL 실험을 진행하지 않고 원인을 진단한다. Backbone 변경이 필요하면 새로운 계획 revision과 모든 기준선 재실험이 필요하다.

## 3. 현재 코드와 구현 경계

| 현재 구현 | 확인한 계약 | 후속 구현 결정 |
|---|---|---|
| `sema_vqa/readout_model.py`: `ReadoutMoEVQA`, `LoRALinear` | expert append, old expert 동결, top1 선택, backbone 공유 | 재사용. 기존 VQA checkpoint/호출 계약은 변경하지 않음 |
| `sema_vqa/vision_model.py`: `_question_states`, `loss`, `answer` | question→image cross-attention, answer token-loss 합, 32-token greedy | 새 task-aware 경로에서 손실/생성만 명시적으로 override. 기존 동작 불변 |
| `sema_vqa/readout_router.py`: `FeatureRouter` | frozen feature ridge, task당128 train features, joint/image/question | 동일 router 사용. 모든 평가에서 target/metadata 제거된 view만 전달 |
| `sema_vqa/readout_run.py`: `train_phase`, `routed_answers`, `evaluate` | QA row, VQA per-example scorer, task matrix, config/source hashes | 새로운 runner가 task adapter 및 corpus scorer를 사용. 기존 CLI에 암묵적 caption 분기를 넣지 않음 |
| `scripts/prepare_readout_data.py` | 전역 이미지 hash와 과거 split 제외 | 중복차단 원칙 재사용. Caption references와 family별 source adapter 추가 |
| `scripts/summarize_readout.py` | 예측 ID 검증, VQA 재채점 | Caption corpus scoring이 가능하도록 새 summary 경로 작성. CIDEr를 VQA scorer에 억지로 넣지 않음 |

구현 모듈은 `sema_multimodal/`의 `data.py`, `model.py`, `run.py`, `metrics.py`와 `scripts/prepare_multimodal_data.py`, `scripts/summarize_multimodal.py`다. 모델은 기존 `ReadoutMoEVQA`를 상속하여 token 평균 loss와 32/64-token 생성만 확장했다. 기존 VQA 모듈은 변경하지 않았다. Runner는 batch별 optimizer/scheduler/RNG/cursor checkpoint를 저장하며 interrupted resume를 실제 GPU에서 검증했다. 데이터 준비는 공식 source 파일과 명시적인 로컬 ZIP만 사용하며 부족/손상된 이미지를 다른 예제로 대체하지 않는다.

## 4. 데이터와 공개 입력 계약

새 manifest는 version, protocol, source revisions/hashes, split policy, task order와 ordered tasks를 포함한다. 각 task에는 `task_id`(학습/평가 집계 전용), `family`, `metric`, train/validation/eval row 목록을 둔다.

Row 필수 필드:

- `id`: source namespace를 포함한 유일 ID.
- `image`, `image_sha256`, `source_image_id`: 실제 이미지와 identity.
- `input_text`: 원 VQA question 또는 고정 caption instruction.
- `target`: 학습에 사용할 단일 문자열.
- `references`: 원 source의 전체 정답/캡션 목록. 평가 reference를 학습 target으로 가져오지 않음.
- `source`, `source_split`, `family`: provenance/집계 전용 metadata.

Model public view는 이미지와 input text만 노출한다. Teacher-forcing loss에만 target을 추가한다. Router view에는 target/references/task_id/source/family를 전달하지 않는다. Task-family에 따른 생성 token limit은 요청 형식에서 정하며 stream task index로 고르지 않는다.

Caption은 이미지당 한 row와 모든 reference captions를 보존한다. Train에서는 `(seed, epoch, image_id)`로 reference를 결정적으로 순환 선택해 여러 caption을 활용하되 이미지 exposure는 한 번으로 센다. 예측은 이미지당 하나 생성하고 모든 정답 caption으로 채점한다. VQA도 이번 후속 cohort는 이미지당 최대 한 QA로 제한하여 caption과 unique-image budget을 맞춘다. 이는 기존 VQA cohort와 다른 데이터 계약임을 명시한다.

같은 이미지의 caption과 question이 서로 다른 split이나 task에 배치되지 않도록 **이미지 identity와 실제 byte SHA256 단위로 먼저 split, 이후 task 할당**한다. 같은 split 안에서도 task 간 이미지 중복을 금지한다. 여러 source에서 후보인 이미지는 잠긴 hash 순위로 task에 한 번만 배정한다. 부족하면 중복 보충하지 않고 보고한다. Confirmation eval은 기존 모든 VQA train/validation/eval 이미지도 제외한다. 이미지 hash만으로 near-duplicate와 backbone pretraining 누출까지 제거했다고 하지 않는다.

## 5. 손실·생성·평가 계약

### 손실

기존 VQA `loss`는 BLIP의 answer token-loss 합을 예제 평균한다. Caption 길이가 더 길어 생기는 가중 편향을 피하기 위해 **후속 cohort의 모든 방법/두 family에 동일하게 예제별 유효 target token 평균 → batch 평균**을 적용한다. Shift 이후 유효 token 수를 사용하고 padding과 시작 token을 제외하며 EOS는 포함한다. Instruction은 encoder 입력이라 decoder target loss에 포함하지 않는다. BLIP smoothing은 유지한다. Microbatch 분할/gradient accumulation 후에도 effective batch의 예제 평균과 일치해야 한다.

이 변경 때문에 기존 AP와 새 AP를 직접 비교해 방법 개선량을 계산하지 않는다. 기존 VQA 경로는 그대로 남겨 archived run 재현성을 보존한다.

### 생성

Greedy, VQA 최대32/caption 최대64 new tokens, EOS 종료, batching/cache는 동일 계약. 생성 길이·EOS 미출력·빈 출력·반복·상한 도달 비율을 family별로 보고한다. 빈 caption을 임의 문장으로 치환하지 않는다. 길이 설정은 validation 전에 고정하고 eval을 보고 바꾸지 않는다.

### 평가

- VQA: 각 source의 고정 scorer. VQAv2는 확보한 official references 기반 consensus accuracy, GQA는 official normalized accuracy를 목표로 한다. 이전 VQA exact-match와 혼용하지 않는다. 평가 reference 확보가 안 되면 source/scorer gate 실패다.
- Caption: COCO-style CIDEr-D를 주 지표, BLEU-4 보조. Scorer 구현 revision/tokenizer/배율을 lock에 포함한다. CIDEr는 corpus metric이므로 개별 score 평균으로 대체하지 않는다. CIDEr-D document frequency는 task별 **train reference corpus로 미리 고정**하고 validation/eval에서 재추정하지 않는다. 따라서 standard test-corpus-IDF 수치와 직접 SOTA 비교하지 않는다.
- Corpus reference/ID 정렬과 image당 단일 prediction을 검증한다. 모델에는 reference를 주지 않는다.
- 원 단위 task matrix와 pre-to-post gain을 우선 보존한다. 서로 다른 accuracy와 CIDEr를 생으로 평균하지 않는다. 정규화·통계는 실험계획에 정의한다.

## 6. Router와 expert 상태

Frozen image/text mean feature, ridge=1.0, task당128 train vectors, seed별 deterministic sampling을 유지한다. Router 재학습은 현재까지의 train feature memory와 training task labels만 사용한다. Expert 개수에 따라 feature bank/weights/metadata byte 수를 보고하며 memory-free라고 부르지 않는다.

입력 instruction만으로 VQA/caption family가 쉽게 드러난다. 따라서 두 task pilot만으로 어려운 routing 문제를 해결했다고 주장하지 않는다. 본 평가에는 family당 두 domain task를 포함하고, image-only/question-only/family-restricted/task-ID oracle 진단으로 family 식별과 domain 선택을 구분한다. Oracle은 주 성능이 아니다. 이전 expert가 byte-identical해도 router 갱신으로 이전 task 성능은 바뀔 수 있다.

Checkpoint에는 schema version, expert bank, active expert, router memory/weights, stage/optimizer/RNG state, source/config/manifest/asset/tokenizer/scorer hashes와 학습 예제/token 수를 보존한다. Expert 확장 후 optimizer를 새로 만들고 이전 expert gradient가 없는지 검증한다. Resume는 같은 계약만 허용하고 새 결과로 이어붙이는 우회는 금지한다.

## 7. 구현 순서와 수용 조건

1. **Data/scorer:** source 접근·라이선스·reference 검증 → 이미지 기준 전역 분할 → scorer fixture와 corpus 재채점 검증. 다운로드/서버 작업은 해당 서버 지침을 읽은 뒤 수행한다.
2. **Model adapter:** single-task VQA/caption의 실제 GPU 학습과 생성. Frozen backbone/old expert 불변, new expert update, target-token loss, EOS/긴 caption을 확인한다.
3. **Runner:** effective-batch/partial replay, task matrix, router metadata 차단, interrupted checkpoint resume와 prediction parity를 실제 작은 run으로 검증한다.
4. **VQA 보호:** archived VQA config/source는 수정하지 않는다. 공통 코드 변경 시 별도 출력의 기존 VQA smoke와 checkpoint 재생 결과 parity가 필수다. 기존 artifact 경로를 출력으로 받으면 실패시킨다.
5. **Pilot/본 실험:** 아래 실험계획의 gate를 통과한 경우에만 실행한다. 기술 smoke 성공을 held-out 학습 성공으로 대체하지 않는다.

기본 출력은 새 `artifacts/multimodal_runs/`, `artifacts/multimodal_report/`로 분리한다. 기존 `readout_runs/`, `readout_report/`, vision/legacy 결과는 읽기 전용이다. DC1의 새 source/environment는 `/home/faransansj/CLMoE-Multimodal-20261013/`다. 기존 venv의 의존성은 read-only 경로로 사용하고 추가 scorer 의존성은 새 venv에 설치했다. 실제 기술 smoke는 완료했지만 captioning held-out learnability와 본 CL 성능은 아직 검증하지 않았다.
