# 보존 기록: SEMA → LLM-side Continual VQA POC

이 문서는 이전 구현의 재현 기록이다. VQA readout을 전부 동결한 설계는 원본 SEMA의 no-expansion 학습 경로와 다르며, 아래 결과를 원래 mechanism의 검증으로 사용하지 않는다. 현재 vision-side 구현은 [README.md](README.md)에 있다.

## 연구 질문과 범위

Image continual learning의 SEMA expert expansion이 VQA continual learning에서도 **정확도를 높이고 catastrophic forgetting을 줄이는가?**

현재 단계는 VQA만 다룬다. VQAv2 내부 question-type CL과, VQAv2가 하나의 task인 VQA 데이터셋 간 CL을 별도로 실행한다. 분류·captioning을 섞는 전체 multimodal continual learning은 이번 실험의 결과가 나온 뒤 판단할 후속 범위다.

실험 조건은 `configs/poc.json`, 논문 SHA-256과 공식 저장소 commit은 `configs/sources.json`에 고정한다. 긍정적인 결과를 전제하지 않는다. Test 결과로 threshold나 checkpoint를 고르지 않는다.

## 실제 결과

**16개 실행 완료; 현재 조건에서는 가설을 지지하지 못했다.** 66,560개 예측을 재채점했다. 아래 값은 주 비교 3-seed 평균이다.

| Protocol | Sequential AP (%) | SEMA-derived AP (%) | Sequential AF (pp) | SEMA-derived AF (pp) |
|---|---:|---:|---:|---:|
| VQAv2 10-task | 27.40 | 4.61 | 2.86 | 0.00 |
| ScienceQA → TextVQA → GQA → VQAv2 | 29.84 | 14.70 | 4.07 | 0.00 |

SEMA 6개 실행 모두 자동 확장이 없었다. **AF=0은 첫 task 이후 동결된 결과이며 성공적인 continual learning의 증거가 아니다.** 강제 task-boundary expansion도 AP가 각각 3.52%, 15.04%로 순차학습보다 낮았다. 전체 해석·표준편차·ablation·한계는 [RESULTS.md](RESULTS.md), 원 수치는 [summary.csv](artifacts/report/summary.csv)에 있다.

## 근거

| 역할 | 논문 / 공식 코드 | 실제 사용 부분 |
|---|---|---|
| Source method | [SEMA, 2403.18886v3](https://arxiv.org/abs/2403.18886v3), [SEMA-CL](https://github.com/huiyiwang01/SEMA-CL) | Functional adapter, linear AE descriptor, z-score expansion, frozen previous experts/router, shallow-to-deep expansion |
| VQAv2 CL 평가 | [CL-MoE, 2503.00413v1](https://arxiv.org/abs/2503.00413v1), [CL-MoE](https://github.com/ECNU-ICALK/CL-MoE) | 공개 CL4VQA train/test 분할, 10-task 순서, stage-one 초기화, VQAv2 evaluator, AP/AF |
| Dataset-level VQA 순서와 후속 범위 | [SMoLoRA, 2411.13949v2](https://arxiv.org/abs/2411.13949v2), [SMoLoRA](https://github.com/Minato-Zackie/SMoLoRA) | 공개 single-instruction 데이터와 task별 evaluator; upstream에서 VQA task만 선택 |

`third_party/`는 공식 소스 snapshot이다. 기존 launcher는 연구 출처로 보존하며 직접 실행하지 않는다. 이번 runner는 `sema_vqa/`에 있다. **CL-MoE의 dual-momentum MoE나 SMoLoRA 자체를 재현했다는 의미가 아니다.**

## 방법과 명시적인 이식 차이

- Backbone: `lmsys/vicuna-7b-v1.5`, `openai/clip-vit-large-patch14-336`, `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` projector. 이미 VQA instruction tuning된 LLaVA checkpoint로 대체하지 않는다.
- Vision/language backbone와 projector는 동결한다. 마지막 **language transformer MLP 3개** 옆에 source-derived down-projection → ReLU → up-projection adapter를 둔다. Bottleneck 16, up-projection은 zero initialization. 이는 ViT에서 multimodal language representation으로의 연구 이식이다. 앞선 층에 별도 adapter는 두지 않는다.
- Frozen backbone은 bfloat16, trainable adapter/RD/router와 optimizer update는 float32를 사용한다. 특히 RD의 작은 gradient가 저정밀 가중치에서 사라지는 위험을 피한다.
- 이미지는 source training의 square padding을 train/eval 모두에 적용한다. 공개 CL-MoE eval utility의 직접 center crop과는 다른, 명시적인 일관 preprocessing 선택이다.
- Descriptor는 linear encoder/decoder AE, hidden dimension 128. Image/question prefix의 masked mean을 reconstruct한다. Input은 detach하여 RD loss가 functional adapter나 backbone을 바꾸지 못하게 한다.
- Router는 prefix mean에 대한 linear logits와 softmax weighted mixture다. 답변 토큰과 padding은 pooling에서 제외한다. Teacher-forced 정답이 expansion 또는 routing의 입력으로 새어 들어가지 않게 한다.
- 공식 `sema_block.py`는 논문에 적힌 signed z-score와 달리 **absolute z-score의 sample 평균**, 그다음 expert 간 minimum을 사용한다. 이 POC는 **공식 코드의 reduction**을 사용한다. Threshold는 1.0이며 test로 조정하지 않는다.
- 기존 adapter/descriptor/router row를 동결하고 새 row만 학습한다. Forward 안에서 모델을 바꾸지 않고 runner가 expansion과 optimizer 재구성을 담당한다.
- 각 task에서 층당 최대 한 번 추가한다. 얕은 층 추가를 학습한 뒤 깊은 층을 검사한다. 이전 task 데이터는 replay하지 않는다.
- Source의 functional/RD 별도 학습을 따른다. Functional은 VQA CE와 AdamW, RD는 SGD와 cosine schedule이다. Functional이 고정된 시점에 현재 task의 pooled feature만 캐시하여 RD를 학습한다. Task가 끝나면 feature cache를 재사용하지 않는다.
- Rolling RD 통계는 float32, unbiased standard deviation, 길이 500이다. Zero variance에는 epsilon을 적용한다. Source의 초기 buffer overflow 누락은 수정했고, 동적 expert 구조·통계·동결 상태를 checkpoint에 보존한다.
- 생성 시 prefix router context와 language KV를 캐시한다. 실제 이미지 smoke에서 uncached 생성과 같은 답을 확인했다.

### 비교군

1. `sequential`: 같은 단일 adapter를 모든 task에서 계속 학습.
2. `sema`: RD가 감지한 층만 확장; 기존 expert 동결.
3. `no_expansion`: 첫 task의 adapter를 학습한 뒤 전부 동결. 낮은 forgetting이 학습하지 않은 결과일 수 있음을 확인하는 대조군.
4. `task_expansion`: 매 task마다 모든 대상 층에 expert 하나를 강제 추가하고 학습.

**Compute confound:** source 방식의 SEMA는 추가된 층마다 현재 task를 다시 학습하므로 최대 3번의 functional pass를 수행한다. 다른 비교군의 pass 수는 1이다. 따라서 긍정적인 POC도 capacity와 추가 optimization 효과를 완전히 분리하는 compute-matched 인과 검증은 아니다. 로그의 phase/step와 wall time을 함께 해석해야 한다.

## 데이터와 평가

### A. `cl4vqa`

공식 CL4VQA 분할을 그대로 사용한다.

`recognition → location → judge → commonsense → count → action → color → type → subcategory → causal`

### B. `vqa_stream`

SMoLoRA 공개 upstream 순서에서 VQA만 남긴다.

`ScienceQA → TextVQA → GQA → VQAv2`

ScienceQA는 image-bearing example만 사용한다. 이는 **전체 SMoLoRA benchmark 재현이 아닌 VQA-only 파생 프로토콜**이다. 원래 후속 평가용 VizWiz/OCRVQA를 임의로 upstream에 추가하지 않는다.

### 고정 POC 표본

- Task당 최대 train 512 QA turn, eval 128 question. Subset seed 2026.
- Model seed 17/29/43에서 `sequential`과 `sema`를 비교한다. Ablation 두 개는 seed 17.
- 모든 방법과 model seed가 동일한 train/eval ID를 사용한다.
- Random seed를 고정하지만 CUDA 연산의 bitwise determinism을 강제하지 않는다. 동일 seed의 방법 간 첫 task training loss에 작은 수치 차이가 관측되었다.
- 공개 multi-turn conversation은 모든 human/assistant pair로 분해한 다음 sampling한다. 첫 pair만 취하지 않지만 이전 turn의 대화 context는 유지하지 않는 명시적인 POC 변환이다.
- 질문의 OCR 문자열과 공개 task assignment를 보존한다. 새로운 question-type taxonomy를 만들지 않는다.
- Manifest에 원본 hash, 전체/선택 개수, source ID, turn ID, train/eval image overlap을 기록한다. 누락 이미지나 정답을 합성하지 않는다.
- `--train-limit 0 --eval-limit 0`은 전체 eligible 데이터 준비를 지원한다. 현재 POC 수치는 전체 benchmark 또는 논문 표의 수치와 직접 비교할 수 없다.
- 검증된 이미지: `cl4vqa` 6,042개, `vqa_stream` 2,528개. 모든 선택 예제가 2,048-token 이내다. 경로/ID 중복은 0이나, **동일 바이트 이미지의 train/eval 중복은 각각 1종/8종**이다. 원본 분할을 변경하지 않았으며 이미지 독립성을 주장하지 않는다. 상세 pair는 `artifacts/image-overlap-audit.json`, 전체 이미지 hash는 `artifacts/manifests/*.validation.json`에 있다.

### 지표

행 `i`는 task `i` 학습 직후, 열 `j`는 평가 task다. 모든 seen task를 매 stage 평가한다. `results.json`의 matrix는 **0–1**, 요약 metric은 **percent / percentage point**다.

- AP: 마지막 행의 task 평균.
- AF: 이전 task들에 대한 `diagonal accuracy − final accuracy` 평균. 음수도 허용하며 max-forgetting으로 바꾸지 않는다.
- BWT: `−AF`.
- MAP: 각 stage의 seen-task 평균을 다시 평균.
- 단일 task의 AF/BWT는 코드에서 0으로 정의한다. 최종 판정은 multi-task 결과로만 한다.

Task별 scorer도 구분한다.

| Task | Scorer |
|---|---|
| CL4VQA, SMoLoRA VQAv2 | 공개 evaluator 그대로 대소문자를 무시한 단일 정답 exact match. 일반적인 VQA consensus가 아님 |
| TextVQA | 공식 10개 annotator 답 + vendored EvalAI normalization/leave-one-annotator-out consensus |
| GQA | 공개 conversion의 `rstrip('.').lower()` 후 original balanced-question 정답과 비교 |
| ScienceQA | 공개 option-letter parser와 original choices/answer index. Invalid는 오답; random fallback 없음 |

`scripts/summarize.py`는 최종 수치를 모으기 전에 **모든 stage의 prediction ID coverage와 재계산 점수**를 검증한다. 누락 prediction을 제외해서 accuracy를 올리는 동작은 허용하지 않는다.

## 실행

### 설치

Python 3.12 환경에서 먼저 CUDA wheel을 설치한 뒤 dependency를 설치한다.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt
```

### 모델·데이터 준비

Yonsei에서는 대용량 다운로드를 **login node에서만** 실행한다. 아래 `ROOT`는 본인 ID의 scratch 경로로 설정한다.

```bash
ROOT=/scratch2/faransansj/CLMoE-VQE
.venv/bin/python scripts/download_models.py --output "$ROOT/models"
.venv/bin/gdown --folder https://drive.google.com/drive/folders/1mcAjzmCU1UVW0TKvsHAy9Sr1hmwsJudo -O "$ROOT/CL4VQA"
.venv/bin/python scripts/prepare_data.py --root "$ROOT" --protocol cl4vqa --download-images
.venv/bin/python scripts/prepare_data.py --root "$ROOT" --protocol vqa_stream --download-images
```

선택 이미지의 ZIP range extraction을 우선 사용한다. 서버가 range를 지원하지 않으면 전체 archive 다운로드를 명시적으로 출력한다. COCO의 공식 HTTP host를 사용하며 전역 TLS 검증을 끄지 않는다. 모델 revision은 `models/sources.json`, 원본 dataset fingerprint는 manifest에 남는다. NAS retention 지침에 따라 다운로드 후 atime을 갱신해야 한다.

### 검증·학습

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
sbatch scripts/smoke.sbatch
sbatch scripts/experiments.sbatch
```

SLURM만 GPU를 배정한다. `CUDA_VISIBLE_DEVICES`를 직접 설정하지 않는다. Launcher 경로는 현재 Yonsei 계정 기준이며 다른 환경에서는 SBATCH 경로를 수정한다. `cs-gpu-01`과 `node40`에서는 할당 후 CUDA 초기화가 실제 실패하여 해당 node를 제외했다. 본 실험 array `2444862`의 실패 index `7,8,9,10,11`은 학습 시작 전 GPU 오류였으며, 이 5개만 동일 Python 코드·설정으로 다시 제출했다.

Array는 16개 실행이며 동시 GPU job 수는 최대 4다. Task 완료 checkpoint가 있으면 같은 code/config/manifest로 task 경계에서 resume한다. 실패 중간 task는 마지막 완료 checkpoint부터 다시 수행한다.

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize.py \
  --runs "$ROOT/runs" --output "$ROOT/report" --plan configs/poc.json
```

## 검증 상태

- 실제 실행한 unit/regression tests: **59 passed**.
- 현재 float32-adapter VQAv2 GPU smoke: SLURM job **2444813**, RTX 4090, peak allocated memory **14.63 GB**. Functional parameter update, dynamic checkpoint restoration, cached/uncached generation equality를 확인했다. 실제 이미지 feature에서 3개 RD loss가 모두 감소했다 (`artifacts/smoke-fp32-result.json`).
- Smoke의 단일 질문은 학습 전후 모두 오답이었다. 이를 정확도 향상 근거로 사용하지 않는다.
- 초기 bfloat16-adapter recognition→location transition smoke: job **2444642**, task당 train 4/eval 2. RD 학습 뒤 두 번째 task의 z-score가 threshold 1.0을 넘어 expert가 `[1,1,1] → [2,2,2]`로 증가했다. 두 stage의 AP는 0으로, 기능 실행만 입증하며 성능 결론에 포함하지 않는다. 최종 실험은 trainable module을 float32로 유지한다.
- 본 실험 **16/16 완료**: SLURM arrays `2444862`, `2445312`. 520개 파일의 **66,560개 prediction**을 원본 정답으로 재채점했고 모든 기록 matrix와 일치했다. 모든 run이 동일한 Python/config source hash를 사용했다. `artifacts/report/verification.json`에 완료·source·detector 검증을 기록했다.

실행 증거와 16개 최종 adapter checkpoint는 local `artifacts/runs/`, 최종 집계는 `artifacts/report/`에 보존했다. Remote 원본은 `$ROOT/runs`, 집계는 `$ROOT/report`에 있다. 연구 결론은 [RESULTS.md](RESULTS.md)에 기록했다.
