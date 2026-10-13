# 보존 기록: 초기 LLM-side SEMA 변형의 VQA POC 결과

**해석 정정:** 아래 수치와 원본 산출물은 그대로 보존하지만, 이 실험을 source-faithful SEMA 검증으로 해석하지 않는다. 적용 위치를 ViT에서 LLM으로 바꿨고, 원본이 확장 없이도 분류 head를 계속 학습하는 경로(`models/sema.py:81–83,205–207`)를 VQA에 대응시키지 않아 전체 readout까지 동결했다. 따라서 AF=0과 후속 학습 중단은 원래 SEMA의 필연적 동작이 아니다. 현재 vision-side 구현과 실험 절차는 [README.md](README.md)를 참조한다.

**최신 결과는 별도다:** Vision-side74개 실험과 후속 readout8-seed/128개 학습·160개 routing 조건의 결과는 [README.md](README.md), 최종 종료 판정은 [VQA_CLOSURE.md](VQA_CLOSURE.md)에 기록했다. 이 문서의16개 LLM-side 실험 수치를 새 결과와 합치거나 source SEMA의 결론으로 재사용하지 않는다.

## 결론

**현재 이식 방식과 고정 실험 조건에서는 성능 개선 가설을 지지하지 못했다.** 계획한 16개 실행을 완료했고, 520개 prediction 파일의 66,560개 예측을 원본 manifest의 정답으로 재채점했다. SEMA식 모델은 두 protocol 모두 순차학습보다 최종 평균 성능이 크게 낮았다. 자동 확장이 발생하지 않아 첫 task 이후 학습이 중단되었으며, 이 경우 AF=0은 성공적인 continual learning의 증거가 아니다.

이는 SEMA 전체 또는 모든 expert-expansion 접근의 불가능성을 입증한 결과가 아니다. **아래에 명시한 multimodal 이식, 데이터 크기, 학습량, threshold에 대한 부정적 POC 결과**다. 결과를 본 뒤 threshold나 표본을 바꾸어 성공 수치로 대체하지 않았다.

## 무엇을 검증했는가

- [SEMA 논문](https://arxiv.org/abs/2403.18886v3)과 [공식 코드](https://github.com/huiyiwang01/SEMA-CL)의 functional adapter, reconstruction descriptor(RD), frozen old experts, learned routing, shallow-to-deep expansion을 가져왔다. 원래 ViT에 넣던 모듈을 frozen Vicuna의 마지막 3개 MLP에 이식했다. 따라서 원논문 재현이 아니라 **SEMA-derived VQA adaptation**이다.
- [CL-MoE 논문](https://arxiv.org/abs/2503.00413v1)과 [공식 코드](https://github.com/ECNU-ICALK/CL-MoE)의 VQAv2 task 분할·순서·exact-match 평가를 사용했다. CL-MoE의 dual-momentum 학습 자체를 구현하거나 비교한 실험은 아니다.
- [SMoLoRA 논문](https://arxiv.org/abs/2411.13949v2)과 [공식 코드](https://github.com/Minato-Zackie/SMoLoRA)의 공개 stream 중 VQA task만 사용했다. Flickr30k/ImageNet과 instruction-following forgetting은 이번 실험 범위가 아니다. SMoLoRA 방법 자체와의 성능 비교도 아니다.
- 논문 hash, 코드 commit, 데이터 revision: [`configs/sources.json`](configs/sources.json). 이 과거 이식안의 차이와 재실행 명령: [`LEGACY_LLM.md`](LEGACY_LLM.md).

### 고정 조건

| 항목 | 실제 실행 조건 |
|---|---|
| Backbone | Vicuna-7B-v1.5 + CLIP ViT-L/14-336 + LLaVA stage-one pretrained projector; 모두 frozen |
| 정밀도 | Backbone bfloat16, adapter/RD/router float32 |
| Adaptation | 마지막 3개 language MLP, functional bottleneck 16, RD dimension 128 |
| Functional 학습 | AdamW, LR 2e-4, task당 1 epoch, effective batch 16 |
| RD 학습 | SGD, LR 0.01, 20 epochs, batch 32, statistics buffer 500 |
| Detection | 공식 코드의 absolute z-score, batch 128, threshold 1.0 |
| 데이터 | task당 train 512 QA turn / eval 128 question, subset seed 2026 |
| 반복 | sequential/SEMA: model seed 17·29·43; ablation: seed 17 |
| 생성 | Greedy, 최대 32 tokens, inference task ID 없음 |
| Replay | 없음 |
| 하드웨어 | SLURM 할당 RTX 4090, run당 GPU 1개 |

**Protocol A — VQAv2 question-type CL:** recognition → location → judge → commonsense → count → action → color → type → subcategory → causal.

**Protocol B — VQA dataset stream:** ScienceQA → TextVQA → GQA → VQAv2. 공개 stream에서 VQA가 아닌 task를 제외한 순서이며, ScienceQA는 이미지가 있는 예제만 선택했다.

모든 방법이 동일한 protocol별 manifest를 사용했다. Functional adapter와 routing은 image/question prefix만 사용하며, 정답 token은 descriptor와 routing의 입력에서 제외했다.

## 주 비교 결과

AP는 마지막 stage에서 모든 task의 score를 평균한 값이다. AF는 마지막 task를 제외하고 `학습 직후 score − 마지막 stage score`를 평균한 signed 값이다. 낮을수록 좋으며 음수도 가능하다. BWT = −AF. `±`는 **3개 seed의 sample standard deviation**이지 confidence interval이 아니다.

| Protocol | 방법 | AP ↑ (%) | AF ↓ (pp) | MAP ↑ (%) | 최종 layer별 expert 수 |
|---|---|---:|---:|---:|---|
| VQAv2 10-task | Sequential adapter | **27.40 ± 0.33** | 2.86 ± 1.14 | 23.26 ± 0.76 | [1, 1, 1] |
| VQAv2 10-task | SEMA-derived | 4.61 ± 0.16 | 0.00 ± 0.00 | 3.19 ± 0.16 | [1, 1, 1] |
| VQA dataset stream | Sequential adapter | **29.84 ± 0.46** | 4.07 ± 0.10 | 37.51 ± 0.25 | [1, 1, 1] |
| VQA dataset stream | SEMA-derived | 14.70 ± 0.23 | 0.00 ± 0.00 | 27.29 ± 0.22 | [1, 1, 1] |

SEMA − sequential의 paired-seed AP 차이:

- VQAv2: **−22.79 ± 0.24 pp**. 각 seed 차이 −23.046875, −22.734375, −22.578125 pp.
- Dataset stream: **−15.14 ± 0.64 pp**. 각 seed 차이 −15.781250, −14.492188, −15.136719 pp.

미리 정한 판정 조건은 두 protocol 모두에서 평균 AP 증가와 평균 AF 감소를 함께 얻는 것이었다. **두 protocol 모두 이 조건을 충족하지 못했다.** AF만 선택해 긍정적 결론을 내리지 않는다.

### Ablation: 낮은 AF가 실제 학습을 의미하는가

`no_expansion`은 첫 task만 학습한 뒤 전체 adapter를 동결한다. `task_expansion`은 task 경계마다 3개 layer 모두에 expert를 하나씩 추가하고 새 expert/router를 학습한다. 두 ablation은 seed 17 한 번씩이므로 다중 seed 결론이 아니다.

| Protocol | 방법 | AP (%) | AF (pp) | 최종 layer별 expert 수 | Adapter system parameter 수 |
|---|---|---:|---:|---|---:|
| VQAv2 10-task | No expansion / frozen | 4.53 | 0.00 | [1, 1, 1] | 3,576,243 |
| VQAv2 10-task | Task-boundary expansion | 3.52 | 0.00 | [10, 10, 10] | 35,762,430 |
| VQA dataset stream | No expansion / frozen | 14.57 | 0.00 | [1, 1, 1] | 3,576,243 |
| VQA dataset stream | Task-boundary expansion | 15.04 | −0.36 | [4, 4, 4] | 14,304,972 |

Parameter 수는 functional adapter·RD·router의 전체 저장 parameter를 센 것이며 backbone은 제외한다. 현재 학습 가능한 parameter 수와 같지 않다. 공통 module 구조상 sequential/no-expansion에도 사용하지 않는 RD가 포함된다.

자동 확장 모델은 frozen 대조군과 유사한 결과를 냈다. 강제 확장은 실제로 expert를 늘렸지만 순차학습의 AP를 회복하지 못했다. **이 조건에서는 capacity를 추가하는 것만으로 문제를 해결하지 못했다.** 강제 확장의 낮은 AF도 낮은 신규 task 성능과 함께 해석해야 한다.

## 실패 양상의 실측 근거

### 1. 자동 확장 detector가 한 번도 발화하지 않았다

6개 SEMA 실행에서 총 108개 layer/task 판단, 432개 detection-batch score를 기록했다.

| Protocol | 전체 seed의 batch score 범위 | Threshold | 실제 확장 |
|---|---:|---:|---:|
| VQAv2 10-task | 0.62745–0.87221 | 1.0 | 0회 |
| VQA dataset stream | 0.58706–0.93946 | 1.0 | 0회 |

각 실행은 `[1,1,1]` expert로 끝났다. 기존 expert가 frozen인 구현에서 확장이 없으면 두 번째 task 이후 functional update도 없다. 따라서 기존 task prediction이 유지되는 AF=0과 신규 task 학습 실패가 동시에 나타난다. Threshold 미달은 관측 사실이지만, representation·descriptor calibration·학습량 중 무엇이 근본 원인인지는 이 실험만으로 분리할 수 없다.

### 2. 순차학습에는 실제 task별 forgetting이 있었다

VQAv2 sequential의 3-seed 평균이다. 평균 AF는 일부 task의 positive transfer와 상쇄되므로, task별 하락도 함께 본다.

| Task | 학습 직후 (%) | 최종 (%) | 하락 (pp) |
|---|---:|---:|---:|
| judge | 56.77 | 46.88 | 9.90 |
| count | 22.40 | 13.02 | 9.38 |
| action | 45.57 | 35.94 | 9.64 |
| color | 45.83 | 31.51 | 14.32 |

반면 SEMA의 최종 judge/action score는 모두 0%, color는 0.78%였다. 이는 이미 습득한 높은 성능을 보존한 결과가 아니다. 모든 task의 전체 triangular matrix는 각 run의 `results.json`과 집계 JSON에 보존했다.

### 3. Dataset stream에서는 첫 task 보존과 이후 task 부진이 함께 나타났다

최종 stage의 3-seed 평균:

| Task | Sequential (%) | SEMA-derived (%) |
|---|---:|---:|
| ScienceQA | 37.24 | **49.22** |
| TextVQA | **25.08** | 4.64 |
| GQA | **26.56** | 1.04 |
| VQAv2 | **30.47** | 3.91 |

첫 task인 ScienceQA를 보존했지만 다른 VQA task를 배우지 못했다. AP와 AF를 함께 확인해야 하는 이유다.

## 평가 정의와 해석 한계

1. **전체 benchmark 재현이 아니다.** task당 512/128의 고정 부분집합, 한 task 순서, 주 비교 3 seeds다. CL-MoE·SEMA·SMoLoRA의 논문 표와 직접 비교하지 않는다.
2. **원래 SEMA의 ViT 표현이 아니다.** 마지막 3개 LLM MLP와 image/question prefix pooling으로 이식했다. 논문의 signed z-score가 아니라 공식 코드의 absolute z-score를 선택했다. 세부 adaptation 목록은 README에 있다.
3. **Score 종류가 다르다.** VQAv2는 공개 benchmark 코드의 case-insensitive 단일 정답 exact match이며 official VQA consensus accuracy가 아니다. TextVQA는 10개 정답의 EvalAI consensus, GQA는 source exact match, ScienceQA는 source choice parsing을 사용했다. Dataset stream AP는 이 task score들의 macro average다.
4. **학습 노출량을 완전히 맞춘 실험이 아니다.** SEMA는 확장 시 layer마다 추가 functional pass를 허용하므로 최대 3 passes/task, 다른 비교군은 1 pass/task다. 이번 자동 확장에서는 반대로 task 0 이후 pass가 0이었다. AP 차이를 expert 구조만의 인과 효과로 분리할 수 없다.
5. **데이터 변환이 있다.** 공개 conversation의 모든 QA pair를 분해해 sampling하되 이전 turn context는 생략했다. 평가에도 square padding을 적용했고 생성 길이는 32로 제한했다. CL-MoE source evaluation의 center crop/더 긴 생성 설정과 다르다.
6. **독립 이미지 분할을 주장하지 않는다.** 경로/ID overlap은 0이지만 byte SHA-256 기준 train/eval 동일 이미지는 VQAv2 1종(3 row pairs), dataset stream 8종(13 row pairs)이었다. 동일한 question 문자열은 이 pair에서 0개지만 semantic QA 독립성을 보장하지 않는다. 원본 공식 split과 고정 sample을 유지했다. [`image-overlap-audit.json`](artifacts/image-overlap-audit.json).
7. **Bitwise determinism은 강제하지 않았다.** seed를 고정했지만 방법 간 첫 task training loss에 작은 CUDA 수치 차이가 있었다. 같은 seed의 frozen 대조군과 SEMA 수치가 완전히 일치하지 않는 점을 숨기지 않는다.
8. **현재 결과는 후속 multimodal/instruction-following 성능을 말해주지 않는다.** 해당 task나 지표를 실행하지 않았다.

다음 연구에서는 현재 평가 표본에 맞춰 threshold를 낮추는 대신, 별도의 train-only calibration/validation과 새 고정 평가 계획으로 representation·detector·학습량을 분리해야 한다. 이는 이번 결과에서 검증된 해결책이 아니라 후속 실험 설계상의 요구다.

## 실행 및 검증 증거

- **16/16 완료**, 520개 prediction 파일, **66,560개 예측 재채점**. ID 누락·중복, manifest hash, 계획한 run 조합, 재계산 score와 기록 matrix의 일치를 확인했다.
- 모든 run의 Python/config source fingerprint가 동일하며 현재 repository 파일의 hash와 일치한다. 모든 matrix의 정확한 triangular shape를 별도로 확인했다.
- Unit/regression tests: **59 passed**. 실제 GPU smoke에서 parameter update, RD loss 감소, dynamic checkpoint 복원, cached/uncached generation 일치를 확인했다. Smoke의 정확도는 연구 결과에 포함하지 않았다.
- 초기 array **2444862**에서 `node40`에 배정된 index 7·8·9·10·11은 학습 시작 전 CUDA GPU 오류로 실패했다. 해당 node를 제외하고 **2445312**에서 이 5개만 같은 코드·설정으로 완료했다. 실패 로그도 보존했다. 성공한 실험을 다시 골라 실행하지 않았다.
- 16개 최종 adapter checkpoint, 원본 예측, event logs, config, source hash를 local `artifacts/runs/`에 모두 저장했다. Backbone/data는 remote `/scratch2/faransansj/CLMoE-VQE/`에 보존했다.

주요 산출물:

- [`artifacts/report/summary.csv`](artifacts/report/summary.csv): 16개 run의 원 수치.
- [`artifacts/report/summary.json`](artifacts/report/summary.json): 전체 matrix, paired-seed 차이, 사전 판정 결과.
- [`artifacts/report/verification.json`](artifacts/report/verification.json): 완료·재채점·source 일치와 detector 통계.
- `artifacts/runs/<protocol>/<method>/seed<seed>/`: `latest.pt`, `results.json`, `events.jsonl`, `predictions_after_*.jsonl`, `config.json`, `code_sha256.json`.
- `artifacts/manifests/`: 선택한 실제 데이터 ID, 정답, source hash, 이미지 검증 결과.
- [`artifacts/environment-freeze.txt`](artifacts/environment-freeze.txt): 실제 실행 환경.

**연구 판단:** 실행 가능한 VQA continual expert-expansion 실험과 대조군을 구축하고 실제 forgetting을 측정했지만, 이식한 SEMA 조건에서 성능이 좋다는 주장은 성립하지 않는다. 자동 detector의 비발화와 강제 확장의 낮은 신규 task 성능을 모두 해결·재검증하기 전에는 긍정적 성능 근거로 사용할 수 없다.
