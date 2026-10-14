# Continual VQA with expandable experts

**VQA 실험 종료:** 8-seed readout confirmation 128개 학습과 160개 routing 조건을 완료했다. 최종 판정·산출물 무결성 확인·남은 제한은 [VQA_CLOSURE.md](VQA_CLOSURE.md)에 정리했다. 아래 결과와 기존 원 산출물은 후속 연구에서 덮어쓰지 않는다.

**후속 멀티모달 CL 연구:** `research/multimodal-cl`에서 VQA/captioning 경로를 구현하고 테스트307개, 실제 GPU 학습·routing 재현·기존 VQA parity를 검증했다. 공개 HF Flickr30k를 확보한 뒤 G1~G3 개발92개 학습과 gate 판정을 완료했다. Expansion은4-task 개발에서 shared/capacity보다 높았지만 **common/tuned replay보다 낮았고, 두 Flickr 후속 전이는 음수**였다. 코드·데이터·통계를 pre-eval lock한 새8-seed/224개 confirmation 조건을 실행 중이다. [설계](MULTIMODAL_DESIGN.md), [실험계획·결과](MULTIMODAL_EXPERIMENTS.md), [설정](configs/multimodal_research.json). 기존 VQA 결과는 변경하지 않는다.

**기존 vision-side 검증 결론:** Vision-side SEMA의 실제 확장·학습은 확인했다. 다만 학습량을 맞춘 no-expansion보다 우월하다는 근거는 확보하지 못했다. 아래에 완료된74개 실험과 실패·편향 진단을 함께 기록한다.

## 후속 연구: VQA-aware readout expansion

목표를 원본 SEMA 이식 검증에서 **VQA continual learning의 실질적 성능개선과 확장 고유 효과 검증**으로 확장한다. 기존74개 실험·모델·결과는 변경하지 않는다. 계획은 [`configs/readout_research.json`](configs/readout_research.json)에 기록했다.

**후속 연구의 확인 결과:** 새 공식 평가에서8-seed/128개 학습을 완료했다. 총 adapter 파라미터와 학습 exposure를 맞춘 단일 LoRA 대비 확장군의AP가 CL4VQA에서+10.33pp, dataset stream에서+6.53pp 높았다(두 비교 모두 주 AP family의 Holm-adjusted `p=0.046875`). 단, CL4VQA의 주5-epoch 확장은 최강 validation baseline인 replay에 졌고, adaptive growth도 정확도 유지에 실패했다. 아래에 성공과 실패를 함께 기록한다.

- 구현: `sema_vqa/readout_model.py`, `readout_router.py`, `readout_run.py`. 원본 BLIP vision/readout은 동결하고, 질문 encoder·답변 decoder의 self/cross-attention query/value에 rank8 LoRA expert를 추가한다. Expert당1,179,648개 파라미터; 공유 backbone은 복제하지 않는다.
- 기존 expert는 동결한다. Frozen image/text 특징에 대한 ridge router로 추론 expert를 선택하며, 추론에 task ID·정답을 입력하지 않는다. 학습 task 경계는 주어진다. Router는 task당128개 **학습 특징 벡터**를 저장하므로 memory-free CL이 아니다.
- 현재 비교: 동일 LoRA 하나의 sequential 학습, raw-example replay, 전체 readout 학습, task별 확장, train-answer competence 기반 adaptive expansion. Task-ID oracle은 별도 진단이며 주 결과가 아니다.
- 새 공식 평가 표본은 과거 train/validation/eval에 사용한 이미지·ID를 제외해 고정한다. Fit/validation은 원본 train에서 분리하고, 두 protocol의 train/validation/eval **분할 사이**에서 이미지 SHA256 중복을 차단한다. 평가 결과를 보고 방법을 선택하지 않는다.
- 실제 확인: `pytest tests -q`에서134 tests 통과. GPU job2448303에서 실제 학습4개 예제의 loss `21.1945 → 3.2533`, 새 expert 학습 후 기존 가중치·답변 불변, 정답과 무관한 고정 routing 특징, checkpoint 생성 결과 일치를 확인했다. 증거: `artifacts/readout_smoke/result.json`. Job2448424에서는 실제 runner의 partial replay batch와 checkpoint 기반 결과 복구를, job2448449에서는 checkpoint 재사용 routing 진단의 native 예측5개 파일이 원본과 byte-identical함을 확인했다. 이 smoke들은 held-out 성능 근거가 아니다.
- 단일 expert 대조군(`shared_lora`, `replay_lora`, `frozen_first`)에는 router를 만들지 않고 특징 계산·저장과 무의미한 oracle 재평가를 생략한다. GPU job2448777에서 수정 전후 expand/replay의 원 예측10개 파일이 byte-identical했고 replay의 routing 메모리는0 bytes였다. 수정 전 소스는 `artifacts/readout-pre-single-expert-source.tar.gz`(SHA256 `ae275c43ccce3d16072b9cd22eaabbf222ec10eef4de8b2d073c214b5d6a7134`)에 보존했다. 확장 모델은 별도 frozen-feature 추출 비용이 있으므로 top1 선택을 단일 LoRA와 동일한 추론 연산량이라고 주장하지 않는다.
- 선행연구: [SMoLoRA](https://arxiv.org/abs/2411.13949), [Dynamic Routing Between Experts](https://arxiv.org/abs/2511.01831), [SAMoRA](https://arxiv.org/abs/2604.19048)에서 관련 LoRA expert·routing을 이미 다룬다. 이 조합 자체를 새로운 기여라고 주장하지 않는다.

### 새 데이터 준비

개별 HTTP 요청의 지연을 줄이기 위해 공식 COCO2014 ZIP을 사용한다. 선택된 **동일 파일**만 추출하며 candidate 순서·제외 규칙은 변경하지 않는다. `sources.json`에 원본 URL·크기·SHA256을 기록한다. 아래 다운로드는 DC1 **login node**에서만 실행한다. 기존 output directory는 덮어쓰지 않는다.

새 분할을 실제 완성했다. 14개 task 모두 fit2,048/validation256개이며, 공식 eval은 총6,620개다(CL4VQA location403, causal73; 나머지 task512). 두 protocol 전체의 분할 간 이미지 SHA256 overlap과 새 eval/과거 사용 이미지 overlap은 모두0이다. BLIP tokenizer preflight에서 최대 질문446 tokens, 512-position 위반0개, 32-token 생성 한도 초과 target0개를 확인했다. 근거는 `artifacts/readout_data_20261008/`와 `artifacts/readout-data-preflight.json`이다. 이 데이터 검증 자체는 모델 성능 측정이 아니다.

```bash
ROOT=/scratch2/faransansj/CLMoE-VQE
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.download_readout_coco \
  --output "$ROOT/cache/coco2014_official"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.prepare_readout_data \
  --root "$ROOT" --output-dir "$ROOT/readout_data_20261008" --protocol cl4vqa vqa_stream \
  --old-manifests "$ROOT/manifests/cl4vqa.json" "$ROOT/manifests/vqa_stream.json" \
    "$ROOT/vision_manifests_referenced/cl4vqa.json" "$ROOT/vision_manifests_referenced/vqa_stream.json" \
  --download-images --coco-archives "$ROOT/cache/coco2014_official"
```

TextVQA의 동일 이미지·질문에 서로 다른10인 응답 집합이 있는 실제 사례(qid16138/16139)에서 문자열 매칭이 모호해져 준비가 한 번 중단됐다. 새 경로는 pinned instruction/원본 train annotation34,602개 **전체**의 행 수·이미지·질문·기존 target 정합성을 확인한 뒤 원본 행 위치로 참조 ID를 복원한다. 실제 전체 source 검증에서 학습 target과 각10인 응답 집합이 모두 보존됐다(`artifacts/readout-reference-recovery.json`). 모호한 예제를 제거하거나 답변 집합을 합치거나 모델 예측으로 참조를 선택하지 않는다.

프로젝트 검증은 `pytest tests -q`로 범위를 고정한다. Repository 전체를 무조건 수집하면 vendored SMoLoRA의 LLaVA registry와 현재 Transformers가 충돌한다. 이를 프로젝트 테스트 통과로 숨기거나 upstream 코드를 변경하지 않았다.

### 개발용 pilot: 처음3개 task, seed17

Job2448334의10개 pilot을 완료하고14,015개 예측(oracle·학습 전 평가 포함)을 재채점했다. **기존 train-derived validation 결과이며 새 공식 평가가 아니다.** 아래 LoRA는 LR3e-4, 전체 readout은2e-5, 모두10 epochs다. LR1e-4의 부정적/낮은 결과도 `artifacts/readout_report/pilot/`에 보존한다.

| Protocol prefix | 방법 | AP ↑ | AF ↓ |
|---|---|---:|---:|
| CL4VQA 첫3 tasks | 단일 shared LoRA | 18.38 | 7.55 |
| CL4VQA 첫3 tasks | 전체 readout 학습 | 20.56 | 13.30 |
| CL4VQA 첫3 tasks | 확장 LoRA | 23.68 | 0.00 |
| Dataset stream 첫3 tasks | 단일 shared LoRA | 7.36 | 21.95 |
| Dataset stream 첫3 tasks | 전체 readout 학습 | 15.36 | 16.30 |
| Dataset stream 첫3 tasks | 확장 LoRA | 23.03 | 0.00 |

확장군과 단일 LoRA의 첫 task 원 예측은 두 protocol·두 LR 모두 byte-identical했다. 그러나 CL4VQA recognition 정확도는 확장군 LR3e-4에서도1.10%이고, ScienceQA 첫 task의 pre-to-post gain은 음수였다. 보존만으로 약한 신규 학습을 성공으로 처리하지 않는다. 전체 protocol·더 큰 fit·강한 replay 대조·다중 seed 검증이 필요하다.

Routing 진단 job2448372에서는 전체4-task stream의 validation task-identification 정확도가 raw joint92.85%, image-only69.01%, question-stem joint82.32%였다. **VQA 정확도가 아니다.** Source별 질문 형식·OCR/options context가 routing에 영향을 준다. `configs/readout_routing_ablation.json`에 질문 stem routing 대조를 고정했으며 VQA 모델 입력 자체는 변경하지 않는다.

Pilot의 정확한 코드와 분석 도구는 `artifacts/readout-pilot-complete-source.tar.gz`(SHA256 `82b7f5623c9c97b1a0c7c266a1e727fab183abfc7fce04b0fcddc9dea7a1d05d`)에 보존했다. 이후 runner는 `--router-text raw|stem`이 추가된 별도 cohort다. 기존 pilot 재채점/재현에는 별도 복사본에서 이 snapshot을 사용한다.

### 전체 task validation: seed17, 공식 평가 미사용

Job2449994의32개 학습을 완료하고390,144개 예측을 재채점했다. Fit2,048/validation256개씩 사용한 CL4VQA10 tasks와 dataset stream4 tasks다. **전체 원본 benchmark 크기나 다중 seed 공식 평가 결과를 뜻하지 않는다.** 사전 선언한 grid 전체와 부정적 결과를 `artifacts/readout_report/development/`에 보존하고 `selection.json`으로 선택을 고정했다.

| Protocol | 방법 / 선택 | LR | Epochs | AP ↑ | AF ↓ |
|---|---|---:|---:|---:|---:|
| CL4VQA | shared LoRA 최적 | 3e-4 | 5 | 21.76 | 17.01 |
| CL4VQA | replay LoRA 최적 | 3e-4 | 10 | 34.49 | 4.56 |
| CL4VQA | 전체 readout 최적 | 2e-5 | 5 | 26.84 | 19.57 |
| CL4VQA | 확장: shared와 동일 예산 | 3e-4 | 5 | 31.76 | 0.91 |
| CL4VQA | 확장: 독립 최적 | 3e-4 | 10 | 36.91 | 1.22 |
| Dataset stream | shared LoRA 최적 | 3e-4 | 5 | 16.62 | 14.01 |
| Dataset stream | replay LoRA 최적 | 3e-4 | 10 | 24.60 | 3.15 |
| Dataset stream | 전체 readout 최적 | 2e-5 | 10 | 24.76 | 8.10 |
| Dataset stream | 확장: shared와 동일 예산 | 3e-4 | 5 | 26.90 | 0.26 |
| Dataset stream | 확장: 독립 최적 | 3e-4 | 10 | 28.53 | 0.00 |

독립 선택의 확장은 최강 대조군 대비 CL4VQA에서AP+2.42pp/AF−3.34pp, stream에서AP+3.77pp/AF−8.10pp였다. 그러나 CL4VQA5-epoch 확장은 같은5-epoch replay(AP33.28)보다 낮았다. `primary_common_budget.validation_joint_success`는 **matched shared와의 비교**이지 최강 replay를 이겼다는 뜻이 아니다. 모든8개 protocol/LR/epoch 조합에서 shared/expand의 첫 task 예측 파일은 byte-identical했다.

약점도 남았다. 독립 선택 확장의 CL4VQA recognition 최종13.67%, causal6.64%; stream TextVQA9.02%, ScienceQA pre-to-post gain−1.56pp다. 전체 task별 점수·routing·답변 분포는 audit에 보존했다. 이 validation 결과만으로 통계적 유의성·SOTA·새 알고리즘 기여를 주장하지 않는다. Capacity-matched shared adapter, frozen-first, adaptive growth 및8-seed confirmation은 아래에 별도로 보고한다.

첫 전체 audit은 TextVQA 평균의 `1.39e-17` 반올림 차이로 중단됐다. 개별 예제 점수는 모두 같았으며, audit의 naive `+=` 누적을 `math.fsum`으로 수정한 뒤32개 run 전체가 통과했다. Matrix/learning-gain의 정확한 일치 검사는 완화하지 않았고 학습·예측 결과도 바꾸지 않았다. 원인과 수치는 `artifacts/readout-audit-aggregation-diagnosis.json`에 보존했다.

학습 전 고정 소스는 `artifacts/readout-development-source.tar.gz`(SHA256 `20095b4ebfd7dfe4ac48242126a915ea53aa33b4385304de7213ba2a0b1bdb31`), 수정된 재채점 도구까지 포함한 재현 소스는 `artifacts/readout-development-complete-source.tar.gz`(SHA256 `0100461801efd32feea866a260db33d6f217a7420893c5aeecf24dbe7f81cc66`)다. 학습 조건·manifest·배포 소스 checksum은 `artifacts/readout-development-lock.json`에 고정했다.

### Expert 가중치를 고정한 routing 대조

Job2451271에서8개 학습된 bank를 재사용했고 추가 expert 학습은0회다. 모든 stage의 native 예측316개 파일이 원 학습 run과 byte-identical했고, variant 전체404,480개 예측을 독립 재채점했다. 아래는 LR3e-4/10 epochs bank의 validation 결과다.

| Routing 입력 | CL4VQA AP / AF | Dataset stream AP / AF |
|---|---:|---:|
| Raw image + question | 36.91 / 1.22 | 28.53 / 0.00 |
| Raw question only | 37.23 / 1.39 | 28.43 / −0.26 |
| Image only | 14.88 / 4.12 | 24.59 / 2.51 |
| Question stem + image | 37.42 / 1.04 | 28.01 / 0.69 |
| Question stem only | 37.89 / 0.61 | 27.69 / 0.60 |

CL4VQA에서는 같은 expert 가중치라도 image-only routing이 크게 낮았다. Stem 대조에서도 이 차이가 유지됐다. 다만 stem은 형식뿐 아니라 OCR/options 등 context도 제거하므로 순수한 형식의 인과효과로 해석하지 않는다. VQA 모델의 질문 입력은 바꾸지 않았으며, 사전 선언한 주 결과는 raw joint로 유지한다. 전체 결과는 `artifacts/readout_report/routing_development/audit.json`이다.

### Capacity·보존-only·adaptive validation 대조

Job2451628의8개 대조와114,176개 예측 재채점을 완료했다. 모두 primary common-budget 설정(LR3e-4,5 epochs)이다.

| Protocol | 대조 | Experts | AP ↑ | AF ↓ |
|---|---|---:|---:|---:|
| CL4VQA | Rank80 단일 LoRA: 총 adapter 파라미터 일치 | 1 | 24.49 | 20.27 |
| CL4VQA | 첫 expert만 학습 후 유지 | 1 | 3.98 | 0.00 |
| CL4VQA | Adaptive margin0.25 / 0.5 | 8 | 30.23 | −0.13 |
| Dataset stream | Rank32 단일 LoRA: 총 adapter 파라미터 일치 | 1 | 18.46 | 14.56 |
| Dataset stream | 첫 expert만 학습 후 유지 | 1 | 9.40 | 0.00 |
| Dataset stream | Adaptive margin0.25 / 0.5 | 2 | 19.87 | 0.00 |

확장/대용량 단일 adapter는 CL4VQA에서 각각11,796,480개, stream에서4,718,592개 adaptation 파라미터를 갖는다. 학습 exposure/step도 각각102,400/6,400과40,960/2,560으로 일치했다. 확장의AP 차이는+7.27/+8.45pp였지만 **active 파라미터·FLOPs·routing memory 일치는 아니다**. 대용량 단일 adapter는 matched setting 대조이지 별도로 최적화한 rank별 최강 baseline이라는 뜻도 아니다.

Adaptive는 두 margin의 validation 결과가 같아 사전 tie-break에 따라0.25를 고정했다. CL4VQA의 commonsense/color, stream의 GQA/VQAv2에서 gradient 학습을 생략했다. Router 업데이트 자체의 효과는 남지만, unconditional 대비AP가1.52/7.03pp 낮아 유지된 정확도의 효율 개선으로 주장하지 않는다. 상세 선택·학습량·파라미터 검증은 `artifacts/readout_report/development_controls/`에 보존했다.

### 고정된 confirmation 조건

새 공식 평가를 측정하기 전에 `configs/readout_confirmation_jobs.json`의128개 학습 조건(8 seeds ×2 protocols ×8 methods/settings), `readout_confirmation_routing_jobs.json`의32개 frozen-bank routing 조건, `readout_confirmation_comparisons.json`의20개 paired 비교를 고정했다. 주 AP 비교6개 전체에 Holm 보정을 적용하고, 나머지는 exploratory로 구분한다. Seed17은 포함하지 않으며 새 평가로 조건을 재선택하지 않는다. 이 선언은 **실행 결과나 성공 판정이 아니다**.

### 새 공식 평가: 고정된8-seed confirmation

학습 job2451800의128개 run을 모두 완료했다. Seed는 `29,43,59,71,83,97,109,127`이며 개발 seed17은 제외했다. 모든 run의 설정·manifest·초기화 자산·모델 소스가 평가 전 lock과 일치했다. 학습 run의 원 예측3,227,360개를 재채점했고,16개 protocol/seed 조합 모두 shared/주 확장의 첫 task 예측 파일이 byte-identical했다. 실패한 seed를 제외하거나 평가 결과에 따라 설정을 바꾸지 않았다.

아래는 **8개 training seed의 평균 ± sample SD**, AP는%, AF는percentage points다. AP는 마지막 stage의 task-macro 평균이다. AF는 처음T−1개 task의 `학습 직후 점수 − 마지막 점수` 평균이며 최대 forgetting이 아니다. BWT는 그 음수, MAP는 각 stage의 seen-task 평균을 다시 평균한 값이다. 전체 matrix·task별 결과도 원 audit에 보존했다.

| Protocol | 조건 | Epochs | AP ↑ | AF ↓ |
|---|---|---:|---:|---:|
| CL4VQA | Shared LoRA rank8 | 5 | 21.95 ±1.34 | 18.53 ±1.72 |
| CL4VQA | Capacity-matched shared rank80 | 5 | 20.59 ±2.61 | 26.71 ±2.65 |
| CL4VQA | Replay LoRA | 10 | 34.44 ±0.47 | 4.94 ±0.58 |
| CL4VQA | Full readout | 5 | 22.20 ±2.54 | 25.34 ±2.74 |
| CL4VQA | Frozen first expert | 5, 첫 task만 | 3.42 ±0.16 | 0.00 ±0.00 |
| CL4VQA | **Expansion: 주 common-budget** | **5** | **30.92 ±0.28** | **1.33 ±0.44** |
| CL4VQA | Expansion: 독립 validation 선택 | 10 | 35.87 ±0.19 | 1.84 ±0.36 |
| CL4VQA | Adaptive margin0.25 | 5, 선택된 task만 | 28.96 ±0.47 | −0.02 ±0.39 |
| Dataset stream | Shared LoRA rank8 | 5 | 14.91 ±0.92 | 13.82 ±1.09 |
| Dataset stream | Capacity-matched shared rank32 | 5 | 17.47 ±1.24 | 12.65 ±1.25 |
| Dataset stream | Replay LoRA | 10 | 22.95 ±2.64 | 1.92 ±0.57 |
| Dataset stream | Full readout | 10 | 21.64 ±0.82 | 8.01 ±3.47 |
| Dataset stream | Frozen first expert | 5, 첫 task만 | 9.07 ±0.41 | 0.00 ±0.00 |
| Dataset stream | **Expansion: 주 common-budget** | **5** | **23.99 ±0.55** | **1.02 ±0.49** |
| Dataset stream | Expansion: 독립 validation 선택 | 10 | 24.93 ±0.61 | 1.96 ±0.28 |
| Dataset stream | Adaptive margin0.25 | 5, 선택된 task만 | 19.08 ±0.93 | 0.02 ±0.05 |

#### 사전 고정한6개 주 AP 검정

모두 **주5-epoch expansion − baseline**이다. CI는20,000회 paired-seed bootstrap의95% percentile 구간이며 동시 신뢰구간이 아니다. Exact two-sided sign-flip 검정은 귀무가설 하 paired method-label exchangeability를 가정한다. 여덟 seed에서 가능한 최소 양측 p는 `2/256`이다. 아래6개 AP 검정 전체에 Holm 보정을 적용했다. AF와 나머지 비교는 exploratory다.

| Protocol | Baseline | ΔAP [95% seed CI] | AP Holm p | ΔAF |
|---|---|---:|---:|---:|
| CL4VQA | Matched shared rank8 | +8.97 [8.01, 9.93] | 0.046875 | −17.20 |
| CL4VQA | Validation-selected replay | **−3.52 [−3.89, −3.18]** | 0.046875 | −3.61 |
| CL4VQA | Capacity-matched shared rank80 | +10.33 [8.57, 11.77] | 0.046875 | −25.37 |
| Dataset stream | Matched shared rank8 | +9.08 [8.63, 9.47] | 0.046875 | −12.80 |
| Dataset stream | Validation-selected full readout | +2.35 [1.45, 3.09] | 0.046875 | −6.99 |
| Dataset stream | Capacity-matched shared rank32 | +6.53 [5.85, 7.40] | 0.046875 | −11.63 |

주 비교6개 중5개는AP 증가,1개는AP 감소다. **CL4VQA에서 최강 validation baseline을 이긴다는 주 가설은 실패했다.** Stream은 validation-selected full readout 대비AP/AF가 개선됐지만, eval에서 더 높게 나온 replay와의AP 차이는+1.04pp, CI[−0.23,3.14], unadjusted `p=0.421875`였다. 따라서 모든 강한 baseline을 유의하게 이겼다고 하지 않는다.

독립 validation 선택의10-epoch 확장은 CL4VQA replay와 동일한 총 exposure/step에서AP+1.43pp, CI[1.11,1.74], AF−3.10pp였다. Stream에서는 full readout 대비AP+3.29pp/AF−6.05pp, replay 대비AP+1.98pp/AF+0.04pp였다. 이 AP 비교들의 unadjusted p는0.0078125이나 **모두 exploratory이며 주 검정으로 승격하지 않는다**. Stream replay 대비 AF 차이의 CI는[−0.42,0.46]으로, 비열등성이나 동일성을 입증하지 않는다.

#### 가중치 고정 routing 대조: 추가 expert 학습0회

Job2454898은32개 confirmation expert bank 각각에서 사전 선언한5개 routing variant를 평가했다. Native 예측1,264개 파일이 부모 run과 byte-identical했고, 모든32개 native matrix/지표도 같았다. 추가3,078,400개 예측을 재채점했다. 학습·routing 합계 **6,305,760개 예측**을 검증했다.

각 칸은 **AP 평균 ± SD / AF 평균**이다. 모든 조건은 같은8개 seed의 bank를 사용한다. VQA 모델에 주는 질문은 바꾸지 않고 router 입력만 바꾼다.

| Router | CL4VQA 5 epochs | CL4VQA 10 epochs | Stream 5 epochs | Stream 10 epochs |
|---|---:|---:|---:|---:|
| Raw joint, 주 결과 | 30.92 ±0.28 / 1.33 | 35.87 ±0.19 / 1.84 | 23.99 ±0.55 / 1.02 | 24.93 ±0.61 / 1.96 |
| Raw question only | 31.55 ±0.27 / 1.30 | 36.61 ±0.38 / 1.79 | 24.00 ±0.44 / 0.86 | 24.98 ±0.54 / 1.69 |
| Image only | 12.77 ±0.63 / 3.42 | 14.21 ±0.48 / 4.24 | 22.73 ±0.63 / 1.62 | 23.46 ±0.92 / 2.39 |
| Stem joint | 32.63 ±0.41 / 0.54 | 37.77 ±0.21 / 0.95 | 23.75 ±0.65 / 0.73 | 24.95 ±0.53 / 1.33 |
| Stem question only | 33.01 ±0.36 / 0.52 | 38.30 ±0.35 / 0.93 | 23.65 ±0.62 / 0.79 | 24.65 ±0.51 / 1.24 |

같은 가중치에서 image-only의AP 손실은 CL4VQA5/10 epochs에서18.15/21.66pp, stream에서1.26/1.48pp였다. 네 비교 모두8개 seed 전부에서 손실이었다. 이는 이 protocol에서 질문 조건의 expert 선택이 중요하다는 근거다. 하지만 질문만으로 원 task를 식별하는 shortcut 가능성을 제거한 것은 아니다. Stem은 OCR/options 등 context도 제거하므로 순수한 형식 효과가 아니다. CL4VQA stem-question의38.30을 새로운 주 모델 점수로 선택하지 않는다.

#### 학습·보존·자원에 대한 해석

- **실제 신규 학습:** 첫 task를 제외한 pre-to-post gain은 주 확장에서 CL4VQA20.09pp/stream6.85pp였다. 각8개 seed의 최솟값도19.21/5.23pp로 양수였다. Frozen-first의 후속 gain은0이며AP3.42/9.07에 그쳤다. 따라서 낮은AF만으로 성능 개선을 주장한 것이 아니다.
- **총량/학습량 대조:** 모든16개 protocol/seed pair에서 확장과 capacity-matched shared의 총 adapter/모델 파라미터, 실제 optimizer step, 처리 예제 수가 같았다. CL4VQA는adapter11,796,480개/전체373,633,919개/102,400 exposures/6,400 steps, stream은4,718,592개/366,556,031개/40,960 exposures/2,560 steps다. 이 matched setting의 차이는 단순한 총 파라미터 증가만으로 설명되지 않는다. 다만 대용량 shared의 LR/rank를 별도 최적화한 최강 baseline은 아니다.
- **동일 FLOPs·memory 비교는 아님:** 확장의 학습 가능 LoRA는한 번에1,179,648개지만 capacity control은11,796,480/4,718,592개다. 확장의 저장 routing feature memory는7,864,400/3,145,760 bytes이며 ridge 계수·전체 runtime memory 수치가 아니다. Replay는64 raw examples/task, 최종640/256개를 유지하고 batch 절반을 대체한다. 총 exposure가 같더라도 current/replay 구성은 다르다. Full readout의 학습 가능 파라미터는275,139,644개다.
- **Adaptive 실패:** 모든 seed에서 CL4VQA8 experts/81,920 exposures, stream2 experts/20,480 exposures로 줄었지만 unconditional 주 확장 대비AP가1.96/4.91pp 낮았다. Router 업데이트로 일부 적응은 있으나 유지된 정확도의 효율 개선은 아니다.
- **보존과 routing의 구분:** Unconditional task-ID oracle의AF는 모든 run에서0이었다. 실제 routing의AF는0이 아니므로 expert 가중치 보존과 task-agnostic 시스템 전체의 망각 방지를 동일시하지 않는다. Oracle은 추론 방법이나 주 점수가 아니다.
- **남은 약점:** 주/독립 확장의 TextVQA 최종 점수는5.39/7.13, recognition10.01/13.26, causal7.36/11.13이었다. ScienceQA의10-epoch pre-to-post gain은−0.68pp다. 주 확장 TextVQA의 한 seed에서는 단일 답변 비율이80.86%였다. 전역 collapse guard 통과를 양호한 OCR/추론 능력의 증명으로 취급하지 않는다.
- **외적 타당성:** Fit2,048/task와 고정 task order의 부분집합 실험이다. Causal eval은73개뿐이다. CI는 고정된 benchmark에서 training seed 변동만 반영하며 새로운 이미지·task order·dataset의 불확실성을 포함하지 않는다. VQAv2/CL4VQA 점수는 source의 단일 정답 exact match이지 official consensus accuracy가 아니다. Pretraining에서의 이미지 노출 독립성도 보장하지 않는다.

논문에서 방어 가능한 결론은 **고정 BLIP의 readout expert 보존·질문 조건 routing이 matched shared adapter 대비 VQA 정확도와 망각을 개선하며, 이 이득은 총 파라미터만 늘린 대조에서도 유지된다는 통제된 실증 결과**다. 새 LoRA/MoE 아키텍처, 성공한 adaptive growth, 모든 강한 baseline에 대한 우위, SOTA를 입증한 연구로 쓰지 않는다. 기존 vision-side cohort와는 데이터·학습량이 다르므로 두 cohort의 수치 차이를 readout 위치만의 인과효과로 해석하지 않는다.

#### 재현 자료와 명령

- 평가 전 lock: `artifacts/readout-confirmation-lock.json`.
- 학습 당시 source/config/tests snapshot: `artifacts/readout-confirmation-source.tar.gz`, SHA256 `7917aaacfcbaa998881ea173bcba96b4f675a11fe89b01a5ef553d9f383f8a8e`.
- 실제 환경: `artifacts/readout-confirmation-environment.json`(Python3.12.14, torch2.5.1+cu124).
- 전체128개 run: `artifacts/readout_runs/confirmation/`; 학습 audit/summary/20개 비교: `artifacts/readout_report/confirmation/`.
- 논문용 CSV: 위 report의 `summary.csv`, `comparisons.csv`, `task_results.csv`(112개 condition/task 행), `resources.csv`. `diagnostics.json`에16개 first-task parity와 parameter/exposure 검증을 보존했다.
- 모든160개 routing condition: `artifacts/readout_routing/confirmation/`; 재채점과20개8-seed 요약: `artifacts/readout_report/routing_confirmation/`. Routing paired 분석도 exploratory로 표시했다.
- 최종 checkpoint128개와 원 예측은 DC1 `/home/faransansj/CLMoE-VQE/artifacts/readout_runs/confirmation/`에도 복사했고 모든 checkpoint의 scratch/home SHA256 일치를 확인했다. 실행·체크섬 증거는 `artifacts/readout-confirmation-execution.json`이다. Local에는 원 예측/보고서만 복사했다. Home도 독립 백업은 아니며, 이미지·초기화 자산 등 scratch 데이터에는30일 atime 정리 정책이 남아 있다.
- 원 SLURM 로그: `artifacts/readout-2451800_*.log`, `artifacts/readout-2454898_*.log`. `sacct`는 accounting DB connection-refused로 실패했으므로 할당 GPU-hours나 처리속도 향상을 보고하지 않는다.

다음은 사용한 실행 경로다. 재실행은 보존 결과를 덮어쓰지 않는 **별도 source 복사본·output 경로**에서 수행한다. SLURM launcher의 chdir/log 경로 및 config의 절대경로를 새 위치에 맞춘다. 원 manifest와 assets의 checksum은 lock과 대조한다. 모든 데이터 다운로드는 login node에서만 하고 GPU는 SLURM으로 할당한다.

```bash
ROOT=/scratch2/faransansj/CLMoE-VQE
sbatch --wait --array=0-127%4 \
  --export=ALL,READOUT_JOBS=configs/readout_confirmation_jobs.json,READOUT_MODULE=sema_vqa.readout_run \
  scripts/readout_array.sbatch
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m scripts.summarize_readout \
  --runs "$ROOT/readout_runs/confirmation" --output "$ROOT/readout_report/confirmation"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m scripts.compare_readout \
  --audit "$ROOT/readout_report/confirmation/audit.json" \
  --specifications configs/readout_confirmation_comparisons.json \
  --plan configs/readout_research.json --output "$ROOT/readout_report/confirmation/comparisons.json"
sbatch --wait --array=0-31%4 \
  --export=ALL,READOUT_JOBS=configs/readout_confirmation_routing_jobs.json,READOUT_MODULE=scripts.ablate_readout_routing \
  scripts/readout_array.sbatch
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m scripts.summarize_readout_routing \
  --jobs configs/readout_confirmation_routing_jobs.json --output "$ROOT/readout_report/routing_confirmation"
```

## 기존 vision-side 실험의 목표

Image continual learning의 **SEMA expert-expansion mechanism**을 VQA continual learning에 적용하고, 실제 신규 task 학습과 정확도·forgetting을 함께 개선할 수 있는지 검증한다. VQA 외 분류·captioning을 포함하는 후속 multimodal continual learning은 아직 실행하지 않는다.

**현재 구현은 LLM-side adapter가 아니다.** 공식 BLIP의 VQA 구조를 사용하되, VQA fine-tuning 이전의 pretrained checkpoint에서 시작한다. SEMA는 **ViT 내부**에 적용한다.

```text
이미지 → pretrained ViT-B/16 + SEMA adapters → 이미지 표현 ─┐
                                                          ├→ 질문 encoder → 답변 decoder
질문 ─────────────────────────────────────────────────────┘
```

이전 LLM-side 실험은 [LEGACY_LLM.md](LEGACY_LLM.md), [RESULTS.md](RESULTS.md), `artifacts/runs/`에 보존한다. **그 결과는 이번 source-faithful 구현의 검증 결과가 아니다.** 적용 위치 변경뿐 아니라, 확장이 없을 때도 readout을 학습하는 SEMA의 경로를 이전 구현에서 누락했다. 이전 AF=0을 source method의 성능 근거로 사용하지 않는다.

## Source에 맞춘 부분

- [SEMA 공식 코드](https://github.com/huiyiwang01/SEMA-CL), commit `5a4551262d659c6d93dd6c50ff814e028554fbf3`.
- [BLIP 공식 코드](https://github.com/salesforce/BLIP), commit `056a169437371659074aa2732649d5de3bffb4a8`.
- ViT의 **12개 block 모두**에 down/ReLU/up functional adapter를 둔다. 원본 `vit_sema.py`처럼 attention residual **이후, norm2 이전**의 표현이 adapter 입력이다.
- SEMA `exps/sema_vtab.json`에 맞춰 **zero-based block 9–11**에서 expert/RD/router를 확장한다. Bottleneck 16, RD dimension 128, buffer 500.
- Descriptor와 router는 **이미지 token 평균**만 사용한다. 질문·정답은 들어가지 않는다.
- 원본 코드의 `min_expert(mean_batch(abs(z-score))) > threshold`를 사용한다. 얕은 층부터 확장하고 새 모듈 학습 후 다시 감지한다. 층당 task별 최대 한 번 추가한다.
- 이전 expert·descriptor·router row와 비확장 층 adapter는 task 종료 후 동결한다. 새 모듈만 vision 측에서 학습하며 replay는 없다.
- **확장이 없어도 readout은 계속 학습한다.** 원본 `models/sema.py:81–83,205–207`의 학습 경로를 VQA의 질문 encoder·답변 decoder 학습으로 대응시킨다.
- Functional과 RD 학습은 분리한다. Functional의 gradient가 descriptor에 들어가지 않으며, RD loss가 vision backbone이나 functional expert를 갱신하지 않는다.

### VQA 적용에 필요한 차이와 제한

SEMA의 작은 분류 head를 BLIP의 질문 encoder·답변 decoder로 바꾼다. VQA는 task 간 답변 vocabulary가 겹치므로 원본 class-incremental old-class logit masking은 적용하지 않는다. 따라서 이것도 모델 가중치까지 동일한 원논문 재현은 아니며, **vision-side mechanism을 보존한 VQA 적용**이다.

BLIP pretrained checkpoint는 `model_base_capfilt_large.pth`이고, `model_base_vqa_capfilt_large.pth` 같은 VQA-tuned checkpoint는 사용하지 않는다. 원본 BLIP VQA 구성과 달리 입력 224px, deterministic resize, greedy 최대32 tokens를 사용한다. 224px는 SEMA ViT 설정과 일치하지만 OCR에는 불리할 수 있다. 모든 비교군에 같은 설정을 적용한다. 질문/답변은 512-token 한도를 넘으면 오류로 종료하며, 조용히 잘라내지 않는다.

## 데이터·설정 선택

- VQAv2 10-task: CL-MoE의 공개 CL4VQA 순서와 평가 split.
- VQA dataset stream: SMoLoRA 공개 순서 중 `ScienceQA → TextVQA → GQA → VQAv2`.
- 원본 선택 train 512개/task에서 이미지 SHA-256을 기준으로 전체 task에 걸쳐 약80/20 fit/validation 분할한다. 공식 eval과 byte-identical한 train 이미지는 먼저 제외한다. **세 split 사이 동일 바이트 이미지 중복은 없다.**
- Official eval 128개/task는 변경하지 않는다. Pretraining 단계의 이미지 노출 여부까지 독립성을 보장하는 것은 아니다.
- 실제 fit 크기: VQAv2 388–431개/task, dataset stream 393–411개/task. Validation 81–124개/task.
- 설정 선택은 validation만 사용한다. 최종 평가에서도 validation을 train에 합치지 않는다.
- VQAv2는 CL-MoE 코드의 단일 정답 exact match이며 official VQA consensus accuracy가 아니다. TextVQA/GQA/ScienceQA는 각각 공개 source scorer를 사용한다.

초기 계획은 [`configs/vision_poc.json`](configs/vision_poc.json)이다. 초기 validation에서 큰 forgetting이 관측되어, **공식 SEMA CIFAR 설정의 functional 5 epochs**와 중간 head LR `1e-5`를 추가 비교한다. 확장 mechanism과 모델 코드는 바꾸지 않는다. 전체 36개 train-only 조건은 [`configs/vision_search_poc.json`](configs/vision_search_poc.json)에 선언했다. Sequential validation AP로 공통 head LR·epoch budget을 먼저 고른 후, 같은 조건에서 forgetting을 고려해 SEMA threshold를 고른다. Official eval로 설정을 선택하지 않는다.

TextVQA/GQA의 공개 instruction train에는 평가용 원본 reference가 빠져 있다. `enrich_vision_references.py`가 공식 **train** annotation을 연결한다. TextVQA의 실제10개 정답을 보존하고, GQA validation에는 공식 canonical answer를 사용한다. Image ID·question으로 연결하되 GQA의 동일 질문에 상충하는 annotation이 있으면 기존 instruction 정답으로만 구분하며 모델 예측은 사용하지 않는다. Fit target은 변경하지 않는다.

36개 validation 완료 후, official eval **이전**에 [`configs/vision_independent_poc.json`](configs/vision_independent_poc.json)의 secondary 비교도 고정했다. 최선의 sequential baseline은 유지하되 SEMA의 LR·epoch를 독립적으로 고른다. 공통 budget의 primary 비교를 대체하지 않는다. CL4VQA의 5-epoch 후보는 validation AP·AF가 개선됐지만 확장이 없었으므로, no-expansion ablation 없이 이를 expansion 효과라고 해석하지 않는다. Primary16개와 secondary10개를 seeds `17/29/43`으로 평가한다(ablation은 seed17).

비교군:

| 방법 | Vision adapter의 후속 task 학습 | VQA readout |
|---|---|---|
| `sequential` | 같은 12개 adapter 계속 학습 | 계속 학습 |
| `sema` | 기존 expert 동결, 감지된 층만 확장·학습 | 확장 유무와 무관하게 학습 |
| `no_expansion` | 첫 task 이후 동결 | 계속 학습 |
| `task_expansion` | 매 task마다 block9–11에 하나씩 추가 | 계속 학습 |

**학습량 대조:** SEMA는 추가된 층마다 functional phase를 실행하므로 task당 최대3회의 phase가 가능하다. Dataset stream의 두 조건에 대해, 실제 SEMA의 task별 phase 수를 그대로 재현한 sequential·no-expansion 대조군12개도 실행했다. LR·epoch·effective batch·optimizer/scheduler 재시작을 맞췄다. RD 학습, 전체 FLOPs, wall time까지 동일하게 맞춘 실험은 아니다.

## 실행 명령과 재현 조건

아래는 Yonsei-DC1에서 사용한 실행 명령이다. 새로 재현할 때는 별도 작업 복사본·결과 경로를 사용하고 SLURM의 `--chdir`, log 경로와 `ROOT`도 함께 바꾼다. 보존된 결과에 다른 코드로 resume하지 않는다. GPU는 SLURM으로만 할당하며 `CUDA_VISIBLE_DEVICES`를 직접 설정하지 않는다. 다운로드는 login node에서만 수행한다.

처음62개(calibration36 + confirmation26)의 정확한 core4개 파일은 `artifacts/vision-core-before-exposure.tar.gz`에 보존했다. 이 실험을 그대로 재현·재개하려면 **별도 작업 복사본에 해당 snapshot을 적용**한다. 학습량 대조군12개는 현재 runner의 `--match-exposure-from` 확장으로 실행했다. 모델·SEMA 모듈·scorer는 byte-identical하며, runner revision이 다른 두 cohort의 audit은 분리한다.

```bash
ROOT=/scratch2/faransansj/CLMoE-VQE
uv pip install --python .venv/bin/python -r requirements-vision.txt
.venv/bin/python scripts/download_blip.py --output "$ROOT/models/blip"

# 원본 데이터 획득 절차는 LEGACY_LLM.md의 prepare_data.py 명령 참조.
.venv/bin/python scripts/prepare_vision_splits.py \
  --manifest "$ROOT/manifests/cl4vqa.json" \
  --validation "$ROOT/manifests/cl4vqa.validation.json" \
  --output "$ROOT/vision_manifests_referenced/cl4vqa.json"
PYTHONPATH=. .venv/bin/python scripts/enrich_vision_references.py \
  --root "$ROOT" --manifest "$ROOT/manifests/vqa_stream.json" \
  --output "$ROOT/vision_reference_manifests/vqa_stream.json"
.venv/bin/python scripts/prepare_vision_splits.py \
  --manifest "$ROOT/vision_reference_manifests/vqa_stream.json" \
  --validation "$ROOT/manifests/vqa_stream.validation.json" \
  --output "$ROOT/vision_manifests_referenced/vqa_stream.json"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest tests -q
sbatch scripts/vision_smoke.sbatch
# 다음은 원본62개 cohort의 명령이다. 새 결과 경로에서 실행한다.
sbatch --wait --array=0-5%4 scripts/vision_calibration.sbatch
sbatch --wait --array=6-11%4 --export=ALL,VISION_RUN_GROUP=calibration_references scripts/vision_calibration.sbatch
sbatch --wait scripts/vision_refinement.sbatch

# 모든 calibration/refinement 완료 후에만 실행
RUNS=("$ROOT/vision_runs/calibration/cl4vqa"
      "$ROOT/vision_runs/calibration_references/vqa_stream"
      "$ROOT/vision_runs/calibration_short" "$ROOT/vision_runs/calibration_midrate")
PYTHONPATH=. .venv/bin/python scripts/summarize_vision.py \
  --runs "${RUNS[@]}" --output "$ROOT/vision_report/calibration" \
  --plan configs/vision_search_poc.json --select
PYTHONPATH=. .venv/bin/python scripts/summarize_vision.py \
  --runs "${RUNS[@]}" --output "$ROOT/vision_report/calibration_independent" \
  --plan configs/vision_independent_poc.json --select

# 두 선택을 고정한 뒤26개 confirmation; 결과를 보고 설정을 다시 고르지 않는다.
sbatch --wait scripts/vision_confirmation.sbatch
PYTHONPATH=. .venv/bin/python scripts/summarize_vision.py \
  --runs "$ROOT/vision_runs/confirmation" "$ROOT/vision_runs/confirmation_independent" \
  --output "$ROOT/vision_report/confirmation" --plan configs/vision_search_poc.json --compare
```

Core runner: `python -m sema_vqa.vision_run --help`. Task 경계 checkpoint는 `latest.pt`, 원 예측은 `predictions_after_*.jsonl`, matrix/지표는 `results.json`에 저장한다. Resume는 manifest·모델 source·설정·핵심 코드 hash가 같은 경우만 허용한다.

학습량 대조군은 **현재 runner 코드**에서 다음과 같이 실행한다. Parent SEMA 결과가 먼저 완성되어 있어야 한다. `configs/vision_exposure_poc.json`은 원본 parent runner hash를 고정하며, source config·events·results hash와 실제 phase 수도 검사한다.

```bash
sbatch --wait scripts/vision_exposure.sbatch
PYTHONPATH=. .venv/bin/python scripts/summarize_vision.py \
  --runs "$ROOT/vision_runs/exposure_confirmation" \
         "$ROOT/vision_runs/exposure_confirmation_independent" \
  --output "$ROOT/vision_report/exposure" --plan configs/vision_exposure_poc.json \
  --compare-exposure "$ROOT/vision_report/confirmation/audit.json"
```

## 실제 확인된 동작

- **83 tests passed**: gradient accumulation, source insertion 위치, no-expansion readout 학습, image/reference split, validation-only 선택, paired-seed 통계·task-end accuracy, exposure replay·optimizer 재시작·중단된 attempt 제외·부정확한 비교 거부.
- GPU smoke **2446528**: 실제 train 4개에서 VQA loss `21.1945 → 5.9240`, vision adapter와 readout의 실제 update, RD loss 감소, 새 expert update/기존 expert 불변, dynamic checkpoint 생성 결과 일치. 이 4개는 학습 예제이며 held-out 성능 근거가 아니다.
- Source parity GPU job **2446531**: 수정하지 않은 공식 `SEMAModules`와 1/2/3 experts의 출력·z-score를 비교했다. 최대 forward 오차 `2.24e-8` 이내.
- 모든 선택 질문·정답의 BLIP token 길이를 검사했다. 최대 질문 길이359, 최대 정답 길이22로 모두512 이내다.
- 증거: `artifacts/vision_smoke/result.json`, `artifacts/vision-parity-2446531.log`, `artifacts/vision-token-validation.json`, `artifacts/vision_manifests/`.
- Calibration36개 완료: 초기 CL4VQA array `2446533`, TextVQA/GQA reference를 복구한 dataset-stream array `2446864`, 추가24개 array `2446959`. 원본 예측 재채점과 전체 grid 확인을 통과했다. 최초 dataset-stream 실패 산출물은 보존하고 선택에서 제외한다. 현재 manifest는 `artifacts/vision_manifests_referenced/`, 선택 근거는 `artifacts/vision_report/calibration*/`다.

## 최종 결과: vision-side 실험74개

**구현·확장·신규 task fitting은 확인했다. Dataset stream에서는 sequential 대비 개선 신호가 있지만, 학습량을 맞춘 no-expansion보다 우월하지 않아 expert expansion의 고유한 성능 이득은 입증하지 못했다. CL4VQA의 짧은 학습 조건은 답변 붕괴 때문에 성공으로 보지 않는다.**

완료된 calibration36 + confirmation26 + exposure-control12개를 보고한다. 이전 LLM-side16개 및 reference 누락으로 중단된 초기6개는 이 합계에 넣지 않는다. 여러 task stage의 반복 평가를 포함한 원 예측 **245,200건**을 다시 채점했다.

### 최종 평가: validation에서 고정한 설정

평균 ± sample SD, seeds `17/29/43`. AP는 마지막 stage의 task 평균 점수(%), AF는 이전 task들의 `task 종료 직후 점수 − 마지막 점수` 평균(pp)이다. AP는 높을수록, AF는 낮을수록 좋다. Protocol 간 절대 점수는 서로 비교하지 않는다.

| Protocol | 조건 | AP ↑ | AF ↓ | 추가 expert 수(seed 순서) |
|---|---|---:|---:|---|
| CL4VQA | 최선 validation sequential | 10.44 ± 1.17 | 12.91 ± 1.67 | 0 / 0 / 0 |
| CL4VQA | SEMA, 공통 budget | 9.87 ± 1.38 | 17.82 ± 1.65 | 0 / 0 / 0 |
| CL4VQA | SEMA, 독립 선택 | 13.33 ± 0.16 | 6.94 ± 0.48 | 0 / 1 / 1 |
| Dataset stream | 최선 validation sequential | 13.25 ± 3.25 | 13.97 ± 2.89 | 0 / 0 / 0 |
| Dataset stream | SEMA, 공통 budget | 14.19 ± 0.51 | 13.84 ± 0.94 | 5 / 5 / 6 |
| Dataset stream | SEMA, 독립 선택 | 15.07 ± 0.62 | 11.14 ± 1.72 | 7 / 6 / 6 |

- CL4VQA 공통 조건: head LR `2e-6`,20 epochs, threshold1.0. 독립 선택은 같은 LR/threshold의5 epochs.
- Dataset stream 공통 조건: head LR `2e-5`,20 epochs, threshold0.75. 독립 선택은 LR `1e-5`,5 epochs, threshold0.75.
- Dataset stream 독립 선택의 paired 변화는 AP **+1.82 ± 3.80pp**, AF **−2.83 ± 4.57pp**다. 세 seed 모두 동시 개선된 것은 아니다. 작은3-seed 결과를 통계적으로 확립된 개선으로 해석하지 않는다.
- CL4VQA 독립 선택은 최종1,280문항의 **99.77% / 97.97% / 100%**에 `no`라고 답했다. Recognition/location/count/color/type의 평균 최종 정확도는0이다. 평균 점수 상승을 유용한 VQA 개선으로 판정하지 않는다.
- CL4VQA seed17의 SEMA와 no-expansion은 공통·독립 조건 모두 AP/AF/MAP가 동일했다. 실제 확장이 없는 조건의 점수는 확장 효과가 아니다.
- 전체26개 confirmation에서 첫 task 이후의173개 functional phase 모두 마지막 epoch의 **train loss**가 첫 epoch보다 낮았다. 이는 이전의 완전 동결 실패와 다르지만, held-out 신규 task 성능 향상 자체를 증명하는 지표는 아니다.
- Task 종료 직후 평균 정확도(matrix diagonal)는 CL4VQA baseline22.06%, 독립 SEMA19.58%였다. Dataset stream은23.72%와23.42%였다. 낮은 AF만으로 plasticity까지 개선됐다고 결론내리지 않는다.

### 학습량을 맞춘 대조: Dataset stream

각 seed의 SEMA에서 기록한 phase 수를 그대로 사용했다. 공통 조건은 총6/6/7 phases, 독립 조건은8/7/7 phases다. 모든 control의 첫 task 원 예측은 parent SEMA와 **byte-identical**했고, 실제 phase 수·model/scorer hash·source provenance 검증도 통과했다.

| Parent 조건 | 방법 | AP ↑ | AF ↓ |
|---|---|---:|---:|
| 공통20 epochs | SEMA | 14.19 ± 0.51 | 13.84 ± 0.94 |
| 공통20 epochs | 노출량 일치 sequential | 12.28 ± 3.22 | 14.77 ± 1.05 |
| 공통20 epochs | 노출량 일치 no-expansion | 14.55 ± 2.43 | 14.17 ± 1.40 |
| 독립5 epochs | SEMA | 15.07 ± 0.62 | 11.14 ± 1.72 |
| 독립5 epochs | 노출량 일치 sequential | 13.86 ± 0.38 | 11.94 ± 1.42 |
| 독립5 epochs | 노출량 일치 no-expansion | **15.70 ± 1.12** | **10.68 ± 1.77** |

독립 조건의 SEMA는 노출량 일치 sequential보다 세 seed 모두 AP가 높고 AF가 낮았다. 그러나 동일 노출량의 no-expansion 대비 평균 AP는 **−0.625pp**, AF는 **+0.460pp**로 나빴다. 공통 조건도 no-expansion보다 AP가 낮았다. 따라서 “단순 sequential보다 개선”과 “확장이 필요하거나 우월함”은 구분해야 한다. **현재 결과에서는 전자는 관측했지만 후자는 지지하지 않는다.**

이것은 새 no-expansion 방법을 평가 결과로 선택해 성공 모델로 대체한 것이 아니라, 원래 계획에 명시한 학습량 confound를 확인한 대조다. 두 조건의 LR·epoch·threshold는 최종 평가를 본 뒤 바꾸지 않았다.

### 범위와 재현 산출물

- 제한된 fit388–431개/task, 고정 task 순서,3 seeds, eval128개/task의 POC다. Full benchmark 재현이나 발표된 CL-MoE/SMoLoRA 수치와의 동등 비교가 아니다.
- BLIP의 큰 trainable readout, old-class masking 미적용,224px, greedy generation, 사전학습 이미지 노출 가능성이 남는다. 이 실험으로 SEMA 일반이나 모든 VQA expansion 접근의 가능성을 부정하지 않는다.
- [INFERENCE] Image-only detector는 질문 자체의 변화는 직접 감지하지 못하며, shared readout의 forgetting이 vision expert 보존 이득을 제한할 수 있다. 원인 확정을 위한 readout-capacity 개입 실험은 하지 않았다.
- GPU jobs: smoke `2446528`, source parity `2446531`, calibration `2446533/2446864/2446959`, confirmation `2447187`, exposure controls `2447416`.
- 원 예측·config·events·matrices: `artifacts/vision_runs/`. Validation121,680건, confirmation108,160건, exposure15,360건.
- 선택 및 paired 비교: `artifacts/vision_report/calibration/`, `calibration_independent/`, `confirmation/`, `exposure/`의 `audit.json`, `summary.csv`, `selection.json` 또는 `comparison.json`.
- 답변 붕괴·fit 학습 진단: `artifacts/vision-validation-diagnostics.json`, `artifacts/vision-confirmation-learning.json`.
- 모델·환경: `artifacts/blip-sources.json`, `artifacts/environment-vision-freeze.txt`; 원본 core snapshot은 위 tar.gz. Model checkpoint는 DC1 `/scratch2/faransansj/CLMoE-VQE/vision_runs/`의 각 run `latest.pt`에 보존했고 local로 복사하지 않았다. Scratch의30일 atime 정리 정책에 유의한다.
