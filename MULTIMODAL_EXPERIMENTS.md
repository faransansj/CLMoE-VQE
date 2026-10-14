# 멀티모달 task 확장 실험계획

작성: 2026-10-13. 실행 갱신: 2026-10-14. 상태: **개발92개 학습 및 G1~G3 gate 통과; 실제 routing 검증과 pre-eval lock 후 G4 새8-seed/224개 조건 실행 중**. [설계](MULTIMODAL_DESIGN.md)와 [기계 판독 계획](configs/multimodal_research.json)을 함께 따른다. 아래 본 실험 조건은 변경하지 않았다. 변경은 validation trial 이전에 revision으로 기록한다. Confirmation eval을 본 뒤 조건·지표·표본·seed를 바꾸지 않는다.

## 1. 검증할 주장

- **H1 보존/최종 성능:** heterogeneous vision-language CL에서 expansion이 동일 학습량의 shared adapter, capacity-matched shared, replay 대비 최종 성능을 개선하는가?
- **H2 신규 학습:** 낮은 망각이 새 task를 안 배운 결과가 아닌가? 후속 task pre-to-post gain과 frozen-first 대조로 확인한다.
- **H3 task 간 전이(탐색):** stream의 새 task 학습 직후 성능이 같은 예산의 독립 task 학습보다 높은가? 보존 성공과 분리한다.
- **H4 routing(탐색):** 효과가 family instruction 식별만으로 설명되는가, 같은 family 안의 domain 구분도 필요한가?

주 범위는 VQA와 captioning. Audio/video, 새로운 backbone 탐색, adaptive expansion, SOTA 비교는 포함하지 않는다.

## 2. 데이터와 task stream

| 기호 | Task | 입력/출력 | 주 지표 | source 확보 시작점 |
|---|---|---|---|---|
| V | VQAv2 | image+question → answer | official consensus accuracy | https://visualqa.org/download.html |
| C | COCO2014 captions | image+고정 instruction → caption | 고정 train-IDF CIDEr-D | https://cocodataset.org/#download |
| G | GQA balanced | image+question → answer | official normalized accuracy | https://cs.stanford.edu/people/dorarad/gqa/download.html |
| F | Flickr30k captions | image+동일 instruction → caption | 고정 train-IDF CIDEr-D | https://shannon.cs.illinois.edu/DenotationGraph/ |

이 표는 **사용 예정 source**이며 실제 접근·라이선스·가용 image/reference 수를 이번에 확인했다는 뜻이 아니다. Source gate에서 revision, 다운로드 원본, SHA256, 허용 사용범위와 공식 split을 고정한다. Flickr30k 등의 접근 승인이 필요하면 획득 전 본 평가를 실행하지 않는다. 다른 dataset으로 조용히 대체하지 않는다.

- Image-group sampling seed `20261013`. Task당 train 2,048 unique images, validation 256, 새 eval 512 목표. VQA도 image당 한 QA; caption은 image당 모든 reference 유지.
- V/C의 COCO 이미지와 G/F를 포함해 전 task·split 간 identity/SHA256 중복을 차단한다. 같은 이미지의 질문/캡션을 train/eval에 나누지 않는다.
- Source train에서 train/validation을 만들고, source의 공개된 label-bearing held-out split에서 eval을 만든다. VQAv2/COCO2014는 공식 val, GQA는 공개 validation, Flickr30k는 사전 고정한 공개 split recipe의 test를 사용한다. Recipe와 image ID 목록을 lock에 포함한다. Test 서버의 비공개 label을 가정하지 않는다.
- 새 eval은 기존 모든 VQA cohort의 train/validation/eval 이미지·ID도 제외한다. 후속 train/validation은 기존 eval 이미지를 제외한다. 실제 파일을 decode/hash한 뒤 확정한다.
- 필터 뒤 가용 수가 부족하면 중복 증식하거나 다른 split에서 충당하지 않는다. 가용 수와 부족 원인을 기록하고, 새 데이터 계획을 validation 전에 재고정한다. 목표 수 충족 또는 명시적인 계획 수정 전에는 confirmation으로 진행하지 않는다.
- Pilot과 본 평가의 데이터 분할은 동일한 전역 manifest에서 가져오고, pilot에서는 train/validation만 연다. Test prediction은 confirmation 시작 전 생성하지 않는다.

### 순서

Pilot: `V→C`, `C→V` 두 순서. 이것만으로 복잡한 routing 효과를 주장하지 않는다.

본 평가의 사전 고정 네 순서:

1. `V→C→G→F` (교차)
2. `F→G→C→V` (역방향 교차)
3. `V→G→C→F` (VQA family 먼저)
4. `F→C→G→V` (caption family 먼저)

같은 dataset/split을 순서만 바꿔 사용한다. 따라서 네 순서를 독립 dataset 네 개로 세지 않는다. 순서별 결과를 모두 공개하고 통계 단위는 seed로 둔다.

## 3. 방법과 공정성

모든 방법에 같은 초기화·이미지 처리·target-token 평균 loss·생성 설정을 적용한다. 기본 LoRA rank8/alpha16, AdamW LR3e-4, 5 epochs/task, effective batch16, microbatch8, image224다. 공통 optimizer 세부값과 precision은 구현 때 기존 runner와 대조해 config에 명시하고 lock한다. VQA32/caption64 max new tokens, greedy.

| 조건 | 목적 | 용량/학습 계약 |
|---|---|---|
| Shared LoRA | 비확장 순차학습 | rank8 단일 adapter, 모든 task 업데이트 |
| Capacity-matched shared | 단순 총 용량 증가 배제 | 본 평가 rank32/alpha64, pilot rank16/alpha32; 마지막 expansion bank와 총 파라미터 동일 |
| Replay LoRA | 강한 CL 대조 | rank8, 과거 task당64 train images; batch 절반을 과거로 대체, 첫 task는 모두 current |
| Expansion | 주 방법 | task마다 rank8 expert 추가, old frozen, joint ridge routing |
| Frozen-first | 학습 없는 보존 대조 | 첫 expert만 학습; 첫 task router/expert 고정, 후속 업데이트 없음 |
| Independent task | 간섭 없는 참조/전이 대조 | task별 fresh backbone 초기화+rank8, 다른 task 지식 없음 |
| Validation-tuned replay | 공통 설정이 약한 replay를 만드는지 확인 | 동일 grid에서 독립 선택; 예산 차이 명시, 탐색적 비교 |

공통 4방법은 phase별 steps/총 image exposures를 맞춘다. Replay는 replacement라 current exposure는 절반이며, 이 차이를 별도 보고한다. 여러 caption reference는 새로운 이미지 exposure로 중복 계산하지 않는다. 길이가 다른 task에서는 같은 예제 수가 같은 token/FLOPs가 아니므로 target tokens, steps, current/replay exposures, trainable/total params, router memory, raw replay bytes, peak GPU memory와 실제 측정 시간도 기록한다. Router 특징 계산 비용을 포함하지 않은 시간을 전체 속도라 하지 않는다.

Independent는 전이 비교를 위해 반드시 공통 5-epoch 조건으로도 저장한다. Validation 정규화에 사용하는 선택된 독립 모델과 혼동하지 않는다. Frozen-first/independent는 CL 우월성의 주 대조가 아니라 진단이다. Separate task models의 저장비도 보고한다.

## 4. 단계와 진입/종료 gate

### G0 — 데이터·코드 계약

Source/라이선스/이미지/reference 확보, 전역 split audit, scorer fixture, lock 가능한 config를 준비한다. 실제 소규모 GPU smoke에서 caption 생성, 신규 expert update/기존 expert 불변, oracle와 learned routing 분리, interrupted resume parity를 검증한다. 기존 VQA run 재생 경로는 불변이어야 한다. 실패하면 다음 단계로 넘어가지 않는다.

### G1 — 독립 task learnability

개발 seed17, task4개 × LR `{1e-4,3e-4}` × epochs `{5,10}` = **16개 독립 학습**. Frozen-base의 같은 validation도 평가한다. 각 task에서 공통 3e-4/5epoch 조건의 validation 점수가 frozen-base보다 높고, paired-image bootstrap 2,000회의 score gain 95% CI 하한이 0보다 커야 한다. Caption bootstrap은 resample마다 corpus score를 다시 계산하고 IDF는 고정한다. 이는 본 가설 검정이 아닌 feasibility gate다.

추가로 image-shuffled 진단을 해서 이미지와 무관한 generic caption/답변으로 보이는지 보고한다. 빈 출력률1% 초과 또는 최빈 전체 출력 비율90% 이상이면 실패다. 이 guard 통과만으로 의미적 품질을 증명하지 않는다. Caption reference 표본을 직접 점검한다. Gate 실패 시 architecture/학습 문제를 먼저 해결하고 모든 validation revision을 기록한다. Replay/CL을 돌려 단일 task 실패를 가리지 않는다.

Task별 가장 높은 validation score의 독립 모델을 정규화 reference로 선택한다. 동률은 적은 epochs, 낮은 LR 순. Reference score가 0이면 정규화가 정의되지 않으므로 중단한다. 이 4개 양수 상수는 confirmation 전에 고정한다.

### G2 — 양방향 pilot

Seed17, 2순서 × 공통 4방법 = **8개 CL run**. Eval은 validation만 사용한다. Expansion은 각 순서에서 후속 task gain이 양수이고, 이전 task 망각이 shared보다 작으며, 두 task 최종 정규화 점수 평균이 shared보다 높아야 진입 성공으로 본다. 모든 방법의 완전한 matrix와 생성 진단을 보존한다. 실패하면 실패 이유·다음 validation 계획을 기록하고 G3를 자동 실행하지 않는다.

### G3 — 4-task 개발/선택

Seed17, 4순서 × 4방법 × 2 LR × 2 epoch = **64개 run**. Frozen-first 4개 진단 run 추가. G1/G2와 합쳐 **92개 training job 계획**이며 동일 config의 결과는 중복 학습/증거로 세지 않는다.

주 비교 조건은 3e-4/5epoch로 고정한다. Grid는 설정 민감도 공개와 independently tuned replay 선택에 사용한다. Tuned replay는 네 순서의 family-balanced normalized final score 평균이 가장 높은 조건을 선택한다. 동률은 적은 epochs, 낮은 LR 순. Negative gain/collapse를 함께 표시하고 실패 trial도 공개한다. 주 expansion 조건을 grid 최댓값으로 바꾸지 않는다.

G1~G3의 모든 matrix·실패·선택을 보존한 뒤 데이터/config/source/scorer/normalizer/seed/비교 목록을 hash lock한다. 주 조건 expansion에서 어느 family든 후속 task 평균 gain≤0 또는 생성 guard 실패면 confirmation으로 넘어가지 않는다.

### G4 — 새 seed confirmation

개발 seed17은 제외하고 **137,149,163,179,191,211,227,241**을 사용한다. 이번 연구의 새 8개 paired seeds이며 예전 VQA 결과의 seed 선택을 재사용하지 않는다.

- 공통 4방법 × 4순서 × 8seeds = **128개 CL run**.
- Frozen-first × 4순서 × 8seeds = **32개**.
- 공통 조건 independent × 4task × 8seeds = **32개**; 순서마다 재학습하지 않는다.
- Tuned replay × 4순서 × 8seeds = **최대32개**. 공통 replay와 설정이 같으면 원 결과를 재사용한다.
- 총 **192~224개 학습 job**. Seed 제거·유리한 순서 선택 금지. 인프라 실패는 동일 lock으로 resume/retry하고 로그를 남긴다. Method failure는 방법 실패로 보고하며 조용히 제외하지 않는다.

G4 새 eval은 원 예측을 저장한 뒤 독립 재채점한다. 여기서 나온 성능으로 다시 설정을 고르면 새 연구 cohort가 필요하다.

### G5 — frozen-bank routing/진단

G4 expansion의 **32개 bank**에 native joint, image-only, question-only, family-restricted joint, task-ID oracle의 **5조건=160개 평가 조건**. Expert 학습은 추가하지 않는다. Native predictions와 matrix가 부모 run과 일치해야 한다. 다른 feature modality는 같은 train memory에서만 router를 다시 fit한다.

Family-restricted joint는 요청 형식으로 알고 있는 VQA/caption family에 해당하는 **이미 학습한** expert만 후보로 남긴 진단이다. Native joint와 같은 feature/계수에서 후보 mask만 바꾸고, 정답 task ID는 쓰지 않는다. Oracle은 정답 task expert를 강제하므로 별도 상한 진단이다. Family당 두 task가 모두 도입된 마지막 stage의 router confusion과 family 내 domain 혼동을 반드시 보고한다. 모든 routing 비교는 탐색적이며 최고 점수를 주 모델 점수로 재선택하지 않는다.

## 5. 지표와 통계 사전 정의

Task j의 stage t 점수를 `S[t,j]`로 저장한다. 원 단위(accuracy points, CIDEr-D scorer units)를 유지한다. `R[j]`는 G1에서 고정한 independent validation score. 정규화 점수는 `N[t,j]=S[t,j]/R[j]`이며 1보다 커도 자르지 않는다. Test reference로 분모를 재선택하지 않는다.

- Family final score: 마지막 stage에서 그 family 두 task의 N 평균.
- 선택용 전체 점수: VQA family와 caption family final score의 1:1 평균.
- Task AF: `S[j,j]−S[T,j]`; 마지막 task 제외. Family AF는 해당 이전 task들만 평균하며 해당 task가 없으면 NA, 0으로 채우지 않는다. 정규화 AF도 별도 기록.
- Pre-to-post gain: 현재 task의 학습 직전 점수와 직후 점수 차이. 첫 task를 제외한 결과와 task별 결과를 모두 보고한다.
- Transfer: CL task 학습 직후 점수 minus 같은 seed/공통예산 independent 점수. 단순 final retention과 구분한다.
- 각 stage의 seen-task matrix, native/oracle routing confusion, 생성 분포와 resource curve를 함께 공개한다.

**주 AP 검정 family는 총6개:** 두 task family 각각에서 expansion minus (shared, capacity-matched, common-budget replay). Seed마다 네 task 순서의 paired 차이를 먼저 평균한다. **8개 seed가 통계 단위**이며 32개의 독립 관측처럼 취급하지 않는다. Exact two-sided sign-flip, 20,000회 paired-seed bootstrap 95% percentile CI, 6개 p 전체 Holm 보정. CI는 개별 구간이며 동시 CI가 아니다. 유한한 네 순서 밖의 task-order 일반화/새 dataset 불확실성을 포함하지 않는다.

AF, transfer, tuned replay, router, 개별 task/order 결과는 exploratory다. Tuned replay의 family별 AP 두 검정은 별도 Holm family로 표시하며 주 결과로 승격하지 않는다. 검정 단위/seed 수를 늘려 p를 유리하게 만들지 않는다.

## 6. 성공/실패 판정

- **비확장 대비 성능 개선:** family별 shared/capacity 대조의 양의 평균 AP 차이, CI 하한>0, 주 Holm p<0.05를 각각 보고한다. 한 family만 통과하면 그 범위의 결과로 제한한다.
- **망각 완화:** task별/seed별 AF와 차이 CI를 보고한다. 평균 AF가 낮다는 관찰과 유의한 감소를 구분한다. AP 유의성으로 AF 유의성을 대신하지 않는다.
- **강한 CL baseline 우위:** replay 대조가 실패하면 shared 대비 성공과 분리해 명시한다. 주 조건 replay 및 tuned replay의 결과를 모두 보고, 모든 baseline을 이겼다는 표현을 자동 부여하지 않는다.
- **실질적인 CL 성공:** 위 AP 근거에 더해 후속 task gain이 두 family에서 양수이고, expansion의 family 평균 AF가 shared보다 낮으며 생성 guard가 통과해야 한다. 이는 운영 판정이며 formal 비열등성 검정으로 부르지 않는다.
- **Transfer 성공:** 독립 학습 대조의 paired 개선이 실제로 나타날 때만 제한적으로 주장한다. Old expert를 보존했다는 이유만으로 전이를 주장하지 않는다.
- Caption 경로 불학습, dataset 접근/분할 불가능, mixed-task 간섭 없음, routing shortcut만의 효과도 유효한 부정적 결론이다. 실패가 나오면 가설에 맞게 데이터를 바꾸지 않는다.

## 7. 실행·보존·최종 산출물

G0 기술 smoke와 G1~G3 개발92개 학습을 완료했다. Tiny smoke의 메모리·시간을 본 실험 GPU-hours로 확대하지 않는다. 원격 실행은 서버별 지침 확인 후 scheduler로만 수행한다.

실행 시작 전 체크리스트:

- source 접근/라이선스, manifest 실제 표본 수와 전체 split overlap audit
- 공통 초기화, optimizer 세부값, loss/생성/scorer 구현 및 revision 고정
- G0 실제 smoke와 archived VQA parity 증거
- G1~G3 validation 결과/실패 기록, normalization/selection lock
- 계산된 job matrix와 비교 목록의 기계 검증, 독립 보존 공간 확보

최종 공개 산출물은 source/config/environment lock, 원 예측과 references 연결용 manifest, 전 task matrix, summary/task/resource CSV, paired 비교/CI, routing confusion, checkpoint hash inventory다. 모델/데이터/원 예측의 독립 백업은 Git과 별도로 계획해야 한다. 기존 VQA 산출물에는 어떤 쓰기도 하지 않는다.

## 8. 실행 기록: G0

- 공식 VQAv2 train/val questions·10인 annotation, COCO2014 captions, GQA v1.2 balanced train/val questions를 DC1 login node에서 확보했다. URL/원 ZIP·JSON SHA256과 실제 row 수는 [source-access 증거](artifacts/multimodal-source-access.json)에 보존한다.
- V/C 개발 manifest는 task당 train2,048/validation256/eval512, 총5,632개 unique image SHA256이다. Cross-task/split 및 과거 이미지 제외 조건의 overlap은0이며 과거 manifest6개를 사용했다. [분할 audit](artifacts/multimodal-development-data-audit.json). 이 2-task manifest는 **confirmation_eligible=false**다. G1/본 평가의 4-task manifest로 대체하지 않는다.
- 통합 검증에서 같은 크기로 변경된 이미지의 stat 기반 cache가 오래된 hash를 반환하는 오류를 발견했다. Timestamp를 신뢰하는 cache를 제거하여 매번 실제 bytes를 hash/decode한다. 수정 후 `pytest tests -q`: **199 passed**.
- SLURM **2477517**, RTX4090: 실제 VQA/caption expert 업데이트, backbone·old expert 불변, public-view routing 불변, EOS caption 생성, checkpoint roundtrip, **epoch 중간 중단/재개 후 가중치·예측·RNG 일치**, partial replay batch 보존을 확인했다. [GPU smoke 증거](artifacts/multimodal-g0-smoke.json).
- 기존 VQA expand/replay smoke의 예측 각각5개 파일, 총10개가 이전 GPU smoke와 byte-identical했다. [VQA parity](artifacts/multimodal-vqa-parity.json). 기존 핵심7개 source SHA256도 confirmation lock과 일치한다.
- 실제 새 replay uninterrupted/resumed 두 run의 원 예측 **20개를 독립 재채점**했다. 이는 tiny technical acceptance이며 정확도 향상 근거가 아니다.
- Caption smoke의 두 train 이미지에서 같은 문장 `a woman in a black top and white pants`를 생성했다. Caption loss도 단조 감소하지 않았다. 새 expert update·생성 기능은 확인했지만 caption 품질/학습 성공이나 G1 통과를 주장하지 않는다.
- 당시 Flickr30k 공식 신청 경로만 확인해 source 접근을 blocker로 기록했으나, 이후 사용자가 지정한 공개 [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) mirror를 확보하여 **접근 차단을 해소했다**. 원본 이미지 이용 조건은 유지하며 CSV의 실제 split을 사용한다. 이전 실행 증거 JSON의 blocker는 당시 상태 기록이지 현재 blocker가 아니다.

### 재현 실행

DC1에서 새 checkout 및 `requirements-multimodal.txt`의 의존성, Java PATH를 준비한 후:

```bash
# 다운로드는 login node에서만; 기존 source 파일을 변경하지 않음.
.venv/bin/python -m scripts.download_multimodal_sources \
  --output /scratch2/faransansj/multimodal_sources_20261013
# 실제 source 경로/라이선스와 과거 manifest를 지정해 새 데이터만 준비.
.venv/bin/python -m scripts.prepare_multimodal_data --help
# 기존 결과를 덮어쓰지 않는 새 SLURM job/output 경로 사용.
sbatch --wait scripts/multimodal_g0.sbatch
# 완성 run은 scripts.summarize_multimodal로 독립 재채점.
```

Source config 예시는 `configs/multimodal_sources_dc1.json`(V/C), `configs/multimodal_sources_vcg_dc1.json`(V/C/G)이다. 둘 다 개발용 source 부분집합이지 완성된 4-task 본 평가 설정은 아니다. 서버별 source/output/SLURM 경로를 확인하고 수정한 다음 사용한다.

### 나머지 방법의 실제 실행 검증

SLURM **2477586**, RTX4090에서 shared/capacity-matched/expand/frozen-first/independent의 두 task 전이를 추가로 실행했다. 각 학습 방법의10 exposures/4 steps, frozen-first의5 exposures/2 steps, 독립 모델의 이전 task 점수 보존, native/oracle 분리, router/replay memory 계약을 확인했다. Capacity-matched와 expansion의 최종 총 adapter/model 파라미터도 일치했다. [방법별 smoke 증거](artifacts/multimodal-methods-smoke.json). 모두5개 train/2개 validation 이미지/task를 사용한 기술 검증이며, 점수로 방법 순위를 매기지 않는다.

V/C/G의 실제3-source 준비도 완료했다. 세 task 모두2,048/256/512개이며 unique image SHA2568,448개, task/split 및 과거 이미지 제외 조건 overlap0이다. 두 개발 manifest는 모두 non-confirmation이다. 최종 독립 audit은 위 다섯 방법과 replay uninterrupted/resumed를 합친 **7개 run/76개 원 예측**을 검증했다. [전체 GPU smoke audit](artifacts/multimodal-g0-audit.json). 평가 단위는 원래 accuracy/CIDEr로 유지했으며 unlike-unit AP를 계산하지 않았다.

## 9. Flickr30k source 고정과 G1 실행

사용자가 지정한 `nlphuji/flickr30k` revision `2b239befc81b6e3f035ce6bd52f5f4d60f5625f7`의 CSV와 ZIP을 login node에서 다운로드하고 Hub LFS SHA256과 일치를 확인했다. `scripts/prepare_flickr30k_hf.py`가31,014개 JPEG 전체를 decode/hash하고5개 raw caption을 모두 보존했다. Train29,000/val1,014/test1,000의 **CSV row split**을 유지한다. HF loader의 top-level `test`는 split으로 쓰지 않는다. Mirror는 원 collection31,783개의 부분집합이며 이를 전체 원본과 동일하다고 부르지 않는다. 출처·개별 이미지 hash는 [Flickr source 증거](artifacts/multimodal-flickr30k-sources.json)에 보존했다.

새 source config는 `configs/multimodal_sources_full_dc1.json`이다. Flickr train만 개발 train/validation 후보로 사용하고 test에서 eval을 고정하는 기존 recipe를 유지한다. G1 grid는 `configs/multimodal_g1_jobs.json`의16개 독립 학습이며 전부 validation-only다. G1을 통과할 때만 `configs/multimodal_g2_jobs.json`의8개 양방향 pilot을 제출한다.

### 학습 전 검증과 재현 계약

- Source 변환·G1 adjudicator 추가 후 통합 테스트는 **279 passed**였다.
- 최초4-task manifest에서 GQA eval2개가 이전2-task G0의 train/validation 이미지 bytes와 겹쳐, **어떤 G1 학습/평가도 시작하기 전에** 이전2/3-task manifest를 추가 제외했다. 기존 manifest는 보존하고 `manifest_v2.json`을 새로 만들었다. 예전 상대 image-root manifest4개는 원본 row를 바꾸지 않은 root-resolved 사본을 사용하며 [원본/사본 hash](artifacts/multimodal-history-roots.json)를 보존했다.
- 최종4-task manifest는 task당2,048/256/512, unique image SHA256 **11,264개**, task/split/history overlap0이다. 과거 manifest8개를 제외했고 새 eval은 이전 멀티모달의 모든 split과도 disjoint다. [최종 분할 audit](artifacts/multimodal-full-data-audit.json).
- Tokenizer preflight에서 모든 입력/target이512-position 이내였다. Flickr reference2개는64-token 생성 한도보다 길다. 학습 target은 잘라내지 않고 사전 고정한 생성 한도도 바꾸지 않았다. [Preflight](artifacts/multimodal-full-preflight.json).
- COCO/Flickr train 이미지 각1개를 실제로 열어5개 reference와 내용 정합성을 확인했다. 이는 전수 semantic 검증이 아니다.
- G1 실행 전 source/config/manifest와16-job grid, bootstrap seed20261013을 [G1 lock](artifacts/multimodal-g1-lock.json)에 고정했다. Future eval 예측은 만들지 않는다.

```bash
sbatch --wait --array=0-15%4 \
  --export=ALL,MULTIMODAL_JOBS=configs/multimodal_g1_jobs.json \
  scripts/multimodal_array.sbatch
.venv/bin/python -m scripts.evaluate_multimodal_g1 \
  --runs artifacts/multimodal_runs/g1 \
  --manifest /scratch2/faransansj/multimodal_data_full_20261013/manifest_v2.json \
  --output artifacts/multimodal_report/g1
```

G1 adjudicator는16개 complete run의 config/source/scorer·예측을 독립 재채점하고 frozen-base/native/image-shuffled 결과를 모두 보존한다. Paired-image bootstrap2,000회는 고정 train-IDF CIDEr의 정확한 image별 기여를 사전 계산한 뒤 동일 draw의 corpus 평균을 재계산한다. 실제 corpus scorer와 중복 image draw의 일치 regression을 통과했으며 BLEU에는 이 분해를 사용하지 않는다. CLI exit0은 gate pass,1은 과학적 gate 실패,2는 불완전/변경된 증거다. 모든 task가 통과할 때만 normalizers를 발행하며 pilot을 자동 제출하지 않는다.

### G1 결과: 통과

SLURM **2478261**의16개 독립 학습을 완료하고 frozen-base/native/image-shuffled **12,288개 예측을 독립 재채점**했다. 공통 LR3e-4/5epoch의 네 task가 모두 gate를 통과했다. 아래 accuracy는0–100, CIDEr-D는 scorer 원 단위0–10이며 서로 평균하지 않는다.

| Task | Frozen base | 학습 후 | Gain 95% paired-image CI | Native − shuffled |
|---|---:|---:|---:|---:|
| V: VQAv2 | 0.0000 | 33.0469 | [27.5381, 38.8672] | 6.5234 |
| C: COCO captions | 0.02891 | 0.49321 | [0.40217, 0.52952] | 0.44902 |
| G: GQA | 0.0000 | 32.0313 | [26.5527, 37.8906] | 5.0781 |
| F: Flickr30k captions | 0.04067 | 0.11628 | [0.04211, 0.11502] | 0.10137 |

공통 조건의 빈 출력률은 모두0, 최빈 전체 출력 비율은 V21.48%/C5.47%/G19.92%/F2.34%였다. 이는 독립 학습 가능성의 증거이지 CL 이득·caption의 높은 절대 품질·여러 seed 재현성을 증명하지 않는다.

정규화 reference는 네 task 모두 LR3e-4/10epochs가 선택됐다. **V35.9765625, C0.6802684683477754, G33.59375, F0.15263336791624926**을 고정했다. [전체 G1 판정](artifacts/multimodal-g1-report.json), [정규화 상수](artifacts/multimodal-g1-normalizers.json)(SHA256 `9db5200bdd988a13dc0811bf6c599db77a6ee41688b07f195f9a67fcf2eba928`).

원 예측/config/events는 DC1의 `artifacts/multimodal_runs/g1/`와 로컬/원격 `artifacts/multimodal-g1-raw.tar.gz`에 보존했다. Checkpoint는 DC1 home의 각 run에 유지하며 압축 증거에는 포함하지 않는다. 최종 및 사용하지 않은 최초 manifest와 역사적 root-resolved 사본은 `artifacts/multimodal-full-manifests.tar.gz`로 보존했다.

## 10. G2 양방향 pilot: 통과

SLURM **2478596**에서7개 run을 완료했다. `V_C_replay_lora_seed17`은 모델/학습/config 생성 이전 scorer 초기화에서 PTB tokenizer가10,242개 caption에10,243줄을 반환해 중단됐다. 초과 stdout 내용이 저장되지 않아 근본 원인은 확인하지 못했다. 코드·환경·설정을 변경하지 않은 **2478885**의 동일 array index2 재실행은 완료됐다. 원 실패 로그도 raw archive에 보존했으며 실패 trial을 방법 성능에서 조용히 제외한 것이 아니다.

`scripts/evaluate_multimodal_g2.py`가8개 run의 **10,240개 원 예측을 독립 재채점**했다. Shared/replay/expansion의 첫 task 원 예측은 해당 G1 공통 조건과 byte-identical했다. 모든 방법이 task당640 optimizer steps/10,240 image exposures를 사용했다. Replay의 후속 current exposure는 절반이며, 나머지는 과거 이미지 replacement다.

| 순서 | 방법 | 최종 정규화 평균 ↑ | 첫 task 망각 ↓ | 후속 task gain ↑ |
|---|---|---:|---:|---:|
| V→C | shared | 0.38743 | 33.04688 | 0.52646 |
| V→C | capacity-matched | 0.46494 | 35.15625 | 0.61180 |
| V→C | replay | 0.69765 | 3.35938 | 0.38718 |
| V→C | expansion | **0.82445** | **0.00000** | 0.49618 |
| C→V | shared | 0.56834 | 0.44972 | 38.59375 |
| C→V | capacity-matched | 0.54617 | 0.59826 | 38.32031 |
| C→V | replay | 0.79649 | 0.03148 | 32.89063 |
| C→V | expansion | **0.87554** | **0.00000** | 36.91406 |

망각/gain은 해당 task의 원 단위다(V accuracy points, C CIDEr-D). 두 순서 모두 후속 gain>0, shared보다 작은 망각, 높은 최종 정규화 평균을 만족했다. 정규화 평균의 expansion−shared 차이는 V→C **+0.43703**, C→V **+0.30720**이다. Expansion은 이 pilot에서 replay/capacity 대조보다도 최종 점수가 높지만, seed17/두 family pilot이므로 일반적 CL 우월성·family 내 routing·positive transfer를 확정하지 않는다.

[전체 G2 판정](artifacts/multimodal-g2-report.json), [실행/재시도 기록](artifacts/multimodal-g2-execution.json). 예측/config/events 및 원/재시도 job 로그는 로컬/원격 `artifacts/multimodal-g2-raw.tar.gz`에 보존했다. Checkpoint는 DC1 home의 원 run에 남겨둔다.

## 11. G3 실행 계약

G2 통과 후 `configs/multimodal_g3_jobs.json`의64개 grid+4개 frozen-first를 제출했다. 기존 네 순서·LR/epoch grid·seed17·validation-only·공통 주 조건을 유지한다. [G3 lock](artifacts/multimodal-g3-lock.json)에 G1/G2 report, normalizers, source, jobs와 adjudicator hash를 실행 전에 고정했다.

`scripts/evaluate_multimodal_g3.py`는68개 전체 run을 독립 재채점하고, 네 순서 평균으로 replay 설정을 선택한다. G3 gate의 family 후속 gain은 **첫 task를 제외한 family별6개 전이의 원 단위 평균**으로 계산한다. 공통 조건 expansion의 모든 stage/task 생성 guard도 확인한다. Validation 결과만으로 confirmation 실행을 대체하지 않는다. G3 통과와 pre-eval hash lock 전에는 새 seed/eval을 실행하지 않는다.

G3 학습은 SLURM **2479153**에서68개 모두 완료했다. 원 matrix/config/events 및68개 job 로그를 로컬/원격 `artifacts/multimodal-g3-raw.tar.gz`에 보존했다. Confirmation 전에 동일 frozen G3 bank로 routing 진단 도구의 validation smoke를 수행하고 새 분석/진단 도구도 hash lock한다.

### G3 결과: family-average gate 통과, 개별 실패 보존

68개 run의 **243,712개 원 예측을 독립 재채점**했다. 공통 expansion의 후속 family 평균 gain은 VQA **+14.16016 accuracy points**, caption **+0.27547 CIDEr-D**였고 모든 stage/task 생성 guard를 통과했다. 따라서 사전 G3 gate는 통과했다. 단, Flickr의 후속 gain은 `V→C→G→F`에서 **−0.03630**, `V→G→C→F`에서 **−0.02637**이었다. Family 평균 통과를 모든 task의 신규 학습 성공으로 표현하지 않는다.

네 고정 순서의 family-balanced normalized final score 평균:

| 공통 조건 방법 | Validation 평균 ↑ |
|---|---:|
| Shared | 0.53799 |
| Capacity-matched shared | 0.68014 |
| Replay | **0.95554** |
| Expansion | 0.85036 |
| Frozen-first | 0.31819 |

Validation-tuned replay는 LR3e-4/10epochs, 평균 **0.99755**가 선택됐다. Expansion의 주 조건은3e-4/5epochs 그대로다. **이 개발 결과에서 expansion은 common/tuned replay를 이기지 못했다.** G4는 이 실패를 숨기지 않고 사전 고정8개 새 seed로 검증하며, tuned replay32개를 포함한 총224개 조건을 사용한다.

[G3 전체 판정](artifacts/multimodal-g3-report.json), [실행/부정적 전이](artifacts/multimodal-g3-execution.json). G1~G3의108개 checkpoint,11,264개 선택 이미지, 원 BLIP 초기화 자산도 DC1과 독립적인 로컬 백업을 만들어 SHA256 일치를 확인했다. [백업 위치·hash·보존 정책](artifacts/multimodal-backups.json). 대용량 데이터/모델 archive는 Git에 넣지 않으며 NAS는 읽기만으로 atime이 갱신되지 않는다는 서버 정책을 따른다.

G4의 paired-seed bootstrap RNG seed는 **20261014**로 eval 이전에 고정했다. 기존8 seed/4 order/20,000 resamples/6개 주 Holm 및 별도 tuned-replay2개 Holm 설계는 바꾸지 않는다. 분석·routing 도구 통합 후 `pytest tests -q`: **305 passed**. 실제 GPU routing smoke는 별도이며 테스트 수로 대체하지 않는다.

## 12. Pre-eval lock와 G4 실행

G3의 `V→C→G→F` native expansion bank로 전체256개 validation/task를 사용한 다섯 routing 조건을 실행했다. SLURM **2482136**에서 native의14개 pre/after 예측 파일이 부모와 byte-identical했고, router checkpoint tensor 및 expert bank 불변을 확인했다. **16,896개 예측을 독립 재채점**했다. Family-restricted는 별도 refit이 아니라 native per-stage 계수의 고정 사본에 후보 mask만 적용한다. 도입 전 같은 family expert가 없으면 base-only로 평가하고 명시적으로 unroutable로 기록한다. Oracle은 아직 없는 expert를 사용하지 않으며 pre-task 점수를 정의하지 않는다.

첫 smoke **2482127**은 boolean generation flag를 잘못 거부하는 진단 reader 오류로 중단됐다. 이를 고쳤고, 독립 rescorer에서 발견한 final-stage confusion 집계, route-error 분모, AF 부호도 eval 이전에 수정했다. Confirmation lock preflight는 실제 structured scorer-lock을 수용하도록 교정했다. 원 실패 로그·부분 예측은 보존하며, 최종 테스트는 **307 passed**다. [실제 검증/수정 기록](artifacts/multimodal-routing-validation-execution.json), [독립 routing audit](artifacts/multimodal-routing-validation-audit.json).

이 validation bank에서 question-only caption 점수는 task-ID oracle보다 높았다. **Task-ID oracle은 expert/task 식별 정확도의 상한 진단이지 task 성능의 수학적 상한이 아니다.** 다른 task의 expert가 정답 expert보다 더 잘 답할 수 있다. 이 탐색 결과로 주 native joint router를 바꾸지 않는다.

[Confirmation lock](artifacts/multimodal-confirmation-lock.json) SHA256: `4294af072de6385c28192829b60edab5f8d4d24fec1fd4c927b7c3ee1c1f2161`. 코드·자산·scorer·정규화·환경·bootstrap·G1/G2/G3 보고서와224개 training job 및32-bank/160-condition routing template을 고정했다. 최초 confirmation 예측이 없는 상태에서 remote bytes와 모든 lock 조건 일치를 확인한 뒤 제출했다. [정확한 source snapshot](artifacts/multimodal-confirmation-source.tar.gz), [의존성 inventory](artifacts/multimodal-confirmation-environment.json).

G4는 `configs/multimodal_g4_jobs.json`/`scripts/multimodal_array.sbatch`로 실행한다. G5의 `configs/multimodal_g5_jobs.json`은 자기 해시 순환을 피하기 위해 lock 경로/SHA를 job template 안에 넣지 않는다. `scripts/multimodal_routing_array.sbatch`가 외부 `MULTIMODAL_CONFIRMATION_LOCK`/`MULTIMODAL_CONFIRMATION_LOCK_SHA256`에서 주입하며 evaluator가 부모 run과의 binding을 검증한다. 모델 재학습은 없다.
