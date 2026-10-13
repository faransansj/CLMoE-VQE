# 멀티모달 task 확장 실험계획

작성: 2026-10-13. 상태: **실험 전 계획, 데이터 확보·구현·GPU 실행 전**. [설계](MULTIMODAL_DESIGN.md)와 [기계 판독 계획](configs/multimodal_research.json)을 함께 따른다. 아래 조건은 시작안이며, 변경은 validation trial 이전에 revision으로 기록한다. Confirmation eval을 본 뒤 조건·지표·표본·seed를 바꾸지 않는다.

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

이번 작업은 설계까지다. 새 학습을 제출하거나 데이터를 다운로드하지 않았다. Run당 GPU 메모리·walltime을 G0/G1에서 측정하기 전에는 GPU-hours/완료 날짜를 추정하지 않는다. 원격 실행은 서버별 지침 확인 후 scheduler로만 수행한다.

실행 시작 전 체크리스트:

- source 접근/라이선스, manifest 실제 표본 수와 전체 split overlap audit
- 공통 초기화, optimizer 세부값, loss/생성/scorer 구현 및 revision 고정
- G0 실제 smoke와 archived VQA parity 증거
- G1~G3 validation 결과/실패 기록, normalization/selection lock
- 계산된 job matrix와 비교 목록의 기계 검증, 독립 보존 공간 확보

최종 공개 산출물은 source/config/environment lock, 원 예측과 references 연결용 manifest, 전 task matrix, summary/task/resource CSV, paired 비교/CI, routing confusion, checkpoint hash inventory다. 모델/데이터/원 예측의 독립 백업은 Git과 별도로 계획해야 한다. 기존 VQA 산출물에는 어떤 쓰기도 하지 않는다.
