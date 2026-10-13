# VQA 실험 종료 기록

기준일: 2026-10-13. 상태: **계획한 readout 8-seed confirmation과 routing 대조를 종료한다.** 새로운 학습·조건 재선택은 하지 않는다. 현재 코드·평가 결과는 멀티모달 후속 연구의 변경 불가 비교 기준선이다.

## 결론과 범위

고정 BLIP의 질문 encoder·답변 decoder에 task별 rank8 LoRA expert를 추가하고 이전 expert를 동결하며 이미지/질문 특징으로 expert를 선택하면, 이번 VQA CL 조건에서 shared adapter보다 최종 정확도가 높고 망각이 작았다. 총 adapter 파라미터·optimizer step·처리 예제 수를 맞춘 shared 대조에서도 개선했다. 이는 **확장·보존·routing을 결합한 방식**의 근거이지 각 요소의 독립적인 인과효과나 모든 MoE의 우수성 증거가 아니다. CL-MoE 논문의 방법 자체를 재현한 결과로 부르지 않는다.

8-seed 평균, AP는 %, AF는 pp:

| Protocol | Shared AP / AF | 주 5-epoch expansion AP / AF | Capacity-matched shared AP |
|---|---:|---:|---:|
| CL4VQA | 21.95 / 18.53 | 30.92 / 1.33 | 20.59 |
| Dataset stream | 14.91 / 13.82 | 23.99 / 1.02 | 17.47 |

Shared 대비 AP 차이는 +8.97/+9.08pp, capacity-matched 대비 +10.33/+6.53pp. 네 비교 모두 주 AP family의 Holm p=0.046875. 후속 task 학습 전후 평균 gain은 +20.09/+6.85pp여서 낮은 망각만의 결과가 아니다. 정확한 CI와 모든 비교는 [README](README.md#새-공식-평가-고정된8-seed-confirmation)와 [비교 CSV](artifacts/readout_report/confirmation/comparisons.csv)를 따른다.

### 성공과 함께 보존할 부정적 결과

- 주 AP 비교 6개 중 5개 증가, 1개 감소. CL4VQA 5-epoch expansion은 validation-selected 10-epoch replay보다 AP가 3.52pp 낮다. 최강 baseline 대비 주 가설은 이 protocol에서 실패했다.
- 독립 validation 선택 10-epoch expansion의 replay 대비 CL4VQA +1.43pp는 같은 exposure/step이지만 exploratory다. 주 검정으로 바꾸지 않는다.
- Adaptive growth는 expert 수를 줄였지만 주 확장보다 AP가 CL4VQA 1.96pp/stream 4.91pp 낮다. 효율 향상 성공으로 부르지 않는다.
- 질문 routing의 효과는 관찰했지만 task 식별 shortcut을 제거하지 못했다. Oracle AF=0은 실제 router AF=0을 뜻하지 않는다.
- TextVQA·recognition·causal 약점, ScienceQA 10-epoch 음의 gain을 숨기지 않는다.
- 초기 vision-side 74개 실험은 학습량 일치 비확장군 대비 우위 미입증. [RESULTS.md](RESULTS.md)는 더 이전 LLM-side 실패 기록이다. Cohort를 합치거나 readout 위치만의 효과로 해석하지 않는다.

## 완료 증거

| 항목 | 완료 기록 | 근거 |
|---|---|---|
| 학습 | 128 runs; 8 seeds × 2 protocols × 8 조건 | `artifacts/readout_report/confirmation/audit.json` |
| 평가 전 고정 | 모든 run contract가 lock과 일치 | `artifacts/readout-confirmation-lock.json`, `artifacts/readout_report/confirmation/diagnostics.json` |
| 독립 재채점 | 학습 3,227,360개 예측 | 학습 `audit.json` |
| 첫 task parity | 16 protocol/seed 조합 | `diagnostics.json` |
| 용량·학습량 대조 | 16 pair에서 total parameters/exposures/steps 일치 | `diagnostics.json`; active parameters/FLOPs/memory는 다름 |
| Routing | 32 banks × 5 variants = 160 조건; 추가 expert 학습 0회 | `artifacts/readout_report/routing_confirmation/audit.json` |
| Routing 재채점 | 3,078,400개 예측; native 1,264개 파일 일치 기록 | routing `audit.json`, `artifacts/readout-confirmation-execution.json` |
| 합계 | 6,305,760개 재채점 예측 | 위 두 audit |
| 모델 보존 | 최종 checkpoint 128개 scratch→DC1 home 복사 및 checksum 기록 | `readout-confirmation-execution.json` |
| 보고서 | summary/comparisons/task_results/resources CSV, 원 matrix와 진단 | `artifacts/readout_report/confirmation/` |

### 이번 마무리에서 실제 재확인한 것

로컬 Python 검증으로 실행 기록에 저장된 **7개 artifact SHA256**(lock, environment, 학습 audit/comparisons/diagnostics, routing audit/summary)이 현재 파일과 일치함을 확인했다. 학습 128개, 16 protocol/seed 조합당 8조건, 지정 seed 집합, routing 160조건/32 banks, 재채점 수 합계, 주 비교 6개(5개 양수/1개 음수)와 보정 p를 확인했다.

이는 **기존 audit와 산출물 무결성 확인**이다. 이번에 630만 예측을 다시 전수 재채점하거나 학습·테스트 suite를 재실행한 것은 아니다. 모델/runner 코드는 바꾸지 않았다.

## 남아 있는 제한: 실험 미완료와 구분

- Fit 2,048/task, 고정 task 순서, 제한된 eval 표본. Seed CI는 새로운 이미지/task order의 불확실성을 포함하지 않는다. VQAv2/CL4VQA는 source 단일 정답 exact match이며 official consensus accuracy가 아니다.
- Backbone 사전학습 이미지 노출 독립성, SOTA, 새로운 MoE 구조, 성공한 adaptive growth를 주장하지 않는다.
- 서버 실시간 큐·현재 checkpoint 존재 여부는 이번에 조회하지 않았다. 보존 상태는 저장 당시 기록이다. DC1 home 복사도 독립 백업은 아니다. Scratch의 30일 atime 정리 위험은 남는다.
- `sacct` accounting DB 실패 기록 때문에 GPU-hours/throughput 우위를 보고하지 않는다. 알려진 실패를 재실행해 확인하지 않았다.
- Git에는 코드·설정·최종 보고서를 보존한다. 대용량 원 예측/모델/데이터와 third-party checkout은 Git 밖의 기존 위치에 유지한다. Git baseline은 완전한 데이터 백업이나 단독 실행 패키지가 아니다.

## 후속 연구와 경계

이 cohort는 종료하며 새 결과로 기존 표·설정·원 예측을 덮어쓰지 않는다. 새 데이터, 코드 계약, 출력 namespace에서 heterogeneous vision-language task CL을 연구한다. 기존 VQA 점수와 후속 결과를 직접 연결해 개선량을 계산하지 않고, 후속 공통 backbone/전처리/손실로 baseline과 expansion을 다시 비교한다. 후속 연구는 별도 브랜치의 설계·실험계획에 따르며 아직 실행하지 않았다.
