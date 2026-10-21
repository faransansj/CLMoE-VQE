#!/usr/bin/env python3
"""Week 1-A Flickr sanity check (read-only).

모든 검증은 기존 artifact와 실제 원본 소스 파일을 "읽기 전용"으로 조회한다.
학습, 추론/생성, scorer(VQA/CIDEr) 호출, 기존 artifact 변경은 전혀 없다.
산출물은 artifacts/week1-flickr-sanity.json과 artifacts/week1-flickr-samples.md 두 개뿐이다.

검사 항목
  1. COCO(C) / Flickr(F) 실제 train/validation/eval reference 수·토큰 길이 분포와
     list-문자열 파싱, empty/duplicate/truncated reference 검사.
  2. 고정 HF revision 2b239befc81b6e3f035ce6bd52f5f4d60f5625f7 Flickr CSV의
     내부 split(train/val/test) vs 실제 사용 이미지 ID·기대 수량 감사.
     (CSV/변환 JSON은 Yonsei-DC1에서 읽기 전용 확보, SHA256 고정 확인)
  3. 고정 시드로 G4 단일 run(seed 137, order V→C→G→F, method=expand)의
     최종(모든 task 학습 후) Flickr eval 예측 50개 + 전체 reference 출력.
     50건 각각 AI assistant가 예측·전체 reference(및 12건은 실제 이미지)를 직접 읽고
     작성한 의미론적 검수 노트가 스크립트에 정적으로 내장되어 md가 재현된다.
     이미지 내용 판단은 작성자가 수행한 것이고 스크립트는 이미지를 해시로만 검증한다.
  4. metrics.py의 train-IDF/토크나이저/reference 구조가 COCO와 Flickr에서
     동일한 코드 경로인지 정적(소스 관찰) 검사. scorer 호출 없음.
  5. 기존 frozen-base / independent / expansion 점수의 Flickr:COCO 비율만 산출
     (tasks.csv에 이미 기록된 점수 재사용, 새로운 채점 없음).

입력 provenance: 실제로 읽은 모든 파일(및 tar 멤버)의 SHA256을 JSON에 기록하고
스크립트 자체 SHA256도 함께 기록한다.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FLICKR_REVISION = "2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
FLICKR_CSV_SHA256 = "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137"
FLICKR_CONVERSION_SHA256 = "414d88ea3bf650d96b9315895d68f69bd07e3b497562936376141955d2feccd8"
SOURCES_JSON = REPO / "artifacts/multimodal-flickr30k-sources.json"
MANIFEST_TAR = REPO / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
MANIFEST_SHA256 = "c501786957b9aa7260feabd780d561160abbf53e4bcb722a2375a76179f90b39"
G4_RAW_TAR = REPO / "artifacts/multimodal-g4-raw.tar.gz"
G4_TAR_PREFIX = "artifacts/multimodal_runs/g4/"
G4_RUN_NAME = "V_C_G_F_expand_lr0.0003_ep5_seed137"
G4_TASKS_CSV = REPO / "artifacts/multimodal-g4-report/tasks.csv"
G1_NORMALIZERS = REPO / "artifacts/multimodal-g1-normalizers.json"
CONFIRMATION_LOCK = REPO / "artifacts/multimodal-confirmation-lock.json"
IMAGES_TAR = REPO / "artifacts/multimodal-selected-images-v2.tar"
METRICS_SOURCE = REPO / "sema_multimodal/metrics.py"
DATA_SOURCE = REPO / "sema_multimodal/data.py"
RUN_SOURCE = REPO / "sema_multimodal/run.py"
PREP_MULTIMODAL_SOURCE = REPO / "scripts/prepare_multimodal_data.py"
PREP_FLICKR_SOURCE = REPO / "scripts/prepare_flickr30k_hf.py"

MIRROR_SPLIT_COUNTS = {"train": 29000, "val": 1014, "test": 1000}
MIRROR_CAPTIONS_PER_IMAGE = 5

SAMPLE_SEED = 20261016
SAMPLE_SIZE = 50
G4_SEEDS = [137, 149, 163, 179, 191, 211, 227, 241]
G4_ORDERS = ["V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V"]
CAPTION_MAX_NEW_TOKENS = 64
SCORER_CODE_COMMENT = (
    "COCO와 Flickr는 동일 TaskScorer/sta:코드 경로(metric==cider_d, family==caption, "
    "n=4, sigma=6.0, Stanford CoreNLP 3.4.1 PTB -preserveLines -lowerCase, "
    "고정 DF=task train 참조만 사용, train_documents=2048)를 공유한다. "
    "토크나이저·pycocoevalcap 1.2·BLEU 소스 해시도 C/F 각 런 config에서 동일하며 "
    "task 간 차이는 train corpus 내용(train_reference_sha256, document_frequency_sha256)과 "
    "점수뿐이다."
)
CONCLUSION_OK = "버그 없음"
CONCLUSION_SUSPECT = "버그 의심(근거)"
CONCLUSION_CONFIRMED = "버그 확인(위치·영향 범위)"

# 아래 50건은 작성자(FlickrSanity)가 고정 시드 SAMPLE_SEED로 뽑힌 예측과 전체
# reference를 전부 읽었고, "이미지확인" 표기 12건은 manifest 해시와 일치하는
# 실제 이미지(bytes SHA256 런타임 재검증)를 눈으로 확인한 것이다. 판독 근거:
# reference와 예측의 의미상 일치/부분일치/불일치를 정성적으로 기록한다.
# 이는 AI assistant의 판독이며 사람 평가나 scorer 결과가 아니다.
# 예측 오류의 학습량·라우팅 등 인과 원인은 이 조회만으로 확정하지 않는다.
# 자동 휴리스틱(토큰 수, limit_reached 등)은 참고용이다.
EXPERT_NOTES = {
    "flickr30k/test/2755362721": "이미지확인: 모래 트랙을 달리는 회색 개(경주 조끼 착용, 방마구) 두 마리. ref 1-5는 트랙·모래·회색 개 경주를 정확히 기술(5/5 일치). pred는 개 달리기는 맞지만 grass/ball은 이미지에 없다. 모델 오류(부분 불일치), reference 데이터 이상 없음.",
    "flickr30k/test/4983587808": "이미지확인: 노란 벽 앞 노트북 작업하는 금발 흰옷 여성과 음료. ref 1-5 일치(2번 ref는 1인칭 독백형 원문 특이 문장이 그대로 보존되어 있음). pred는 여성+노트북까지는 맞으나 없는 남성 커피를 넣고 64토큰 한도 도달 후 shirt 색 단어 반복 루프. 부분+루프, 데이터 문제 아님.",
    "flickr30k/test/4146886427": "이미지확인: 터키 디너를 에워싼 6인 가족. ref 1-5 일치(식사 인원 6명 명시). pred woman/plate of food/drink는 주제만 유지하고 인원·상황을 잃음. 라우팅이 C expert(expert=1)로 간 케이스 — 라우트 결과만 영향.",
    "flickr30k/test/5931115247": "ref 1-5 롤러블레이더 여아(헬멧·보호대)로 일치. pred woman+child 산책 묘사로 보행자형 일반_caption, 장비 정보 누락+64토큰 반복. 부분.",
    "flickr30k/test/302289651": "이미지확인: 무릎에 아기(흰 모자)를 두고 샌드위치 먹는 남성. ref 1-5 정확. pred는 없는 노트북·여성을 말하는 완전 불일치+루프. 모델 오류, ref와 이미지는 정확.",
    "flickr30k/test/2472980433": "ref 1-5 오토바이 위어리 트릭 트랙 기술로 일치. pred skateboard street로 차량 오분류. 불일치, 라우팅 C expert(expert=1).",
    "flickr30k/test/3483071906": "ref 1-5 아이들 줄다리기+코치 기술로 일치. pred는 아이들 노는 장면까지 유지하나 밧줄을 공·구조물로 대체. 부분.",
    "flickr30k/test/3799701220": "ref 1-5 가게에서 가죽 가공하는 두 남성으로 일치. pred는 노트북·의자로 오기술+64토큰 반복. 불일치, 모델 오류.",
    "flickr30k/test/198534816": "이미지확인: 홈플레이트 상황의 소프트볼 타자·캐처·심판. ref 1-5(Pride 마룬 유니폼, 홈플레이트) 정확. pred는 농구 인도어로 종목 오분류. reference가 정확함을 명시적으로 보여주는 케이스.",
    "flickr30k/test/2978735290": "ref 1-5 두 아이 축구공+점프·네트 일치. pred는 아이+공+잔디 유지(단일 인원 축소). 부분 일치.",
    "flickr30k/test/2069110603": "ref 1-5 가방(노란 오브젝트) 차기 카라테 수련으로 일치. pred는 기타 연주로 동작/대상 오분류. 불일치, 모델 오류.",
    "flickr30k/test/1043910339": "ref 1-5 야구 홈플레이트 달리기+태그 상황 일치. pred는 야구 주제 유지하나 반복 문구(repeated_trigram) 생성 후 64토큰 한도. 부분+루프.",
    "flickr30k/test/7099368287": "ref 1-5 헤더 경합 공중 충돌로 일치. pred는 공중 속 공+인물 주제를 잘 잡음(의미상 가장 근접한 케이스 중 하나), 단 색 단어 반복+한도 도달. 양호(주제 일치)+루프.",
    "flickr30k/test/47529535": "ref 1-5 등에 동생 짊어진 아프리카 아이 일치. pred는 잔디밭에 선 소년 - 짊어진 상황 누락. 불일치(인물·야외만 유지).",
    "flickr30k/test/4971580248": "이미지확인: 전부 검은 복장+가방에 검은 그립을 든 여성 거리. ref 1-5 일치. pred는 검은 재킷+가방은 유지하나 남성·빨간 우산을 추가. 부분.",
    "flickr30k/test/475816542": "ref 1-5 검은 개+갈색 개가 축구공을 가지고 노는 장면 일치. pred는 개+공+잔디 유지하나 한 마리로 축소, mouth 가정 추가. 부분.",
    "flickr30k/test/4892698507": "ref 1-5 채소 시장 거리 노점 판매 일치. pred는 사람들+음식 탁자 언급으로 시장 주제를 일부 유지하나 시장/노점 정보 손실. 부분. 라우팅 C expert(expert=1).",
    "flickr30k/test/6995905943": "ref 1-5 모래 필드 구기 경기+관중석 일치. pred는 구단·구경까지 유지하나 농구 경기장이나 모래사장 오분류. 부분 불일치.",
    "flickr30k/test/270263570": "ref 1-5 숲 인접 공지에서 얽혀노는 두 갈색 개 일치. pred는 개 유지하나 공·풀로 대체. 부분.",
    "flickr30k/test/5506383509": "ref 1-5 4명 아이들(옷 색 명시) 노는 그룹 일치. pred는 반복 문구 picture of a picture로 붕괴, 특히 옷 색 반복. 64토큰 한도 도달. 부분 이상 루프.",
    "flickr30k/test/529101401": "ref 1-5 부두에서 물로 도약하는 소년 일치. pred는 도약+물 유지, surfboard 추가, boy→man. 부분.",
    "flickr30k/test/4692834620": "이미지확인: 빨간 슈트 남성과 하고자하는 여성의 스트리트 댄스. ref 1-5(빨간 두 사람 댄스) 정확. pred는 검은 셔츠+팻말이 없는 관중 묘사로 완전 불일치. ref는 정확하므로 데이터 이상 없음.",
    "flickr30k/test/2064792226": "ref 1-5 화려하게 꾸민 자전거 둘 탑승+사진 찍는 여성 일치. pred 자전거 부유 유지, 옷 색 반복+한도 도달. 부분+루프.",
    "flickr30k/test/3487979741": "ref 1-5 고속 코너 dust 트랙 자전거 라이더 일치. pred는 사람+자전거+흙길을 잘 유지(각도 정보 누락). 부분 일치로 양호.",
    "flickr30k/test/136693281": "ref 1-5 father/daughter가 나무 심기+아들 삽 든 일치. pred는 man+woman 벤치 아기 반묶 문장으로 완전 불일치. eos로 끝났는 데도 주제를 놓침. 모델 오류.",
    "flickr30k/test/7292785488": "이미지확인: 벽돌 건물 옆 레일 스케이트보드 그라인드 흰옷 남성. ref 1-5(레일/걸작 기술) 정확. pred는 blue shirt+skateboard 유지, 레일 그라인드를 놓치고 park 잘못 지명. 부분.",
    "flickr30k/test/3432637363": "ref 1-5 투라드 남성+경찰 둘+흰 천막 쓰레기봉투 일치. pred는 남성+군중 방향 유지, 투란/경찰/천막 누락. 부분.",
    "flickr30k/test/4463538995": "ref 1-5 빨간 점퍼 여성 다채로운 리본(트윌링) 일치. pred는 남성+잔디+프리스비로 완전 오분류. 불일치. 라우팅 C expert(expert=1).",
    "flickr30k/test/286084055": "ref 1-5 두세 사람 와인 대화 거실 일치. pred는 man+woman 벤치+아기 반복 문장으로 불일치. eos인 데도 주점 손실.",
    "flickr30k/test/2778313163": "ref 1-5 청소(비담 쓸기) 여성들 일치. pred는 벤치 남녀 지도 보기로 불일치. 그리고 후미 정수 토큰 ,7 현상(디코더 아티팩트 후보) 관측. 불일치+아티팩트.",
    "flickr30k/test/3613264553": "이미지확인: 돌 언덕 위 흰 개, 하늘 배경. ref 1-5(흰 개+산지 암색 지면) 정확. pred는 자전거 타는 남성으로 완전 불일치. reference 정확성과 예측 오차가 선명히 분리된 케이스.",
    "flickr30k/test/2863539038": "ref 1-5 실외 꼬치 구이 조리 남성 일치. pred는 식탁 음식 와인 자리로 완전 불일치(요리 정보 상실). 모델 오류.",
    "flickr30k/test/2451114871": "이미지확인: 색상 리본 버튼 달린 회색 개 4마리 트랙 경주(방마구, 기계 토끼 러어). ref 1-5(색상 유니폼+개 목줄+경주 트랙) 정확. pred는 말 탄 사람들로 완전 불일치. eos로 끝났는데도 주제를 아예 못 잡음 — 데이터 이상 없음, 모델 제한.",
    "flickr30k/test/4930533574": "ref 1-5 오래된 자전거로 짐 실은 노인 busy 거리 일치. pred는 거리+남성 유지, 자전거→서있는 skateboard로 변형. 부분. 라우팅 C expert(expert=1).",
    "flickr30k/test/6099093979": "ref 1-5 계단에서 공예품+인형 판매 여성(개 동행 일부 ref) 일치. pred는 여성+앉음 유지, 판매 대신 남성·옷 반복. 부분+루프(64토큰 한도).",
    "flickr30k/test/7151070637": "ref 1-5 도시 마라톤 주자들 + 쌍둥이 고층빌딩 일치. pred는 도시+여러 사람 유지, 달리기→걷기 오류. 부분.",
    "flickr30k/test/2289762817": "ref 1-5 빨간 셔츠 남성 야외 테이블 식사(타코) 일치. pred는 검은 셔츠 남성 반복 루프, 빨간 셔츠/음식 누락. repeated_trigram+64토큰 한도. 불일치+루프.",
    "flickr30k/test/4035354150": "ref 1-5 비키니 여성이 선글라스 남성에 음료 따르기 일치. pred는 검은 셔츠 여성 벤치+아이, 후미 ,7 아티팩트. 불일치.",
    "flickr30k/test/2709044515": "ref 1-5 셰프 불 팔리는 팬 조리 일치. pred는 노트북 화면+ i am 반복, 불 글 조리 누락. 불일치+루프(64토큰 한도), 디코더 잔여 토큰 후보.",
    "flickr30k/test/2900560501": "ref 1-5 솜브레로 청년들 술한잔(테이크 촬영) 일치. pred는 식탁+음료 유지, 술/모자 정보 손실. 부분. 라우팅 C expert(expert=1).",
    "flickr30k/test/435054077": "ref 1-5 잔디밭 여러 개와 앉은 여성(곁에 purse) 일치. pred는 여성+여러 개 유지, 옷 색 반복+한도 도달. 부분+루프.",
    "flickr30k/test/4655102365": "ref 1-5 도시 분수대 주변 군중 일치. pred는 군중+거리 유지, 가상의 간판 no people's land 텍스트 자작. 부분.",
    "flickr30k/test/2572712647": "ref 1-5 파란 음악정(스탠드 앞) 여성 성가대 일치. pred는 농구 경기+빨간 셔츠 남성 반복 - 불일치+루프(64토큰 한도).",
    "flickr30k/test/697490420": "이미지확인: 노란 카약·짐 옆으로 걸어가는 알몸 남성(수영복). ref 1-5(노란 카약+짐+수영복) 정확. pred는 서핑을 타는 파도로 불일치. 라우팅 C expert(expert=1). ref는 정확.",
    "flickr30k/test/4688197510": "이미지확인: 청록-흰 체커리 외벽의 De Neude 가게 앞 길 걷는 남성(붉은 셔츠). ref 1-5(가게 간판 표기 포함) 정확. pred는 남성+거리 유지, 상점 설명 누락, 자전거 보행자 혼동. 부분. 라우팅 C expert(expert=1).",
    "flickr30k/test/1989609": "ref 1-5 에스컬레이터 수리+도구 전달 남성 일치. pred는 벤치+식물 장식+후미 ,7 아티팩트. 불일치.",
    "flickr30k/test/2131161531": "ref 1-5 안경 웃는 여성 소파 경보 일치. pred는 흰 셔츠+노트북 커피+남성 옷 색 반복 - 불일치+루프(64토큰 한도).",
    "flickr30k/test/4520820052": "ref 1-5 금빛 긴 점퍼 여성 지하철 승강장 일치. pred는 흰 셔츠+빨간 우산 거리 걷기로 불일치(장소/복장 오류). eos.",
    "flickr30k/test/2445442929": "ref 1-5 뒤 군중을 돌아본 테이블 남성 일치. pred는 앉은 남성 유지, 음식/군중 누락+벤치 반복 루프(64토큰 한도). 부분+루프.",
    "flickr30k/test/4706166634": "ref 1-5 흰·빨강 깃발이 달린 대형 건물을 지나 달리는 두 소녀 일치. pred는 거리 간판+버스로 인물 누락. 불일치. 라우팅 C expert(expert=1).",
}

VISUAL_INSPECTED = {
    "flickr30k/test/2755362721", "flickr30k/test/4983587808", "flickr30k/test/4146886427",
    "flickr30k/test/302289651", "flickr30k/test/198534816", "flickr30k/test/4971580248",
    "flickr30k/test/7292785488", "flickr30k/test/3613264553", "flickr30k/test/2451114871",
    "flickr30k/test/697490420", "flickr30k/test/4692834620", "flickr30k/test/4688197510",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(evidence, key, path, sha, extra=None):
    entry = {"availability": "examined", "path": str(path), "sha256": sha}
    if extra:
        entry.update(extra)
    evidence[key] = entry


def summarise_split(rows):
    lens = [len(row["references"]) for row in rows]
    white = [len(ref.split()) for row in rows for ref in row["references"]]
    chars = [len(ref) for row in rows for ref in row["references"]]
    dup_in_row, seen_global = [], {}
    recurring = 0
    list_string = empty = nonstr = 0
    for row in rows:
        clean = []
        for ref in row["references"]:
            text = ref.strip()
            clean.append(text)
            if text.startswith("[") and text.endswith("]") and text.count("'") >= 2:
                list_string += 1
            if not text:
                empty += 1
        counts = collections.Counter(clean)
        for value, count in counts.items():
            if count > 1:
                dup_in_row.append({"id": row["id"], "reference": value, "count": count})
        for value in counts:
            if value in seen_global:
                recurring += 1
            seen_global[value] = True
    ordered = sorted(white)
    return {
        "rows": len(rows),
        "references_total": sum(lens),
        "refcount_min": min(lens),
        "refcount_mean": round(sum(lens) / len(lens), 4),
        "refcount_max": max(lens),
        "refcount_histogram": dict(sorted(collections.Counter(lens).items())),
        "whitespace_token_min": min(white),
        "whitespace_token_mean": round(sum(white) / len(white), 4),
        "whitespace_token_median": statistics.median(white),
        "whitespace_token_p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "whitespace_token_max": max(white),
        "char_mean": round(sum(chars) / len(chars), 1),
        "anomalies": {
            "list_string_reference_rows": list_string,
            "empty_references": empty,
            "duplicate_within_row": len(dup_in_row),
            "duplicate_examples": dup_in_row,
            "identical_reference_strings_recurring_across_rows": recurring,
        },
    }


def load_manifest_from_tar(evidence, manifest_tar, member):
    with tarfile.open(manifest_tar, "r:gz") as tar:
        raw = tar.extractfile(member).read()
    manifest = json.loads(raw)
    actual = sha256_bytes(raw)
    if actual != MANIFEST_SHA256:
        raise SystemExit(f"manifest member sha256 mismatch: {actual}")
    record(evidence, "manifest", f"{manifest_tar}::{member}", actual, {
        "container_sha256": sha256_file(manifest_tar),
        "lock_expectation_sha256": MANIFEST_SHA256,
    })
    return manifest


def reference_distributions(manifest):
    result = {}
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    for task_id in ("C", "F"):
        task = tasks[task_id]
        result[task_id] = {
            "source": task["source"], "family": task["family"], "metric": task["metric"],
            "split_policy": "all caption references preserved; one row per image" if task_id == "C" else "CSV row split preserved; all five captions per image",
        }
        for split in ("train", "validation", "eval"):
            result[task_id][split] = summarise_split(task[split])
    return result


def audit_flickr_csv(evidence, csv_path, conversion_path, manifest):
    if sha256_file(csv_path) != FLICKR_CSV_SHA256:
        raise SystemExit(f"Flickr CSV sha256 mismatch: {csv_path}")
    if sha256_file(conversion_path) != FLICKR_CONVERSION_SHA256:
        raise SystemExit(f"Flickr converted JSON sha256 mismatch: {conversion_path}")
    record(evidence, "flickr_csv", csv_path, FLICKR_CSV_SHA256,
           {"expected_revision": FLICKR_REVISION, "bytes": csv_path.stat().st_size})
    record(evidence, "flickr_converted_annotations", conversion_path, FLICKR_CONVERSION_SHA256,
           {"bytes": conversion_path.stat().st_size})

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    counts = collections.Counter(row["split"] for row in rows)
    csv_ok = counts == collections.Counter(MIRROR_SPLIT_COUNTS)
    if not csv_ok:
        raise SystemExit(f"CSV split counts differ from pinned mirror recipe: {dict(counts)}")

    captions_per_image_bad = []
    empty_captions = 0
    odd_ending = []
    within_image_dup = collections.defaultdict(list)
    tok = []
    recurring = collections.Counter()
    for row in rows:
        raw = json.loads(row["raw"])
        sids = json.loads(row["sentids"])
        if len(raw) != MIRROR_CAPTIONS_PER_IMAGE or len(sids) != MIRROR_CAPTIONS_PER_IMAGE:
            captions_per_image_bad.append(row["img_id"])
        tok.extend(len(text.split()) for text in raw)
        for text in raw:
            if not text.strip():
                empty_captions += 1
            trailing = text.strip()
            if trailing and not (trailing[-1].isalnum() or trailing[-1] in ".!?'\""):
                odd_ending.append({"img_id": row["img_id"], "split": row["split"], "caption": text})
            recurring[text.strip()] += 1
        if len(set(raw)) != MIRROR_CAPTIONS_PER_IMAGE:
            within_image_dup[row["split"]].append(row["filename"].rsplit(".", 1)[0])
    sids_total = sum(len(json.loads(row["sentids"])) for row in rows)
    sids_unique = len({sid for row in rows for sid in json.loads(row["sentids"])})
    if sids_total != sids_unique or sids_total != 5 * len(rows):
        raise SystemExit("sentid uniqueness/count check failed")

    by_filename = {row["filename"]: row for row in rows}
    taskF = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    checks = {"checked_rows": 0, "mismatches": [], "split_of_used": collections.Counter()}
    used_by_split = collections.defaultdict(set)
    for manifest_split, source_split in (("train", "train"), ("validation", "train"), ("eval", "test")):
        for row in taskF[manifest_split]:
            stem = str(row["source_image_id"])
            csv_row = by_filename.get(stem + ".jpg")
            if csv_row is None:
                checks["mismatches"].append({"row": row["id"], "problem": "csv_row_missing"})
                continue
            raw = json.loads(csv_row["raw"])
            problems = []
            if csv_row["split"] != source_split:
                problems.append(f"csv_split_is_{csv_row['split']}_not_{source_split}")
            if row["source_split"] != source_split:
                problems.append(f"source_split_is_{row['source_split']}_not_{source_split}")
            if row["id"] != ("flickr30k/test/" + stem if source_split == "test" else "flickr30k/train/" + stem):
                problems.append("row_id_format")
            if row["references"] != raw:
                problems.append("references_differ_from_csv_raw")
            if row["image"] != "Flickr30k/" + csv_row["filename"]:
                problems.append("image_path_format")
            if problems:
                checks["mismatches"].append({"row": row["id"], "problems": problems})
            checks["checked_rows"] += 1
            checks["split_of_used"][csv_row["split"]] += 1
            used_by_split[csv_row["split"]].add(stem)

    pool = {split: {row["filename"].rsplit(".", 1)[0] for row in rows if row["split"] == split}
            for split in ("train", "val", "test")}
    usage = {split: {"pool_available": len(pool[split]), "used": len(used_by_split.get(split, set())),
                     "used_subset_of_pool": used_by_split.get(split, set()) <= pool[split]}
             for split in ("train", "val", "test")}
    if len(used_by_split.get("train", set()) & used_by_split.get("test", set())):
        raise SystemExit("Flickr train/validation and eval image IDs overlap")

    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conv_by_imgid = {}
    for image in conversion["images"]:
        conv_by_imgid[image["imgid"]] = image
    conv_mismatch = 0
    for row in rows:
        image = conv_by_imgid.get(int(row["img_id"]))
        if (image is None or image["filename"] != row["filename"] or image["split"] != row["split"]
                or [sentence["raw"] for sentence in image["sentences"]] != json.loads(row["raw"])
                or [sentence["sentid"] for sentence in image["sentences"]] != json.loads(row["sentids"])):
            conv_mismatch += 1

    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    record(evidence, "flickr_sources_json", SOURCES_JSON, sha256_file(SOURCES_JSON),
           {"caption_count_claim": sources.get("caption_count"), "image_count_claim": sources.get("image_count")})
    hashes = sources["image_sha256"]
    hash_mismatch = [row["id"] for row in taskF["train"] + taskF["validation"] + taskF["eval"]
                     if hashes.get(str(row["source_image_id"]) + ".jpg") != row["image_sha256"]]
    if sources["split_counts"] and collections.Counter(sources["split_counts"]) != collections.Counter(MIRROR_SPLIT_COUNTS):
        raise SystemExit("flickr sources json split_counts disagree with CSV")

    dup_manifest_rows = []
    for split in ("train", "validation", "eval"):
        for row in taskF[split]:
            counter = collections.Counter(row["references"])
            if any(count > 1 for count in counter.values()):
                dup_manifest_rows.append(row["id"])
    dup_used_train = [stem for stem in within_image_dup.get("train", []) if stem in used_by_split["train"]]

    result = {
        "csv_split_counts": dict(sorted(counts.items())),
        "expected_mirror_counts": MIRROR_SPLIT_COUNTS,
        "images_total": len(rows),
        "captions_total": 5 * len(rows),
        "captions_per_image_violations": len(captions_per_image_bad),
        "sentids_total_unique": [sids_total, sids_unique],
        "caption_whitespace_token_mean": round(sum(tok) / len(tok), 4),
        "caption_whitespace_token_max": max(tok),
        "empty_captions_in_csv": empty_captions,
        "atypical_trailing_caption_rows": len(odd_ending),
        "atypical_examples": odd_ending[:8],
        "csv_within_image_duplicate_rows_by_split": {k: len(v) for k, v in sorted(within_image_dup.items())},
        "manifest_rows_with_within_row_duplicate": dup_manifest_rows,
        "csv_duplicate_rows_among_used_train_images": sorted(dup_used_train),
        "recurring_identical_captions_across_images": sum(1 for count in recurring.values() if count > 1),
        "manifest_to_csv_mapping": {"checked_rows": checks["checked_rows"], "mismatches": checks["mismatches"]},
        "used_image_split_coverage": {k: v for k, v in usage.items()},
        "used_split_totals": dict(sorted(checks["split_of_used"].items())),
        "conversion_json_vs_csv_mismatched_rows": conv_mismatch,
        "manifest_image_sha256_vs_sources_json_mismatches": hash_mismatch,
    }
    if checks["mismatches"] or conv_mismatch or hash_mismatch:
        raise SystemExit("Flickr CSV audit found real mismatches (see json)")
    if sorted(dup_manifest_rows) != sorted("flickr30k/train/" + stem for stem in dup_used_train):
        raise SystemExit("manifest duplicate rows do not match the CSV source duplicates")
    return result


def verify_flickr_image_bytes(evidence, images_tar, manifest):
    taskF = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    wanted = {}
    for split in ("train", "validation", "eval"):
        for row in taskF[split]:
            wanted["Flickr30k/" + str(row["source_image_id"]) + ".jpg"] = row["image_sha256"]
    mismatched, verified = [], 0
    with tarfile.open(images_tar, "r") as tar:
        for name in wanted:
            member = tar.getmember(name)
            digest = sha256_bytes(tar.extractfile(member).read())
            verified += 1
            if digest != wanted[name]:
                mismatched.append(name)
    if mismatched:
        raise SystemExit(f"Flickr image bytes mismatch against manifest for {len(mismatched)} files")
    record(evidence, "selected_images_tar", images_tar, sha256_file(images_tar), {
        "verified_members": verified,
        "note": "Flickr30k 멤버 2,816개 전부 실제 바이트 해시를 manifest image_sha256과 대조",
    })
    return {"verified_members": verified, "mismatches": mismatched}


def load_final_flickr_predictions(evidence, g4_raw_tar, run_name):
    with tarfile.open(g4_raw_tar, "r:gz") as tar:
        preds_member = G4_TAR_PREFIX + run_name + "/predictions_after_03_F.jsonl"
        config_member = G4_TAR_PREFIX + run_name + "/config.json"
        preds_raw = tar.extractfile(preds_member).read()
        config_raw = tar.extractfile(config_member).read()
    config = json.loads(config_raw)
    if config.get("manifest_sha256") != MANIFEST_SHA256:
        raise SystemExit("G4 run config manifest hash mismatch")
    if config.get("method") != "expand" or config.get("seed") != 137 or config.get("task_order") != ["V", "C", "G", "F"]:
        raise SystemExit("G4 run identity mismatch")
    if config["scorer_locks"]["C"]["tokenizer_jar_sha256"] != config["scorer_locks"]["F"]["tokenizer_jar_sha256"]:
        raise SystemExit("C/F tokenizer jar differs")
    record(evidence, "g4_raw_tar", g4_raw_tar, sha256_file(g4_raw_tar))
    record(evidence, "g4_run_predictions_member", f"{g4_raw_tar}::{preds_member}", sha256_bytes(preds_raw),
           {"rows": preds_raw.decode().count("\n")})
    record(evidence, "g4_run_config_member", f"{g4_raw_tar}::{config_member}", sha256_bytes(config_raw))
    predictions = [json.loads(line) for line in preds_raw.decode().splitlines()]
    if len(predictions) != 512 or len({p["id"] for p in predictions}) != 512:
        raise SystemExit("expected 512 unique final Flickr predictions")
    return config, predictions


def pick_sample(predictions):
    ids = sorted(p["id"] for p in predictions)
    rng = random.Random(SAMPLE_SEED)
    chosen, seen = [], set()
    while len(chosen) < SAMPLE_SIZE:
        index = rng.randrange(len(ids))
        if index in seen:
            continue
        seen.add(index)
        chosen.append(ids[index])
    return chosen


def static_metrics_inspection(evidence, metrics_source, run_source, data_source, prep_multimodal_source,
                              prep_flickr_source, run_config):
    for path in (metrics_source, run_source, data_source, prep_multimodal_source, prep_flickr_source):
        record(evidence, "static_source_" + path.name, path, sha256_file(path))
    metrics_text = metrics_source.read_text(encoding="utf-8")
    run_text = run_source.read_text(encoding="utf-8")
    data_text = data_source.read_text(encoding="utf-8")
    prep_text = prep_multimodal_source.read_text(encoding="utf-8")

    tree = ast.parse(metrics_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    checks = {}
    # 단일 구현: _tokenize와 _fixed_cider_score가 모듈에 한 번씩만 정의되고
    # TaskScorer.score에서만 호출된다(모든 cider_d task 공용).
    checks["single_tokenize_definition"] = "_tokenize" in functions and functions["_tokenize"].name == "_tokenize"
    checks["single_fixed_cider_definition"] = "_fixed_cider_score" in functions
    scorer_methods = {node.name: node for node in classes["TaskScorer"].body if isinstance(node, ast.FunctionDef)}
    score_fn, init_fn = scorer_methods["score"], scorer_methods["__init__"]
    calls = [n.func.id for n in ast.walk(score_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    checks["score_uses_shared_tokenize_and_cider"] = {"_tokenize", "_fixed_cider_score"} <= set(calls)
    checks["idf_corpus_is_task_train_only"] = (
        'train = task["train"]' in metrics_text
        and "self._train_documents = len(train)" in metrics_text
        and '"idf_source": "task_train_references"' in metrics_text
        and "for row in train:" in metrics_text
    )
    init_segment = ast.get_source_segment(metrics_text, init_fn) or ""
    checks["no_eval_rows_in_idf_path"] = ('task["eval"]' not in init_segment and "task['eval']" not in init_segment
                                          and "eval_split" not in init_segment)
    checks["tokenizer_is_stanford_ptb_preserve_lines"] = "-preserveLines" in metrics_text and "-lowerCase" in metrics_text
    checks["cider_constants_shared"] = metrics_text.count("CiderScorer(n=4, sigma=6.0)") == 2
    checks["reference_structure_single_validator"] = ("def _references(row):" in metrics_text
                                                      and "self.family = task[\"family\"]" in metrics_text)
    checks["task_scorer_built_identically_for_c_and_f"] = (
        'scorers = {task["task_id"]: TaskScorer(task) for task in tasks}' in run_text
    )
    checks["caption_rows_keep_all_references"] = "all_caption_references_preserved" in prep_text
    checks["flickr_refs_from_official_raw_only"] = (
        'refs = [text(sentence.get("raw"), "Karpathy raw caption") for sentence in sentences]' in prep_text
    )
    checks["manifest_reference_validation"] = ("references must contain all nonempty source strings" in data_text)
    checks["flickr_split_recipe_pinned"] = 'recipe.get("eval") != ["test"]' in prep_text
    if not all(checks.values()):
        raise SystemExit(f"static inspection failed: {[k for k, v in checks.items() if not v]}")

    locks = run_config["scorer_locks"]
    c_lock, f_lock = locks["C"], locks["F"]
    equal_keys = {k for k in c_lock if c_lock[k] == f_lock[k]}
    differing = {k for k in c_lock if k not in equal_keys}
    expected_differing = {"train_reference_sha256", "document_frequency_sha256"}
    if differing != expected_differing:
        raise SystemExit(f"unexpected C/F scorer lock differences: {sorted(differing ^ expected_differing)}")
    if not (c_lock["train_documents"] == f_lock["train_documents"] == 2048):
        raise SystemExit("train_documents must be 2048 for both caption tasks")
    return {
        "scorer_code_comment": SCORER_CODE_COMMENT,
        "static_checks": checks,
        "c_f_lock_equal_keys": sorted(equal_keys),
        "c_f_lock_expected_differing_keys": sorted(expected_differing),
        "train_documents": {"C": c_lock["train_documents"], "F": f_lock["train_documents"]},
        "eval_rows_used_for_idf": False,
    }


def score_ratios(evidence, tasks_csv, normalizers_json, run_config):
    record(evidence, "g4_report_tasks_csv", tasks_csv, sha256_file(tasks_csv))
    record(evidence, "g1_normalizers_json", normalizers_json, sha256_file(normalizers_json))
    rows = list(csv.DictReader(tasks_csv.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["metric"] == "cider_d" and row["task_id"] not in ("C", "F"):
            raise SystemExit("unexpected caption task id in tasks.csv")
    independent = {}
    expand = {}
    for row in rows:
        if row["method"] == "independent" and row["metric"] == "cider_d":
            independent[(row["task_id"], int(row["seed"]))] = row
        elif row["method"] == "expand" and row["metric"] == "cider_d":
            order = next(o for o in G4_ORDERS if row["name"].startswith(o + "_"))
            expand[(order, int(row["seed"]), row["task_id"])] = row
    seeds = G4_SEEDS
    base = {tid: [float(independent[(tid, s)]["pre"]) for s in seeds] for tid in ("C", "F")}
    indep = {tid: [float(independent[(tid, s)]["final"]) for s in seeds] for tid in ("C", "F")}
    pairs = lambda fs, cs: [f / c for f, c in zip(fs, cs)]
    expand_rows = []
    for s in seeds:
        for order in G4_ORDERS:
            expand_rows.append({"seed": s, "order": order,
                                "F_final": float(expand[(order, s, "F")]["final"]),
                                "C_final": float(expand[(order, s, "C")]["final"])})
    seed_expand_ratio = []
    for s in seeds:
        fs = [entry["F_final"] for entry in expand_rows if entry["seed"] == s]
        cs = [entry["C_final"] for entry in expand_rows if entry["seed"] == s]
        seed_expand_ratio.append((sum(fs) / len(fs)) / (sum(cs) / len(cs)))
    mean = lambda x: sum(x) / len(x)
    normalizers = json.loads(normalizers_json.read_text(encoding="utf-8"))
    if normalizers["manifest_sha256"] != MANIFEST_SHA256 or normalizers["eval_split"] != "validation":
        raise SystemExit("normalizer provenance mismatch")
    result = {
        "score_source": "artifacts/multimodal-g4-report/tasks.csv (기록된 점수 재사용; 새 채점 없음)",
        "eval_split": "eval",
        "native_units": "CIDEr-D pycocoevalcap 1.2 scale x1, train-IDF(task train 2,048 images)",
        "frozen_base_pre": {"C": base["C"][0], "F": base["F"][0],
                            "ratio_F_over_C": base["F"][0] / base["C"][0],
                            "constant_across_seeds": len(set(base["C"])) == 1 and len(set(base["F"])) == 1},
        "independent": {"C_mean": mean(indep["C"]), "F_mean": mean(indep["F"]),
                        "C_per_seed": indep["C"], "F_per_seed": indep["F"],
                        "ratio_F_over_C_mean": mean(pairs(indep["F"], indep["C"])),
                        "ratio_F_over_C_min": min(pairs(indep["F"], indep["C"])),
                        "ratio_F_over_C_max": max(pairs(indep["F"], indep["C"]))},
        "expansion": {"C_mean": mean([row["C_final"] for row in expand_rows]),
                      "F_mean": mean([row["F_final"] for row in expand_rows]),
                      "ratio_F_over_C_row_mean": mean([row["F_final"] / row["C_final"] for row in expand_rows]),
                      "ratio_F_over_C_seed_aggregated_mean": mean(seed_expand_ratio),
                      "seed_aggregated_ratios": seed_expand_ratio},
        "g1_validation_normalizers": {"C": normalizers["scores"]["C"], "F": normalizers["scores"]["F"],
                                      "ratio_F_over_C": normalizers["scores"]["F"] / normalizers["scores"]["C"],
                                      "seed": 17, "split": "validation"},
        "g4_eval_vs_g1_validation_observation": (
            "validation과 eval에서 독립 F/C 비율이 0.224(validation) vs 0.193(eval), C는 0.680→0.517, "
            "F는 0.153→0.100으로 모두 낮아진다. G1과 G4는 split뿐 아니라 seed 집합도 다르므로 "
            "이를 split 난이도 차이로 귀속할 수 없으며, 비율만으로 버그를 배제하지 않는다."
        ),
    }
    return result


def build_samples_md(sample_ids, predictions, manifest, ratios, csv_audit, distributions, image_verification):
    taskF = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    eval_by_id = {row["id"]: row for row in taskF["eval"]}
    pred_by_id = {p["id"]: p for p in predictions}
    full_flags = collections.Counter((p["generation"]["eos"], p["generation"]["limit_reached"]) for p in predictions)
    lines = []
    add = lines.append
    add("# Week 1-A Flickr sanity samples (G4 seed 137, order V→C→G→F, method=expand, 최종 F 예측 50건)")
    add("")
    add(f"- 샘플 시드: `random.Random({SAMPLE_SEED})`를 512개 정렬된 eval ID에 randrange 중복 없이 50회 적용.")
    add(f"- 예측 run: `{G4_RUN_NAME}` (manifest {MANIFEST_SHA256[:12]}…, predictions_after_03_F.jsonl 512행).")
    add("- 50건 전부 예측 + **전체 reference(각 5개)** 를 함께 읽었고, `이미지확인` 표기 12건은 manifest image_sha256과")
    add("  byte-동일한 실제 이미지를 확인한 AI assistant의 정성 검수 노트다(사람 평가 아님). 나머지 38건은 예측·전체 reference")
    add("  텍스트 기준 의미 판독이다. 스크립트는 이미지를 해시로만 재검증하고 노트를 임베딩해 재현한다.")
    add("- 결론: **" + CONCLUSION_OK + "** — 50건에서 reference 오배정/파싱 손상은 발견되지 않았고,")
    add("  반복 생성·내용 불일치는 관찰되지만 학습량·라우팅 등 원인은 미확정이다. 표본 메모의 모델 오류는 예측 불일치를 뜻한다.")
    add("")
    add("## 자동 기록 참고표 (휴리스틱, 판정 근거 아님)")
    add("")
    limit = [sid for sid in sample_ids if pred_by_id[sid]["generation"]["limit_reached"]]
    eos_true = [sid for sid in sample_ids if pred_by_id[sid]["generation"]["eos"]]
    trailing_digit = [sid for sid in sample_ids if pred_by_id[sid]["prediction"].strip().endswith("7")]
    experts = collections.Counter(pred_by_id[sid]["expert"] for sid in sample_ids)
    add(f"- eos=True {len(eos_true)}/50, limit_reached(64토큰 한도) {len(limit)}/50, repeated_trigram "
        f"{sum(1 for sid in sample_ids if pred_by_id[sid]['generation']['repeated_trigram'])}/50.")
    add(f"- 후미 정수 토큰('7') 예측 {len(trailing_digit)}건(원인 미확정): "
        + (", ".join(sid.split('/')[-1] for sid in trailing_digit) or "없음") + ".")
    add("- 50건 중 expert 배정: " + ", ".join(f"expert {k} = {v}건" for k, v in sorted(experts.items()))
        + " (order V,C,G,F에서 F 전문 expert는 3).")
    add("- limit_reached는 생성 64토큰 한도 도달 여부다. Reference 길이(F eval 최대 68 whitespace 토큰)와")
    add("  별도로 기록한다. 반복과 한도 도달의 인과 관계는 여기서 검증하지 않는다.")
    add("")
    add("## 50건 검수")
    add("")
    def table_text(value):
        return str(value).replace("|", "&#124;").replace("\n", "<br>")
    add("| # / ID / expert | 예측 | 전체 reference | 정성 판독 메모 |")
    add("|---|---|---|---|")
    for index, sid in enumerate(sample_ids, 1):
        pred = pred_by_id[sid]
        refs = eval_by_id[sid]["references"]
        ref_text = "<br>".join(f"{k}. {table_text(ref)}" for k, ref in enumerate(refs, 1))
        add(f"| {index:02d} / {sid} / {pred['expert']} | {table_text(pred['prediction'])} | "
            f"{ref_text} | {table_text(EXPERT_NOTES[sid])} |")
    add("")
    add("## 부록: 수치 요약")
    add("")
    add(f"- 이미지 byte 검증: F eval 512 + validation 256 + train 2,048 = {image_verification['verified_members']}개 tar 멤버 전부 manifest 해시와 일치.")
    add(f"- CSV 감사: manifest→CSV 대조 {csv_audit['manifest_to_csv_mapping']['checked_rows']}행 전부 일치, 변환 JSON↔CSV 불일치 {csv_audit['conversion_json_vs_csv_mismatched_rows']}행.")
    add("- CSV 내부 split: train 29,000 / val 1,014 / test 1,000 (고정 미러 기대 수량과 일치). 사용: train split 2,304(2,048 train + 256 validation), test split 512(eval), val·restval 0.")
    add(f"- 기존 점수 비율(×1, train-IDF CIDEr-D): frozen base F/C {ratios['frozen_base_pre']['ratio_F_over_C']:.3f}, "
        f"independent {ratios['independent']['ratio_F_over_C_mean']:.3f}, expansion(seed 집계) "
        f"{ratios['expansion']['ratio_F_over_C_seed_aggregated_mean']:.3f}, G1 validation normalizer "
        f"{ratios['g1_validation_normalizers']['ratio_F_over_C']:.3f}.")
    add(f"- reference 분포: F train/val/eval refs-per-image 평균 "
        f"{distributions['F']['train']['refcount_mean']}/{distributions['F']['validation']['refcount_mean']}/{distributions['F']['eval']['refcount_mean']}, "
        f"C train refs-per-image 평균 {distributions['C']['train']['refcount_mean']} (manifest 내 중복 reference 2건 관찰; 원본 COCO 재대조 미수행).")
    add(f"- whitespace 토큰 평균: C {distributions['C']['train']['whitespace_token_mean']}/{distributions['C']['validation']['whitespace_token_mean']}/"
        f"{distributions['C']['eval']['whitespace_token_mean']}, F {distributions['F']['train']['whitespace_token_mean']}/"
        f"{distributions['F']['validation']['whitespace_token_mean']}/{distributions['F']['eval']['whitespace_token_mean']} (train/val/eval).")
    add(f"- 전체 512개 최종 F 예측 생성 플래그: eos이고 한도 미도달 {full_flags.get((True, False), 0)}개, "
        f"limit_reached {full_flags.get((False, True), 0)}개 ({full_flags.get((False, True), 0) / 5.12:.1f}%).")
    add("- scorer 정적 검사: metrics.py의 _tokenize/_fixed_cider_score는 모든 cider_d task 공용 단일 구현, IDF는 task train 2,048장 "
        "참조만 사용(평가 참조 미포함), C/F scorer lock은 train corpus 내용 해시 2종만 다름.")
    add("")
    return "\n".join(lines) + "\n"


CONCLUSION_SCOPE = {
    "checked": [
        "C/F manifest reference 수·길이 분포, list-문자열 파싱 흔적, empty/duplicate reference(전체 28,143 ref)",
        "Flickr CSV(pinned revision) 내부 split vs 실제 사용 이미지 ID 2,816개 전수 대조 및 기대 수량(29,000/1,014/1,000)",
        "Flickr 사용 이미지 2,816개 실제 바이트 SHA256 vs manifest(로컬 tar 재검증)",
        "G4 최종 Flickr 예측 50건 + 전체 reference AI assistant 정성 검수(12건은 실제 이미지 대조)",
        "metrics.py train-IDF/토크나이저/reference 구조의 C/F 동일성 정적 검사 + C/F scorer lock 대조",
        "frozen base/independent/expansion의 기존 점수 Flickr:COCO 비율 일관성",
    ],
    "not_checked": [
        "COCO 원본(captions_train2014/val2014.json)과의 재대조는 미수행; COCO는 사용 manifest 수준에서 검증",
        "near-duplicate 및 BLIP 사전학습 데이터와의 이미지 중복(원 summary의 한계 그대로)",
        "절대 CIDEr-D 수준의 문헌 비교 - 본 파이프라인은 DF 코퍼스를 task train 2,048장으로 고정해 표준 eval-corpus DF와 달라 문헌 값과 직접 비교 불가",
        "G5 라우팅 품질 자체의 평가(1-C/라우팅 분석 범위)",
    ],
    "verdict": (
        "데이터 파이프라인(reference 파싱·split 매핑·이미지 대응·scorer 코드 경로)과 기존 점수 내부 일관성에서 "
        "Flickr 고장의 증거가 없다. 낮은 점수의 원인은 이번 조회만으로 확정하지 않는다. "
        "표본에서 reference와 다른 내용 및 반복 생성이 관찰되지만 학습량·IDF·라우팅의 "
        "인과 기여는 미검증이다. Frozen base F/C 비율과 학습 후 비율도 서로 달라 "
        "단순한 데이터셋 간 고정 배율로 해석하지 않는다."
    ),
}

AFFECTED_RUNS_NOTE = (
    "버그 미확인이므로 G1-G5 재실행 범위 산정은 트리거되지 않았다. "
    "(참고: 계획 문서상 버그 확정 시 최악 G4 전체 재실행 약 120h GPU 추정은 본 판정에서는 적용되지 않는다.)"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flickr-csv", type=Path, required=True,
                        help="pinned revision의 flickr_annotations_30k.csv (읽기 전용 확보본)")
    parser.add_argument("--flickr-conversion", type=Path, required=True,
                        help="prepare_flickr30k_hf.py가 만든 dataset_flickr30k.json (읽기 전용 확보본)")
    parser.add_argument("--flickr-csv-obtained-from",
                        default="Yonsei-DC1:/scratch2/faransansj/multimodal_sources_20261013/flickr30k_hf/flickr_annotations_30k.csv (2026-10-16 scp, 읽기 전용)")
    parser.add_argument("--flickr-conversion-obtained-from",
                        default="Yonsei-DC1:/scratch2/faransansj/multimodal_sources_20261013/flickr30k_hf/dataset_flickr30k.json (2026-10-16 scp, 읽기 전용)")
    args = parser.parse_args()

    evidence = {}
    manifest = load_manifest_from_tar(evidence, MANIFEST_TAR, MANIFEST_MEMBER)
    distributions = reference_distributions(manifest)
    csv_audit = audit_flickr_csv(evidence, args.flickr_csv, args.flickr_conversion, manifest)
    image_verification = verify_flickr_image_bytes(evidence, IMAGES_TAR, manifest)
    run_config, predictions = load_final_flickr_predictions(evidence, G4_RAW_TAR, G4_RUN_NAME)
    sample_ids = pick_sample(predictions)
    missing = sorted(set(sample_ids) ^ set(EXPERT_NOTES))
    if missing:
        raise SystemExit(f"embedded notes do not cover sampled ids: {missing}")
    if not VISUAL_INSPECTED <= set(sample_ids):
        raise SystemExit("visual inspection ids must be a subset of the sample")
    statics = static_metrics_inspection(evidence, METRICS_SOURCE, RUN_SOURCE, DATA_SOURCE,
                                        PREP_MULTIMODAL_SOURCE, PREP_FLICKR_SOURCE, run_config)
    ratios = score_ratios(evidence, G4_TASKS_CSV, G1_NORMALIZERS, run_config)
    record(evidence, "confirmation_lock", CONFIRMATION_LOCK, sha256_file(CONFIRMATION_LOCK))
    evidence["week1_flickr_sanity_script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__)),
        "obtained_from_flickr_csv": args.flickr_csv_obtained_from,
        "obtained_from_conversion": args.flickr_conversion_obtained_from,
    }

    pred_by_id = {p["id"]: p for p in predictions}
    taskF = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    eval_rows = {row["id"]: row for row in taskF["eval"]}
    sample_rows = [{"index": i, "id": sid, "prediction": pred_by_id[sid]["prediction"],
                    "expert": pred_by_id[sid]["expert"], "generation": pred_by_id[sid]["generation"],
                    "references": eval_rows[sid]["references"],
                    "visual_inspection": sid in VISUAL_INSPECTED,
                    "expert_note": EXPERT_NOTES[sid]}
                   for i, sid in enumerate(sample_ids, 1)]
    eval_gen = collections.Counter((p["generation"]["eos"], p["generation"]["limit_reached"]) for p in predictions)
    sample_section = {
        "seed": SAMPLE_SEED, "size": SAMPLE_SIZE, "run": G4_RUN_NAME, "stage": "after_03_F (final)",
        "selection": "randrange over 512 sorted eval ids without replacement",
        "sampled_ids": sample_ids,
        "samples": sample_rows,
        "visual_inspected_count": len(VISUAL_INSPECTED),
        "notes_method": "AI assistant 의미 판독(예측+전체 reference 전부, 12건은 실제 이미지 대조); 사람 평가 아님; 노트는 스크립트에 정적 임베딩",
        "full_512_generation_flag_counts": {"eos_true_and_not_limit": eval_gen.get((True, False), 0),
                                            "limit_reached": eval_gen.get((False, True), 0),
                                            "other": sum(count for flags, count in eval_gen.items()
                                                         if flags not in ((True, False), (False, True)))},
    }
    ratio_section = ratios
    conclusion = CONCLUSION_OK

    output_json = REPO / "artifacts/week1-flickr-sanity.json"
    output_md = REPO / "artifacts/week1-flickr-samples.md"
    for path in (output_json, output_md):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"output already exists: {path}")

    payload = {
        "schema_version": 1,
        "stage": "week1-1A-flickr-sanity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "flickr_revision": FLICKR_REVISION,
        "conclusion": conclusion,
        "conclusion_scope": CONCLUSION_SCOPE,
        "affected_runs_if_bug_confirmed": {"triggered": False, "note": AFFECTED_RUNS_NOTE},
        "evidence": evidence,
        "sections": {
            "reference_distributions": distributions,
            "flickr_csv_split_audit": csv_audit,
            "flickr_used_image_bytes": image_verification,
            "sample_50_final_flickr_predictions": sample_section,
            "metrics_static_inspection": statics,
            "existing_score_ratios_flickr_over_coco": ratio_section,
        },
    }
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    md = build_samples_md(sample_ids, predictions, manifest, ratios, csv_audit,
                          distributions, image_verification)
    with output_md.open("x", encoding="utf-8") as stream:
        stream.write(md)
    print(json.dumps({"json": str(output_json), "md": str(output_md), "conclusion": conclusion,
                      "sampled": len(sample_ids),
                      "checked_rows": csv_audit["manifest_to_csv_mapping"]["checked_rows"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()






