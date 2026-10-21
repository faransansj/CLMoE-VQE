#!/usr/bin/env python3
"""Stage 0.5 task F: post-hoc truncation rescoring of saved Flickr predictions.

No training, no inference, no scorer/method change. Saved predictions are
truncated by predeclared text rules and rescored with the existing fixed
train-IDF CIDEr-D, per-image and corpus, exactly the implementation used by
scripts/week1_per_example_rescore.py (sema_multimodal.metrics.TaskScorer,
M._tokenize, M._fixed_cider_score, frozen IDF from task train references).

DIAGNOSTIC LABEL (binding): the truncation rules are applied to eval-split
saved outputs post hoc. Diagnostic only — NOT method selection, NOT a
decoding change, NOT a config proposal. Saved artifacts are read-only
inputs; this script never modifies them.

Cohorts (one row = one saved prediction on one eval image):
  C1 g4_independent_F: kind=independent, task=F, stage=0 (8 seeds x 512 rows).
     No paired old C: independent C outputs are saved on the COCO eval split
     (disjoint image set), so paired deltas/wins/both-EOS are reported only
     for cohort 2; cohort 1 reports corpus scores, truncation counts, lengths.
  C2 flickr_last_cells: 16 cells = orders {V_C_G_F, V_G_C_F} x 8 seeds;
     F = saved post-stage task-ID-oracle F-expert outputs
         (kind=oracle, task=F, stage=3, phase=after),
     old C = saved pre-stage native outputs (kind=native, task=F, stage=3,
         phase=before; expert=C), paired per (order, seed, id).

Truncation rules, on whitespace tokens of the saved prediction string,
applied ONLY to non-EOS F outputs (generation.eos true -> unchanged):
  none   : untruncated baseline (parity anchor; all rows unchanged).
  R1     : cut at start of first repeated token trigram, defined exactly like
           sema_multimodal/model.py generate() repeated_trigram:
           earliest pos with tokens[pos:pos+3] == tokens[pos+3:pos+6];
           keep text before the repeat begins; unchanged if none.
  R2     : earliest marker occurrence among ", and" and " and a " whose
           preceding text has >= 8 whitespace words; keep the text before
           the marker start, rstripped of trailing whitespace; unchanged
           if no qualifying marker.
  R3_12  : keep first 12 words; unchanged if <= 12 words.
  R3_16  : keep first 16 words; unchanged if <= 16 words.
  R3_20  : keep first 20 words; unchanged if <= 20 words.
           (Flickr train references: median 11 words, p95 22 words.)
  R4     : R1 first, then R3 with N=20 on the R1 output.

Rules only delete text; they never insert characters (in particular no '.'
can be introduced). Every rule output is rescored per-image (singleton
CiderScorer + M._fixed_cider_score, exactly week1_per_example_rescore) and
per-cell corpus (TaskScorer.score over the original 512-row corpus, exact
existing scorer). Untruncated rescoring is verified against the stored
values (per-image row scores and stored corpus scalars) before any rule
result is reported.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sema_multimodal import metrics as M  # metrics module only; never .model / .run

SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
EVAL_ROWS = 512
FLICKR_LAST_ORDERS = ("V_C_G_F", "V_G_C_F")
VARIANTS = ("none", "R1", "R2", "R3_12", "R3_16", "R3_20", "R4")
RULES = ("R1", "R2", "R3_12", "R3_16", "R3_20", "R4")

ROWS_GZ = ROOT / "artifacts/week1-per-example-scores.jsonl.gz"
MANIFESTS_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
G4_TAR = ROOT / "artifacts/multimodal-g4-raw.tar.gz"
G4_ROOT = "artifacts/multimodal_runs/g4"
AUDIT_G4 = ROOT / "artifacts/multimodal-g4-report/audit.json"
AUDIT_G5 = ROOT / "artifacts/multimodal-g5-report/routing_audit.json"
CONFIRMATION_LOCK = ROOT / "artifacts/multimodal-confirmation-lock.json"
TARGET_FORMAT = ROOT / "artifacts/stage05-target-format.json"

OUT_JSON = ROOT / "artifacts/stage05-truncation-rescore.json"

PER_IMAGE_TOL = 1e-12     # fresh per-image vs stored row score
CORPUS_TOL = 1e-9         # fresh corpus / stored per-image mean vs stored scalar
CACHED_TOL = 1e-9         # per-image mean vs fresh TaskScorer corpus
R2_MIN_WORDS = 8
R4_N = 20
CHUNK = 50000

TIMING: dict[str, object] = {}
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)
        print("CHECK FAILED:", message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return float("nan")
    return math.fsum(values) / len(values)


def one(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected finite scalar, got {value!r}")
    return float(value)


def bank_name(order_name: str, seed: int) -> str:
    return f"{order_name}_expand_lr0.0003_ep5_seed{seed}"


def independent_name(task_id: str, seed: int) -> str:
    return f"{task_id}_independent_lr0.0003_ep5_seed{seed}"


def percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list of counts."""
    if not sorted_values:
        return 0
    index = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


# ---------------------------------------------------------------- truncation rules
def rule_R1(prediction: str) -> str:
    """Earliest pos with tokens[pos:pos+3] == tokens[pos+3:pos+6]; keep tokens[:pos]."""
    tokens = prediction.split()
    n = len(tokens)
    for pos in range(max(0, n - 5)):
        if tokens[pos:pos + 3] == tokens[pos + 3:pos + 6]:
            return " ".join(tokens[:pos])
    return prediction


def rule_R2(prediction: str) -> str:
    """Earliest ', and' / ' and a ' marker with >= 8 words before it; cut before marker."""
    markers: list[int] = []
    start = 0
    while True:
        found = prediction.find(", and", start)
        if found < 0:
            break
        markers.append(found)
        start = found + 1
    start = 0
    while True:
        found = prediction.find(" and a ", start)
        if found < 0:
            break
        markers.append(found)
        start = found + 1
    for position in sorted(markers):
        if len(prediction[:position].split()) >= R2_MIN_WORDS:
            return prediction[:position].rstrip()
    return prediction


def rule_R3(prediction: str, n_words: int) -> str:
    tokens = prediction.split()
    if len(tokens) <= n_words:
        return prediction
    return " ".join(tokens[:n_words])


def apply_rule(rule: str, prediction: str) -> str:
    if rule == "none":
        return prediction
    if rule == "R1":
        return rule_R1(prediction)
    if rule == "R2":
        return rule_R2(prediction)
    if rule.startswith("R3_"):
        return rule_R3(prediction, int(rule[3:]))
    if rule == "R4":
        return rule_R3(rule_R1(prediction), R4_N)
    raise ValueError(f"unknown rule {rule}")


def normalize(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------------- input loading
def load_cohort_rows() -> dict:
    """Stream the week1 per-example rows into minimal cohort tables.

    Cell keys: (None, seed) for independent rows, (order_name, seed) otherwise.
    """
    cohort: dict[str, dict] = {"ind_F": {}, "pre_C": {}, "orac_F": {}}
    counts = Counter()
    started = time.perf_counter()
    with gzip.open(ROWS_GZ, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            generation = row["generation"]
            check(isinstance(generation.get("eos"), bool),
                  f"generation.eos not bool for {row.get('id')}")
            record = {"id": row["id"], "prediction": row["prediction"],
                      "score_stored": one(row["score"]), "eos": generation["eos"],
                      "expert_name": row["expert_name"],
                      "repeated_trigram": bool(generation["repeated_trigram"]),
                      "limit_reached": bool(generation["limit_reached"]),
                      "generated_tokens": int(generation["generated_tokens"])}
            kind, task, stage, phase = row["kind"], row["task"], row["stage"], row["phase"]
            order_name = None if row["order"] is None else "_".join(row["order"])
            if kind == "independent":
                if task != "F":
                    counts[(kind, phase, task, stage)] += 1  # independent V/C/G: out of scope
                    continue
                check(stage == 0 and phase == "after",
                      f"unexpected independent F row stage={stage} phase={phase}")
                cohort["ind_F"].setdefault((None, row["seed"]), {})[record["id"]] = record
            elif kind == "oracle" and task == "F" and stage == 3 and phase == "after" \
                    and row["order"] is not None and row["order"][3] == "F":
                key = (order_name, row["seed"])
                cohort["orac_F"].setdefault(key, {})[record["id"]] = record
            elif kind == "native" and task == "F" and stage == 3 and phase == "before" \
                    and row["order"] is not None and row["order"][3] == "F":
                key = (order_name, row["seed"])
                cohort["pre_C"].setdefault(key, {})[record["id"]] = record
            else:
                continue
            counts[(kind, phase, task, stage)] += 1
    TIMING["load_rows_seconds"] = round(time.perf_counter() - started, 3)
    expected = {"independent_after_F_0": 4096, "independent_after_C_0": 4096,
                "independent_after_V_0": 4096, "independent_after_G_0": 4096,
                "oracle_after_F_3": 8192, "native_before_F_3": 8192}
    got = {"independent_after_F_0": counts[("independent", "after", "F", 0)],
           "independent_after_C_0": counts[("independent", "after", "C", 0)],
           "independent_after_V_0": counts[("independent", "after", "V", 0)],
           "independent_after_G_0": counts[("independent", "after", "G", 0)],
           "oracle_after_F_3": counts[("oracle", "after", "F", 3)],
           "native_before_F_3": counts[("native", "before", "F", 3)]}
    check(got == expected, f"cohort row counts {got} != {expected}")
    check(len(cohort["ind_F"]) == len(SEEDS),
          "independent F cells must cover exactly the 8 seeds")
    check(len(cohort["pre_C"]) == 16 and len(cohort["orac_F"]) == 16,
          "flickr-last cells must cover exactly 16 (order, seed) pairs")
    for key in cohort["ind_F"]:
        check(len(cohort["ind_F"][key]) == EVAL_ROWS,
              f"independent F cell {key} must hold exactly {EVAL_ROWS} rows")
    for key in cohort["pre_C"]:
        check(key in cohort["orac_F"], f"oracle cell missing for {key}")
        check(set(cohort["pre_C"][key]) == set(cohort["orac_F"][key]),
              f"pre/oracle id sets disagree for {key}")
        check(len(cohort["pre_C"][key]) == EVAL_ROWS,
              f"pre C cell {key} must hold exactly {EVAL_ROWS} rows")
    cohort["counts"] = dict(counts)
    return cohort


def load_task_views() -> tuple[dict, dict]:
    """Task views (train/eval rows) from the manifest archive, as in week1."""
    with tarfile.open(MANIFESTS_TAR, "r:gz") as tar:
        raw = tar.extractfile(MANIFEST_MEMBER).read()
    manifest = json.loads(raw.decode("utf-8"))
    views = {}
    for record in manifest["tasks"]:
        views[record["task_id"]] = {"task_id": record["task_id"], "family": record["family"],
                                    "metric": record["metric"], "train": record["train"],
                                    "eval": record["eval"]}
    check(set(views) >= {"C", "F"}, f"manifest tasks missing C/F: {sorted(views)}")
    eval_ids = {task: [row["id"] for row in view["eval"]] for task, view in views.items()}
    check(all(len(ids) == EVAL_ROWS for ids in eval_ids.values()),
          f"manifest eval sizes {[(t, len(v)) for t, v in eval_ids.items()]}")
    row_maps = {task: {row["id"]: row for row in view["eval"]} for task, view in views.items()}
    return views, {"manifest_sha256": sha256_bytes(raw), "eval_ids": eval_ids,
                   "row_maps": row_maps}


def stored_scalars() -> dict[tuple, float]:
    """Stored corpus scalars for the four baseline corpus families."""
    scalars: dict[tuple, float] = {}
    audit_g4 = json.loads(AUDIT_G4.read_text())
    per_seed: dict[tuple[int, str], set[float]] = {}
    for record in audit_g4["transfer_records"]:
        if record["kind"] != "common_cl" or record["method"] != "expand":
            continue
        for row in record["rows"]:
            if row["task_id"] != "F":
                continue  # C/V/G independent rows are out of scope for this stage
            per_seed.setdefault((record["seed"], row["task_id"]), set()).add(
                one(row["independent_score"]))
    check(set(per_seed) == {(seed, "F") for seed in SEEDS},
          "audit independent rows must cover (8 seeds) x F")
    for (seed, task), values in per_seed.items():
        check(len(values) == 1, f"independent scores not order-invariant for {task}/{seed}")
        scalars[("independent", independent_name(task, seed))] = values.pop()

    with tarfile.open(G4_TAR, "r:gz") as tar:
        for order_name in FLICKR_LAST_ORDERS:
            for seed in SEEDS:
                member = f"{G4_ROOT}/{bank_name(order_name, seed)}/results.json"
                results = json.loads(tar.extractfile(member).read().decode("utf-8"))
                scalars[("pre_native", bank_name(order_name, seed))] = \
                    one(results["pre_task_scores"][3])
    audit_g5 = json.loads(AUDIT_G5.read_text())
    seen_oracle = 0
    for run in audit_g5["runs"]:
        bank = run["directory"].rstrip("/").split("/")[-1]
        if bank.startswith(FLICKR_LAST_ORDERS):
            scalars[("oracle", bank)] = one(run["variants"]["task_id_oracle"]["matrix"][3][3])
            seen_oracle += 1
    check(seen_oracle == 16, f"expected 16 flickr-last oracle scalars, got {seen_oracle}")
    return scalars


# ---------------------------------------------------------------- tokenization + scoring
TOKEN_CACHE: dict[str, dict[str, list[str]]] = {}
SCORERS: dict[str, object] = {}
PER_IMAGE_BASELINE: dict[str, tuple[list[str], list[float], list[float]]] = {}
STORED: dict[tuple, float] = {}


def build_token_cache(tasks: tuple[str, ...], strings_by_task: dict[str, set[str]]) -> dict:
    report = {}
    for task in tasks:
        values = sorted(strings_by_task[task])
        cache: dict[str, list[str]] = {}
        started = time.perf_counter()
        for start in range(0, len(values), CHUNK):
            chunk = values[start:start + CHUNK]
            mapping = {f"s{i}": [value] for i, value in enumerate(chunk)}
            tokenized = M._tokenize(mapping)
            if set(tokenized) != set(mapping):
                raise ValueError(f"PTB chunk lost captions for task {task}")
            for i, value in enumerate(chunk):
                cache[value] = tokenized[f"s{i}"][0]
        TOKEN_CACHE[task] = cache
        report[task] = {"unique_strings": len(values),
                        "chunks": math.ceil(len(values) / CHUNK),
                        "seconds": round(time.perf_counter() - started, 3)}
    return report


def tokenize_report(task: str, predictions: list[str], row_map: dict, ids: list[str]) -> dict:
    """Rebuild exactly TaskScorer.score's captions dict and compare to the cache."""
    captions = {("reference", index): row_map[ids[index]]["references"]
                for index in range(len(ids))}
    captions.update({("prediction", index): [predictions[index]]
                     for index in range(len(ids))})
    tokenized = M._tokenize(captions)
    cache = TOKEN_CACHE[task]
    mismatches = 0
    for key, tokens in tokenized.items():
        kind, index = key
        sources = (row_map[ids[index]]["references"] if kind == "reference"
                   else [predictions[index]])
        for source, tokens_line in zip(sources, tokens):
            if cache.get(normalize(source)) != tokens_line:
                mismatches += 1
    return {"captions_compared": len(tokenized), "mismatch_total": mismatches}


def per_image_scores(task: str, predictions: list[str], row_map: dict,
                     ids: list[str]) -> list[float]:
    """week1 per-image method: singleton CiderScorer + fixed score, cached PTB tokens."""
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    scorer = SCORERS[task]
    df, train_documents = scorer._df, scorer._train_documents
    cache = TOKEN_CACHE[task]
    values = []
    for prediction, row_id in zip(predictions, ids):
        row = row_map[row_id]
        singleton = CiderScorer(n=4, sigma=6.0)
        singleton += (cache[normalize(prediction)],
                      [cache[normalize(ref)] for ref in row["references"]])
        values.append(M._fixed_cider_score(singleton, df, train_documents))
    return values


def corpus_score(task: str, predictions: list[str], row_map: dict, ids: list[str]) -> float:
    rows = [row_map[row_id] for row_id in ids]
    return one(SCORERS[task].score(predictions, rows)["score"])


def collect_strings(cohort: dict, views: dict) -> dict[str, set[str]]:
    """All strings that will pass through PTB tokenization (cache keys).

    Eval references of task F plus every prediction string that will be
    scored: saved baselines and every rule output of non-EOS F rows. All
    cohort predictions (independent F, pre-stage native, task-ID oracle) are
    scored as task-F rows. Train references are tokenized inside
    TaskScorer.__init__ (frozen IDF) and are not part of this cache.
    """
    strings: dict[str, set[str]] = {"F": set()}
    for row in views["F"]["eval"]:
        for ref in row["references"]:
            strings["F"].add(normalize(ref))
    for bucket in ("ind_F", "pre_C", "orac_F"):
        for cells in cohort[bucket].values():
            for record in cells.values():
                strings["F"].add(normalize(record["prediction"]))
    variant_strings: set[str] = set()
    for bucket in ("ind_F", "orac_F"):
        for cells in cohort[bucket].values():
            for record in cells.values():
                if record["eos"]:
                    continue
                for rule in RULES:
                    variant_strings.add(normalize(apply_rule(rule, record["prediction"])))
    strings["F"].update(variant_strings)
    return strings


# ---------------------------------------------------------------- cohort evaluation
def length_stats(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    return {"mean": mean(lengths), "median": statistics.median(ordered),
            "p95_nearest_rank": percentile(ordered, 0.95),
            "min": ordered[0] if ordered else 0, "max": ordered[-1] if ordered else 0}


def paired_stats(f_scores: list[float], c_scores: list[float],
                 mask: list[bool] | None) -> dict:
    if mask is None:
        pairs = list(zip(f_scores, c_scores))
    else:
        pairs = [(f, c) for (f, c), flag in zip(zip(f_scores, c_scores), mask) if flag]
    if not pairs:
        return {"n": 0}
    deltas = [f - c for f, c in pairs]
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)
    return {"n": len(pairs), "f_mean": mean(f for f, _ in pairs),
            "c_mean": mean(c for _, c in pairs), "delta_mean": mean(deltas),
            "delta_sum": math.fsum(deltas), "wins": wins, "losses": losses,
            "ties": len(pairs) - wins - losses}


def evaluate_cohort(cohort_key: str, cohort: dict, task: str, views: dict, meta: dict,
                    paired: bool) -> dict:
    """Rescore one cohort under every variant and compute paired statistics.

    paired=True (flickr_last_cells): old C = saved pre-stage native outputs on the
    same images, rescored per-image and paired per row.
    paired=False (g4_independent_F): no same-image old C exists (independent C is
    scored on the COCO eval split, disjoint from the Flickr30k eval split), so all
    paired statistics are None.
    """
    row_map = meta["row_maps"][task]
    if cohort_key == "g4_independent_F":
        cell_keys = [(None, seed) for seed in SEEDS]
        f_bucket = "ind_F"
    elif cohort_key == "flickr_last_cells":
        cell_keys = [(order_name, seed) for order_name in FLICKR_LAST_ORDERS
                     for seed in SEEDS]
        f_bucket = "orac_F"
    else:
        raise ValueError(f"unknown cohort {cohort_key}")
    if paired:
        old_row_map = meta["row_maps"][task]  # pre-stage native rows are task-F rows
    cells_f = cohort[f_bucket]

    # ---- baseline (rule none) with stored-value parity, before any rule result
    baseline_cells = []
    for order_name, seed in cell_keys:
        cell = cells_f[(order_name, seed)]
        ids = list(cell)  # insertion order == saved file order
        check(ids == meta["eval_ids"][task],
              f"{cohort_key} F id order mismatch vs manifest for ({order_name},{seed})")
        predictions = [cell[i]["prediction"] for i in ids]
        per_image = per_image_scores(task, predictions, row_map, ids)
        fresh_corpus = corpus_score(task, predictions, row_map, ids)
        per_image_mean = mean(per_image)
        stored_per_image = [cell[i]["score_stored"] for i in ids]
        max_per_image_diff = max(abs(a - b) for a, b in zip(per_image, stored_per_image))
        check(max_per_image_diff <= PER_IMAGE_TOL,
              f"baseline per-image parity {cohort_key} ({order_name},{seed}): "
              f"{max_per_image_diff:.3e}")
        check(abs(per_image_mean - fresh_corpus) <= CACHED_TOL,
              f"per-image mean vs TaskScorer.score corpus {cohort_key} ({order_name},{seed})")
        old_per_image = None
        old_max_diff = None
        if paired:
            old_cell = cohort["pre_C"][(order_name, seed)]
            old_ids = list(old_cell)
            check(old_ids == meta["eval_ids"][task],
                  f"{cohort_key} old C id order mismatch ({order_name},{seed})")
            old_predictions = [old_cell[i]["prediction"] for i in old_ids]
            old_per_image = per_image_scores(task, old_predictions, old_row_map, old_ids)
            old_max_diff = max(abs(a - b) for a, b in
                               zip(old_per_image,
                                   [old_cell[i]["score_stored"] for i in old_ids]))
            check(old_max_diff <= PER_IMAGE_TOL,
                  f"old C per-image parity {cohort_key} ({order_name},{seed}): "
                  f"{old_max_diff:.3e}")
        if cohort_key == "g4_independent_F":
            stored_corpus = STORED[("independent", independent_name(task, seed))]
        else:
            stored_corpus = STORED[("oracle", bank_name(order_name, seed))]
        check(abs(fresh_corpus - stored_corpus) <= CORPUS_TOL,
              f"F baseline corpus parity {cohort_key} ({order_name},{seed})")
        record = {
            "order": order_name, "seed": seed, "n_rows": len(ids),
            "corpus_cider_d_fresh": fresh_corpus,
            "corpus_cider_d_stored": stored_corpus,
            "fresh_minus_stored": fresh_corpus - stored_corpus,
            "per_image_mean": per_image_mean,
            "per_image_mean_minus_corpus": per_image_mean - fresh_corpus,
            "stored_per_image_mean": mean(stored_per_image),
            "fresh_minus_stored_max_per_image": max_per_image_diff,
        }
        if paired:
            bank = bank_name(order_name, seed)
            old_stored_corpus = STORED[("pre_native", bank)]
            check(abs(mean(old_per_image) - old_stored_corpus) <= CORPUS_TOL,
                  f"old C stored mean vs stored scalar parity ({order_name},{seed})")
            record["old_c_corpus_stored"] = old_stored_corpus
            record["old_c_per_image_mean"] = mean(old_per_image)
            record["old_c_fresh_minus_stored_max_per_image"] = old_max_diff
        baseline_cells.append(record)
        PER_IMAGE_BASELINE[f"{cohort_key}|{order_name}|{seed}"] = (ids, per_image,
                                                                  old_per_image)

    pooled_ids = [i for order_name, seed in cell_keys
                  for i in PER_IMAGE_BASELINE[f"{cohort_key}|{order_name}|{seed}"][0]]
    baseline_records: dict[tuple, tuple[float, float | None]] = {}
    for order_name, seed in cell_keys:
        ids, per_image, old_per_image = PER_IMAGE_BASELINE[f"{cohort_key}|{order_name}|{seed}"]
        for index, row_id in enumerate(ids):
            c_val = old_per_image[index] if paired else None
            baseline_records[(order_name, seed, row_id)] = (per_image[index], c_val)

    token_reports = []
    for order_name, seed in cell_keys:
        ids, _per_image, _old = PER_IMAGE_BASELINE[f"{cohort_key}|{order_name}|{seed}"]
        predictions = [cells_f[(order_name, seed)][i]["prediction"] for i in ids]
        token_reports.append(tokenize_report(task, predictions, row_map, ids))
    token_summary = {"cells": len(token_reports),
                     "mismatch_total": sum(r["mismatch_total"] for r in token_reports),
                     "captions_compared": sum(r["captions_compared"] for r in token_reports)}
    check(token_summary["mismatch_total"] == 0, f"token cache mismatch {cohort_key}")

    eos_flags: dict[tuple, bool] = {}
    old_eos: dict[tuple, bool] = {}
    for order_name, seed in cell_keys:
        cell = cells_f[(order_name, seed)]
        old_cell = cohort["pre_C"][(order_name, seed)] if paired else {}
        for row_id in cell:
            eos_flags[(order_name, seed, row_id)] = cell[row_id]["eos"]
            if paired:
                old_eos[(order_name, seed, row_id)] = old_cell[row_id]["eos"]
    n_non_eos = sum(not flag for flag in eos_flags.values())
    n_eos = len(eos_flags) - n_non_eos
    n_both_eos = sum(1 for key in eos_flags if eos_flags[key] and old_eos[key]) if paired else None
    baseline_words = {key: len(cells_f[(key[0], key[1])][key[2]]["prediction"].split())
                      for key in eos_flags}

    pooled_fresh_baseline = corpus_score(
        task, [cells_f[(o, s)][i]["prediction"] for o, s in cell_keys
               for i in PER_IMAGE_BASELINE[f"{cohort_key}|{o}|{s}"][0]],
        row_map, pooled_ids)
    check(abs(mean([baseline_records[(o, s, i)][0] for o, s in cell_keys
                    for i in PER_IMAGE_BASELINE[f"{cohort_key}|{o}|{s}"][0]])
              - pooled_fresh_baseline) <= CACHED_TOL,
          f"pooled baseline per-image mean vs corpus {cohort_key}")

    variant_results = {}
    for rule in VARIANTS:
        started = time.perf_counter()
        changed_keys: list[tuple] = []
        kept_empty = 0
        f_per_image: dict[tuple, float] = {}
        lengths_after: list[int] = []
        texts_after: dict[tuple, str] = {}
        for order_name, seed in cell_keys:
            ids, baseline_per_image, _old = PER_IMAGE_BASELINE[
                f"{cohort_key}|{order_name}|{seed}"]
            cell = cells_f[(order_name, seed)]
            predictions = []
            for row_id in ids:
                key = (order_name, seed, row_id)
                text = cell[row_id]["prediction"]
                if eos_flags[key]:
                    new_text = text
                else:
                    new_text = apply_rule(rule, text)
                    if new_text != text:
                        changed_keys.append(key)
                    if not new_text.strip():
                        kept_empty += 1
                texts_after[key] = new_text
                predictions.append(new_text)
            if rule == "none":
                per_image = baseline_per_image
            else:
                per_image = per_image_scores(task, predictions, row_map, ids)
                fresh = corpus_score(task, predictions, row_map, ids)
                check(abs(mean(per_image) - fresh) <= CACHED_TOL,
                      f"per-image mean vs corpus {cohort_key}/{rule} ({order_name},{seed})")
            for row_id, value in zip(ids, per_image):
                f_per_image[(order_name, seed, row_id)] = value
            for row_id in ids:
                key = (order_name, seed, row_id)
                if not eos_flags[key]:
                    lengths_after.append(len(texts_after[key].split()))
        cell_corpora = []
        pooled_predictions = []
        for order_name, seed in cell_keys:
            ids, _b, _o = PER_IMAGE_BASELINE[f"{cohort_key}|{order_name}|{seed}"]
            cell_predictions = [texts_after[(order_name, seed, i)] for i in ids]
            pooled_predictions.extend(cell_predictions)
            if rule == "none":
                fresh = next(c["corpus_cider_d_fresh"] for c in baseline_cells
                             if c["order"] == order_name and c["seed"] == seed)
            else:
                fresh = corpus_score(task, cell_predictions, row_map, ids)
            cell_corpora.append({"order": order_name, "seed": seed, "n_rows": len(ids),
                                 "corpus_cider_d": fresh})
        pooled_corpus = corpus_score(task, pooled_predictions, row_map, pooled_ids)
        f_all = [f_per_image[key] for key in eos_flags]
        non_eos_mask = [not eos_flags[key] for key in eos_flags]
        result = {
            "rule": rule,
            "n_truncated": len(changed_keys),
            "kept_empty_rows": kept_empty,
            "cells": cell_corpora,
            "corpus_mean_of_cells": mean(c["corpus_cider_d"] for c in cell_corpora),
            "pooled_corpus_cider_d": pooled_corpus,
            "per_image_mean_all": mean(f_all),
            "per_image_mean_non_eos": mean([f for f, flag in zip(f_all, non_eos_mask) if flag]),
            "paired_all": None, "paired_non_eos": None, "paired_eos": None,
            "paired_both_eos": None,
            "lengths": {
                "non_eos_before": length_stats([baseline_words[key] for key in eos_flags
                                                if not eos_flags[key]]),
                "non_eos_after": length_stats(lengths_after),
                "corpus_mean_words_all_rows": mean(
                    len(texts_after[key].split()) for key in eos_flags),
            },
            "seconds": round(time.perf_counter() - started, 3),
        }
        if paired:
            c_all = [baseline_records[key][1] for key in eos_flags]
            result["paired_all"] = paired_stats(f_all, c_all, None)
            result["paired_non_eos"] = paired_stats(f_all, c_all, non_eos_mask)
            result["paired_eos"] = paired_stats(f_all, c_all, [not flag for flag in non_eos_mask])
            result["paired_both_eos"] = paired_stats(
                f_all, c_all, [eos_flags[key] and old_eos[key] for key in eos_flags])
        if rule == "R1":
            result["saved_repeated_trigram_rows_in_non_eos"] = sum(
                1 for key in eos_flags if not eos_flags[key]
                and cells_f[(key[0], key[1])][key[2]]["repeated_trigram"])
            result["cuts_matching_saved_flag"] = sum(
                1 for key in changed_keys
                if cells_f[(key[0], key[1])][key[2]]["repeated_trigram"])
        variant_results[rule] = result

    cohort_rows = sum(len(cells_f[key]) for key in cell_keys)
    summary = {
        "cohort": cohort_key, "task": task,
        "paired_vs_old_c": paired,
        "n_cells": len(cell_keys), "n_rows": cohort_rows,
        "n_eos": n_eos, "n_non_eos": n_non_eos, "eos_share": n_eos / cohort_rows,
        "n_both_eos_rows": n_both_eos,
        "f_expert_composition": dict(Counter(
            cells_f[key][i]["expert_name"] for key in cell_keys
            for i in cells_f[key])),
        "baseline_cells": baseline_cells,
        "baseline_pooled_corpus_cider_d": pooled_fresh_baseline,
        "token_parity": token_summary,
        "variants": variant_results,
    }
    if paired:
        summary["old_c_expert_composition"] = dict(Counter(
            cohort["pre_C"][key][i]["expert_name"] for key in cell_keys
            for i in cohort["pre_C"][key]))
    return summary


# ---------------------------------------------------------------- period diagnostics
def period_diagnostics(cohort: dict) -> dict:
    """Count '.'/',' in saved predictions (no inference; pure data audit)."""
    sides = {"ind_F": "F", "pre_C": "C", "orac_F": "F"}
    report = {}
    for bucket, task in sides.items():
        rows = [record for cells in cohort[bucket].values() for record in cells.values()]
        eos_rows = [record for record in rows if record["eos"]]
        non_eos_rows = [record for record in rows if not record["eos"]]
        report[bucket] = {
            "task": task, "rows": len(rows),
            "contains_period_rows": sum("." in r["prediction"] for r in rows),
            "contains_comma_rows": sum("," in r["prediction"] for r in rows),
            "eos_rows": len(eos_rows),
            "eos_contains_period_rows": sum("." in r["prediction"] for r in eos_rows),
            "non_eos_rows": len(non_eos_rows),
            "non_eos_contains_period_rows": sum("." in r["prediction"] for r in non_eos_rows),
        }
    variant_period_rows = 0
    variant_outputs = 0
    for bucket in ("ind_F", "orac_F"):
        for cells in cohort[bucket].values():
            for record in cells.values():
                if record["eos"]:
                    continue
                for rule in RULES:
                    variant_outputs += 1
                    if "." in apply_rule(rule, record["prediction"]):
                        variant_period_rows += 1
    report["truncated_variant_outputs"] = {
        "outputs_checked": variant_outputs,
        "contains_period_rows": variant_period_rows,
        "note": "truncation rules only delete text, so a '.' can never be introduced; "
                "verified explicitly over every rule output",
    }
    # whole-input audit: every saved prediction row in the week1 per-example file
    started = time.perf_counter()
    total_rows = 0
    per_task = {"C": {"rows": 0, "contains_period_rows": 0, "contains_comma_rows": 0},
                "F": {"rows": 0, "contains_period_rows": 0, "contains_comma_rows": 0},
                "other": {"rows": 0, "contains_period_rows": 0, "contains_comma_rows": 0}}
    with gzip.open(ROWS_GZ, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row["task"] if row["task"] in ("C", "F") else "other"
            bucket = per_task[task]
            bucket["rows"] += 1
            if "." in row["prediction"]:
                bucket["contains_period_rows"] += 1
            if "," in row["prediction"]:
                bucket["contains_comma_rows"] += 1
            total_rows += 1
    report["whole_input_file"] = {
        "rows": total_rows,
        "per_task": per_task,
        "seconds": round(time.perf_counter() - started, 3),
        "note": "every saved prediction row in artifacts/week1-per-example-scores.jsonl.gz "
                "(all kinds/stages/phases); archive-wide totals (incl. G1) are in "
                "artifacts/stage05-target-format.json",
    }
    return report


# ---------------------------------------------------------------- main
def main() -> int:
    if OUT_JSON.exists() or OUT_JSON.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_JSON}")
    total_started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    input_paths = [ROWS_GZ, MANIFESTS_TAR, G4_TAR, AUDIT_G4, AUDIT_G5, CONFIRMATION_LOCK,
                   TARGET_FORMAT, ROOT / "sema_multimodal/metrics.py"]
    inputs = []
    for path in input_paths:
        if path.exists():
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size})
        else:
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": None,
                           "bytes": None, "note": "not present in this environment"})

    script_sha = sha256_file(Path(__file__))

    java_path = shutil.which("java")
    java_version = None
    if java_path:
        try:
            probe = subprocess.run([java_path, "-version"], capture_output=True, text=True,
                                   timeout=30)
            java_version = ((probe.stderr or "").strip() or (probe.stdout or "").strip()
                            ).splitlines()[0] if (probe.stderr or probe.stdout) else "unknown"
        except Exception as error:  # record, never abort scoring on a probe failure
            java_version = f"unavailable: {error!r}"

    cohort = load_cohort_rows()
    views, meta = load_task_views()
    STORED.update(stored_scalars())

    TIMING["phase"] = "scorers"
    started = time.perf_counter()
    SCORERS["F"] = M.TaskScorer(views["F"])
    locks_report = json.loads(CONFIRMATION_LOCK.read_text())["scorer_locks"]
    lock_drift = {}
    recorded, fresh = locks_report["F"], SCORERS["F"].lock
    drift = {key: [recorded.get(key), fresh.get(key)]
             for key in sorted(set(recorded) & set(fresh)) if recorded[key] != fresh[key]}
    check(not drift, f"scorer lock drift for F: {drift}")
    lock_drift["F"] = {"drift": drift, "fields_match": set(recorded) == set(fresh),
                       "implementation_sha256": fresh.get("implementation_sha256"),
                       "document_frequency_sha256": fresh.get("document_frequency_sha256"),
                       "train_documents": fresh.get("train_documents")}
    TIMING["build_scorers_seconds"] = round(time.perf_counter() - started, 3)

    TIMING["phase"] = "tokenize"
    strings_by_task = collect_strings(cohort, views)
    token_cache_report = build_token_cache(("F",), strings_by_task)

    TIMING["phase"] = "score"
    summary_independent = evaluate_cohort("g4_independent_F", cohort, "F", views, meta,
                                          paired=False)
    summary_flickr = evaluate_cohort("flickr_last_cells", cohort, "F", views, meta,
                                     paired=True)
    period = period_diagnostics(cohort)

    envelope = {
        "stage": "stage0.5-truncation-rescore",
        "generated_by": {
            "script": "scripts/stage05_truncation_rescore.py",
            "script_sha256": script_sha,
            "generated_at_utc": run_started_utc,
            "python": platform.python_version(),
            "pycocoevalcap": importlib.metadata.version("pycocoevalcap"),
            "numpy": importlib.metadata.version("numpy"),
            "platform": platform.platform(),
            "java_path": java_path,
            "java_version": java_version,
        },
        "environment_note": {
            "built_by_parent_local_env": True,
            "versions_matched_to_remote": {
                "python": "3.12.14", "numpy": "1.26.4", "pycocoevalcap": "1.2"},
            "java_difference": "local OpenJDK 21 (nixos) vs remote 22.0.2; PTBTokenizer "
                               "-preserveLines -lowerCase assumed version-stable; the parity "
                               "gates against stored values verify this empirically",
            "remote_run_replaced": "sbatch on Yonsei-DC1 not used for this stage because SSH "
                                   "was reset before banner for all agents (see "
                                   "artifacts/stage05-attempts.log); scoring ran locally, "
                                   "CPU only, single process",
        },
        "execution": {
            "training": False, "inference": False,
            "scoring": "rescoring of saved predictions only (no model loaded)",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURM_JOB_NODELIST"),
            "attempt": os.environ.get("STAGE05_ATTEMPT"),
        },
        "inputs": inputs,
        "labels": {
            "post_hoc_on_eval_split": True,
            "diagnostic_only": True,
            "not_method_selection": True,
            "rules_apply_only_to_non_eos_outputs": True,
            "saved_artifacts_unmodified": True,
            "note": "All truncation rules are applied post hoc to saved eval-split Flickr "
                    "predictions. Diagnostic only; no method/config/scorer is selected or "
                    "changed, and no saved artifact is modified.",
        },
        "cohorts": {
            "g4_independent_F": {
                "definition": "kind=independent, task=F, stage=0 (8 seeds x 512 eval rows); "
                              "no same-image old C exists (independent C is saved on the COCO "
                              "eval split), so paired statistics are None for this cohort",
                "stored_corpus_source": "audit_g4 transfer_records independent_score "
                                        "(order-invariant per seed/task)",
            },
            "flickr_last_cells": {
                "definition": "16 cells = orders {V_C_G_F, V_G_C_F} x 8 seeds; F = saved "
                              "post-stage task-ID-oracle F-expert outputs; old C = saved "
                              "pre-stage native outputs (expert C), paired per (order, seed, id)",
                "stored_corpus_source": "oracle: routing_audit task_id_oracle matrix[3][3]; "
                                        "pre native: g4 results.json pre_task_scores[3]",
            },
        },
        "truncation_rules": {
            "none": "untruncated baseline (parity anchor)",
            "R1": "cut at start of first repeated token trigram "
                  "(tokens[pos:pos+3] == tokens[pos+3:pos+6], the exact repeated_trigram "
                  "definition of sema_multimodal/model.py); keep text before the repeat; "
                  "unchanged if none",
            "R2": "cut at earliest ', and' or ' and a ' marker with >= 8 words before it; "
                  "keep text before the marker start, rstripped; unchanged if none",
            "R3_12": "keep first 12 words (unchanged if <= 12)",
            "R3_16": "keep first 16 words (unchanged if <= 16)",
            "R3_20": "keep first 20 words (unchanged if <= 20)",
            "R4": "R1 then R3 with N=20",
            "scope": "applied ONLY to non-EOS outputs (generation.eos == true stays "
                     "byte-identical); rules only delete text, never insert characters",
        },
        "row_counts": {f"{k[0]}|{k[1]}|{k[2]}|p{k[3]:02d}": v
                       for k, v in sorted(cohort["counts"].items(), key=str)},
        "scorer_locks_parity": lock_drift,
        "token_cache": token_cache_report,
        "period_diagnostics": period,
        "results": {
            "g4_independent_F": summary_independent,
            "flickr_last_cells": summary_flickr,
        },
        "parity": {
            "baseline_per_image_max_abs_diff": max(
                cell["fresh_minus_stored_max_per_image"]
                for summary in (summary_independent, summary_flickr)
                for cell in summary["baseline_cells"]),
            "old_c_per_image_max_abs_diff": max(
                (cell["old_c_fresh_minus_stored_max_per_image"]
                 for summary in (summary_independent, summary_flickr)
                 for cell in summary["baseline_cells"]
                 if "old_c_fresh_minus_stored_max_per_image" in cell),
                default=None),
            "baseline_corpus_max_abs_diff": max(
                abs(cell["fresh_minus_stored"]) for summary in (summary_independent,
                                                                summary_flickr)
                for cell in summary["baseline_cells"]),
            "tolerances": {"per_image": PER_IMAGE_TOL, "corpus": CORPUS_TOL,
                           "per_image_mean_vs_corpus": CACHED_TOL},
        },
        "runtimes": {},
        "failures": failures,
        "status": "ok" if not failures else "failed",
    }
    envelope["runtimes"] = {
        "total_seconds": round(time.perf_counter() - total_started, 3),
        **{key: value for key, value in TIMING.items() if key.endswith("_seconds")},
    }

    OUT_JSON.write_text(json.dumps(envelope, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print("wrote", OUT_JSON, flush=True)
    print("status:", envelope["status"], "| failures:", len(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
