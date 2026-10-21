"""Week-1 item 1-C: per-example rescoring of saved G4/G5 multimodal predictions.

CPU only. Reuses the frozen ``sema_multimodal`` scorer protocol verbatim (the metrics
module implementation SHA is pinned against the original run-time lock), executes NO
model inference and imports no model/run module. Per-example VQA accuracies come from
the same ``vqav2_accuracy``/``gqa_accuracy`` functions; per-image CIDEr-D comes from
the frozen-DF ``_fixed_cider_score`` on a singleton CIDEr corpus built from cached PTB
tokens of the exact strings TaskScorer.score would tokenize (line identity is proven
per corpus against the in-corpus PTB batch).

Requested scope (exact saved prediction files; each corpus is one 512-image eval):
  * diagonal, all 128 cells (4 orders x 8 seeds x 4 tasks): native (G4), forced
    task-ID oracle (G5 ``task_id_oracle``), same-seed independent (G4).
  * caption family final stage, 64 cells: native + task-ID oracle for both caption
    tasks (files shared with the diagonal corpus where the member is identical).
  * Flickr30k before/after for the 16 later-task banks (2 orders x 8 seeds): native
    pre (``predictions_before_03_F``) vs final native, plus the saved caption-family
    restricted pre rows (the only stored pre-state caption-family predictions). The
    G5 task-ID oracle has no before-stage diagnostics by construction, so that
    pairing is reported missing and never imputed.

Verification contract:
  * corpus mean of per-example scores == fresh ``TaskScorer.score`` (strict) ==
    stored audit/results corpus scalar (strict).
  * PTB token cache proven line-identical against the in-corpus batch tokenization
    for every requested caption corpus.
  * per-(task, row id, prediction) computation is reused across variants when the
    prediction text is identical; every hit/miss is counted in the per-corpus record.
  * G5 native prediction files verified row-identical against G4 native files, so the
    archived G4 file is the canonical native corpus.

Outputs (new files only; never overwrites an existing artifact):
  artifacts/week1-per-example-scores.jsonl.gz   one row per scored occurrence
  artifacts/week1-per-example-rescore.json      evidence envelope with all checks
  artifacts/week1-per-example-report.md         concise Korean narrative

Static safety: no torch/model import, no scorer edit, read-only on all existing
artifacts, and no import of scripts that execute at import time.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sema_multimodal import metrics as M  # metrics module only; never .model / .run

ORDERS = ("V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V")
SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
TASKS = ("V", "C", "G", "F")
CAPTION_TASKS = ("C", "F")
EVAL_ROWS = 512

G4_TAR = ROOT / "artifacts/multimodal-g4-raw.tar.gz"
G5_TAR = ROOT / "artifacts/multimodal-g5-raw.tar.gz"
G4_ROOT = "artifacts/multimodal_runs/g4"
G5_ROOT = "artifacts/multimodal_runs/routing_g4"
MANIFESTS_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
AUDIT_G4 = ROOT / "artifacts/multimodal-g4-report/audit.json"
AUDIT_G5 = ROOT / "artifacts/multimodal-g5-report/routing_audit.json"
NORMALIZERS = ROOT / "artifacts/multimodal-g1-normalizers.json"
CONFIRMATION_LOCK = ROOT / "artifacts/multimodal-confirmation-lock.json"
ROUTING_REVIEW = ROOT / "artifacts/routing-review-routing-partition.json"

OUT_JSON = ROOT / "artifacts/week1-per-example-rescore.json"
OUT_ROWS = ROOT / "artifacts/week1-per-example-scores.jsonl.gz"
OUT_REPORT = ROOT / "artifacts/week1-per-example-report.md"

ROW_MEAN_TOL = 1e-9      # per-image mean vs fresh TaskScorer.score corpus value
STORED_TOL = 1e-9        # fresh value vs stored audit/results scalar
MIRROR_TOL = 1e-12       # stored cross-copy equality (results vs audit vs g5 matrix)
IDENTITY_TOL = 1e-12     # T == R + K identity per cell
TEXT_EQUAL_TOL = 1e-12   # own text-equal paired-delta sum must be exactly 0
REVIEW_TOL = 1e-6        # review-artifact count/delta reconciliation

TIMING: dict[str, object] = {}
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("empty mean")
    return math.fsum(values) / len(values)


def one(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected finite scalar, got {value!r}")
    return float(value)


def bank_name(order_name: str, seed: int) -> str:
    return f"{order_name}_expand_lr0.0003_ep5_seed{seed}"


def independent_name(task_id: str, seed: int) -> str:
    return f"{task_id}_independent_lr0.0003_ep5_seed{seed}"


def seed_of(bank: str) -> int:
    return int(bank.rsplit("seed", 1)[1])


def corpora_key(entry: dict) -> str:
    return f"{entry['kind']}|{entry['bank']}|{entry['task']}_p{entry['stage']:02d}|{entry['phase']}"


# ---------------------------------------------------------------- wanted members
WANTED: dict[Path, set[str]] = {
    G4_TAR: {f"{G4_ROOT}/{bank_name(o, s)}/config.json" for o in ORDERS for s in SEEDS} |
            {f"{G4_ROOT}/{independent_name(t, s)}/config.json" for t in TASKS for s in SEEDS},
    G5_TAR: set(),
    MANIFESTS_TAR: {MANIFEST_MEMBER},
}
for _o in ORDERS:
    _seq = _o.split("_")
    _pos = {task: p for p, task in enumerate(_seq)}
    for _s in SEEDS:
        _bank = bank_name(_o, _s)
        WANTED[G4_TAR].add(f"{G4_ROOT}/{_bank}/results.json")
        for _p in range(4):
            _task = _seq[_p]
            WANTED[G4_TAR].add(f"{G4_ROOT}/{_bank}/predictions_after_{_p:02d}_{_task}.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/task_id_oracle/predictions_after_{_p:02d}_{_task}.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/native_joint/predictions_after_{_p:02d}_{_task}.jsonl")
            WANTED[G4_TAR].add(f"{G4_ROOT}/{independent_name(_task, _s)}/predictions_after_00_{_task}.jsonl")
            WANTED[G4_TAR].add(f"{G4_ROOT}/{independent_name(_task, _s)}/results.json")
        for _task in CAPTION_TASKS:
            WANTED[G4_TAR].add(f"{G4_ROOT}/{_bank}/predictions_after_03_{_task}.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/task_id_oracle/predictions_after_03_{_task}.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/native_joint/predictions_after_03_{_task}.jsonl")
        if _pos["F"] == 3:
            WANTED[G4_TAR].add(f"{G4_ROOT}/{_bank}/predictions_before_03_F.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/native_joint/predictions_before_03_F.jsonl")
            WANTED[G5_TAR].add(f"{G5_ROOT}/{_bank}/family_restricted_joint/predictions_before_03_F.jsonl")


class Archives:
    """Read each wanted member exactly once from the gzip archives."""

    def __init__(self):
        self.data: dict[tuple[Path, str], bytes] = {}
        self.uncompressed_read = 0

    def preload(self) -> None:
        started = time.perf_counter()
        for archive in (G4_TAR, G5_TAR, MANIFESTS_TAR):
            with tarfile.open(archive, "r:gz") as tar:
                wanted = WANTED[archive]
                for info in tar:
                    if info.name in wanted:
                        stream =tar.extractfile(info)
                        self.data[(archive, info.name)] = stream.read() if stream else b""
                        self.uncompressed_read += 1
        TIMING["archive_preload_seconds"] = round(time.perf_counter() - started, 3)
        missing = [f"{a.name}@{m}" for a in (G4_TAR, G5_TAR, MANIFESTS_TAR)
                   for m in WANTED[a] if (a, m) not in self.data]
        if missing:
            raise FileNotFoundError(f"missing wanted members, first 8: {missing[:8]}")

    def read(self, archive: Path, member: str) -> bytes:
        try:
            return self.data[(archive, member)]
        except KeyError as error:
            raise FileNotFoundError(f"{archive.name}: {member}") from error

    def g4(self, member: str) -> bytes:
        return self.read(G4_TAR, f"{G4_ROOT}/{member}")

    def g5(self, member: str) -> bytes:
        return self.read(G5_TAR, f"{G5_ROOT}/{member}")

    def json_g4(self, member: str) -> dict:
        return json.loads(self.g4(member).decode("utf-8"))

    def jsonl_g4(self, member: str) -> list[dict]:
        return parse_jsonl(self.g4(member))

    def jsonl_g5(self, member: str) -> list[dict]:
        return parse_jsonl(self.g5(member))


ARCHIVES = Archives()
MANIFEST_SHA256 = None


# ---------------------------------------------------------------- manifest + configs
def load_manifest_and_configs() -> tuple[dict, dict]:
    global MANIFEST_SHA256
    raw = ARCHIVES.read(MANIFESTS_TAR, MANIFEST_MEMBER)
    manifest = json.loads(raw.decode("utf-8"))
    MANIFEST_SHA256 = sha256_bytes(raw)
    configs = {}
    for order_name in ORDERS:
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            configs[f"expand/{bank}"] = ARCHIVES.json_g4(f"{bank}/config.json")
            for task in TASKS:
                name = independent_name(task, seed)
                configs.setdefault(f"independent/{name}",
                                   ARCHIVES.json_g4(f"{name}/config.json"))
    drift = {name: config["manifest_sha256"] for name, config in configs.items()
             if config["manifest_sha256"] != MANIFEST_SHA256}
    check(not drift, f"manifest sha mismatch in run configs: {sorted(drift)[:4]}")
    task_views = {}
    for record in manifest["tasks"]:
        task_views[record["task_id"]] = {"task_id": record["task_id"], "family": record["family"],
                                         "metric": record["metric"], "train": record["train"],
                                         "eval": record["eval"]}
    return task_views, configs


TASK_VIEWS: dict[str, dict] = {}
EVAL_ROW_MAPS: dict[str, dict] = {}
EVAL_ID_LISTS: dict[str, list[str]] = {}
BANK_ORDERS: dict[str, tuple] = {}
PER_EXAMPLE_CACHE: dict[tuple[str, str, str], float] = {}
SCORERS: dict[str, object] = {}
TOKEN_CACHE: dict[str, dict[str, list[str]]] = {}


# ---------------------------------------------------------------- corpora grid
def enumerate_corpora() -> list[dict]:
    corpora: dict[tuple, dict] = {}

    def add(kind, bank, task, stage, role, member, phase):
        key = (kind, bank, task, stage, phase)
        entry = corpora.setdefault(key, {"kind": kind, "bank": bank, "task": task, "stage": stage,
                                         "roles": [], "member": member, "phase": phase})
        if role not in entry["roles"]:
            entry["roles"].append(role)

    for order_name in ORDERS:
        seq = order_name.split("_")
        pos = {task: p for p, task in enumerate(seq)}
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            for p, task in enumerate(seq):
                add("native", bank, task, p, "diag",
                    f"predictions_after_{p:02d}_{task}.jsonl", "after")
                add("oracle", bank, task, p, "diag",
                    f"task_id_oracle/predictions_after_{p:02d}_{task}.jsonl", "after")
            for task in CAPTION_TASKS:
                if pos[task] != 3:
                    add("native", bank, task, 3, "cap_final",
                        f"predictions_after_03_{task}.jsonl", "after")
                else:
                    existing = corpora[("native", bank, task, 3, "after")]
                    if "cap_final" not in existing["roles"]:
                        existing["roles"].append("cap_final")
                add("oracle", bank, task, 3, "cap_final",
                    f"task_id_oracle/predictions_after_03_{task}.jsonl", "after")
            if pos["F"] == 3:
                add("native", bank, "F", 3, "flickr_pre", "predictions_before_03_F.jsonl", "before")
                add("family_restricted", bank, "F", 3, "flickr_pre",
                    "family_restricted_joint/predictions_before_03_F.jsonl", "before")
    for task in TASKS:
        for seed in SEEDS:
            name = independent_name(task, seed)
            add("independent", name, task, 0, "independent",
                f"predictions_after_00_{task}.jsonl", "after")
    entries = []
    for key in sorted(corpora):
        entry = dict(corpora[key])
        entry["roles"] = sorted(set(entry["roles"]))
        entries.append(entry)
    return entries


def corpus_rows(entry: dict) -> list[dict]:
    if entry["kind"] in ("oracle", "family_restricted"):
        return ARCHIVES.jsonl_g5(f"{entry['bank']}/{entry['member']}")
    return ARCHIVES.jsonl_g4(f"{entry['bank']}/{entry['member']}")


def collect_caption_strings(entries: list[dict]) -> dict:
    strings: dict[str, set[str]] = {"C": set(), "F": set()}
    for task in CAPTION_TASKS:
        for split in ("train", "eval"):
            for row in TASK_VIEWS[task][split]:
                for ref in row["references"]:
                    strings[task].add(ref.replace("\n", " ").replace("\r", " "))
    for entry in entries:
        task = entry["task"]
        if task not in strings:
            continue
        for row in corpus_rows(entry):
            strings[task].add(row["prediction"].replace("\n", " ").replace("\r", " "))
    return {task: sorted(values) for task, values in strings.items()}


def build_token_cache(caption_strings: dict, chunk_size: int = 50000) -> dict:
    report = {}
    for task, values in caption_strings.items():
        cache: dict[str, list[str]] = {}
        started = time.perf_counter()
        for start in range(0, len(values), chunk_size):
            chunk = values[start:start + chunk_size]
            mapping = {f"s{i}": [value] for i, value in enumerate(chunk)}
            tokenized = M._tokenize(mapping)
            if set(tokenized) != set(mapping):
                raise ValueError(f"PTB chunk lost captions for task {task}")
            for i, value in enumerate(chunk):
                cache[value] = tokenized[f"s{i}"][0]
        TIMING[f"tokenize_seconds_{task}"] = round(time.perf_counter() - started, 3)
        report[task] = {"unique_strings": len(values), "chunks": math.ceil(len(values) / chunk_size)}
        TOKEN_CACHE[task] = cache
    return report


def build_scorers() -> None:
    started = time.perf_counter()
    for task_id, view in TASK_VIEWS.items():
        SCORERS[task_id] = M.TaskScorer(view)
    TIMING["build_scorers_seconds"] = round(time.perf_counter() - started, 3)


# ---------------------------------------------------------------- stored values
def stored_corpus_scores() -> tuple[dict, dict]:
    audit_g4 = json.loads(AUDIT_G4.read_text())
    audit_g5 = json.loads(AUDIT_G5.read_text())
    stored: dict[tuple, dict] = {}
    transfers = {}
    for record in audit_g4["transfer_records"]:
        if record["kind"] == "common_cl" and record["method"] == "expand":
            for row in record["rows"]:
                transfers[(tuple(record["order"]), record["seed"], row["task_id"])] = row
    check(len(transfers) == 128, f"expected 128 common-CL transfer rows, got {len(transfers)}")
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            results = ARCHIVES.json_g4(f"{bank}/results.json")
            for p, task in enumerate(order):
                diag_entry = stored.setdefault(("native_diag", bank, task, p), {})
                diag_entry.update({
                    "results_matrix": one(results["matrix"][p][p]),
                    "audit_transfer_post_train": one(transfers[(order, seed, task)]["post_train"])})
            for task in CAPTION_TASKS:
                p = order.index(task)
                if p != 3:
                    # the final-stage native file for C is always distinct from diag C
                    stored.setdefault(("native_final", bank, task, 3), {})["results_matrix"] = \
                        one(results["matrix"][3][p])
                else:
                    stored[("native_diag", bank, task, 3)]["results_matrix_final"] = \
                        one(results["matrix"][3][p])
            if order.index("F") == 3:
                stored.setdefault(("native_pre", bank, "F", 3), {})["results_pre_task_scores"] = \
                    one(results["pre_task_scores"][3])
            for task in TASKS:
                name = independent_name(task, seed)
                stored[("independent", name, task, 0)] = {
                    "results_matrix": one(ARCHIVES.json_g4(f"{name}/results.json")["matrix"][0][0]),
                    "audit_independent_score": one(transfers[(order, seed, task)]["independent_score"])}
    for run in audit_g5["runs"]:
        bank = run["directory"].rstrip("/").split("/")[-1]
        order = tuple(run["parent_config"]["task_ids"])
        parent = run["parent_config"]
        if "manifest_sha256" in parent:
            check(parent["manifest_sha256"] == MANIFEST_SHA256,
                  f"G5 parent config manifest drift {bank}")
        variants = run["variants"]
        for p, task in enumerate(order):
            stored[("oracle_diag", bank, task, p)] = {
                "g5_routing_audit_matrix": one(variants["task_id_oracle"]["matrix"][p][p])}
            stored[("native_diag", bank, task, p)]["g5_native_matrix"] = \
                one(variants["native_joint"]["matrix"][p][p])
        for task in CAPTION_TASKS:
            p = order.index(task)
            stored[("oracle_final", bank, task, 3)] = {
                "g5_routing_audit_matrix": one(variants["task_id_oracle"]["matrix"][3][p])}
            if p != 3:
                stored[("native_final", bank, task, 3)]["g5_native_matrix"] = \
                    one(variants["native_joint"]["matrix"][3][p])
            else:
                stored[("native_diag", bank, task, 3)]["g5_native_matrix_final"] = \
                    one(variants["native_joint"]["matrix"][3][p])
        if order.index("F") == 3:
            family = variants.get("family_restricted_joint", {})
            pre_scores = family.get("pre_task_scores")
            stored.setdefault(("family_restricted", bank, "F", 3), {})
            if isinstance(pre_scores, list) and pre_scores[3] is not None:
                stored[("family_restricted", bank, "F", 3)]["g5_routing_pre_task_scores"] = \
                    one(pre_scores[3])
    return stored, transfers


# ---------------------------------------------------------------- locks
def verify_locks(configs: dict, lock_doc: dict) -> dict:
    recorded = lock_doc["scorer_locks"]
    report = {}
    for task_id, scorer in sorted(SCORERS.items()):
        fresh = scorer.lock
        stale = recorded[task_id]
        check(set(stale) == set(fresh), f"scorer lock field drift for {task_id}")
        drift = {key: [stale.get(key), fresh.get(key)] for key in sorted(set(stale) & set(fresh))
                 if stale[key] != fresh[key]}
        check(not drift, f"scorer lock value drift for {task_id}: {drift}")
        run_drift = {name: config["scorer_locks"][task_id] for name, config in configs.items()
                     if task_id in config["scorer_locks"] and config["scorer_locks"][task_id] != stale}
        check(not run_drift, f"run-config scorer lock drift for {task_id}: {sorted(run_drift)[:4]}")
        report[task_id] = {"drift": drift, "run_configs_parity": not run_drift,
                           "units": stale.get("units"),
                           "train_documents": stale.get("train_documents"),
                           "implementation_sha256": stale.get("implementation_sha256"),
                           "document_frequency_sha256": stale.get("document_frequency_sha256")}
    return report


# ---------------------------------------------------------------- per-corpus scoring
def score_corpus(entry: dict) -> dict:
    key = corpora_key(entry)
    rows = corpus_rows(entry)
    ids = [row["id"] for row in rows]
    check(ids == EVAL_ID_LISTS[entry["task"]], f"id order mismatch {key}")
    predictions = [row["prediction"] for row in rows]
    experts = [row["expert"] for row in rows]
    generations = [row.get("generation") for row in rows]
    view = TASK_VIEWS[entry["task"]]
    metric = view["metric"]
    rows_manifest = [EVAL_ROW_MAPS[entry["task"]][i] for i in ids]
    scorer = SCORERS[entry["task"]]

    started = time.perf_counter()
    per_example: list[float] = []
    computed = reused = 0
    token_report = None
    if metric == "vqav2":
        for index, (row, prediction) in enumerate(zip(rows_manifest, predictions)):
            cache_key = (entry["task"], ids[index], prediction)
            hit = PER_EXAMPLE_CACHE.get(cache_key)
            if hit is not None:
                per_example.append(hit)
                reused += 1
            else:
                value = 100.0 * M.vqav2_accuracy(prediction, row["references"])
                per_example.append(value)
                PER_EXAMPLE_CACHE[cache_key] = value
                computed += 1
        units = "accuracy_points"
    elif metric == "gqa":
        for index, (row, prediction) in enumerate(zip(rows_manifest, predictions)):
            cache_key = (entry["task"], ids[index], prediction)
            hit = PER_EXAMPLE_CACHE.get(cache_key)
            if hit is not None:
                per_example.append(hit)
                reused += 1
            else:
                value = 100.0 * M.gqa_accuracy(prediction, row["references"][0])
                per_example.append(value)
                PER_EXAMPLE_CACHE[cache_key] = value
                computed += 1
        units = "accuracy_points"
    else:
        from pycocoevalcap.cider.cider_scorer import CiderScorer
        df = scorer._df
        train_documents = scorer._train_documents
        cache = TOKEN_CACHE[entry["task"]]
        for index, (prediction, row) in enumerate(zip(predictions, rows_manifest)):
            cache_key = (entry["task"], ids[index], prediction)
            hit = PER_EXAMPLE_CACHE.get(cache_key)
            if hit is not None:
                per_example.append(hit)
                reused += 1
            else:
                singleton = CiderScorer(n=4, sigma=6.0)
                singleton += (cache[prediction.replace("\n", " ").replace("\r", " ")],
                              [cache[ref.replace("\n", " ").replace("\r", " ")]
                               for ref in row["references"]])
                value = M._fixed_cider_score(singleton, df, train_documents)
                per_example.append(value)
                PER_EXAMPLE_CACHE[cache_key] = value
                computed += 1
        units = "native_cider_d"
    scoring_seconds = time.perf_counter() - started

    started = time.perf_counter()
    fresh = scorer.score(predictions, rows_manifest)
    scorer_score_seconds = time.perf_counter() - started
    fresh_score = one(fresh["score"])
    per_example_mean = math.fsum(per_example) / len(rows_manifest)
    check(abs(per_example_mean - fresh_score) <= ROW_MEAN_TOL,
          f"per-image mean vs fresh TaskScorer.score mismatch {key}")
    if units == "native_cider_d":
        token_report = token_parity(entry["task"], rows_manifest, predictions)
        check(token_report["mismatch_total"] == 0, f"token cache mismatch {key}")
    return {"entry": dict(entry), "key": key, "ids": ids, "experts": experts,
            "predictions": predictions, "generations": generations, "per_example": per_example,
            "per_example_mean": per_example_mean, "fresh_score": fresh_score, "units": units,
            "computed": computed, "reused": reused,
            "scoring_seconds": scoring_seconds, "scorer_score_seconds": scorer_score_seconds,
            "token_report": token_report}


def token_parity(task: str, rows_manifest: list, predictions: list) -> dict:
    """Rebuild exactly TaskScorer.score's captions dict and compare against the cache."""
    captions = {("reference", index): row["references"] for index, row in enumerate(rows_manifest)}
    captions.update({("prediction", index): [pred] for index, pred in enumerate(predictions)})
    tokenized = M._tokenize(captions)
    cache = TOKEN_CACHE[task]
    matches = 0
    total_miss = 0
    misses: list[dict] = []
    for key, tokens in tokenized.items():
        kind, index = key
        sources = rows_manifest[index]["references"] if kind == "reference" else [predictions[index]]
        for source, token in zip(sources, tokens):
            normalized = source.replace("\n", " ").replace("\r", " ")
            if cache.get(normalized) == token:
                matches += 1
            else:
                total_miss += 1
                if len(misses) < 8:
                    misses.append({"cached": cache.get(normalized, "")[:48], "batch": token[:48]})
    return {"captions_compared": len(tokenized), "cache_line_matches": matches,
            "mismatch_total": total_miss, "mismatch_preview": misses}


def g5_native_parity(entries: list[dict]) -> dict:
    identical = 0
    diffs = []
    checked = 0
    for entry in entries:
        if entry["kind"] != "native":
            continue
        g4_rows = corpus_rows(entry)
        g5_rows = ARCHIVES.jsonl_g5(f"{entry['bank']}/native_joint/{entry['member']}")
        checked += 1
        if ([(r["id"], r["prediction"], r["expert"]) for r in g4_rows] ==
                [(r["id"], r["prediction"], r["expert"]) for r in g5_rows]):
            identical += 1
        else:
            diffs.append({"key": f"{entry['bank']}/{entry['member']}",
                          "prediction_diff_rows": sum(1 for a, b in zip(g4_rows, g5_rows)
                                                      if a["prediction"] != b["prediction"]),
                          "expert_diff_rows": sum(1 for a, b in zip(g4_rows, g5_rows)
                                                  if a["expert"] != b["expert"])})
    check(not diffs, f"g4/g5 native prediction files disagree: {diffs[:4]}")
    return {"files_checked": checked, "files_row_identical": identical, "files_disagreeing": diffs}


def stored_parity(scored: dict, stored: dict) -> dict:
    maxima = {"fresh_vs_stored_max": 0.0, "stored_mirror_max": 0.0}
    per_corpus = []
    for key_union in sorted(scored):
        kind, bank, task, stage, phase = key_union
        entry = scored[key_union]
        roles = entry["entry"]["roles"]
        if kind == "native" and phase == "before":
            stored_key = "native_pre"
        elif kind == "native":
            stored_key = "native_diag" if "diag" in roles else "native_final"
        elif kind == "oracle":
            stored_key = "oracle_diag" if "diag" in roles else "oracle_final"
        else:
            stored_key = kind
        sources = stored.get((stored_key, bank, task, stage), None)
        comparisons: list[dict] = []
        note = {}
        if sources is None:
            note["stored_missing"] = f"no stored entry for {stored_key}|{bank}|{task}|p{stage}"
        else:
            values = []
            for source, value in sources.items():
                comparisons.append({"source": source, "stored": value, "fresh": entry["fresh_score"],
                                    "delta": abs(value - entry["fresh_score"])})
                values.append(value)
            for i, a in enumerate(values):
                for b in values[i + 1:]:
                    maxima["stored_mirror_max"] = max(maxima["stored_mirror_max"], abs(a - b))
            if comparisons:
                maxima["fresh_vs_stored_max"] = max(
                    maxima["fresh_vs_stored_max"], max(comparison["delta"] for comparison in comparisons))
        check(not note.get("stored_missing"), note.get("stored_missing", ""))
        per_corpus.append({"key": entry["key"], "sources": comparisons,
                           "stored_key": stored_key,
                           "stored_missing": note.get("stored_missing")})
    check(maxima["fresh_vs_stored_max"] <= STORED_TOL,
          f"fresh scores do not reconcile with stored: max {maxima['fresh_vs_stored_max']}")
    return {"maxima": maxima, "per_corpus": per_corpus}


# ---------------------------------------------------------------- cell analysis
def paired_cell(native: dict, oracle: dict, independent: dict | None,
                order_name: str, task: str) -> dict:
    order = tuple(order_name.split("_"))
    pos = order.index(task)
    other = "F" if task == "C" else "C"
    pos_other = order.index(other)
    rows = len(native["per_example"])
    experts = native["experts"]
    own_mask = [x == pos for x in experts]
    mis_mask = [x != pos for x in experts]
    same_family_other_mask = [x == pos_other for x in experts]
    cross_mask = [x not in (pos, pos_other) for x in experts]
    own_equal_mask = [own and a == b for own, a, b in
                      zip(own_mask, native["predictions"], oracle["predictions"])]
    route_differ_text_equal = [mis and a == b for mis, a, b in
                               zip(mis_mask, native["predictions"], oracle["predictions"])]
    own_mismatch = [(i, native["ids"][i], native["predictions"][i], oracle["predictions"][i])
                    for i in range(rows) if own_mask[i]
                    and native["predictions"][i] != oracle["predictions"][i]]

    record = {
        "order": order_name, "task": task, "n_rows": rows,
        "corpus_native": native["key"], "corpus_oracle": oracle["key"],
        "counts": {"own": int(sum(own_mask)), "misrouted": int(sum(mis_mask)),
                   "same_family_other": int(sum(same_family_other_mask)),
                   "cross_family": int(sum(cross_mask))},
        "own_route_text_mismatch_rows": len(own_mismatch),
        "own_route_text_mismatch_ids_preview": [occurrence[1] for occurrence in own_mismatch[:8]],
        "own_text_equal_own_rows": int(sum(own_equal_mask)),
        "route_differ_text_equal_rows": int(sum(route_differ_text_equal)),
        "mis_targets": dict(Counter(order[x] for x in experts if x != pos)),
        "group_score_means": {}, "group_paired": {},
    }
    for name, mask in (("own", own_mask), ("same_family_other", same_family_other_mask),
                       ("cross_family", cross_mask), ("misrouted", mis_mask)):
        indexes = [i for i, flag in enumerate(mask) if flag]
        if not indexes:
            continue
        n_vals = [native["per_example"][i] for i in indexes]
        o_vals = [oracle["per_example"][i] for i in indexes]
        record["group_score_means"][name] = {"native": mean(n_vals), "oracle": mean(o_vals)}
        record["group_paired"][name] = {"n_minus_o_mean": mean([n - o for n, o in zip(n_vals, o_vals)]),
                                        "n_minus_o_sum": math.fsum(n - o for n, o in zip(n_vals, o_vals))}
    indexes = [i for i, flag in enumerate(own_equal_mask) if flag]
    record["own_text_equal_paired_sum"] = math.fsum(
        native["per_example"][i] - oracle["per_example"][i] for i in indexes)
    check(abs(record["own_text_equal_paired_sum"]) <= TEXT_EQUAL_TOL,
          f"own text-equal paired delta nonzero {order_name}/{task}")

    if independent is not None:
        eq_own_ind = [own and a == b for own, a, b in
                      zip(own_mask, native["predictions"], independent["predictions"])]
        eq_mis_ind = [mis and a == b for mis, a, b in
                      zip(mis_mask, native["predictions"], independent["predictions"])]
        record["corpus_independent"] = independent["key"]
        record["identity_vs_independent"] = {
            "own_text_eq_independent_rows": int(sum(eq_own_ind)),
            "mis_text_eq_independent_rows": int(sum(eq_mis_ind)),
        }
        n_mean, o_mean, i_mean = (native["per_example_mean"], oracle["per_example_mean"],
                                  independent["per_example_mean"])
        record["t_r_k"] = {
            "native_mean": n_mean, "oracle_mean": o_mean, "independent_mean": i_mean,
            "T": n_mean - i_mean, "R": n_mean - o_mean, "K": o_mean - i_mean,
            "n_minus_o_own_mean": _delta_group(native, oracle, own_mask),
            "n_minus_o_mis_mean": _delta_group(native, oracle, mis_mask),
            "n_minus_i_own_mean": _delta_group(native, independent, own_mask),
            "n_minus_i_mis_mean": _delta_group(native, independent, mis_mask),
            "o_minus_i_own_mean": _delta_group(oracle, independent, own_mask),
            "o_minus_i_mis_mean": _delta_group(oracle, independent, mis_mask)}
        residual = record["t_r_k"]["T"] - (record["t_r_k"]["R"] + record["t_r_k"]["K"])
        check(abs(residual) <= IDENTITY_TOL, f"T != R + K for {order_name}/{task}")
        record["t_r_k"]["identity_residual"] = residual
    return record


def _delta_group(a: dict, b: dict, mask: list) -> float | None:
    indexes = [i for i, flag in enumerate(mask) if flag]
    if not indexes:
        return None
    return mean([a["per_example"][i] - b["per_example"][i] for i in indexes])


def diag_cells(scored: dict) -> list[dict]:
    records = []
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            for p, task in enumerate(order):
                native = scored[("native", bank, task, p, "after")]
                oracle = scored[("oracle", bank, task, p, "after")]
                independent = scored[("independent", independent_name(task, seed), task, 0, "after")]
                record = paired_cell(native, oracle, independent, order_name, task)
                record["seed"] = seed
                record["stage"] = p
                records.append(record)
    return records


def caption_final_cells(scored: dict) -> list[dict]:
    records = []
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            for task in CAPTION_TASKS:
                p = order.index(task)
                native = scored[("native", bank, task, 3, "after")]
                oracle = scored[("oracle", bank, task, 3, "after")]
                record = paired_cell(native, oracle, None, order_name, task)
                record["seed"] = seed
                record["shared_with_diag_corpus"] = (p == 3)
                records.append(record)
    return records


def flickr_pre_cells(scored: dict) -> list[dict]:
    records = []
    for order_name in ORDERS:
        seq = order_name.split("_")
        pos = {task: p for p, task in enumerate(seq)}
        if pos["F"] != 3:
            continue
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            pre = scored[("native", bank, "F", 3, "before")]
            post = scored[("native", bank, "F", 3, "after")]
            oracle_post = scored[("oracle", bank, "F", 3, "after")]
            family_pre = scored[("family_restricted", bank, "F", 3, "before")]
            n = len(pre["per_example"])
            wins = losses = ties = 0
            post_minus_pre = []
            for i in range(n):
                delta = post["per_example"][i] - pre["per_example"][i]
                post_minus_pre.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            record = {
                "order": order_name, "seed": seed, "n_rows": n,
                "pre_native_corpus": corpora_key(pre["entry"]),
                "post_native_corpus": corpora_key(post["entry"]),
                "pre_native_mean": pre["per_example_mean"],
                "post_native_mean": post["per_example_mean"],
                "corpus_delta_native_minus_pre": post["per_example_mean"] - pre["per_example_mean"],
                "corpus_delta_mean_per_image": mean(post_minus_pre),
                "per_image_wins": wins, "per_image_losses": losses, "per_image_ties": ties,
                "pre_expert_counts": dict(Counter(seq[e] if e >= 0 else "unroutable"
                                                  for e in pre["experts"])),
                "post_expert_counts": dict(Counter(seq[e] if e >= 0 else "unroutable"
                                                   for e in post["experts"])),
                "pre_post_expert_transitions": {
                    f"{seq[a]}->{seq[b]}": count for (a, b), count in
                    Counter((a, b) for a, b in zip(pre["experts"], post["experts"])).items()},
                "same_text_pre_post_rows": sum(1 for a, b in zip(pre["predictions"], post["predictions"])
                                               if a == b),
                "oracle_post": {"corpus_mean": oracle_post["per_example_mean"],
                                "native_minus_oracle_rows_mean": mean(
                                    [n - o for n, o in zip(post["per_example"],
                                                           oracle_post["per_example"])])},
                "family_restricted_pre": {
                    "corpus_mean": family_pre["per_example_mean"],
                    "expert_counts": dict(Counter(seq[e] if e >= 0 else "unroutable"
                                                  for e in family_pre["experts"])),
                    "familyC_minus_native_rows_mean": mean(
                        [f - n for f, n in zip(family_pre["per_example"], pre["per_example"])])},
                "task_id_oracle_pre": "missing_by_design (task-ID oracle has no before-stage predictions)",
            }
            records.append(record)
    return records


# ---------------------------------------------------------------- aggregates
def pooled_weighted(diag: list[dict], scored: dict) -> dict:
    """Occurrence-pooled group means per task (rows over 4 orders x 8 seeds x 512)."""
    report = {}
    for task in TASKS:
        own_scores: list[float] = []
        mis_scores: list[float] = []
        o_own: list[float] = []
        o_mis: list[float] = []
        i_own: list[float] = []
        i_mis: list[float] = []
        d_no_own: list[float] = []
        d_no_mis: list[float] = []
        native_all: list[float] = []
        oracle_all: list[float] = []
        indep_all: list[float] = []
        for order_name in ORDERS:
            seq = order_name.split("_")
            for seed in SEEDS:
                bank = bank_name(order_name, seed)
                p = seq.index(task)
                native = scored[("native", bank, task, p, "after")]
                oracle = scored[("oracle", bank, task, p, "after")]
                independent = scored[("independent", independent_name(task, seed), task, 0, "after")]
                n_rows = len(native["per_example"])
                for i in range(n_rows):
                    expert = native["experts"][i]
                    native_all.append(native["per_example"][i])
                    oracle_all.append(oracle["per_example"][i])
                    indep_all.append(independent["per_example"][i])
                    if expert == p:
                        own_scores.append(native["per_example"][i])
                        o_own.append(oracle["per_example"][i])
                        i_own.append(independent["per_example"][i])
                        d_no_own.append(native["per_example"][i] - oracle["per_example"][i])
                    else:
                        mis_scores.append(native["per_example"][i])
                        o_mis.append(oracle["per_example"][i])
                        i_mis.append(independent["per_example"][i])
                        d_no_mis.append(native["per_example"][i] - oracle["per_example"][i])
        report[task] = {
            "rows_per_cell": EVAL_ROWS, "own_occurrence_rows": len(own_scores),
            "mis_occurrence_rows": len(mis_scores),
            "native_own_mean": mean(own_scores) if own_scores else None,
            "native_misrouted_mean": mean(mis_scores) if mis_scores else None,
            "oracle_own_mean": mean(o_own) if o_own else None,
            "oracle_misrouted_mean": mean(o_mis) if o_mis else None,
            "independent_own_mean": mean(i_own) if i_own else None,
            "independent_misrouted_mean": mean(i_mis) if i_mis else None,
            "n_minus_o_own_mean": mean(d_no_own) if d_no_own else None,
            "n_minus_o_misrouted_mean": mean(d_no_mis) if d_no_mis else None,
            "native_all_mean": mean(native_all),
            "oracle_all_mean": mean(oracle_all),
            "independent_all_mean": mean(indep_all),
        }
    return report


def cell_equal_weighted(diag: list[dict]) -> dict:
    """Per (order, seed) equal-weight task means (summary-critical aggregates)."""
    report = {}
    for task in TASKS:
        cells = [cell for cell in diag if cell["task"] == task]
        aggregated = defaultdict(list)
        for cell in cells:
            for label, value in cell["t_r_k"].items():
                if value is not None:
                    aggregated[label].append(value)
        report[task] = {
            "cells": len(cells),
            "rows_per_cell": EVAL_ROWS,
            "cell_equal_means": {label: {"value": mean(values), "n_cells": len(values)}
                                 for label, values in aggregated.items()},
            "own_rows": sum(cell["counts"]["own"] for cell in cells),
            "misrouted_rows": sum(cell["counts"]["misrouted"] for cell in cells),
            "own_text_mismatch_rows": sum(cell["own_route_text_mismatch_rows"] for cell in cells),
        }
    return report


def caption_final_weighted(cap_final: list[dict]) -> dict:
    report = {}
    for task in CAPTION_TASKS:
        cells = [cell for cell in cap_final if cell["task"] == task]
        per_order = {}
        for order_name in ORDERS:
            sub = [cell for cell in cells if cell["order"] == order_name]
            per_order[order_name] = {
                "cells": len(sub), "rows": sum(cell["n_rows"] for cell in sub),
                "own_rows": sum(cell["counts"]["own"] for cell in sub),
                "same_family_other_rows": sum(cell["counts"]["same_family_other"] for cell in sub),
                "cross_family_rows": sum(cell["counts"]["cross_family"] for cell in sub),
                "own_text_mismatch_rows": sum(cell["own_route_text_mismatch_rows"] for cell in sub),
                "per_seed_cells": [{"seed": cell["seed"],
                                    "own_rows": cell["counts"]["own"],
                                    "same_family_other_rows": cell["counts"]["same_family_other"],
                                    "cross_family_rows": cell["counts"]["cross_family"],
                                    "own_text_mismatch_rows": cell["own_route_text_mismatch_rows"],
                                    "group_score_means": cell["group_score_means"],
                                    "group_paired": cell["group_paired"]}
                                   for cell in sub],
            }
        report[task] = {"per_order": per_order,
                        "rows_pooled": {
                            "own_rows": sum(entry["own_rows"] for entry in per_order.values()),
                            "same_family_other_rows": sum(entry["same_family_other_rows"]
                                                          for entry in per_order.values()),
                            "cross_family_rows": sum(entry["cross_family_rows"]
                                                     for entry in per_order.values()),
                            "own_text_mismatch_rows": sum(entry["own_text_mismatch_rows"]
                                                          for entry in per_order.values())}}
    return report


def caption_final_full_subgroups(scored: dict) -> dict:
    """Full own/other absolute scores and paired deltas for every caption final cell."""
    records = []
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        for seed in SEEDS:
            bank = bank_name(order_name, seed)
            for task in CAPTION_TASKS:
                p = order.index(task)
                other = "F" if task == "C" else "C"
                pos_other = order.index(other)
                native = scored[("native", bank, task, 3, "after")]
                oracle = scored[("oracle", bank, task, 3, "after")]
                own_mask = [x == p for x in native["experts"]]
                other_mask = [x != p for x in native["experts"]]
                cross_mask = [x not in (p, pos_other) for x in native["experts"]]
                subgroups = {}
                for name, mask in (("own", own_mask), ("other", other_mask),
                                   ("same_family_other",
                                    [x == pos_other for x in native["experts"]]),
                                   ("cross_family", cross_mask)):
                    indexes = [i for i, flag in enumerate(mask) if flag]
                    if indexes:
                        n_vals = [native["per_example"][i] for i in indexes]
                        o_vals = [oracle["per_example"][i] for i in indexes]
                        subgroups[name] = {
                            "rows": len(indexes),
                            "native_mean": mean(n_vals), "oracle_mean": mean(o_vals),
                            "paired_delta_mean": mean([n - o for n, o in zip(n_vals, o_vals)]),
                            "paired_delta_sum": math.fsum(n - o for n, o in zip(n_vals, o_vals))}
                records.append({"order": order_name, "seed": seed, "task": task,
                                "shared_with_diag_corpus": (p == 3),
                                "subgroups": subgroups,
                                "native_corpus_mean": native["per_example_mean"],
                                "oracle_corpus_mean": oracle["per_example_mean"]})
    return records


def caption_diagnostics(scored: dict) -> dict:
    """Descriptive native final-stage caption associations; 32 cells per task."""
    table = {}
    for task in CAPTION_TASKS:
        flags = ("eos", "empty", "repeated_trigram", "limit_reached")
        pooled_rows = {flag: {"True": [], "False": []} for flag in flags}
        pooled_len: list[float] = []
        length_bucket_rows = {"<=8": [], "9-16": [], "17-32": [], "33-64": []}
        cell_pools: list[dict] = []
        buckets_of = lambda token_count: "<=8" if token_count <= 8 else (
            "9-16" if token_count <= 16 else "17-32" if token_count <= 32 else "33-64")
        for (kind, bank, task_loop, stage, phase), entity in sorted(scored.items()):
            if kind != "native" or task_loop != task or stage != 3 or phase != "after":
                continue
            cell = {"flags": {flag: {"True": [], "False": []} for flag in flags},
                    "lengths": []}
            for i, generation in enumerate(entity["generations"]):
                if not generation:
                    continue
                value = entity["per_example"][i]
                tokens = int(generation.get("generated_tokens", 0) or 0)
                bucket = buckets_of(tokens)
                for flag in flags:
                    key = "True" if bool(generation.get(flag, False)) else "False"
                    pooled_rows[flag][key].append(value)
                    cell["flags"][flag][key].append(value)
                pooled_len.append(tokens)
                length_bucket_rows[bucket].append(value)
                cell["lengths"].append(tokens)
            cell_pools.append(cell)
        summarized = {
            "scope": "native final stage (after index 3), all 32 order/seed cells per task",
            "cellular_lookups": len(cell_pools),
            "occurrence_pooled": {},
            "cell_equal": {},
            "length_buckets_occurrence_pooled": {},
        }
        for flag in flags:
            for key, values in pooled_rows[flag].items():
                if values:
                    summarized["occurrence_pooled"][f"{flag}={key}"] = mean(values)
                per_cell = [mean(cell["flags"][flag][key]) for cell in cell_pools
                            if cell["flags"][flag][key]]
                if per_cell:
                    summarized["cell_equal"][f"{flag}={key}"] = mean(per_cell)
        for bucket, values in length_bucket_rows.items():
            if values:
                summarized["length_buckets_occurrence_pooled"][bucket] = {"rows": len(values),
                                                                          "mean": mean(values)}
        summarized["length_pooled_mean"] = mean(pooled_len) if pooled_len else None
        table[task] = summarized
    return table


# ---------------------------------------------------------------- review cross-checks
def review_partition_checks(diag: list[dict], cap_final: list[dict]) -> dict:
    review = json.loads(ROUTING_REVIEW.read_text()) if ROUTING_REVIEW.exists() else None
    checks = {"review_available": review is not None}
    if review is None:
        return checks
    # VQA diagonal numbers
    vqa_rows = []
    for task in ("V", "G"):
        per_order = {}
        for order_name in ORDERS:
            sub = [cell for cell in diag if cell["task"] == task and cell["order"] == order_name]
            computed = {
                "rows_observed": sum(cell["n_rows"] for cell in sub),
                "own_routed_rows": sum(cell["counts"]["own"] for cell in sub),
                "misrouted_rows": sum(cell["counts"]["misrouted"] for cell in sub),
                "own_route_text_mismatch_rows": sum(cell["own_route_text_mismatch_rows"]
                                                    for cell in sub),
                "own_text_eq_independent_rows": sum(cell["identity_vs_independent"]
                                                    ["own_text_eq_independent_rows"] for cell in sub),
                "mis_text_eq_independent_rows": sum(cell["identity_vs_independent"]
                                                    ["mis_text_eq_independent_rows"] for cell in sub),
                "mis_text_eq_oracle_rows": sum(cell["route_differ_text_equal_rows"] for cell in sub),
            }
            per_order[order_name] = computed
            vqa_rows.append((task, order_name, computed,
                             review["vqa_diagonal_partition"]["per_task"][task]["per_order"][order_name]))
        checks.setdefault("vqa_diagonal", {})[task] = per_order
    drift = []
    for task, order_name, computed, expected in vqa_rows:
        for field, value in expected.items():
            if field not in computed:
                continue
            if isinstance(value, float):
                if computed[field] is None or abs(computed[field] - value) > REVIEW_TOL:
                    drift.append((task, order_name, field, computed[field], value))
            elif computed[field] != value:
                drift.append((task, order_name, field, computed[field], value))
    checks["vqa_diagonal_drift"] = drift
    check(not drift, f"review vqa diagonal count drift: {drift[:4]}")
    # caption final
    cap_rows = []
    for task in CAPTION_TASKS:
        per_order = {}
        for order_name in ORDERS:
            sub = [cell for cell in cap_final if cell["task"] == task and cell["order"] == order_name]
            computed = {
                "rows_observed": sum(cell["n_rows"] for cell in sub),
                "own_routed_rows": sum(cell["counts"]["own"] for cell in sub),
                "same_family_other_rows": sum(cell["counts"]["same_family_other"] for cell in sub),
                "cross_family_rows": sum(cell["counts"]["cross_family"] for cell in sub),
                "own_route_text_mismatch_rows": sum(cell["own_route_text_mismatch_rows"]
                                                    for cell in sub),
                "route_differ_text_eq_oracle_rows": sum(cell["route_differ_text_equal_rows"]
                                                        for cell in sub),
            }
            eligible_cells = [cell for cell in sub if cell["own_route_text_mismatch_rows"] == 0
                              and (cell["counts"]["same_family_other"] + cell["counts"]["cross_family"]) > 0]
            computed["eligible_seed_count"] = len(eligible_cells)
            computed["identified_rows_eligible"] = sum(cell["n_rows"] for cell in eligible_cells)
            computed["identified_mean_paired_delta_eligible_only"] = (mean(
                [cell["group_paired"]["misrouted"]["n_minus_o_mean"] for cell in eligible_cells])
                if eligible_cells else None)
            per_order[order_name] = computed
            cap_rows.append((task, order_name, computed,
                             review["caption_final_stage"]["per_task"][task]["per_order"][order_name]))
        checks.setdefault("caption_final", {})[task] = per_order
    cap_drift = []
    for task, order_name, computed, expected in cap_rows:
        for field, value in expected.items():
            if field not in computed or value is None:
                continue
            if isinstance(value, float):
                if computed[field] is None or abs(computed[field] - value) > REVIEW_TOL:
                    cap_drift.append((task, order_name, field, computed[field], value))
            elif computed[field] != value:
                cap_drift.append((task, order_name, field, computed[field], value))
    checks["caption_final_drift"] = cap_drift
    check(not cap_drift, f"review caption final count drift: {cap_drift[:4]}")
    checks["review_cross_family_total_expected"] = \
        review["caption_final_stage"]["cross_family_rows_total"]
    return checks


def build_mismatch_registry(diag: list[dict], cap_final: list[dict], scored: dict) -> list[dict]:
    entries = []
    for category, cells in (("diag", diag), ("cap_final", cap_final)):
        for cell in cells:
            if not cell["own_route_text_mismatch_rows"]:
                continue
            kind, bank, task_stage, _phase = cell["corpus_native"].split("|", 3)
            task = cell["task"]
            stage = int(task_stage.split("_p")[1].split("_")[0])
            shared = task == "F" and stage == 3
            independent_entity = None
            if cell.get("corpus_independent"):
                i_kind, i_bank, i_task_stage, i_phase = cell["corpus_independent"].split("|", 3)
                independent_entity = scored.get(
                    (i_kind, i_bank, cell["task"], int(i_task_stage.split("_p")[1].split("_")[0]), i_phase))
            native_lookup = scored[(kind, bank, task, stage, "after")]
            oracle_lookup = scored[("oracle", bank, task, stage, "after")]
            for i, row_id, n_pred, o_pred in _mismatch_occurrences(cell, native_lookup, oracle_lookup):
                entries.append({
                    "occurrence_key": f"{bank}|{task}|p{stage:02d}|{row_id}",
                    "order": cell["order"], "seed": cell["seed"], "bank": bank, "task": task,
                    "stage": stage, "category": category,
                    "shared_file_with": ("caption_final" if (category == "diag" and shared)
                                         else "diagonal" if (category == "cap_final" and shared)
                                         else None),
                    "in_batch1_62_scope": (category == "diag" and task in ("V", "G"))
                                          or category == "cap_final",
                    "id": row_id, "native_prediction": n_pred, "oracle_prediction": o_pred,
                    "native_score": native_lookup["per_example"][i],
                    "oracle_score": oracle_lookup["per_example"][i],
                    "independent_score": (independent_entity["per_example"][i]
                                          if independent_entity else None)})
    return entries


def batch1_manifest_crosscheck(registry: list[dict]) -> dict:
    """Read-only cross-check of the 62 shared occurrences against the 1-D manifest."""
    path = ROOT / "artifacts/week1-batch1-cases.json"
    if not path.exists():
        return {"available": False, "reason": "artifacts/week1-batch1-cases.json not present"}
    manifest = json.loads(path.read_text())
    def stage_int(value):
        if isinstance(value, int):
            return value
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else -1
    theirs = {(occ["bank"], occ["task"], stage_int(occ["stage"]), occ["row_id"]): occ
              for occ in manifest["occurrences"]}
    mine = [entry for entry in registry if entry["in_batch1_62_scope"]]
    matched = text_agree = 0
    diffs = []
    for entry in mine:
        key = (entry["bank"], entry["task"], entry["stage"], entry["id"])
        other = theirs.get(key)
        if other is None:
            diffs.append({"occurrence_key": entry["occurrence_key"], "issue": "missing_in_batch1"})
            continue
        matched += 1
        if (entry["native_prediction"] == other["native_prediction"] and
                entry["oracle_prediction"] == other["oracle_prediction"]):
            text_agree += 1
        else:
            diffs.append({"occurrence_key": entry["occurrence_key"], "issue": "text_disagreement"})
    check(matched == 62 and text_agree == 62 and not diffs,
          f"batch1 crosscheck matched={matched} text_agree={text_agree} diffs={diffs[:4]}")
    return {"available": True, "manifest_occurrences": len(manifest["occurrences"]),
            "my_batch1_scope_rows": len(mine), "matched_keys": matched,
            "prediction_text_agreements": text_agree, "diffs": diffs[:8]}


def _mismatch_occurrences(cell: dict, native_lookup: dict, oracle_lookup: dict):
    rows = len(native_lookup["per_example"])
    pos = tuple(cell["order"].split("_")).index(cell["task"])
    for i in range(rows):
        expert = native_lookup["experts"][i]
        if expert == pos and native_lookup["predictions"][i] != oracle_lookup["predictions"][i]:
            yield i, native_lookup["ids"][i], native_lookup["predictions"][i], \
                oracle_lookup["predictions"][i]


def caption_final_pooled(scored: dict) -> dict:
    """Occurrence-weighted own/other absolute scores and paired deltas, all 64 cells."""
    report = {}
    for task in CAPTION_TASKS:
        own_n: list[float] = []
        own_o: list[float] = []
        other_n: list[float] = []
        other_o: list[float] = []
        cross_n: list[float] = []
        per_order = {order_name: {"own": [], "other": []} for order_name in ORDERS}
        for order_name in ORDERS:
            order = tuple(order_name.split("_"))
            pos_other = order.index("F" if task == "C" else "C")
            for seed in SEEDS:
                bank = bank_name(order_name, seed)
                native = scored[("native", bank, task, 3, "after")]
                oracle = scored[("oracle", bank, task, 3, "after")]
                for i, expert in enumerate(native["experts"]):
                    is_own = expert == order.index(task)
                    is_cross = expert not in (order.index(task), pos_other)
                    n_value = native["per_example"][i]
                    o_value = oracle["per_example"][i]
                    if is_own:
                        own_n.append(n_value)
                        own_o.append(o_value)
                        per_order[order_name]["own"].append(n_value - o_value)
                    else:
                        other_n.append(n_value)
                        other_o.append(o_value)
                        per_order[order_name]["other"].append(n_value - o_value)
                    if is_cross:
                        cross_n.append(n_value)
        entries = {
            "own_rows": len(own_n), "other_rows": len(other_n),
            "cross_family_rows": len(cross_n),
            "own_native_mean": mean(own_n) if own_n else None,
            "own_oracle_mean": mean(own_o) if own_o else None,
            "own_paired_delta_mean": mean([n - o for n, o in zip(own_n, own_o)]) if own_n else None,
            "other_native_mean": mean(other_n) if other_n else None,
            "other_oracle_mean": mean(other_o) if other_o else None,
            "other_paired_delta_mean": mean([n - o for n, o in zip(other_n, other_o)]) if other_n else None,
        }
        report[task] = {
            "rows_pooled": entries,
            "per_order_paired_delta_mean": {
                order_name: {
                    "own": mean(values["own"]) if values["own"] else None,
                    "other": mean(values["other"]) if values["other"] else None}
                for order_name, values in per_order.items()},
        }
    return report


# ---------------------------------------------------------------- static safety scan
def static_import_scan() -> dict:
    import ast
    def module_imports(path: str) -> list[str]:
        tree = ast.parse(Path(path).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        return sorted(set(names))
    own = module_imports(str(Path(__file__)))
    metrics = module_imports(str(ROOT / "sema_multimodal/metrics.py"))
    vqa = module_imports(str(ROOT / "sema_vqa/metrics.py"))
    module_names = set(own) | set(metrics) | set(vqa)
    torch_free = not any(name == "torch" or name.startswith("torch.") for name in module_names)
    model_modules_loaded = sorted(name for name in sys.modules
                                  if name.startswith("sema_") and "model" in name)
    check(torch_free, "torch import detected in scorer/reports module imports")
    check(not model_modules_loaded, f"model modules loaded: {model_modules_loaded}")
    return {"own_script_imports": own, "sema_multimodal_metrics_imports": metrics,
            "sema_vqa_metrics_imports": vqa, "torch_free": torch_free,
            "model_modules_loaded": model_modules_loaded}


# ---------------------------------------------------------------- osc report
def write_report(envelope: dict) -> None:
    lines = []
    append = lines.append
    append("# Week-1 1-C 예별(per-example) 재채점 리포트")
    append("")
    append("- 실행(UTC): " + envelope["generated_by"]["generated_at_utc"] +
           " · 스크립트: `" + envelope["generated_by"]["script"] + "` · sha256 " +
           envelope["generated_by"]["script_sha256"][:16] + "…")
    append("- 총 런타임: " + str(envelope["runtimes"]["total_seconds"]) + "초 (계측 단계: " +
           ", ".join(f"{key}={value}" for key, value in envelope["runtimes"].items()
                     if key.endswith("_seconds") and key != "total_seconds") + ")")
    append("- 상태: " + envelope["status"] + " · 실패 레코드 " + str(len(envelope["failures"])) + "건")
    append("- 재채점 corpus " + str(len(envelope["per_corpus"])) + "개 · occurrence rows " +
           str(envelope["rows_written"]) + "개 (artifacts/week1-per-example-scores.jsonl.gz)")
    append("")
    append("## 검증 (strict tolerance)")
    append("")
    append("- per-image 평균 == fresh TaskScorer.score == 저장값: 최대 오차 "
           "max |fresh−stored| = " + f"{envelope['stored_parity_maxima']['fresh_vs_stored_max']:.3e}"
           f", 저장 복사본 간 최대 = {envelope['stored_parity_maxima']['stored_mirror_max']:.3e} (허용 1e-12/1e-9)")
    append("- PTB 토큰 캐시: 요청된 모든 caption corpus에서 배치 토큰과 라인 단위 일치, 불일치 0건")
    append("- G5 native 파일 ↔ G4 native 파일: " +
           str(envelope["native_parity_g5_vs_g4"]["files_row_identical"]) + "/" +
           str(envelope["native_parity_g5_vs_g4"]["files_checked"]) + "개 파일 행 단위 동일")
    append("- scorer lock: 구현 SHA·문서빈도 SHA 포함 필드 전부 원 lock/런 구성과 일치: " +
           str(sorted(envelope["scorer_locks_parity"])))
    append("- 메모: occurrence는 4 orders × 8 seeds × 512 eval images의 재평가 건수이며 서로 다른 이미지 수가 아님; 통계 검정·방법 선택·인과 주장 없음")
    append("")
    append("## 불일치(own-route text mismatch) 정합")
    append("")
    totals = envelope["mismatch_totals"]
    append(f"- VQA diagonal {totals['vqa_diagonal']}/21, caption final {totals['caption_final']}/41"
           f" (review 기록과 정합; 배치-1 스코프 62 = 21 + 41)")
    append(f"- caption diagonal 총 {totals['caption_diagonal']} rows: 그중 "
           f"{totals['caption_diagonal_shared_with_final']} rows는 F@stage3 셀로 caption final corpus와 "
           f"동일 파일·동일 occurrence(역할 중복, 계산 1회 재사용), 나머지 "
           f"{totals['caption_diagonal_additional_beyond_62']} rows는 62 범위 밖의 diagonal-stage 추가 불일치")
    append(f"- 고유 occurrence 합계(역할 중복 제외): {totals['unique_occurrences']}")
    append("")
    append("## 주요 집단 절대값 (occurrence pooled, 원 단위)")
    append("")
    units = envelope["task_units"]
    for task, values in envelope["diag_occurrence_weighted_task"].items():
        append(f"- {task} [{units[task]}]: diag native mean {values['native_all_mean']:.6f}, "
               f"oracle {values['oracle_all_mean']:.6f}, independent {values['independent_all_mean']:.6f}; "
               f"own_rows {values['own_occurrence_rows']}, mis_rows {values['mis_occurrence_rows']}; "
               f"N−O own {values['n_minus_o_own_mean']}, mis {values['n_minus_o_misrouted_mean']}")
    append("")
    append("## caption final (native vs task-ID oracle, 자기/타집단)")
    append("")
    for task, table in envelope["caption_final_weighted"].items():
        rows = table["rows_pooled"]
        append(f"- {task} [{units[task]}]: own {rows['own_rows']} rows, other {rows['same_family_other_rows']}"
               f" + cross {rows['cross_family_rows']}; own-route text mismatch {rows['own_text_mismatch_rows']}/41."
               "셀별 own/other 절대값·paired Δ는 envelope의 caption_final_full_subgroups 참조")
    append("")
    append("## Flickr later-task 16 cells (before vs after)")
    append("")
    wins = sum(cell["per_image_wins"] for cell in envelope["flickr_pre_post_cells"])
    losses = sum(cell["per_image_losses"] for cell in envelope["flickr_pre_post_cells"])
    ties = sum(cell["per_image_ties"] for cell in envelope["flickr_pre_post_cells"])
    corpus_delta = mean([cell["corpus_delta_native_minus_pre"] for cell in envelope["flickr_pre_post_cells"]])
    append(f"- per-image: wins {wins}, losses {losses}, ties {ties} (post−pre, occurrence pooled); "
           f"corpus mean post−pre {corpus_delta:.6f}")
    append("- task-ID oracle의 before-stage 예측은 저장 산출물에 없음(by construction) — 명시적 결측; "
           "family-restricted 사전행이 유일하게 저장된 사전-상태 caption-family 출력 (셀별 "
           "family_restricted_pre 전사/사후 전이·Δ는 envelope 참조)")
    append("")
    append("## 한계")
    append("")
    for limit in envelope["limitations"]:
        append("- " + limit)
    append("")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> int:
    for path in (OUT_JSON, OUT_ROWS, OUT_REPORT):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    total_started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    TIMING["wall_start"] = round(total_started, 3)

    inputs = []
    for path in (G4_TAR, G5_TAR, MANIFESTS_TAR, AUDIT_G4, AUDIT_G5, NORMALIZERS,
                 CONFIRMATION_LOCK, ROUTING_REVIEW, ROOT / "sema_multimodal/metrics.py",
                 ROOT / "sema_vqa/metrics.py"):
        if path.exists():
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size})

    global TASK_VIEWS
    TIMING["phase"] = "load"
    ARCHIVES.preload()
    task_views, configs = load_manifest_and_configs()
    globals()["TASK_VIEWS"] = task_views
    for task_id, view in task_views.items():
        EVAL_ID_LISTS[task_id] = [row["id"] for row in view["eval"]]
        EVAL_ROW_MAPS[task_id] = {row["id"]: row for row in view["eval"]}
    for order_name in ORDERS:
        for seed in SEEDS:
            BANK_ORDERS[bank_name(order_name, seed)] = tuple(order_name.split("_"))
    check(all(len(ids) == EVAL_ROWS for ids in EVAL_ID_LISTS.values()),
          "manifest eval sizes must be exactly 512 per task")

    TIMING["phase"] = "enumerate"
    entries = enumerate_corpora()
    coverage = {"unique_corpora": len(entries),
                "by_kind": dict(Counter(entry["kind"] for entry in entries)),
                "by_role": dict(Counter(role for entry in entries for role in entry["roles"]))}
    check(coverage["by_kind"].get("native", 0) == 192,
          f"native corpus count {coverage['by_kind'].get('native')}")
    check(coverage["by_kind"].get("oracle", 0) == 176,
          f"oracle corpus count {coverage['by_kind'].get('oracle')}")
    check(coverage["by_kind"].get("independent", 0) == 32, "independent count mismatch")
    check(coverage["by_kind"].get("family_restricted", 0) == 16, "family-restricted count mismatch")
    check(coverage["by_role"].get("diag", 0) == 256, "diag role count mismatch")
    check(coverage["by_role"].get("cap_final", 0) == 128, "cap_final role count mismatch")
    check(coverage["by_role"].get("flickr_pre", 0) == 32, "flickr_pre role count mismatch")

    TIMING["phase"] = "tokenize"
    build_token_cache(collect_caption_strings(entries))
    TIMING["phase"] = "scorers"
    build_scorers()
    lock_report = verify_locks(configs, json.loads(CONFIRMATION_LOCK.read_text()))
    TIMING["phase"] = "stored_sources"
    stored, _transfers = stored_corpus_scores()

    TIMING["phase"] = "score"
    started = time.perf_counter()
    scored: dict[tuple, dict] = {}
    for entry in entries:
        entity = score_corpus(entry)
        scored[(entry["kind"], entry["bank"], entry["task"], entry["stage"], entry["phase"])] = entity
    TIMING["scoring_loop_seconds"] = round(time.perf_counter() - started, 3)

    TIMING["phase"] = "parity"
    native_parity = g5_native_parity(entries)
    stored_report = stored_parity(scored, stored)

    TIMING["phase"] = "aggregate"
    diag = diag_cells(scored)
    cap_final = caption_final_cells(scored)
    flickr = flickr_pre_cells(scored)
    occurrence_pooled = pooled_weighted(diag, scored)
    cell_equal = cell_equal_weighted(diag)
    cap_weighted = caption_final_weighted(cap_final)
    cap_full = caption_final_full_subgroups(scored)

    diag_vqa_mismatch_rows = sum(cell["own_route_text_mismatch_rows"] for cell in diag
                                 if cell["task"] in ("V", "G"))
    diag_caption_cells = [cell for cell in diag if cell["task"] in CAPTION_TASKS]
    diag_caption_mismatch_rows = sum(cell["own_route_text_mismatch_rows"] for cell in diag_caption_cells)
    diag_caption_shared_rows = sum(cell["own_route_text_mismatch_rows"] for cell in diag_caption_cells
                                   if cell["task"] == "F" and cell["stage"] == 3)
    cap_final_mismatch_rows = sum(cell["own_route_text_mismatch_rows"] for cell in cap_final)
    mismatch_totals = {
        "vqa_diagonal": diag_vqa_mismatch_rows,
        "caption_diagonal": diag_caption_mismatch_rows,
        "caption_diagonal_shared_with_final": diag_caption_shared_rows,
        "caption_diagonal_additional_beyond_62": diag_caption_mismatch_rows - diag_caption_shared_rows,
        "caption_final": cap_final_mismatch_rows,
        "unique_occurrences": diag_vqa_mismatch_rows + cap_final_mismatch_rows +
                              (diag_caption_mismatch_rows - diag_caption_shared_rows),
        "batch1_scope_vqa_diag_plus_caption_final": diag_vqa_mismatch_rows + cap_final_mismatch_rows,
        "note": "the recorded 62 = VQA diagonal (21) + caption final (41); the caption diagonal "
                "corpus for F at stage 3 IS the caption final corpus, so those mismatch rows are "
                "shared occurrences (one deterministic compute, two roles), and the remaining "
                "caption diagonal rows are additional diagonal-stage mismatches outside the 62",
    }
    check(diag_vqa_mismatch_rows == 21,
          f"VQA diag own-route text mismatches expected 21, got {diag_vqa_mismatch_rows}")
    check(cap_final_mismatch_rows == 41,
          f"caption final own-route text mismatches expected 41, got {cap_final_mismatch_rows}")

    review_checks = review_partition_checks(diag, cap_final)
    registry = build_mismatch_registry(diag, cap_final, scored)
    batch1_check = batch1_manifest_crosscheck(registry)
    diagnostics = caption_diagnostics(scored)

    TIMING["phase"] = "rows_write"
    rows_started = time.perf_counter()
    row_count = 0
    with OUT_ROWS.open("xb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as handle:
            for (kind, bank, task, stage, phase), entity in sorted(scored.items()):
                entry = entity["entry"]
                if kind == "independent":
                    order_name = None
                    pos = None
                else:
                    order_name = BANK_ORDERS[bank]
                    pos = order_name.index(task)
                for i, (row_id, prediction, expert, score) in enumerate(
                        zip(entity["ids"], entity["predictions"], entity["experts"],
                            entity["per_example"])):
                    if kind == "independent":
                        cell = "independent"
                        expert_name = "single_expert"
                    else:
                        cell = "flickr_pre" if phase == "before" else (
                            "diag" if "diag" in entry["roles"] else "cap_final")
                        expert_name = order_name[expert] if expert >= 0 else "unroutable"
                    if kind == "native":
                        if phase == "before":
                            own = None
                            label = "pre_state"
                        else:
                            own = bool(expert == pos)
                            label = "own" if own else "misrouted"
                    elif kind == "oracle":
                        own = bool(expert == pos)
                        label = "forced_own" if own else "oracle_misrouted"
                    elif kind == "family_restricted":
                        own = None
                        label = expert_name if expert >= 0 else "unroutable"
                    else:
                        own = None
                        label = "independent"
                    record = {"cell": cell, "order": order_name, "seed": seed_of(bank),
                              "task": task, "stage": stage, "phase": phase, "kind": kind,
                              "roles": entry["roles"], "row_index": i, "id": row_id,
                              "expert": expert, "expert_name": expert_name, "own": own,
                              "label": label, "prediction": prediction, "score": score,
                              "generation": entity["generations"][i]}
                    handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                    row_count += 1
    TIMING["rows_write_seconds"] = round(time.perf_counter() - rows_started, 3)
    check(row_count == len(scored) * EVAL_ROWS, f"rows written {row_count} != corpora*512")

    envelope = {
        "schema_version": 1,
        "generated_by": {"script": "scripts/week1_per_example_rescore.py",
                         "script_sha256": sha256_file(Path(__file__)),
                         "generated_at_utc": run_started_utc,
                         "argument": "single-process CPU-only rescoring of saved predictions"},
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                        "numpy": importlib.metadata.version("numpy"),
                        "pycocoevalcap": importlib.metadata.version("pycocoevalcap"),
                        "cpu_count": os.cpu_count() or 1},
        "static_import_scan": static_import_scan(),
        "runtimes": dict(TIMING, total_seconds=round(time.perf_counter() - total_started, 3)),
        "inputs": inputs,
        "manifest_sha256": MANIFEST_SHA256,
        "grid_coverage": coverage,
        "scorer_locks_parity": lock_report,
        "task_units": {task: lock_report[task]["units"] for task in TASKS},
        "native_parity_g5_vs_g4": native_parity,
        "stored_parity_maxima": stored_report["maxima"],
        "per_corpus": stored_report["per_corpus"],
        "diag_cells": diag,
        "caption_final_cells": cap_final,
        "flickr_pre_post_cells": flickr,
        "diag_occurrence_weighted_task": occurrence_pooled,
        "diag_cell_equal_weighted_task": cell_equal,
        "caption_final_weighted": cap_weighted,
        "caption_final_full_subgroups": cap_full,
        "caption_final_pooled_subgroups": caption_final_pooled(scored),
        "caption_diagnostic_associations": diagnostics,
        "mismatch_totals": mismatch_totals,
        "mismatch_registry": registry,
        "review_cross_checks": review_checks,
        "batch1_manifest_crosscheck": batch1_check,
        "rows_written": row_count,
        "output_rows_sha256": sha256_file(OUT_ROWS),
        "failures": failures,
        "status": "pass" if not failures else "fail",
        "limitations": [
            "rows are evaluation occurrences (4 orders x 8 seeds x 512 eval images per stage); the same image recurs per order and seed, so rows are not independent images",
            "the G5 task-ID oracle has no before-stage prediction files by construction, so the pre-Flickr paired oracle evidence is reported missing and never imputed; the saved caption-family restricted pre rows are the only stored pre-state caption-family outputs",
            "no significance test, no method selection, no causal claim; all deltas are descriptive",
            "original source hashes and scorer locks must match; the artifacts record the strict tolerances and maxima",
            "caption scoring uses the frozen train references IDF only; no evaluation-reference IDF enters the scorer path",
        ],
    }
    OUT_JSON.write_text(json.dumps(envelope, ensure_ascii=False, indent=1, sort_keys=False))
    write_report(envelope)
    TIMING["total_seconds"] = round(time.perf_counter() - total_started, 3)
    print(f"status={envelope['status']} corpora={len(entries)} rows={row_count} "
          f"max_fresh_vs_stored={stored_report['maxima']['fresh_vs_stored_max']:.3e} "
          f"wall={TIMING['total_seconds']}s failures={len(failures)}")
    for message in failures[:20]:
        print("FAIL:", message)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
