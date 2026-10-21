"""Week-1 slice 1-D: batch-size-1 regeneration of the 62 own-route native/oracle
text mismatches identified by the shipped G4/G5 routing-review artifacts.

Scope (locked by the parent ticket; no method/scorer/config changes, no training):
exactly the shipped occurrences are re-run. They are

  * 21 VQA diagonal own-route string mismatches (VQAv2 1 + GQA 20): rows whose
    saved native expert equals the task's own expert index at the task's
    diagonal stage (G4 training-time native evaluation, row-identical to the
    G5 native_joint replay) whose prediction differs from the same row's G5
    ``task_id_oracle`` prediction, and
  * 41 caption final own-route native/oracle text mismatches (COCO 14 +
    Flickr 27): rows of ``G5 native_joint`` at the final stage (after_03)
    whose native expert is the gold task expert and whose native prediction
    differs from the G5 ``task_id_oracle`` prediction of the same row.

Both saved variants of every occurrence already used the SAME expert weights
(the own task expert), so batch-size-1 regeneration shares one deterministic
compute per (bank, task, stage, row) and maps every occurrence onto it. Each
regenerated answer is compared literally to the saved native text and the
saved oracle text (native / oracle / both / neither), and the saved per-row
generation diagnostics are compared field by field. No router is re-fitted,
no scorer runs, no bank expert updates, and no prediction outside the 62
occurrences is produced (parent-instructed deterministic compute reuse applies
when one compute would otherwise be duplicated per occurrence).

Subcommands
-----------
``cases``       CPU, read-only on the inputs: build a self-binding case manifest
                from the shipped G4/G5 raw tars, the manifest tar, the
                confirmation checkpoint index and the shipped routing-review
                partition artifact; exclusive-create one JSON.
``regenerate``  CUDA, exclusive-create outputs: verify every recorded input hash
                against the current files, load each affected saved bank
                checkpoint (shared bank load, backbone assets loaded once),
                regenerate exactly the 62 rows at micro_batch_size=1 with the
                saved shared expert forced, and write ``predictions.jsonl`` plus
                a run report; no evaluation outside the 62 rows.
``merge``       local: attach scheduler evidence (sacct), re-verify the fetched
                predictions/report and inputs, and exclusive-create the final
                local report.

Immutability: input artifacts are only read and re-hashed before and after
every run; every created file refuses to overwrite an existing one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT_DEFAULT = Path(__file__).resolve().parents[1]

ORDERS = ("V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V")
SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
DIAG_TASKS = ("V", "G")
CAPTION_TASKS = ("C", "F")
FAMILY_OF = {"V": "vqa", "G": "vqa", "C": "caption", "F": "caption"}
EVAL_ROWS = 512
ORIGINAL_MICRO_BATCH = 8
GEN_KEYS = ("generated_tokens", "max_new_tokens", "eos", "empty", "repeated_trigram", "limit_reached")

G4_TAR = "artifacts/multimodal-g4-raw.tar.gz"
G5_TAR = "artifacts/multimodal-g5-raw.tar.gz"
G4_ROOT = "artifacts/multimodal_runs/g4"
G5_ROOT = "artifacts/multimodal_runs/routing_g4"
MANIFESTS_TAR = "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
CHECKPOINT_INDEX = "artifacts/multimodal-confirmation-checkpoints.json"
PARTITION_JSON = "artifacts/routing-review-routing-partition.json"
CODE_CONDITIONS_JSON = "artifacts/routing-review-code-conditions.json"
G5_SUMMARY_JSON = "artifacts/multimodal-g5-summary.json"

SCHEMA = 1
TOOL = "scripts/week1_batch1_regenerate.py"


def bank_name(order_name: str, seed: int) -> str:
    return f"{order_name}_expand_lr0.0003_ep5_seed{seed}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(data: bytes) -> list[dict]:
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    for row in rows:
        if (not isinstance(row, dict) or not {"id", "prediction", "expert", "generation"} <= set(row)
                or not isinstance(row["prediction"], str)
                or not isinstance(row["generation"], dict)
                or tuple(sorted(row["generation"])) != tuple(sorted(GEN_KEYS))):
            raise ValueError("Malformed saved prediction record")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Saved prediction file contains duplicate ids")
    return rows


class TarLoader:
    """One-pass preload of named members from a gzip tar (read-only)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._bytes: dict[str, bytes] = {}

    def want(self, members) -> None:
        for member in members:
            self._bytes[member] = None

    @property
    def pending(self) -> list[str]:
        return [name for name, value in self._bytes.items() if value is None]

    def preload(self) -> None:
        pending = set(self.pending)
        with tarfile.open(self.path, "r:gz") as tar:
            for member in tar:
                name = member.name
                if name in pending:
                    if not member.isreg() or member.size == 0:
                        raise ValueError(f"Unexpected member {name} in {self.path}")
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable member {name} in {self.path}")
                    self._bytes[name] = stream.read()
                    pending.discard(name)
            if pending:
                raise ValueError(f"Missing tar members in {self.path}: {sorted(pending)[:8]}")

    def read(self, member: str) -> bytes:
        data = self._bytes.get(member)
        if data is None:
            raise ValueError(f"Tar member not preloaded: {member}")
        return data


def exclusive_create_text(path: Path, payload: str) -> None:
    """Exclusive create of the final file; never overwrites anything present."""
    final = Path(path)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing week-1 output: {final}")
    temporary = final.with_name(final.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary sheet already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as descriptor:
        descriptor.write(payload)
        descriptor.flush()
    temporary.replace(final)


def row_equal(left: dict, right: dict) -> bool:
    return (left["id"] == right["id"] and left["prediction"] == right["prediction"]
            and left["expert"] == right["expert"] and left["generation"] == right["generation"])


def original_batching(rows: list[dict], index: int, kind: str) -> dict:
    """Reconstruct the original run's micro-batch composition around one row.

    ``native`` mirrors runner.evaluate's grouped generation: rows are visited
    in chunks of micro_batch_size=8 within the row's expert group (expert
    groups in sorted order). ``oracle`` mirrors the forced single-expert pass:
    all rows in file order chunked into consecutive groups of 8. Recorded as
    original-run diagnostics; the batch-1 pass regenerates every row alone.
    """
    if kind == "oracle":
        start = ORIGINAL_MICRO_BATCH * (index // ORIGINAL_MICRO_BATCH)
        chunk = rows[start:start + ORIGINAL_MICRO_BATCH]
        return {"row_offset_in_pass": index,
                "micro_index_in_pass": index // ORIGINAL_MICRO_BATCH,
                "row_ids_in_micro_batch": [row["id"] for row in chunk]}
    expert = rows[index]["expert"]
    positions = [position for position, row in enumerate(rows) if row["expert"] == expert]
    position_in_group = positions.index(index)
    start = ORIGINAL_MICRO_BATCH * (position_in_group // ORIGINAL_MICRO_BATCH)
    chunk_positions = positions[start:start + ORIGINAL_MICRO_BATCH]
    return {"expert_group_size": len(positions),
            "micro_index_within_group": position_in_group // ORIGINAL_MICRO_BATCH,
            "row_offset_within_group": position_in_group % ORIGINAL_MICRO_BATCH,
            "row_ids_in_micro_batch": [rows[position]["id"] for position in chunk_positions]}


def _occurrence(occurrence_id, set_name, task_id, order_name, bank, seed, index, position,
                rows_native, rows_oracle, manifest_row, source_info, stage) -> dict:
    row, oracle = rows_native[index], rows_oracle[index]
    if row["id"] != oracle["id"]:
        raise ValueError(f"Saved native/oracle id alignment broke at {bank} {task_id} index {index}")
    if row["expert"] != position or oracle["expert"] != position:
        raise ValueError(f"Between-variant expert identity broken at {bank} {task_id} index {index}")
    source = dict(source_info)
    if set_name == "caption_final":
        source["g4_final_text_equal"] = source_info["g4_final_text_equal"]
        source["g4_final_expert_equal"] = source_info["g4_final_expert_equal"]
    return {
        "occurrence_id": occurrence_id,
        "set": set_name, "family": FAMILY_OF[task_id], "task": task_id,
        "bank": bank, "order_name": order_name, "order": order_name.split("_"), "seed": seed,
        "stream_position": position, "stage": stage,
        "stage_note": ("stream_position is the task's own diagonal position (the shared own-expert "
                       "index); stage is the evaluation stage of the saved files compared: "
                       "diagonal own-route for the VQA set, final stage (after_03) for caption"),
        "shared_expert": position, "gold_expert": position,
        "native_expert": row["expert"], "oracle_expert": oracle["expert"],
        "row_id": row["id"], "image_sha256": manifest_row["image_sha256"],
        "image": manifest_row["image"], "input_text": manifest_row["input_text"],
        "native_prediction": row["prediction"], "oracle_prediction": oracle["prediction"],
        "native_generation": dict(row["generation"]), "oracle_generation": dict(oracle["generation"]),
        "original_micro_batch_size": ORIGINAL_MICRO_BATCH,
        "original_batching": {
            "native": original_batching(rows_native, index, "native"),
            "oracle": original_batching(rows_oracle, index, "oracle"),
            "note": "the original native pass chunks rows per expert group in chunks of 8; "
                    "the forced oracle pass visits all 512 rows in file order in chunks of 8; "
                    "batch-1 regeneration runs the same weights on the same row alone",
        },
        "source_members": source,
        "compute_key": f"{bank}|{task_id}|{stage}|{row['id']}",
    }


def command_cases(root: Path, output: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sema_multimodal.data import validate_manifest

    started = time.perf_counter()
    root = root.resolve()
    input_paths = {
        G4_TAR: root / G4_TAR,
        G5_TAR: root / G5_TAR,
        MANIFESTS_TAR: root / MANIFESTS_TAR,
        CHECKPOINT_INDEX: root / CHECKPOINT_INDEX,
        PARTITION_JSON: root / PARTITION_JSON,
        CODE_CONDITIONS_JSON: root / CODE_CONDITIONS_JSON,
        G5_SUMMARY_JSON: root / G5_SUMMARY_JSON,
    }
    inputs_initial = {name: digest_file(path) for name, path in input_paths.items()}

    partition = json.loads((root / PARTITION_JSON).read_text())
    if partition.get("schema_version") != 1 or partition.get("status") != "pass":
        raise ValueError("Shipped routing-review partition artifact must be a passing revision")
    diag_partition = partition["vqa_diagonal_partition"]["per_task"]
    caption_partition = partition["caption_final_stage"]["per_task"]
    numerics = partition["caption_final_stage"]["own_route_generation_numerics"]
    shipped_diag_counts = {f"{task}/{order}": cell["own_route_text_mismatch_rows"]
                           for task in DIAG_TASKS for order, cell in diag_partition[task]["per_order"].items()}
    shipped_diag_own = {f"{task}/{order}": cell["own_routed_rows"]
                        for task in DIAG_TASKS for order, cell in diag_partition[task]["per_order"].items()}
    shipped_final_counts = {f"{task}/{order}": cell["own_route_text_mismatch_rows"]
                            for task in CAPTION_TASKS for order, cell in caption_partition[task]["per_order"].items()}
    shipped_final_own = {f"{task}/{order}": cell["own_routed_rows"]
                         for task in CAPTION_TASKS for order, cell in caption_partition[task]["per_order"].items()}
    shipped_final_seed_counts = {}
    shipped_final_seed_own = {}
    for task in CAPTION_TASKS:
        for order, cell in caption_partition[task]["per_order"].items():
            for entry in cell["per_seed"]:
                key = f"{task}/{order}/{entry['seed']}"
                shipped_final_seed_counts[key] = entry["own_route_text_mismatched_vs_oracle"]
                shipped_final_seed_own[key] = entry["own_rows"]

    g4, g5 = TarLoader(root / G4_TAR), TarLoader(root / G5_TAR)
    for order_name in ORDERS:
        for seed in SEEDS:
            name = bank_name(order_name, seed)
            positions = {task_id: index for index, task_id in enumerate(order_name.split("_"))}
            g4.want([f"{G4_ROOT}/{name}/config.json"])
            for task_id in DIAG_TASKS:
                position = positions[task_id]
                g4.want([f"{G4_ROOT}/{name}/predictions_after_{position:02d}_{task_id}.jsonl"])
                g5.want([f"{G5_ROOT}/{name}/native_joint/predictions_after_{position:02d}_{task_id}.jsonl",
                         f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{position:02d}_{task_id}.jsonl"])
            for task_id in CAPTION_TASKS:
                position = positions[task_id]
                g5.want([f"{G5_ROOT}/{name}/native_joint/predictions_after_{position:02d}_{task_id}.jsonl",
                         f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{position:02d}_{task_id}.jsonl"])
                g4.want([f"{G4_ROOT}/{name}/predictions_after_03_{task_id}.jsonl"])
                g5.want([f"{G5_ROOT}/{name}/native_joint/predictions_after_03_{task_id}.jsonl",
                         f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_03_{task_id}.jsonl"])
    g4.preload()
    g5.preload()
    files_record: dict[str, str] = {}

    def recordread(loader, member) -> bytes:
        data = loader.read(member)
        files_record.setdefault(member, sha256_bytes(data))
        return data

    manifest_loader = TarLoader(root / MANIFESTS_TAR)
    manifest_loader.want([MANIFEST_MEMBER])
    manifest_loader.preload()
    manifest_bytes = manifest_loader.read(MANIFEST_MEMBER)
    manifest_sha = sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)  # structure-only locally; the configured image_root is byte-verified on CUDA
    eval_rows = {task["task_id"]: task["eval"] for task in manifest["tasks"]}
    for task_id, rows in eval_rows.items():
        if len(rows) != EVAL_ROWS:
            raise ValueError(f"Manifest eval corpus is not {EVAL_ROWS} rows: {task_id}")
    manifest_rows = {task_id: {row["id"]: row for row in rows} for task_id, rows in eval_rows.items()}

    checkpoints = json.loads((root / CHECKPOINT_INDEX).read_text())["files"]

    CONFIG_COMMON_FIELDS = ("method", "eval_split", "lr", "epochs", "rank", "alpha", "batch_size",
                            "micro_batch_size", "image_size", "device", "router_modality", "router_memory",
                            "router_ridge", "replay_per_task", "manifest", "assets", "image_root",
                            "manifest_sha256", "assets_sha256", "tokenizer_sha256", "source_sha256",
                            "schema_version", "protocol", "oracle_diagnostic", "image_shuffled_diagnostic",
                            "smoke_output", "train_limit", "eval_limit", "generation", "optimizer", "loss",
                            "torch", "numpy", "cuda", "deterministic_algorithms", "tf32")

    banks = {}
    diag_mismatch_per = Counter()
    diag_own_per = Counter()
    caption_diag_mismatch_rows = 0
    caption_diag_own_rows = 0
    final_mismatch_per = Counter()
    final_own_per = Counter()
    per_seed_mine = Counter()
    final_seed_own_mine = Counter()
    occurrences: list[dict] = []

    for order_name in ORDERS:
        for seed in SEEDS:
            name = bank_name(order_name, seed)
            order = order_name.split("_")
            positions = {task_id: index for index, task_id in enumerate(order)}
            config_member = f"{G4_ROOT}/{name}/config.json"
            config_raw = recordread(g4, config_member)
            config = json.loads(config_raw)
            if config.get("schema_version") != 1 or config.get("method") != "expand":
                raise ValueError(f"Bank {name} config is not a schema-1 expansion parent")
            if config.get("task_ids") != order or config.get("task_order") != order:
                raise ValueError(f"Bank {name} order mismatch")
            missing = [field for field in CONFIG_COMMON_FIELDS if field not in config]
            if missing:
                raise ValueError(f"Bank {name} config lacks {missing}")
            if config["micro_batch_size"] != ORIGINAL_MICRO_BATCH:
                raise ValueError(f"Bank {name} original micro_batch_size is not {ORIGINAL_MICRO_BATCH}")
            latest_key = f"{G4_ROOT}/{name}/latest.pt"
            if latest_key not in checkpoints:
                raise ValueError(f"Checkpoint index does not cover {latest_key}")
            banks[name] = {
                "bank": name, "order_name": order_name, "order": order, "seed": seed,
                "config_sha256": sha256_bytes(config_raw),
                "config_common": {field: config[field] for field in CONFIG_COMMON_FIELDS},
                "latest_sha256": checkpoints[latest_key]["sha256"],
                "latest_bytes": checkpoints[latest_key]["bytes"],
                "manifest_sha256": config["manifest_sha256"],
            }
            if banks[name]["manifest_sha256"] != manifest_sha:
                raise ValueError(f"Bank {name} manifest differs from the shipped manifest")

            for task_id in DIAG_TASKS:
                position = positions[task_id]
                native_raw = recordread(g4, f"{G4_ROOT}/{name}/predictions_after_{position:02d}_{task_id}.jsonl")
                replay_raw = recordread(g5, f"{G5_ROOT}/{name}/native_joint/predictions_after_{position:02d}_{task_id}.jsonl")
                oracle_raw = recordread(g5, f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{position:02d}_{task_id}.jsonl")
                rows_native, rows_replay, rows_oracle = jsonl(native_raw), jsonl(replay_raw), jsonl(oracle_raw)
                if (len(rows_native) != EVAL_ROWS or len(rows_replay) != EVAL_ROWS or len(rows_oracle) != EVAL_ROWS
                        or not all(row_equal(a, b) for a, b in zip(rows_native, rows_replay))
                        or [row["id"] for row in rows_native] != [row["id"] for row in rows_oracle]):
                    raise ValueError(f"Diagonal replay/identity mismatch: {name} {task_id}")
                if any(row["expert"] != position for row in rows_oracle):
                    raise ValueError(f"Diagonal oracle does not force the gold expert: {name} {task_id}")
                diag_own_per[f"{task_id}/{order_name}"] += sum(1 for row in rows_native if row["expert"] == position)
                for index, (row, oracle) in enumerate(zip(rows_native, rows_oracle)):
                    if row["expert"] == position and row["prediction"] != oracle["prediction"]:
                        manifest_row = manifest_rows[task_id][row["id"]]
                        diag_mismatch_per[f"{task_id}/{order_name}"] += 1
                        occurrences.append(_occurrence(
                            "", "vqa_diagonal", task_id, order_name, name, seed, index, position,
                            rows_native, rows_oracle, manifest_row,
                            {"g4_native": f"{G4_ROOT}/{name}/predictions_after_{position:02d}_{task_id}.jsonl",
                             "g5_native_equal_to_g4_rows": True,
                             "g5_oracle": f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{position:02d}_{task_id}.jsonl"},
                            stage=f"after_{position:02d}"))

            for task_id in CAPTION_TASKS:
                position = positions[task_id]
                diag_native_raw = recordread(g5, f"{G5_ROOT}/{name}/native_joint/predictions_after_{position:02d}_{task_id}.jsonl")
                diag_oracle_raw = recordread(g5, f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{position:02d}_{task_id}.jsonl")
                rows_diag_native, rows_diag_oracle = jsonl(diag_native_raw), jsonl(diag_oracle_raw)
                if any(row["expert"] != position for row in rows_diag_oracle):
                    raise ValueError(f"Caption diagonal oracle does not force gold: {name} {task_id}")
                caption_diag_own_rows += sum(1 for row in rows_diag_native if row["expert"] == position)
                caption_diag_mismatch_rows += sum(
                    1 for row, oracle in zip(rows_diag_native, rows_diag_oracle)
                    if row["expert"] == position and row["prediction"] != oracle["prediction"])

                final_native_raw = recordread(g5, f"{G5_ROOT}/{name}/native_joint/predictions_after_03_{task_id}.jsonl")
                final_oracle_raw = recordread(g5, f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_03_{task_id}.jsonl")
                g4_final_raw = recordread(g4, f"{G4_ROOT}/{name}/predictions_after_03_{task_id}.jsonl")
                rows_native, rows_oracle, rows_g4_final = jsonl(final_native_raw), jsonl(final_oracle_raw), jsonl(g4_final_raw)
                if (len(rows_native) != EVAL_ROWS or len(rows_oracle) != EVAL_ROWS or len(rows_g4_final) != EVAL_ROWS
                        or [row["id"] for row in rows_native] != [row["id"] for row in rows_oracle]):
                    raise ValueError(f"Caption final pairing differs: {name} {task_id}")
                if any(row["expert"] != position for row in rows_oracle):
                    raise ValueError(f"Caption final oracle does not force gold: {name} {task_id}")
                own_count = sum(1 for row in rows_native if row["expert"] == position)
                final_own_per[f"{task_id}/{order_name}"] += own_count
                final_seed_own_mine[f"{task_id}/{order_name}/{seed}"] = own_count
                for index, (row, oracle) in enumerate(zip(rows_native, rows_oracle)):
                    if row["expert"] == position and row["prediction"] != oracle["prediction"]:
                        manifest_row = manifest_rows[task_id][row["id"]]
                        final_mismatch_per[f"{task_id}/{order_name}"] += 1
                        per_seed_mine[f"{task_id}/{order_name}/{seed}"] += 1
                        occurrences.append(_occurrence(
                            "", "caption_final", task_id, order_name, name, seed, index, position,
                            rows_native, rows_oracle, manifest_row,
                            {"g5_native": f"{G5_ROOT}/{name}/native_joint/predictions_after_03_{task_id}.jsonl",
                             "g5_oracle": f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_03_{task_id}.jsonl",
                             "g4_final": f"{G4_ROOT}/{name}/predictions_after_03_{task_id}.jsonl",
                             "g4_final_text_equal": rows_g4_final[index]["prediction"] == row["prediction"],
                             "g4_final_expert_equal": rows_g4_final[index]["expert"] == row["expert"]},
                            stage="after_03"))

    observed_diag_counts = Counter()
    observed_final_counts = Counter()
    per_bank_occ = Counter()
    for entry in occurrences:
        if entry["set"] == "vqa_diagonal":
            observed_diag_counts[f"{entry['task']}/{entry['order_name']}"] += 1
        else:
            observed_final_counts[f"{entry['task']}/{entry['order_name']}"] += 1
        per_bank_occ[entry["bank"]] += 1
    by_task_bound = partition["vqa_transfer"]["R_own_route_worstcase_bound"]["own_route_text_mismatch_rows_by_task"]
    observed_diag_by_task = Counter(entry["task"] for entry in occurrences if entry["set"] == "vqa_diagonal")

    failures = []
    for key, expected in shipped_diag_counts.items():
        if observed_diag_counts.get(key, 0) != expected:
            failures.append(f"shipped diag count {key}: {expected} != observed {observed_diag_counts.get(key, 0)}")
    for key, expected in shipped_final_counts.items():
        if observed_final_counts.get(key, 0) != expected:
            failures.append(f"shipped final count {key}: {expected} != observed {observed_final_counts.get(key, 0)}")
    for key, expected in shipped_final_seed_counts.items():
        if per_seed_mine.get(key, 0) != expected:
            failures.append(f"shipped final seed count {key}: {expected} != {per_seed_mine.get(key, 0)}")
    for key, expected in shipped_final_seed_own.items():
        if final_seed_own_mine.get(key, 0) != expected:
            failures.append(f"shipped final seed own rows {key}: {expected} != {final_seed_own_mine.get(key, 0)}")
    for key, expected in shipped_diag_own.items():
        if diag_own_per.get(key, 0) != expected:
            failures.append(f"shipped diag own rows {key}: {expected} != {diag_own_per.get(key, 0)}")
    for key, expected in shipped_final_own.items():
        if final_own_per.get(key, 0) != expected:
            failures.append(f"shipped final own rows {key}: {expected} != {final_own_per.get(key, 0)}")
    for task_id, expected in by_task_bound.items():
        if observed_diag_by_task.get(task_id, 0) != expected:
            failures.append(f"shipped vqa bound rows by task {task_id}: {expected} != {observed_diag_by_task.get(task_id, 0)}")
    task_counts = Counter(entry["task"] for entry in occurrences)
    if (len(occurrences) != 62 or task_counts.get("V") != 1 or task_counts.get("G") != 20
            or task_counts.get("C") != 14 or task_counts.get("F") != 27):
        failures.append(f"occurrence total/split not 62 (V1/G20/C14/F27): {len(occurrences)} {dict(task_counts)}")
    if sum(observed_diag_counts.values()) != numerics["vqa_diag_cells_own_route_text_mismatch_rows"]:
        failures.append("vqa diagonal mismatches differ from shipped numerics")
    if caption_diag_mismatch_rows != numerics["caption_diag_cells_own_route_text_mismatch_rows"]:
        failures.append("caption diagonal mismatches differ from shipped numerics")
    if sum(observed_diag_counts.values()) + caption_diag_mismatch_rows != numerics["all_diag_cells_own_route_text_mismatch_rows"]:
        failures.append("all diagonal own-route mismatches differ from shipped numerics")
    if sum(diag_own_per.values()) + caption_diag_own_rows != numerics["all_diag_cells_own_route_rows"]:
        failures.append("all diagonal own-route rows differ from shipped numerics")
    if sum(observed_final_counts.values()) != numerics["caption_final_own_route_text_mismatch_rows"]:
        failures.append("caption final mismatches differ from shipped numerics")
    if sum(final_own_per.values()) != numerics["caption_final_own_route_rows"]:
        failures.append("caption final own-route rows differ from shipped numerics")
    if failures:
        raise ValueError("Cases binding failures:\n" + "\n".join(failures))

    occurrences.sort(key=lambda item: (item["bank"], item["task"], item["row_id"]))
    for index, entry in enumerate(occurrences, start=1):
        entry["occurrence_id"] = f"B1-{index:03d}"
    computes = {}
    for entry in occurrences:
        computes.setdefault(entry["compute_key"], {"key": entry["compute_key"], "occurrence_ids": []})
        computes[entry["compute_key"]]["occurrence_ids"].append(entry["occurrence_id"])

    image_counter = Counter(entry["image_sha256"] for entry in occurrences)
    counts_payload = {
        "total": len(occurrences),
        "by_set": dict(Counter(entry["set"] for entry in occurrences)),
        "by_task": dict(task_counts),
        "by_family": dict(Counter(entry["family"] for entry in occurrences)),
        "by_bank": dict(per_bank_occ),
        "distinct_row_ids": len({entry["row_id"] for entry in occurrences}),
        "distinct_image_sha256": len(image_counter),
        "max_occurrences_per_image_sha256": max(image_counter.values()),
        "note": "the evaluation grid is 4 orders x 8 seeds x 512-image eval corpora per task: these are "
                "evaluation OCCURRENCES, not independent image sets; distinct rows/bytes are reported separately",
    }
    binding_payload = {
        "shipped_sources": [PARTITION_JSON, CODE_CONDITIONS_JSON, G5_SUMMARY_JSON, G4_TAR, G5_TAR],
        "shipped_diag_counts": shipped_diag_counts,
        "shipped_diag_own_rows": shipped_diag_own,
        "shipped_final_counts": shipped_final_counts,
        "shipped_final_own_rows": shipped_final_own,
        "shipped_final_seed_counts": sorted(shipped_final_seed_counts.items()),
        "shipped_final_seed_own": sorted(shipped_final_seed_own.items()),
        "shipped_numerics": numerics,
        "shipped_vqa_rows_by_task": by_task_bound,
        "observed_diag_counts": dict(sorted(observed_diag_counts.items())),
        "observed_final_counts": dict(sorted(observed_final_counts.items())),
        "observed_final_seed_counts": sorted(per_seed_mine.items()),
        "observed_final_seed_own_rows": sorted(final_seed_own_mine.items()),
        "observed_diag_own_rows": dict(sorted(diag_own_per.items())),
        "observed_final_own_rows": dict(sorted(final_own_per.items())),
        "observed_caption_diag_mismatch_rows": caption_diag_mismatch_rows,
        "binding_status": "pass",
        "caption_diag_note": "caption DIAGONAL routing-timing own-route text mismatches belong to the sibling "
                             "per-example analysis, not to this batch; the 62 here are 21 VQA diagonal + "
                             "41 caption FINAL occurrences.",
    }

    inputs_final = {}
    for name, path in input_paths.items():
        current = digest_file(path)
        if current != inputs_initial[name]:
            raise ValueError(f"Input changed during the cases phase: {name}")
        inputs_final[name] = current

    regions = {
        "schema_version": SCHEMA, "phase": "cases", "tool": TOOL,
        "script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "binding": binding_payload,
        "manifest": {"archive": MANIFESTS_TAR, "member": MANIFEST_MEMBER, "sha256": manifest_sha,
                     "eval_rows_per_task": {task: len(rows) for task, rows in eval_rows.items()}},
        "banks": {name: banks[name] for name in sorted(banks)},
        "files": dict(sorted(files_record.items())),
        "occurrences": occurrences,
        "computes": [computes[key] for key in sorted(computes)],
        "counts": counts_payload,
        "inputs": inputs_final,
        "exclusive_outputs": True,
        "runtime_seconds_cases_phase": round(time.perf_counter() - started, 3),
        "platform": platform.platform(), "python": sys.version.split()[0],
    }
    exclusive_create_text(output, json.dumps(regions, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"cases_written": str(output), "occurrences": len(occurrences),
                      "computes": len(computes), "banks": len(banks), "files": len(files_record),
                      "seconds": regions["runtime_seconds_cases_phase"]}), flush=True)


def _digest_value(digest, name, value) -> None:
    import torch
    if torch.is_tensor(value):
        digest.update(name.encode("utf-8"))
        cpu = value.detach().cpu().contiguous()
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.view(torch.uint8)
        digest.update(cpu.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _digest_value(digest, f"{name}/{key}", value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"{name}/len={len(value)}".encode("utf-8"))
        for index, item in enumerate(value):
            _digest_value(digest, f"{name}/{index}", item)
    elif isinstance(value, (bool, int, float, str, type(None))):
        digest.update(f"{name}={value!r}".encode("utf-8"))
    else:
        digest.update(f"{name}<class:{type(value).__name__}>".encode("utf-8"))


def _tensor_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _digest_value(digest, name, value)
    return digest.hexdigest()


def _expert_digest(model) -> str:
    digest = hashlib.sha256()
    for name, projection in sorted(model._projections.items()):
        for key, tensor in sorted(projection.experts.state_dict().items()):
            _digest_value(digest, f"{name}/{key}", tensor)
    return digest.hexdigest()


def _device_identity() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for week-1 batch-1 regeneration")
    identity = {
        "cuda_visible_devices_os": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count_visible": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=False)
    identity["nvidia_smi"] = smi.stdout.strip().splitlines()
    identity["scheduler_env"] = {"SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                                 "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
                                 "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION")}
    return identity


def _diag_transition(saved: dict, regenerated: dict) -> dict:
    differing = {key: {"saved": saved[key], "batch1": regenerated[key]}
                 for key in GEN_KEYS if saved[key] != regenerated[key]}
    return {"fields_equal": not differing, "differing_fields": differing}


def command_regenerate(args: argparse.Namespace) -> None:
    import math
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import sema_multimodal.run as runner
    from sema_multimodal.data import ImageVerifier, public_rows, validate_manifest
    from sema_multimodal.model import MultimodalReadout

    started = time.perf_counter()
    cases_path = Path(args.cases).resolve()
    output = Path(args.output).resolve()
    repo = Path(args.root).resolve() if args.root else cases_path.resolve().parents[1]
    payload = json.loads(cases_path.read_text())
    if payload.get("schema_version") != SCHEMA or payload.get("phase") != "cases":
        raise ValueError("Expected a week-1 batch-1 cases manifest as input")
    if payload["binding"]["binding_status"] != "pass":
        raise ValueError("Cases manifest binding status must pass")
    cases_sha = digest_file(cases_path)
    occurrences = payload["occurrences"]
    computes = payload["computes"]
    if len(occurrences) != 62 or len(computes) < 1:
        raise ValueError("Cases manifest must carry exactly 62 occurrences")
    covered = Counter(entry["compute_key"] for entry in occurrences)
    for compute in computes:
        if covered[compute["key"]] != len(compute["occurrence_ids"]) or not compute["occurrence_ids"]:
            raise ValueError(f"Cases compute mapping broken for {compute['key']}")
    if sum(len(compute["occurrence_ids"]) for compute in computes) != len(occurrences):
        raise ValueError("Cases Occurrences must map onto exactly one compute each")
    occurrences_by_key = {entry["compute_key"]: entry for entry in occurrences}
    compute_occurrence_ids = {compute["key"]: compute["occurrence_ids"] for compute in computes}

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing run directory: {output}")
    protected_parts = {"readout", "vision", "legacy", "sema_runs", "vqa_runs", "multimodal_runs"}
    if any(part in protected_parts for part in output.parts) or \
            (repo in output.parents and output.name in {"g4", "routing_g4"}):
        raise ValueError(f"Run directory lands inside protected space: {output}")
    output.mkdir(parents=True, exist_ok=False)

    banks = payload["banks"]
    files_record = payload["files"]

    def write_log(record: dict) -> None:
        with (output / "verification.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    initial_files_before = {}
    for member, expected in sorted(files_record.items()):
        target = repo / member
        if not target.is_file():
            raise FileNotFoundError(f"Saved prediction file missing on disk: {target}")
        actual = digest_file(target)
        if actual != expected:
            raise ValueError(f"Original prediction file changed before inference: {member}")
        initial_files_before[member] = actual
    for name in sorted(banks):
        if digest_file(repo / f"{G4_ROOT}/{name}/config.json") != banks[name]["config_sha256"]:
            raise ValueError(f"Bank config changed before inference: {name}")
        if digest_file(repo / f"{G4_ROOT}/{name}/latest.pt") != banks[name]["latest_sha256"]:
            raise ValueError(f"Bank checkpoint changed before inference: {name}")

    first_config = json.loads((repo / f"{G4_ROOT}/{sorted(banks)[0]}/config.json").read_text())
    common_fields = ("method", "eval_split", "lr", "epochs", "rank", "alpha", "batch_size",
                     "micro_batch_size", "image_size", "device", "router_modality", "router_memory",
                     "router_ridge", "replay_per_task", "manifest", "assets", "image_root",
                     "manifest_sha256", "assets_sha256", "tokenizer_sha256", "source_sha256",
                     "schema_version", "protocol", "oracle_diagnostic", "image_shuffled_diagnostic",
                     "smoke_output", "train_limit", "eval_limit", "generation", "optimizer", "loss",
                     "torch", "numpy", "cuda", "deterministic_algorithms", "tf32")
    for name in sorted(banks):
        config = json.loads((repo / f"{G4_ROOT}/{name}/config.json").read_text())
        for field in common_fields:
            if config[field] != first_config[field]:
                raise ValueError(f"Recorded settings drift across banks at {field}: {name}")
    if first_config["micro_batch_size"] != ORIGINAL_MICRO_BATCH:
        raise ValueError("Original micro_batch_size is not 8")
    if first_config["eval_split"] != "eval":
        raise ValueError("Week-1 batch-1 regeneration covers confirmation-eval banks only")

    manifest_file = Path(first_config["manifest"])
    manifest_bytes = manifest_file.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != payload["manifest"]["sha256"]:
        raise ValueError(f"Manifest differs from the shipped hash: {manifest_file}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("protocol") != first_config["protocol"]:
        raise ValueError("Manifest protocol differs from the recorded configuration")
    image_root = Path(first_config["image_root"]).resolve()
    validate_manifest(manifest, image_root=image_root)
    manifest_eval = {task["task_id"]: task["eval"] for task in manifest["tasks"]}
    for task_id, rows in manifest_eval.items():
        if len(rows) != EVAL_ROWS:
            raise ValueError(f"Manifest eval corpus is not {EVAL_ROWS}: {task_id}")
    row_index = {task_id: {row["id"]: index for index, row in enumerate(rows)}
                 for task_id, rows in manifest_eval.items()}
    view_rows = {task_id: public_rows(rows) for task_id, rows in manifest_eval.items()}
    for entry in occurrences:
        if entry["row_id"] not in row_index[entry["task"]]:
            raise ValueError(f"Occurrence row is outside the manifest eval corpus: {entry['row_id']}")
        row = manifest_eval[entry["task"]][row_index[entry["task"]][entry["row_id"]]]
        if (row.get("image") != entry["image"] or row.get("input_text") != entry["input_text"]
                or row.get("image_sha256") != entry["image_sha256"]):
            raise ValueError(f"Manifest row binding differs from the cases manifest: {entry['row_id']}")
        if view_rows[entry["task"]][row_index[entry["task"]][entry["row_id"]]] != \
                {"image": entry["image"], "question": entry["input_text"]}:
            raise ValueError(f"Public view binding differs for {entry['row_id']}")

    asset_root = Path(first_config["assets"])
    assets_verified = {}
    for relative_name, expected_sha in sorted(first_config["assets_sha256"].items()):
        candidate = Path(relative_name) if Path(relative_name).is_absolute() else asset_root / relative_name
        if not candidate.is_file():
            raise FileNotFoundError(f"Model asset missing: {candidate}")
        actual = digest_file(candidate)
        if actual != expected_sha:
            raise ValueError(f"Changed model asset file: {candidate}")
        assets_verified[relative_name] = actual
    if first_config["tokenizer_sha256"] != {key: value for key, value in first_config["assets_sha256"].items()
                                            if key.startswith("tokenizer/")}:
        raise ValueError("Tokenizer manifest differs from the record")

    write_log({"event": "verifications_ok", "banks": len(banks), "files": len(initial_files_before),
               "manifest_sha256": manifest_sha, "assets": len(assets_verified),
               "cases": sha256_bytes(cases_path.read_bytes())})
    image_verifier = ImageVerifier(image_root)

    device_identity = _device_identity()
    seed = first_config["seed"]
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_load_start = time.perf_counter()
    model = MultimodalReadout(first_config["assets"], image_size=first_config["image_size"],
                              rank=first_config["rank"], alpha=first_config["alpha"],
                              device=first_config["device"])
    model_load_seconds = round(time.perf_counter() - model_load_start, 3)
    model._freeze_all()
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    args_b1 = SimpleNamespace(**first_config)
    args_b1.micro_batch_size = 1

    by_bank = {name: [] for name in sorted(banks)}
    for entry in sorted(occurrences, key=lambda item: (item["bank"], item["task"], item["row_id"])):
        by_bank[entry["bank"]].append(entry)

    predictions_path = output / "predictions.jsonl"
    records: list[dict] = []
    bank_blocks: dict[str, dict] = {}
    generation_seconds_total = 0.0
    with predictions_path.open("x", encoding="utf-8") as predictions:
        for name in sorted(by_bank):
            entries = by_bank[name]
            if not entries:
                continue  # the grid holds 32 banks; only banks with occurrences are loaded
            parent = repo / f"{G4_ROOT}/{name}"
            checkpoint = torch.load(parent / "latest.pt", map_location="cpu", weights_only=False)
            bank_config = json.loads((repo / f"{G4_ROOT}/{name}/config.json").read_text())
            if checkpoint.get("schema_version") != 1 or checkpoint.get("settings") != bank_config:
                raise ValueError(f"Checkpoint settings differ from the recorded configuration: {name}")
            if checkpoint.get("state", {}).get("assignments") != list(range(4)):
                raise ValueError(f"Checkpoint expert assignments are not the identity: {name}")
            bank_state = checkpoint["model"]
            if (bank_state.get("num_experts") != 4 or bank_state.get("rank") != first_config["rank"]
                    or not math.isclose(bank_state.get("alpha"), first_config["alpha"], rel_tol=1e-12)):
                raise ValueError(f"Checkpoint bank does not match the declared capacity: {name}")
            bank_block = {
                "bank": name, "order_name": banks[name]["order_name"], "seed": banks[name]["seed"],
                "latest_sha256": banks[name]["latest_sha256"], "latest_bytes": banks[name]["latest_bytes"],
                "config_sha256": banks[name]["config_sha256"], "occurrences": len(entries),
                "expert_digest": {"model_rank_alpha": [first_config["rank"], first_config["alpha"]]},
                "load_seconds": None,
            }
            load_started = time.perf_counter()
            model.load_checkpoint_state(bank_state)
            if model.num_experts != 4:
                raise ValueError(f"Reconstructed bank size differences {name}")
            model._freeze_all()
            bank_block["load_seconds"] = round(time.perf_counter() - load_started, 3)
            bank_block["full_parameter_digest_before_this_bank"] = _tensor_digest(model)
            del bank_block["expert_digest"]
            bank_block["expert_digest_before"] = _expert_digest(model)
            if torch.cuda.is_available() and model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for entry in entries:
                task_id = entry["task"]
                row = manifest_eval[task_id][row_index[task_id][entry["row_id"]]]
                image_sha_before = image_verifier.verify(entry["image"])
                if image_sha_before != entry["image_sha256"]:
                    raise ValueError(f"Image bytes differ from the manifest record: {entry['image']}")
                started_row = time.perf_counter()
                answers, assignments, diagnostics = runner.routed_predictions(
                    model, None, [row], None, args_b1, forced_expert=entry["gold_expert"])
                row_seconds = round(time.perf_counter() - started_row, 3)
                generation_seconds_total += row_seconds
                if len(answers) != 1 or len(diagnostics) != 1:
                    raise ValueError("Batch-1 generation lost the requested input row")
                if assignments[0] != entry["gold_expert"]:
                    raise ValueError("Forced expert disagrees with the shared expert")
                regenerated = answers[0]
                regen_diag = diagnostics[0]
                saved_native = entry["native_prediction"]
                saved_oracle = entry["oracle_prediction"]
                matches = {"native": regenerated == saved_native, "oracle": regenerated == saved_oracle}
                matches["both"] = matches["native"] and matches["oracle"]
                matches["neither"] = not (matches["native"] or matches["oracle"])
                match_label = ("both" if matches["both"] else "native" if matches["native"]
                               else "oracle" if matches["oracle"] else "neither")
                record = {
                    "occurrence_ids": list(compute_occurrence_ids[entry["compute_key"]]),
                    "compute_key": entry["compute_key"],
                    "set": entry["set"], "family": entry["family"], "task": task_id,
                    "bank": entry["bank"], "order_name": entry["order_name"], "seed": entry["seed"],
                    "stage": entry["stage"], "shared_expert": entry["gold_expert"],
                    "row_id": entry["row_id"], "image_sha256": entry["image_sha256"],
                    "image_sha256_verified": image_sha_before,
                    "match_class": match_label, "matches": matches,
                    "regenerated_prediction": regenerated,
                    "regenerated_generation": regen_diag,
                    "generation_diagnostic_transitions": {
                        "native": _diag_transition(entry["native_generation"], regen_diag),
                        "oracle": _diag_transition(entry["oracle_generation"], regen_diag)},
                    "micro_batch_sizes": {"original": ORIGINAL_MICRO_BATCH, "run": 1},
                    "compute_seconds": row_seconds,
                    "peak_gpu_bytes_cumulative_at_row": int(torch.cuda.max_memory_allocated()),
                }
                records.append(record)
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()
            bank_block["expert_digest_after"] = _expert_digest(model)
            bank_block["expert_tensors_identical_pre_post"] = (
                bank_block["expert_digest_before"] == bank_block["expert_digest_after"])
            if not bank_block["expert_tensors_identical_pre_post"]:
                raise ValueError(f"Expert bank changed during batch-1 inference: {name}")
            bank_block["full_parameter_digest_after_this_bank"] = _tensor_digest(model)
            bank_block["whole_model_identical_pre_post_bank"] = (
                bank_block["full_parameter_digest_before_this_bank"]
                == bank_block["full_parameter_digest_after_this_bank"])
            if not bank_block["whole_model_identical_pre_post_bank"]:
                raise ValueError(f"Base parameters changed during batch-1 inference: {name}")
            if model.rank != first_config["rank"] or not math.isclose(model.alpha, first_config["alpha"], rel_tol=1e-12):
                raise ValueError("LoRA capacity changed during batch-1 inference")
            bank_block["peak_gpu_bytes_at_bank"] = int(torch.cuda.max_memory_allocated())
            bank_blocks[name] = bank_block
            del checkpoint, bank_state
            torch.cuda.empty_cache()

    if "sema_multimodal.metrics" in sys.modules:
        raise ValueError("Metrics module must not be imported (no scorer path)")
    if len(records) != len(computes):
        raise RuntimeError("Compute coverage assertion failed after inference")

    after_files = {}
    for member, expected in sorted(files_record.items()):
        actual = digest_file(repo / member)
        if actual != expected:
            raise ValueError(f"Original prediction file changed during inference: {member}")
        after_files[member] = actual
    for name in sorted(banks):
        if digest_file(repo / f"{G4_ROOT}/{name}/config.json") != banks[name]["config_sha256"]:
            raise ValueError(f"Bank config changed during inference: {name}")
        if digest_file(repo / f"{G4_ROOT}/{name}/latest.pt") != banks[name]["latest_sha256"]:
            raise ValueError(f"Bank checkpoint changed during inference: {name}")

    match_counter = Counter(record["match_class"] for record in records)
    per_occurrence_matches = Counter()
    for record in records:
        for occurrence in record["occurrence_ids"]:
            per_occurrence_matches[record["match_class"]] += 1
    family_task_bank = {"by_family": Counter(), "by_task": Counter(), "by_bank": Counter()}
    for record in records:
        family_task_bank["by_family"][record["family"]] += 1
        family_task_bank["by_task"][f"{record['family']}/{record['task']}"] += 1
        family_task_bank["by_bank"][record["bank"]] += 1
    transitions_counter = {"native": Counter(), "oracle": Counter()}
    for record in records:
        for variant in ("native", "oracle"):
            transition = record["generation_diagnostic_transitions"][variant]
            transitions_counter[variant]["fields_equal" if transition["fields_equal"] else "fields_changed"] += 1

    report = {
        "schema_version": SCHEMA, "phase": "regenerate", "tool": TOOL,
        "cases_path": str(cases_path), "cases_sha256": cases_sha,
        "run_directory": str(output),
        "repo_root": str(repo),
        "binding": {
            "manifest_path": str(manifest_file), "manifest_sha256": manifest_sha,
            "image_root": str(image_root), "asset_root": str(asset_root),
            "banks_total_in_grid": 32, "banks_used": len(banks),
            "verified_prediction_files": len(files_record),
            "verified_asset_files": len(assets_verified),
        },
        "started_wall_seconds": round(started, 3), "finished_wall_seconds": round(time.perf_counter(), 3),
        "runtime_seconds_total": round(time.perf_counter() - started, 3),
        "runtime_seconds_generation_sum": round(generation_seconds_total, 3),
        "occurrences": len(occurrences), "computes": len(records),
        "compute_reuse_applied": len(records) != len(occurrences),
        "run_micro_batch_size": 1, "original_micro_batch_size": ORIGINAL_MICRO_BATCH,
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "tokenizer_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
            "cpu_threads_torch": args.cpu_threads,
            "platform": platform.platform(),
        },
        "device": device_identity,
        "model_identity": {
            "backbone": "BLIP model_base_capfilt_large (frozen) + ViT-L/14-336 (pos_embedding interpolated)",
            "assets_verified": assets_verified,
            "rank": first_config["rank"], "alpha": first_config["alpha"], "num_experts": 4,
            "generation_settings": first_config["generation"],
            "model_load_seconds": model_load_seconds,
            "all_banks_expert_tensors_identical_pre_post": all(
                block["expert_tensors_identical_pre_post"] for block in bank_blocks.values()),
            "all_banks_whole_parameters_identical_pre_post": all(
                block["whole_model_identical_pre_post_bank"] for block in bank_blocks.values()),
            "shared_bank_load_seconds": {
                name: bank_blocks[name]["load_seconds"] for name in bank_blocks},
        },
        "settings": {
            "eval_split": first_config["eval_split"], "eval_rows_per_task": EVAL_ROWS,
            "vqa_max_new_tokens": first_config["generation"]["vqa_max_new_tokens"],
            "caption_max_new_tokens": first_config["generation"]["caption_max_new_tokens"],
            "router_variants_in_scope": ["native_joint", "task_id_oracle"],
            "forced_expert_only": True, "router_refit_steps": 0, "feature_extraction": "skipped (forced expert)",
        },
        "banks": bank_blocks,
        "match_counts": {"by_computes": dict(match_counter), "by_occurrences": dict(per_occurrence_matches)},
        "match_by_family_task": dict(family_task_bank["by_task"].items()),
        "match_by_family": dict(family_task_bank["by_family"].items()),
        "match_by_bank": dict(sorted(family_task_bank["by_bank"].items())),
        "diagnostic_field_transitions_by_variant": {
            variant: dict(counter.items()) for variant, counter in transitions_counter.items()},
        "per_occurrence": [
            {"occurrence_id": record["occurrence_ids"][0], "compute_key": record["compute_key"],
             "set": record["set"], "task": record["task"], "family": record["family"], "bank": record["bank"],
             "order_name": record["order_name"], "seed": record["seed"], "row_id": record["row_id"],
             "match_class": record["match_class"], "compute_seconds": record["compute_seconds"]}
            for record in records],
        "originals": {
            "prediction_files_verified_before_and_after": len(files_record),
            "unchanged_after_inference": True,
            "bank_configs_unchanged_after_inference": True,
            "checkpoints_unchanged_after_inference": True,
        },
        "constraints": {
            "scorer_executed": False, "metrics_module_imported": "sema_multimodal.metrics" in sys.modules,
            "fresh_predictions_outside_62": 0, "expert_retraining_steps": 0, "router_refit_steps": 0,
        },
        "limitations": [
            "ONE batch-1 pass per occurrence cannot causally attribute the saved native/oracle same-expert "
            "text differences: padding/batch compositional (bf16 distributed round-trip), run-order, "
            "checkpointing, or cudnn algorithm selection differences may all act on the original training-time "
            "and G5 replay evaluations. Text equalities observed at batch-1 are only observations, not causes.",
            "The 21 VQA diagonal and 41 caption final analyses evaluate the same trained banks (shared across "
            "two sets); the pair (bank, task, stage) is unique per occurrence and the shared expert is the "
            "task's own expert. No new router evaluation, no rescoring and no model selection was performed.",
            "One-token outputs for VQA (generated_tokens can be 1 to 3) are strongly constrained by "
            "near-tied greedy logits: exact match rates are intentionally reported literally and without "
            "model-selection interpretation.",
            "Wall/GPU allocation time is reported by SLURM sacct; GPU utilization/utilized capacity is NOT "
            "claimed because nvidia-smi utilization was not recorded during the passes.",
        ],
        "exclusive_outputs": True,
    }
    exclusive_create_text(output / "report.json",
                          json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    write_log({"event": "batch1_done", "computes": len(records), "match_counts": dict(match_counter),
               "seconds": report["runtime_seconds_total"]})
    print(json.dumps({"output": str(output), "computes": len(records),
                      "match_counts": dict(match_counter),
                      "elapsed_seconds": report["runtime_seconds_total"]}), flush=True)


def command_merge(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    regeneration_path = Path(args.regeneration).resolve()
    predictions_path = Path(args.predictions).resolve()
    sacct_path = Path(args.sacct).resolve()
    cases_path = Path(args.cases).resolve()
    output = Path(args.output).resolve()
    predictions_output = Path(args.predictions_output).resolve()
    inputs_root = Path(args.root).resolve() if args.root else cases_path.resolve().parents[1]

    report = json.loads(regeneration_path.read_text())
    if report.get("schema_version") != SCHEMA or report.get("phase") != "regenerate":
        raise ValueError("Expected a week-1 batch-1 regenerate report")
    predictions_text = predictions_path.read_text()
    predicted_lines = [json.loads(line) for line in predictions_text.splitlines() if line.strip()]
    if len(predicted_lines) != report["computes"]:
        raise ValueError("Fetched predictions line count differs from the run report")
    match_counter = Counter(line["match_class"] for line in predicted_lines)
    if dict(match_counter) != report["match_counts"]["by_computes"]:
        raise ValueError("Fetched prediction match counts differ from the report")
    if match_counter.get("both", 0) > 0:
        raise ValueError("match_class both cannot happen for differing saved native/oracle texts")
    if report["occurrences"] != len(report["per_occurrence"]):
        raise ValueError("Report occurrence rows differ from the records")
    covered = Counter()
    for line in predicted_lines:
        for occurrence in line.get("occurrence_ids", []):
            covered[occurrence] += 1
    if not covered or set(covered) != {entry["occurrence_id"] for entry in report["per_occurrence"]} \
            or any(value != 1 for value in covered.values()):
        raise ValueError("Occurrence mapping across the fetched predictions is not exactly-once")
    cases = json.loads(cases_path.read_text())
    if digest_file(cases_path) != report["cases_sha256"]:
        raise ValueError("Cases manifest differs from the run report binding")
    if len(cases["occurrences"]) != report["occurrences"]:
        raise ValueError("Cases occurrence count differs from the report")

    sacct_lines = []
    for line in sacct_path.read_text().splitlines():
        if line.startswith("JobID|") or not line.strip():
            continue
        record = {}
        for item in line.split("|"):
            if "=" in item:
                key, _, value = item.partition("=")
                record[key.strip()] = value.strip()
        sacct_lines.append(record)
    elapsed = None
    for record in sacct_lines:
        if record.get("JobID", "").split(".")[0] == args.job_id and record.get("Elapsed"):
            elapsed = elapsed or record["Elapsed"]

    local_inputs = {}
    for name, entry in cases["inputs"].items():
        actual = digest_file(inputs_root / name)
        if actual != entry["sha256"]:
            raise ValueError(f"Local input changed after the cases phase: {name}")
        local_inputs[name] = actual

    regions = dict(report)
    regions["phase"] = "final"
    regions["tool"] = TOOL
    regions["runtime_seconds_total_run_phase"] = report["runtime_seconds_total"]
    regions["runtime_seconds_final_phase"] = round(time.perf_counter() - started, 3)
    regions["scheduler"] = {
        "job_id": args.job_id,
        "records": sacct_lines,
        "job_elapsed_first_seen": elapsed,
        "source": "sacct --parsable2 on the submitting login node: allocation wall/cpus/gres only; "
                  "GPU compute utilization is intentionally not claimed",
    }
    regions["local_verifications"] = {
        "inputs_unchanged_after_the_cases_phase": True,
        "inputs": local_inputs,
        "fetched_predictions_sha256": sha256_bytes(predictions_text.encode()),
        "fetched_predictions_lines": len(predicted_lines),
        "fetched_predictions_match_counts": dict(match_counter),
        "fetched_predictions_occurrence_coverage": len(covered),
        "cases_local_sha256": digest_file(cases_path),
        "regeneration_local_sha256": digest_file(regeneration_path),
    }
    exclusive_create_text(output, json.dumps(regions, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if predictions_output.exists() or predictions_output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite local predictions: {predictions_output}")
    with predictions_output.open("x", encoding="utf-8") as stream:
        stream.write(predictions_text)
    print(json.dumps({"merge_written": str(output), "predictions_written": str(predictions_output),
                      "scheduler_elapsed": elapsed, "match_counts": dict(match_counter)}), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cases_parser = sub.add_parser("cases", help="Build the 62-occurrence case manifest from shipped artifacts")
    cases_parser.add_argument("--output", type=Path, default=Path("artifacts/week1-batch1-cases.json"))
    cases_parser.add_argument("--root", type=Path, default=ROOT_DEFAULT.resolve())
    regenerate_parser = sub.add_parser("regenerate", help="Batch-1 CUDA regeneration on the DC1 GPU node")
    regenerate_parser.add_argument("--cases", type=Path, required=True)
    regenerate_parser.add_argument("--output", type=Path, required=True)
    regenerate_parser.add_argument("--root", type=Path, default=None)
    regenerate_parser.add_argument("--cpu-threads", type=int, default=4)
    merge_parser = sub.add_parser("merge", help="Attach sacct evidence and finalize local artifacts")
    merge_parser.add_argument("--regeneration", type=Path, required=True)
    merge_parser.add_argument("--predictions", type=Path, required=True)
    merge_parser.add_argument("--predictions-output", type=Path, required=True)
    merge_parser.add_argument("--sacct", type=Path, required=True)
    merge_parser.add_argument("--cases", type=Path, required=True)
    merge_parser.add_argument("--job-id", type=str, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "cases":
        command_cases(args.root.resolve(), args.output.resolve())
    elif args.command == "regenerate":
        command_regenerate(args)
    elif args.command == "merge":
        command_merge(args)
    else:
        raise SystemExit("Unknown stage")


if __name__ == "__main__":
    main()
