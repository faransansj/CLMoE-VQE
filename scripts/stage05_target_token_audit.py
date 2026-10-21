#!/usr/bin/env python3
"""Stage 0.5: verify which caption target strings were actually trained.

Reconstructs every logged training batch of the G4 independent C/F runs from the
locked manifest and the deterministic batch plan (sema_multimodal.run.epoch_batches
and sema_multimodal.data.training_rows), tokenizes the cycled targets with the
run's BertTokenizer, and compares the per-batch token count with the logged
`target_tokens`. Also reports how often the terminal '.' token appears right
before [SEP] in trained targets. No model, no training, no scoring.

Usage: python scripts/stage05_target_token_audit.py --tokenizer DIR --events-root DIR --output PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sema_multimodal.data import training_rows  # noqa: E402
from sema_multimodal.run import epoch_batches  # noqa: E402

MANIFEST_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--events-root", type=Path, required=True,
                        help="directory holding g4/<task>_independent_lr0.0003_ep5_seed<seed>/events.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    period_id = tokenizer.convert_tokens_to_ids(".")
    with tarfile.open(MANIFEST_TAR, "r:gz") as tar:
        raw = tar.extractfile(MANIFEST_MEMBER).read()
    manifest = json.loads(raw)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    cache: dict[str, list[int]] = {}

    def ids(text: str) -> list[int]:
        if text not in cache:
            cache[text] = tokenizer(text, add_special_tokens=False)["input_ids"]
        return cache[text]

    report = {"script_sha256": digest_bytes(Path(__file__).read_bytes()),
              "manifest_sha256": digest_bytes(raw), "period_token_id": period_id,
              "sep_token_id": tokenizer.sep_token_id, "tasks": {}}
    for task_id in ("C", "F"):
        rows = tasks[task_id]["train"]
        batches_checked = mismatched = 0
        trained_targets = 0
        last_token = Counter()
        for seed in SEEDS:
            events_path = args.events_root / f"g4/{task_id}_independent_lr0.0003_ep5_seed{seed}/events.jsonl"
            logged = [json.loads(line) for line in events_path.read_text().splitlines()
                      if '"train_batch"' in line]
            logged = {(e["epoch"], e["batch"]): e["target_tokens"] for e in logged}
            for epoch in range(5):
                plan = epoch_batches(rows, [], seed=seed, task=0, epoch=epoch, batch_size=16)
                for index, (batch, _, _) in enumerate(plan):
                    targets = [row["answer"] for row in training_rows(batch, seed=seed, epoch=epoch)]
                    tokens = sum(len(ids(text)) + 1 for text in targets)  # wordpieces + [SEP]
                    batches_checked += 1
                    mismatched += tokens != logged[(epoch, index)]
                    for text in targets:
                        piece = ids(text)
                        last_token["period" if piece and piece[-1] == period_id else "other"] += 1
                        trained_targets += 1
            assert len(logged) == 5 * len(plan)
        report["tasks"][task_id] = {
            "batches_checked": batches_checked,
            "batches_token_count_mismatch": mismatched,
            "trained_target_occurrences": trained_targets,
            "share_targets_ending_with_period_token_before_sep": last_token["period"] / trained_targets,
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["tasks"]))


if __name__ == "__main__":
    main()
