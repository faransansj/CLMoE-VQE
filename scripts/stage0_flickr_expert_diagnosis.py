#!/usr/bin/env python3
"""Stage 0: diagnose Flickr expert generation from existing code/logs only.

No training, inference, or scoring. Reads saved events/results from the G1
development grid and G4 independent runs, plus the existing per-example score
rows, and summarizes training targets, loss trajectories, and generation
length/EOS behavior for COCO (C) vs Flickr (F).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import statistics as st
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/stage0-flickr-expert-diagnosis.json"
G1_TAR = ROOT / "artifacts/multimodal-g1-raw.tar.gz"
G4_TAR = ROOT / "artifacts/multimodal-g4-raw.tar.gz"
MANIFEST_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
SCORES = ROOT / "artifacts/week1-per-example-scores.jsonl.gz"
NORMALIZERS = ROOT / "artifacts/multimodal-g1-normalizers.json"
CODE = {
    "sema_multimodal/model.py": [r"generation_limit", r"normalize_tokens=True"],
    "sema_multimodal/data.py": [r"def training_rows", r"refs\[\(offset \+ epoch\) % len\(refs\)\]"],
    "sema_vqa/readout_model.py": [r"masked_fill\(answers.attention_mask == 0, -100\)",
                                  r"includes the supervised answer-ending SEP"],
    "sema_vqa/vision_model.py": [r"truncation=False", r"exceeds 512 positions"],
}
G1_RE = re.compile(r"artifacts/multimodal_runs/g1/([CF])_lr([0-9.]+)_ep(\d+)_seed17/(events\.jsonl|results\.json)$")
G4_RE = re.compile(r"artifacts/multimodal_runs/g4/([CF])_independent_lr0\.0003_ep5_seed(\d+)/events\.jsonl$")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def summarize_events(lines: list[dict]) -> dict:
    batches = [e for e in lines if e["event"] == "train_batch"]
    start = next(e for e in lines if e["event"] == "task_start")
    final = next(e for e in lines if e["event"] == "evaluation")
    by_epoch = defaultdict(list)
    for e in batches:
        by_epoch[e["epoch"]].append(e)
    epochs = sorted(by_epoch)
    return {
        "epochs": len(epochs),
        "steps": len(batches),
        "loss_mean_by_epoch": [mean(b["loss"] for b in by_epoch[k]) for k in epochs],
        "target_tokens_per_image_incl_eos": (sum(b["target_tokens"] for b in batches)
                                             / sum(b["current_images"] for b in batches)),
        "lr_first_last": [batches[0]["lr"], batches[-1]["lr"]],
        "pre_score": start["pre_evaluation"]["scoring"]["score"],
        "pre_generation": start["pre_evaluation"]["generation"],
        "post_score": final["scoring"]["score"],
        "post_generation": final["generation"],
    }


def read_tar(path: Path, pattern: re.Pattern) -> dict:
    found = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            match = pattern.match(member.name)
            if match and member.isfile():
                found[match.groups()] = tar.extractfile(member).read().decode("utf-8")
    return found


def code_checks() -> dict:
    result = {}
    for rel, patterns in CODE.items():
        text = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        result[rel] = {"sha256": digest(ROOT / rel), "matches": {}}
        for pattern in patterns:
            lines = [i + 1 for i, line in enumerate(text) if re.search(pattern, line)]
            assert lines, f"{rel}: {pattern}"
            result[rel]["matches"][pattern] = lines
    return result


def reference_lengths() -> dict:
    with tarfile.open(MANIFEST_TAR, "r:gz") as tar:
        manifest = json.loads(tar.extractfile(MANIFEST_MEMBER).read())
    out = {}
    for task in manifest["tasks"]:
        if task["task_id"] not in ("C", "F"):
            continue
        lengths = [len(ref.split()) for row in task["train"] for ref in row["references"]]
        out[task["task_id"]] = {
            "train_references": len(lengths),
            "whitespace_tokens_mean": mean(lengths),
            "median": st.median(lengths),
            "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            "max": max(lengths),
            "share_over_20_words": sum(n > 20 for n in lengths) / len(lengths),
            "share_over_30_words": sum(n > 30 for n in lengths) / len(lengths),
        }
    return out


def final_native_length_tables() -> dict:
    cells = defaultdict(lambda: defaultdict(list))
    with gzip.open(SCORES, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["task"] not in ("C", "F") or row["phase"] != "after":
                continue
            if row["kind"] == "independent":
                group = "independent"
            elif row["kind"] == "oracle" and row["stage"] == 3:
                group = "final_oracle_own_expert"
            else:
                continue
            gen = row["generation"]
            cells[row["task"]][group].append((gen["generated_tokens"], gen["eos"],
                                              gen["limit_reached"], gen["repeated_trigram"],
                                              row["score"]))
    result = {}
    for task, groups in cells.items():
        result[task] = {}
        for group, rows in groups.items():
            ended = [r for r in rows if r[1]]
            result[task][group] = {
                "occurrences": len(rows),
                "generated_tokens_mean": mean(r[0] for r in rows),
                "eos_rate": mean(float(r[1]) for r in rows),
                "limit_reached_rate": mean(float(r[2]) for r in rows),
                "repeated_trigram_rate": mean(float(r[3]) for r in rows),
                "score_mean": mean(r[4] for r in rows),
                "eos_rows_tokens_mean": mean(r[0] for r in ended),
                "eos_rows_score_mean": mean(r[4] for r in ended),
                "non_eos_rows_score_mean": mean(r[4] for r in rows if not r[1]),
            }
    return result


def old_c_vs_new_f_by_termination() -> dict:
    """Pair saved pre-stage C-expert outputs with post-stage F-expert outputs on the
    same Flickr eval images (16 later-task cells), split by whether F emitted EOS."""
    pre_rows, post_rows = defaultdict(dict), defaultdict(dict)
    with gzip.open(SCORES, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["task"] != "F" or row["stage"] != 3 or not row["order"]:
                continue
            cell = ("_".join(row["order"]), row["seed"])
            if row["kind"] == "native" and row["phase"] == "before":
                pre_rows[cell][row["id"]] = row
            elif row["kind"] == "oracle" and row["phase"] == "after":
                post_rows[cell][row["id"]] = row
    assert len(pre_rows) == 16 and set(pre_rows) <= set(post_rows)
    groups = defaultdict(list)
    for cell, rows in pre_rows.items():
        assert len(rows) == 512
        for image_id, old in rows.items():
            new = post_rows[cell][image_id]
            assert old["expert_name"] == "C" and new["expert_name"] == "F"
            key = "new_f_emitted_eos" if new["generation"]["eos"] else "new_f_no_eos"
            groups[key].append((old["score"], new["score"], old["generation"]["eos"]))
    return {key: {"occurrences": len(rows),
                  "old_c_score_mean": mean(r[0] for r in rows),
                  "new_f_score_mean": mean(r[1] for r in rows),
                  "mean_new_f_minus_old_c": mean(r[1] - r[0] for r in rows),
                  "new_f_wins": sum(r[1] > r[0] for r in rows),
                  "old_c_wins": sum(r[1] < r[0] for r in rows),
                  "ties": sum(r[1] == r[0] for r in rows),
                  "old_c_eos_rate": mean(float(r[2]) for r in rows)}
            for key, rows in sorted(groups.items())}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    g1_raw = read_tar(G1_TAR, G1_RE)
    g1 = {}
    for (task, lr, epochs, kind), text in sorted(g1_raw.items()):
        if kind != "events.jsonl":
            continue
        summary = summarize_events([json.loads(l) for l in text.splitlines() if l.strip()])
        g1[f"{task}_lr{lr}_ep{epochs}"] = summary
    g4_raw = read_tar(G4_TAR, G4_RE)
    g4 = defaultdict(dict)
    for (task, seed), text in sorted(g4_raw.items()):
        g4[task][seed] = summarize_events([json.loads(l) for l in text.splitlines() if l.strip()])
    assert len(g1) == 8 and all(len(v) == 8 for v in g4.values())
    g4_summary = {}
    for task, runs in g4.items():
        g4_summary[task] = {
            "seeds": len(runs),
            "loss_mean_by_epoch": [mean(r["loss_mean_by_epoch"][k] for r in runs.values()) for k in range(5)],
            "target_tokens_per_image_incl_eos": mean(r["target_tokens_per_image_incl_eos"] for r in runs.values()),
            "pre_generated_tokens": mean(r["pre_generation"]["generated_tokens"] for r in runs.values()),
            "post_generated_tokens": mean(r["post_generation"]["generated_tokens"] for r in runs.values()),
            "post_eos_rate": mean(r["post_generation"]["eos"] for r in runs.values()),
            "post_limit_rate": mean(r["post_generation"]["limit_reached"] for r in runs.values()),
            "post_repeated_trigram_rate": mean(r["post_generation"]["repeated_trigram"] for r in runs.values()),
            "post_score": mean(r["post_score"] for r in runs.values()),
        }
    payload = {
        "stage": "stage0-flickr-expert-diagnosis",
        "execution": {"training": False, "inference": False, "scoring": False},
        "script": {"path": "scripts/stage0_flickr_expert_diagnosis.py",
                   "sha256": digest(Path(__file__))},
        "inputs": {str(p.relative_to(ROOT)): digest(p)
                   for p in (G1_TAR, G4_TAR, MANIFEST_TAR, SCORES, NORMALIZERS)},
        "code_checks": code_checks(),
        "train_reference_lengths": reference_lengths(),
        "g1_development_grid_seed17_validation": g1,
        "g4_independent_eval": g4_summary,
        "per_example_generation_eval": final_native_length_tables(),
        "old_c_vs_new_f_by_new_f_termination": old_c_vs_new_f_by_termination(),
        "caveat": ("Termination split is observational; F-EOS rows are a nonrandom subset. "
                   "G1 grid is a single development seed. No new training/inference/scoring."),
        "normalizers": json.loads(NORMALIZERS.read_text())["scores"],
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"written": str(OUTPUT.relative_to(ROOT)), "g1_runs": len(g1),
                      "g4_runs": sum(len(v) for v in g4.values())}))


if __name__ == "__main__":
    main()
