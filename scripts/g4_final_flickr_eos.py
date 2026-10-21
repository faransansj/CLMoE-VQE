"""Aggregate final-evaluation Flickr termination stats from the saved G4 predictions.

Read-only over artifacts/multimodal-g4-raw.tar.gz (streamed, never extracted).
For every G4 run the last `predictions_after_*_F.jsonl` is the final Flickr evaluation.
Expansion rows are split by the routed expert; expert index == stream position of its task.
"""

import argparse
import collections
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path

RUN_RE = re.compile(r"multimodal_runs/g4/(?P<run>[^/]+)/predictions_after_(?P<step>\d+)_F\.jsonl$")
SEED_RE = re.compile(r"_seed(?P<seed>\d+)$")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(run):
    seed = int(SEED_RE.search(run)["seed"])
    if run.startswith("F_independent"):
        return "independent", None, seed
    order = run[:7]
    method = run[8:].split("_lr")[0]
    if "_ep10_" in run:
        method += "_ep10"
    return method, order, seed


def summarize(rows):
    return {
        "rows": len(rows),
        "eos_rate": sum(r["generation"]["eos"] for r in rows) / len(rows),
        "limit_rate": sum(r["generation"]["limit_reached"] for r in rows) / len(rows),
        "mean_generated_tokens": statistics.fmean(r["generation"]["generated_tokens"] for r in rows),
    }


def describe(values):
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="artifacts/multimodal-g4-raw.tar.gz")
    parser.add_argument("--report", default="artifacts/g4-final-flickr-eos.json")
    args = parser.parse_args()

    finals = {}
    with tarfile.open(args.archive) as archive:
        for member in archive:
            match = RUN_RE.search(member.name)
            if match and member.isfile():
                step = int(match["step"])
                if match["run"] not in finals or step > finals[match["run"]][0]:
                    finals[match["run"]] = (step, member.name)
        wanted = {name: run for run, (_, name) in finals.items()}
        per_run = {}
    with tarfile.open(args.archive) as archive:
        for member in archive:
            run = wanted.get(member.name)
            if run is None:
                continue
            rows = [json.loads(line) for line in archive.extractfile(member)]
            method, order, seed = parse_run(run)
            entry = {"method": method, "order": order, "seed": seed, "file": member.name, "all": summarize(rows)}
            if method == "expand":
                by_expert = collections.defaultdict(list)
                for row in rows:
                    by_expert[order.split("_")[row["expert"]]].append(row)
                entry["by_routed_expert_task"] = {task: summarize(group) for task, group in sorted(by_expert.items())}
            per_run[run] = entry

    groups = collections.defaultdict(list)
    for entry in per_run.values():
        position = None if entry["order"] is None else ("F_first" if entry["order"].startswith("F") else "F_last")
        groups[(entry["method"], position)].append(entry)

    aggregate = {}
    for (method, position), entries in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        key = method if position is None else f"{method}/{position}"
        block = {
            "runs": len(entries),
            "eos_rate": describe([e["all"]["eos_rate"] for e in entries]),
            "mean_generated_tokens": describe([e["all"]["mean_generated_tokens"] for e in entries]),
        }
        if method == "expand":
            for task in ("F", "C"):
                rates = [e["by_routed_expert_task"][task]["eos_rate"] for e in entries if task in e["by_routed_expert_task"]]
                shares = [e["by_routed_expert_task"].get(task, {"rows": 0})["rows"] / e["all"]["rows"] for e in entries]
                block[f"routed_to_{task}_expert"] = {"eos_rate": describe(rates), "row_share": describe(shares)}
        aggregate[key] = block

    report = {
        "schema_version": 1,
        "stage": "g4-final-flickr-eos",
        "labels": {
            "diagnostic_only": True,
            "post_hoc": True,
            "source": "generation.eos of the last predictions_after_*_F.jsonl per G4 run (G4 confirmation eval split, already saved)",
            "no_generation_or_scoring": True,
        },
        "inputs": {"archive": args.archive, "archive_sha256": sha256(args.archive)},
        "tool": {"path": "scripts/g4_final_flickr_eos.py", "sha256": sha256(__file__)},
        "aggregate": aggregate,
        "per_run": dict(sorted(per_run.items())),
    }
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    for key, block in aggregate.items():
        line = f"{key:34s} runs={block['runs']:2d} eos={block['eos_rate']['mean']:.4f} [{block['eos_rate']['min']:.4f},{block['eos_rate']['max']:.4f}] len={block['mean_generated_tokens']['mean']:.1f}"
        for task in ("F", "C"):
            sub = block.get(f"routed_to_{task}_expert")
            if sub:
                line += f" | {task}-expert eos={sub['eos_rate']['mean']:.4f} share={sub['row_share']['mean']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
