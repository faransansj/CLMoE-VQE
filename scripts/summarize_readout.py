"""Audit every observed readout prediction and report development/confirmation separately."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from sema_vqa.metrics import continual_metrics, score_prediction


def rescore(path, rows, metric):
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows) or not rows:
        raise ValueError("Duplicate/empty expected evaluation rows")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != len(rows) or len({r["id"] for r in records}) != len(rows):
        raise ValueError(f"Missing/duplicate predictions: {path}")
    if {r["id"] for r in records} != set(expected):
        raise ValueError(f"Prediction IDs differ: {path}")
    scores = []
    for record in records:
        score = score_prediction(record["prediction"], expected[record["id"]], metric)
        if not math.isclose(score, record["score"], abs_tol=1e-12):
            raise ValueError(f"Prediction score mismatch: {path}/{record['id']}")
        scores.append(score)
    return math.fsum(scores) / len(rows), records


def audit(directory):
    config = json.loads((directory / "config.json").read_text())
    raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"Manifest changed: {directory}")
    manifest = json.loads(raw)
    tasks = manifest["tasks"][:config["task_limit"] or None]
    result = json.loads((directory / "results.json").read_text())
    if result["completed"] != len(tasks) or len(result["matrix"]) != len(tasks):
        raise ValueError(f"Incomplete task sequence: {directory}")
    matrix, before, oracle = [], [], []
    count = 0
    final_counts = Counter()
    final_task_reports = []
    for stage, task in enumerate(tasks):
        score, records = rescore(directory / f"predictions_before_{stage:02d}.jsonl", task[config["eval_split"]], task["metric"])
        before.append(score)
        count += len(records)
        scores, oracle_scores = [], []
        for seen, previous in enumerate(tasks[:stage + 1]):
            rows = previous[config["eval_split"]]
            score, records = rescore(directory / f"predictions_after_{stage:02d}_{previous['name']}.jsonl", rows, previous["metric"])
            count += len(records)
            scores.append(score)
            if config["oracle_diagnostic"] and config["method"] in ("expand", "adaptive"):
                oracle_score, oracle_records = rescore(directory / f"oracle_after_{stage:02d}_{previous['name']}.jsonl", rows, previous["metric"])
                oracle_scores.append(oracle_score)
                count += len(oracle_records)
            if stage == len(tasks) - 1:
                answers = Counter(record["prediction"] for record in records)
                final_counts.update(answers)
                final_task_reports.append({"task": previous["name"], "score": score,
                    "route_to_assigned_expert_fraction": sum(record["expert"] == result["task_expert_assignments"][seen] for record in records) / len(records),
                    "answer_counts": dict(answers), "examples": len(records)})
        matrix.append(scores)
        oracle.append(oracle_scores)
    metrics = continual_metrics(matrix)
    for key, recovered in (("matrix", matrix), ("oracle_matrix", oracle), ("pre_task_scores", before)):
        if recovered != result[key]:
            raise ValueError(f"Recorded {key} differs from predictions: {directory}")
    if any(not math.isclose(metrics[key], value, abs_tol=1e-10) for key, value in result["metrics"].items()):
        raise ValueError(f"Recorded metrics differ: {directory}")
    plasticity = [100 * (matrix[i][i] - before[i]) for i in range(len(tasks))]
    if plasticity != result["plasticity_pp"]:
        raise ValueError("Recorded learning gains differ")
    # Only the latest attempt per task counts toward completed training exposure.
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    attempts, ended = {}, {}
    for event in events:
        task = event["task"]
        if event["event"] == "task_start":
            attempts[task] = []
            ended.pop(task, None)
        elif event["event"] == "train":
            attempts[task].append(event)
        elif event["event"] == "task_end":
            ended[task] = event
    if set(ended) != set(range(len(tasks))):
        raise ValueError(f"Task-end events incomplete: {directory}")
    for task, end in ended.items():
        observed = attempts[task]
        expected = list(range(1, config["epochs"] + 1)) if end["trained"] else []
        if [row["epoch"] for row in observed] != expected:
            raise ValueError("Incomplete/duplicated training phase")
        examples = len(tasks[task]["train"])
        if any(row["examples"] != examples or row["steps"] != math.ceil(examples / config["batch_size"]) for row in observed):
            raise ValueError("Training exposure does not match declared task")
    return {"directory": str(directory), "config": config, "metrics": metrics, "matrix": matrix,
            "oracle_metrics": continual_metrics(oracle) if all(oracle) else None,
            "mean_plasticity_pp": statistics.mean(plasticity), "plasticity_pp": plasticity,
            "expert_count": result["expert_count"], "parameter_counts": result["parameter_counts"],
            "router_memory_bytes": result["router_memory_bytes"], "replay_examples": result["replay_examples"],
            "training_examples": sum(row["examples"] for rows in attempts.values() for row in rows),
            "optimizer_steps": sum(row["steps"] for rows in attempts.values() for row in rows),
            "completed_tasks": len(tasks), "full_protocol": len(tasks) == len(manifest["tasks"]),
            "audited_predictions": count, "final_answer_counts": dict(final_counts),
            "largest_answer_fraction": max(final_counts.values()) / sum(final_counts.values()),
            "final_task_reports": final_task_reports}


def summarize(runs):
    groups = defaultdict(list)
    fields = ("protocol", "eval_split", "method", "lr", "epochs", "rank", "alpha",
              "router_modality", "router_text", "router_memory", "router_ridge", "growth_margin",
              "replay_per_task", "probe_size", "image_size", "batch_size", "micro_batch_size",
              "manifest_sha256", "task_limit")
    for run in runs:
        config = run["config"]
        key = tuple(config[field] for field in fields)
        groups[key].append(run)
    summaries = []
    for key, values in sorted(groups.items()):
        if len({row["config"]["seed"] for row in values}) != len(values):
            raise ValueError("Duplicate seed within scientific condition")
        if len({json.dumps(row["config"]["source_sha256"], sort_keys=True) for row in values}) != 1:
            raise ValueError("Mixed source revisions within condition")
        record = dict(zip(fields, key))
        record["seeds"] = [row["config"]["seed"] for row in values]
        for metric in ("AP", "AF", "BWT", "MAP"):
            numbers = [row["metrics"][metric] for row in values]
            record[metric] = statistics.mean(numbers)
            record[metric + "_sd"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        record["mean_plasticity_pp"] = statistics.mean(row["mean_plasticity_pp"] for row in values)
        record["expert_counts"] = [row["expert_count"] for row in values]
        record["largest_answer_fractions"] = [row["largest_answer_fraction"] for row in values]
        record["oracle_AP"] = statistics.mean(row["oracle_metrics"]["AP"] for row in values) if all(row["oracle_metrics"] for row in values) else None
        summaries.append(record)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted({path.parent for root in args.runs for path in root.rglob("config.json")})
    if not directories:
        raise ValueError("No declared runs found")
    runs = [audit(directory) for directory in directories]
    summaries = summarize(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"runs": runs, "audited_predictions": sum(row["audited_predictions"] for row in runs)}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"runs": len(runs), "audited_predictions": report["audited_predictions"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
