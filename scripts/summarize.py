"""Audit every prediction against declared held-out rows before comparing methods."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from sema_vqa.metrics import continual_metrics, score_prediction


def audit(directory):
    config = json.loads((directory / "config.json").read_text())
    manifest_path = Path(config["manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"Manifest changed: {directory}")
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    result = json.loads((directory / "results.json").read_text())
    if result["completed"] != len(tasks) or len(result["matrix"]) != len(tasks):
        raise ValueError(f"Incomplete task sequence: {directory}")
    recovered = []
    for stage in range(len(tasks)):
        scores = []
        for task in tasks[:stage + 1]:
            rows = {row["id"]: row for row in task["eval"]}
            if len(rows) != len(task["eval"]):
                raise ValueError("Duplicate evaluation IDs")
            path = directory / f"predictions_after_{stage:02d}_{task['name']}.jsonl"
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            ids = [row["id"] for row in predictions]
            if len(ids) != len(set(ids)) or set(ids) != set(rows):
                raise ValueError(f"Prediction coverage differs: {path}")
            scores.append(statistics.mean(score_prediction(pred["prediction"], rows[pred["id"]], task["metric"])
                                          for pred in predictions))
        for actual, stored in zip(scores, result["matrix"][stage]):
            if not math.isclose(actual, stored, abs_tol=1e-12):
                raise ValueError(f"Recorded matrix differs from predictions: {directory}")
        recovered.append(scores)
    return {"protocol": manifest["protocol"], "method": config["method"], "seed": config["seed"],
            "manifest_sha256": config["manifest_sha256"], "metrics": continual_metrics(recovered),
            "experts": result["experts"], "adapter_parameters": result["adapter_parameters"],
            "directory": str(directory), "matrix": recovered}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--plan", default="configs/poc.json")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    runs = [audit(path.parent) for path in sorted(Path(args.runs).glob("*/*/seed*/results.json"))]
    expected = {(protocol, method, seed) for protocol in plan["protocols"]
                for method in plan["primary_methods"] for seed in plan["seeds"]}
    expected |= {(protocol, method, 17) for protocol in plan["protocols"]
                 for method in plan["ablation_methods_seed17"]}
    indexed = {(run["protocol"], run["method"], run["seed"]): run for run in runs}
    if set(indexed) != expected or len(indexed) != len(runs):
        raise ValueError(f"Run coverage differs; missing={expected - set(indexed)}, extra={set(indexed) - expected}")
    comparisons = []
    for protocol in plan["protocols"]:
        selected = [run for run in runs if run["protocol"] == protocol]
        if len({run["manifest_sha256"] for run in selected}) != 1:
            raise ValueError(f"Methods used different subsets: {protocol}")
        deltas = {metric: [indexed[protocol, "sema", seed]["metrics"][metric]
                           - indexed[protocol, "sequential", seed]["metrics"][metric]
                           for seed in plan["seeds"]] for metric in ("AP", "AF")}
        means = {metric: statistics.mean(values) for metric, values in deltas.items()}
        comparisons.append({"protocol": protocol, "paired_seed_deltas_pp": deltas,
                            "mean_delta_pp": means,
                            "seed_sd_pp": {metric: statistics.stdev(values) for metric, values in deltas.items()},
                            "supports_directional_poc": means["AP"] > 0 and means["AF"] < 0})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"runs": runs, "comparisons": comparisons,
              "supports_both_protocols": all(row["supports_directional_poc"] for row in comparisons),
              "claim_limit": "Fixed-subset three-seed POC; not full-benchmark replication or statistical proof."}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["protocol", "method", "seed", "AP", "AF", "BWT", "MAP", "experts", "parameters"])
        for run in runs:
            writer.writerow([run["protocol"], run["method"], run["seed"],
                             *[run["metrics"][metric] for metric in ("AP", "AF", "BWT", "MAP")],
                             run["experts"], run["adapter_parameters"]])
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
