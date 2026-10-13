"""Independently rescore every declared inference-only routing ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from sema_vqa.metrics import continual_metrics
from scripts.summarize_readout import rescore


def audit(directory):
    report = json.loads((directory / "routing_ablation.json").read_text())
    config = report["parent_config"]
    raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Routing ablation manifest changed")
    parent = Path(report["parent"])
    if json.loads((parent / "config.json").read_text()) != config:
        raise ValueError("Routing ablation parent configuration changed")
    if hashlib.sha256((parent / "results.json").read_bytes()).hexdigest() != report["parent_results_sha256"]:
        raise ValueError("Routing ablation parent results changed")
    if hashlib.sha256((parent / "latest.pt").read_bytes()).hexdigest() != report["parent_checkpoint_sha256"]:
        raise ValueError("Routing ablation parent expert weights changed")
    tasks = json.loads(raw)["tasks"][:config["task_limit"] or None]
    expected_files = len(tasks) + len(tasks) * (len(tasks) + 1) // 2
    if report["native_prediction_files_byte_identical"] != expected_files or report["expert_retraining_steps"] != 0:
        raise ValueError("Native stage reconstruction not fully verified")
    native = f"{config['router_text']}_{config['router_modality']}"
    summaries, count = [], 0
    for variant, state in report["variants"].items():
        before, matrix = [], []
        final_answers = Counter()
        directory_variant = directory / variant
        for stage, task in enumerate(tasks):
            path = directory_variant / f"predictions_before_{stage:02d}.jsonl"
            score, records = rescore(path, task[config["eval_split"]], task["metric"])
            before.append(score)
            count += len(records)
            if variant == native and path.read_bytes() != (parent / path.name).read_bytes():
                raise ValueError("Native pre-task prediction parity changed")
            scores = []
            for previous in tasks[:stage + 1]:
                path = directory_variant / f"predictions_after_{stage:02d}_{previous['name']}.jsonl"
                score, records = rescore(path, previous[config["eval_split"]], previous["metric"])
                scores.append(score)
                count += len(records)
                if stage == len(tasks) - 1:
                    final_answers.update(record["prediction"] for record in records)
                if variant == native and path.read_bytes() != (parent / path.name).read_bytes():
                    raise ValueError("Native after-task prediction parity changed")
            matrix.append(scores)
        metrics = continual_metrics(matrix)
        if matrix != state["matrix"] or before != state["pre_task_scores"] or metrics != state["metrics"]:
            raise ValueError("Routing ablation reported metrics differ from raw predictions")
        plasticity = [100 * (row[i] - before[i]) for i, row in enumerate(matrix)]
        largest_fraction = max(final_answers.values()) / sum(final_answers.values())
        if (plasticity != state["plasticity_pp"] or dict(final_answers) != state["final_answer_counts"]
                or largest_fraction != state["largest_answer_fraction"]):
            raise ValueError("Routing learning/answer-distribution statistics differ")
        summaries.append({"directory": str(directory_variant), "parent": str(parent), "parent_config": config,
                          "variant": variant, "metrics": metrics, "matrix": matrix,
                          "post_first_plasticity_pp": statistics.mean(plasticity[1:] or plasticity),
                          "largest_answer_fraction": largest_fraction,
                          "expert_retraining_steps": 0})
    return {"runs": summaries, "audited_predictions": count, "native_files_reproduced": expected_files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if len({job["output"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate declared routing ablation")
    reports = [audit(Path(job["output"])) for job in jobs]
    result = {"runs": [run for report in reports for run in report["runs"]],
              "audited_predictions": sum(report["audited_predictions"] for report in reports),
              "independent_training_runs": 0, "parent_banks": len(reports)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parent_banks": len(reports), "audited_predictions": result["audited_predictions"],
                      "summary": [{"protocol": r["parent_config"]["protocol"], "lr": r["parent_config"]["lr"],
                                   "epochs": r["parent_config"]["epochs"], "variant": r["variant"], **r["metrics"]}
                                  for r in result["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
