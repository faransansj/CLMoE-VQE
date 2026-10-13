"""Re-score raw predictions; select hyperparameters using validation ONLY."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from sema_vqa.metrics import continual_metrics, score_prediction
from scripts.vision_array import calibration_jobs


def audit(directory):
    config = json.loads((directory / "config.json").read_text())
    raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"Manifest changed: {directory}")
    manifest = json.loads(raw)
    tasks = manifest["tasks"]
    result = json.loads((directory / "results.json").read_text())
    if result["completed"] != len(tasks) or len(result["matrix"]) != len(tasks):
        raise ValueError(f"Incomplete sequence: {directory}")
    recovered, count = [], 0
    final_answers = Counter()
    for stage, stored in enumerate(result["matrix"]):
        if len(stored) != stage + 1:
            raise ValueError("Nontriangular result matrix")
        scores = []
        for task in tasks[:stage + 1]:
            rows = {row["id"]: row for row in task[config["eval_split"]]}
            predictions = [json.loads(line) for line in (directory / f"predictions_after_{stage:02d}_{task['name']}.jsonl").read_text().splitlines()]
            ids = [prediction["id"] for prediction in predictions]
            if len(rows) != len(task[config["eval_split"]]) or len(ids) != len(set(ids)) or set(ids) != set(rows):
                raise ValueError("Prediction ID coverage mismatch")
            actual = [score_prediction(pred["prediction"], rows[pred["id"]], task["metric"]) for pred in predictions]
            if any(not math.isclose(pred["score"], score, abs_tol=1e-12) for pred, score in zip(predictions, actual)):
                raise ValueError("Stored per-example score differs")
            scores.append(statistics.mean(actual))
            count += len(predictions)
            if stage == len(tasks) - 1:
                final_answers.update(p["prediction"] for p in predictions)
        if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(scores, stored)):
            raise ValueError("Stored matrix differs from re-scored predictions")
        recovered.append(scores)
    metrics = continual_metrics(recovered)
    if any(not math.isclose(value, result["metrics"][key], abs_tol=1e-10) for key, value in metrics.items()):
        raise ValueError("Stored aggregate metrics differ")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    exposure = config.get("exposure_match")
    initial_match = None
    if exposure:
        source = Path(exposure["source_run"])
        for name, digest in exposure["source_sha256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Exposure source changed after fitting: {name}")
        first = f"predictions_after_00_{tasks[0]['name']}.jsonl"
        initial_match = (directory / first).read_bytes() == (source / first).read_bytes()
        if not initial_match:
            raise ValueError("Exposure control initial-task predictions differ from parent SEMA")
    return {"cohort": directory.parents[2].name,
            "protocol": manifest["protocol"], "method": config["method"], "seed": config["seed"],
            "lr": config["lr"], "threshold": config["threshold"], "eval_split": config["eval_split"],
            "epochs": config["epochs"],
            "manifest_sha256": config["manifest_sha256"], "source_sha256": config["source_sha256"],
            "matrix": recovered, "metrics": metrics, "experts": result["experts"], "parameters": result["parameters"],
            "final_answer_counts": dict(final_answers.most_common()),
            "exposure_match": exposure, "initial_predictions_match_parent": initial_match,
            "prediction_count": count, "functional_phases": {str(i): sorted({e["phase"] for e in events if e["event"] == "train" and e["task"] == i}) for i in range(len(tasks))},
            "directory": str(directory)}


def select(runs, plan):
    if any(run["eval_split"] != "validation" for run in runs):
        raise ValueError("Selection from official eval is prohibited")
    expected = {(j["protocol"], j["method"], j["seed"], j["lr"], j["threshold"], j["epochs"]) for j in calibration_jobs(plan)}
    observed = {(r["protocol"], r["method"], r["seed"], r["lr"], r["threshold"], r["epochs"]) for r in runs}
    if expected != observed or len(runs) != len(expected):
        raise ValueError(f"Calibration grid differs: missing {expected - observed}, extra {observed - expected}")
    scope = plan["calibration"].get("selection_scope", "shared")
    if scope not in ("shared", "independent"):
        raise ValueError(f"Unknown selection scope: {scope}")
    selected, evidence = {}, {}
    for protocol in plan["protocols"]:
        subset = [r for r in runs if r["protocol"] == protocol]
        baseline = max((r for r in subset if r["method"] == "sequential"), key=lambda r: (r["metrics"]["AP"], -r["metrics"]["AF"], -r["epochs"], r["lr"]))
        candidates = [r for r in subset if r["method"] == "sema" and
                      (scope == "independent" or (r["lr"] == baseline["lr"] and r["epochs"] == baseline["epochs"]))]
        eligible = [r for r in candidates if r["metrics"]["AF"] <= baseline["metrics"]["AF"]]
        winner = max(eligible or candidates, key=lambda r:
                     (r["metrics"]["AP"], -r["metrics"]["AF"], r["threshold"], -r["epochs"], -r["lr"])
                     if scope == "independent" else (r["metrics"]["AP"], r["threshold"]))
        selected[protocol] = {"lr": winner["lr"], "threshold": winner["threshold"], "epochs": winner["epochs"]}
        evidence[protocol] = {"baseline": baseline["metrics"], "sema": winner["metrics"],
                              "baseline_setting": {k: baseline[k] for k in ("lr", "threshold", "epochs")},
                              "joint_criterion": winner["metrics"]["AP"] > baseline["metrics"]["AP"] and winner["metrics"]["AF"] < baseline["metrics"]["AF"]}
    return {"selected": selected, "validation_evidence": evidence, "rule": plan["calibration"]["selection"],
            "selection_scope": scope, "eval_split": "validation"}


def compare(runs, plan):
    """Paired seed comparisons against the strongest validation-selected baseline."""
    if any(r["eval_split"] != "eval" for r in runs):
        raise ValueError("Confirmation comparison requires official eval predictions")
    confirmation = plan["confirmation"]
    seeds = confirmation["seeds"]
    cohorts = ("confirmation", "confirmation_independent")
    expected = set()
    for protocol in plan["protocols"]:
        for cohort in cohorts:
            methods = confirmation["primary_methods"] if cohort == "confirmation" else ["sema"]
            expected.update((cohort, protocol, method, seed) for method in methods for seed in seeds)
            expected.update((cohort, protocol, method, 17) for method in confirmation["ablation_methods_seed17"])
    index = {(r["cohort"], r["protocol"], r["method"], r["seed"]): r for r in runs}
    if set(index) != expected or len(index) != len(runs):
        raise ValueError("Incomplete, duplicated, or unexpected confirmation grid")

    def stats(values):
        return {"mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None}

    comparisons = []
    for protocol in plan["protocols"]:
        baseline = [index["confirmation", protocol, "sequential", seed] for seed in seeds]
        for cohort in cohorts:
            sema = [index[cohort, protocol, "sema", seed] for seed in seeds]
            entry = {"protocol": protocol, "cohort": cohort, "baseline": {}, "sema": {}, "paired_delta": {}}
            for metric in ("AP", "AF", "BWT", "MAP", "task_end_AP"):
                def value(run):
                    if metric == "task_end_AP":
                        return 100 * statistics.mean(row[i] for i, row in enumerate(run["matrix"]))
                    return run["metrics"][metric]
                b = [value(r) for r in baseline]
                s = [value(r) for r in sema]
                delta = [after - before for before, after in zip(b, s)]
                entry["baseline"][metric] = stats(b)
                entry["sema"][metric] = stats(s)
                entry["paired_delta"][metric] = stats(delta) | {"by_seed": dict(zip(map(str, seeds), delta))}
            entry["expert_additions_by_seed"] = {str(seed): sum(r["experts"]) - 3 for seed, r in zip(seeds, sema)}
            entry["functional_phases_by_seed"] = {
                str(seed): {task: len(phases) for task, phases in r["functional_phases"].items()}
                for seed, r in zip(seeds, sema)}
            entry["final_generation_by_seed"] = {}
            for seed, run in zip(seeds, sema):
                counts = run["final_answer_counts"]
                answer, frequency = max(counts.items(), key=lambda item: item[1])
                entry["final_generation_by_seed"][str(seed)] = {
                    "distinct_answers": len(counts), "most_common_answer": answer,
                    "most_common_fraction": frequency / sum(counts.values())}
            entry["joint_criterion"] = entry["paired_delta"]["AP"]["mean"] > 0 and entry["paired_delta"]["AF"]["mean"] < 0
            comparisons.append(entry)
    ablations = [{k: r[k] for k in ("protocol", "cohort", "method", "seed", "lr", "epochs", "metrics", "experts")}
                 for r in runs if r["method"] in confirmation["ablation_methods_seed17"]]
    return {"seeds": seeds, "comparisons": comparisons, "ablations": ablations,
            "interpretation": "Delta = SEMA minus strongest validation-selected sequential. Lower AF is better. task_end_AP is the mean matrix diagonal, not a pre/post learning gain. Mean joint improvement is not statistical significance or proof that expansion caused it; inspect no-expansion ablation, task-end accuracy and training exposures."}


def compare_exposure(controls, parents, plan):
    expected_parents = {(group, protocol, seed) for group in plan["source_cohorts"]
                        for protocol in plan["protocols"] for seed in plan["confirmation"]["seeds"]}
    selected = [r for r in parents if r["method"] == "sema" and
                (r["cohort"], r["protocol"], r["seed"]) in expected_parents]
    parent_index = {(r["cohort"], r["protocol"], r["seed"]): r for r in selected}
    if set(parent_index) != expected_parents or len(selected) != len(expected_parents):
        raise ValueError("Incomplete or duplicated parent SEMA grid")
    by_directory = {r["directory"]: r for r in selected}
    methods = plan["confirmation"]["primary_methods"]
    expected_controls = {key + (method,) for key in expected_parents for method in methods}
    index = {}
    for control in controls:
        exposure = control["exposure_match"]
        if not exposure or exposure["source_run"] not in by_directory:
            raise ValueError("Control has no matching audited SEMA parent")
        parent = by_directory[exposure["source_run"]]
        if control["eval_split"] != "eval" or parent["eval_split"] != "eval":
            raise ValueError("Exposure comparison requires official eval")
        for key in ("manifest_sha256", "seed", "lr", "epochs", "threshold"):
            if control[key] != parent[key]:
                raise ValueError(f"Exposure comparison mismatch: {key}")
        if parent["source_sha256"]["sema_vqa/vision_run.py"] != plan["source_runner_before_extension_sha256"]:
            raise ValueError("Unexpected parent runner revision")
        for name, digest in parent["source_sha256"].items():
            if name != "sema_vqa/vision_run.py" and control["source_sha256"][name] != digest:
                raise ValueError("Exposure comparison model/scorer differs")
        counts = exposure["phase_counts"]
        for run in (parent, control):
            if [len(run["functional_phases"][str(i)]) for i in range(len(run["matrix"]))] != counts:
                raise ValueError("Actual functional phase counts do not match")
        if control["initial_predictions_match_parent"] is not True:
            raise ValueError("Initial-task equivalence was not verified")
        index[parent["cohort"], parent["protocol"], parent["seed"], control["method"]] = control
    if set(index) != expected_controls or len(index) != len(controls):
        raise ValueError("Incomplete, duplicated, or unexpected exposure controls")

    def stats(values):
        return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}

    comparisons = []
    seeds = plan["confirmation"]["seeds"]
    for cohort in plan["source_cohorts"]:
        for protocol in plan["protocols"]:
            sema = [parent_index[cohort, protocol, seed] for seed in seeds]
            for method in methods:
                matched = [index[cohort, protocol, seed, method] for seed in seeds]
                entry = {"source_cohort": cohort, "protocol": protocol, "control_method": method,
                         "sema": {}, "control": {}, "paired_delta": {}, "initial_predictions_identical": True}
                for metric in ("AP", "AF", "BWT", "MAP", "task_end_AP"):
                    def value(run):
                        return (100 * statistics.mean(row[i] for i, row in enumerate(run["matrix"]))
                                if metric == "task_end_AP" else run["metrics"][metric])
                    s, c = [value(r) for r in sema], [value(r) for r in matched]
                    delta = [a - b for a, b in zip(s, c)]
                    entry["sema"][metric], entry["control"][metric] = stats(s), stats(c)
                    entry["paired_delta"][metric] = stats(delta) | {"by_seed": dict(zip(map(str, seeds), delta))}
                entry["joint_criterion"] = entry["paired_delta"]["AP"]["mean"] > 0 and entry["paired_delta"]["AF"]["mean"] < 0
                comparisons.append(entry)
    return {"seeds": seeds, "comparisons": comparisons,
            "interpretation": "SEMA minus phase-matched control, not minus the original independently optimized sequential baseline. Matching covers functional example exposure and readout optimizer steps, not total FLOPs or RD work. Model/scorer hashes and all initial-task predictions match; the archived parent runner predates only the exposure-control extension."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--compare", action="store_true")
    mode.add_argument("--compare-exposure", help="Previously audited parent confirmation JSON")
    parser.add_argument("--plan", default="configs/vision_poc.json")
    args = parser.parse_args()
    paths = sorted(p for root in args.runs for p in Path(root).rglob("results.json"))
    if len(paths) != len(set(p.resolve() for p in paths)):
        raise ValueError("Overlapping run roots")
    runs = [audit(p.parent) for p in paths]
    if not runs:
        raise ValueError("No completed runs")
    if len({json.dumps(r["source_sha256"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("Runs used different scientific code")
    for protocol in {r["protocol"] for r in runs}:
        if len({r["manifest_sha256"] for r in runs if r["protocol"] == protocol}) != 1:
            raise ValueError("Different task subsets across runs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"runs": runs, "audited_predictions": sum(r["prediction_count"] for r in runs)}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cohort", "protocol", "method", "seed", "lr", "threshold", "epochs", "split", "AP", "AF", "BWT", "MAP", "experts"])
        for r in runs:
            writer.writerow([r["cohort"], r["protocol"], r["method"], r["seed"], r["lr"], r["threshold"], r["epochs"], r["eval_split"],
                             *[r["metrics"][m] for m in ("AP", "AF", "BWT", "MAP")], r["experts"]])
    if args.select:
        selection = select(runs, json.loads(Path(args.plan).read_text()))
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
    elif args.compare:
        comparison = compare(runs, json.loads(Path(args.plan).read_text()))
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))
    elif args.compare_exposure:
        parents = json.loads(Path(args.compare_exposure).read_text())["runs"]
        comparison = compare_exposure(runs, parents, json.loads(Path(args.plan).read_text()))
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))
    else:
        print(json.dumps({"runs": len(runs), "audited_predictions": report["audited_predictions"]}))


if __name__ == "__main__":
    main()
