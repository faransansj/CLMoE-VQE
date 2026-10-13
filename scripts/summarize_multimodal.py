"""Independently audit multimodal predictions and retain each metric's native units.

Optional normalization requires a separately fixed G1 validation artifact:
{"schema_version": 1, "eval_split": "validation", "manifest_sha256": "...",
 "scores": {"V": 50.0, "C": 3.0, ...}} and its explicit SHA256 on the CLI.
No raw heterogeneous AP is computed and normalization never clips at one.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from sema_multimodal.data import validate_manifest
from sema_multimodal.metrics import TaskScorer, continual_metrics


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_same(actual, recorded, label):
    """Reject nonfinite scores, changed auxiliary metrics, IDs and shapes."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(actual) != set(recorded):
            raise ValueError(f"Recorded {label} keys differ")
        for key in actual:
            _assert_same(actual[key], recorded[key], f"{label}/{key}")
    elif isinstance(actual, list):
        if not isinstance(recorded, list) or len(actual) != len(recorded):
            raise ValueError(f"Recorded {label} shape differs")
        for index, (left, right) in enumerate(zip(actual, recorded)):
            _assert_same(left, right, f"{label}/{index}")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                or not math.isfinite(actual) or not math.isfinite(recorded)
                or not math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError(f"Recorded {label} score differs")
    elif actual != recorded:
        raise ValueError(f"Recorded {label} differs")


def rescore(path, rows, scorer):
    """Require exactly one prediction per manifest image; order by manifest ID."""
    path = Path(path)
    expected = {row["id"]: row for row in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("Evaluation corpus must have nonempty unique row IDs")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if any(not isinstance(record, dict) or not isinstance(record.get("id"), str)
           or not isinstance(record.get("prediction"), str) for record in records):
        raise ValueError(f"Malformed prediction record: {path}")
    observed = {record["id"]: record for record in records}
    if len(records) != len(rows) or len(observed) != len(records) or set(observed) != set(expected):
        raise ValueError(f"Missing/duplicate/mismatched prediction IDs: {path}")
    ordered = [observed[row["id"]] for row in rows]
    for record, row in zip(ordered, rows):
        if "image_sha256" in record and record["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"Prediction image hash mismatch: {path}/{row['id']}")
    scores = scorer.score([record["prediction"] for record in ordered], rows)
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError(f"Nonfinite corpus metric: {path}")
    return scores, ordered


def _resolve(path, base=ROOT):
    path = Path(path).expanduser()
    return path if path.is_absolute() else base / path


def _verify_hashes(mapping, label, base=ROOT):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Missing {label} file hash inventory")
    for name, digest in mapping.items():
        path = _resolve(name, base)
        if not isinstance(digest, str) or _sha256(path) != digest:
            raise ValueError(f"Changed {label} file: {path}")


def _generation_report(records):
    counts = Counter(record["prediction"] for record in records)
    generation = [record.get("generation", {}) for record in records]
    report = {"examples": len(records), "empty_fraction": sum(not record["prediction"].strip() for record in records) / len(records),
        "largest_output_fraction": max(counts.values()) / len(records), "output_counts": dict(counts)}
    for key in ("eos", "repeated_trigram", "limit_reached"):
        if all(type(item.get(key)) is bool for item in generation):
            report[key + "_fraction"] = sum(item[key] for item in generation) / len(records)
    if all(type(item.get("generated_tokens")) is int for item in generation):
        report["mean_generated_tokens"] = math.fsum(item["generated_tokens"] for item in generation) / len(records)
    return report


def _check_evaluation(recorded, task, path, scoring, records):
    """Audit the saved corpus auxiliaries and generation/routing distributions."""
    generation = {}
    for key in ("generated_tokens", "eos", "empty", "repeated_trigram", "limit_reached"):
        values = [record["generation"][key] for record in records]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid saved generation diagnostic: {path}")
        generation[key] = math.fsum(values) / len(records)
    predictions = Counter(record["prediction"] for record in records)
    expected = {"task_id": task["task_id"], "family": task["family"], "metric": task["metric"],
        "scoring": scoring, "generation": generation, "prediction_counts": dict(predictions),
        "most_frequent_output_fraction": max(predictions.values()) / len(records),
        "route_counts": dict(Counter(str(record["expert"]) for record in records)), "predictions_file": path.name}
    for key, value in expected.items():
        _assert_same(value, recorded[key], f"{path.name}/{key}")


def audit(directory, *, normalizer_lock=None, image_root=None, asset_root=None):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    if config.get("schema_version") != 1 or result.get("schema_version") != 1:
        raise ValueError("Expected multimodal run schema_version=1")
    manifest_path = _resolve(config["manifest"])
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {directory}")
    manifest = json.loads(raw)
    actual_image_root = image_root if image_root is not None else config.get("image_root")
    validate_manifest(manifest, image_root=actual_image_root)
    if config["protocol"] != manifest["protocol"]:
        raise ValueError("Run and manifest protocol differ")
    _verify_hashes(config["source_sha256"], "source")
    actual_asset_root = _resolve(asset_root if asset_root is not None else config["assets"])
    _verify_hashes(config["assets_sha256"], "asset", actual_asset_root)
    _assert_same({key: value for key, value in config["assets_sha256"].items() if key.startswith("tokenizer/")},
                 config["tokenizer_sha256"], "tokenizer_sha256")
    if not config["tokenizer_sha256"]:
        raise ValueError("Missing tokenizer asset hashes")
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    order = config["task_order"]
    if not order or len(set(order)) != len(order) or any(task_id not in by_id for task_id in order):
        raise ValueError("Run has invalid/duplicate task order")
    if any(not task_id.replace("-", "").replace("_", "").isalnum() for task_id in order):
        raise ValueError("Unsafe prediction filename task ID")
    tasks = [by_id[task_id] for task_id in order]
    _assert_same(order, result["task_order"], "task_order")
    _assert_same([{key: task[key] for key in ("task_id", "family", "metric")} for task in tasks],
                 result["task_metadata"], "task_metadata")
    split = config["eval_split"]
    if split not in {"validation", "eval"}:
        raise ValueError("Rescoring requires validation or eval rows")
    limit = config.get("eval_limit", 0)
    if type(limit) is not int or limit < 0 or (limit and (not config.get("smoke_output") or split != "validation")):
        raise ValueError("Evaluation prefixes are only valid for explicit validation smoke")
    evaluation_rows = {task["task_id"]: task[split][:limit or None] for task in tasks}
    scorers = {task["task_id"]: TaskScorer(task) for task in tasks}
    _assert_same({task_id: scorer.lock for task_id, scorer in scorers.items()}, config["scorer_locks"], "scorer_locks")
    n = len(tasks)
    if (result.get("status") != "complete" or result.get("completed") != n
            or len(result["matrix"]) != n or len(result["pre_task_scores"]) != n
            or len(result["stage_evaluations"]) != n or len(result["oracle_matrix"]) != n):
        raise ValueError("Incomplete declared task sequence")
    matrix, before, oracle, evaluations, diagnostics = [], [], [], [], []
    count = 0
    use_oracle = config.get("oracle_diagnostic", False) and config["method"] == "expand"
    scored_files = {}
    for stage, task in enumerate(tasks):
        task_id = task["task_id"]
        before_path = directory / f"predictions_before_{stage:02d}_{task_id}.jsonl"
        scoring, records = rescore(before_path, evaluation_rows[task_id], scorers[task_id])
        scored_files[before_path.name] = (task, scoring, records)
        before.append(scoring["score"])
        count += len(records)
        stage_scores, oracle_scores = [], []
        evaluations.append({"stage": stage, "kind": "before", "task_id": task_id, "scoring": scoring})
        recorded_stage = result["stage_evaluations"][stage]
        expected_count = stage + 1 + int(config.get("image_shuffled_diagnostic", False))
        if len(recorded_stage) != expected_count:
            raise ValueError("Missing/extra stage corpus evaluations")
        for seen, previous in enumerate(tasks[:stage + 1]):
            previous_id = previous["task_id"]
            after_path = directory / f"predictions_after_{stage:02d}_{previous_id}.jsonl"
            scoring, records = rescore(after_path, evaluation_rows[previous_id], scorers[previous_id])
            scored_files[after_path.name] = (previous, scoring, records)
            _check_evaluation(recorded_stage[seen], previous, after_path, scoring, records)
            stage_scores.append(scoring["score"])
            count += len(records)
            evaluations.append({"stage": stage, "kind": "after", "task_id": previous_id, "scoring": scoring})
            diagnostics.append({"stage": stage, "task_id": previous_id, "family": previous["family"],
                                **_generation_report(records)})
            if use_oracle:
                oracle_path = directory / f"oracle_after_{stage:02d}_{previous_id}.jsonl"
                scoring, records = rescore(oracle_path, evaluation_rows[previous_id], scorers[previous_id])
                scored_files[oracle_path.name] = (previous, scoring, records)
                oracle_scores.append(scoring["score"])
                count += len(records)
                evaluations.append({"stage": stage, "kind": "oracle", "task_id": previous_id, "scoring": scoring})
        if config.get("image_shuffled_diagnostic", False):
            shuffled_path = directory / f"image_shuffled_{stage:02d}_{task_id}.jsonl"
            scoring, records = rescore(shuffled_path, evaluation_rows[task_id], scorers[task_id])
            if recorded_stage[-1].get("diagnostic") != "image_shuffled":
                raise ValueError("Missing image-shuffled diagnostic declaration")
            _check_evaluation(recorded_stage[-1], task, shuffled_path, scoring, records)
            scored_files[shuffled_path.name] = (task, scoring, records)
            count += len(records)
            evaluations.append({"stage": stage, "kind": "image_shuffled", "task_id": task_id, "scoring": scoring})
        matrix.append(stage_scores)
        oracle.append(oracle_scores)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    audited_files = set()
    for event in events:
        recorded = event.get("pre_evaluation") if event.get("event") == "task_start" else event
        if not recorded or "predictions_file" not in recorded:
            continue
        filename = recorded["predictions_file"]
        if filename not in scored_files:
            raise ValueError(f"Undeclared event prediction file: {filename}")
        task, scoring, records = scored_files[filename]
        _check_evaluation(recorded, task, directory / filename, scoring, records)
        audited_files.add(filename)
    if audited_files != set(scored_files):
        raise ValueError("Missing recorded evaluation events")
    _assert_same(matrix, result["matrix"], "matrix")
    _assert_same(before, result["pre_task_scores"], "pre_task_scores")
    _assert_same(oracle, result["oracle_matrix"], "oracle_matrix")
    gains = [matrix[index][index] - before[index] for index in range(n)]
    _assert_same(gains, result["pre_to_post_gain"], "pre_to_post_gain")
    normalizers = None
    if normalizer_lock is not None:
        if (normalizer_lock.get("schema_version") != 1 or normalizer_lock.get("eval_split") != "validation"
                or normalizer_lock.get("manifest_sha256") != config["manifest_sha256"]):
            raise ValueError("Normalizer must be a fixed matching G1 validation artifact")
        constants = normalizer_lock["scores"]
        if not isinstance(constants, dict) or set(constants) != set(by_id):
            raise ValueError("Normalizer artifact must cover the full manifest task set")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in constants.values()):
            raise ValueError("Every full-manifest normalizer must be finite and positive")
        normalizers = {task_id: constants[task_id] for task_id in order}
    metrics = continual_metrics(matrix, tasks, pre_task_scores=before, normalizers=normalizers)
    native_metrics = continual_metrics(matrix, tasks, pre_task_scores=before) if normalizers is not None else metrics
    _assert_same(native_metrics, result["metrics"], "metrics")
    return {"directory": str(directory), "config": config, "matrix": matrix,
        "pre_task_scores": before, "pre_to_post_gain": gains, "oracle_matrix": oracle,
        "metrics": metrics, "oracle_metrics": continual_metrics(oracle, tasks, normalizers=normalizers) if use_oracle else None,
        "corpus_evaluations": evaluations, "generation": diagnostics, "audited_predictions": count,
        "image_bytes_verified": actual_image_root is not None,
        "stage_resources": result["stage_resources"]}


def _check_output(path):
    path = Path(path).expanduser().resolve()
    forbidden = ("readout", "vision", "legacy", "sema_runs", "vqa_runs")
    if any(part.casefold().startswith(forbidden) for part in path.parts):
        raise ValueError("Archived artifact namespaces are read-only")
    report_root = (ROOT / "artifacts" / "multimodal_report").resolve()
    if path.is_relative_to(report_root):
        return path
    if path == ROOT or path in ROOT.parents or path.is_relative_to(ROOT):
        raise ValueError("Reports must stay under artifacts/multimodal_report, outside source/run evidence")
    if not any("multimodal" in part.casefold() for part in path.parts):
        raise ValueError("External report output must use an explicit multimodal namespace")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/multimodal_report"))
    parser.add_argument("--image-root", type=Path, help="Override image location; real SHA256/decode verification")
    parser.add_argument("--asset-root", type=Path, help="Override base for relative model asset hash paths")
    parser.add_argument("--normalizers", type=Path)
    parser.add_argument("--normalizers-sha256", help="Explicit pre-fixed hash; required with --normalizers")
    args = parser.parse_args()
    if bool(args.normalizers) != bool(args.normalizers_sha256):
        parser.error("--normalizers and --normalizers-sha256 must be supplied together")
    normalizer_lock = None
    if args.normalizers is not None:
        if _sha256(args.normalizers) != args.normalizers_sha256:
            raise ValueError("Fixed normalizer file hash mismatch")
        normalizer_lock = json.loads(args.normalizers.read_text())
    directories = sorted({path.parent for root in args.runs for path in root.rglob("config.json")})
    if not directories:
        raise ValueError("No declared multimodal runs found")
    runs = [audit(directory, normalizer_lock=normalizer_lock, image_root=args.image_root, asset_root=args.asset_root)
            for directory in directories]
    output = _check_output(args.output)
    if any(output == directory.resolve() or output.is_relative_to(directory.resolve()) for directory in directories):
        raise ValueError("Report output must not overwrite run evidence")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "runs": runs,
              "normalizer_sha256": args.normalizers_sha256,
              "audited_predictions": sum(run["audited_predictions"] for run in runs)}
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    task_records = [{"directory": run["directory"], "method": run["config"]["method"],
                     "seed": run["config"]["seed"], "task_id": task_id, **metrics}
                    for run in runs for task_id, metrics in run["metrics"]["tasks"].items()]
    with (output / "tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_records[0]))
        writer.writeheader()
        writer.writerows(task_records)
    family_records = [{"directory": run["directory"], "family": family, **metrics}
                      for run in runs for family, metrics in run["metrics"]["families"].items()]
    with (output / "families.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(family_records[0]))
        writer.writeheader()
        writer.writerows(family_records)
    print(json.dumps({"runs": len(runs), "audited_predictions": report["audited_predictions"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
