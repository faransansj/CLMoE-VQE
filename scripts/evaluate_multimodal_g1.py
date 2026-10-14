"""Adjudicate the locked G1 grid without launching any training or pilot jobs.

Usage: python -m scripts.evaluate_multimodal_g1 --runs RUN_ROOT [RUN_ROOT ...]
    --manifest MANIFEST --output artifacts/multimodal_report/NEW_G1_REPORT

The output directory must be new. g1.json retains every independently audited
trial, including negative gains; normalizers.json exists only on a passing gate.
Exit status is 0 for a passing gate, 1 for a scientific gate failure, and 2 for
invalid/incomplete evidence. Caption bootstrap uses the exact fixed-train-IDF
CIDEr-D decomposition, not mean sentence BLEU or evaluation-estimated IDF.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import summarize_multimodal as summary
from sema_multimodal.data import SOURCE_SPECS, image_identity, validate_manifest
from sema_multimodal.metrics import TaskScorer, _caption_dependencies, _fixed_cider_score, _tokenize


TASKS = {"V": ("vqa", "vqav2"), "C": ("caption", "cider_d"),
         "G": ("vqa", "gqa"), "F": ("caption", "cider_d")}
SPLIT_SIZES = {"train": 2048, "validation": 256, "eval": 512}
LEARNING_RATES = (1e-4, 3e-4)
EPOCHS = (5, 10)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261013
COMMON = {"lr": 3e-4, "epochs": 5}
# Only these fields may vary across the 16 runs. In particular assets, source,
# generation, optimizer, precision and loss settings must remain identical.
GRID_FIELDS = {"lr", "epochs", "task_order", "task_ids", "scorer_locks"}
FIXED_SETTINGS = {
    "schema_version": 1, "method": "independent", "seed": 17,
    "eval_split": "validation", "train_limit": 0, "eval_limit": 0,
    "smoke_output": False, "image_shuffled_diagnostic": True,
    "rank": 8, "alpha": 16, "batch_size": 16, "micro_batch_size": 8,
    "image_size": 224, "deterministic_algorithms": True, "tf32": False,
    "generation": {"strategy": "cached_greedy", "vqa_max_new_tokens": 32, "caption_max_new_tokens": 64},
    "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                  "weight_decay": 0.01, "gradient_clip_norm": 1.0,
                  "scheduler": "epoch_cosine", "precision": "cuda_bfloat16_or_cpu_float32"},
    "loss": "BLIP smoothing 0.1; shifted target token mean per row, then effective-batch example mean",
}


def discover_runs(roots):
    directories = []
    seen = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Missing run root: {root}")
        found = sorted(path.parent.resolve() for path in root.rglob("config.json"))
        if not found:
            raise ValueError(f"No declared runs in root: {root}")
        for directory in found:
            if directory in seen:
                raise ValueError(f"Duplicate run directory (including overlapping roots): {directory}")
            seen.add(directory)
            directories.append(directory)
    if len(directories) != 16:
        raise ValueError(f"G1 requires exactly 16 distinct runs; found {len(directories)}")
    return sorted(directories)


def validate_g1_manifest(manifest):
    validate_manifest(manifest)
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if manifest["protocol"] != "multimodal_cl" or set(by_id) != set(TASKS):
        raise ValueError("G1 requires the full four-task V/C/G/F multimodal_cl manifest")
    for task_id, task in by_id.items():
        if ((task["family"], task["metric"]) != TASKS[task_id]
                or SOURCE_SPECS[task["source"]][2] != task_id):
            raise ValueError(f"Wrong source/family/metric for G1 task {task_id}")
        for split, count in SPLIT_SIZES.items():
            if len(task[split]) != count:
                raise ValueError(f"G1 requires {count} unique images for {task_id}/{split}")
    return by_id


def validate_grid(directories, manifest, manifest_sha256):
    """Check the complete grid before expensive rescoring; never drop a trial."""
    by_id = validate_g1_manifest(manifest)
    expected = {(task_id, lr, epochs) for task_id in by_id for lr in LEARNING_RATES for epochs in EPOCHS}
    grid, common_settings, scorer_locks = {}, None, {}
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        for field, value in FIXED_SETTINGS.items():
            if field not in config or type(config[field]) is not type(value) or config[field] != value:
                # Alpha is serialized as a float by argparse, unlike its integer plan value.
                if field == "alpha" and type(config.get(field)) in (int, float) and config[field] == value:
                    continue
                raise ValueError(f"Wrong G1 config {field}: {directory}")
        if config.get("manifest_sha256") != manifest_sha256 or config.get("protocol") != manifest["protocol"]:
            raise ValueError(f"G1 manifest hash/protocol mismatch: {directory}")
        order = config.get("task_order")
        if not isinstance(order, list) or len(order) != 1 or order[0] not in by_id:
            raise ValueError(f"Each G1 run must independently train exactly one task: {directory}")
        if config.get("task_ids") != order:
            raise ValueError(f"G1 task selection differs from task order: {directory}")
        task_id = order[0]
        if type(config.get("lr")) not in (int, float) or type(config.get("epochs")) is not int:
            raise ValueError(f"Malformed G1 learning rate/epoch config: {directory}")
        key = (task_id, config["lr"], config["epochs"])
        if key not in expected:
            raise ValueError(f"Unexpected G1 grid condition {key}: {directory}")
        if key in grid:
            raise ValueError(f"Duplicate G1 grid condition {key}: {directory}")
        locks = config.get("scorer_locks")
        if not isinstance(locks, dict) or set(locks) != {task_id}:
            raise ValueError(f"Missing/wrong G1 scorer locks: {directory}")
        if task_id in scorer_locks and locks[task_id] != scorer_locks[task_id]:
            raise ValueError(f"Changed scorer settings for task {task_id}")
        scorer_locks[task_id] = locks[task_id]
        settings = {field: value for field, value in config.items() if field not in GRID_FIELDS}
        if common_settings is not None and settings != common_settings:
            raise ValueError(f"Runs differ in source/manifest/assets or common settings: {directory}")
        common_settings = settings
        grid[key] = (directory, config)
    if set(grid) != expected:
        raise ValueError(f"Incomplete G1 grid; missing {sorted(expected - set(grid))}")
    return grid


def primary_image_scores(scorer, predictions, rows):
    """Prepare exact primary corpus contributions once, retaining image order.

    _fixed_cider_score is explicitly mean_i(10 * mean_orders(sum_references
    similarity_i) / n_references_i). With train DF, train document count and
    sigma fixed, each inner term depends only on image i. Consequently the
    corpus on any repeated-index bootstrap draw is EXACTLY the mean of those
    terms with the draw's multiplicities. PTB tokens/cooked ngrams are unchanged
    by duplication. We invoke the same implementation, not a second formula.
    Regression tests compare repeated-index resamples against TaskScorer.score.
    This identity does NOT apply to BLEU; BLEU is retained only from corpus audit.
    """
    if not rows or len(predictions) != len(rows) or not all(isinstance(pred, str) for pred in predictions):
        raise ValueError("One prediction per unique image is required")
    for key in ("id", "image_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError("Bootstrap input must contain unique image rows, not annotation duplicates")
    if len({image_identity(row) for row in rows}) != len(rows):
        raise ValueError("Bootstrap input must contain unique image identities")
    if any(row["family"] != scorer.family for row in rows):
        raise ValueError("Bootstrap rows belong to the wrong task family")
    if scorer.metric != "cider_d":
        return np.asarray([scorer.score([pred], [row])["score"]
                           for pred, row in zip(predictions, rows)], dtype=np.float64)
    for row in rows:
        if ((row["id"] in scorer._train_images and scorer._train_images[row["id"]] != row["image_sha256"])
                or (row["id"] not in scorer._train_images and row["image_sha256"] in scorer._train_hashes)):
            raise ValueError("Caption row ID/image hash disagrees with train corpus identity")
    captions = {("reference", index): scorer._references(row) for index, row in enumerate(rows)}
    captions.update({("prediction", index): [pred] for index, pred in enumerate(predictions)})
    tokenized = _tokenize(captions)
    CiderScorer, _, _, _, _ = _caption_dependencies()
    corpus = CiderScorer(n=4, sigma=6.0)
    for index in range(len(rows)):
        corpus += (tokenized[("prediction", index)][0], tokenized[("reference", index)])
    return np.asarray([_fixed_cider_score(SimpleNamespace(n=corpus.n, sigma=corpus.sigma,
        ctest=[prediction], crefs=[references]), scorer._df, scorer._train_documents)
        for prediction, references in zip(corpus.ctest, corpus.crefs)], dtype=np.float64)


def paired_bootstrap(before, after):
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.ndim != 1 or before.size == 0 or before.shape != after.shape:
        raise ValueError("Paired bootstrap requires matching nonempty image score vectors")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Nonfinite paired bootstrap score")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(before), size=len(before))
        # Recompute both primary corpus scores on exactly the same image draw.
        gains[index] = np.mean(after[draw]) - np.mean(before[draw])
    lower, upper = np.quantile(gains, [0.025, 0.975], method="linear")
    return {"unit": "unique_image", "images": len(before), "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED, "confidence": 0.95, "interval": "paired_percentile_linear",
            "mean_gain": float(np.mean(after) - np.mean(before)),
            "bootstrap_mean_gain": float(np.mean(gains)), "ci_lower": float(lower), "ci_upper": float(upper),
            "gain_samples": gains.tolist(),
            "caption_semantics": "fixed_train_IDF; exact resampled primary corpus decomposition; no BLEU averaging"}


def generation_guards(records):
    report = summary._generation_report(records)
    checks = {"empty_fraction_at_most_0.01": report["empty_fraction"] <= 0.01,
              "mode_fraction_below_0.9": report["largest_output_fraction"] < 0.9}
    return {**report, "checks": checks, "passed": all(checks.values())}


def _ordered_predictions(path, rows, expert):
    """Read already-audited evidence in manifest order, not JSONL file order."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {record["id"]: record for record in records}
    if len(records) != len(rows) or len(by_id) != len(records) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError(f"Missing/duplicate prediction IDs: {path}")
    ordered = [by_id[row["id"]] for row in rows]
    if any(type(record.get("expert")) is not int or record["expert"] != expert for record in ordered):
        raise ValueError(f"Expected frozen-base expert -1 or independent expert 0: {path}")
    return ordered


def _check_training_budget(audited, task, epochs):
    resources = audited["stage_resources"]
    if len(resources) != 1:
        raise ValueError("Independent G1 requires exactly one completed training stage")
    count = len(task["train"])
    expected = {"task_id": task["task_id"], "steps": epochs * math.ceil(count / 16),
                "image_exposures": epochs * count, "current_exposures": epochs * count,
                "replay_exposures": 0, "router_memory_bytes": 0, "raw_replay_bytes": 0}
    for key, value in expected.items():
        if type(resources[0].get(key)) is not type(value) or resources[0][key] != value:
            raise ValueError(f"Incomplete/non-independent G1 training budget: {key}")


def adjudicate(roots, manifest_path, *, image_root=None, asset_root=None):
    manifest_path = Path(manifest_path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    directories = discover_runs(roots)
    grid = validate_grid(directories, manifest, manifest_sha256)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    scorers = {task_id: TaskScorer(task) for task_id, task in tasks.items()}
    trials, audits, baselines = [], [], {}
    common_records = {}
    for (task_id, lr, epochs), (directory, config) in sorted(grid.items()):
        audited = summary.audit(directory, image_root=image_root, asset_root=asset_root)
        summary._assert_same(config, audited["config"], "G1 config changed during audit")
        summary._assert_same(scorers[task_id].lock, config["scorer_locks"][task_id], "G1 scorer lock")
        task = tasks[task_id]
        _check_training_budget(audited, task, epochs)
        rows = task["validation"]
        files = {"before": directory / f"predictions_before_00_{task_id}.jsonl",
                 "after": directory / f"predictions_after_00_{task_id}.jsonl",
                 "image_shuffled": directory / f"image_shuffled_00_{task_id}.jsonl"}
        records = {kind: _ordered_predictions(path, rows, -1 if kind == "before" else 0)
                   for kind, path in files.items()}
        baseline = [record["prediction"] for record in records["before"]]
        if task_id in baselines and baselines[task_id] != baseline:
            raise ValueError(f"Frozen-base predictions changed across grid trials: {task_id}")
        baselines[task_id] = baseline
        evaluations = {item["kind"]: item["scoring"] for item in audited["corpus_evaluations"]}
        if set(evaluations) != {"before", "after", "image_shuffled"}:
            raise ValueError(f"Missing/extra G1 corpus diagnostic: {directory}")
        guards = {kind: generation_guards(items) for kind, items in records.items()}
        trial = {"directory": str(directory), "task_id": task_id, "family": task["family"],
                 "metric": task["metric"], "units": scorers[task_id].lock["units"],
                 "seed": 17, "lr": lr, "epochs": epochs,
                 "common_condition": lr == COMMON["lr"] and epochs == COMMON["epochs"],
                 "scores": evaluations, "gain": evaluations["after"]["score"] - evaluations["before"]["score"],
                 "image_shuffled_delta": evaluations["after"]["score"] - evaluations["image_shuffled"]["score"],
                 "guards": guards, "stage_resources": audited["stage_resources"],
                 "evidence_sha256": {path.name: summary._sha256(path) for path in
                     [directory / "config.json", directory / "results.json", directory / "events.jsonl", *files.values()]}}
        if trial["common_condition"]:
            common_records[task_id] = records
        trials.append(trial)
        audits.append(audited)
    task_reports, selections, failures = {}, {}, []
    for task_id, task in tasks.items():
        task_trials = [trial for trial in trials if trial["task_id"] == task_id]
        common = next(trial for trial in task_trials if trial["common_condition"])
        records = common_records[task_id]
        contributions = {kind: primary_image_scores(scorers[task_id],
            [record["prediction"] for record in records[kind]], task["validation"])
            for kind in ("before", "after")}
        for kind, values in contributions.items():
            summary._assert_same(float(np.mean(values)), common["scores"][kind]["score"],
                                 f"{task_id}/{kind} exact corpus decomposition")
        bootstrap = paired_bootstrap(contributions["before"], contributions["after"])
        checks = {"positive_mean_gain": common["gain"] > 0,
                  "positive_ci_lower": bootstrap["ci_lower"] > 0,
                  **common["guards"]["after"]["checks"]}
        selected = min(task_trials, key=lambda trial: (-trial["scores"]["after"]["score"], trial["epochs"], trial["lr"]))
        selection = {key: selected[key] for key in ("directory", "lr", "epochs")}
        selection["score"] = selected["scores"]["after"]["score"]
        selection["positive"] = math.isfinite(selection["score"]) and selection["score"] > 0
        selections[task_id] = selection
        reasons = [name for name, passed in checks.items() if not passed]
        if not selection["positive"]:
            reasons.append("normalizer_must_be_positive")
        failures.extend(f"{task_id}: {reason}" for reason in reasons)
        task_reports[task_id] = {"family": task["family"], "metric": task["metric"], "units": common["units"],
            "common_run": common["directory"], "scores": common["scores"], "gain": common["gain"],
            "bootstrap": bootstrap, "guards": common["guards"], "checks": checks,
            "image_shuffled_delta": common["image_shuffled_delta"], "normalizer_selection": selection,
            "passed": not reasons, "failure_reasons": reasons}
    passed = not failures
    normalizers = None
    if passed:
        normalizers = {"schema_version": 1, "eval_split": "validation", "manifest_sha256": manifest_sha256,
                       "scores": {task_id: selection["score"] for task_id, selection in selections.items()}}
    return {"schema_version": 1, "stage": "G1", "input_valid": True, "gate_passed": passed,
            "pilot_allowed": passed, "pilot_launched": False, "failure_reasons": failures,
            "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
            "eval_split": "validation", "seed": 17, "common_condition": COMMON,
            "grid": {"tasks": list(tasks), "learning_rates": list(LEARNING_RATES), "epochs": list(EPOCHS), "runs": 16},
            "image_counts": {task_id: {split: len(task[split]) for split in SPLIT_SIZES} for task_id, task in tasks.items()},
            "scorer_locks": {task_id: scorer.lock for task_id, scorer in scorers.items()},
            "trials": trials, "tasks": task_reports, "audits": audits,
            "normalizers_usable": normalizers is not None,
            "image_shuffled_delta_definition": "trained native minus trained image-shuffled; descriptive, not a gate threshold",
            "interpretation": "Validation feasibility only; guards do not establish semantic quality or CL benefit."}, normalizers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args(argv)
    output = summary._check_output(args.output)
    if output.exists():
        raise ValueError("G1 output must be a new report directory; existing evidence is immutable")
    for root in args.runs:
        root = root.expanduser().resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("G1 report output must not overlap run evidence")
    try:
        report, normalizers = adjudicate(args.runs, args.manifest, image_root=args.image_root, asset_root=args.asset_root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        report = {"schema_version": 1, "stage": "G1", "input_valid": False, "gate_passed": False,
                  "pilot_allowed": False, "pilot_launched": False, "normalizers_usable": False,
                  "failure_reasons": [f"{type(error).__name__}: {error}"],
                  "manifest": str(args.manifest), "run_roots": [str(root) for root in args.runs]}
        normalizers = None
    output.mkdir(parents=True, exist_ok=False)
    if normalizers is not None:
        normalizer_path = output / "normalizers.json"
        normalizer_path.write_text(json.dumps(normalizers, indent=2, allow_nan=False) + "\n")
        report["normalizers_file"] = normalizer_path.name
        report["normalizers_sha256"] = summary._sha256(normalizer_path)
    else:
        report["normalizers_file"] = None
        report["normalizers_sha256"] = None
    (output / "g1.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "gate_passed": report["gate_passed"],
                      "input_valid": report["input_valid"], "failure_reasons": report["failure_reasons"],
                      "normalizers_sha256": report["normalizers_sha256"]}, indent=2))
    return 2 if not report["input_valid"] else int(not report["gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
