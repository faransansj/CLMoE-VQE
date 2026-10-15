"""G4 new-seed confirmation audit: locked-grid adjudication, independent rescoring
of every complete run, and the predeclared paired seed statistics.

The G4 lock is predeclared content: the eight confirmation seeds, the four fixed
task orders, the common training condition, the G3-selected tuned replay
condition, inline G1 validation normalizers and every literal training job argv
(128 common-method runs + 32 frozen-first + 32 common-budget independent + the
tuned replay runs; 192-224 total). The audit requires the exact locked grid with
no missing, duplicate or extra trial, rescores every complete run through
scripts.summarize_multimodal.audit with the locked normalizers, validates the
scientific settings, source/asset/scorer/manifest contracts and training
budgets, then exports summary/task/resource CSVs plus an audit JSON and a
statistics JSON in a fresh output directory.

Primary endpoints are per-family normalized final scores only. Primary
comparisons (six): expansion minus each of shared, capacity-matched and
common-budget replay, separately per family; paired within seed/order, averaged
over the four orders per seed (eight inferential units); exact two-sided
sign-flip tests, 20000-replicate paired-seed percentile 95% bootstraps, Holm
correction across all six. Tuned replay yields two further exploratory
comparisons carried in a separate Holm family of two. AF, transfer and per-run
diagnostics stay exploratory paired-interval reports. The bootstrap seed comes
from the G4 lock and must never silently fall back to the readout-cohort
constant of scripts/compare_readout.py. Positive estimates, intervals and
p-values do not establish novelty; intervals are individual, not simultaneous.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

from scripts import summarize_multimodal as summary

ROOT = Path(__file__).resolve().parents[1]

TASK_IDS = ("V", "C", "G", "F")
FAMILIES = ("vqa", "caption")
COMMON_METHODS = ("shared_lora", "capacity_matched_shared", "replay_lora", "expand")
FROZEN_METHOD = "frozen_first"
INDEPENDENT_METHOD = "independent"
TUNED_METHOD = "replay_lora"
ORDERS = (("V", "C", "G", "F"), ("F", "G", "C", "V"),
          ("V", "G", "C", "F"), ("F", "C", "G", "V"))
REPLAY_CANDIDATES = tuple((lr, epochs) for lr in (0.0001, 0.0003) for epochs in (5, 10))
COMMON_CONDITION = (0.0003, 5)
SLOT_SIZES = {"train": 2048, "validation": 256, "eval": 512}
FIRST_TASK_GROUP = ("shared_lora", "replay_lora", "expand", FROZEN_METHOD)
GUARD_EMPTY_FRACTION_MAX = 0.01
GUARD_LARGEST_OUTPUT_FRACTION_STRICT_MAX = 0.9
CONFIRMATION_SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 20000
SIGNFLIP_TOLERANCE = 1e-12
NORMALIZERS_ARTIFACT = "artifacts/multimodal-g1-normalizers.json"
G1_REPORT_ARTIFACT = "artifacts/multimodal-g1-report.json"
G2_REPORT_ARTIFACT = "artifacts/multimodal-g2-report.json"
G3_REPORT_ARTIFACT = "artifacts/multimodal-g3-report.json"
REQUIRED_PRIOR_ARTIFACTS = (G1_REPORT_ARTIFACT, G2_REPORT_ARTIFACT,
                            G3_REPORT_ARTIFACT, NORMALIZERS_ARTIFACT)
REQUIRED_RUNTIME_KEYS = ("optimizer", "generation", "loss", "torch", "numpy",
                         "cuda", "deterministic_algorithms", "tf32")
SINGLE_PATH_ARGS = {"--manifest": "manifest", "--assets": "assets",
                    "--image-root": "image_root", "--output": "output"}
CLASSIFIER_ARGS = {
    "--method": ("method", 1),
    "--eval-split": ("eval_split", 1),
    "--seed": ("seed", 1),
    "--lr": ("lr", 1),
    "--epochs": ("epochs", 1),
    "--batch-size": ("batch_size", 1),
    "--micro-batch-size": ("micro_batch_size", 1),
}
FORBIDDEN_FLAGS = ("--rank", "--alpha", "--image-size", "--device", "--router-modality",
                   "--router-memory", "--router-ridge", "--replay-per-task", "--train-limit",
                   "--eval-limit", "--stop-after-steps", "--oracle-diagnostic",
                   "--image-shuffled-diagnostic", "--smoke-output")

_HEX = set("0123456789abcdef")


def _is_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or not set(value.lower()) <= _HEX:
        raise ValueError(f"Expected lowercase sha256 for {label}")
    return value


def _number_matches(actual, expected):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)


def parse_job(argv):
    """Parse one literal locked job argv into its condition declaration."""
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        raise ValueError("Locked G4 job argv must be a nonempty list of literal strings")
    if any(flag in argv for flag in FORBIDDEN_FLAGS):
        raise ValueError(f"Locked G4 job must not change the frozen scientific contract: {argv}")
    values = {}
    seen = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argv token {token!r} inside locked G4 job")
        if token == "--task-ids":
            if token in seen:
                raise ValueError(f"Duplicate job flag {token}")
            index += 1
            tasks = []
            while index < len(argv) and not argv[index].startswith("--"):
                tasks.append(argv[index])
                index += 1
            if not tasks:
                raise ValueError("Locked G4 job needs a nonempty --task-ids list")
            seen.add(token)
            values["task_ids"] = tasks
            continue
        if token in SINGLE_PATH_ARGS:
            following = argv[index + 1] if index + 1 < len(argv) else None
            if token in seen or not following or following.startswith("--"):
                raise ValueError(f"Locked G4 job needs one path value after {token}")
            values[SINGLE_PATH_ARGS[token]] = following
            seen.add(token)
            index += 2
            continue
        if token in CLASSIFIER_ARGS:
            key, count = CLASSIFIER_ARGS[token]
            pieces = argv[index + 1:index + 1 + count]
            if token in seen or len(pieces) != count or pieces[0].startswith("--"):
                raise ValueError(f"Locked G4 job needs one value after {token}")
            values[key] = pieces[0]
            seen.add(token)
            index += 1 + count
            continue
        raise ValueError(f"Unknown flag {token} inside locked G4 job argv: {argv}")
    required = {"manifest", "assets", "image_root", "output", "method", "task_ids",
                "seed", "lr", "epochs", "eval_split", "batch_size", "micro_batch_size"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Locked G4 job misses required flags {sorted(missing)}: {argv}")
    if len(values["task_ids"]) not in (1, len(TASK_IDS)):
        raise ValueError("G4 runs are four-task CL or single-task independent conditions")
    try:
        values["seed"] = int(values["seed"])
        values["lr"] = float(values["lr"])
        values["epochs"] = int(values["epochs"])
        values["batch_size"] = int(values["batch_size"])
        values["micro_batch_size"] = int(values["micro_batch_size"])
    except ValueError as error:
        raise ValueError(f"Malformed numeric job value in locked G4 job: {argv}") from error
    return values


def _run_name(order, method, lr, epochs, seed):
    return f"{'_'.join(order)}_{method}_lr{lr}_ep{epochs}_seed{seed}"


def _validate_inline_normalizers(lock):
    normalizers = lock.get("normalizers")
    if not isinstance(normalizers, dict) or normalizers.get("schema_version") != 1 \
            or normalizers.get("eval_split") != "validation" \
            or normalizers.get("manifest_sha256") != lock["manifest_sha256"]:
        raise ValueError("G4 lock normalizers must be the fixed G1 validation artifact for this manifest")
    scores = normalizers.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(TASK_IDS) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in scores.values()):
        raise ValueError("G4 inline normalizers need positive finite G1 scores for all four tasks")


def _validate_selected_replay(lock):
    selected = lock.get("selected_replay")
    if not isinstance(selected, dict) or set(selected) != {"lr", "epochs"}:
        raise ValueError("G4 lock selected_replay must be exactly {lr, epochs}")
    condition = (selected["lr"], selected["epochs"])
    if not any(_number_matches(condition[0], candidate[0]) and condition[1] == candidate[1]
               for candidate in REPLAY_CANDIDATES):
        raise ValueError(f"Selected replay condition {condition} is outside the locked G3 replay grid")


def _parse_lock(path, expected_hash):
    sha256 = summary._sha256(path)
    if sha256 != expected_hash:
        raise ValueError(f"G4 lock file hash mismatch: expected {expected_hash}, computed {sha256}")
    lock = json.loads(Path(path).read_text())
    if not isinstance(lock, dict) or lock.get("schema_version") != 1 or lock.get("stage") != "G4":
        raise ValueError("G4 lock must be {schema_version:1, stage:G4}")
    _is_sha256(lock.get("manifest_sha256"), "lock manifest_sha256")
    for key, label in (("source_sha256", "source"), ("assets_sha256", "assets"),
                       ("tokenizer_sha256", "tokenizer")):
        if not isinstance(lock.get(key), dict) or not lock[key]:
            raise ValueError(f"G4 lock {key} must be the complete {label} hash map")
        if not all(isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX
                   for value in lock[key].values()):
            raise ValueError(f"G4 lock {key} values must be lowercase sha256 strings")
    if any(not key.startswith("tokenizer/") for key in lock["tokenizer_sha256"]):
        raise ValueError("G4 lock tokenizer_sha256 keys must live under tokenizer/")
    hashed = {key: value for key, value in lock["assets_sha256"].items() if key.startswith("tokenizer/")}
    if hashed != lock["tokenizer_sha256"]:
        raise ValueError("G4 lock assets_sha256 and tokenizer_sha256 disagree over tokenizer files")
    scorer_locks = lock.get("scorer_locks")
    if (not isinstance(scorer_locks, dict) or set(scorer_locks) != set(TASK_IDS)
            or any(not isinstance(value, dict) or not value for value in scorer_locks.values())):
        raise ValueError(f"G4 lock scorer_locks must cover all four fixed tasks {TASK_IDS}")
    _validate_inline_normalizers(lock)
    _is_sha256(lock.get("normalizers_sha256"), "lock normalizers_sha256")
    seeds = lock.get("seeds")
    if (not isinstance(seeds, list) or sorted(seeds) != sorted(set(seeds))
            or sorted(seeds) != sorted(CONFIRMATION_SEEDS)):
        raise ValueError(f"G4 confirmation seeds must be exactly {sorted(CONFIRMATION_SEEDS)}")
    orders = lock.get("orders")
    if (not isinstance(orders, list) or len(orders) != len(ORDERS)
            or sorted(tuple(order) for order in orders) != sorted(ORDERS)):
        raise ValueError(f"G4 lock orders must be exactly the predeclared four {list(ORDERS)}")
    common = lock.get("common_training")
    if not isinstance(common, dict) or set(common) != {"lr", "epochs", "batch_size", "micro_batch_size"}:
        raise ValueError("G4 lock common_training must declare lr, epochs, batch_size and micro_batch_size")
    if not _number_matches(common["lr"], COMMON_CONDITION[0]) or common["epochs"] != COMMON_CONDITION[1] \
            or common["batch_size"] != 16 or common["micro_batch_size"] != 8:
        raise ValueError("G4 common training is predeclared as AdamW lr3e-4, 5 epochs, batch16, micro8")
    _validate_selected_replay(lock)
    if lock.get("bootstrap_seed") != BOOTSTRAP_SEED or lock.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError("G4 bootstrap is predeclared as the confirmation seed with 20000 replicates")
    jobs = lock.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("G4 lock jobs must be a nonempty array of {name, argv}")
    if any(not isinstance(job, dict) or set(job) != {"name", "argv"} or not isinstance(job["name"], str)
           for job in jobs):
        raise ValueError("Every G4 lock job must be exactly {name, argv}")
    names = [job["name"] for job in jobs]
    if sorted(names) != sorted(set(names)):
        raise ValueError("G4 lock job names must be unique")
    inventory = lock.get("prior_gate_reports")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("G4 lock requires a prior_gate_reports hash inventory")
    scientific = lock.get("scientific_config")
    if (not isinstance(scientific, dict) or set(scientific) != set(REQUIRED_RUNTIME_KEYS)):
        raise ValueError("G4 lock scientific_config must carry the G1 runtime contract "
                         f"{sorted(REQUIRED_RUNTIME_KEYS)}")
    inventory = lock.get("prior_gate_reports")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("G4 lock requires a prior_gate_reports {relative file path: sha256} inventory")
    for name, value in inventory.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError("Prior gate report inventory must map one file path to one sha256 string")
        _is_sha256(value, f"prior gate report {name}")
    missing_aliases = [alias for alias in REQUIRED_PRIOR_ARTIFACTS if alias not in inventory]
    if missing_aliases:
        raise ValueError(f"Prior gate report inventory is missing required artifacts: {missing_aliases}")
    explicit = lock.get("g3_report_sha256")
    if explicit is not None:
        _is_sha256(explicit, "lock g3_report_sha256")
        if inventory[G3_REPORT_ARTIFACT] != explicit:
            raise ValueError("lock g3_report_sha256 must equal its prior_gate_reports inventory entry")
    lock["_seeds"] = tuple(int(seed) for seed in seeds)
    lock["_orders"] = tuple(tuple(order) for order in orders)
    lock["_lock_sha256"] = expected_hash
    return lock


def _load_prior_reports(lock, artifacts_root=ROOT):
    """Resolve, hash-verify and load every declared prior-gate artifact.

    The inventory is shared with the routing evaluator: {relative-file-path:
    sha256}. Every file resolves or the audit fails; the four required aliases
    carry the G1/G2/G3 reports (identified by their JSON stage field) and the
    G1 normalizer artifact whose file content must equal the inline lock
    normalizers.
    """
    earlier = lock["prior_gate_reports"]
    loaded_by_alias = {}
    records = {}
    for name in sorted(earlier):
        expected = earlier[name]
        candidate = Path(name) if Path(name).is_absolute() else Path(artifacts_root) / name
        if not candidate.is_file():
            raise ValueError(f"Prior gate report file is missing on this machine: {candidate}")
        actual = summary._sha256(candidate)
        if actual != expected:
            raise ValueError(f"Changed prior gate report {name}: {candidate}")
        try:
            loaded_by_alias[name] = json.loads(candidate.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Prior gate report must be readable JSON: {candidate}") from error
        records[name] = {"sha256": expected, "path": str(candidate), "status": "verified"}
    missing_aliases = [alias for alias in REQUIRED_PRIOR_ARTIFACTS if alias not in loaded_by_alias]
    if missing_aliases:
        raise ValueError(f"Prior gate report inventory is missing required artifacts: {missing_aliases}")
    stage_by_alias = {G1_REPORT_ARTIFACT: "G1", G2_REPORT_ARTIFACT: "G2", G3_REPORT_ARTIFACT: "G3"}
    canonical_name_by_alias = {G1_REPORT_ARTIFACT: "g1_report", G2_REPORT_ARTIFACT: "g2_report",
                               G3_REPORT_ARTIFACT: "g3_report"}
    canonical = {}
    for alias, visible_stage in stage_by_alias.items():
        report = loaded_by_alias[alias]
        if not isinstance(report, dict) or report.get("stage") != visible_stage:
            raise ValueError(f"Prior gate report {alias} must be the {visible_stage} stage JSON")
        records[alias]["stage"] = report.get("stage")
        canonical[canonical_name_by_alias[alias]] = report
    normalizers = loaded_by_alias[NORMALIZERS_ARTIFACT]
    if not isinstance(normalizers, dict) or normalizers != lock["normalizers"]:
        raise ValueError(f"Inline G4 normalizers differ from the prior artifact {NORMALIZERS_ARTIFACT}")
    records[NORMALIZERS_ARTIFACT]["stage"] = "G1 normalizer artifact"
    canonical["g1_normalizers"] = normalizers
    return canonical, records


def _validate_gate_chain(lock, loaded):
    validations = {}
    chain = (("g1_report", ("input_valid", "gate_passed", "pilot_allowed")),
             ("g2_report", ("input_valid", "gate_passed", "g3_allowed")),
             ("g3_report", ("input_valid", "gate_passed", "g4_allowed")))
    for name, fields in chain:
        report = loaded.get(name)
        if not isinstance(report, dict):
            raise ValueError(f"Loaded prior gate report {name} is missing; incomplete evidence must fail")
        if not all(report.get(field) for field in fields):
            raise ValueError(f"Locked prior gate report {name} does not pass its own gate")
        if report.get("manifest_sha256") != lock["manifest_sha256"]:
            raise ValueError(f"Prior gate report {name} manifest differs from the G4 lock")
        validations[name] = f"loaded; {report.get('stage', name)} gate passed"
    g3 = loaded.get("g3_report")
    if g3 is None:
        return validations
    if g3.get("normalizers_sha256") != lock["normalizers_sha256"]:
        raise ValueError("G3 report normalizer hash differs from the G4 lock")
    selected = g3.get("selected_replay")
    if not isinstance(selected, dict) or "lr" not in selected or "epochs" not in selected:
        raise ValueError("G3 report selection lacks lr/epochs")
    if not _number_matches(selected["lr"], lock["selected_replay"]["lr"]) \
            or int(selected["epochs"]) != int(lock["selected_replay"]["epochs"]):
        raise ValueError("Locked selected_replay differs from the audited G3 selection")
    trials = g3.get("trials")
    if not isinstance(trials, list) or not trials:
        return validations
    candidates, counts = {}, {}
    for trial in trials:
        if trial.get("method") != "replay_lora":
            continue
        key = (float(trial["lr"]), int(trial["epochs"]))
        candidates[key] = candidates.get(key, 0.0) + float(trial["family_balanced_normalized_final_score"])
        counts[key] = counts.get(key, 0) + 1
    if not candidates or any(count != len(ORDERS) for count in counts.values()):
        raise ValueError("G3 replay trials must span all four orders for every candidate condition")
    ranking = sorted(candidates, key=lambda key: (-(candidates[key] / counts[key]), key[1], key[0]))
    locked = (float(lock["selected_replay"]["lr"]), int(lock["selected_replay"]["epochs"]))
    if ranking[0] != locked:
        raise ValueError("Locked selected_replay does not reproduce the G3 selection rule")
    return validations


def expected_conditions(lock):
    """Exact locked condition grid; tuned replay reuses common runs when identical."""
    seeds = lock["_seeds"]
    common_lr = lock["common_training"]["lr"]
    common_epochs = lock["common_training"]["epochs"]
    selected = (lock["selected_replay"]["lr"], lock["selected_replay"]["epochs"])
    tuned_is_common = selected == (common_lr, common_epochs)
    conditions = []

    def add(kind, method, order, lr, epochs):
        for seed in seeds:
            conditions.append({"kind": kind, "method": method, "order": tuple(order),
                               "lr": float(lr), "epochs": int(epochs), "seed": int(seed),
                               "name": _run_name(order, method, lr, epochs, seed)})

    for order in lock["_orders"]:
        for method in COMMON_METHODS:
            add("common_cl", method, order, common_lr, common_epochs)
        add("frozen_first", FROZEN_METHOD, order, common_lr, common_epochs)
    for task_id in TASK_IDS:
        add("independent", INDEPENDENT_METHOD, (task_id,), common_lr, common_epochs)
    if not tuned_is_common:
        for order in lock["_orders"]:
            add("tuned_replay", TUNED_METHOD, order, selected[0], selected[1])
    names = [condition["name"] for condition in conditions]
    if sorted(names) != sorted(set(names)):
        raise ValueError("Derived G4 condition names collide; the lock is inconsistent with run naming")
    if tuned_is_common and len(names) != 192:
        raise ValueError(f"Tuned replay reuses the common condition, so the lock must hold exactly 192 jobs, not {len(names)}")
    if not tuned_is_common and len(names) != 224:
        raise ValueError(f"Tuned replay differs from common, so the lock must hold exactly 224 jobs, not {len(names)}")
    locked_names = sorted(job["name"] for job in lock["jobs"])
    if locked_names != sorted(names):
        raise ValueError("Locked jobs differ from the derived condition grid: "
                         f"extra {sorted(set(locked_names) - set(names))}, "
                         f"missing {sorted(set(names) - set(locked_names))}")
    return {condition["name"]: condition for condition in conditions}, tuned_is_common


def validate_locked_jobs(lock, grid):
    for job in lock["jobs"]:
        condition = grid[job["name"]]
        parsed = parse_job(job["argv"])
        if tuple(parsed["task_ids"]) != condition["order"] or parsed["method"] != condition["method"]:
            raise ValueError(f"Locked job argv contradicts its name-derived condition: {job['name']}")
        if not _number_matches(parsed["lr"], condition["lr"]) or parsed["epochs"] != condition["epochs"] \
                or parsed["seed"] != condition["seed"] or parsed["eval_split"] != "eval" \
                or parsed["batch_size"] != 16 or parsed["micro_batch_size"] != 8:
            raise ValueError(f"Locked job argv differs from the predeclared condition: {job['name']}")


def discover_runs(roots):
    directories = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"G4 --runs root is not a directory: {root}")
        directories.update(path.parent for path in root.rglob("config.json"))
    if not directories:
        raise ValueError("No G4 run directories found under --runs")
    return sorted(directories, key=lambda path: str(path))


def match_runs(directories, grid):
    matched = []
    seen = {}
    for directory in directories:
        name = directory.name
        if name not in grid:
            raise ValueError(f"G4 run directory is not a locked job output: {directory}")
        if name in seen:
            raise ValueError(f"Duplicate G4 run directory for locked job {name}: {directory}")
        seen[name] = directory
        matched.append((directory, grid[name]))
    if len(matched) != len(grid):
        raise ValueError(f"Locked G4 conditions without a complete run: {sorted(set(grid) - set(seen))}")
    return matched


def _validate_run_config(config, condition, lock):
    where = condition["name"]
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("protocol") != "multimodal_cl":
        raise ValueError(f"G4 run config schema/protocol contract broken: {where}")
    if config.get("eval_split") != "eval" or config.get("smoke_output") or config.get("eval_limit") \
            or config.get("train_limit"):
        raise ValueError(f"G4 run must score the full official eval split without smoke controls: {where}")
    if config.get("method") != condition["method"]:
        raise ValueError(f"G4 run method differs from the locked condition: {where}")
    if list(config.get("task_order", [])) != list(condition["order"]) \
            or list(config.get("task_ids", [])) != list(condition["order"]):
        raise ValueError(f"G4 run task ids/order differ from the locked condition: {where}")
    if config.get("seed") != condition["seed"]:
        raise ValueError(f"G4 run seed differs from the locked condition: {where}")
    if not _number_matches(config.get("lr"), condition["lr"]):
        raise ValueError(f"G4 run lr differs from the locked condition: {where}")
    if config.get("epochs") != condition["epochs"]:
        raise ValueError(f"G4 run epochs differ from the locked condition: {where}")
    if config.get("manifest_sha256") != lock["manifest_sha256"]:
        raise ValueError(f"G4 run manifest hash differs from the lock: {where}")
    if config.get("source_sha256") != lock["source_sha256"]:
        raise ValueError(f"G4 run source hash differs from the lock: {where}")
    if config.get("assets_sha256") != lock["assets_sha256"]:
        raise ValueError(f"G4 run asset hashes differ from the lock: {where}")
    if config.get("tokenizer_sha256") != lock["tokenizer_sha256"]:
        raise ValueError(f"G4 run tokenizer hash differs from the lock: {where}")
    needed = {task_id: lock["scorer_locks"][task_id] for task_id in condition["order"]}
    if config.get("scorer_locks") != needed:
        raise ValueError(f"G4 run scorer locks differ from the lock for the run task set: {where}")
    for key, expected in (("batch_size", 16), ("micro_batch_size", 8), ("rank", 8),
                          ("alpha", 16), ("image_size", 224), ("router_memory", 128),
                          ("router_ridge", 1.0), ("replay_per_task", 64)):
        if not _number_matches(config.get(key), expected):
            raise ValueError(f"G4 run config field {key} differs from the frozen contract: {where}")
    if config.get("router_modality") != "joint" or config.get("device") != "cuda":
        raise ValueError(f"G4 routing/inference contract is frozen at the joint router on cuda: {where}")
    if config.get("oracle_diagnostic") or config.get("image_shuffled_diagnostic"):
        raise ValueError(f"G4 confirmation runs must not enable diagnostic scoring: {where}")
    for key, value in lock["scientific_config"].items():
        if config.get(key) != value:
            raise ValueError(f"G4 run {key} differs from the locked G1 runtime contract: {where}")
    return config


def _load_manifest(path, expected_sha256, cache):
    manifest = cache.get(path)
    if manifest is not None:
        return manifest
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Manifest hash differs from the locked manifest_sha256: {path}")
    manifest = json.loads(raw)
    if manifest.get("confirmation_eligible") is False:
        raise ValueError("G4 manifest must be confirmation-eligible")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if set(tasks) != set(TASK_IDS):
        raise ValueError(f"G4 manifest must cover tasks {TASK_IDS}")
    for task_id in TASK_IDS:
        task = tasks[task_id]
        if task["family"] not in FAMILIES:
            raise ValueError(f"G4 manifest task {task_id} family {task['family']} is outside vqa/caption")
        for split, expected_rows in SLOT_SIZES.items():
            if len(task[split]) != expected_rows:
                raise ValueError(f"G4 manifest {task_id}/{split} must provide exactly {expected_rows} rows")
    cache[path] = manifest
    return manifest


def validate_stage_budgets(audited, condition, train_rows):
    stages = audited["stage_resources"]
    method = condition["method"]
    epochs = condition["epochs"]
    name = condition["name"]
    expected_exposures = train_rows * epochs
    if method == INDEPENDENT_METHOD:
        if len(stages) != 1:
            raise ValueError(f"Independent transfer control must complete one task: {name}")
        if stages[0]["image_exposures"] != expected_exposures:
            raise ValueError(f"Independent budget differs from the common condition: {name}")
    elif method == FROZEN_METHOD:
        if stages[0]["image_exposures"] != expected_exposures:
            raise ValueError(f"Frozen-first first-stage budget differs: {name}")
        for stage in stages[1:]:
            if stage["steps"] or stage["image_exposures"] or stage["current_exposures"] or stage["replay_exposures"]:
                raise ValueError(f"Frozen-first must be update-free after the first task: {name}")
    else:
        for stage in stages:
            if stage["image_exposures"] != expected_exposures:
                raise ValueError(f"CL stage budget differs from the common condition: {name}")
    for stage in stages:
        if stage["steps"] != -(-stage["image_exposures"] // 16):
            raise ValueError(f"Stage optimizer steps mismatch effective-batch arithmetic: {name}")
        if stage["current_exposures"] + stage["replay_exposures"] != stage["image_exposures"]:
            raise ValueError(f"Replay plus current exposures do not add up: {name}")
    if method == "replay_lora":
        stored = [stage["replay_images_stored"] for stage in stages]
        if stored != [(index + 1) * 64 for index in range(len(stages))]:
            raise ValueError(f"Stage resources must record end-stage replay storage "
                             f"(current task extended before the stage snapshot): {name}")
    else:
        for stage in stages:
            if stage["replay_exposures"] or stage["replay_images_stored"] or stage["raw_replay_bytes"]:
                raise ValueError(f"Method {method} must not consume replay rows: {name}")


def _guard_failures(audited):
    failures = []
    for record in audited["generation"]:
        if record["empty_fraction"] > GUARD_EMPTY_FRACTION_MAX \
                or record["largest_output_fraction"] >= GUARD_LARGEST_OUTPUT_FRACTION_STRICT_MAX:
            failures.append({"stage": record["stage"], "task_id": record["task_id"],
                             "family": record["family"], "empty_fraction": record["empty_fraction"],
                             "largest_output_fraction": record["largest_output_fraction"]})
    return failures


def _subsequent_task_gains(audited):
    tasks = audited["metrics"]["tasks"]
    gains = []
    for index, value in enumerate(audited["pre_to_post_gain"]):
        if index == 0:
            continue
        task_id = audited["config"]["task_order"][index]
        gains.append({"task_id": task_id, "family": tasks[task_id]["family"], "gain": value,
                      "normalized_gain": value / tasks[task_id]["normalizer"]})
    return gains


def audit_trials(matched, lock, manifest_cache):
    """Rescore every complete run with the shared independent audit and the locked normalizers."""
    trials = {}
    for directory, condition in matched:
        label = condition["name"]
        audited = summary.audit(directory, normalizer_lock=lock["normalizers"])
        if not audited["image_bytes_verified"]:
            raise ValueError(f"Rescoring did not verify real image bytes: {directory}")
        config = json.loads((directory / "config.json").read_text())
        resources = json.loads((directory / "results.json").read_text())["resources"]
        _validate_run_config(config, condition, lock)
        manifest = _load_manifest(config["manifest"], config["manifest_sha256"], manifest_cache)
        train_rows = len({task["task_id"]: task for task in manifest["tasks"]}
                         [condition["order"][0]]["train"])
        validate_stage_budgets(audited, condition, train_rows)
        metrics = audited["metrics"]
        trial = {"directory": str(directory), "name": condition["name"], "kind": condition["kind"],
                 "method": condition["method"], "task_order": list(condition["order"]),
                 "seed": condition["seed"], "lr": condition["lr"], "epochs": condition["epochs"],
                 "family_final": _family_endpoints(condition["kind"], metrics),
                 "family_native_AF": _family_af(condition["kind"], metrics, "native_AF"),
                 "family_normalized_AF": _family_af(condition["kind"], metrics, "normalized_AF"),
                 "family_balanced_normalized_final": (None if condition["kind"] == "independent"
                                                      else metrics.get("family_balanced_normalized_final")),
                 "subsequent_task_gains": _subsequent_task_gains(audited),
                 "generation_guard_failures": _guard_failures(audited),
                 "audited_predictions": audited["audited_predictions"], "resources": resources}
        if condition["kind"] == "independent":
            task_id = condition["order"][0]
            trial["independent_score"] = metrics["tasks"][task_id]["final"]
            trial["independent_metric"] = metrics["tasks"][task_id]["metric"]
        trial["audit"] = audited
        trials[label] = trial
    return trials


def _family_endpoints(kind, metrics):
    if kind == "independent":
        return {family: None for family in FAMILIES}
    return {family: metrics["families"][family]["normalized_final"] for family in FAMILIES}


def _family_af(kind, metrics, field):
    if kind == "independent":
        return {family: None for family in FAMILIES}
    return {family: metrics["families"][family].get(field) for family in FAMILIES}


def _cl_inventory(trials):
    inventory = {}
    for trial in trials.values():
        if trial["kind"] in ("common_cl", "tuned_replay", "frozen_first"):
            inventory[("_".join(trial["task_order"]), trial["method"], trial["lr"],
                       trial["epochs"], trial["seed"])] = trial
    return inventory


def validate_capacity_parity(trials, lock):
    """Capacity-matched and expansion adapter budgets match at every (order, seed)."""
    inventory = _cl_inventory(trials)
    records = {}
    for order in lock["_orders"]:
        tag = "_".join(order)
        for seed in lock["_seeds"]:
            capacity = inventory[(tag, "capacity_matched_shared", lock["common_training"]["lr"],
                                  lock["common_training"]["epochs"], seed)]
            expansion = inventory[(tag, "expand", lock["common_training"]["lr"],
                                   lock["common_training"]["epochs"], seed)]
            capacity_experts = capacity["audit"]["stage_resources"][-1]["parameter_counts"]["experts"]
            expansion_experts = expansion["audit"]["stage_resources"][-1]["parameter_counts"]["experts"]
            if capacity_experts != expansion_experts:
                raise ValueError(f"Capacity-matched and expansion adapters diverge: {tag}/seed{seed}")
            records[f"{tag}/seed{seed}"] = {"capacity_experts": capacity_experts,
                                            "expansion_experts": expansion_experts,
                                            "equal": True}
    return records


def validate_first_task_parity(trials, lock):
    """Compare every rank-8 first-task run to its same-seed independent control."""
    controls = {(trial["task_order"][0], trial["seed"]): trial
                for trial in trials.values() if trial["kind"] == "independent"}
    groups = {}
    for trial in trials.values():
        if trial["kind"] not in ("common_cl", "frozen_first"):
            continue
        if trial["method"] not in FIRST_TASK_GROUP:
            continue
        key = (trial["task_order"][0], trial["seed"])
        groups.setdefault(key, []).append(trial)
    identical_map, report = {}, {}
    for key in sorted(groups):
        if key not in controls:
            raise ValueError(f"Same-seed first-task parity group lacks independent control: {key}")
        control = controls[key]
        filename = f"predictions_after_00_{key[0]}.jsonl"
        payload = (Path(control["directory"]) / filename).read_bytes()
        members = sorted(groups[key], key=lambda trial: trial["name"])
        identical, divergent = [], []
        for trial in members:
            matches = (Path(trial["directory"]) / filename).read_bytes() == payload
            identical_map[trial["name"]] = matches
            (identical if matches else divergent).append(trial["name"])
        identical_map[control["name"]] = True
        report[f"{key[0]}/seed{key[1]}"] = {
            "control": control["name"], "members": [trial["name"] for trial in members],
            "identical": identical, "divergent": divergent}
        if divergent:
            raise ValueError(f"Same-seed first-task predictions diverged in {divergent}: {key}")
    return report, identical_map


def paired_statistics(deltas, *, bootstrap_seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Seed-unit paired statistics with compare_readout conventions; the bootstrap
    seed must come from the G4 lock, never from the readout constant."""
    if len(deltas) < 2 or not all(math.isfinite(value) for value in deltas):
        raise ValueError("Need at least two finite paired seed deltas")
    if len(deltas) > 16:
        raise ValueError("Exact sign-flip enumeration is limited to 16 predeclared seeds")
    mean = statistics.mean(deltas)
    observed = abs(sum(deltas))
    permutations = itertools.product((-1, 1), repeat=len(deltas))
    count = sum(abs(sum(sign * delta for sign, delta in zip(signs, deltas))) >= observed - SIGNFLIP_TOLERANCE
                for signs in permutations)
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = np.asarray(deltas)[rng.integers(0, len(deltas), size=(replicates, len(deltas)))].mean(1)
    return {"mean": mean,
            "sample_sd": statistics.stdev(deltas),
            "per_seed": deltas,
            "seed_bootstrap_95_percentile": np.quantile(samples, [0.025, 0.975]).tolist(),
            "exact_two_sided_sign_flip_p": count / (2 ** len(deltas)),
            "positive_seed_count": sum(value > 0 for value in deltas),
            "negative_seed_count": sum(value < 0 for value in deltas),
            "bootstrap": {"seed": int(bootstrap_seed), "replicates": int(replicates)}}


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [None] * len(pvalues), 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def _per_seed_order_means(values_by_order_seed, seeds):
    keyed = {}
    for (order_index, seed), value in values_by_order_seed.items():
        keyed.setdefault(seed, {})[order_index] = value
    if set(keyed) != set(seeds) or any(set(slot) != set(range(len(ORDERS))) for slot in keyed.values()):
        raise ValueError("Per-family endpoints need all four orders for every seed")
    return {seed: statistics.mean(keyed[seed][order_index] for order_index in range(len(ORDERS)))
            for seed in seeds}


def primary_statistics(trials, lock):
    """Six primary AP comparisons plus two tuned-replay exploratory comparisons."""
    inventory = _cl_inventory(trials)
    seeds = lock["_seeds"]
    common_lr = lock["common_training"]["lr"]
    common_epochs = lock["common_training"]["epochs"]
    selected_lr = lock["selected_replay"]["lr"]
    selected_epochs = lock["selected_replay"]["epochs"]
    rows = []
    for family in FAMILIES:
        for baseline in ("shared_lora", "capacity_matched_shared", "replay_lora"):
            candidate_values = {}
            baseline_values = {}
            for order_index, order in enumerate(lock["_orders"]):
                tag = "_".join(order)
                for seed in seeds:
                    candidate = inventory[(tag, "expand", common_lr, common_epochs, seed)]
                    base = inventory[(tag, baseline, common_lr, common_epochs, seed)]
                    candidate_values[(order_index, seed)] = candidate["family_final"][family]
                    baseline_values[(order_index, seed)] = base["family_final"][family]
            candidate_means = _per_seed_order_means(candidate_values, seeds)
            baseline_means = _per_seed_order_means(baseline_values, seeds)
            deltas = [candidate_means[seed] - baseline_means[seed] for seed in seeds]
            rows.append({"label": f"expand_minus_{baseline}_{family}", "family": family,
                         "method": "expand", "baseline": baseline, "endpoint": "family_normalized_final_score",
                         "candidate_seed_means": candidate_means, "baseline_seed_means": baseline_means,
                         "statistics": paired_statistics(deltas, bootstrap_seed=lock["bootstrap_seed"],
                                                         replicates=lock["bootstrap_replicates"])})
    tuned_rows = []
    for family in FAMILIES:
        candidate_values = {}
        baseline_values = {}
        for order_index, order in enumerate(lock["_orders"]):
            tag = "_".join(order)
            for seed in seeds:
                candidate = inventory[(tag, "expand", common_lr, common_epochs, seed)]
                base = inventory[(tag, TUNED_METHOD, selected_lr, selected_epochs, seed)]
                candidate_values[(order_index, seed)] = candidate["family_final"][family]
                baseline_values[(order_index, seed)] = base["family_final"][family]
        candidate_means = _per_seed_order_means(candidate_values, seeds)
        baseline_means = _per_seed_order_means(baseline_values, seeds)
        deltas = [candidate_means[seed] - baseline_means[seed] for seed in seeds]
        tuned_rows.append({"label": f"expand_minus_tuned_replay_{family}", "family": family,
                           "method": "expand", "baseline": "tuned_replay",
                           "baseline_condition": {"lr": selected_lr, "epochs": selected_epochs},
                           "endpoint": "family_normalized_final_score",
                           "candidate_seed_means": candidate_means, "baseline_seed_means": baseline_means,
                           "statistics": paired_statistics(deltas, bootstrap_seed=lock["bootstrap_seed"],
                                                           replicates=lock["bootstrap_replicates"])})
    for row, value in zip(rows, holm([row["statistics"]["exact_two_sided_sign_flip_p"] for row in rows])):
        row["holm_adjusted_p"] = value
        row["significant_at_0_05"] = value < 0.05
        row["AP_mean_positive"] = row["statistics"]["mean"] > 0
        row["CI_lower_positive"] = row["statistics"]["seed_bootstrap_95_percentile"][0] > 0
    for row, value in zip(tuned_rows, holm([row["statistics"]["exact_two_sided_sign_flip_p"] for row in tuned_rows])):
        row["holm_adjusted_p"] = value
        row["significant_at_0_05"] = value < 0.05
        row["AP_mean_positive"] = row["statistics"]["mean"] > 0
        row["CI_lower_positive"] = row["statistics"]["seed_bootstrap_95_percentile"][0] > 0
        row["exploratory"] = "separate Holm family of two"
    return rows, tuned_rows


def _family_of(task_id):
    return "vqa" if task_id in ("V", "G") else "caption"


def transfer_records(trials):
    """Same-seed common-budget independent transfer controls for every CL trial."""
    independent = {}
    for trial in trials.values():
        if trial["kind"] == "independent":
            independent[(trial["task_order"][0], trial["seed"])] = trial
    expected_controls = {(task_id, seed) for task_id in TASK_IDS for seed in CONFIRMATION_SEEDS}
    if set(independent) != expected_controls:
        raise ValueError(f"Same-seed independent common-budget controls incomplete: "
                         f"{sorted(expected_controls - set(independent))}")
    records = []
    for name in sorted(trials):
        trial = trials[name]
        if trial["kind"] not in ("common_cl", "tuned_replay", "frozen_first"):
            continue
        rows = []
        for index, task_id in enumerate(trial["task_order"]):
            post_train = trial["audit"]["matrix"][index][index]
            control = independent[(task_id, trial["seed"])]
            independent_score = control["independent_score"]
            normalizer = trial["audit"]["metrics"]["tasks"][task_id]["normalizer"]
            rows.append({"task_id": task_id, "post_train": post_train,
                         "independent_score": independent_score,
                         "independent_directory": control["directory"],
                         "native_delta": post_train - independent_score,
                         "normalized_delta": (post_train - independent_score) / normalizer})
        records.append({"trial": name, "kind": trial["kind"], "method": trial["method"],
                        "seed": trial["seed"], "order": list(trial["task_order"]), "rows": rows})
    return records


def transfer_statistics(records, lock):
    """Exploratory per-family transfer intervals for every method condition."""
    seeds = lock["_seeds"]
    aggregated = {}
    for record in records:
        order_index = ORDERS.index(tuple(record["order"]))
        for family in FAMILIES:
            deltas = [row["normalized_delta"] for row in record["rows"]
                      if _family_of(row["task_id"]) == family]
            if deltas:
                slot = aggregated.setdefault((family, record["method"]), {})
                slot[(order_index, record["seed"])] = statistics.mean(deltas)
    expected_keys = {(order_index, seed) for order_index in range(len(ORDERS)) for seed in seeds}
    rows = []
    for key in sorted(aggregated):
        family, method = key
        slot = aggregated[key]
        missing = expected_keys - set(slot)
        if missing:
            raise ValueError(f"Transfer statistics incomplete for {family}/{method}: "
                             f"{len(missing)} (order, seed) pairs missing")
        per_seed = [statistics.mean(slot[(order_index, seed)] for order_index in range(len(ORDERS)))
                    for seed in seeds]
        rows.append({"family": family, "method": method,
                     "endpoint": "normalized_post_train_minus_independent",
                     "statistics": paired_statistics(per_seed, bootstrap_seed=lock["bootstrap_seed"],
                                                     replicates=lock["bootstrap_replicates"])})
    return rows


def af_statistics(trials, lock):
    """Exploratory per-family normalized AF deltas: expansion minus every baseline."""
    inventory = _cl_inventory(trials)
    seeds = lock["_seeds"]
    common_lr = lock["common_training"]["lr"]
    common_epochs = lock["common_training"]["epochs"]
    selected_lr = lock["selected_replay"]["lr"]
    selected_epochs = lock["selected_replay"]["epochs"]
    rows = []
    for family in FAMILIES:
        for baseline, label, baseline_lr, baseline_epochs in (
                ("shared_lora", "shared_lora", common_lr, common_epochs),
                ("capacity_matched_shared", "capacity_matched_shared", common_lr, common_epochs),
                ("replay_lora", "replay_lora", common_lr, common_epochs),
                ("replay_lora", "tuned_replay", selected_lr, selected_epochs)):
            candidate_values = {}
            baseline_values = {}
            for order_index, order in enumerate(lock["_orders"]):
                tag = "_".join(order)
                for seed in seeds:
                    candidate = inventory[(tag, "expand", common_lr, common_epochs, seed)]
                    base = inventory[(tag, baseline, baseline_lr, baseline_epochs, seed)]
                    candidate_value = candidate["family_normalized_AF"][family]
                    baseline_value = base["family_normalized_AF"][family]
                    if candidate_value is None or baseline_value is None:
                        raise ValueError(f"Family AF missing for order {order} seed {seed} ({family})")
                    candidate_values[(order_index, seed)] = candidate_value
                    baseline_values[(order_index, seed)] = baseline_value
            candidate_means = _per_seed_order_means(candidate_values, seeds)
            baseline_means = _per_seed_order_means(baseline_values, seeds)
            deltas = [candidate_means[seed] - baseline_means[seed] for seed in seeds]
            rows.append({"family": family, "candidate": "expand", "baseline": label,
                         "endpoint": "family_normalized_AF_delta_negative_desirable",
                         "candidate_seed_means": candidate_means, "baseline_seed_means": baseline_means,
                         "statistics": paired_statistics(deltas, bootstrap_seed=lock["bootstrap_seed"],
                                                         replicates=lock["bootstrap_replicates"])})
    return rows


def subsequent_gain_statistics(trials, lock):
    """Exploratory per-family subsequent-task gain summaries for the primary expansion."""
    seeds = lock["_seeds"]
    assembled = {}
    for trial in trials.values():
        if trial["kind"] != "common_cl" or trial["method"] != "expand":
            continue
        order_index = ORDERS.index(tuple(trial["task_order"]))
        for gain in trial["subsequent_task_gains"]:
            assembled.setdefault((gain["family"], order_index, trial["seed"]), []).append(gain)
    rows = []
    for family in FAMILIES:
        expected = {(family, order_index, seed) for order_index in range(len(ORDERS)) for seed in seeds}
        missing = expected - set(assembled)
        if missing:
            raise ValueError(f"Subsequent-task transitions incomplete for family {family}: "
                             f"{len(missing)} (order, seed) slots missing")
        transitions = [values for key in sorted(expected) for values in assembled[key]]
        per_order_slot = {}
        for order_index, seed in _ordered_slots(seeds):
            per_order_slot[(order_index, seed)] = statistics.mean(
                values["normalized_gain"] for values in assembled[(family, order_index, seed)])
        per_seed = [statistics.mean(per_order_slot[(order_index, seed)]
                                    for order_index in range(len(ORDERS)))
                    for seed in seeds]
        rows.append({"family": family, "method": "expand",
                     "mean_native_gain": statistics.mean(values["gain"] for values in transitions),
                     "mean_normalized_gain": statistics.mean(values["normalized_gain"] for values in transitions),
                     "transition_count": len(transitions),
                     "statistics": paired_statistics(per_seed, bootstrap_seed=lock["bootstrap_seed"],
                                                     replicates=lock["bootstrap_replicates"])})
    return rows


def _ordered_slots(seeds):
    for order_index in range(len(ORDERS)):
        for seed in seeds:
            yield order_index, seed


def family_guard_failures(trials):
    """Generation guard failures per family aggregated over all expansion runs."""
    failures = {}
    for trial in trials.values():
        if trial["method"] != "expand":
            continue
        for record in trial["generation_guard_failures"]:
            entry = {key: record[key] for key in ("stage", "task_id", "family", "empty_fraction",
                                                  "largest_output_fraction")}
            entry["trial"] = trial["name"]
            failures.setdefault(record["family"], []).append(entry)
    for family in FAMILIES:
        failures.setdefault(family, [])
    return failures


def operational_determination(primary_rows, tuned_rows, af_rows, subsequent_rows, guard_failures):
    operations = {}
    for family in FAMILIES:
        family_primary = [row for row in primary_rows if row["family"] == family]
        family_tuned = next((row for row in tuned_rows if row["family"] == family), None)
        shared_af = next((row for row in af_rows if row["family"] == family and row["baseline"] == "shared_lora"),
                         None)
        family_gain = next((row for row in subsequent_rows if row["family"] == family), None)
        checks_by_baseline = {row["baseline"]: {"mean": row["statistics"]["mean"],
                                                "CI_95": row["statistics"]["seed_bootstrap_95_percentile"],
                                                "holm_p": row["holm_adjusted_p"],
                                                "pass": row["AP_mean_positive"] and row["CI_lower_positive"]
                                                and row["significant_at_0_05"]}
                              for row in family_primary}
        if family_tuned is not None:
            checks_by_baseline["tuned_replay"] = {"mean": family_tuned["statistics"]["mean"],
                                                  "CI_95": family_tuned["statistics"]["seed_bootstrap_95_percentile"],
                                                  "holm_p": family_tuned["holm_adjusted_p"],
                                                  "pass": family_tuned["AP_mean_positive"]
                                                  and family_tuned["CI_lower_positive"]
                                                  and family_tuned["significant_at_0_05"]}
        AP_all_pass = all(checks_by_baseline[baseline]["pass"]
                          for baseline in ("shared_lora", "capacity_matched_shared", "replay_lora"))
        expansion_af_mean = shared_af_mean = None
        af_pass = False
        if shared_af is not None:
            expansion_af_mean = statistics.mean(shared_af["candidate_seed_means"].values())
            shared_af_mean = statistics.mean(shared_af["baseline_seed_means"].values())
            af_pass = expansion_af_mean < shared_af_mean
        gain_pass = family_gain is not None and family_gain["mean_native_gain"] > 0
        guard_pass = not guard_failures.get(family)
        success = all((AP_all_pass, gain_pass, af_pass, guard_pass))
        notes = []
        if AP_all_pass:
            notes.append("AP improved over shared, capacity-matched and common replay "
                         "(mean>0, CI lower>0, Holm p<0.05)")
        else:
            passed = [row["baseline"] for row in family_primary
                      if row["AP_mean_positive"] and row["CI_lower_positive"] and row["significant_at_0_05"]]
            failed = [row["baseline"] for row in family_primary
                      if not (row["AP_mean_positive"] and row["CI_lower_positive"] and row["significant_at_0_05"])]
            notes.append(f"AP contrasts passed {passed} and were not established over {failed}")
        if family_tuned is not None:
            notes.append(f"tuned-replay contrast {'passed' if checks_by_baseline['tuned_replay']['pass'] else 'not established'} "
                         "(exploratory)")
        notes.append("subsequent-task gains positive" if gain_pass else
                     f"subsequent gains nonpositive "
                     f"({family_gain['mean_native_gain'] if family_gain is not None else 0.0})")
        if expansion_af_mean is not None:
            notes.append(f"family AF mean expansion={expansion_af_mean} vs shared={shared_af_mean}")
        notes.append("generation guard passed" if guard_pass
                     else f"{len(guard_failures.get(family, []))} generation guard failures")
        operations[family] = {"checks_by_baseline": checks_by_baseline, "success": success,
                              "mean_family_AF_expansion": expansion_af_mean,
                              "mean_family_AF_shared": shared_af_mean,
                              "subsequent_gain_mean_native": family_gain["mean_native_gain"]
                              if family_gain is not None else 0.0,
                              "determination": "; ".join(notes)}
    return operations


def _csv_value(value):
    if value is None:
        return "NA"
    return json.dumps(value) if isinstance(value, (dict, list)) else str(value)


SUMMARY_COLUMNS = ["directory", "name", "kind", "method", "task_order", "seed", "lr", "epochs",
                   "family_vqa_normalized_final", "family_caption_normalized_final",
                   "family_balanced_normalized_final",
                   "family_vqa_native_AF", "family_caption_native_AF",
                   "family_vqa_normalized_AF", "family_caption_normalized_AF",
                   "guard_failed", "budget_valid", "capacity_experts_equal", "first_task_identical",
                   "audited_predictions"]
TASK_COLUMNS = ["directory", "name", "kind", "method", "task_order", "seed", "stage_index", "task_id",
                "family", "metric", "pre", "post_train", "final", "AF", "pre_to_post_gain",
                "normalizer", "normalized_pre", "normalized_post_train", "normalized_final",
                "normalized_AF", "normalized_pre_to_post_gain",
                "transfer_native_delta", "transfer_normalized_delta",
                "transfer_independent_score", "transfer_independent_directory"]
RESOURCE_COLUMNS = ["directory", "name", "kind", "method", "task_order", "seed", "stage_index", "task_id",
                    "steps", "image_exposures", "current_exposures", "replay_exposures", "target_tokens",
                    "training_seconds", "router_memory_bytes", "replay_images_stored",
                    "replay_row_bytes", "replay_image_bytes", "raw_replay_bytes", "peak_gpu_memory_bytes",
                    "parameter_counts_base", "parameter_counts_experts",
                    "parameter_counts_active_trainable", "parameter_counts_total",
                    "parameter_counts_trainable",
                    "total_steps", "total_image_exposures", "total_target_tokens",
                    "total_current_exposures", "total_replay_exposures", "total_training_seconds",
                    "total_feature_seconds", "total_evaluation_seconds",
                    "total_independent_storage_bytes", "total_wall_seconds", "total_peak_gpu_memory_bytes"]


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def summary_rows(trials, capacity_equal, first_task_equal):
    rows = []
    for name in sorted(trials):
        trial = trials[name]
        rows.append({"directory": trial["directory"], "name": name, "kind": trial["kind"],
                     "method": trial["method"], "task_order": list(trial["task_order"]),
                     "seed": trial["seed"], "lr": trial["lr"], "epochs": trial["epochs"],
                     "family_vqa_normalized_final": trial["family_final"]["vqa"],
                     "family_caption_normalized_final": trial["family_final"]["caption"],
                     "family_balanced_normalized_final": trial["family_balanced_normalized_final"],
                     "family_vqa_native_AF": trial["family_native_AF"]["vqa"],
                     "family_caption_native_AF": trial["family_native_AF"]["caption"],
                     "family_vqa_normalized_AF": trial["family_normalized_AF"]["vqa"],
                     "family_caption_normalized_AF": trial["family_normalized_AF"]["caption"],
                     "guard_failed": len(trial["generation_guard_failures"]),
                     "budget_valid": True,
                     "capacity_experts_equal": capacity_equal.get(name),
                     "first_task_identical": first_task_equal.get(name),
                     "audited_predictions": trial["audited_predictions"]})
    return rows


def task_rows(trials, transfer_by_trial):
    rows = []
    for name in sorted(trials):
        trial = trials[name]
        transfer_rows = transfer_by_trial.get(name, [])
        for index, task_id in enumerate(trial["task_order"]):
            stage = trial["audit"]["metrics"]["tasks"][task_id]
            transfer = next((row for row in transfer_rows if row["task_id"] == task_id), {})
            pre = trial["audit"]["pre_task_scores"][index] if index < len(trial["audit"]["pre_task_scores"]) else ""
            normalizer = stage.get("normalizer")
            rows.append({"directory": trial["directory"], "name": name, "kind": trial["kind"],
                         "method": trial["method"], "task_order": list(trial["task_order"]),
                         "seed": trial["seed"], "stage_index": index, "task_id": task_id,
                         "family": stage["family"], "metric": stage["metric"],
                         "pre": pre, "post_train": stage["post_train"], "final": stage["final"],
                         "AF": stage.get("AF"), "pre_to_post_gain": stage.get("pre_to_post_gain"),
                         "normalizer": normalizer,
                         "normalized_pre": pre / normalizer if normalizer is not None and pre != "" else None,
                         "normalized_post_train": stage["post_train"] / normalizer
                                                  if normalizer is not None else None,
                         "normalized_final": stage.get("normalized_final"),
                         "normalized_AF": stage.get("normalized_AF"),
                         "normalized_pre_to_post_gain": stage.get("normalized_gain"),
                         "transfer_native_delta": transfer.get("native_delta"),
                         "transfer_normalized_delta": transfer.get("normalized_delta"),
                         "transfer_independent_score": transfer.get("independent_score"),
                         "transfer_independent_directory": transfer.get("independent_directory")})
    return rows


def resource_rows(trials):
    rows = []
    for name in sorted(trials):
        trial = trials[name]
        resources = trial["resources"]
        for stage_index, stage in enumerate(trial["audit"]["stage_resources"]):
            counts = stage.get("parameter_counts") or {}
            rows.append({"directory": trial["directory"], "name": name, "kind": trial["kind"],
                         "method": trial["method"], "task_order": list(trial["task_order"]),
                         "seed": trial["seed"], "stage_index": stage_index,
                         "task_id": stage.get("task_id", ""),
                         "steps": stage.get("steps"), "image_exposures": stage.get("image_exposures"),
                         "current_exposures": stage.get("current_exposures"),
                         "replay_exposures": stage.get("replay_exposures"),
                         "target_tokens": stage.get("target_tokens"),
                         "training_seconds": stage.get("training_seconds"),
                         "router_memory_bytes": stage.get("router_memory_bytes"),
                         "replay_images_stored": stage.get("replay_images_stored"),
                         "replay_row_bytes": stage.get("replay_row_bytes"),
                         "replay_image_bytes": stage.get("replay_image_bytes"),
                         "raw_replay_bytes": stage.get("raw_replay_bytes"),
                         "peak_gpu_memory_bytes": stage.get("peak_gpu_memory_bytes"),
                         "parameter_counts_base": counts.get("base"),
                         "parameter_counts_experts": counts.get("experts"),
                         "parameter_counts_active_trainable": counts.get("active_trainable"),
                         "parameter_counts_total": counts.get("total"),
                         "parameter_counts_trainable": counts.get("trainable"),
                         "total_steps": resources.get("steps"),
                         "total_image_exposures": resources.get("image_exposures"),
                         "total_target_tokens": resources.get("target_tokens"),
                         "total_current_exposures": resources.get("current_exposures"),
                         "total_replay_exposures": resources.get("replay_exposures"),
                         "total_training_seconds": resources.get("training_seconds"),
                         "total_feature_seconds": resources.get("feature_seconds"),
                         "total_evaluation_seconds": resources.get("evaluation_seconds"),
                         "total_independent_storage_bytes": resources.get("independent_storage_bytes"),
                         "total_wall_seconds": resources.get("wall_seconds"),
                         "total_peak_gpu_memory_bytes": resources.get("peak_gpu_memory_bytes")})
    return rows


def adjudicate(run_roots, lock, lock_sha256):
    conditions, tuned_is_common = expected_conditions(lock)
    validate_locked_jobs(lock, conditions)
    directories = discover_runs(run_roots)
    matched = match_runs(directories, conditions)
    manifest_cache = {}
    trials = audit_trials(matched, lock, manifest_cache)
    audited_predictions = sum(trial["audited_predictions"] for trial in trials.values())
    capacity_records = validate_capacity_parity(trials, lock)
    first_task_report, first_task_equal = validate_first_task_parity(trials, lock)
    primary_rows, tuned_rows = primary_statistics(trials, lock)
    records = transfer_records(trials)
    transfer_rows = transfer_statistics(records, lock)
    af_rows = af_statistics(trials, lock)
    subsequent_rows = subsequent_gain_statistics(trials, lock)
    guard_failures = family_guard_failures(trials)
    operational = operational_determination(primary_rows, tuned_rows, af_rows, subsequent_rows, guard_failures)
    grid = {"expected_conditions": len(conditions), "locked_jobs": len(lock["jobs"]),
            "tuned_replay_reuses_common": tuned_is_common,
            "kinds": {kind: sum(1 for condition in conditions.values() if condition["kind"] == kind)
                      for kind in sorted({condition["kind"] for condition in conditions.values()})}}
    report = {"schema_version": 1, "stage": "G4", "lock_sha256": lock_sha256,
              "lock": {"seeds": list(lock["_seeds"]), "orders": [list(order) for order in lock["_orders"]],
                       "common_training": lock["common_training"],
                       "selected_replay": lock["selected_replay"],
                       "bootstrap": {"seed": lock["bootstrap_seed"], "replicates": lock["bootstrap_replicates"]},
                       "manifest_sha256": lock["manifest_sha256"],
                       "source_sha256": lock["source_sha256"],
                       "assets_sha256": lock["assets_sha256"],
                       "tokenizer_sha256": lock["tokenizer_sha256"],
                       "scorer_locks": lock["scorer_locks"]},
              "grid": grid,
              "capacity_parity": capacity_records,
              "first_task_parity": first_task_report,
              "first_task_parity_by_trial": first_task_equal,
              "guard_failures": guard_failures,
              "trials": [trials[name] for name in sorted(trials)],
              "audited_predictions": audited_predictions,
              "transfer_records": records,
              "operational": operational,
              "interpretation": "Everything audited here is predeclared lock content; the audit performs "
                                "no data-driven selection, no re-tuned hyperparameters and no post-hoc "
                                "additions. Intervals are paired-seed percentile lists, p-values are "
                                "exact sign-flip enumerations, and neither establishes novelty."}
    statistics = {"schema_version": 1, "stage": "G4",
                  "bootstrap": {"seed": lock["bootstrap_seed"], "replicates": lock["bootstrap_replicates"]},
                  "unit": "seed; per-family endpoints average the four fixed orders inside a seed",
                  "primary_comparisons": primary_rows,
                  "tuned_replay_comparisons": tuned_rows,
                  "exploratory_transfer": transfer_rows,
                  "exploratory_AF": af_rows,
                  "subsequent_gain_statistics": subsequent_rows,
                  "operational": operational,
                  "interpretation": "Holm correction covers the six primary comparisons; the tuned replay "
                                    "family uses its own Holm family of two and stays exploratory. "
                                    "Seed bootstrap intervals are paired percentiles, not simultaneous "
                                    "intervals."}
    return report, statistics


def build_compact_summary(report, statistics, output):
    return {"runs": len(report["trials"]),
            "audited_predictions": report["audited_predictions"],
            "primary_comparisons": [{"label": row["label"], "mean": row["statistics"]["mean"],
                                     "CI_95": row["statistics"]["seed_bootstrap_95_percentile"],
                                     "holm_p": row["holm_adjusted_p"],
                                     "AP_mean_positive": row["AP_mean_positive"],
                                     "CI_lower_positive": row["CI_lower_positive"],
                                     "significant_at_0_05": row["significant_at_0_05"]}
                                    for row in statistics["primary_comparisons"]],
            "tuned_replay_comparisons": [{"label": row["label"], "mean": row["statistics"]["mean"],
                                          "CI_95": row["statistics"]["seed_bootstrap_95_percentile"],
                                          "holm_p": row["holm_adjusted_p"]}
                                         for row in statistics["tuned_replay_comparisons"]],
            "operational": {family: {"success": values["success"], "determination": values["determination"]}
                            for family, values in statistics["operational"].items()},
            "files": ["audit.json", "statistics.json", "summary.csv", "tasks.csv", "resources.csv"],
            "output": str(output),
            "lock_sha256": report["lock_sha256"]}


def write_outputs(output, report, statistics, transfer_by_trial, capacity_equal, first_task_equal):
    trials = {trial["name"]: trial for trial in report["trials"]}
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "statistics.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    _write_csv(output / "summary.csv", SUMMARY_COLUMNS, summary_rows(trials, capacity_equal, first_task_equal))
    _write_csv(output / "tasks.csv", TASK_COLUMNS, task_rows(trials, transfer_by_trial))
    _write_csv(output / "resources.csv", RESOURCE_COLUMNS, resource_rows(trials))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lock = _parse_lock(args.lock, args.lock_sha256)
    output = summary._check_output(args.output)
    if output.exists():
        raise ValueError(f"G4 confirmation output already exists; refusing overwrite: {output}")
    loaded, prior_inventory = _load_prior_reports(lock)
    prior_validations = _validate_gate_chain(lock, loaded)
    report, statistics = adjudicate(args.runs, lock, args.lock_sha256)
    report["prior_gate_reports"] = prior_inventory
    report["prior_gate_validations"] = prior_validations
    output = summary._check_output(args.output)
    output.mkdir(parents=True)
    transfer_by_trial = {record["trial"]: record["rows"] for record in report["transfer_records"]}
    capacity_equal = {}
    for key in report["capacity_parity"]:
        tag, seed_text = key.split("/")
        seed = int(seed_text[len("seed"):])
        numerics = (lock["common_training"]["lr"], lock["common_training"]["epochs"], seed)
        for method in ("capacity_matched_shared", "expand"):
            capacity_equal[_run_name(tag.split("_"), method, *numerics)] = True
    first_task_equal = report["first_task_parity_by_trial"]
    write_outputs(output, report, statistics, transfer_by_trial, capacity_equal, first_task_equal)
    print(json.dumps(build_compact_summary(report, statistics, output), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
