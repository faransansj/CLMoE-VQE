"""Independent rescorer for the frozen-bank routing ablation outputs.

Every declared G5 routing ablation is re-verified end to end: the parent run's
checkpoint/config/results/source/asset/manifest/scorer hashes, byte parity of all
native stage reconstructions (plus parent oracle files when the parent kept
them), the five declared variants, every prediction ID, corpus rescores through
TaskScorer, saved generation diagnostics, and the no-future-routing contract.
Only then are task/family normalized scores and paired variant summaries emitted
in the locked G1 normalization units. This report is exploratory routing
diagnostics: it never claims primary significance and never re-selects a model.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts import summarize_multimodal as summary
from scripts.ablate_multimodal_routing import VARIANTS, digest_file, predictions_path, request_family, \
    verify_confirmation_lock
from sema_multimodal.data import validate_manifest
from sema_multimodal.metrics import TaskScorer, continual_metrics

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("vqa", "caption")
TASK_FIELDS = ("directory", "variant", "task_id", "family", "pre_raw", "pre_normalized",
               "raw_s0", "raw_s1", "raw_s2", "raw_s3", "normalized_s0", "normalized_s1",
               "normalized_s2", "normalized_s3", "raw_af", "normalized_af",
               "native_delta_normalized_final", "pretask_unroutable")
FAMILY_FIELDS = ("directory", "variant", "family", "raw_final_mean", "normalized_final_mean",
                 "native_delta_raw_final_mean", "native_delta_normalized_final_mean",
                 "normalized_af_mean", "normalized_pre_to_post_gain_mean",
                 "other_expert_route_fraction_final", "other_family_route_fraction_final",
                 "unroutable_pretask_cells", "family_balanced_normalized_final")


def resolve_path(value, base=ROOT):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _assert_same(actual, recorded, label):
    summary._assert_same(actual, recorded, label)


def _verify_hashes(mapping, label, base=ROOT):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Missing {label} file hash inventory")
    for name, digest in mapping.items():
        if not isinstance(digest, str) or digest_file(resolve_path(name, base)) != digest:
            raise ValueError(f"Changed {label} file: {resolve_path(name, base)}")


def _positive_constants(scores, by_id):
    if not isinstance(scores, dict) or set(scores) != set(by_id):
        raise ValueError("Locked normalizer artifact must cover the full manifest task set")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in scores.values()):
        raise ValueError("Every locked normalizer must be finite and positive")


def _routing_prepare(report):
    """Structural gates on the routing report itself."""
    if report.get("schema_version") != 1 or report.get("stage") != "G5_frozen_bank_routing":
        raise ValueError("Expected a frozen-bank routing ablation artifact")
    if report.get("native_variant") != "native_joint":
        raise ValueError("The native routing variant declaration is fixed")
    if set(report.get("variants", {})) != set(VARIANTS):
        raise ValueError("Exactly the five locked routing variants must be declared")
    if report.get("expert_retraining_steps") != 0:
        raise ValueError("Frozen-bank routing must never update an expert")
    if report.get("model_projection_tensors_byte_identical_to_checkpoint") is not True:
        raise ValueError("The frozen bank must be verified unchanged for routing evidence")
    if report.get("router_checkpoint_tensors_identical") is not True:
        raise ValueError("The replayed router must equal the parent checkpoint router")
    config = report.get("parent_config")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("The routing report is missing the parent configuration")
    if config["method"] != "expand" or config.get("router_modality") != "joint":
        raise ValueError("Only expansion banks with the native joint router keep this contract")
    if config.get("eval_split") not in ("validation", "eval"):
        raise ValueError("Routing parents must use validation or eval splits")
    if report.get("eval_split") != config["eval_split"] or report.get("allow_eval") != bool(
            config["eval_split"] == "eval"):
        raise ValueError("Routing report eval declaration disagrees with the parent configuration")
    if report.get("script_sha256") != digest_file(ROOT / "scripts/ablate_multimodal_routing.py"):
        raise ValueError("Routing ablation script hash differs from the deployed evaluator")
    return config


def _verify_parent_bindings(report, config, asset_root):
    parent = Path(report["parent"])
    if json.loads((parent / "config.json").read_text()) != config:
        raise ValueError("Routing ablation parent configuration changed")
    if digest_file(parent / "results.json") != report["parent_results_sha256"]:
        raise ValueError("Routing ablation parent results changed")
    if digest_file(parent / "latest.pt") != report["parent_checkpoint_sha256"]:
        raise ValueError("Routing ablation parent expert weights changed")
    result = json.loads((parent / "results.json").read_text())
    if result.get("status") != "complete" or result.get("completed") != len(config["task_ids"]):
        raise ValueError("The routing parent run is incomplete")
    if report.get("manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("Routing ablation manifest binding disagrees with the parent configuration")
    raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Routing ablation manifest changed")
    manifest = json.loads(raw)
    if config.get("protocol") != manifest["protocol"]:
        raise ValueError("Routing parent protocol differs from the manifest")
    if config["eval_split"] == "eval" and manifest.get("confirmation_eligible") is False:
        raise ValueError("Development manifests are not confirmation eligible")
    _verify_hashes(config["source_sha256"], "source")
    _verify_hashes(config["assets_sha256"], "asset",
                   resolve_path(asset_root) if asset_root is not None else Path(config["assets"]).resolve())
    _assert_same(
        {key: value for key, value in config["assets_sha256"].items() if key.startswith("tokenizer/")},
        config["tokenizer_sha256"], "routing parent tokenizer hashes")
    return result, manifest


def _entry_for(report, variant, kind, stage, task_id):
    for entry in report["variants"][variant]["stage_evaluations"]:
        if entry["kind"] == kind and entry["stage"] == stage and entry["task_id"] == task_id:
            return entry
    raise ValueError(f"Routing report is missing the {variant} {kind} entry stage {stage}: {task_id}")


def _check_routing_records(report, variant, kind, stage, seen, task, records, expert_family,
                           assignments=None):
    """Enforce the routing contract read back from the emitted prediction records."""
    experts = {record["expert"] for record in records}
    if kind == "after":
        if variant == "task_id_oracle":
            if experts != {assignments[seen]}:
                raise ValueError("Oracle routing must use only the gold task's already-trained expert")
            return
        if -1 in experts:
            raise ValueError("Seen tasks are never routed base-only")
        if any(not 0 <= expert <= stage for expert in experts):
            raise ValueError("Router touched a future/unseen expert")
        if variant == "family_restricted_joint" and any(expert_family[expert] != task["own_family"]
                                                        for expert in experts):
            raise ValueError("Family-restricted routing left the public request family")
        return
    if variant == "task_id_oracle":
        raise ValueError("The oracle variant must not produce pre-task diagnostics")
    if stage == 0:
        if experts != {-1}:
            raise ValueError("Stage-0 pre-task reconstruction must be exactly base-only")
        return
    if any(not -1 <= expert < stage for expert in experts):
        raise ValueError("Pre-task router touched a future/unseen expert")
    if -1 in experts:
        if variant != "family_restricted_joint":
            raise ValueError("Only the family-restricted diagnostic may report unroutable requests")
        entry = _entry_for(report, variant, "before", stage, task["task_id"])
        if entry["routing"] != "base_only_no_eligible_family" or entry["family_available"] is not False:
            raise ValueError("Unroutable family restriction must be declared in the routing report")
        return
    if variant == "family_restricted_joint":
        entry = _entry_for(report, variant, "before", stage, task["task_id"])
        if entry["routing"] != "family_masked" or entry["family_available"] is not True:
            raise ValueError("Eligible family restriction must be declared in the routing report")


def _update_guard(guard, records):
    count = len(records)
    generation = [record["generation"] for record in records]
    empty = sum(not record["prediction"].strip() for record in records) / count
    repeated = sum(bool(item.get("repeated_trigram")) for item in generation) / count
    predictions = Counter(record["prediction"] for record in records)
    largest = max(predictions.values()) / count
    guard["max_empty_fraction"] = max(guard["max_empty_fraction"], empty)
    guard["max_repeated_trigram_fraction"] = max(guard["max_repeated_trigram_fraction"], repeated)
    guard["max_largest_output_fraction"] = max(guard["max_largest_output_fraction"], largest)
    return guard


def audit(directory, *, normalizer_lock, confirmation_lock=None, image_root=None, asset_root=None,
          manifest_override=None):
    """Independently rescore one frozen-bank routing ablation directory."""
    directory = Path(directory)
    report = json.loads((directory / "routing_ablation.json").read_text())
    config = _routing_prepare(report)
    result, manifest = _verify_parent_bindings(report, config, asset_root)
    if manifest_override is not None:
        override_raw = Path(manifest_override).read_bytes()
        if hashlib.sha256(override_raw).hexdigest() != config["manifest_sha256"]:
            raise ValueError("Authoritative manifest differs from the routing parent")
        manifest = json.loads(override_raw)
    validate_manifest(manifest, image_root=resolve_path(image_root) if image_root is not None else
                      Path(config["image_root"]).resolve())
    if (normalizer_lock.get("schema_version") != 1 or normalizer_lock.get("eval_split") != "validation"
            or normalizer_lock.get("manifest_sha256") != config["manifest_sha256"]):
        raise ValueError("Normalizer must be a fixed matching G1 validation artifact")
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if set(config["task_ids"]) != set(by_id):
        raise ValueError("Routing ablation task list does not cover the manifest")
    tasks = []
    for task_id in config["task_ids"]:
        task = dict(by_id[task_id])
        task["own_family"] = request_family(task, task[config["eval_split"]])
        tasks.append(task)
    _positive_constants(normalizer_lock["scores"], by_id)
    scorers = {task["task_id"]: TaskScorer(task) for task in tasks}
    _assert_same({task["task_id"]: scorer.lock for task, scorer in zip(tasks, scorers.values())},
                 config["scorer_locks"], "routing parent scorer locks")
    assignments = result.get("task_expert_assignments")
    if assignments != list(range(len(tasks))):
        raise ValueError("Routing parent assignments disagree with the expansion contract")
    expert_family = {}
    for value, family in report.get("expert_family", {}).items():
        index = int(value)
        if index in expert_family or not 0 <= index < len(tasks):
            raise ValueError("Malformed expert family declaration in the routing report")
        expert_family[index] = family
    for expert, task in zip(assignments, tasks):
        if expert_family.get(expert) != task["own_family"]:
            raise ValueError("Expert family declaration disagrees with the parent stream")
    for field in ("manifest_sha256", "source_sha256", "assets_sha256", "tokenizer_sha256", "scorer_locks"):
        if confirmation_lock is not None and confirmation_lock.get(field) != config[field]:
            raise ValueError(f"Locked confirmation {field} does not bind this routing parent")
    parent = Path(report["parent"])
    eval_limit = config.get("eval_limit", 0)
    eval_rows = {task["task_id"]: task[config["eval_split"]][:eval_limit or None] for task in tasks}
    depth = len(tasks)
    expected_native = depth + sum(stage + 1 for stage in range(depth))
    use_parent_oracle = bool(config.get("oracle_diagnostic"))
    if report.get("native_prediction_files_byte_identical") != expected_native:
        raise ValueError("Native prediction parity coverage is incomplete")
    variant_records = {}
    audited_predictions = 0
    for variant in VARIANTS:
        variant_dir = directory / variant
        expected = set()
        if variant != "task_id_oracle":
            for stage, task in enumerate(tasks):
                expected.add(predictions_path(variant_dir, "before", stage, task["task_id"]))
        for stage in range(depth):
            for previous in tasks[:stage + 1]:
                expected.add(predictions_path(variant_dir, "after", stage, previous["task_id"]))
        present = {path for path in variant_dir.iterdir() if path.is_file()}
        if present != expected:
            raise ValueError(f"Undeclared routing prediction files under {variant_dir}")
        for path in sorted(expected):
            parent_file = None
            if variant == "native_joint":
                parent_file = parent / path.name
            elif variant == "task_id_oracle" and use_parent_oracle:
                parent_file = parent / path.name.replace("predictions_after", "oracle_after", 1)
            if parent_file is not None:
                if not parent_file.is_file():
                    raise ValueError(f"Parent prediction file missing: {parent_file}")
                if digest_file(path) != digest_file(parent_file):
                    reason = "Native reconstruction does not byte-match the parent" \
                        if variant == "native_joint" else "Oracle reconstruction does not byte-match the parent"
                    raise ValueError(f"{reason}: {path.name}")
        files = {}
        for stage, task in enumerate(tasks):
            if variant != "task_id_oracle":
                path = predictions_path(variant_dir, "before", stage, task["task_id"])
                scoring, records = summary.rescore(path, eval_rows[task["task_id"]], scorers[task["task_id"]])
                audited_predictions += len(records)
                _check_routing_records(report, variant, "before", stage, None, task, records, expert_family)
                summary._check_evaluation(_entry_for(report, variant, "before", stage, task["task_id"]),
                                          task, path, scoring, records)
                files[f"before/{stage}"] = {"scoring": scoring, "records": records}
            for seen, previous in enumerate(tasks[:stage + 1]):
                path = predictions_path(variant_dir, "after", stage, previous["task_id"])
                scoring, records = summary.rescore(path, eval_rows[previous["task_id"]],
                                                   scorers[previous["task_id"]])
                audited_predictions += len(records)
                _check_routing_records(report, variant, "after", stage, seen, previous, records,
                                       expert_family, assignments)
                summary._check_evaluation(_entry_for(report, variant, "after", stage, previous["task_id"]),
                                          previous, path, scoring, records)
                files[f"after/{stage}/{seen}"] = {"scoring": scoring, "records": records}
        variant_records[variant] = files
    observed_statistics = {}
    for variant in VARIANTS:
        files = variant_records[variant]
        matrix_rows, pre_rows = [], []
        confusion = {"all": {}, "vqa": {}, "caption": {}}
        final_answer_counts = Counter()
        guard = {"max_empty_fraction": 0.0, "max_largest_output_fraction": 0.0,
                 "max_repeated_trigram_fraction": 0.0}
        for stage in range(depth):
            if variant != "task_id_oracle":
                before = files[f"before/{stage}"]
                pre_rows.append(before["scoring"]["score"])
                guard = _update_guard(guard, before["records"])
            row = []
            for seen in range(stage + 1):
                cell = files[f"after/{stage}/{seen}"]
                row.append(cell["scoring"]["score"])
                guard = _update_guard(guard, cell["records"])
                gold = assignments[seen]
                if stage == depth - 1:
                    for record in cell["records"]:
                        routed = record["expert"]
                        for name in ("all", tasks[seen]["own_family"]):
                            confusion[name].setdefault(str(gold), Counter())[str(routed)] += 1
                        final_answer_counts.update([record["prediction"]])
            matrix_rows.append(row)
        observed_statistics[variant] = {
            "matrix": matrix_rows,
            "pre_task_scores": pre_rows or None,
            "final_answer_counts": dict(final_answer_counts),
            "largest_answer_fraction": (max(final_answer_counts.values())
                                        / sum(final_answer_counts.values()))
            if final_answer_counts else 0.0,
            "confusion": confusion,
        }
        matrix_copy = observed_statistics[variant]
        matrix_copy["metrics"] = (continual_metrics(matrix_rows, tasks, pre_task_scores=pre_rows)
                                  if variant != "task_id_oracle" else continual_metrics(matrix_rows, tasks))
        matrix_copy["guard"] = guard
    _verify_reported_evidence(report, observed_statistics, tasks, depth)
    unroutable_by_variant = {}
    for variant in VARIANTS:
        unroutable_by_variant[variant] = [
            (entry["stage"], entry["task_id"]) for entry in report["variants"][variant]["stage_evaluations"]
            if entry["kind"] == "before" and entry["routing"] == "base_only_no_eligible_family"]
    declared_unroutable = {}
    for item in report["unroutable_stage_tasks"]:
        declared_unroutable.setdefault(item["variant"], set()).add((item["stage"], item["task_id"]))
    for variant in VARIANTS:
        if set(unroutable_by_variant[variant]) != declared_unroutable.get(variant, set()):
            raise ValueError("Unroutable family declaration disagrees with the routing report")
        if variant != "family_restricted_joint" and unroutable_by_variant[variant]:
            raise ValueError("Only the family-restricted variant may declare unroutable pre-task requests")
        observed_statistics[variant]["unroutable_pretask_cells"] = unroutable_by_variant[variant]
    task_ids = [task["task_id"] for task in tasks]
    summary_per_variant = {}
    family_final, deltas, observations, guards = {}, {}, {}, {}
    for variant in VARIANTS:
        observed = observed_statistics[variant]
        matrix = observed["matrix"]
        normalized_matrix = [[value / normalizer_lock["scores"][task_ids[seen]]
                              for seen, value in enumerate(row)] for row in matrix]
        pre = observed["pre_task_scores"]
        normalized_pre = [value / normalizer_lock["scores"][task_ids[index]] for index, value in
                          enumerate(pre)] if pre is not None else None
        if pre is not None:
            gains_raw = [matrix[index][index] - pre[index] for index in range(depth)]
            gains_normalized = [gains_raw[index] / normalizer_lock["scores"][task_ids[index]]
                                for index in range(depth)]
        else:
            gains_raw = gains_normalized = None
        af_raw, af_normalized = {}, {}
        for index in range(depth - 1):
            af_raw[task_ids[index]] = matrix[index][index] - matrix[depth - 1][index]
            af_normalized[task_ids[index]] = af_raw[task_ids[index]] / normalizer_lock["scores"][task_ids[index]]
        family_final[variant] = {}
        for family in FAMILIES:
            members = [index for index, task in enumerate(tasks) if task["own_family"] == family]
            family_final[variant][family] = sum(normalized_matrix[depth - 1][index] for index in members) \
                / len(members)
        family_final[variant]["balanced"] = (family_final[variant]["vqa"] + family_final[variant]["caption"]) / 2
        summary_per_variant[variant] = {
            "matrix": matrix, "matrix_normalized": normalized_matrix,
            "pre_task_scores": pre, "pre_task_scores_normalized": normalized_pre,
            "pre_to_post_gain_raw": gains_raw, "pre_to_post_gain_normalized": gains_normalized,
            "task_af_raw": af_raw, "task_af_normalized": af_normalized,
            "final_answer_counts": observed["final_answer_counts"],
            "largest_answer_fraction": observed["largest_answer_fraction"],
            "final_stage_confusion": {name: {gold: dict(routes) for gold, routes in gold_map.items()}
                                      for name, gold_map in observed["confusion"].items()},
            "unroutable_pretask_cells": observed_statistics[variant]["unroutable_pretask_cells"],
            "metrics": observed["metrics"],
        }
        guards[variant] = observed["guard"]
        observations[variant] = final_route_fractions(observed["confusion"], expert_family)
    for variant in VARIANTS:
        deltas[variant] = {
            "normalized_delta_per_task": {task_ids[index]: summary_per_variant[variant]["matrix_normalized"]
                                          [depth - 1][index]
                                          - summary_per_variant["native_joint"]["matrix_normalized"]
                                          [depth - 1][index] for index in range(depth)},
            "family_balanced_delta": family_final[variant]["balanced"]
            - family_final["native_joint"]["balanced"],
        }
    return {
        "directory": str(directory), "parent": str(parent), "parent_config": config,
        "eval_split": config["eval_split"], "allow_eval": report["allow_eval"],
        "confirmation_lock": confirmation_lock,
        "native_prediction_files_byte_identical": expected_native,
        "oracle_parent_parity_declared": use_parent_oracle,
        "expert_retraining_steps": report["expert_retraining_steps"],
        "variants": summary_per_variant,
        "family_final_normalized": family_final,
        "paired_variant_deltas": deltas,
        "routing_observations": observations,
        "generation_summary": guards,
        "unroutable_stage_tasks": report["unroutable_stage_tasks"],
        "audited_predictions": audited_predictions,
    }


def final_route_fractions(confusion, expert_family):
    """Final-stage route errors divided by all evaluated images, not just errors."""
    other_expert, other_family = {}, {}
    for family, gold_rows in confusion.items():
        total = sum(count for routes in gold_rows.values() for count in routes.values())
        other_expert[family] = sum(count for gold, routes in gold_rows.items()
                                   for routed, count in routes.items() if int(gold) != int(routed)) / total
        other_family[family] = sum(count for gold, routes in gold_rows.items()
                                   for routed, count in routes.items()
                                   if expert_family[int(gold)] != expert_family[int(routed)]) / total
    return {"other_expert_route_fraction_final": other_expert,
            "other_family_route_fraction_final": other_family}


def _verify_reported_evidence(report, observed_statistics, tasks, depth):
    """Compare every recorded variant statistic against the independently rescored values."""
    for variant in VARIANTS:
        declared = report["variants"][variant]
        observed = observed_statistics[variant]
        _assert_same(observed["matrix"], declared["matrix"], f"{variant}/matrix")
        if variant == "task_id_oracle":
            if declared.get("pre_task_scores") is not None:
                raise ValueError("The oracle variant must not declare pre-task scores")
        else:
            _assert_same(observed["pre_task_scores"], declared["pre_task_scores"], f"{variant}/pre_task_scores")
            _assert_same([matrix_row[index] - observed["pre_task_scores"][index]
                          for index, matrix_row in enumerate(observed["matrix"])],
                         declared["pre_to_post_gain"], f"{variant}/pre_to_post_gain")
        if dict(observed["final_answer_counts"]) != declared["final_answer_counts"]:
            raise ValueError(f"{variant}/final_answer_counts differs")
        _assert_same(observed["largest_answer_fraction"], declared["largest_answer_fraction"],
                     f"{variant}/largest_answer_fraction")
        for kind in ("all", "vqa", "caption"):
            recomputed = {gold: dict(routes) for gold, routes in observed["confusion"][kind].items()}
            if recomputed != declared["final_stage_confusion"][kind]:
                raise ValueError(f"{variant}/final_stage_confusion({kind}) differs")
        _assert_same(observed["metrics"], declared["metrics"], f"{variant}/metrics")


def _check_output(path):
    resolved = resolve_path(path).resolve()
    forbidden = ("readout", "vision", "legacy", "sema_runs", "vqa_runs")
    if any(part.casefold().startswith(forbidden) for part in resolved.parts):
        raise ValueError("Archived artifact namespaces are read-only")
    report_root = (ROOT / "artifacts" / "multimodal_report").resolve()
    if not resolved.is_relative_to(report_root):
        raise ValueError("Routing report outputs must stay under artifacts/multimodal_report")
    if resolved.exists():
        raise ValueError("Routing report output is new evidence; refuse overwriting an existing tree")
    return resolved


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablations", type=Path, nargs="+", required=True,
                        help="Routing ablation output directories (each carries routing_ablation.json)")
    parser.add_argument("--normalizers", type=Path, required=True, help="Locked normalization artifact")
    parser.add_argument("--normalizers-sha256", required=True, help="Explicit SHA256 of --normalizers")
    parser.add_argument("--confirmation-lock", type=Path,
                        help="Locked G4 confirmation artifact; required when any parent used eval predictions")
    parser.add_argument("--confirmation-lock-sha256")
    parser.add_argument("--manifest", type=Path, help="Authoritative manifest; must match every parent")
    parser.add_argument("--image-root", type=Path, help="Override image location; real SHA256/decode verification")
    parser.add_argument("--asset-root", type=Path, help="Override base for relative model asset hash paths")
    parser.add_argument("--output", type=Path, required=True, help="New routing report directory")
    args = parser.parse_args()
    if bool(args.confirmation_lock) != bool(args.confirmation_lock_sha256):
        parser.error("--confirmation-lock and --confirmation-lock-sha256 must be supplied together")
    normalizers_path = resolve_path(args.normalizers)
    if digest_file(normalizers_path) != args.normalizers_sha256:
        raise ValueError("Fixed normalizer file hash mismatch")
    normalizer_lock = json.loads(normalizers_path.read_text())
    directories, reports = [], []
    for declared in args.ablations:
        directory = Path(declared).resolve()
        if directory in directories:
            raise ValueError("Duplicate declared routing ablation")
        if not (directory / "routing_ablation.json").is_file():
            raise ValueError(f"Not a routing ablation directory: {directory}")
        directories.append(directory)
        reports.append(json.loads((directory / "routing_ablation.json").read_text()))
    if any(report.get("eval_split", "validation") == "eval" or report.get("allow_eval")
           for report in reports) and args.confirmation_lock is None:
        parser.error("--confirmation-lock is required for confirmation-eval routing evidence")
    if args.confirmation_lock is not None:
        if digest_file(resolve_path(args.confirmation_lock)) != args.confirmation_lock_sha256:
            raise ValueError("Explicit confirmation-lock SHA256 mismatch")
    runs = []
    verified_lock = None
    for directory, report in zip(directories, reports):
        config = report["parent_config"]
        if args.confirmation_lock is not None:
            verified_lock = verify_confirmation_lock(resolve_path(args.confirmation_lock),
                                                     args.confirmation_lock_sha256, config)
        runs.append(audit(directory, normalizer_lock=normalizer_lock, confirmation_lock=verified_lock,
                          image_root=args.image_root, asset_root=args.asset_root,
                          manifest_override=args.manifest))
    output = _check_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    container = {"schema_version": 1, "stage": "G5_routing",
                 "scope": "Exploratory frozen-bank routing diagnostics; no primary significance is claimed "
                          "and no model or config is re-selected from these scores.",
                 "normalizers_sha256": args.normalizers_sha256,
                 "confirmation_lock": {"path": str(resolve_path(args.confirmation_lock)),
                                       "sha256": args.confirmation_lock_sha256}
                 if verified_lock is not None else None,
                 "runs": runs, "parent_banks": len(runs),
                 "audited_predictions": sum(run["audited_predictions"] for run in runs)}
    (output / "routing_audit.json").write_text(json.dumps(container, indent=2, allow_nan=False) + "\n")
    task_rows, family_rows = [], []
    for run in runs:
        config = run["parent_config"]
        manifest = json.loads(Path(config["manifest"]).read_bytes())
        by_id = {task["task_id"]: task for task in manifest["tasks"]}
        tasks = [by_id[task_id] for task_id in config["task_ids"]]
        depth = len(tasks)
        for variant, scores in run["variants"].items():
            deltas = run["paired_variant_deltas"][variant]
            for index, task in enumerate(tasks):
                row = {"directory": run["directory"], "variant": variant, "task_id": task["task_id"],
                       "family": task["family"],
                       "pre_raw": None, "pre_normalized": None,
                       "raw_af": scores["task_af_raw"].get(task["task_id"]),
                       "normalized_af": scores["task_af_normalized"].get(task["task_id"]),
                       "native_delta_normalized_final": deltas["normalized_delta_per_task"][task["task_id"]],
                       "pretask_unroutable": any(stage_cell == index
                                                 for stage_cell, _ in scores["unroutable_pretask_cells"])}
                if scores["pre_task_scores"] is not None:
                    row["pre_raw"] = scores["pre_task_scores"][index]
                    row["pre_normalized"] = scores["pre_task_scores_normalized"][index]
                for stage in range(depth):
                    row[f"raw_s{stage}"] = (scores["matrix"][stage][index]
                                            if index < len(scores["matrix"][stage]) else None)
                    row[f"normalized_s{stage}"] = (scores["matrix_normalized"][stage][index]
                                                   if index < len(scores["matrix_normalized"][stage])
                                                   else None)
                task_rows.append(row)
            natives = run["variants"]["native_joint"]
            for family in FAMILIES:
                members = [index for index, task in enumerate(tasks) if task["family"] == family]
                raw_family = sum(scores["matrix"][depth - 1][index] for index in members) / len(members)
                raw_native_family = sum(natives["matrix"][depth - 1][index] for index in members) / len(members)
                member_af = [scores["task_af_normalized"][tasks[index]["task_id"]] for index in members
                             if index < depth - 1 and tasks[index]["task_id"] in scores["task_af_normalized"]]
                gains = scores["pre_to_post_gain_normalized"]
                member_gains = [gains[index] for index in members] if gains is not None else []
                family_ids = {task["task_id"] for task in tasks if task["family"] == family}
                unroutable_count = sum(1 for _, task_id in scores["unroutable_pretask_cells"]
                                       if task_id in family_ids)
                family_rows.append({
                    "directory": run["directory"], "variant": variant, "family": family,
                    "raw_final_mean": raw_family,
                    "normalized_final_mean": run["family_final_normalized"][variant][family],
                    "native_delta_raw_final_mean": raw_family - raw_native_family,
                    "native_delta_normalized_final_mean": run["family_final_normalized"][variant][family]
                        - run["family_final_normalized"]["native_joint"][family],
                    "normalized_af_mean": sum(member_af) / len(member_af) if member_af else None,
                    "normalized_pre_to_post_gain_mean": (sum(member_gains) / len(member_gains)
                                                         if member_gains else None),
                    "other_expert_route_fraction_final":
                        run["routing_observations"][variant]["other_expert_route_fraction_final"][family],
                    "other_family_route_fraction_final":
                        run["routing_observations"][variant]["other_family_route_fraction_final"][family],
                    "unroutable_pretask_cells": unroutable_count,
                })
            family_rows.append({
                "directory": run["directory"], "variant": variant, "family": "balanced",
                "raw_final_mean": None, "normalized_final_mean": None,
                "native_delta_raw_final_mean": None,
                "native_delta_normalized_final_mean": deltas["family_balanced_delta"],
                "normalized_af_mean": None, "normalized_pre_to_post_gain_mean": None,
                "other_expert_route_fraction_final": None,
                "other_family_route_fraction_final": None,
                "unroutable_pretask_cells": len(scores["unroutable_pretask_cells"]),
                "family_balanced_normalized_final": run["family_final_normalized"][variant]["balanced"],
            })
    _write_csv(output / "routing_tasks.csv", task_rows, TASK_FIELDS)
    _write_csv(output / "routing_families.csv", family_rows, FAMILY_FIELDS)
    print(json.dumps({"parent_banks": len(runs), "audited_predictions": container["audited_predictions"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
