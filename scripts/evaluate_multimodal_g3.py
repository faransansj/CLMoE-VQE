"""Audit the locked 68-run G3 grid, select replay, and judge the G4 gate."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts import summarize_multimodal as summary
from scripts.evaluate_multimodal_g2 import METHODS, locked_json

ORDERS = (("V", "C", "G", "F"), ("F", "G", "C", "V"),
          ("V", "G", "C", "F"), ("F", "C", "G", "V"))
LRS = (0.0001, 0.0003)
EPOCHS = (5, 10)
COMMON = (0.0003, 5)


def adjudicate(runs, manifest_path, g1_path, g1_hash, g2_path, g2_hash,
               normalizer_path, normalizer_hash):
    g1 = locked_json(g1_path, g1_hash)
    g2 = locked_json(g2_path, g2_hash)
    normalizers = locked_json(normalizer_path, normalizer_hash)
    manifest_raw = Path(manifest_path).read_bytes()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    if not all(report.get("input_valid") and report.get("gate_passed") for report in (g1, g2)):
        raise ValueError("Passing G1 and G2 evidence is required")
    if not g2["g3_allowed"] or g2["g1_report_sha256"] != g1_hash:
        raise ValueError("G2 does not authorize this development cohort")
    if any(report["manifest_sha256"] != manifest_hash for report in (g1, g2, normalizers)):
        raise ValueError("Manifest differs from prior gates")
    if any(report["normalizers_sha256"] != normalizer_hash for report in (g1, g2)):
        raise ValueError("Normalizer differs from prior gates")
    scores = normalizers["scores"]
    if set(scores) != {"V", "C", "G", "F"} or any(not math.isfinite(v) or v <= 0 for v in scores.values()):
        raise ValueError("Four positive normalizers are required")
    expected = set(itertools.product(ORDERS, METHODS, LRS, EPOCHS))
    expected.update((order, "frozen_first", *COMMON) for order in ORDERS)
    reference = g1["audits"][0]["config"]
    varying = {"method", "task_ids", "task_order", "lr", "epochs", "scorer_locks", "image_shuffled_diagnostic"}
    invariant = {key: value for key, value in reference.items() if key not in varying}
    grid = {}
    for config_path in sorted(Path(runs).resolve().rglob("config.json")):
        config = json.loads(config_path.read_text())
        key = (tuple(config["task_order"]), config["method"], config["lr"], config["epochs"])
        if key not in expected or key in grid or config["task_ids"] != list(key[0]):
            raise ValueError(f"Unexpected/duplicate G3 condition: {config_path}")
        if {name: value for name, value in config.items() if name not in varying} != invariant:
            raise ValueError(f"Changed G1 scientific contract: {config_path}")
        if config["image_shuffled_diagnostic"] or config["scorer_locks"] != g1["scorer_locks"]:
            raise ValueError(f"Changed scorer/diagnostic condition: {config_path}")
        grid[key] = config_path.parent
    if set(grid) != expected:
        raise ValueError(f"Missing G3 conditions: {sorted(expected - set(grid))}")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    independent = {(trial["task_id"], trial["lr"], trial["epochs"]): trial for trial in g1["trials"]}
    trials, audits = [], []
    for (order, method, lr, epochs), directory in sorted(grid.items()):
        audited = summary.audit(directory, normalizer_lock=normalizers)
        resources = audited["stage_resources"]
        if len(resources) != 4:
            raise ValueError("Missing stage resources")
        for index, stage in enumerate(resources):
            exposures = 0 if method == "frozen_first" and index > 0 else 2048 * epochs
            if stage["image_exposures"] != exposures or stage["steps"] != exposures // 16:
                raise ValueError(f"Changed effective training budget: {directory}")
        parity = None
        if method != "capacity_matched_shared":
            filename = f"predictions_after_00_{order[0]}.jsonl"
            reference_file = Path(independent[(order[0], lr, epochs)]["directory"]) / filename
            parity = (directory / filename).read_bytes() == reference_file.read_bytes()
            if not parity:
                raise ValueError(f"First-task predictions differ from matched G1: {directory}")
        matrix = audited["matrix"]
        normalized_final = {task_id: matrix[-1][index] / scores[task_id] for index, task_id in enumerate(order)}
        family_final = {family: sum(normalized_final[tid] for tid in order if tasks[tid]["family"] == family) / 2
                        for family in ("vqa", "caption")}
        failed_guards = [{key: generation[key] for key in ("stage", "task_id", "empty_fraction", "largest_output_fraction")}
                         for generation in audited["generation"]
                         if generation["empty_fraction"] > 0.01 or generation["largest_output_fraction"] >= 0.9]
        gains = [{"task_id": task_id, "family": tasks[task_id]["family"], "gain": audited["pre_to_post_gain"][index],
                  "normalized_gain": audited["pre_to_post_gain"][index] / scores[task_id]}
                 for index, task_id in enumerate(order) if index > 0]
        trials.append({"directory": str(directory), "order": list(order), "method": method, "lr": lr, "epochs": epochs,
                       "common_condition": (lr, epochs) == COMMON, "matrix": matrix,
                       "pre_task_scores": audited["pre_task_scores"], "subsequent_task_gains": gains,
                       "normalized_final_by_task": normalized_final, "family_normalized_final_score": family_final,
                       "family_balanced_normalized_final_score": sum(family_final.values()) / 2,
                       "generation_guard_failures": failed_guards, "first_task_predictions_identical_to_g1": parity,
                       "stage_resources": resources})
        audits.append(audited)
    for order, lr, epochs in itertools.product(ORDERS, LRS, EPOCHS):
        pair = [trial for trial in trials if trial["order"] == list(order) and trial["lr"] == lr and trial["epochs"] == epochs
                and trial["method"] in ("expand", "capacity_matched_shared")]
        if len({trial["stage_resources"][-1]["parameter_counts"]["experts"] for trial in pair}) != 1:
            raise ValueError("Expansion/capacity-matched final adapter budgets differ")
    replay_candidates = []
    for lr, epochs in itertools.product(LRS, EPOCHS):
        selected = [trial for trial in trials if trial["method"] == "replay_lora" and trial["lr"] == lr and trial["epochs"] == epochs]
        replay_candidates.append({"lr": lr, "epochs": epochs,
                                  "mean_family_balanced_normalized_final_score": sum(trial["family_balanced_normalized_final_score"] for trial in selected) / 4})
    selected_replay = min(replay_candidates, key=lambda row: (-row["mean_family_balanced_normalized_final_score"], row["epochs"], row["lr"]))
    primary = [trial for trial in trials if trial["method"] == "expand" and trial["common_condition"]]
    family_gains = {}
    for family in ("vqa", "caption"):
        gains = [gain for trial in primary for gain in trial["subsequent_task_gains"] if gain["family"] == family]
        family_gains[family] = {"task_transitions": len(gains), "mean_gain": sum(gain["gain"] for gain in gains) / len(gains),
                                "mean_normalized_gain": sum(gain["normalized_gain"] for gain in gains) / len(gains)}
    checks = {family + "_positive_subsequent_gain": result["mean_gain"] > 0 for family, result in family_gains.items()}
    checks["no_primary_expansion_generation_guard_failure"] = all(not trial["generation_guard_failures"] for trial in primary)
    passed = all(checks.values())
    return {"schema_version": 1, "stage": "G3", "input_valid": True, "gate_passed": passed, "g4_allowed": passed,
            "confirmation_requires_pre_eval_hash_lock": True, "manifest_sha256": manifest_hash,
            "g1_report_sha256": g1_hash, "g2_report_sha256": g2_hash, "normalizers_sha256": normalizer_hash,
            "seed": 17, "eval_split": "validation", "checks": checks, "primary_expansion_family_gains": family_gains,
            "gain_aggregation": "All six subsequent task transitions per family across the four fixed orders; native units within family only",
            "replay_candidates": replay_candidates, "selected_replay": selected_replay,
            "reuse_common_replay": (selected_replay["lr"], selected_replay["epochs"]) == COMMON,
            "trials": trials, "audits": audits, "audited_predictions": sum(audit["audited_predictions"] for audit in audits),
            "interpretation": "Development validation only; no new-seed confirmation or general CL superiority claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "manifest", "g1-report", "g2-report", "normalizers", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("g1-sha256", "g2-sha256", "normalizers-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = summary._check_output(args.output)
    if output.exists():
        raise ValueError("G3 report output must be new")
    runs = args.runs.resolve()
    if output == runs or output.is_relative_to(runs) or runs.is_relative_to(output):
        raise ValueError("G3 report must not overlap input runs")
    report = adjudicate(args.runs, args.manifest, args.g1_report, args.g1_sha256, args.g2_report, args.g2_sha256,
                        args.normalizers, args.normalizers_sha256)
    output.mkdir(parents=True)
    (output / "g3.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("gate_passed", "g4_allowed", "checks", "primary_expansion_family_gains", "selected_replay", "audited_predictions")}, indent=2))
    return int(not report["gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
