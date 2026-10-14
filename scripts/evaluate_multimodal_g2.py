"""Independently rescore and adjudicate the predeclared VQA/caption G2 pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts import summarize_multimodal as summary

METHODS = ("shared_lora", "capacity_matched_shared", "replay_lora", "expand")
ORDERS = (("V", "C"), ("C", "V"))


def locked_json(path, expected_hash):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Changed pre-fixed evidence: {path}")
    return json.loads(raw)


def adjudicate(runs, manifest_path, g1_path, g1_hash, normalizer_path, normalizer_hash):
    g1 = locked_json(g1_path, g1_hash)
    normalizers = locked_json(normalizer_path, normalizer_hash)
    manifest_raw = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    if not g1.get("input_valid") or not g1.get("gate_passed") or not g1.get("pilot_allowed"):
        raise ValueError("A passing independently audited G1 is required")
    if g1["manifest_sha256"] != manifest_hash or g1["normalizers_sha256"] != normalizer_hash:
        raise ValueError("Pilot manifest/normalizers differ from passing G1")
    if normalizers["manifest_sha256"] != manifest_hash or normalizers["eval_split"] != "validation":
        raise ValueError("Normalizers must be fixed G1 validation scores for this manifest")
    scores = normalizers["scores"]
    if set(scores) != {"V", "C", "G", "F"} or any(not math.isfinite(v) or v <= 0 for v in scores.values()):
        raise ValueError("Four positive task normalizers are required")
    for task_id, value in scores.items():
        if value != g1["tasks"][task_id]["normalizer_selection"]["score"]:
            raise ValueError("Normalizer differs from the G1 selection")
    expected = {(order, method) for order in ORDERS for method in METHODS}
    root = Path(runs).resolve()
    directories = sorted({path.parent for path in root.rglob("config.json")})
    reference = g1["audits"][0]["config"]
    invariant_fields = ("manifest_sha256", "source_sha256", "assets_sha256", "tokenizer_sha256",
                        "optimizer", "generation", "loss", "torch", "numpy", "cuda",
                        "deterministic_algorithms", "tf32", "assets", "image_root")
    fixed = {"seed": 17, "lr": 0.0003, "epochs": 5, "rank": 8, "alpha": 16.0,
             "batch_size": 16, "micro_batch_size": 8, "image_size": 224,
             "eval_split": "validation", "smoke_output": False, "train_limit": 0,
             "eval_limit": 0, "router_modality": "joint", "router_memory": 128,
             "router_ridge": 1.0, "replay_per_task": 64}
    grid = {}
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        key = (tuple(config["task_order"]), config["method"])
        if key not in expected or key in grid or config["task_ids"] != list(key[0]):
            raise ValueError(f"Unexpected/duplicate pilot condition: {directory}")
        for field, value in fixed.items():
            if config.get(field) != value:
                raise ValueError(f"Wrong G2 setting {field}: {directory}")
        for field in invariant_fields:
            if config.get(field) != reference[field]:
                raise ValueError(f"Changed G1 scientific contract {field}: {directory}")
        if set(config["scorer_locks"]) != set(key[0]):
            raise ValueError("Wrong pilot scorer task set")
        for task_id, lock in config["scorer_locks"].items():
            if lock != g1["scorer_locks"][task_id]:
                raise ValueError(f"Changed scorer: {task_id}")
        grid[key] = directory
    if set(grid) != expected:
        raise ValueError(f"Missing G2 conditions: {sorted(expected - set(grid))}")
    audits = {key: summary.audit(directory, normalizer_lock=normalizers)
              for key, directory in grid.items()}
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    rows, order_reports = [], {}
    for order in ORDERS:
        per_method = {}
        for method in METHODS:
            audited = audits[(order, method)]
            result = json.loads((grid[(order, method)] / "results.json").read_text())
            if result["status"] != "complete" or result["completed"] != 2:
                raise ValueError("Incomplete pilot result")
            for stage in result["stage_resources"]:
                if stage["steps"] != 640 or stage["image_exposures"] != 10240:
                    raise ValueError("Pilot effective training budget differs")
            parity = None
            if method in ("shared_lora", "replay_lora", "expand"):
                filename = f"predictions_after_00_{order[0]}.jsonl"
                current = grid[(order, method)] / filename
                independent = Path(g1["tasks"][order[0]]["common_run"]) / filename
                parity = current.read_bytes() == independent.read_bytes()
                if not parity:
                    raise ValueError(f"First-task predictions differ from matched G1: {current}")
            matrix = result["matrix"]
            row = {"order": list(order), "method": method, "directory": str(grid[(order, method)]),
                   "matrix": matrix, "pre_task_scores": result["pre_task_scores"],
                   "normalized_final_score": sum(matrix[-1][i] / scores[task_id]
                                                 for i, task_id in enumerate(order)) / 2,
                   "first_task_forgetting": matrix[0][0] - matrix[1][0],
                   "first_task_metric": tasks[order[0]]["metric"],
                   "second_task_gain": result["pre_to_post_gain"][1],
                   "second_task_metric": tasks[order[1]]["metric"],
                   "audited_predictions": audited["audited_predictions"],
                   "first_task_predictions_identical_to_g1": parity,
                   "stage_resources": result["stage_resources"]}
            per_method[method] = row
            rows.append(row)
        expansion, shared = per_method["expand"], per_method["shared_lora"]
        checks = {"positive_second_task_gain": expansion["second_task_gain"] > 0,
                  "less_first_task_forgetting_than_shared": expansion["first_task_forgetting"] < shared["first_task_forgetting"],
                  "higher_normalized_final_score_than_shared": expansion["normalized_final_score"] > shared["normalized_final_score"]}
        order_reports["_".join(order)] = {"checks": checks, "passed": all(checks.values()),
             "normalized_final_delta_vs_shared": expansion["normalized_final_score"] - shared["normalized_final_score"],
             "forgetting_delta_vs_shared": expansion["first_task_forgetting"] - shared["first_task_forgetting"]}
    passed = all(value["passed"] for value in order_reports.values())
    return {"schema_version": 1, "stage": "G2", "input_valid": True, "gate_passed": passed,
            "g3_allowed": passed, "g3_launched": False,
            "g1_report_sha256": g1_hash, "normalizers_sha256": normalizer_hash,
            "manifest_sha256": manifest_hash, "seed": 17, "eval_split": "validation",
            "orders": order_reports, "trials": rows,
            "audited_predictions": sum(row["audited_predictions"] for row in rows),
            "audits": list(audits.values()),
            "interpretation": "Validation pilot only, not multi-seed confirmation or general CL superiority."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "manifest", "g1-report", "normalizers", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--g1-sha256", required=True)
    parser.add_argument("--normalizers-sha256", required=True)
    args = parser.parse_args()
    output = summary._check_output(args.output)
    if output.exists():
        raise ValueError("Pilot report output must be new")
    runs = args.runs.resolve()
    if output == runs or output.is_relative_to(runs) or runs.is_relative_to(output):
        raise ValueError("Pilot report must not overlap input runs")
    report = adjudicate(args.runs, args.manifest, args.g1_report, args.g1_sha256,
                        args.normalizers, args.normalizers_sha256)
    output.mkdir(parents=True)
    (output / "g2.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("gate_passed", "g3_allowed", "orders", "audited_predictions")}, indent=2))
    return int(not report["gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
