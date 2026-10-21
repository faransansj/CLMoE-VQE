#!/usr/bin/env python3
"""Compare explicitly collected P0 validation gates on CPU, without training.

l40s-card requires distinct L40S UUIDs and byte-identical training tensors and
validation diagnostics. l40s-4090 measures differences only: it never applies a
numerical tolerance or declares acceptance. Checkpoints are trusted PyTorch
pickle inputs; report digests bind files but are not cryptographic attestations.
Prediction arguments are repeated TASK_ID=PATH pairs for every --tasks entry.
Token diagnostics mean generated-token COUNTS, not unavailable token IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE_TRAINING_FIELDS = (
    "assignments", "replay", "steps", "image_exposures", "target_tokens",
    "current_exposures", "replay_exposures", "task_index", "stage", "epoch", "batch_cursor",
)
DEPLOYMENT_FIELDS = {"manifest", "image_root", "assets", "output", "device"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def scientific_config(config):
    return {key: value for key, value in config.items() if key not in DEPLOYMENT_FIELDS}


def report_identity(report):
    require(report.get("schema_version") == 2 and report.get("stage") == "p0-strip-gate",
            "Only schema-2 explicitly collected P0 gate reports are supported")
    labels, environment, provenance = report["labels"], report["environment"], report["provenance"]
    require(labels["strip"] is False and labels["internal_evaluation_split"] == "validation"
            and labels["eval_split_generated_or_scored"] is False, "Gate must be strip-OFF validation-only")
    require(report["gate"].get("collection_only") is True, "Report must be an independent gate collection")
    require(report["run"]["status"] == "complete", "Gate collection must be complete")
    require(isinstance(provenance.get("lock_verified"), bool), "Provenance must state whether the lock was verified")
    require(labels["prereg_commit"] == provenance["prereg_commit"], "Preregistration commit labels disagree")
    if provenance["lock_verified"]:
        for key in ("prereg_commit", "locked_code_commit", "source_commit"):
            require(isinstance(provenance.get(key), str) and COMMIT.fullmatch(provenance[key]),
                    f"Missing full provenance commit: {key}")
        require(SHA256.fullmatch(provenance.get("prereg_sha256") or ""), "Missing preregistration SHA256")
    hashes = provenance.get("execution_sha256")
    require(isinstance(hashes, dict) and hashes and all(isinstance(v, str) and SHA256.fullmatch(v) for v in hashes.values()),
            "Missing execution file hashes")
    require(SHA256.fullmatch(environment.get("environment_sha256", "")), "Missing environment SHA256")
    for value, label in ((environment.get("gpu_uuid"), "GPU UUID"),
                         (environment.get("gpu_name"), "GPU model"),
                         (report["scheduler"].get("node"), "node")):
        require(isinstance(value, str) and value.strip(), f"Missing {label}")
    config = report["scientific_config"]
    require(type(config.get("seed")) is int, "Scientific config must identify the seed")
    require(config.get("eval_split") == "validation", "Scientific config must use validation")
    require(not config.get("train_limit") and not config.get("eval_limit"), "Partial smoke collections are not gates")
    for key in ("manifest_sha256", "assets_sha256", "tokenizer_sha256"):
        require(config.get(key) and report["input_hashes"].get(key) == config[key],
                f"Missing or inconsistent input hash contract: {key}")
    require(SHA256.fullmatch(config["manifest_sha256"]), "Invalid manifest SHA256")
    for field in ("assets_sha256", "tokenizer_sha256", "source_sha256"):
        require(isinstance(config.get(field), dict) and config[field]
                and all(isinstance(value, str) and SHA256.fullmatch(value) for value in config[field].values()),
                f"Missing or invalid {field}")
    return {"gpu_name": environment["gpu_name"], "gpu_uuid": environment["gpu_uuid"],
            "environment_sha256": environment["environment_sha256"], "node": report["scheduler"]["node"],
            **{key: provenance.get(key) for key in ("lock_verified", "prereg_commit", "prereg_sha256",
                                                    "locked_code_commit", "source_commit", "source_commit_status")}}


def gpu_family(name):
    if name in {"NVIDIA L40S", "L40S"}:
        return "l40s"
    if name in {"NVIDIA GeForce RTX 4090", "GeForce RTX 4090", "RTX 4090"}:
        return "4090"
    raise ValueError(f"Unsupported GPU model (L40 is not L40S): {name}")


def validate_pair(left, right, mode):
    identities = [report_identity(report) for report in (left, right)]
    families = [gpu_family(identity["gpu_name"]) for identity in identities]
    require(families == (["l40s", "l40s"] if mode == "l40s-card" else ["l40s", "4090"]),
            "Left must be L40S; right must match the comparison mode")
    if mode == "l40s-card":
        require(identities[0]["gpu_uuid"] != identities[1]["gpu_uuid"], "Card comparison requires distinct GPU UUIDs")
    require(scientific_config(left["scientific_config"]) == scientific_config(right["scientific_config"]),
            "Scientific configurations differ beyond deployment paths/device")
    require(left["input_hashes"] == right["input_hashes"], "Input hashes differ")
    require(identities[0]["lock_verified"] == identities[1]["lock_verified"], "Provenance mismatch: lock_verified")
    if identities[0]["lock_verified"]:
        for key in ("prereg_commit", "prereg_sha256", "locked_code_commit", "source_commit"):
            require(identities[0][key] == identities[1][key], f"Provenance mismatch: {key}")

    def executed(report):
        # The selected deployment jobs file legitimately differs per host (paths only;
        # the runner already proved scientific argv equality). Every other file must match.
        return {name: digest for name, digest in report["provenance"]["execution_sha256"].items()
                if not name.startswith("configs/") or name == "configs/multimodal_g4_jobs.json"}
    require(executed(left) == executed(right), "Execution source file hashes differ between the two collections")
    return identities


def bound_digest(report, path, expected_name):
    path = Path(path)
    require(path.name == expected_name, f"Expected explicitly collected file named {expected_name}")
    digest = sha256_file(path)
    require(report["run"]["files_sha256"].get(expected_name) == digest,
            f"Input is not bound by the report digest: {path}")
    return digest


def compare_values(left, right):
    """Exact logical tensor bytes (including signed zero/NaN payload), not torch.equal."""
    import numpy as np
    import torch
    result = {"bitwise_equal": True, "compared_tensors": 0, "mismatches": 0,
              "first_mismatch": None, "max_abs_diff_finite": 0.0, "nonfinite_pairs": 0}

    def mismatch(path, reason):
        result["bitwise_equal"] = False
        result["mismatches"] += 1
        if result["first_mismatch"] is None:
            result["first_mismatch"] = {"path": path, "reason": reason}

    def visit(a, b, path):
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                mismatch(path, "tensor/type")
                return
            if a.dtype != b.dtype or a.shape != b.shape or a.layout != b.layout:
                mismatch(path, "tensor dtype/shape/layout")
                return
            require(a.layout == torch.strided and not a.is_quantized, "Unsupported checkpoint tensor layout")
            require(a.device.type == b.device.type == "cpu", "Comparator only accepts CPU-loaded tensors")
            result["compared_tensors"] += 1
            x, y = a.detach().resolve_conj().resolve_neg().contiguous(), b.detach().resolve_conj().resolve_neg().contiguous()
            if x.reshape(-1).view(torch.uint8).numpy().tobytes() != y.reshape(-1).view(torch.uint8).numpy().tobytes():
                mismatch(path, "tensor bytes")
            if a.numel() and (a.is_floating_point() or a.is_complex()):
                x, y = x.to(torch.complex128 if a.is_complex() else torch.float64), y.to(torch.complex128 if b.is_complex() else torch.float64)
                finite = torch.isfinite(x) & torch.isfinite(y)
                result["nonfinite_pairs"] += int((~finite).sum())
                if bool(finite.any()):
                    delta = float((x[finite] - y[finite]).abs().max())
                    if math.isfinite(delta):
                        result["max_abs_diff_finite"] = max(result["max_abs_diff_finite"], delta)
                    else:
                        result["nonfinite_pairs"] += 1
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                mismatch(path, "array/type")
            elif a.dtype != b.dtype or a.shape != b.shape:
                mismatch(path, "array dtype/shape")
            else:
                require(not a.dtype.hasobject, "Object arrays are not a byte comparison contract")
                if a.tobytes() != b.tobytes():
                    mismatch(path, "array bytes")
        elif type(a) is not type(b):
            mismatch(path, "value type")
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b), key=str):
                if key not in a or key not in b:
                    mismatch(f"{path}/{key}", "missing key")
                else:
                    visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                mismatch(path, "sequence length")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}/{index}")
        elif isinstance(a, float):
            if struct.pack("!d", a) != struct.pack("!d", b):
                mismatch(path, "scalar float bytes")
        elif isinstance(a, (str, int, bool, bytes, type(None))):
            if a != b:
                mismatch(path, "scalar value")
        elif isinstance(a, np.generic):
            if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                mismatch(path, "numpy scalar bytes")
        else:
            raise ValueError(f"Unsupported training-state value at {path}: {type(a).__name__}")

    visit(left, right, "/")
    return result


def compare_checkpoints(left_path, right_path, left_report, right_report):
    import torch
    digests = [bound_digest(report, path, "latest.pt") for report, path in
               ((left_report, left_path), (right_report, right_path))]
    # Existing training checkpoints include Python/NumPy RNG pickle objects.
    # Only load explicitly supplied trusted local inputs; never evaluate score fields.
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in (left_path, right_path)]
    for checkpoint, report in zip(checkpoints, (left_report, right_report)):
        require(checkpoint["schema_version"] == 1, "Unsupported training checkpoint schema")
        require(checkpoint["settings"] == report["scientific_config"], "Checkpoint settings do not match report")
        require(checkpoint["state"]["task_index"] == len(report["scientific_config"]["task_order"]),
                "Checkpoint has not completed every task")
    a, b = checkpoints
    result = {key: compare_values(a[key], b[key]) for key in ("model", "router")}
    result["training_state"] = compare_values(
        {key: a["state"][key] for key in STATE_TRAINING_FIELDS},
        {key: b["state"][key] for key in STATE_TRAINING_FIELDS})
    result["bitwise_equal"] = all(value["bitwise_equal"] for value in result.values())
    result["sha256"] = {"left": digests[0], "right": digests[1]}
    result["selected_training_fields"] = list(STATE_TRAINING_FIELDS)
    return result


def prediction_paths(values, tasks):
    result = {}
    for value in values:
        task, separator, path = value.partition("=")
        require(separator and task and path and task not in result, "Predictions require unique TASK_ID=PATH pairs")
        result[task] = Path(path)
    require(set(result) == set(tasks), "Supply exactly one explicit prediction path for every task")
    return result


def load_predictions(path, rows):
    with path.open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream]
    require([row["id"] for row in predictions] == [row["id"] for row in rows],
            "Prediction IDs/order must exactly cover manifest validation rows")
    for row in predictions:
        require(isinstance(row.get("prediction"), str) and "expert" in row, "Malformed prediction row")
        generation = row["generation"]
        for field in ("generated_tokens", "max_new_tokens"):
            require(type(generation.get(field)) is int and generation[field] > 0, f"Missing generation {field}")
        require(generation["generated_tokens"] <= generation["max_new_tokens"], "Invalid token count")
        for field in ("eos", "empty", "repeated_trigram", "limit_reached"):
            require(type(generation.get(field)) is bool, f"Missing generation {field}")
    return predictions


def per_image_cider(task, left, right, expected_lock):
    """Reuse the project's actual fixed-train-IDF implementation, with batched PTB.

    Its public score API averages a corpus. Singleton CiderScorer corpora recover
    per-image values from that same implementation without changing IDF or
    launching one Java tokenizer process per image. No held-out IDF is fitted.
    """
    from sema_multimodal.metrics import TaskScorer, _caption_dependencies, _fixed_cider_score, _tokenize
    scorer = TaskScorer(task)
    require(scorer.lock == expected_lock, "Local CIDEr scorer does not match collected scorer lock")
    rows = task["validation"]
    captions = {("reference", index): scorer._references(row) for index, row in enumerate(rows)}
    for label, predictions in (("left", left), ("right", right)):
        captions.update({(label, index): [row["prediction"]] for index, row in enumerate(predictions)})
    tokenized = _tokenize(captions)
    CiderScorer, _, _, _, _ = _caption_dependencies()
    scores = []
    for index, row in enumerate(rows):
        require(row["id"] not in scorer._train_images and row["image_sha256"] not in scorer._train_hashes,
                "Validation image overlaps the fixed train-IDF corpus")
        values = []
        for label in ("left", "right"):
            corpus = CiderScorer(n=4, sigma=6.0)
            corpus += (tokenized[(label, index)][0], tokenized[("reference", index)])
            values.append(_fixed_cider_score(corpus, scorer._df, scorer._train_documents))
        scores.append({"id": row["id"], "left": values[0], "right": values[1], "delta_right_minus_left": values[1] - values[0]})
    return {"units": "native_cider_d", "scorer_lock": scorer.lock, "per_image": scores,
            "mean_delta_right_minus_left": math.fsum(row["delta_right_minus_left"] for row in scores) / len(scores)}


def compare_predictions(task, left_path, right_path, reports):
    task_id = task["task_id"]
    last = len(reports[0]["scientific_config"]["task_order"]) - 1
    name = f"predictions_after_{last:02d}_{task_id}.jsonl"
    digests = [bound_digest(report, path, name) for report, path in zip(reports, (left_path, right_path))]
    left, right = [load_predictions(path, task["validation"]) for path in (left_path, right_path)]
    images = []
    for a, b in zip(left, right):
        ga, gb = a["generation"], b["generation"]
        images.append({"id": a["id"], "prediction_equal": a["prediction"] == b["prediction"],
                       "expert_equal": a["expert"] == b["expert"], "eos_equal": ga["eos"] == gb["eos"],
                       "eos_left": ga["eos"], "eos_right": gb["eos"],
                       "generated_tokens_left": ga["generated_tokens"], "generated_tokens_right": gb["generated_tokens"],
                       "generated_tokens_delta_right_minus_left": gb["generated_tokens"] - ga["generated_tokens"],
                       "generation_diagnostics_equal": ga == gb})
    result = {"sha256": {"left": digests[0], "right": digests[1]},
              "file_bytes_equal": digests[0] == digests[1], "prediction_records_equal": left == right,
              "token_comparison": "generated_token_count_only; token IDs are not collected",
              "per_image": images}
    if task["metric"] == "cider_d":
        result["cider_d"] = per_image_cider(task, left, right, reports[0]["scientific_config"]["scorer_locks"][task_id])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("l40s-card", "l40s-4090"))
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-report", required=True, type=Path)
        parser.add_argument(f"--{side}-checkpoint", required=True, type=Path, help="Trusted collected latest.pt")
        parser.add_argument(f"--{side}-predictions", required=True, action="append", metavar="TASK_ID=PATH")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tasks", required=True, nargs="+", help="Exact ordered task IDs from both scientific configs")
    parser.add_argument("--output", required=True, type=Path, help="Exclusive-create comparison JSON")
    args = parser.parse_args(argv)
    require(not args.output.exists(), f"Refusing existing output: {args.output}")
    reports = [load_json(path) for path in (args.left_report, args.right_report)]
    identities = validate_pair(*reports, args.mode)
    config = reports[0]["scientific_config"]
    require(args.tasks == config["task_order"] and len(set(args.tasks)) == len(args.tasks), "Task order differs from collected run")
    require(sha256_file(args.manifest) == config["manifest_sha256"], "Explicit manifest hash differs from collected inputs")
    from sema_multimodal.data import validate_manifest
    manifest = validate_manifest(load_json(args.manifest))
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    require(all(task in by_id for task in args.tasks), "Requested task missing from manifest")
    require(any(by_id[task]["metric"] == "cider_d" for task in args.tasks), "Comparison requires a caption task for per-image CIDEr")
    paths = [prediction_paths(values, args.tasks) for values in (args.left_predictions, args.right_predictions)]
    checkpoint = compare_checkpoints(args.left_checkpoint, args.right_checkpoint, *reports)
    predictions = {task: compare_predictions(by_id[task], paths[0][task], paths[1][task], reports) for task in args.tasks}
    result = {"schema_version": 1, "mode": args.mode, "comparison_device": "cpu",
              "left": {**identities[0], "report_path": str(args.left_report), "report_sha256": sha256_file(args.left_report)},
              "right": {**identities[1], "report_path": str(args.right_report), "report_sha256": sha256_file(args.right_report)},
              "seed": config["seed"], "manifest_sha256": config["manifest_sha256"], "tasks": args.tasks,
              "checkpoint": checkpoint, "validation_predictions": predictions}
    exit_code = 0
    if args.mode == "l40s-card":
        result["label"] = "same_model_distinct_card_bitwise_gate"
        result["passed"] = checkpoint["bitwise_equal"] and all(value["file_bytes_equal"] and value["prediction_records_equal"] for value in predictions.values())
        exit_code = 0 if result["passed"] else 3
    else:
        result["label"] = "cross_gpu_tolerance_measurement_only"
        result["acceptance_thresholds"] = None
        result["acceptance_decision"] = "not_made"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "label": result["label"]}))
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(f"compare_p0_gate: {error}", file=sys.stderr)
        sys.exit(2)
