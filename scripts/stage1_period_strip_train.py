#!/usr/bin/env python3
"""Stage 1: single-factor causal test of the terminal-period hypothesis.

Trains the G4 independent Flickr (F) caption expert through the UNTOUCHED
sema_multimodal.run training path with exactly the original G4 F-independent
settings (lr 3e-4, 5 epochs, batch 16 / micro 8, rank 8 / alpha 16, image size
224, joint router defaults, replay_per_task 64, deterministic flags, same full
manifest/assets), task F only, task index 0 — with ONLY ONE factor changed in
mode "strip": the training target. A runtime monkeypatch of
sema_multimodal.data.training_rows makes caption answers lose trailing
whitespace and ONE terminal '.' before the tokenizer sees them. Scoring
references (train IDF, validation references) are untouched. No sema_* source
is edited; nothing else differs.

Subcommands (one SLURM job each; exclusive outputs):
  fidelity  unpatched training, seed 137, internal evaluations on the Flickr
            VALIDATION split (never the confirmation eval split). The trained
            adapter must be BITWISE equal to G4 F_independent_..._seed137
            latest.pt expert tensors (per-tensor torch.equal; max abs diff
            reported); its D0 decoding on validation must reproduce the
            stage-0.5 probe's F/137 D0 rows. On failure: report, exit 3, and
            the strip runs must not be submitted.
  strip     seeds 137/149/163 with the period-strip patch (requires a passed
            fidelity gate), then evaluation on F VALIDATION ONLY: D0 greedy
            (64) with per-step P([SEP])/P('.') diagnostics plus D1 (EOS bias
            +4) as reference, scored with the untouched fixed train-IDF
            CIDEr-D, paired against the stage-0.5 F-D0 and C-D0 of the same
            seed on the same 256 images (overall and both-EOS-only).
  evaluate  recovery: evaluate/score existing strip run dirs without training.

Labels: single-factor contrast on the validation split; no causal claim
beyond this contrast; no variant is selected as a method on eval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

G4_BANK_PREFIX = "artifacts/multimodal_runs/g4"
F_BANK = "F_independent_lr0.0003_ep5_seed137"
F_BANK_CONFIG_SHA256 = "353a152604a7bd4087ae9a82bc52d3af9ceaf5106ed47c92c1ec5ae2f929d6cb"
F_BANK_LATEST_SHA256 = "f65d61252faf22940a2598412f529dee32894aca46c40a473ff9c17a94df014e"
PROBE_GZ_SHA256 = "9a2cbbc3acf8398633585c866ec9451e725da76fc09591d52c92f93fb76893a9"
PROBE_REPORT_SHA256 = "5779a9d8bbdaddd92f601ea004f1da3f7f506cd9528c7d5282ca4075e2839e6a"
STRIP_SEEDS = (137, 149, 163)
VALIDATION_ROWS = 256
MICRO_BATCH = 8
SCHEMA = 1
TOOL = "scripts/stage1_period_strip_train.py"

EVAL_VARIANTS = (
    {"name": "D0", "limit": 64, "eos_bias": 0.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D1_b4", "limit": 64, "eos_bias": 4.0, "repetition_penalty": 1.0, "period_stop": False},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def round6(value: float) -> float:
    return round(float(value), 6)


def exclusive_create_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        temporary.unlink()
        raise FileExistsError(f"Refusing to overwrite existing output: {path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.unlink()


def exclusive_create_gzip(path: Path, records) -> None:
    import gzip
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as text:
                for record in records:
                    text.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                text.flush()
                gz.flush()


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")

    def write(self, record: dict) -> None:
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "stage05_probe", ROOT / "scripts/stage05_decoding_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scheduler_allocation() -> dict:
    job_id = os.environ.get("SLURM_JOB_ID")
    allocation = {"slurm_job_id": job_id,
                  "slurm_env": {key: os.environ.get(key) for key in
                                ("SLURM_NODELIST", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
                                 "SLURM_MEM_PER_NODE", "SLURM_TIMELIMIT", "SLURM_QOS")}
                  if job_id else None, "scontrol": None}
    if job_id:
        result = subprocess.run(["scontrol", "show", "job", job_id],
                                capture_output=True, text=True, check=False)
        fields = {}
        for token in (result.stdout + " " + result.stderr).split():
            if "=" in token:
                key, _, value = token.partition("=")
                fields.setdefault(key, value)
        if fields:
            allocation["scontrol"] = {key: fields[key] for key in
                                      ("JobId", "Partition", "NodeList", "TresPerNode", "NumCPUs",
                                       "Mem", "Timelimit", "Gres", "QOS", "StartTime", "JobState")
                                      if key in fields}
    return allocation


# ---------------------------------------------------------------- patch
def install_strip_patch():
    """Monkeypatch sema_multimodal.data.training_rows: caption answers lose
    trailing whitespace and ONE terminal '.'; everything else passes through.
    Returns the original function for restoration."""
    import sema_multimodal.data as mmdata

    original = mmdata.training_rows

    def strip_terminal_period(text: str) -> str:
        stripped = text.rstrip()
        if stripped.endswith("."):
            stripped = stripped[:-1]
        return stripped

    def patched_training_rows(rows, seed, epoch):
        result = original(rows, seed, epoch)
        return [dict(item, answer=(strip_terminal_period(item["answer"])
                                   if row.get("family") == "caption" else item["answer"]))
                for row, item in zip(rows, result)]

    mmdata.training_rows = patched_training_rows
    return original


# ---------------------------------------------------------------- training
def bank_config() -> dict:
    path = ROOT / f"{G4_BANK_PREFIX}/{F_BANK}/config.json"
    if not path.is_file():
        raise FileNotFoundError(f"G4 bank config missing: {path}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != F_BANK_CONFIG_SHA256:
        raise ValueError(f"G4 bank config differs from the recorded binding: {path}")
    return json.loads(raw)


def training_args(config: dict, output: Path) -> list[str]:
    return ["--manifest", config["manifest"], "--assets", config["assets"],
            "--image-root", config["image_root"], "--output", str(output),
            "--method", config["method"], "--task-ids", config["task_order"][0],
            "--eval-split", "validation", "--seed", str(config["seed"]),
            "--lr", repr(config["lr"]), "--epochs", str(config["epochs"]),
            "--rank", str(config["rank"]), "--alpha", repr(config["alpha"]),
            "--batch-size", str(config["batch_size"]),
            "--micro-batch-size", str(config["micro_batch_size"]),
            "--image-size", str(config["image_size"]), "--device", config["device"],
            "--router-modality", config["router_modality"],
            "--router-memory", str(config["router_memory"]),
            "--router-ridge", repr(config["router_ridge"]),
            "--replay-per-task", str(config["replay_per_task"])]


def run_training(output: Path, config: dict, patch: bool, log: JsonlLog) -> tuple[dict, float]:
    """Call the untouched sema_multimodal.run.main with the original settings;
    eval-split validation keeps every internal evaluation off the eval split."""
    import sema_multimodal.run as runner

    check(not output.exists(), f"Refusing existing run dir (exclusive create): {output}")
    restore = install_strip_patch() if patch else None
    argv = training_args(config, output)
    log.write({"event": "training_start", "output": str(output), "patched": bool(patch),
               "argv": argv})
    started = time.perf_counter()
    try:
        result = runner.main(argv)
    finally:
        if restore is not None:
            import sema_multimodal.data as mmdata
            mmdata.training_rows = restore
    seconds = round(time.perf_counter() - started, 3)
    log.write({"event": "training_done", "output": str(output), "seconds": seconds,
               "steps": result["resources"]["steps"],
               "training_seconds": round(result["resources"]["training_seconds"], 3),
               "status": result["status"]})
    return result, seconds


def compare_expert_tensors(mine_path: Path, g4_path: Path) -> dict:
    import torch

    mine = torch.load(mine_path, map_location="cpu", weights_only=False)
    reference = torch.load(g4_path, map_location="cpu", weights_only=False)
    mine_state, reference_state = mine["model"], reference["model"]
    scalar_fields = {}
    for field in ("rank", "alpha", "num_experts", "active_expert", "trainable_expert", "training"):
        scalar_fields[field] = {"mine": mine_state.get(field), "g4": reference_state.get(field),
                                "equal": mine_state.get(field) == reference_state.get(field)}
    max_abs, tensor_count, bitwise, differing = 0.0, 0, True, []
    for name in sorted(set(mine_state["projections"]) | set(reference_state["projections"])):
        mine_proj = mine_state["projections"].get(name, {})
        reference_proj = reference_state["projections"].get(name, {})
        for key in sorted(set(mine_proj) | set(reference_proj)):
            if key not in mine_proj or key not in reference_proj:
                bitwise = False
                differing.append({"tensor": f"{name}/{key}", "missing": True})
                continue
            a, b = mine_proj[key], reference_proj[key]
            tensor_count += 1
            equal = torch.equal(a, b)
            bitwise = bitwise and equal
            diff = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
            max_abs = max(max_abs, diff)
            if not equal and len(differing) < 8:
                differing.append({"tensor": f"{name}/{key}", "max_abs_diff": diff})
    return {"bitwise_equal": bitwise and all(value["equal"] for value in scalar_fields.values()),
            "tensors_compared": tensor_count, "max_abs_diff": max_abs,
            "scalar_fields": scalar_fields, "differing_preview": differing,
            "mine_sha256": sha256_bytes(mine_path.read_bytes()),
            "g4_sha256": sha256_bytes(g4_path.read_bytes()),
            "g4_expected_sha256": F_BANK_LATEST_SHA256,
            "g4_sha_matches_binding": sha256_bytes(g4_path.read_bytes()) == F_BANK_LATEST_SHA256}


# ---------------------------------------------------------------- evaluation
def load_validation(data_config: dict) -> dict:
    import sema_multimodal.data as mmdata

    manifest_path = Path(data_config["manifest"])
    raw = manifest_path.read_bytes()
    if sha256_bytes(raw) != data_config["manifest_sha256"]:
        raise ValueError("Manifest bytes differ from the recorded hash")
    manifest = json.loads(raw)
    mmdata.validate_manifest(manifest)  # structure-only; the used images are verified below
    task_F = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    validation, train = task_F["validation"], task_F["train"]
    check(len(validation) == VALIDATION_ROWS, f"F validation must have {VALIDATION_ROWS} rows")
    verifier = mmdata.ImageVerifier(data_config["image_root"])
    for row in validation:
        if verifier.verify(row["image"]) != row["image_sha256"]:
            raise ValueError(f"Validation image bytes differ: {row['id']}")
    return {"manifest": manifest, "validation": validation, "train": train,
            "rows": mmdata.public_rows(validation), "image_root": data_config["image_root"],
            "images_verified": len(validation)}


def decode_eval(probe, model, token_ids, rows, image_root, device) -> dict:
    import torch

    variant_rows = {variant["name"]: [] for variant in EVAL_VARIANTS}
    seconds = {variant["name"]: 0.0 for variant in EVAL_VARIANTS}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        for start in range(0, len(rows), MICRO_BATCH):
            batch_rows = rows[start:start + MICRO_BATCH]
            states, mask = model._question_states(batch_rows, image_root)
            for variant in EVAL_VARIANTS:
                started = time.perf_counter()
                outcome = probe.decode_variant(
                    model, states, mask, limit=variant["limit"], eos_bias=variant["eos_bias"],
                    repetition_penalty=variant["repetition_penalty"],
                    period_stop=variant["period_stop"],
                    diagnostics=variant["name"] == "D0", token_ids=token_ids)
                seconds[variant["name"]] += time.perf_counter() - started
                for offset, (prediction, info) in enumerate(zip(outcome["predictions"],
                                                                outcome["diagnostics"])):
                    record = {"variant": variant["name"], "row_index": start + offset,
                              "prediction": prediction, **info}
                    if variant["name"] == "D0":
                        record["d0_step_diag"] = outcome["steps"][offset]
                    variant_rows[variant["name"]].append(record)
            del states, mask
    return {"variant_rows": variant_rows,
            "seconds": {name: round(value, 3) for name, value in seconds.items()}}


def load_stage05_baselines(gz_path: str) -> dict:
    import gzip

    raw = Path(gz_path).read_bytes()
    if sha256_bytes(raw) != PROBE_GZ_SHA256:
        raise ValueError(f"Stage-0.5 predictions differ from the recorded hash: {gz_path}")
    baselines = {"F_D0": {}, "C_D0": {}}
    with gzip.open(gz_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["variant"] != "D0" or record["model"] not in ("F", "C"):
                continue
            cell = baselines[f"{record['model']}_D0"].setdefault(record["seed"], {})
            check(record["row_index"] not in cell, "Duplicate baseline row")
            cell[record["row_index"]] = record
    for key, per_seed in baselines.items():
        for seed, cell in per_seed.items():
            check(len(cell) == VALIDATION_ROWS, f"{key}/{seed} must have {VALIDATION_ROWS} rows")
    return baselines


def paired_delta(per_image, baseline_per_image, mask=None) -> dict:
    deltas = [a - b for a, b in zip(per_image, baseline_per_image)]
    selected = deltas if mask is None else [d for d, keep in zip(deltas, mask) if keep]
    return {"rows": len(selected),
            "mean": round6(math.fsum(selected) / len(selected)) if selected else None}


def evaluate_run_dir(run_dir: Path, run_label: str, probe, data: dict, baselines: dict,
                     log: JsonlLog) -> dict:
    import random

    import numpy as np
    import torch

    from sema_multimodal.metrics import TaskScorer
    from sema_multimodal.model import MultimodalReadout

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    check(config["method"] == "independent" and config["task_order"] == ["F"],
          f"Unexpected stage-1 run config: {run_dir}")
    check(config["eval_split"] == "validation", f"Stage-1 runs must evaluate on validation: {run_dir}")
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = MultimodalReadout(config["assets"], image_size=config["image_size"],
                              rank=config["rank"], alpha=config["alpha"],
                              device=config["device"])
    token_ids = probe.resolve_token_ids(model.tokenizer)
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    check(checkpoint.get("schema_version") == 1, f"Checkpoint schema: {run_dir}")
    check(checkpoint.get("settings") == config, f"Checkpoint settings mismatch: {run_dir}")
    check(checkpoint.get("state", {}).get("assignments") == [0], f"Assignment must be [0]: {run_dir}")
    bank_state = checkpoint["model"]
    check(bank_state.get("num_experts") == 1 and bank_state.get("rank") == config["rank"]
          and math.isclose(bank_state.get("alpha"), config["alpha"], rel_tol=1e-12),
          f"Checkpoint capacity mismatch: {run_dir}")
    model.load_checkpoint_state(bank_state)
    model._freeze_all()
    model.eval()
    model.set_expert(0)
    digest_before = probe._tensor_digest(model)
    expert_before = probe._expert_digest(model)
    rows, validation = data["rows"], data["validation"]
    device = torch.device(config["device"])
    decode_started = time.perf_counter()
    decoded = decode_eval(probe, model, token_ids, rows, data["image_root"], device)
    decode_seconds = round(time.perf_counter() - decode_started, 3)
    expert_after = probe._expert_digest(model)
    digest_after = probe._tensor_digest(model)
    check(expert_before == expert_after, f"Expert tensors changed during evaluation: {run_dir}")
    check(digest_before == digest_after, f"Whole model changed during evaluation: {run_dir}")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    scorer = TaskScorer({"task_id": "F", "family": "caption", "metric": "cider_d",
                         "train": data["train"]})
    check(scorer.lock == config["scorer_locks"]["F"],
          f"Scorer lock drift against the run config: {run_dir}")
    log.write({"event": "scorer_built", "run_dir": str(run_dir),
               "document_frequency_sha256": scorer.lock["document_frequency_sha256"]})

    strings = {probe.normalize_caption(ref) for row in data["train"] for ref in row["references"]}
    strings.update(probe.normalize_caption(ref) for row in validation
                   for ref in row["references"])
    for records in decoded["variant_rows"].values():
        strings.update(probe.normalize_caption(record["prediction"]) for record in records)
    cache = probe.caption_cache(sorted(strings))

    results, prediction_records = {}, []
    for variant in EVAL_VARIANTS:
        name = variant["name"]
        records = decoded["variant_rows"][name]
        predictions = [record["prediction"] for record in records]
        scores = probe.score_variant(scorer, cache, validation, predictions,
                                     f"{run_label}/{name}")
        summary = probe.variant_summary(records)
        summary.update({"cider_d": round6(scores["cider_d"]), "bleu4": round6(scores["bleu4"]),
                        "per_image_mean": round6(scores["mean_per_image"])})
        base_F = [baselines["F_D0"][seed][i]["cider_d_per_image"] for i in range(VALIDATION_ROWS)]
        base_C = [baselines["C_D0"][seed][i]["cider_d_per_image"] for i in range(VALIDATION_ROWS)]
        eos_F = [baselines["F_D0"][seed][i]["eos"] for i in range(VALIDATION_ROWS)]
        eos_C = [baselines["C_D0"][seed][i]["eos"] for i in range(VALIDATION_ROWS)]
        check(all(baselines["F_D0"][seed][i]["id"] == validation[i]["id"]
                  and baselines["C_D0"][seed][i]["id"] == validation[i]["id"]
                  for i in range(0, VALIDATION_ROWS, 64)),
              "Baseline ids disagree with the validation split")
        mask_F = [record["eos"] and base for record, base in zip(records, eos_F)]
        mask_C = [record["eos"] and base for record, base in zip(records, eos_C)]
        summary["paired_vs_F_D0_same_seed"] = paired_delta(scores["per_image"], base_F)
        summary["paired_vs_C_D0_same_seed"] = paired_delta(scores["per_image"], base_C)
        summary["paired_vs_F_D0_both_eos"] = paired_delta(scores["per_image"], base_F, mask_F)
        summary["paired_vs_C_D0_both_eos"] = paired_delta(scores["per_image"], base_C, mask_C)
        if name == "D0":
            summary["d0_diagnostics"] = probe.summarize_steps(
                [record["d0_step_diag"] for record in records],
                [record["eos"] for record in records])
        results[name] = {"summary": summary}
        for index, record in enumerate(records):
            out = {"run": run_label, "seed": seed, "run_dir": str(run_dir),
                   "variant": name, "row_index": index,
                   "id": validation[index]["id"],
                   "image_sha256": validation[index]["image_sha256"],
                   "prediction": record["prediction"],
                   "generated_tokens": record["generated_tokens"],
                   "max_new_tokens": record["max_new_tokens"],
                   "eos": record["eos"], "period_stop": record["period_stop"],
                   "empty": record["empty"], "repeated_trigram": record["repeated_trigram"],
                   "limit_reached": record["limit_reached"],
                   "period_argmax_steps": record["period_argmax_steps"],
                   "contains_period": record["contains_period"],
                   "cider_d_per_image": round6(scores["per_image"][index])}
            if name == "D0":
                out["d0_step_diag"] = record["d0_step_diag"]
            prediction_records.append(out)
    return {"run_dir": str(run_dir), "run_label": run_label, "seed": seed,
            "config_sha256": sha256_bytes((run_dir / "config.json").read_bytes()),
            "latest_sha256": sha256_bytes((run_dir / "latest.pt").read_bytes()),
            "adapter_proof": {"whole_model_digest_pre": digest_before,
                              "whole_model_digest_post": digest_after,
                              "whole_model_identical_pre_post": digest_before == digest_after,
                              "expert_digest_pre": expert_before,
                              "expert_digest_post": expert_after,
                              "expert_tensors_identical_pre_post": expert_before == expert_after},
            "decode_seconds": decode_seconds, "variant_seconds": decoded["seconds"],
            "peak_gpu_bytes": peak, "results": results,
            "prediction_records": prediction_records}


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    fidelity = sub.add_parser("fidelity")
    fidelity.add_argument("--output-dir", required=True)
    fidelity.add_argument("--report", required=True)
    fidelity.add_argument("--log", required=True)
    fidelity.add_argument("--probe-gz", required=True)

    strip = sub.add_parser("strip")
    strip.add_argument("--output-root", required=True)
    strip.add_argument("--report", required=True)
    strip.add_argument("--predictions", required=True)
    strip.add_argument("--log", required=True)
    strip.add_argument("--probe-gz", required=True)
    strip.add_argument("--fidelity-report", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--log", required=True)
    evaluate.add_argument("--probe-gz", required=True)
    evaluate.add_argument("--fidelity-report", required=True)

    args = parser.parse_args(argv)
    started = time.perf_counter()
    started_utc = utc_now()
    probe = load_probe_module()
    log = JsonlLog(Path(args.log))
    log.write({"event": "start", "mode": args.mode, "utc": started_utc,
               "script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
               "job_id": os.environ.get("SLURM_JOB_ID")})

    config = bank_config()
    if args.mode == "fidelity":
        output = Path(args.output_dir)
        result, seconds = run_training(output, config, patch=False, log=log)
        comparison = compare_expert_tensors(output / "latest.pt",
                                            ROOT / f"{G4_BANK_PREFIX}/{F_BANK}/latest.pt")
        data = load_validation(config)
        baselines = load_stage05_baselines(args.probe_gz)
        fidelity_eval = evaluate_run_dir(output, "fidelity_seed137", probe, data,
                                         baselines, log)
        d0_records = fidelity_eval["prediction_records"]
        probe_records = [baselines["F_D0"][137][i] for i in range(VALIDATION_ROWS)]
        predictions_match = all(
            d0_records[i]["prediction"] == probe_records[i]["prediction"]
            and d0_records[i]["generated_tokens"] == probe_records[i]["generated_tokens"]
            and d0_records[i]["eos"] == probe_records[i]["eos"]
            for i in range(VALIDATION_ROWS))
        run_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        settings_diff = {field: {"stage1": run_config.get(field), "g4": config.get(field)}
                         for field in sorted(set(run_config) | set(config))
                         if run_config.get(field) != config.get(field)}
        report = {
            "schema_version": SCHEMA, "stage": "stage1-fidelity",
            "mode": "fidelity", "seed": 137,
            "started_utc": started_utc, "finished_utc": utc_now(),
            "runtime_seconds_total": round(time.perf_counter() - started, 3),
            "scheduler": scheduler_allocation(),
            "tool": {"path": TOOL, "sha256": sha256_bytes(Path(__file__).resolve().read_bytes())},
            "unpatched_training": {"output_dir": str(output),
                                   "wall_seconds": seconds,
                                   "resources": result["resources"],
                                   "pre_task_scores": result["pre_task_scores"],
                                   "matrix": result["matrix"],
                                   "metrics": result["metrics"]},
            "adapter_comparison": comparison,
            "gate": {"bitwise_equal": comparison["bitwise_equal"],
                     "max_abs_diff": comparison["max_abs_diff"],
                     "validation_D0_matches_stage05_F_D0": predictions_match,
                     "passed": comparison["bitwise_equal"]
                               and comparison["max_abs_diff"] == 0.0
                               and predictions_match},
            "settings_diff_vs_g4_config": settings_diff,
            "fidelity_validation_D0_summary": fidelity_eval["results"]["D0"]["summary"],
            "labels": {"training": True, "diagnostic_only": True,
                       "note": ("fidelity gate for the strip runs; the confirmation eval split "
                                "was never generated or scored")},
        }
        exclusive_create_text(Path(args.report),
                              json.dumps(report, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n")
        log.write({"event": "fidelity_gate", "gate": report["gate"]})
        print(json.dumps({"report": str(args.report), "gate": report["gate"]}), flush=True)
        if not report["gate"]["passed"]:
            print("FIDELITY GATE FAILED — strip runs not authorized", flush=True)
            raise SystemExit(3)
        return

    # strip / evaluate: require a passed fidelity gate
    fidelity_report_path = Path(args.fidelity_report)
    fidelity_record = json.loads(fidelity_report_path.read_text(encoding="utf-8"))
    check(fidelity_record.get("stage") == "stage1-fidelity", "Fidelity report mismatch")
    check(fidelity_record["gate"]["passed"] is True,
          "Fidelity gate has not passed; strip runs are not authorized")
    output_root = Path(args.output_root)
    data = load_validation(config)
    baselines = load_stage05_baselines(args.probe_gz)
    runs, all_records = {}, []
    for seed in STRIP_SEEDS:
        run_dir = output_root / f"strip_seed{seed}"
        if args.mode == "strip":
            if run_dir.exists():
                check((run_dir / "latest.pt").is_file(),
                      f"Existing run dir without checkpoint; use a new dir: {run_dir}")
                log.write({"event": "reuse_existing_run", "run_dir": str(run_dir)})
            else:
                seed_config = dict(config)
                seed_config["seed"] = seed
                run_training(run_dir, seed_config, patch=True, log=log)
        else:
            check((run_dir / "latest.pt").is_file(), f"Run dir has no checkpoint: {run_dir}")
        entry = evaluate_run_dir(run_dir, f"strip_seed{seed}", probe, data, baselines, log)
        runs[f"STRIP/{seed}"] = entry
        all_records.extend(entry.pop("prediction_records"))
    report = {
        "schema_version": SCHEMA, "stage": "stage1-period-strip",
        "labels": {
            "diagnostic_only": True,
            "single_factor": ("only the training target changed: caption answers lost trailing "
                              "whitespace and ONE terminal '.'; training path, data order, "
                              "scorer, references and all other settings identical to G4 "
                              "independent F"),
            "variant_selected_as_method_on_eval": False,
            "causal_scope": ("single-factor contrast conditional on this pipeline, the 640-step "
                             "budget and the validation split; no claim beyond it"),
            "confirmation_eval_split_read_for_generation_or_scoring": False,
        },
        "mode": args.mode,
        "started_utc": started_utc, "finished_utc": utc_now(),
        "runtime_seconds_total": round(time.perf_counter() - started, 3),
        "scheduler": scheduler_allocation(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "tool": {"path": TOOL, "sha256": sha256_bytes(Path(__file__).resolve().read_bytes())},
        "fidelity_gate": {"report": str(fidelity_report_path),
                          "sha256": sha256_bytes(fidelity_report_path.read_bytes()),
                          "gate": fidelity_record["gate"]},
        "probe_inputs": {"predictions_gz": {"path": str(args.probe_gz),
                                            "sha256": PROBE_GZ_SHA256},
                         "probe_report_sha256": PROBE_REPORT_SHA256},
        "data": {"task": "F", "split": "validation", "rows": len(data["validation"]),
                 "images_byte_verified": data["images_verified"],
                 "train_rows": len(data["train"])},
        "target_transform": {
            "strip": "rstrip(); remove ONE terminal '.'; caption answers only",
            "scoring_references_untouched": True,
            "applied_via": ("runtime monkeypatch of sema_multimodal.data.training_rows "
                            "(no sema_* source edit)")},
        "runs": runs,
        "korean_summary": korean_summary(runs),
    }
    exclusive_create_gzip(Path(args.predictions), all_records)
    exclusive_create_text(Path(args.report),
                          json.dumps(report, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n")
    log.write({"event": "done", "report": str(args.report),
               "predictions": str(args.predictions), "rows": len(all_records),
               "total_seconds": round(time.perf_counter() - started, 3)})
    log.close()
    print(json.dumps({"report": str(args.report), "predictions": str(args.predictions),
                      "rows": len(all_records),
                      "total_seconds": round(time.perf_counter() - started, 3)}), flush=True)


def korean_summary(runs: dict) -> list[str]:
    lines = []
    for key in sorted(runs):
        entry = runs[key]
        d0 = entry["results"]["D0"]["summary"]
        diag = d0["d0_diagnostics"]
        lines.append(
            f"- {key}: D0 EOS 도달률 {d0['eos_rate']:.3f}, 한계 도달률 {d0['limit_rate']:.3f}, "
            f"평균 길이 {d0['mean_length']:.2f}, 반복 3-gram {d0['repeated_trigram_rate']:.3f}, "
            f"CIDEr-D {d0['cider_d']:.4f}; '.' argmax {d0['period_argmax_steps_total']}단계.")
        for group, label in (("eos_rows", "EOS 행"), ("non_eos_rows", "한계 행")):
            if diag[group].get("rows"):
                lines.append(
                    f"- {key} {label} {diag[group]['rows']}행: [SEP] 최고확률 중앙값 "
                    f"{diag[group]['sep_max_prob']['median']:.4f}, top-2 비율 "
                    f"{diag[group]['sep_top2_at_some_step_share']:.3f}; '.' 최고확률 중앙값 "
                    f"{diag[group]['period_max_prob']['median']:.4f}, top-2 비율 "
                    f"{diag[group]['period_top2_at_some_step_share']:.3f}.")
        f_all = d0["paired_vs_F_D0_same_seed"]
        f_both = d0["paired_vs_F_D0_both_eos"]
        c_all = d0["paired_vs_C_D0_same_seed"]
        c_both = d0["paired_vs_C_D0_both_eos"]
        b4 = entry["results"]["D1_b4"]["summary"]
        lines.append(
            f"- {key} paired Δ (validation 256장, 동일 이미지): stage0.5 F-D0 대비 "
            f"{f_all['mean']:+.4f} (both-EOS {f_both['mean'] if f_both['mean'] is not None else 'n/a'}, "
            f"n={f_both['rows']}); C-D0 대비 {c_all['mean']:+.4f} (both-EOS "
            f"{c_both['mean'] if c_both['mean'] is not None else 'n/a'}, n={c_both['rows']}).")
        lines.append(
            f"- {key} D1(EOS 바이어스+4, 참고용): EOS {b4['eos_rate']:.3f}, "
            f"CIDEr-D {b4['cider_d']:.4f}, F-D0 대비 Δ "
            f"{b4['paired_vs_F_D0_same_seed']['mean']:+.4f}.")
    lines.append("- 단일 요인 대조(학습 타깃 마침표 제거) 결과이며, eval(확인 평가) 분할 방법 선택 없음; "
                 "원인 해석은 이 대조 범위 안에서만 유지.")
    return lines


if __name__ == "__main__":
    main()
