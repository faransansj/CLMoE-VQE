#!/usr/bin/env python3
"""Stage 0.5 item 4: decoding-variant probe on the Flickr validation split.

Diagnostic only. Runs the two caption experts of the G4 independent banks
(F = Flickr-trained, C = COCO-trained; seeds 137/149/163) on the 256-row Flickr
VALIDATION split — never the confirmation eval split — with the untouched frozen
BLIP backbone and the original image/prompt/precision settings, and applies
decoding variants implemented in this script only (no sema_* source edits, no
scorer edits):

  D0           baseline greedy, cached, max_new_tokens 64 (original decoding)
  D1_b1/2/4    greedy + EOS([SEP]) logit bias +b, b in {1, 2, 4}
  D2_r1.2/1.5  greedy + HF-style repetition penalty on generated ids only
  D3_n32/24    greedy with max_new_tokens 32 / 24
  D4_period    greedy that additionally stops on the '.' token; under the
               original decoding '.' has never been an argmax (0 of 625,664
               saved C/F predictions), so D4 is expected to equal D0

Along the greedy D0 path each step records P([SEP]) and P('.') and their ranks
in the vocabulary; summaries (per-row max-probability quantiles, top-2/top-1 hit
fractions, best-rank medians) are reported separately for EOS-ended and
limit-reached rows, per adapter and seed.

Every variant is scored with the untouched fixed train-IDF CIDEr-D
(sema_multimodal.metrics.TaskScorer built from the F training references only;
per-image singleton scores reuse the same _fixed_cider_score), and every variant
is paired against the seed-matched C-adapter D0 on the same 256 images, overall
and restricted to rows where both runs ended in EOS.

Adapter/base tensors are proven unchanged pre/post inference by SHA256 digests
over the full state dict and the expert banks.

Subcommands:
  probe  full pipeline: verify -> decode (GPU) -> score (CPU) -> report
  score  recovery: rescore an existing run dir's predictions.jsonl, then report

New files only; every final artifact is created exclusively (never overwritten).
Labels: diagnostic only; no variant is selected as a method on eval.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Byte bindings recorded from the local confirmation archives (2026-10-20).
G4_TAR_SHA256 = "8cdb944de983e62b5b759d4ff010151b71144439e6655db89642731a601413d6"
CHECKPOINTS_TAR_SHA256 = "6b288479429c8498b2e517a22175486ce16fae4ca1c342f7c7e76f2007b8cb38"
INDEX_SHA256 = "f7300e31054d009888e498d99e5c70814e9a3d96aad51c8731373bca7e662ba5"
# config.json digests for the six banks, computed from multimodal-g4-raw.tar.gz
# (SHA256 above); the binding of last resort when that tar is not present next
# to the remote repo copy.
CONFIG_SHA256 = {
    "F_independent_lr0.0003_ep5_seed137": "353a152604a7bd4087ae9a82bc52d3af9ceaf5106ed47c92c1ec5ae2f929d6cb",
    "F_independent_lr0.0003_ep5_seed149": "acd601263a47e0ebf0c658777be9e8b4846d4ab1615d4ac1d452840953a49df9",
    "F_independent_lr0.0003_ep5_seed163": "6dbc9759180f1815268c0cd3595551465d931e65bfde673f6df98f15c55103dc",
    "C_independent_lr0.0003_ep5_seed137": "586241e9fd8e556868cc06024f75786ab50b46cd510f03429563992dcaed1571",
    "C_independent_lr0.0003_ep5_seed149": "e365bbb8255f0c38b426ff4986233a39b3d2555967be4469cdd28c078fe5008c",
    "C_independent_lr0.0003_ep5_seed163": "579c9b72ec13d143d208a9c024e7c1129a85fedd8e8bed78eb9f6c6e0be7e23e",
}

G4_ROOT = "artifacts/multimodal_runs/g4"
G5_TAR = ROOT / "artifacts/multimodal-g5-raw.tar.gz"
G4_TAR = ROOT / "artifacts/multimodal-g4-raw.tar.gz"
CHECKPOINTS_TAR = ROOT / "artifacts/multimodal-confirmation-checkpoints.tar"
MANIFESTS_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"

SEEDS = (137, 149, 163)
MODELS = ("F", "C")
BANKS = tuple(f"{task}_independent_lr0.0003_ep5_seed{seed}"
              for task in MODELS for seed in SEEDS)
VALIDATION_ROWS = 256
MICRO_BATCH = 8
CAPTION_LIMIT = 64
SCHEMA = 1
TOOL = "scripts/stage05_decoding_probe.py"

VARIANTS = (
    {"name": "D0", "limit": 64, "eos_bias": 0.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D1_b1", "limit": 64, "eos_bias": 1.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D1_b2", "limit": 64, "eos_bias": 2.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D1_b4", "limit": 64, "eos_bias": 4.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D2_r1.2", "limit": 64, "eos_bias": 0.0, "repetition_penalty": 1.2, "period_stop": False},
    {"name": "D2_r1.5", "limit": 64, "eos_bias": 0.0, "repetition_penalty": 1.5, "period_stop": False},
    {"name": "D3_n32", "limit": 32, "eos_bias": 0.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D3_n24", "limit": 24, "eos_bias": 0.0, "repetition_penalty": 1.0, "period_stop": False},
    {"name": "D4_period", "limit": 64, "eos_bias": 0.0, "repetition_penalty": 1.0, "period_stop": True},
)
ROW_MEAN_TOL = 1e-9


# ---------------------------------------------------------------- small utils
def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exclusive_create_text(path: Path, payload: str) -> None:
    """Exclusive create of the final file; never overwrites anything present."""
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def round6(value: float) -> float:
    return round(float(value), 6)


def normalize_caption(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------------- digests
def _digest_value(digest, name, value) -> None:
    import torch

    if torch.is_tensor(value):
        digest.update(name.encode("utf-8"))
        cpu = value.detach().cpu().contiguous()
        if cpu.dtype is torch.bfloat16:
            cpu = cpu.view(torch.uint8)
        digest.update(cpu.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _digest_value(digest, f"{name}/{key}", value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"{name}/len={len(value)}".encode("utf-8"))
        for index, item in enumerate(value):
            _digest_value(digest, f"{name}/{index}", item)
    elif isinstance(value, (bool, int, float, str, type(None))):
        digest.update(f"{name}={value!r}".encode("utf-8"))
    else:
        digest.update(f"{name}<class:{type(value).__name__}>".encode("utf-8"))


def _tensor_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _digest_value(digest, name, value)
    return digest.hexdigest()


def _expert_digest(model) -> str:
    digest = hashlib.sha256()
    for name, projection in sorted(model._projections.items()):
        for key, tensor in sorted(projection.experts.state_dict().items()):
            _digest_value(digest, f"{name}/{key}", tensor)
    return digest.hexdigest()


# ---------------------------------------------------------------- environment
def _device_identity() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required: decoding variants run on the SLURM GPU allocation")
    identity = {
        "cuda_visible_devices_os": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count_visible": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=False)
    identity["nvidia_smi"] = smi.stdout.strip().splitlines()
    return identity


def _scheduler_allocation() -> dict:
    job_id = os.environ.get("SLURM_JOB_ID")
    allocation = {"slurm_job_id": job_id,
                  "slurm_env": {key: os.environ.get(key) for key in
                                ("SLURM_NODELIST", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
                                 "SLURM_MEM_PER_NODE", "SLURM_TIMELIMIT", "SLURM_QOS")}
                  if job_id else None,
                  "scontrol": None, "scontrol_available": False}
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
                                      ("JobId", "Partition", "JobName", "NodeList", "Nodes",
                                       "TresPerNode", "NumCPUs", "NumNodes", "NumTasks", "Mem",
                                       "Timelimit", "Gres", "QOS", "StartTime", "SubmitTime",
                                       "JobState") if key in fields}
            allocation["scontrol_available"] = True
    return allocation


def attempts_seen() -> list[dict]:
    """Report earlier stage-05 outputs already on disk (names/sizes only, never read)."""
    seen: dict[str, dict] = {}
    artifacts = ROOT / "artifacts"
    for pattern in ("stage05-decoding-probe-run*", "stage05-decoding-probe-smoke*"):
        for path in sorted(artifacts.glob(pattern)):
            seen[str(path.relative_to(ROOT))] = {"kind": "dir"}
    logs = ROOT / "artifacts/stage05-logs"
    if logs.is_dir():
        for path in sorted(logs.glob("stage05-decoding-probe*")):
            seen[str(path.relative_to(ROOT))] = {"kind": "log", "bytes": path.stat().st_size}
    return [{"path": key, **value} for key, value in sorted(seen.items())]


# ---------------------------------------------------------------- inputs
def member_bytes(tar_path: Path, member: str) -> bytes:
    with tarfile.open(tar_path, "r:*") as tar:
        return tar.extractfile(member).read()


def bank_member_name(bank: str, leaf: str) -> str:
    return f"{G4_ROOT}/{bank}/{leaf}"


def acquire_bank_file(bank: str, leaf: str, expected_sha: str, expected_bytes, run_dir: Path,
                      log: JsonlLog) -> tuple[Path, str]:
    """Return (path, source) for a bank file whose SHA256 matches the expectation.

    latest.pt: on-disk run dir first, else the confirmation checkpoints tar.
    config.json: on-disk run dir first, else the g4 raw tar member.
    """
    if leaf == "latest.pt":
        tar_sources = [(CHECKPOINTS_TAR, CHECKPOINTS_TAR_SHA256, bank_member_name(bank, leaf))]
    elif leaf == "config.json":
        tar_sources = [(G4_TAR, G4_TAR_SHA256, bank_member_name(bank, leaf))]
    else:
        raise ValueError(leaf)
    disk = ROOT / bank_member_name(bank, leaf)
    if disk.is_file():
        actual = digest_file(disk)
        if actual == expected_sha:
            return disk, "disk"
        log.write({"event": "bank_file_disk_mismatch", "bank": bank, "leaf": leaf,
                   "actual_sha256": actual, "expected_sha256": expected_sha})
    for tar_path, tar_sha, member in tar_sources:
        if not tar_path.is_file():
            continue
        actual_tar = digest_file(tar_path)
        if actual_tar != tar_sha:
            raise ValueError(f"Tar bytes differ from the recorded binding: {tar_path}")
        data = member_bytes(tar_path, member)
        if sha256_bytes(data) != expected_sha:
            raise ValueError(f"Tar member does not match the recorded binding: {member}")
        if expected_bytes is not None and len(data) != expected_bytes:
            raise ValueError(f"Tar member byte count differs from the index: {member}")
        target = run_dir / "checkpoints" / bank / leaf
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if sha256_bytes(target.read_bytes()) == expected_sha:
                return target, "extracted:previous"
            raise ValueError(f"Extracted bank file differs from the binding: {target}")
        with target.open("xb") as stream:
            stream.write(data)
        log.write({"event": "bank_file_extracted", "bank": bank, "leaf": leaf,
                   "tar": tar_path.name, "sha256": expected_sha})
        return target, f"extracted:{tar_path.name}"
    raise FileNotFoundError(f"No source with matching SHA256 for {bank}/{leaf}")


def bank_config_checked(bank: str, path: Path) -> dict:
    raw = path.read_bytes()
    if sha256_bytes(raw) != CONFIG_SHA256[bank]:
        raise ValueError(f"config.json differs from the g4-tar binding: {bank}")
    config = json.loads(raw)
    if config.get("method") != "independent" or config.get("task_order") != [bank[0]]:
        raise ValueError(f"Unexpected independent bank configuration: {bank}")
    if config.get("eval_split") != "eval":
        raise ValueError(f"Confirmation banks must record eval_split=eval: {bank}")
    if config.get("rank") != 8 or config.get("alpha") != 16 or config.get("micro_batch_size") != MICRO_BATCH:
        raise ValueError(f"Unexpected capacity/micro-batch in {bank}")
    if config["generation"] != {"strategy": "cached_greedy", "vqa_max_new_tokens": 32,
                                "caption_max_new_tokens": CAPTION_LIMIT}:
        raise ValueError(f"Unexpected recorded generation settings: {bank}")
    return config


def load_manifest_from_config(config: dict) -> tuple[dict, str]:
    manifest_path = Path(config["manifest"])
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        if sha256_bytes(raw) != config["manifest_sha256"]:
            raise ValueError(f"Manifest bytes differ from the recorded hash: {manifest_path}")
        return json.loads(raw), f"file:{manifest_path}"
    if MANIFESTS_TAR.is_file():
        raw = member_bytes(MANIFESTS_TAR, MANIFEST_MEMBER)
        if sha256_bytes(raw) != config["manifest_sha256"]:
            raise ValueError("Manifest tar member differs from the recorded hash")
        return json.loads(raw), f"tar:{MANIFESTS_TAR.name}:{MANIFEST_MEMBER}"
    raise FileNotFoundError("Manifest neither on disk nor in the manifests tar")


def verify_assets(config: dict) -> dict:
    verified = {}
    asset_root = Path(config["assets"])
    for relative_name, expected_sha in sorted(config["assets_sha256"].items()):
        candidate = Path(relative_name) if Path(relative_name).is_absolute() else asset_root / relative_name
        if not candidate.is_file():
            raise FileNotFoundError(f"Model asset missing: {candidate}")
        actual = digest_file(candidate)
        if actual != expected_sha:
            raise ValueError(f"Changed model asset file: {candidate}")
        verified[relative_name] = actual
    if config["tokenizer_sha256"] != {key: value for key, value in config["assets_sha256"].items()
                                      if key.startswith("tokenizer/")}:
        raise ValueError("Tokenizer manifest differs from the record")
    return verified


def verify_sources(config: dict) -> dict:
    verified = {}
    for relative_name, expected_sha in sorted(config["source_sha256"].items()):
        candidate = ROOT / relative_name
        if not candidate.is_file():
            raise FileNotFoundError(f"Repository source missing: {candidate}")
        actual = sha256_bytes(candidate.read_bytes())
        if actual != expected_sha:
            raise ValueError(f"Source file differs from the locked record: {relative_name}")
        verified[relative_name] = actual
    return verified


def verify_inputs_and_banks(index_path: str, run_dir: Path, log: JsonlLog,
                            hash_checkpoints_tar: bool) -> dict:
    index_file = Path(index_path)
    if not index_file.is_file():
        raise FileNotFoundError(f"Checkpoint index not found: {index_path}")
    raw_index = index_file.read_bytes()
    if sha256_bytes(raw_index) != INDEX_SHA256:
        raise ValueError("Checkpoint index differs from the recorded confirmation index")
    index = json.loads(raw_index)
    for bank in BANKS:
        member = bank_member_name(bank, "latest.pt")
        if member not in index["files"]:
            raise ValueError(f"Index misses {member}")
    inputs = {"index": {"path": str(index_file), "sha256": sha256_bytes(raw_index),
                        "bytes": len(raw_index)}}
    if G4_TAR.is_file():
        inputs["g4_tar"] = {"path": str(G4_TAR), "sha256": digest_file(G4_TAR),
                            "bytes": G4_TAR.stat().st_size}
    else:
        inputs["g4_tar"] = {"path": str(G4_TAR), "present": False}
    extraction_needed = any(not (ROOT / bank_member_name(bank, "latest.pt")).is_file() for bank in BANKS)
    if CHECKPOINTS_TAR.is_file() and (hash_checkpoints_tar or extraction_needed):
        inputs["checkpoints_tar"] = {"path": str(CHECKPOINTS_TAR), "sha256": digest_file(CHECKPOINTS_TAR),
                                     "bytes": CHECKPOINTS_TAR.stat().st_size}
    elif CHECKPOINTS_TAR.is_file():
        inputs["checkpoints_tar"] = {"path": str(CHECKPOINTS_TAR), "present": True,
                                     "sha256": None,
                                     "note": "not re-hashed; disk checkpoints matched the index"}
    else:
        inputs["checkpoints_tar"] = {"path": str(CHECKPOINTS_TAR), "present": False}
    bank_records, configs = {}, {}
    for bank in BANKS:
        latest_member = bank_member_name(bank, "latest.pt")
        latest_path, latest_source = acquire_bank_file(
            bank, "latest.pt", index["files"][latest_member]["sha256"],
            index["files"][latest_member]["bytes"], run_dir, log)
        config_path, config_source = acquire_bank_file(
            bank, "config.json", CONFIG_SHA256[bank], None, run_dir, log)
        config = bank_config_checked(bank, config_path)
        configs[bank] = config
        bank_records[bank] = {
            "latest": {"path": str(latest_path), "source": latest_source,
                       "sha256": index["files"][latest_member]["sha256"],
                       "bytes": index["files"][latest_member]["bytes"],
                       "sha_matches_index": True},
            "config": {"path": str(config_path), "source": config_source,
                       "sha256": CONFIG_SHA256[bank]}}
    first = configs[BANKS[0]]
    for bank in BANKS[1:]:
        for field in ("method", "eval_split", "lr", "epochs", "rank", "alpha", "batch_size",
                      "micro_batch_size", "image_size", "device", "manifest", "assets",
                      "image_root", "manifest_sha256", "assets_sha256", "tokenizer_sha256",
                      "source_sha256"):
            if configs[bank][field] != first[field]:
                raise ValueError(f"Config drift across banks at {field}: {bank}")
    # Each independent bank records only its own task's scorer lock; compare per task.
    first_bank_of_task = {}
    for bank in BANKS:
        task = bank[0]
        first_bank_of_task.setdefault(task, bank)
        reference = first_bank_of_task[task]
        if bank != reference and configs[reference]["scorer_locks"][task] != configs[bank]["scorer_locks"][task]:
            raise ValueError(f"Scorer lock drift across {task} banks at {bank}")
    log.write({"event": "banks_verified",
               "banks": {bank: bank_records[bank] for bank in BANKS}})
    return {"index": index, "configs": configs, "bank_records": bank_records, "inputs": inputs}


def resolve_token_ids(tokenizer) -> dict:
    period_id = tokenizer.convert_tokens_to_ids(".")
    if not isinstance(period_id, int) or period_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no '.' token")
    if tokenizer.decode([period_id]) != ".":
        raise ValueError("Tokenizer '.' token does not decode to '.'")
    return {"sep": int(tokenizer.sep_token_id), "pad": int(tokenizer.pad_token_id),
            "bos": int(tokenizer.bos_token_id), "period": int(period_id)}


def saved_validation_predictions_check(validation_ids: set[str]) -> dict:
    """Scan the g4/g5 prediction archives for any prediction file of the six banks
    whose row id set matches the Flickr validation split (D0 reproduction)."""
    result = {"banks": len(BANKS), "files_scanned": 0,
              "files_matching_validation_ids": [], "max_validation_id_overlap": 0,
              "archive_sources": []}
    for tar_path in (G4_TAR, G5_TAR):
        if not tar_path.is_file():
            continue
        result["archive_sources"].append(tar_path.name)
        prefixes = tuple(f"{G4_ROOT}/{bank}/" for bank in BANKS)
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".jsonl"):
                    continue
                if not member.name.startswith(prefixes):
                    continue
                if not member.name.rsplit("/", 1)[-1].startswith("predictions_"):
                    continue
                result["files_scanned"] += 1
                raw = tar.extractfile(member).read()
                ids = [json.loads(line)["id"] for line in raw.decode("utf-8").splitlines()
                       if line.strip()]
                overlap = len(set(ids) & validation_ids)
                result["max_validation_id_overlap"] = max(result["max_validation_id_overlap"], overlap)
                if set(ids) == validation_ids:
                    result["files_matching_validation_ids"].append(member.name)
    if not result["archive_sources"]:
        result["note"] = ("g4/g5 raw tars absent next to this repo copy; checked locally on "
                          "2026-10-20 over the same archives: zero validation-id overlap for the "
                          "six banks (their saved prediction files cover the confirmation eval "
                          "split only)")
    return result


# ---------------------------------------------------------------- decoding
def decode_variant(model, states, mask, *, limit, eos_bias, repetition_penalty,
                   period_stop, diagnostics, token_ids) -> dict:
    """One decoding variant over a pre-encoded micro-batch.

    Mirrors MultimodalReadout.generate step for step (cached greedy, bf16
    autocast, per-row stop, pad after finish, identical length/trigram
    bookkeeping) with the variant knobs applied on top. For D0 the argmax runs on
    the raw bf16 logits exactly like the original; diagnostics use a float copy.
    """
    import torch

    tokenizer = model.tokenizer
    batch = states.shape[0]
    device = model.device
    sep_id, pad_id, bos_id = token_ids["sep"], token_ids["pad"], token_ids["bos"]
    period_id = token_ids["period"]
    token = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    eos = torch.zeros(batch, dtype=torch.bool, device=device)
    period_stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)
    generated, past = [], None
    history = [[] for _ in range(batch)]
    step_records = [{"sep_prob": [], "period_prob": [], "sep_rank": [], "period_rank": []}
                    for _ in range(batch)] if diagnostics else None
    period_argmax = [0] * batch
    penalize = repetition_penalty != 1.0
    biased = eos_bias != 0.0
    for step in range(limit):
        output = model.text_decoder(
            token, past_key_values=past, use_cache=True,
            attention_mask=torch.ones((batch, step + 1), dtype=torch.long, device=device),
            encoder_hidden_states=states, encoder_attention_mask=mask, return_dict=True)
        raw = output.logits[:, -1]
        active = ~finished
        active_list = active.nonzero(as_tuple=True)[0].tolist()
        if biased or penalize:
            adjusted = raw.float()
            if penalize:
                penalty = repetition_penalty
                for row in active_list:
                    ids = sorted(set(history[row]))
                    if ids:
                        index_tensor = torch.tensor(ids, dtype=torch.long, device=device)
                        values = adjusted[row, index_tensor]
                        adjusted[row, index_tensor] = torch.where(
                            values < 0, values * penalty, values / penalty)
            if biased:
                adjusted[:, sep_id] = adjusted[:, sep_id] + eos_bias
            next_token = adjusted.argmax(dim=-1)
        else:
            next_token = raw.argmax(dim=-1)
        if diagnostics:
            probs = torch.softmax(raw.float(), dim=-1)
            sep_prob = probs[:, sep_id]
            period_prob = probs[:, period_id]
            sep_rank = (probs > sep_prob[:, None]).sum(dim=-1)
            period_rank = (probs > period_prob[:, None]).sum(dim=-1)
            for row in active_list:
                step_records[row]["sep_prob"].append(round(float(sep_prob[row]), 6))
                step_records[row]["period_prob"].append(round(float(period_prob[row]), 6))
                step_records[row]["sep_rank"].append(int(sep_rank[row]))
                step_records[row]["period_rank"].append(int(period_rank[row]))
        for row in active_list:
            chosen = int(next_token[row])
            if chosen == period_id:
                period_argmax[row] += 1
            history[row].append(chosen)
        next_token = torch.where(finished, pad_id, next_token)
        generated.append(next_token)
        lengths += active.long()
        reached_eos = active & (next_token == sep_id)
        reached_period = active & (next_token == period_id) if period_stop else torch.zeros_like(active)
        eos |= reached_eos
        period_stopped |= reached_period
        finished |= reached_eos | reached_period | (lengths >= limit)
        past, token = output.past_key_values, next_token[:, None]
        del output, raw
        if bool(finished.all()):
            break
    tokens = torch.stack(generated, dim=1)
    predictions = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    generated_rows = tokens.tolist()
    del tokens, past
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lengths_list = lengths.tolist()
    eos_list = eos.tolist()
    period_end_list = period_stopped.tolist()
    rows = []
    for index in range(batch):
        prediction = predictions[index]
        valid = generated_rows[index][:lengths_list[index]]
        repeated = any(valid[pos:pos + 3] == valid[pos + 3:pos + 6]
                       for pos in range(max(0, len(valid) - 5)))
        rows.append({
            "generated_tokens": lengths_list[index],
            "max_new_tokens": limit,
            "eos": bool(eos_list[index]),
            "period_stop": bool(period_end_list[index]),
            "empty": not prediction.strip(),
            "repeated_trigram": repeated,
            "limit_reached": lengths_list[index] == limit,
            "period_argmax_steps": period_argmax[index],
            "contains_period": "." in prediction,
        })
    return {"predictions": predictions, "diagnostics": rows, "steps": step_records}


# ---------------------------------------------------------------- scoring
def caption_cache(strings: list[str], chunk_size: int = 50000) -> dict:
    from sema_multimodal.metrics import _tokenize

    cache: dict[str, list[str]] = {}
    for start in range(0, len(strings), chunk_size):
        part = strings[start:start + chunk_size]
        mapping = {f"s{index}": [value] for index, value in enumerate(part)}
        tokenized = _tokenize(mapping)
        if set(tokenized) != set(mapping):
            raise ValueError("PTB chunk lost captions")
        for index, value in enumerate(part):
            cache[value] = tokenized[f"s{index}"][0]
    return cache


def score_variant(scorer, cache, rows, predictions, variant_key) -> dict:
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from sema_multimodal.metrics import _fixed_cider_score, _tokenize

    fresh = scorer.score(predictions, rows)
    per_image = []
    for prediction, row in zip(predictions, rows):
        hyp = cache[normalize_caption(prediction)]
        refs = [cache[normalize_caption(ref)] for ref in row["references"]]
        singleton = CiderScorer(n=4, sigma=6.0)
        singleton += (hyp, refs)
        per_image.append(_fixed_cider_score(singleton, scorer._df, scorer._train_documents))
    mean_per_image = math.fsum(per_image) / len(rows)
    if abs(mean_per_image - fresh["score"]) > ROW_MEAN_TOL:
        raise ValueError(f"per-image mean vs fresh TaskScorer.score mismatch {variant_key}: "
                         f"{mean_per_image!r} vs {fresh['score']!r}")
    captions = {("reference", index): row["references"] for index, row in enumerate(rows)}
    captions.update({("prediction", index): [pred] for index, pred in enumerate(predictions)})
    tokenized = _tokenize(captions)
    mismatches = 0
    for (kind, index), tokens in tokenized.items():
        sources = rows[index]["references"] if kind == "reference" else [predictions[index]]
        for source, token in zip(sources, tokens):
            if cache.get(normalize_caption(source)) != token:
                mismatches += 1
    if mismatches:
        raise ValueError(f"PTB cache parity failed for {variant_key}: {mismatches} lines")
    return {"cider_d": fresh["score"], "bleu4": fresh["bleu4"],
            "per_image": per_image, "mean_per_image": mean_per_image}


# ---------------------------------------------------------------- summaries
def quantiles(values: list[float]) -> dict:
    if not values:
        return {"rows": 0}
    ordered = sorted(values)

    def q(fraction):
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round6(ordered[index])

    return {"rows": len(ordered), "p10": q(0.10), "median": q(0.50), "p90": q(0.90)}


def summarize_steps(step_rows: list[dict], eos_flags: list[bool]) -> dict:
    groups = {"all": [], "eos_rows": [], "non_eos_rows": []}
    for index, row in enumerate(step_rows):
        if not row["sep_prob"]:
            continue
        groups["all"].append(index)
        (groups["eos_rows"] if eos_flags[index] else groups["non_eos_rows"]).append(index)
    summary = {}
    for group, indices in groups.items():
        if not indices:
            summary[group] = {"rows": 0}
            continue
        selected = [step_rows[i] for i in indices]
        sep_max = [max(row["sep_prob"]) for row in selected]
        period_max = [max(row["period_prob"]) for row in selected]
        sep_best_rank = [min(row["sep_rank"]) for row in selected]
        period_best_rank = [min(row["period_rank"]) for row in selected]
        summary[group] = {
            "rows": len(indices),
            "sep_max_prob": quantiles(sep_max),
            "period_max_prob": quantiles(period_max),
            "sep_top2_at_some_step_share": sum(any(r <= 1 for r in row["sep_rank"])
                                               for row in selected) / len(indices),
            "sep_top1_at_some_step_share": sum(any(r == 0 for r in row["sep_rank"])
                                               for row in selected) / len(indices),
            "period_top2_at_some_step_share": sum(any(r <= 1 for r in row["period_rank"])
                                                  for row in selected) / len(indices),
            "period_top1_at_some_step_share": sum(any(r == 0 for r in row["period_rank"])
                                                  for row in selected) / len(indices),
            "sep_best_rank_median": statistics.median(sep_best_rank),
            "period_best_rank_median": statistics.median(period_best_rank),
        }
    return summary


def variant_summary(diag: list[dict]) -> dict:
    count = len(diag)
    return {
        "rows": count,
        "eos_rate": sum(row["eos"] for row in diag) / count,
        "period_stop_rate": sum(row["period_stop"] for row in diag) / count,
        "limit_rate": sum(row["limit_reached"] for row in diag) / count,
        "mean_length": math.fsum(row["generated_tokens"] for row in diag) / count,
        "repeated_trigram_rate": sum(row["repeated_trigram"] for row in diag) / count,
        "empty_rate": sum(row["empty"] for row in diag) / count,
        "contains_period_share": sum(row["contains_period"] for row in diag) / count,
        "period_argmax_rows": sum(row["period_argmax_steps"] > 0 for row in diag),
        "period_argmax_steps_total": sum(row["period_argmax_steps"] for row in diag),
    }


# ---------------------------------------------------------------- data loading
def load_validation_data(first_config: dict) -> dict:
    from sema_multimodal.data import ImageVerifier, public_rows, validate_manifest

    manifest, manifest_source = load_manifest_from_config(first_config)
    validate_manifest(manifest)  # structure-only; the 256 used images are byte-verified below
    task_F = next(task for task in manifest["tasks"] if task["task_id"] == "F")
    validation, train = task_F["validation"], task_F["train"]
    if len(validation) != VALIDATION_ROWS:
        raise ValueError(f"F validation split must have {VALIDATION_ROWS} rows")
    image_root = Path(first_config["image_root"]).resolve()
    verifier = ImageVerifier(image_root)
    hashes = [verifier.verify(row["image"]) for row in validation]
    mismatched = [row["id"] for row, actual in zip(validation, hashes)
                  if actual != row["image_sha256"]]
    if mismatched:
        raise ValueError(f"Image bytes differ from the manifest record: {mismatched[:4]}")
    endings = Counter()
    for row in validation:
        for ref in row["references"]:
            text = ref.rstrip()
            if text.endswith("."):
                endings["period"] += 1
            elif text[-1:] in ("!", "?"):
                endings["other_terminal_punct"] += 1
            else:
                endings["no_terminal_punct"] += 1
    total_refs = sum(endings.values())
    return {"manifest": manifest, "manifest_source": manifest_source,
            "validation": validation, "train": train,
            "rows": public_rows(validation), "image_root": str(image_root),
            "images_verified": len(validation), "image_sha_mismatches": 0,
            "validation_reference_endings": {name: count / total_refs
                                             for name, count in sorted(endings.items())}}


# ---------------------------------------------------------------- decode phase
def decode_all_banks(run_dir: Path, log: JsonlLog, verified: dict, data: dict,
                     banks: tuple = BANKS, subset: int | None = None) -> dict:
    import random

    import numpy as np
    import torch

    from sema_multimodal.model import MultimodalReadout

    first_config = verified["configs"][BANKS[0]]
    model = MultimodalReadout(first_config["assets"], image_size=first_config["image_size"],
                              rank=first_config["rank"], alpha=first_config["alpha"],
                              device=first_config["device"])
    token_ids = resolve_token_ids(model.tokenizer)
    rows = data["rows"] if subset is None else data["rows"][:subset]
    validation = data["validation"] if subset is None else data["validation"][:subset]
    check(len(rows) == len(validation), "Row/public-view length mismatch")
    image_root = data["image_root"]
    device = torch.device(first_config["device"])
    records_path = run_dir / "predictions.jsonl"
    runs: dict[str, dict] = {}
    overall_started = time.perf_counter()
    with records_path.open("x", encoding="utf-8") as stream:
        for bank in banks:
            model_family, seed = bank[0], int(bank.rsplit("seed", 1)[1])
            config = verified["configs"][bank]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            checkpoint = torch.load(verified["bank_records"][bank]["latest"]["path"],
                                    map_location="cpu", weights_only=False)
            if checkpoint.get("schema_version") != 1 or checkpoint.get("settings") != config:
                raise ValueError(f"Checkpoint settings differ from the recorded configuration: {bank}")
            if checkpoint.get("state", {}).get("assignments") != [0]:
                raise ValueError(f"Independent bank expert assignment is not [0]: {bank}")
            bank_state = checkpoint["model"]
            if (bank_state.get("num_experts") != 1 or bank_state.get("rank") != first_config["rank"]
                    or not math.isclose(bank_state.get("alpha"), first_config["alpha"], rel_tol=1e-12)):
                raise ValueError(f"Checkpoint bank does not match the declared capacity: {bank}")
            model.load_checkpoint_state(bank_state)
            model._freeze_all()
            model.eval()
            model.set_expert(0)
            check(model.num_experts == 1, f"Reconstructed bank size differs: {bank}")
            load_seconds = round(time.perf_counter() - load_started, 3)
            digest_before = _tensor_digest(model)
            expert_before = _expert_digest(model)
            bank_started = time.perf_counter()
            variant_rows: dict[str, list] = {variant["name"]: [] for variant in VARIANTS}
            variant_seconds: dict[str, float] = {variant["name"]: 0.0 for variant in VARIANTS}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                for start in range(0, len(rows), MICRO_BATCH):
                    batch_rows = rows[start:start + MICRO_BATCH]
                    states, mask = model._question_states(batch_rows, image_root)
                    for variant in VARIANTS:
                        started = time.perf_counter()
                        outcome = decode_variant(
                            model, states, mask, limit=variant["limit"],
                            eos_bias=variant["eos_bias"],
                            repetition_penalty=variant["repetition_penalty"],
                            period_stop=variant["period_stop"],
                            diagnostics=variant["name"] == "D0", token_ids=token_ids)
                        variant_seconds[variant["name"]] += time.perf_counter() - started
                        for offset, (prediction, info) in enumerate(zip(outcome["predictions"],
                                                                        outcome["diagnostics"])):
                            manifest_row = validation[start + offset]
                            record = {"model": model_family, "seed": seed, "bank": bank,
                                      "variant": variant["name"], "row_index": start + offset,
                                      "id": manifest_row["id"],
                                      "image_sha256": manifest_row["image_sha256"],
                                      "prediction": prediction, **info}
                            if variant["name"] == "D0":
                                record["d0_step_diag"] = outcome["steps"][offset]
                            variant_rows[variant["name"]].append(record)
                            stream.write(json.dumps(record, ensure_ascii=False,
                                                    allow_nan=False) + "\n")
                    del states, mask
            stream.flush()
            expert_after = _expert_digest(model)
            digest_after = _tensor_digest(model)
            expert_identical = expert_before == expert_after
            model_identical = digest_before == digest_after
            check(expert_identical, f"Expert tensors changed during inference: {bank}")
            check(model_identical, f"Whole model changed during inference: {bank}")
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            d0_diag = variant_rows["D0"]
            eos_flags = [row["eos"] for row in d0_diag]
            step_rows = [row["d0_step_diag"] for row in d0_diag]
            key = f"{model_family}/{seed}"
            runs[key] = {
                "bank": bank,
                "d0_diagnostics_summary": summarize_steps(step_rows, eos_flags),
                "adapter_proof": {
                    "whole_model_digest_pre": digest_before,
                    "whole_model_digest_post": digest_after,
                    "whole_model_identical_pre_post": model_identical,
                    "expert_digest_pre": expert_before,
                    "expert_digest_post": expert_after,
                    "expert_tensors_identical_pre_post": expert_identical,
                    "rank_alpha_pre_post": [model.rank, model.alpha],
                },
                "load_seconds": load_seconds,
                "variant_seconds": {name: round(seconds, 3) for name, seconds in variant_seconds.items()},
                "peak_gpu_bytes": peak,
                "bank_seconds": round(time.perf_counter() - bank_started, 3),
                "_variant_rows": variant_rows,
            }
            log.write({"event": "bank_done", "bank": bank, "seconds": runs[key]["bank_seconds"],
                       "peak_gpu_bytes": peak})
            del checkpoint, bank_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"runs": runs, "token_ids": token_ids, "validation_rows": validation,
            "decode_seconds": round(time.perf_counter() - overall_started, 3)}


# ---------------------------------------------------------------- score phase
def score_all(runs: dict, scoring_rows: list, data: dict, config: dict, log: JsonlLog) -> dict:
    from sema_multimodal.metrics import TaskScorer

    scorer = TaskScorer({"task_id": "F", "family": "caption", "metric": "cider_d",
                         "train": data["train"]})
    recorded_lock = config["scorer_locks"]["F"]
    if scorer.lock != recorded_lock:
        drift = {key: (recorded_lock.get(key), scorer.lock.get(key))
                 for key in sorted(set(recorded_lock) | set(scorer.lock))
                 if recorded_lock.get(key) != scorer.lock.get(key)}
        raise ValueError(f"Scorer lock drift against the recorded configuration: {drift}")
    log.write({"event": "scorer_built", "train_documents": scorer._train_documents,
               "document_frequency_sha256": scorer.lock["document_frequency_sha256"]})

    strings = {normalize_caption(ref) for row in data["train"] for ref in row["references"]}
    strings.update(normalize_caption(ref) for row in data["validation"]
                   for ref in row["references"])
    for key in runs:
        for records in runs[key]["_variant_rows"].values():
            strings.update(normalize_caption(record["prediction"]) for record in records)
    string_list = sorted(strings)
    cache_started = time.perf_counter()
    cache = caption_cache(string_list)
    cache_seconds = round(time.perf_counter() - cache_started, 3)
    log.write({"event": "token_cache_built", "unique_strings": len(string_list),
               "seconds": cache_seconds})

    results = {}
    for key in sorted(runs):
        for variant in VARIANTS:
            name = variant["name"]
            records = runs[key]["_variant_rows"][name]
            predictions = [record["prediction"] for record in records]
            started = time.perf_counter()
            scores = score_variant(scorer, cache, scoring_rows, predictions,
                                   f"{key}/{name}")
            results[f"{key}/{name}"] = {"scores": scores,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "diag": [dict(record) for record in records]}
    return {"cache_seconds": cache_seconds, "unique_strings": len(string_list), "results": results}


# ---------------------------------------------------------------- aggregation
def aggregate(runs: dict, results: dict, data: dict) -> dict:
    present = sorted(runs)
    seeds_by_model = {model: sorted({int(key.split("/")[1]) for key in present
                                     if key.split("/")[0] == model})
                      for model in MODELS}
    base_per_image, base_diag = {}, {}
    for seed in seeds_by_model["C"]:
        base_key = f"C/{seed}"
        base_per_image[seed] = results[f"{base_key}/D0"]["scores"]["per_image"]
        base_diag[seed] = runs[base_key]["_variant_rows"]["D0"]

    runs_out = {}
    for key in sorted(runs):
        model_family, seed_text = key.split("/")
        seed = int(seed_text)
        entry = {field: runs[key].get(field) for field in
                 ("bank", "d0_diagnostics_summary", "adapter_proof", "load_seconds",
                  "variant_seconds", "peak_gpu_bytes", "bank_seconds")}
        if entry["adapter_proof"] is None:
            entry["adapter_proof"] = {"note": ("not recomputed in score-recovery mode; the probe "
                                               "run's verification.jsonl carries the digests")}
        entry["variants"] = {}
        for variant in VARIANTS:
            name = variant["name"]
            cell = results[f"{key}/{name}"]
            diag = cell["diag"]
            summary = variant_summary(diag)
            summary.update({"cider_d": round6(cell["scores"]["cider_d"]),
                            "bleu4": round6(cell["scores"]["bleu4"]),
                            "per_image_mean": round6(cell["scores"]["mean_per_image"]),
                            "scoring_seconds": cell["seconds"]})
            base = base_per_image[seed]
            deltas = [value - base_value for value, base_value
                      in zip(cell["scores"]["per_image"], base)]
            summary["paired_delta_vs_C_D0_mean"] = round6(math.fsum(deltas) / len(deltas))
            both = [index for index in range(len(diag))
                    if diag[index]["eos"] and base_diag[seed][index]["eos"]]
            summary["paired_delta_both_eos_rows"] = len(both)
            summary["paired_delta_both_eos_mean"] = (
                round6(math.fsum(deltas[i] for i in both) / len(both)) if both else None)
            if name == "D4_period":
                d0_diag = runs[key]["_variant_rows"]["D0"]
                summary["d4_period_equals_d0_rows"] = sum(
                    diag[index]["prediction"] == d0_diag[index]["prediction"]
                    for index in range(len(diag)))
            entry["variants"][name] = summary
        runs_out[key] = entry

    pooled = {}
    for model_family in MODELS:
        keys = [f"{model_family}/{seed}" for seed in seeds_by_model[model_family]]
        if not keys:
            continue
        pooled[model_family] = {}
        for variant in VARIANTS:
            diag = []
            for key in keys:
                diag.extend(results[f"{key}/{variant['name']}"]["diag"])
            pooled[model_family][variant["name"]] = variant_summary(diag)
        pooled[model_family]["d0_cider_d_mean_over_seeds"] = round6(
            math.fsum(runs_out[key]["variants"]["D0"]["cider_d"] for key in keys) / len(keys))
        for group in ("eos_rows", "non_eos_rows"):
            merged = {"rows": 0}
            for key in keys:
                summary = runs[key]["d0_diagnostics_summary"][group]
                if not summary.get("rows"):
                    continue
                merged["rows"] += summary["rows"]
                for field in ("sep_top2_at_some_step_share", "sep_top1_at_some_step_share",
                              "period_top2_at_some_step_share", "period_top1_at_some_step_share"):
                    merged[field] = merged.get(field, 0.0) + summary[field] * summary["rows"]
                for field in ("sep_best_rank_median", "period_best_rank_median"):
                    merged.setdefault(field + "_values", []).append(summary[field])
                for field in ("sep_max_prob", "period_max_prob"):
                    merged.setdefault(field, {"p10": [], "median": [], "p90": []})
                    for quant in ("p10", "median", "p90"):
                        merged[field][quant].append(summary[field][quant])
            for field in ("sep_top2_at_some_step_share", "sep_top1_at_some_step_share",
                          "period_top2_at_some_step_share", "period_top1_at_some_step_share"):
                if merged["rows"]:
                    merged[field] = round6(merged[field] / merged["rows"])
                else:
                    merged.pop(field, None)
            for field in ("sep_best_rank_median", "period_best_rank_median"):
                if field + "_values" in merged:
                    merged[field] = statistics.median(merged.pop(field + "_values"))
            for field in ("sep_max_prob", "period_max_prob"):
                if field in merged:
                    merged[field] = {quant: round6(statistics.mean(values))
                                     for quant, values in merged[field].items()}
            pooled[model_family][f"d0_diagnostics_{group}"] = merged
    return {"runs": runs_out, "pooled": pooled}


def korean_summary(aggregate_out: dict) -> list[str]:
    lines = []
    seeds_by_model = {model: sorted({int(key.split("/")[1]) for key in aggregate_out["runs"]
                                     if key.split("/")[0] == model})
                      for model in MODELS}
    for model_family in MODELS:
        seeds = seeds_by_model[model_family]
        if not seeds:
            continue
        pooled = aggregate_out["pooled"][model_family]
        d0 = pooled["D0"]
        runs_out = aggregate_out["runs"]
        cider_mean = round6(math.fsum(runs_out[f"{model_family}/{seed}"]["variants"]["D0"]["cider_d"]
                                      for seed in seeds) / len(seeds))
        lines.append(
            f"- {model_family} 어댑터 D0(그리디, 최대 64토큰): validation 256장 x seed 3개에서 "
            f"EOS 도달률 {d0['eos_rate']:.3f}, 한계(64토큰) 도달률 {d0['limit_rate']:.3f}, "
            f"평균 길이 {d0['mean_length']:.2f}, 반복 3-gram 행 비율 {d0['repeated_trigram_rate']:.3f}, "
            f"corpus CIDEr-D seed 평균 {cider_mean:.4f}.")
        for group, label in (("eos_rows", "EOS 종료 행"), ("non_eos_rows", "한계 도달 행")):
            summary = pooled.get(f"d0_diagnostics_{group}", {})
            if summary.get("rows"):
                lines.append(
                    f"- {model_family} {label} {summary['rows']}행: [SEP] 최고확률 중앙값 "
                    f"{summary['sep_max_prob']['median']:.4f}, [SEP] 어느 단계에서든 top-2 "
                    f"{summary['sep_top2_at_some_step_share']:.3f}; '.' 최고확률 중앙값 "
                    f"{summary['period_max_prob']['median']:.4f}, '.' top-2 "
                    f"{summary['period_top2_at_some_step_share']:.3f}.")
        for name, label in (("D1_b4", "D1(바이어스+4)"), ("D2_r1.5", "D2(반복페널티 1.5)"),
                            ("D3_n24", "D3(최대 24토큰)"), ("D4_period", "D4('.' 정지)")):
            cells = [aggregate_out["runs"][f"{model_family}/{seed}"]["variants"][name]
                     for seed in seeds]
            lines.append(
                f"- {model_family} {label}: EOS 도달률 seed 평균 "
                f"{round6(math.fsum(cell['eos_rate'] for cell in cells) / len(cells)):.3f}, "
                f"CIDEr-D seed 평균 {round6(math.fsum(cell['cider_d'] for cell in cells) / len(cells)):.4f}, "
                f"C-D0 대비 paired Δ(seed 평균) "
                f"{round6(math.fsum(cell['paired_delta_vs_C_D0_mean'] for cell in cells) / len(cells)):+.4f}.")
        d4_equals = [aggregate_out["runs"][f"{model_family}/{seed}"]["variants"]["D4_period"]
                     ["d4_period_equals_d0_rows"] for seed in seeds]
        lines.append(f"- {model_family} D4('.' 정지 토큰 추가) 예측이 D0과 동일한 행: "
                     f"{sum(d4_equals)}/{len(seeds) * 256} (seed {len(seeds)}개 x 256장); "
                     f"'.' 토큰 argmax 선택 여부는 각 변형의 period_argmax counts 참조.")
    lines.append("- 진단용 결과이며, 어떤 변형도 eval(확인 평가) 분할의 방법으로 선택되지 않음.")
    return lines


# ---------------------------------------------------------------- artifacts
def prediction_records(runs: dict, results: dict) -> list[dict]:
    records = []
    for key in sorted(runs):
        for variant in VARIANTS:
            name = variant["name"]
            cell = results[f"{key}/{name}"]
            for record in runs[key]["_variant_rows"][name]:
                out = dict(record)
                out["cider_d_per_image"] = round6(cell["scores"]["per_image"][record["row_index"]])
                records.append(out)
    return records


def build_report(args, started: float, verified: dict, data: dict, runs_state: dict,
                 scored: dict, aggregate_out: dict, mode: str) -> dict:
    import torch

    finished = time.perf_counter()
    config = verified["configs"][BANKS[0]]
    report = {
        "schema_version": SCHEMA,
        "stage": "stage0.5-decoding-probe",
        "labels": {
            "diagnostic_only": True,
            "variant_selected_as_method_on_eval": False,
            "confirmation_eval_split_read_for_generation_or_scoring": False,
            "note": ("Flickr validation split (256 rows) only; the confirmation eval split was "
                     "never read for generation or scoring in this probe. Decoding variants are "
                     "diagnostics; no variant is selected as a method on eval."),
        },
        "tool": {"path": TOOL, "sha256": sha256_bytes(Path(__file__).resolve().read_bytes())},
        "sbatch": ({"path": "scripts/stage05_decoding_probe.sbatch",
                    "sha256": sha256_bytes((ROOT / "scripts/stage05_decoding_probe.sbatch").read_bytes())}
                   if (ROOT / "scripts/stage05_decoding_probe.sbatch").is_file() else None),
        "mode": mode,
        "started_utc": args.started_utc,
        "finished_utc": utc_now(),
        "runtime_seconds_total": round(finished - started, 3),
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32
                             and not torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
            "hf_transformers": importlib.metadata.version("transformers"),
            "numpy": importlib.metadata.version("numpy"),
            "pycocoevalcap": importlib.metadata.version("pycocoevalcap"),
        },
        "scheduler": _scheduler_allocation(),
        "device": _device_identity() if mode == "probe" else None,
        "inputs": verified["inputs"],
        "checkpoints": verified["bank_records"],
        "model": {
            "backbone": "BLIP model_base_capfilt_large (frozen) + readout LoRA experts (rank 8, alpha 16)",
            "assets": config["assets"], "assets_verified": data.get("assets_verified"),
            "sources_verified": data.get("sources_verified"),
            "manifest": {"source": data.get("manifest_source"),
                         "sha256": config["manifest_sha256"],
                         "image_root": data.get("image_root")},
            "token_ids": runs_state["token_ids"],
        },
        "data": {
            "task": "F", "split": "validation", "rows": len(data["validation"]),
            "images_byte_verified": data.get("images_verified"),
            "image_sha_mismatches": data.get("image_sha_mismatches"),
            "validation_reference_endings": data["validation_reference_endings"],
            "scorer": {"metric": "cider_d", "idf_source": "F task train references (2048 rows)",
                       "lock_matches_recorded_config": True,
                       "document_frequency_sha256":
                           config["scorer_locks"]["F"]["document_frequency_sha256"]},
        },
        "decoding_variants": VARIANTS,
        "decoding_notes": [
            "D0 mirrors MultimodalReadout.generate exactly: cached greedy argmax on the raw "
            "bf16 logits under bf16 autocast, stop on [SEP], caption limit 64, pad after "
            "finish, identical length/trigram bookkeeping.",
            "D1 adds a constant to the [SEP] logit (fp32) at every step before the argmax; "
            "stopping rules unchanged.",
            "D2 applies the HF RepetitionPenaltyLogitsProcessor transform "
            "(score<0 ? score*p : score/p) to the tokens present in the generated ids only "
            "(the prompt/BOS seed is not penalized).",
            "D3 only shortens max_new_tokens (32 / 24).",
            "D4 additionally stops when the '.' token is selected; '.' has never been an "
            "argmax under the original decoding (0/625,664 saved C/F predictions), so D4 is "
            "expected to equal D0.",
            "D0 step diagnostics record softmax probabilities of the UNMODIFIED logits.",
        ],
        "d0_reproduction": {
            "saved_validation_predictions": data.get("saved_validation_check"),
            "status": "reference",
            "note": ("No saved validation-split predictions exist for the six confirmation "
                     "independent banks (their stored prediction files cover the confirmation "
                     "eval split only, zero id overlap). D0 mirrors MultimodalReadout.generate "
                     "exactly and is reported as the reference decoding."),
        },
        "runs": aggregate_out["runs"],
        "pooled": aggregate_out["pooled"],
        "korean_summary": korean_summary(aggregate_out),
        "timings": {"decode_seconds": runs_state.get("decode_seconds"),
                    "tokenize_seconds": scored.get("cache_seconds"),
                    "scoring_seconds": None,
                    "total_seconds": round(finished - started, 3)},
        "artifacts": {"predictions": str(Path(args.predictions)),
                      "report": str(Path(args.report))},
        "attempts_seen_before_this_attempt": attempts_seen(),
    }
    return report


def rebuild_runs_from_records(records: list[dict], data: dict, verified: dict) -> dict:
    """Reconstruct the runs structure from a run dir's predictions.jsonl (score mode)."""
    runs = {}
    validation_ids = [row["id"] for row in data["validation"]]
    by_cell: dict[tuple, dict[int, dict]] = {}
    for record in records:
        cell_key = (record["model"], record["seed"], record["variant"])
        cell = by_cell.setdefault(cell_key, {})
        if record["row_index"] in cell:
            raise ValueError(f"Duplicate row in records: {cell_key}/{record['row_index']}")
        if record["id"] != validation_ids[record["row_index"]]:
            raise ValueError(f"Record row does not match the validation split: {record}")
        cell[record["row_index"]] = record
    for model_family in MODELS:
        for seed in SEEDS:
            variant_rows = {}
            for variant in VARIANTS:
                cell = by_cell.get((model_family, seed, variant["name"]))
                if cell is None or len(cell) != len(validation_ids):
                    raise ValueError(f"Run dir misses {model_family}/{seed}/{variant['name']}")
                ordered = [cell[index] for index in range(len(validation_ids))]
                if variant["name"] == "D0" and any("d0_step_diag" not in row for row in ordered):
                    raise ValueError(f"Run dir D0 rows lack step diagnostics: {model_family}/{seed}")
                variant_rows[variant["name"]] = ordered
            key = f"{model_family}/{seed}"
            eos_flags = [row["eos"] for row in variant_rows["D0"]]
            step_rows = [row["d0_step_diag"] for row in variant_rows["D0"]]
            runs[key] = {"bank": f"{model_family}_independent_lr0.0003_ep5_seed{seed}",
                         "d0_diagnostics_summary": summarize_steps(step_rows, eos_flags),
                         "_variant_rows": variant_rows}
    return {"runs": runs, "token_ids": rebuild_token_ids(verified),
            "validation_rows": data["validation"]}


def rebuild_token_ids(verified: dict) -> dict:
    from transformers import BertTokenizer

    config = verified["configs"][BANKS[0]]
    tokenizer = BertTokenizer.from_pretrained(str(Path(config["assets"]) / "tokenizer"),
                                              local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    return resolve_token_ids(tokenizer)


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    def add_common(child):
        child.add_argument("--index", default=str(ROOT / "artifacts/multimodal-confirmation-checkpoints.json"))
        child.add_argument("--report", default=str(ROOT / "artifacts/stage05-decoding-probe.json"))
        child.add_argument("--predictions",
                           default=str(ROOT / "artifacts/stage05-decoding-predictions.jsonl.gz"))
        child.add_argument("--cpu-threads", type=int, default=4)

    probe = sub.add_parser("probe")
    probe.add_argument("--output-dir", default=str(ROOT / "artifacts/stage05-decoding-probe-run"))
    add_common(probe)

    score = sub.add_parser("score")
    score.add_argument("--run-dir", required=True)
    add_common(score)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-dir", default=str(ROOT / "artifacts/stage05-decoding-probe-smoke"))
    smoke.add_argument("--rows", type=int, default=16)
    add_common(smoke)

    args = parser.parse_args(argv)
    started = time.perf_counter()
    args.started_utc = utc_now()

    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if args.stage == "probe":
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        log = JsonlLog(run_dir / "verification.jsonl")
    elif args.stage == "score":
        run_dir = Path(args.run_dir)
        log = JsonlLog(run_dir / "verification-score.jsonl")
    else:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        log = JsonlLog(run_dir / "verification.jsonl")
    log.write({"event": "start", "mode": args.stage, "utc": args.started_utc,
               "script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
               "job_id": os.environ.get("SLURM_JOB_ID")})

    verified = verify_inputs_and_banks(args.index, run_dir, log,
                                       hash_checkpoints_tar=args.stage == "probe")
    data = load_validation_data(verified["configs"][BANKS[0]])
    data["assets_verified"] = verify_assets(verified["configs"][BANKS[0]])
    data["sources_verified"] = verify_sources(verified["configs"][BANKS[0]])
    log.write({"event": "data_and_sources_verified",
               "rows": len(data["validation"]),
               "manifest_source": data["manifest_source"]})

    if args.stage == "probe":
        data["saved_validation_check"] = saved_validation_predictions_check(
            {row["id"] for row in data["validation"]})
        runs_state = decode_all_banks(run_dir, log, verified, data)
    elif args.stage == "score":
        records = [json.loads(line) for line in
                   (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        runs_state = rebuild_runs_from_records(records, data, verified)
    else:
        runs_state = decode_all_banks(run_dir, log, verified, data,
                                      banks=(BANKS[0], BANKS[3]), subset=args.rows)

    score_started = time.perf_counter()
    scored = score_all(runs_state["runs"], runs_state["validation_rows"], data,
                       verified["configs"][BANKS[0]], log)
    score_seconds = round(time.perf_counter() - score_started, 3)
    aggregate_out = aggregate(runs_state["runs"], scored["results"], data)

    if args.stage == "smoke":
        summary = {"mode": "smoke", "rows": args.rows,
                   "banks": sorted(runs_state["runs"]),
                   "variants": {f"{key}/{name}": {
                       "cider_d": round6(scored["results"][f"{key}/{name}"]["scores"]["cider_d"]),
                       "eos_rate": round6(aggregate_out["runs"][key]["variants"][name]["eos_rate"]),
                       "mean_length": round6(aggregate_out["runs"][key]["variants"][name]["mean_length"]),
                       "contains_period_share": round6(aggregate_out["runs"][key]["variants"][name]["contains_period_share"])}
                       for key in sorted(runs_state["runs"]) for name in
                       (variant["name"] for variant in VARIANTS)},
                   "decode_seconds": runs_state["decode_seconds"],
                   "total_seconds": round(time.perf_counter() - started, 3)}
        log.write({"event": "done", "mode": "smoke", **summary})
        log.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    report = build_report(args, started, verified, data, runs_state, scored,
                          aggregate_out, args.stage)
    report["timings"]["scoring_seconds"] = score_seconds
    records_out = prediction_records(runs_state["runs"], scored["results"])

    exclusive_create_gzip(Path(args.predictions), records_out)
    exclusive_create_text(Path(args.report),
                          json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    log.write({"event": "done", "report": str(args.report),
               "predictions": str(args.predictions), "rows": len(records_out),
               "total_seconds": round(time.perf_counter() - started, 3)})
    log.close()
    print(json.dumps({"report": str(args.report), "predictions": str(args.predictions),
                      "rows": len(records_out),
                      "total_seconds": round(time.perf_counter() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
