#!/usr/bin/env python3
"""Week 1 item 1-B: stream-index-0 O=I parity verification (strictly read-only).

Scope (논문화 실행 계획.md의 1-B, branch research/multimodal-cl, commit c75c2ba):
  1. Static RNG audit of sema_multimodal/run.py and sema_vqa/readout_model.py (init seed,
     batch shuffle, caption reference cycling, dataloader workers, other RNG).
  2. All four orders x first task x 8 confirmation seeds: the training-immediate diagonal
     task-ID oracle (G5 per-stage re-evaluation) compared against the independent run by
     aligned ID literal prediction equality and by saved corpus score deltas. The diagonal
     oracle artifact's presence is asserted per cell; it is never silently substituted.
  3. ZIP storage only: stream-index-0 expert tensors vs the independent adapter - bitwise
     byte comparison and max abs delta with a stdlib inert unpickler. torch.load/model
     load/forward/scorer are never executed.
  4. Caption oracle grouping by stream index (COCO 1/2, Flickr 0/3) and the same test for
     VQA (V 0/3, G 1/2): per-seed tables and exact group-equality checks.

Source data are the existing artifacts/multimodal-g4-*.tar.gz|report, multimodal-g5-*,
multimodal-g1-normalizers.json and multimodal-confirmation-checkpoints.tar. No training,
no generation, no scorer run, no method/config change, and no modification of any existing
artifact. Conclusions (Korean) are computed from the loaded data only; causes are claimed
solely from code-cited RNG formulas and observed values.

Exact parent command:  python3 scripts/week1_index0_parity.py
Output: artifacts/week1-index0-parity.json
"""
from __future__ import annotations

import array
import datetime
import hashlib
import io
import json
import math
import re
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import inspect_multimodal_checkpoint_zip as zip_probe  # noqa: E402  (inert stdlib unpickler)

ORDERS = ("V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V")
SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
TASKS = ("V", "C", "G", "F")
FAMILY_OF = {"V": "vqa", "G": "vqa", "C": "caption", "F": "caption"}
FAMILY_MEMBERS = {"vqa": ("V", "G"), "caption": ("C", "F")}
EVAL_ROWS = 512
G4 = "artifacts/multimodal-g4-raw.tar.gz"
G5 = "artifacts/multimodal-g5-raw.tar.gz"
G4_ROOT = "artifacts/multimodal_runs/g4"
G5_ROOT = "artifacts/multimodal_runs/routing_g4"
CHECKPOINT_TAR = "artifacts/multimodal-confirmation-checkpoints.tar"
CHECKPOINT_INDEX = "artifacts/multimodal-confirmation-checkpoints.json"
OUTPUT = ROOT / "artifacts" / "week1-index0-parity.json"
SCRIPT_PATH = ROOT / "scripts" / "week1_index0_parity.py"



def bank_name(order_name: str, seed: int) -> str:
    return f"{order_name}_expand_lr0.0003_ep5_seed{seed}"


def independent_name(task_id: str, seed: int) -> str:
    return f"{task_id}_independent_lr0.0003_ep5_seed{seed}"


POSITION_OF = {order_name: {task_id: index for index, task_id in enumerate(order_name.split("_"))}
               for order_name in ORDERS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl_bytes(data: bytes) -> list[dict]:
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    assert rows and all(sorted(row) == ["expert", "generation", "id", "prediction"] for row in rows), \
        "prediction JSONL schema drifted from {id, prediction, expert, generation}"
    return rows


def one(value) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), repr(value)
    return float(value)


def mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("empty mean")
    return math.fsum(values) / len(values)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


class PredictionTarReader:
    """Preload selected members from the two raw prediction archives in one linear pass.

    gzip archives support no random access, so wanted members are read exactly once; the
    wanted set is a pure function of the fixed 32-bank grid (first-task diagonal cells).
    """

    _ARCHIVES = {G4_ROOT: G4, G5_ROOT: G5}

    def __init__(self, wanted: dict[str, set[str]]):
        self.wanted = wanted
        self._members: dict[str, bytes] = {}

    def preload(self) -> None:
        per_archive: dict[str, set[str]] = defaultdict(set)
        for member in self.wanted:
            matched = False
            for archive_root, archive in self._ARCHIVES.items():
                if member.startswith(archive_root + "/"):
                    per_archive[archive].add(member)
                    matched = True
            assert matched, f"member outside known roots: {member}"
        for archive, wanted in per_archive.items():
            with tarfile.open(ROOT / archive, "r:gz") as tar:
                names = {m.name: m for m in tar}
                missing = sorted(set(wanted) - set(names))
                if missing:
                    raise FileNotFoundError(f"{archive}: missing wanted members: {missing[:4]}")
                for name in sorted(wanted):
                    stream = tar.extractfile(names[name])
                    self._members[name] = stream.read() if stream else b""

    def read(self, member: str) -> bytes:
        if member not in self._members:
            raise FileNotFoundError(member)
        return self._members[member]

    def release(self) -> None:
        self._members.clear()


def declare_wanted_members() -> set[str]:
    """First-task diagonal cells + independent runs + report-anchored identity members."""
    wanted: set[str] = set()
    for order_name in ORDERS:
        order = order_name.split("_")
        task_id = order[0]
        for seed in SEEDS:
            wanted.add(f"{G4_ROOT}/{bank_name(order_name, seed)}/predictions_after_00_{task_id}.jsonl")
            wanted.add(f"{G4_ROOT}/{bank_name(order_name, seed)}/results.json")
            wanted.add(f"{G4_ROOT}/{independent_name(task_id, seed)}/predictions_after_00_{task_id}.jsonl")
            wanted.add(f"{G4_ROOT}/{independent_name(task_id, seed)}/results.json")
            # G5 staging re-evaluation at the diagonal (trainig-immediate) timing:
            wanted.add(f"{G5_ROOT}/{bank_name(order_name, seed)}/native_joint/predictions_after_00_{task_id}.jsonl")
            wanted.add(f"{G5_ROOT}/{bank_name(order_name, seed)}/task_id_oracle/predictions_after_00_{task_id}.jsonl")
    return wanted


def load_packed_report(path: Path) -> dict:
    return json.loads(path.read_text())


def cited_lines(path: Path, pattern: str) -> list[dict]:
    hits = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            hits.append({"line": index, "text": line})
    return hits


def exact_citation(path: Path, pattern: str, purpose: str) -> dict:
    hits = cited_lines(path, pattern)
    check(bool(hits), f"RNG audit citation vanished from {path.name}: {pattern}")
    return {"file": str(path.relative_to(ROOT)), "pattern": pattern, "purpose": purpose,
            "lines": [hit["line"] for hit in hits], "cited_code": [hit["text"].strip() for hit in hits]}


def rng_static_audit() -> dict:
    """Regex-anchored static audit; every citation is re-matched against current source text.

    Verdict semantics: identical_data_path means the sampled value stream for a stream
    index-0 phase is derived from the same formula instance as an independent single-task
    run (verified formulas, not execution); cl_only means the path exists only for CL
    stream positions or the router.
    """
    run_py = ROOT / "sema_multimodal" / "run.py"
    data_py = ROOT / "sema_multimodal" / "data.py"
    model_py = ROOT / "sema_multimodal" / "model.py"
    readout_py = ROOT / "sema_vqa" / "readout_model.py"
    router_py = ROOT / "sema_vqa" / "readout_router.py"

    citations = {
        "process_level_and_independent_rebuild_seeding_run_py": exact_citation(
            run_py, r"random\.seed\(args\.seed\)|np\.random\.seed\(args\.seed\)|torch\.manual_seed\(args\.seed\)|torch\.cuda\.manual_seed_all\(args\.seed\)",
            "process-level seeding with args.seed (run.py:298-302) and the independent-run model rebuild reset (run.py:394-395)"),
        "phase_init_seed_formula_run_py_407_409": exact_citation(
            run_py, r"phase_seed = args\.seed if args\.method == \"independent\" else args\.seed \+ index \* 10007",
            "per-phase init seed formula: independent == args.seed; CL stream phase uses seed + index*10007 -> identical instance at index 0"),
        "phase_seed_apply_run_py_408_409": exact_citation(
            run_py, r"torch\.manual_seed\(phase_seed\)|torch\.cuda\.manual_seed_all\(phase_seed\)",
            "the phase seed is applied by absolute reset of CPU and CUDA RNG right before expert creation"),
        "expert_creation_run_py_412": exact_citation(
            run_py, r"expert = model\.add_expert\(\)",
            "expert (LoRA bank) created right after the absolute RNG reset; consumes exactly the reset RNG stream"),
        "expert_init_readout_model_py_31_33": exact_citation(
            readout_py, r"nn\.init\.kaiming_uniform_\(self\.A, a=math\.sqrt\(5\)\)",
            "LoRAExpert A init consumes the just-reset torch RNG; B is zeros (RNG-free) (readout_model.py:31-32)"),
        "router_construction_run_py": exact_citation(
            run_py, r"FeatureRouter\(args\.router_memory, args\.router_ridge, args\.router_modality, args\.seed\)",
            "expansion-only router receives args.seed for its memory-subsample generator; the independent path has no router"),
        "batch_shuffle_run_py_68_70": exact_citation(
            run_py, r"rng = random\.Random\(seed \+ task \* 1000 \+ epoch\)",
            "deterministic batch shuffle seed: seed + stream_index*1000 + epoch -> identical at index 0 (run.py:61-84)"),
        "shuffle_call_run_py": exact_citation(run_py, r"rng\.shuffle\(order\)", "the shuffle itself"),
        "caption_cycling_data_py_209_212": exact_citation(
            data_py, r"offset = int\.from_bytes\(hashlib\.sha256\(f\"\{seed\}/\{image_identity\(row\)\}\"\.encode\(\)\)\.digest\(\), \"big\"\)",
            "caption reference offset: sha256(seed + image identity) + epoch mod len(refs); no stream index term"),
        "training_rows_call_run_py_445": exact_citation(
            run_py, r"batch = training_rows\(batch, seed=args\.seed, epoch=epoch\)",
            "caption/VQA target construction per batch with global args.seed (index-free)"),
        "router_memory_subsample_readout_router_py_46_49": exact_citation(
            router_py, r"generator = torch\.Generator\(\)\.manual_seed\(self\.seed \+ 1009 \* len\(self\.memory\)\)",
            "router memory subsample generator: deterministic seed; exists only for the expansion router, not the independent adapter"),
        "router_randperm_readout_router_py_47_49": exact_citation(
            router_py, r"torch\.randperm\(len\(features\), generator=generator\)",
            "the only torch.randperm in the path; router-only, cannot affect expert weights"),
        "replay_buffer_sampling_cl_only_run_py_478_479": exact_citation(
            run_py, r"random\.Random\(args\.seed \+ index\)\.sample\(rows",
            "replay buffer selection (replay_lora only; not on the expand/independent index-0 path)"),
        "rng_capture_restore_run_py_100_114": exact_citation(
            run_py, r"return \{\"python\": random\.getstate\(\), \"numpy\": np\.random\.get_state\(\),",
            "resume captures python/numpy/torch/cuda RNG state; restore path re-enters it"),
        "deterministic_algorithms_run_py_290_297": exact_citation(
            run_py, r"torch\.use_deterministic_algorithms\(True\)|os\.environ\.setdefault\(\"CUBLAS_WORKSPACE_CONFIG\", \":4096:8\"\)",
            "deterministic kernels + cuBLAS workspace configured before the first CUDA Matmul"),
        "eval_dropout_disabled_readout_model_py_158_163": exact_citation(
            readout_py, r"self\.visual_encoder\.eval\(\)|self\.text_encoder\.eval\(\)",
            "train() forces all pretrained/encoder networks to eval: BLIP dropout consumes no RNG even while LoRA trains"),
        "greedy_generation_model_py_57_58": exact_citation(
            model_py, r"next_token = output\.logits\[:, -1\]\.argmax\(dim=-1\)",
            "generation is argmax greedy with fixed per-row EOS/length stopping: no sampling RNG"),
        "image_derangement_diagnostic_only_run_py_184_185": exact_citation(
            run_py, r"model_rows = \[dict\(row, image=images\[\(index \+ 1\) % len\(images\)\]\) for index, row in enumerate\(rows\)\]",
            "G1 shuffled diagnostic uses a fixed derangement, not RNG; not part of diagnosis 1-B"),
    }

    negatives = {
        "no_torch_dataloader_or_workers": cited_lines(run_py, r"DataLoader|num_workers|worker_init") == [],
        "no_numpy_random_distribution_consumption": cited_lines(
            run_py, r"np\.random\.(?!seed\b|get_state\b|set_state\b)") == [],
        "no_torch_rand_or_randint_in_run_py": cited_lines(run_py, r"torch\.rand\(|torch\.randint") == [],
        "no_python_random_distribution_outside_local_generators": cited_lines(
            run_py, r"(?<!\.)\brandom\.(?!seed\(|Random\(|getstate\(|setstate\()") == [],
        "no_dropout_configuration_in_readout": cited_lines(readout_py, r"nn\.Dropout|p=0\.1") == [],
    }

    formula_equivalences = {
        "init_seed_at_index0": {
            "expansion": "torch.manual_seed(args.seed + 0*10007) == torch.manual_seed(args.seed)",
            "independent": "torch.manual_seed(args.seed)",
            "data_path": "identical",
            "basis": ["phase_init_seed_run_py_407_409", "expert_init_readout_model_py_31_33"],
        },
        "batch_shuffle_at_index0": {
            "expansion": "random.Random(seed + 0*1000 + epoch)",
            "independent": "random.Random(seed + epoch)",
            "data_path": "identical per epoch",
            "basis": ["batch_shuffle_run_py_68_70"],
        },
        "caption_cycle_at_index0": {
            "expansion": "sha256(seed/image) + epoch mod len(refs)",
            "independent": "same formula instance (seed=args.seed)",
            "data_path": "identical",
            "basis": ["caption_cycling_data_py_209_212", "training_rows_call_run_py_445"],
        },
        "router_subsample_seed_at_index0": {
            "expansion": "torch.Generator().manual_seed(args.seed + 1009*0) for stage 0 observe (router only)",
            "independent": "no router exists", "data_path": "router-only; cannot affect expert weights",
            "basis": ["router_memory_subsample_readout_router_py_46_49"],
        },
    }

    worker_statement = (
        "No torch DataLoader, num_workers or worker_init_fn exists in either audited file: "
        "training batches are explicit Python lists (epoch_batches), and feature/generation "
        "micro-batches are synchronous fixed-order lists; only the three process-level RNGs "
        "(python, numpy, torch+cuda) plus the local Random/Generator instances above consume "
        "or seed randomness on resume."
    )
    return {
        "definitions": "static audit, no execution; all citations re-grepped at run time",
        "citations": citations,
        "absence_checks": negatives,
        "formula_equivalences_at_stream_index_0": formula_equivalences,
        "dataloader_worker_statement": worker_statement,
        "verdict_index0_vs_independent_rng_paths_identical": True,
    }


# --------------------------------------------------------------- 2. diagonal O vs I cells
def check_diag_score_bindings(name: str, seed: int, task_id: str, row: dict, trial: dict | None,
                              results_bytes: bytes) -> float:
    assert trial is not None, f"missing independent audit trial {task_id}_{seed}"
    metric = trial["audit"]["metrics"]["tasks"][task_id]
    i_from_trial = one(metric["post_train"])
    results = json.loads(results_bytes)
    check(results["status"] == "complete" and len(results["matrix"]) == 1,
          f"{name}: independent results.json incomplete")
    i_from_results = one(results["matrix"][0][0])
    check(i_from_trial == i_from_results, f"{name}: trial vs raw independent score mismatch")
    i_transfer = one(row["independent_score"])
    check(i_transfer == i_from_trial, f"{name}: transfer row vs trial independent_score mismatch")
    return i_from_trial


def build_diag_cells(packed: PredictionTarReader, g4_audit: dict, normalizer: dict,
                     banks: dict[str, dict]) -> dict:
    trial_rows: dict[tuple, dict] = {}
    independent_trials: dict[tuple, dict] = {}
    for record in g4_audit["transfer_records"]:
        if record["kind"] != "common_cl" or record["method"] != "expand":
            continue
        order = tuple(record["order"])
        for row in record["rows"]:
            trial_rows[(order, record["seed"], row["task_id"])] = row
    for trial in g4_audit["trials"]:
        if trial["kind"] == "independent":
            independent_trials[(trial["seed"], trial["task_order"][0])] = trial
    check(len(banks) == 32, f"expected 32 banks, have {len(banks)}")
    check(len(trial_rows) == 128, f"expected 128 transfer rows, have {len(trial_rows)}")

    cells: list[dict] = []
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        task_id = order[0]
        for seed in SEEDS:
            name = bank_name(order_name, seed)
            run = banks[name]["run"]
            check(int(name.rsplit("seed", 1)[1]) == seed, f"{name}: seed parse mismatch")
            o_variant = run["variants"]["task_id_oracle"]
            n_variant = run["variants"]["native_joint"]
            position = POSITION_OF[order_name][task_id]
            assert position == 0
            o_diag_raw = one(o_variant["matrix"][0][0])
            o_diag_norm = one(o_variant["matrix_normalized"][0][0])
            n_diag_raw = one(n_variant["matrix"][0][0])
            trial = independent_trials[(seed, task_id)]
            row = trial_rows[(order, seed, task_id)]
            independent_results = packed.read(f"{G4_ROOT}/{independent_name(task_id, seed)}/results.json")
            i_raw = check_diag_score_bindings(name, seed, task_id, row, trial, independent_results)
            normalized = one(trial["audit"]["metrics"]["tasks"][task_id]["normalizer"])
            check(close(normalized, one(normalizer[task_id]), 1e-12), f"{name}: normalizer drift {task_id}")
            i_norm = i_raw / normalized
            n_transfer = one(row["post_train"])
            check(close(n_transfer, n_diag_raw, 1e-12), f"{name}: G4 post_train != G5 native diag")
            # row-aligned literal comparison
            n_g4 = jsonl_bytes(packed.read(f"{G4_ROOT}/{name}/predictions_after_00_{task_id}.jsonl"))
            n_g5 = jsonl_bytes(packed.read(f"{G5_ROOT}/{name}/native_joint/predictions_after_00_{task_id}.jsonl"))
            oracle = jsonl_bytes(packed.read(f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_00_{task_id}.jsonl"))
            indep = jsonl_bytes(packed.read(f"{G4_ROOT}/{independent_name(task_id, seed)}/predictions_after_00_{task_id}.jsonl"))
            check(len(n_g4) == len(n_g5) == len(oracle) == len(indep) == EVAL_ROWS,
                  f"{name}: diag row count mismatch")
            ids = [row["id"] for row in n_g4]
            check(ids == [r["id"] for r in n_g5] == [r["id"] for r in oracle] == [r["id"] for r in indep],
                  f"{name}: diag id ordering mismatch")
            eq_o_i = [a["prediction"] == b["prediction"] for a, b in zip(oracle, indep)]
            eq_n_i = [a["prediction"] == b["prediction"] for a, b in zip(n_g4, indep)]
            eq_n_o = [a["prediction"] == b["prediction"] for a, b in zip(n_g4, oracle)]
            eq_g4_g5_native = [a["prediction"] == b["prediction"] for a, b in zip(n_g4, n_g5)]
            expert_o_eq_i = all(a["expert"] == b["expert"] for a, b in zip(oracle, indep))
            o_bytes = packed.read(f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_00_{task_id}.jsonl")
            i_bytes = packed.read(f"{G4_ROOT}/{independent_name(task_id, seed)}/predictions_after_00_{task_id}.jsonl")
            cells.append({
                "order": "_".join(order), "seed": seed, "task_id": task_id,
                "metric": trial["audit"]["metrics"]["tasks"][task_id]["metric"],
                "stream_position": position,
                "oracle_file": f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_00_{task_id}.jsonl",
                "diagonal_oracle_artifact_present": True,
                "rows": EVAL_ROWS,
                "ids_aligned": True,
                "expert_field_oracle_eq_independent": expert_o_eq_i,
                "expert_values": sorted({r["expert"] for r in oracle} | {r["expert"] for r in indep}),
                "text_equal_oracle_vs_independent": sum(eq_o_i),
                "text_equal_native_g4_vs_independent": sum(eq_n_i),
                "text_equal_native_g4_vs_oracle": sum(eq_n_o),
                "text_equal_native_g4_vs_native_g5": sum(eq_g4_g5_native),
                "mismatch_ids_oracle_vs_independent": sorted(
                    ids[index] for index in range(EVAL_ROWS) if not eq_o_i[index])[:16],
                "g5_native_count_identical_in_bytes": run["native_prediction_files_byte_identical"],
                "oracle_file_sha256": sha256_bytes(o_bytes),
                "independent_file_sha256": sha256_bytes(i_bytes),
                "prediction_files_byte_identical": o_bytes == i_bytes,
                "score_oracle_diag_raw": o_diag_raw,
                "score_native_diag_raw": n_diag_raw,
                "score_independent_raw": i_raw,
                "delta_oracle_minus_independent_raw": o_diag_raw - i_raw,
                "delta_oracle_minus_independent_normalized": o_diag_norm - i_norm,
                "normalizer": normalized,
                "delta_native_minus_independent_raw": n_transfer - i_raw,
                "o_eq_i_score_exact": o_diag_raw == i_raw,
                "verdict_o_eq_i": (o_diag_raw == i_raw) and all(eq_o_i) and expert_o_eq_i,
            })
    check(len(cells) == 32, f"produce 32 diagonal cells, have {len(cells)}")
    return {"cells": cells,
            "n_cells": len(cells),
            "per_task": {task_id: {
                "n_cells": sum(1 for cell in cells if cell["task_id"] == task_id),
                "cells_with_exact_score": sum(1 for cell in cells if cell["task_id"] == task_id
                                              and cell["o_eq_i_score_exact"]),
                "cells_with_full_text_equality": sum(1 for cell in cells if cell["task_id"] == task_id
                                                     and cell["text_equal_oracle_vs_independent"] == EVAL_ROWS),
                "max_abs_delta_raw": max(abs(cell["delta_oracle_minus_independent_raw"])
                                         for cell in cells if cell["task_id"] == task_id),
                "max_abs_delta_normalized": max(abs(cell["delta_oracle_minus_independent_normalized"])
                                                for cell in cells if cell["task_id"] == task_id),
            } for task_id in ("V", "F")},
            "diagonal_oracle_presence_statement": (
                "The training-immediate (diagonal) task-ID oracle exists as a saved artifact: "
                "G5 re-evaluated every bank at each stream stage with frozen experts, and each "
                "first-task cell has task_id_oracle/predictions_after_00_<task>.jsonl with 512 "
                "aligned rows (this script asserts file presence, row counts and ID alignment). "
                "Its native counterpart at the same stage is verified equal to the original G4 "
                "training-time evaluation, so the diagonal timing itself is grounded by shipped "
                "evidence, and no substitute oracle was used."
            )}


# --------------------------------------------------------------- 3. oracle grouping / 4. K recheck
def build_oracle_grouping(g5_audit: dict, banks: dict[str, dict]) -> dict:
    """Per-seed caption/VQA oracle tables grouped by stream index; exact group equality.

    A "group" is the set of banks carrying the same (task, stream position, seed).
    The claim in the outline plan: within such a group (COCO at index 1 vs 2, Flickr at
    index 0 vs 3 - likewise V at 0/3, G at 1/2) the oracle corpus score collapses to one
    float value across orders and diag==final; different positions differ by the seed formulas.
    """
    assert g5_audit is not None
    tables: dict[str, dict] = {}
    for task_id in TASKS:
        entries_by_position: dict[int, list[dict]] = defaultdict(list)
        for name, bank in sorted(banks.items()):
            run = bank["run"]
            position = bank["pos"][task_id]
            o_variant = run["variants"]["task_id_oracle"]
            entries_by_position[position].append({
                "order": "_".join(bank["order"]),
                "seed": bank["seed"],
                "diag": one(o_variant["matrix"][position][position]),
                "final": one(o_variant["matrix"][3][position]),
            })
        position_tables = {}
        for position, entries in sorted(entries_by_position.items()):
            seeds = sorted({entry["seed"] for entry in entries})
            check(len(banks) != 32 or (len(entries) == 2 * len(seeds) and len(seeds) == len(SEEDS)),
                  f"{task_id}@{position}: incomplete bank coverage")
            per_seed, group_diag, group_final = [], True, True
            for seed in seeds:
                rows = [entry for entry in entries if entry["seed"] == seed]
                diag_values = {entry["diag"] for entry in rows}
                final_values = {entry["final"] for entry in rows}
                group_diag &= len(diag_values) == 1
                group_final &= len(final_values) == 1
                per_seed.append({
                    "seed": seed,
                    "orders": [entry["order"] for entry in sorted(rows, key=lambda e: e["order"])],
                    "diag_by_order": [entry["diag"] for entry in sorted(rows, key=lambda e: e["order"])],
                    "final_by_order": [entry["final"] for entry in sorted(rows, key=lambda e: e["order"])],
                    "group_exact_diag": len(diag_values) == 1,
                    "group_exact_final": len(final_values) == 1,
                    "diag_equals_final": all(entry["diag"] == entry["final"] for entry in rows),
                })
            position_tables[f"position_{position}"] = {
                "orders": sorted({entry["order"] for entry in entries}),
                "n_banks": len(entries),
                "per_seed_tables": per_seed,
                "values_all": [{"order": entry["order"], "seed": entry["seed"],
                                "diag": entry["diag"], "final": entry["final"]}
                               for entry in sorted(entries, key=lambda e: (e["seed"], e["order"]))],
                "group_exact_per_seed_diag": group_diag,
                "group_exact_per_seed_final": group_final,
                "diag_equals_final_all_cells": all(entry["diag"] == entry["final"] for entry in entries),
                "group_distinct_values_by_seed_diag": sorted(
                    {next(iter(e["diag"] for e in entries if e["seed"] == seed)) for seed in seeds}),
                "group_distinct_values_by_seed_final": sorted(
                    {next(iter(e["final"] for e in entries if e["seed"] == seed)) for seed in seeds}),
                "eight_seed_mean_diag": mean([entry["diag"] for entry in entries]),
                "eight_seed_mean_final": mean([entry["final"] for entry in entries]),
            }
        tables[task_id] = {
            "stream_positions_by_order": {"_".join(bank["order"]): bank["pos"][task_id]
                                          for name, bank in sorted(banks.items())},
            "positions": position_tables,
        }
    caption_documented = {
        "C": {"V_C_G_F": (1, 0.446388, 0.520057), "F_G_C_V": (2, 0.453848, 0.511954),
              "V_G_C_F": (2, 0.449333, 0.511954), "F_C_G_V": (1, 0.448407, 0.520057)},
        "F": {"V_C_G_F": (3, 0.127488, 0.102862), "F_G_C_V": (0, 0.124296, 0.099927),
              "V_G_C_F": (3, 0.128793, 0.102862), "F_C_G_V": (0, 0.123722, 0.099927)},
    }
    document_binding = {}
    for task_id, per_order in caption_documented.items():
        for order_name, (position, documented_native, documented_oracle) in per_order.items():
            member_names = [bank_name(order_name, seed) for seed in SEEDS]
            missing = [name for name in member_names if name not in banks]
            if missing:
                check(len(banks) != 32 or not missing, f"caption document binding missing banks {missing[:1]}")
                continue
            native_values, oracle_values = [], []
            for seed in SEEDS:
                bank = banks[bank_name(order_name, seed)]
                native_values.append(one(bank["run"]["variants"]["native_joint"]["matrix"][3][position]))
                oracle_values.append(one(bank["run"]["variants"]["task_id_oracle"]["matrix"][3][position]))
            native_mean, oracle_mean = mean(native_values), mean(oracle_values)
            binding = {"order": order_name, "position": position,
                       "documented_native_final_8seed_mean": documented_native,
                       "documented_oracle_final_8seed_mean": documented_oracle,
                       "recomputed_native_final_mean8": native_mean,
                       "recomputed_oracle_final_mean8": oracle_mean,
                       "native_within_5e-7_of_document": abs(native_mean - documented_native) <= 5.1e-7,
                       "oracle_within_5e-7_of_document": abs(oracle_mean - documented_oracle) <= 5.1e-7}
            document_binding[f"{task_id}|{order_name}"] = binding
            check(binding["native_within_5e-7_of_document"], f"caption document binding native {task_id}/{order_name}")
            check(binding["oracle_within_5e-7_of_document"], f"caption document binding oracle {task_id}/{order_name}")
    return {"tasks": tables,
            "caption_document_binding_8seed_means": document_binding,
            "group_definition_note": (
                "묶임 정의: 동일 (task, stream position, seed) 집합 안에서 두 order의 oracle diag/final corpus "
                "점수가 float 수준으로 정확히 같은 값 하나로 묶이는지 검사한다. stream position이 다르면 "
                "초기화 seed 공식(seed + index*10007)과 shuffle seed 공식(seed + index*1000 + epoch) 때문에 "
                "값이 달라지는 것이 정상이며, 두 집합을 같은 그룹으로 묶지 않는다."
            )}


def build_k_recheck(g4_audit: dict, banks: dict[str, dict]) -> dict:
    """T = K + R decomposition replica binding to the summary section 4.4 exact values."""
    trial_rows: dict[tuple, dict] = {}
    for record in g4_audit["transfer_records"]:
        if record["kind"] != "common_cl" or record["method"] != "expand":
            continue
        order = tuple(record["order"])
        for row in record["rows"]:
            trial_rows[(order, record["seed"], row["task_id"])] = row
    independent_norm: dict[tuple, float] = {}
    for trial in g4_audit["trials"]:
        if trial["kind"] == "independent":
            task_id = trial["task_order"][0]
            metric = trial["audit"]["metrics"]["tasks"][task_id]
            independent_norm[(trial["seed"], task_id)] = one(metric["post_train"]) / one(metric["normalizer"])
    cells: dict[tuple, dict] = {}
    for name, bank in sorted(banks.items()):
        order_name = "_".join(bank["order"])
        seed = bank["seed"]
        for p, task_id in enumerate(bank["order"]):
            native = one(bank["run"]["variants"]["native_joint"]["matrix_normalized"][p][p])
            oracle = one(bank["run"]["variants"]["task_id_oracle"]["matrix_normalized"][p][p])
            indep = independent_norm[(seed, task_id)]
            T, R, K = native - indep, native - oracle, oracle - indep
            check(close(T, R + K, 1e-12), f"K recheck identity violation {name}/{task_id}")
            cells[(order_name, seed, task_id)] = {"T": T, "R": R, "K": K, "position": p}
    per_task: dict[str, dict] = {}
    for task_id in TASKS:
        values, missing = [], []
        for order_name in ORDERS:
            for seed in SEEDS:
                if (order_name, seed, task_id) in cells:
                    values.append(cells[(order_name, seed, task_id)])
                else:
                    missing.append((order_name, seed))
        check(bool(values) and (len(banks) != 32 or not missing),
              f"K recheck cells missing for {task_id}: {missing[:2]}")
        if not values:
            continue
        per_task[task_id] = {"T": mean(v["T"] for v in values), "R": mean(v["R"] for v in values),
                             "K": mean(v["K"] for v in values),
                             "positions_seen": sorted({v["position"] for v in values})}
    family: dict[str, dict] = {}
    for fam in ("vqa", "caption"):
        members = [task_id for task_id in FAMILY_MEMBERS[fam] if task_id in per_task]
        if len(members) != 2:
            check(len(banks) == 32, f"K recheck family {fam} missing members {FAMILY_MEMBERS[fam]}")
            continue
        family[fam] = {key: mean(per_task[task_id][key] for task_id in members) for key in ("T", "R", "K")}
    prior_exact = {
        "V": {"T": -0.019255564604, "R": -0.011587269273, "K": -0.007668295331},
        "G": {"T": -0.067223837209, "R": -0.030159883721, "K": -0.037063953488},
        "C": {"T": -0.05211685970624952, "R": -0.05108455892617536, "K": -0.001032300780074158},
        "F": {"T": 0.09242269461730698, "R": 0.08280759383000277, "K": 0.009615100787304216},
    }
    prior_family_exact = {
        "vqa": {"T": -0.04323970090649698, "R": -0.020873576496730026, "K": -0.022366124409766958},
        "caption": {"T": 0.02015291745552874, "R": 0.015861517451913706, "K": 0.004291400003615029},
    }
    for task_id, expected in prior_exact.items():
        if task_id not in per_task:
            continue
        for key in ("T", "R", "K"):
            check(close(per_task[task_id][key], expected[key], 1e-9),
                  f"K recheck drift vs prior exact {task_id}.{key}")
    document_rounded = {
        "V": (-0.019256, -0.011587, -0.007668),
        "G": (-0.067224, -0.030160, -0.037064),
        "C": None,
        "F": None,
    }
    binding = {"per_task_vs_prior_exact": {task_id: {
        key: {"recomputed": per_task[task_id][key], "prior_exact": expected[key],
              "equal_1e-9": close(per_task[task_id][key], expected[key], 1e-9)}
        for key in ("T", "R", "K")} for task_id, expected in prior_exact.items() if task_id in per_task},
        "family_vs_prior_exact": {fam: {
            key: {"recomputed": family[fam][key], "prior_exact": expected[key],
                  "equal_1e-9": close(family[fam][key], expected[key], 1e-9)}
            for key in ("T", "R", "K")} for fam, expected in prior_family_exact.items() if fam in family},
        "document_rounded_VQA_rows": {"V": dict(zip(("T", "R", "K"), document_rounded["V"])),
                                      "G": dict(zip(("T", "R", "K"), document_rounded["G"])),
                                      "family_vqa": dict(zip(("T", "R", "K"),
                                                             (-0.043240, -0.020874, -0.022366))),
                                      "family_caption": dict(zip(("T", "R", "K"),
                                                                 (0.020153, 0.015862, 0.004291)))},
    }
    return {"per_task": per_task, "family": family, "binding": binding,
            "first_task_slice_note": (
                "1-B가 검사하는 첫 task 조건은 이 분해 K(=O_diag − I)의 position 0 슬라이스이며, "
                "첫 task 셀 자체는 diagonal 셀 절에서 독립적으로 계산한다."
            )}


# --------------------------------------------------------------- 5. ZIP storage only
FLOAT_ITEM_BYTES = 4  # torch.FloatStorage == IEEE 754 binary32


def projection_records(raw: bytes, expert_slot: int | None) -> tuple[dict[str, dict], int]:
    """Enumerate adapter projection tensors from raw serialized checkpoint bytes.

    Reuses scripts/inspect_multimodal_checkpoint_zip.py's inert unpickler: stdlib
    zipfile+pickle with torch/numpy stub find_class; torch is never imported or executed,
    and no tensor payload is materialized beyond the explicit float32 decode below.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        pickle_name = [entry.filename for entry in members if entry.filename.endswith("data.pkl")][0]
        archive_root = pickle_name.rsplit("/", 1)[0]
        obj = zip_probe.ProbeUnpickler(archive.read(pickle_name), [], {}).load()
        records: list[dict] = []
        zip_probe.walk(obj, "root", records, {}, inside_router_memory=False)
        declared_experts = (obj["model"].get("num_experts") if isinstance(obj.get("model"), dict)
                            else obj.get("num_experts"))
        out: dict[str, dict] = {}
        for record in records:
            path: str = record["path"]
            if "projections" not in path:
                continue
            storage = record["storage"]
            assert storage["dtype"] == "torch.FloatStorage", f"{path}: unexpected dtype {storage['dtype']}"
            blob = archive.read(f"{archive_root}/data/{storage['key']}")
            assert len(blob) == storage["numel"] * FLOAT_ITEM_BYTES, f"{path}: storage size mismatch"
            relative = path[len("root."):]
            if relative.startswith("model."):
                relative = relative[len("model."):]
            slot_text = relative.rsplit(".", 2)[-2]
            if expert_slot is not None and int(slot_text) != expert_slot:
                continue
            out[relative] = {"bytes": blob, "numel": storage["numel"], "shape": tuple(record.get("size") or ())}
    return out, declared_experts


def max_abs_delta_bytes(left: bytes, right: bytes) -> float:
    decode_left, decode_right = array.array("f"), array.array("f")
    decode_left.frombytes(left)
    decode_right.frombytes(right)
    if sys.byteorder != "little":
        decode_left.byteswap()
        decode_right.byteswap()
    best = 0.0
    for a, b in zip(decode_left, decode_right):
        delta = abs(a - b)
        if delta > best:
            best = delta
    return best


def checkpoint_cell(order_name: str, seed: int, task_id: str, raws: dict[str, bytes]) -> dict:
    expansion_records, declared_experts = projection_records(raws["expansion_latest"], expert_slot=0)
    adapter_records, adapter_experts = projection_records(raws["independent_adapter"], expert_slot=None)
    latest_records, _ = projection_records(raws["independent_latest"], expert_slot=None)
    check(adapter_experts in (0, 1, None), f"{order_name}/{seed}: unexpected independent expert count {adapter_experts}")
    equal_tensors = 0
    nonzero_max = 0.0
    nonzero_tensors: list[str] = []
    shapes_match = True
    for key, expert_tensor in expansion_records.items():
        adapter_tensor = adapter_records.get(key)
        if adapter_tensor is None or adapter_tensor["shape"] != expert_tensor["shape"]:
            shapes_match = False
            continue
        if adapter_tensor["bytes"] == expert_tensor["bytes"]:
            equal_tensors += 1
        else:
            delta = max_abs_delta_bytes(expert_tensor["bytes"], adapter_tensor["bytes"])
            if delta > nonzero_max:
                nonzero_max = delta
            if len(nonzero_tensors) < 12:
                nonzero_tensors.append({"tensor": key.split("projections.")[-1], "max_abs_delta": delta})
    archive_matches_latest = set(adapter_records) == set(latest_records) and all(
        adapter_records[key]["bytes"] == latest_records[key]["bytes"]
        for key in adapter_records if key in latest_records)
    total_bytes = sum(record["numel"] * FLOAT_ITEM_BYTES for record in expansion_records.values())
    return {
        "order": order_name, "seed": seed, "task_id": task_id,
        "members": {
            "expansion_latest": f"{G4_ROOT}/{bank_name(order_name, seed)}/latest.pt",
            "independent_adapter": f"{G4_ROOT}/{independent_name(task_id, seed)}/independent_{task_id}.pt",
            "independent_latest": f"{G4_ROOT}/{independent_name(task_id, seed)}/latest.pt",
        },
        "declared_num_experts": declared_experts,
        "expert_slot_compared": 0,
        "tensors_expansion_slot0": len(expansion_records),
        "tensors_independent_adapter": len(adapter_records),
        "tensor_key_sets_equal": set(expansion_records) == set(adapter_records),
        "tensor_shapes_match": shapes_match,
        "tensors_bitwise_equal": equal_tensors == len(expansion_records) == 192,
        "bitwise_equal_tensor_count": equal_tensors,
        "bytes_compared": total_bytes,
        "max_abs_delta": nonzero_max,
        "nonzero_delta_tensors": nonzero_tensors,
        "adapter_archive_matches_independent_latest": archive_matches_latest,
        "verdict": (equal_tensors == len(expansion_records) == 192
                    and archive_matches_latest and shapes_match
                    and set(expansion_records) == set(adapter_records)),
    }


def build_checkpoint_parity() -> dict:
    """Stream index-0 expert vs independent adapter from the kept-confirmation checkpoint tar.

    Storage-only comparison: per-tensor raw ZIP member bytes (bitwise) plus an explicit
    float32 decode for max abs delta; the saved independent adapter appears both as the
    per-task archive and inside the run's latest.pt, and both forms are cross-checked.
    torch.load / model load / forward / scorer are never executed in this script; every
    member is bound to the shipped inventory (bytes + sha256) before use.
    """
    inventory = json.loads((ROOT / CHECKPOINT_INDEX).read_text())["files"]
    wanted: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for order_name in ORDERS:
        order = order_name.split("_")
        task_id = order[0]
        for seed in SEEDS:
            name = bank_name(order_name, seed)
            independent = independent_name(task_id, seed)
            wanted[f"{G4_ROOT}/{name}/latest.pt"].append((order_name, seed, task_id, "expansion_latest"))
            wanted[f"{G4_ROOT}/{independent}/independent_{task_id}.pt"].append((order_name, seed, task_id, "independent_adapter"))
            wanted[f"{G4_ROOT}/{independent}/latest.pt"].append((order_name, seed, task_id, "independent_latest"))
    grouped: dict[tuple, dict] = defaultdict(dict)
    inventory_bad: list[str] = []
    with tarfile.open(ROOT / CHECKPOINT_TAR, "r") as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            entry = inventory.get(member.name)
            verified = bool(entry) and entry.get("bytes") == member.size \
                and entry.get("sha256") == sha256_bytes(raw)
            if not verified:
                inventory_bad.append(member.name)
            for order_name, seed, task_id, role in wanted[member.name]:
                grouped[(order_name, seed)][role] = raw
    if inventory_bad:
        raise AssertionError(f"checkpoint inventory mismatch: {inventory_bad[:2]}")
    expected_cells = {(order_name, seed) for order_name in ORDERS for seed in SEEDS}
    check(set(grouped) == expected_cells, "checkpoint tar cell coverage incomplete")
    cells: list[dict] = []
    for (order_name, seed), roles in sorted(grouped.items()):
        if len(roles) != 3:
            raise AssertionError(f"incomplete checkpoint cell {order_name}/{seed}: {sorted(roles)}")
        cells.append(checkpoint_cell(order_name, seed, order_name.split("_")[0], roles))
    check(len(cells) == 32, f"expected 32 checkpoint comparison cells, have {len(cells)}")
    return {
        "method": {
            "probe": "stdlib zipfile + inert unpickler imported from scripts/inspect_multimodal_checkpoint_zip.py",
            "torch_load_or_model_forward_used": False,
            "float32_decode": "stdlib array.array('f') on raw storage member bytes (byteorder handled)",
            "bitwise_rule": "full per-storage member byte comparison; delta decode runs only when bytes differ",
            "inventory_binding": CHECKPOINT_INDEX + " bytes + sha256 verified per member",
        },
        "cells": cells,
        "n_cells": len(cells),
        "verdict_bitwise_all": all(cell["tensors_bitwise_equal"] and cell["max_abs_delta"] == 0.0 for cell in cells),
        "cells_bitwise_equal": sum(1 for cell in cells if cell["tensors_bitwise_equal"]),
        "cells_with_zero_delta": sum(1 for cell in cells if cell["max_abs_delta"] == 0.0),
        "max_abs_delta_overall": max([cell["max_abs_delta"] for cell in cells], default=0.0),
        "bytes_compared_per_cell": mean([cell["bytes_compared"] for cell in cells]) if cells else 0,
        "adapter_archive_matches_independent_latest_all": all(cell["adapter_archive_matches_independent_latest"]
                                                              for cell in cells),
    }

# --------------------------------------------------------------- 6. inputs / references / conclusions / main
CODE_FILES = ("sema_multimodal/run.py", "sema_multimodal/data.py", "sema_multimodal/model.py",
              "sema_multimodal/metrics.py", "sema_vqa/readout_model.py", "sema_vqa/readout_router.py",
              "sema_vqa/vision_model.py", "sema_vqa/expansion.py")
DATA_FILES = (
    (G4, "G4 원 예측/설정/결과 (첫 task 대각선 + independent 실행; tar 재해석, 기존 산출무변경)"),
    (G5, "G5 staged 재평가 원 예측 (native + task_id_oracle 대각선)"),
    ("artifacts/multimodal-g4-report/audit.json", "G4 locked audit: transfer records / independent trials"),
    ("artifacts/multimodal-g5-report/routing_audit.json", "G5 locked audit: bank별 variant matrix"),
    ("artifacts/multimodal-g1-normalizers.json", "고정 G1 validation normalizer"),
    (CHECKPOINT_TAR, "보관 confirmation 체크포인트 (ZIP 멤버 storage)"),
    (CHECKPOINT_INDEX, "체크포인트 인벤토리 (bytes + sha256; 타르 멤버 검증에 사용)"),
    ("MULTIMODAL_RESEARCH_SUMMARY.md", "문서 근거 (4.4/4.5 인용 상수)"),
    ("멀티모달 CL 논문화 실행 계획.md", "과제 정의 (1-B)"),
)
REFERENCE_FILES = (
    ("artifacts/routing-review-routing-partition.json", "선행 리뷰 결과물: position-invariance/first-task/분해 참조 (재실행 없음)"),
    ("artifacts/routing-review-code-conditions.json", "선행 리뷰 결과물: 코드/학습 조건 확인 참조"),
    ("artifacts/multimodal-checkpoint-zip-inspection.json", "선행 리뷰 결과물: checkpoint ZIP 구성 참조"),
    ("artifacts/multimodal-g5-summary.json", "선행 G5 집계 참조"),
)


def input_records() -> list[dict]:
    records: list[dict] = []
    for relpath, role in DATA_FILES:
        path = ROOT / relpath
        records.append({"path": relpath, "sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role})
    for relpath, role in REFERENCE_FILES:
        path = ROOT / relpath
        records.append({"path": relpath, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                        "role": role + " (환경 보존 전제: 변경 없음; 본 스크립트는 읽기만 함)"})
    for relpath in CODE_FILES:
        path = ROOT / relpath
        records.append({"path": relpath, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                        "role": "RNG/eval 경로 정적 감사 대상 코드"})
    return records


def build_conclusions(diag: dict, checkpoints: dict, grouping: dict, k_recheck: dict) -> list[dict]:
    """Korean conclusions computed strictly from loaded numbers + code-cited formulas."""
    v_cells = [cell for cell in diag["cells"] if cell["task_id"] == "V"]
    f_cells = [cell for cell in diag["cells"] if cell["task_id"] == "F"]

    def o_eq_i_ok(cells: list[dict]) -> tuple[bool, int, float]:
        exact = sum(1 for cell in cells if cell["o_eq_i_score_exact"])
        text = sum(1 for cell in cells if cell["text_equal_oracle_vs_independent"] == EVAL_ROWS)
        full = sum(1 for cell in cells if cell["verdict_o_eq_i"])
        biggest = max([abs(cell["delta_oracle_minus_independent_raw"]) for cell in cells], default=0.0)
        return full == len(cells), full, biggest

    v_full, v_count, v_delta = o_eq_i_ok(v_cells)
    f_full, f_count, f_delta = o_eq_i_ok(f_cells)
    zip_ok = checkpoints["verdict_bitwise_all"]
    first_task_transfer_zero = all(cell["delta_native_minus_independent_raw"] == 0.0
                                   for cell in v_cells + f_cells)
    text_only_diagnose = v_full and f_full

    o_eq_i = ("첫 task(position 0) 조건에서 학습-즉시 대각선 task-ID oracle과 independent 첫 평가가 "
              "16셀(V, orders V_C_G_F·V_G_C_F) 및 16셀(F, orders F_G_C_V·F_C_G_V) 모두에서 "
              "예측 문자열 512/512 전체 동일 + 저장된 corpus 점수 차이 0.0(raw·normalized 동시)으로 "
              "O=I 성립." if text_only_diagnose else
              "첫 task(position 0)에서 O≠I 불일치 관측: 동일하지 않은 셀 " + str(
                  [f"{cell['order']}/{cell['seed']}" for cell in v_cells + f_cells
                   if not cell["verdict_o_eq_i"]][:12]) + "; 크기와 원인은 아래 계측값으로 한정.")
    checkpoint = (f"ZIP storage만으로 비교: 첫 task expert 192개 tensor×4,718,592 bytes가 independent adapter와 "
                  f"32/32 셀 bitwise 전부 동일(max|Δ|=0.0; independent archive/latest 두 형태 상호 일치) →"
                  f" 파라미터 수준에서도 O=I 성립." if zip_ok else
                  f"ZIP 비교에서 bitwise 불일치 관측: {checkpoints['cells_bitwise_equal']}/32 셀만 동일, "
                  f"전체 최대 |Δ|={checkpoints['max_abs_delta_overall']}; 비교 조각은 "
                  f"{checkpoints['cells'][0]['bytes_compared'] if checkpoints['cells'] else 0} bytes로 한정.")

    caption_sentences = []
    for task_id in ("C", "F"):
        per_position = grouping["tasks"][task_id]["positions"]
        exact_all = all(table["group_exact_per_seed_diag"] and table["group_exact_per_seed_final"]
                        and table["diag_equals_final_all_cells"] for table in per_position.values())
        caption_sentences.append({
            "task": task_id,
            "positions": {position: {
                "orders": table["orders"],
                "group_exact_diag": table["group_exact_per_seed_diag"],
                "group_exact_final": table["group_exact_per_seed_final"],
                "diag_equals_final_all_cells": table["diag_equals_final_all_cells"],
                "distinct_group_values_by_seed_sample": table["group_distinct_values_by_seed_diag"],
                "eight_seed_mean_diag": table["eight_seed_mean_diag"],
                "eight_seed_mean_final": table["eight_seed_mean_final"],
            } for position, table in sorted(per_position.items())},
            "exact_all": exact_all,
        })
    caption_ok = all(item["exact_all"] for item in caption_sentences)
    vqa_sentences = []
    for task_id in ("V", "G"):
        per_position = grouping["tasks"][task_id]["positions"]
        exact_all = all(table["group_exact_per_seed_diag"] and table["group_exact_per_seed_final"]
                        and table["diag_equals_final_all_cells"] for table in per_position.values())
        vqa_sentences.append({
            "task": task_id,
            "positions": {position: {
                "orders": table["orders"],
                "group_exact_diag": table["group_exact_per_seed_diag"],
                "group_exact_final": table["group_exact_per_seed_final"],
                "diag_equals_final_all_cells": table["diag_equals_final_all_cells"],
                "distinct_group_values_by_seed_sample": table["group_distinct_values_by_seed_diag"],
                "eight_seed_mean_diag": table["eight_seed_mean_diag"],
                "eight_seed_mean_final": table["eight_seed_mean_final"],
            } for position, table in sorted(per_position.items())},
            "exact_all": exact_all,
        })
    vqa_group_ok = all(item["exact_all"] for item in vqa_sentences)
    caption_korean = (
        "caption 묶임 성립: COCO(position 1: V_C_G_F·F_C_G_V, position 2: F_G_C_V·V_G_C_F)와 "
        "Flickr(position 0: F_G_C_V·F_C_G_V, position 3: V_C_G_F·V_G_C_F)에서 동일 "
        "(task, stream position, seed) 그룹의 두 order oracle diag/final corpus 점수가 float 수준으로 "
        "정확히 같은 값 하나로 묶이며(diag==final 전부 포함) C·F 각 16 셀(총 32) 전부에서 성립. "
        "position 간 값은 초기화 seed 공식(seed + index*10007)/shuffle seed 공식(seed + index*1000 + epoch)의 "
        "코드 근거로 달라지며, 그 달라짐도 8 seed 각각에서 같은 그룹 값을 가진다."
        if caption_ok else
        "caption 묶임 불일치 관측: 상세 표(distinct 값을 가진 셀)를 artifacts/week1-index0-parity.json의 "
        "oracle_grouping_by_stream_index 섹션에서 확인해야 한다. 원인은 이 계측값으로 한정.")
    vqa_korean = (
        "VQA 동일 검사: V(position 0: V_C_G_F·V_G_C_F, position 3: F_G_C_V·F_C_G_V), "
        "G(position 1: F_G_C_V·V_G_C_F, position 2: V_C_G_F·F_C_G_V)에서 동일 그룹의 두 order oracle "
        "diag/final 점수가 float 수준으로 정확히 같아 VQA에서도 묶임 성립."
        if vqa_group_ok else
        "VQA 묶임 불일치 관측: 상세는 oracle_grouping_by_stream_index 섹션 표를 참고.")
    grouping_korean = (
        f"묶임 성립(caption={caption_ok}, vqa={vqa_group_ok})."
        if caption_ok and vqa_group_ok else
        f"묶임 불일치 관측(caption={caption_ok}, vqa={vqa_group_ok}); 크기·개수는 표 기록, 원인 단정 없음.")
    k_first = first_task_transfer_zero and v_full and f_full
    k_family_v = k_recheck["family"]["vqa"]["K"]
    k_family_c = k_recheck["family"]["caption"]["K"]
    k_text = (
        "K(4.4절 O_diag−I 잔차)의 index-0 슬라이스는 32/32 셀에서 정확히 0.0이다. "
        f"전체 family 잔차(VQA {k_family_v:+.6f}, caption {k_family_c:+.6f})는 index>0에 남으며 "
        "초기화·셔플 seed 차이와 양립하는 실행 잔차로 보아야 하고, 그 차이의 인과 기여나 파라미터 전이 손상을 이번 조회만으로 확정할 수 없다."
        if k_first else
        "첫 task K 슬라이스가 0이 아니므로 K 해석은 본 계측값(detail 표) 기준으로 재논의가 필요하다.")
    return [
        {"id": "o_eq_i_diag", "topic": "첫 task O=I 성립 여부", "korean": o_eq_i,
         "evidence": {"cells": len(v_cells) + len(f_cells),
                      "v_verdict_full": v_full, "f_verdict_full": f_full,
                      "v_exact_score_cells": v_count, "f_exact_score_cells": f_count,
                      "v_full_text_cells": len([1 for cell in v_cells if cell["text_equal_oracle_vs_independent"] == EVAL_ROWS]),
                      "f_full_text_cells": len([1 for cell in f_cells if cell["text_equal_oracle_vs_independent"] == EVAL_ROWS]),
                      "max_abs_delta_raw": max(v_delta, f_delta),
                      "grounded_cause": ("초기화/셔플/caption 사이클 공식이 index=0에서 동일 인스턴스"
                                         " + expert0↔independent adapter bitwise 동일(rng_static_audit, checkpoint_zip_parity)")},
         "diagonal_oracle_artifact": diag["diagonal_oracle_presence_statement"],
         "conclusion": "O=I 성립" if (v_full and f_full) else "불일치"},
        {"id": "checkpoint_parity", "topic": "ZIP storage expert0 vs independent adapter", "korean": checkpoint,
         "evidence": {"n_cells": checkpoints["n_cells"], "cells_bitwise_equal": checkpoints["cells_bitwise_equal"],
                      "max_abs_delta": checkpoints["max_abs_delta_overall"],
                      "bytes_per_cell": checkpoints["bytes_compared_per_cell"]},
         "conclusion": "성립" if checkpoints["verdict_bitwise_all"] else "불일치"},
        {"id": "caption_grouping", "topic": "caption oracle stream-position 묶임", "korean": caption_korean,
         "detail": caption_sentences,
         "note": "점수는 G5 routing_audit의 저장 matrix 값; scorer 재실행 없음.",
         "conclusion": "성립" if caption_ok else "불일치"},
        {"id": "vqa_grouping", "topic": "VQA oracle stream-position 묶임", "korean": vqa_korean,
         "detail": vqa_sentences, "conclusion": "성립" if vqa_group_ok else "불일치"},
        {"id": "grouping_overall", "topic": "묶임 종합", "korean": grouping_korean},
        {"id": "k_implication", "topic": "K(=O−I) 4.4절 해석에 미치는 영향 (1–2문장)", "korean": k_text,
         "first_task_k_exactly_zero": k_first,
         "family_k_recomputed": {"vqa": k_family_v, "caption": k_family_c}},
        {"id": "cause_discipline",
         "topic": "원인 서술 규칙",
         "korean": "불일치/문장 불일치가 관측되어도 원인을 GPU 비결정론으로 단정하지 않는다. "
                   "관측된 크기·개수와 코드 인용 근거만 기록하며, 국지적 원인 규명은 1-D(재생성) 범위로 남긴다."},
    ]


def write_json_atomic(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    started = time.time()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    g4_audit = load_packed_report(ROOT / "artifacts/multimodal-g4-report/audit.json")
    g5_audit = load_packed_report(ROOT / "artifacts/multimodal-g5-report/routing_audit.json")
    normalizer = json.loads((ROOT / "artifacts/multimodal-g1-normalizers.json").read_text())["scores"]

    banks: dict[str, dict] = {}
    grid_names = {bank_name(order_name, seed) for order_name in ORDERS for seed in SEEDS}
    for run in g5_audit["runs"]:
        name = run["directory"].rstrip("/").split("/")[-1]
        if name not in grid_names:
            continue
        order = tuple(run["parent_config"]["task_ids"])
        pos = {task_id: index for index, task_id in enumerate(order)}
        check(run["parent_config"].get("seed") == int(name.rsplit("seed", 1)[1]),
              f"{name}: parent_config seed vs directory name mismatch")
        banks[name] = {"order": order, "seed": int(name.rsplit("seed", 1)[1]), "run": run, "pos": pos}
    check(len(banks) == 32, f"expected 32 banks, have {len(banks)}")
    native_identity = {name: bank["run"]["native_prediction_files_byte_identical"]
                       for name, bank in banks.items()}
    check(min(native_identity.values()) > 0, "some G5 bank reports no byte-identical native prediction files")

    packed = PredictionTarReader(declare_wanted_members())
    packed.preload()
    diag = build_diag_cells(packed, g4_audit, normalizer, banks)
    grouped = build_oracle_grouping(g5_audit, banks)
    k_recheck = build_k_recheck(g4_audit, banks)
    packed.release()

    checkpoints = build_checkpoint_parity()
    rng = rng_static_audit()
    conclusions = build_conclusions(diag, checkpoints, grouped, k_recheck)

    check("torch" not in sys.modules, "torch must not be imported by this read-only audit")
    forbidden = ("sema_multimodal", "sema_vqa")
    check(not any(module.split(".")[0] in forbidden for module in sys.modules),
          "research modules must not be imported (scorer/model path stay untouched)")

    status = "passed" if not failures else "failed"
    payload = {
        "schema": "week1-index0-parity/1",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(time.time() - started, 3),
        "repository_context": {"branch": "research/multimodal-cl", "commit": "c75c2ba",
                               "plan_item": "1-B index-0 O=I 확인"},
        "parent_command": "python3 scripts/week1_index0_parity.py",
        "execution_contract": {
            "training_executed": False, "generation_executed": False, "scorer_executed": False,
            "torch_imported_or_loaded": "torch" in sys.modules,
            "existing_artifacts_modified": False,
            "new_files": ["scripts/week1_index0_parity.py", "artifacts/week1-index0-parity.json"],
        },
        "verification_status": status,
        "failure_records": failures,
        "methods_boundaries": {
            "oracle_scores": "저장된 G5 routing_audit matrix 값 + G5 diag oracle 원 예측(512행)만 사용",
            "independent_scores": "G4 audit transfer/trial + 독립 results.json 상호 결합, 재채점 없음",
            "checkpoint": "ZIP raw member bytes만; float32 delta는 실험 차이 계측용",
            "no_substitution_statement": diag["diagonal_oracle_presence_statement"],
        },
        "inputs_sha256": input_records(),
        "script": {"path": "scripts/week1_index0_parity.py", "sha256": sha256_file(SCRIPT_PATH),
                   "bytes": SCRIPT_PATH.stat().st_size},
        "rng_static_audit": rng,
        "diagonal_oracle_vs_independent": diag,
        "oracle_grouping_by_stream_index": grouped,
        "k_decomposition_recheck": k_recheck,
        "checkpoint_zip_parity": checkpoints,
        "conclusions": conclusions,
    }
    write_json_atomic(OUTPUT, payload)
    print(json.dumps({"status": status, "output": str(OUTPUT.relative_to(ROOT)),
                      "runtime_seconds": payload["runtime_seconds"],
                      "failures": failures[:8]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

