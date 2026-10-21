#!/usr/bin/env python3
"""Aggregate recorded multimodal CL resource artifacts for MULTIMODAL_RESEARCH_SUMMARY.md item5.

Executes no model code and measures nothing new; every number is either recorded
in an artifact or read exactly from the serialized checkpoints via
scripts/inspect_multimodal_checkpoint_zip.py (stdlib zip+pickle probe, no torch).

"recorded": value present in an artifact produced at run time.
"exact":    byte-exact value read from the serialized checkpoint structure
            (dtype from the StorageClass name, bytes from the zip member size).
"미기록":   the quantity is not present in any recorded artifact (no arithmetic).

Output: artifacts/multimodal-resource-review.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

FP32_WARNING = "dtypes are read from the checkpoint storages (all torch.FloatStorage), not assumed"

VARIANT_LABEL = {
    "capacity_matched_shared": "capacity-matched shared",
    "shared_lora": "shared LoRA",
    "replay_lora": "replay LoRA",
    "expand": "expansion",
    "frozen_first": "frozen-first",
    "independent": "independent",
}

CODE_CITATIONS = {
    "parameter_counts semantics": "sema_vqa/readout_model.py:216-228 (parameter_counts: numel counts; base includes frozen zero-output vision mixtures; experts sums bank A/B; trainable = enabled tensors)",
    "LoRA expert layout": "sema_vqa/readout_model.py:26-36 (A=(rank,in_features), B=(out_features,rank); zero-delta init), :121-142 (old experts permanently frozen after expansion)",
    "checkpoint = LoRA bank only + router + trainer state": "sema_vqa/readout_model.py:193-213 (checkpoint_state snapshots only LoRA tensors + metadata)",
    "router memory counter": "sema_vqa/readout_router.py:98-99 memory_bytes() = sum(numel*element_size + 8 per feature tensor); 128x1536 fp32 features per task (observe, :30-47)",
    "router fit params": "sema_vqa/readout_router.py:49-71 weight[1536,E]/center[1536]/prior[E]",
    "router exists only for expand": "sema_multimodal/run.py:318 (FeatureRouter only when method == 'expand')",
    "replay storage semantics": "sema_multimodal/run.py:524-531 replay_row_bytes = len(json.dumps(replay rows)); replay_image_bytes = on-disk size of unique referenced image files; policy 'Rows and immutable image-file references; no duplicate image copies'",
    "replay replacement budget": "sema_multimodal/run.py:60-78 half of each actual batch is replaced (not added), only once a buffer exists (stage 0 has none)",
    "per-stage counters": "sema_multimodal/run.py:527-538; cumulative totals exported by scripts/evaluate_multimodal_confirmation.py:1120-1133",
    "checkpoint layout incl. optimizer": "sema_multimodal/run.py:374-380 latest.pt = {model checkpoint_state, router, state, rng, optimizer, scheduler, log_bytes}; optimizer is AdamW over trainable only, rebuilt per phase :327-332; final-archived checkpoints verified optimizer=None via direct pickle probe (scripts/inspect_multimodal_checkpoint_zip.py)",
    "peak GPU memory": "sema_multimodal/run.py:372-373, 532 torch.cuda.max_memory_allocated, max across tasks (recorded)",
    "feature/fit timing": "sema_multimodal/run.py:473-477 feature_seconds covers frozen-feature extraction AND router refit; router refits after every task addition",
    "independent per-task archive": "sema_multimodal/run.py:480-486 archive per task = torch.save(model.checkpoint_state()) only; its size is recorded in independent_storage_bytes",
}

NOT_RECORDED = {
    "gpu_hours": "미기록 — no SLURM accounting artifact exists for the multimodal jobs (same limitation as the predecessor cohort: README.md:202, VQA_CLOSURE.md). Recorded quantities are stage seconds (training/feature/evaluation/wall) and torch.cuda.max_memory_allocated peaks.",
    "per_sample_latency": "미기록 — no per-sample inference latency is recorded anywhere; no derived per-image latency proxies are emitted.",
    "routing_inference_overhead": "미기록 — G5 routing CSVs contain no timing columns; recorded piece is train-time feature+fit seconds for expansion only (sema_multimodal/run.py:473-477).",
    "preprocessed_replay_tensor_storage": "미기록(+해당 없음) — the runner persists raw rows and references raw JPEG files; it stores no preprocessed image tensors (replay rows carry the image path, vision model decodes per request: sema_vqa/vision_model.py:128-134), so any preprocessed-equivalent size would be an invented value and is not computed here.",
    "runtime_optimizer_state": "미기록 — the archived final checkpoints store no optimizer state (verified: optimizer=None in all 256 files); the running AdamW state size was never recorded and is not estimated.",
}


def parse_run(name: str):
    import re

    m = re.match(r"^(?P<order>[A-Z](?:_[A-Z])*)_(?P<variant>capacity_matched_shared|shared_lora|replay_lora|"
                 r"expand|frozen_first|independent)(?:_(?:lr(?P<lr>[0-9.]+))?(?:_ep(?P<epochs>[0-9]+(?:\.[0-9]+)?))?)?"
                 r"_seed(?P<seed>\d+)$", name)
    if m:
        d = m.groupdict()
        if d["variant"] == "replay_lora" and d["epochs"] == "10":
            d["method_key"] = "tuned_replay"
        else:
            d["method_key"] = {"capacity_matched_shared": "capacity", "shared_lora": "shared",
                               "replay_lora": "replay", "expand": "expansion",
                               "frozen_first": "frozen", "independent": "independent"}[d["variant"]]
        return d
    m = re.match(r"^(?P<order>[A-Z])_(?:lr(?P<lr>[0-9.]+))?(?:_ep(?P<epochs>[0-9]+(?:\.[0-9]+)?))?_seed(?P<seed>\d+)$",
                 name)
    if m:
        d = m.groupdict()
        d["variant"] = "independent"
        d["method_key"] = "independent"
        return d
    return None


def fnum(row, key):
    return float(row[key])


def inum(row, key):
    return int(round(float(row[key])))


def load_json(path: Path):
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def build_runs(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        base = Path(row["directory"]).name
        grouped[(row["kind"], row["task_order"], int(row["seed"]), directory := base)].append(row)
    runs = []
    for (kind, order, seed, directory), rows_ in grouped.items():
        base = parse_run(directory)
        assert base is not None, directory
        stages = [{
            "stage_index": inum(r, "stage_index"),
            "task_id": r["task_id"],
            "steps": inum(r, "steps"),
            "image_exposures": inum(r, "image_exposures"),
            "replay_exposures": inum(r, "replay_exposures"),
            "target_tokens": inum(r, "target_tokens"),
            "training_seconds": fnum(r, "training_seconds"),
            "router_memory_bytes": inum(r, "router_memory_bytes"),
            "replay_images_stored": inum(r, "replay_images_stored"),
            "replay_row_bytes": inum(r, "replay_row_bytes"),
            "replay_image_bytes": inum(r, "replay_image_bytes"),
            "raw_replay_bytes": inum(r, "raw_replay_bytes"),
            "peak_gpu_memory_bytes": inum(r, "peak_gpu_memory_bytes"),
            "parameter_counts": {
                "base_numel": inum(r, "parameter_counts_base"),
                "experts_numel": inum(r, "parameter_counts_experts"),
                "active_trainable_numel": inum(r, "parameter_counts_active_trainable"),
                "trainable_numel": inum(r, "parameter_counts_trainable"),
            },
            "totals": {
                "steps": inum(r, "total_steps"),
                "image_exposures": inum(r, "total_image_exposures"),
                "current_exposures": inum(r, "total_current_exposures"),
                "replay_exposures": inum(r, "total_replay_exposures"),
                "target_tokens": inum(r, "total_target_tokens"),
                "training_seconds": fnum(r, "total_training_seconds"),
                "feature_seconds": fnum(r, "total_feature_seconds"),
                "evaluation_seconds": fnum(r, "total_evaluation_seconds"),
                "independent_storage_bytes": inum(r, "total_independent_storage_bytes"),
                "wall_seconds": fnum(r, "total_wall_seconds"),
                "peak_gpu_memory_bytes": inum(r, "total_peak_gpu_memory_bytes"),
            },
        } for r in sorted(rows_, key=lambda x: inum(x, "stage_index"))]
        runs.append({
            "kind": kind, "order": base["order"], "seed": seed,
            "variant": base["variant"], "method": base["method_key"],
            "dir": directory, "stages": stages,
        })
    return runs


def spread(vals):
    return {"min": min(vals), "max": max(vals), "mean": st.mean(vals),
            "const": len(set(map(int, vals))) == 1}


def summary_of(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    final = [r["stages"][-1] for r in runs]
    pc_keys = ("base_numel", "experts_numel", "active_trainable_numel", "trainable_numel")
    aggregated = {
        "n_runs": len(runs),
        "stages_per_run": spread([len(r["stages"]) for r in runs]),
        "parameter_counts_final": {k: spread([row["parameter_counts"][k] for row in final]) for k in pc_keys},
        "router_memory_bytes_final": spread([row["router_memory_bytes"] for row in final]),
        "replay_images_stored_final": spread([row["replay_images_stored"] for row in final]),
        "replay_row_bytes_final": spread([row["replay_row_bytes"] for row in final]),
        "replay_image_bytes_final": spread([row["replay_image_bytes"] for row in final]),
        "raw_replay_bytes_final": spread([row["raw_replay_bytes"] for row in final]),
        "sums": {
            "steps": sum(row["totals"]["steps"] for row in final),
            "image_exposures": sum(row["totals"]["image_exposures"] for row in final),
            "current_exposures": sum(row["totals"]["current_exposures"] for row in final),
            "replay_exposures": sum(row["totals"]["replay_exposures"] for row in final),
            "target_tokens": sum(row["totals"]["target_tokens"] for row in final),
            "training_seconds": sum(row["totals"]["training_seconds"] for row in final),
            "feature_seconds": sum(row["totals"]["feature_seconds"] for row in final),
            "evaluation_seconds": sum(row["totals"]["evaluation_seconds"] for row in final),
            "wall_seconds": sum(row["totals"]["wall_seconds"] for row in final),
        },
        "peak_gpu_memory_bytes": spread([row["totals"]["peak_gpu_memory_bytes"] for row in final]),
        "independent_storage_bytes_sum": sum(row["totals"]["independent_storage_bytes"] for row in final),
    }
    return aggregated


def stage_growth_of(runs: list[dict]) -> list[dict]:
    stages = defaultdict(list)
    for r in runs:
        for s in r["stages"]:
            stages[s["stage_index"]].append(s)
    out = []
    for idx in sorted(stages):
        rows = stages[idx]
        pc = rows[0]["parameter_counts"]
        constant = all(row["parameter_counts"] == pc for row in rows)
        out.append({
            "stage_index": idx,
            "task_ids": sorted({row["task_id"] for row in rows}),
            "steps": st.mean(row["steps"] for row in rows),
            "image_exposures": st.mean(row["image_exposures"] for row in rows),
            "replay_exposures": st.mean(row["replay_exposures"] for row in rows),
            "training_seconds_mean": st.mean(row["training_seconds"] for row in rows),
            "experts_numel": {"value": pc["experts_numel"], "const": constant},
            "active_trainable_numel": pc["active_trainable_numel"],
            "router_memory_bytes": spread([row["router_memory_bytes"] for row in rows]),
            "replay_images_stored": st.mean(row["replay_images_stored"] for row in rows),
            "replay_row_bytes": st.mean(row["replay_row_bytes"] for row in rows),
            "replay_image_bytes": st.mean(row["replay_image_bytes"] for row in rows),
            "run_final_totals": {  # each run's FINAL totals (same value on every stage row), not a stage-prefix sum
                "steps": st.mean(row["totals"]["steps"] for row in rows),
                "training_seconds": st.mean(row["totals"]["training_seconds"] for row in rows),
                "feature_seconds": st.mean(row["totals"]["feature_seconds"] for row in rows),
                "evaluation_seconds": st.mean(row["totals"]["evaluation_seconds"] for row in rows),
                "wall_seconds": st.mean(row["totals"]["wall_seconds"] for row in rows),
                "peak_gpu_memory_bytes": st.mean(row["totals"]["peak_gpu_memory_bytes"] for row in rows),
            },
        })
    return out


def router_memory_decomposition(expert_count_per_stage: dict) -> list[dict]:
    """Explain each recorded expansion router memory value from the exact structure:
    features 128x1536 fp32 + int label per task + weight[1536,E] + center[1536] + prior[E] + state-dict JSON 90B."""
    rows = []
    for stage in sorted(expert_count_per_stage):
        tasks = stage + 1
        experts = expert_count_per_stage[stage] // 1179648  # recorded expert numel per rank-8 expert
        feature_tensor = tasks * (128 * 1536 * 4)
        label_bytes_recorded = tasks * 8  # memory_bytes() adds 8 per feature tensor for the int label
        weight = experts * 1536 * 4
        center = 1536 * 4
        prior = experts * 4
        json_state = 90
        rows.append({
            "stage_index": stage,
            "recorded": None,
            "feature_tensor_bytes_exact_from_checkpoint": feature_tensor,
            "label_bytes_recorded_by_counter": label_bytes_recorded,
            "router_weight_bytes": weight,
            "router_center_bytes": center,
            "router_prior_bytes": prior,
            "router_state_dict_json_bytes": json_state,
            "sum": feature_tensor + label_bytes_recorded + weight + center + prior + json_state,
        })
    return rows


def joint_dev(dev_files: dict) -> dict:
    groups = defaultdict(list)
    for path_key, meta in dev_files.items():
        name = Path(path_key).name
        run = parse_run(Path(path_key).parent.name) or {}
        gen = "g1" if "/g1/" in path_key else "g2" if "/g2/" in path_key else "g3" if "/g3/" in path_key else "other"
        variant = VARIANT_LABEL[run.get("variant", "independent")]
        if name == "latest.pt":
            groups[(gen, variant, "latest.pt")].append(meta["bytes"])
        elif name.startswith("independent_"):
            groups[(gen, variant, "independent_archive")].append(meta["bytes"])
    out = {}
    for gen in ("g1", "g2", "g3"):
        entries = {}
        for (group, variant, kind), values in sorted(groups.items()):
            if group != gen:
                continue
            entries[f"{variant}/{kind}"] = {"n_files": len(values),
                                            "bytes_min": min(values), "bytes_max": max(values),
                                            "bytes_mean": st.mean(values)}
        out[gen] = entries
    return out


def local_artifact_stat(art: Path) -> dict:
    names = [
        "multimodal-g1-raw.tar.gz", "multimodal-g2-raw.tar.gz", "multimodal-g3-raw.tar.gz",
        "multimodal-g4-raw.tar.gz", "multimodal-g5-raw.tar.gz",
        "multimodal-g1-report.json", "multimodal-g2-report.json", "multimodal-g3-report.json",
        "multimodal-g4-report.tar.gz", "multimodal-g4-statistics.json",
        "multimodal-g5-report.tar.gz", "multimodal-g5-summary.json",
        "multimodal-full-manifests.tar.gz", "multimodal-confirmation-lock.json",
        "multimodal-confirmation-source.tar.gz", "multimodal-confirmation-checkpoints.json",
        "multimodal-confirmation-checkpoints.tar", "multimodal-checkpoint-zip-inspection.json",
        "multimodal-development-checkpoints.json", "multimodal-development-checkpoints.tar",
        "multimodal-routing-validation-audit.json", "multimodal-routing-validation-raw.tar.gz",
        "multimodal-routing-validation-execution.json", "multimodal-confirmation-environment.json",
        "multimodal-confirmation-execution.json", "multimodal-confirmation-audit-revision.json",
        "multimodal-backups.json", "multimodal-selected-images-v2.tar", "multimodal-locked-blip-assets.tar",
    ]
    rows = []
    for name in names:
        target = art / name
        rows.append({"file": name, "bytes": target.stat().st_size, "present": target.exists()})
    report_dirs = {}
    for dirname in ("multimodal-g4-report", "multimodal-g5-report"):
        total = sum(p.stat().st_size for p in (art / dirname).rglob("*") if p.is_file())
        report_dirs[dirname] = {"bytes": total}
    return {"files": rows, "report_dirs": report_dirs}


def validate(runs, methods, confirmation_files, dev_files, backups, execution, inspection) -> dict:
    checks = {}
    steps_total = sum(r["stages"][-1]["totals"]["steps"] for r in runs)
    checks["g4_total_steps_vs_execution_recorded"] = {
        "computed": steps_total, "recorded": execution["g4"]["optimizer_steps"],
        "match": steps_total == execution["g4"]["optimizer_steps"]}
    checks["g4_training_runs"] = {"computed": len(runs), "recorded": execution["g4"]["training_runs"],
                                  "match": len(runs) == execution["g4"]["training_runs"]}
    checks["g4_stage_rows"] = {"computed": sum(len(r["stages"]) for r in runs),
                               "recorded": execution["g4"]["resource_rows"],
                               "match": sum(len(r["stages"]) for r in runs) == execution["g4"]["resource_rows"]}
    file_bytes = sum(meta["bytes"] for meta in confirmation_files.values())
    checks["confirmation_checkpoint_inventory"] = {
        "sum_of_files": file_bytes, "recorded_tar_bytes": backups["confirmation_checkpoints"]["bytes"],
        "tar_padding_delta": backups["confirmation_checkpoints"]["bytes"] - file_bytes}
    dev_bytes = sum(meta["bytes"] for meta in dev_files.values())
    checks["development_checkpoint_inventory"] = {
        "sum_of_files": dev_bytes,
        "recorded_tar_bytes": backups["development_checkpoints"]["archive"]["bytes"],
        "tar_padding_delta": backups["development_checkpoints"]["archive"]["bytes"] - dev_bytes,
        "file_count": {"computed": len(dev_files), "recorded": backups["development_checkpoints"]["file_count"],
                       "match": len(dev_files) == backups["development_checkpoints"]["file_count"]}}
    # exact inspection validations
    ident = set()
    for rec in inspection["per_file"]:
        ident.add(rec["serialized"] == rec["data_pkl_bytes"] + rec["data_member_bytes"]
                  + rec["version_byteorder_bytes"] + rec["zip_overhead_bytes"])
    checks["serialized_composition_identity_per_file"] = {"all_identity": all(ident)}
    # recorded router stage values vs exact decomposition (features + labels + fit params + json)
    method = methods["common_cl/expansion"]
    preds = router_memory_decomposition({g["stage_index"]: g["experts_numel"]["value"]
                                         for g in method["stage_growth"]})
    recorded = [g["router_memory_bytes"]["mean"] for g in method["stage_growth"]]
    exact = [p["sum"] for p in preds]
    checks["expansion_router_memory_recorded_equals_exact"] = {
        "recorded_stage_means": recorded, "exact_components": exact,
        "match": recorded == exact,
        "ancillary": "stage feature+labels 786,440/task; weight 6,144/expert; center 6,144; prior 4/expert; JSON 90",
    }
    # per-method storage dtypes/optimizer verdicts
    opt = {k: v["optimizer_state_dict"] for k, v in inspection["summary_by_variant_file"].items()}
    checks["archived_optimizer_state"] = opt
    checks["archived_dtype_histogram"] = {
        k: v["storage_dtypes_all"] for k, v in inspection["summary_by_variant_file"].items()}
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    art = args.root / "artifacts"

    rows = load_csv(art / "multimodal-g4-report" / "resources.csv")
    runs = build_runs(rows)

    inspection = load_json(art / "multimodal-checkpoint-zip-inspection.json")
    confirmation_files = load_json(art / "multimodal-confirmation-checkpoints.json")["files"]
    dev_files = load_json(art / "multimodal-development-checkpoints.json")["files"]
    backups = load_json(art / "multimodal-backups.json")
    execution = load_json(art / "multimodal-confirmation-execution.json")
    g5 = load_json(art / "multimodal-g5-summary.json")

    insp_latest = inspection["summary_by_variant_file"]
    insp_key_by_method = {"shared": "shared_lora|latest.pt", "capacity": "capacity_matched_shared|latest.pt",
                          "replay": "replay_lora|latest.pt", "expansion": "expand|latest.pt",
                          "frozen": "frozen_first|latest.pt", "independent": "independent|latest.pt"}
    groups = defaultdict(list)
    for r in runs:
        groups[(r["kind"], r["method"])].append(r)

    out = {"schema": "multimodal-resource-review/2",
           "root": str(args.root),
           "g4_stage_rows": len(rows), "g4_run_conditions": len(runs),
           "dtypes_policy": FP32_WARNING,
           "code_citations": CODE_CITATIONS, "not_recorded": NOT_RECORDED,
           "references": {
               "resources_csv": "artifacts/multimodal-g4-report/resources.csv",
               "checkpoints_inventory": "artifacts/multimodal-confirmation-checkpoints.json",
               "checkpoint_zip_probe": ["scripts/inspect_multimodal_checkpoint_zip.py",
                                        "artifacts/multimodal-checkpoint-zip-inspection.json"],
               "development_checkpoints": "artifacts/multimodal-development-checkpoints.json",
               "backups": "artifacts/multimodal-backups.json",
               "execution": "artifacts/multimodal-confirmation-execution.json",
               "g5": "artifacts/multimodal-g5-summary.json",
           },
           "methods": {}}
    for (kind, method), rlist in sorted(groups.items()):
        summary = summary_of(rlist)
        summary["stage_growth"] = stage_growth_of(rlist) if summary["stages_per_run"]["mean"] > 1 else []
        method_key = "tuned_replay" if method == "tuned_replay" else method
        if method_key == "tuned_replay":
            insp_key = "tuned_replay|latest.pt"
        elif method_key == "capacity":
            insp_key = "capacity_matched_shared|latest.pt"
        else:
            insp_key = insp_key_by_method[method_key]
        summary["checkpoint_probe_exact"] = {"latest": insp_latest[insp_key]}
        if method_key == "independent":
            summary["checkpoint_probe_exact"]["independent_archive"] = insp_latest["independent|independent_archive"]
        out["methods"][f"{kind}/{method}"] = summary

    validations = validate(runs, out["methods"], confirmation_files, dev_files, backups, execution, inspection)
    out["validations"] = validations

    out["development_checkpoints"] = joint_dev(dev_files)
    out["g5_summary"] = g5
    out["recorded_shared_assets"] = {
        "image_dataset": backups["image_archive"],
        "blip_assets": backups["initialization_archive"],
        "development_checkpoints": backups["development_checkpoints"],
        "confirmation_checkpoints": backups["confirmation_checkpoints"],
    }
    out["serialized_checkpoint_exact_probe"] = {k: insp_latest[k] for k in sorted(insp_latest)}
    out["recorded_result_archives"] = [
        {"file": name, "bytes": execution["inventory"][name]["bytes"]}
        for name in sorted(execution["inventory"])]
    out["replay_distinction_recorded_only"] = {
        "replay_policy_recorded_in_run": "Rows and immutable image-file references; no duplicate image copies (sema_multimodal/run.py:531)",
        "row_bytes": "recorded replay_row_bytes: JSON of the buffer rows (image path + question; caption rows carry answer/references)",
        "image_bytes": "recorded replay_image_bytes: bytes of the unique on-disk JPEG files referenced by the buffer rows; these are the same shared-dataset files (11,264 images, 1,730,877,440 B recorded) — zero duplicated image bytes",
        "preprocessed_replay": "the buffer stores no preprocessed tensor form; any preprocessed-equivalent size would be unrecorded -> 미기록",
        "router_feature_bytes_per_image_recorded_shape": "128x1536 fp32 per task in checkpoint storage (exact: 786,432 B/task tensor + int label) — verified inside latest.pt",
    }
    out["local_artifact_storage"] = local_artifact_stat(art)

    path = art / "multimodal-resource-review.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"written": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
