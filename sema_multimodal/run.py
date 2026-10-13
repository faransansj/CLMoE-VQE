"""Train heterogeneous frozen-BLIP readouts without writing archived VQA outputs.

Example (validation only): python -m sema_multimodal.run --manifest MANIFEST
--assets ASSETS --image-root IMAGES --method independent --task-ids C --output
artifacts/multimodal_runs/g1-C. Evaluation units remain task-specific; this
runner deliberately never computes an average of raw accuracy and CIDEr-D.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

METHODS = ("shared_lora", "capacity_matched_shared", "replay_lora", "expand", "frozen_first", "independent")
ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    files = [*sorted((ROOT / "sema_multimodal").glob("*.py")),
             *(ROOT / "sema_vqa" / name for name in
               ("readout_model.py", "readout_router.py", "vision_model.py", "expansion.py", "metrics.py")),
             *(ROOT / "third_party" / "BLIP" / "models" / name for name in
               ("med.py", "vit.py")), ROOT / "third_party" / "BLIP" / "configs" / "med_config.json"]
    return {str(path.relative_to(ROOT)): digest_file(path) for path in files}


def safe_output(path, smoke=False):
    path = Path(path).resolve()
    forbidden = ("readout", "vision", "legacy", "sema_runs", "vqa_runs")
    archived = {"runs", "report", "transition_smoke", "manifests"}
    if any(part.casefold().startswith(forbidden) or part.casefold() in archived for part in path.parts):
        raise ValueError("Archived VQA/vision/readout output namespaces are read-only")
    runs = (ROOT / "artifacts" / "multimodal_runs").resolve()
    if path == runs or runs in path.parents:
        return path
    if smoke and any("multimodal" in part.casefold() for part in path.parts):
        # An explicit named smoke directory is allowed, never a source/assets root.
        if path == ROOT or path in ROOT.parents or (ROOT in path.parents and "artifacts" not in path.parts):
            raise ValueError("Smoke output must not write into source directories")
        return path
    raise ValueError("Output must be under artifacts/multimodal_runs; --smoke-output permits an explicit multimodal smoke directory")


def epoch_batches(rows, replay, *, seed, task, epoch, batch_size):
    """Deterministic replacement replay, including the final partial batch.

    The current dataset defines the phase's step/image budget. Half of each
    actual batch (floor for odd sizes) is replaced, not added or discarded.
    Reconstructing this pure plan leaves global RNG untouched on resume.
    """
    rng = random.Random(seed + task * 1000 + epoch)
    order = list(rows)
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        count = len(batch) // 2 if replay else 0
        if count:
            batch[-count:] = rng.choices(replay, k=count)
        batches.append((batch, len(batch) - count, count))
    return batches


def optimizer_step(model, batch, image_root, micro_batch_size, optimizer):
    """Exactly one effective-batch example mean, including unequal microbatches."""
    import torch
    optimizer.zero_grad(set_to_none=True)
    total, tokens = 0.0, 0
    for start in range(0, len(batch), micro_batch_size):
        micro = batch[start:start + micro_batch_size]
        losses = model.loss_per_example(micro, image_root)
        (losses.sum() / len(batch)).backward()
        total += float(losses.detach().sum())
        tokens += int(model.target_token_counts(micro).sum())
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    if any(parameter.grad is not None for parameter in frozen):
        raise RuntimeError("Frozen backbone/old expert acquired a gradient")
    torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
    optimizer.step()
    return total / len(batch), tokens


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def emit(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def router_state(router):
    if router is None:
        return None
    state = router.state_dict()
    # The legacy router persists memory, not fitted weights. This new schema
    # snapshots both so resume does not introduce an extra numerical refit.
    state.update({key: None if getattr(router, key) is None else getattr(router, key).detach().cpu().clone()
                  for key in ("weight", "center", "prior")})
    return state


def restore_router(state):
    if state is None:
        return None
    from sema_vqa.readout_router import FeatureRouter
    router = FeatureRouter(**{key: state[key] for key in ("memory_per_task", "ridge", "modality", "seed")})
    router.memory, router.num_experts = state["memory"], state["num_experts"]
    for key in ("weight", "center", "prior"):
        setattr(router, key, state[key])
    return router


def features(model, rows, args):
    import torch
    from .data import public_rows
    view = public_rows(rows)
    return torch.cat([model.routing_features(view[start:start + args.micro_batch_size], args.image_root).cpu()
                      for start in range(0, len(view), args.micro_batch_size)])


def routed_predictions(model, router, rows, cached, args, forced_expert=None):
    import torch
    from .data import public_rows
    view = public_rows(rows)
    model.eval()
    if forced_expert is not None:
        assignments = torch.full((len(view),), forced_expert, dtype=torch.long)
    elif router is not None and router.memory:
        assignments = router.predict(cached).cpu()
    else:
        assignments = torch.full((len(view),), 0 if model.num_experts else -1, dtype=torch.long)
    predictions, diagnostics = [None] * len(view), [None] * len(view)
    for expert in assignments.unique(sorted=True).tolist():
        model.set_expert(None if expert == -1 else expert)
        indices = (assignments == expert).nonzero(as_tuple=True)[0].tolist()
        for start in range(0, len(indices), args.micro_batch_size):
            group = indices[start:start + args.micro_batch_size]
            answers, information = model.generate([view[index] for index in group], args.image_root)
            if len(answers) != len(group) or len(information) != len(group):
                raise ValueError("Generation lost an input row")
            for index, answer, info in zip(group, answers, information):
                predictions[index], diagnostics[index] = answer, info
    if any(value is None for value in predictions):
        raise ValueError("Routing lost an input row")
    return predictions, assignments.tolist(), diagnostics


def evaluate(model, router, task, scorer, cached, args, path, forced_expert=None, shuffled=False):
    rows = task[args.eval_split][:args.eval_limit or None]
    model_rows = rows
    if shuffled:
        images = [row["image"] for row in rows]
        # Fixed derangement when N>1, never rescore against changed references.
        model_rows = [dict(row, image=images[(index + 1) % len(images)]) for index, row in enumerate(rows)]
    predictions, assignments, diagnostics = routed_predictions(model, router, model_rows, cached, args, forced_expert)
    scoring = scorer.score(predictions, rows)
    if not math.isfinite(scoring["score"]):
        raise ValueError("Nonfinite task score")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row, prediction, expert, generation in zip(rows, predictions, assignments, diagnostics):
            stream.write(json.dumps({"id": row["id"], "prediction": prediction,
                                     "expert": expert, "generation": generation}) + "\n")
    temporary.replace(path)
    count = len(rows)
    generation = {key: sum(info[key] for info in diagnostics) / count
                  for key in ("generated_tokens", "eos", "empty", "repeated_trigram", "limit_reached")}
    frequencies = Counter(predictions)
    return {"task_id": task["task_id"], "family": task["family"], "metric": task["metric"],
            "scoring": scoring, "generation": generation, "prediction_counts": dict(frequencies),
            "most_frequent_output_fraction": max(frequencies.values()) / count,
            "route_counts": dict(Counter(assignments)), "predictions_file": path.name}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ("manifest", "assets", "image-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multimodal_runs/default")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--task-ids", nargs="+", help="Configured task IDs in stream order; independent G1 accepts one ID")
    parser.add_argument("--eval-split", choices=("validation", "eval"), default="validation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=8, help="Per-expert rank; capacity baseline multiplies by selected task count")
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--batch-size", type=int, default=16, help="Effective batch, replay replaces half of its actual rows")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--router-modality", choices=("joint", "image", "question"), default="joint")
    parser.add_argument("--router-memory", type=int, default=128)
    parser.add_argument("--router-ridge", type=float, default=1.0)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--oracle-diagnostic", action="store_true", help="Separate task-ID diagnostic, never native routing")
    parser.add_argument("--image-shuffled-diagnostic", action="store_true", help="G1 validation image derangement")
    parser.add_argument("--smoke-output", action="store_true", help="Allow explicit safe multimodal smoke namespace and tiny subsets")
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke-only train prefix; scorer still locks full train reference corpus")
    parser.add_argument("--eval-limit", type=int, default=0, help="Smoke-only validation prefix")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="Commit then stop at this total step count; resume with 0")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_started = time.perf_counter()
    if min(args.epochs, args.rank, args.batch_size, args.micro_batch_size, args.router_memory, args.image_size) < 1:
        parser.error("Epochs, rank, image and batch/memory sizes must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.alpha, args.router_ridge)):
        parser.error("Learning rate, alpha and ridge must be finite and positive")
    if min(args.replay_per_task, args.train_limit, args.eval_limit, args.stop_after_steps) < 0:
        parser.error("Budgets/limits must be nonnegative")
    if (args.train_limit or args.eval_limit) and (not args.smoke_output or args.eval_split != "validation"):
        parser.error("Tiny subsets are explicit validation-only smoke, not research results")
    if args.image_shuffled_diagnostic and args.eval_split != "validation":
        parser.error("G1 shuffled diagnostics may not inspect confirmation eval")
    output = safe_output(args.output, args.smoke_output)
    from .data import validate_manifest
    from .metrics import TaskScorer
    raw = args.manifest.read_bytes()
    manifest = validate_manifest(json.loads(raw), args.image_root)
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    task_order = args.task_ids or list(by_id)
    if len(set(task_order)) != len(task_order) or any(task_id not in by_id for task_id in task_order):
        parser.error("Task IDs must be a unique ordered selection of manifest tasks")
    if args.eval_split == "eval" and args.method != "independent" and set(task_order) != set(by_id):
        parser.error("Confirmation CL must cover the complete declared task set")
    if args.eval_split == "eval" and manifest.get("confirmation_eligible") is False:
        parser.error("Development manifests are not confirmation eligible")
    tasks = [by_id[task_id] for task_id in task_order]
    for task in tasks:
        if not task["train"] or not task[args.eval_split]:
            raise ValueError("Every selected task needs nonempty train and scoring rows")
        if not task["task_id"].replace("-", "").replace("_", "").isalnum():
            raise ValueError("Task IDs must be safe filename components")
    scorers = {task["task_id"]: TaskScorer(task) for task in tasks}
    assets = args.assets.resolve()
    asset_files = [assets / "model_base_capfilt_large.pth", assets / "sources.json", *sorted((assets / "tokenizer").rglob("*"))]
    asset_hashes = {str(path.relative_to(assets)): digest_file(path) for path in asset_files if path.is_file()}
    if "model_base_capfilt_large.pth" not in asset_hashes or not any(key.startswith("tokenizer/") for key in asset_hashes):
        raise ValueError("Real pure-pretraining BLIP weights and tokenizer assets are required")
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("output", "stop_after_steps")}
    settings.update(schema_version=1, protocol=manifest["protocol"], task_order=task_order,
                    manifest_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=source_hashes(),
                    assets_sha256=asset_hashes,
                    tokenizer_sha256={key: value for key, value in asset_hashes.items() if key.startswith("tokenizer/")},
                    scorer_locks={task_id: scorer.lock for task_id, scorer in scorers.items()},
                    optimizer={"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                               "gradient_clip_norm": 1.0, "scheduler": "epoch_cosine", "precision": "cuda_bfloat16_or_cpu_float32"},
                    generation={"strategy": "cached_greedy", "vqa_max_new_tokens": 32, "caption_max_new_tokens": 64},
                    loss="BLIP smoothing 0.1; shifted target token mean per row, then effective-batch example mean")
    import numpy as np
    import torch
    settings.update(torch=str(torch.__version__), numpy=str(np.__version__),
                    cuda=torch.version.cuda, deterministic_algorithms=True, tf32=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # cuBLAS needs this before the first CUDA matrix operation (model construction).
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)
    config_path, checkpoint_path, log = output / "config.json", output / "latest.pt", output / "events.jsonl"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Refusing changed config/source/assets/manifest/scorer on resume")
    atomic_json(config_path, settings)
    from .model import MultimodalReadout
    from sema_vqa.readout_router import FeatureRouter
    multiplier = len(tasks) if args.method == "capacity_matched_shared" else 1

    def new_model():
        return MultimodalReadout(args.assets, image_size=args.image_size, rank=args.rank * multiplier,
                                 alpha=args.alpha * multiplier, device=args.device)

    model = new_model()
    router = FeatureRouter(args.router_memory, args.router_ridge, args.router_modality, args.seed) if args.method == "expand" else None
    state = {"task_index": 0, "stage": "prepare", "epoch": 0, "batch_cursor": 0,
             "matrix": [], "oracle_matrix": [], "pre_scores": [], "assignments": [], "replay": [],
             "stage_evaluations": [], "stage_resources": [], "steps": 0, "image_exposures": 0,
             "target_tokens": 0, "current_exposures": 0, "replay_exposures": 0,
             "training_seconds": 0.0, "feature_seconds": 0.0, "evaluation_seconds": 0.0,
             "independent_storage_bytes": 0, "wall_seconds": 0.0, "peak_gpu_memory_bytes": 0}
    optimizer, scheduler = None, None

    def make_optimizer():
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable new/current expert")
        opt = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["schema_version"] != 1 or saved["settings"] != settings:
            raise ValueError("Checkpoint schema/scientific contract changed")
        model.load_checkpoint_state(saved["model"])
        router, state = restore_router(saved["router"]), saved["state"]
        if saved["optimizer"] is not None:
            optimizer, scheduler = make_optimizer()
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
        # Discard events written after the last committed transaction.
        if log.exists():
            with log.open("r+b") as stream:
                stream.truncate(saved["log_bytes"])
        restore_rng(saved["rng"])
    elif log.exists():
        raise ValueError("Events without a checkpoint cannot be silently resumed")
    wall_before = state["wall_seconds"]

    def result():
        from .metrics import continual_metrics
        return {"schema_version": 1, "completed": len(state["matrix"]), "task_order": task_order,
                "task_metadata": [{key: task[key] for key in ("task_id", "family", "metric")} for task in tasks],
                "matrix": state["matrix"], "oracle_matrix": state["oracle_matrix"],
                "pre_task_scores": state["pre_scores"],
                "pre_to_post_gain": [row[index] - state["pre_scores"][index] for index, row in enumerate(state["matrix"])],
                "task_expert_assignments": state["assignments"],
                "stage_evaluations": state["stage_evaluations"], "stage_resources": state["stage_resources"],
                "resources": {key: state[key] for key in ("steps", "image_exposures", "target_tokens", "current_exposures",
                                                          "replay_exposures", "training_seconds", "feature_seconds", "evaluation_seconds",
                                                          "independent_storage_bytes", "wall_seconds", "peak_gpu_memory_bytes")},
                "metrics": continual_metrics(state["matrix"], tasks, pre_task_scores=state["pre_scores"])
                           if len(state["matrix"]) == len(tasks) else None,
                "status": "complete" if state["stage"] == "complete" else "interrupted",
                "raw_units_not_averaged": True}

    def commit():
        state["wall_seconds"] = wall_before + time.perf_counter() - run_started
        if model.device.type == "cuda":
            state["peak_gpu_memory_bytes"] = max(state["peak_gpu_memory_bytes"], torch.cuda.max_memory_allocated())
        saved = {"schema_version": 1, "settings": settings, "model": model.checkpoint_state(),
                 "router": router_state(router), "state": state, "rng": capture_rng(),
                 "optimizer": optimizer.state_dict() if optimizer else None,
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "log_bytes": log.stat().st_size if log.exists() else 0}
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(saved, temporary)
        temporary.replace(checkpoint_path)
        atomic_json(output / "results.json", result())

    eval_cache = {}
    while state["task_index"] < len(tasks):
        index = state["task_index"]
        task = tasks[index]
        rows = task["train"][:args.train_limit or None]
        if state["stage"] == "prepare":
            if args.method == "independent" and index:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                torch.manual_seed(args.seed)
                model = new_model()
            started = time.perf_counter()
            cached = None
            if router is not None:
                cached = features(model, task[args.eval_split][:args.eval_limit or None], args)
                eval_cache[index] = cached
            state["feature_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            before = evaluate(model, router, task, scorers[task["task_id"]], cached, args,
                              output / f"predictions_before_{index:02d}_{task['task_id']}.jsonl")
            state["evaluation_seconds"] += time.perf_counter() - started
            state["pre_scores"].append(before["scoring"]["score"])
            phase_seed = args.seed if args.method == "independent" else args.seed + index * 10007
            torch.manual_seed(phase_seed)
            torch.cuda.manual_seed_all(phase_seed)
            training = args.method != "frozen_first" or index == 0
            if index == 0 or args.method in ("expand", "independent"):
                expert = model.add_expert()
            else:
                expert = 0
            state["assignments"].append(expert)
            state["phase_steps_start"] = state["steps"]
            state["phase_images_start"] = state["image_exposures"]
            state["phase_tokens_start"] = state["target_tokens"]
            state["phase_current_start"] = state["current_exposures"]
            state["phase_replay_start"] = state["replay_exposures"]
            state["phase_training_seconds_start"] = state["training_seconds"]
            state["epoch"], state["batch_cursor"] = 0, 0
            if training:
                model.train_expert(expert)
                optimizer, scheduler = make_optimizer()
                state["stage"] = "train"
            else:
                model._freeze_all()
                model.set_expert(expert)
                optimizer, scheduler = None, None
                state["stage"] = "evaluate"
            emit(log, {"event": "task_start", "task": index, "task_id": task["task_id"],
                       "pre_evaluation": before, "trained": training, "expert": expert})
            commit()
        if state["stage"] == "train":
            from .data import training_rows
            model.train()
            model.set_expert(state["assignments"][index])
            replay = state["replay"] if args.method == "replay_lora" else []
            while state["epoch"] < args.epochs:
                epoch = state["epoch"]
                batches = epoch_batches(rows, replay, seed=args.seed, task=index, epoch=epoch, batch_size=args.batch_size)
                for batch_index in range(state["batch_cursor"], len(batches)):
                    batch, current_count, replay_count = batches[batch_index]
                    batch = training_rows(batch, seed=args.seed, epoch=epoch)
                    started = time.perf_counter()
                    loss, tokens = optimizer_step(model, batch, args.image_root, args.micro_batch_size, optimizer)
                    if torch.cuda.is_available() and model.device.type == "cuda":
                        torch.cuda.synchronize()
                    state["training_seconds"] += time.perf_counter() - started
                    state["steps"] += 1
                    state["image_exposures"] += len(batch)
                    state["target_tokens"] += tokens
                    state["current_exposures"] += current_count
                    state["replay_exposures"] += replay_count
                    state["batch_cursor"] = batch_index + 1
                    emit(log, {"event": "train_batch", "task": index, "epoch": epoch,
                               "batch": batch_index, "loss": loss, "steps": state["steps"],
                               "images": len(batch), "current_images": current_count,
                               "replay_images": replay_count, "target_tokens": tokens,
                               "lr": optimizer.param_groups[0]["lr"]})
                    commit()
                    if args.stop_after_steps and state["steps"] >= args.stop_after_steps:
                        return result()
                scheduler.step()
                state["epoch"], state["batch_cursor"] = epoch + 1, 0
                commit()
            state["stage"] = "evaluate"
            optimizer, scheduler = None, None
            commit()
        if state["stage"] == "evaluate":
            if router is not None:
                # A stage checkpoint is committed before/after this atomic phase;
                # a failed evaluation replays the whole phase from pre-observe memory.
                started = time.perf_counter()
                router.observe(features(model, rows, args), state["assignments"][index], model.num_experts)
                state["feature_seconds"] += time.perf_counter() - started
            if args.method == "replay_lora":
                state["replay"].extend(random.Random(args.seed + index).sample(rows, min(args.replay_per_task, len(rows))))
            if args.method == "independent":
                archive = output / f"independent_{task['task_id']}.pt"
                temporary = archive.with_suffix(".pt.tmp")
                torch.save(model.checkpoint_state(), temporary)
                temporary.replace(archive)
                state["independent_storage_bytes"] += archive.stat().st_size
            scores, oracle_scores, evaluations = [], [], []
            selected_state = model.checkpoint_state() if args.method == "independent" else None
            for seen, previous in enumerate(tasks[:index + 1]):
                if args.method == "independent" and seen != index:
                    model.load_checkpoint_state(torch.load(output / f"independent_{previous['task_id']}.pt", map_location="cpu", weights_only=False))
                elif args.method == "independent":
                    model.load_checkpoint_state(selected_state)
                if router is not None and seen not in eval_cache:
                    started = time.perf_counter()
                    eval_cache[seen] = features(model, previous[args.eval_split][:args.eval_limit or None], args)
                    state["feature_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                evaluation = evaluate(model, router, previous, scorers[previous["task_id"]], eval_cache.get(seen), args,
                                      output / f"predictions_after_{index:02d}_{previous['task_id']}.jsonl")
                scores.append(evaluation["scoring"]["score"])
                evaluations.append(evaluation)
                emit(log, {"event": "evaluation", "task": index, **evaluation})
                if args.oracle_diagnostic and router is not None:
                    oracle = evaluate(model, router, previous, scorers[previous["task_id"]], eval_cache[seen], args,
                                      output / f"oracle_after_{index:02d}_{previous['task_id']}.jsonl", state["assignments"][seen])
                    oracle_scores.append(oracle["scoring"]["score"])
                    emit(log, {"event": "oracle_evaluation", "task": index, **oracle})
                state["evaluation_seconds"] += time.perf_counter() - started
            if args.method == "independent":
                model.load_checkpoint_state(selected_state)
            if args.image_shuffled_diagnostic:
                shuffled = evaluate(model, None, task, scorers[task["task_id"]], None, args,
                                    output / f"image_shuffled_{index:02d}_{task['task_id']}.jsonl", shuffled=True)
                emit(log, {"event": "image_shuffled", "task": index, **shuffled})
                evaluations.append({"diagnostic": "image_shuffled", **shuffled})
            state["matrix"].append(scores)
            state["oracle_matrix"].append(oracle_scores)
            state["stage_evaluations"].append(evaluations)
            memory = router.memory_bytes() if router is not None else 0
            if router is not None:
                memory += sum(value.numel() * value.element_size() for value in (router.weight, router.center, router.prior) if value is not None)
                memory += len(json.dumps({key: router.state_dict()[key] for key in
                                          ("memory_per_task", "ridge", "modality", "seed", "num_experts")}).encode())
            replay_row_bytes = len(json.dumps(state["replay"], sort_keys=True).encode()) if state["replay"] else 0
            replay_image_bytes = sum((args.image_root / image).stat().st_size for image in
                                     {row["image"] for row in state["replay"]})
            resources = {"task_id": task["task_id"], "parameter_counts": model.parameter_counts(),
                         "router_memory_bytes": memory, "replay_images_stored": len(state["replay"]),
                         "replay_row_bytes": replay_row_bytes, "replay_image_bytes": replay_image_bytes,
                         "raw_replay_bytes": replay_row_bytes + replay_image_bytes,
                         "replay_storage": "Rows and immutable image-file references; no duplicate image copies",
                         "peak_gpu_memory_bytes": max(state["peak_gpu_memory_bytes"], torch.cuda.max_memory_allocated()) if model.device.type == "cuda" else 0,
                         "steps": state["steps"] - state["phase_steps_start"],
                         "image_exposures": state["image_exposures"] - state["phase_images_start"],
                         "target_tokens": state["target_tokens"] - state["phase_tokens_start"],
                         "current_exposures": state["current_exposures"] - state["phase_current_start"],
                         "replay_exposures": state["replay_exposures"] - state["phase_replay_start"],
                         "training_seconds": state["training_seconds"] - state["phase_training_seconds_start"]}
            state["stage_resources"].append(resources)
            emit(log, {"event": "task_end", "task": index, "resources": resources,
                       "raw_score_row": scores, "pre_to_post_gain": scores[index] - state["pre_scores"][index]})
            state["task_index"] += 1
            state["stage"] = "complete" if state["task_index"] == len(tasks) else "prepare"
            commit()
    atomic_json(output / "results.json", result())
    return result()


if __name__ == "__main__":
    main()
