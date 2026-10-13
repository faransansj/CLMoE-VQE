"""VQA-aware readout expert growth with task-agnostic routing and explicit controls.

Training task boundaries are known. Inference uses only frozen image/question
features. Expert selection for adaptive growth uses CURRENT TRAIN labels only;
validation/eval labels are restricted to scoring. Original SEMA remains separate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .metrics import continual_metrics, score_prediction
from .readout_router import FeatureRouter, question_stem
from .vision_run import emit, optimizer_step


METHODS = ("shared_lora", "replay_lora", "frozen_first", "expand", "adaptive", "full_readout")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def train_phase(model, rows, replay, args, log, task):
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Training phase has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + task * 1000 + epoch)
        order = list(rows)
        rng.shuffle(order)
        total, steps, replay_examples = 0.0, 0, 0
        for start in range(0, len(order), args.batch_size):
            batch = list(order[start:start + args.batch_size])
            # Preserve the total examples/steps; replay replaces half of this
            # phase's examples, not an unreported extra pass over old data.
            if replay:
                count = len(batch) // 2
                if count:
                    batch[-count:] = rng.choices(replay, k=count)
                    replay_examples += count
            total += len(batch) * optimizer_step(lambda b: model.loss(b, args.image_root), batch,
                                                  args.micro_batch_size, [optimizer])
            steps += 1
        emit(log, {"event": "train", "task": task, "epoch": epoch + 1,
                   "loss": total / len(order), "steps": steps, "examples": len(order),
                   "replay_examples": replay_examples, "trainable": sum(p.numel() for p in parameters),
                   "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()


@torch.no_grad()
def features(model, rows, args):
    if args.router_text == "stem":
        rows = [dict(row, question=question_stem(row["question"])) for row in rows]
    return torch.cat([model.routing_features(rows[start:start + args.micro_batch_size], args.image_root).cpu()
                      for start in range(0, len(rows), args.micro_batch_size)])


@torch.no_grad()
def competence(model, rows, args, expert):
    model.eval()
    model.set_expert(expert)
    losses = []
    for start in range(0, len(rows), args.micro_batch_size):
        losses.append(model.loss_per_example(rows[start:start + args.micro_batch_size], args.image_root,
                                             normalize_tokens=True).detach().cpu())
    return float(torch.cat(losses).mean())


@torch.no_grad()
def routed_answers(model, router, rows, cached_features, args, forced_expert=None):
    model.eval()
    if args.method == "full_readout":
        assignments = torch.zeros(len(rows), dtype=torch.long)
    elif forced_expert is not None:
        assignments = torch.full((len(rows),), forced_expert, dtype=torch.long)
    elif router is None:
        assignments = torch.full((len(rows),), 0 if model.num_experts else -1, dtype=torch.long)
    elif not router.memory:
        assignments = torch.full((len(rows),), -1, dtype=torch.long)
    else:
        assignments = router.predict(cached_features).cpu()
    predictions = [None] * len(rows)
    for expert in assignments.unique(sorted=True).tolist():
        if args.method != "full_readout":
            model.set_expert(None if expert == -1 else expert)
        indices = (assignments == expert).nonzero(as_tuple=True)[0].tolist()
        for start in range(0, len(indices), args.micro_batch_size):
            group = indices[start:start + args.micro_batch_size]
            answers = model.answer([rows[index] for index in group], args.image_root)
            if len(answers) != len(group):
                raise ValueError("Incomplete answer generation")
            for index, answer in zip(group, answers):
                predictions[index] = answer
    if any(answer is None for answer in predictions):
        raise ValueError("Routing lost an example")
    return predictions, assignments.tolist()


def evaluate(model, router, task, cache, args, path, forced_expert=None):
    rows = task[args.eval_split]
    predictions, assignments = routed_answers(model, router, rows, cache, args, forced_expert)
    scores = [score_prediction(answer, row, task["metric"]) for answer, row in zip(predictions, rows)]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row, answer, expert, score in zip(rows, predictions, assignments, scores):
            stream.write(json.dumps({"id": row["id"], "prediction": answer, "expert": expert, "score": score}) + "\n")
    temporary.replace(path)
    return sum(scores) / len(scores), dict(Counter(predictions)), dict(Counter(assignments))


def source_hashes():
    files = ("readout_run.py", "readout_model.py", "readout_router.py", "vision_model.py", "vision_run.py", "expansion.py", "metrics.py")
    root = Path(__file__).parent
    return {str(Path("sema_vqa") / name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "assets", "image-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--eval-split", choices=("validation", "eval"), default="validation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--router-modality", choices=("joint", "image", "question"), default="joint")
    parser.add_argument("--router-text", choices=("raw", "stem"), default="raw")
    parser.add_argument("--router-memory", type=int, default=128)
    parser.add_argument("--router-ridge", type=float, default=1.0)
    parser.add_argument("--replay-per-task", type=int, default=64)
    parser.add_argument("--growth-margin", type=float, default=0.5,
                        help="Grow if every expert exceeds its own train-reference smoothed token CE by this margin")
    parser.add_argument("--probe-size", type=int, default=128)
    parser.add_argument("--task-limit", type=int, default=0, help="Development prefix only; 0 means complete protocol")
    parser.add_argument("--oracle-diagnostic", action="store_true", help="Also report task-ID oracle separately, never primary")
    args = parser.parse_args()
    if min(args.epochs, args.rank, args.batch_size, args.micro_batch_size, args.router_memory, args.probe_size) < 1:
        parser.error("Epochs, ranks and batch/memory sizes must be positive")
    if args.lr <= 0 or args.growth_margin < 0 or args.replay_per_task < 0 or args.task_limit < 0:
        parser.error("Invalid learning rate, growth margin, replay budget or task limit")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    tasks = manifest["tasks"][:args.task_limit or None]
    if args.eval_split == "eval" and args.task_limit:
        parser.error("Final evaluation must cover complete declared protocol")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), source_sha256=source_hashes(),
                    torch=torch.__version__, protocol=manifest["protocol"],
                    assets_sources=json.loads((args.assets / "sources.json").read_text()),
                    inference=("Frozen image/question feature router; no task IDs or answers; optional oracle is diagnostic only"
                               if args.method in ("expand", "adaptive") else
                               "No routing; one evolving or frozen readout; no task ID input"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Refusing resume with changed scientific settings or source")
    atomic_json(config_path, settings)
    log = output / "events.jsonl"
    from .readout_model import ReadoutMoEVQA
    if args.method == "full_readout":
        from .vision_model import VisionVQA
        model = VisionVQA(args.assets, image_size=args.image_size)
        model.freeze_vision()
        router = None
    else:
        model = ReadoutMoEVQA(args.assets, image_size=args.image_size, rank=args.rank, alpha=args.alpha)
        router = (FeatureRouter(args.router_memory, args.router_ridge, args.router_modality, args.seed)
                  if args.method in ("expand", "adaptive") else None)
    matrix, oracle_matrix, pre_scores, assignments, references, replay = [], [], [], [], [], []
    checkpoint_path = output / "latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["settings"] != settings:
            raise ValueError("Checkpoint scientific settings mismatch")
        model.load_checkpoint_state(checkpoint["model"])
        if router is not None:
            router = FeatureRouter.from_state_dict(checkpoint["router"])
        matrix, oracle_matrix = checkpoint["matrix"], checkpoint["oracle_matrix"]
        pre_scores, assignments, references, replay = (checkpoint[key] for key in ("pre_scores", "assignments", "references", "replay"))
        # Recover the derived report if interruption followed checkpoint commit.
        atomic_json(output / "results.json", checkpoint["result"])
    eval_features = {}
    started = time.time()
    for index in range(len(matrix), len(tasks)):
        task = tasks[index]
        torch.manual_seed(args.seed + index * 10007)
        torch.cuda.manual_seed_all(args.seed + index * 10007)
        emit(log, {"event": "task_start", "task": index, "name": task["name"]})
        rows = task["train"]
        train_features = None if router is None else features(model, rows, args)
        if router is not None:
            for seen in range(index + 1):
                if seen not in eval_features:
                    eval_features[seen] = features(model, tasks[seen][args.eval_split], args)
        pre_score, _, _ = evaluate(model, router, task, eval_features.get(index), args,
                                   output / f"predictions_before_{index:02d}.jsonl")
        pre_scores.append(pre_score)
        probe = random.Random(args.seed + index).sample(rows, min(args.probe_size, len(rows)))
        grow, expert, train = False, 0, True
        if args.method == "full_readout":
            model.text_encoder.requires_grad_(True)
            model.text_decoder.requires_grad_(True)
        elif index == 0 or args.method == "expand":
            expert = model.add_expert()
            grow = True
            model.train_expert(expert)
        elif args.method == "adaptive":
            losses = [competence(model, probe, args, k) for k in range(model.num_experts)]
            differences = [loss - reference for loss, reference in zip(losses, references)]
            expert = min(range(len(losses)), key=losses.__getitem__)
            grow = min(differences) > args.growth_margin
            emit(log, {"event": "growth_probe", "task": index, "losses": losses,
                       "reference_smoothed_token_ce": references, "differences": differences, "grow": grow,
                       "probe_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in probe]).encode()).hexdigest()})
            if grow:
                expert = model.add_expert()
                model.train_expert(expert)
            else:
                model.set_expert(expert)
                train = False
        elif args.method == "frozen_first":
            model.set_expert(0)
            train = False
        else:
            model.train_expert(0)
        if train:
            train_phase(model, rows, replay if args.method == "replay_lora" else [], args, log, index)
        if router is not None:
            if grow:
                references.append(competence(model, probe, args, expert))
            router.observe(train_features, expert, model.num_experts)
        assignments.append(expert)
        if args.method == "replay_lora":
            replay.extend(random.Random(args.seed + index).sample(rows, min(args.replay_per_task, len(rows))))
        scores, oracle_scores = [], []
        for seen, previous in enumerate(tasks[:index + 1]):
            score, answers, routes = evaluate(model, router, previous, eval_features.get(seen), args,
                                               output / f"predictions_after_{index:02d}_{previous['name']}.jsonl")
            scores.append(score)
            emit(log, {"event": "evaluation", "task": index, "evaluated_task": seen, "score": score,
                       "answer_counts": answers, "route_counts": routes})
            if args.oracle_diagnostic and router is not None:
                oracle_score, _, _ = evaluate(model, router, previous, eval_features[seen], args,
                                              output / f"oracle_after_{index:02d}_{previous['name']}.jsonl", assignments[seen])
                oracle_scores.append(oracle_score)
        matrix.append(scores)
        oracle_matrix.append(oracle_scores)
        metrics = continual_metrics(matrix)
        result = {"completed": len(matrix), "matrix": matrix, "metrics": metrics,
                  "oracle_matrix": oracle_matrix, "pre_task_scores": pre_scores,
                  "plasticity_pp": [100 * (row[i] - pre_scores[i]) for i, row in enumerate(matrix)],
                  "task_expert_assignments": assignments,
                  "expert_count": model.num_experts if router else 1,
                  "router_memory_bytes": router.memory_bytes() if router else 0,
                  "replay_examples": len(replay), "parameter_counts": model.parameter_counts()}
        emit(log, {"event": "task_end", "task": index, "metrics": metrics,
                   "experts": result["expert_count"], "grew": grow, "trained": train,
                   "plasticity_pp": result["plasticity_pp"][-1], "seconds": time.time() - started})
        checkpoint = {"settings": settings, "model": model.checkpoint_state(),
                      "router": router.state_dict() if router else None, "matrix": matrix,
                      "oracle_matrix": oracle_matrix, "pre_scores": pre_scores,
                      "assignments": assignments, "references": references, "replay": replay, "result": result}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        atomic_json(output / "results.json", result)


if __name__ == "__main__":
    main()
