"""Source-derived sequential VQA experiment; no replay or evaluation-driven tuning."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .model import ContinualVQA
from .metrics import continual_metrics, score_prediction


def emit(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def functional_train(model, rows, root, args, log, task, phase):
    model.set_mode("functional")
    params = [p for layer in model.adapted for module in (layer.mixture.functional, layer.mixture.router)
              for p in module.parameters() if p.requires_grad]
    if not params:
        return
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(rows) / args.batch_size) * args.epochs
    step = 0
    for epoch in range(args.epochs):
        order = list(rows)
        random.Random(args.seed + task * 100 + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            warmup = max(1, int(total * 0.03))
            scale = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in batch:
                loss = model.loss(row, root)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite CE at {row['id']}")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0 or step == total:
                emit(log, {"event": "train", "task": task, "phase": phase, "step": step,
                           "steps": total, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"]})


@torch.no_grad()
def collect_features(model, rows, root):
    collected = [[] for _ in model.adapted]
    for row in rows:
        values = model.describe(row, root, "features")
        for target, value in zip(collected, values):
            target.append(value.cpu())
    return [torch.cat(values).unsqueeze(1) for values in collected]


def descriptor_train(model, rows, root, args, log, task):
    features = collect_features(model, rows, root)
    for index, (layer, data) in enumerate(zip(model.adapted, features)):
        mixture = layer.mixture
        params = [p for p in mixture.rd.parameters() if p.requires_grad]
        if not params:
            continue
        dtype = next(mixture.parameters()).dtype
        optimizer = torch.optim.SGD(params, lr=args.rd_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rd_epochs)
        for epoch in range(args.rd_epochs):
            order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed + epoch))
            losses = []
            for ids in order.split(32):
                x = data[ids].to(device=model.device, dtype=dtype)
                optimizer.zero_grad(set_to_none=True)
                loss = mixture.rd_loss(x)
                mixture.observe(x)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite RD reconstruction loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        emit(log, {"event": "rd", "task": task, "layer": index,
                   "epochs": args.rd_epochs, "final_loss": float(np.mean(losses))})


def evaluate(model, tasks, root, output, stage, args):
    scores = []
    for task in tasks:
        correct = 0
        path = output / f"predictions_after_{stage:02d}_{task['name']}.jsonl"
        with path.open("w") as stream:
            for row in task["eval"]:
                prediction = model.answer(row, root, args.max_new_tokens)
                hit = score_prediction(prediction, row, task["metric"])
                correct += hit
                stream.write(json.dumps({"id": row["id"], "prediction": prediction,
                                         "answer": row["answer"], "score": hit}) + "\n")
        scores.append(correct / len(task["eval"]))
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["sema", "sequential", "no_expansion", "task_expansion"], required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--bottleneck", type=int, default=16)
    p.add_argument("--rd-dim", type=int, default=128)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--detect-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rd-lr", type=float, default=0.01)
    p.add_argument("--rd-epochs", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    if (output / "config.json").exists() and not args.resume:
        raise FileExistsError("Existing experiment: use --resume or a new output path")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    settings = vars(args) | {"manifest_sha256": digest, "torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(),
                             "metrics_by_task": {task["name"]: task["metric"] for task in tasks}}
    if args.resume:
        previous = json.loads((output / "config.json").read_text())
        for key in ("manifest_sha256", "method", "seed", "layers", "bottleneck", "rd_dim",
                    "threshold", "epochs", "lr", "rd_lr", "rd_epochs", "batch_size",
                    "detect_batch_size", "max_new_tokens", "assets"):
            if previous[key] != settings[key]:
                raise ValueError(f"Resume configuration differs: {key}")
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    log = output / "events.jsonl"
    model = ContinualVQA(args.assets, args.layers, args.bottleneck, args.rd_dim, args.threshold)
    matrix = []
    start_task = 0
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_adapter_state(saved["adapters"])
        matrix = saved["matrix"]
        start_task = saved["completed"]
    elif args.resume:
        raise FileNotFoundError("Resume requires a completed-task checkpoint")
    for task_index in range(start_task, len(tasks)):
        # Task-local initialization makes task-boundary resume independent of
        # random numbers consumed while reconstructing dynamic checkpoint modules.
        torch.manual_seed(args.seed + 100000 * task_index)
        torch.cuda.manual_seed_all(args.seed + 100000 * task_index)
        started = time.time()
        task = tasks[task_index]
        rows = task["train"]
        emit(log, {"event": "task_start", "index": task_index, "name": task["name"], "train_n": len(rows)})
        if task_index == 0 or args.method == "sequential":
            if args.method == "sequential":
                for layer in model.adapted:
                    layer.mixture.functional.requires_grad_(True)
            functional_train(model, rows, args.image_root, args, log, task_index, "functional")
            if args.method in ("sema", "task_expansion"):
                descriptor_train(model, rows, args.image_root, args, log, task_index)
        elif args.method == "task_expansion":
            for layer in model.adapted:
                layer.mixture.expand()
            functional_train(model, rows, args.image_root, args, log, task_index, "forced_expansion")
            descriptor_train(model, rows, args.image_root, args, log, task_index)
        elif args.method == "sema":
            # At most one addition per layer/task; train a shallow addition before
            # testing deeper representations, matching SEMA's sequential policy.
            for index, layer in enumerate(model.adapted):
                data = collect_features(model, rows, args.image_root)[index]
                scores = []
                for batch in data.split(args.detect_batch_size):
                    score = layer.mixture.expansion_score(batch.to(model.device, dtype=next(layer.mixture.parameters()).dtype))
                    scores.append(float(score))
                expand = any(score > args.threshold for score in scores)
                emit(log, {"event": "detection", "task": task_index, "layer": index,
                           "scores": scores, "threshold": args.threshold, "expand": expand})
                if expand:
                    layer.mixture.expand()
                    functional_train(model, rows, args.image_root, args, log, task_index, f"expansion_layer_{index}")
                    descriptor_train(model, rows, args.image_root, args, log, task_index)
                    layer.mixture.freeze()
        for layer in model.adapted:
            layer.mixture.freeze()
        scores = evaluate(model, tasks[:task_index + 1], args.image_root, output, task_index, args)
        matrix.append(scores)
        result = {"matrix": matrix, "task_names": [task["name"] for task in tasks],
                  "completed": task_index + 1, "metrics": continual_metrics(matrix), "matrix_unit": "fraction",
                  "experts": [layer.mixture.num_experts for layer in model.adapted],
                  "adapter_parameters": sum(p.numel() for layer in model.adapted for p in layer.mixture.parameters())}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary = output / "latest.tmp.pt"
        torch.save(result | {"adapters": model.adapter_state()}, temporary)
        temporary.replace(checkpoint)
        emit(log, {"event": "task_end", "index": task_index, "scores": scores,
                   "metrics": result["metrics"], "experts": result["experts"], "seconds": time.time() - started})


if __name__ == "__main__":
    main()
