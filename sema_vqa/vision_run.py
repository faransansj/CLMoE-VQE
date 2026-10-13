"""Source SEMA task loop with trainable VQA readout, no replay or test tuning."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .metrics import continual_metrics, score_prediction


def emit(log, record):
    with log.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def optimizer_step(loss_fn, rows, micro_batch_size, optimizers):
    """Mean over examples, including an incomplete last effective batch."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    mean = 0.0
    for start in range(0, len(rows), micro_batch_size):
        batch = rows[start:start + micro_batch_size]
        loss = loss_fn(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite VQA loss")
        weight = len(batch) / len(rows)
        (loss * weight).backward()
        mean += float(loss.detach()) * weight
    for optimizer in optimizers:
        optimizer.step()
    return mean


def functional_train(model, rows, args, log, task, phase):
    model.train()
    functional = list(model.functional_parameters())
    head = list(model.head_parameters())
    if not head:
        raise ValueError("Source SEMA keeps a trainable readout even without expansion")
    optimizers = [torch.optim.AdamW(head, lr=args.lr, weight_decay=0.05)]
    if functional:
        optimizers.append(torch.optim.SGD(functional, lr=args.adapter_lr, momentum=0.9, weight_decay=0.0005))
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs) for o in optimizers]
    steps = 0
    for epoch in range(args.epochs):
        order = list(rows)
        random.Random(args.seed + task * 1000 + epoch).shuffle(order)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            loss_sum += len(batch) * optimizer_step(lambda b: model.loss(b, args.image_root), batch,
                                                    args.micro_batch_size, optimizers)
            steps += 1
        emit(log, {"event": "train", "task": task, "phase": phase, "epoch": epoch + 1,
                   "loss": loss_sum / len(rows), "steps": steps, "examples": len(rows),
                   "head_lr": optimizers[0].param_groups[0]["lr"],
                   "vision_trainable": sum(p.numel() for p in functional)})
        for schedule in schedules:
            schedule.step()


@torch.no_grad()
def collect_features(model, rows, args):
    model.eval()
    values = [[] for _ in model.mixtures]
    for start in range(0, len(rows), args.micro_batch_size):
        for target, feature in zip(values, model.features(rows[start:start + args.micro_batch_size], args.image_root)):
            target.append(feature.cpu())
    return [torch.cat(group) for group in values]


def descriptor_train(model, rows, args, log, task):
    features = collect_features(model, rows, args)
    for layer, (mixture, data) in enumerate(zip(model.mixtures, features)):
        params = [p for p in mixture.rd.parameters() if p.requires_grad]
        if not params:
            continue
        optimizer = torch.optim.SGD(params, lr=args.rd_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rd_epochs)
        for epoch in range(args.rd_epochs):
            order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed + epoch))
            losses = []
            for indices in order.split(32):
                x = data[indices].to(model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = mixture.rd_loss(x)
                mixture.observe(x)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite descriptor loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        emit(log, {"event": "rd", "task": task, "layer": layer + 9, "loss": sum(losses) / len(losses),
                   "epochs": args.rd_epochs, "observations": int(mixture.stats[-1].count)})


def learn_task(model, task, index, args, log):
    rows = task["train"]
    model.prepare_task(args.method, index)
    if index == 0 or args.method == "sequential":
        for repeat in range(args.phase_counts[index]):
            phase = "functional" if repeat == 0 else f"functional_repeat_{repeat + 1}"
            functional_train(model, rows, args, log, index, phase)
        if args.method in ("sema", "task_expansion"):
            descriptor_train(model, rows, args, log, index)
    elif args.method == "task_expansion":
        for mixture in model.mixtures:
            mixture.expand()
        functional_train(model, rows, args, log, index, "forced_expansion")
        descriptor_train(model, rows, args, log, index)
    elif args.method == "no_expansion":
        for repeat in range(args.phase_counts[index]):
            phase = "readout_only" if repeat == 0 else f"readout_only_repeat_{repeat + 1}"
            functional_train(model, rows, args, log, index, phase)
    else:
        added = set()
        # Upstream scans batches, adds at the first eligible shallow layer,
        # trains, then re-detects with the updated representations.
        while True:
            features = collect_features(model, rows, args)
            selected = None
            for start in range(0, len(rows), args.detect_batch_size):
                for layer, (mixture, data) in enumerate(zip(model.mixtures, features)):
                    if layer in added:
                        continue
                    score = float(mixture.expansion_score(data[start:start + args.detect_batch_size].to(model.device)))
                    expands = score > args.threshold
                    emit(log, {"event": "detection", "task": index, "layer": layer + 9,
                               "batch_start": start, "score": score, "threshold": args.threshold, "expand": expands})
                    if expands:
                        selected = layer
                        break
                if selected is not None:
                    break
            if selected is None:
                break
            model.mixtures[selected].expand()
            added.add(selected)
            functional_train(model, rows, args, log, index, f"expansion_layer_{selected + 9}")
            descriptor_train(model, rows, args, log, index)
            model.freeze_vision()
            if len(added) == len(model.mixtures):
                break
        if not added:
            functional_train(model, rows, args, log, index, "readout_only")
    model.freeze_vision()


@torch.no_grad()
def evaluate(model, tasks, stage, args, output):
    model.eval()
    scores = []
    for task in tasks:
        rows = task[args.eval_split]
        total = 0.0
        path = output / f"predictions_after_{stage:02d}_{task['name']}.jsonl"
        with path.open("w") as stream:
            for start in range(0, len(rows), args.micro_batch_size):
                batch = rows[start:start + args.micro_batch_size]
                predictions = model.answer(batch, args.image_root)
                if len(predictions) != len(batch):
                    raise ValueError("Incomplete generation batch")
                for row, prediction in zip(batch, predictions):
                    score = score_prediction(prediction, row, task["metric"])
                    total += score
                    stream.write(json.dumps({"id": row["id"], "prediction": prediction, "score": score}) + "\n")
        scores.append(total / len(rows))
    return scores


def load_exposure(source, settings, tasks):
    """Replay completed SEMA functional phases, not RD work or evaluation scores."""
    files = {name: (source / name).read_bytes() for name in ("config.json", "events.jsonl", "results.json")}
    config = json.loads(files["config.json"])
    result = json.loads(files["results.json"])
    if settings["method"] not in ("sequential", "no_expansion") or config["method"] != "sema":
        raise ValueError("Exposure matching requires sequential/no_expansion control and SEMA source")
    for key in ("seed", "epochs", "lr", "adapter_lr", "batch_size", "micro_batch_size", "image_size",
                "threshold", "rd_epochs", "rd_lr", "detect_batch_size", "manifest_sha256",
                "assets_sources", "eval_split", "torch", "variant"):
        if settings[key] != config[key]:
            raise ValueError(f"Exposure source configuration differs: {key}")
    if set(config["source_sha256"]) != set(settings["source_sha256"]):
        raise ValueError("Exposure source file set differs")
    for name, digest in settings["source_sha256"].items():
        if name != "sema_vqa/vision_run.py" and config["source_sha256"][name] != digest:
            raise ValueError(f"Exposure source model/scorer changed: {name}")
    if result["completed"] != len(tasks):
        raise ValueError("Exposure source sequence is incomplete")
    phases, ended = {}, set()
    for line in files["events.jsonl"].splitlines():
        event = json.loads(line)
        task = event["task"]
        if event["event"] == "task_start":
            phases[task] = {}
            ended.discard(task)
        elif event["event"] == "train":
            if task not in phases:
                raise ValueError("Exposure training event has no task start")
            phases[task].setdefault(event["phase"], []).append(event)
        elif event["event"] == "task_end":
            ended.add(task)
    if ended != set(range(len(tasks))) or set(phases) != ended:
        raise ValueError("Exposure source task events are incomplete")
    counts = []
    for index, task in enumerate(tasks):
        if not 1 <= len(phases[index]) <= 3:
            raise ValueError("Invalid SEMA phase count")
        for events in phases[index].values():
            if [e["epoch"] for e in events] != list(range(1, settings["epochs"] + 1)):
                raise ValueError("Incomplete or duplicated exposure phase epochs")
            if any(e["examples"] != len(task["train"]) or
                   e["steps"] != e["epoch"] * math.ceil(len(task["train"]) / settings["batch_size"]) for e in events):
                raise ValueError("Exposure source examples or optimizer steps differ")
        counts.append(len(phases[index]))
    if counts[0] != 1:
        raise ValueError("Initial SEMA task must have one functional phase")
    return {"source_run": str(source), "phase_counts": counts,
            "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "assets", "image-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--method", choices=("sequential", "sema", "no_expansion", "task_expansion"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adapter-lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--rd-epochs", type=int, default=20)
    parser.add_argument("--rd-lr", type=float, default=0.01)
    parser.add_argument("--detect-batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--eval-split", choices=("validation", "eval"), default="validation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--match-exposure-from", help="Completed SEMA run; only sequential/no_expansion controls")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch_size, args.rd_epochs, args.detect_batch_size) <= 0:
        raise ValueError("Epochs and batch sizes must be positive")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    tasks = manifest["tasks"]
    args.phase_counts = [1] * len(tasks)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args) | {"manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                             "assets_sources": json.loads((Path(args.assets) / "sources.json").read_text()),
                             "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                             "variant": "SEMA vision BLIP pure-pretrained trainable-readout"}
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / "sema_vqa/vision_model.py", root / "sema_vqa/expansion.py", root / "sema_vqa/metrics.py"]
    settings["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.match_exposure_from:
        exposure = load_exposure(Path(args.match_exposure_from), settings, tasks)
        args.phase_counts = exposure["phase_counts"]
        settings.update(phase_counts=args.phase_counts, exposure_match=exposure)
    config_path = output / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("Existing experiment; use new output or explicit resume")
        old = json.loads(config_path.read_text())
        for key in settings:
            if key not in ("resume", "gpu") and (key not in old or settings[key] != old[key]):
                raise ValueError(f"Resume scientific configuration changed: {key}")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    from .vision_model import VisionVQA
    model = VisionVQA(args.assets, args.image_size, args.threshold)
    matrix, start_task = [], 0
    checkpoint = output / "latest.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_checkpoint_state(saved["model"])
        matrix, start_task = saved["matrix"], saved["completed"]
        del saved
    log = output / "events.jsonl"
    for index in range(start_task, len(tasks)):
        torch.manual_seed(args.seed + 100000 * index)
        torch.cuda.manual_seed_all(args.seed + 100000 * index)
        began = time.time()
        emit(log, {"event": "task_start", "task": index, "name": tasks[index]["name"], "train_n": len(tasks[index]["train"])})
        learn_task(model, tasks[index], index, args, log)
        matrix.append(evaluate(model, tasks[:index + 1], index, args, output))
        result = {"completed": index + 1, "task_names": [t["name"] for t in tasks], "matrix": matrix,
                  "matrix_unit": "fraction", "metrics": continual_metrics(matrix), "experts": model.expert_counts,
                  "parameters": model.parameter_counts(), "eval_split": args.eval_split}
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"completed": index + 1, "matrix": matrix, "model": model.checkpoint_state()}, temporary)
        temporary.replace(checkpoint)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        emit(log, {"event": "task_end", "task": index, "scores": matrix[-1], "experts": model.expert_counts,
                   "metrics": result["metrics"], "seconds": time.time() - began})


if __name__ == "__main__":
    main()
