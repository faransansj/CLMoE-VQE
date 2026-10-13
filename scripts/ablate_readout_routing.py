"""Re-evaluate frozen expert banks with alternate routers, without retraining.

For expand/adaptive, expert k never changes after its training phase. A final
bank therefore reconstructs earlier stages when routing is restricted to that
stage's available experts. Native-router predictions MUST reproduce the parent
byte-for-byte at every stage before alternative results are accepted. Future
experts are never routed to. This is an inference ablation, not new independent
training runs or permission to inspect official evaluation during development.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sema_vqa.metrics import continual_metrics
from sema_vqa.readout_model import ReadoutMoEVQA
from sema_vqa.readout_router import FeatureRouter
from sema_vqa.readout_run import evaluate, features


VARIANTS = (("raw", "joint"), ("raw", "question"), ("raw", "image"), ("stem", "joint"), ("stem", "question"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-eval", action="store_true", help="Only after all confirmation settings/ablations are locked")
    cli = parser.parse_args()
    if cli.output.exists():
        raise ValueError("Inference ablation requires an unused output path")
    config = json.loads((cli.parent / "config.json").read_text())
    if config["method"] not in ("expand", "adaptive"):
        raise ValueError("A changing shared expert cannot reconstruct earlier stages from its final checkpoint")
    if config["eval_split"] == "eval" and not cli.allow_eval:
        raise ValueError("Official evaluation requires an explicit locked-confirmation invocation")
    # Core model/features/scorer must remain identical. The pilot archive has
    # a different runner/router revision; use its isolated snapshot for pilots.
    for name, digest in config["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Parent source revision differs: {name}")
    manifest_raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    tasks = json.loads(manifest_raw)["tasks"][:config["task_limit"] or None]
    checkpoint = torch.load(cli.parent / "latest.pt", map_location="cpu", weights_only=False)
    parent_result = json.loads((cli.parent / "results.json").read_text())
    if checkpoint["settings"] != config or parent_result["completed"] != len(tasks):
        raise ValueError("Incomplete or mismatched parent checkpoint")
    assignments = parent_result["task_expert_assignments"]
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    model = ReadoutMoEVQA(config["assets"], image_size=config["image_size"], rank=config["rank"], alpha=config["alpha"])
    model.load_checkpoint_state(checkpoint["model"])
    native = (config["router_text"], config["router_modality"])
    variants = list(VARIANTS)
    if native not in variants:
        variants.append(native)
    args = SimpleNamespace(**config)
    cli.output.mkdir(parents=True, exist_ok=False)
    routers, states = {}, {}
    for text, modality in variants:
        key = f"{text}_{modality}"
        routers[key] = FeatureRouter(config["router_memory"], config["router_ridge"], modality, config["seed"])
        states[key] = {"matrix": [], "pre_task_scores": [], "final_answer_counts": Counter()}
        (cli.output / key).mkdir()
    eval_features = {text: {} for text in {text for text, _ in variants}}
    reproduced = 0
    for stage, task in enumerate(tasks):
        train_features = {}
        for text in eval_features:
            args.router_text = text
            train_features[text] = features(model, task["train"], args)
            eval_features[text][stage] = features(model, task[config["eval_split"]], args)
        available = max(assignments[:stage + 1]) + 1
        for text, modality in variants:
            args.router_text, args.router_modality = text, modality
            key = f"{text}_{modality}"
            router, state = routers[key], states[key]
            directory = cli.output / key
            before_path = directory / f"predictions_before_{stage:02d}.jsonl"
            before, _, _ = evaluate(model, router, task, eval_features[text][stage], args, before_path)
            state["pre_task_scores"].append(before)
            if (text, modality) == native:
                if before_path.read_bytes() != (cli.parent / before_path.name).read_bytes():
                    raise AssertionError(f"Native pre-task predictions do not reconstruct stage{stage}")
                reproduced += 1
            router.observe(train_features[text], assignments[stage], available)
            scores = []
            for seen, previous in enumerate(tasks[:stage + 1]):
                path = directory / f"predictions_after_{stage:02d}_{previous['name']}.jsonl"
                score, answers, routes = evaluate(model, router, previous, eval_features[text][seen], args, path)
                if any(int(expert) >= available for expert in routes):
                    raise AssertionError("Routed into a future expert")
                scores.append(score)
                if stage == len(tasks) - 1:
                    state["final_answer_counts"].update(answers)
                if (text, modality) == native:
                    if path.read_bytes() != (cli.parent / path.name).read_bytes():
                        raise AssertionError(f"Native predictions do not reconstruct {path.name}")
                    reproduced += 1
            state["matrix"].append(scores)
    for state in states.values():
        state["metrics"] = continual_metrics(state["matrix"])
        state["plasticity_pp"] = [100 * (row[i] - state["pre_task_scores"][i]) for i, row in enumerate(state["matrix"])]
        state["largest_answer_fraction"] = max(state["final_answer_counts"].values()) / sum(state["final_answer_counts"].values())
    report = {"parent": str(cli.parent), "parent_config": config, "parent_checkpoint_sha256": sha(cli.parent / "latest.pt"),
              "parent_results_sha256": sha(cli.parent / "results.json"), "script_sha256": sha(Path(__file__)),
              "native_prediction_files_byte_identical": reproduced,
              "expert_retraining_steps": 0, "variants": states,
              "scope": "Same frozen trained experts; alternate TRAIN-feature routers. Inference-only paired ablations, not independent model fits. No future expert used."}
    (cli.output / "routing_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"parent": str(cli.parent), "native_files_reproduced": reproduced,
                      "variants": {key: state["metrics"] for key, state in states.items()}}, indent=2))


if __name__ == "__main__":
    main()
