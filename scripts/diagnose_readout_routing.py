"""Validation-only router diagnostic, including source-format-neutral question stems.

This measures task identification, NOT VQA accuracy. No experts are fitted and
no official evaluation split is inspected. Native prompt formatting may identify
a dataset, so raw routing accuracy alone is not evidence of semantic skill
recognition or robustness to reformatted questions.
"""
import argparse
import json
from pathlib import Path

import torch

from sema_vqa.readout_model import ReadoutMoEVQA
from sema_vqa.readout_router import FeatureRouter, question_stem


@torch.no_grad()
def collect(model, rows, root, stem):
    if stem:
        rows = [dict(row, question=question_stem(row["question"])) for row in rows]
    return torch.cat([model.routing_features(rows[start:start + 8], root).cpu() for start in range(0, len(rows), 8)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "assets", "image-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = ReadoutMoEVQA(args.assets)
    tasks = json.loads(args.manifest.read_text())["tasks"]
    results = {}
    for stem in (False, True):
        training = [collect(model, task["train"], args.image_root, stem) for task in tasks]
        validation = [collect(model, task["validation"], args.image_root, stem) for task in tasks]
        for modality in ("image", "question", "joint"):
            router = FeatureRouter(modality=modality)
            stages = []
            for stage, data in enumerate(training):
                router.observe(data, stage, stage + 1)
                confusion = torch.zeros(stage + 1, stage + 1, dtype=torch.long)
                for target, x in enumerate(validation[:stage + 1]):
                    prediction = router.predict(x)
                    confusion[target] = torch.bincount(prediction, minlength=stage + 1)
                stages.append({"confusion": confusion.tolist(),
                               "macro_accuracy": float((confusion.diagonal() / confusion.sum(1)).mean()),
                               "per_task_accuracy": (confusion.diagonal() / confusion.sum(1)).tolist()})
            results[f"{'stem' if stem else 'raw'}_{modality}"] = stages
    report = {"manifest": str(args.manifest), "split": "validation", "task_names": [task["name"] for task in tasks],
              "results": results, "interpretation": "Task identification only; stem removes source directives, OCR list, option lists and terminal punctuation from routing input, not from the VQA model input. Raw-vs-stem gap measures dependence on formatting/context jointly, not a pure causal isolation of formatting."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
