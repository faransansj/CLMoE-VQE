import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sema_vqa.vision_run import functional_train, learn_task, load_exposure


def exposure_fixture(tmp_path):
    config = {"method": "sema", "seed": 17, "epochs": 2, "lr": 2e-5,
              "adapter_lr": .005, "batch_size": 2, "micro_batch_size": 1,
              "image_size": 224, "threshold": .75, "rd_epochs": 20, "rd_lr": .01,
              "detect_batch_size": 128, "manifest_sha256": "same-manifest",
              "assets_sources": {"checkpoint": "same-weights"}, "eval_split": "eval",
              "torch": "same-version", "variant": "vision",
              "source_sha256": {"sema_vqa/vision_run.py": "old-loop", "sema_vqa/vision_model.py": "same-model"}}
    tasks = [{"train": [0, 1, 2]}, {"train": [0, 1]}]
    events = []
    for task, phases in enumerate((1, 2)):
        events.append({"event": "task_start", "task": task})
        # A preempted attempt must not inflate the completed task's exposure.
        if task == 1:
            events.extend([{"event": "train", "task": task, "phase": "abandoned", "epoch": 1},
                           {"event": "task_start", "task": task}])
        for phase in range(phases):
            for epoch in (1, 2):
                events.append({"event": "train", "task": task, "phase": str(phase), "epoch": epoch,
                               "examples": len(tasks[task]["train"]),
                               "steps": epoch * math.ceil(len(tasks[task]["train"]) / 2)})
        events.append({"event": "task_end", "task": task})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "results.json").write_text(json.dumps({"completed": 2}))
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    settings = copy.deepcopy(config)
    settings["method"] = "no_expansion"
    settings["source_sha256"]["sema_vqa/vision_run.py"] = "exposure-loop"
    return settings, tasks, events


def test_exposure_uses_only_last_completed_attempt(tmp_path):
    settings, tasks, _ = exposure_fixture(tmp_path)
    assert load_exposure(tmp_path, settings, tasks)["phase_counts"] == [1, 2]


@pytest.mark.parametrize("problem", ["partial_epoch", "wrong_seed", "changed_model"])
def test_exposure_refuses_unmatched_or_incomplete_source(tmp_path, problem):
    settings, tasks, events = exposure_fixture(tmp_path)
    if problem == "partial_epoch":
        events.pop(-2)
        (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    elif problem == "wrong_seed":
        settings["seed"] = 29
    else:
        settings["source_sha256"]["sema_vqa/vision_model.py"] = "different-model"
    with pytest.raises(ValueError, match="phase epochs|seed|model/scorer"):
        load_exposure(tmp_path, settings, tasks)


class RegressionReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Parameter(torch.tensor(.25))
        self.head = nn.Parameter(torch.tensor(-.5))

    def prepare_task(self, method, index):
        self.vision.requires_grad_(index == 0 or method == "sequential")

    def freeze_vision(self):
        self.vision.requires_grad_(False)

    def functional_parameters(self):
        return [self.vision] if self.vision.requires_grad else []

    def head_parameters(self):
        return [self.head]

    def loss(self, rows, image_root):
        data = torch.tensor(rows)
        return (self.vision * data[:, 0] + self.head - data[:, 1]).square().mean()


def test_exposure_restarts_optimizer_and_cosine_each_phase(tmp_path):
    task = {"train": [(1., 1.2), (-2., -.3), (.5, 2.1)]}
    args = SimpleNamespace(method="no_expansion", phase_counts=[1, 3], epochs=2,
                           lr=.1, adapter_lr=.005, seed=17, batch_size=2,
                           micro_batch_size=1, image_root="unused")
    actual, expected, continuous = RegressionReadout(), RegressionReadout(), RegressionReadout()
    learn_task(actual, task, 1, args, tmp_path / "actual.jsonl")
    expected.prepare_task("no_expansion", 1)
    for repeat in range(3):
        functional_train(expected, task["train"], args, tmp_path / "expected.jsonl", 1, str(repeat))
    long_args = copy.copy(args)
    long_args.epochs, long_args.phase_counts = 6, [1, 1]
    learn_task(continuous, task, 1, long_args, tmp_path / "continuous.jsonl")
    torch.testing.assert_close(actual.head, expected.head, rtol=0, atol=0)
    assert not torch.allclose(actual.head, continuous.head)
    assert actual.vision.item() == .25
