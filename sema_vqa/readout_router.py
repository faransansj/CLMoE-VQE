"""Frozen-feature ridge routing; training task boundaries, never inference task IDs.

Memory contains a fixed-size deterministic sample of training feature vectors per
observed task and expert assignments, not images, questions or answers. New
experts can change the decision boundary: frozen experts do not imply frozen
end-to-end predictions. This memory must be counted in comparisons.
"""
import re

import torch


def question_stem(question):
    lines = [line.strip() for line in question.replace("<image>", "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty question stem")
    return re.sub(r"\s+", " ", lines[0]).rstrip(" .?!").casefold()


class FeatureRouter:
    def __init__(self, memory_per_task=128, ridge=1.0, modality="joint", seed=17):
        if memory_per_task < 1 or ridge <= 0 or modality not in ("joint", "image", "question"):
            raise ValueError("Invalid router memory, regularization or modality")
        self.memory_per_task = memory_per_task
        self.ridge = ridge
        self.modality = modality
        self.seed = seed
        self.memory = []
        self.weight = None
        self.center = None
        self.prior = None
        self.num_experts = 0

    def _features(self, features):
        if features.ndim != 2 or features.shape[1] != 1536 or not torch.isfinite(features).all():
            raise ValueError("Expected finite image/question features [N,1536]")
        if self.modality == "image":
            return features[:, :768].float()
        if self.modality == "question":
            return features[:, 768:].float()
        return features.float()

    def observe(self, features, expert, num_experts):
        if not 0 <= expert < num_experts or len(features) == 0:
            raise ValueError("Invalid training expert or empty feature set")
        self._features(features)
        generator = torch.Generator().manual_seed(self.seed + 1009 * len(self.memory))
        indices = torch.randperm(len(features), generator=generator)[:self.memory_per_task]
        self.memory.append((features.detach().cpu()[indices].clone(), expert))
        self.fit(num_experts, features.device)

    def fit(self, num_experts, device):
        if not self.memory or any(not 0 <= expert < num_experts for _, expert in self.memory):
            raise ValueError("Router needs valid observed training features")
        # Equal task weight even when a task has fewer stored examples.
        features = torch.cat([x for x, _ in self.memory]).to(device)
        x = self._features(features)
        labels = torch.cat([torch.full((len(x),), expert, dtype=torch.long) for x, expert in self.memory]).to(device)
        y = torch.nn.functional.one_hot(labels, num_classes=num_experts).float()
        sample_weights = torch.cat([torch.full((len(x),), 1.0 / len(x)) for x, _ in self.memory]).to(device)
        sample_weights *= len(sample_weights) / sample_weights.sum()
        self.center = (x * sample_weights[:, None]).sum(0) / sample_weights.sum()
        self.prior = (y * sample_weights[:, None]).sum(0) / sample_weights.sum()
        x = (x - self.center) * sample_weights.sqrt()[:, None]
        y = (y - self.prior) * sample_weights.sqrt()[:, None]
        # Solve the smaller primal/dual system; no inverse is materialized.
        if x.shape[0] < x.shape[1]:
            gram = x @ x.T
            gram.diagonal().add_(self.ridge)
            self.weight = x.T @ torch.linalg.solve(gram, y)
        else:
            gram = x.T @ x
            gram.diagonal().add_(self.ridge)
            self.weight = torch.linalg.solve(gram, x.T @ y)
        self.num_experts = num_experts

    def predict(self, features):
        if self.weight is None:
            raise ValueError("Router has not been fitted")
        x = self._features(features)
        device = x.device
        scores = (x - self.center.to(device)) @ self.weight.to(device) + self.prior.to(device)
        return scores.argmax(-1)

    def state_dict(self):
        return {"memory_per_task": self.memory_per_task, "ridge": self.ridge,
                "modality": self.modality, "seed": self.seed, "memory": self.memory,
                "num_experts": self.num_experts}

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        router = cls(**{key: state[key] for key in ("memory_per_task", "ridge", "modality", "seed")})
        router.memory = state["memory"]
        if router.memory:
            router.fit(state["num_experts"], device)
        return router

    def memory_bytes(self):
        return sum(x.numel() * x.element_size() + 8 for x, _ in self.memory)
