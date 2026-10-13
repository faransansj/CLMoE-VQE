"""Expandable BLIP readout LoRA with one shared, permanently frozen backbone.

Each expert contains only query/value LoRA pairs in both text networks. The
caller groups rows by its predicted expert and uses ``set_expert`` before the
inherited loss/generation methods; neither task IDs nor answers enter routing.
Checkpoints contain the LoRA bank only and require the original pretrained
assets when reconstructing a model. Optimizers must be rebuilt after expansion
or checkpoint loading.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .vision_model import VisionVQA


def _validate_configuration(rank, alpha):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")


class LoRAExpert(nn.Module):
    """A bias-free low-rank delta, initialized to exactly zero output."""

    def __init__(self, base, rank):
        super().__init__()
        self.A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)


class LoRALinear(nn.Module):
    """One frozen projection and an expandable bank; no dense expert mixing."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        _validate_configuration(rank, alpha)
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.experts = nn.ModuleList()
        self.active_expert = None

    def _append_expert(self, copy_from=None):
        expert = LoRAExpert(self.base, self.rank)
        if copy_from is not None:
            with torch.no_grad():
                expert.A.copy_(self.experts[copy_from].A.detach())
                expert.B.copy_(self.experts[copy_from].B.detach())
        expert.requires_grad_(False)
        expert.train(self.training)
        self.experts.append(expert)

    def forward(self, x):
        result = self.base(x)
        if self.active_expert is not None:
            result = result + self.experts[self.active_expert](x) * (self.alpha / self.rank)
        return result


class ReadoutMoEVQA(VisionVQA):
    """Frozen BLIP with globally indexed, sparse-dispatched readout experts.

    Construction starts with no experts and base-only inference. Expansion
    freezes all existing parameters, preserves the selected expert and appends
    a frozen expert. ``train_expert`` selects/enables only the newest expert;
    older experts cannot be reopened for training after expansion. Selection
    alone never changes trainability. The original vision adapters stay frozen
    at their zero-output initialization, and every backbone stays in eval mode
    even while the LoRA parameters receive gradients.
    """

    def __init__(self, assets, image_size=224, rank=8, alpha=16, device="cuda"):
        _validate_configuration(rank, alpha)
        super().__init__(assets, image_size=image_size, device=device)
        self.rank = rank
        self.alpha = float(alpha)
        self.active_expert = None
        self._trainable_expert = None
        # Plain references: projections remain registered exactly once, at
        # their original location inside the encoder/decoder module trees.
        self._projections = {}
        for network_name in ("text_encoder", "text_decoder"):
            network = getattr(self, network_name)
            targets = [(name, module) for name, module in network.named_modules()
                       if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("query", "value")]
            if not targets:
                raise ValueError(f"No query/value projections found in {network_name}")
            for name, base in targets:
                parent_name, _, leaf = name.rpartition(".")
                parent = network.get_submodule(parent_name) if parent_name else network
                projection = LoRALinear(base, rank=rank, alpha=alpha)
                setattr(parent, leaf, projection)
                self._projections[f"{network_name}.{name}"] = projection
        self._freeze_all()
        self.train(False)

    @property
    def num_experts(self):
        return len(next(iter(self._projections.values())).experts)

    def _check_expert(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Expert index must be an integer")
        if not 0 <= index < self.num_experts:
            raise IndexError(f"Expert index {index} outside [0, {self.num_experts})")

    def _freeze_all(self):
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self._trainable_expert = None

    def add_expert(self, copy_from=None):
        """Append a zero-delta (or detached copied) expert; preserve selection."""
        if copy_from is not None:
            self._check_expert(copy_from)
        self._freeze_all()
        for projection in self._projections.values():
            projection._append_expert(copy_from)
        return self.num_experts - 1

    def set_expert(self, index):
        """Select a global expert, or disable all readout LoRA with ``None``."""
        if index is not None:
            self._check_expert(index)
        self.active_expert = index
        for projection in self._projections.values():
            projection.active_expert = index

    def expert_parameters(self, index):
        """Iterate only this expert's A/B tensors, regardless of trainability."""
        self._check_expert(index)
        return (parameter for projection in self._projections.values()
                for parameter in projection.experts[index].parameters())

    def train_expert(self, index):
        """Select/train the newest expert without reactivating any old expert."""
        self._check_expert(index)
        if index != self.num_experts - 1:
            raise ValueError("Old experts are permanently frozen after expansion")
        self._freeze_all()
        for parameter in self.expert_parameters(index):
            parameter.requires_grad_(True)
        self._trainable_expert = index
        self.set_expert(index)
        self.train(True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # Eval disables pretrained dropout but does not disable autograd through
        # frozen layers, which is required for gradients into earlier LoRA.
        self.visual_encoder.eval()
        self.text_encoder.eval()
        self.text_decoder.eval()
        return self

    def loss_per_example(self, rows, image_root, normalize_tokens=False):
        """Return BLIP's per-answer sums, optionally averaged over next tokens.

        This preserves the vendored decoder's label-smoothed cross-entropy,
        rather than claiming an unsmoothed NLL. Normalization excludes the BOS
        position and padding but includes the supervised answer-ending SEP.
        The caller controls no-grad/detachment for training-only competence
        probes; this method is also the differentiable training loss source.
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            states, mask = self._question_states(rows, image_root)
            answers = self._tokens([row["answer"] for row in rows], self.tokenizer.bos_token_id)
            targets = answers.input_ids.masked_fill(answers.attention_mask == 0, -100)
            output = self.text_decoder(answers.input_ids, attention_mask=answers.attention_mask,
                                       encoder_hidden_states=states, encoder_attention_mask=mask,
                                       labels=targets, reduction="none", return_dict=True)
            losses = output.loss.float()
        if normalize_tokens:
            losses = losses / (targets[:, 1:] != -100).sum(dim=1).clamp_min(1)
        return losses

    def loss(self, rows, image_root):
        """Keep the original BLIP mean-over-answers training objective."""
        return self.loss_per_example(rows, image_root).mean()

    @torch.no_grad()
    def routing_features(self, rows, image_root):
        """Return [B, 1536] frozen image/text means, separately L2-normalized.

        Question encoding is text-only (no cross-attention), excludes padding,
        and reads no answer/task metadata. LoRA is disabled throughout feature
        extraction; the fixed zero-output vision adapters contribute nothing.
        Restore selection and mode even if image loading/tokenization fails.
        """
        selected, training = self.active_expert, self.training
        try:
            self.set_expert(None)
            self.eval()
            if not rows:
                return torch.empty((0, 1536), device=self.device, dtype=torch.float32)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                image_states = self.visual_encoder(self._images(rows, image_root))
                questions = self._tokens([row["question"].replace("<image>", "").strip() for row in rows],
                                         self.enc_token_id)
                text_states = self.text_encoder(questions.input_ids,
                                                attention_mask=questions.attention_mask,
                                                return_dict=True, mode="text").last_hidden_state
            image_mean = image_states.float().mean(dim=1)
            mask = questions.attention_mask.unsqueeze(-1).to(dtype=torch.float32)
            text_mean = (text_states.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return torch.cat((F.normalize(image_mean, dim=-1), F.normalize(text_mean, dim=-1)), dim=-1)
        finally:
            self.set_expert(selected)
            self.train(training)

    def checkpoint_state(self):
        """Snapshot only LoRA tensors plus reconstruction/lifecycle metadata."""
        return {
            "version": 1,
            "rank": self.rank,
            "alpha": self.alpha,
            "num_experts": self.num_experts,
            "active_expert": self.active_expert,
            "trainable_expert": self._trainable_expert,
            "training": self.training,
            "projections": {
                name: {key: value.detach().to(device="cpu", copy=True)
                       for key, value in projection.experts.state_dict().items()}
                for name, projection in self._projections.items()
            },
        }

    def load_checkpoint_state(self, state):
        """Rebuild the dynamic bank, including saved rank, without loading base weights."""
        if state["version"] != 1:
            raise ValueError("Unsupported readout checkpoint version")
        rank, alpha, count = state["rank"], state["alpha"], state["num_experts"]
        _validate_configuration(rank, alpha)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Checkpoint expert count must be a nonnegative integer")
        if set(state["projections"]) != set(self._projections):
            raise ValueError("Checkpoint query/value projections do not match the backbone")
        selected, trainable = state["active_expert"], state["trainable_expert"]
        for index in (selected, trainable):
            if index is not None and (isinstance(index, bool) or not isinstance(index, int)
                                      or not 0 <= index < count):
                raise ValueError("Checkpoint contains an invalid expert index")
        if trainable is not None and trainable != count - 1:
            raise ValueError("Checkpoint attempts to train a permanently frozen old expert")
        self._freeze_all()
        self.set_expert(None)
        self.rank, self.alpha = rank, float(alpha)
        for name, projection in self._projections.items():
            projection.rank, projection.alpha = self.rank, self.alpha
            projection.experts = nn.ModuleList()
            for _ in range(count):
                projection._append_expert()
            projection.experts.load_state_dict(state["projections"][name], strict=True)
        if trainable is not None:
            self.train_expert(trainable)
        self.set_expert(selected)
        self.train(state["training"])

    def parameter_counts(self):
        """Count real tensors; base includes the frozen zero-output vision adapters.

        ``experts`` sums every bank's A/B tensors. ``active_trainable`` counts
        trainable tensors in the selected expert, while ``trainable`` counts
        all enabled tensors even when a different expert is selected to infer.
        """
        total = sum(parameter.numel() for parameter in self.parameters())
        experts = sum(parameter.numel() for projection in self._projections.values()
                      for parameter in projection.experts.parameters())
        active = (sum(parameter.numel() for parameter in self.expert_parameters(self.active_expert)
                      if parameter.requires_grad) if self.active_expert is not None else 0)
        return {"base": total - experts, "experts": experts, "active_trainable": active,
                "total": total,
                "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)}
