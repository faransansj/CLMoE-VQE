# Adapted from SEMA-CL backbone/sema_components.py, backbone/sema_block.py,
# and models/sema.py (vendored in third_party/SEMA-CL).
#
# MIT License
#
# Copyright (c) 2023 SEMA-CL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Explicit SEMA adapter growth for token features, without forward-time mutation.

The caller supplies a boolean [batch, sequence] mask selecting visual/question
context, excluding both padding and answer tokens. The same context pools router
logits and representation descriptors (RDs). ``forward`` returns an additive
adapter delta, not a residual sum; the caller adds it at its chosen backbone site.

Compared with upstream, descriptor inputs are detached, padding/answer masking is
supported, router rows remain independent modules, and growth, RD optimization,
and rolling-stat observation are explicit operations. Statistics and trainability
are checkpointed. Rolling windows retain the latest samples even when a batch
crosses capacity (upstream discards that batch's overflow during initial fill).
There is no optional layer normalization, scalar, or dropout: the functional
expert is the upstream down/ReLU/up path with its LoRA-style initialization.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Expected nonempty token features of shape [B, S, D]")
    if mask is None:
        return x.mean(dim=1)
    if mask.shape != x.shape[:2]:
        raise ValueError("mask must have shape [B, S]")
    mask = mask.to(device=x.device, dtype=torch.bool)
    counts = mask.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Each sample must select at least one context token")
    # masked_fill also excludes nonfinite values in unselected padding tokens.
    return x.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / counts


class _Functional(nn.Module):
    def __init__(self, d_model: int, bottleneck: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.up_proj(self.non_linear_func(self.down_proj(x)))


class _Descriptor(nn.Module):
    def __init__(self, d_model: int, rd_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_model, rd_dim)
        self.decoder = nn.Linear(rd_dim, d_model)
        for layer in (self.encoder, self.decoder):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.bias)

    def forward(self, pooled: Tensor) -> Tensor:
        reconstruction = self.decoder(self.encoder(pooled))
        # Accumulate MSE in float32 even under mixed precision.
        return (reconstruction.float() - pooled.float()).square().mean(dim=-1)


class _Records(nn.Module):
    """Bounded sample ring; buffers stay float32 under module dtype conversion."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.register_buffer("values", torch.zeros(capacity, dtype=torch.float32))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))

    def _apply(self, fn, recurse=True):
        values = self.values
        result = super()._apply(fn, recurse=recurse)
        # Reuse the original float32 values, not a rounded half-precision copy.
        self.values = values.to(device=self.values.device, dtype=torch.float32)
        return result

    @torch.no_grad()
    def observe(self, losses: Tensor) -> None:
        losses = losses.detach().to(device=self.values.device, dtype=torch.float32)
        capacity = self.values.numel()
        n = losses.numel()
        if n >= capacity:
            self.values.copy_(losses[-capacity:])
            self.count.fill_(capacity)
            self.cursor.zero_()
            return
        cursor = int(self.cursor.item())
        first = min(n, capacity - cursor)
        self.values[cursor:cursor + first].copy_(losses[:first])
        if first < n:
            self.values[:n - first].copy_(losses[first:])
        self.count.fill_(min(capacity, int(self.count.item()) + n))
        self.cursor.fill_((cursor + n) % capacity)

    def z_score(self, losses: Tensor) -> Tensor:
        count = int(self.count.item())
        if count <= 2:
            return torch.zeros_like(losses, dtype=torch.float32)
        values = self.values[:count]
        std, mean = torch.std_mean(values, unbiased=True)
        # Equal observations yield zero; a changed observation remains detectable.
        std = std.clamp_min(torch.finfo(torch.float32).eps)
        return ((losses.float() - mean) / std).abs()


class SEMAMixture(nn.Module):
    """Source-derived SEMA mixture, initially containing one trainable expert.

    Public module lists ``functional``, ``rd``, and ``router`` separate optimizer
    parameter groups. Rebuild optimizers after ``expand()``; old parameters are
    frozen and have their accumulated gradients cleared. ``threshold`` is stored
    policy metadata: the caller decides whether ``expansion_score(x, mask) >
    threshold`` permits expansion under its task/layer protocol.

    ``load_state_dict`` also works as part of a parent module: it reconstructs the
    dynamic expert count before loading child tensors and restores per-parameter
    requires-grad flags. Use ``from_state_dict`` to reconstruct a standalone
    mixture without separately storing its constructor arguments.
    """

    def __init__(self, d_model: int, bottleneck: int = 16, rd_dim: int = 64,
                 threshold: float = 1.5, buffer_size: int = 500) -> None:
        super().__init__()
        if min(d_model, bottleneck, rd_dim, buffer_size) <= 0:
            raise ValueError("Model dimensions and buffer_size must be positive")
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.rd_dim = rd_dim
        self.threshold = float(threshold)
        self.buffer_size = buffer_size
        self.functional = nn.ModuleList()
        self.rd = nn.ModuleList()
        self.router = nn.ModuleList()
        self.stats = nn.ModuleList()
        self._append_expert()

    @property
    def num_experts(self) -> int:
        return len(self.functional)

    def _append_expert(self) -> None:
        reference = next(self.parameters(), None)
        modules = (_Functional(self.d_model, self.bottleneck),
                   _Descriptor(self.d_model, self.rd_dim),
                   nn.Linear(self.d_model, 1), _Records(self.buffer_size))
        for collection, module in zip(
                (self.functional, self.rd, self.router, self.stats), modules):
            if reference is not None:
                module.to(device=reference.device, dtype=reference.dtype)
            module.train(self.training)
            collection.append(module)

    def _new_rd_is_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.rd[-1].parameters())

    def forward(self, x: Tensor, mask: Tensor | None = None,
                context: Tensor | None = None) -> Tensor:
        """Return additive delta; optional cached prompt context supports decoding."""
        pooled = _masked_mean(x, mask) if context is None else context
        logits = torch.cat([row(pooled) for row in self.router], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        # Avoid stacking a full [experts, batch, sequence, width] activation tensor.
        result = self.functional[0](x) * weights[:, 0, None, None]
        for index in range(1, self.num_experts):
            result = result + self.functional[index](x) * weights[:, index, None, None]
        return result

    def rd_loss(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Scalar newest-RD loss, with no gradient into the input/backbone.

        Returns a detached float32 scalar zero if all RDs are frozen; callers
        should skip RD backward/optimizer work in that case. Does not observe.
        """
        if not self._new_rd_is_trainable():
            return x.new_zeros((), dtype=torch.float32)
        return self.rd[-1](_masked_mean(x.detach(), mask)).mean()

    @torch.no_grad()
    def observe(self, x: Tensor, mask: Tensor | None = None) -> None:
        """Record newest trainable RD per-sample losses; ignore frozen RDs."""
        if self._new_rd_is_trainable():
            self.stats[-1].observe(self.rd[-1](_masked_mean(x.detach(), mask)))

    @torch.no_grad()
    def expansion_score(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Minimum across experts of mean absolute per-sample z-scores.

        Matches upstream's >2 observations warmup; scoring never records samples.
        The caller compares this scalar against ``threshold`` and controls growth.
        """
        pooled = _masked_mean(x.detach(), mask)
        scores = [records.z_score(descriptor(pooled)).mean()
                  for descriptor, records in zip(self.rd, self.stats)]
        return torch.stack(scores).min()

    def freeze(self) -> None:
        """Freeze every expert, descriptor and router row, clearing old grads."""
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None

    def expand(self) -> int:
        """Freeze all old state and add a trainable pair/router row; return index."""
        self.freeze()
        self._append_expert()
        return self.num_experts - 1

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": {"d_model": self.d_model, "bottleneck": self.bottleneck,
                       "rd_dim": self.rd_dim, "threshold": self.threshold,
                       "buffer_size": self.buffer_size},
            "num_experts": self.num_experts,
            "requires_grad": {name: parameter.requires_grad
                              for name, parameter in self.named_parameters()},
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.threshold = float(state["config"]["threshold"])
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(state["requires_grad"][name])
            parameter.grad = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state = state_dict.get(prefix + "_extra_state")
        if state is not None:
            if state["version"] != 1:
                raise ValueError("Unsupported SEMAMixture checkpoint version")
            for name in ("d_model", "bottleneck", "rd_dim", "buffer_size"):
                if state["config"][name] != getattr(self, name):
                    raise ValueError(f"SEMAMixture checkpoint {name} mismatch")
            count = state["num_experts"]
            if count < 1:
                raise ValueError("SEMAMixture checkpoint must contain an expert")
            while self.num_experts < count:
                self._append_expert()
            for collection in (self.functional, self.rd, self.router, self.stats):
                del collection[count:]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any], *, device=None,
                        dtype=None) -> "SEMAMixture":
        """Reconstruct an isolated mixture checkpoint, strictly loading all state."""
        mixture = cls(**state_dict["_extra_state"]["config"])
        reference = state_dict["functional.0.down_proj.weight"]
        mixture.to(device=reference.device if device is None else device,
                   dtype=reference.dtype if dtype is None else dtype)
        mixture.load_state_dict(state_dict)
        return mixture
