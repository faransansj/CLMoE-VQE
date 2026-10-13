from types import SimpleNamespace

import torch
from torch import nn

from sema_vqa.vision_run import optimizer_step
from sema_vqa.vision_model import VisionAdapterBlock, VisionVQA


def test_partial_accumulation_matches_full_batch():
    x = torch.tensor([[1., 2.], [3., 1.], [2., -1.], [0., 3.], [-1., 4.]])
    y = torch.tensor([[2.], [1.], [0.], [3.], [4.]])
    full = nn.Linear(2, 1)
    micro = nn.Linear(2, 1)
    micro.load_state_dict(full.state_dict())
    first = torch.optim.SGD(full.parameters(), lr=0.03)
    second = torch.optim.SGD(micro.parameters(), lr=0.03)
    expected = (full(x) - y).square().mean()
    expected.backward()
    first.step()
    observed = optimizer_step(lambda ids: (micro(x[ids]) - y[ids]).square().mean(), list(range(5)), 2, [second])
    torch.testing.assert_close(torch.tensor(observed), expected.detach())
    for left, right in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(left, right)


class Attention(nn.Module):
    def forward(self, x, register_hook=False):
        return 0.25 * x


class BaseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(768), nn.LayerNorm(768)
        self.attn = Attention()
        self.drop_path = nn.Identity()
        self.mlp = nn.Linear(768, 768)


def test_adapter_consumes_post_attention_residual_not_normalized_input():
    block = VisionAdapterBlock(BaseBlock(), False, 1.0)
    nn.init.normal_(block.adapter.up_proj.weight, std=0.1)
    x = torch.randn(2, 4, 768) * 3 + 2
    residual = x + block.base.attn(block.base.norm1(x))
    expected = residual + block.base.mlp(block.base.norm2(residual)) + block.adapter(residual)
    torch.testing.assert_close(block(x), expected)
    wrong = residual + block.base.mlp(block.base.norm2(residual)) + block.adapter(block.base.norm2(residual))
    assert not torch.allclose(block(x), wrong)


def test_no_expansion_keeps_readout_learning_but_freezes_vision():
    # Exercise the real task policy without downloading a pretrained model.
    model = VisionVQA.__new__(VisionVQA)
    nn.Module.__init__(model)
    model.visual_encoder = nn.Module()
    model.visual_encoder.blocks = nn.ModuleList([VisionAdapterBlock(BaseBlock(), i >= 9, 1.0) for i in range(12)])
    model.text_encoder = nn.Linear(2, 2)
    model.text_decoder = nn.Linear(2, 1)
    model.prepare_task("sema", 1)
    assert not list(model.functional_parameters())
    assert all(not p.requires_grad for p in model.visual_encoder.parameters())
    head = list(model.head_parameters())
    before = [p.detach().clone() for p in head]
    optimizer = torch.optim.SGD(head, lr=0.01)
    loss_fn = lambda rows: (model.text_decoder(model.text_encoder(torch.ones(len(rows), 2))) - 1).square().mean()
    old_loss = float(loss_fn([0, 1]).detach())
    optimizer_step(loss_fn, [0, 1], 1, [optimizer])
    assert float(loss_fn([0, 1]).detach()) < old_loss
    assert any(not torch.equal(a, b) for a, b in zip(before, head))
