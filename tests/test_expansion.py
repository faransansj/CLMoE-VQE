import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from sema_vqa.expansion import SEMAMixture


def test_matches_official_adapter_and_autoencoder():
    path = Path(__file__).parents[1] / "third_party/SEMA-CL/backbone/sema_components.py"
    spec = importlib.util.spec_from_file_location("official_components", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    mixture = SEMAMixture(8, bottleneck=3, rd_dim=2)
    adapter = source.Adapter(d_model=8, bottleneck=3, init_option="lora", adapter_layernorm_option="none")
    descriptor = source.AE(SimpleNamespace(d_model=8, rd_dim=2))
    with torch.no_grad():
        mixture.functional[0].up_proj.weight.normal_()
    adapter.load_state_dict(mixture.functional[0].state_dict())
    descriptor.load_state_dict(mixture.rd[0].state_dict())
    x = torch.randn(4, 7, 8)
    torch.testing.assert_close(mixture(x), adapter(x))
    torch.testing.assert_close(mixture.rd_loss(x), descriptor.compute_reconstruction_loss(x).mean())


def test_expansion_preserves_old_parameters_under_weight_decay():
    mixture = SEMAMixture(8, bottleneck=3, rd_dim=2)
    mixture.observe(torch.randn(5, 3, 8))
    old_parameters = {name: p.detach().clone() for name, p in mixture.named_parameters()}
    old_stats = mixture.stats[0].values.clone()
    mixture.expand()
    mixture.train()
    optimizer = torch.optim.AdamW(mixture.parameters(), lr=0.01, weight_decay=0.2)
    x = torch.randn(5, 3, 8)
    (mixture(x).sum() + mixture.rd_loss(x)).backward()
    optimizer.step()
    mixture.observe(x)
    for name, old in old_parameters.items():
        parameter = dict(mixture.named_parameters())[name]
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
    torch.testing.assert_close(mixture.stats[0].values, old_stats, rtol=0, atol=0)
    assert mixture.functional[1].up_proj.weight.abs().sum() > 0


def test_answer_tokens_cannot_affect_prompt_routing_or_descriptor():
    mixture = SEMAMixture(8, bottleneck=3, rd_dim=2)
    mixture.expand()
    with torch.no_grad():
        for expert in mixture.functional:
            expert.up_proj.weight.normal_()
    x = torch.randn(3, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]] * 3)
    changed = x.detach().clone()
    changed[:, 3:] = 1000 * torch.randn_like(changed[:, 3:])
    torch.testing.assert_close(mixture(x, mask)[:, :3], mixture(changed, mask)[:, :3])
    torch.testing.assert_close(mixture.rd_loss(x, mask), mixture.rd_loss(changed, mask))
    mixture.rd_loss(x, mask).backward()
    assert x.grad is None
    assert mixture.rd[-1].encoder.weight.grad is not None


def test_dynamic_checkpoint_preserves_outputs_freezing_and_scores():
    mixture = SEMAMixture(8, bottleneck=3, rd_dim=2, buffer_size=7)
    mixture.observe(torch.randn(5, 2, 8))
    mixture.expand()
    mixture.observe(torch.randn(6, 2, 8))
    mixture.freeze()
    state = copy.deepcopy(mixture.state_dict())
    restored = SEMAMixture.from_state_dict(state)
    x = torch.randn(4, 3, 8)
    torch.testing.assert_close(restored(x), mixture(x))
    torch.testing.assert_close(restored.expansion_score(x), mixture.expansion_score(x))
    assert not any(p.requires_grad for p in restored.parameters())
    parent = torch.nn.ModuleDict({"mix": SEMAMixture(8, bottleneck=3, rd_dim=2, buffer_size=7)})
    parent.load_state_dict({"mix." + k: v for k, v in state.items()})
    assert parent["mix"].num_experts == 2
    torch.testing.assert_close(parent["mix"].expansion_score(x), mixture.expansion_score(x))


def test_rolling_window_and_float32_statistics():
    mixture = SEMAMixture(2, buffer_size=5)
    records = mixture.stats[0]
    records.observe(torch.tensor([1., 2., 3.]))
    records.observe(torch.tensor([4., 5., 6., 7.]))
    torch.testing.assert_close(records.values.sort().values, torch.tensor([3., 4., 5., 6., 7.]))
    expected = ((torch.tensor([2., 8.]) - 5) / torch.tensor([3., 4., 5., 6., 7.]).std()).abs()
    torch.testing.assert_close(records.z_score(torch.tensor([2., 8.])), expected)
    mixture.to(dtype=torch.bfloat16)
    assert records.values.dtype == torch.float32
    records.observe(torch.ones(5))
    assert torch.isfinite(records.z_score(torch.tensor([1., 2.]))).all()
    assert records.z_score(torch.tensor([1.])).item() == 0
