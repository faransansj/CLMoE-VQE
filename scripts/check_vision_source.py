"""GPU parity against untouched official SEMAModules; not an accuracy result."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from sema_vqa.expansion import SEMAMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/SEMA-CL"))
from backbone.sema_block import SEMAModules


def main():
    torch.manual_seed(19)
    config = SimpleNamespace(d_model=768, ffn_num=16, ffn_adapter_init_option="lora",
                             ffn_adapter_scalar="0.1", ffn_adapter_layernorm_option="none",
                             adapt_start_layer=9, adapt_end_layer=11, rd_dim=128,
                             buffer_size=500, exp_threshold=1.0)
    source = SEMAModules(config, 9, None).cuda().eval()
    ours = SEMAMixture(768, 16, 128, 1.0, 500).cuda().eval()
    records = []
    for count in (1, 2, 3):
        if count > 1:
            source.add_adapter()
            source.fix_router()
            ours.expand()
        with torch.no_grad():
            for index in range(count):
                torch.nn.init.normal_(ours.functional[index].up_proj.weight, std=0.03)
                source.adapters[index].functional.load_state_dict(ours.functional[index].state_dict())
                source.adapters[index].rd.load_state_dict(ours.rd[index].state_dict())
                history = torch.linspace(0.1, 0.5, 500) * (index + 1)
                source.adapters[index].rd_loss_record.add_record(history)
                ours.stats[index].observe(history)
            source.router.weight.copy_(torch.cat([row.weight for row in ours.router]))
            source.router.bias.copy_(torch.cat([row.bias for row in ours.router]))
        source.eval()
        ours.eval()
        x = torch.randn(8, 17, 768, device="cuda")
        with torch.no_grad():
            expected = source(x)["func_out"]
            actual = ours(x)
            source_scores = torch.stack([adapter.get_z_score_deviation(adapter.rd.compute_reconstruction_loss(x)) for adapter in source.adapters]).mean(dim=1).min()
            actual_score = ours.expansion_score(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_score, source_scores, rtol=1e-5, atol=1e-6)
        records.append({"experts": count, "max_forward_error": float((actual - expected).abs().max()),
                        "source_z_score": float(source_scores), "implementation_z_score": float(actual_score)})
    print(json.dumps({"official_SEMA_parity": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
