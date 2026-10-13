"""Exercise the actual readout lifecycle without loading pretrained assets."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sema_vqa.readout_model import LoRALinear, ReadoutMoEVQA
from sema_vqa.vision_model import VisionVQA


class TinyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        return self.dropout((self.query(x) + self.key(x) + self.value(x)) / 3)


class TinyText(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embeddings = nn.Embedding(64, width)
        self.attention = TinyAttention(width)
        self.crossattention = TinyAttention(width)
        self.head = nn.Linear(width, 64)

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, return_dict=True, mode="multimodal",
                labels=None, reduction="none"):
        hidden = self.attention(self.embeddings(input_ids))
        if mode == "multimodal":
            hidden = hidden + self.crossattention(encoder_hidden_states.mean(dim=1, keepdim=True))
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            assert reduction == "none"
            loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:],
                                   reduction="none", label_smoothing=0.1).sum(dim=1)
        return SimpleNamespace(last_hidden_state=hidden, logits=logits, loss=loss)


class TinyVision(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.7)
        self.adapter = nn.Linear(width, width)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, images):
        return self.dropout(self.projection(images)) + self.adapter(images)


@pytest.fixture
def make_model(monkeypatch):
    def build(rank=2, alpha=6):
        def init(backbone, assets, image_size=224, device="cpu"):
            nn.Module.__init__(backbone)
            backbone.device = torch.device(device)
            backbone.image_size = image_size
            backbone.enc_token_id = 1
            backbone.tokenizer = SimpleNamespace(bos_token_id=1, sep_token_id=2, pad_token_id=0)
            backbone.visual_encoder = TinyVision(4)
            backbone.text_encoder = TinyText(4)
            backbone.text_decoder = TinyText(4)
            backbone._original_projections = {
                f"{network_name}.{name}": module
                for network_name in ("text_encoder", "text_decoder")
                for name, module in getattr(backbone, network_name).named_modules()
                if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("query", "value")
            }

            def images(rows, image_root):
                return torch.stack([torch.arange(8, dtype=torch.float32).reshape(2, 4)
                                    + sum(map(ord, row["image"])) % 7 for row in rows])

            def tokens(texts, first_token):
                values = [[first_token] + [3 + sum(map(ord, word)) % 61 for word in text.split()] + [2]
                          for text in texts]
                ids = torch.zeros((len(values), max(map(len, values))), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for index, value in enumerate(values):
                    ids[index, :len(value)] = torch.tensor(value)
                    mask[index, :len(value)] = 1
                return SimpleNamespace(input_ids=ids, attention_mask=mask)

            backbone._images = images
            backbone._tokens = tokens

        monkeypatch.setattr(VisionVQA, "__init__", init)
        # Identical frozen assets for adapter-only checkpoint reconstruction.
        torch.manual_seed(21)
        return ReadoutMoEVQA("unused-assets", rank=rank, alpha=alpha, device="cpu")

    return build


@pytest.fixture
def rows():
    return [{"image": "one.jpg", "question": "<image> what color", "answer": "red", "task": "a"},
            {"image": "two.jpg", "question": "which shade is shown", "answer": "very dark blue", "task": "b"}]


def prediction(model, rows):
    states, mask = model._question_states(rows, "unused")
    answers = model._tokens([row["answer"] for row in rows], model.tokenizer.bos_token_id)
    return model.text_decoder(answers.input_ids, attention_mask=answers.attention_mask,
                              encoder_hidden_states=states, encoder_attention_mask=mask).logits.detach()


def snapshot(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def test_all_query_value_projections_share_one_frozen_backbone(make_model):
    model = make_model(rank=3, alpha=9)
    expected = {f"{network}.{attention}.{projection}"
                for network in ("text_encoder", "text_decoder")
                for attention in ("attention", "crossattention")
                for projection in ("query", "value")}
    assert set(model._projections) == expected
    assert model.num_experts == 0
    assert model.active_expert is None
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert ReadoutMoEVQA.answer is VisionVQA.answer
    original_ids = {id(parameter) for parameter in model.parameters()}
    assert model.add_expert() == 0
    for name, projection in model._projections.items():
        assert projection.base is model._original_projections[name]
        assert projection.experts[0].A.shape == (3, 4)
        assert projection.experts[0].B.shape == (4, 3)
        assert torch.count_nonzero(projection.experts[0].A)
        assert not torch.count_nonzero(projection.experts[0].B)
    assert original_ids <= {id(parameter) for parameter in model.parameters()}
    assert isinstance(model.text_encoder.attention.key, nn.Linear)
    assert isinstance(model.text_decoder.crossattention.key, nn.Linear)


def test_lora_rank_scaling_and_base_only_rectangular_projection():
    base = nn.Linear(5, 3, dtype=torch.float64)
    projection = LoRALinear(base, rank=2, alpha=10)
    x = torch.randn(2, 7, 5, dtype=torch.float64)
    original = base(x).detach()
    projection._append_expert()
    projection.active_expert = 0
    torch.testing.assert_close(projection(x), original, rtol=0, atol=0)
    expert = projection.experts[0]
    assert expert.A.shape == (2, 5)
    assert expert.B.shape == (3, 2)
    assert expert.A.dtype == base.weight.dtype
    with torch.no_grad():
        expert.B.normal_()
    expected = base(x) + ((x @ expert.A.T) @ expert.B.T) * 5
    torch.testing.assert_close(projection(x), expected)
    projection.active_expert = None
    torch.testing.assert_close(projection(x), original, rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in base.parameters())


def test_training_expansion_preserves_old_predictions_and_base(make_model, rows):
    model = make_model()
    base_prediction = prediction(model, rows)
    model.add_expert()
    model.train_expert(0)
    initial_expert = snapshot(model.expert_parameters(0))
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    # Include frozen parameters deliberately: cleared gradients must also keep
    # them safe from an optimizer with momentum/decoupled weight decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.loss(rows, "unused").backward()
        assert all(parameter.grad is not None for parameter in model.expert_parameters(0))
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        optimizer.step()
    assert any(not torch.equal(old, current) for old, current in zip(initial_expert, model.expert_parameters(0)))
    for name, old in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], old, rtol=0, atol=0)
    old_prediction = prediction(model, rows)
    old_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    assert model.add_expert() == 1
    assert model.active_expert == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    optimizer.step()  # A stale optimizer cannot update frozen old experts.
    torch.testing.assert_close(prediction(model, rows), old_prediction, rtol=0, atol=0)
    model.train_expert(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.2)
    optimizer.zero_grad(set_to_none=True)
    model.loss(rows, "unused").backward()
    assert all(parameter.grad is not None for parameter in model.expert_parameters(1))
    assert all(parameter.grad is None for parameter in model.expert_parameters(0))
    optimizer.step()
    for name, old in old_parameters.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], old, rtol=0, atol=0)
    trainability = [parameter.requires_grad for parameter in model.parameters()]
    model.set_expert(0)
    torch.testing.assert_close(prediction(model, rows), old_prediction, rtol=0, atol=0)
    model.set_expert(None)
    torch.testing.assert_close(prediction(model, rows), base_prediction, rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == trainability
    with pytest.raises(ValueError, match="permanently frozen"):
        model.train_expert(0)


def test_copied_expert_has_detached_independent_storage(make_model, rows):
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    with torch.no_grad():
        for parameter in model.expert_parameters(0):
            parameter.normal_(std=0.2)
    original = prediction(model, rows)
    assert model.add_expert(copy_from=0) == 1
    for first, second in zip(model.expert_parameters(0), model.expert_parameters(1)):
        assert first.data_ptr() != second.data_ptr()
        assert second.is_leaf and second.grad_fn is None
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.set_expert(1)
    torch.testing.assert_close(prediction(model, rows), original, rtol=0, atol=0)
    model.add_expert()
    for projection in model._projections.values():
        assert not torch.count_nonzero(projection.experts[2].B)


def test_backbone_dropout_is_disabled_in_training_mode(make_model, rows):
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    assert model.training
    for backbone in (model.visual_encoder, model.text_encoder, model.text_decoder):
        assert all(not module.training for module in backbone.modules())
    torch.testing.assert_close(prediction(model, rows), prediction(model, rows), rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in model.expert_parameters(0))
    model.eval()
    model.train()
    assert not model.text_encoder.attention.dropout.training
    assert not model.text_decoder.crossattention.dropout.training


def test_dynamic_checkpoint_roundtrip_restores_rank_selection_and_trainability(make_model, rows):
    model = make_model(rank=3, alpha=12)
    model.add_expert()
    model.add_expert(copy_from=0)
    model.train_expert(1)
    with torch.no_grad():
        for index in range(2):
            for parameter in model.expert_parameters(index):
                parameter.normal_(std=0.1)
    model.set_expert(0)  # Selection and trainability are deliberately different.
    saved = model.checkpoint_state()
    expected = prediction(model, rows)
    assert saved["num_experts"] == 2
    assert saved["active_expert"] == 0 and saved["trainable_expert"] == 1
    assert sum(value.numel() for bank in saved["projections"].values() for value in bank.values()) == model.parameter_counts()["experts"]
    assert all(key.split(".")[1] in ("A", "B") for bank in saved["projections"].values() for key in bank)
    assert all(not value.requires_grad and value.device.type == "cpu"
               for bank in saved["projections"].values() for value in bank.values())
    restored = make_model(rank=1, alpha=2)
    for _ in range(3):
        restored.add_expert()
    base = {name: parameter.detach().clone() for name, parameter in restored.named_parameters()
            if ".experts." not in name}
    restored.load_checkpoint_state(saved)
    assert restored.num_experts == 2 and restored.rank == 3 and restored.alpha == 12
    assert restored.active_expert == 0 and restored.training
    assert not any(parameter.requires_grad for parameter in restored.expert_parameters(0))
    assert all(parameter.requires_grad for parameter in restored.expert_parameters(1))
    for name, value in base.items():
        torch.testing.assert_close(dict(restored.named_parameters())[name], value, rtol=0, atol=0)
    torch.testing.assert_close(prediction(restored, rows), expected, rtol=0, atol=0)
    for index in range(2):
        model.set_expert(index)
        restored.set_expert(index)
        torch.testing.assert_close(prediction(restored, rows), prediction(model, rows), rtol=0, atol=0)
    # A checkpoint is a snapshot, not an alias to the live bank.
    saved_tensor = next(iter(next(iter(saved["projections"].values())).values()))
    expected_tensor = saved_tensor.clone()
    with torch.no_grad():
        next(model.expert_parameters(0)).add_(1)
    torch.testing.assert_close(saved_tensor, expected_tensor, rtol=0, atol=0)


def test_empty_checkpoint_can_remove_an_existing_bank(make_model):
    original = make_model(rank=3, alpha=9)
    saved = original.checkpoint_state()
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    model.load_checkpoint_state(saved)
    assert model.num_experts == 0 and model.active_expert is None
    assert not model.training
    assert model.rank == 3 and model.alpha == 9
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_counts_match_actual_parameters_and_selected_trainability(make_model):
    model = make_model(rank=2)
    base = sum(parameter.numel() for parameter in model.parameters())
    per_expert = sum(2 * (projection.base.in_features + projection.base.out_features)
                     for projection in model._projections.values())
    assert model.parameter_counts() == {"base": base, "experts": 0, "active_trainable": 0,
                                         "total": base, "trainable": 0}
    model.add_expert()
    model.add_expert()
    model.train_expert(1)
    assert model.parameter_counts() == {"base": base, "experts": 2 * per_expert,
                                         "active_trainable": per_expert,
                                         "total": base + 2 * per_expert, "trainable": per_expert}
    model.set_expert(0)
    assert model.parameter_counts()["active_trainable"] == 0
    assert model.parameter_counts()["trainable"] == per_expert
    model.set_expert(None)
    assert model.parameter_counts()["active_trainable"] == 0


def test_routing_uses_only_frozen_image_and_text_means_and_restores_state(make_model, monkeypatch, rows):
    model = make_model()
    model.add_expert()
    model.add_expert()
    model.train_expert(1)
    model.set_expert(0)
    trainability = [parameter.requires_grad for parameter in model.parameters()]
    image_states = torch.arange(2 * 3 * 768, dtype=torch.float32).reshape(2, 3, 768) / 100
    observed = {}
    tokenize = model._tokens

    def images(inputs):
        assert not model.visual_encoder.training and not torch.is_grad_enabled()
        assert all(projection.active_expert is None for projection in model._projections.values())
        return image_states

    def tokens(texts, first_token):
        observed["texts"] = texts
        return tokenize(texts, first_token)

    def text(input_ids, attention_mask, return_dict, mode):
        assert mode == "text" and return_dict
        assert not model.text_encoder.training and not torch.is_grad_enabled()
        assert all(projection.active_expert is None for projection in model._projections.values())
        # Padded positions are intentionally extreme so an unmasked average
        # cannot accidentally pass. No image states can enter this signature.
        states = torch.arange(768, dtype=torch.float32)[None, None, :] + input_ids[..., None].square()
        states = states + (1 - attention_mask[..., None]) * 100000
        observed["states"], observed["mask"] = states, attention_mask
        return SimpleNamespace(last_hidden_state=states)

    monkeypatch.setattr(model.visual_encoder, "forward", images)
    monkeypatch.setattr(model.text_encoder, "forward", text)
    monkeypatch.setattr(model, "_tokens", tokens)
    features = model.routing_features(rows, "unused")
    mask = observed["mask"].unsqueeze(-1)
    expected_text = F.normalize((observed["states"] * mask).sum(1) / mask.sum(1), dim=-1)
    expected_image = F.normalize(image_states.mean(1), dim=-1)
    torch.testing.assert_close(features, torch.cat((expected_image, expected_text), dim=-1))
    assert features.shape == (2, 1536) and not features.requires_grad
    torch.testing.assert_close(features[:, :768].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(features[:, 768:].norm(dim=-1), torch.ones(2))
    assert observed["texts"] == ["what color", "which shade is shown"]
    no_labels = [{"image": row["image"], "question": row["question"]} for row in rows]
    torch.testing.assert_close(model.routing_features(no_labels, "unused"), features, rtol=0, atol=0)
    changed = [dict(row, answer="unrelated", task="new-task", task_id=99) for row in rows]
    torch.testing.assert_close(model.routing_features(changed, "unused"), features, rtol=0, atol=0)
    assert model.active_expert == 0 and model.training
    assert [parameter.requires_grad for parameter in model.parameters()] == trainability
    assert not model.text_encoder.training and not model.visual_encoder.training


@pytest.mark.parametrize("training", [False, True])
def test_routing_restores_selection_and_mode_after_exception(make_model, monkeypatch, rows, training):
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    model.train(training)

    def fail(*args):
        raise OSError("unreadable image")

    monkeypatch.setattr(model, "_images", fail)
    with pytest.raises(OSError, match="unreadable image"):
        model.routing_features(rows, "unused")
    assert model.active_expert == 0 and model.training == training
    assert all(projection.active_expert == 0 for projection in model._projections.values())
    assert all(parameter.requires_grad for parameter in model.expert_parameters(0))
    assert not model.text_encoder.training
    assert model.routing_features([], "unused").shape == (0, 1536)
    assert model.active_expert == 0 and model.training == training


def test_per_example_loss_matches_blip_sums_and_next_token_normalization(make_model, rows):
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    logits = prediction(model, rows)
    answers = model._tokens([row["answer"] for row in rows], model.tokenizer.bos_token_id)
    targets = answers.input_ids.masked_fill(answers.attention_mask == 0, -100)
    token_losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets[:, 1:],
                                  reduction="none", label_smoothing=0.1)
    summed = model.loss_per_example(rows, "unused")
    assert summed.shape == (2,) and summed.requires_grad
    torch.testing.assert_close(summed, token_losses.sum(1))
    torch.testing.assert_close(model.loss(rows, "unused"), summed.mean())
    torch.testing.assert_close(model.loss_per_example(rows, "unused", normalize_tokens=True),
                               summed / torch.tensor([2, 4]))
    summed.mean().backward()
    assert all(parameter.grad is not None for parameter in model.expert_parameters(0))
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


@pytest.mark.parametrize("rank,alpha", [(0, 16), (-1, 16), (1.5, 16), (True, 16),
                                         (2, 0), (2, -1), (2, float("inf")), (2, float("nan"))])
def test_invalid_lora_configuration_is_rejected(rank, alpha):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(2, 3), rank=rank, alpha=alpha)


def test_invalid_selection_and_checkpoint_metadata_are_rejected(make_model):
    model = make_model()
    with pytest.raises(IndexError):
        model.set_expert(0)
    model.add_expert()
    for index in (-1, 1):
        with pytest.raises(IndexError):
            model.set_expert(index)
        with pytest.raises(IndexError):
            model.add_expert(copy_from=index)
    with pytest.raises(TypeError):
        model.set_expert(True)
    state = model.checkpoint_state()
    state["projections"] = {}
    with pytest.raises(ValueError, match="projections"):
        model.load_checkpoint_state(state)
    model.add_expert()
    state = model.checkpoint_state()
    state["trainable_expert"] = 0
    with pytest.raises(ValueError, match="permanently frozen"):
        model.load_checkpoint_state(state)
