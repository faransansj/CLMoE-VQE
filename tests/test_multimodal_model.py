"""Behavioral loss, decode, replay and resume invariants without BLIP assets."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sema_multimodal.data import CAPTION_INSTRUCTION
from sema_multimodal.model import MultimodalReadout
from sema_multimodal.run import epoch_batches, optimizer_step, capture_rng, restore_rng, safe_output
from sema_vqa.vision_model import VisionVQA


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)

    def forward(self, values):
        return self.query(values) + self.value(values)


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(32, 4)
        self.attention = Attention()
        self.crossattention = Attention()
        self.head = nn.Linear(4, 32)

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, return_dict=True, mode="multimodal",
                labels=None, reduction="none", **kwargs):
        hidden = self.attention(self.embeddings(input_ids))
        if mode != "text":
            if encoder_attention_mask is None:
                pooled = encoder_hidden_states.mean(dim=1, keepdim=True)
            else:
                mask = encoder_attention_mask[:, :, None]
                pooled = (encoder_hidden_states * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
            hidden = hidden + self.crossattention(pooled)
        logits = self.head(hidden)
        loss = None if labels is None else F.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none", label_smoothing=0.1).sum(dim=1)
        return SimpleNamespace(last_hidden_state=hidden, logits=logits, loss=loss)


@pytest.fixture
def make_model(monkeypatch):
    def initialize(self, assets, image_size=224, device="cpu"):
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.enc_token_id = 1
        self.tokenizer = SimpleNamespace(bos_token_id=1, sep_token_id=2, pad_token_id=0)
        self.visual_encoder = nn.Linear(4, 4)
        self.text_encoder, self.text_decoder = Text(), Text()

        def images(rows, image_root):
            return torch.stack([torch.arange(8, dtype=torch.float32).reshape(2, 4)
                                + len(row["image"]) for row in rows])

        def tokens(texts, first_token):
            values = [[first_token] + [3 + sum(map(ord, word)) % 29 for word in text.split()] + [2] for text in texts]
            ids = torch.zeros((len(values), max(map(len, values))), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, value in enumerate(values):
                ids[index, :len(value)] = torch.tensor(value)
                mask[index, :len(value)] = 1
            return SimpleNamespace(input_ids=ids, attention_mask=mask)

        self._images, self._tokens = images, tokens

    monkeypatch.setattr(VisionVQA, "__init__", initialize)

    def build():
        torch.manual_seed(31)
        return MultimodalReadout("not-needed", rank=2, alpha=4, device="cpu")

    return build


@pytest.fixture
def rows():
    return [{"image": "one.jpg", "question": "What color?", "answer": "red"},
            {"image": "two.jpg", "question": CAPTION_INSTRUCTION,
             "answer": "a bright red car near a large tree"},
            {"image": "three.jpg", "question": CAPTION_INSTRUCTION, "answer": ""}]


def test_target_token_mean_matches_smoothed_shifted_loss_including_eos(make_model, rows):
    model = make_model()
    states, mask = model._question_states(rows, "unused")
    answers = model._tokens([row["answer"] for row in rows], model.tokenizer.bos_token_id)
    targets = answers.input_ids.masked_fill(answers.attention_mask == 0, -100)
    logits = model.text_decoder(answers.input_ids, encoder_hidden_states=states, encoder_attention_mask=mask).logits
    per_token = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets[:, 1:],
                                reduction="none", label_smoothing=0.1)
    counts = (targets[:, 1:] != -100).sum(1)
    assert counts.tolist() == [2, 9, 1]  # EOS alone remains supervised for empty target.
    torch.testing.assert_close(model.target_token_counts(rows), counts)
    expected = per_token.sum(1) / counts
    torch.testing.assert_close(model.loss_per_example(rows, "unused"), expected)
    torch.testing.assert_close(model.loss(rows, "unused"), expected.mean())
    alone = model.loss_per_example([rows[0]], "unused")
    torch.testing.assert_close(alone[0], expected[0])  # Caption padding cannot dilute VQA loss.


def test_unequal_microbatches_have_effective_batch_example_mean(make_model, rows):
    full, split = make_model(), make_model()
    for model in (full, split):
        torch.manual_seed(11)
        model.add_expert()
        model.train_expert(0)
    full_optimizer = torch.optim.SGD(list(full.expert_parameters(0)), lr=0.01)
    split_optimizer = torch.optim.SGD(list(split.expert_parameters(0)), lr=0.01)
    full_loss, full_tokens = optimizer_step(full, rows, "unused", 3, full_optimizer)
    split_loss, split_tokens = optimizer_step(split, rows, "unused", 2, split_optimizer)
    assert full_loss == pytest.approx(split_loss, abs=1e-6)
    assert full_tokens == split_tokens == 12
    for first, second in zip(full.expert_parameters(0), split.expert_parameters(0)):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)


class GreedyDecoder(nn.Module):
    def __init__(self, eos_at=None):
        super().__init__()
        self.calls = []
        self.eos_at = eos_at

    def forward(self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs):
        index = len(self.calls)
        self.calls.append((input_ids.clone(), past_key_values, attention_mask.shape, use_cache))
        logits = torch.full((len(input_ids), 1, 8), -10.0)
        logits[:, 0, 3] = 10.0
        if self.eos_at is not None and index >= self.eos_at:
            logits[0, 0, 2] = 20.0
        return SimpleNamespace(logits=logits, past_key_values=(index,))


def configure_generation(model, decoder):
    model.text_decoder = decoder
    model._question_states = lambda rows, root: (torch.zeros((len(rows), 1, 4)), torch.ones((len(rows), 1), dtype=torch.long))
    model.tokenizer.batch_decode = lambda values, skip_special_tokens=True: [
        " ".join(str(token) for token in row.tolist() if token not in (0, 1, 2)) for row in values]


def test_mixed_public_requests_stop_at_32_and_64_using_cache_not_family(make_model):
    model = make_model()
    decoder = GreedyDecoder()
    configure_generation(model, decoder)
    requests = [{"image": "x", "question": "What?", "family": "caption", "task_id": "secret"},
                {"image": "y", "question": CAPTION_INSTRUCTION, "family": "vqa"}]
    predictions, info = model.generate(requests, "unused")
    assert [len(prediction.split()) for prediction in predictions] == [32, 64]
    assert [row["generated_tokens"] for row in info] == [32, 64]
    assert all(not row["eos"] and row["limit_reached"] for row in info)
    assert len(decoder.calls) == 64
    assert decoder.calls[0][1] is None
    for index, (token, past, shape, cached) in enumerate(decoder.calls):
        assert token.shape == (2, 1) and shape == (2, index + 1) and cached
        if index:
            assert past == (index - 1,)
    assert decoder.calls[33][0][0, 0] == 0  # Finished VQA row is padded while caption continues.


def test_eos_stopping_does_not_generate_after_finished_row(make_model):
    model = make_model()
    decoder = GreedyDecoder(eos_at=0)
    configure_generation(model, decoder)
    predictions, info = model.generate([{"image": "a", "question": "What?"},
                                        {"image": "b", "question": CAPTION_INSTRUCTION}], "unused")
    assert predictions[0] == ""
    assert info[0]["eos"] and info[0]["generated_tokens"] == 1 and info[0]["empty"]
    assert not info[0]["limit_reached"]
    assert decoder.calls[1][0][0, 0] == 2
    assert decoder.calls[2][0][0, 0] == 0
    empty_predictions, empty_info = model.generate([], "unused")
    assert empty_predictions == empty_info == []


def test_partial_replay_keeps_steps_images_and_global_rng():
    current = [{"id": str(index)} for index in range(5)]
    memory = [{"id": "old"}]
    before = capture_rng()
    batches = epoch_batches(current, memory, seed=17, task=1, epoch=0, batch_size=3)
    after = capture_rng()
    assert before["python"] == after["python"] and torch.equal(before["torch"], after["torch"])
    assert [len(batch) for batch, _, _ in batches] == [3, 2]
    assert [(current_count, replay_count) for _, current_count, replay_count in batches] == [(2, 1), (1, 1)]
    assert sum(len(batch) for batch, _, _ in batches) == len(current)
    assert batches == epoch_batches(current, memory, seed=17, task=1, epoch=0, batch_size=3)
    assert [row["id"] for row in batches[1][0]].count("old") == 1


def test_optimizer_model_rng_roundtrip_continues_identical_next_batch(make_model, rows, tmp_path):
    model = make_model()
    model.add_expert()
    model.train_expert(0)
    optimizer = torch.optim.AdamW(list(model.expert_parameters(0)), lr=0.001)
    optimizer_step(model, rows[:2], "unused", 1, optimizer)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.checkpoint_state(), "optimizer": optimizer.state_dict(), "rng": capture_rng()}, checkpoint)
    optimizer_step(model, rows[1:], "unused", 1, optimizer)
    expected = model.checkpoint_state()
    expected_random = torch.rand(3)
    restored = make_model()
    saved = torch.load(checkpoint, weights_only=False)
    restored.load_checkpoint_state(saved["model"])
    resumed_optimizer = torch.optim.AdamW(list(restored.expert_parameters(0)), lr=0.001)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    optimizer_step(restored, rows[1:], "unused", 1, resumed_optimizer)
    actual = restored.checkpoint_state()
    for name, bank in expected["projections"].items():
        for key, value in bank.items():
            assert torch.equal(value, actual["projections"][name][key])
    assert torch.equal(expected_random, torch.rand(3))
    old = [parameter.detach().clone() for parameter in restored.expert_parameters(0)]
    restored.add_expert()
    restored.train_expert(1)
    optimizer_step(restored, rows, "unused", 2,
                   torch.optim.AdamW(list(restored.expert_parameters(1)), lr=0.01))
    assert all(torch.equal(first, second) for first, second in zip(old, restored.expert_parameters(0)))
    assert all(parameter.grad is None for parameter in restored.expert_parameters(0))


def test_output_guard_never_allows_archived_namespaces(tmp_path):
    for name in ("readout_runs", "readout_report", "vision_runs", "legacy"):
        with pytest.raises(ValueError, match="read-only"):
            safe_output(tmp_path / name / "multimodal_smoke", smoke=True)
    with pytest.raises(ValueError, match="Output must"):
        safe_output(tmp_path / "unmarked-output")
    assert safe_output(tmp_path / "multimodal_smoke", smoke=True) == (tmp_path / "multimodal_smoke").resolve()
