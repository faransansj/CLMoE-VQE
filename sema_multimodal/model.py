"""The archived BLIP readout graph with token-balanced heterogeneous targets."""
import torch

from sema_vqa.readout_model import ReadoutMoEVQA

from .data import CAPTION_INSTRUCTION


class MultimodalReadout(ReadoutMoEVQA):
    """One shared frozen backbone; only loss/generation differ from VQA.

    Public rows contain only ``image`` and ``question``. Training adds ``answer``.
    Caption generation limits are chosen from the public instruction, never
    task/family/source metadata. The inherited decoder objective is BLIP's
    label-smoothed shifted token sum; its verified normalization counts EOS,
    excludes BOS and padding, and is applied to both families here.
    """

    def loss_per_example(self, rows, image_root):
        return super().loss_per_example(rows, image_root, normalize_tokens=True)

    def loss(self, rows, image_root):
        return self.loss_per_example(rows, image_root).mean()

    def target_token_counts(self, rows):
        tokens = self._tokens([row["answer"] for row in rows], self.tokenizer.bos_token_id)
        return tokens.attention_mask[:, 1:].sum(dim=1)

    @staticmethod
    def generation_limit(row):
        return 64 if row["question"].strip() == CAPTION_INSTRUCTION else 32

    @torch.no_grad()
    def generate(self, rows, image_root):
        """Cached greedy decoding with per-request EOS/length stopping.

        A mixed replay/evaluation batch can contain both families. Rows reaching
        their own limit are padded while the other rows continue with the same
        cache; no family/task label is accepted by this method.
        """
        if not rows:
            return [], []
        limits = torch.tensor([self.generation_limit(row) for row in rows], device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            states, mask = self._question_states(rows, image_root)
            token = torch.full((len(rows), 1), self.tokenizer.bos_token_id,
                               dtype=torch.long, device=self.device)
            finished = torch.zeros(len(rows), dtype=torch.bool, device=self.device)
            eos = torch.zeros_like(finished)
            lengths = torch.zeros(len(rows), dtype=torch.long, device=self.device)
            generated, past = [], None
            for index in range(int(limits.max())):
                output = self.text_decoder(
                    token, past_key_values=past, use_cache=True,
                    attention_mask=torch.ones((len(rows), index + 1), dtype=torch.long, device=self.device),
                    encoder_hidden_states=states, encoder_attention_mask=mask, return_dict=True)
                next_token = output.logits[:, -1].argmax(dim=-1)
                next_token = torch.where(finished, self.tokenizer.pad_token_id, next_token)
                generated.append(next_token)
                lengths += (~finished).long()
                reached_eos = (~finished) & (next_token == self.tokenizer.sep_token_id)
                eos |= reached_eos
                finished |= reached_eos | (lengths >= limits)
                past, token = output.past_key_values, next_token[:, None]
                if bool(finished.all()):
                    break
        tokens = torch.stack(generated, dim=1)
        predictions = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        diagnostics = []
        for prediction, length, limit, ended, values in zip(
                predictions, lengths.tolist(), limits.tolist(), eos.tolist(), tokens.tolist()):
            valid = values[:length]
            repeated = any(valid[pos:pos + 3] == valid[pos + 3:pos + 6]
                           for pos in range(max(0, len(valid) - 5)))
            diagnostics.append({"generated_tokens": length, "max_new_tokens": limit,
                                "eos": ended, "empty": not prediction.strip(),
                                "repeated_trigram": repeated, "limit_reached": length == limit})
        return predictions, diagnostics

    def answer(self, rows, image_root):
        return self.generate(rows, image_root)[0]
