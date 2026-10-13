"""SEMA in BLIP's ViT; the pretrained VQA readout remains trainable.

Block insertion follows SEMA backbone/vit_sema.py:86-104, not an LLM MLP.
BLIP components and loss follow its official models/blip_vqa.py. Only pure
pretraining weights are accepted by the acquisition script. Both source repos
and licenses are preserved under third_party/. Legacy experiments are untouched.
"""
import itertools
from pathlib import Path
import sys

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BertTokenizer

from .expansion import SEMAMixture, _Functional

BLIP_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "BLIP"
sys.path.insert(0, str(BLIP_ROOT))
from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel


class VisionAdapterBlock(nn.Module):
    """Parallel adapter consumes the post-attention residual BEFORE norm2."""
    def __init__(self, base, expandable, threshold):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.mixture = SEMAMixture(768, 16, 128, threshold, 500) if expandable else None
        self.adapter = None if expandable else _Functional(768, 16)
        self.capture = False
        self.pooled = None

    def forward(self, x, register_hook=False):
        x = x + self.base.drop_path(self.base.attn(self.base.norm1(x), register_hook=register_hook))
        if self.capture:
            self.pooled = x.detach().float().mean(dim=1, keepdim=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = self.mixture(x.float()) if self.mixture is not None else self.adapter(x.float())
        return x + self.base.drop_path(self.base.mlp(self.base.norm2(x))) + delta


class VisionVQA(nn.Module):
    def __init__(self, assets, image_size=224, threshold=1.0, device="cuda"):
        super().__init__()
        assets = Path(assets)
        self.device = torch.device(device)
        self.image_size = image_size
        self.visual_encoder = VisionTransformer(img_size=image_size, patch_size=16,
                                                embed_dim=768, depth=12, num_heads=12,
                                                drop_path_rate=0.0)
        config = BertConfig.from_json_file(str(BLIP_ROOT / "configs" / "med_config.json"))
        self.text_encoder = BertModel(config=config, add_pooling_layer=False)
        self.text_decoder = BertLMHeadModel(config=config)
        self.tokenizer = BertTokenizer.from_pretrained(str(assets / "tokenizer"), local_files_only=True)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        self.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]
        checkpoint = torch.load(assets / "model_base_capfilt_large.pth", map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        state["visual_encoder.pos_embed"] = interpolate_pos_embed(state["visual_encoder.pos_embed"], self.visual_encoder)
        # The official pretraining checkpoint includes momentum encoders, queues,
        # projections and ITM heads. They are not part of the official VQA model.
        current = self.state_dict()
        usable = {key: value for key, value in state.items() if key in current}
        missing = set(current) - set(usable)
        if missing:
            raise ValueError(f"Pretraining checkpoint misses VQA parameters: {sorted(missing)}")
        self.load_state_dict(usable, strict=True)
        del checkpoint, state, usable, current
        self.visual_encoder.requires_grad_(False)
        for index, block in enumerate(self.visual_encoder.blocks):
            self.visual_encoder.blocks[index] = VisionAdapterBlock(block, index >= 9, threshold)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.to(self.device)

    @property
    def mixtures(self):
        return [block.mixture for block in self.visual_encoder.blocks[9:]]

    @property
    def expert_counts(self):
        return [mixture.num_experts for mixture in self.mixtures]

    def train(self, mode=True):
        super().train(mode)
        for block in self.visual_encoder.blocks:
            block.base.eval()
        return self

    def head_parameters(self):
        return (p for p in itertools.chain(self.text_encoder.parameters(), self.text_decoder.parameters()) if p.requires_grad)

    def functional_parameters(self):
        for block in self.visual_encoder.blocks:
            modules = (block.adapter,) if block.adapter is not None else (block.mixture.functional, block.mixture.router)
            for module in modules:
                yield from (p for p in module.parameters() if p.requires_grad)

    def freeze_vision(self):
        self.visual_encoder.requires_grad_(False)
        for parameter in self.visual_encoder.parameters():
            parameter.grad = None

    def prepare_task(self, method, task_index):
        self.text_encoder.requires_grad_(True)
        self.text_decoder.requires_grad_(True)
        if task_index > 0:
            self.freeze_vision()
        if task_index == 0 or method == "sequential":
            for block in self.visual_encoder.blocks:
                if block.adapter is not None:
                    block.adapter.requires_grad_(True)
                else:
                    block.mixture.functional.requires_grad_(True)
                    block.mixture.router.requires_grad_(True)
        if task_index == 0 and method in ("sema", "task_expansion"):
            for mixture in self.mixtures:
                mixture.rd.requires_grad_(True)

    def _images(self, rows, image_root):
        images = []
        for row in rows:
            with Image.open(Path(image_root) / row["image"]) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images).to(self.device)

    def _tokens(self, texts, first_token):
        tokens = self.tokenizer(texts, padding=True, return_tensors="pt", truncation=False)
        if tokens.input_ids.shape[1] > 512:
            raise ValueError("BLIP text exceeds 512 positions; do not silently truncate questions/options")
        tokens.input_ids[:, 0] = first_token
        return tokens.to(self.device)

    def _question_states(self, rows, image_root):
        images = self._images(rows, image_root)
        image_states = self.visual_encoder(images)
        questions = self._tokens([r["question"].replace("<image>", "").strip() for r in rows], self.enc_token_id)
        output = self.text_encoder(questions.input_ids, attention_mask=questions.attention_mask,
                                   encoder_hidden_states=image_states,
                                   encoder_attention_mask=torch.ones(image_states.shape[:2], dtype=torch.long, device=self.device),
                                   return_dict=True)
        return output.last_hidden_state, questions.attention_mask

    def loss(self, rows, image_root):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            states, mask = self._question_states(rows, image_root)
            answers = self._tokens([r["answer"] for r in rows], self.tokenizer.bos_token_id)
            targets = answers.input_ids.masked_fill(answers.attention_mask == 0, -100)
            output = self.text_decoder(answers.input_ids, attention_mask=answers.attention_mask,
                                       encoder_hidden_states=states, encoder_attention_mask=mask,
                                       labels=targets, reduction="none", return_dict=True)
            # Official BLIP: sum token losses per answer, mean over questions.
            return output.loss.float().mean()

    @torch.no_grad()
    def features(self, rows, image_root):
        for block in self.visual_encoder.blocks[9:]:
            block.capture = True
        try:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                self.visual_encoder(self._images(rows, image_root))
            return [block.pooled for block in self.visual_encoder.blocks[9:]]
        finally:
            for block in self.visual_encoder.blocks[9:]:
                block.capture = False
                block.pooled = None

    @torch.no_grad()
    def answer(self, rows, image_root):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            states, mask = self._question_states(rows, image_root)
            token = torch.full((len(rows), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
            generated = []
            finished = torch.zeros(len(rows), dtype=torch.bool, device=self.device)
            past = None
            # Explicit cache avoids upstream's version-dependent generate hook.
            for index in range(32):
                output = self.text_decoder(token, past_key_values=past, use_cache=True,
                                           attention_mask=torch.ones((len(rows), index + 1), dtype=torch.long, device=self.device),
                                           encoder_hidden_states=states, encoder_attention_mask=mask, return_dict=True)
                token = output.logits[:, -1].argmax(dim=-1)
                token = torch.where(finished, self.tokenizer.pad_token_id, token)
                generated.append(token)
                finished |= token == self.tokenizer.sep_token_id
                past = output.past_key_values
                token = token[:, None]
                if bool(finished.all()):
                    break
        return self.tokenizer.batch_decode(torch.stack(generated, dim=1), skip_special_tokens=True)

    def checkpoint_state(self):
        return {"text_encoder": self.text_encoder.state_dict(), "text_decoder": self.text_decoder.state_dict(),
                "adapters": [block.adapter.state_dict() if block.adapter is not None else block.mixture.state_dict()
                             for block in self.visual_encoder.blocks]}

    def load_checkpoint_state(self, state):
        self.text_encoder.load_state_dict(state["text_encoder"], strict=True)
        self.text_decoder.load_state_dict(state["text_decoder"], strict=True)
        if len(state["adapters"]) != 12:
            raise ValueError("Expected all twelve vision adapter states")
        for block, saved in zip(self.visual_encoder.blocks, state["adapters"]):
            (block.adapter if block.adapter is not None else block.mixture).load_state_dict(saved, strict=True)
        self.freeze_vision()

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "vision_adapters": sum(p.numel() for block in self.visual_encoder.blocks
                                       for p in (block.adapter if block.adapter is not None else block.mixture).parameters()),
                "readout": sum(p.numel() for p in itertools.chain(self.text_encoder.parameters(), self.text_decoder.parameters()))}
