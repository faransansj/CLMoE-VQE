"""LLaVA-1.5 stage-one composition with SEMA on the last language MLPs.

Composition, image padding, patch selection and Vicuna-v1 conversation follow
third_party/CL-MoE/llava/{conversation,mm_utils,model/llava_arch}.py.
The transfer from ViT MLPs to multimodal language MLPs is our experimental
adaptation, not an implementation claimed by either source paper.
"""
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from .expansion import SEMAMixture

SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")


class AdaptedMLP(nn.Module):
    def __init__(self, base, mixture):
        super().__init__()
        self.base = base
        self.mixture = mixture
        self.prompt_length = 0
        self.mode = "functional"
        self.rd_value = None
        self.score = None
        self.pooled = None
        self.cached_context = None

    def forward(self, x):
        adapter_x = x.float()
        if x.shape[1] == 1 and self.cached_context is not None:
            delta = self.mixture(adapter_x, context=self.cached_context)
            return self.base(x) + delta.to(dtype=x.dtype)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < self.prompt_length
        self.cached_context = (
            adapter_x.detach().masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)
        )
        if self.mode == "rd":
            self.rd_value = self.mixture.rd_loss(adapter_x, mask)
            self.mixture.observe(adapter_x, mask)
        elif self.mode == "detect":
            self.score = self.mixture.expansion_score(adapter_x, mask)
        elif self.mode == "features":
            self.pooled = self.cached_context
        return self.base(x) + self.mixture(adapter_x, mask).to(dtype=x.dtype)


class ContinualVQA(nn.Module):
    def __init__(self, assets, layers=3, bottleneck=16, rd_dim=128,
                 threshold=1.0, buffer_size=500, device="cuda"):
        super().__init__()
        assets = Path(assets)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(assets / "vicuna", use_fast=False)
        self.image_processor = CLIPImageProcessor.from_pretrained(assets / "clip")
        self.language = AutoModelForCausalLM.from_pretrained(
            assets / "vicuna", torch_dtype=dtype, attn_implementation="sdpa")
        self.vision = CLIPVisionModel.from_pretrained(assets / "clip", torch_dtype=dtype)
        self.projector = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096))
        state = torch.load(assets / "projector" / "mm_projector.bin", map_location="cpu", weights_only=True)
        state = {key.split("mm_projector.")[-1]: value for key, value in state.items()}
        self.projector.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.to(device=device, dtype=dtype)
        self.adapted = []
        for block in self.language.model.layers[-layers:]:
            mixture = SEMAMixture(self.language.config.hidden_size, bottleneck=bottleneck,
                                  rd_dim=rd_dim, threshold=threshold, buffer_size=buffer_size)
            # Keep small trainable parameters and optimizer updates in float32;
            # low-magnitude RD gradients must not round away in bfloat16 weights.
            mixture.to(device=device, dtype=torch.float32)
            wrapper = AdaptedMLP(block.mlp, mixture)
            block.mlp = wrapper
            self.adapted.append(wrapper)
        self.eval()

    @property
    def device(self):
        return next(self.language.parameters()).device

    def set_mode(self, mode):
        for layer in self.adapted:
            layer.mode = mode
            layer.rd_value = None
            layer.score = None

    def prepare(self, row, image_root, with_answer):
        question = row["question"].replace("<image>", "").strip()
        prefix = SYSTEM + " USER: "
        suffix = "\n" + question + " ASSISTANT:"
        before = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
        after_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
        image_path = Path(image_root) / row["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        side = max(image.size)
        color = tuple(int(x * 255) for x in self.image_processor.image_mean)
        square = Image.new("RGB", (side, side), color)
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        pixels = self.image_processor(square, return_tensors="pt").pixel_values.to(
            device=self.device, dtype=next(self.vision.parameters()).dtype)
        with torch.no_grad():
            patches = self.vision(pixels, output_hidden_states=True).hidden_states[-2][:, 1:]
            visual = self.projector(patches)
            embed = self.language.get_input_embeddings()
            after = embed(torch.tensor([after_ids], device=self.device))
            prompt = torch.cat((embed(before), visual, after), dim=1)
        prompt_length = prompt.shape[1]
        for layer in self.adapted:
            layer.prompt_length = prompt_length
            layer.cached_context = None
        if not with_answer:
            return prompt, None
        # Leading space matches the Vicuna-v1 assistant separator. Tokenize the
        # response separately so no image/question token can enter the CE loss.
        answer_ids = self.tokenizer(" " + row["answer"], add_special_tokens=False).input_ids
        answer_ids += [self.tokenizer.eos_token_id]
        answer = torch.tensor([answer_ids], device=self.device)
        inputs = torch.cat((prompt, embed(answer).detach()), dim=1)
        labels = torch.cat((torch.full((1, prompt_length), -100, device=self.device), answer), dim=1)
        if inputs.shape[1] > 2048:
            raise ValueError(f"Source 2048-token limit exceeded: {row['id']}")
        return inputs, labels

    def loss(self, row, image_root):
        inputs, labels = self.prepare(row, image_root, with_answer=True)
        return self.language(inputs_embeds=inputs, labels=labels, use_cache=False).loss

    def describe(self, row, image_root, mode):
        self.set_mode(mode)
        inputs, _ = self.prepare(row, image_root, with_answer=False)
        # Frozen prompt prefix; RD detaches its input independently. Functional
        # adapters are frozen during descriptor optimization by the runner.
        self.language(inputs_embeds=inputs, use_cache=False)
        if mode == "features":
            return [layer.pooled for layer in self.adapted]
        if mode == "detect":
            return [layer.score for layer in self.adapted]
        values = [layer.rd_value for layer in self.adapted if layer.rd_value.requires_grad]
        return torch.stack(values).sum() if values else None

    @torch.inference_mode()
    def answer(self, row, image_root, max_new_tokens=16):
        self.set_mode("functional")
        inputs, _ = self.prepare(row, image_root, with_answer=False)
        # Cache context-only router inputs per layer along with language KVs;
        # generated answer tokens never change the router's conditioning.
        output = self.language.generate(
            inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def adapter_state(self):
        return [layer.mixture.state_dict() for layer in self.adapted]

    def load_adapter_state(self, states):
        if len(states) != len(self.adapted):
            raise ValueError("Checkpoint adapter-layer count differs")
        for layer, state in zip(self.adapted, states):
            layer.mixture.load_state_dict(state, strict=True)
