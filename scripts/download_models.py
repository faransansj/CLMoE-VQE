"""Download the exact stage-one components linked by the official VQA repos.
Run on the login node, never inside a GPU allocation on Yonsei.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPOS = {
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "clip": "openai/clip-vit-large-patch14-336",
    "projector": "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    api = HfApi()
    for name, repo in REPOS.items():
        info = api.model_info(repo)
        names = [item.rfilename for item in info.siblings]
        weights = "*.safetensors" if any(n.endswith(".safetensors") for n in names) else "pytorch_model*.bin"
        patterns = ["*.json", "tokenizer.model", weights]
        if name == "projector":
            patterns = ["mm_projector.bin", "config.json"]
        snapshot_download(repo, revision=info.sha, local_dir=root / name,
                          allow_patterns=patterns, max_workers=2)
        records[name] = {"repo": repo, "revision": info.sha, "patterns": patterns}
        (root / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[name]), flush=True)


if __name__ == "__main__":
    main()
