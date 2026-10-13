"""Download ONLY on login nodes; use official pretraining, never VQA-tuned weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "model_base_capfilt_large.pth"
    if not target.exists():
        temporary = target.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = "8f5187458d4d47bb87876faf3038d5947eff17475edf52cf47b62e84da0b235f"
    if digest != expected:
        raise ValueError("Official pretrained checkpoint differs from recorded experiment bytes")
    revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
    snapshot_download("google-bert/bert-base-uncased", revision=revision,
                      allow_patterns=["vocab.txt", "tokenizer_config.json", "config.json"],
                      local_dir=root / "tokenizer")
    sources = {"checkpoint": {"url": URL, "sha256": digest,
                              "bytes": target.stat().st_size, "stage": "pretraining, not VQA finetuning"},
               "tokenizer": {"repo": "google-bert/bert-base-uncased", "revision": revision},
               "BLIP_commit": "056a169437371659074aa2732649d5de3bffb4a8"}
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps(sources), flush=True)
    for path in root.rglob("*"):
        if path.is_file():
            path.touch()


if __name__ == "__main__":
    main()
