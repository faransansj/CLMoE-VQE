"""Validate every selected real image and the source context-length contract."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from sema_vqa.model import SYSTEM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--protocol", choices=["cl4vqa", "vqa_stream"], required=True)
    args = p.parse_args()
    root = Path(args.root)
    path = root / "manifests" / f"{args.protocol}.json"
    manifest = json.loads(path.read_text())
    if not manifest["images"]["ready"]:
        raise ValueError("Manifest image acquisition is not complete")
    tokenizer = AutoTokenizer.from_pretrained(root / "models" / "vicuna", use_fast=False)
    prefix_length = len(tokenizer(SYSTEM + " USER: ").input_ids) + 576
    checked = set()
    report = {"protocol": args.protocol, "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "tasks": [], "image_sha256": {}}
    for task in manifest["tasks"]:
        lengths = []
        for split in ("train", "eval"):
            ids = [row["id"] for row in task[split]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs: {task['name']}/{split}")
            for row in task[split]:
                image = root / "images" / row["image"]
                if row["image"] not in checked:
                    with Image.open(image) as decoded:
                        decoded.verify()
                    report["image_sha256"][row["image"]] = hashlib.sha256(image.read_bytes()).hexdigest()
                    checked.add(row["image"])
                question = row["question"].replace("<image>", "").strip()
                length = prefix_length + len(tokenizer("\n" + question + " ASSISTANT:", add_special_tokens=False).input_ids)
                length += len(tokenizer(" " + row["answer"], add_special_tokens=False).input_ids) + 1
                if length > 2048:
                    raise ValueError(f"Source context limit exceeded ({length}): {row['id']}")
                lengths.append(length)
        report["tasks"].append({"name": task["name"], "train": len(task["train"]),
                                "eval": len(task["eval"]), "max_tokens": max(lengths)})
    report["verified_unique_images"] = len(checked)
    output = root / "manifests" / f"{args.protocol}.validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "image_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
