"""Make image-disjoint train/validation from train only; preserve official eval."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def split_manifest(manifest, image_hashes, seed=2027):
    result = copy.deepcopy(manifest)
    eval_hashes = {image_hashes[r["image"]] for t in manifest["tasks"] for r in t["eval"]}
    eval_ids = {r["id"] for t in manifest["tasks"] for r in t["eval"]}
    groups = {image_hashes[r["image"]] for t in manifest["tasks"] for r in t["train"]} - eval_hashes
    validation_groups = {h for h in groups if int(hashlib.sha256(f"{seed}:{h}".encode()).hexdigest(), 16) / 2**256 < 0.2}
    excluded = []
    for task in result["tasks"]:
        training, validation = [], []
        for row in task["train"]:
            digest = image_hashes[row["image"]]
            if digest in eval_hashes or row["id"] in eval_ids:
                excluded.append({"task": task["name"], "id": row["id"], "image": row["image"], "reason": "official_eval_image_or_id_overlap"})
            elif digest in validation_groups:
                validation.append(row)
            else:
                training.append(row)
        if not training or not validation:
            raise ValueError(f"Empty fit/validation partition: {task['name']}")
        for row in validation:
            if task["metric"] == "textvqa" and (
                    not isinstance(row.get("answers"), list) or len(row["answers"]) != 10):
                raise ValueError(f"Validation requires ten original TextVQA answers: {row['id']}")
            if task["metric"] == "gqa":
                if "reference_answer" not in row:
                    raise ValueError(f"Validation requires original GQA training reference: {row['id']}")
                row["instruction_answer"] = row["answer"]
                row["answer"] = row["reference_answer"]
        task["train"], task["validation"] = training, validation
    split_hashes = {split: {image_hashes[r["image"]] for t in result["tasks"] for r in t[split]}
                   for split in ("train", "validation", "eval")}
    for a, b in (("train", "validation"), ("train", "eval"), ("validation", "eval")):
        if split_hashes[a] & split_hashes[b]:
            raise AssertionError(f"Image byte overlap: {a}/{b}")
    result["vision_partition"] = {"seed": seed, "validation_fraction": 0.2,
        "policy": "Global image-byte SHA256 groups across all tasks; train only; remove eval image/ID overlaps before deterministic hash split. Official eval unchanged.",
        "excluded": excluded, "image_groups": {key: len(value) for key, value in split_hashes.items()},
        "counts": {t["name"]: {s: len(t[s]) for s in split_hashes} for t in result["tasks"]}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation", required=True, help="Existing image SHA256 audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    verification = json.loads(Path(args.validation).read_text())
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if verification["manifest_sha256"] not in (digest, manifest.get("reference_base_manifest_sha256")):
        raise ValueError("Image audit is not for this manifest or its reference-enriched source")
    result = split_manifest(manifest, verification["image_sha256"], args.seed)
    result["vision_partition"]["source_manifest_sha256"] = digest
    result["vision_partition"]["image_audit_sha256"] = hashlib.sha256(Path(args.validation).read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["vision_partition"], indent=2))


if __name__ == "__main__":
    main()
