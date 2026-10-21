#!/usr/bin/env python3
"""Stage 0.5 item 3: caption reference ending formats vs saved prediction punctuation.

Data-only: reads the locked manifest and saved prediction archives. No training,
inference, or scoring.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/stage05-target-format.json"
MANIFEST_TAR = ROOT / "artifacts/multimodal-full-manifests.tar.gz"
MANIFEST_MEMBER = "multimodal_data_full_20261013/manifest_v2.json"
PREDICTION_TARS = [ROOT / "artifacts/multimodal-g1-raw.tar.gz",
                   ROOT / "artifacts/multimodal-g4-raw.tar.gz",
                   ROOT / "artifacts/multimodal-g5-raw.tar.gz"]
CAPTION_FILE = re.compile(r"(predictions|image_shuffled)_[a-z]*_?\d\d_[CF]\.jsonl$|image_shuffled_\d\d_[CF]\.jsonl$")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ending(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith(" ."):
        return "space_period"
    if stripped.endswith("."):
        return "period"
    if stripped[-1:] in "!?":
        return "other_terminal_punct"
    return "no_terminal_punct"


def reference_formats() -> dict:
    with tarfile.open(MANIFEST_TAR, "r:gz") as tar:
        manifest = json.loads(tar.extractfile(MANIFEST_MEMBER).read())
    result = {}
    for task in manifest["tasks"]:
        if task["task_id"] not in ("C", "F"):
            continue
        result[task["task_id"]] = {}
        for split in ("train", "validation", "eval"):
            refs = [ref for row in task[split] for ref in row["references"]]
            ends = Counter(ending(ref) for ref in refs)
            result[task["task_id"]][split] = {
                "references": len(refs),
                "ending_share": {k: v / len(refs) for k, v in sorted(ends.items())},
                "trailing_whitespace_share": sum(r != r.rstrip() for r in refs) / len(refs),
                "capitalized_first_char_share": sum(r[:1].isupper() for r in refs) / len(refs),
                "contains_period_share": sum("." in r for r in refs) / len(refs),
            }
    return result


def prediction_punctuation() -> dict:
    totals = Counter()
    per_archive = {}
    for path in PREDICTION_TARS:
        counts = Counter()
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".jsonl"):
                    continue
                name = member.name.rsplit("/", 1)[-1]
                if not (name.endswith("_C.jsonl") or name.endswith("_F.jsonl")):
                    continue
                if not (name.startswith("predictions_") or name.startswith("image_shuffled_")):
                    continue
                task = name[-7]
                for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    prediction = json.loads(line)["prediction"]
                    counts[(task, "rows")] += 1
                    counts[(task, "contains_period")] += "." in prediction
                    counts[(task, "contains_comma")] += "," in prediction
        per_archive[path.name] = {f"{t}/{k}": v for (t, k), v in sorted(counts.items())}
        totals.update(counts)
    return {"per_archive": per_archive,
            "total": {f"{t}/{k}": v for (t, k), v in sorted(totals.items())}}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = {
        "stage": "stage0.5-target-format",
        "execution": {"training": False, "inference": False, "scoring": False},
        "script": {"path": "scripts/stage05_target_format.py", "sha256": digest(Path(__file__))},
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in [MANIFEST_TAR, *PREDICTION_TARS]},
        "reference_formats": reference_formats(),
        "saved_caption_prediction_punctuation": prediction_punctuation(),
        "note": ("BERT basic tokenization splits punctuation and drops whitespace, so 'x .' and 'x.' "
                 "yield the same token ids; the relevant difference is presence of a terminal period "
                 "before [SEP]. Mechanism linking this to non-termination is a hypothesis."),
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"written": str(OUTPUT.relative_to(ROOT)),
                      "prediction_totals": payload["saved_caption_prediction_punctuation"]["total"]}))


if __name__ == "__main__":
    main()
