"""Acquire official COCO2014 ZIPs on a LOGIN node, avoiding per-image HTTP latency.

This changes transport only, never sample selection. Existing partial transfers
resume through the source acquisition helper; archive members are extracted and
image-hashed by prepare_readout_data.py. Do not run downloads on compute nodes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from scripts.prepare_data import download, sha256


def acquire(directory, split):
    url = f"http://images.cocodataset.org/zips/{split}.zip"
    destination = directory / f"{split}.zip"
    download(url, destination)
    destination.touch()
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda split: acquire(args.output, split), ("train2014", "val2014")))
    for record in records:
        record["sha256"] = sha256(Path(record["path"]))
    (args.output / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
