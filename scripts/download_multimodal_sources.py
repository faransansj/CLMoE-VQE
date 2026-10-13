"""Acquire official VQA/COCO annotations on a login node, without touching old data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCES = {
    'coco_annotations': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'vqa_train_questions': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
    'vqa_val_questions': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
    'vqa_train_annotations': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
    'vqa_val_annotations': 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in SOURCES.items():
        destination = args.output / (name + '.zip')
        if not destination.exists():
            temporary = destination.with_suffix('.partial')
            request = urllib.request.Request(url, headers={'User-Agent': 'CLMoE-VQE research'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
        digest = hashlib.sha256()
        with destination.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        members = []
        with zipfile.ZipFile(destination) as archive:
            for member in archive.infolist():
                if not member.filename.endswith('.json'):
                    continue
                if name == 'coco_annotations' and not Path(member.filename).name.startswith('captions_'):
                    continue
                target = args.output / Path(member.filename).name
                content = archive.read(member)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError(f'Refusing to replace different source: {target}')
                if not target.exists():
                    target.write_bytes(content)
                members.append({'path': target.name, 'sha256': hashlib.sha256(content).hexdigest()})
        records[name] = {'url': url, 'bytes': destination.stat().st_size,
                         'sha256': digest.hexdigest(), 'members': members}
        (args.output / 'sources.json').write_text(json.dumps(records, indent=2) + '\n')
        print(name, destination.stat().st_size, digest.hexdigest(), flush=True)


if __name__ == '__main__':
    main()
