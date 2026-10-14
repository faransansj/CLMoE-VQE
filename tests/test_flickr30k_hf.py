import csv
import hashlib
import io
import json
import os
import stat
import zipfile

from PIL import Image
import pytest

from scripts import prepare_flickr30k_hf as prep
from scripts import prepare_multimodal_data as multimodal


def rows():
    return [{"raw": json.dumps(["  Café, a \"quoted\" caption.\n", "Second.", "Third.", "Fourth.", "Fifth. "], ensure_ascii=False),
             "sentids": json.dumps(list(range(index * 5, index * 5 + 5))),
             "split": split, "filename": f"{100 + index}.jpg", "img_id": str(index)}
            for index, split in enumerate(("train", "val", "test"))]


def csv_file(tmp_path, records=None):
    path = tmp_path / "annotations.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["raw", "sentids", "split", "filename", "img_id"])
        writer.writeheader()
        writer.writerows(rows() if records is None else records)
    return path


def jpeg_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), (20, 40, 60)).save(stream, format="JPEG")
    return stream.getvalue()


def archive_file(tmp_path, records=None, extra=None, payload=None):
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for row in rows() if records is None else records:
            archive.writestr(f"flickr30k-images/{row['filename']}", jpeg_bytes() if payload is None else payload)
        if extra is not None:
            archive.writestr(extra, b"unexpected")
    return path


def test_caption_split_id_preservation_and_adapter(tmp_path):
    original = rows()
    source = csv_file(tmp_path, original)
    images = prep.read_annotations(source, expected_counts={"train": 1, "val": 1, "test": 1})
    for before, after in zip(original, images):
        assert after == {"imgid": int(before["img_id"]), "filename": before["filename"], "split": before["split"],
                         "sentences": [{"raw": raw, "sentid": sid} for raw, sid in
                                       zip(json.loads(before["raw"]), json.loads(before["sentids"]))]}
    dataset = tmp_path / "dataset.json"
    prep.write_new_json(dataset, {"images": images})
    spec = {"type": "flickr30k", "revision": prep.REVISION, "license": "fixture",
            "dataset": dataset.name, "images": "Flickr30k", "split_recipe": {"train": ["train"], "eval": ["test"]}}
    candidates = multimodal.normalize_source(spec, tmp_path, {}, 17)
    assert len(candidates["train"]) == len(candidates["eval"]) == 1
    assert candidates["train"][0]["references"] == json.loads(original[0]["raw"])
    assert candidates["eval"][0]["source_image_id"] == "102"
    assert candidates["eval"][0]["image"] == "Flickr30k/102.jpg"


@pytest.mark.parametrize("field,value", [
    ("filename", "../100.jpg"), ("filename", "/100.jpg"), ("filename", "a\\100.jpg"),
    ("filename", "name.jpg"), ("img_id", "-1"), ("img_id", "0.0"),
    ("split", "TEST"), ("split", "restval"), ("raw", "not json"), ("raw", "{}"),
    ("raw", '["a", "b", "c", "d"]'), ("raw", '["a", "b", "c", "d", "  "]'),
    ("raw", '["a", "b", "c", "d", 5]'), ("sentids", "bad json"),
    ("sentids", "[0, 1, 2, 3]"), ("sentids", "[0, 1, 2, 3, true]"),
    ("sentids", '[0, 1, 2, 3, "4"]'), ("sentids", "[0, 1, 2, 3, -1]"),
    ("sentids", "[0, 1, 2, 3, 3]"),
])
def test_malformed_records(tmp_path, field, value):
    records = rows()
    records[0][field] = value
    with pytest.raises(ValueError):
        prep.read_annotations(csv_file(tmp_path, records))


@pytest.mark.parametrize("field", ["filename", "img_id", "sentids"])
def test_global_uniqueness(tmp_path, field):
    records = rows()
    records[1][field] = records[0][field]
    with pytest.raises(ValueError, match="Duplicate"):
        prep.read_annotations(csv_file(tmp_path, records))


@pytest.mark.parametrize("text", ["raw,sentids,split,filename\n", "raw,sentids,split,filename,img_id,img_id\n",
                                  "raw,sentids,split,filename,img_id\n[]\n",
                                  "raw,sentids,split,filename,img_id\n[],[],train,100.jpg,0,extra\n"])
def test_csv_schema_and_field_counts(tmp_path, text):
    source = tmp_path / "bad.csv"
    source.write_text(text)
    with pytest.raises(ValueError):
        prep.read_annotations(source)


def test_empty_and_wrong_counts(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        prep.read_annotations(csv_file(tmp_path, []))
    with pytest.raises(ValueError, match="split counts mismatch"):
        prep.read_annotations(csv_file(tmp_path), expected_counts=prep.SPLIT_COUNTS)


def test_extraction_hashes_resume_preserves_sources_and_refreshes_atime(tmp_path):
    source = csv_file(tmp_path)
    archive = archive_file(tmp_path)
    before = [source.read_bytes(), archive.read_bytes()]
    images = prep.read_annotations(source)
    root = tmp_path / "new_images"
    hashes = prep.extract_images(archive, images, root)
    expected = hashlib.sha256(jpeg_bytes()).hexdigest()
    assert hashes == {row["filename"]: expected for row in rows()}
    assert sorted(path.name for path in root.iterdir()) == sorted(hashes)
    for path in root.iterdir():
        assert path.read_bytes() == jpeg_bytes()
        os.utime(path, (1, 1))
    assert prep.extract_images(archive, images, root) == hashes
    assert all(path.stat().st_atime > 1 for path in root.iterdir())
    assert before == [source.read_bytes(), archive.read_bytes()]


@pytest.mark.parametrize("member", ["../escape.jpg", "/escape.jpg", "flickr30k-images/../escape.jpg",
                                    "flickr30k-images\\escape.jpg", "C:/escape.jpg", "./escape.jpg"])
def test_archive_traversal_rejected_before_extraction(tmp_path, member):
    archive = archive_file(tmp_path, extra=member)
    root = tmp_path / "images"
    with pytest.raises(ValueError, match="Unsafe archive member"):
        prep.extract_images(archive, prep.read_annotations(csv_file(tmp_path)), root)
    assert not root.exists()
    assert not (tmp_path / "escape.jpg").exists()


def test_archive_symlink_and_duplicate_members(tmp_path):
    info = zipfile.ZipInfo("flickr30k-images/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = archive_file(tmp_path, extra=info)
    images = prep.read_annotations(csv_file(tmp_path))
    with pytest.raises(ValueError, match="Non-regular"):
        prep.extract_images(archive, images, tmp_path / "images")
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = archive_file(tmp_path, extra="flickr30k-images/100.jpg")
    with pytest.raises(ValueError, match="Duplicate archive"):
        prep.extract_images(archive, images, tmp_path / "images")


def test_exact_archive_prefix_required(tmp_path):
    archive = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for row in rows():
            stream.writestr(row["filename"], jpeg_bytes())
    with pytest.raises(ValueError, match="Missing exact"):
        prep.extract_images(archive, prep.read_annotations(csv_file(tmp_path)), tmp_path / "images")


def test_corrupt_image_and_wrong_format(tmp_path):
    images = prep.read_annotations(csv_file(tmp_path))
    for payload in (b"not an image", jpeg_bytes()[:100]):
        with pytest.raises((ValueError, OSError)):
            prep.extract_images(archive_file(tmp_path, payload=payload), images, tmp_path / "images")
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, format="PNG")
    with pytest.raises(ValueError, match="Expected JPEG"):
        prep.extract_images(archive_file(tmp_path, payload=png.getvalue()), images, tmp_path / "images")
    assert not list((tmp_path / "images").iterdir())


def test_conflicting_image_not_overwritten(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    conflict = root / "100.jpg"
    conflict.write_bytes(b"must survive")
    with pytest.raises(ValueError, match="Conflicting existing"):
        prep.extract_images(archive_file(tmp_path), prep.read_annotations(csv_file(tmp_path)), root)
    assert conflict.read_bytes() == b"must survive"


def test_destination_symlinks_rejected(tmp_path):
    archive = archive_file(tmp_path)
    images = prep.read_annotations(csv_file(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "images"
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Image root"):
        prep.extract_images(archive, images, root)
    root.unlink()
    root.mkdir()
    (root / "100.jpg").symlink_to(outside / "missing.jpg")
    with pytest.raises(ValueError, match="Unsafe image destination"):
        prep.extract_images(archive, images, root)
    assert not list(outside.iterdir())


def test_helper_rejects_unvalidated_traversal(tmp_path):
    images = [{"filename": "../escape.jpg"}]
    with pytest.raises(ValueError, match="Unsafe image filename"):
        prep.extract_images(archive_file(tmp_path), images, tmp_path / "images")


def test_real_fixture_hash_and_production_hash_failure(tmp_path):
    source = csv_file(tmp_path)
    archive = archive_file(tmp_path)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert prep.verify_source(source, actual)["sha256"] == actual
    for path, pinned in ((source, prep.CSV_SHA256), (archive, prep.ARCHIVE_SHA256)):
        with pytest.raises(ValueError, match="Source SHA256 mismatch"):
            prep.verify_source(path, pinned)
    output, root = tmp_path / "dataset.json", tmp_path / "images"
    with pytest.raises(ValueError, match="Source SHA256 mismatch"):
        prep.convert(source, archive, output, root)
    assert not output.exists()
    assert not output.with_suffix(".sources.json").exists()
    assert not root.exists()


def test_outputs_never_overwritten(tmp_path):
    output = tmp_path / "dataset.json"
    prep.write_new_json(output, {"preserve": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prep.write_new_json(output, {"preserve": False})
    with pytest.raises(ValueError, match="Output must be new"):
        prep.convert(tmp_path / "absent.csv", tmp_path / "absent.zip", output, tmp_path / "images")
    assert output.read_bytes() == before
    output.unlink()
    provenance = output.with_suffix(".sources.json")
    provenance.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="Output must be new"):
        prep.convert(tmp_path / "absent.csv", tmp_path / "absent.zip", output, tmp_path / "images")
