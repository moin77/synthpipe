"""Validation checks on a small hand-made dataset with known problems."""

import json
import os

import pytest

from synthpipe.cli import build_parser
from synthpipe.validate import validate_dataset

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

CLASSES = ["wrench", "screwdriver", "hammer", "pliers"]


def make_dataset(root, labels, broken_images=(), extra_images=(), classes=True):
    """labels: {name: text}. Every name gets a valid 64x48 image unless listed in broken_images."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for name, text in labels.items():
        (root / "labels" / f"{name}.txt").write_text(text)
    for name in list(labels) + list(extra_images):
        p = root / "images" / f"{name}.jpg"
        if name in broken_images:
            p.write_bytes(b"not a jpeg")
        elif name not in ("orphan",):
            Image.new("RGB", (64, 48), (120, 90, 60)).save(p)
    if classes:
        (root / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    return str(root)


GOOD = {
    "a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.2\n",
    "b": "2 0.6 0.4 0.3 0.3\n3 0.2 0.7 0.2 0.1\n",
}


def counts(report):
    return {c["check"]: c["count"] for c in report["checks"]}


def test_clean_dataset_passes(tmp_path):
    report = validate_dataset(make_dataset(tmp_path / "ds", GOOD))
    assert report["status"] == "pass"
    assert report["stats"]["boxes"] == 4
    assert report["stats"]["boxes_per_class"] == {"wrench": 1, "screwdriver": 1, "hammer": 1, "pliers": 1}
    assert report["stats"]["resolutions"] == {"64x48": 2}


def test_every_problem_is_found(tmp_path):
    labels = dict(GOOD)
    labels.update({
        "malformed": "0 0.5 0.5 0.2\n",                 # 4 values
        "badclass": "7 0.5 0.5 0.2 0.2\n",              # class 7 of 4
        "outside": "0 0.95 0.5 0.2 0.2\n",              # right edge at 1.05
        "zero": "1 0.5 0.5 0 0.2\n",                    # zero width
        "tiny": "2 0.5 0.5 0.004 0.2\n",                # 0.4% wide
        "dupe": "3 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n",
        "empty": "",
        "broken": "0 0.5 0.5 0.2 0.2\n",
        "orphan": "0 0.5 0.5 0.2 0.2\n",                # label without image
    })
    ds = make_dataset(tmp_path / "ds", labels, broken_images=("broken",), extra_images=("nolabel",))
    c = counts(validate_dataset(ds))
    assert c["malformed_line"] == 1
    assert c["bad_class_id"] == 1
    assert c["out_of_bounds"] == 1
    assert c["zero_size_box"] == 1
    assert c["tiny_box"] == 1
    assert c["duplicate_box"] == 1
    assert c["empty_label"] == 1
    assert c["unreadable_image"] == 1
    assert c["orphan_label"] == 1
    assert c["missing_label"] == 1


def test_missing_class_and_imbalance(tmp_path):
    labels = {f"img{i}": "0 0.5 0.5 0.2 0.2\n" for i in range(8)}
    labels["x"] = "1 0.5 0.5 0.2 0.2\n"
    report = validate_dataset(make_dataset(tmp_path / "ds", labels))
    c = counts(report)
    assert c["missing_class"] == 2        # hammer and pliers have none
    assert c["class_imbalance"] == 1      # 8 wrench vs 1 screwdriver
    assert report["status"] == "fail"


def test_class_names_guessed_without_classes_txt(tmp_path):
    report = validate_dataset(make_dataset(tmp_path / "ds", GOOD, classes=False))
    assert counts(report)["no_class_names"] == 1
    assert report["stats"]["classes"] == ["class_0", "class_1", "class_2", "class_3"]


def test_subfolders_like_train_val(tmp_path):
    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (64, 48)).save(root / "images" / split / "a.jpg")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    report = validate_dataset(str(root), class_names=["tool"])
    assert report["stats"]["images"] == 2
    assert report["status"] == "pass"


def test_cli_writes_report_and_exit_codes(tmp_path):
    ds = make_dataset(tmp_path / "ds", GOOD)
    args = build_parser().parse_args(["validate", ds, "--samples", "2"])
    assert args.func(args) == 0
    html = (tmp_path / "ds" / "quality_report.html").read_text(encoding="utf-8")
    assert "Dataset quality report" in html and "data:image/jpeg;base64," in html
    data = json.loads((tmp_path / "ds" / "validation.json").read_text())
    assert data["status"] == "pass" and "_files" not in data

    bad = make_dataset(tmp_path / "bad", {"a": "9 0.5 0.5 0.2 0.2\n"})
    args = build_parser().parse_args(["validate", bad])
    assert args.func(args) == 1


def test_strict_fails_on_warnings(tmp_path):
    labels = dict(GOOD, empty="")
    ds = make_dataset(tmp_path / "ds", labels)
    assert build_parser().parse_args(["validate", ds]).func(build_parser().parse_args(["validate", ds])) == 0
    args = build_parser().parse_args(["validate", ds, "--strict"])
    assert args.func(args) == 1
