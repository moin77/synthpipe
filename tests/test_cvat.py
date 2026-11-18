"""CVAT round-trip without CVAT: export, fake the reviewer's CVAT download, import."""

import json
import os
import zipfile

import pytest

from synthpipe.autolabel import run_autolabel
from synthpipe.cli import build_parser
from synthpipe.cvat import export_for_cvat

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from test_autolabel import NAMES, FakePredictor  # noqa: E402

ANSWERS = {
    "LocB_001": [(0, 0.5, 0.5, 0.2, 0.2, 0.92)],                                  # fine
    "LocB_002": [(1, 0.4, 0.4, 0.2, 0.3, 0.31)],                                  # low conf
    "LocC_003": [],                                                               # nothing found
    "LocC_004": [(2, 0.3, 0.3, 0.2, 0.2, 0.88), (3, 0.8, 0.8, 0.1, 0.1, 0.77)],   # one extra
}


def run(argv):
    args = build_parser().parse_args(argv)
    return args.func(args)


@pytest.fixture
def prelabeled(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    for n in ANSWERS:
        Image.new("RGB", (400, 300), (80, 100, 120)).save(src / f"{n}.jpg")
    out = tmp_path / "pre"
    run_autolabel(str(src), str(out), FakePredictor(ANSWERS), review_conf=0.5, echo=False)
    return tmp_path, out


def test_export_orders_review_queue_first_and_shrinks(prelabeled):
    tmp, pre = prelabeled
    r = export_for_cvat(str(pre), str(tmp / "task"), max_side=200)
    assert r["images"] == 4 and r["boxes"] == 4 and r["review_first"] == 2
    with zipfile.ZipFile(tmp / "task" / "images.zip") as z:
        names = sorted(z.namelist())
        assert names[0] == "0001_LocC_003.jpg"          # no boxes -> first
        assert names[1] == "0002_LocB_002.jpg"          # low confidence -> second
        with z.open(names[0]) as f:
            assert max(Image.open(f).size) == 200
    with zipfile.ZipFile(tmp / "task" / "annotations_yolo11.zip") as z:
        assert z.read("obj.names").decode().split() == NAMES
        assert "obj_train_data/0004_LocC_004.txt" in z.namelist()
        assert z.read("train.txt").decode().splitlines()[0] == "obj_train_data/0001_LocC_003.jpg"
    labels = json.loads((tmp / "task" / "cvat_labels.json").read_text())
    assert [l["name"] for l in labels] == NAMES
    assert "Upload annotations" in (tmp / "task" / "HOW_TO_REVIEW.txt").read_text()


def fake_cvat_download(path, stems_to_lines, names):
    """What CVAT's 'Export task dataset -> YOLO 1.1' zip looks like."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("obj.names", "\n".join(names) + "\n")
        z.writestr("obj.data", "classes = 4\nnames = data/obj.names\ntrain = data/train.txt\n")
        z.writestr("train.txt", "")
        for stem, lines in stems_to_lines.items():
            z.writestr(f"obj_train_data/{stem}.txt", "".join(l + "\n" for l in lines))


def test_import_maps_names_back_and_counts_changes(prelabeled, capsys):
    tmp, pre = prelabeled
    assert run(["cvat-export", str(pre), "--out", str(tmp / "task"), "--max-side", "0"]) == 0

    # The reviewer's CVAT task has the labels in a DIFFERENT order: ids must be mapped by name
    cvat_names = ["pliers", "hammer", "screwdriver", "wrench"]
    cid = {n: i for i, n in enumerate(cvat_names)}
    reviewed = {
        "0004_LocB_001": [f"{cid['wrench']} 0.5 0.5 0.2 0.2"],                    # kept
        "0002_LocB_002": [f"{cid['screwdriver']} 0.4 0.4 0.2 0.3",                # kept
                          f"{cid['pliers']} 0.8 0.2 0.1 0.1"],                    # drawn (missed)
        "0001_LocC_003": [f"{cid['hammer']} 0.6 0.6 0.3 0.3"],                    # drawn (missed)
        "0003_LocC_004": [f"{cid['hammer']} 0.3 0.3 0.2 0.2"],                    # extra pliers deleted
    }
    # (order numbers come from the export; read them instead of guessing)
    manifest = json.loads((tmp / "task" / "manifest.json").read_text())
    by_key = {v["key"]: s for s, v in manifest["images"].items()}
    reviewed = {by_key[s.split("_", 1)[1]]: v for s, v in reviewed.items()}
    fake_cvat_download(tmp / "cvat.zip", reviewed, cvat_names)

    assert run(["cvat-import", str(tmp / "cvat.zip"), "--task-dir", str(tmp / "task"),
                "--out", str(tmp / "final"), "--minutes", "2"]) == 0
    final = tmp / "final"
    assert (final / "labels" / "LocB_001.txt").read_text().split()[0] == "0"      # wrench is id 0 again
    assert (final / "labels" / "LocC_003.txt").read_text().split()[0] == "2"      # hammer is id 2
    assert (final / "images" / "LocC_004.jpg").exists()                           # original names back
    meta = json.loads((final / "metadata.json").read_text())
    ch = meta["review"]["changes"]["totals"]
    assert (ch["correct"], ch["missed"], ch["extra"]) == (3, 2, 1)
    assert meta["review"]["seconds_per_image"] == 30.0
    html = (final / "quality_report.html").read_text(encoding="utf-8")
    assert "Human review in CVAT" in html
    out = capsys.readouterr().out
    assert "drawn (model missed)        : 2" in out


def test_import_rejects_wrong_zip(prelabeled):
    tmp, pre = prelabeled
    export_for_cvat(str(pre), str(tmp / "task"), max_side=0)
    with zipfile.ZipFile(tmp / "bad.zip", "w") as z:
        z.writestr("annotations.xml", "<annotations/>")
    assert run(["cvat-import", str(tmp / "bad.zip"), "-t", str(tmp / "task"), "-o", str(tmp / "final")]) == 1


def test_unknown_cvat_label_is_dropped_with_warning(prelabeled, capsys):
    tmp, pre = prelabeled
    export_for_cvat(str(pre), str(tmp / "task"), max_side=0)
    manifest = json.loads((tmp / "task" / "manifest.json").read_text())
    stem = next(iter(manifest["images"]))
    fake_cvat_download(tmp / "cvat.zip", {stem: ["4 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.2 0.2"]},
                       NAMES + ["drill"])
    run(["cvat-import", str(tmp / "cvat.zip"), "-t", str(tmp / "task"), "-o", str(tmp / "final")])
    assert "drill" in capsys.readouterr().out
    key = manifest["images"][stem]["key"]
    assert (tmp / "final" / "labels" / f"{key}.txt").read_text().strip() == "0 0.5 0.5 0.2 0.2"
