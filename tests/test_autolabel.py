"""Auto-labeling with a fake model, so no GPU, Ultralytics or weights are needed."""

import json
import os

import pytest

from synthpipe.autolabel import compare_labels, run_autolabel
from synthpipe.cli import build_parser, cmd_autolabel

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

NAMES = ["wrench", "screwdriver", "hammer", "pliers"]


class FakePredictor:
    """Returns fixed boxes per file name."""

    def __init__(self, answers):
        self.names = NAMES
        self.answers = answers          # {"img1": [(cls, x, y, w, h, conf), ...]}

    def predict(self, paths):
        for p in paths:
            yield p, self.answers.get(os.path.splitext(os.path.basename(p))[0], [])


def make_images(folder, names):
    os.makedirs(folder, exist_ok=True)
    for n in names:
        Image.new("RGB", (64, 48), (90, 120, 150)).save(os.path.join(folder, n + ".jpg"))


ANSWERS = {
    "a": [(0, 0.5, 0.5, 0.2, 0.2, 0.91), (2, 0.2, 0.3, 0.1, 0.1, 0.88)],
    "b": [(1, 0.4, 0.4, 0.2, 0.3, 0.35)],     # low confidence -> review
    "c": [],                                   # nothing found -> review
}


def test_run_writes_dataset_and_review_queue(tmp_path):
    make_images(tmp_path / "in", ["a", "b", "c"])
    out = tmp_path / "out"
    s = run_autolabel(str(tmp_path / "in"), str(out), FakePredictor(ANSWERS), review_conf=0.5, echo=False)
    assert s["images"] == 3 and s["boxes"] == 3
    assert s["needs_review"] == 2 and s["no_boxes"] == 1
    assert (out / "labels" / "a.txt").read_text().splitlines()[0] == "0 0.500000 0.500000 0.200000 0.200000"
    assert (out / "labels" / "c.txt").read_text() == ""
    assert (out / "images" / "a.jpg").exists()
    queue = [l for l in (out / "review_queue.txt").read_text().splitlines() if not l.startswith("#")]
    assert queue[0].startswith("c.jpg") and queue[1].startswith("b.jpg")   # empty first, then lowest conf
    pred = json.loads((out / "predictions.json").read_text())
    assert pred["images"]["a"]["boxes"][0]["conf"] == 0.91
    assert (out / "classes.txt").read_text().split() == NAMES


def write_labels(folder, labels):
    os.makedirs(folder, exist_ok=True)
    for k, lines in labels.items():
        with open(os.path.join(folder, k + ".txt"), "w") as f:
            f.write("".join(l + "\n" for l in lines))


def test_compare_counts_each_kind_of_edit(tmp_path):
    truth = {
        "img1": ["0 0.5 0.5 0.2 0.2",          # predicted exactly -> correct
                 "1 0.2 0.2 0.1 0.1"],         # predicted as hammer -> wrong class
        "img2": ["2 0.5 0.5 0.4 0.4",          # predicted too small -> loose
                 "3 0.8 0.8 0.1 0.1"],         # not predicted -> missed
        "img3": ["0 0.3 0.3 0.2 0.2"],         # predicted exactly -> correct, image has no edits
    }
    pred = {
        "img1": ["0 0.5 0.5 0.2 0.2", "2 0.2 0.2 0.1 0.1"],
        "img2": ["2 0.5 0.5 0.15 0.15",
                 "1 0.1 0.9 0.05 0.05"],       # nothing there -> extra
        "img3": ["0 0.3 0.3 0.2 0.2"],
    }
    write_labels(tmp_path / "truth", truth)
    write_labels(tmp_path / "pred", pred)
    c = compare_labels(str(tmp_path / "pred"), str(tmp_path / "truth"), NAMES)
    t = c["totals"]
    assert (t["truth"], t["correct"], t["wrong_class"], t["loose"], t["missed"], t["extra"]) == (5, 2, 1, 1, 1, 1)
    assert c["edits"] == 4
    assert c["boxes_kept_pct"] == 40.0
    assert c["images_without_edits"] == 1
    assert c["per_class"]["screwdriver"]["wrong_class"] == 1
    assert c["confusion"]["matrix"][1][2] == 1          # screwdriver labelled as hammer


def test_cli_end_to_end_with_comparison(tmp_path, capsys):
    make_images(tmp_path / "in", ["a", "b", "c"])
    write_labels(tmp_path / "truth", {
        "a": ["0 0.5 0.5 0.2 0.2", "2 0.2 0.3 0.1 0.1"],
        "b": ["1 0.4 0.4 0.2 0.3"],
        "c": ["3 0.5 0.5 0.3 0.3"],
    })
    args = build_parser().parse_args([
        "autolabel", "--model", "fake.pt", "--images", str(tmp_path / "in"),
        "--out", str(tmp_path / "out"), "--compare-to", str(tmp_path / "truth"), "-q"])
    assert cmd_autolabel(args, predictor=FakePredictor(ANSWERS)) == 0
    out = tmp_path / "out"
    comp = json.loads((out / "comparison.json").read_text())
    assert comp["totals"]["correct"] == 3 and comp["totals"]["missed"] == 1
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["autolabel"]["needs_review"] == 2
    html = (out / "quality_report.html").read_text(encoding="utf-8")
    assert "Auto-labeling" in html and "Compared with human labels" in html
    assert "75.0%" in capsys.readouterr().out          # 3 of 4 human boxes kept

    # second run into the same folder needs --overwrite
    assert cmd_autolabel(args, predictor=FakePredictor(ANSWERS)) == 1


def test_out_inside_images_is_refused(tmp_path):
    make_images(tmp_path / "in", ["a"])
    args = build_parser().parse_args([
        "autolabel", "-m", "fake.pt", "-i", str(tmp_path / "in"), "-o", str(tmp_path / "in" / "out")])
    assert cmd_autolabel(args, predictor=FakePredictor(ANSWERS)) == 1
