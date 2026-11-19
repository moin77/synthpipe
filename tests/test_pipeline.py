"""The whole pipeline in one test, as a user would run it:

generate -> validate -> autolabel -> cvat-export -> (reviewer in CVAT) -> cvat-import -> publish

Blender, the YOLO model, CVAT and Hugging Face are replaced by small fakes.
"""

import json
import os
import random
import sys
import zipfile

import pytest

from synthpipe.cli import build_parser, cmd_autolabel, cmd_publish

pytest.importorskip("PIL")

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["wrench", "screwdriver", "hammer", "pliers"]


def run(argv, **kw):
    args = build_parser().parse_args(argv)
    if args.func is cmd_autolabel or args.func is cmd_publish:
        return args.func(args, **kw)
    return args.func(args)


def fake_blender(tmp_path):
    fake = os.path.join(HERE, "fake_blender.py")
    if sys.platform == "win32":
        exe = tmp_path / "blender.bat"
        exe.write_text(f'@"{sys.executable}" "{fake}" %*\n')
    else:
        exe = tmp_path / "blender"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        exe.chmod(0o755)
    return exe


def read_labels(folder):
    out = {}
    for f in sorted(os.listdir(folder)):
        with open(os.path.join(folder, f)) as fh:
            out[os.path.splitext(f)[0]] = [l.split() for l in fh if l.strip()]
    return out


class NoisyModel:
    """Pretends to be a detector: returns the true boxes with some misses, wrong classes and extras."""

    def __init__(self, truth, seed=3):
        self.names = NAMES
        self.truth = truth
        self.rng = random.Random(seed)

    def predict(self, paths):
        for p in paths:
            key = os.path.splitext(os.path.basename(p))[0]
            boxes = []
            for c, x, y, w, h in self.truth.get(key, []):
                r = self.rng.random()
                if r < 0.15:
                    continue                                  # missed
                cls = (int(c) + 1) % 4 if r < 0.25 else int(c)  # wrong class
                boxes.append((cls, float(x), float(y), float(w), float(h), self.rng.uniform(0.3, 0.95)))
            if self.rng.random() < 0.3:
                boxes.append((0, 0.95, 0.05, 0.04, 0.04, 0.27))  # extra
            yield p, boxes


def test_full_pipeline(tmp_path):
    # 1. generate a synthetic dataset (fake Blender), validated automatically
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "name": "pipeline-test", "count": 8,
        "output": {"dir": "synth"},
        "blender": {"exe": str(fake_blender(tmp_path)), "blend_file": "gen.blend", "script": "gen.py"},
    }))
    (tmp_path / "gen.blend").write_text("")
    (tmp_path / "gen.py").write_text("")
    assert run(["generate", "-c", str(cfg), "-q"]) == 0
    synth = tmp_path / "synth"
    meta = json.loads((synth / "metadata.json").read_text())
    assert meta["validation"]["errors"] == 0

    # 2. the "real photos" are the synthetic images; their labels are the truth
    truth = read_labels(synth / "labels")
    model = NoisyModel(truth)
    assert run(["autolabel", "-m", "fake.pt", "-i", str(synth / "images"), "-o", str(tmp_path / "pre"),
                "--compare-to", str(synth / "labels"), "-q"], predictor=model) == 0
    predicted = json.loads((tmp_path / "pre" / "comparison.json").read_text())["totals"]

    # 3. package for CVAT
    assert run(["cvat-export", str(tmp_path / "pre"), "-o", str(tmp_path / "task"), "--max-side", "0"]) == 0

    # 4. the reviewer fixes everything back to the truth and exports YOLO 1.1 from CVAT
    manifest = json.loads((tmp_path / "task" / "manifest.json").read_text())
    with zipfile.ZipFile(tmp_path / "cvat.zip", "w") as z:
        z.writestr("obj.names", "\n".join(NAMES) + "\n")
        for stem, entry in manifest["images"].items():
            lines = truth.get(entry["key"], [])
            z.writestr(f"obj_train_data/{stem}.txt", "".join(" ".join(l) + "\n" for l in lines))

    # 5. bring the reviewed labels back
    assert run(["cvat-import", str(tmp_path / "cvat.zip"), "-t", str(tmp_path / "task"),
                "-o", str(tmp_path / "reviewed"), "--minutes", "4",
                "--compare-to", str(synth / "labels")]) == 0
    reviewed = tmp_path / "reviewed"
    assert read_labels(reviewed / "labels") == truth          # names and class ids survived the trip
    review = json.loads((reviewed / "metadata.json").read_text())["review"]
    assert review["changes"]["totals"] == predicted            # review edits == what autolabel predicted
    assert review["agreement"]["boxes_kept_pct"] == 100.0

    # 6. publish: the synthetic set uploads; the reviewed photos need --confirm-rights
    class Hub:
        def __init__(self):
            self.uploaded = []

        def create_repo(self, **kw):
            pass

        def upload_folder(self, **kw):
            self.uploaded.append(kw["folder_path"])

    hub = Hub()
    assert run(["publish", str(synth), "-r", "me/synth"], api=hub) == 0
    assert run(["publish", str(reviewed), "-r", "me/reviewed"], api=hub) == 1
    assert run(["publish", str(reviewed), "-r", "me/reviewed", "--confirm-rights"], api=hub) == 0
    assert hub.uploaded == [str(synth), str(reviewed)]
    card = (reviewed / "README.md").read_text(encoding="utf-8")
    assert "Pre-labeled by a YOLO model" in card and "reviewed by a person in CVAT" in card
