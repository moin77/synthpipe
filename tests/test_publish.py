"""Dataset card + Hugging Face upload, with a fake Hub API (no network)."""

import json
import os

import pytest

from synthpipe.cli import build_parser, cmd_publish
from synthpipe.publish import build_card, centre_share, dataset_kind

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

NAMES = ["wrench", "screwdriver", "hammer", "pliers"]


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def make_dataset(root, meta, boxes):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i, lines in enumerate(boxes):
        Image.new("RGB", (64, 48)).save(root / "images" / f"img{i}.jpg")
        (root / "labels" / f"img{i}.txt").write_text("".join(l + "\n" for l in lines))
    (root / "classes.txt").write_text("\n".join(NAMES) + "\n")
    (root / "metadata.json").write_text(json.dumps(meta))
    return root


SYNTH_META = {
    "generator": "ToolVision toolvision_blender.py v1.2", "blender_version": "4.5.3", "seed": 42,
    "resolution": [1024, 768], "samples": 64, "classes": NAMES, "num_hdris": 12,
    "randomization": {"tools_per_image": [1, 4], "cam_distance_m": [0.3, 0.9]},
    "label_rule": "full tool outline, clipped to image edge",
    "synthpipe": {"rerun": "synthpipe generate --config configs/tools.yaml --seed 42 --count 3 --start 0",
                  "config_sha256": "ab" * 32},
}
BOXES = [["0 0.5 0.5 0.2 0.2", "1 0.52 0.48 0.1 0.3"], ["2 0.5 0.5 0.3 0.3"], ["3 0.45 0.55 0.2 0.1"]]


def run(argv, api=None):
    return cmd_publish(build_parser().parse_args(argv), api=api)


def test_kind_detection():
    assert dataset_kind(SYNTH_META) == "synthetic"
    assert dataset_kind({"generator": "synthpipe autolabel", "autolabel": {}}) == "autolabeled"
    assert dataset_kind({"generator": "synthpipe cvat-import", "review": {}}) == "reviewed"
    assert dataset_kind({}) == "unknown"


def test_centre_share():
    assert centre_share([(0.5, 0.5), (0.1, 0.1)]) == 0.5


def test_dry_run_writes_card_and_data_yaml(tmp_path):
    ds = make_dataset(tmp_path / "ds", SYNTH_META, BOXES)
    api = FakeApi()
    assert run(["publish", str(ds), "--repo", "moindedroper/toolvision-test", "--dry-run"], api) == 0
    assert api.calls == []                                   # nothing uploaded
    card = (ds / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: cc-by-4.0")
    assert "- object-detection" in card and "- synthetic" in card and "- n<1K" in card
    assert "| Images | 3 |" in card
    assert "| 0 | `wrench` |" in card
    assert "synthpipe generate --config configs/tools.yaml --seed 42" in card
    assert "Objects sit mostly near the middle" in card       # all centres are central here
    assert "huggingface.co/datasets/moindedroper/toolvision-test" in card
    data_yaml = (ds / "data.yaml").read_text()
    assert "  0: wrench" in data_yaml and "train: images" in data_yaml


def test_upload_calls_hub_with_ignore_patterns(tmp_path):
    ds = make_dataset(tmp_path / "ds", SYNTH_META, BOXES)
    (ds / "blender.log").write_text("noise")
    api = FakeApi()
    assert run(["publish", str(ds), "-r", "moindedroper/toolvision-test", "--private"], api) == 0
    (name1, kw1), (name2, kw2) = api.calls
    assert name1 == "create_repo" and kw1["repo_type"] == "dataset" and kw1["private"] is True
    assert name2 == "upload_folder" and kw2["folder_path"] == str(ds) and "*.log" in kw2["ignore_patterns"]


def test_real_photos_need_confirm_rights(tmp_path):
    meta = {"generator": "synthpipe cvat-import", "classes": NAMES,
            "review": {"minutes": 40, "seconds_per_image": 23.0,
                       "changes": {"totals": {"correct": 3, "wrong_class": 1, "loose": 0, "missed": 1, "extra": 2}},
                       "agreement": {"boxes_kept_pct": 93.3, "iou_ok": 0.5}}}
    ds = make_dataset(tmp_path / "ds", meta, BOXES)
    api = FakeApi()
    assert run(["publish", str(ds), "-r", "me/real-tools"], api) == 1
    assert api.calls == []
    assert run(["publish", str(ds), "-r", "me/real-tools", "--confirm-rights"], api) == 0
    card = (ds / "README.md").read_text(encoding="utf-8")
    assert "- human-reviewed" in card
    assert "Review time: 40 minutes (23.0 s per image)" in card
    assert "93.3% of boxes" in card


def test_validation_errors_block_publish(tmp_path):
    ds = make_dataset(tmp_path / "ds", SYNTH_META, [["9 0.5 0.5 0.2 0.2"]])   # class 9 doesn't exist
    api = FakeApi()
    assert run(["publish", str(ds), "-r", "me/x"], api) == 1
    assert run(["publish", str(ds), "-r", "me/x", "--force"], api) == 0


def test_existing_readme_is_backed_up(tmp_path):
    ds = make_dataset(tmp_path / "ds", SYNTH_META, BOXES)
    (ds / "README.md").write_text("my own notes")
    assert run(["publish", str(ds), "-r", "me/x", "--dry-run"]) == 0
    assert (ds / "README.backup.md").read_text() == "my own notes"
    # a second run replaces the generated card without another backup
    assert run(["publish", str(ds), "-r", "me/x", "--dry-run", "--note", "Only 4 tool models."]) == 0
    assert "- Only 4 tool models." in (ds / "README.md").read_text(encoding="utf-8")


def test_card_handles_missing_files(tmp_path):
    (tmp_path / "ds" / "images").mkdir(parents=True)
    card = build_card(str(tmp_path / "ds"), "me/empty")
    assert "# Empty" in card and "license: cc-by-4.0" in card
