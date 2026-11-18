"""End-to-end test of `synthpipe generate` and `synthpipe compare` with a fake Blender."""

import json
import os
import sys

import pytest

from synthpipe.cli import build_parser

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_blender.py")


@pytest.fixture
def setup(tmp_path):
    # A launcher that looks like blender.exe but runs fake_blender.py
    if sys.platform == "win32":
        exe = tmp_path / "blender.bat"
        exe.write_text(f'@"{sys.executable}" "{FAKE}" %*\n')
    else:
        exe = tmp_path / "blender"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        exe.chmod(0o755)
    (tmp_path / "gen.blend").write_text("")
    (tmp_path / "gen.py").write_text("")

    def make_cfg(name="test", **extra):
        data = {
            "name": name,
            "count": 5,
            "output": {"dir": "run"},
            "blender": {"exe": str(exe), "blend_file": "gen.blend", "script": "gen.py"},
        }
        data.update(extra)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(data))
        return str(p)

    return tmp_path, make_cfg


def run(argv):
    args = build_parser().parse_args(argv)
    return args.func(args)


def test_generate_writes_dataset_and_metadata(setup):
    tmp, make_cfg = setup
    assert run(["generate", "-c", make_cfg(), "-q"]) == 0
    out = tmp / "run"
    assert len(os.listdir(out / "images")) == 5
    assert len(os.listdir(out / "labels")) == 5
    assert (out / "run_config.json").exists()
    assert (out / "blender.log").exists()
    meta = json.loads((out / "metadata.json").read_text())
    sp = meta["synthpipe"]
    assert sp["status"] == "ok"
    assert len(sp["config_sha256"]) == 64
    assert "--python-exit-code" in sp["command"]


def test_refuses_to_mix_into_existing_run(setup):
    tmp, make_cfg = setup
    cfg = make_cfg()
    assert run(["generate", "-c", cfg, "-q"]) == 0
    assert run(["generate", "-c", cfg, "-q"]) == 1
    assert run(["generate", "-c", cfg, "-q", "--overwrite", "--count", "3"]) == 0
    assert len(os.listdir(tmp / "run" / "images")) == 3


def test_failed_generator_is_marked(setup):
    tmp, make_cfg = setup
    assert run(["generate", "-c", make_cfg("crash"), "-q"]) == 1
    marker = tmp / "run" / "RUN_FAILED.txt"
    assert marker.exists()
    assert "pliers_a" in marker.read_text()


def test_same_seed_identical_other_seed_not(setup):
    tmp, make_cfg = setup
    cfg = make_cfg()
    a, b, c = str(tmp / "a"), str(tmp / "b"), str(tmp / "c")
    assert run(["generate", "-c", cfg, "-q", "--out", a]) == 0
    assert run(["generate", "-c", cfg, "-q", "--out", b]) == 0
    assert run(["generate", "-c", cfg, "-q", "--out", c, "--seed", "7"]) == 0
    assert run(["compare", a, b]) == 0
    assert run(["compare", a, c]) == 2


def test_out_is_relative_to_current_folder(setup, monkeypatch):
    tmp, make_cfg = setup
    work = tmp / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert run(["generate", "-c", make_cfg(), "-q", "--out", "runs/a"]) == 0
    assert (work / "runs" / "a" / "labels").is_dir()
    assert run(["compare", "runs/a", "runs/a"]) == 0


def test_dry_run_runs_nothing(setup):
    tmp, make_cfg = setup
    assert run(["generate", "-c", make_cfg(), "--dry-run"]) == 0
    assert not (tmp / "run" / "images").exists()
