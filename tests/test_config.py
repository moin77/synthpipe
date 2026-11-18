import json

import pytest

from synthpipe.config import ConfigError, config_hash, load_config


def write_cfg(tmp_path, data, name="cfg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


BASE = {
    "output": {"dir": "out"},
    "blender": {"exe": "blender.exe", "blend_file": "gen.blend", "script": "gen.py"},
}


def test_defaults_are_filled_in(tmp_path):
    cfg = load_config(write_cfg(tmp_path, BASE))
    assert cfg["seed"] == 42
    assert cfg["randomization"]["spot_chance"] == 0.35
    assert cfg["classes"] == ["wrench", "screwdriver", "hammer", "pliers"]


def test_relative_paths_resolve_from_config_folder(tmp_path):
    cfg = load_config(write_cfg(tmp_path, BASE))
    assert cfg["output"]["dir"] == str(tmp_path / "out")
    assert cfg["blender"]["script"] == str(tmp_path / "gen.py")


def test_overrides_win(tmp_path):
    cfg = load_config(write_cfg(tmp_path, BASE), {"count": 5, "seed": 7, "output.dir": None})
    assert cfg["count"] == 5 and cfg["seed"] == 7
    assert cfg["output"]["dir"] == str(tmp_path / "out")   # None override is ignored


def test_typo_is_reported(tmp_path):
    data = dict(BASE, randomization={"spot_chanse": 0.2})
    with pytest.raises(ConfigError, match="spot_chanse"):
        load_config(write_cfg(tmp_path, data))


def test_reversed_range_is_rejected(tmp_path):
    data = dict(BASE, randomization={"cam_distance_m": [0.9, 0.3]})
    with pytest.raises(ConfigError, match="min .* is bigger than max"):
        load_config(write_cfg(tmp_path, data))


def test_chance_out_of_range_is_rejected(tmp_path):
    data = dict(BASE, randomization={"flip_chance": 1.5})
    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_config(write_cfg(tmp_path, data))


def test_all_problems_reported_together(tmp_path):
    data = dict(BASE, count=0, randomization={"flip_chance": 2, "exposure": [1, 0]})
    with pytest.raises(ConfigError) as e:
        load_config(write_cfg(tmp_path, data))
    msg = str(e.value)
    assert "count" in msg and "flip_chance" in msg and "exposure" in msg


def test_missing_required_paths(tmp_path):
    with pytest.raises(ConfigError, match="blender.exe: required"):
        load_config(write_cfg(tmp_path, {"output": {"dir": "out"}}))


def test_hash_ignores_machine_paths_but_not_settings(tmp_path):
    a = load_config(write_cfg(tmp_path, BASE, "a.json"))
    b = load_config(write_cfg(tmp_path, dict(BASE, output={"dir": "elsewhere"}), "b.json"))
    c = load_config(write_cfg(tmp_path, dict(BASE, seed=1), "c.json"))
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_yaml_example_config_is_valid():
    pytest.importorskip("yaml")
    import os
    here = os.path.dirname(__file__)
    cfg = load_config(os.path.join(here, "..", "configs", "tools.yaml"))
    assert cfg["name"] == "toolvision-v1.2"
