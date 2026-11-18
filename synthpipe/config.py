"""Load a SynthPipe YAML config, fill in defaults and catch bad values early.

The goal: a typo or an impossible range fails here with a clear message,
not 20 minutes later inside Blender.
"""

import copy
import hashlib
import json
import os

# ---------------------------------------------------------------------
# Defaults. Any key the YAML leaves out gets these values.
# They match ToolVision generator v1.1, so an almost-empty config
# reproduces the published dataset settings.
# ---------------------------------------------------------------------
DEFAULTS = {
    "name": "dataset",
    "backend": "blender",
    "seed": 42,
    "count": 10,
    "start": 0,
    "output": {
        "dir": None,          # required
        "format": "yolo",
        "overwrite": False,   # refuse to write into a folder that already has images
    },
    "blender": {
        "exe": None,          # required
        "blend_file": None,   # required
        "script": None,       # required
        "extra_args": [],     # e.g. ["--factory-startup"]
    },
    "assets": {
        "hdri_dir": "",
        "texture_dir": "",
    },
    "classes": ["wrench", "screwdriver", "hammer", "pliers"],
    "objects": [
        "combination_wrench",
        "screwdriver",
        "screwdrivers_02_a",
        "screwdrivers_02_b",
        "wooden_hammer_01",
        "pliers_a",
    ],
    "render": {
        "resolution": [1024, 768],
        "samples": 64,
        "lens_mm": 26,
    },
    "randomization": {
        "tools_per_image": [1, 4],
        "scatter_radius_m": 0.20,
        "min_tool_spacing_m": 0.12,
        "tool_scale": [0.85, 1.15],
        "flip_chance": 0.5,
        "cam_distance_m": [0.30, 0.90],
        "cam_elevation_deg": [40, 85],
        "portrait_chance": 0.4,
        "hdri_strength": [0.2, 1.8],
        "spot_chance": 0.35,
        "exposure": [-0.3, 0.8],
        "textured_floor_chance": 0.75,
        "min_box_fraction": 0.01,
    },
}

# Which randomization keys are [min, max] ranges and which are 0-1 chances
RANGE_KEYS = {
    "tools_per_image", "tool_scale", "cam_distance_m",
    "cam_elevation_deg", "hdri_strength", "exposure",
}
CHANCE_KEYS = {
    "flip_chance", "portrait_chance", "spot_chance",
    "textured_floor_chance", "min_box_fraction",
}
POSITIVE_KEYS = {"scatter_radius_m", "min_tool_spacing_m"}

SUPPORTED_BACKENDS = {"blender"}
SUPPORTED_FORMATS = {"yolo"}


class ConfigError(ValueError):
    """Raised for any problem in the config. The message lists every problem found."""


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_file(path):
    """Read a .yaml/.yml (needs PyYAML) or .json config into a dict."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:
            raise ConfigError("PyYAML is not installed. Run: pip install pyyaml") from e
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping (key: value pairs)")
    return data


def merge(defaults, user, where="", unknown=None):
    """Deep-merge user values over defaults. Unknown keys are collected, not ignored,
    so a typo like 'spot_chanse' is reported instead of silently doing nothing."""
    if unknown is None:
        unknown = []
    out = copy.deepcopy(defaults)
    for key, value in user.items():
        path = f"{where}.{key}" if where else key
        if key not in defaults:
            unknown.append(path)
            continue
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                unknown.append(f"{path} (expected a section with keys)")
                continue
            out[key], _ = merge(defaults[key], value, path, unknown)
        else:
            out[key] = value
    return out, unknown


def resolve_path(value, base_dir):
    """Relative paths in the config are relative to the config file's folder."""
    if not value:
        return value
    value = os.path.expanduser(str(value))
    if not os.path.isabs(value):
        value = os.path.join(base_dir, value)
    return os.path.normpath(value)


def load_config(path, overrides=None):
    """Load, merge with defaults, apply command-line overrides, resolve paths, validate.

    overrides: dict like {"count": 5, "seed": 7, "output.dir": "..."}.
    Returns the complete config dict. Raises ConfigError on any problem.
    """
    path = os.path.abspath(path)
    user = load_file(path)
    cfg, unknown = merge(DEFAULTS, user)
    if unknown:
        raise ConfigError("Unknown config keys (typo?):\n  - " + "\n  - ".join(unknown))

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value

    base = os.path.dirname(path)
    cfg["output"]["dir"] = resolve_path(cfg["output"]["dir"], base)
    for k in ("exe", "blend_file", "script"):
        cfg["blender"][k] = resolve_path(cfg["blender"][k], base)
    for k in ("hdri_dir", "texture_dir"):
        cfg["assets"][k] = resolve_path(cfg["assets"][k], base)

    validate(cfg)
    return cfg


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(cfg):
    """Check values only (no file-system checks). Collects every problem, then raises once."""
    errors = []

    if cfg["backend"] not in SUPPORTED_BACKENDS:
        errors.append(f"backend: '{cfg['backend']}' is not supported (use: {', '.join(sorted(SUPPORTED_BACKENDS))})")
    if cfg["output"]["format"] not in SUPPORTED_FORMATS:
        errors.append(f"output.format: '{cfg['output']['format']}' is not supported (use: yolo)")

    for key in ("seed", "count", "start"):
        v = cfg[key]
        if not isinstance(v, int) or isinstance(v, bool):
            errors.append(f"{key}: must be a whole number, got {v!r}")
    if isinstance(cfg["count"], int) and cfg["count"] < 1:
        errors.append("count: must be at least 1")
    if isinstance(cfg["start"], int) and cfg["start"] < 0:
        errors.append("start: cannot be negative")

    if not cfg["output"]["dir"]:
        errors.append("output.dir: required")
    for k in ("exe", "blend_file", "script"):
        if not cfg["blender"][k]:
            errors.append(f"blender.{k}: required")
    if not isinstance(cfg["blender"]["extra_args"], list):
        errors.append("blender.extra_args: must be a list")

    classes = cfg["classes"]
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        errors.append("classes: must be a non-empty list of names")
    elif len(set(classes)) != len(classes):
        errors.append("classes: names must be unique")
    if not isinstance(cfg["objects"], list) or not cfg["objects"]:
        errors.append("objects: must be a non-empty list of Blender object names")

    res = cfg["render"]["resolution"]
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(r, int) and r > 0 for r in res)):
        errors.append(f"render.resolution: must be [long_side, short_side] in pixels, got {res!r}")
    elif res[0] < res[1]:
        errors.append("render.resolution: first value is the LONG side (portrait is handled by portrait_chance)")
    if not (isinstance(cfg["render"]["samples"], int) and cfg["render"]["samples"] > 0):
        errors.append("render.samples: must be a positive whole number")
    if not (_is_number(cfg["render"]["lens_mm"]) and cfg["render"]["lens_mm"] > 0):
        errors.append("render.lens_mm: must be a positive number")

    r = cfg["randomization"]
    for key in RANGE_KEYS:
        v = r[key]
        if not (isinstance(v, list) and len(v) == 2 and all(_is_number(x) for x in v)):
            errors.append(f"randomization.{key}: must be [min, max], got {v!r}")
        elif v[0] > v[1]:
            errors.append(f"randomization.{key}: min ({v[0]}) is bigger than max ({v[1]})")
    for key in CHANCE_KEYS:
        v = r[key]
        if not (_is_number(v) and 0 <= v <= 1):
            errors.append(f"randomization.{key}: must be between 0 and 1, got {v!r}")
    for key in POSITIVE_KEYS:
        v = r[key]
        if not (_is_number(v) and v > 0):
            errors.append(f"randomization.{key}: must be a positive number, got {v!r}")

    tpi = r["tools_per_image"]
    if isinstance(tpi, list) and len(tpi) == 2:
        if not all(isinstance(x, int) for x in tpi) or tpi[0] < 1:
            errors.append("randomization.tools_per_image: must be whole numbers, minimum 1")
    el = r["cam_elevation_deg"]
    if isinstance(el, list) and len(el) == 2 and all(_is_number(x) for x in el):
        if el[0] < 0 or el[1] > 90:
            errors.append("randomization.cam_elevation_deg: must stay within 0-90 (90 = straight down)")

    if errors:
        raise ConfigError("Config problems:\n  - " + "\n  - ".join(errors))


def check_paths(cfg):
    """File-system checks, done right before a run (kept separate so tests need no files)."""
    errors = []
    b = cfg["blender"]
    if not os.path.isfile(b["exe"]):
        errors.append(f"blender.exe not found: {b['exe']}")
    if not os.path.isfile(b["blend_file"]):
        errors.append(f"blender.blend_file not found: {b['blend_file']}")
    if not os.path.isfile(b["script"]):
        errors.append(f"blender.script not found: {b['script']}")
    for k in ("hdri_dir", "texture_dir"):
        d = cfg["assets"][k]
        if d and not os.path.isdir(d):
            errors.append(f"assets.{k} folder not found: {d}")
    if errors:
        raise ConfigError("Path problems:\n  - " + "\n  - ".join(errors))


def config_hash(cfg):
    """SHA-256 of the settings that change the pixels and labels.
    Output folder and Blender install path are left out, so the same dataset
    gets the same hash on any machine."""
    relevant = copy.deepcopy(cfg)
    relevant.pop("output", None)
    relevant.pop("blender", None)
    relevant.pop("assets", None)
    text = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
