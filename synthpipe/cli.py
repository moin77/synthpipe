"""SynthPipe command line.

    synthpipe generate --config configs/tools.yaml [--count 5] [--seed 7] [--out DIR] [--overwrite]
    synthpipe compare RUN_A RUN_B
    synthpipe show-config --config configs/tools.yaml
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys

from . import __version__
from .config import ConfigError, check_paths, config_hash, load_config
from .backends import blender as blender_backend
from . import compare as compare_mod

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _git_commit(folder):
    """Commit of the SynthPipe repo, so a dataset points to the exact code that made it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=folder,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _count_images(folder):
    img_dir = os.path.join(folder, "images")
    if not os.path.isdir(img_dir):
        return 0
    return sum(1 for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS))


def _fail(message, code=1):
    print(f"\nERROR: {message}", file=sys.stderr)
    return code


# ---------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------
def cmd_generate(args):
    # --out is relative to where you run the command (paths inside the YAML are
    # relative to the YAML file), so make it absolute before it reaches the config
    out = os.path.abspath(args.out) if args.out else None
    overrides = {"count": args.count, "seed": args.seed, "start": args.start, "output.dir": out}
    if args.overwrite:
        overrides["output.overwrite"] = True
    try:
        cfg = load_config(args.config, overrides)
        check_paths(cfg)
    except (ConfigError, OSError) as e:
        return _fail(str(e))

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    existing = _count_images(out_dir)
    if existing and not cfg["output"]["overwrite"] and cfg["start"] == 0:
        return _fail(
            f"{out_dir} already has {existing} images. Use a new --out folder, "
            "or --overwrite to replace them, or set start to continue a run."
        )
    if existing and cfg["output"]["overwrite"] and cfg["start"] == 0:
        # Only the generated subfolders are removed, never anything else in out_dir
        for sub in ("images", "labels"):
            shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
        print(f"Removed {existing} old images (--overwrite)")

    # Remove the marker of an earlier failed run in this folder
    failed_marker = os.path.join(out_dir, "RUN_FAILED.txt")
    if os.path.exists(failed_marker):
        os.remove(failed_marker)

    # 1. The full, resolved config the generator will read
    run_config_path = os.path.join(out_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    chash = config_hash(cfg)
    print(f"SynthPipe {__version__}")
    print(f"  dataset : {cfg['name']}")
    print(f"  images  : {cfg['count']} (start {cfg['start']}), seed {cfg['seed']}")
    print(f"  output  : {out_dir}")
    print(f"  config  : sha256 {chash[:12]}...")
    if args.dry_run:
        cmd = blender_backend.build_command(cfg, run_config_path)
        print("\nDry run. Command that would run:\n  " + subprocess.list2cmdline(cmd))
        return 0

    # 2. Run Blender
    started = _now()
    log_path = os.path.join(out_dir, "blender.log")
    print("\nRunning Blender (full log: blender.log)...")
    try:
        code, tail = blender_backend.run(cfg, run_config_path, log_path, echo=not args.quiet)
    except FileNotFoundError:
        return _fail(f"Could not start Blender: {cfg['blender']['exe']}")
    except KeyboardInterrupt:
        with open(failed_marker, "w", encoding="utf-8") as f:
            f.write(f"Stopped by user at {_now()}\n")
        return _fail("Stopped. The folder is marked with RUN_FAILED.txt.", 130)
    finished = _now()

    # 3. Did it work?
    meta_path = os.path.join(out_dir, "metadata.json")
    problems = []
    if code != 0:
        problems.append(f"Blender exited with code {code}")
    if not os.path.isfile(meta_path):
        problems.append("the generator did not write metadata.json")
    if problems:
        with open(failed_marker, "w", encoding="utf-8") as f:
            f.write(f"Run failed at {finished}\n")
            f.write("\n".join(problems) + "\n\nLast lines of blender.log:\n")
            f.write("\n".join(tail) + "\n")
        print("\nLast lines from Blender:\n  " + "\n  ".join(tail[-15:]), file=sys.stderr)
        return _fail("; ".join(problems) + ". See blender.log and RUN_FAILED.txt.")

    # 4. Add SynthPipe's own reproducibility record to metadata.json
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta["synthpipe"] = {
        "version": __version__,
        "config_name": cfg["name"],
        "config_file": os.path.abspath(args.config),
        "config_sha256": chash,
        "command": subprocess.list2cmdline(
            blender_backend.build_command(cfg, run_config_path)),
        "rerun": f"synthpipe generate --config {args.config} --seed {cfg['seed']} "
                 f"--count {cfg['count']} --start {cfg['start']}",
        "git_commit": _git_commit(repo_dir),
        "python": platform.python_version(),
        "os": platform.platform(),
        "started": started,
        "finished": finished,
        "status": "ok",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    n = _count_images(out_dir)
    print(f"\nDone: {n} images in {out_dir}")
    if "boxes_per_class" in meta:
        print("Boxes per class: " + ", ".join(f"{k} {v}" for k, v in meta["boxes_per_class"].items()))
    return 0


# ---------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------
def cmd_compare(args):
    try:
        report = compare_mod.compare_runs(args.run_a, args.run_b, pixel_tolerance=args.tolerance)
    except FileNotFoundError as e:
        return _fail(str(e))
    print(compare_mod.format_report(report))
    return 0 if report["identical"] else 2


# ---------------------------------------------------------------------
# show-config
# ---------------------------------------------------------------------
def cmd_show_config(args):
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as e:
        return _fail(str(e))
    print(json.dumps(cfg, indent=2))
    print(f"\nconfig sha256: {config_hash(cfg)}")
    return 0


# ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="synthpipe",
        description="Config file to labeled synthetic dataset, in one command.",
    )
    p.add_argument("--version", action="version", version=f"synthpipe {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="render a dataset from a config file")
    g.add_argument("--config", "-c", required=True, help="path to the YAML config")
    g.add_argument("--count", type=int, help="override: number of images")
    g.add_argument("--seed", type=int, help="override: master seed")
    g.add_argument("--start", type=int, help="override: first image index (to continue a run)")
    g.add_argument("--out", help="override: output folder")
    g.add_argument("--overwrite", action="store_true", help="allow writing into a folder that has images")
    g.add_argument("--dry-run", action="store_true", help="check the config and print the command only")
    g.add_argument("--quiet", "-q", action="store_true", help="don't echo Blender output")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("compare", help="check two runs are identical (reproducibility test)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--tolerance", type=float, default=0.0,
                   help="max mean pixel difference (0-255) still counted as identical")
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("show-config", help="print the full config after defaults are filled in")
    s.add_argument("--config", "-c", required=True)
    s.set_defaults(func=cmd_show_config)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
