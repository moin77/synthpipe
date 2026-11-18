"""SynthPipe command line.

    synthpipe generate --config configs/tools.yaml [--count 5] [--seed 7] [--out DIR] [--overwrite]
    synthpipe validate DATASET [--config FILE] [--classes a,b,c] [--strict]
    synthpipe autolabel --model best.pt --images DIR --out DIR [--compare-to LABELS]
    synthpipe cvat-export PRELABELED --out TASK_DIR
    synthpipe cvat-import CVAT_EXPORT.zip --task-dir TASK_DIR --out DIR [--minutes M]
    synthpipe publish DATASET --repo user/name [--dry-run]
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
import zipfile

from . import __version__
from .config import DEFAULTS, ConfigError, check_paths, config_hash, load_config
from .backends import blender as blender_backend
from . import compare as compare_mod
from . import report as report_mod
from . import validate as validate_mod
from . import autolabel as autolabel_mod
from . import cvat as cvat_mod
from . import publish as publish_mod

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

    # 5. Validate the new dataset and write the HTML quality report
    vcfg = cfg["validation"]
    report = None
    if vcfg["enabled"] and not args.no_validate:
        print("\nValidating...")
        report = _run_validation(out_dir, cfg["classes"], vcfg, title=cfg["name"])
        meta["validation"] = validate_mod.summary(report)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    n = _count_images(out_dir)
    print(f"\nDone: {n} images in {out_dir}")
    if "boxes_per_class" in meta:
        print("Boxes per class: " + ", ".join(f"{k} {v}" for k, v in meta["boxes_per_class"].items()))
    if report is not None:
        print(f"Validation: {report['status'].upper()} ({report['errors']} errors, "
              f"{report['warnings']} warnings) - see quality_report.html")
        if report["errors"]:
            return _fail("The dataset rendered but has errors. Open quality_report.html for details.")
    return 0


def _run_validation(dataset_dir, class_names, vcfg, title=None, echo=False, extra_html=""):
    """Run the checks, save validation.json and quality_report.html in the dataset folder."""
    settings = {k: vcfg[k] for k in ("tiny_box", "imbalance_ratio", "duplicate_iou")}
    report = validate_mod.validate_dataset(dataset_dir, class_names, settings)
    validate_mod.save_json(report, os.path.join(dataset_dir, "validation.json"))
    report_mod.save_html(report, os.path.join(dataset_dir, "quality_report.html"),
                         title=title, samples=vcfg["samples"], extra_html=extra_html)
    if echo:
        print(validate_mod.format_text(report))
    return report


# ---------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------
def cmd_validate(args):
    vcfg = dict(DEFAULTS["validation"])
    classes = None
    if args.config:
        try:
            cfg = load_config(args.config)
        except (ConfigError, OSError) as e:
            return _fail(str(e))
        vcfg.update(cfg["validation"])
        classes = cfg["classes"]
    for key in ("tiny_box", "imbalance_ratio", "duplicate_iou", "samples"):
        value = getattr(args, key)
        if value is not None:
            vcfg[key] = value
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        report = _run_validation(os.path.abspath(args.dataset), classes, vcfg, echo=True)
    except FileNotFoundError as e:
        return _fail(str(e))
    print(f"\nReport: {os.path.join(os.path.abspath(args.dataset), 'quality_report.html')}")
    if report["errors"] or (args.strict and report["warnings"]):
        return 1
    return 0


# ---------------------------------------------------------------------
# autolabel
# ---------------------------------------------------------------------
def cmd_autolabel(args, predictor=None):
    """predictor can be passed in by tests; normally it is built from --model."""
    images_dir = os.path.abspath(args.images)
    out_dir = os.path.abspath(args.out)
    if not os.path.isdir(images_dir):
        return _fail(f"Images folder not found: {images_dir}")
    if os.path.abspath(images_dir) == out_dir or out_dir.startswith(images_dir + os.sep):
        return _fail("--out must be a different folder from --images")
    existing = _count_images(out_dir)
    if existing and not args.overwrite:
        return _fail(f"{out_dir} already has {existing} images. Use a new --out folder or --overwrite.")
    if existing:
        for sub in ("images", "labels"):
            shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)

    model_path = os.path.abspath(args.model) if args.model else None
    if predictor is None:
        if not model_path or not os.path.isfile(model_path):
            return _fail(f"Model file not found: {args.model}")
        try:
            predictor = autolabel_mod.YoloPredictor(
                model_path, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                device=args.device, batch=args.batch)
        except RuntimeError as e:
            return _fail(str(e))

    os.makedirs(out_dir, exist_ok=True)
    print(f"SynthPipe {__version__} auto-label")
    print(f"  model   : {model_path}")
    print(f"  classes : {', '.join(predictor.names)}")
    print(f"  images  : {images_dir}")
    print(f"  output  : {out_dir}")
    print(f"  conf {args.conf}, review below {args.review_conf}\n")
    started = _now()
    summary = autolabel_mod.run_autolabel(images_dir, out_dir, predictor, review_conf=args.review_conf,
                                          echo=not args.quiet)

    comparison = None
    if args.compare_to:
        try:
            comparison = autolabel_mod.compare_labels(
                os.path.join(out_dir, "labels"), os.path.abspath(args.compare_to), list(predictor.names))
        except FileNotFoundError as e:
            print(f"\nWARNING: comparison skipped: {e}")
        else:
            with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)

    meta = {
        "generator": "synthpipe autolabel",
        "date": started,
        "classes": list(predictor.names),
        "autolabel": {
            "synthpipe_version": __version__,
            "model": model_path,
            "model_sha256": autolabel_mod.file_sha256(model_path) if model_path and os.path.isfile(model_path) else None,
            "source_images": images_dir,
            "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "device": args.device,
            **{k: v for k, v in summary.items() if k != "confidences"},
            "compared_to": os.path.abspath(args.compare_to) if comparison else None,
        },
        "boxes_per_class": summary["boxes_per_class"],
    }

    vcfg = dict(DEFAULTS["validation"])
    vcfg["samples"] = args.samples
    panel = report_mod.autolabel_panel(summary, comparison)
    report = _run_validation(out_dir, list(predictor.names), vcfg, title="auto-labeled", extra_html=panel)
    meta["validation"] = validate_mod.summary(report)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {summary['images']} images, {summary['boxes']} boxes "
          f"({summary['minutes']} min). {summary['needs_review']} images to review first "
          f"(review_queue.txt), {summary['no_boxes']} with no boxes.")
    if comparison:
        print("\n" + autolabel_mod.format_comparison(comparison))
    print(f"\nReport: {os.path.join(out_dir, 'quality_report.html')}")
    return 0


# ---------------------------------------------------------------------
# CVAT round-trip
# ---------------------------------------------------------------------
def cmd_cvat_export(args):
    out_dir = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out_dir, "manifest.json")) and not args.overwrite:
        return _fail(f"{out_dir} already holds a CVAT task. Use a new --out folder or --overwrite.")
    try:
        r = cvat_mod.export_for_cvat(args.dataset, out_dir, max_side=args.max_side, task_name=args.name)
    except (FileNotFoundError, OSError) as e:
        return _fail(str(e))
    print(f"CVAT task ready in {out_dir}")
    print(f"  {r['images']} images ({r['images_zip_mb']} MB), {r['boxes']} pre-labeled boxes, "
          f"{r['review_first']} to review first")
    print(f"  labels: {', '.join(r['classes'])}")
    print("\nNext: follow HOW_TO_REVIEW.txt in that folder.")
    return 0


def cmd_cvat_import(args):
    out_dir = os.path.abspath(args.out)
    task_dir = os.path.abspath(args.task_dir)
    if not os.path.isfile(os.path.join(task_dir, "manifest.json")):
        return _fail(f"No manifest.json in {task_dir}. Use the folder made by cvat-export.")
    if _count_images(out_dir) and not args.overwrite:
        return _fail(f"{out_dir} already has images. Use a new --out folder or --overwrite.")
    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    try:
        r = cvat_mod.import_from_cvat(args.zip, task_dir, out_dir)
    except (ValueError, OSError, zipfile.BadZipFile) as e:
        return _fail(str(e))
    if r["unknown_labels"]:
        print(f"WARNING: labels in CVAT that are not in the class list were dropped: "
              f"{', '.join(r['unknown_labels'])} ({r['boxes_dropped_unknown_label']} boxes)")
    if r["label_files_not_in_task"]:
        print(f"WARNING: {len(r['label_files_not_in_task'])} label files don't belong to this task "
              "(wrong --task-dir?)")

    pre_labels = os.path.join(r["prelabeled_dataset"], "labels")
    changes = None
    if os.path.isdir(pre_labels):
        changes = autolabel_mod.compare_labels(pre_labels, os.path.join(out_dir, "labels"), r["classes"])
    agreement = None
    if args.compare_to:
        agreement = autolabel_mod.compare_labels(
            os.path.join(out_dir, "labels"), os.path.abspath(args.compare_to), r["classes"])

    meta = {
        "generator": "synthpipe cvat-import",
        "date": _now(),
        "classes": r["classes"],
        "review": {
            "synthpipe_version": __version__,
            "cvat_export": os.path.abspath(args.zip),
            "prelabeled_dataset": r["prelabeled_dataset"],
            "minutes": args.minutes,
            "seconds_per_image": round(60 * args.minutes / r["images"], 1) if args.minutes else None,
            "changes": {k: v for k, v in changes.items() if k != "edits_per_image"} if changes else None,
            "agreement_with": os.path.abspath(args.compare_to) if agreement else None,
            "agreement": {k: v for k, v in agreement.items() if k != "edits_per_image"} if agreement else None,
        },
    }
    panel = report_mod.review_panel(changes, args.minutes, agreement) if changes else ""
    vcfg = dict(DEFAULTS["validation"])
    report = _run_validation(out_dir, r["classes"], vcfg, title="reviewed", extra_html=panel)
    meta["validation"] = validate_mod.summary(report)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Imported {r['images']} reviewed images into {out_dir}")
    if changes:
        t = changes["totals"]
        print(f"\nWhat the reviewer changed ({t['truth']} final boxes):")
        print(f"  kept as the model drew them : {t['correct']} ({changes['boxes_kept_pct']}%)")
        print(f"  class changed               : {t['wrong_class']}")
        print(f"  box adjusted                : {t['loose']}")
        print(f"  drawn (model missed)        : {t['missed']}")
        print(f"  deleted (model extra)       : {t['extra']}")
        print(f"  images untouched            : {changes['images_without_edits']} "
              f"({changes['images_without_edits_pct']}%)")
    if args.minutes:
        print(f"\nReview time {args.minutes:g} min = {60 * args.minutes / r['images']:.0f} s per image")
        if changes and changes["edits"]:
            print(f"  ~{60 * args.minutes / changes['edits']:.0f} s per edit")
    if agreement:
        print(f"\nAgreement with {args.compare_to}: {agreement['boxes_kept_pct']}% of original boxes match")
    print(f"\nValidation: {report['status'].upper()} ({report['errors']} errors, {report['warnings']} warnings)")
    print(f"Report: {os.path.join(out_dir, 'quality_report.html')}")
    return 1 if report["errors"] else 0


# ---------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------
def _folder_size(folder):
    total, files = 0, 0
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if n.endswith(".log") or n == "RUN_FAILED.txt":
                continue
            total += os.path.getsize(os.path.join(root, n))
            files += 1
    return total, files


def cmd_publish(args, api=None):
    """api can be passed in by tests; normally huggingface_hub.HfApi is used."""
    ds = os.path.abspath(args.dataset)
    if not os.path.isdir(os.path.join(ds, "images")) or not os.path.isdir(os.path.join(ds, "labels")):
        return _fail(f"{ds} needs images/ and labels/ folders")
    if args.repo.count("/") != 1:
        return _fail("--repo must look like your-username/dataset-name")
    if os.path.isfile(os.path.join(ds, "RUN_FAILED.txt")):
        return _fail("This folder is marked RUN_FAILED.txt; it is not a finished dataset.")

    names = validate_mod.read_class_names(ds)
    validation = publish_mod._load(os.path.join(ds, "validation.json"))
    if validation is None:
        print("No validation.json yet, running the checks first...")
        report = _run_validation(ds, names, dict(DEFAULTS["validation"]))
        validation = {k: v for k, v in report.items() if not k.startswith("_")}
        names = names or report["stats"]["classes"]
    if validation["errors"] and not args.force:
        return _fail(f"The dataset has {validation['errors']} validation errors (see quality_report.html). "
                     "Fix them, or use --force to publish anyway.")

    meta = publish_mod._load(os.path.join(ds, "metadata.json"))
    kind = publish_mod.dataset_kind(meta)
    if kind != "synthetic" and not args.confirm_rights and not args.dry_run:
        return _fail("These are not SynthPipe renders. Only publish photos you took yourself or have the "
                     "right to share. If that's the case, add --confirm-rights.")

    card_path = os.path.join(ds, "README.md")
    if os.path.isfile(card_path):
        with open(card_path, encoding="utf-8") as f:
            if "Card generated by SynthPipe" not in f.read():
                os.replace(card_path, os.path.join(ds, "README.backup.md"))
                print("Your existing README.md was kept as README.backup.md")
    card = publish_mod.build_card(ds, args.repo, license_id=args.license, title=args.title,
                                  summary=args.summary, extra_notes=args.note or ())
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    publish_mod.write_data_yaml(ds, names)

    size, files = _folder_size(ds)
    print(f"Dataset card written: {card_path}")
    print(f"  kind: {kind}, {files} files, {size / 1e6:.1f} MB, license {args.license}")
    if args.dry_run:
        print("\nDry run: nothing uploaded. Open README.md to check the card, then run again without --dry-run.")
        return 0

    print(f"\nUploading to https://huggingface.co/datasets/{args.repo} ...")
    try:
        url = publish_mod.upload(ds, args.repo, private=args.private, message=args.message, api=api)
    except RuntimeError as e:
        return _fail(str(e))
    except Exception as e:   # network or auth errors from huggingface_hub
        msg = str(e)
        if "401" in msg or "403" in msg or "token" in msg.lower():
            msg += "\nLog in first: hf auth login  (token with write access from huggingface.co/settings/tokens)"
        return _fail(msg)
    print(f"Done: {url}")
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
    g.add_argument("--no-validate", action="store_true", help="skip the checks and HTML report")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="check a YOLO dataset and write an HTML quality report")
    v.add_argument("dataset", help="folder with images/ and labels/")
    v.add_argument("--config", "-c", help="take class names and limits from a SynthPipe config")
    v.add_argument("--classes", help="class names in id order, comma separated (else classes.txt)")
    v.add_argument("--tiny-box", type=float, help="tiny-box limit as a fraction of the image (default 0.01)")
    v.add_argument("--imbalance-ratio", type=float, help="class imbalance limit (default 3.0)")
    v.add_argument("--duplicate-iou", type=float, help="duplicate-box overlap limit (default 0.95)")
    v.add_argument("--samples", type=int, help="images in the report gallery (default 12)")
    v.add_argument("--strict", action="store_true", help="exit with an error on warnings too")
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("autolabel", help="pre-label images with a trained YOLO model for human review")
    a.add_argument("--model", "-m", required=True, help="trained weights, e.g. runs/.../weights/best.pt")
    a.add_argument("--images", "-i", required=True, help="folder of images to label")
    a.add_argument("--out", "-o", required=True, help="output folder (images/, labels/, report)")
    a.add_argument("--conf", type=float, default=0.25, help="keep boxes at or above this confidence (default 0.25)")
    a.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default 0.7)")
    a.add_argument("--imgsz", type=int, default=640, help="inference size (default 640, as trained)")
    a.add_argument("--device", help="0 for the first GPU, or cpu (default: automatic)")
    a.add_argument("--batch", type=int, default=8, help="images per batch (default 8)")
    a.add_argument("--review-conf", type=float, default=0.5,
                   help="images with a box below this go on the review queue (default 0.5)")
    a.add_argument("--compare-to", help="folder of human YOLO labels for the same images, to score the pre-labels")
    a.add_argument("--samples", type=int, default=12, help="images in the report gallery")
    a.add_argument("--overwrite", action="store_true", help="replace an earlier run in --out")
    a.add_argument("--quiet", "-q", action="store_true", help="don't print a line per image")
    a.set_defaults(func=cmd_autolabel)

    ce = sub.add_parser("cvat-export", help="package pre-labels for review in CVAT")
    ce.add_argument("dataset", help="pre-labeled dataset (output of autolabel)")
    ce.add_argument("--out", "-o", required=True, help="folder for the CVAT task files")
    ce.add_argument("--max-side", type=int, default=1600,
                    help="shrink images to this many pixels on the long side for a fast upload (0 = keep)")
    ce.add_argument("--name", help="task name shown in the instructions")
    ce.add_argument("--overwrite", action="store_true")
    ce.set_defaults(func=cmd_cvat_export)

    ci = sub.add_parser("cvat-import", help="bring corrected labels back from CVAT (YOLO 1.1 export)")
    ci.add_argument("zip", help="the zip downloaded from CVAT (Export task dataset, YOLO 1.1)")
    ci.add_argument("--task-dir", "-t", required=True, help="the folder made by cvat-export")
    ci.add_argument("--out", "-o", required=True, help="folder for the corrected dataset")
    ci.add_argument("--minutes", type=float, help="how long the review took, for the time-per-image numbers")
    ci.add_argument("--compare-to", help="original human labels for the same images, to check agreement")
    ci.add_argument("--overwrite", action="store_true")
    ci.set_defaults(func=cmd_cvat_import)

    pb = sub.add_parser("publish", help="write a dataset card and upload the dataset to Hugging Face")
    pb.add_argument("dataset", help="dataset folder (images/, labels/)")
    pb.add_argument("--repo", "-r", required=True, help="Hugging Face dataset id, e.g. moindedroper/toolvision-synthetic")
    pb.add_argument("--license", default="cc-by-4.0", help="license id for the card (default cc-by-4.0)")
    pb.add_argument("--title", help="dataset title (default: from the repo name)")
    pb.add_argument("--summary", help="one or two sentences at the top of the card")
    pb.add_argument("--note", action="append", help="extra known limitation (repeatable)")
    pb.add_argument("--private", action="store_true", help="create the repo as private")
    pb.add_argument("--message", help="commit message on the Hub")
    pb.add_argument("--dry-run", action="store_true", help="write README.md and data.yaml only, no upload")
    pb.add_argument("--confirm-rights", action="store_true",
                    help="required for real photos: you took them or may share them")
    pb.add_argument("--force", action="store_true", help="publish even if validation found errors")
    pb.set_defaults(func=cmd_publish)

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
