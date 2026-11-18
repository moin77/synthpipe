"""Publish a dataset to the Hugging Face Hub with an auto-generated dataset card.

The card (README.md) is written from what SynthPipe already recorded:
metadata.json (how the data was made), validation.json (what is in it and what the
checks found) and, for real photos, the auto-label and CVAT review results.
"""

import datetime
import json
import os

from .validate import read_class_names

UPLOAD_IGNORE = ["*.log", "RUN_FAILED.txt", "**/__pycache__/**", "**/.DS_Store", "**/Thumbs.db"]


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _size_category(n):
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                         (1_000_000, "100K<n<1M")):
        if n < limit:
            return label
    return "n>1M"


def dataset_kind(meta):
    """'synthetic', 'autolabeled', 'reviewed' or 'unknown'."""
    if not meta:
        return "unknown"
    gen = str(meta.get("generator", ""))
    if "cvat-import" in gen or "review" in meta:
        return "reviewed"
    if "autolabel" in gen or "autolabel" in meta:
        return "autolabeled"
    if "synthpipe" in meta or "randomization" in meta:
        return "synthetic"
    return "unknown"


def centre_share(centers):
    """Share of box centres inside the middle 40% x 40% of the frame (16% of its area)."""
    if not centers:
        return None
    inside = sum(1 for x, y in centers if 0.3 <= x < 0.7 and 0.3 <= y < 0.7)
    return inside / len(centers)


def detected_limitations(validation):
    """Facts from the data itself, written as limitations a user should know about."""
    out = []
    if not validation:
        return out
    st = validation["stats"]
    counts = {k: v for k, v in st["boxes_per_class"].items() if v > 0}
    if len(counts) >= 2:
        hi, lo = max(counts, key=counts.get), min(counts, key=counts.get)
        ratio = counts[hi] / counts[lo]
        if ratio >= 1.25:
            out.append(f"Classes are not balanced: `{hi}` has {counts[hi]} boxes, `{lo}` has "
                       f"{counts[lo]} ({ratio:.1f}x).")
    share = centre_share(st.get("box_centers"))
    if share is not None and share >= 0.35:
        out.append(f"Objects sit mostly near the middle of the frame: {share:.0%} of box centres fall "
                   "in the central 40% x 40% of the image (16% of its area). Expect weaker detection "
                   "near image edges.")
    if st.get("empty_images"):
        out.append(f"{st['empty_images']} images have no boxes (background images or missed objects).")
    for c in validation.get("checks", []):
        if c["count"] and c["severity"] == "warning" and c["check"] not in ("empty_label", "class_imbalance"):
            out.append(f"Validation warning `{c['check']}`: {c['count']} ({c['description'].lower()}).")
    return out


def _fmt_range(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]:g} to {v[1]:g}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


RAND_LABELS = {
    "tools_per_image": "Objects per image",
    "tool_scale": "Object scale",
    "cam_distance_m": "Camera distance (m)",
    "cam_elevation_deg": "Camera elevation (deg, 90 = top-down)",
    "cam_lens_mm": "Lens (mm)",
    "hdri_strength": "HDRI light strength",
    "spot_chance": "Chance of a spotlight-only scene",
    "exposure": "Exposure",
    "textured_floor_chance": "Chance of a textured floor",
    "flip_chance": "Chance of flipping an object",
    "portrait_chance": "Chance of portrait framing",
}


def build_card(dataset_dir, repo_id, license_id="cc-by-4.0", title=None, summary=None, extra_notes=()):
    """Return the README.md text for the Hub."""
    meta = _load(os.path.join(dataset_dir, "metadata.json")) or {}
    validation = _load(os.path.join(dataset_dir, "validation.json"))
    comparison = _load(os.path.join(dataset_dir, "comparison.json"))
    names = read_class_names(dataset_dir) or meta.get("classes") or []
    kind = dataset_kind(meta)
    st = validation["stats"] if validation else {}
    n_img = st.get("images") or meta.get("count") or 0
    n_box = st.get("boxes")
    title = title or repo_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

    tags = ["object-detection", "yolo"]
    tags += {"synthetic": ["synthetic", "synthetic-data", "blender", "computer-vision"],
             "autolabeled": ["auto-labeled", "computer-vision"],
             "reviewed": ["human-reviewed", "model-assisted-labeling", "cvat", "computer-vision"]}.get(kind, [])

    front = ["---", f"license: {license_id}", f"pretty_name: {title}", "task_categories:",
             "- object-detection", "tags:"] + [f"- {t}" for t in tags] + \
            ["size_categories:", f"- {_size_category(n_img)}", "---", ""]

    default_summary = {
        "synthetic": "Synthetic images rendered in Blender with randomized scenes, with YOLO bounding-box labels generated automatically from the 3D scene.",
        "autolabeled": "Images pre-labeled by a trained detector. Labels have not been reviewed by a person.",
        "reviewed": "Images pre-labeled by a trained detector, then reviewed and corrected by a person in CVAT.",
    }.get(kind, "An object-detection dataset in YOLO format.")

    L = front + [f"# {title}", "", summary or default_summary, ""]

    # At a glance
    L += ["## At a glance", "", "| | |", "|---|---|", f"| Images | {n_img:,} |"]
    if n_box is not None:
        L.append(f"| Boxes | {n_box:,} ({st.get('boxes_per_image_mean', 0)} per image) |")
    L.append(f"| Classes | {len(names)}: {', '.join(f'`{n}`' for n in names)} |")
    if st.get("resolutions"):
        top = sorted(st["resolutions"].items(), key=lambda kv: -kv[1])[:3]
        L.append(f"| Resolution | {', '.join(k for k, _ in top)} |")
    L.append("| Format | YOLO (`class x_center y_center width height`, normalized 0-1) |")
    if validation:
        badge = {"pass": "passed", "warnings": "passed with warnings", "fail": "FAILED"}[validation["status"]]
        L.append(f"| Validation | {badge} ({validation['errors']} errors, {validation['warnings']} warnings) |")
    L += ["", ""]

    if st.get("boxes_per_class"):
        total = sum(st["boxes_per_class"].values()) or 1
        L += ["## Classes", "", "| id | class | boxes | share |", "|---:|---|---:|---:|"]
        for i, n in enumerate(names):
            c = st["boxes_per_class"].get(n, 0)
            L.append(f"| {i} | `{n}` | {c:,} | {100 * c / total:.1f}% |")
        L += ["", ""]

    # Files
    L += ["## Files", "", "```",
          "images/              images",
          "labels/              one .txt per image (YOLO boxes)",
          "classes.txt          class names, line number = class id",
          "data.yaml            Ultralytics dataset file",
          "metadata.json        how the data was made (settings, seed, versions)",
          "validation.json      automatic checks and statistics",
          "quality_report.html  visual report: charts and sample images with boxes"]
    if os.path.isfile(os.path.join(dataset_dir, "run_config.json")):
        L.append("run_config.json      the exact generator config")
    if os.path.isfile(os.path.join(dataset_dir, "predictions.json")):
        L.append("predictions.json     model confidences per box")
    L += ["```", "", "Train with Ultralytics:", "", "```bash",
          f"huggingface-cli download {repo_id} --repo-type dataset --local-dir {repo_id.split('/')[-1]}",
          f"cd {repo_id.split('/')[-1]} && yolo detect train data=data.yaml model=yolo11n.pt",
          "```", "",
          "`data.yaml` uses all images for both `train` and `val`. Split off your own validation set "
          "(and test on real images) before trusting the scores.", "", ""]

    # How it was made
    L += ["## How it was made", ""]
    sp = meta.get("synthpipe") or {}
    if kind == "synthetic":
        gen = meta.get("generator", "Blender generator")
        L.append(f"Rendered with {gen} in Blender {meta.get('blender_version', '')} (Cycles), "
                 "through [SynthPipe](https://github.com/moin77/synthpipe).".replace("  ", " "))
        L.append("")
        rows = [("Seed", meta.get("seed"))]
        if meta.get("resolution"):
            rows.append(("Render size", "x".join(str(v) for v in meta["resolution"])))
        if meta.get("samples"):
            rows.append(("Cycles samples", meta["samples"]))
        if meta.get("num_hdris") is not None:
            rows.append(("HDRI environments", meta["num_hdris"]))
        if meta.get("num_floor_textures") is not None:
            rows.append(("Floor textures", meta["num_floor_textures"]))
        for k, label in RAND_LABELS.items():
            if k in meta.get("randomization", {}):
                rows.append((label, _fmt_range(meta["randomization"][k])))
        L += ["| Setting | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows if b is not None]
        L.append("")
        if meta.get("label_rule"):
            L += [f"Label rule: {meta['label_rule']}.", ""]
        if sp:
            L += ["**Reproduce it exactly:**", "", "```bash", sp.get("rerun", ""), "```", "",
                  f"Config fingerprint (SHA-256): `{sp.get('config_sha256', '')}`. "
                  "Same config + same seed gives identical labels.", ""]
    elif kind in ("autolabeled", "reviewed"):
        al = meta.get("autolabel") or {}
        rv = meta.get("review") or {}
        if rv and not al and rv.get("prelabeled_dataset"):
            # the reviewed set points back at the auto-labeled set it came from
            pre = _load(os.path.join(rv["prelabeled_dataset"], "metadata.json")) or {}
            al = pre.get("autolabel") or {}
        if al:
            L.append(f"Pre-labeled by a YOLO model (confidence ≥ {al.get('conf')}, image size "
                     f"{al.get('imgsz')}). Model SHA-256: `{al.get('model_sha256', 'n/a')}`.")
        if rv:
            ch = (rv.get("changes") or {}).get("totals") or {}
            L.append("")
            L.append(("Every image was then" if al else "Every image was") +
                     " reviewed by a person in CVAT. Compared with the model's boxes:")
            L += ["", "| What happened | Boxes |", "|---|---:|",
                  f"| Kept as the model drew them | {ch.get('correct', 0)} |",
                  f"| Class changed | {ch.get('wrong_class', 0)} |",
                  f"| Box adjusted | {ch.get('loose', 0)} |",
                  f"| Drawn by the reviewer (missed by the model) | {ch.get('missed', 0)} |",
                  f"| Deleted by the reviewer (model false positive) | {ch.get('extra', 0)} |", ""]
            if rv.get("minutes"):
                L.append(f"Review time: {rv['minutes']:g} minutes ({rv.get('seconds_per_image')} s per image).")
            ag = rv.get("agreement")
            if ag:
                L.append(f"The reviewed labels match an earlier independent labeling of the same images "
                         f"for {ag['boxes_kept_pct']}% of boxes (IoU ≥ {ag['iou_ok']}).")
            L.append("")
        elif comparison:
            L.append(f"Against existing human labels, {comparison['boxes_kept_pct']}% of the model's "
                     "boxes were correct as drawn.")
            L.append("")
        L += ["Pipeline: [SynthPipe](https://github.com/moin77/synthpipe) `autolabel` → CVAT review → `cvat-import`.", ""]
    L.append("")

    # Limitations
    lims = list(extra_notes) + detected_limitations(validation)
    if kind == "synthetic":
        lims.append("Synthetic images differ from real photos (lighting, materials, camera noise). "
                    "Models trained only on this data should be tested, and ideally fine-tuned, on real images.")
    if kind == "autolabeled":
        lims.append("Labels come from a model and were not checked by a person: expect missed objects "
                    "and false positives.")
    if lims:
        L += ["## Known limitations", ""] + [f"- {x}" for x in lims] + ["", ""]

    L += ["## License", "",
          f"Released under `{license_id}`. Check the license of any third-party assets "
          "(3D models, textures, HDRIs, photos) before commercial use.", "", "",
          "## Citation", "", "```bibtex",
          f"@misc{{{repo_id.split('/')[-1].replace('-', '_')},",
          f"  title  = {{{title}}},",
          f"  author = {{{repo_id.split('/')[0]}}},",
          f"  year   = {{{datetime.date.today().year}}},",
          f"  url    = {{https://huggingface.co/datasets/{repo_id}}}",
          "}", "```", "",
          f"*Card generated by SynthPipe on {datetime.date.today().isoformat()}.*", ""]
    return "\n".join(x for x in L if x is not None)


def write_data_yaml(dataset_dir, names):
    lines = ["# Ultralytics dataset file. No separate validation split: make your own.",
             "path: .", "train: images", "val: images", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    with open(os.path.join(dataset_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def upload(dataset_dir, repo_id, private=False, message=None, token=None, api=None):
    """Create the dataset repo if needed and upload the folder. Returns the dataset URL."""
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub") from e
        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message or "Upload dataset with SynthPipe",
        ignore_patterns=UPLOAD_IGNORE,
    )
    return f"https://huggingface.co/datasets/{repo_id}"
