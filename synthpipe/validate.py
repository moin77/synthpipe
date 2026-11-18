"""Automatic dataset validation for YOLO-format datasets.

Works on any folder with images/ and labels/ (subfolders such as train/ and
val/ are fine), so it checks real hand-labeled data as well as SynthPipe output.

Errors mean the dataset is broken and training would be wrong or crash.
Warnings mean something deserves a look but may be intentional.
"""

import datetime
import json
import math
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EPS = 1e-4   # rounding slack for coordinates written with 6 decimals

ERROR, WARNING = "error", "warning"

# Every check, in report order: (code, severity, what it means)
CHECKS = [
    ("missing_label",   ERROR,   "Image has no label file"),
    ("orphan_label",    ERROR,   "Label file has no image"),
    ("unreadable_image", ERROR,  "Image file can't be opened"),
    ("malformed_line",  ERROR,   "Label line is not 'class x y w h' with numbers"),
    ("bad_class_id",    ERROR,   "Class id is not in the class list"),
    ("out_of_bounds",   ERROR,   "Box goes outside the image (coordinates outside 0-1)"),
    ("zero_size_box",   ERROR,   "Box has zero or negative width or height"),
    ("missing_class",   ERROR,   "A class has no boxes at all"),
    ("empty_label",     WARNING, "Image has no boxes (fine if it's a deliberate background image)"),
    ("tiny_box",        WARNING, "Box is very small and hard to learn from"),
    ("duplicate_box",   WARNING, "Two boxes of the same class almost completely overlap"),
    ("class_imbalance", WARNING, "Largest class has many more boxes than the smallest"),
    ("no_class_names",  WARNING, "No classes.txt found, class names are guessed"),
]
SEVERITY = {code: sev for code, sev, _ in CHECKS}
DESCRIPTION = {code: desc for code, _, desc in CHECKS}

DEFAULT_SETTINGS = {
    "tiny_box": 0.01,          # box width or height below this fraction of the image
    "imbalance_ratio": 3.0,    # warn when max class / min class is above this
    "duplicate_iou": 0.95,     # same-class boxes overlapping more than this
}


# ---------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------
def _index(folder, exts):
    """{relative path without extension: full path}, walking subfolders."""
    found = {}
    if not os.path.isdir(folder):
        return found
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(root, f)
                key = os.path.splitext(os.path.relpath(full, folder))[0].replace("\\", "/")
                found[key] = full
    return found


def read_class_names(dataset_dir):
    """classes.txt, else metadata.json's classes, else None."""
    p = os.path.join(dataset_dir, "classes.txt")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    m = os.path.join(dataset_dir, "metadata.json")
    if os.path.isfile(m):
        try:
            with open(m, encoding="utf-8") as f:
                names = json.load(f).get("classes")
            if names:
                return list(names)
        except (ValueError, OSError):
            pass
    return None


def _iou(a, b):
    """IoU of two YOLO boxes (x, y, w, h)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _image_size(path):
    """(width, height), None if Pillow is missing, or raises OSError if the image is broken."""
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as im:
        size = im.size
        im.verify()          # reads the whole file, catches truncated images
    return size


# ---------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------
def validate_dataset(dataset_dir, class_names=None, settings=None, check_images=True):
    """Check a YOLO dataset. Returns a report dict (also the content of validation.json)."""
    s = dict(DEFAULT_SETTINGS, **(settings or {}))
    dataset_dir = os.path.abspath(dataset_dir)
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")
    if not os.path.isdir(img_dir) and not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"No images/ or labels/ folder in {dataset_dir}")

    issues = []

    def add(code, file=None, line=None, message=""):
        issues.append({"check": code, "severity": SEVERITY[code],
                       "file": file, "line": line, "message": message})

    names = class_names or read_class_names(dataset_dir)
    names_guessed = names is None

    images = _index(img_dir, IMAGE_EXTS)
    labels = _index(lbl_dir, (".txt",))

    for key in sorted(set(images) - set(labels)):
        add("missing_label", f"images/{os.path.relpath(images[key], img_dir)}".replace("\\", "/"),
            message="no matching .txt in labels/")
    for key in sorted(set(labels) - set(images)):
        add("orphan_label", f"labels/{key}.txt", message="no matching image in images/")

    # Image checks
    resolutions = {}
    images_checked = check_images
    if check_images:
        for key in sorted(images):
            rel = f"images/{os.path.relpath(images[key], img_dir)}".replace("\\", "/")
            try:
                size = _image_size(images[key])
            except Exception as e:      # PIL raises several error types for broken files
                add("unreadable_image", rel, message=str(e)[:200])
                continue
            if size is None:            # Pillow not installed
                images_checked = False
                break
            label = f"{size[0]}x{size[1]}"
            resolutions[label] = resolutions.get(label, 0) + 1

    # Label checks
    per_class = {}
    boxes_per_image = []
    box_sizes = []        # sqrt(w*h): side of a square with the same area, as fraction of image
    box_centers = []
    max_id_seen = -1
    file_boxes = {}       # key -> list of (cls, x, y, w, h), used for the sample gallery

    for key in sorted(labels):
        rel = f"labels/{key}.txt"
        with open(labels[key], encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        boxes = []
        for n, raw in enumerate(lines, start=1):
            text = raw.strip()
            if not text:
                continue
            parts = text.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                cls = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:])
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError
            except ValueError:
                add("malformed_line", rel, n, f"'{text[:60]}'")
                continue

            if cls < 0 or (names and cls >= len(names)):
                add("bad_class_id", rel, n, f"class {cls}, but only {len(names) if names else '?'} classes")
                continue
            max_id_seen = max(max_id_seen, cls)

            if w <= 0 or h <= 0:
                add("zero_size_box", rel, n, f"w={w:g}, h={h:g}")
                continue
            if (x - w / 2 < -EPS or x + w / 2 > 1 + EPS or
                    y - h / 2 < -EPS or y + h / 2 > 1 + EPS):
                add("out_of_bounds", rel, n,
                    f"box spans x {x - w / 2:.3f}-{x + w / 2:.3f}, y {y - h / 2:.3f}-{y + h / 2:.3f}")
            if w < s["tiny_box"] or h < s["tiny_box"]:
                add("tiny_box", rel, n, f"w={w:.4f}, h={h:.4f} (limit {s['tiny_box']})")

            boxes.append((cls, x, y, w, h))
            per_class[cls] = per_class.get(cls, 0) + 1
            box_sizes.append(math.sqrt(w * h))
            box_centers.append((x, y))

        # Duplicates inside one image
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] == boxes[j][0]:
                    iou = _iou(boxes[i][1:], boxes[j][1:])
                    if iou > s["duplicate_iou"]:
                        add("duplicate_box", rel, None, f"two class-{boxes[i][0]} boxes, IoU {iou:.2f}")

        if not any(l.strip() for l in lines) and key in images:
            add("empty_label", rel, message="no boxes")
        boxes_per_image.append(len(boxes))
        file_boxes[key] = boxes

    # Class names and class balance
    if names_guessed:
        names = [f"class_{i}" for i in range(max_id_seen + 1)]
        add("no_class_names", message="add classes.txt (one name per line) next to images/ and labels/")
    counts = {name: per_class.get(i, 0) for i, name in enumerate(names)}
    for name, c in counts.items():
        if c == 0:
            add("missing_class", message=f"'{name}' has 0 boxes")
    nonzero = [c for c in counts.values() if c > 0]
    if len(nonzero) >= 2:
        ratio = max(nonzero) / min(nonzero)
        if ratio > s["imbalance_ratio"]:
            biggest = max(counts, key=counts.get)
            smallest = min((k for k in counts if counts[k] > 0), key=counts.get)
            add("class_imbalance",
                message=f"'{biggest}' has {counts[biggest]} boxes, '{smallest}' has {counts[smallest]} "
                        f"({ratio:.1f}x, limit {s['imbalance_ratio']}x)")

    # Summary
    totals = {code: 0 for code, _, _ in CHECKS}
    for it in issues:
        totals[it["check"]] += 1
    n_err = sum(1 for it in issues if it["severity"] == ERROR)
    n_warn = sum(1 for it in issues if it["severity"] == WARNING)
    status = "fail" if n_err else ("warnings" if n_warn else "pass")
    n_boxes = sum(per_class.values())

    return {
        "dataset": dataset_dir,
        "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "errors": n_err,
        "warnings": n_warn,
        "settings": s,
        "images_checked": images_checked,
        "stats": {
            "images": len(images),
            "label_files": len(labels),
            "boxes": n_boxes,
            "boxes_per_image_mean": round(n_boxes / len(boxes_per_image), 2) if boxes_per_image else 0,
            "boxes_per_image_max": max(boxes_per_image) if boxes_per_image else 0,
            "empty_images": sum(1 for b in boxes_per_image if b == 0),
            "classes": names,
            "boxes_per_class": counts,
            "resolutions": resolutions,
            "boxes_per_image": boxes_per_image,
            "box_sizes": [round(v, 4) for v in box_sizes],
            "box_centers": [(round(x, 3), round(y, 3)) for x, y in box_centers],
        },
        "checks": [{"check": c, "severity": sev, "description": d, "count": totals[c]}
                   for c, sev, d in CHECKS],
        "issues": issues,
        "_files": {"images": images, "boxes": file_boxes},   # for the HTML gallery, not saved
    }


def save_json(report, path):
    data = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summary(report):
    """Short dict stored in metadata.json after a generate run."""
    return {
        "status": report["status"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "checked_at": report["checked_at"],
        "report": "quality_report.html",
    }


def format_text(report, limit=15):
    st = report["stats"]
    icon = {"pass": "PASS", "warnings": "PASS WITH WARNINGS", "fail": "FAIL"}[report["status"]]
    lines = [
        f"Dataset: {report['dataset']}",
        f"Images {st['images']}, boxes {st['boxes']} "
        f"({st['boxes_per_image_mean']} per image), empty images {st['empty_images']}",
        "Boxes per class: " + ", ".join(f"{k} {v}" for k, v in st["boxes_per_class"].items()),
    ]
    if not report["images_checked"]:
        lines.append("(image files not opened: install pillow to check for broken images)")
    for c in report["checks"]:
        if c["count"]:
            tag = "ERROR  " if c["severity"] == ERROR else "WARNING"
            lines.append(f"  {tag} {c['check']}: {c['count']}  - {c['description']}")
    shown = report["issues"][:limit]
    if shown:
        lines.append("")
        for it in shown:
            where = it["file"] or ""
            if it["line"]:
                where += f":{it['line']}"
            lines.append(f"  [{it['check']}] {where} {it['message']}".rstrip())
        if len(report["issues"]) > limit:
            lines.append(f"  ... and {len(report['issues']) - limit} more (see validation.json)")
    lines.append("")
    lines.append(f"RESULT: {icon} ({report['errors']} errors, {report['warnings']} warnings)")
    return "\n".join(lines)
