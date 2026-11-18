"""Auto-labeling: a trained detector pre-labels new images, a human then corrects them.

The model's boxes are written as a normal YOLO dataset (images/, labels/, classes.txt),
plus predictions.json with every confidence score and review_queue.txt listing the
images a person should check first (no boxes found, or a low-confidence box).

If human labels already exist for the same images, compare_labels() measures how
many pre-labels were right and how many edits a person would still need to make.
"""

import datetime
import hashlib
import json
import os
import shutil

from .validate import IMAGE_EXTS, _index, _iou


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
class YoloPredictor:
    """Thin wrapper around Ultralytics YOLO so the rest of the code has no Ultralytics imports.

    Any object with .names (list of class names) and .predict(paths) works, which is how
    the tests run without a GPU or a model file.
    """

    def __init__(self, weights, conf=0.25, iou=0.7, imgsz=640, device=None, batch=8):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from e
        self.model = YOLO(weights)
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        self.settings = {"conf": conf, "iou": iou, "imgsz": imgsz, "device": device}
        self.batch = batch

    def predict(self, paths):
        """Yield (path, [(cls, x, y, w, h, conf), ...]) with YOLO-normalised boxes."""
        kwargs = {k: v for k, v in self.settings.items() if v is not None}
        for start in range(0, len(paths), self.batch):
            chunk = paths[start:start + self.batch]
            for path, r in zip(chunk, self.model.predict(source=chunk, verbose=False, **kwargs)):
                b = r.boxes
                boxes = [
                    (int(c), float(x), float(y), float(w), float(h), float(p))
                    for (x, y, w, h), c, p in zip(b.xywhn.tolist(), b.cls.tolist(), b.conf.tolist())
                ]
                yield path, boxes


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# pre-labeling
# ---------------------------------------------------------------------
def run_autolabel(images_dir, out_dir, predictor, review_conf=0.5, copy_images=True, echo=True):
    """Pre-label every image in images_dir and write a YOLO dataset to out_dir.

    review_conf: an image goes on the review queue if it has no boxes or any box below this.
    Returns a summary dict (also saved inside metadata.json under "autolabel").
    """
    images = _index(images_dir, IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    keys = sorted(images)

    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    names = list(predictor.names)
    per_image = {}
    n_boxes = 0
    per_class = {n: 0 for n in names}
    confidences = []
    started = datetime.datetime.now()

    key_by_path = {images[k]: k for k in keys}
    for done, (path, boxes) in enumerate(predictor.predict([images[k] for k in keys]), start=1):
        key = key_by_path[path]
        ext = os.path.splitext(path)[1]
        if copy_images:
            dst = os.path.join(img_out, key + ext)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)
        lbl = os.path.join(lbl_out, key + ".txt")
        os.makedirs(os.path.dirname(lbl), exist_ok=True)
        with open(lbl, "w", encoding="utf-8") as f:
            for c, x, y, w, h, _ in boxes:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        min_conf = min((b[5] for b in boxes), default=None)
        needs_review = not boxes or min_conf < review_conf
        per_image[key] = {
            "file": key + ext,
            "boxes": [{"class": names[c] if c < len(names) else str(c), "class_id": c,
                       "xywhn": [round(x, 6), round(y, 6), round(w, 6), round(h, 6)],
                       "conf": round(p, 4)} for c, x, y, w, h, p in boxes],
            "min_conf": round(min_conf, 4) if min_conf is not None else None,
            "review": needs_review,
            "reason": "no boxes found" if not boxes else
                      (f"lowest confidence {min_conf:.2f}" if needs_review else ""),
        }
        n_boxes += len(boxes)
        for c, *_, p in boxes:
            if c < len(names):
                per_class[names[c]] += 1
            confidences.append(p)
        if echo:
            flag = "  REVIEW" if needs_review else ""
            print(f"  [{done}/{len(keys)}] {key}: {len(boxes)} boxes{flag}")

    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    # Review queue: empty images first, then lowest confidence first
    queue = sorted((k for k in per_image if per_image[k]["review"]),
                   key=lambda k: (per_image[k]["min_conf"] is not None, per_image[k]["min_conf"] or 0))
    with open(os.path.join(out_dir, "review_queue.txt"), "w", encoding="utf-8") as f:
        f.write("# Check these images first (no boxes, or a box below the review confidence)\n")
        for k in queue:
            f.write(f"{per_image[k]['file']}\t{per_image[k]['reason']}\n")

    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": names, "review_conf": review_conf, "images": per_image}, f, indent=2)

    summary = {
        "images": len(keys),
        "boxes": n_boxes,
        "boxes_per_class": per_class,
        "needs_review": len(queue),
        "no_boxes": sum(1 for v in per_image.values() if not v["boxes"]),
        "review_conf": review_conf,
        "confidences": [round(c, 4) for c in confidences],
        "minutes": round((datetime.datetime.now() - started).total_seconds() / 60, 2),
    }
    return summary


# ---------------------------------------------------------------------
# compare with human labels
# ---------------------------------------------------------------------
def _read_labels(path):
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:5])))
                except ValueError:
                    pass
    return boxes


def compare_labels(pred_dir, truth_dir, names, iou_ok=0.5, iou_touch=0.1):
    """Score pre-labels against human labels, as the edits a reviewer would make.

    For each human box, the best unused predicted box (by IoU) is matched:
      correct     same class, IoU >= iou_ok              -> keep as is
      wrong_class IoU >= iou_ok, other class             -> change the class
      loose       same class, iou_touch <= IoU < iou_ok  -> drag the box edges
      missed      no prediction overlaps                 -> draw a new box
    Predicted boxes left unmatched are 'extra'           -> delete
    """
    truth = _index(truth_dir, (".txt",))
    pred = _index(pred_dir, (".txt",))
    keys = sorted(set(truth) & set(pred))
    if not keys:
        raise FileNotFoundError("No label files with matching names in both folders")

    n = len(names)
    zero = lambda: {"correct": 0, "wrong_class": 0, "loose": 0, "missed": 0, "extra": 0, "truth": 0}
    per_class = {name: zero() for name in names}
    total = zero()
    confusion = [[0] * n for _ in range(n)]      # truth class -> predicted class, for IoU >= iou_ok
    per_image = {}

    def bump(cls, field):
        name = names[cls] if 0 <= cls < n else f"class_{cls}"
        per_class.setdefault(name, zero())[field] += 1
        total[field] += 1

    for key in keys:
        t_boxes = _read_labels(truth[key])
        p_boxes = _read_labels(pred[key])
        pairs = []
        for i, t in enumerate(t_boxes):
            for j, p in enumerate(p_boxes):
                v = _iou(t[1:], p[1:])
                if v >= iou_touch:
                    pairs.append((v, t[0] == p[0], i, j))
        # Best overlaps first; same-class beats other-class at equal IoU
        pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        used_t, used_p, edits = set(), set(), 0
        for v, same, i, j in pairs:
            if i in used_t or j in used_p:
                continue
            if v < iou_ok and not same:
                continue            # a weak overlap with another class is not a match
            used_t.add(i)
            used_p.add(j)
            tc, pc = t_boxes[i][0], p_boxes[j][0]
            bump(tc, "truth")
            if v >= iou_ok:
                if 0 <= tc < n and 0 <= pc < n:
                    confusion[tc][pc] += 1
                if same:
                    bump(tc, "correct")
                else:
                    bump(tc, "wrong_class")
                    edits += 1
            else:
                bump(tc, "loose")
                edits += 1
        for i, t in enumerate(t_boxes):
            if i not in used_t:
                bump(t[0], "truth")
                bump(t[0], "missed")
                edits += 1
        for j, p in enumerate(p_boxes):
            if j not in used_p:
                bump(p[0], "extra")
                edits += 1
        per_image[key] = edits

    edits = total["wrong_class"] + total["loose"] + total["missed"] + total["extra"]
    untouched = sum(1 for e in per_image.values() if e == 0)
    return {
        "images": len(keys),
        "iou_ok": iou_ok,
        "totals": total,
        "per_class": per_class,
        "confusion": {"classes": names, "matrix": confusion},
        "edits": edits,
        "boxes_kept_pct": round(100 * total["correct"] / total["truth"], 1) if total["truth"] else 0.0,
        "images_without_edits": untouched,
        "images_without_edits_pct": round(100 * untouched / len(keys), 1),
        "edits_per_image": per_image,
    }


def format_comparison(c):
    t = c["totals"]
    lines = [
        f"Compared {c['images']} images with human labels (match = IoU >= {c['iou_ok']}).",
        f"Human boxes: {t['truth']}",
        f"  correct, keep as is : {t['correct']} ({c['boxes_kept_pct']}%)",
        f"  wrong class         : {t['wrong_class']}",
        f"  box too loose/tight : {t['loose']}",
        f"  missed, must draw   : {t['missed']}",
        f"  extra, must delete  : {t['extra']}",
        f"Edits needed: {c['edits']} in total; {c['images_without_edits']} images "
        f"({c['images_without_edits_pct']}%) need no edits at all.",
        "",
        f"{'class':<14}{'human':>7}{'correct':>9}{'wrong':>7}{'loose':>7}{'missed':>8}{'extra':>7}",
    ]
    for name, v in c["per_class"].items():
        lines.append(f"{name:<14}{v['truth']:>7}{v['correct']:>9}{v['wrong_class']:>7}"
                     f"{v['loose']:>7}{v['missed']:>8}{v['extra']:>7}")
    return "\n".join(lines)
