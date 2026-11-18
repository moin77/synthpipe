"""Human-in-the-loop round-trip with CVAT (YOLO 1.1 format).

export: pre-labeled dataset -> folder you upload to CVAT
    images.zip              images renamed 0001_<name>.jpg in review order (worst first),
                            so CVAT's default sorting shows them in that order
    annotations_yolo11.zip  the model's boxes, uploaded with "Upload annotations -> YOLO 1.1"
    cvat_labels.json        paste into the task's Labels -> Raw tab (same names, same order)
    manifest.json           maps each CVAT name back to the original image
    HOW_TO_REVIEW.txt       the steps

import: CVAT's "Export task dataset -> YOLO 1.1" zip -> corrected YOLO dataset with
    original image names, a validation report, and a record of what the reviewer changed.
"""

import datetime
import io
import json
import os
import re
import shutil
import zipfile

from .validate import IMAGE_EXTS, _index, read_class_names

PREFIX = re.compile(r"^\d{4,}_")


def _review_order(dataset_dir, keys):
    """Review queue first (as written by autolabel), then the rest by name."""
    order = []
    q = os.path.join(dataset_dir, "review_queue.txt")
    if os.path.isfile(q):
        with open(q, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                key = os.path.splitext(line.split("\t")[0].strip())[0].replace("\\", "/")
                if key in keys and key not in order:
                    order.append(key)
    return order + sorted(k for k in keys if k not in order), len(order)


def _write_image(src, max_side, quality=92):
    """Bytes for CVAT. With max_side, the image is shrunk (EXIF rotation applied first, so the
    picture CVAT shows matches the boxes). Normalised YOLO boxes don't change with size."""
    if not max_side:
        with open(src, "rb") as f:
            return f.read(), os.path.splitext(src)[1].lower()
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            r = max_side / max(im.size)
            im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), ".jpg"


def export_for_cvat(dataset_dir, out_dir, max_side=1600, task_name=None):
    dataset_dir = os.path.abspath(dataset_dir)
    names = read_class_names(dataset_dir)
    if not names:
        raise FileNotFoundError(f"No classes.txt in {dataset_dir}")
    images = _index(os.path.join(dataset_dir, "images"), IMAGE_EXTS)
    labels = _index(os.path.join(dataset_dir, "labels"), (".txt",))
    if not images:
        raise FileNotFoundError(f"No images in {dataset_dir}/images")

    order, n_review = _review_order(dataset_dir, set(images))
    width = max(4, len(str(len(order))))
    os.makedirs(out_dir, exist_ok=True)

    mapping = {}
    n_boxes = 0
    img_zip = os.path.join(out_dir, "images.zip")
    ann_zip = os.path.join(out_dir, "annotations_yolo11.zip")
    with zipfile.ZipFile(img_zip, "w", zipfile.ZIP_STORED) as zi, \
            zipfile.ZipFile(ann_zip, "w", zipfile.ZIP_DEFLATED) as za:
        train_lines = []
        for i, key in enumerate(order, start=1):
            base = os.path.basename(key)
            data, ext = _write_image(images[key], max_side)
            stem = f"{i:0{width}d}_{base}"
            zi.writestr(stem + ext, data)
            text = ""
            if key in labels:
                with open(labels[key], encoding="utf-8") as f:
                    text = f.read()
            n_boxes += sum(1 for line in text.splitlines() if line.strip())
            za.writestr(f"obj_train_data/{stem}.txt", text)
            train_lines.append(f"obj_train_data/{stem}{ext}")
            mapping[stem] = {"key": key, "source_image": images[key]}
        za.writestr("obj.names", "\n".join(names) + "\n")
        za.writestr("obj.data", f"classes = {len(names)}\nnames = obj.names\ntrain = train.txt\n")
        za.writestr("train.txt", "\n".join(train_lines) + "\n")

    with open(os.path.join(out_dir, "cvat_labels.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": n, "type": "rectangle", "attributes": []} for n in names], f, indent=2)

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "prelabeled_dataset": dataset_dir,
        "classes": names,
        "max_side": max_side,
        "images": mapping,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    task = task_name or os.path.basename(dataset_dir.rstrip("/\\"))
    with open(os.path.join(out_dir, "HOW_TO_REVIEW.txt"), "w", encoding="utf-8") as f:
        f.write(HOW_TO.format(task=task, labels=", ".join(names), n=len(order), r=n_review))
    return {"images": len(order), "boxes": n_boxes, "classes": names,
            "images_zip_mb": round(os.path.getsize(img_zip) / 1e6, 1), "out_dir": out_dir,
            "review_first": n_review}


HOW_TO = """Review the pre-labels in CVAT ({n} images)
==========================================

1. Create the task (app.cvat.ai, or your own CVAT)
   - Tasks -> + -> Create a new task. Name: {task}
   - Labels: open the "Raw" tab, replace its content with everything in
     cvat_labels.json, click Done. You should see: {labels}
     (same names, same order: the class numbers depend on it)
   - Select files: drag in images.zip
   - Advanced configuration -> Sorting method: Lexicographical (the default)
   - Submit & Open

2. Load the model's boxes
   - Open the task -> Actions -> Upload annotations
   - Format: YOLO 1.1, file: annotations_yolo11.zip -> OK

3. Review
   - Open the job. Images are numbered so the {r} most likely to need work
     (no boxes found, or low confidence) come first.
   - Start a timer. For each image: delete wrong boxes (Del), fix classes,
     drag box edges, draw missed tools (N, then drag). Save often (Ctrl+S).
   - Stop the timer when done and note the minutes.

4. Export the corrected labels
   - Task -> Actions -> Export task dataset -> Format: YOLO 1.1,
     "Save images" OFF -> download the zip.

5. Bring them back
   python -m synthpipe cvat-import <downloaded.zip> --task-dir <this folder> --out <new folder> --minutes <review minutes>
"""


# ---------------------------------------------------------------------
# import
# ---------------------------------------------------------------------
def read_cvat_yolo_zip(zip_path):
    """Returns (names from obj.names, {stem: label text})."""
    names, files = None, {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            path = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            base = path.rsplit("/", 1)[-1]
            if base == "obj.names":
                names = [l.strip() for l in z.read(info).decode("utf-8").splitlines() if l.strip()]
            elif path.endswith(".txt") and re.search(r"(^|/)obj_[^/]+_data/", path):
                stem = os.path.splitext(base)[0]
                files[stem] = z.read(info).decode("utf-8")
    if names is None:
        raise ValueError("No obj.names in the zip. Export from CVAT with format 'YOLO 1.1'.")
    if not files:
        raise ValueError("No label files (obj_train_data/*.txt) in the zip.")
    return names, files


def import_from_cvat(zip_path, task_dir, out_dir):
    """Write a corrected YOLO dataset. Classes are matched by NAME, so a different label
    order in CVAT can't silently swap classes. Returns a summary dict."""
    with open(os.path.join(task_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    classes = manifest["classes"]
    cvat_names, files = read_cvat_yolo_zip(zip_path)

    unknown = [n for n in cvat_names if n not in classes]
    remap = {i: classes.index(n) for i, n in enumerate(cvat_names) if n in classes}

    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    imported, missing, dropped = 0, [], 0
    for stem, entry in manifest["images"].items():
        key, src = entry["key"], entry["source_image"]
        text = files.get(stem)
        if text is None:
            # CVAT leaves out files for images it has no annotations for in some versions
            text = ""
            missing.append(stem)
        lines = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            if cid not in remap:
                dropped += 1
                continue
            lines.append(" ".join([str(remap[cid])] + parts[1:]))
        dst_lbl = os.path.join(lbl_out, key + ".txt")
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        dst_img = os.path.join(img_out, key + os.path.splitext(src)[1])
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src, dst_img)
        imported += 1

    extra = sorted(set(files) - set(manifest["images"]))
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")
    return {
        "images": imported,
        "cvat_labels": cvat_names,
        "unknown_labels": unknown,
        "boxes_dropped_unknown_label": dropped,
        "images_without_label_file": len(missing),
        "label_files_not_in_task": extra,
        "prelabeled_dataset": manifest["prelabeled_dataset"],
        "classes": classes,
    }
