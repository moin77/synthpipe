"""Compare two generated runs: the reproducibility test.

Labels must match byte for byte. Images are compared by file hash first;
if hashes differ (GPU denoising can change a few pixels between runs),
the mean pixel difference is measured when Pillow and NumPy are installed.
"""

import hashlib
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(folder, exts):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


def _pixel_diff(a, b):
    """Mean absolute difference per pixel channel (0-255), or None if it can't be measured."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
        ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    except OSError:          # unreadable or broken image: count it as different
        return float("inf")
    if ia.shape != ib.shape:
        return float("inf")
    return float(np.abs(ia - ib).mean())


def compare_runs(run_a, run_b, pixel_tolerance=0.0):
    report = {"run_a": run_a, "run_b": run_b, "labels": {}, "images": {}}

    # Labels
    la = _files(os.path.join(run_a, "labels"), (".txt",))
    lb = _files(os.path.join(run_b, "labels"), (".txt",))
    common = sorted(set(la) & set(lb))
    lab_diff = [f for f in common
                if _sha256(os.path.join(run_a, "labels", f)) != _sha256(os.path.join(run_b, "labels", f))]
    report["labels"] = {
        "compared": len(common),
        "only_in_a": sorted(set(la) - set(lb)),
        "only_in_b": sorted(set(lb) - set(la)),
        "different": lab_diff,
    }

    # Images
    ia = _files(os.path.join(run_a, "images"), IMAGE_EXTS)
    ib = _files(os.path.join(run_b, "images"), IMAGE_EXTS)
    common_i = sorted(set(ia) & set(ib))
    exact, close, different, unmeasured = 0, [], [], []
    worst = 0.0
    for f in common_i:
        pa, pb = os.path.join(run_a, "images", f), os.path.join(run_b, "images", f)
        if _sha256(pa) == _sha256(pb):
            exact += 1
            continue
        d = _pixel_diff(pa, pb)
        if d is None:
            unmeasured.append(f)
            continue
        worst = max(worst, d)
        (close if d <= pixel_tolerance else different).append((f, round(d, 4)))
    report["images"] = {
        "compared": len(common_i),
        "only_in_a": sorted(set(ia) - set(ib)),
        "only_in_b": sorted(set(ib) - set(ia)),
        "byte_identical": exact,
        "within_tolerance": close,
        "different": different,
        "not_measured": unmeasured,
        "worst_mean_pixel_diff": round(worst, 4),
    }

    L, I = report["labels"], report["images"]
    report["identical"] = (
        L["compared"] > 0
        and not (L["only_in_a"] or L["only_in_b"] or L["different"])
        and not (I["only_in_a"] or I["only_in_b"] or I["different"] or I["not_measured"])
    )
    return report


def format_report(r):
    L, I = r["labels"], r["images"]
    lines = [
        f"Run A: {r['run_a']}",
        f"Run B: {r['run_b']}",
        "",
        f"Labels: {L['compared']} compared, {len(L['different'])} different",
    ]
    if L["only_in_a"] or L["only_in_b"]:
        lines.append(f"  missing: {len(L['only_in_a'])} only in A, {len(L['only_in_b'])} only in B")
    for f in L["different"][:10]:
        lines.append(f"  differs: {f}")
    lines.append(
        f"Images: {I['compared']} compared, {I['byte_identical']} byte-identical, "
        f"{len(I['within_tolerance'])} within tolerance, {len(I['different'])} different"
    )
    if I["worst_mean_pixel_diff"]:
        lines.append(f"  worst mean pixel difference: {I['worst_mean_pixel_diff']} (out of 255)")
    if I["only_in_a"] or I["only_in_b"]:
        lines.append(f"  missing: {len(I['only_in_a'])} only in A, {len(I['only_in_b'])} only in B")
    for f, d in I["different"][:10]:
        lines.append(f"  differs: {f} (mean diff {d})")
    if I["not_measured"]:
        lines.append(f"  {len(I['not_measured'])} images differ by hash; install pillow + numpy to measure")
    lines.append("")
    lines.append("RESULT: IDENTICAL" if r["identical"] else "RESULT: NOT IDENTICAL")
    return "\n".join(lines)
