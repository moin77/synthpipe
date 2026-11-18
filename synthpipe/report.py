"""Turn a validation report into one self-contained HTML file (no internet needed to view it)."""

import base64
import html
import io
import os

from .validate import ERROR

# Reference palette (dataviz skill): one series hue, sequential ramp, status colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

e = html.escape


# ---------------------------------------------------------------------
# charts (inline SVG)
# ---------------------------------------------------------------------
def _nice_max(v):
    if v <= 0:
        return 1
    mag = 10 ** (len(str(int(v))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def class_bars(counts):
    """Horizontal bars, one per class, value labelled at the bar end."""
    names = list(counts)
    if not names:
        return "<p class='muted'>No classes.</p>"
    row, left, width, right = 34, 120, 980, 70
    h = row * len(names) + 10
    top = max(counts.values()) or 1
    out = [f"<svg viewBox='0 0 {width} {h}' class='chart' role='img' aria-label='Boxes per class'>"]
    plot = width - left - right
    for i, name in enumerate(names):
        c = counts[name]
        y = 5 + i * row
        w = max(2.0, plot * c / top) if c else 0
        tip = f"{e(name)}: {c} boxes"
        out.append(f"<text x='{left - 10}' y='{y + 15}' class='lbl' text-anchor='end'>{e(name)}</text>")
        out.append(f"<rect x='{left}' y='{y}' width='{plot}' height='{row - 8}' class='hit' data-tip='{tip}'/>")
        if w:
            out.append(f"<path d='{_bar_h(left, y + 2, w, row - 12)}' class='bar' data-tip='{tip}'/>")
        out.append(f"<text x='{left + w + 8}' y='{y + 15}' class='val'>{c}</text>")
    out.append(f"<line x1='{left}' y1='0' x2='{left}' y2='{h - 5}' class='axis'/>")
    out.append("</svg>")
    return "".join(out)


def _bar_h(x, y, w, h, r=4):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    r = min(r, w / 2, h / 2)
    return (f"M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} "
            f"V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z")


def _bar_v(x, y_base, w, h, r=4):
    """Vertical bar, square at the baseline, rounded at the top."""
    r = min(r, w / 2, h / 2)
    top = y_base - h
    return (f"M{x},{y_base} V{top + r} Q{x},{top} {x + r},{top} "
            f"H{x + w - r} Q{x + w},{top} {x + w},{top + r} V{y_base} Z")


def histogram(bins, labels, aria, tips):
    """Vertical bars. bins: counts, labels: x-axis label per bin (some may be ''),
    tips: hover text per bin."""
    width, height, left, bottom, top_pad = 470, 210, 36, 30, 10
    plot_w, plot_h = width - left - 10, height - bottom - top_pad
    ymax = _nice_max(max(bins) if bins else 0)
    if ymax % 4:                       # keep the 4 grid steps on whole numbers
        ymax = -(-ymax // 4) * 4
    n = max(1, len(bins))
    slot = plot_w / n
    gap = 2 if slot > 6 else 0
    base = top_pad + plot_h
    out = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' aria-label='{e(aria)}'>"]
    for k in range(5):       # recessive grid
        v = ymax * k / 4
        y = base - plot_h * k / 4
        out.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width - 10}' y2='{y:.1f}' class='grid'/>")
        out.append(f"<text x='{left - 6}' y='{y + 4:.1f}' class='tick' text-anchor='end'>{v:g}</text>")
    for i, c in enumerate(bins):
        x = left + i * slot + gap / 2
        bh = plot_h * c / ymax if ymax else 0
        tip = e(tips[i])
        out.append(f"<rect x='{left + i * slot:.1f}' y='{top_pad}' width='{slot:.1f}' height='{plot_h}' "
                   f"class='hit' data-tip='{tip}'/>")
        if c:
            out.append(f"<path d='{_bar_v(x, base, slot - gap, max(bh, 2))}' class='bar' data-tip='{tip}'/>")
        if labels[i]:
            out.append(f"<text x='{left + i * slot + slot / 2:.1f}' y='{base + 18}' class='tick' "
                       f"text-anchor='middle'>{e(labels[i])}</text>")
    out.append(f"<line x1='{left}' y1='{base}' x2='{width - 10}' y2='{base}' class='axis'/>")
    out.append("</svg>")
    return "".join(out)


def center_heatmap(centers, grid=10):
    """Where box centres land in the frame. Synthetic data that always centres
    objects shows up here as one hot spot."""
    cells = [[0] * grid for _ in range(grid)]
    for x, y in centers:
        i = min(grid - 1, max(0, int(x * grid)))
        j = min(grid - 1, max(0, int(y * grid)))
        cells[j][i] += 1
    top = max(max(r) for r in cells) or 1
    size, cell = 300, 30
    out = [f"<svg viewBox='0 0 {size + 2} {size + 2}' class='chart heat' role='img' "
           "aria-label='Box centre positions'>"]
    for j in range(grid):
        for i in range(grid):
            c = cells[j][i]
            col = SEQ[min(len(SEQ) - 1, int(c / top * (len(SEQ) - 1)))] if c else "var(--empty)"
            tip = f"x {i * 10}-{i * 10 + 10}%, y {j * 10}-{j * 10 + 10}%: {c} box{'es' if c != 1 else ''}"
            out.append(f"<rect x='{1 + i * cell}' y='{1 + j * cell}' width='{cell - 2}' height='{cell - 2}' "
                       f"rx='3' fill='{col}' data-tip='{tip}'/>")
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------
def _thumb(path, width=360):
    """Small base64 JPEG. Without Pillow, the original file is embedded."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos store rotation in EXIF
            ratio = width / im.width
            im = im.resize((width, max(1, round(im.height * ratio))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
            data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        with open(path, "rb") as f:
            data = f.read()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    except Exception:
        return None
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def pick_samples(report, n):
    """Images with issues first, then evenly spaced ones, so the gallery shows problems and variety."""
    files = report["_files"]
    keys = sorted(files["images"])
    flagged = []
    for it in report["issues"]:
        f = it["file"] or ""
        key = os.path.splitext(f.split("/", 1)[1])[0] if "/" in f else None
        if key in files["images"] and key not in flagged:
            flagged.append(key)
    picks = flagged[: n // 2]
    rest = [k for k in keys if k not in picks]
    need = n - len(picks)
    if rest and need > 0:
        step = max(1, len(rest) // need)
        picks += rest[::step][:need]
    return picks, set(flagged)


def gallery(report, n):
    if n <= 0:
        return ""
    picks, flagged = pick_samples(report, n)
    names = report["stats"]["classes"]
    issues_by_key = {}
    for it in report["issues"]:
        f = it["file"] or ""
        if "/" in f:
            issues_by_key.setdefault(os.path.splitext(f.split("/", 1)[1])[0], []).append(it["check"])
    cards = []
    for key in picks:
        src = _thumb(report["_files"]["images"][key])
        if not src:
            continue
        boxes = report["_files"]["boxes"].get(key, [])
        shapes = []
        for cls, x, y, w, h in boxes:
            k = cls if cls < len(SERIES) else 0
            name = names[cls] if cls < len(names) else str(cls)
            shapes.append(
                f"<rect x='{x - w / 2:.4f}' y='{y - h / 2:.4f}' width='{w:.4f}' height='{h:.4f}' "
                f"class='box c{k}'/>")
            shapes.append(f"<span class='tag c{k}' style='left:{max(0, x - w / 2) * 100:.2f}%;"
                          f"top:{max(0, y - h / 2) * 100:.2f}%'>{e(name)}</span>")
        svg = "".join(s for s in shapes if s.startswith("<rect"))
        tags = "".join(s for s in shapes if s.startswith("<span"))
        badge = ""
        if key in flagged:
            badge = "<span class='flag'>⚠ " + e(", ".join(sorted(set(issues_by_key.get(key, []))))) + "</span>"
        cards.append(
            "<figure class='card'><div class='frame'>"
            f"<img src='{src}' alt='{e(key)}'/>"
            f"<svg viewBox='0 0 1 1' preserveAspectRatio='none' class='overlay'>{svg}</svg>{tags}"
            f"</div><figcaption>{e(key)} · {len(boxes)} box{'es' if len(boxes) != 1 else ''} {badge}</figcaption></figure>")
    if not cards:
        return ""
    return "<div class='gallery'>" + "".join(cards) + "</div>"


# ---------------------------------------------------------------------
# page
# ---------------------------------------------------------------------
STATUS_TEXT = {
    "pass": ("good", "✓", "Passed"),
    "warnings": ("warning", "!", "Passed with warnings"),
    "fail": ("critical", "✕", "Failed"),
}


def render_html(report, title=None, samples=12, extra_html=""):
    st = report["stats"]
    kind, icon, label = STATUS_TEXT[report["status"]]
    title = title or os.path.basename(report["dataset"].rstrip("/\\")) or "dataset"

    # boxes per image histogram (0..max)
    bpi = st["boxes_per_image"]
    mx = max(bpi) if bpi else 0
    bpi_bins = [sum(1 for b in bpi if b == k) for k in range(mx + 1)]
    bpi_chart = histogram(bpi_bins, [str(k) for k in range(mx + 1)], "Boxes per image",
                          [f"{c} image{'s' if c != 1 else ''} with {k} box{'es' if k != 1 else ''}"
                           for k, c in enumerate(bpi_bins)])

    # box size histogram: sqrt(area) as % of image, 5% bins
    nb = 20
    size_bins = [0] * nb
    for v in st["box_sizes"]:
        size_bins[min(nb - 1, int(v * nb))] += 1
    size_labels = [f"{k * 5}%" if k % 2 == 0 else "" for k in range(nb)]
    size_tips = [f"{c} box{'es' if c != 1 else ''} with side {k * 5}-{k * 5 + 5}% of the image"
                 for k, c in enumerate(size_bins)]
    size_chart = histogram(size_bins, size_labels, "Box size", size_tips)

    tiles = [
        ("Images", st["images"]),
        ("Boxes", st["boxes"]),
        ("Boxes per image", st["boxes_per_image_mean"]),
        ("Empty images", st["empty_images"]),
        ("Errors", report["errors"]),
        ("Warnings", report["warnings"]),
    ]
    tiles_html = "".join(f"<div class='tile'><div class='num'>{e(str(v))}</div>"
                         f"<div class='cap'>{e(k)}</div></div>" for k, v in tiles)

    rows = []
    for c in report["checks"]:
        if c["count"] == 0:
            s = "<span class='st good'>✓ OK</span>"
        elif c["severity"] == ERROR:
            s = "<span class='st critical'>✕ Error</span>"
        else:
            s = "<span class='st warning'>! Warning</span>"
        rows.append(f"<tr><td>{s}</td><td><code>{e(c['check'])}</code></td>"
                    f"<td>{e(c['description'])}</td><td class='n'>{c['count']}</td></tr>")

    issue_rows = []
    for it in report["issues"][:300]:
        where = (it["file"] or "") + (f":{it['line']}" if it["line"] else "")
        sev = "✕ Error" if it["severity"] == ERROR else "! Warning"
        cls = "critical" if it["severity"] == ERROR else "warning"
        issue_rows.append(f"<tr><td><span class='st {cls}'>{sev}</span></td><td><code>{e(it['check'])}</code></td>"
                          f"<td><code>{e(where)}</code></td><td>{e(it['message'])}</td></tr>")
    more = len(report["issues"]) - len(issue_rows)
    issues_html = (
        "<table><thead><tr><th>Severity</th><th>Check</th><th>Where</th><th>Detail</th></tr></thead>"
        f"<tbody>{''.join(issue_rows)}</tbody></table>"
        + (f"<p class='muted'>…and {more} more in validation.json</p>" if more > 0 else "")
        if issue_rows else "<p class='muted'>No issues found.</p>"
    )

    class_table = "".join(f"<tr><td>{e(k)}</td><td class='n'>{v}</td>"
                          f"<td class='n'>{(100 * v / st['boxes']) if st['boxes'] else 0:.1f}%</td></tr>"
                          for k, v in st["boxes_per_class"].items())
    res = ", ".join(f"{k} ({v})" for k, v in sorted(st["resolutions"].items(), key=lambda x: -x[1])) \
        or ("not checked (install pillow)" if not report["images_checked"] else "none")

    s = report["settings"]
    meta = (f"Checked {e(report['checked_at'])} · tiny box &lt; {s['tiny_box']:g} of image · "
            f"imbalance limit {s['imbalance_ratio']:g}× · duplicate IoU &gt; {s['duplicate_iou']:g}")

    box_css = "".join(
        f".c{i}{{--c:{SERIES[i]}}}" for i in range(len(SERIES))) + \
        "@media (prefers-color-scheme: dark){" + "".join(
        f".c{i}{{--c:{SERIES_DARK[i]}}}" for i in range(len(SERIES))) + "}"

    gallery_html = gallery(report, samples)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality report · {e(title)}</title>
<style>
:root {{
  color-scheme: light;
  --surface-0:#f4f3f0; --surface-1:#fcfcfb; --border:#e3e2dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7974;
  --series-1:#2a78d6; --grid:#ebeae6; --axis:#c9c8c2; --empty:#efeee9;
  --good:#0ca30c; --good-text:#006300; --warning:#fab219; --warning-text:#8a5d00; --critical:#d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --surface-0:#121211; --surface-1:#1a1a19; --border:#2e2e2c;
    --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
    --series-1:#3987e5; --grid:#2a2a28; --axis:#45453f; --empty:#262624;
    --good-text:#0ca30c; --warning-text:#fab219;
  }}
}}
{box_css}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface-0); color:var(--text-primary);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:32px 20px 60px; }}
h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:17px; margin:0 0 4px; }}
.sub {{ color:var(--text-secondary); margin:0 0 20px; word-break:break-all; }}
.muted {{ color:var(--text-muted); font-size:13px; }}
.status {{ display:inline-flex; gap:8px; align-items:center; font-weight:600; padding:6px 14px;
  border-radius:999px; border:1px solid var(--border); background:var(--surface-1); margin-bottom:18px; }}
.dot {{ width:22px; height:22px; border-radius:50%; display:inline-grid; place-items:center;
  color:#fff; font-size:13px; }}
.dot.good {{ background:var(--good); }} .dot.warning {{ background:var(--warning); color:#0b0b0b; }}
.dot.critical {{ background:var(--critical); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:20px; }}
.tile, .panel {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; }}
.tile {{ padding:14px 16px; }}
.num {{ font-size:26px; font-weight:650; font-variant-numeric:tabular-nums; }}
.cap {{ color:var(--text-secondary); font-size:13px; }}
.panel {{ padding:18px 20px; margin-bottom:16px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:800px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.chart {{ width:100%; height:auto; display:block; margin-top:10px; }}
.heat {{ max-width:300px; }}
.bar {{ fill:var(--series-1); }}
.hit {{ fill:transparent; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.axis {{ stroke:var(--axis); stroke-width:1; }}
.lbl {{ fill:var(--text-primary); font-size:13px; }}
.val {{ fill:var(--text-secondary); font-size:13px; font-variant-numeric:tabular-nums; }}
.tick {{ fill:var(--text-muted); font-size:11px; font-variant-numeric:tabular-nums; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; margin-top:10px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
th {{ color:var(--text-secondary); font-weight:550; font-size:13px; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ font:12.5px ui-monospace,Consolas,monospace; }}
.st {{ white-space:nowrap; font-weight:600; font-size:13px; }}
.st.good {{ color:var(--good-text); }} .st.warning {{ color:var(--warning-text); }}
.st.critical {{ color:var(--critical); }}
details summary {{ cursor:pointer; color:var(--text-secondary); font-size:13px; margin-top:8px; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; margin-top:12px; }}
.card {{ margin:0; }}
.frame {{ position:relative; border-radius:8px; overflow:hidden; background:var(--empty); }}
.frame img {{ width:100%; display:block; }}
.overlay {{ position:absolute; inset:0; width:100%; height:100%; }}
.box {{ fill:none; stroke:var(--c); stroke-width:2.5; vector-effect:non-scaling-stroke; }}
.tag {{ position:absolute; background:var(--c); color:#fff; font-size:10.5px; line-height:1;
  padding:3px 5px; border-radius:0 0 4px 0; font-weight:600; text-shadow:0 0 2px rgba(0,0,0,.35); }}
figcaption {{ font-size:12.5px; color:var(--text-secondary); margin-top:6px; }}
.flag {{ color:var(--text-primary); font-weight:600; }}
#tip {{ position:fixed; pointer-events:none; background:var(--text-primary); color:var(--surface-1);
  font-size:12.5px; padding:5px 8px; border-radius:6px; opacity:0; transition:opacity .08s; z-index:9; }}
footer {{ color:var(--text-muted); font-size:12.5px; margin-top:24px; }}
</style></head>
<body><main>
<h1>Dataset quality report</h1>
<p class="sub">{e(report['dataset'])}</p>
<div class="status"><span class="dot {kind}">{icon}</span>{label}
  <span class="muted">· {report['errors']} errors · {report['warnings']} warnings</span></div>
<div class="tiles">{tiles_html}</div>

{extra_html}

<div class="panel">
  <h2>Checks</h2>
  <table><thead><tr><th>Result</th><th>Check</th><th>What it means</th><th class="n">Found</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
</div>

<div class="panel">
  <h2>Boxes per class</h2>
  <p class="muted">Balanced classes train more evenly. Limit before a warning: {s['imbalance_ratio']:g}× between largest and smallest.</p>
  {class_bars(st['boxes_per_class'])}
  <details><summary>Table view</summary>
  <table><thead><tr><th>Class</th><th class="n">Boxes</th><th class="n">Share</th></tr></thead>
  <tbody>{class_table}</tbody></table></details>
</div>

<div class="grid2">
  <div class="panel"><h2>Boxes per image</h2>
    <p class="muted">How many objects each image holds.</p>{bpi_chart}</div>
  <div class="panel"><h2>Box size</h2>
    <p class="muted">Box side length as a share of the image (√ of box area).</p>{size_chart}</div>
</div>

<div class="panel">
  <h2>Where boxes appear in the frame</h2>
  <p class="muted">Darker = more box centres. One hot spot in the middle means objects are always centred,
  which can hurt detection near the edges.</p>
  {center_heatmap(st['box_centers'])}
</div>

<div class="panel">
  <h2>Sample images</h2>
  <p class="muted">Images with issues first (marked ⚠), then an even spread across the dataset. Resolutions: {e(res)}.</p>
  {gallery_html or "<p class='muted'>No samples.</p>"}
</div>

<div class="panel">
  <h2>All issues</h2>
  {issues_html}
</div>

<footer>Generated by SynthPipe · {meta}</footer>
</main>
<div id="tip" role="tooltip"></div>
<script>
(function(){{
  var tip=document.getElementById('tip');
  document.addEventListener('mousemove',function(ev){{
    var t=ev.target.closest('[data-tip]');
    if(!t){{tip.style.opacity=0;return;}}
    tip.textContent=t.getAttribute('data-tip');
    tip.style.left=Math.min(ev.clientX+12,window.innerWidth-tip.offsetWidth-8)+'px';
    tip.style.top=(ev.clientY+14)+'px';
    tip.style.opacity=1;
  }});
}})();
</script>
</body></html>"""


def save_html(report, path, title=None, samples=12, extra_html=""):
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(report, title, samples, extra_html))


def autolabel_panel(summary, comparison=None):
    """Extra report section for an auto-labeled dataset."""
    conf = summary["confidences"]
    nb = 20
    bins = [0] * nb
    for c in conf:
        bins[min(nb - 1, int(c * nb))] += 1
    labels = [f"{k / nb:.1f}" if k % 2 == 0 else "" for k in range(nb)]
    tips = [f"{b} box{'es' if b != 1 else ''} with confidence {k / nb:.2f}-{(k + 1) / nb:.2f}"
            for k, b in enumerate(bins)]
    mean = sum(conf) / len(conf) if conf else 0
    tiles = [
        ("Pre-labeled images", summary["images"]),
        ("Boxes found", summary["boxes"]),
        ("Mean confidence", f"{mean:.2f}"),
        ("To review first", summary["needs_review"]),
        ("No boxes found", summary["no_boxes"]),
    ]
    out = ["<div class='panel'><h2>Auto-labeling</h2>",
           f"<p class='muted'>Boxes drawn by the model. Images with no boxes or a box below "
           f"{summary['review_conf']:g} confidence are listed in review_queue.txt.</p>",
           "<div class='tiles' style='margin-top:12px'>",
           "".join(f"<div class='tile'><div class='num'>{e(str(v))}</div><div class='cap'>{e(k)}</div></div>"
                   for k, v in tiles),
           "</div><h2 style='margin-top:8px'>Confidence of predicted boxes</h2>",
           "<div style='max-width:560px'>",
           histogram(bins, labels, "Confidence of predicted boxes", tips), "</div>"]
    if comparison:
        t = comparison["totals"]
        rows = [
            ("Correct, keep as is", t["correct"], "good", "✓"),
            ("Wrong class, change it", t["wrong_class"], "warning", "!"),
            ("Box too loose or tight, adjust", t["loose"], "warning", "!"),
            ("Missed, draw a new box", t["missed"], "critical", "✕"),
            ("Extra box, delete", t["extra"], "critical", "✕"),
        ]
        body = "".join(f"<tr><td><span class='st {k}'>{i}</span> {e(n)}</td><td class='n'>{v}</td></tr>"
                       for n, v, k, i in rows)
        cls_rows = "".join(
            f"<tr><td>{e(name)}</td><td class='n'>{v['truth']}</td><td class='n'>{v['correct']}</td>"
            f"<td class='n'>{v['wrong_class']}</td><td class='n'>{v['loose']}</td>"
            f"<td class='n'>{v['missed']}</td><td class='n'>{v['extra']}</td></tr>"
            for name, v in comparison["per_class"].items())
        out.append(
            "<h2 style='margin-top:18px'>Compared with human labels</h2>"
            f"<p class='muted'>{comparison['images']} images, {t['truth']} human boxes, match = IoU ≥ "
            f"{comparison['iou_ok']:g}. <b>{comparison['boxes_kept_pct']}%</b> of boxes needed no change; "
            f"{comparison['images_without_edits']} images ({comparison['images_without_edits_pct']}%) "
            f"needed no edits at all; {comparison['edits']} edits in total.</p>"
            f"<table><thead><tr><th>What the reviewer does</th><th class='n'>Boxes</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
            "<details><summary>Per class</summary><table><thead><tr><th>Class</th><th class='n'>Human</th>"
            "<th class='n'>Correct</th><th class='n'>Wrong class</th><th class='n'>Loose</th>"
            f"<th class='n'>Missed</th><th class='n'>Extra</th></tr></thead><tbody>{cls_rows}</tbody></table></details>")
    out.append("</div>")
    return "".join(out)


def review_panel(changes, minutes=None, agreement=None):
    """Extra report section after a human review in CVAT: what the reviewer changed."""
    t = changes["totals"]
    rows = [
        ("Kept as the model drew them", t["correct"], "good", "✓"),
        ("Class changed", t["wrong_class"], "warning", "!"),
        ("Box adjusted", t["loose"], "warning", "!"),
        ("Drawn by the reviewer (model missed)", t["missed"], "critical", "✕"),
        ("Deleted by the reviewer (model extra)", t["extra"], "critical", "✕"),
    ]
    body = "".join(f"<tr><td><span class='st {k}'>{i}</span> {e(n)}</td><td class='n'>{v}</td></tr>"
                   for n, v, k, i in rows)
    tiles = [("Final boxes", t["truth"]), ("Kept as drawn", f"{changes['boxes_kept_pct']}%"),
             ("Edits made", changes["edits"]),
             ("Images untouched", f"{changes['images_without_edits_pct']}%")]
    if minutes:
        tiles.append(("Review time", f"{minutes:g} min"))
        tiles.append(("Seconds per image", f"{60 * minutes / changes['images']:.0f}"))
    out = ["<div class='panel'><h2>Human review in CVAT</h2>",
           "<p class='muted'>Model pre-labels compared with the reviewed labels (match = IoU ≥ "
           f"{changes['iou_ok']:g}).</p><div class='tiles' style='margin-top:12px'>",
           "".join(f"<div class='tile'><div class='num'>{e(str(v))}</div><div class='cap'>{e(k)}</div></div>"
                   for k, v in tiles),
           "</div><table><thead><tr><th>What happened to each box</th><th class='n'>Boxes</th></tr></thead>"
           f"<tbody>{body}</tbody></table>"]
    if agreement:
        a = agreement["totals"]
        out.append(
            "<h2 style='margin-top:18px'>Agreement with the original human labels</h2>"
            f"<p class='muted'>{agreement['boxes_kept_pct']}% of the original labels match the reviewed ones "
            f"({a['correct']} of {a['truth']}); {a['missed']} original boxes have no match, "
            f"{a['extra']} reviewed boxes are new. Differences show labeling disagreement, not "
            "necessarily errors.</p>")
    out.append("</div>")
    return "".join(out)
