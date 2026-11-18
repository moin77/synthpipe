"""Stand-in for blender.exe in tests: no Blender needed.

Accepts the same command line SynthPipe builds, reads --config and writes
a tiny deterministic dataset in the same layout as the real generator.
A config named "crash" makes it fail like a broken generator script.
"""

import json
import os
import random
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
cfg_path = argv[argv.index("--config") + 1]
with open(cfg_path, encoding="utf-8") as f:
    cfg = json.load(f)

print(f"Loaded config: {cfg_path}")
if cfg["name"] == "crash":
    print("Traceback (most recent call last):")
    print("RuntimeError: Object 'pliers_a' not found in this .blend file")
    sys.exit(1)

out = cfg["output"]["dir"]
os.makedirs(os.path.join(out, "images"), exist_ok=True)
os.makedirs(os.path.join(out, "labels"), exist_ok=True)
counts = {c: 0 for c in cfg["classes"]}
for i in range(cfg["start"], cfg["start"] + cfg["count"]):
    rng = random.Random(cfg["seed"] * 1_000_003 + i)
    name = f"synth_{i:05d}"
    with open(os.path.join(out, "images", name + ".jpg"), "wb") as f:
        f.write(bytes(rng.randrange(256) for _ in range(64)))
    lines = []
    for _ in range(rng.randint(*cfg["randomization"]["tools_per_image"])):
        cid = rng.randrange(len(cfg["classes"]))
        counts[cfg["classes"][cid]] += 1
        lines.append(f"{cid} {rng.random():.6f} {rng.random():.6f} 0.100000 0.100000")
    with open(os.path.join(out, "labels", name + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{i - cfg['start'] + 1}/{cfg['count']}] {name}: {len(lines)} boxes")
    print("Fra:1 Mem:12M | Time:00:01.00 | Rendering")   # noise line, should be hidden

with open(os.path.join(out, "classes.txt"), "w") as f:
    f.write("\n".join(cfg["classes"]) + "\n")
with open(os.path.join(out, "metadata.json"), "w") as f:
    json.dump({"generator": "fake", "seed": cfg["seed"], "boxes_per_class": counts}, f, indent=2)
print("Done. Boxes per class:", counts)
