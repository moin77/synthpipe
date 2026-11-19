# Changelog

## v0.5.0: publish to Hugging Face
- `synthpipe publish`: writes a dataset card (README.md) and `data.yaml`, then uploads to the Hugging Face Hub.
- The card is built from metadata, validation results and review records: counts, class table, every randomization range, the rerun command, and limitations found in the data (class imbalance, objects bunched in the middle of the frame, empty images).
- Refuses datasets with validation errors (`--force` overrides) and real photos without `--confirm-rights`.
- First dataset published: [moindedroper/toolvision-synthetic](https://huggingface.co/datasets/moindedroper/toolvision-synthetic).

## v0.4.0: human review in CVAT
- `synthpipe cvat-export`: images renamed in review order (worst first), shrunk for upload, model boxes as a YOLO 1.1 zip, label list for CVAT's Raw tab, step-by-step instructions.
- `synthpipe cvat-import`: corrected labels back under original names; classes matched by name; records what the reviewer kept, changed, drew and deleted, and seconds per image.
- First real review: 106 photos in 40 minutes (23 s per image), 64% of model boxes kept, 93% agreement with earlier hand labels.

## v0.3.0: auto-labeling
- `synthpipe autolabel`: a trained YOLO model pre-labels images; writes confidences and a review queue.
- `--compare-to` scores pre-labels against human labels as reviewer edits (correct, wrong class, loose, missed, extra).
- Phone photos: EXIF rotation applied in report thumbnails.

## v0.2.0: validation and quality report
- `synthpipe validate` for any YOLO dataset; runs automatically after `generate`.
- 8 error checks and 5 warning checks; `validation.json` and a self-contained `quality_report.html` with charts, a box-position heatmap and sample images with boxes.

## v0.1.0: config-driven generation
- `synthpipe generate --config tools.yaml`: YAML config to labeled YOLO dataset through Blender in background mode.
- Config validation before Blender starts; `metadata.json` with config fingerprint, versions and rerun command; `RUN_FAILED.txt` on failure.
- `synthpipe compare`: reproducibility check. Same seed gave identical labels and a 0.0049/255 mean pixel difference (GPU denoiser noise).
