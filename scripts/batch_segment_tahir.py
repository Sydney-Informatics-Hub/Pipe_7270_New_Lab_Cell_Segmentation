#!/usr/bin/env python
"""
Batch cell segmentation — multiple contributor folders
======================================================

A **local, non-app** counterpart to ``app/cell_seg_app.py``. It loads the *same*
trained YOLOv8-seg model the app loads and reproduces the app's exact processing
(grayscale preprocessing → 1024 px inference → mask cleanup → coloured overlay +
16-bit label mask), then runs it over **several** folders, each with its own
filename filter and its own output folder.

What it does
------------
For each configured job it selects every image whose filename contains that job's
filter and writes, under a per-job output folder
``E:\\CELL_SEG\\RDS_15_06\\<Name>_all``:

* ``overlays/<name>_overlay.png``  — coloured segmentation overlay (like the app)
* ``labels/<name>_labels.png``     — 16-bit label mask, unique id per cell
* ``<Name>_all_animation.mp4``     — a **single** animation stepping through every
                                     image in that job (original → segmented
                                     crossfade), one after the other
* ``cell_counts.csv``              — per-image cell counts

The default jobs (filter → output folder):

* **Tahir** — ``From Tahir/2_8-bit tiff``      · ``_Ch2`` → ``Tahir_all``
* **Billy** — ``From Billy/2. 8-bit tiffs``    · ``_Ch2`` → ``Billy_all``
* **Prem**  — ``From Prem/2_8-bit tiff``       · ``_Ch3`` → ``Prem_all``
* **Jay**   — ``From Jay/TIFF images``         · ``_DIC`` → ``Jay_all``

Inference runs at 1024 px (the size the model was trained at): each image is
resized so its long side is 1024, segmented, and the masks are mapped back to the
image's native resolution. Images already 1024 px are left as-is.

Run it (from WSL, with the E: drive mounted)::

    /home/rhar4542/anaconda3/envs/yolovenv/bin/python \\
        Pipe_7270_New_Lab_Cell_Segmentation/scripts/batch_segment_tahir.py

Process only some jobs with ``--only Prem Jay``. Inference settings and the model
can be overridden on the command line — see ``--help``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Microscopy images can be large; don't trip Pillow's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None



# ---------------------------------------------------------------------------
# Defaults (match the app)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent          # .../Pipe_7270_New_Lab_Cell_Segmentation
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"

# Where the contributor folders live, and where the per-job outputs go.
RAW_ROOT = Path("/mnt/e/CELL_SEG/RDS_15_06/Cell Segmentation raw images_New group")
OUTPUT_ROOT = Path("/mnt/e/CELL_SEG/RDS_15_06")


@dataclass
class Job:
    """One folder to segment: a name, an input dir and a filename filter."""
    name: str          # used for the output folder "<name>_all"
    input_dir: Path
    filter: str        # only files whose name contains this are processed

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / f"{self.name}_all"


# Default jobs — each contributor folder with its own channel filter.
JOBS: list[Job] = [
    Job("Tahir", RAW_ROOT / "From Tahir/2_8-bit tiff",   "_Ch2"),
    Job("Billy", RAW_ROOT / "From Billy/2. 8-bit tiffs", "_Ch2"),
    Job("Prem",  RAW_ROOT / "From Prem/2_8-bit tiff",    "_Ch3"),
    Job("Jay",   RAW_ROOT / "From Jay/TIFF images",      "_DIC"),
]

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Inference / rendering defaults — identical to the app's sidebar defaults.
INFER_SIZE = 1024     # resize the long side to this for inference (model's train size)
CONF = 0.25
IOU = 0.50
MAX_DET = 1000
RETINA_MASKS = True
MATCH_TRAINING = True
CLEAN_ENABLED = True
SMOOTH_RADIUS = 4
KEEP_LARGEST = True
FILL_HOLES = False
OVERLAY_ALPHA = 0.45

# Animation defaults (one combined video for all images)
VIDEO_FPS = 25
VIDEO_HOLD_S = 0.6     # seconds holding on the original, then on the overlay
VIDEO_FADE_S = 0.6     # seconds crossfading original -> overlay
VIDEO_SIDE = 1024      # square canvas side for the video (PNGs stay full-res)

# Distinct overlay colours (RGB), cycled per instance — same palette as the app.
PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Model discovery (mirrors app.find_models — prefer best.pt over last.pt)
# ---------------------------------------------------------------------------
def resolve_model(models_dir: Path, explicit: str | None) -> Path:
    """Return the weights file to use, preferring ``best.pt`` like the app does."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            sys.exit(f"ERROR: --model not found: {p}")
        return p

    pts = sorted(models_dir.rglob("*.pt"))
    if not pts:
        sys.exit(f"ERROR: no .pt weights found under {models_dir}")

    def rank(p: Path) -> tuple[int, str]:
        name = p.stem.lower()
        return (0 if "best" in name else (1 if "last" in name else 2), str(p).lower())

    return sorted(pts, key=rank)[0]


# ---------------------------------------------------------------------------
# Image preprocessing (copied verbatim from the app)
# ---------------------------------------------------------------------------
def to_model_rgb(image: Image.Image, match_training: bool) -> np.ndarray:
    """Convert a PIL image to an HxWx3 uint8 array for the model.

    With ``match_training`` True: collapse to luma, percentile-stretch any
    non-8-bit data to 8-bit, replicate to 3 identical channels (what the model
    was trained on). Otherwise a plain RGB conversion.
    """
    if not match_training:
        return np.array(image.convert("RGB"))

    arr = np.array(image)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 3:
        rgb = arr[:, :, :3].astype(np.float32)
        arr = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return np.stack([arr, arr, arr], axis=-1)


def to_display_rgb(image: Image.Image) -> np.ndarray:
    """A nice 8-bit RGB for display. 8-bit images pass through; anything else is
    percentile-stretched via ``to_model_rgb`` so 16-bit data still looks right.
    """
    arr = np.array(image)
    if arr.dtype == np.uint8:
        return np.array(image.convert("RGB"))
    return to_model_rgb(image, match_training=True)


# ---------------------------------------------------------------------------
# Mask cleanup (copied verbatim from the app)
# ---------------------------------------------------------------------------
def clean_instance_mask(
    mask: np.ndarray, smooth_radius: int, keep_largest: bool, fill_holes: bool
) -> np.ndarray:
    """Tidy a single boolean instance mask (smooth, keep-largest, fill-holes)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask

    if smooth_radius >= 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * smooth_radius + 1, 2 * smooth_radius + 1)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if keep_largest:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lbl == biggest).astype(np.uint8)

    if fill_holes:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        m = filled

    return m.astype(bool)


# ---------------------------------------------------------------------------
# Segmentation (mirrors app.segment_image, native-size inference)
# ---------------------------------------------------------------------------
def segment(model: YOLO, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run YOLO at 1024 px and return
    ``(display_rgb, overlay_rgb, label_mask_uint16, cell_count)``.

    The image is resized so its long side is ``INFER_SIZE`` (1024 — what the model
    was trained at), segmented, and the masks are mapped back to the image's
    native resolution. Images already 1024 px on the long side are left as-is.
    """
    full_input = to_model_rgb(image, MATCH_TRAINING)
    height, width = full_input.shape[:2]            # native size — used for ALL outputs

    # Resize to INFER_SIZE on the long side for inference, then masks are scaled
    # back to (height, width) below. No-op if already at INFER_SIZE.
    long_side = max(height, width)
    if long_side != INFER_SIZE:
        scale = INFER_SIZE / long_side
        inf_w = max(1, round(width * scale))
        inf_h = max(1, round(height * scale))
        infer_input = cv2.resize(full_input, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    else:
        inf_h, inf_w = height, width
        infer_input = full_input

    result = model.predict(
        source=infer_input,
        imgsz=(inf_h, inf_w),       # the resized (≈1024) size
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        retina_masks=RETINA_MASKS,
        verbose=False,
    )[0]

    # Cleaned instance masks. Masks come out at the inference size and are scaled
    # back up to the native (height, width) here, so all outputs are native-res.
    clean_masks: dict[int, np.ndarray] = {}
    if result.masks is not None and len(result.masks) > 0:
        raw = result.masks.data.cpu().numpy()
        for idx in range(len(raw)):
            m = raw[idx] > 0.5
            if m.shape != (height, width):
                m = np.array(
                    Image.fromarray((m.astype(np.uint8) * 255)).resize(
                        (width, height), Image.NEAREST
                    )
                ) > 127
            if CLEAN_ENABLED:
                m = clean_instance_mask(m, SMOOTH_RADIUS, KEEP_LARGEST, FILL_HOLES)
            clean_masks[idx] = m

    if result.boxes is not None and result.boxes.conf is not None:
        conf_arr = result.boxes.conf.cpu().numpy()
    else:
        conf_arr = np.zeros(len(clean_masks))
    order = [i for i in np.argsort(conf_arr)
             if clean_masks.get(i) is not None and clean_masks[i].any()]

    # Display / overlay base.
    display_rgb = to_display_rgb(image)
    if display_rgb.shape[:2] != (height, width):
        display_rgb = np.array(
            Image.fromarray(display_rgb).resize((width, height), Image.BILINEAR)
        )

    overlay = display_rgb.copy()
    lw = max(1, round((height + width) / 1500))
    for new_id, idx in enumerate(order, start=1):
        m = clean_masks[idx]
        colour = PALETTE[(new_id - 1) % len(PALETTE)]
        overlay[m] = (OVERLAY_ALPHA * colour + (1 - OVERLAY_ALPHA) * overlay[m]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, [int(c) for c in colour], lw)

    # 16-bit label mask: unique id per cell, background 0, highest-conf wins overlaps.
    label_mask = np.zeros((height, width), dtype=np.uint16)
    for new_id, idx in enumerate(order, start=1):
        label_mask[clean_masks[idx]] = new_id

    return display_rgb, overlay, label_mask, len(order)


# ---------------------------------------------------------------------------
# Animation (one combined video stepping through all images)
# ---------------------------------------------------------------------------
def fit_canvas(rgb: np.ndarray, side: int) -> np.ndarray:
    """Letterbox an RGB image onto a fixed ``side``×``side`` black canvas.

    Keeps aspect ratio and pads, so every frame in the combined video is the same
    size regardless of the source image's dimensions.
    """
    h, w = rgb.shape[:2]
    scale = side / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw ``text`` with a dark background strip in the top-left for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, frame.shape[1] / 1400)
    thick = max(1, int(round(scale * 1.5)))
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = int(round(8 * scale))
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + bl + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def clip_frames(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    name: str,
    count: int,
    side: int = VIDEO_SIDE,
    fps: int = VIDEO_FPS,
    hold_s: float = VIDEO_HOLD_S,
    fade_s: float = VIDEO_FADE_S,
) -> list[np.ndarray]:
    """Build the frames for one image: hold original → crossfade → hold overlay.

    Returns a list of RGB frames, each ``side``×``side`` with a caption.
    """
    orig = fit_canvas(original_rgb, side)
    over = fit_canvas(overlay_rgb, side)
    n_hold = max(1, round(hold_s * fps))
    n_fade = max(1, round(fade_s * fps))

    cap = f"{name}  ({count} cells)"
    frames: list[np.ndarray] = []
    for _ in range(n_hold):
        frames.append(draw_label(orig.copy(), cap + "  [original]"))
    for i in range(1, n_fade + 1):
        a = i / (n_fade + 1)
        frames.append(draw_label(cv2.addWeighted(orig, 1 - a, over, a, 0), cap))
    for _ in range(n_hold):
        frames.append(draw_label(over.copy(), cap + "  [segmented]"))
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-segment multiple contributor folders with the cell-seg YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    job_names = [j.name for j in JOBS]
    p.add_argument(
        "--only", nargs="+", metavar="NAME", choices=job_names, default=None,
        help=f"Run only these jobs (choices: {', '.join(job_names)}). Default: all.",
    )
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Folder of YOLO runs.")
    p.add_argument("--model", default=None, help="Explicit .pt weights path (overrides --models-dir).")
    p.add_argument("--no-video", action="store_true", help="Skip the mp4 animations.")
    p.add_argument("--overwrite", action="store_true", help="Re-process even if outputs exist.")
    return p.parse_args()


def process_job(job: Job, model: YOLO, make_video: bool, overwrite: bool) -> dict:
    """Segment every matching image in one job. Returns a summary dict."""
    print(f"\n{'=' * 70}\nJOB: {job.name}   filter='{job.filter}'\n{'=' * 70}")

    if not job.input_dir.exists():
        print(f"⚠  input folder not found, skipping: {job.input_dir}")
        return {"job": job.name, "matched": 0, "processed": 0, "reused": 0,
                "errors": 0, "cells": 0, "status": "INPUT MISSING"}

    overlays_dir = job.output_dir / "overlays"
    labels_dir = job.output_dir / "labels"
    for d in (overlays_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in job.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and job.filter in p.name
    )
    print(f"Input   : {job.input_dir}")
    print(f"Output  : {job.output_dir}")
    print(f"Matched : {len(images)} files containing '{job.filter}'\n")
    if not images:
        print("  (nothing to do)")
        return {"job": job.name, "matched": 0, "processed": 0, "reused": 0,
                "errors": 0, "cells": 0, "status": "NO MATCHES"}

    # One combined animation for all images in this job.
    anim_mp4 = job.output_dir / f"{job.name}_all_animation.mp4"
    video_writer = None
    if make_video:
        side = VIDEO_SIDE + (VIDEO_SIDE % 2)   # ensure even (codec-safe)
        video_writer = cv2.VideoWriter(
            str(anim_mp4), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (side, side)
        )
        if not video_writer.isOpened():
            print("⚠  Could not open the video writer — animation will be skipped.\n")
            video_writer = None

    counts_rows: list[dict] = []
    n_ok = n_skip = n_err = 0

    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        overlay_png = overlays_dir / f"{stem}_overlay.png"
        label_png = labels_dir / f"{stem}_labels.png"
        pngs_exist = overlay_png.exists() and label_png.exists()

        try:
            if overwrite or not pngs_exist:
                image = Image.open(img_path)
                display_rgb, overlay_rgb, label_mask, count = segment(model, image)
                Image.fromarray(overlay_rgb).save(overlay_png)
                Image.fromarray(label_mask).save(label_png)   # 16-bit (mode I;16)
                n_ok += 1
                status = f"{count} cells"
            else:
                # Re-use existing outputs (no re-inference); still feed the video.
                label_mask = np.array(Image.open(label_png))
                count = int((np.unique(label_mask) != 0).sum())
                n_skip += 1
                status = f"reuse ({count} cells)"
                if video_writer is not None:
                    overlay_rgb = np.array(Image.open(overlay_png).convert("RGB"))
                    display_rgb = to_display_rgb(Image.open(img_path))

            if video_writer is not None:
                for frame in clip_frames(display_rgb, overlay_rgb, img_path.name, count):
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            counts_rows.append({"image": img_path.name, "cells": count})
            print(f"[{i:>3}/{len(images)}] {img_path.name}: {status}")
        except Exception as exc:  # keep going on a bad file
            print(f"[{i:>3}/{len(images)}] ERROR {img_path.name}: {type(exc).__name__}: {exc}")
            counts_rows.append({"image": img_path.name, "cells": -1})
            n_err += 1

    if video_writer is not None:
        video_writer.release()

    # Per-image counts CSV
    csv_path = job.output_dir / "cell_counts.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "cells"])
        w.writeheader()
        w.writerows(counts_rows)

    total_cells = sum(r["cells"] for r in counts_rows if r["cells"] > 0)
    print(f"\n  {job.name}: processed={n_ok} reused={n_skip} errors={n_err} "
          f"| total cells={total_cells}")
    print(f"  overlays : {overlays_dir}")
    print(f"  labels   : {labels_dir}")
    if video_writer is not None or (make_video and anim_mp4.exists()):
        print(f"  animation: {anim_mp4}")
    print(f"  counts   : {csv_path}")

    return {"job": job.name, "matched": len(images), "processed": n_ok,
            "reused": n_skip, "errors": n_err, "cells": total_cells, "status": "ok"}


def main() -> None:
    args = parse_args()

    jobs = JOBS if not args.only else [j for j in JOBS if j.name in set(args.only)]

    weights = resolve_model(args.models_dir, args.model)
    print(f"Model   : {weights}")
    model = YOLO(str(weights))
    print(f"Loaded  : task={model.task} | classes={model.names}")
    print(f"Jobs    : {', '.join(j.name for j in jobs)}")

    summaries = [process_job(j, model, not args.no_video, args.overwrite) for j in jobs]

    # Grand summary across all jobs.
    print(f"\n{'=' * 70}\nALL JOBS DONE\n{'=' * 70}")
    print(f"{'job':8s} {'matched':>8s} {'done':>6s} {'reuse':>6s} {'err':>5s} {'cells':>7s}  status")
    for s in summaries:
        print(f"{s['job']:8s} {s['matched']:>8d} {s['processed']:>6d} {s['reused']:>6d} "
              f"{s['errors']:>5d} {s['cells']:>7d}  {s['status']}")
    print(f"\nTotal cells across all jobs: {sum(s['cells'] for s in summaries)}")


if __name__ == "__main__":
    main()
