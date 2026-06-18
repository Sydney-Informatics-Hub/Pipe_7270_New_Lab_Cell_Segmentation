"""
Cell Segmentation App (YOLOv8-seg)
==================================

A small Streamlit app for the PIPE-7270 cell-segmentation project.

What it does
------------
1. Points at the project's ``models/`` folder (each subfolder is a trained YOLO
   run containing ``weights/best.pt`` / ``weights/last.pt``).
2. Lets you pick which model to use from a dropdown.
3. Lets you drag-and-drop (or browse for) an image to segment.
4. Runs YOLOv8 instance segmentation and shows the original next to the
   segmented output, plus a live cell count.
5. Lets you save the segmented image (and an optional binary mask) as a PNG.

Run it with::

    cd Pipe_7270_New_Lab_Cell_Segmentation/app
    streamlit run cell_seg_app.py

(See ``run_app.sh`` for a one-liner.)
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Microscopy images can be large; don't trip Pillow's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
# The app lives in ``<project>/app/`` so the models folder is one level up.
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = APP_DIR.parent / "models"

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]

# Distinct overlay colours (RGB), cycled per instance so adjacent cells differ
# and overlaps can't darken (no alpha stacking like ``result.plot()``).
PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
], dtype=np.uint8)



# ---------------------------------------------------------------------------
# Model discovery & loading
# ---------------------------------------------------------------------------
def find_models(models_dir: Path) -> dict[str, str]:
    """Return ``{friendly_label: absolute_path_to_.pt}`` for every weight file.

    YOLO runs store weights at ``<run>/weights/best.pt`` (and ``last.pt``), so a
    weight inside a ``weights`` folder is labelled ``<run> (best)`` for a tidy
    dropdown. Any other loose ``*.pt`` file is labelled by its relative path.
    Results are sorted so ``best`` checkpoints appear before ``last``.
    """
    models: dict[str, str] = {}
    if not models_dir.exists():
        return models

    for pt in sorted(models_dir.rglob("*.pt")):
        if pt.parent.name == "weights":
            run_name = pt.parent.parent.name
            label = f"{run_name} ({pt.stem})"  # e.g. "cell_seg_just_cells_v1 (best)"
        else:
            label = str(pt.relative_to(models_dir))
        models[label] = str(pt)

    # Sort: prefer "best" before "last", then alphabetical.
    def sort_key(item: tuple[str, str]):
        label = item[0].lower()
        rank = 0 if "best" in label else (1 if "last" in label else 2)
        return (rank, label)

    return dict(sorted(models.items(), key=sort_key))


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str) -> YOLO:
    """Load (and cache) a YOLO model so it isn't reloaded on every rerun."""
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def to_model_rgb(image: Image.Image, match_training: bool) -> np.ndarray:
    """Convert a PIL image to an HxWx3 uint8 array for the model.

    When ``match_training`` is True we reproduce the notebook's preprocessing:
    collapse to single-channel luma, percentile-stretch any non-8-bit data to
    8-bit, then replicate to 3 identical channels. The model was trained on
    grayscale cells, so this gives the most faithful results. When False we just
    do a plain RGB conversion.
    """
    if not match_training:
        return np.array(image.convert("RGB"))

    arr = np.array(image)

    # Drop a singleton channel dim, e.g. HxWx1 -> HxW.
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # Colour -> luma (BT.601), ignoring any alpha channel.
    if arr.ndim == 3:
        rgb = arr[:, :, :3].astype(np.float32)
        arr = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

    # Stretch to 8-bit if needed (e.g. 16-bit microscopy TIFFs).
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return np.stack([arr, arr, arr], axis=-1)


# ---------------------------------------------------------------------------
# Mask cleanup
# ---------------------------------------------------------------------------
def clean_instance_mask(
    mask: np.ndarray,
    smooth_radius: int,
    keep_largest: bool,
    fill_holes: bool,
) -> np.ndarray:
    """Tidy a single boolean instance mask.

    YOLO clips each mask to its bounding box, so at low inference resolution
    masks fill their box and leave blocky rectangular edges (and stray fragments
    where they spill into a neighbour's box). This:

    * **Smooths** with a morphological open-then-close (rounds box-crop corners
      and removes thin rectangular spurs) using an elliptical kernel of the
      given radius in pixels.
    * **Keeps the largest connected piece**, dropping detached rectangular bits
      that otherwise "flow" into adjacent cells.
    * Optionally **fills interior holes**.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask

    if smooth_radius >= 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * smooth_radius + 1, 2 * smooth_radius + 1)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)   # remove spurs/slivers
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)  # round dents, de-block

    if keep_largest:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:  # background + >1 foreground component
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (lbl == biggest).astype(np.uint8)

    if fill_holes:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        m = filled

    return m.astype(bool)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running segmentation…")
def segment_image(
    model_path: str,
    file_bytes: bytes,
    res_mode: str,
    infer_size: int,
    conf: float,
    iou: float,
    max_det: int,
    retina_masks: bool,
    show_boxes: bool,
    show_labels: bool,
    line_width: int,
    match_training: bool,
    clean_enabled: bool,
    smooth_radius: int,
    keep_largest: bool,
    fill_holes: bool,
    overlay_alpha: float,
) -> tuple[bytes, bytes, int]:
    """Run YOLO segmentation and return ``(segmented_png, label_mask_png, cell_count)``.

    ``res_mode`` controls the inference resolution:

    * ``"Native size"`` — run at the image's full resolution, ``imgsz=(H, W)``.
    * ``"Downscale, then restore to native"`` — run at ``infer_size`` (capped so
      it never exceeds the native long side); YOLO maps the results back onto
      the native-resolution image automatically.
    * ``"Custom size"`` — run at exactly ``infer_size`` (letterboxed), results
      mapped back to native.

    Both the overlay preview and the exported label mask are built from the
    **same** cleaned per-instance masks, so they always agree. Each cell is drawn
    in its own colour with a single (non-stacking) blend, so overlaps no longer
    darken, and ``clean_instance_mask`` removes the blocky box-crop artifacts
    that otherwise bleed into the exported mask.

    In every mode the outputs are at the loaded image's native size. Cached on
    its inputs so clicking a download button (which reruns the script) doesn't
    recompute the prediction.
    """
    model = load_model(model_path)

    image = Image.open(io.BytesIO(file_bytes))
    model_input = to_model_rgb(image, match_training)
    height, width = model_input.shape[:2]

    # Pick the inference size from the chosen mode. YOLO always rescales
    # detections/masks back onto the native-resolution image, so the outputs are
    # native-size regardless of the value used here.
    if res_mode == "Downscale, then restore to native":
        imgsz: tuple[int, int] | int = min(int(infer_size), max(height, width))
    elif res_mode == "Custom size":
        imgsz = int(infer_size)
    else:  # "Native size"
        imgsz = (height, width)

    result = model.predict(
        source=model_input,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        retina_masks=retina_masks,
        verbose=False,
    )[0]

    # --- Build cleaned, native-resolution instance masks (used for BOTH the
    # overlay and the exported label mask so they always agree) ---------------
    radius = smooth_radius if clean_enabled else 0
    clean_masks: dict[int, np.ndarray] = {}
    if result.masks is not None and len(result.masks) > 0:
        raw = result.masks.data.cpu().numpy()  # (N, h, w) in {0, 1}
        for idx in range(len(raw)):
            m = raw[idx] > 0.5
            if m.shape != (height, width):  # e.g. non-retina prototype masks
                m = np.array(
                    Image.fromarray((m.astype(np.uint8) * 255)).resize(
                        (width, height), Image.NEAREST
                    )
                ) > 127
            if clean_enabled:
                m = clean_instance_mask(m, radius, keep_largest, fill_holes)
            clean_masks[idx] = m

    # Paint lowest- to highest-confidence so the most confident cell wins any
    # overlap, and skip instances cleaned away to nothing.
    if result.boxes is not None and result.boxes.conf is not None:
        conf_arr = result.boxes.conf.cpu().numpy()
    else:
        conf_arr = np.zeros(len(clean_masks))
    order = [i for i in np.argsort(conf_arr) if clean_masks.get(i) is not None
             and clean_masks[i].any()]

    # --- Overlay: grayscale base + one distinct colour per cell --------------
    base_rgb = np.array(image.convert("RGB"))
    if base_rgb.shape[:2] != (height, width):
        base_rgb = np.array(
            Image.fromarray(base_rgb).resize((width, height), Image.BILINEAR)
        )
    overlay = base_rgb.copy()
    lw = line_width if line_width > 0 else max(1, round((height + width) / 1500))
    for new_id, idx in enumerate(order, start=1):
        m = clean_masks[idx]
        colour = PALETTE[(new_id - 1) % len(PALETTE)]
        overlay[m] = (overlay_alpha * colour + (1 - overlay_alpha) * overlay[m]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, [int(c) for c in colour], lw)
        if show_boxes:
            x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), [int(c) for c in colour], lw)
        if show_labels:
            ys, xs = np.where(m)
            cv2.putText(
                overlay, str(new_id), (int(xs.mean()), int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, lw * 0.4),
                (255, 255, 255), max(1, lw), cv2.LINE_AA,
            )

    seg_buf = io.BytesIO()
    Image.fromarray(overlay).save(seg_buf, format="PNG")

    # --- Labelled mask: unique grayscale id per cell (background = 0) ---------
    # 16-bit PNG so up to 65535 cells stay uniquely identifiable.
    label_mask = np.zeros((height, width), dtype=np.uint16)
    for new_id, idx in enumerate(order, start=1):
        label_mask[clean_masks[idx]] = new_id

    mask_buf = io.BytesIO()
    Image.fromarray(label_mask).save(mask_buf, format="PNG")  # 16-bit (mode I;16)

    return seg_buf.getvalue(), mask_buf.getvalue(), len(order)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cell Segmentation", page_icon="🔬", layout="wide")
st.title("🔬 Cell Segmentation (YOLOv8-seg)")
st.caption(
    "Pick a trained model, drop in an image, and segment the cells. "
    "Then save the result as a PNG."
)

# --- Sidebar: model + inference settings ----------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    models_dir_str = st.text_input(
        "Models folder",
        value=str(DEFAULT_MODELS_DIR),
        help="Folder containing your trained YOLO run folders.",
    )
    models_dir = Path(models_dir_str).expanduser()

    if st.button("🔄 Rescan models folder"):
        st.cache_resource.clear()

    models = find_models(models_dir)

    if not models:
        st.error(
            f"No `.pt` weights found under:\n\n`{models_dir}`\n\n"
            "Check the path above, or train a model first."
        )
        selected_label = None
        selected_model_path = None
    else:
        selected_label = st.selectbox("Model", list(models.keys()))
        selected_model_path = models[selected_label]
        st.caption(f"`{selected_model_path}`")

    st.divider()
    st.subheader("Inference")
    res_mode = st.radio(
        "Resolution",
        (
            "Native size",
            "Downscale, then restore to native",
            "Custom size",
        ),
        index=0,
        help=(
            "**Native size** — infer at the image's full resolution "
            "(most detail, slowest).\n\n"
            "**Downscale, then restore to native** — infer at a lower "
            "resolution for speed, then map the results back onto the "
            "native-size image.\n\n"
            "**Custom size** — infer at a fixed size you choose."
        ),
    )

    infer_size = 640
    if res_mode == "Downscale, then restore to native":
        infer_size = st.select_slider(
            "Inference size (px, long side)",
            options=[256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280],
            value=640,
            help="Image is downscaled to this size for inference (never "
                 "upscaled past native), then results are restored to native.",
        )
    elif res_mode == "Custom size":
        infer_size = st.number_input(
            "Inference size (px, multiple of 32)",
            min_value=64, max_value=4096, value=1024, step=32,
            help="Image is resized to this size for inference (letterboxed); "
                 "results are mapped back to the native resolution.",
        )

    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU (NMS) threshold", 0.0, 1.0, 0.50, 0.05)
    max_det = st.number_input("Max detections", 10, 5000, 1000, 10)
    retina_masks = st.checkbox(
        "High-res masks (retina)", value=True,
        help="Render masks at full image resolution.",
    )

    st.subheader("Mask cleanup")
    clean_enabled = st.checkbox(
        "Clean up masks", value=True,
        help="Remove the blocky box-crop artifacts that bleed into the exported "
             "mask. Applies to both the preview and the saved label PNG.",
    )
    smooth_radius = st.slider(
        "Smoothing radius (px)", 0, 25, 4,
        help="Morphological open+close radius. Higher = smoother, rounder cell "
             "edges; too high erodes fine detail. 0 = off.",
        disabled=not clean_enabled,
    )
    keep_largest = st.checkbox(
        "Keep largest piece per cell", value=True,
        help="Drop detached fragments so a cell can't spill a rectangular bit "
             "into its neighbour.",
        disabled=not clean_enabled,
    )
    fill_holes = st.checkbox(
        "Fill interior holes", value=False,
        help="Fill gaps inside a cell. Leave off to keep genuine holes.",
        disabled=not clean_enabled,
    )

    st.subheader("Overlay")
    match_training = st.checkbox(
        "Match training preprocessing (grayscale 8-bit)", value=True,
        help="Recommended — the model was trained on grayscale cells.",
    )
    overlay_alpha = st.slider(
        "Mask opacity", 0.0, 1.0, 0.45, 0.05,
        help="How strongly each cell's colour is blended over the image.",
    )
    show_boxes = st.checkbox("Show boxes", value=False)
    show_labels = st.checkbox("Show labels (cell id)", value=False)
    line_width = st.slider(
        "Line width (0 = auto)", 0, 10, 0,
        help="Thickness of mask/box outlines.",
    )

# --- Main: upload + results -----------------------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop an image here (or browse)",
    type=SUPPORTED_IMAGE_TYPES,
    help="JPG, PNG, BMP or TIFF.",
)

if uploaded_file is None:
    st.info("⬆️ Upload an image to get started.")
    st.stop()

if not selected_model_path:
    st.warning("Select a model in the sidebar to run segmentation.")
    st.stop()

file_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(file_bytes))

try:
    seg_png, mask_png, count = segment_image(
        model_path=selected_model_path,
        file_bytes=file_bytes,
        res_mode=str(res_mode),
        infer_size=int(infer_size),
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        retina_masks=bool(retina_masks),
        show_boxes=bool(show_boxes),
        show_labels=bool(show_labels),
        line_width=int(line_width),
        match_training=bool(match_training),
        clean_enabled=bool(clean_enabled),
        smooth_radius=int(smooth_radius),
        keep_largest=bool(keep_largest),
        fill_holes=bool(fill_holes),
        overlay_alpha=float(overlay_alpha),
    )
except Exception as exc:  # noqa: BLE001 - surface any inference error to the UI
    st.error(f"Segmentation failed: {type(exc).__name__}: {exc}")
    st.stop()

st.metric("Cells detected", count)

# Show the resolution actually used so the chosen mode is transparent.
nat_w, nat_h = original_image.size
if res_mode == "Downscale, then restore to native":
    _eff = min(int(infer_size), max(nat_w, nat_h))
    st.caption(f"Inference: downscaled to ≤{_eff}px long side → restored to {nat_w}×{nat_h}.")
elif res_mode == "Custom size":
    st.caption(f"Inference: {int(infer_size)}px (letterboxed) → restored to {nat_w}×{nat_h}.")
else:
    st.caption(f"Inference: native {nat_w}×{nat_h}.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(original_image, width="stretch")
with col2:
    st.subheader("Segmented")
    st.image(seg_png, width="stretch")

# --- Save as PNG -----------------------------------------------------------
st.divider()
st.subheader("💾 Save")

stem = Path(uploaded_file.name).stem
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Save segmented image (PNG)",
        data=seg_png,
        file_name=f"{stem}_segmented.png",
        mime="image/png",
        width="stretch",
    )
with dl2:
    st.download_button(
        "⬇️ Save labelled mask (PNG)",
        data=mask_png,
        file_name=f"{stem}_labels.png",
        mime="image/png",
        width="stretch",
        help="16-bit PNG: background = 0, each cell a unique grayscale id.",
    )
