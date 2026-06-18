# Cell Segmentation App

A small [Streamlit](https://streamlit.io) app for running the trained YOLOv8
cell-segmentation models on a single image and saving the result as a PNG.

## What it does

1. **Points at the `models/` folder** — each subfolder is a trained YOLO run
   containing `weights/best.pt` (and `weights/last.pt`).
2. **Choose a model** from a dropdown in the sidebar.
3. **Drag and drop** (or browse for) an image to segment.
4. Shows the **original next to the segmented output**, with a live **cell count**.
5. **Save** the segmented image — and an optional **labelled mask** — as a **PNG**.

## Folder layout it expects

```
Pipe_7270_New_Lab_Cell_Segmentation/
├── app/
│   ├── cell_seg_app.py     <- this app
│   ├── requirements.txt
│   └── run_app.sh
└── models/
    └── cell_seg_just_cells_v1/
        └── weights/
            ├── best.pt
            └── last.pt
```

The app auto-discovers every `*.pt` under `models/`, so dropping in new run
folders just works. You can also point it at a different folder using the
**Models folder** box in the sidebar.

## Running it

From WSL / a terminal, with your YOLO environment activated:

```bash
cd Pipe_7270_New_Lab_Cell_Segmentation/app
streamlit run cell_seg_app.py
```

or use the helper script:

```bash
./run_app.sh
```

Streamlit prints a local URL (default <http://localhost:8501>) — open it in a
browser.

> **First run:** if Streamlit isn't installed yet, install the app deps into the
> same environment you trained with:
> ```bash
> pip install -r requirements.txt
> ```

## Sidebar settings

| Setting | Default | Notes |
|---|---|---|
| Models folder | `../models` | Where to look for `*.pt` weights. |
| Model | first `best.pt` | Which checkpoint to run. |
| Resolution | Native size | How to size the image for inference (see below). |
| Confidence | `0.25` | Minimum detection confidence. |
| IoU (NMS) | `0.50` | Overlap threshold for merging detections. |
| Max detections | `1000` | Upper bound on cells per image. |
| High-res masks | on | Render masks at full resolution. |
| Clean up masks | on | Remove blocky box-crop artifacts (see below). |
| Smoothing radius | `4` px | Morphological open+close radius for cleanup. |
| Keep largest piece | on | Drop detached fragments per cell. |
| Fill interior holes | off | Fill gaps inside a cell. |
| Match training preprocessing | on | Grayscale 8-bit, like the training data. |
| Mask opacity | `0.45` | Blend strength of each cell's colour. |
| Show boxes / labels | off | Toggle boxes / per-cell id numbers. |
| Line width | auto | Outline thickness. |

### Resolution modes

The overlay and label mask are **always returned at the uploaded image's native
size** — the mode only changes the resolution YOLO runs inference at (it maps
the results back to native automatically):

- **Native size** — infer at the image's full resolution (rounded up to the
  model's stride of 32). Most detail, slowest, most memory.
- **Downscale, then restore to native** — infer at a lower resolution (pick the
  long-side size, capped so it never upscales past native) for speed, then map
  the results back onto the native-size image.
- **Custom size** — infer at a fixed size you choose (multiple of 32).

### Mask cleanup (fixing blocky / overlapping masks)

YOLO clips every instance mask to its bounding box, so at lower inference
resolution masks fill their box and leave **rectangular ("blocky") edges**, and
where two cells overlap one can spill a rectangular bit into the other — which
shows up as dark patches in the preview and bleeds into the exported label mask.

To counter this, the app builds **both** the preview and the exported label mask
from the *same* cleaned masks, drawing each cell in its own colour with a single
(non-stacking) blend, and applies:

- **Smoothing radius** — a morphological open-then-close that rounds box-crop
  corners and removes thin rectangular spurs. Higher = smoother; too high erodes
  fine detail.
- **Keep largest piece per cell** — drops detached fragments so a cell can't
  leak a rectangular block into its neighbour.
- **Fill interior holes** — optional; leave off to preserve genuine holes.

> The best fix for blockiness is still **resolution**: prefer **Native size** +
> **High-res masks** so masks are computed finely in the first place. Cleanup
> then just tidies the edges.

## Notes

- **Match training preprocessing** is on by default. The models were trained on
  grayscale cells, so the app converts uploads to 8-bit grayscale (percentile-
  stretching 16-bit TIFFs) before inference. Turn it off to feed raw RGB.
- The **labelled mask** is a 16-bit grayscale PNG: background is `0` and every
  detected cell gets a unique id (`1, 2, 3, …`). It's an instance/label image
  (not a colour overlay), so most viewers show it as near-black — load it in
  Python (e.g. `numpy`/`skimage`) for counting or `regionprops`, or colourise it
  with `skimage.color.label2rgb` to view. Overlaps are resolved in favour of the
  higher-confidence cell.
- Predictions are cached, so adjusting the download filename or re-clicking a
  button won't re-run the model.
