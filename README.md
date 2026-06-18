# Pipe_7270_New_Lab_Cell_Segmentation

YOLO-based **instance segmentation of cells** in microscopy images for the
PIPE-7270 project. The repo covers the full workflow: preparing the annotated
data, training a YOLO segmentation model, batch-segmenting whole folders of
images, and an interactive Streamlit app for segmenting one image at a time.

The models are trained on **8-bit grayscale** cell images (16-bit fluorescence
TIFFs and RGB images are normalised to grayscale and replicated to 3 channels so
the pretrained YOLO weights load cleanly). The current production model is a
**single-class (`Cell`) YOLO11n-seg** model.

## Project layout

```
Pipe_7270_New_Lab_Cell_Segmentation/
├── requirements_yolovenv.txt           # Python deps for the yolovenv environment
├── app/                                # Streamlit single-image segmentation app
│   ├── cell_seg_app.py
│   ├── requirements.txt
│   ├── run_app.sh
│   └── README.md
├── notebooks/                          # Training + evaluation notebooks
│   ├── train_cell_seg_yolo.ipynb            # 2-class (Cell, dead-cell)
│   ├── train_cell_seg_yolo_just_cells.ipynb # single-class (Cell only)
│   ├── data_just_cells.yaml                 # YOLO dataset config (single class)
│   ├── runs/                                # training outputs (weights, plots)
│   └── *.pt                                 # base/pretrained YOLO checkpoints
├── scripts/
│   └── batch_segment_tahir.py          # batch segmentation of many folders
├── models/
│   └── cell_seg_just_cells_v1/weights/{best.pt,last.pt}
├── training_data/                      # raw annotated data (COCO-style)
├── training_data_processed/            # cleaned data
├── training_data_processed_gray8_just_cells/  # YOLO-formatted, 8-bit, single class
└── test_imgs/                          # sample images for trying the app
```

## Environment setup

The project runs in a conda environment called **`yolovenv`** (Python 3.9, an
NVIDIA GPU and CUDA 12.x are assumed; CPU-only works but training will be slow).

### 1. Create the conda environment

```bash
conda create -n yolovenv python=3.9 -y
conda activate yolovenv
```

### 2. Install PyTorch with CUDA

Install the CUDA build of PyTorch first, from the official PyTorch index, so you
get GPU support (the plain `pip install torch` may give a CPU-only build):

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

> For a different CUDA version or CPU-only, pick the matching command from
> <https://pytorch.org/get-started/locally/>.

### 3. Install the remaining dependencies

```bash
pip install -r requirements_yolovenv.txt
```

### 4. Verify the GPU is visible

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

This should print `CUDA: True`. Versions verified to work together: Python 3.9,
`torch 2.8.0+cu129`, `ultralytics 8.4.68`, `numpy 1.26`, `opencv-python 4.7`,
`pillow 11`, `streamlit 1.50`.

### Note on data locations

The notebooks and the batch script were written for **WSL**, where the Windows
`E:` drive is auto-mounted at `/mnt/e`. The single-class notebook now reads its
data from inside the repo (`training_data_processed/`), but the 2-class notebook
and `batch_segment_tahir.py` still point at `/mnt/e/CELL_SEG/...`. Adjust those
paths for your machine.

## Notebooks

Both notebooks live in `notebooks/` and are meant to be run from WSL with the
`yolovenv` kernel selected.

### `train_cell_seg_yolo.ipynb` — 2-class model (`Cell`, `dead-cell`)

The original training notebook. It:

1. Verifies the YOLO-formatted dataset on the mounted `E:` drive
   (`train/`, `val/`, `test/` with `images/` + `labels/`).
2. **Normalises every image to 8-bit grayscale** (RGB → luma, 16-bit →
   percentile-stretched 8-bit) and replicates it to 3 identical channels so the
   pretrained weights load — this fixes the channel-mismatch error YOLO's mosaic
   augmentation throws on mixed-channel data. Output goes to a sibling
   `*_gray8/` folder.
3. Writes a `data.yaml`, trains a YOLO segmentation model, and evaluates it on
   the held-out test split.

In practice the `dead-cell` class was too rare to learn (~0 % mAP), which is
why the single-class notebook below is now preferred.

### `train_cell_seg_yolo_just_cells.ipynb` — single-class model (`Cell` only)

The current production training notebook. Same idea as above but it collapses
**every** annotation (`Cell`, `Dead cell`, `dead-cell`) into a single `Cell`
class, so the model only learns *where* cells are. It:

1. Reads the COCO-style data from `training_data_processed/` (each split has the
   images plus one `annotations.json` exported from AnyLabeling).
2. **Converts COCO polygons → YOLO segmentation labels**, normalises images to
   8-bit grayscale, and rewrites every class id to `0`, writing the result to
   `training_data_processed_gray8_just_cells/{train,val,test}/{images,labels}`.
3. Writes `data_just_cells.yaml` and trains **YOLO11n-seg** at `imgsz=1024`
   (AdamW + cosine LR, augmentation tuned for microscopy) with early stopping.
4. Evaluates the best checkpoint and runs predictions on the test images.

The trained weights are saved under `notebooks/runs/<run_name>/weights/` and the
production copy is in `models/cell_seg_just_cells_v1/weights/best.pt`.

### `data_just_cells.yaml`

The YOLO dataset config used by the single-class notebook — points at
`training_data_processed_gray8_just_cells/` and declares the single class
`0: Cell`.

## Scripts

### `scripts/batch_segment_tahir.py`

A **local, non-app** batch runner that loads the *same* trained model the app
uses and reproduces the app's exact processing (grayscale preprocessing → 1024 px
inference → mask cleanup → coloured overlay + 16-bit label mask), then runs it
over **several** contributor folders, each with its own filename filter and
output folder.

For each configured job it writes, per image:

- `overlays/<name>_overlay.png` — coloured segmentation overlay,
- `labels/<name>_labels.png` — 16-bit label mask (unique id per cell),
- a single `<Name>_all_animation.mp4` stepping through every image, and
- `cell_counts.csv` — per-image cell counts.

Run it (from WSL, with the `E:` drive mounted):

```bash
/home/rhar4542/anaconda3/envs/yolovenv/bin/python \
    scripts/batch_segment_tahir.py
```

Process only some jobs with `--only Prem Jay`; the model and inference settings
(confidence, IoU, image size, etc.) can be overridden — see `--help`. Input/output
roots and the per-contributor jobs are defined near the top of the script.

## App

### `app/cell_seg_app.py`

An interactive [Streamlit](https://streamlit.io) app for segmenting **one image
at a time**. It auto-discovers every `*.pt` under `models/`, lets you pick a
model and drag-and-drop an image, shows the original next to the segmented output
with a live cell count, and lets you save the overlay (and an optional 16-bit
label mask) as PNG. It exposes the inference and mask-cleanup settings in the
sidebar.

Run it:

```bash
conda activate yolovenv
cd app
streamlit run cell_seg_app.py
# or: ./run_app.sh
```

Streamlit prints a local URL (default <http://localhost:8501>). See
[app/README.md](app/README.md) for the full list of sidebar settings,
resolution modes, and the mask-cleanup explanation. The app's own
`app/requirements.txt` lists only the app-layer deps — installing
`requirements_yolovenv.txt` already covers them.