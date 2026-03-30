# AI-Driven DTM Generator

**Automated Digital Terrain Model generation from LiDAR point clouds using deep learning.**  
Removes vegetation, buildings, and all above-ground objects to produce a clean, usable terrain surface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Environment & Hardware](#3-environment--hardware)
4. [Dataset Structure](#4-dataset-structure)
5. [Notebook Walkthrough — Cell by Cell](#5-notebook-walkthrough--cell-by-cell)
   - [Cell 1 — P100 Environment Fix](#cell-1--p100-environment-fix)
   - [Cell 2 — GPU Verification & Memory Helpers](#cell-2--gpu-verification--memory-helpers)
   - [Cell 3 — Dataset Detection](#cell-3--dataset-detection)
   - [Cell 4 — Global Configuration](#cell-4--global-configuration)
   - [Cell 5 — Geospatial Feature Cache](#cell-5--geospatial-feature-cache)
   - [Cell 6 — Model: DTMNet (PointNet++ MSG)](#cell-6--model-dtmnet-pointnet-msg)
   - [Cell 7 — Dataset Class](#cell-7--dataset-class)
   - [Cell 8 — Training Loop](#cell-8--training-loop)
   - [Cell 9 — Threshold Optimisation](#cell-9--threshold-optimisation)
   - [Cell 10 — DTM Post-Processing](#cell-10--dtm-post-processing)
   - [Cell 11 — Visualisation & Export](#cell-11--visualisation--export)
   - [Cell 12 — Output Packaging](#cell-12--output-packaging)
6. [Model Architecture Deep Dive](#6-model-architecture-deep-dive)
7. [Feature Engineering](#7-feature-engineering)
8. [Training Strategy](#8-training-strategy)
9. [DTM Post-Processing Pipeline](#9-dtm-post-processing-pipeline)
10. [Configuration Reference](#10-configuration-reference)
11. [Outputs & Artifacts](#11-outputs--artifacts)
12. [Web Interface Integration](#12-web-interface-integration)
13. [Memory Budget & Performance](#13-memory-budget--performance)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

This notebook implements an end-to-end AI pipeline that takes raw LiDAR point cloud tiles as input and produces clean, research-grade Digital Terrain Models (DTMs) as output.

**Problem Statement:** Raw LiDAR scans capture everything — ground, vegetation, buildings, vehicles, and other objects. A DTM requires only the bare earth surface. Classical filtering methods (Progressive Morphological Filter, Cloth Simulation Filter) struggle on complex terrain such as Indian villages with dense vegetation, steep slopes, and irregular built structures.

**Solution:** A deep learning segmentation model trained on labelled point cloud tiles classifies each point as *ground* or *non-ground* with high accuracy. The predicted ground points are then post-processed into a smooth, hole-free terrain grid.

**Target Performance:**
- Ground classification accuracy ≥ 95%
- Precision / Recall balance via Focal Loss + threshold optimisation
- Clean DTM: no pits, no voids, continuous elevation surface

---

## 2. Pipeline Architecture

```
Raw LiDAR Tiles  (points.npy + labels.npy per tile)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Feature Engineering  (Cell 5)                       │
│  Per-point: [ΔZ, roughness, slope, density]          │
│  Stored as float32 cache — computed once             │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  DTMNet — PointNet++ MSG  (Cell 6)                   │
│  Input : (B, N, 7)  [x,y,z, ΔZ, rough, slope, dens] │
│  SA1-MSG → SA2-MSG → SA3 → FP3 → FP2 → FP1 → Head  │
│  Output: (B, N, 2)  [non-ground, ground] logits      │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Threshold Optimisation  (Cell 9)                    │
│  Sweep [0.20 … 0.85], pick threshold maximising F1   │
└──────────────────────────────────────────────────────┘
        │  predicted ground XYZ
        ▼
┌──────────────────────────────────────────────────────┐
│  DTM Post-Processing  (Cell 10)                      │
│  1. Rasterize  → min-elevation grid (0.5 m cells)   │
│  2. Fill holes → scipy griddata interpolation        │
│  3. Fill pits  → morphological grey-closing          │
│  4. Smooth     → Gaussian filter (σ = 1.0)           │
└──────────────────────────────────────────────────────┘
        │
        ▼
✅  Clean DTM  per tile (.npy)
    + inference_config.json
    + dtm_inference.py
    + dtm_model_outputs.zip
```

---

## 3. Environment & Hardware

### Target Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA Tesla P100-PCIe 16 GB (CUDA sm_60) |
| System RAM | 19 GB (Kaggle free tier) |
| Storage | `/kaggle/working` (20 GB) |
| Platform | Kaggle Notebooks |

### Critical Compatibility Note — P100 & PyTorch

The P100 uses CUDA compute capability **sm_60 (Pascal architecture)**. PyTorch ≥ 2.1 dropped binary kernel support for sm_60. If the wrong PyTorch version is installed, the GPU is detected but all tensor operations silently fail at runtime with:

```
CUDA error: no kernel image is available for execution on the device
```

**Fix:** Cell 1 installs **PyTorch 2.0.1+cu117** — the last release that includes sm_60 kernels — into a Python 3.10 virtual environment and registers it as a Jupyter kernel named `DTM-P100 (py3.10)`.

### Software Versions

| Package | Version | Why |
|---|---|---|
| Python | 3.10 (venv) | Last Python version with torch 2.0.1 cp3xx wheels |
| PyTorch | 2.0.1+cu117 | Last version with sm_60 (P100) support |
| CUDA toolkit | 11.7 | Matches the wheel |
| NumPy | ≥ 1.23 | |
| SciPy | ≥ 1.9 | Hole filling, morphological ops |
| Matplotlib | ≥ 3.6 | Visualisation |
| tqdm | ≥ 4.64 | Progress bars |
| psutil | ≥ 5.9 | RAM monitoring |

### Alternative GPUs (No Cell 1 needed)

| GPU | Compute Cap | Notes |
|---|---|---|
| T4 | sm_75 | Works with PyTorch 2.x out of the box |
| A100 | sm_80 | Fastest option, skip Cell 1 |
| V100 | sm_70 | Skip Cell 1 |
| P100 | sm_60 | Requires Cell 1 |

---

## 4. Dataset Structure

### Expected Layout

```
Training/
├── train/
│   ├── tile_0001/
│   │   ├── points.npy      # (N, 3+) float32 — XYZ (+ optional intensity)
│   │   └── labels.npy      # (N,)    int64   — 0=non-ground, 1=ground
│   ├── tile_0002/
│   │   └── ...
│   └── ...
└── val/
    ├── tile_0001/
    └── ...
```

### Dataset Statistics (10 Indian Villages)

| Split | Tiles | Notes |
|---|---|---|
| Train | 14,418 | Augmented during training |
| Val | 3,955 | Used for validation and threshold search |
| Total | 18,373 | |

### Tile Format

- `points.npy`: NumPy array `(N, ≥3)` float32. Only the first 3 columns (XYZ) are used. N varies per tile, typically 500–50,000 points.
- `labels.npy`: NumPy array `(N,)` int64. Binary: `0 = non-ground`, `1 = ground`.
- Coordinates are in projected metres (local coordinate system per village).

### ZIP Mode

If the dataset is not extracted, the notebook can stream directly from a `.zip` file without full extraction — critical for datasets >10 GB. Set `DATASET_ZIP_PATH` in Cell 3.

---

## 5. Notebook Walkthrough — Cell by Cell

### Cell 1 — P100 Environment Fix

**Purpose:** Installs PyTorch 2.0.1 with sm_60 support. Run once, ~5 minutes.

**What it does, step by step:**

1. Checks if the environment already works (imports torch, runs a real CUDA op). If yes, skips everything.
2. Installs Python 3.10 via the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) using `apt-get`.
3. Creates a virtual environment at `/kaggle/working/dtm_env`.
4. Downloads and installs the PyTorch 2.0.1+cu117 wheel directly from `download.pytorch.org` (bypasses PyPI, which does not have this package).
5. Installs `ipykernel` and registers the environment as a Jupyter kernel named `DTM-P100 (py3.10)`.
6. Runs a real GPU smoke test (`x @ x` matrix multiply) to confirm kernels execute.

**After Cell 1 completes:**
> **Kernel menu → Change Kernel → "DTM-P100 (py3.10)"**
> Then run all cells from Cell 2 onward. Do NOT click Restart — you need to switch kernels, not restart the same one.

**Key implementation detail:** The wheel URL includes the Python version tag (`cp310`) and CUDA version (`cu117`). This specific combination (`torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl`) is required — other combinations either don't exist or don't support sm_60.

---

### Cell 2 — GPU Verification & Memory Helpers

**Purpose:** Confirms the GPU is truly operational (not just detected) and defines shared memory utilities.

**GPU verification strategy:** `torch.cuda.is_available()` returns `True` even when sm_60 kernels are missing. This cell performs an actual matrix multiplication and calls `.item()` to force synchronous kernel execution. If this fails, a clear error message is shown.

**Shared utilities defined here (used by all later cells):**

```python
mem_report(tag="")   # Prints GPU VRAM + system RAM usage at any point
free_memory()        # gc.collect() + cuda.empty_cache() + synchronize()
```

These are called after every validation pass and stage transition to prevent VRAM fragmentation over long training runs.

---

### Cell 3 — Dataset Detection

**Purpose:** Locates the Kaggle dataset without requiring a hardcoded path.

**Detection logic (priority order):**
1. If `DATASET_ROOT` is set and exists with `train/val` subdirs → use it directly.
2. If `DATASET_ZIP_PATH` is set → use ZIP streaming mode.
3. Auto-scan `/kaggle/input/**` for any directory containing `train/` and `val/`. If multiple matches, prefer the one with `split_manifest.json`.
4. Auto-scan for a single `.zip` file.
5. If nothing found → raise a clear error.

**Set your dataset path here:**
```python
DATASET_ROOT = "/kaggle/input/your-dataset-name/Training"
```

---

### Cell 4 — Global Configuration

**Purpose:** Single source of truth for all hyperparameters and paths. The only cell you need to edit for common experiments.

**Stage plan system:** Training is divided into stages with increasing tile counts. This allows the model to converge on small data first before scaling, fitting the full pipeline within a 10-hour Kaggle session.

```python
STAGE_PLAN = [
    {"name":"sanity", "epochs":10, "max_train":200,   ...},  # ~2 min
    {"name":"medium", "epochs":20, "max_train":2000,  ...},  # ~40 min
    {"name":"full",   "epochs":35, "max_train":14418, ...},  # ~8.5 h
]
START_PROFILE = "full"  # Change to "sanity" for a quick smoke-test
```

**Key parameters to tune:**

| Parameter | Default | Effect |
|---|---|---|
| `max_pts` | 2048 | Points sampled per tile. Higher = more accurate but slower. |
| `max_lr` | 0.01 | OneCycleLR peak. Decrease if training is unstable. |
| `focal_alpha` | 0.75 | Class weight for ground. Increase if ground recall is too low. |
| `focal_gamma` | 2.0 | Focusing exponent. Higher = more focus on hard examples. |
| `dtm_resolution` | 0.5 | Output DTM cell size in metres. |
| `dtm_smooth_sigma` | 1.0 | Gaussian smoothing strength. Increase for noisier terrain. |

---

### Cell 5 — Geospatial Feature Cache

**Purpose:** Pre-computes 4 terrain-discriminative features for every point in every tile and saves them as float32 `.npy` files. This is a one-time cost (30–60 min for 18K tiles) that makes subsequent epochs fast (< 1 ms per tile).

**The 4 features:**

| Feature | Description | DTM Relevance |
|---|---|---|
| **ΔZ** (delta_z) | Height of each point above the local DTM estimate. Ground points have ΔZ ≈ 0. Non-ground objects are elevated. | Single strongest discriminator |
| **Roughness** | Local Z standard deviation within each 2m grid cell. Ground is smooth (σ ≈ 0). Vegetation and buildings are rough. | Separates ground from vegetation |
| **Slope** | Surface gradient derived from the DTM grid via `numpy.gradient`. Ground is planar. Buildings have vertical faces. | Separates ground from buildings |
| **Density** | Normalised point count per grid cell. Dense clusters often indicate vegetation canopy. | Separates open ground from canopy |

**Implementation details:**

- Grid resolution adapts to tile extent: cells are ~2 m wide, clipped to 16–64 grid size.
- All operations use **float32** (original code used float64, doubling RAM usage).
- Point files are loaded with `mmap_mode="r"` — the file is memory-mapped without copying its contents into RAM.
- GC is called every 1000 tiles to prevent RAM accumulation during the build.
- All 4 features are **z-scored** (zero mean, unit variance) before saving.

**Cache layout:**
```
/kaggle/working/feat_cache/
    train/
        tile_0001.npy    # (N_orig, 4) float32
        tile_0002.npy
        ...
    val/
        tile_0001.npy
        ...
```

---

### Cell 6 — Model: DTMNet (PointNet++ MSG)

**Purpose:** Defines the neural network architecture and loss function.

**Architecture summary:**

```
Input (B, N, 7): [x, y, z, ΔZ, roughness, slope, density]
         │
    SA1-MSG  ──  512 centroids
         │         ├── radius=0.5m, k=32  → MLP[32→64]  ─┐
         │         └── radius=1.5m, k=64  → MLP[64→128] ─┴→ 192-ch
    SA2-MSG  ──  128 centroids
         │         ├── radius=3.0m, k=64  → MLP[128→128] ─┐
         │         └── radius=6.0m, k=128 → MLP[128→256] ─┴→ 384-ch
    SA3      ──   32 centroids  radius=12m, k=128 → MLP[256→512]
         │
    FP3: interpolate SA3→SA2 level  (384+512 → 256-ch)
    FP2: interpolate FP3→SA1 level  (192+256 → 128-ch)
    FP1: interpolate FP2→Input level  (4+128 → 128-ch)
         │
    Head: Conv1D 128→128→64→2 + Dropout(0.5)
         │
Output (B, N, 2): [non-ground logit, ground logit]
```

**Total parameters: ~1.14 million**

**Key design decisions:**

- **Multi-Scale Grouping (MSG):** Each SA layer groups points at two different radii simultaneously. Fine radii (0.5–1.5 m) capture fine point texture; coarse radii (3–12 m) capture global terrain shape. This is critical for villages where ground patches and vegetation are interleaved at multiple scales.
- **Shared cdist:** Both MSG radii share a single `torch.cdist` computation, halving the VRAM cost of the ball-query step.
- **Gradient Checkpointing:** SA1 and SA2 (the VRAM-heaviest layers due to grouped feature tensors) use `torch.utils.checkpoint`. Activations are not stored during the forward pass and are recomputed during backpropagation. This reduces peak VRAM by ~40% at a ~20% compute cost.
- **Explicit tensor deletion:** `del dist, gxyz, gpts` after each scale — forces immediate VRAM release rather than waiting for the garbage collector.

**Loss function — Focal Loss:**

```
FL(p_t) = α_t × (1 - p_t)^γ × CE(logits, targets)
```

- `α = 0.75`: Upweights the ground class (minority in some tiles).
- `γ = 2.0`: Down-weights easy-to-classify points (most non-ground objects). The model focuses its capacity on hard boundary cases — points at the edge of vegetation, building overhangs, low shrubs — which are the primary source of error.

---

### Cell 7 — Dataset Class

**Purpose:** `DTMDataset` — a `torch.utils.data.Dataset` subclass that handles tile loading, feature assembly, and augmentation.

**Feature assembly per tile:**
```python
features = [x, y, z]                    # (N, 3) raw XYZ
           + [ΔZ, roughness, slope, density]  # (N, 4) from cache or on-the-fly
# → (N, 7) float32 tensor
```

XY coordinates are **centred** (subtract mean) before concatenation so the model is invariant to absolute position within the coordinate system.

**Augmentation pipeline (training only):**

| # | Augmentation | Parameters | Rationale |
|---|---|---|---|
| 1 | Random Z-rotation | θ ~ Uniform[0, 2π] | Terrain is rotationally symmetric |
| 2 | Gaussian XYZ jitter | σ = 0.02 m | Simulates LiDAR sensor noise |
| 3 | Uniform scale | s ~ Uniform[0.90, 1.10] | Simulates varying flight altitude |
| 4 | Random XY flip | p = 0.5 per axis | Doubles effective dataset size |
| 5 | Anisotropic XY scale | s ~ Uniform[0.85, 1.15] per axis | Village layouts are elongated |
| 6 | Z stretch | s ~ Uniform[0.80, 1.20] | Varying tree/building heights across villages |
| 7 | Point dropout | p = 0.3, keep 75–100% | Simulates sparse scan areas |

**Memory optimisations:**
- `mmap_mode="r"` on `points.npy`: file is memory-mapped, not copied to RAM.
- Feature cache loaded as `float32` directly from disk (no conversion cost).
- Sub-sampling with `np.random.choice` to `max_pts=2048` per tile.

---

### Cell 8 — Training Loop

**Purpose:** Full training loop with Focal Loss, OneCycleLR, AMP (fp16), staged auto-promotion, gradient clipping, and checkpointing.

**Optimizer:** AdamW with `weight_decay=1e-4`, `betas=(0.9, 0.999)`.

**Learning rate schedule — OneCycleLR:**
- Starts at `max_lr / 25 = 0.0004`
- Rises to `max_lr = 0.01` over the first 30% of steps (warm-up)
- Decays cosinely to `max_lr / 25000 ≈ 4e-7`
- Steps **per batch**, not per epoch — provides smooth LR throughout.

**AMP (Automatic Mixed Precision):**
- `torch.cuda.amp.GradScaler` manages gradient scaling to prevent fp16 underflow.
- `torch.cuda.amp.autocast` wraps forward pass and loss computation.
- Compatible with PyTorch 2.0.1 on P100/sm_60. Note: `torch.amp.*` (the newer API) is intentionally avoided as it has subtle differences in 2.0.x.

**Staged auto-promotion logic:**
```
After every validation run, check the last `promo_window=3` validation scores:
  if mean(val_acc) ≥ 0.82
  AND mean(F1) ≥ 0.72
  AND std(val_acc) ≤ 0.02    (metrics have plateaued)
  → Promote to next stage (more tiles, same batch size)
  → Rebuild DataLoader with new tile cap
  → Rebuild OneCycleLR scheduler for remaining epochs
  → Reset patience counter
```

**Checkpoint format:**
```python
{
    "ep"    : epoch_number,
    "bva"   : best_val_accuracy,
    "pat"   : patience_counter,
    "sidx"  : stage_index,
    "hist"  : list_of_epoch_dicts,
    "model" : model.state_dict(),
    "opt"   : optimizer.state_dict(),
    "sched" : scheduler.state_dict(),
    "scaler": scaler.state_dict(),
}
```

Checkpoints are saved every epoch to `/kaggle/working/logs/checkpoint.pth`. Training resumes automatically if the file exists (`CFG["resume"] = True`).

**Epoch log format:**
```
Ep 023/035 [full] | T 0.1823/0.9412 | V 0.1631/0.9521 | P0.941 R0.963 F10.952 | lr4.2e-04 187s  *** BEST ***
```

---

### Cell 9 — Threshold Optimisation

**Purpose:** The model outputs a continuous probability P(ground). The default threshold of 0.5 is rarely optimal for imbalanced datasets. This cell sweeps thresholds on the full validation set and selects the one maximising the ground-class F1 score.

**Sweep range:** 0.20 to 0.85 in steps of 0.01 (65 thresholds evaluated).

**Why F1 rather than accuracy?**
Accuracy can be misleadingly high even with poor Precision/Recall balance. F1 = 2·P·R/(P+R) penalises both false positives (mis-classifying non-ground as ground, causing building outlines to appear in the DTM) and false negatives (missing ground points, causing holes).

**Outputs:**
- `threshold.json`: optimal threshold + metrics at that threshold + full sweep table
- `threshold_curve.png`: Accuracy and F1 vs. threshold plot

---

### Cell 10 — DTM Post-Processing

**Purpose:** Converts per-tile predicted ground points into a clean, gap-free, pit-free elevation raster.

This is the step that turns a point cloud segmentation result into a directly usable DTM. Four sub-steps run in sequence:

#### Step 1: Rasterization
```
Ground XYZ → regular elevation grid
```
- Grid resolution: `dtm_resolution = 0.5 m` per cell.
- Elevation per cell: **20th percentile** of Z values within the cell (more robust than pure minimum — resistant to the occasional residual mis-classified point that sits at near-ground level).
- Cells with no ground points are set to `NaN`.

#### Step 2: Hole Filling
```
NaN cells → interpolated elevation
```
- Uses `scipy.interpolate.griddata` with `method="linear"` (barycentric interpolation over Delaunay triangulation of known-elevation cells).
- Cells outside the convex hull of known data fall back to nearest-neighbour interpolation.
- Eliminates no-data gaps caused by occluded areas or low point density.

#### Step 3: Pit Removal
```
Small negative artifacts → morphological fill
```
- `scipy.ndimage.grey_closing(size=3)` applies morphological grey closing (dilation then erosion).
- Fills isolated low pixels that are significantly below their neighbours — artefacts from residual mis-classifications at the boundary of buildings or dense vegetation.
- Applied 2 iterations.

#### Step 4: Gaussian Smoothing
```
High-frequency noise → smooth terrain surface
```
- `scipy.ndimage.gaussian_filter(sigma=1.0)` removes sub-metre noise while preserving real terrain features (ridges, slopes, valleys).
- σ is tunable in `CFG["dtm_smooth_sigma"]`.

**Outputs per tile:**
- `{tile_name}_dtm.npy`: `(H, W)` float32 clean DTM array.
- `{tile_name}_meta.json`: geo-reference info, grid shape, tile name, threshold used.

---

### Cell 11 — Visualisation & Export

**Purpose:** Generates quality-check visualisations and writes the inference artefacts needed for web integration.

**Sample DTM grid:** A matplotlib figure showing 6 randomly selected DTM tiles rendered with the `terrain` colourmap and elevation colourbars. Saved to `logs/sample_dtm_tiles.png`.

**`inference_config.json`:**
```json
{
  "model_arch"       : "DTMNet",
  "model_params"     : {"num_cls": 2, "use_grad_ckpt": false},
  "n_input_features" : 7,
  "feature_order"    : ["x","y","z","delta_z","roughness","slope","density"],
  "max_pts_per_tile" : 2048,
  "optimal_threshold": 0.47,
  "dtm_resolution"   : 0.5,
  "dtm_smooth_sigma" : 1.0,
  "dtm_fill_method"  : "linear",
  "classification_metrics": { "acc": 0.9531, "p": 0.941, "r": 0.963, "f1": 0.952 }
}
```

**`dtm_inference.py`:** Standalone command-line script. Has no dependency on the notebook — copy it alongside `best_model.pth` and `inference_config.json` to any server. Example usage:

```bash
python dtm_inference.py \
    --points  /path/to/tile/points.npy \
    --out     /path/to/output/dtm.npy  \
    --model   best_model.pth           \
    --config  inference_config.json    \
    --max_pts 2048
```

---

### Cell 12 — Output Packaging

**Purpose:** Collects all outputs into a single downloadable zip for easy transfer.

**`dtm_model_outputs.zip` contents:**
```
model/
    best_model.pth          ← PyTorch weights (load with DTMNet)
    inference_config.json   ← All inference parameters
    dtm_inference.py        ← Standalone CLI script
training/
    history.json            ← Per-epoch metrics
    training_curves.png     ← Loss / Accuracy / P-R-F1 plots
    threshold.json          ← Threshold sweep results
    threshold_curve.png     ← Threshold vs accuracy plot
    sample_dtm_tiles.png    ← Visual quality check
dtm_samples/
    tile_XXXX_dtm.npy × 10  ← Sample clean DTM tiles
```

---

## 6. Model Architecture Deep Dive

### Set Abstraction (SA)

Each SA layer samples a set of **centroids** via Farthest Point Sampling (FPS) and groups nearby points within a radius. FPS maximises coverage of the point cloud — important for tiles with non-uniform density.

**FPS implementation:** Iterative greedy algorithm. At each step the point farthest from all already-selected centroids is chosen. Runs entirely on GPU with `torch.Tensor` operations.

**Ball query:** `torch.cdist` computes all pairwise distances between centroids and input points in one operation. Points within the radius are selected with `topk(k, largest=False)`. The cdist matrix is computed **once** and reused across all radii in MSG layers.

### Feature Propagation (FP)

Trilinear interpolation from coarser to finer resolution:
1. For each point at level *l*, find 3 nearest neighbours at level *l+1*.
2. Weight their features inversely by distance.
3. Concatenate with skip-connection features from level *l*.
4. Pass through a shared MLP.

### Why MSG over single-scale SA?

Single-scale SA captures only one neighbourhood size. In Indian village terrain:
- Fine scale (0.5 m): captures individual tree trunks, building edges.
- Medium scale (1.5–3 m): captures tree crowns, small structures.
- Coarse scale (6–12 m): captures building footprints, forest patches.

MSG outputs all scales concatenated, giving the model full multi-resolution context at each layer.

---

## 7. Feature Engineering

### Why Not Raw XYZ Alone?

Raw XYZ gives the model only absolute position. The model must then infer terrain features implicitly from the point pattern, which requires much more capacity and data.

Adding explicit terrain features — derived from classical DTM domain knowledge — provides the model with the discriminators that classical algorithms use, but in a learnable context:

| Classical Algorithm | Feature Used | Our Equivalent |
|---|---|---|
| Progressive Morphological Filter | Height above lowest neighbour | **ΔZ** |
| Multiscale Curvature Classification | Local surface roughness | **Roughness** |
| Cloth Simulation Filter | Terrain inclination | **Slope** |
| Density-based filters | Point count per cell | **Density** |

### Grid Resolution Adaptation

The grid used for feature computation adapts to tile size:
```python
GW = clip(tile_width_m / 2.0,  min=16, max=64)   # cell width ≈ 2 m
GH = clip(tile_height_m / 2.0, min=16, max=64)
```

This ensures features are always computed at a physically meaningful scale regardless of tile extent.

---

## 8. Training Strategy

### Staged Training

| Stage | Tiles (train/val) | Batch | Epochs | Est. Time |
|---|---|---|---|---|
| sanity | 200 / 40 | 6 | 10 | ~2 min |
| medium | 2,000 / 400 | 6 | 20 | ~40 min |
| full | 14,418 / 3,955 | 6 | 35 | ~8.5 h |

Training starts at the `START_PROFILE` stage and automatically advances when metrics stabilise:
- Rolling window of last 3 validation scores
- Mean val_acc ≥ 0.82, mean F1 ≥ 0.72, std(val_acc) ≤ 0.02

### Early Stopping

Patience = 8 validation checks (every 2 epochs = 16 epochs without improvement). Triggered only within the current stage; patience resets on stage promotion.

### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` applied after `scaler.unscale_()` and before `scaler.step()`. Prevents gradient explosions during early training on heterogeneous tile sizes.

---

## 9. DTM Post-Processing Pipeline

### Robust Minimum (20th Percentile)

Using the true minimum Z per cell makes the raster vulnerable to a single residual mis-classification (one non-ground point classified as ground at near-ground height causes a localised low spike). The **20th percentile** is robust against up to 20% mis-classification within any cell while still representing near-ground elevation.

### Hole Filling Method Choice

| Method | Quality | Speed | Use when |
|---|---|---|---|
| `linear` (default) | Smooth, physically plausible | Medium | Normal terrain |
| `nearest` | Sharp boundaries, no smoothing | Fast | Very sparse data |
| `cubic` | Highest quality | Slow | Dense data, high-quality output |

Change via `CFG["dtm_fill_method"]`.

### Smoothing Sigma Guidelines

| Terrain Type | Recommended σ |
|---|---|
| Urban / flat | 0.5 – 1.0 |
| Rural village (default) | 1.0 |
| Steep / complex terrain | 0.5 |
| Very noisy scan | 1.5 – 2.0 |

---

## 10. Configuration Reference

All parameters live in `CFG` (Cell 4). Complete reference:

```python
CFG = {
    # ── Data ──────────────────────────────────────────────────────
    "use_zip"         : False,         # True = stream from zip (no extraction)
    "zip_path"        : "",            # Path to .zip if use_zip=True
    "zip_prefix"      : "",            # Root prefix inside zip
    "training_dir"    : "/kaggle/input/.../Training",

    # ── Paths ─────────────────────────────────────────────────────
    "feat_cache_dir"  : "/kaggle/working/feat_cache",
    "logs_dir"        : "/kaggle/working/logs",
    "model_path"      : "/kaggle/working/logs/best_model.pth",
    "ckpt_path"       : "/kaggle/working/logs/checkpoint.pth",
    "history_path"    : "/kaggle/working/logs/history.json",
    "curves_path"     : "/kaggle/working/logs/curves.png",
    "threshold_path"  : "/kaggle/working/logs/threshold.json",
    "dtm_dir"         : "/kaggle/working/dtm_output",
    "inference_cfg"   : "/kaggle/working/dtm_output/inference_config.json",

    # ── Training ──────────────────────────────────────────────────
    "resume"          : True,          # Auto-resume from ckpt_path if it exists
    "max_pts"         : 2048,          # Points sampled per tile per epoch
    "seed"            : 42,

    # ── Optimiser ─────────────────────────────────────────────────
    "max_lr"          : 0.01,          # OneCycleLR peak learning rate
    "weight_decay"    : 1e-4,          # AdamW L2 regularisation
    "grad_clip"       : 5.0,           # Max gradient norm

    # ── Focal Loss ────────────────────────────────────────────────
    "focal_alpha"     : 0.75,          # Ground class weight (> 0.5 = upweight ground)
    "focal_gamma"     : 2.0,           # Focusing exponent (> 0 = focus on hard points)

    # ── Early Stopping ────────────────────────────────────────────
    "patience"        : 8,             # Val checks without improvement before stopping
    "val_every"       : 2,             # Run validation every N epochs

    # ── Stage Promotion ───────────────────────────────────────────
    "promo_window"    : 3,             # Rolling window size for stability check
    "promo_min_acc"   : 0.82,          # Minimum mean val accuracy to promote
    "promo_min_f1"    : 0.72,          # Minimum mean F1 to promote
    "promo_max_std"   : 0.02,          # Maximum val accuracy std (plateau check)
    "promo_min_eps"   : 3,             # Minimum epochs in stage before promotion

    # ── DataLoader ────────────────────────────────────────────────
    "workers"         : 2,             # DataLoader worker processes
    "prefetch"        : 2,             # Prefetch factor per worker
    "use_amp"         : True,          # fp16 automatic mixed precision
    "grad_ckpt"       : True,          # Gradient checkpointing on SA1+SA2

    # ── DTM Post-Processing ───────────────────────────────────────
    "dtm_resolution"  : 0.5,           # Grid cell size in metres
    "dtm_smooth_sigma": 1.0,           # Gaussian smoothing σ
    "dtm_fill_method" : "linear",      # Hole-fill: "linear"|"nearest"|"cubic"
}
```

---

## 11. Outputs & Artifacts

### During Training (auto-saved)

| File | Description |
|---|---|
| `logs/checkpoint.pth` | Full checkpoint every epoch (model + optimizer + scheduler) |
| `logs/best_model.pth` | Weights at best validation accuracy |
| `logs/history.json` | Per-epoch: loss, accuracy, P, R, F1, LR, stage |
| `logs/curves.png` | Training curves: Loss / Accuracy / P-R-F1 |

### After Threshold Optimisation

| File | Description |
|---|---|
| `logs/threshold.json` | Optimal threshold + full sweep table |
| `logs/threshold_curve.png` | Accuracy & F1 vs. threshold visualisation |

### After DTM Generation

| File | Description |
|---|---|
| `dtm_output/val/{tile}_dtm.npy` | Clean DTM per tile, `(H,W)` float32 |
| `dtm_output/val/{tile}_meta.json` | Geo-reference, shape, n_ground_points |
| `dtm_output/inference_config.json` | All parameters needed for inference |
| `dtm_output/dtm_inference.py` | Standalone CLI inference script |

### Final Package

| File | Description |
|---|---|
| `dtm_model_outputs.zip` | All of the above in one downloadable archive |

---

## 12. Web Interface Integration

### Loading the Model in Python

```python
import torch, json, numpy as np

# Load config
with open("inference_config.json") as f:
    cfg = json.load(f)

# Define DTMNet (copy Cell 6 class definitions) and load weights
model = DTMNet(**cfg["model_params"])
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
```

### Inference on a New Tile

```python
from dtm_inference import compute_features, rasterize, fill_and_smooth
import torch.nn.functional as F

# 1. Load raw point cloud
xyz = np.load("new_tile/points.npy")[:, :3].astype(np.float32)

# 2. Feature engineering (same as training)
feat4 = compute_features(xyz)

# 3. Subsample to max_pts
N = xyz.shape[0]
ch = np.random.choice(N, cfg["max_pts_per_tile"], replace=(N < cfg["max_pts_per_tile"]))
xyz_sub = xyz[ch]; feat_sub = feat4[ch]

# 4. Centre XY
xyz_sub[:, 0] -= xyz_sub[:, 0].mean()
xyz_sub[:, 1] -= xyz_sub[:, 1].mean()
feat = np.concatenate([xyz_sub, feat_sub], axis=1)

# 5. Model inference
with torch.no_grad():
    logits = model(torch.from_numpy(feat).unsqueeze(0))
    prob_ground = F.softmax(logits, dim=-1)[0, :, 1].numpy()

# 6. Extract ground points
threshold = cfg["optimal_threshold"]
xyz_ground = xyz[ch][prob_ground >= threshold]

# 7. Build DTM
dtm_raw, geo = rasterize(xyz_ground, cfg["dtm_resolution"])
dtm_clean = fill_and_smooth(dtm_raw, cfg["dtm_fill_method"], cfg["dtm_smooth_sigma"])

# dtm_clean: (H, W) float32 elevation array, ready to display/download
# geo: (xmin, ymin, resolution) for georeferencing
```

### REST API Example (Flask)

```python
from flask import Flask, request, jsonify, send_file
import io, numpy as np

app = Flask(__name__)

@app.route("/generate_dtm", methods=["POST"])
def generate_dtm():
    # Receive point cloud as binary
    data   = np.frombuffer(request.data, dtype=np.float32).reshape(-1, 3)
    dtm    = run_inference(data)           # your wrapper around the code above
    buf    = io.BytesIO(); np.save(buf, dtm); buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                     download_name="dtm.npy")
```

### Output Interpretation

- `dtm[row, col]` = elevation in metres at position `(xmin + col*res, ymin + row*res)`.
- `NaN` values indicate areas with no coverage (should not appear after hole-filling).
- Multiply by display scale for visualisation; use `extent=[xmin, xmax, ymin, ymax]` with `imshow` for correct georeferencing.

---

## 13. Memory Budget & Performance

### GPU VRAM Usage (P100 16 GB)

| Component | VRAM | Notes |
|---|---|---|
| Model weights | ~4.5 MB | 1.14M params × 4 bytes |
| SA1 cdist matrix (B=6, M=512, N=2048) | ~25 MB fp16 | Shared across both MSG radii |
| SA1 grouped features (B=6, M=512, k=64, C=7) | ~14 MB | |
| SA2 cdist matrix (B=6, M=128, N=512) | ~1.6 MB fp16 | |
| Grad checkpointing saving (SA1+SA2) | -~3 GB | Activations not stored |
| Typical peak usage | ~6–8 GB | Safe on 16 GB P100 |

### System RAM Usage (19 GB budget)

| Component | RAM | Notes |
|---|---|---|
| DataLoader buffer (2 workers × prefetch 2) | ~100 MB | 4 batches × 6 tiles × 2048 pts × 7 feats × 4 bytes |
| Feature cache (all 18K tiles) | ~1.2 GB | (N_mean≈2000) × 4 feats × float32 |
| Tile points (mmap) | ~0 MB | Memory-mapped, not copied |
| Python + PyTorch overhead | ~2–3 GB | |
| Peak during cache build | ~4 GB | float32 ops, GC every 1000 tiles |

### Timing Estimates (P100, AMP fp16, batch=6, max_pts=2048)

| Stage | Tiles | Batches/epoch | Time/epoch | Total |
|---|---|---|---|---|
| sanity | 200 | 33 | ~12 s | ~2 min |
| medium | 2,000 | 333 | ~2 min | ~40 min |
| full | 14,418 | 2,403 | ~14 min | ~8.2 h |

DTM generation (Cell 10) for 3,955 val tiles: ~20–40 minutes depending on tile density.

---

## 14. Troubleshooting

### `CUDA error: no kernel image is available`

**Cause:** Wrong PyTorch version for P100 (sm_60). PyTorch ≥ 2.1 dropped sm_60 support.  
**Fix:** Run Cell 1, then switch to kernel `DTM-P100 (py3.10)`.

### `torch==2.0.1+cu117 not found` in pip

**Cause:** The `--extra-index-url` approach only works if pip can reach the index. Cell 1 uses a direct wheel URL instead of PyPI lookup, which bypasses this problem.

### `FileNotFoundError: conda`

**Cause:** Kaggle GPU notebooks do not have conda installed.  
**Fix:** Cell 1 uses `apt` + `venv` + direct pip wheel — no conda required.

### GPU not detected after changing kernel

**Cause:** Kaggle may require a page refresh after kernel change.  
**Fix:** Refresh the notebook page, select `DTM-P100 (py3.10)` again, run from Cell 2.

### Feature cache takes too long

**Cause:** 18K tiles × compute_features takes 30–90 minutes depending on tile size.  
**Fix:** This runs once only. Subsequent sessions skip cached tiles (`force=False` by default). Run `build_cache(CFG, force=False)` to resume an interrupted build.

### Val accuracy stuck at ~91–92%

**Cause:** Model may be at the `sanity` or `medium` stage, or `focal_alpha` may need tuning.  
**Fix:**
- Set `START_PROFILE = "full"` and ensure the full stage runs for 20+ epochs.
- If precision is high but recall is low, increase `focal_alpha` to 0.80.
- If recall is high but precision is low, decrease `focal_alpha` to 0.70.
- Check that `grad_ckpt=True` — without it you may be hitting OOM that causes silent batch drops.

### DTM has many holes / nodata regions

**Cause:** Too few ground points predicted (recall too low), or `dtm_resolution` too fine.  
**Fix:**
- Lower `OPT_THRESH` in Cell 10 (accept more false-positive ground points to reduce holes).
- Increase `dtm_resolution` to 1.0 m (larger cells capture more sparse points).
- Change `dtm_fill_method` to `"nearest"` for more aggressive gap filling.

### OOM during training

**Cause:** Batch too large, or gradient checkpointing disabled.  
**Fix:**
- Confirm `CFG["grad_ckpt"] = True`.
- Reduce `CFG["batch"]` from 6 to 4.
- Reduce `CFG["max_pts"]` from 2048 to 1024 (will be faster but slightly less accurate).
- Call `free_memory()` after Cell 5 before starting training.

### Training not resuming

**Cause:** `ckpt_path` doesn't exist yet (first run), or `CFG["resume"] = False`.  
**Fix:** This is expected on first run — the log `ℹ️ Training from scratch` is correct. On subsequent runs with a saved checkpoint, set `CFG["resume"] = True`.

---

*This README corresponds to `dtm_generator_final.ipynb`. For questions or issues, refer to the inline comments in each notebook cell.*
