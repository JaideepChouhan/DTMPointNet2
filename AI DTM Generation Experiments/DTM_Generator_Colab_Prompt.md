# 🛰️ LLM Prompt: AI-Driven DTM Generator on Google Colab (TPU)
> Copy everything between the triple-dashed lines and paste it directly into your LLM of choice (Claude, GPT-4o, Gemini, etc.)

---

## ═══════════════════════════════════════════════════════════════
## PROMPT START — COPY FROM HERE
## ═══════════════════════════════════════════════════════════════

You are an expert geospatial AI engineer and Python developer. I need you to build a **complete, production-quality, AI-driven Digital Terrain Model (DTM) generator** that runs on **Google Colab connected via VS Code, using a TPU accelerator**. I am an absolute beginner to AI/ML modelling, so every piece of code must be:

- Fully commented with plain-English explanations of what each block does and **why**
- Modular (one clearly named function per task)
- Robust with try/except error handling and informative print statements
- Self-contained in a **single Jupyter notebook** (`.ipynb`) structured as numbered sections

---

## 1. CONTEXT AND GOAL

### 1.1 What We Are Building
A full end-to-end pipeline that:
1. Reads raw LiDAR point cloud files (`.las` / `.laz` format) from Indian village drone surveys
2. Classifies every 3D point as **ground** or **non-ground** (vegetation, buildings, structures) using a trained PointNet++ AI model
3. Keeps **only ground points**, preserving their **original measured elevation values exactly** — do NOT alter Z coordinates
4. Interpolates those ground points onto a smooth, regular-grid **Digital Terrain Model (DTM) raster** at 1 m resolution
5. Fills all pits, sinks, and holes in the terrain (hydrological conditioning) to make the surface hydrologically correct and ready for drainage network modelling
6. Exports the final result as a **Cloud-Optimised GeoTIFF (COG)** per village
7. Produces validation metrics (RMSE, MAE, bias) for each output

### 1.2 What a DTM Is
A DTM is a **bare-earth elevation model** — imagine you removed every tree, building, and structure and took a photograph of the exposed ground from directly above, coloured by height. It is **not** a Digital Surface Model (DSM), which would show rooftops and tree canopies. We need the bare earth because later I will build drainage network and water-logging analysis on top of it.

### 1.3 Critical Constraint: Preserve Original Elevations
The AI model only **classifies** points (ground vs non-ground). The elevation (Z) values of ground points must remain exactly as measured by the LiDAR sensor. We interpolate from real measured points. We do not predict or modify elevation values.

---

## 2. HARDWARE AND RUNTIME CONFIGURATION

- **Platform**: Google Colab (accessed via VS Code remote)
- **Accelerator**: TPU v2/v3 (Google Colab free or Pro TPU)
- **PyTorch XLA**: Use `torch_xla` for TPU-accelerated training
- **Fallback**: If TPU is unavailable at runtime, automatically fall back to CPU with `torch.set_num_threads(8)`
- **RAM budget**: Assume ~12 GB usable RAM — use streaming/chunking to avoid OOM errors
- **Storage**: Use Google Drive mount at `/content/drive/MyDrive/` for persistent storage of models and outputs

At the top of the notebook, auto-detect the device:
```python
# Auto-detect TPU → GPU → CPU in that priority order
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    DEVICE_NAME = "TPU"
except ImportError:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Using device: {DEVICE_NAME} → {device}")
```

---

## 3. DATA DIRECTORY STRUCTURE

The data is mounted from Google Drive. The exact directory tree is:

```
/content/drive/MyDrive/GSI_2026/Data/
├── Andaman_and_Nicobar_Islands_1/
│   └── Andaman and Nicobar Islands 1/
│       └── Kadamtala_Rangat_A&N_02022022_group1_densified_point_cloud.laz
├── Andaman_and_Nicobar_Islands_2/
│   └── Andaman and Nicobar Islands 2/
│       └── Gandhinagar_Diglipur_group1_densified_point_cloud.laz
├── Gujrat_Point_Cloud/
│   ├── DEVDI_POINT CLOUD (511671).las
│   └── KHAPRETA_510206.laz
├── Punjab_Point_Cloud/
│   ├── Dhal_Hoshiarpur_31235.las
│   └── DHUNDA_FATEHGARH SAHIB_32619.laz
├── Rajasthan_Point_Cloud/
│   ├── 64334_2H (REFLIGHT)_POINT CLOUD.LAS
│   └── 67169_5NKR_CHAKHIRASINGH.las
└── Tamil_Nadu_Point_Cloud/
    └── Tamil Nadu_Point_Cloud/
        ├── PIRAYANKUPPAM.las
        └── THANDALAM.las
```

Write a function `discover_las_files(base_dir)` that **recursively walks** this tree, finds every `.las` and `.laz` file (case-insensitive), and returns a list of dictionaries:

```python
[
  {
    "path": "/content/drive/.../DEVDI_POINT CLOUD (511671).las",
    "name": "DEVDI_POINT_CLOUD_511671",   # sanitised name (no spaces/parens)
    "state": "Gujarat",                   # inferred from parent folder name
    "epsg": 32643                         # derived from state (see EPSG table below)
  },
  ...
]
```

### EPSG Code Mapping (must be hard-coded):
| State folder keyword | State name | UTM Zone | EPSG |
|---|---|---|---|
| `Gujrat` | Gujarat | 43N | 32643 |
| `Punjab` | Punjab | 43N | 32643 |
| `Rajasthan` | Rajasthan | 43N | 32643 |
| `Tamil` | Tamil Nadu | 44N | 32644 |
| `Andaman` | Andaman & Nicobar | 47N | 32647 |

---

## 4. NOTEBOOK STRUCTURE

Create a single Jupyter notebook named `DTM_Generator_TPU.ipynb` with the following **numbered sections as markdown headers**. Each section must have:
- A markdown cell explaining what this step does, why it matters, and what inputs/outputs it has
- A code cell that implements it
- A print summary at the end confirming success

### Section 0: Setup and Installation
Install all dependencies. Handle the special case of TPU vs non-TPU installs.

```
pip install laspy[lazrs,laszip]==2.5.4 numpy pandas matplotlib scipy
pip install scikit-learn rasterio whitebox tqdm pyproj
pip install cloth-simulation-filter rio-cogeo
pip install torch torchvision tqdm
# TPU-specific (only if on TPU runtime):
pip install cloud-tpu-client torch_xla
```

Also install `open3d` for optional 3D visualisation.

Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Section 1: Configuration
Define a single `CONFIG` dictionary at the top that controls the entire pipeline. The user should only need to edit this one cell. Include:
```python
CONFIG = {
    "data_root": "/content/drive/MyDrive/GSI_2026/Data",
    "output_root": "/content/drive/MyDrive/GSI_2026/Outputs",
    "preprocessed_dir": "/content/drive/MyDrive/GSI_2026/Preprocessed",
    "training_dir": "/content/drive/MyDrive/GSI_2026/Training",
    "logs_dir": "/content/drive/MyDrive/GSI_2026/Logs",
    "model_path": "/content/drive/MyDrive/GSI_2026/Logs/best_model.pth",
    
    # Preprocessing
    "block_size_m": 50,           # spatial tile size in metres
    "stride_m": 40,               # stride between tiles (overlap = block - stride)
    "min_pts_per_block": 200,     # discard tiles with fewer points
    "outlier_pct": (1.0, 99.0),   # Z percentiles for noise removal
    "max_points_per_block": 4096, # subsample to this for model input
    
    # Training
    "train_val_split": 0.80,      # 80% train, 20% val
    "epochs": 50,
    "batch_size": 2,              # reduced for TPU memory management
    "learning_rate": 0.001,
    "early_stop_patience": 10,    # stop if val_acc doesn't improve for N epochs
    "random_seed": 42,
    
    # DTM interpolation
    "dtm_resolution_m": 1.0,     # output raster resolution in metres
    "idw_power": 2.0,            # IDW power parameter
    "idw_k_neighbours": 12,      # number of nearest neighbours for IDW
    "idw_radius_factor": 5,      # search radius = resolution × this factor
    "idw_chunk_size": 50000,     # process this many grid cells at a time
    
    # Hydrological conditioning
    "breach_dist": 5,            # max breach distance for depression filling
    "stream_threshold": 1000,    # flow accumulation threshold for stream extraction
    
    # CSF pseudo-labelling (used when no Class 2 labels exist in LAS)
    "csf_cloth_resolution": 0.5,
    "csf_class_threshold": 0.3,
    "csf_iterations": 500,
    
    # Smoothing
    "smoothing_sigma": 0.5,       # Gaussian smoothing sigma (applied only to NoData-filled areas)
}
```

### Section 2: Data Discovery and Exploration
Implement `discover_las_files(base_dir)` as described in Section 3 above.

Then implement `explore_las_file(path)` that prints a structured summary including:
- Total point count
- X/Y/Z min/max ranges
- CRS/coordinate reference system from header (if present)
- All available classification classes with counts and percentages
- Whether Class 2 (Ground) labels already exist and how many
- File size on disk
- Estimated memory footprint if fully loaded

Run this on all discovered files and save a summary CSV to `outputs/exploration_summary.csv`.

Explain in comments:
- Class 2 in LAS standard = Ground (bare earth)
- Class 1 = Unclassified
- Classes 3/4/5 = Low/Medium/High Vegetation
- Class 6 = Building
- Class 7 = Noise
- Class 9 = Water

### Section 3: Preprocessing (Tiling and Normalisation)
Implement `preprocess_las_file(las_path, out_dir, config)` that:

1. **Reads** the LAS/LAZ file using `laspy`
2. **Removes Z outliers** using percentile clipping (config `outlier_pct`) — explain that extreme Z values are sensor errors, not real terrain
3. **Tiles** the point cloud into `block_size × block_size` metre blocks using a sliding window with `stride` overlap — explain why overlap avoids edge artefacts
4. For each tile block:
   - **Saves** `points.npy` — normalised XYZ (X and Y shifted to local tile origin, Z shifted to tile Z minimum). Explain: normalisation makes the model translation-invariant
   - **Saves** `global_points.npy` — original world coordinate XYZ (unchanged) — these are the real elevations we use for DTM output
   - **Saves** `origin.npy` — the world coordinate origin of this tile
   - **Saves** `labels.npy` (binary: 1=ground, 0=non-ground) — only if Class 2 classifications exist in the source file
   - Skips tiles with fewer than `min_pts_per_block` points

Explain the separation between normalised (model input) and global (DTM output) coordinates in a markdown cell.

### Section 4: Label Generation
Implement three labelling strategies, clearly explaining each:

**Strategy A — Use Existing LAS Classifications (Best Quality)**
If `labels.npy` was already saved by the preprocessor (Class 2 existed), use it directly. This is the recommended first choice.

**Strategy B — Cloth Simulation Filter (CSF) Pseudo-Labels**
For files without Class 2 labels:
```python
from CSF import CSF
```
Implement `generate_csf_labels(block_dir, config)` that processes each tile.
Explain in comments: CSF simulates a cloth falling from above onto an inverted point cloud. Points near where the cloth settles are ground.

**Strategy C — Guidance for Manual Labelling**
Print instructions pointing the user to CloudCompare (free tool) for high-quality manual labelling. No code needed, just a markdown explanation.

After labelling, implement `build_training_split(preprocessed_root, training_dir, config)` that:
- Finds all blocks with `labels.npy`
- Shuffles with `random_seed`
- Splits 80/20 train/val
- Copies to `training_dir/train/` and `training_dir/val/`
- Saves `split_manifest.json`
- Prints counts

### Section 5: AI Model Definition (PointNet++)
Implement the `PointNet2GroundSeg` model in pure PyTorch (no custom CUDA extensions needed, making it compatible with TPU via PyTorch XLA).

The model must have:

**5.1 — `farthest_point_sample(xyz, n_sample)`**
Explain: Selects spatially spread-out "centroid" points using iterative farthest-point sampling. Like placing survey markers spread evenly across a field.

**5.2 — `index_points(pts, idx)`**
Utility to gather points by index (batch-aware).

**5.3 — `SetAbstraction(n_ctr, radius, n_samp, in_ch, mlp_dims)`**
Explain: The "zooming out" layer. Selects centroids, groups nearby neighbours, and learns local shape features (like learning that "these points form a flat surface = ground" vs "these curve upward = tree").

Uses ball-query grouping via `topk` on squared distances (no custom CUDA ops).

**5.4 — `FeaturePropagation(in_ch, mlp_dims)`**
Explain: The "zooming back in" layer. Interpolates learned features back to every original point using inverse-distance weighted averaging. Like spreading knowledge from centroid points back to all their neighbours.

**5.5 — `PointNet2GroundSeg(num_classes=2)`**
Full model:
```
SA1: 512 centroids, radius 0.5m, 32 neighbours → MLP [32, 32, 64]
SA2: 128 centroids, radius 2.0m, 64 neighbours → MLP [64, 64, 128]
SA3:  32 centroids, radius 8.0m, 128 neighbours → MLP [128, 128, 256]
FP3: in=256+128 → MLP [256, 256]
FP2: in=256+64  → MLP [256, 128]
FP1: in=128     → MLP [128, 128, 128]
Head: Linear(128→128) → BN → ReLU → Dropout(0.5) → Linear(128→2)
```

Explain the architecture is like a hierarchy: SA1 learns tiny local features (1-2 points), SA2 learns neighbourhood features (a few square metres), SA3 learns large-scale terrain features (tens of metres). Then FP restores per-point predictions.

Add a function `count_parameters(model)` that prints total trainable parameters.

### Section 6: Dataset Loader
Implement `PointCloudDataset(root_dir, augment=False, max_points=4096)` as a PyTorch `Dataset`:

- Loads `points.npy` (normalised, model input) and `labels.npy`
- Subsamples/oversamples to exactly `max_points` points (random choice with/without replacement)
- If `augment=True`:
  - Random rotation around Z axis (0–360°) — terrain is rotationally symmetric when viewed from above
  - Small random Gaussian jitter (σ=0.01 m) — simulates sensor noise variation
  - Random uniform scale (0.9–1.1×) — simulates altitude variation
  - Explain each augmentation and why it helps generalisation

Include class imbalance handling: compute ground point ratio and print a warning if it's below 5% or above 95%.

### Section 7: Training
Implement `train_model(config, device)` with full training loop:

**TPU-specific handling:**
```python
# For TPU, use torch_xla's optimised mark_step
if DEVICE_NAME == "TPU":
    import torch_xla.core.xla_model as xm
    # After each batch backward, call xm.mark_step()
    # Use xm.optimizer_step(optimizer) instead of optimizer.step()
    # Use xm.master_print() for logging from TPU process 0 only
```

**Training loop features:**
- Weighted cross-entropy loss to handle class imbalance (compute class weights from training set)
- Adam optimiser with weight decay 1e-4
- Cosine annealing learning rate scheduler
- Early stopping: save best model based on validation accuracy; stop if no improvement for `early_stop_patience` epochs
- Log per-epoch: train loss, train acc, val loss, val acc, elapsed time
- Save history to `logs/history.json`
- Save best model to `logs/best_model.pth`
- Print estimated remaining time per epoch

**Metrics to compute and display per epoch:**
- Overall accuracy
- Ground point precision (of points predicted ground, how many really are ground)
- Ground point recall (of real ground points, how many did we find)
- F1 score for ground class

Explain each metric in comments: recall matters most for DTM quality — missing ground points creates holes; false positives add noise but IDW can handle sparse errors.

After training, plot and save training curves (loss and accuracy vs epoch) to `logs/training_curves.png`.

### Section 8: Inference
Implement `run_inference(village_info, model_path, config, device)` that:

1. Loads the trained model weights
2. Iterates over all preprocessed tile blocks for the village
3. For each block:
   - Loads `points.npy` (normalised, model input)
   - Loads `global_points.npy` (original world coordinates)
   - Subsamples to `max_points` if needed
   - Runs model forward pass (no gradient, eval mode)
   - Gets per-point predictions (0=non-ground, 1=ground)
   - Extracts `global_points[predictions == 1]` — these are the ground points in original world coordinates
4. Concatenates all ground points from all tiles (deduplicate spatially with a 0.1 m grid to remove overlap duplicates)
5. Saves to `outputs/{name}_ground.npy`
6. Prints: total input points, total ground points found, ground percentage

Explain: We use global (world) coordinates for DTM output, not normalised ones, so the final DTM has real-world geographic coordinates and correct elevations.

### Section 9: DTM Generation (IDW Interpolation)
Implement `generate_dtm(ground_pts_path, epsg, config, out_path)` using **Inverse Distance Weighting (IDW)**:

Explain the IDW formula:
```
For each grid cell at position p:
  z_hat(p) = Σ(w_i × z_i) / Σ(w_i)
  where w_i = 1 / (distance(p, p_i)^power + epsilon)
```

Plain English: "Nearby measured ground points have more influence on the grid cell's estimated elevation than far-away ones. The power parameter controls how quickly influence drops with distance."

Implementation:
- Build a regular grid from xmin to xmax, ymin to ymax at `dtm_resolution_m` spacing
- Use `scipy.spatial.cKDTree` for fast nearest-neighbour search
- Process grid in chunks of `idw_chunk_size` cells to avoid RAM overflow
- Use `workers=-1` in `tree.query()` for parallelism
- Set cells with no neighbours within `idw_radius_factor × resolution` as NoData (NaN)
- Write GeoTIFF with correct CRS (using EPSG code), transform, and NaN nodata value

After IDW, implement **NoData hole filling**:
- Detect NaN cells (areas with no ground points — usually inside buildings or dense canopy)
- Fill using iterative nearest-neighbour propagation from surrounding valid cells
- Apply gentle Gaussian smoothing (sigma=0.5) **only** to filled/interpolated areas, not to cells interpolated directly from LiDAR measurements — explain this preserves real terrain texture while making filled areas blend smoothly

### Section 10: Hydrological Conditioning
Implement `condition_hydrology(dtm_path, out_dir, config)` using WhiteboxTools:

Explain the problem: Raw DTMs have tiny depressions called "sinks" — cells that are lower than all their neighbours. These are often LiDAR noise or interpolation artefacts. If we ran water flow modelling on a raw DTM, water would get stuck in these sinks and never reach the drainage outlet. This step fixes that.

Steps (in order — do not change order):
1. **Breach depressions** (`breach_depressions_least_cost`) — carves minimal channels through depression rims. Preferred over filling because it preserves terrain morphology better
2. **Fill remaining depressions** (`fill_depressions`, `fix_flats=True`) — fills anything the breach step couldn't resolve
3. **D8 flow direction** (`d8_pointer`) — for each cell, determines which of its 8 neighbours the water flows to (the steepest downhill direction)
4. **Flow accumulation** (`d8_flow_accumulation`) — counts how many upstream cells drain through each cell (large values = main channels)
5. **Stream extraction** (`extract_streams`, threshold from config) — cells with flow_accumulation > threshold are classified as streams

Save all intermediate rasters to `outputs/hydro/{name}/` and the final conditioned DTM as `{name}_conditioned.tif`.

Include a comment explaining: "The conditioned DTM is hydrologically 'correct' — if you poured water anywhere on it, it would always flow downhill to an outlet without getting stuck. This is the version we use for drainage network and water-logging analysis later."

### Section 11: Cloud-Optimised GeoTIFF Export
Implement `export_cog(input_tif, output_cog, config)` using `rio-cogeo`:

```python
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
```

Settings:
- Compression: Deflate with predictor=3 (lossless, optimal for floating-point elevation data)
- Block size: 512×512 pixels (standard for cloud streaming)
- Overview levels: 5 (for fast previewing at multiple zoom levels)
- Data type: float32

Explain: A COG is a regular GeoTIFF but reorganised internally so that web GIS tools can read just the portion they need (e.g., one tile on a map) without downloading the whole file. This is the delivery format expected by GIS professionals.

### Section 12: Validation
Implement `validate_dtm(cog_path, ref_las_path, n_samples=5000)` that:

1. Loads reference LAS file and extracts Class 2 (Ground) points as ground truth
2. Randomly samples up to `n_samples` reference points
3. For each reference point, looks up its corresponding DTM cell value
4. Computes:
   - **RMSE** (Root Mean Square Error) — target: < 0.15 m
   - **MAE** (Mean Absolute Error) — target: < 0.10 m
   - **Bias** (mean signed error) — target: ≈ 0
   - **Std** (standard deviation of errors)
   - **P95** (95th percentile of absolute errors) — target: < 0.30 m
5. Saves results to `outputs/{name}_validation.json`
6. Prints a colour-coded pass/fail for each metric (use ✅ / ❌ emoji)

Explain each metric in plain English:
- RMSE: "On average, our DTM elevation is within X metres of the true ground elevation"
- Bias: "Our model systematically over/underestimates elevation by X metres"
- P95: "In 95% of locations, our error is less than X metres"

### Section 13: Visualisation
Implement `visualise_dtm(cog_path, streams_path=None, out_png=None)`:

Create a 2-panel matplotlib figure:
- Left panel: elevation colourmap (`terrain` colourmap)
- Right panel: hillshade view (simulated sun from NW at 45° elevation, vertical exaggeration 2×)
- If `streams_path` provided: overlay stream network in blue on hillshade panel
- Add colourbar with elevation range
- Save to PNG

Also implement `plot_point_cloud_preview(las_path, n=50000)` that:
- Samples up to 50,000 points from the LAS file
- Colours by classification (Ground=brown, Vegetation=green, Building=red, Unclassified=grey)
- Creates a scatter plot (X vs Z — a side profile view)
- Useful for quick quality inspection before processing

### Section 14: Full Pipeline Runner
Implement `run_full_pipeline(config, device)` that calls all previous sections in sequence for every discovered LAS/LAZ file:

```
For each village file:
  1. explore_las_file()        → log summary
  2. preprocess_las_file()     → tiles in preprocessed/
  3. generate_csf_labels()     → if no Class 2 labels exist
  4. (training skipped — uses existing model or trains once)
  5. run_inference()            → ground points in outputs/
  6. generate_dtm()            → raw DTM raster
  7. condition_hydrology()     → conditioned DTM + streams
  8. export_cog()              → final deliverable COG
  9. validate_dtm()            → accuracy report
 10. visualise_dtm()           → preview PNG
```

Also implement `train_pipeline(config, device)` separately, which:
- Runs preprocessing for all files
- Generates CSF labels where needed
- Builds the training split
- Trains the model
- Saves the best model checkpoint

Print a final summary table at the end: one row per village showing RMSE, MAE, file size of COG, processing time.

### Section 15: Troubleshooting and Tips
Include a markdown section with:

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `MemoryError` | Tile too large | Reduce `block_size_m` to 30 |
| `ModuleNotFoundError: laspy` | Not installed | `pip install laspy[lazrs,laszip]` |
| `NaN DTM everywhere` | No ground points found | Check CSF parameters or use CloudCompare labels |
| `WhiteboxTools binary missing` | First run | Call `wbt.refresh_tools()` |
| TPU `xla` errors | Not a TPU runtime | Pipeline auto-falls back to CPU |
| `Large white holes in DTM` | Dense vegetation/buildings | Increase `idw_radius_factor` or run hole-fill step |
| `Training loss not decreasing` | LR too high | Reduce `learning_rate` to 0.0005 |
| `CSF labels all non-ground` | Very flat terrain | Reduce `csf_cloth_resolution` to 0.3 |
| `CRS error on GeoTIFF write` | Wrong EPSG | Check EPSG table in CONFIG section |

Also include TPU-specific tips:
- Use `xm.mark_step()` after every batch backward pass to flush XLA operations
- Keep batch sizes small (2–4) on TPU to avoid recompilation overhead
- The first epoch is always slow on TPU (JIT compilation) — subsequent epochs are much faster
- Use `xm.rendezvous('init')` if using multi-core TPU

---

## 5. CODE QUALITY REQUIREMENTS

Every function must follow this pattern:
```python
def function_name(param1, param2, config):
    """
    One-sentence summary of what this does.
    
    Why this step exists:
        Plain-English explanation of the problem this solves.
    
    Args:
        param1: description
        param2: description
        config: the global CONFIG dictionary
    
    Returns:
        description of return value
    """
    print(f"\n{'='*60}")
    print(f"  STEP: function_name")
    print(f"  Input: {param1}")
    print(f"{'='*60}")
    
    try:
        # ... implementation ...
        print(f"  ✅ Success: ...")
        return result
    except Exception as e:
        print(f"  ❌ Error in function_name: {e}")
        raise
```

---

## 6. FINAL DELIVERABLES FROM THE NOTEBOOK

After running the full pipeline, the `Outputs/` directory must contain:

```
GSI_2026/Outputs/
├── exploration_summary.csv
├── {village_name}_ground.npy            ← classified ground points (world coords)
├── {village_name}_dtm_raw.tif           ← raw IDW-interpolated DTM
├── {village_name}_COG.tif              ← final Cloud-Optimised GeoTIFF
├── {village_name}_validation.json       ← accuracy metrics
├── {village_name}_preview.png           ← visualisation
├── hydro/
│   └── {village_name}/
│       ├── {name}_breached.tif
│       ├── {name}_filled.tif            ← hydrologically conditioned DTM
│       ├── {name}_d8_dir.tif
│       ├── {name}_d8_acc.tif
│       └── {name}_streams.tif          ← drainage network raster
└── logs/
    ├── best_model.pth                   ← trained model weights
    ├── history.json                     ← training loss/accuracy per epoch
    └── training_curves.png             ← plot of training history
```

---

## 7. IMPORTANT CONSTRAINTS — DO NOT VIOLATE THESE

1. **Never modify Z values of ground points** — they are real measurements
2. **Use float32 throughout** — float64 doubles RAM usage unnecessarily
3. **Never load the entire LAS file into a model at once** — always tile/chunk
4. **The Gaussian smoothing in the hole-fill step** must only apply to cells that had NoData (no LiDAR measurement), not to IDW-interpolated cells
5. **TPU batch dimensions must be static** (fixed size) — pad batches if the last batch is smaller, then trim predictions
6. **All file paths must use `pathlib.Path`** not string concatenation — handles spaces in filenames (e.g., `DEVDI_POINT CLOUD (511671).las`)
7. **The conditioned DTM (output of WhiteboxTools `fill_depressions`)** is the correct input for future drainage network analysis — always use this, not the raw DTM
8. **EPSG codes are critical** — wrong CRS makes the output geographically displaced. Use the mapping table exactly as provided

---

## 8. TRAINING STRATEGY FOR THE HACKATHON

Since some LAS files may already have Class 2 labels (pre-classified by the survey agency), and others may not, implement this logic:

1. First pass: preprocess all files and check which have Class 2 labels
2. Files WITH labels → use as primary training data (high-quality supervision)
3. Files WITHOUT labels → generate CSF pseudo-labels → use as secondary training data
4. Train on a combined dataset
5. Run inference on ALL files (including those used for training) to produce DTMs

This is valid because the model learns a generalised concept of "ground geometry" and will perform better even on its training files than CSF alone.

---

## 9. WHAT TO BUILD FOR LATER (DO NOT BUILD NOW, BUT PREPARE FOR IT)

Leave clearly marked placeholder cells (with `# TODO: DRAINAGE NETWORK` and `# TODO: WATER LOGGING` comments) after Section 10, explaining that:
- **Drainage network vectorisation**: The `_streams.tif` raster from WhiteboxTools can be converted to vector lines using `wbt.raster_streams_to_vector()` — this will be Phase 2
- **Water logging / flood inundation**: Requires setting an inlet water level and computing which cells would be inundated — this will use the conditioned DTM as the terrain base — Phase 3

---

Now generate the complete, fully working `DTM_Generator_TPU.ipynb` Jupyter notebook following all specifications above. Start with Section 0 and go through all 15 sections. Do not skip any section. Include all helper functions. Make sure all imports are at the top of the relevant cells. The notebook should run from top to bottom without errors on a fresh Google Colab TPU runtime.

## ═══════════════════════════════════════════════════════════════
## PROMPT END — STOP COPYING HERE
## ═══════════════════════════════════════════════════════════════

---

## 📋 HOW TO USE THIS PROMPT

1. **Copy everything between the `═══` lines** (the PROMPT START / PROMPT END markers)
2. **Paste into your LLM** (Claude Opus, GPT-4o, or Gemini 1.5 Pro recommended for long outputs)
3. **If the LLM truncates output**, follow up with: `"Continue from Section X where you left off. Do not repeat previous sections."`
4. **Save the generated code** as `DTM_Generator_TPU.ipynb` in your Google Drive at `MyDrive/GSI_2026/`
5. **Update Section 1 CONFIG** to point to your actual data directory before running

## ⚙️ RECOMMENDED LLM SETTINGS
- Temperature: 0.2 (lower = more precise, less creative)
- Max output tokens: Maximum available (this is a long generation)
- Model: Claude Opus 4 / GPT-4o / Gemini 1.5 Pro (any frontier model)

## 🗂️ YOUR DATA PATH IN COLAB
After mounting Google Drive, your data will be at:
```
/content/drive/MyDrive/GSI_2026/Data/
```
Update the `data_root` in CONFIG accordingly.

## ⚠️ COMMON MISTAKES TO AVOID WHEN RUNNING THE NOTEBOOK
- Do NOT run Section 13 (Full Pipeline) before running Section 7 (Train) at least once
- Make sure the Colab runtime is set to **TPU** before starting (Runtime → Change runtime type → TPU)
- The first epoch will appear slow — this is normal (XLA JIT compilation)
- Keep the Colab tab active during training or use `nohup` / tmux via VS Code terminal

---
*Generated for: MoPR Geospatial Intelligence Hackathon 2026 — DTM from LiDAR Problem Statement*
*Handbook Version: 1.0 | Prompt Version: 1.0*
