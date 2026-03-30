# 🔧 Modification Prompt: Local Storage + Google Colab TPU Hybrid Setup
> This prompt tells the LLM exactly what to change in the already-generated notebook.
> Copy everything between the triple-dashed lines and paste into your LLM.

---

## ═══════════════════════════════════════════════════════════════
## PROMPT START — COPY FROM HERE
## ═══════════════════════════════════════════════════════════════

I have already generated a notebook called `DTM_Generator_TPU.ipynb` using your previous instructions. Now I need you to **modify specific sections only** — do not regenerate sections that are unchanged.

## THE PROBLEM TO SOLVE

My data lives on my **local Ubuntu 24.04 laptop** at:
```
/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Data/
```

I **do not** want to use Google Drive. My machine has:
- CPU: Intel Core i7 8th Gen (4 cores / 8 threads)
- RAM: 16 GB DDR4
- Storage: 512 GB SSD
- OS: Ubuntu 24.04
- GPU: None

I **do** want to use Google Colab's TPU for model training only.

The conflict is: **Google Colab TPU runs on Google's cloud servers — it cannot directly read files from my laptop's hard drive.** You must solve this with the hybrid architecture described below.

---

## THE SOLUTION: A TWO-ENVIRONMENT HYBRID PIPELINE

Split the pipeline into two environments:

### Environment A — LOCAL (runs on my Ubuntu laptop in VS Code / Jupyter Lab)
All steps that touch large raw files run locally:
- Section 1: CONFIG (updated paths)
- Section 2: Data Discovery and Exploration
- Section 3: Preprocessing (LAS → tiled .npy blocks)
- Section 4: Label Generation (CSF pseudo-labels)
- Section 4b: Build Training Split + **COMPRESS and UPLOAD to Colab**
- Section 8: Inference (load model weights back, classify all villages)
- Section 9: DTM Generation
- Section 10: Hydrological Conditioning
- Section 11: COG Export
- Section 12: Validation
- Section 13: Visualisation

### Environment B — GOOGLE COLAB TPU (runs in browser or VS Code remote Colab)
Only the training step runs on Colab:
- Section 5: Model Definition (PointNet++)
- Section 6: Dataset Loader
- Section 7: Training on TPU ← **this is the only reason we use Colab**
- Section 7b: **DOWNLOAD model weights back to local machine**

The training data (preprocessed `.npy` tiles) is typically **10–30 MB total** even for all 10 villages — small enough to upload to Colab in seconds. The raw LAS files (100 MB – 2 GB each) never leave the laptop.

---

## WHAT TO MODIFY: SECTION-BY-SECTION INSTRUCTIONS

### MODIFY Section 0: Setup — Split into Two Sub-Sections

**Section 0-A: Local Environment Setup**
Add a cell at the top of the local notebook with a `pip install` list for Ubuntu:
```bash
pip install laspy[lazrs,laszip]==2.5.4 numpy pandas matplotlib scipy \
            scikit-learn rasterio whitebox tqdm pyproj \
            cloth-simulation-filter rio-cogeo torch torchvision \
            jupyterlab ipywidgets
```
Add a clear markdown warning:
> ⚠️ Run this notebook LOCALLY in VS Code or JupyterLab, NOT in Google Colab.
> Only the Colab Training Notebook (separate file) runs in the cloud.

**Section 0-B: Colab Training Notebook Setup**
Create a **second, separate, standalone notebook** called `DTM_Colab_Training_TPU.ipynb`.
This notebook is self-contained and runs 100% in Google Colab with TPU.
Its setup cell:
```python
# Run this in Google Colab with TPU runtime
!pip install torch torchvision tqdm numpy --quiet

# Detect TPU
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    DEVICE_NAME = "TPU"
    print("✅ TPU detected and ready")
except ImportError:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"⚠️  TPU not available, using {DEVICE_NAME}")
```
Do NOT mount Google Drive in this notebook. Do NOT reference any Google Drive paths.

---

### MODIFY Section 1: CONFIG — Two Separate Configs

**Local CONFIG** (in the main local notebook):
```python
CONFIG = {
    # ── LOCAL PATHS (your Ubuntu laptop) ──────────────────────────────────────
    "data_root":         "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Data",
    "output_root":       "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Outputs",
    "preprocessed_dir":  "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Preprocessed",
    "training_dir":      "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Training",
    "logs_dir":          "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs",
    "model_path":        "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs/best_model.pth",

    # ── UPLOAD / DOWNLOAD ─────────────────────────────────────────────────────
    # Path where training data zip will be saved for uploading to Colab
    "upload_zip_path":   "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/colab_upload/training_data.zip",
    # Path where the downloaded model weights from Colab should be placed
    "colab_weights_download_dir": "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs",

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    "block_size_m":      50,
    "stride_m":          40,
    "min_pts_per_block": 200,
    "outlier_pct":       (1.0, 99.0),
    "max_points_per_block": 4096,

    # ── TRAINING (handled in Colab) ───────────────────────────────────────────
    "train_val_split":   0.80,
    "epochs":            50,
    "batch_size":        2,          # TPU-friendly small batch
    "learning_rate":     0.001,
    "early_stop_patience": 10,
    "random_seed":       42,

    # ── DTM ───────────────────────────────────────────────────────────────────
    "dtm_resolution_m":  1.0,
    "idw_power":         2.0,
    "idw_k_neighbours":  12,
    "idw_radius_factor": 5,
    "idw_chunk_size":    50000,

    # ── HYDROLOGY ────────────────────────────────────────────────────────────
    "breach_dist":       5,
    "stream_threshold":  1000,

    # ── CSF ──────────────────────────────────────────────────────────────────
    "csf_cloth_resolution": 0.5,
    "csf_class_threshold":  0.3,
    "csf_iterations":       500,

    # ── SMOOTHING ────────────────────────────────────────────────────────────
    "smoothing_sigma":   0.5,

    # ── CPU THREADS (local machine) ──────────────────────────────────────────
    "cpu_threads":       8,   # i7 8th gen has 8 logical threads
}
```

**Colab CONFIG** (inside `DTM_Colab_Training_TPU.ipynb`):
```python
# This config lives ONLY in the Colab notebook.
# Paths point to Colab's local /content/ filesystem (temporary, session only).
COLAB_CONFIG = {
    "training_dir":       "/content/training_data",   # extracted from uploaded zip
    "logs_dir":           "/content/logs",
    "model_save_path":    "/content/logs/best_model.pth",
    "history_save_path":  "/content/logs/history.json",
    "curves_save_path":   "/content/logs/training_curves.png",
    "epochs":             50,
    "batch_size":         2,
    "learning_rate":      0.001,
    "early_stop_patience":10,
    "random_seed":        42,
    "max_points_per_block": 4096,
}
```

---

### ADD New Section 4b: Package and Upload Training Data to Colab

Insert this as a new section between Section 4 (Label Generation) and Section 5 (Model Definition) in the **local notebook**.

```python
"""
SECTION 4b — Package Training Data for Colab Upload
====================================================
WHY THIS EXISTS:
  Google Colab TPU cannot access your local hard drive.
  However, the preprocessed training tiles (.npy files) are tiny —
  typically 30–80 KB each. 300 tiles = ~15 MB total.
  We zip them up and manually upload the zip to Colab.

  Raw LAS files (which can be 500 MB – 2 GB each) NEVER leave your machine.

WHAT THIS PRODUCES:
  A single zip file at CONFIG['upload_zip_path'] containing:
    training_data/
      train/
        block_00000/
          points.npy    ← normalised XYZ (model input, ~50 KB)
          labels.npy    ← binary ground/non-ground labels
        block_00001/
          ...
      val/
        ...
      split_manifest.json
"""

import zipfile
import os
from pathlib import Path

def package_training_data(config):
    """
    Zips the entire training_data/ directory into a single archive
    ready to be uploaded to Google Colab.
    
    Why zip? Colab's file upload widget works with single files.
    A zip of .npy tiles compresses well (numpy arrays are compressible).
    
    Args:
        config: global CONFIG dictionary
    
    Returns:
        Path to the created zip file and its size in MB
    """
    training_dir = Path(config["training_dir"])
    zip_path = Path(config["upload_zip_path"])
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}\n"
            "Run Section 4 (Build Training Split) first."
        )

    print(f"\n{'='*60}")
    print(f"  SECTION 4b: Packaging Training Data for Colab Upload")
    print(f"{'='*60}")
    print(f"  Source : {training_dir}")
    print(f"  Output : {zip_path}")

    # Count files before zipping
    npy_files = list(training_dir.rglob("*.npy"))
    print(f"  Files to zip: {len(npy_files)} .npy files")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(npy_files):
            # Store with relative path so Colab extracts cleanly
            arcname = file_path.relative_to(training_dir.parent)
            zf.write(file_path, arcname)
            
        # Also include the split manifest
        manifest = training_dir / "split_manifest.json"
        if manifest.exists():
            zf.write(manifest, manifest.relative_to(training_dir.parent))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n  ✅ Zip created: {zip_path}")
    print(f"  📦 Size: {size_mb:.1f} MB")
    print(f"\n{'='*60}")
    print(f"  NEXT STEP — Upload to Google Colab:")
    print(f"{'='*60}")
    print(f"""
  1. Open Google Colab in your browser:
       https://colab.research.google.com

  2. Create a new notebook OR open DTM_Colab_Training_TPU.ipynb
  
  3. Set runtime to TPU:
       Runtime → Change runtime type → Hardware accelerator → TPU
  
  4. In Colab, run this cell to upload the zip:
       from google.colab import files
       uploaded = files.upload()   # select: training_data.zip
       
       import zipfile, os
       with zipfile.ZipFile('training_data.zip', 'r') as z:
           z.extractall('/content/')
       print("Extracted:", os.listdir('/content/training_data'))

  5. Run all cells in DTM_Colab_Training_TPU.ipynb

  6. When training is done, download best_model.pth from Colab:
       from google.colab import files
       files.download('/content/logs/best_model.pth')
       files.download('/content/logs/history.json')
       files.download('/content/logs/training_curves.png')

  7. Place best_model.pth at:
       {config['model_path']}
  
  8. Return to THIS local notebook and run Section 8 (Inference) onward.
""")
    return zip_path, size_mb

# ── RUN IT ───────────────────────────────────────────────────────────────────
zip_path, size_mb = package_training_data(CONFIG)
```

---

### REPLACE Section 7: Training — Move Entirely to `DTM_Colab_Training_TPU.ipynb`

In the local notebook, **replace the training cell** with a markdown-only placeholder:

```markdown
## Section 7: Model Training (Runs in Google Colab — NOT Here)

Training is performed in Google Colab with a TPU accelerator.
This keeps the large LiDAR data on your laptop while using cloud GPU/TPU
for the computationally expensive deep learning step.

### Why not train locally?
- No GPU on your machine
- PointNet++ on CPU: ~5–8 hours for 150 blocks × 50 epochs
- PointNet++ on TPU: ~20–40 minutes for the same job

### Instructions
1. Run Section 4b above to create `training_data.zip`
2. Follow the printed upload instructions to run `DTM_Colab_Training_TPU.ipynb`
3. Download `best_model.pth` and save it to:
   `/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs/best_model.pth`
4. Continue to Section 8 (Inference) below

### Check: Is the model ready?
Run the cell below to confirm the weights file exists before proceeding.
```

```python
# Sanity check before running inference
from pathlib import Path
model_path = Path(CONFIG["model_path"])
if model_path.exists():
    size_mb = model_path.stat().st_size / 1e6
    print(f"✅ Model weights found: {model_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Modified: {model_path.stat().st_mtime}")
    print("\n▶ Ready to run Section 8 (Inference)")
else:
    print(f"❌ Model weights NOT found at: {model_path}")
    print("\n   Complete the Colab training steps first, then:")
    print(f"   Place best_model.pth at: {model_path}")
```

---

### CREATE `DTM_Colab_Training_TPU.ipynb` — The Standalone Colab Notebook

This is a **new, second notebook** that contains only what's needed for Colab TPU training. It must be completely self-contained — no references to local paths, no Google Drive, no laspy, no rasterio. Write all sections of this notebook now:

**Cell 1 — Instructions Markdown:**
Explain clearly at the top:
> This notebook is designed to run ONLY in Google Colab with a TPU runtime.
> It receives training data as a zip file uploaded from your local machine
> and produces a trained model weight file to download back.
> Data never needs to go to Google Drive.

**Cell 2 — Install dependencies:**
```python
!pip install torch torchvision tqdm numpy --quiet
```

**Cell 3 — TPU / device detection** (as shown above in Section 0-B)

**Cell 4 — Upload and extract training data:**
```python
print("📂 Step 1: Upload your training_data.zip file")
print("   This was created on your local machine by Section 4b of the local notebook.")
print()

from google.colab import files
import zipfile, os

print("Select training_data.zip when the upload dialog appears...")
uploaded = files.upload()

zip_filename = list(uploaded.keys())[0]
print(f"\n✅ Uploaded: {zip_filename} ({len(uploaded[zip_filename])/1e6:.1f} MB)")

print("\n📦 Extracting...")
with zipfile.ZipFile(zip_filename, 'r') as z:
    z.extractall('/content/')

# Verify extraction
train_blocks = len(list(Path('/content/training_data/train').glob('block_*')))
val_blocks   = len(list(Path('/content/training_data/val').glob('block_*')))
print(f"✅ Extracted successfully")
print(f"   Train blocks : {train_blocks}")
print(f"   Val blocks   : {val_blocks}")
```

**Cell 5 — COLAB_CONFIG** (as defined above)

**Cell 6 — Model definition** (copy PointNet2GroundSeg, SetAbstraction, FeaturePropagation, farthest_point_sample, index_points verbatim from the local notebook — these are pure PyTorch, no changes needed)

**Cell 7 — Dataset loader** (copy PointCloudDataset verbatim — it only uses numpy and torch, works identically on Colab)

**Cell 8 — Training loop with full TPU support:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json, time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_on_colab(colab_config, device, device_name):
    """
    Full training loop optimised for Google Colab TPU.
    
    TPU vs CPU/GPU differences handled here:
    - TPU requires xm.mark_step() after each backward pass to flush
      the XLA lazy execution graph (think of it as "commit this batch")
    - TPU requires xm.optimizer_step() instead of optimizer.step()
    - Batch sizes must stay constant — we pad the last batch if needed
    - First epoch is slow (XLA traces + compiles the graph), 
      subsequent epochs are much faster
    
    Args:
        colab_config : COLAB_CONFIG dictionary
        device       : xla device or torch.device
        device_name  : "TPU", "GPU", or "CPU" (string)
    """
    torch.manual_seed(colab_config["random_seed"])
    os.makedirs(colab_config["logs_dir"], exist_ok=True)

    # ── Datasets and loaders ───────────────────────────────────────────────
    train_ds = PointCloudDataset(
        colab_config["training_dir"] + "/train",
        augment=True,
        max_points=colab_config["max_points_per_block"]
    )
    val_ds = PointCloudDataset(
        colab_config["training_dir"] + "/val",
        augment=False,
        max_points=colab_config["max_points_per_block"]
    )

    # drop_last=True is CRITICAL for TPU — keeps batch size constant
    # which avoids expensive XLA recompilation on the last smaller batch
    train_ld = DataLoader(
        train_ds, batch_size=colab_config["batch_size"],
        shuffle=True, num_workers=2, drop_last=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=colab_config["batch_size"],
        shuffle=False, num_workers=2, drop_last=False
    )

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model = PointNet2GroundSeg(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Compute class weights to handle imbalance
    # (flat terrain = many ground; forested = few ground)
    all_labels = np.concatenate([
        np.load(str(Path(colab_config["training_dir"]) / "train" / b / "labels.npy"))
        for b in os.listdir(Path(colab_config["training_dir"]) / "train")
        if (Path(colab_config["training_dir"]) / "train" / b / "labels.npy").exists()
    ])
    n_ground     = (all_labels == 1).sum()
    n_non_ground = (all_labels == 0).sum()
    w_ground     = len(all_labels) / (2 * n_ground + 1e-9)
    w_non_ground = len(all_labels) / (2 * n_non_ground + 1e-9)
    weights = torch.tensor([w_non_ground, w_ground], dtype=torch.float32).to(device)
    print(f"  Ground points   : {n_ground:,} ({100*n_ground/len(all_labels):.1f}%)")
    print(f"  Non-ground pts  : {n_non_ground:,}")
    print(f"  Class weights   : non-ground={w_non_ground:.2f}, ground={w_ground:.2f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=colab_config["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=colab_config["epochs"],
        eta_min=1e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, colab_config["epochs"] + 1):
        t0 = time.time()

        # — Train phase —
        model.train()
        train_loss = train_ok = train_n = 0
        for pts, labels in tqdm(train_ld, desc=f"Ep {epoch:02d} Train", leave=False):
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pts)                          # [B, N, 2]
            B, N, C = logits.shape
            loss = criterion(logits.view(B*N, C), labels.view(B*N))
            loss.backward()

            # TPU: flush the XLA lazy graph after backward
            if device_name == "TPU":
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer)             # TPU-aware step
                xm.mark_step()                           # flush XLA ops
            else:
                optimizer.step()

            preds = logits.argmax(-1)
            train_loss += loss.item() * B
            train_ok   += (preds == labels).sum().item()
            train_n    += B * N

        # — Validation phase —
        model.eval()
        val_loss = val_ok = val_n = 0
        val_tp = val_fp = val_fn = 0
        with torch.no_grad():
            for pts, labels in tqdm(val_ld, desc=f"Ep {epoch:02d} Val  ", leave=False):
                pts, labels = pts.to(device), labels.to(device)
                logits = model(pts)
                B, N, C = logits.shape
                loss = criterion(logits.view(B*N, C), labels.view(B*N))
                preds = logits.argmax(-1)
                val_loss += loss.item() * B
                val_ok   += (preds == labels).sum().item()
                val_n    += B * N
                # Ground class metrics (class=1)
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                
                if device_name == "TPU":
                    xm.mark_step()

        t_loss = train_loss / len(train_ds)
        t_acc  = train_ok  / train_n
        v_loss = val_loss  / len(val_ds)
        v_acc  = val_ok    / val_n
        precision = val_tp / (val_tp + val_fp + 1e-9)
        recall    = val_tp / (val_tp + val_fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        elapsed   = time.time() - t0

        print(f"Ep {epoch:3d}/{colab_config['epochs']} | "
              f"T-loss {t_loss:.4f} T-acc {t_acc:.4f} | "
              f"V-loss {v_loss:.4f} V-acc {v_acc:.4f} | "
              f"P {precision:.3f} R {recall:.3f} F1 {f1:.3f} | "
              f"{elapsed:.0f}s")

        scheduler.step()
        history.append({"ep": epoch, "tl": t_loss, "ta": t_acc,
                         "vl": v_loss, "va": v_acc, "f1": f1})

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save(model.state_dict(), colab_config["model_save_path"])
            print(f"  ** Best model saved (val_acc={best_val_acc:.4f}) **")
        else:
            patience_counter += 1
            if patience_counter >= colab_config["early_stop_patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience_counter} epochs)")
                break

    # ── Save artifacts ────────────────────────────────────────────────────
    with open(colab_config["history_save_path"], "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    eps = [h["ep"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, [h["tl"] for h in history], label="Train Loss")
    ax1.plot(eps, [h["vl"] for h in history], label="Val Loss")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(True)
    ax2.plot(eps, [h["ta"] for h in history], label="Train Acc")
    ax2.plot(eps, [h["va"] for h in history], label="Val Acc")
    ax2.plot(eps, [h["f1"] for h in history], label="Val F1 (Ground)", linestyle="--")
    ax2.set(title="Accuracy & F1", xlabel="Epoch", ylabel="Score")
    ax2.legend(); ax2.grid(True)
    plt.suptitle("DTM PointNet++ Training — Google Colab TPU", fontsize=12)
    plt.tight_layout()
    plt.savefig(colab_config["curves_save_path"], dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"   Model: {colab_config['model_save_path']}")
    return best_val_acc

# ── TRAIN ──────────────────────────────────────────────────────────────────
best_acc = train_on_colab(COLAB_CONFIG, device, DEVICE_NAME)
```

**Cell 9 — Download model weights back to local machine:**
```python
"""
STEP: Download Trained Model to Your Local Machine
===================================================
After this cell runs, your browser will automatically download:
  1. best_model.pth      ← the trained weights (most important)
  2. history.json        ← training loss/accuracy per epoch
  3. training_curves.png ← plot of training progress

Save best_model.pth to:
  /home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs/best_model.pth

Then return to your LOCAL notebook (DTM_Generator_TPU.ipynb) 
and run Section 8 (Inference) onward.
"""

from google.colab import files
import os

print("📥 Downloading trained model files...")
print("   Your browser will download each file in sequence.\n")

for filename in [
    COLAB_CONFIG["model_save_path"],
    COLAB_CONFIG["history_save_path"],
    COLAB_CONFIG["curves_save_path"],
]:
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / 1e6
        print(f"  ⬇  {filename} ({size_mb:.2f} MB)")
        files.download(filename)
    else:
        print(f"  ❌ Not found: {filename}")

print("""
✅ Downloads complete!

NEXT STEPS on your LOCAL machine:
──────────────────────────────────
1. Move best_model.pth to:
     /home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs/best_model.pth

2. Open DTM_Generator_TPU.ipynb in VS Code (local Jupyter kernel)

3. Run the sanity-check cell in Section 7 to confirm weights exist

4. Run Section 8 (Inference) → Section 9 (DTM) → Section 10 (Hydrology) → ...
""")
```

---

### MODIFY Section 2: Data Discovery — Update Paths

Replace the `base_dir` default in `discover_las_files()` to use:
```python
BASE_DIR = Path("/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Data")
```

Make sure `discover_las_files()` uses `pathlib.Path` throughout (not `os.path.join`) because the path contains spaces (`AIR Docs`, `GSI 2026 `). Add this critical note:

```python
# IMPORTANT: Path has spaces ("AIR Docs", "GSI 2026 ")
# pathlib.Path handles spaces automatically without escaping
# Do NOT use string concatenation with + for these paths
data_root = Path(config["data_root"])  # ✅ correct
# data_root = config["data_root"] + "/Gujrat"  # ❌ error-prone
```

---

### MODIFY Section 8: Inference — Set CPU Threads for Local Machine

At the start of the inference function, add:
```python
import torch
torch.set_num_threads(CONFIG["cpu_threads"])  # use all 8 logical cores on i7 8th gen
print(f"Using {CONFIG['cpu_threads']} CPU threads for inference")
```

Also add a model-not-found guard with a clear error message:
```python
model_path = Path(config["model_path"])
if not model_path.exists():
    raise FileNotFoundError(
        f"\n❌ Model weights not found at: {model_path}\n\n"
        "You need to:\n"
        "  1. Run the Colab training notebook (DTM_Colab_Training_TPU.ipynb)\n"
        "  2. Download best_model.pth from Colab\n"
        f"  3. Place it at: {model_path}\n"
        "  4. Re-run this cell"
    )
```

---

### ADD Section 0-C: Workflow Overview Diagram

Add this as the very first markdown cell in the local notebook so the user always sees the big picture:

```markdown
## 🗺️ Overall Workflow: Two-Machine Architecture

```
YOUR UBUNTU LAPTOP (local Jupyter in VS Code)
┌──────────────────────────────────────────────────────────────────┐
│  RAW DATA:  /home/jaideepchouhan/Documents/AIR Docs/GSI 2026/   │
│             ├── Gujrat_Point_Cloud/*.las                         │
│             ├── Punjab_Point_Cloud/*.las/.laz                    │
│             ├── Rajasthan_Point_Cloud/*.LAS                      │
│             ├── Tamil_Nadu_Point_Cloud/**/*.las                  │
│             └── Andaman_and_Nicobar_Islands_*/**/*.laz           │
│                                                                  │
│  Section 2 → Explore LAS files                                   │
│  Section 3 → Preprocess: LAS → 50m tiles → .npy blocks          │
│  Section 4 → Generate CSF labels (ground/non-ground)            │
│  Section 4b→ Zip training_data/ (10–30 MB) ─────────────────┐   │
└─────────────────────────────────────────────────────────────────┘
                                                              │ UPLOAD ZIP
                                                              ▼
GOOGLE COLAB (cloud, TPU runtime, browser or VS Code remote)
┌──────────────────────────────────────────────────────────────────┐
│  DTM_Colab_Training_TPU.ipynb                                    │
│                                                                  │
│  Upload training_data.zip → extract → /content/training_data/   │
│  Train PointNet++ on TPU (~30 min for 50 epochs)                 │
│  Download: best_model.pth, history.json, training_curves.png     │
└──────────────────────────────────────────────────────────────────┘
                                                              │ DOWNLOAD WEIGHTS
                                                              ▼
YOUR UBUNTU LAPTOP (continue in local notebook)
┌──────────────────────────────────────────────────────────────────┐
│  Place best_model.pth in Logs/                                   │
│  Section 8 → Inference: classify all villages                    │
│  Section 9 → IDW interpolation → raw DTM raster                  │
│  Section 10→ Hydrological conditioning + stream extraction       │
│  Section 11→ Export Cloud-Optimised GeoTIFF (COG)                │
│  Section 12→ Validate accuracy (RMSE, MAE, bias)                 │
│  Section 13→ Visualise in matplotlib / open in QGIS              │
│                                                                  │
│  OUTPUTS: /home/jaideepchouhan/Documents/AIR Docs/GSI 2026/      │
│           Outputs/{village_name}_COG.tif  ← final deliverable   │
└──────────────────────────────────────────────────────────────────┘
```

Key insight: Only the 10–30 MB of preprocessed .npy training tiles ever 
leave your machine. The raw LAS files (500 MB – 2 GB each) stay local.
The trained model (< 5 MB) comes back to your machine.
```

---

## SUMMARY OF ALL CHANGES

Please implement exactly these changes and no others:

| Change | Location | What |
|--------|----------|------|
| Split into 2 notebooks | New | Create `DTM_Colab_Training_TPU.ipynb` |
| Section 0 | Local notebook | Remove Drive mount; add local pip installs |
| Section 0 | Colab notebook | TPU setup, pip installs only |
| Section 0-C | Local notebook | Add workflow diagram markdown cell at top |
| Section 1 | Local notebook | Replace CONFIG with local Ubuntu paths |
| Section 1 | Colab notebook | Add COLAB_CONFIG with /content/ paths |
| Section 2 | Local notebook | Use Ubuntu path; use pathlib.Path throughout |
| Section 4b | Local notebook | NEW — zip and upload instructions |
| Section 7 | Local notebook | Replace with placeholder + sanity check cell |
| Section 7 | Colab notebook | Full training loop with TPU xm.mark_step() |
| Colab Cell 9 | Colab notebook | Download weights back to local machine |
| Section 8 | Local notebook | Add torch.set_num_threads(8); add model guard |

Do not change Sections 3, 4, 5 (model definition), 6 (dataset loader), 
9, 10, 11, 12, 13, 14, 15 — these are already correct from the original notebook.

Output both files completely:
1. The modified `DTM_Generator_TPU.ipynb` (local notebook — all sections)
2. The new `DTM_Colab_Training_TPU.ipynb` (Colab notebook — 9 cells)

## ═══════════════════════════════════════════════════════════════
## PROMPT END — STOP COPYING HERE
## ═══════════════════════════════════════════════════════════════

---

## 📌 QUICK REFERENCE: YOUR RUN ORDER

### On your Ubuntu laptop (VS Code → local Jupyter kernel):
```bash
# Activate your Python environment first
cd "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026"
jupyter lab   # or open the .ipynb in VS Code directly

# Run in order: Sections 0-A, 1, 2, 3, 4, 4b
# Stop here — go to Colab
```

### In Google Colab (browser):
```
1. Go to colab.research.google.com
2. Upload DTM_Colab_Training_TPU.ipynb
3. Runtime → Change runtime type → TPU
4. Run all cells top to bottom
5. Download best_model.pth when prompted
```

### Back on your Ubuntu laptop:
```bash
# Copy best_model.pth to Logs/
cp ~/Downloads/best_model.pth \
   "/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Logs/"

# Continue in VS Code Jupyter: Sections 8, 9, 10, 11, 12, 13
```

## ⚠️ KEY THINGS TO REMEMBER

- The path `/home/jaideepchouhan/Documents/AIR Docs/GSI 2026/Data` has **spaces** — always use `pathlib.Path`, never string concatenation
- `drop_last=True` in the Colab DataLoader is **mandatory** for TPU — variable batch sizes cause XLA recompilation
- The Colab session is **temporary** — if it disconnects during training, the zip is still on Colab's `/content/` but you'd need to re-upload. To be safe, Colab also auto-saves the model after every improved epoch.
- WhiteboxTools is **local only** — do not try to install it in Colab (it's not needed there)

---
*Modification prompt for: MoPR Geospatial Intelligence Hackathon 2026*
*Architecture: Ubuntu 24.04 Local ↔ Google Colab TPU Hybrid*
