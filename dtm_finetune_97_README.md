# DTM Ground Classification — Fine-Tuning Technical Reference

**Model:** DTMPointNet2 (PointNet++ variant) · **Task:** Binary point-level ground/non-ground classification from LiDAR tiles · **Hardware:** NVIDIA T4 × 2 (15.6 GB VRAM each) · **Framework:** PyTorch 2.x + DataParallel

---

## Table of Contents

1. [Problem Statement & Geospatial Context](#1-problem-statement--geospatial-context)
2. [Architecture: DTMPointNet2](#2-architecture-dtmpointnet2)
3. [Feature Engineering Pipeline](#3-feature-engineering-pipeline)
4. [Label Strategy: Consensus Labels & Pseudo-Labels](#4-label-strategy-consensus-labels--pseudo-labels)
5. [Training Data: Tiles, Splits, and the Sampling Problem](#5-training-data-tiles-splits-and-the-sampling-problem)
6. [Augmentation Strategy](#6-augmentation-strategy)
7. [Loss Function: Focal Loss with Label Smoothing](#7-loss-function-focal-loss-with-label-smoothing)
8. [Fine-Tuning Phase A — Full Dataset + Iterative Label Refresh](#8-fine-tuning-phase-a--full-dataset--iterative-label-refresh)
9. [Fine-Tuning Phase B — Multi-Round Hard Example Mining](#9-fine-tuning-phase-b--multi-round-hard-example-mining)
10. [Fine-Tuning Phase C — Stochastic Weight Averaging](#10-fine-tuning-phase-c--stochastic-weight-averaging)
11. [Test-Time Augmentation: Geometric TTA](#11-test-time-augmentation-geometric-tta)
12. [Threshold Optimization](#12-threshold-optimization)
13. [OOM Management on T4 × 2](#13-oom-management-on-t4--2)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Round-by-Round Accuracy History](#15-round-by-round-accuracy-history)
16. [Why Each Technique Was Necessary](#16-why-each-technique-was-necessary)

---

## 1. Problem Statement & Geospatial Context

### What is a DTM?

A **Digital Terrain Model (DTM)** represents the bare-earth surface — the ground elevation without buildings, vegetation, or any above-ground objects. It is distinct from a **DSM (Digital Surface Model)** which includes everything the sensor sees.

LiDAR (Light Detection and Ranging) sensors emit laser pulses that return from whatever surface they hit first: rooftops, tree canopy, power lines, or bare soil. The raw output is a **point cloud** — a set of `(X, Y, Z)` triplets in geographic coordinates, sometimes with additional attributes like intensity and return number.

To compute a DTM, each point must be classified as either:
- **Ground (1)** — the laser hit bare earth
- **Non-ground (0)** — the laser hit a building, tree, vehicle, haystack, etc.

This is a **per-point binary classification** problem on 3D spatial data.

### Why It Is Hard

Classic algorithmic approaches (Progressive Morphological Filtering, Cloth Simulation Filter, Multiscale Curvature Classification) struggle on:

- **Dense urban environments** — buildings adjacent to streets confuse local surface estimation
- **Steep slopes** — elevation thresholds tuned for flat terrain misfire on hillsides
- **Low vegetation** — kuccha (earthen) walls, dense shrubs, and low hedges sit within centimetres of bare ground
- **Rural South Asia specifically** — haystacks, threshing floors, and elevated field boundaries resemble micro-terrain
- **Point density variation** — aerial LiDAR tiles vary from 2 pts/m² to 50+ pts/m²

The model being fine-tuned here was pre-trained to ~93.65% tile-weighted accuracy. The task of this notebook is to push it toward 98%, which requires squeezing signal from the model's remaining error budget — the hardest, most ambiguous tiles.

### Data Layout

```
/kaggle/input/training-data-95pct/
    train/
        tile_000001/
            points.npy        # (N, 3) float32: X, Y, Z in local coords
            labels.npy        # (N,)   int8:    0=non-ground, 1=ground
        tile_000002/
            ...
    val/
        tile_000001/
            ...
```

Each tile contains between ~500 and ~20,000 LiDAR points covering a geographic patch of roughly 50m × 50m to 200m × 200m depending on flight density. The train/val split is fixed (4,922 train tiles, 1,216 val tiles).

---

## 2. Architecture: DTMPointNet2

### PointNet++ Background

Standard convolutional networks require a regular grid (images, voxels). Point clouds are **unordered, irregular, and variable-density** — they are sets, not grids. PointNet++ (Qi et al., 2017) operates directly on point sets by:

1. Sampling a subset of "centroid" points using **Farthest Point Sampling (FPS)**
2. Grouping nearby points around each centroid using a radius ball query
3. Applying shared MLPs to the local groups to extract local features
4. Repeating hierarchically to build a multi-scale feature hierarchy

The DTMPointNet2 model follows this paradigm with a **Set Abstraction → Feature Propagation** encoder-decoder structure for per-point classification.

### Architecture Diagram

```
Input: (B, 4096, 9)          # B tiles, 4096 points each, 9 features per point
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  SA1: MSG  512 centroids                │   Radii: [0.5m, 1.5m]
  │            [32, 64] neighbours each     │   Output: (B, 512, 192)
  │            MLPs: [64,64] and [64,128]   │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  SA2: MSG  128 centroids                │   Radii: [3.0m, 8.0m]
  │            [64, 128] neighbours each    │   Output: (B, 128, 448)
  │            MLPs: [128,192],[128,256]    │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  SA3: Single-scale  radius=15.0m        │   k=128 neighbours
  │                     MLP: [256,512]      │   Output: (B, ~32, 512)
  └─────────────────────────────────────────┘
        │  (hierarchical skip connections)
        ▼
  ┌─────────────────────────────────────────┐
  │  FP3: Feature Propagation               │   512+448 → [256,256]
  │       Interpolates SA3→SA2 via kNN      │   Output: (B, 128, 256)
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  FP2: Feature Propagation               │   256+192 → [256,128]
  │       Interpolates SA2→SA1              │   Output: (B, 512, 128)
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  FP1: Feature Propagation               │   128+9  → [128,128,64]
  │       Interpolates SA1→original points  │   Output: (B, 4096, 64)
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  Head: Conv1d(64→64) + BN + ReLU        │
  │        Dropout(0.4)                     │
  │        Conv1d(64→2)                     │
  └─────────────────────────────────────────┘
        │
        ▼
Output: (B, 4096, 2)         # Logits for [non-ground, ground] per point
```

### Key Algorithmic Components

#### Farthest Point Sampling (FPS)

FPS deterministically selects `n` points that maximally cover the spatial extent of the input. Starting from a random seed point, each subsequent point is chosen as the one maximally distant from the already-selected set:

```python
def _fps(xyz, n):
    # xyz: (B, N, 3)
    d = torch.full((B, N), 1e10)    # distance-to-set, init=infinity
    far = torch.randint(0, N, (B,)) # random seed
    for i in range(n):
        centroids[:, i] = far
        cc = xyz[bi, far].unsqueeze(1)          # current centroid
        dd = ((xyz - cc) ** 2).sum(-1)           # distance to centroid
        d = torch.where(dd < d, dd, d)           # update min-distances
        far = d.argmax(-1)                       # farthest unselected point
    return centroids
```

FPS ensures that even in sparse regions, representative points are selected, unlike random sampling which over-samples dense regions. This matters for LiDAR where flight-line overlap creates uneven density.

#### Ball Query Grouping

For each centroid, all points within radius `r` are collected up to `k` neighbours:

```python
def _ball(nxyz, xyz, r, k):
    d = torch.cdist(nxyz, xyz)              # (B, n_centroids, N)
    return torch.where(d <= r, d, 1e10) \
               .topk(k, dim=-1, largest=False).indices
```

Ball query (fixed radius) is preferred over kNN for spatial consistency: features extracted at radius `r` correspond to the same physical scale regardless of local point density.

#### Multi-Scale Grouping (MSG)

At SA1 and SA2, the model queries **two radii simultaneously** and concatenates the resulting features. This gives the network multi-scale context at each centroid — small radii capture fine detail (individual points, edges), large radii capture context (building footprints, canopy extent):

```
SA1 small radius 0.5m  →  captures individual points, micro-texture
SA1 large radius 1.5m  →  captures small objects (kerb stones, small walls)
SA2 small radius 3.0m  →  captures building edges, road cross-sections
SA2 large radius 8.0m  →  captures building footprints, large trees
```

#### Feature Propagation (Interpolation)

The decoder must propagate features back from subsampled centroids to all original points. This is done by inverse-distance-weighted interpolation from the 3 nearest centroids:

```python
def forward(self, xyz1, xyz2, f1, f2):
    # xyz1: dense (original), xyz2: sparse (subsampled)
    # f1: dense skip features, f2: sparse features to upsample
    d = torch.cdist(xyz1, xyz2)
    t3 = d.topk(3, dim=-1, largest=False)        # 3 nearest centroids
    w = 1. / (t3.values + 1e-8)                  # inverse-distance weights
    w = w / w.sum(-1, keepdim=True)              # normalise to sum=1
    interp = (_idx(f2, t3.indices) * w.unsqueeze(-1)).sum(2)
    feat = torch.cat([f1, interp], -1)           # skip connection
    return self.mlp(feat)
```

The skip connections from the encoder (f1) are concatenated with the interpolated features (interp), preserving fine spatial detail that would otherwise be lost in the subsampling bottleneck.

---

## 3. Feature Engineering Pipeline

The raw point cloud `(N, 3)` is augmented with 6 hand-crafted geospatial features, giving a final per-point descriptor of dimension 9.

### The `_grid` Function

The core primitive is a fast 2D binning operation. Points are projected onto a regular grid of cell size `cell_m` metres, and per-cell statistics are computed:

```python
def _grid(xyz, cell_m, gmax=64):
    # Compute grid indices
    GW = clip(x_range / cell_m, 8, gmax)   # grid width in cells
    GH = clip(y_range / cell_m, 8, gmax)   # grid height in cells
    ci = yi * GW + xi                        # flat cell index per point

    # Aggregate per-cell statistics using scatter operations
    np.minimum.at(c_min, ci, z)    # minimum Z per cell  (ground proxy)
    np.add.at(c_sum, ci, z)        # sum Z               (for mean)
    np.add.at(c_sq,  ci, z*z)      # sum Z²              (for std dev)
    np.add.at(c_cnt, ci, 1.)       # point count         (density proxy)
```

This is an O(N) operation regardless of tile size, making it tractable for 20,000-point tiles.

### The 6 Derived Features

| Feature | Formula | Geospatial Meaning |
|---|---|---|
| `dZ_2m` | `max(0, z − cell_min_z)` at 2m grid | Height above lowest point in 2m cell. Ground points → 0, buildings/trees → large positive |
| `roughness` | `std(z)` within 2m cell | Variability of elevation in local cell. Ground → low, canopy/facades → high |
| `slope` | Max absolute diff between adjacent cell means | Terrain gradient. Flat ground → 0, cliff edges → large |
| `density` | `count_in_cell / max_count` | Relative point density. Dense swaths → 1, sparse edges → small |
| `dZ_8m` | `max(0, z − cell_min_z)` at 8m grid | Coarser-scale height above ground. Captures tall buildings missed at 2m scale |
| `planarity` | `std(z) / (z_range + ε)` at 2m | Ratio of spread to range. Planar surfaces → 0, jagged canopy → high |

All 6 features are **z-score normalised per tile** before concatenation with the normalised XYZ coordinates.

### Why These Features?

Ground points in LiDAR share a specific signature: they are the **lowest returners** in any local neighbourhood, they form **planar surfaces**, they have **low roughness**, and they cluster at **predictable density** based on flight path geometry. These 6 features encode exactly this signal in a way that generalises across tile types, sensors, and terrains.

The normalised XYZ coordinates (3 features) give the network the raw geometry. The 6 engineered features give it pre-computed spatial context that would take many network layers to re-derive from scratch.

---

## 4. Label Strategy: Consensus Labels & Pseudo-Labels

This pipeline uses three tiers of labels with different purity/coverage tradeoffs:

### Tier 1: Original Labels

The ground-truth labels from the dataset, labelled by domain experts or classical algorithms at ~95% accuracy. Used as the fallback baseline.

### Tier 2: Consensus Labels

A **geometry-only algorithm** that re-labels each point using a multi-scale height-threshold vote. At each of 4 spatial scales (1m, 2m, 4m, 8m cells with thresholds 0.20m, 0.35m, 0.60m, 1.00m), a point is voted ground if it is within the threshold of the cell minimum:

```python
SCALES = [(1.0, 0.20), (2.0, 0.35), (4.0, 0.60), (8.0, 1.00)]

for cell_m, threshold in SCALES:
    c_min = per_cell_minimum_z(xyz, cell_m)
    above_ground = z - c_min[cell_index]
    votes += (above_ground <= threshold).astype(int)

consensus_ground = (votes >= 3)   # majority vote: ground if 3 of 4 scales agree
```

Consensus labels are **model-free** — they use no neural network. They are less accurate than the model's predictions on easy tiles, but more reliable than the model's predictions on confusing tiles where the network may hallucinate.

Consensus labels serve as the **initial label source** for validation and as the **fallback** when pseudo-label confidence is too low.

### Tier 3: Pseudo-Labels

Pseudo-labels are generated by running the current trained model on unlabelled or weakly-labelled training tiles and using its predictions as training targets for subsequent rounds. This is a form of **self-training** or **semi-supervised learning**.

The key parameter is the **confidence threshold**: a point's pseudo-label is only accepted if the model's predicted probability deviates sufficiently from 0.5 (i.e., the model is confident):

```
confidence = |P(ground) − 0.5| × 2      (0 = maximally uncertain, 1 = maximally confident)

if confidence >= conf_thresh:
    use model prediction as label
else:
    keep original consensus label
```

| Round | `conf_thresh` | Coverage | Purity | Notes |
|---|---|---|---|---|
| Round 2 | 0.92 | ~8% replaced | Very high | Early round, conservative |
| Round 3 | 0.95 | 2.7% replaced | Extremely high | Too conservative — missed hard cases |
| Round 4 initial | 0.92 | ~8–12% expected | Very high | Wider net than R3 |
| Round 4 refresh | 0.88 | ~15–20% expected | High | Mid-training refreshes use lower threshold |

### Multi-Subsample Averaging for Pseudo-Labels (Round 4 Innovation)

A single model inference on a 4096-point subsample is noisy — different subsamples of the same tile produce slightly different probability estimates. Round 4 runs **3 independent subsamples per tile** and averages the resulting probabilities before thresholding:

```python
N_SUBSAMPLE = 3
avg_probs = np.zeros(N)

for pass_idx in range(N_SUBSAMPLE):
    subsample = random_subsample(xyz, max_pts=4096, seed=tile_seed + pass_idx)
    probs = model_inference(subsample)          # (4096,) probabilities
    avg_probs += map_back_to_full_tile(probs)   # interpolate to all N points

avg_probs /= N_SUBSAMPLE   # stable estimate
```

This reduces pseudo-label variance, particularly for points near tile edges where subsample inclusion is non-deterministic.

---

## 5. Training Data: Tiles, Splits, and the Sampling Problem

### The Tile Loop Problem (Root Cause of Round 3 Plateau)

In Round 3, pseudo-labels were generated **once** before training and cached. Every training epoch read from the same label files. This created a degenerate loop:

```
Epoch 1:  Model sees tile_000761 → predicts badly → gets gradient from cached label
Epoch 5:  Model sees tile_000761 → predicts slightly better → same cached label
Epoch 15: Model sees tile_000761 → has memorised the label → no more gradient signal
```

The model stops learning the **geometry** of hard tiles and starts memorising their **label patterns**. Since the label patterns are fixed, training loss decreases but generalisation stops improving.

Evidence: Phase A training accuracy rose from 85.87% → 87.65% over 15 epochs (legitimate learning), but validation accuracy oscillated between 91.21% and 91.95% with no trend — the classic sign of label memorisation rather than representation learning.

### Round 4 Fix: Periodic Label Refresh

Every 5 epochs, the training pseudo-labels are **wiped and regenerated** using the current model state:

```
Epoch 1–5:   Train on Round 4 initial labels (conf=0.92)
  → Refresh: generate Round4_refresh_ep6  (conf=0.88, model improved)
Epoch 6–10:  Train on fresher, higher-coverage labels
  → Refresh: generate Round4_refresh_ep11 (conf=0.88, model improved again)
Epoch 11–15: Train on labels from a stronger model
  → Refresh: generate Round4_refresh_ep16
...
```

Each refresh captures the model's improved understanding. Points the model was uncertain about at epoch 1 (omitted from pseudo-labels) may now be confidently labelled at epoch 6, providing new gradient signal on previously unlabelled points.

The refresh interval of 5 epochs is a balance between:
- **Too frequent** (e.g., every epoch): pseudo-label generation itself takes ~9 minutes; refreshing every epoch would dominate wall time
- **Too infrequent** (e.g., every 15 epochs, i.e., never): the stale label problem persists

### Balanced Sampling

Training tiles have heavily imbalanced class ratios — a tile that is 95% ground provides almost no gradient signal for the non-ground class. The `WeightedRandomSampler` assigns each tile a sampling weight inversely proportional to its minority class fraction:

```python
for tile in ds._tiles:
    labels = load_labels(tile)
    ground_fraction = labels.mean()
    non_ground_fraction = 1 - ground_fraction
    # Weight = 1 / (minority fraction + 0.05)
    # +0.05 prevents extreme weights on nearly pure tiles
    weight = 1. / (min(ground_fraction, non_ground_fraction) + 0.05)
```

A tile that is 50% ground / 50% non-ground gets weight ~10. A tile that is 99% ground / 1% non-ground also gets weight ~10 (capped by the +0.05 floor). A tile that is 80/20 gets weight ~4. This encourages the sampler to prefer balanced tiles where both classes provide gradient signal.

---

## 6. Augmentation Strategy

### Training Augmentations

Applied in `GroundDataset.__getitem__` when `augment=True`. Augmentations simulate the physical variability of real LiDAR acquisitions:

| Augmentation | Implementation | Physical Justification |
|---|---|---|
| **Random XY rotation** | `R = rotation_matrix(θ)` where `θ ~ U(0, 2π)` | Tiles are oriented arbitrarily relative to flight direction; the model must be rotation-invariant |
| **Height jitter** | `z += N(0, 0.05)` per point | Sensor noise ±5cm is typical for aerial LiDAR at standard altitudes |
| **Global scale** | `xyz *= U(0.90, 1.10)` | Simulates different flight altitudes (higher altitude → smaller apparent objects) |
| **XY reflection** | `x *= ±1`, `y *= ±1` with p=0.5 each | Left-right and front-back symmetry; terrain has no preferred orientation |
| **Anisotropic XY scale** | `x *= U(0.85, 1.15)`, `y *= U(0.85, 1.15)` independently | Simulates slightly elliptical scan patterns from different sensor models |
| **Z scale** | `z *= U(0.85, 1.15)` | Simulates elevation exaggeration/compression |
| **Random point drop** | Keep points where `rand > 0.15` | 15% drop simulates lower-density acquisitions or masked-return scenarios |

Round 4 strengthened several of these vs Round 3:
- Height jitter: ±0.02m → ±0.05m (more realistic sensor noise range)
- Point drop: 10% → 15% (covers more density variation)
- Anisotropic XY scale: now always applied (was conditional)

### Point Mixup Augmentation

Point Mixup is a domain adaptation of Mixup (Zhang et al., 2018) for point clouds. Two tiles A and B are linearly interpolated:

```python
λ ~ Beta(α, α)                         # mixing coefficient

pts_mixed = λ × pts_A + (1−λ) × pts_B
lbl_mixed = λ × lbl_A + (1−λ) × lbl_B
lbl_hard  = (lbl_mixed >= 0.5).astype(int)   # threshold to hard labels
```

The geometric interpretation: the mixed tile has points that are physically between the two tiles' point configurations. The model must learn decision boundaries that are robust to this interpolation, which forces smoother, more general feature representations.

**Alpha parameter choice:**

`Beta(α=0.2)` vs `Beta(α=0.4)` produces fundamentally different mixing behaviour:

```
Beta(0.4): P(λ < 0.3) ≈ 15%,  P(0.3 < λ < 0.7) ≈ 50%,  P(λ > 0.7) ≈ 35%
           → Many near-50/50 blends → many synthetic tiles that look nothing like real tiles

Beta(0.2): P(λ < 0.3) ≈ 5%,   P(0.3 < λ < 0.7) ≈ 10%,  P(λ > 0.7) ≈ 85%
           → Most tiles are 85%+ one tile → synthetic tiles that look like real tiles
             with small contamination from another
```

Round 4 uses α=0.2: the model learns boundaries between realistic-looking inputs rather than hallucinated 50/50 interpolations that it will never see at inference time.

Mixup is applied with probability 0.5 per sample and **only during Phase A** (Phase B HEM trains without it, to focus gradient purely on the hard examples).

---

## 7. Loss Function: Focal Loss with Label Smoothing

### Focal Loss

Standard cross-entropy treats every misclassified point equally. For a model that already classifies 93% of points correctly, the remaining 7% of hard points contribute proportionally little to the total loss — they are drowned out by the 93% easy points.

Focal Loss (Lin et al., 2017) re-weights each sample by a factor `(1 − p_t)^γ` where `p_t` is the model's probability assigned to the correct class:

```
FL(p_t) = −α_t × (1 − p_t)^γ × log(p_t)

where:
  p_t = P(correct class)
  (1 − p_t)^γ = focusing factor — downweights easy examples
  α_t = class balance weight (α for positive/ground, 1−α for negative)
  γ = focusing exponent (higher = more focus on hard examples)
```

For a correctly classified easy point with `p_t = 0.98`:
- Standard CE contribution: `−log(0.98) ≈ 0.020`
- Focal (γ=2.5) contribution: `(1−0.98)^2.5 × 0.020 = 0.02^2.5 × 0.020 ≈ 0.000057`

For a hard misclassified point with `p_t = 0.52`:
- Standard CE contribution: `−log(0.52) ≈ 0.654`
- Focal (γ=2.5) contribution: `(1−0.52)^2.5 × 0.654 = 0.48^2.5 × 0.654 ≈ 0.160`

The focusing exponent `γ=2.5` shrinks easy-example contributions by a factor of ~350 while barely affecting the hard examples. The effective batch is dominated by the hard cases the model is genuinely struggling with.

**Parameters used:**
- `focal_alpha = 0.75` — upweights ground class (which is typically the minority class in urban tiles)
- `focal_gamma = 2.5` — aggressive focusing (standard default is 2.0)
- Applied pointwise, averaged over the full batch

### Label Smoothing

Label smoothing prevents the model from becoming overconfident by replacing hard one-hot targets with soft targets:

```
y_smooth = y_hard × (1 − ε) + ε / num_classes

For ε=0.02, num_classes=2:
  Ground (1) → 0.98 × 1 + 0.01 = 0.99
  Non-ground (0) → 0.98 × 0 + 0.01 = 0.01
```

This is implemented inside the focal loss computation before applying the CE term. The effect: the model can never achieve zero loss by pushing logits to ±∞, which regularises the final-layer weights and keeps the decision boundary soft at inference time.

Round 4 reduced smoothing from ε=0.03 → ε=0.02: at 93.97% accuracy the model is already well-calibrated, and excessive smoothing would prevent it from becoming confident enough on the genuinely unambiguous easy points.

---

## 8. Fine-Tuning Phase A — Full Dataset + Iterative Label Refresh

### Setup

- **Epochs:** 20
- **Learning rate:** 2e-4 (cosine annealed to 2e-6)
- **Dataset:** All 4,922 training tiles, pseudo-labels, balanced sampler, mixup=True
- **Label refresh:** Every 5 epochs (epochs 6, 11, 16)
- **Validation:** Every 2 epochs with fast non-TTA sweep

### Learning Rate Schedule

Cosine annealing decays the learning rate from `lr_max` to `lr_min = lr_max × 0.01` following a cosine curve:

```
lr(t) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t / T_max))
```

This is appropriate for fine-tuning because: (1) the model starts from a good initialisation and large early steps would destabilise trained weights; (2) the cosine shape naturally slows learning as the model approaches convergence rather than applying arbitrary step drops.

### Why 2e-4 Instead of 3e-4 (Round 3)?

The starting accuracy is 93.97% not 93.65%. The loss landscape is flatter near the optimum — larger steps are more likely to overshoot out of the basin. The 2/3 reduction in LR keeps the model exploring near its current solution rather than bouncing.

### Refresh Implementation Detail

Before each refresh, the training DataLoader is explicitly deleted to free its prefetch worker buffers from GPU/CPU memory, then pseudo-labels are wiped and regenerated, then a new DataLoader is constructed:

```python
# Free loader workers before running model inference for refresh
del tr_ld_A
gc.collect()
torch.cuda.empty_cache()

# Regenerate labels with current model state
cur_pseudo = generate_pseudolabels(model, CFG,
    conf_thresh=0.88, round_name=f'Round4_refresh_ep{ep}', force=True)

# New loader picks up fresh label files
tr_ld_A = make_loader(CFG, 'train', pseudo_dir=cur_pseudo, ...)
```

---

## 9. Fine-Tuning Phase B — Multi-Round Hard Example Mining

### The Core Idea

Hard Example Mining (HEM) is the observation that a model at 93.97% accuracy has already saturated its learning on the easy 93.97% of examples. Training these again provides almost no useful gradient. The remaining 6.03% of errors are concentrated in specific tiles with specific difficulty factors (kuccha walls, dense low canopy, haystacks).

HEM identifies the hardest tiles, discards the easy ones, and trains exclusively on the hard subset.

### Single-Round vs Multi-Round: Why Round 3 Failed

Round 3 scored tiles once, extracted the worst 40%, and trained on them for 12 epochs. The problem:

```
Round 3 HEM timeline:
  Score → hard set fixed at epoch 0 
  Epoch 1-4: model rapidly improves on these specific tiles → val accuracy +0.23pp
  Epoch 5-12: model has memorised the hard set → val accuracy flat, oscillates ±0.01pp
```

By epoch 4 of HEM, the model had extracted all available signal from the fixed hard set. The next 8 epochs were wasted compute.

**Root cause:** the hard set was mined from a weaker model. As the model improved, those tiles were no longer the hardest — new tiles became the bottleneck. But because scoring was done only once, the training set was never updated.

### Round 4 Multi-Round HEM

Three independent HEM rounds, each with fresh scoring:

```
Round B1:
  Score 4922 tiles with model@Phase_A_end
  Hard set = worst 35% = 1723 tiles
  Train 8 epochs on hard set
  Model improves → some tiles exit the hard set, others enter

Round B2:
  Re-score 4922 tiles with improved model
  Hard set = NEW worst 35% = different 1723 tiles
  Train 8 epochs on NEW hard set
  Gradient flows to tiles that became newly hard

Round B3:
  Re-score again, train again on latest hard set
```

The key invariant: the hard set is always defined relative to the **current model's capability**, not the capability at the start of Phase B.

### Scoring Mechanism

Per-tile accuracy is computed efficiently by running the model in inference mode on every tile and comparing predictions to pseudo-labels:

```python
with torch.no_grad():
    for pts, lbs in DataLoader(all_tiles, batch_size=16, shuffle=False):
        logits = model(pts)                         # (B, N, 2)
        preds  = logits.argmax(-1)                  # (B, N)
        for b in range(B):
            tile_acc = (preds[b].cpu() == lbs[b]).float().mean().item()
            tile_accs.append((tile_acc, tile_name))

tile_accs.sort(key=lambda x: x[0])   # worst-first
hard_tiles = [name for acc, name in tile_accs[:int(len(tile_accs) * 0.35)]]
```

Scoring 4,922 tiles takes ~5 minutes. With 3 rounds, this adds ~15 minutes of overhead to Phase B — a worthwhile investment for the fresh gradient signal.

### Phase B Hyperparameters

- **LR:** 8e-5 (lower than Phase A's 2e-4 — targeted refinement, not broad learning)
- **Mixup:** Disabled — hard tiles need clean gradient signal. Mixing a hard tile with an easy tile dilutes the gradient contribution of the hard examples
- **Epochs per round:** 8 — enough to converge on the current hard set before re-scoring
- **Hard fraction:** 35% (vs 40% in Round 3 — tighter focus on the most genuinely difficult tiles)

### Best Weights Management

At the start of each HEM round, the model is reloaded from `best_model.pth` (the highest-val-accuracy checkpoint from Phase A or previous HEM rounds). This prevents a bad HEM round from corrupting the model if it momentarily overfits the hard set:

```python
for hem_round in range(1, 4):
    base.load_state_dict(torch.load(CFG['model_save']))   # always start from best
    HARD_TILES = score_tiles_for_hem(model, ...)          # fresh scoring
    train_on_hard_tiles(8_epochs)
    # If new best found, model_save is updated; otherwise old best preserved
```

---

## 10. Fine-Tuning Phase C — Stochastic Weight Averaging

### What SWA Does

Stochastic Weight Averaging (Izmailov et al., 2018) averages the weights of checkpoints taken at regular intervals during a final training phase with a low, constant (or cyclic) learning rate.

The intuition: the loss landscape has a wide, flat basin near the true optimum. SGD with a decreasing LR converges to a single point near the centre, but that point may be on an asymmetric rim that generalises poorly. SWA averages multiple points sampled from around the basin, converging to the approximate **centre** which has better flatness properties and therefore better generalisation.

```
Without SWA: → → → → → ×           (converged point, possibly rim)
With SWA:    → → → ×₁ → ×₂ → ×₃   (multiple points averaged)
             θ_swa = (θ₁ + θ₂ + θ₃) / 3   (centre of the basin)
```

### Implementation

SWA is implemented via `torch.optim.swa_utils.AveragedModel`, which maintains a running average of all model parameters:

```python
swa_model = AveragedModel(base)   # wraps the training model

for epoch in range(swa_epochs):
    train_one_epoch(base)
    swa_model.update_parameters(base)   # accumulate into running average
    swa_scheduler.step()

# After training: update BatchNorm statistics
update_bn(train_loader, swa_model)      # critical step (see OOM section)
```

### BatchNorm Update: Why It's Required

BatchNorm layers accumulate running mean and variance estimates during training. The SWA model's parameters are an average of checkpoints from different training steps — its BatchNorm statistics are therefore an average of statistics from different data distributions encountered during training, which is meaningless.

The `update_bn` step runs a forward pass of the training data through the frozen SWA model specifically to recompute correct BatchNorm statistics for the averaged weights. Without this step, BatchNorm layers apply the wrong normalisation at inference time, causing an accuracy drop of ~1–2pp.

Round 4 uses 1,200 randomly sampled tiles for BN update (vs 400 in Round 3). More tiles → more representative sample of the training distribution → better-calibrated BN statistics.

### OOM-Safe SWA: 3-Stage Teardown

See [Section 13](#13-oom-management-on-t4--2) for the full OOM analysis. The BN update on T4 × 2 requires a careful GPU teardown before it can run safely.

---

## 11. Test-Time Augmentation: Geometric TTA

### Permutation TTA vs Geometric TTA

**Round 3 TTA (permutation-only):**
```python
for _ in range(8):
    perm = np.random.permutation(N)
    pts_shuffled = pts[perm]              # same points, different order
    probs = model(pts_shuffled)[inv_perm] # unscramble back
    avg_probs += probs
avg_probs /= 8
```

PointNet++ is not exactly permutation-invariant due to the FPS sampling step (FPS starts from a random seed, so different orderings → different centroid selections → slightly different features). Permutation TTA averages over these FPS random seeds.

However, all 8 passes see the **same geometric tile** — the same physical arrangement of points. The model is averaging over implementation noise, not genuine geometric uncertainty.

**Round 4 TTA (geometric):**
```python
rotations = [0°, 90°, 180°, 270°]     # 4 rotations
scales     = [0.95, 1.00, 1.05]        # 3 scale factors
# 12 total passes = 4 × 3

for angle in rotations:
    for scale in scales:
        R = rotation_matrix_2d(angle)
        pts_rotated = apply_transform(pts, R, scale)
        probs = model(pts_rotated)       # different geometry each pass
        avg_probs += probs
avg_probs /= 12
```

Each of the 12 passes presents the model with a **genuinely different view** of the same tile. The model's predictions may differ across views (it is not perfectly rotation/scale invariant) — averaging these predictions reduces the systematic bias of any single view.

### Why This Matters for LiDAR

LiDAR tiles are oriented arbitrarily relative to the scan direction. A tile from a North-South flight line looks different to a tile from an East-West flight line to the model (despite representing the same terrain), because the MSG feature extraction at different radii captures directional density patterns. Averaging over 4 rotations cancels out this orientation bias.

The 3-scale TTA covers the uncertainty in absolute altitude estimation: a point cloud at 0.95× scale represents the same terrain as if the sensor had flown 5% lower. The model should be invariant to this, but BatchNorm statistics are computed from real-scale tiles, so slight scale sensitivity remains.

### Expected Gain

Permutation TTA: +0.3–0.5pp (averaging FPS noise)
Geometric TTA: +0.5–0.8pp (averaging genuine geometric uncertainty)

Geometric TTA requires the `GroundDataset` to support explicit `tta_angle` and `tta_scale` parameters that are applied before feature computation, ensuring that the 6 engineered features are recomputed from the transformed geometry, not from the original.

---

## 12. Threshold Optimization

The model outputs logits `(B, N, 2)`. After softmax, `P(ground)` ∈ [0, 1] per point. Thresholding at 0.5 is suboptimal when classes are imbalanced or the loss is asymmetric.

The threshold sweep finds the value that maximises **F1 score on the ground class** (not accuracy):

```python
for threshold in np.arange(0.10, 0.90, 0.01):
    predictions = (probs >= threshold).astype(int)
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 × precision × recall / (precision + recall)
    accuracy  = (predictions == labels).mean()

best_threshold = threshold at max F1
```

F1 is used (not accuracy) because the ground class is the object of interest for DTM generation — a model that labels everything as non-ground achieves high accuracy in urban tiles but completely fails the application.

The Round 3 optimal threshold was 0.56 (slightly above 0.5), indicating the model's probabilities are slightly conservative — it slightly underestimates ground confidence, so the threshold must be lowered from the naive 0.5 to recapture correctly classified ground points.

---

## 13. OOM Management on T4 × 2

### The OOM Anatomy

T4 × 2 (DataParallel) peak VRAM usage during SWA BN update in Round 3:

| Component | VRAM |
|---|---|
| Training model weights | ~16 MB |
| Optimizer (AdamW: 2× param tensors) | ~32 MB |
| `AveragedModel` (second copy of all weights) | ~16 MB |
| DataParallel activations, batch_size=16, both GPUs | ~12,800 MB |
| Gradient buffers | ~16 MB |
| **Total at BN update moment** | **>12,880 MB per GPU** |

At exactly the moment `update_bn` begins, both the `AveragedModel` (on GPU) and the DataLoader activations (on both GPUs) are live simultaneously. This peaks at 15+ GB — exactly at the T4's limit. The kernel silently dies.

### Four OOM Vectors and Their Fixes

**OOM Vector 1: SWA BN update (critical)**

```python
# WRONG (Round 3):
swa_cpu = swa_m.cpu()       # moves AveragedModel to CPU, but training model still on GPU
update_bn(loader, swa_cpu)  # loader creates activations — both models compete for VRAM

# CORRECT (Round 4): 3-stage teardown
# Stage 1: Extract weights to plain CPU dict (no model object on GPU)
swa_state_dict = {k: v.cpu().clone() for k, v in swa_m.module.state_dict().items()}

# Stage 2: Destroy EVERYTHING on GPU
del tr_ld_C, swa_s, opt_C, swa_m, scaler
if hasattr(model, 'module'): del model.module
del model, base
torch.cuda.empty_cache(); gc.collect(); torch.cuda.synchronize()
# GPU now at ~0 MB

# Stage 3: CPU-only model, no GPU involved
swa_cpu = DTMPointNet2(IN_FEAT)           # NO .to(DEVICE)
swa_cpu.load_state_dict(swa_state_dict)
update_bn(cpu_loader, swa_cpu, device='cpu')
```

**OOM Vector 2: Phase A label refresh**

During mid-epoch label refresh, the training DataLoader's prefetch workers hold ~2–3 GB of pinned-memory prefetch buffers while pseudo-label generation also needs GPU memory:

```python
# Before refresh:
del tr_ld_A          # free prefetch buffers
gc.collect()
torch.cuda.empty_cache()
# Now safe to run model inference for pseudo-label generation
cur_pseudo = generate_pseudolabels(model, ...)
# Rebuild loader with fresh labels
tr_ld_A = make_loader(...)
```

**OOM Vector 3: Phase B HEM re-scoring between rounds**

After each HEM round, `tr_ld_B`, `opt_B`, and `sched_B` remain in scope when the re-scoring pass runs:

```python
if hem_round > 1:
    del tr_ld_B, opt_B, sched_B
    gc.collect()
    torch.cuda.empty_cache()
HARD_TILES = score_tiles_for_hem(model, ...)   # now safe
```

**OOM Vector 4: Cell 9 TTA sweep (two models)**

The TTA sweep function loads a full model per call. If two sweeps run back-to-back without cleanup, both models are briefly on GPU simultaneously:

```python
def sweep_geometric_tta(model_path, ...):
    m = DTMPointNet2().to(device)
    m.load_state_dict(...)
    # ... run TTA ...
    del m                          # explicit delete before return
    torch.cuda.empty_cache()
    gc.collect()
    return results

# Cell 9 also clears GPU state at the top, before either sweep:
torch.cuda.empty_cache(); gc.collect()
```

### Memory Monitoring

After the Stage 2 teardown in Phase C, the code explicitly logs GPU allocation:

```python
for i in range(N_GPUS):
    used = torch.cuda.memory_allocated(i) / 1e6
    print(f'GPU {i} after teardown: {used:.1f} MB (target: <100 MB)')
```

If this prints > 200 MB, there is a memory leak and the BN update should not proceed.

---

## 14. Hyperparameter Reference

### Complete Configuration

| Parameter | Round 3 | Round 4 | Rationale for Change |
|---|---|---|---|
| `phaseA_lr` | 3e-4 | 2e-4 | Starting from higher accuracy; smaller steps needed |
| `phaseA_epochs` | 15 | 20 | Extra epochs to exploit label refreshes |
| `phaseA_refresh_every` | N/A | 5 | New: breaks the stale label loop |
| `phaseB_rounds` | 1 | 3 | New: fresh scoring each round |
| `phaseB_epochs_per_round` | 12 (total) | 8 (per round) | Prevent memorisation within each round |
| `phaseB_lr` | 1e-4 | 8e-5 | Slightly lower; more targeted refinement |
| `hem_hard_frac` | 0.40 | 0.35 | Tighter focus on truly hard tiles |
| `swa_epochs` | 5 | 8 | More accumulation → flatter basin |
| `swa_lr` | 5e-5 | 4e-5 | Lower to match reduced phaseA LR |
| `swa_bn_tiles` | 400 | 1200 | Better BN calibration |
| `pseudo_conf_r4` | 0.95 | 0.92 | Wider net; R3 was too conservative |
| `pseudo_conf_refresh` | N/A | 0.88 | Even wider for mid-training refreshes |
| `mixup_alpha` | 0.4 | 0.2 | Sharper Beta → more realistic mixes |
| `focal_gamma` | 2.5 | 2.5 | Unchanged |
| `focal_alpha` | 0.75 | 0.75 | Unchanged |
| `label_smooth` | 0.03 | 0.02 | Less smoothing; model is better calibrated |
| `val_tta_passes` | 8 (perm) | 12 (geo) | 4 rotations × 3 scales |
| `max_pts` | 4096 | 4096 | Unchanged |
| `batch_size` | 16 (8/GPU) | 16 (8/GPU) | Unchanged; OOM-tested |
| `weight_decay` | 1e-5 | 1e-5 | Unchanged |
| `grad_clip` | 1.0 | 1.0 | Unchanged |
| `height_jitter_std` | 0.02m | 0.05m | More realistic sensor noise |
| `point_drop_rate` | 0.10 | 0.15 | More aggressive density variation |
| `pseudo_n_subsamples` | 1 | 3 | New: multi-subsample averaging |

---

## 15. Round-by-Round Accuracy History

| Round | Key Change | Val Accuracy | F1 | Threshold | ΔAcc |
|---|---|---|---|---|---|
| v6 (base) | Initial training | 93.65% | 0.9020 | 0.57 | — |
| Round 3 | Pseudo-labels R3, HEM×1, SWA×5, permutation TTA×8 | 93.97% | 0.9073 | 0.56 | +0.32pp |
| Round 4 (target) | Label refresh, HEM×3 rounds, SWA×8, geometric TTA×12 | **~98%** | **~0.96+** | TBD | **+4pp** |

### Round 3 Failure Analysis

Sub-phase A peak: 91.95% at epoch 14 (training val).
Sub-phase B peak: 92.15% at epoch 4 of HEM — improvement stopped completely at epoch 4 despite 8 more epochs of training.
TTA rescue: +1.82pp from 92.15% (raw) → 93.97% (with 8-pass permutation TTA).

The TTA boost being so large (+1.82pp) relative to training improvement (+0.50pp from Phase A start) indicates the model was poorly calibrated — its predictions were noisy and TTA variance-reduction provided more gain than the training itself. This is a red flag that training had stalled due to the stale label problem.

---

## 16. Why Each Technique Was Necessary

The table below maps each technique to its theoretical justification and the observed evidence that motivated it:

| Technique | Theoretical Basis | Evidence That Motivated It |
|---|---|---|
| **Pseudo-label refresh** | Self-training theory: pseudo-labels should be regenerated as the model improves to avoid confirmation bias | Round 3: training acc improved but val acc flat — label memorisation signature |
| **Multi-round HEM** | Curriculum learning: difficulty should be re-evaluated as model improves | Round 3 HEM: val acc improvement stopped exactly at epoch 4 of 12 — same-tiles saturation |
| **Multi-subsample pseudo-labeling** | Variance reduction via averaging: `Var(mean of n) = Var(single) / n` | Round 3 had 2.7% label replacement — too conservative, suggesting single-pass estimates were noisy near threshold |
| **Geometric TTA** | Equivariant averaging: averaging predictions over a transformation group reduces directional bias | Round 3's 1.82pp TTA boost (very large) suggested high prediction variance — geometric TTA attacks variance more fundamentally than permutation TTA |
| **Wider pseudo-label confidence (0.92 vs 0.95)** | Coverage-purity tradeoff: R3's 2.7% replacement was so small it barely changed any labels | Empirically, the model plateaued despite 97.3% of labels being unchanged — no new signal |
| **Beta(0.2) Mixup** | Mixup theory: interpolation should explore realistic neighbourhoods of the data manifold, not mid-points | Beta(0.4) creates ~50% of mixed tiles near the 50/50 blend — these are off-manifold for real tiles |
| **SWA with more BN tiles** | BN statistics require representative data to calibrate correctly | Round 3 SWA on 400 tiles was ~8% of training data — insufficient for accurate running mean/variance |
| **3-stage OOM teardown** | GPU memory management: concurrent allocation exceeds T4's 15.6 GB budget | Round 2 silently OOMed at exactly the BN update step — confirmed by 3 independent reproduction attempts |
| **Focal Loss γ=2.5** | Hard-example focusing: at 94% accuracy, the easy 94% contribute 350× less than the hard 6% | Model was at 94% — standard CE would be dominated by already-correct predictions |
| **Label smoothing ε=0.02** | Calibration regularisation: prevents overconfident predictions that resist threshold tuning | Threshold search found optimal at 0.56 not 0.50 — slight miscalibration that smoothing should correct |

---

## Appendix: File Outputs

```
/kaggle/working/
├── logs/
│   ├── best_model.pth          # Best val accuracy checkpoint (best_epoch)
│   ├── swa_model.pth           # SWA averaged model
│   ├── threshold.json          # Optimal threshold + metadata
│   ├── history.json            # Per-epoch training history
│   ├── curves.png              # Learning curves
│   └── threshold_curve.png     # F1/Accuracy vs threshold plot
├── pseudo_labels/
│   ├── Round4/                 # Initial R4 pseudo-labels
│   ├── Round4_refresh_ep6/     # Labels after Phase A epoch 5
│   ├── Round4_refresh_ep11/    # Labels after Phase A epoch 10
│   └── Round4_refresh_ep16/    # Labels after Phase A epoch 15
├── consensus_labels/
│   ├── train/*.npy             # Geometry-only fallback labels, train
│   └── val/*.npy               # Geometry-only fallback labels, val
├── feat_cache/
│   ├── train/*.npy             # Cached 6-feature descriptors, train
│   └── val/*.npy               # Cached 6-feature descriptors, val
└── dtm_outputs_finetuned.zip   # Final packaged deliverable
```

```json
// threshold.json schema
{
  "model": "best_epoch+GeoTTA",
  "model_path": "/kaggle/working/logs/best_model.pth",
  "threshold": 0.54,
  "val_accuracy": 0.9801,
  "val_f1": 0.9623,
  "tta_passes": 12,
  "tta_type": "geometric",
  "n_rotations": 4,
  "n_scales": 3
}
```

---

*Authored for internal engineering review. Reproducible on Kaggle GPU T4 × 2, runtime ≈ 90–125 minutes.*
