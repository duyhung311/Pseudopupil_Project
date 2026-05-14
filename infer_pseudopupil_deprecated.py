"""
Pseudopupil Heatmap Regression — Inference Script
==================================================
Replaces the classical blob detection in detect_pseudopupil.py with
the trained CNN. Loads all .mat files from azimuth subfolders and
writes results.csv with (azimuth, filename, cx, cy, confidence).

Usage:
  python infer_pseudopupil.py

Dependencies:
  pip install torch torchvision timm scipy scikit-image pandas opencv-python-headless
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────
# CONFIGURATION  (keep in sync with train script)
# ─────────────────────────────────────────────
# SCIENTIFIC_NAME = "Aeschna_isoceles"
SCIENTIFIC_NAME = "Aeschna_isoceles"
MAT_DIR    = f"./{SCIENTIFIC_NAME}"
MODEL_PATH = ".\pseudopupil_model.pth"
OUTPUT_CSV = f"./results-v2-{SCIENTIFIC_NAME}.csv"
DEBUG      = False
DEBUG_DIR  = "./debug_infer_images"

ANGLE_VAR_PATTERN = "imagesS{i}"
N_ANGLES          = 5
FLIP_HORIZONTAL = True

# Loaded from checkpoint at runtime — do not set manually
IMAGE_SIZE   = None
HEATMAP_SIZE = None

AZIMUTH_STEP  = 6
AZIMUTH_START = 0
AZIMUTH_END   = 48

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONSISTENCY_THRESH_PX = 30


# ─────────────────────────────────────────────
# MODEL  — must match train_pseudopupil.py exactly
# ─────────────────────────────────────────────

class PseudopupilHeatmapNet(nn.Module):
    """
    Input channels: 3  (I1_n, I2_n, diff_n)
    3 channels matches ImageNet pretrained weights — no conv patching needed.
    """
    def __init__(self, image_size: int, heatmap_size: int):
        super().__init__()
        self.heatmap_size = heatmap_size

        # FIX 1: 3 input channels to match training (was incorrectly 2)
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        # No conv patching needed — 3ch matches ImageNet weights structure

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d((heatmap_size, heatmap_size))

    def forward(self, x):
        features   = self.encoder(x)
        bottleneck = features[-1]
        heatmap    = self.decoder(bottleneck)
        heatmap    = self.pool(heatmap)
        return heatmap


def load_model(model_path: str):
    """Load trained model from checkpoint."""
    checkpoint   = torch.load(model_path, map_location=DEVICE)
    image_size   = checkpoint["image_size"]
    heatmap_size = checkpoint["heatmap_size"]

    model = PseudopupilHeatmapNet(image_size, heatmap_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from {model_path}  "
          f"(epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.5f})")
    return model, image_size, heatmap_size


# ─────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def local_contrast_norm(img: np.ndarray, sigma: int = 32) -> np.ndarray:
    """
    Divide each pixel by its local neighbourhood mean.
    Removes global brightness so the pseudopupil is detectable by
    local contrast alone, regardless of illumination level.
    """
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img.astype(np.float32), size=sigma)
    normed = img / (local_mean + 0.02)
    return rescale_intensity(normed, out_range=(0.0, 1.0)).astype(np.float32)


def letterbox(img: np.ndarray, size: int):
    """
    Resize (H, W, C) → (size, size, C) preserving aspect ratio with zero-padding.
    Returns (out, scale, pad_left, pad_top).
    """
    import cv2
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    new_w   = int(round(w * scale))
    new_h   = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out      = np.zeros((size, size, img.shape[2]), dtype=np.float32)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out, scale, pad_left, pad_top


def load_and_preprocess(mat_path: str, angle_key: str, image_size: int):
    """
    Load I1 + I2 from .mat, build 3-channel input (I1_n, I2_n, diff_n),
    letterbox to (image_size, image_size), return tensor + letterbox params.

    Returns:
      img_tensor            : torch.Tensor (1, 3, image_size, image_size)
      orig_h, orig_w        : original image dimensions
      scale, pad_left, pad_top : letterbox params for coordinate inversion
    """
    mat   = sio.loadmat(mat_path)
    imdat = mat["imdat"]
    I1_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][0]
    I2_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][1]

    I1 = rescale_intensity(img_as_float(np.squeeze(I1_raw)), out_range=(0.0, 1.0)).astype(np.float32)
    I2 = rescale_intensity(img_as_float(np.squeeze(I2_raw)), out_range=(0.0, 1.0)).astype(np.float32)

    orig_h, orig_w = I1.shape[:2]

    # FIX 2: compute diff BEFORE normalisation, then normalise all three channels
    diff = np.abs(I2 - I1).astype(np.float32)
    I1_n   = local_contrast_norm(I1)
    I2_n   = local_contrast_norm(I2)
    diff_n = local_contrast_norm(diff)   # was using raw diff — now correctly uses diff_n

    img_hwc = np.stack([I1_n, I2_n, diff_n], axis=-1)          # (H, W, 3)
    img_lb, scale, pad_left, pad_top = letterbox(img_hwc, image_size)
    if FLIP_HORIZONTAL:
        img_lb = img_lb[:, ::-1, :].copy()
    # FIX 3: transpose axes (2,0,1) for a 3D HWC array — was (3,0,1) which crashes
    img_tensor = torch.from_numpy(
        img_lb.transpose(2, 0, 1)   # (H,W,3) → (3,H,W)
    ).unsqueeze(0).float()           # → (1, 3, H, W)

    return img_tensor, orig_h, orig_w, scale, pad_left, pad_top


# ─────────────────────────────────────────────
# COORDINATE EXTRACTION + CONFIDENCE
# ─────────────────────────────────────────────

def heatmap_to_coords(heatmap: torch.Tensor,
                      orig_w: int, orig_h: int,
                      image_size: int,
                      scale: float, pad_left: int, pad_top: int):
    """
    Soft-argmax on predicted heatmap → (cx, cy) in original pixel space.
    Also returns peak activation as a confidence proxy (0–1).
    """
    hs = heatmap.shape[-1]
    hm = heatmap.squeeze()
    confidence = float(hm.max().item())

    hm = hm / (hm.sum() + 1e-8)

    xs = torch.arange(hs, dtype=torch.float32, device=hm.device)
    ys = torch.arange(hs, dtype=torch.float32, device=hm.device)
    xg, yg = torch.meshgrid(xs, ys, indexing="xy")

    cx_hm = (hm * xg).sum().item()
    cy_hm = (hm * yg).sum().item()

    # heatmap space → letterboxed image space
    cx_lb = cx_hm * image_size / hs
    cy_lb = cy_hm * image_size / hs

    # undo letterbox: remove padding then undo scale
    cx = (cx_lb - pad_left) / scale
    cy = (cy_lb - pad_top)  / scale
    if FLIP_HORIZONTAL:
        cx = orig_w - 1 - cx
    # clamp to original image bounds
    cx = float(np.clip(cx, 0, orig_w - 1))
    cy = float(np.clip(cy, 0, orig_h - 1))

    return cx, cy, confidence


# ─────────────────────────────────────────────
# TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────

def predict_with_tta(model, img_tensor, orig_w, orig_h,
                     image_size, scale, pad_left, pad_top):
    """
    Run 8 augmented versions through the model, average heatmaps, extract coords.
    Improves robustness on unseen data without any retraining.
    """
    fwd_fns = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
        lambda x: x.rot90(1, [-2, -1]),
        lambda x: x.rot90(2, [-2, -1]),
        lambda x: x.rot90(3, [-2, -1]),
        lambda x: x.rot90(1, [-2, -1]).flip(-1),
    ]
    inv_fns = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
        lambda x: x.rot90(-1, [-2, -1]),
        lambda x: x.rot90(-2, [-2, -1]),
        lambda x: x.rot90(-3, [-2, -1]),
        lambda x: x.flip(-1).rot90(-1, [-2, -1]),
    ]
    avg_heatmap = None
    hs = model.pool.output_size  # (heatmap_size, heatmap_size)

    with torch.no_grad():
        for fwd, inv in zip(fwd_fns, inv_fns):
            aug  = fwd(img_tensor)
            pred = model(aug)
            pred = inv(pred)
            # After rot90, spatial dims may be swapped — resize back to canonical
            pred = F.interpolate(pred, size=hs, mode="bilinear", align_corners=False)
            avg_heatmap = pred if avg_heatmap is None else avg_heatmap + pred

    avg_heatmap /= len(fwd_fns)
    return heatmap_to_coords(avg_heatmap[0], orig_w, orig_h,
                             image_size, scale, pad_left, pad_top)


# ─────────────────────────────────────────────
# MULTI-ANGLE CONSISTENCY FILTER
# ─────────────────────────────────────────────

def consistency_filter(candidates: list):
    """
    candidates: list of (cx, cy, confidence) or None — one per angle.
    Returns (cx, cy, mean_confidence, n_inliers).
    Outliers: Euclidean distance from median > CONSISTENCY_THRESH_PX.
    Inliers averaged weighted by per-angle confidence.
    """
    valid = [c for c in candidates if c is not None]

    if not valid:
        return None, None, 0.0, 0

    xs    = np.array([v[0] for v in valid])
    ys    = np.array([v[1] for v in valid])
    confs = np.array([v[2] for v in valid])

    med_x, med_y = np.median(xs), np.median(ys)
    dists = np.sqrt((xs - med_x)**2 + (ys - med_y)**2)
    mask  = dists < CONSISTENCY_THRESH_PX

    if mask.sum() == 0:
        mask = np.ones(len(xs), dtype=bool)  # fallback: use all

    w         = confs[mask]
    cx        = float(np.average(xs[mask], weights=w))
    cy        = float(np.average(ys[mask], weights=w))
    mean_conf = float(w.mean())
    n_inliers = int(mask.sum())

    return cx, cy, mean_conf, n_inliers


# ─────────────────────────────────────────────
# PER-FILE PROCESSING
# ─────────────────────────────────────────────

def process_file(mat_path: str, model, image_size: int, heatmap_size: int, azimuth: int):
    """
    Full pipeline for one .mat file:
      1. Load + preprocess each angle (3-channel letterboxed tensor)
      2. Predict heatmap with CNN (+ TTA)
      3. Extract (cx, cy, confidence) per angle
      4. Multi-angle consistency filter → final (cx, cy)
    """
    candidates = []

    for i in range(N_ANGLES):
        angle_key = ANGLE_VAR_PATTERN.format(i=i)
        try:
            img_tensor, orig_h, orig_w, scale, pad_left, pad_top =                 load_and_preprocess(mat_path, angle_key, image_size)
        except Exception as e:
            print(f"\n    [WARN] angle {i}: {e}")
            candidates.append(None)
            continue

        img_tensor = img_tensor.to(DEVICE)
        cx, cy, conf = predict_with_tta(
            model, img_tensor, orig_w, orig_h,
            image_size, scale, pad_left, pad_top
        )
        candidates.append((cx, cy, conf))

        if DEBUG:
            # For debug we still need the raw heatmap — run one forward pass
            with torch.no_grad():
                raw_heatmap = model(img_tensor)
            _save_debug(
                mat_path, azimuth, i,
                img_tensor, raw_heatmap,
                cx, cy,
                scale, pad_left, pad_top,    # letterbox params
                orig_w, orig_h, image_size   # for coordinate conversion
            )
    cx, cy, conf, n_inliers = consistency_filter(candidates)

    return {
        "azimuth":           azimuth,
        "filename":          os.path.basename(mat_path),
        "cx":                round(cx, 2) if cx is not None else None,
        "cy":                round(cy, 2) if cy is not None else None,
        "confidence":        round(conf, 4) if conf is not None else 0.0,
        "n_inliers":         n_inliers,
        "n_angles_detected": sum(c is not None for c in candidates),
    }


# ─────────────────────────────────────────────
# DEBUG OUTPUT
# ─────────────────────────────────────────────

def _save_debug(mat_path, azimuth, angle_idx, img_tensor, heatmap_tensor, cx, cy,
                scale, pad_left, pad_top, orig_w, orig_h, image_size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    stem = Path(mat_path).stem

    img = img_tensor.squeeze().cpu().numpy()      # (3, H, W)
    hm  = heatmap_tensor.squeeze().cpu().numpy()  # (Hs, Hs)
    hs  = hm.shape[0]                             # e.g. 64

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img[0], cmap="gray");  axes[0].set_title("I1 (low exp, LCN)")
    axes[1].imshow(img[1], cmap="gray");  axes[1].set_title("I2 (high exp, LCN)")
    axes[2].imshow(img[2], cmap="gray");  axes[2].set_title("|I2−I1| (diff, LCN)")
    axes[3].imshow(hm, cmap="hot");       axes[3].set_title("Predicted heatmap")

    # FIX: convert (cx, cy) from original pixel space → heatmap space
    # Pipeline (forward): orig → letterbox → image_size → heatmap_size
    # Step 1: orig → letterboxed image space
    cx_lb = cx * scale + pad_left
    cy_lb = cy * scale + pad_top
    # Step 2: letterboxed image space → heatmap space
    cx_hm = cx_lb * hs / image_size
    cy_hm = cy_lb * hs / image_size

    axes[3].plot(cx_hm, cy_hm, "b+", markersize=14, markeredgewidth=2)
    axes[3].annotate(
        f"({cx:.1f}, {cy:.1f})",         # show original pixel coords in label
        xy=(cx_hm, cy_hm),
        xytext=(5, -12),
        textcoords="offset points",
        color="cyan", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5)
    )
    axes[3].set_xlim(0, hs)
    axes[3].set_ylim(hs, 0)              # flip y to match image convention

    fig.suptitle(f"{stem} | {azimuth}° | angle {angle_idx}")
    fig.tight_layout()
    out = os.path.join(DEBUG_DIR, f"{stem}_{azimuth}deg_angle{angle_idx}.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def collect_mat_files():
    collected = []
    collected_angle = []
    for az in range(AZIMUTH_START, AZIMUTH_END + 1, AZIMUTH_STEP):
        collected_angle.append(az)

    collected_angle.append(39)

    for az in collected_angle:
        subfolder = os.path.join(MAT_DIR, f"{az}Azimuth")
        if not os.path.isdir(subfolder):
            print(f"  [WARN] Missing subfolder: {subfolder}")
            continue
        for path in sorted(glob.glob(os.path.join(subfolder, "*.mat"))):
            collected.append((az, path))
    return collected


def run_inference():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train_pseudopupil.py first."
        )

    model, image_size, heatmap_size = load_model(MODEL_PATH)

    entries = collect_mat_files()
    if not entries:
        print(f"No .mat files found under '{MAT_DIR}'.")
        return

    az_counts = {}
    for az, _ in entries:
        az_counts[az] = az_counts.get(az, 0) + 1
    print(f"\nFound {len(entries)} files across {len(az_counts)} azimuth subfolders")
    for az, n in sorted(az_counts.items()):
        print(f"    {az:3d}Azimuth  →  {n} file(s)")

    results    = []
    current_az = None

    for idx, (az, path) in enumerate(entries, 1):
        if az != current_az:
            current_az = az
            print(f"\n── {az}Azimuth ──────────────────────────")

        print(f"  [{idx:3d}/{len(entries)}] {os.path.basename(path)}", end=" ... ")
        try:
            row = process_file(path, model, image_size, heatmap_size, az)
            results.append(row)
            print(f"cx={row['cx']}, cy={row['cy']}, "
                  f"conf={row['confidence']}, inliers={row['n_inliers']}/{N_ANGLES}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "azimuth": az, "filename": os.path.basename(path),
                "cx": None, "cy": None, "confidence": 0.0,
                "n_inliers": 0, "n_angles_detected": 0,
            })

    cols = ["azimuth", "filename", "cx", "cy", "confidence",
            "n_inliers", "n_angles_detected"]
    df = pd.DataFrame(results)
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["azimuth", "filename"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n\nDone. Results → {OUTPUT_CSV}")
    print("\nMean per azimuth:")
    print(df.groupby("azimuth")[["cx", "cy", "confidence", "n_inliers"]].mean().round(2).to_string())


if __name__ == "__main__":
    run_inference()