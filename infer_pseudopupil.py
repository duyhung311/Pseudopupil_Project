"""
Pseudopupil Detection Pipeline v2
==================================
Two-stage pipeline per angle image:
  Stage 1 — CNN (EfficientNetB0 heatmap regression):
              finds the pseudopupil region in the full image
  Stage 2 — Classical brightness refinement on raw I1:
              within the CNN window, finds the intensity-weighted
              centroid of the brightest pixels (handles triangle/
              polygon configurations automatically)

Output: 5 predictions per .mat file (one per imagesS0–imagesS4),
        each in original image pixel space (2048×1500).

Folder structure expected:
  MAT_DIR/
    0Azimuth/  *.mat
    6Azimuth/  *.mat
    ...
    78Azimuth/ *.mat

Dependencies:
  pip install torch torchvision timm scipy scikit-image pandas opencv-python-headless matplotlib
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these before running
# ═══════════════════════════════════════════════════════════════

SCIENTIFIC_NAME = "Aeschna_isoceles"
MAT_DIR         = f"./{SCIENTIFIC_NAME}"
MODEL_PATH = "./pseudopupil_model.pth"
OUTPUT_CSV = "./results_v2_1.csv"
DEBUG      = True
DEBUG_DIR  = "./debug_v2"

# Azimuth subfolder range
AZIMUTH_START = 0
AZIMUTH_END   = 78
AZIMUTH_STEP  = 6

# Angle variables inside each .mat file
ANGLE_VAR_PATTERN = "imagesS{i}"
N_ANGLES          = 5

# ── Horizontal flip ──────────────────────────────────────────
# Set True if the capture session has the eye on the opposite
# side vs the training data (mirrored composition)
FLIP_HORIZONTAL = False

# ── Stage 2 refinement tuning ────────────────────────────────
# Search window around CNN centre (original pixel radius).
# Rule of thumb: ~3× the expected pseudopupil radius.
# Start at 80, increase if CNN is imprecise, decrease if it
# jumps to nearby specular highlights.
REFINE_WINDOW_PX = 50

# Brightness threshold percentile.
# 95 → top 5% brightest pixels kept.
# Increase (e.g. 99) for a tighter/smaller bright region.
# Decrease (e.g. 90) if the pseudopupil is low-contrast.
REFINE_PERCENTILE = 97

# Minimum blob size in pixels. Blobs smaller than this are
# ignored (removes single-pixel noise spikes).
REFINE_MIN_BLOB_PX = 3

# max allowed shift from coarse → refined
# if refinement moves more than this, fall back to coarse
# tune: set to ~half your pseudopupil radius (50-100px → 40px)
REFINE_MAX_SHIFT_PX = 40  #

# ── Runtime ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# MODEL  — must match train_pseudopupil.py architecture exactly
# ═══════════════════════════════════════════════════════════════

class PseudopupilHeatmapNet(nn.Module):
    """
    EfficientNetB0 encoder + lightweight decoder.
    Input : (B, 3, IMAGE_SIZE, IMAGE_SIZE)  — [I1_lcn, I2_lcn, diff_lcn]
    Output: (B, 1, HEATMAP_SIZE, HEATMAP_SIZE) — predicted heatmap
    """
    def __init__(self, image_size: int, heatmap_size: int):
        super().__init__()
        self.image_size   = image_size
        self.heatmap_size = heatmap_size

        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        # 3-channel input matches ImageNet weights — no patching needed

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
        return self.pool(heatmap)


def load_model(model_path: str):
    """Load trained model from checkpoint. Returns (model, image_size, heatmap_size)."""
    ckpt = torch.load(model_path, map_location=DEVICE)
    image_size   = ckpt["image_size"]
    heatmap_size = ckpt["heatmap_size"]
    model = PseudopupilHeatmapNet(image_size, heatmap_size).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.5f}  "
          f"image_size={image_size}  heatmap_size={heatmap_size}")
    return model, image_size, heatmap_size


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def local_contrast_norm(img: np.ndarray, sigma: int = 32) -> np.ndarray:
    """Normalise by local mean — removes global brightness variation."""
    from scipy.ndimage import uniform_filter
    lm = uniform_filter(img.astype(np.float32), size=sigma)
    return rescale_intensity(img / (lm + 0.02), out_range=(0.0, 1.0)).astype(np.float32)


def letterbox(img: np.ndarray, size: int):
    """
    Resize (H, W, C) → (size, size, C) preserving aspect ratio, zero-pad.
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


def load_raw_images(mat_data: dict, angle_key: str):
    """
    Extract raw I1 and I2 arrays from .mat struct.
    Returns (I1_raw, I2_raw) as float32 in [0, 1], original resolution.
    """
    imdat  = mat_data["imdat"]
    I1_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][0]
    I2_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][1]
    I1 = rescale_intensity(
        img_as_float(np.squeeze(I1_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)
    I2 = rescale_intensity(
        img_as_float(np.squeeze(I2_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)
    return I1, I2


def build_channels(I1: np.ndarray, I2: np.ndarray,
                   exposure: str = "both") -> np.ndarray:
    """
    Build (H, W, 3) input array. Identical to train script.
      "both" -> [I1_lcn, I2_lcn, |I2-I1|_lcn]
      "i1"   -> [I1_lcn, I1_lcn, zeros]
      "i2"   -> [I2_lcn, I2_lcn, zeros]
    """
    if exposure == "i1":
        c = local_contrast_norm(I1)
        return np.stack([c, c, np.zeros_like(c)], axis=-1)
    elif exposure == "i2":
        c = local_contrast_norm(I2)
        return np.stack([c, c, np.zeros_like(c)], axis=-1)
    else:
        diff = np.abs(I2 - I1).astype(np.float32)
        return np.stack([
            local_contrast_norm(I1),
            local_contrast_norm(I2),
            local_contrast_norm(diff),
        ], axis=-1)


def build_model_input(I1: np.ndarray, I2: np.ndarray, image_size: int,
                      exposure: str = "both"):
    """
    Build 3-channel model input using build_channels (identical to train),
    apply optional horizontal flip, letterbox.
    Returns (img_tensor, scale, pad_left, pad_top, orig_h, orig_w).
    """
    orig_h, orig_w = I1.shape[:2]
    img_hwc = build_channels(I1, I2, exposure)   # (H, W, 3)

    if FLIP_HORIZONTAL:
        img_hwc = img_hwc[:, ::-1, :].copy()

    img_lb, scale, pad_left, pad_top = letterbox(img_hwc, image_size)
    img_tensor = torch.from_numpy(
        img_lb.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    return img_tensor, scale, pad_left, pad_top, orig_h, orig_w


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — CNN COARSE PREDICTION
# ═══════════════════════════════════════════════════════════════

def soft_argmax_heatmap(heatmap: torch.Tensor, image_size: int):
    """
    Soft-argmax on (1, Hs, Hs) heatmap.
    Returns (cx_lb, cy_lb) in letterboxed IMAGE_SIZE space, plus confidence.
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

    # Scale from heatmap space → letterboxed image space
    cx_lb = cx_hm * image_size / hs
    cy_lb = cy_hm * image_size / hs
    return cx_lb, cy_lb, confidence


def letterbox_to_orig(cx_lb, cy_lb, scale, pad_left, pad_top, orig_w, orig_h):
    """
    Invert letterbox transform: letterboxed image space → original pixel space.
    Also inverts horizontal flip if FLIP_HORIZONTAL is True.
    """
    cx = (cx_lb - pad_left) / scale
    cy = (cy_lb - pad_top)  / scale

    if FLIP_HORIZONTAL:
        cx = orig_w - 1 - cx

    cx = float(np.clip(cx, 0, orig_w - 1))
    cy = float(np.clip(cy, 0, orig_h - 1))
    return cx, cy


def predict_with_tta(model, img_tensor, image_size):
    """
    8-fold test-time augmentation: average heatmaps from flipped/rotated
    versions of the input. Returns averaged heatmap (1, 1, Hs, Hs).
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
    hs  = model.heatmap_size
    avg = None
    with torch.no_grad():
        for fwd, inv in zip(fwd_fns, inv_fns):
            pred = model(fwd(img_tensor))
            pred = inv(pred)
            pred = F.interpolate(pred, size=(hs, hs),
                                 mode="bilinear", align_corners=False)
            avg = pred if avg is None else avg + pred
    return avg / len(fwd_fns)


def cnn_predict(model, img_tensor, image_size, scale, pad_left, pad_top,
                orig_w, orig_h):
    """
    Full Stage 1: TTA → soft-argmax → invert letterbox.
    Returns (cx_orig, cy_orig, confidence, heatmap).
    """
    heatmap = predict_with_tta(model, img_tensor, image_size)
    cx_lb, cy_lb, conf = soft_argmax_heatmap(heatmap, image_size)
    cx, cy = letterbox_to_orig(cx_lb, cy_lb, scale, pad_left, pad_top,
                               orig_w, orig_h)
    return cx, cy, conf, heatmap


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — CLASSICAL BRIGHTNESS REFINEMENT ON RAW I1
# ═══════════════════════════════════════════════════════════════

def refine_with_brightness_centroid(I1_raw, cx_coarse, cy_coarse):
    h, w = I1_raw.shape[:2]

    # ── 1. Crop window ────────────────────────────────────────────────────
    x0 = int(np.clip(cx_coarse - REFINE_WINDOW_PX, 0, w - 1))
    y0 = int(np.clip(cy_coarse - REFINE_WINDOW_PX, 0, h - 1))
    x1 = int(np.clip(cx_coarse + REFINE_WINDOW_PX, 0, w - 1))
    y1 = int(np.clip(cy_coarse + REFINE_WINDOW_PX, 0, h - 1))

    if (x1 - x0) < 5 or (y1 - y0) < 5:
        return cx_coarse, cy_coarse

    crop = I1_raw[y0:y1, x0:x1].astype(np.float32)
    crop_min, crop_max = crop.min(), crop.max()
    if crop_max - crop_min < 1e-6:
        return cx_coarse, cy_coarse

    crop_norm = (crop - crop_min) / (crop_max - crop_min)

    # ── 2. Adaptive threshold ─────────────────────────────────────────────
    # Instead of fixed percentile, use Otsu-like approach:
    # find natural break between background and pseudopupil brightness
    threshold = np.percentile(crop_norm, REFINE_PERCENTILE)
    bright_mask = crop_norm >= threshold

    # ── 3. Connected components — pick closest blob to CNN centre ─────────
    labeled = label(bright_mask)
    if labeled.max() == 0:
        return cx_coarse, cy_coarse

    cx_local = cx_coarse - x0
    cy_local = cy_coarse - y0

    best_region = None
    best_dist   = float("inf")
    for region in regionprops(labeled):
        if region.area < REFINE_MIN_BLOB_PX:
            continue
        ry, rx = region.centroid
        dist = np.sqrt((rx - cx_local)**2 + (ry - cy_local)**2)
        if dist < best_dist:
            best_dist   = dist
            best_region = region

    if best_region is None:
        return cx_coarse, cy_coarse

    # ── 4. KEY FIX: geometric centroid, not intensity-weighted ────────────
    # Unweighted centroid = centre of the bright region geometry
    # Intensity-weighted centroid = pulled toward brightest pixel (wrong
    # when multiple facets have similar brightness)
    blob_mask    = labeled == best_region.label
    ys_px, xs_px = np.where(blob_mask)

    cx_refined_local = float(np.mean(xs_px))   # unweighted — was np.average with weights
    cy_refined_local = float(np.mean(ys_px))   # unweighted

    # ── 5. Sanity check — only accept if refinement doesn't move too far ──
    # If the geometric centroid is further from coarse than a threshold,
    # the blob is probably a wrong region — fall back to coarse
    shift = np.sqrt((cx_refined_local - cx_local)**2 +
                    (cy_refined_local - cy_local)**2)
    if shift > REFINE_MAX_SHIFT_PX:
        return cx_coarse, cy_coarse

    cx_refined = float(np.clip(cx_refined_local + x0, 0, w - 1))
    cy_refined = float(np.clip(cy_refined_local + y0, 0, h - 1))
    return cx_refined, cy_refined


# ═══════════════════════════════════════════════════════════════
# PER-FILE PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_file(mat_path: str, model, image_size: int,
                 heatmap_size: int, azimuth: int) -> list:
    """
    Run the full two-stage pipeline for one .mat file.
    Returns a list of 5 dicts — one per angle (imagesS0–imagesS4).

    Each dict contains:
      azimuth, filename, angle,
      cx_coarse, cy_coarse,   ← CNN Stage 1 output (original px)
      cx, cy,                 ← Stage 2 refined output (original px)
      confidence              ← CNN heatmap peak activation
    """
    mat_data = sio.loadmat(mat_path)
    rows     = []

    for i in range(N_ANGLES):
        angle_key = ANGLE_VAR_PATTERN.format(i=i)

        # ── Load raw images ───────────────────────────────────────────────
        try:
            I1_raw, I2_raw = load_raw_images(mat_data, angle_key)
        except Exception as e:
            print(f"\n    [WARN] {angle_key} load failed: {e}")
            rows.append(_empty_row(azimuth, mat_path, i))
            continue

        orig_h, orig_w = I1_raw.shape[:2]

        # ── Stage 1: CNN coarse prediction ────────────────────────────────
        try:
            img_tensor, scale, pad_left, pad_top, _, _ = \
                build_model_input(I1_raw, I2_raw, image_size)
            img_tensor = img_tensor.to(DEVICE)

            cx_coarse, cy_coarse, conf, heatmap = cnn_predict(
                model, img_tensor, image_size,
                scale, pad_left, pad_top, orig_w, orig_h
            )
        except Exception as e:
            print(f"\n    [WARN] {angle_key} CNN failed: {e}")
            rows.append(_empty_row(azimuth, mat_path, i))
            continue

        # ── Stage 2: Classical brightness refinement on raw I1 ────────────
        try:
            cx_refined, cy_refined = refine_with_brightness_centroid(
                I1_raw, cx_coarse, cy_coarse
            )
        except Exception as e:
            print(f"\n    [WARN] {angle_key} refinement failed: {e}")
            cx_refined, cy_refined = cx_coarse, cy_coarse

        shift_px = np.sqrt((cx_refined - cx_coarse)**2 +
                           (cy_refined - cy_coarse)**2)

        print(f"      {angle_key}: "
              f"coarse=({cx_coarse:.1f},{cy_coarse:.1f}) → "
              f"refined=({cx_refined:.1f},{cy_refined:.1f})  "
              f"shift={shift_px:.1f}px  conf={conf:.3f}")

        if DEBUG:
            _save_debug(mat_path, azimuth, i,
                        I1_raw, img_tensor, heatmap,
                        cx_coarse, cy_coarse,
                        cx_refined, cy_refined,
                        scale, pad_left, pad_top,
                        orig_w, orig_h, image_size)

        rows.append({
            "azimuth":    azimuth,
            "filename":   os.path.basename(mat_path),
            "angle":      i,
            "cx_coarse":  round(cx_coarse,  2),
            "cy_coarse":  round(cy_coarse,  2),
            "cx":         round(cx_refined, 2),
            "cy":         round(cy_refined, 2),
            "confidence": round(conf, 4),
        })

    return rows


def _empty_row(azimuth, mat_path, angle_idx):
    return {
        "azimuth":   azimuth,
        "filename":  os.path.basename(mat_path),
        "angle":     angle_idx,
        "cx_coarse": None, "cy_coarse": None,
        "cx":        None, "cy":        None,
        "confidence": 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# DEBUG OUTPUT
# ═══════════════════════════════════════════════════════════════

def _save_debug(mat_path, azimuth, angle_idx,
                I1_raw, img_tensor, heatmap_tensor,
                cx_coarse, cy_coarse, cx_refined, cy_refined,
                scale, pad_left, pad_top, orig_w, orig_h, image_size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    stem = Path(mat_path).stem
    hs   = heatmap_tensor.squeeze().cpu().numpy().shape[0]
    hm   = heatmap_tensor.squeeze().cpu().numpy()
    img  = img_tensor.squeeze().cpu().numpy()    # (3, H, W)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    # Panel 0: raw I1 with refinement window + both centroids
    axes[0].imshow(I1_raw, cmap="gray")
    axes[0].set_title("I1 raw (refinement input)")
    win = REFINE_WINDOW_PX
    rect = mpatches.Rectangle(
        (cx_coarse - win, cy_coarse - win), 2*win, 2*win,
        linewidth=1, edgecolor="yellow", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].plot(cx_coarse,  cy_coarse,  "r+", markersize=12,
                 markeredgewidth=2, label="CNN coarse")
    axes[0].plot(cx_refined, cy_refined, "b+", markersize=12,
                 markeredgewidth=2, label="Refined")
    axes[0].legend(fontsize=7, loc="upper right")

    # Panels 1-3: model input channels
    axes[1].imshow(img[0], cmap="gray"); axes[1].set_title("I1 (LCN)")
    axes[2].imshow(img[1], cmap="gray"); axes[2].set_title("I2 (LCN)")
    axes[3].imshow(img[2], cmap="gray"); axes[3].set_title("|I2−I1| (LCN)")

    # Panel 4: predicted heatmap with both points
    axes[4].imshow(hm, cmap="hot")
    axes[4].set_title("Predicted heatmap")
    axes[4].set_xlim(0, hs); axes[4].set_ylim(hs, 0)

    # Map both points to heatmap space.
    # cx_o/cy_o are in original pixel space.
    # Pipeline: orig → (flip if needed) → letterbox → heatmap
    for (cx_o, cy_o), colour in [
        ((cx_coarse,  cy_coarse),  "r"), ((cx_refined, cy_refined), "b")
    ]:
        # Step 1: apply same horizontal flip that was applied during preprocessing
        cx_flipped = (orig_w - 1 - cx_o) if FLIP_HORIZONTAL else cx_o
        # Step 2: orig → letterboxed image space
        cx_lb_dbg = cx_flipped * scale + pad_left
        cy_lb_dbg = cy_o       * scale + pad_top
        # Step 3: letterboxed image → heatmap space
        cx_hm = cx_lb_dbg * hs / image_size
        cy_hm = cy_lb_dbg * hs / image_size
        axes[4].plot(cx_hm, cy_hm, f"{colour}+",
                     markersize=12, markeredgewidth=2)

    fig.suptitle(
        f"{stem}  |  {azimuth}°  |  angle {angle_idx}  |  "
        f"coarse=({cx_coarse:.1f},{cy_coarse:.1f})  "
        f"refined=({cx_refined:.1f},{cy_refined:.1f})"
    )
    fig.tight_layout()
    out = os.path.join(
        DEBUG_DIR, f"{stem}_{azimuth}deg_S{angle_idx}.png"
    )
    plt.savefig(out, dpi=100)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def collect_mat_files():
    collected = []
    for az in range(AZIMUTH_START, AZIMUTH_END + 1, AZIMUTH_STEP):
        subfolder = os.path.join(MAT_DIR, f"{az}Azimuth")
        if not os.path.isdir(subfolder):
            print(f"  [WARN] missing subfolder: {subfolder}")
            continue
        for path in sorted(glob.glob(os.path.join(subfolder, "*.mat"))):
            collected.append((az, path))
    return collected


def run():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: '{MODEL_PATH}'. "
            f"Train with train_pseudopupil.py first."
        )

    model, image_size, heatmap_size = load_model(MODEL_PATH)

    entries = collect_mat_files()
    if not entries:
        print(f"No .mat files found under '{MAT_DIR}'.")
        return

    # Summary
    az_counts = {}
    for az, _ in entries:
        az_counts[az] = az_counts.get(az, 0) + 1
    print(f"\nFound {len(entries)} files across {len(az_counts)} azimuth folders")
    print(f"FLIP_HORIZONTAL = {FLIP_HORIZONTAL}")
    print(f"Refinement: window={REFINE_WINDOW_PX}px  "
          f"percentile={REFINE_PERCENTILE}  "
          f"min_blob={REFINE_MIN_BLOB_PX}px\n")

    all_rows   = []
    current_az = None

    for idx, (az, path) in enumerate(entries, 1):
        if az != current_az:
            current_az = az
            print(f"\n── {az}Azimuth {'─'*40}")

        print(f"  [{idx:3d}/{len(entries)}] {os.path.basename(path)}")
        try:
            rows = process_file(path, model, image_size, heatmap_size, az)
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR: {e}")
            for i in range(N_ANGLES):
                all_rows.append(_empty_row(az, path, i))

    cols = ["azimuth", "filename", "angle",
            "cx_coarse", "cy_coarse", "cx", "cy", "confidence"]
    df = pd.DataFrame(all_rows)
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["azimuth", "filename", "angle"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'─'*60}")
    print(f"Done.  {len(df)} predictions  ({len(entries)} files × {N_ANGLES} angles)")
    print(f"Results → {OUTPUT_CSV}")

    # Per-azimuth summary
    print("\nMean confidence by azimuth:")
    print(df.groupby("azimuth")[["cx", "cy", "confidence"]]
            .mean().round(2).to_string())


if __name__ == "__main__":
    run()