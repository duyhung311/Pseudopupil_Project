"""
Pseudopupil Heatmap Regression — Training Script
=================================================
Trains an EfficientNetB0-based CNN to predict a 2D Gaussian heatmap centred
on the pseudopupil (x, y) coordinate.

Input per sample  : 3-channel [I1_lcn, I2_lcn, abs(I2-I1)_lcn]
Ground truth      : 2D Gaussian heatmap at (cx, cy)
Output            : pseudopupil_model.pth  +  training_log.csv

JSON label format:
  [
    {
      "mat_file": "0Azimuth/sample001.mat",
      "angle":    "imagesS0",
      "cx":       142.5,
      "cy":       98.3,
      "exposure": "both"          <- "both" | "i1" | "i2"  (optional, default "both")
    },
    ...
  ]

Dependencies:
  pip install torch torchvision timm albumentations scipy scikit-image pandas opencv-python-headless
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
import cv2


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SCIENTIFIC_NAME = "Aeschna_isoceles"
MAT_DIR         = f"./{SCIENTIFIC_NAME}"
LABEL_JSON      = "./util/labels_expanded2.json"
MODEL_OUT       = "./pseudopupil_model.pth"
LOG_OUT         = "./training_log.csv"

IMAGE_SIZE    = 256
HEATMAP_SIZE  = 64
GAUSS_SIGMA   = 1.5
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.10
PATIENCE      = 12
NUM_WORKERS   = 4
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG     = False
DEBUG_DIR = "./debug_img/"


# ─────────────────────────────────────────────
# SHARED PREPROCESSING  (identical logic to infer script)
# ─────────────────────────────────────────────

def local_contrast_norm(img: np.ndarray, sigma: int = 32) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img.astype(np.float32), size=sigma)
    # Gate: only normalise pixels above a minimum mean threshold
    # Below threshold, treat as background (output 0)
    # This prevents noise amplification in very dark regions
    gate = (local_mean > 0.02).astype(np.float32)
    normed = (img / (local_mean + 0.02)) * gate
    return rescale_intensity(normed, out_range=(0.0, 1.0)).astype(np.float32)

def load_raw_images(mat_data: dict, angle_key: str):
    """
    Extract I1 (low) and I2 (high) from .mat struct.
    Returns (I1, I2) as float32 in [0,1]. Identical to infer script.
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
    Build (H, W, 3) input array. Identical logic to infer script.
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


def letterbox(img: np.ndarray, size: int):
    """Resize to (size,size) preserving aspect, zero-pad. Identical to infer."""
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


def orig_to_letterbox(cx, cy, orig_w, orig_h, size):
    scale    = size / max(orig_w, orig_h)
    pad_left = (size - int(round(orig_w * scale))) // 2
    pad_top  = (size - int(round(orig_h * scale))) // 2
    return cx * scale + pad_left, cy * scale + pad_top, scale, pad_left, pad_top


def make_heatmap(cx_lb: float, cy_lb: float) -> np.ndarray:
    ratio = HEATMAP_SIZE / IMAGE_SIZE
    cx, cy = cx_lb * ratio, cy_lb * ratio
    xs = np.arange(HEATMAP_SIZE, dtype=np.float32)
    ys = np.arange(HEATMAP_SIZE, dtype=np.float32)
    xg, yg  = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xg-cx)**2 + (yg-cy)**2) / (2 * GAUSS_SIGMA**2))
    return (heatmap / (heatmap.max() + 1e-8)).astype(np.float32)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class PseudopupilDataset(Dataset):
    def __init__(self, labels, mat_dir, transform=None):
        self.labels   = labels
        self.mat_dir  = mat_dir
        geo, colour   = transform if transform is not None else (None, None)
        self.geo_transform    = geo
        self.colour_transform = colour

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry    = self.labels[idx]
        mat_path = os.path.join(self.mat_dir, entry["mat_file"])
        angle    = entry["angle"]
        cx, cy   = float(entry["cx"]), float(entry["cy"])
        exposure = entry.get("exposure", "both")

        mat_data       = sio.loadmat(mat_path)
        I1_raw, I2_raw = load_raw_images(mat_data, angle)
        orig_h, orig_w = I1_raw.shape[:2]

        img_hwc = build_channels(I1_raw, I2_raw, exposure)
        img_lb, _, _, _ = letterbox(img_hwc, IMAGE_SIZE)
        cx_lb, cy_lb, _, _, _ = orig_to_letterbox(cx, cy, orig_w, orig_h, IMAGE_SIZE)

        # Geometric aug (pre-crop)
        if self.geo_transform is not None:
            t = self.geo_transform(
                image=img_lb, keypoints=[(cx_lb, cy_lb)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            img_lb = t["image"]
            kps    = t["keypoints"]
            cx_lb, cy_lb = kps[0] if kps else (cx_lb, cy_lb)

        # Patch-crop (breaks position memorisation)
        if self.geo_transform is not None and np.random.rand() < 0.70:
            img_lb, cx_lb, cy_lb = self._patch_crop(img_lb, cx_lb, cy_lb)

        # Colour aug (post-crop)
        if self.colour_transform is not None:
            t = self.colour_transform(
                image=img_lb, keypoints=[(cx_lb, cy_lb)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            img_lb = t["image"]
            kps    = t["keypoints"]
            cx_t, cy_t = kps[0] if kps else (cx_lb, cy_lb)
        else:
            cx_t, cy_t = cx_lb, cy_lb

        cx_t = float(np.clip(cx_t, 0, IMAGE_SIZE - 1))
        cy_t = float(np.clip(cy_t, 0, IMAGE_SIZE - 1))

        img_tensor     = torch.from_numpy(img_lb.transpose(2, 0, 1)).float()
        heatmap_tensor = torch.from_numpy(make_heatmap(cx_t, cy_t)).unsqueeze(0)
        coord_tensor   = torch.tensor([cx_t, cy_t], dtype=torch.float32)
        return img_tensor, heatmap_tensor, coord_tensor

    def _patch_crop(self, img_lb, cx, cy, patch_size=180):
        """Crop patch_size x patch_size around label with +-50% jitter."""
        h, w  = img_lb.shape[:2]
        half  = patch_size // 2
        max_j = int(patch_size * 0.50)
        jx    = np.random.randint(-max_j, max_j)
        jy    = np.random.randint(-max_j, max_j)
        cx_c  = int(np.clip(cx + jx, half, w - half))
        cy_c  = int(np.clip(cy + jy, half, h - half))
        x0, y0 = cx_c - half, cy_c - half
        crop   = img_lb[y0:y0 + patch_size, x0:x0 + patch_size]
        resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        scale  = IMAGE_SIZE / patch_size
        new_cx = (cx - x0) * scale
        new_cy = (cy - y0) * scale
        if not (5 < new_cx < IMAGE_SIZE - 5 and 5 < new_cy < IMAGE_SIZE - 5):
            return img_lb, cx, cy
        return resized, float(new_cx), float(new_cy)


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────

def get_transforms(train: bool):
    if not train:
        return None, None
    geometric = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.10,
                           rotate_limit=25, border_mode=0, p=0.70),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    colour = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(0.001, 0.015), p=0.4),
        A.RandomGamma(gamma_limit=(70, 150), p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.3),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    return geometric, colour


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class PseudopupilHeatmapNet(nn.Module):
    """EfficientNetB0 encoder + decoder. 3ch input matches ImageNet weights."""
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b0", pretrained=True,
            features_only=True, out_indices=(1, 2, 3, 4),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(320, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d((HEATMAP_SIZE, HEATMAP_SIZE))

    def forward(self, x):
        return self.pool(self.decoder(self.encoder(x)[-1]))


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss — large gradients at peak, stable at background."""
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0,
                 alpha=2.1, focal_weight=50.0):
        super().__init__()
        self.omega, self.theta = omega, theta
        self.epsilon, self.alpha = epsilon, alpha
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        A = self.omega * (
            1 / (1 + torch.pow(self.theta / (self.epsilon + target),
                               self.alpha - target))
        ) * (self.alpha - target) * torch.pow(
            self.theta / (self.epsilon + target), self.alpha - target - 1
        ) * (1 / (self.epsilon + target))
        C    = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target)
        )
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + torch.pow(diff / self.epsilon,
                                                  self.alpha - target)),
            A * diff - C
        )
        focal = (target > 0.1).float() * self.focal_weight + 1.0
        return (loss * focal).mean()


# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────

def soft_argmax(heatmap_batch, image_size, heatmap_size):
    hs = heatmap_size
    hm = heatmap_batch.squeeze(1)
    hm = hm / (hm.sum(dim=(-2,-1), keepdim=True) + 1e-8)
    xs = torch.arange(hs, dtype=torch.float32, device=hm.device)
    ys = torch.arange(hs, dtype=torch.float32, device=hm.device)
    xg, yg = torch.meshgrid(xs, ys, indexing="xy")
    cx = (hm * xg).sum(dim=(-2,-1)) * (image_size / hs)
    cy = (hm * yg).sum(dim=(-2,-1)) * (image_size / hs)
    return cx, cy


def coord_regression_loss(pred_heatmap, cx_gt, cy_gt):
    """L1 on soft-argmax coords vs ground truth — both in IMAGE_SIZE space."""
    px, py = soft_argmax(pred_heatmap, IMAGE_SIZE, HEATMAP_SIZE)
    return (nn.functional.l1_loss(px, cx_gt) +
            nn.functional.l1_loss(py, cy_gt))


# ─────────────────────────────────────────────
# SPLIT  (by mat file to prevent leakage)
# ─────────────────────────────────────────────

def stratified_split_by_file(labels, val_fraction=0.10):
    """
    Group all samples belonging to the same mat file together,
    then split whole files into train/val.
    Prevents I1 and I2 of the same file leaking across the split.
    """
    by_file = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_file[lbl["mat_file"]].append(i)
    files       = list(by_file.keys())
    np.random.shuffle(files)
    n_val_files = max(1, int(len(files) * val_fraction))
    val_files   = set(files[:n_val_files])
    train_labels = [labels[i] for f in files
                    if f not in val_files for i in by_file[f]]
    val_labels   = [labels[i] for f in val_files for i in by_file[f]]
    return train_labels, val_labels


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(LABEL_JSON) as f:
        all_labels = json.load(f)
    print(f"Loaded {len(all_labels)} samples from {LABEL_JSON}")

    train_labels, val_labels = stratified_split_by_file(all_labels, VAL_SPLIT)
    n_train, n_val = len(train_labels), len(val_labels)
    print(f"  Train: {n_train}  |  Val: {n_val}  (split by mat file)")

    train_ds = PseudopupilDataset(train_labels, MAT_DIR, get_transforms(True))
    val_ds   = PseudopupilDataset(val_labels,   MAT_DIR, get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = PseudopupilHeatmapNet().to(DEVICE)
    criterion = AdaptiveWingLoss(focal_weight=50.0)   # inside train(), not module level
    optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS,
                                                      eta_min=1e-6)

    best_val_loss    = float("inf")
    patience_counter = 0
    log_rows         = []

    print(f"\nTraining on {DEVICE}  |  {NUM_EPOCHS} epochs  |  batch {BATCH_SIZE}\n")
    print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  {'LR':>10}")
    print("─" * 46)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, heatmaps, coords in train_loader:
            imgs, heatmaps, coords = (imgs.to(DEVICE), heatmaps.to(DEVICE),
                                      coords.to(DEVICE))
            optimiser.zero_grad()
            preds  = model(imgs)
            loss   = (criterion(preds, heatmaps) +
                      0.1 * coord_regression_loss(preds, coords[:,0], coords[:,1]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _ in val_loader:
                imgs, heatmaps = imgs.to(DEVICE), heatmaps.to(DEVICE)
                val_loss += criterion(model(imgs), heatmaps).item() * imgs.size(0)
        val_loss /= n_val

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6}  {train_loss:>12.5f}  {val_loss:>10.5f}  {lr:>10.2e}")
        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "lr": lr})

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_loss": best_val_loss,
                "image_size": IMAGE_SIZE, "heatmap_size": HEATMAP_SIZE,
                "gauss_sigma": GAUSS_SIGMA,
            }, MODEL_OUT)
            print(f"         ↳ saved best model (val_loss={best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Post-training diagnostic
    print("\nVal-set error diagnostic...")
    errors = []
    model.eval()
    with torch.no_grad():
        for imgs, _, coords in val_loader:
            preds = model(imgs.to(DEVICE))
            cx_p, cy_p = soft_argmax(preds, IMAGE_SIZE, HEATMAP_SIZE)
            for i in range(imgs.size(0)):
                errors.append({
                    "err_x": (cx_p[i].cpu() - coords[i,0]).item(),
                    "err_y": (cy_p[i].cpu() - coords[i,1]).item(),
                })
    df = pd.DataFrame(errors)
    print(df[["err_x","err_y"]].describe().round(2))
    print(f"\nStd  err_x={df.err_x.std():.1f}px  err_y={df.err_y.std():.1f}px")
    print(f"Mean err_x={df.err_x.mean():.1f}px  err_y={df.err_y.mean():.1f}px")

    pd.DataFrame(log_rows).to_csv(LOG_OUT, index=False)
    print(f"\nDone. Best val_loss={best_val_loss:.5f}")
    print(f"Model → {MODEL_OUT}  |  Log → {LOG_OUT}")


if __name__ == "__main__":
    train()