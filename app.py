import io
import json
import base64
import os
import glob
import csv
import re
import threading
import uuid as uuid_module
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from scipy.ndimage import (gaussian_filter, binary_dilation, binary_erosion,
                           label, distance_transform_edt, uniform_filter)

# ---------------------------------------------------------------------------
# Optional GPU acceleration (PyTorch MPS on Apple Silicon / CUDA on servers)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as _F
    if torch.backends.mps.is_available():
        _TORCH_DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda")
    else:
        _TORCH_DEVICE = torch.device("cpu")
    _HAS_TORCH = _TORCH_DEVICE.type != "cpu"
    print(f"PyTorch device: {_TORCH_DEVICE}")
except ImportError:
    _HAS_TORCH = False
    _TORCH_DEVICE = None


def _gaussian_filter_gpu(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via PyTorch on MPS/CUDA; fallback to scipy."""
    if not _HAS_TORCH or sigma <= 0:
        return gaussian_filter(channel, sigma) if sigma > 0 else channel
    ks = int(4 * sigma + 1) | 1  # odd kernel size
    ks = max(3, ks)
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    k1 = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    t = torch.from_numpy(channel).float().unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE)
    kh = k1.view(1, 1, -1, 1).to(_TORCH_DEVICE)
    kw = k1.view(1, 1, 1, -1).to(_TORCH_DEVICE)
    pad = ks // 2
    t = _F.pad(t, (0, 0, pad, pad), mode='reflect')
    t = _F.conv2d(t, kh)
    t = _F.pad(t, (pad, pad, 0, 0), mode='reflect')
    t = _F.conv2d(t, kw)
    return t.squeeze().cpu().numpy()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load shared settings.json (provides defaults for all params)
# ---------------------------------------------------------------------------
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")
try:
    with open(_SETTINGS_PATH) as _sf:
        SETTINGS = json.load(_sf)
    print(f"Loaded settings from {_SETTINGS_PATH}")
except FileNotFoundError:
    SETTINGS = {}
    print("No settings.json found — using hardcoded defaults")

# Batch job tracking: job_id -> state dict
_batch_jobs: dict = {}

IMAGE_DIR = os.path.dirname(__file__) + "/macro_images"
# Discover available macro images
IMAGE_FILES = sorted([os.path.basename(p) for p in glob.glob(os.path.join(IMAGE_DIR, "macro*.tif"))])
if not IMAGE_FILES:
    # Fallback to explicit name if none found
    IMAGE_FILES = ["macro1.tif"]

DEFAULT_IMAGE = IMAGE_FILES[0]

def _image_fullpath(name: str) -> str:
    if not name:
        name = DEFAULT_IMAGE
    # only allow images we discovered
    if os.path.basename(name) not in IMAGE_FILES:
        name = DEFAULT_IMAGE
    return os.path.join(IMAGE_DIR, os.path.basename(name))

print(f"Loading default image {DEFAULT_IMAGE}…")
_orig_pil = Image.open(_image_fullpath(DEFAULT_IMAGE)).convert("RGB")
ORIG = np.array(_orig_pil, dtype=np.uint8)
H, W = ORIG.shape[:2]
print(f"Image loaded: {W}x{H}")

PREVIEW_SCALE = 0.40
_small_pil = _orig_pil.resize((int(W * PREVIEW_SCALE), int(H * PREVIEW_SCALE)), Image.LANCZOS)
SMALL = np.array(_small_pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Colour space utilities
# ---------------------------------------------------------------------------

def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / maxc, 0.0)
    h = np.zeros_like(r)
    m = delta > 0
    rm, gm, bm = m & (maxc == r), m & (maxc == g), m & (maxc == b)
    h[rm] = (60 * ((g[rm] - b[rm]) / delta[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / delta[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / delta[bm]) + 240
    return h, s, v


def _lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _gamma(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)

def _f(t: np.ndarray) -> np.ndarray:
    d = 6.0 / 29.0
    return np.where(t > d ** 3, t ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)

def _f_inv(t: np.ndarray) -> np.ndarray:
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))

def rgb_to_lab(f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f32: float32 [0,1] RGB → L* [0,100], a*, b*"""
    r, g, b = _lin(f32[..., 0]), _lin(f32[..., 1]), _lin(f32[..., 2])
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx, fy, fz = _f(X / 0.95047), _f(Y), _f(Z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def lab_to_rgb_u8(L: np.ndarray, a: np.ndarray, b_ch: np.ndarray) -> np.ndarray:
    fy = (L + 16) / 116
    X = 0.95047 * _f_inv(a / 500 + fy)
    Y =           _f_inv(fy)
    Z = 1.08883 * _f_inv(fy - b_ch / 200)
    r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bv =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([_gamma(np.clip(r, 0, 1)),
                    _gamma(np.clip(g, 0, 1)),
                    _gamma(np.clip(bv, 0, 1))], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------

def build_mask(arr: np.ndarray, params: dict) -> np.ndarray:
    sat_threshold = float(params.get("sat_threshold", 0.08))
    val_min       = float(params.get("val_min",       0.00))
    val_max       = float(params.get("val_max",       1.00))
    blur_sigma    = float(params.get("blur_sigma",    0.0))
    erode_iter    = int(params.get("erode_iter",      0))
    dilate_iter   = int(params.get("dilate_iter",     0))
    mask_mode     = params.get("mask_mode", "hsv_sat")
    hue_center    = float(params.get("hue_center", 0.0))
    hue_width     = float(params.get("hue_width",  60.0))

    h, s, v = rgb_to_hsv(arr)

    if mask_mode == "lab_chroma":
        # LAB C* = sqrt(a²+b²); normalised to ~[0,1] (C* max ≈ 128)
        f = arr.astype(np.float32) / 255.0
        _, a_ch, b_ch = rgb_to_lab(f)
        score = np.sqrt(a_ch ** 2 + b_ch ** 2) / 128.0
    elif mask_mode == "hue_range":
        # Angular distance from hue_center; then use HSV sat as score within range
        hue_diff = np.abs(((h - hue_center + 180) % 360) - 180)
        in_range = (hue_diff <= (hue_width / 2)).astype(np.float32)
        score = s * in_range
    elif mask_mode == "combined":
        # Union: pixel selected if either HSV-sat OR LAB-chroma exceeds threshold
        f = arr.astype(np.float32) / 255.0
        _, a_ch, b_ch = rgb_to_lab(f)
        chroma_n = np.sqrt(a_ch ** 2 + b_ch ** 2) / 128.0
        score = np.maximum(s, chroma_n)
    elif mask_mode == "delta_e":
        # CIE76 ΔE from auto-sampled background: works for gray/achromatic artwork
        # whose saturation is near zero but whose lightness differs from the substrate.
        # Background = median LAB of the 10% most neutral (lowest-chroma) pixels.
        f = arr.astype(np.float32) / 255.0
        L_arr, a_arr, b_arr = rgb_to_lab(f)
        chroma = np.sqrt(a_arr ** 2 + b_arr ** 2)
        n_bg = max(100, chroma.size // 10)
        idx = np.argpartition(chroma.ravel(), n_bg)[:n_bg]
        bg_L = float(np.median(L_arr.ravel()[idx]))
        bg_a = float(np.median(a_arr.ravel()[idx]))
        bg_b = float(np.median(b_arr.ravel()[idx]))
        delta_e = np.sqrt((L_arr - bg_L) ** 2 + (a_arr - bg_a) ** 2 + (b_arr - bg_b) ** 2)
        score = delta_e / 50.0  # normalise: ΔE 5 → 0.10, ΔE 15 → 0.30
    elif mask_mode == "channel_diff":
        # Per-channel deviation from neutral grey (mean of channels).
        # More stable than HSV sat for very bright/dark pixels.
        # max possible value = 2/3 → normalise to 0-1 by ×1.5
        f = arr.astype(np.float32) / 255.0
        mean_c = (f[..., 0] + f[..., 1] + f[..., 2]) / 3.0
        score = np.maximum(
            np.maximum(np.abs(f[..., 0] - mean_c), np.abs(f[..., 1] - mean_c)),
            np.abs(f[..., 2] - mean_c),
        ) * 1.5
    else:  # "hsv_sat"
        score = s

    mask = (score > sat_threshold) & (v > val_min) & (v < val_max)

    if blur_sigma > 0:
        mask = gaussian_filter(mask.astype(np.float32), blur_sigma) > 0.4
    if erode_iter > 0:
        mask = binary_erosion(mask, iterations=erode_iter)
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Strategy A1 – LAB neutral grey
# ---------------------------------------------------------------------------

def remove_lab_neutral(arr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Zero out a* and b* inside the mask, keeping L* (perceptually accurate grey)."""
    f = arr.astype(np.float32) / 255.0
    L, a, b_ch = rgb_to_lab(f)
    a_new  = a.copy();   a_new[mask]  = a[mask]  * (1 - strength)
    b_new  = b_ch.copy(); b_new[mask] = b_ch[mask] * (1 - strength)
    return lab_to_rgb_u8(L, a_new, b_new)


# ---------------------------------------------------------------------------
# Strategy A2 – Reference-tone matching
# ---------------------------------------------------------------------------

def sample_reference_tone(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the median RGB of clean (non-masked, mid-tone) diaper pixels."""
    grey = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    candidates = ~mask & (grey > 40) & (grey < 240)
    if candidates.sum() < 50:
        candidates = ~mask
    return np.median(arr[candidates].astype(np.float32), axis=0)  # shape (3,)


def remove_reference_tone(arr: np.ndarray, mask: np.ndarray,
                          ref_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Keep L*, but pull a* and b* towards the reference material tone."""
    f    = arr.astype(np.float32) / 255.0
    fref = ref_rgb.reshape(1, 1, 3) / 255.0
    L,    a,    b_ch    = rgb_to_lab(f)
    _, ref_a, ref_b_ch  = rgb_to_lab(np.broadcast_to(fref, f.shape))

    a_new  = a.copy();   a_new[mask]  = a[mask]  + strength * (ref_a[mask]    - a[mask])
    b_new  = b_ch.copy(); b_new[mask] = b_ch[mask] + strength * (ref_b_ch[mask] - b_ch[mask])
    return lab_to_rgb_u8(L, a_new, b_new)


# ---------------------------------------------------------------------------
# Strategy A3 – Frequency-layer separation
# ---------------------------------------------------------------------------

def remove_freq_separation(arr: np.ndarray, mask: np.ndarray,
                           freq_sigma: float, use_a2: bool,
                           ref_rgb: np.ndarray | None, strength: float) -> np.ndarray:
    """
    Split image into low-freq (colour/tone) and high-freq (texture/wrinkles).
    Replace the low-freq colour inside the mask; recombine with original high-freq.
    """
    f = arr.astype(np.float32)

    low = np.stack(
        [gaussian_filter(f[..., c], freq_sigma) for c in range(3)], axis=-1
    )
    high = f - low

    low_u8 = np.clip(low, 0, 255).astype(np.uint8)
    if use_a2 and ref_rgb is not None:
        neutral_low = remove_reference_tone(low_u8, mask, ref_rgb, 1.0)
    else:
        neutral_low = remove_lab_neutral(low_u8, mask, 1.0)

    new_low = low.copy()
    for c in range(3):
        new_low[mask, c] = (
            (1 - strength) * low[mask, c]
            + strength * neutral_low[mask, c].astype(np.float32)
        )

    return np.clip(new_low + high, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Strategy A4 – Edge-pixel colour transfer
# ---------------------------------------------------------------------------

def _px_to_lab(px: np.ndarray) -> np.ndarray:
    """(N, 3) float [0-255] → (N, 3) LAB"""
    L, a, b = rgb_to_lab(px / 255.0)
    return np.stack([L, a, b], axis=-1)

def _px_from_lab(lab: np.ndarray) -> np.ndarray:
    """(N, 3) LAB → (N, 3) float [0-255]"""
    return lab_to_rgb_u8(lab[:, 0], lab[:, 1], lab[:, 2]).astype(np.float32)

def _mat_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

def _mat_inv_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T


def remove_edge_transfer(arr: np.ndarray, mask: np.ndarray,
                         params: dict, pixel_scale: float = 1.0) -> np.ndarray:
    """
    For each connected island in the mask:
      1. Sample border pixels (within border_width px outside the island).
         Optionally exclude saturated border pixels (clean_border) to avoid
         sampling adjacent artwork as reference.
      2. Build a colour-distribution model of those border pixels.
      3. Map the island pixels so their distribution matches the border's.
         transform_mode: 'mean' | 'mean_std' | 'covariance' (Monge-Kantorovich OT)
      4. Optionally lock L* (lock_luma) so only a*/b* are changed.
      5. Repeat for n_passes refinement iterations.
    """
    border_width     = max(1, round(float(params.get("et_border_width",   10)) * pixel_scale))
    transform_mode   = params.get("et_transform_mode",  "mean_std")
    colorspace       = params.get("et_colorspace",       "lab")
    blend_strength   = float(params.get("et_blend_strength",  1.0))
    min_border_px    = int(params.get("et_min_border_px",  20))
    min_island_px    = int(params.get("et_min_island_px",   5))
    n_passes         = max(1, int(params.get("et_passes",         1)))
    lock_luma        = bool(params.get("et_lock_luma",        False))
    clean_border     = bool(params.get("et_clean_border",     False))
    clean_border_sat = float(params.get("et_clean_border_sat", 0.15))

    labeled, n_islands = label(mask)
    current = arr.copy().astype(np.float32)

    for _ in range(n_passes):
        pass_result = current.copy()

        for island_id in range(1, n_islands + 1):
            island_mask = labeled == island_id
            n_island = int(island_mask.sum())
            if n_island < min_island_px:
                continue

            # Border = dilated island minus the entire mask
            dilated     = binary_dilation(island_mask, iterations=border_width)
            border_mask = dilated & ~mask
            n_border    = int(border_mask.sum())
            if n_border < min_border_px:
                continue

            border_px = arr[border_mask].astype(np.float32)   # always from original
            island_px = current[island_mask].astype(np.float32)  # from current pass

            # Exclude saturated pixels from the border reference
            if clean_border:
                _, b_sat, _ = rgb_to_hsv(border_px)
                keep = b_sat < clean_border_sat
                if keep.sum() >= min_border_px:
                    border_px = border_px[keep]

            # Convert to working colour space
            if colorspace == "lab":
                border_cs = _px_to_lab(border_px)
                island_cs = _px_to_lab(island_px)
            else:
                border_cs = border_px.copy()
                island_cs = island_px.copy()

            mu_ref = border_cs.mean(axis=0)
            mu_src = island_cs.mean(axis=0)

            if transform_mode == "mean":
                transformed = island_cs - mu_src + mu_ref

            elif transform_mode == "mean_std":
                std_ref = border_cs.std(axis=0) + 1e-6
                std_src = island_cs.std(axis=0) + 1e-6
                transformed = (island_cs - mu_src) / std_src * std_ref + mu_ref

            else:  # covariance — Monge-Kantorovich optimal transport
                n_b = len(border_px)
                if n_b >= 4 and n_island >= 4:
                    cov_src = np.cov(island_cs.T) + np.eye(3) * 1e-6
                    cov_ref = np.cov(border_cs.T) + np.eye(3) * 1e-6
                    A = _mat_sqrt(cov_ref) @ _mat_inv_sqrt(cov_src)
                    transformed = (island_cs - mu_src) @ A.T + mu_ref
                else:
                    transformed = island_cs - mu_src + mu_ref

            # Optionally lock L* — only shift chrominance (a*, b*)
            if lock_luma and colorspace == "lab":
                transformed[:, 0] = island_cs[:, 0]

            # Back to RGB [0-255] float
            if colorspace == "lab":
                transformed = _px_from_lab(np.clip(transformed,
                                                    [-16, -128, -128], [100, 127, 127]))
            transformed = np.clip(transformed, 0, 255)

            for c in range(3):
                pass_result[island_mask, c] = (
                    (1.0 - blend_strength) * current[island_mask, c]
                    + blend_strength * transformed[:, c]
                )

        current = pass_result

    return np.clip(current, 0, 255).astype(np.uint8)


def remove_a4_radiant(arr: np.ndarray, mask: np.ndarray,
                      params: dict, pixel_scale: float = 1.0) -> np.ndarray:
    """
    Outside-in radiant fill.

    For every masked pixel find the nearest non-masked pixel in each of the
    four cardinal directions (N, S, E, W).  Blend the four border colours
    using inverse-distance weights (closer border = more influence).

    Fully vectorised — no Python loops over pixels.
    """
    H, W = mask.shape
    non_mask = ~mask
    result = arr.copy().astype(np.float32)

    y_g = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    x_g = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    r_g = np.broadcast_to(np.arange(H, dtype=np.int32)[:, None], (H, W))
    c_g = np.broadcast_to(np.arange(W, dtype=np.int32)[None, :], (H, W))

    # LEFT  — scan left→right; track nearest non-masked column
    lx = np.where(non_mask, x_g, -np.inf)
    lx = np.maximum.accumulate(lx, axis=1)
    l_dist = np.where(mask & (lx >= 0), x_g - lx, np.inf).astype(np.float32)
    l_vals = arr[r_g, np.clip(np.where(np.isfinite(lx), lx, 0).astype(np.int32), 0, W-1)].astype(np.float32)

    # RIGHT — scan right→left
    rx = np.where(non_mask, x_g, np.inf)
    rx = np.minimum.accumulate(rx[:, ::-1], axis=1)[:, ::-1]
    r_dist = np.where(mask & (rx < W), rx - x_g, np.inf).astype(np.float32)
    r_vals = arr[r_g, np.clip(np.where(np.isfinite(rx), rx, W-1).astype(np.int32), 0, W-1)].astype(np.float32)

    # UP    — scan top→bottom; track nearest non-masked row
    uy = np.where(non_mask, y_g, -np.inf)
    uy = np.maximum.accumulate(uy, axis=0)
    u_dist = np.where(mask & (uy >= 0), y_g - uy, np.inf).astype(np.float32)
    u_vals = arr[np.clip(np.where(np.isfinite(uy), uy, 0).astype(np.int32), 0, H-1), c_g].astype(np.float32)

    # DOWN  — scan bottom→top
    dy = np.where(non_mask, y_g, np.inf)
    dy = np.minimum.accumulate(dy[::-1, :], axis=0)[::-1, :]
    d_dist = np.where(mask & (dy < H), dy - y_g, np.inf).astype(np.float32)
    d_vals = arr[np.clip(np.where(np.isfinite(dy), dy, H-1).astype(np.int32), 0, H-1), c_g].astype(np.float32)

    # Inverse-distance weights; zero when no valid neighbour in that direction
    def _inv(d):
        return np.where(np.isfinite(d) & (d > 0), 1.0 / d, 0.0)

    wl, wr, wu, wd = _inv(l_dist), _inv(r_dist), _inv(u_dist), _inv(d_dist)
    total_w = wl + wr + wu + wd
    valid = mask & (total_w > 0)

    for c in range(3):
        blended = (wl * l_vals[..., c] + wr * r_vals[..., c]
                   + wu * u_vals[..., c] + wd * d_vals[..., c])
        result[valid, c] = blended[valid] / total_w[valid]

    blend = float(params.get("et_blend_strength", 1.0))
    if blend < 1.0:
        orig = arr.astype(np.float32)
        result[valid] = (1 - blend) * orig[valid] + blend * result[valid]

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_a4_propagate(arr: np.ndarray, mask: np.ndarray,
                        params: dict, pixel_scale: float = 1.0) -> np.ndarray:
    """
    Outside-in propagation fill (diffusion inpainting).

    Works from the mask boundary inward in EDT distance layers.
    At each distance level d, every frontier pixel's colour is set to the
    3×3 weighted average of its already-known (non-masked / filled) neighbours.
    Optionally blends with the pixel's own original value for a smooth fade.
    """
    result = arr.copy().astype(np.float32)
    known = ~mask.copy()

    dist = distance_transform_edt(mask).astype(np.int32)
    max_dist = int(dist.max())
    if max_dist == 0:
        return result.astype(np.uint8)

    blend = float(params.get("et_blend_strength", 1.0))

    for d in range(1, max_dist + 1):
        frontier = dist == d
        if not frontier.any():
            break

        known_f = known.astype(np.float32)
        for c in range(3):
            val_sum = uniform_filter(result[..., c] * known_f, 3, mode='nearest') * 9
            cnt     = uniform_filter(known_f,                  3, mode='nearest') * 9
            filled = val_sum[frontier] / np.maximum(cnt[frontier], 1.0)
            if blend < 1.0:
                filled = (1 - blend) * arr[frontier, c].astype(np.float32) + blend * filled
            result[frontier, c] = filled

        known |= frontier

    return np.clip(result, 0, 255).astype(np.uint8)


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)


def apply_filters(arr: np.ndarray, mask: np.ndarray, filters: list) -> np.ndarray:
    """
    Sequentially apply each filter over masked pixels.

    Each filter dict supports:
      color      hex string  colour tint
      opacity    0-1         colour tint blend
      brightness float       multiplicative brightness (1.0 = no change)
      contrast   float       contrast around mid-grey (1.0 = no change)
      saturation float       saturation scale (0 = grey, 1 = unchanged, >1 = boost)
    """
    if not filters:
        return arr
    result = arr.copy().astype(np.float32)
    for filt in filters:
        color      = hex_to_rgb(filt.get("color", "#ffffff"))
        opacity    = max(0.0, min(1.0, float(filt.get("opacity", 0.0))))
        brightness = float(filt.get("brightness", 1.0))
        contrast   = float(filt.get("contrast",   1.0))
        saturation = float(filt.get("saturation", 1.0))

        if opacity > 0:
            for c in range(3):
                result[mask, c] = (1 - opacity) * result[mask, c] + opacity * color[c]

        if abs(brightness - 1.0) > 1e-4:
            result[mask] *= brightness

        if abs(contrast - 1.0) > 1e-4:
            result[mask] = contrast * (result[mask] - 128.0) + 128.0

        if abs(saturation - 1.0) > 1e-4:
            px = result[mask]
            grey = (0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2])[:, np.newaxis]
            result[mask] = grey + saturation * (px - grey)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_strategy(arr: np.ndarray, mask: np.ndarray, params: dict,
                   pixel_scale: float = 1.0) -> np.ndarray:
    use_a2    = bool(params.get("use_a2", False))
    use_a3    = bool(params.get("use_a3", False))
    use_a4    = bool(params.get("use_a4", False))
    a4_mode   = params.get("et_a4_mode", "edge_transfer")
    strength  = float(params.get("removal_strength", 1.0))
    sigma     = float(params.get("freq_sigma", 20.0)) * pixel_scale
    filters   = params.get("filters", [])

    if use_a4:
        if a4_mode == "radiant":
            result = remove_a4_radiant(arr, mask, params, pixel_scale)
        elif a4_mode == "propagate":
            result = remove_a4_propagate(arr, mask, params, pixel_scale)
        else:
            result = remove_edge_transfer(arr, mask, params, pixel_scale)
    elif use_a3:
        ref_rgb = sample_reference_tone(arr, mask) if use_a2 else None
        result  = remove_freq_separation(arr, mask, sigma, use_a2, ref_rgb, strength)
    elif use_a2:
        ref_rgb = sample_reference_tone(arr, mask)
        result  = remove_reference_tone(arr, mask, ref_rgb, strength)
    else:
        result = remove_lab_neutral(arr, mask, strength)

    return apply_filters(result, mask, filters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def array_to_b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def apply_gamma(arr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return arr
    return np.clip((arr.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    # Provide available images to the template so the user can choose
    return render_template("index.html", width=W, height=H,
                           preview_scale=PREVIEW_SCALE,
                           images=IMAGE_FILES,
                           current_image=os.path.basename(DEFAULT_IMAGE))


@app.route("/preview", methods=["POST"])
def preview():
    params = request.get_json(force=True) or {}
    # Allow client to specify which image to use for preview
    image_name = params.get("image") if isinstance(params.get("image"), str) else None
    image_path = _image_fullpath(image_name)
    # Load selected image at requested render size instead of using the global _orig_pil
    render_pct = max(0.05, min(1.0, float(params.get("render_pct", 100)) / 100.0))

    tw = max(1, int(W * render_pct))
    th = max(1, int(H * render_pct))
    # Open image fresh for the preview (keeps memory/state simple when switching images)
    with Image.open(image_path) as pil_img:
        arr = np.array(pil_img.convert("RGB").resize((tw, th), Image.LANCZOS), dtype=np.uint8)

    mask   = build_mask(arr, params)
    result = apply_strategy(arr, mask, params, pixel_scale=render_pct)
    result = apply_gamma(result, float(params.get("gamma", 1.0)))

    overlay = arr.copy()
    overlay[mask] = np.clip(
        overlay[mask].astype(np.float32) * np.array([1.0, 0.3, 0.3]), 0, 255
    ).astype(np.uint8)

    return jsonify({
        "original": array_to_b64_png(arr),
        "mask":     array_to_b64_png(overlay),
        "result":   array_to_b64_png(result),
    })


@app.route("/download", methods=["POST"])
def download():
    params = request.get_json(force=True) or {}
    print("Processing full-resolution image…")
    # Support downloading from a selected image
    image_name = params.get("image") if isinstance(params.get("image"), str) else None
    image_path = _image_fullpath(image_name)
    with Image.open(image_path) as pil_img:
        orig_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    mask   = build_mask(orig_arr, params)
    result = apply_strategy(orig_arr, mask, params)
    result = apply_gamma(result, float(params.get("gamma", 1.0)))

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="TIFF", compression="tiff_deflate")
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff",
                     as_attachment=True,
                     download_name="diaper_no_artwork.tif")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)


def parse_diaper_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    """Return (sample_uuid, size, image_type) parsed from a diaper TIF filename.

    image_type is one of: 'folded_outside' | 'unfolded_outside' |
                          'unfolded_inside' | 'macro_inside' | None
    None means the file does not match any of the four required view types
    and should be skipped.

    View tags embedded in the filename (checked as substrings):
      M_UF_IN  → macro_inside        (checked first — contains UF_IN)
      FO_OUT   → folded_outside
      UF_IN    → unfolded_inside
      UF_OUT   → unfolded_outside
    """
    stem = Path(filename).stem
    upper = stem.upper()
    parts = stem.split('_')

    sample_uuid = next((p.lower() for p in parts if _UUID_RE.match(p)), None)
    size = next((p.upper() for p in parts if re.match(r'^S\d+$', p, re.I)), None)

    if 'M_UF_IN' in upper:
        image_type: str | None = 'macro_inside'
    elif 'FO_OUT' in upper:
        image_type = 'folded_outside'
    elif 'UF_IN' in upper:
        image_type = 'unfolded_inside'
    elif 'UF_OUT' in upper:
        image_type = 'unfolded_outside'
    else:
        image_type = None  # not one of the four required views — skip

    return sample_uuid, size, image_type


def _find_tifs(folder: str, exclude_prefix: str | None = None) -> list[str]:
    paths = []
    for root, _dirs, files in os.walk(folder):
        if exclude_prefix and os.path.abspath(root).startswith(os.path.abspath(exclude_prefix)):
            continue
        for f in sorted(files):
            if f.lower().endswith(('.tif', '.tiff')):
                paths.append(os.path.join(root, f))
    return paths


def _run_batch(job_id: str, input_folder: str, output_folder: str, params: dict) -> None:
    job = _batch_jobs[job_id]
    try:
        tif_files = _find_tifs(input_folder, exclude_prefix=output_folder)
        job['total'] = len(tif_files)
        job['status'] = 'running'

        images_dir = os.path.join(output_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # samples: (uuid_or_filename, size) -> column paths
        COLS = ('folded_outside', 'unfolded_inside', 'unfolded_outside', 'macro_inside')
        samples: dict = {}
        processed_paths: dict = {}  # original abs path -> relative output path

        for i, tif_path in enumerate(tif_files):
            job['progress'] = i
            fname = os.path.basename(tif_path)
            job['current_file'] = fname

            sample_uuid, size, image_type = parse_diaper_filename(fname)
            key = (sample_uuid or fname, size or '')
            if key not in samples:
                samples[key] = {
                    'uuid': sample_uuid or fname,
                    'size': size or '',
                    **{c: None for c in COLS},
                }
            if image_type in COLS:
                samples[key][image_type] = tif_path

            try:
                with Image.open(tif_path) as pil_img:
                    arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
                mask = build_mask(arr, params)
                result = apply_strategy(arr, mask, params)
                result = apply_gamma(result, float(params.get('gamma', 1.0)))
                out_fname = Path(fname).stem + '.jpg'
                out_path = os.path.join(images_dir, out_fname)
                Image.fromarray(result).save(out_path, format='JPEG', quality=100, subsampling=0)
                processed_paths[tif_path] = os.path.join('images', out_fname)
            except Exception as exc:
                job.setdefault('errors', []).append(f'{fname}: {exc}')

        # Write CSV
        csv_path = os.path.join(output_folder, 'problem1_submission.csv')
        fields = [
            'sample_uuid', 'size',
            'path_folded_outside_image',
            'path_unfolded_inside_image',
            'path_unfolded_outside_image',
            'path_unfolded_macro_inside_image',
        ]
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for s in samples.values():
                def _rel(col, _s=s):
                    src = _s.get(col)
                    return processed_paths.get(src, '') if src else ''
                writer.writerow({
                    'sample_uuid': s['uuid'],
                    'size': s['size'],
                    'path_folded_outside_image': _rel('folded_outside'),
                    'path_unfolded_inside_image': _rel('unfolded_inside'),
                    'path_unfolded_outside_image': _rel('unfolded_outside'),
                    'path_unfolded_macro_inside_image': _rel('macro_inside'),
                })

        job['progress'] = len(tif_files)
        job['status'] = 'done'
        job['csv_path'] = csv_path
        job['output_folder'] = output_folder
        job['samples'] = len(samples)
        job['processed'] = len(processed_paths)

    except Exception as exc:
        job['status'] = 'error'
        job['error'] = str(exc)


@app.route('/batch/scan', methods=['POST'])
def batch_scan():
    data = request.get_json(force=True) or {}
    folder = data.get('folder', '').strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({'error': 'Folder not found'}), 400
    files = _find_tifs(folder)
    return jsonify({'files': [os.path.basename(p) for p in files], 'count': len(files)})


@app.route('/batch/start', methods=['POST'])
def batch_start():
    data = request.get_json(force=True) or {}
    input_folder = data.get('input_folder', '').strip()
    output_folder = (data.get('output_folder') or '').strip() or os.path.join(input_folder, 'results')
    params = data.get('params', {})

    if not input_folder or not os.path.isdir(input_folder):
        return jsonify({'error': 'Invalid input folder'}), 400

    job_id = str(uuid_module.uuid4())
    _batch_jobs[job_id] = {
        'status': 'starting', 'progress': 0, 'total': 0,
        'current_file': '', 'output_folder': output_folder,
    }
    t = threading.Thread(
        target=_run_batch,
        args=(job_id, input_folder, output_folder, params),
        daemon=True,
    )
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/batch/status/<job_id>')
def batch_status(job_id):
    job = _batch_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    return jsonify(job)


@app.route('/batch/download_csv/<job_id>')
def batch_download_csv(job_id):
    job = _batch_jobs.get(job_id)
    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Job not ready'}), 400
    csv_path = job.get('csv_path', '')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'CSV not found'}), 404
    return send_file(csv_path, as_attachment=True, download_name='problem1_submission.csv')


if __name__ == "__main__":
    app.run(debug=False, port=5001)
