#!/usr/bin/env python3
"""
HPC batch processor for diaper artwork removal.

Reads image data from HDF5 shards, removes coloured artwork, and writes:

  <output>/
    problem1_submission.csv
    images/
      <original_stem>.jpg   (JPEG 100 %, original resolution)

Usage
-----
  python hpc_batch.py [--input DIR] [--output DIR] [--workers N] [--gpu]

HDF5 layout:
  samples/{uuid}__{Size}/
    @sample_uuid, @Diaper_Size
    components/
      fo_f_out_dl/  fo_f_out_cd/  fo_f_out_mask/
      uf_f_out_dl/  uf_f_in_dl/   m_uf_in_dl/  ...
        0000  shape=() dtype=|V<bytes>
          @content_ext, @original_filename, ...

Preference: _dl > _cd; mask components are skipped entirely.
"""

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation, binary_erosion, gaussian_filter,
    label, distance_transform_edt, uniform_filter,
)

# ---------------------------------------------------------------------------
# Optional GPU acceleration via PyTorch (CUDA or Apple MPS)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as _TF

    if torch.cuda.is_available():
        _TORCH_DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _TORCH_DEVICE = torch.device("mps")
    else:
        _TORCH_DEVICE = torch.device("cpu")

    _HAS_GPU = _TORCH_DEVICE.type != "cpu"
except ImportError:
    _HAS_GPU = False
    _TORCH_DEVICE = None


def _gaussian_filter_gpu(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via PyTorch on GPU; falls back to scipy on CPU."""
    if not _HAS_GPU or sigma <= 0:
        return gaussian_filter(channel, sigma) if sigma > 0 else channel
    ks = max(3, int(4 * sigma + 1) | 1)
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    k1 = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    t = torch.from_numpy(channel).float().unsqueeze(0).unsqueeze(0).to(_TORCH_DEVICE)
    kh = k1.view(1, 1, -1, 1).to(_TORCH_DEVICE)
    kw = k1.view(1, 1, 1, -1).to(_TORCH_DEVICE)
    pad = ks // 2
    t = _TF.pad(t, (0, 0, pad, pad), mode="reflect")
    t = _TF.conv2d(t, kh)
    t = _TF.pad(t, (pad, pad, 0, 0), mode="reflect")
    t = _TF.conv2d(t, kw)
    return t.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Load settings.json (shared defaults with the web app)
# ---------------------------------------------------------------------------
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")
try:
    with open(_SETTINGS_PATH) as _sf:
        _FILE_SETTINGS = json.load(_sf)
except FileNotFoundError:
    _FILE_SETTINGS = {}

_HARDCODED_DEFAULTS: dict = {
    "sat_threshold": 0.06,
    "val_min": 0.00,
    "val_max": 1.00,
    "blur_sigma": 0.0,
    "erode_iter": 0,
    "dilate_iter": 0,
    "mask_mode": "hsv_sat",
    "hue_center": 0.0,
    "hue_width": 60.0,
    "use_a2": False,
    "use_a3": False,
    "use_a4": False,
    "et_a4_mode": "edge_transfer",
    "removal_strength": 1.0,
    "freq_sigma": 20.0,
    "gamma": 1.0,
    "filters": [],
    "et_border_width": 10,
    "et_transform_mode": "covariance",
    "et_colorspace": "lab",
    "et_blend_strength": 1.0,
    "et_edge_feather": False,
    "et_edge_blend_radius": 8,
    "et_min_border_px": 5,
    "et_min_island_px": 1,
    "et_passes": 1,
    "et_clean_border": False,
    "et_clean_border_sat": 0.15,
    "et_lock_luma": False,
}

DEFAULT_PARAMS: dict = {**_HARDCODED_DEFAULTS, **_FILE_SETTINGS}

CSV_FIELDS = [
    "sample_uuid",
    "size",
    "path_folded_outside_image",
    "path_unfolded_inside_image",
    "path_unfolded_outside_image",
    "path_unfolded_macro_inside_image",
]


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _resolve_column(component_name: str) -> str | None:
    name = component_name.lower()
    if "mask" in name:
        return None
    if "m_uf" in name and "in" in name:
        return "path_unfolded_macro_inside_image"
    if "fo" in name and "out" in name:
        return "path_folded_outside_image"
    if "uf" in name and "in" in name:
        return "path_unfolded_inside_image"
    if "uf" in name and "out" in name:
        return "path_unfolded_outside_image"
    return None


def _lighting_score(component_name: str) -> int:
    name = component_name.lower()
    if name.endswith("_dl") or "_dl_" in name:
        return 2
    if name.endswith("_cd") or "_cd_" in name:
        return 1
    return 0


def attrs_to_dict(attrs) -> dict:
    out = {}
    for k, v in attrs.items():
        out[k] = v.decode() if isinstance(v, (bytes, np.bytes_)) else v
    return out


def decode_image_bytes(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def find_shards(root: str) -> list[str]:
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".h5") or f.endswith(".hdf5"):
                paths.append(os.path.join(dirpath, f))
    return paths


# ---------------------------------------------------------------------------
# Algorithm — self-contained (no Flask dependency)
# ---------------------------------------------------------------------------

def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _gamma_enc(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)

def _f(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, t ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)

def _f_inv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))

def rgb_to_hsv(rgb):
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

def rgb_to_lab(f32):
    r, g, b = _lin(f32[..., 0]), _lin(f32[..., 1]), _lin(f32[..., 2])
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx, fy, fz = _f(X / 0.95047), _f(Y), _f(Z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

def lab_to_rgb_u8(L, a, b_ch):
    fy = (L + 16) / 116
    X = 0.95047 * _f_inv(a / 500 + fy)
    Y = _f_inv(fy)
    Z = 1.08883 * _f_inv(fy - b_ch / 200)
    r  =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g  = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bv =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([_gamma_enc(np.clip(r, 0, 1)),
                    _gamma_enc(np.clip(g, 0, 1)),
                    _gamma_enc(np.clip(bv, 0, 1))], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def build_mask(arr, params):
    sat_threshold = float(params.get("sat_threshold", 0.06))
    val_min       = float(params.get("val_min",       0.00))
    val_max       = float(params.get("val_max",       1.00))
    blur_sigma    = float(params.get("blur_sigma",    0.0))
    erode_iter    = int(params.get("erode_iter",      0))
    dilate_iter   = int(params.get("dilate_iter",     0))
    mask_mode     = params.get("mask_mode",           "hsv_sat")
    hue_center    = float(params.get("hue_center",   0.0))
    hue_width     = float(params.get("hue_width",    60.0))

    h, s, v = rgb_to_hsv(arr)

    if mask_mode == "lab_chroma":
        f = arr.astype(np.float32) / 255.0
        _, a_ch, b_ch = rgb_to_lab(f)
        score = np.sqrt(a_ch ** 2 + b_ch ** 2) / 128.0
    elif mask_mode == "hue_range":
        hue_diff = np.abs(((h - hue_center + 180) % 360) - 180)
        score = s * (hue_diff <= (hue_width / 2)).astype(np.float32)
    elif mask_mode == "combined":
        f = arr.astype(np.float32) / 255.0
        _, a_ch, b_ch = rgb_to_lab(f)
        score = np.maximum(s, np.sqrt(a_ch ** 2 + b_ch ** 2) / 128.0)
    elif mask_mode == "delta_e":
        f = arr.astype(np.float32) / 255.0
        L_arr, a_arr, b_arr = rgb_to_lab(f)
        chroma = np.sqrt(a_arr ** 2 + b_arr ** 2)
        n_bg = max(100, chroma.size // 10)
        idx = np.argpartition(chroma.ravel(), n_bg)[:n_bg]
        bg_L = float(np.median(L_arr.ravel()[idx]))
        bg_a = float(np.median(a_arr.ravel()[idx]))
        bg_b = float(np.median(b_arr.ravel()[idx]))
        score = np.sqrt((L_arr - bg_L)**2 + (a_arr - bg_a)**2 + (b_arr - bg_b)**2) / 50.0
    elif mask_mode == "channel_diff":
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


# -- A1 -----------------------------------------------------------------------

def _remove_lab_neutral(arr, mask, strength):
    f = arr.astype(np.float32) / 255.0
    L, a, b_ch = rgb_to_lab(f)
    a_new = a.copy();    a_new[mask]  = a[mask]    * (1 - strength)
    b_new = b_ch.copy(); b_new[mask]  = b_ch[mask] * (1 - strength)
    return lab_to_rgb_u8(L, a_new, b_new)


# -- A2 -----------------------------------------------------------------------

def _sample_reference_tone(arr, mask):
    grey = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    cands = ~mask & (grey > 40) & (grey < 240)
    if cands.sum() < 50:
        cands = ~mask
    return np.median(arr[cands].astype(np.float32), axis=0)

def _remove_reference_tone(arr, mask, ref_rgb, strength):
    f    = arr.astype(np.float32) / 255.0
    fref = ref_rgb.reshape(1, 1, 3) / 255.0
    L, a, b_ch = rgb_to_lab(f)
    _, ref_a, ref_b = rgb_to_lab(np.broadcast_to(fref, f.shape))
    a_new  = a.copy();    a_new[mask]  = a[mask]    + strength * (ref_a[mask]  - a[mask])
    b_new  = b_ch.copy(); b_new[mask]  = b_ch[mask] + strength * (ref_b[mask]  - b_ch[mask])
    return lab_to_rgb_u8(L, a_new, b_new)


# -- A3 -----------------------------------------------------------------------

def _remove_freq_separation(arr, mask, sigma, use_a2, ref_rgb, strength):
    f = arr.astype(np.float32)
    # Use GPU gaussian blur if available
    low = np.stack([_gaussian_filter_gpu(f[..., c], sigma) for c in range(3)], axis=-1)
    high = f - low
    low_u8 = np.clip(low, 0, 255).astype(np.uint8)
    neutral = (_remove_reference_tone(low_u8, mask, ref_rgb, 1.0)
               if use_a2 and ref_rgb is not None
               else _remove_lab_neutral(low_u8, mask, 1.0))
    new_low = low.copy()
    for c in range(3):
        new_low[mask, c] = ((1 - strength) * low[mask, c]
                            + strength * neutral[mask, c].astype(np.float32))
    return np.clip(new_low + high, 0, 255).astype(np.uint8)


# -- A4 edge feather ----------------------------------------------------------

def _apply_edge_feather(arr_orig, result, mask, feather_radius):
    """Eliminate seam at mask boundary via smoothstep blend toward nearest original pixel."""
    if feather_radius <= 0 or not mask.any():
        return result
    dist, idx = distance_transform_edt(mask, return_indices=True)
    feather_zone = mask & (dist <= feather_radius)
    if not feather_zone.any():
        return result
    ry, cx = idx[0][feather_zone], idx[1][feather_zone]
    border_color = arr_orig[ry, cx].astype(np.float32)
    transferred  = result[feather_zone].astype(np.float32)
    t = np.clip((dist[feather_zone] - 1.0) / max(feather_radius - 1, 1), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t))[:, np.newaxis]
    out = result.copy().astype(np.float32)
    out[feather_zone] = (1.0 - w) * border_color + w * transferred
    return np.clip(out, 0, 255).astype(np.uint8)


# -- A4 helpers ---------------------------------------------------------------

def _px_to_lab(px):
    L, a, b = rgb_to_lab(px / 255.0)
    return np.stack([L, a, b], axis=-1)

def _px_from_lab(lab):
    return lab_to_rgb_u8(lab[:, 0], lab[:, 1], lab[:, 2]).astype(np.float32)

def _mat_sqrt(M):
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

def _mat_inv_sqrt(M):
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T


# -- A4 edge transfer ---------------------------------------------------------

def _remove_edge_transfer(arr, mask, params, pixel_scale=1.0):
    border_width     = max(1, round(float(params.get("et_border_width",   10)) * pixel_scale))
    transform_mode   = params.get("et_transform_mode",  "covariance")
    colorspace       = params.get("et_colorspace",       "lab")
    blend_strength   = float(params.get("et_blend_strength",  1.0))
    min_border_px    = int(params.get("et_min_border_px",   5))
    min_island_px    = int(params.get("et_min_island_px",   1))
    n_passes         = max(1, int(params.get("et_passes",   1)))
    clean_border     = bool(params.get("et_clean_border",  False))
    clean_border_sat = float(params.get("et_clean_border_sat", 0.15))
    do_feather       = bool(params.get("et_edge_blend",    True))
    feather_radius   = max(1, round(float(params.get("et_edge_blend_radius", 8)) * pixel_scale))

    labeled, n_islands = label(mask)
    current = arr.copy().astype(np.float32)

    for _ in range(n_passes):
        pass_result = current.copy()
        for island_id in range(1, n_islands + 1):
            island_mask = labeled == island_id
            n_island = int(island_mask.sum())
            if n_island < min_island_px:
                continue
            dilated     = binary_dilation(island_mask, iterations=border_width)
            border_mask = dilated & ~mask
            n_border    = int(border_mask.sum())
            if n_border < min_border_px:
                continue
            border_px = arr[border_mask].astype(np.float32)
            island_px = current[island_mask].astype(np.float32)
            if clean_border:
                _, b_sat, _ = rgb_to_hsv(border_px)
                keep = b_sat < clean_border_sat
                if keep.sum() >= min_border_px:
                    border_px = border_px[keep]
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
            else:  # covariance
                n_b = len(border_px)
                if n_b >= 4 and n_island >= 4:
                    cov_src = np.cov(island_cs.T) + np.eye(3) * 1e-6
                    cov_ref = np.cov(border_cs.T) + np.eye(3) * 1e-6
                    A = _mat_sqrt(cov_ref) @ _mat_inv_sqrt(cov_src)
                    transformed = (island_cs - mu_src) @ A.T + mu_ref
                else:
                    transformed = island_cs - mu_src + mu_ref
            if colorspace == "lab":
                transformed = _px_from_lab(np.clip(transformed, [-16, -128, -128], [100, 127, 127]))
            transformed = np.clip(transformed, 0, 255)
            for c in range(3):
                pass_result[island_mask, c] = (
                    (1.0 - blend_strength) * current[island_mask, c]
                    + blend_strength * transformed[:, c]
                )
        current = pass_result

    result = np.clip(current, 0, 255).astype(np.uint8)
    if do_feather:
        result = _apply_edge_feather(arr, result, mask, feather_radius)
    return result


# -- A4 radiant ---------------------------------------------------------------

def _remove_a4_radiant(arr, mask, params, pixel_scale=1.0):
    """Inverse-distance blend from nearest non-masked pixel in 4 cardinal directions."""
    H, W = mask.shape
    non_mask = ~mask
    result = arr.copy().astype(np.float32)

    y_g = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W))
    x_g = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :], (H, W))
    r_g = np.broadcast_to(np.arange(H, dtype=np.int32)[:, None], (H, W))
    c_g = np.broadcast_to(np.arange(W, dtype=np.int32)[None, :], (H, W))

    lx = np.where(non_mask, x_g, -np.inf)
    lx = np.maximum.accumulate(lx, axis=1)
    l_dist = np.where(mask & (lx >= 0), x_g - lx, np.inf).astype(np.float32)
    l_vals = arr[r_g, np.clip(np.where(np.isfinite(lx), lx, 0).astype(np.int32), 0, W-1)].astype(np.float32)

    rx = np.where(non_mask, x_g, np.inf)
    rx = np.minimum.accumulate(rx[:, ::-1], axis=1)[:, ::-1]
    r_dist = np.where(mask & (rx < W), rx - x_g, np.inf).astype(np.float32)
    r_vals = arr[r_g, np.clip(np.where(np.isfinite(rx), rx, W-1).astype(np.int32), 0, W-1)].astype(np.float32)

    uy = np.where(non_mask, y_g, -np.inf)
    uy = np.maximum.accumulate(uy, axis=0)
    u_dist = np.where(mask & (uy >= 0), y_g - uy, np.inf).astype(np.float32)
    u_vals = arr[np.clip(np.where(np.isfinite(uy), uy, 0).astype(np.int32), 0, H-1), c_g].astype(np.float32)

    dy = np.where(non_mask, y_g, np.inf)
    dy = np.minimum.accumulate(dy[::-1, :], axis=0)[::-1, :]
    d_dist = np.where(mask & (dy < H), dy - y_g, np.inf).astype(np.float32)
    d_vals = arr[np.clip(np.where(np.isfinite(dy), dy, H-1).astype(np.int32), 0, H-1), c_g].astype(np.float32)

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

    out = np.clip(result, 0, 255).astype(np.uint8)
    if bool(params.get("et_edge_blend", True)):
        radius = max(1, round(float(params.get("et_edge_blend_radius", 8)) * pixel_scale))
        out = _apply_edge_feather(arr, out, mask, radius)
    return out


# -- A4 propagate -------------------------------------------------------------

def _remove_a4_propagate(arr, mask, params, pixel_scale=1.0):
    """BFS-style outside-in propagation fill."""
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
            filled  = val_sum[frontier] / np.maximum(cnt[frontier], 1.0)
            if blend < 1.0:
                filled = (1 - blend) * arr[frontier, c].astype(np.float32) + blend * filled
            result[frontier, c] = filled
        known |= frontier

    out = np.clip(result, 0, 255).astype(np.uint8)
    if bool(params.get("et_edge_blend", True)):
        radius = max(1, round(float(params.get("et_edge_blend_radius", 8)) * pixel_scale))
        out = _apply_edge_feather(arr, out, mask, radius)
    return out


# -- Filters ------------------------------------------------------------------

def _apply_filters(arr, mask, filters):
    if not filters:
        return arr
    result = arr.copy().astype(np.float32)
    for filt in filters:
        h = filt.get("color", "#ffffff").lstrip("#")
        color = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
        opacity    = max(0.0, min(1.0, float(filt.get("opacity",    0.0))))
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


# -- Gamma --------------------------------------------------------------------

def apply_gamma(arr, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return arr
    return np.clip((arr.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)


# -- Main dispatcher ----------------------------------------------------------

def process_image(arr, params):
    use_a2   = bool(params.get("use_a2",   False))
    use_a3   = bool(params.get("use_a3",   False))
    use_a4   = bool(params.get("use_a4",   False))
    a4_mode  = params.get("et_a4_mode",   "edge_transfer")
    strength = float(params.get("removal_strength", 1.0))
    sigma    = float(params.get("freq_sigma", 20.0))
    filters  = params.get("filters", [])

    mask = build_mask(arr, params)

    if use_a4:
        if a4_mode == "radiant":
            result = _remove_a4_radiant(arr, mask, params)
        elif a4_mode == "propagate":
            result = _remove_a4_propagate(arr, mask, params)
        else:
            result = _remove_edge_transfer(arr, mask, params)
    elif use_a3:
        ref = _sample_reference_tone(arr, mask) if use_a2 else None
        result = _remove_freq_separation(arr, mask, sigma, use_a2, ref, strength)
    elif use_a2:
        result = _remove_reference_tone(arr, mask, _sample_reference_tone(arr, mask), strength)
    else:
        result = _remove_lab_neutral(arr, mask, strength)

    result = _apply_filters(result, mask, filters)
    return apply_gamma(result, float(params.get("gamma", 1.0)))


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess for parallel processing)
# ---------------------------------------------------------------------------

def _worker(job: dict) -> dict:
    """Process one image component.  Returns updated job dict with out_path or error."""
    try:
        pil_img  = decode_image_bytes(job["payload"])
        arr      = np.array(pil_img, dtype=np.uint8)
        result   = process_image(arr, job["params"])
        Image.fromarray(result).save(job["out_path"], format="JPEG", quality=100, subsampling=0)
        if job.get("write_original"):
            orig_path = job["out_path"].replace(".jpg", ".tif")
            with open(orig_path, "wb") as fout:
                fout.write(job["payload"])
        return {**job, "ok": True}
    except Exception as exc:
        return {**job, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="/home/hpc/k_e06y/e06y0005/hackathon_test1/test_1/",
        help="Directory containing HDF5 shards (searched recursively)",
    )
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <input>/results)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for parallel image processing "
                             "(default: number of CPU cores)")
    parser.add_argument("--write_original", action="store_true",
                        help="Also save the original image bytes next to each processed JPG")
    args = parser.parse_args()

    input_dir      = args.input
    output_dir     = args.output or os.path.join(input_dir, "results")
    images_dir     = os.path.join(output_dir, "images")
    write_original = args.write_original
    os.makedirs(images_dir, exist_ok=True)

    shards = find_shards(input_dir)
    if not shards:
        print(f"ERROR: no .h5/.hdf5 files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    device_info = (f"GPU ({_TORCH_DEVICE})" if _HAS_GPU
                   else "CPU (install PyTorch for GPU acceleration)")
    n_workers = args.workers or os.cpu_count() or 1
    print(f"Found {len(shards)} shard(s)")
    print(f"Compute device : {device_info}")
    print(f"Parallel workers: {n_workers}")

    params = DEFAULT_PARAMS.copy()

    # ── Collect all jobs from shards ──────────────────────────────────────────
    jobs: list[dict] = []
    rows_meta: list[dict] = {}  # group_name -> CSV row skeleton

    for shard_path in shards:
        print(f"  Scanning {Path(shard_path).name}…")
        with h5py.File(shard_path, "r") as handle:
            if "samples" not in handle:
                print("    no 'samples' group — skipped", file=sys.stderr)
                continue

            for group_name in handle["samples"]:
                sample_grp  = handle["samples"][group_name]
                group_attrs = attrs_to_dict(sample_grp.attrs)
                sample_uuid = group_attrs.get("sample_uuid") or group_name.split("__")[0]
                size        = (group_attrs.get("Diaper_Size")
                               or group_attrs.get("diaper_size") or "")

                if "components" not in sample_grp:
                    continue

                best: dict[str, tuple[str, int]] = {}
                for comp_name in sample_grp["components"]:
                    col   = _resolve_column(comp_name)
                    score = _lighting_score(comp_name)
                    if col and (col not in best or score > best[col][1]):
                        best[col] = (comp_name, score)

                row = {f: "" for f in CSV_FIELDS}
                row["sample_uuid"] = sample_uuid
                row["size"]        = size

                for col, (comp_name, _) in best.items():
                    comp_grp = sample_grp["components"][comp_name]
                    keys = sorted(comp_grp.keys())
                    if not keys:
                        continue
                    item_name  = keys[0]
                    dataset    = comp_grp[item_name]
                    payload    = dataset[()].tobytes()
                    dset_attrs = attrs_to_dict(dataset.attrs)
                    orig_fname = dset_attrs.get("original_filename",
                                               f"{group_name}_{comp_name}.tif")
                    stem      = Path(orig_fname).stem
                    out_path  = os.path.join(images_dir, stem + ".jpg")
                    rel_path  = os.path.join("images", stem + ".jpg")

                    jobs.append({
                        "payload":        payload,
                        "params":         params,
                        "out_path":       out_path,
                        "write_original": write_original,
                        "group_name":     group_name,
                        "comp_name":      comp_name,
                        "stem":           stem,
                        "col":            col,
                        "rel_path":       rel_path,
                    })
                    row[col] = rel_path

                if any(row[c] for c in CSV_FIELDS[2:]):
                    rows_meta[group_name] = row

    print(f"\nTotal images to process: {len(jobs)}")

    # ── Process in parallel ───────────────────────────────────────────────────
    n_ok = 0
    n_err = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            res = future.result()
            if res["ok"]:
                n_ok += 1
                print(f"  ✓ {res['group_name'][:16]}… / {res['comp_name']} → {res['stem']}.jpg")
            else:
                n_err += 1
                print(f"  ✗ {res['group_name']}/{res['comp_name']}: {res['error']}",
                      file=sys.stderr)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "problem1_submission.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_meta.values())

    print()
    print("Done.")
    print(f"  Images processed : {n_ok}")
    print(f"  Errors           : {n_err}")
    print(f"  Samples in CSV   : {len(rows_meta)}")
    print(f"  Output folder    : {output_dir}")
    print(f"  Submission CSV   : {csv_path}")


if __name__ == "__main__":
    main()
