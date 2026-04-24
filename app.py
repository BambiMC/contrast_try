import io
import base64
import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from scipy.ndimage import (gaussian_filter, binary_dilation, binary_erosion,
                           label, distance_transform_edt)

app = Flask(__name__)

IMAGE_PATH = os.path.join(os.path.dirname(__file__),
    "macro.tif")

print("Loading image…")
_orig_pil = Image.open(IMAGE_PATH).convert("RGB")
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

    _, s, v = rgb_to_hsv(arr)
    mask = (s > sat_threshold) & (v > val_min) & (v < val_max)

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
      1. Sample border pixels (within `border_width` px outside the island).
      2. Build a statistical model of their colour distribution.
      3. Map the island pixels so their distribution matches the border's.
      4. Blend back with the original using `blend_strength`.

    transform_mode:
      'mean'      – shift mean only  (fast, preserves relative contrasts)
      'mean_std'  – match mean + per-channel std  (good default)
      'covariance'– full 3×3 covariance (Monge-Kantorovich optimal transport)

    colorspace: 'rgb' or 'lab' (LAB is perceptually more uniform)
    """
    border_width      = max(1, round(float(params.get("et_border_width",   10)) * pixel_scale))
    transform_mode    = params.get("et_transform_mode",  "mean_std")
    colorspace        = params.get("et_colorspace",       "lab")
    blend_strength    = float(params.get("et_blend_strength",   1.0))
    edge_blend        = bool(params.get("et_edge_blend",         True))
    edge_blend_radius = max(1.0, float(params.get("et_edge_blend_radius", 8.0)) * pixel_scale)
    min_border_px     = int(params.get("et_min_border_px",  20))
    min_island_px     = int(params.get("et_min_island_px",   5))

    result   = arr.copy().astype(np.float32)
    labeled, n_islands = label(mask)

    for island_id in range(1, n_islands + 1):
        island_mask = labeled == island_id
        n_island = int(island_mask.sum())
        if n_island < min_island_px:
            continue

        # Border = dilated island minus the entire mask (avoids sampling other artwork)
        dilated     = binary_dilation(island_mask, iterations=border_width)
        border_mask = dilated & ~mask
        n_border    = int(border_mask.sum())
        if n_border < min_border_px:
            continue

        border_px = arr[border_mask].astype(np.float32)   # (B, 3)
        island_px = arr[island_mask].astype(np.float32)   # (I, 3)

        # Convert to working colour space
        if colorspace == "lab":
            border_cs = _px_to_lab(border_px)
            island_cs = _px_to_lab(island_px)
        else:
            border_cs = border_px.copy()
            island_cs = island_px.copy()

        mu_ref = border_cs.mean(axis=0)   # (3,)
        mu_src = island_cs.mean(axis=0)   # (3,)

        if transform_mode == "mean":
            transformed = island_cs - mu_src + mu_ref

        elif transform_mode == "mean_std":
            std_ref = border_cs.std(axis=0) + 1e-6
            std_src = island_cs.std(axis=0) + 1e-6
            transformed = (island_cs - mu_src) / std_src * std_ref + mu_ref

        else:  # covariance
            if n_border >= 4 and n_island >= 4:
                cov_src = np.cov(island_cs.T) + np.eye(3) * 1e-6
                cov_ref = np.cov(border_cs.T) + np.eye(3) * 1e-6
                A = _mat_sqrt(cov_ref) @ _mat_inv_sqrt(cov_src)   # optimal transport
                transformed = (island_cs - mu_src) @ A.T + mu_ref
            else:
                transformed = island_cs - mu_src + mu_ref  # fallback

        # Back to RGB [0-255] float
        if colorspace == "lab":
            transformed = _px_from_lab(np.clip(transformed, [-16, -128, -128], [100, 127, 127]))
        transformed = np.clip(transformed, 0, 255)

        # All island pixels are transformed uniformly (w=1).
        # The distribution transform already ensures the island matches the border
        # statistically, so no per-pixel weight tapering is needed — tapering to 0 at
        # the boundary (the old behaviour) kept artwork-coloured pixels at the edges.
        # edge_blend_radius is kept for future post-processing use.
        w = np.ones(n_island)

        eff = blend_strength * w  # (I,)
        for c in range(3):
            result[island_mask, c] = (
                (1.0 - eff) * arr[island_mask, c].astype(np.float32)
                + eff * transformed[:, c]
            )

    return np.clip(result, 0, 255).astype(np.uint8)


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)


def apply_filters(arr: np.ndarray, mask: np.ndarray, filters: list) -> np.ndarray:
    """Sequentially alpha-composite each {color, opacity} filter over masked pixels."""
    if not filters:
        return arr
    result = arr.copy().astype(np.float32)
    for filt in filters:
        color   = hex_to_rgb(filt.get("color", "#ffffff"))
        opacity = max(0.0, min(1.0, float(filt.get("opacity", 0.5))))
        if opacity <= 0:
            continue
        for c in range(3):
            result[mask, c] = (1 - opacity) * result[mask, c] + opacity * color[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_strategy(arr: np.ndarray, mask: np.ndarray, params: dict,
                   pixel_scale: float = 1.0) -> np.ndarray:
    use_a2   = bool(params.get("use_a2", False))
    use_a3   = bool(params.get("use_a3", False))
    use_a4   = bool(params.get("use_a4", False))
    strength = float(params.get("removal_strength", 1.0))
    sigma    = float(params.get("freq_sigma", 20.0)) * pixel_scale
    filters  = params.get("filters", [])

    if use_a4:
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", width=W, height=H,
                           preview_scale=PREVIEW_SCALE)


@app.route("/preview", methods=["POST"])
def preview():
    params = request.get_json(force=True) or {}
    render_pct = max(0.05, min(1.0, float(params.get("render_pct", 100)) / 100.0))

    tw = max(1, int(W * render_pct))
    th = max(1, int(H * render_pct))
    arr = np.array(_orig_pil.resize((tw, th), Image.LANCZOS), dtype=np.uint8)

    mask   = build_mask(arr, params)
    result = apply_strategy(arr, mask, params, pixel_scale=render_pct)

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
    mask   = build_mask(ORIG, params)
    result = apply_strategy(ORIG, mask, params)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="TIFF", compression="tiff_deflate")
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff",
                     as_attachment=True,
                     download_name="diaper_no_artwork.tif")


if __name__ == "__main__":
    app.run(debug=False, port=5001)
