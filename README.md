# Diaper Artwork Remover

Removes printed artwork from a diaper photo while keeping all wrinkles, shadows,
and material texture intact. Works by detecting coloured pixels via HSV saturation,
building a mask, then neutralising the colour inside that mask.

---

## Requirements

Python 3.10+ and pip. Install dependencies once:

```bash
pip3 install Pillow numpy flask scipy
```

---

## Run

```bash
cd /Users/fnberger/Desktop/dAIper/contrast_try
python3 app.py
```

Then open **http://127.0.0.1:5001** in your browser.

The image is loaded once at startup (takes ~2 s). The terminal shows:

```
Loading image…
Image loaded: 2730x2550
 * Running on http://127.0.0.1:5001
```

---

## Restart / refresh after code changes

**Backend change** (`app.py`) — restart the server:

```bash
# kill any running instance
lsof -ti:5001 | xargs kill -9 2>/dev/null
python3 app.py
```

**Frontend change** (`templates/index.html`) — just **hard-refresh the browser**
(`Cmd+Shift+R` / `Ctrl+Shift+R`). Flask serves templates directly from disk,
no restart needed.

---

## Workflow

1. Adjust **Colour Detection Mask** sliders until the red overlay in the
   **Mask overlay** tab covers only the printed artwork.
2. Choose a **Removal Strategy** (see below).
3. Press **Preview** — compare **Result** vs **Original**.
4. When satisfied, press **Download full-res TIF**.

---

## Controls reference

### Colour Detection Mask

| Control | What it does |
|---------|-------------|
| Saturation threshold | Min HSV saturation to be counted as artwork. Lower → more aggressive. |
| Min brightness (V) | Pixels darker than this are never masked (preserves shadows). |
| Max brightness (V) | Blown-out highlights above this are excluded from the mask. |

### Mask Refinement

| Control | What it does |
|---------|-------------|
| Blur / feather | Gaussian blur on the raw mask — smooths jagged edges before removal. |
| Erode | Shrink mask first to remove isolated noisy pixels. |
| Dilate | Expand mask to cover colour bleed and anti-aliased print edges. |

### Removal Strategy

Three strategies can be combined via toggle switches. The active combination
is shown in the badge above the toggles.

| Strategy | Description |
|----------|-------------|
| **A1 — LAB Neutral** (always active as baseline) | Zeroes out the a\* and b\* channels in CIELAB space, leaving only luminance L\*. Produces perceptually correct neutral grey that preserves wrinkles. |
| **A2 — Reference Tone** | Samples the median colour of clean diaper pixels outside the mask, then pulls the masked pixels' a\*/b\* towards that reference instead of zero. Avoids the "grey hole" look on non-white diapers. |
| **A3 — Frequency Separation** | Decomposes the image into a low-frequency colour layer and a high-frequency texture layer. Only the colour layer is neutralised inside the mask; the texture layer is left fully intact and recombined. Best strategy for heavily textured or embossed surfaces. |

**Recommended combo: A2 + A3** — frequency separation with reference-tone matching.

| Removal strength | Effect |
|------------------|--------|
| 1.0 | Fully neutralised inside the mask |
| 0.0 – 0.9 | Partial desaturation — blend between original and neutral |

### Zoom buttons (top-right of image area)

| Button | Action |
|--------|--------|
| − | Zoom out (steps: 25 → 33 → 50 → 67 → 75 → 100 → 125 → 150 → 200 → 300 → 400 %) |
| % label | Shows current zoom level |
| + | Zoom in |
| ⊡ | Fit image to window |

---

## File layout

```
contrast_try/
├── app.py                  # Flask backend + all image processing
├── templates/
│   └── index.html          # Frontend (single-page UI)
├── remove_artwork.md       # Detailed notes on each removal strategy
├── README.md               # This file
└── *.TIF                   # Source image (loaded by app.py)
```
