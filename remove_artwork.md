# Artwork Removal – Approach Plan

The mask already identifies artwork pixels reliably.
The challenge: **replace the color inside the mask** while keeping all luminance structure
(wrinkles, fold-shadows, material texture, print-emboss shading).

---

## Approach 1 – LAB Neutralisation (upgrade of current method)

**How it works**  
Convert the image to CIE L\*a\*b\*. Inside the mask, set a\* = 0 and b\* = 0,
keeping L\* unchanged. Convert back to RGB.

**Why better than RGB greyscale**  
L\* is a perceptually uniform lightness axis — it follows how the eye sees brightness
far better than the Rec.601 luminance formula. Shadows and highlights will be more
natural. The "white" of the diaper won't shift to a cold grey.

**Tradeoffs**  
- Still produces a neutral grey inside the mask; does not match the paper/fabric tone.  
- Requires a LAB conversion (manual XYZ pipeline or scikit-image).  
- Very fast (<1 s on full-res).

**Best when** the diaper base is already near-neutral white.

---

## Approach 2 – Reference-Tone Matching

**How it works**  
1. Sample pixels *outside* the mask (the clean diaper material) to compute a
   reference mean colour (e.g. slightly warm off-white: R≈245, G≈240, B≈235).  
2. Inside the mask: for each pixel compute its L\* (or grey luminance) relative to
   the reference luminance, then tint it with the reference hue/chroma at that
   relative brightness.

**Why this is useful**  
The diaper fabric has a slight warmth or coolness. Plain grey would look like a
"hole" was punched and filled with something different. Matching the surrounding
tone makes the removal invisible.

**Tradeoffs**  
- Slightly more complex; needs a clean reference region (avoid selecting wrinkles
  or other prints).  
- Fails if the diaper has very different zones with different base tones.  
- Still fast; no heavy computation.

**Best when** the diaper is not pure white and you want a seamless look.

---

## Approach 3 – Frequency-Layer Separation

**How it works**  
Decompose the image into two layers:
- **Low-frequency** (large-scale color/tone): Gaussian blur of the full image, e.g. σ=30.
- **High-frequency** (texture, wrinkles, edges): original minus low-frequency.

Inside the mask:
- Replace the *low-frequency* layer with a neutral/reference colour (approaches 1 or 2).
- Keep the *high-frequency* layer completely unchanged.

Recombine: neutral_low + original_high.

**Why this is powerful**  
Wrinkles and shadows live almost entirely in the high-frequency layer. The artwork
color is mainly a low-frequency signal (large colored patches). The split means
color removal can't accidentally destroy the texture.

**Tradeoffs**  
- Sigma of the blur is a new tunable parameter (too small = colored residue remains;
  too large = bleeds outside mask).  
- Does not handle very fine printed lines well (they appear in the HF layer too).  
- Still fast (two Gaussian filters + arithmetic).

**Best when** the artwork has large flat-color areas and the diaper surface is textured.

---

## Approach 4 – Inpainting from Surroundings (Texture Synthesis)

**How it works**  
Treat the mask as a "damaged" region and inpaint it using OpenCV's
`cv2.inpaint` (Telea or Navier-Stokes) or a patch-based method (PatchMatch).
The algorithm fills the masked area with texture synthesised from the immediately
surrounding non-masked pixels.

**Why this is different**  
Instead of manipulating color channels, we *replace* the masked content entirely
with plausible diaper material. Ideal when the artwork sits on a textured surface
(embossed patterns, fabric weave) that should continue underneath.

**Tradeoffs**  
- Computationally heavier (seconds to tens of seconds on full res).  
- Can produce tiling artefacts on large masks.  
- Requires OpenCV (`pip install opencv-python`).  
- Works poorly if the masked region is very large (no nearby reference to copy from).

**Best when** artwork regions are relatively small and surrounded by clean material.

---

## Approach 5 – Gradient-Domain Blending (Poisson Editing)

**How it works**  
1. Create a "target" image identical to the original but with the masked region
   replaced by neutral colour (approach 1 or 2).  
2. Use Poisson image editing (scipy sparse linear system) to blend the *gradients*
   of the original into the target: the luminance gradients (= texture, wrinkles)
   are taken from the original; the absolute color level is taken from the target.

This is conceptually the most principled method: it enforces that **every
luminance edge** from the original is preserved, while the absolute color level
smoothly interpolates from the surrounding clean material.

**Tradeoffs**  
- Requires solving a large sparse linear system — slow on full-res (minutes).  
- Complex to implement correctly (boundary conditions matter).  
- Overkill for flat-color artwork on an essentially uniform background.

**Best when** extreme quality is needed and processing time is acceptable,
or if the artwork is printed *over* visible material texture that must be kept.

---

## Recommendation / Suggested Implementation Order

| Priority | Approach | Effort | Quality |
|----------|----------|--------|---------|
| 1 | **LAB neutralisation** (A1) | Low | Good — quick win over current method |
| 2 | **Reference-tone matching** (A2) | Low | Better — avoids the grey-hole look |
| 3 | **Frequency-layer separation** (A3) | Medium | Best for textured diapers |
| 4 | **OpenCV inpainting** (A4) | Medium | Good for small artwork regions |
| 5 | **Poisson / gradient domain** (A5) | High | Highest quality, slow |

**Suggested combo:**  
Start with **A3 (frequency separation)** using **A2 (reference-tone matching)** for
the low-frequency replacement. This gives a seamless result where shadows and
wrinkles are perfectly preserved and the "blank" diaper area matches the
surrounding material tone — with no new dependencies beyond numpy/scipy.
