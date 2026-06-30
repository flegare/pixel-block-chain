#!/usr/bin/env python3
"""
PBC Multi-Image Evaluation Suite
=================================

Runs core PBC experiments across a diverse set of test images and reports
mean ± std for each metric.  This addresses the single-image generalization
concern for journal submission.

Test images used (in order of preference):
  1. Any real .jpg/.png files found in examples/img/  (leo.jpg, etc.)
  2. Five synthetic images generated with distinct statistical profiles:
       portrait   -- smooth radial gradients (low-frequency, face-like)
       landscape  -- horizon gradient + ground texture (two-zone)
       document   -- white background + dark text-like stripes (bimodal)
       geometric  -- hard-edge geometric shapes (step functions)
       noise      -- high-frequency random noise (hardest for LSB)

Experiments run per image:
  A. Encode + clean verify        -> GREEN rate, PSNR
  B. Tamper 10% region, verify    -> RED/ABSENT detection rate outside mask,
                                     GREEN preservation inside mask
  C. Non-aligned 60%x80% crop     -> surviving tile GREEN rate
  D. Edit Ledger 2-author append  -> all-GREEN after append?

Output: per-image table + aggregate mean ± std across all images.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import math
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, compute_grid, DEFAULT_TILE_SIZE
from pbc.encoder import encode
from pbc.decoder import verify, TileStatus

IMG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
TILE_SIZE  = DEFAULT_TILE_SIZE
TIMESTAMP  = 1_700_000_000
ORIGINATOR_CAM  = "MultiEval-Camera"
ORIGINATOR_EDIT = "MultiEval-Editor"
TAMPER_FRAC     = 0.10   # tamper 10% of pixels (rectangular region)
CROP_W_FRAC     = 0.60
CROP_H_FRAC     = 0.80


# =============================================================================
# Synthetic image generators
# =============================================================================

def make_portrait(size=(512, 384)) -> np.ndarray:
    """Smooth radial gradient — simulates a face-like low-frequency image."""
    H, W = size
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    cx, cy = W // 2, H // 2
    Y, X = np.mgrid[0:H, 0:W]
    r = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    skin = np.clip(255 - r * 80, 60, 255).astype(np.uint8)
    arr[:, :, 0] = np.clip(skin + 20, 0, 255)
    arr[:, :, 1] = np.clip(skin - 30, 0, 255)
    arr[:, :, 2] = np.clip(skin - 60, 0, 255)
    return arr


def make_landscape(size=(512, 384)) -> np.ndarray:
    """Sky-to-ground gradient — two distinct statistical zones."""
    H, W = size
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    horizon = H // 2
    # Sky: blue gradient
    for y in range(horizon):
        v = int(150 + (horizon - y) / horizon * 80)
        arr[y, :, 0] = max(0, v - 60)
        arr[y, :, 1] = max(0, v - 20)
        arr[y, :, 2] = min(255, v + 30)
    # Ground: green-brown noise
    rng = np.random.default_rng(7)
    ground = rng.integers(40, 120, (H - horizon, W, 3), dtype=np.uint8)
    ground[:, :, 1] += 30   # greenish tint
    arr[horizon:, :] = np.clip(ground, 0, 255)
    return arr


def make_document(size=(512, 384)) -> np.ndarray:
    """White background with dark horizontal stripes — bimodal histogram."""
    H, W = size
    arr = np.full((H, W, 3), 245, dtype=np.uint8)
    line_h, gap = 12, 24
    for y in range(0, H, gap):
        arr[y:y + line_h, 40:W - 40] = 30
    return arr


def make_geometric(size=(512, 384)) -> np.ndarray:
    """Hard-edge geometric shapes — step-function pixel values."""
    H, W = size
    arr = np.full((H, W, 3), 200, dtype=np.uint8)
    # Rectangles
    arr[H//4:3*H//4, W//4:3*W//4] = [50, 100, 180]
    arr[H//3:2*H//3, W//3:2*W//3] = [220, 80, 50]
    arr[H//2 - 20:H//2 + 20, :] = [30, 180, 30]
    return arr


def make_noise(size=(512, 384)) -> np.ndarray:
    """High-frequency random noise — worst case for LSB embedding."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (*size, 3), dtype=np.uint8)


SYNTHETIC_GENERATORS = [
    ("portrait",   make_portrait),
    ("landscape",  make_landscape),
    ("document",   make_document),
    ("geometric",  make_geometric),
    ("noise",      make_noise),
]


# =============================================================================
# Helpers
# =============================================================================

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def tile_counts(result):
    g = y = r = a = 0
    for t in result.all_tiles:
        if   t.status == TileStatus.GREEN:  g += 1
        elif t.status == TileStatus.YELLOW: y += 1
        elif t.status == TileStatus.RED:    r += 1
        else:                               a += 1
    return g, y, r, a


def load_real_images():
    """Load real jpg/png images from IMG_DIR (excluding variants)."""
    images = []
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        # Skip generated PBC variants
        if any(tag in fname for tag in ['_pbc', '_altered', '_edited', '_encoded']):
            continue
        path = os.path.join(IMG_DIR, fname)
        try:
            arr = np.array(Image.open(path).convert('RGB'))
            images.append((fname, arr))
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")
    return images


def sep(title=""):
    bar = "=" * 72
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


# =============================================================================
# Experiment A: Clean encode + verify
# =============================================================================

def exp_a(img, name):
    enc = encode(img, ORIGINATOR_CAM, timestamp=TIMESTAMP, tile_size=TILE_SIZE)
    res = verify(enc, tile_size=TILE_SIZE)
    total = res.cols * res.rows
    g, y, r, a = tile_counts(res)
    ps = psnr(img, enc)
    return {
        'green_rate': g / total if total else 0.0,
        'psnr': ps,
        'total_tiles': total,
        '_enc': enc,
    }


# =============================================================================
# Experiment B: Localized tamper detection
# =============================================================================

def exp_b(img, enc_from_a):
    H, W = img.shape[:2]
    # Tamper a 10%-area rectangle at top-left
    tw = int(math.sqrt(TAMPER_FRAC) * W)
    th = int(math.sqrt(TAMPER_FRAC) * H)
    tampered = enc_from_a.copy()
    tampered[:th, :tw] = 128   # flat grey fill

    res = verify(tampered, tile_size=TILE_SIZE)
    total = res.cols * res.rows

    # Which tiles overlap tamper region?
    cols, rows, tile_w, tile_h = compute_grid(W, H, TILE_SIZE)
    tamper_tiles = set()
    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w; x1 = (tx+1)*tile_w if tx < cols-1 else W
            y0 = ty * tile_h; y1 = (ty+1)*tile_h if ty < rows-1 else H
            if x0 < tw and y0 < th:
                tamper_tiles.add((tx, ty))

    # Tiles NOT in tamper region should remain GREEN
    outside_green = 0
    outside_total = 0
    inside_detected = 0
    inside_total = len(tamper_tiles)

    for t in res.all_tiles:
        if (t.tx, t.ty) in tamper_tiles:
            if t.status != TileStatus.GREEN:  # RED, YELLOW, or ABSENT all signal tampering
                inside_detected += 1
        else:
            outside_total += 1
            if t.status == TileStatus.GREEN:
                outside_green += 1

    return {
        'detection_rate': inside_detected / inside_total if inside_total else 0.0,
        'false_positive_rate': 1.0 - (outside_green / outside_total if outside_total else 1.0),
        'tampered_tiles': inside_total,
    }


# =============================================================================
# Experiment C: Non-aligned crop
# =============================================================================

def exp_c(img, enc_from_a):
    H, W = img.shape[:2]
    cw = int(W * CROP_W_FRAC)
    ch = int(H * CROP_H_FRAC)
    cx0 = (W - cw) // 2
    cy0 = (H - ch) // 2
    cropped = enc_from_a[cy0:cy0+ch, cx0:cx0+cw]
    res = verify(cropped, tile_size=TILE_SIZE)
    total = res.cols * res.rows
    g, y, r, a = tile_counts(res)
    return {
        'crop_green_rate': g / total if total else 0.0,
        'crop_tiles': total,
    }


# =============================================================================
# Experiment D: 2-author append
# =============================================================================

def exp_d(img, enc_from_a):
    from pbc.encoder import append_edit
    appended = append_edit(enc_from_a, originator=ORIGINATOR_EDIT,
                           opcode=OpCode.EDIT_COLOR,
                           tile_size=TILE_SIZE, split_fraction=0.5,
                           timestamp=TIMESTAMP + 3600)
    res = verify(appended, tile_size=TILE_SIZE)
    total = res.cols * res.rows
    g, y, r, a = tile_counts(res)
    # After append, tiles should be GREEN (new chain from editor) or YELLOW
    # YELLOW = correct PBC-aware re-encoding. GREEN = full re-encode. Both valid.
    ok = (g + y) / total if total else 0.0
    return {
        'append_ok_rate': ok,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sep("PBC Multi-Image Evaluation Suite")

    # Build image list
    image_list = load_real_images()
    print(f"  Found {len(image_list)} real image(s) in {IMG_DIR}")

    for name, gen_fn in SYNTHETIC_GENERATORS:
        arr = gen_fn()
        image_list.append((f"synth_{name}", arr))

    print(f"  Total images: {len(image_list)} (real + 5 synthetic)")

    all_rows = []

    sep("Per-Image Results")
    hdr = (f"  {'Image':<22}  {'WxH':>10}  {'Tiles':>6}  "
           f"{'PSNR':>6}  {'A:GREEN%':>8}  {'B:det%':>7}  "
           f"{'B:fp%':>6}  {'C:crop%':>8}  {'D:ok%':>7}")
    print(hdr)
    print("  " + "-" * 88)

    for name, img in image_list:
        H, W = img.shape[:2]
        t0 = time.perf_counter()

        ra = exp_a(img, name)
        rb = exp_b(img, ra['_enc'])
        rc = exp_c(img, ra['_enc'])
        rd = exp_d(img, ra['_enc'])
        elapsed = (time.perf_counter() - t0) * 1000

        row = {
            'name': name,
            'W': W, 'H': H,
            'tiles': ra['total_tiles'],
            'psnr': ra['psnr'],
            'green_rate': ra['green_rate'],
            'det_rate': rb['detection_rate'],
            'fp_rate': rb['false_positive_rate'],
            'crop_green': rc['crop_green_rate'],
            'append_ok': rd['append_ok_rate'],
            'ms': elapsed,
        }
        all_rows.append(row)

        print(f"  {name:<22}  {W}x{H:>4}  {ra['total_tiles']:>6}  "
              f"{ra['psnr']:>6.1f}  {ra['green_rate']*100:>7.1f}%  "
              f"{rb['detection_rate']*100:>6.1f}%  "
              f"{rb['false_positive_rate']*100:>5.1f}%  "
              f"{rc['crop_green_rate']*100:>7.1f}%  "
              f"{rd['append_ok_rate']*100:>6.1f}%")

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    sep("Aggregate Statistics (mean ± std across all images)")
    metrics = [
        # (key, label, is_pct)
        ('psnr',       'PSNR (dB)',              False),
        ('green_rate', 'Exp A: clean GREEN%',     True),
        ('det_rate',   'Exp B: tamper detect%',   True),
        ('fp_rate',    'Exp B: false positive%',  True),
        ('crop_green', 'Exp C: crop GREEN%',       True),
        ('append_ok',  'Exp D: append ok%',        True),
    ]

    print()
    print(f"  {'Metric':<35}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 72)
    agg = {}
    for key, label, is_pct in metrics:
        vals = np.array([r[key] for r in all_rows])
        if is_pct:
            vals = vals * 100
        mu, std, mn, mx = vals.mean(), vals.std(), vals.min(), vals.max()
        agg[key] = (mu, std, mn, mx)
        suffix = "%" if is_pct else " dB"
        print(f"  {label:<35}  {mu:>7.1f}{suffix}  {std:>6.2f}{suffix}  "
              f"{mn:>7.1f}{suffix}  {mx:>7.1f}{suffix}")

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    sep("Key Findings for Paper")
    print()
    g_mu, g_std, _, _ = agg['green_rate']
    print(f"  Clean verification: {g_mu:.1f}% ± {g_std:.1f}% tiles GREEN across {len(all_rows)} images")
    d_mu, d_std, _, _ = agg['det_rate']
    fp_mu, fp_std, _, _ = agg['fp_rate']
    print(f"  Tamper detection: {d_mu:.1f}% ± {d_std:.1f}% of tampered tiles flagged")
    print(f"  False positive rate: {fp_mu:.2f}% ± {fp_std:.2f}% of clean tiles flagged")
    c_mu, c_std, _, _ = agg['crop_green']
    print(f"  Non-aligned crop ({CROP_W_FRAC:.0%}x{CROP_H_FRAC:.0%}): "
          f"{c_mu:.1f}% ± {c_std:.1f}% tiles survive")
    a_mu, a_std, _, _ = agg['append_ok']
    print(f"  Append mode: {a_mu:.1f}% ± {a_std:.1f}% tiles valid after 2-author edit")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, 'results', 'multi_image_eval_results.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("PBC Multi-Image Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Image':<22}  {'WxH':>10}  {'PSNR':>6}  "
                f"{'A:G%':>6}  {'B:det%':>7}  {'B:fp%':>6}  "
                f"{'C:cr%':>6}  {'D:ok%':>6}\n")
        f.write("-" * 80 + "\n")
        for r in all_rows:
            f.write(f"{r['name']:<22}  {r['W']}x{r['H']:>4}  "
                    f"{r['psnr']:>6.1f}  {r['green_rate']*100:>5.1f}%  "
                    f"{r['det_rate']*100:>6.1f}%  {r['fp_rate']*100:>5.2f}%  "
                    f"{r['crop_green']*100:>5.1f}%  {r['append_ok']*100:>5.1f}%\n")
        f.write("\nAggregates:\n")
        for key, label, unit in metrics:
            mu, std, mn, mx = agg[key]
            f.write(f"  {label}: {mu:.2f} ± {std:.2f} (min {mn:.2f}, max {mx:.2f})\n")

    print(f"\n  Results saved to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
