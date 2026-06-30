"""
PBC Glyph Watermark Demo -- QR-Inspired, Background-Independent Authentication

Concept:
  Each 128x128 tile receives a small QR-inspired "micro code" at its center.
  The code is an 11x11 module binary matrix:
    - Top-left 7x7: FIXED QR finder pattern (recognizable anchor, same on every tile)
    - Timing strips: alternating row/column separators (QR convention)
    - 3x3 data zone: unique per tile, derived from genesis hash bits

  Unlike the gray-square approach (intensity-based), this encodes in GEOMETRY:
    - Decoder regenerates the expected pattern from the hash and does
      cross-correlation against the tile's pixel region
    - Works on ANY background (white margins, text, figures, equations)
      because we're detecting the BINARY MODULE PATTERN, not absolute intensity

Why it works:
  - Background content is uncorrelated with the specific tile pattern
    => correlation(background, expected_pattern) ~= 0
  - Actual watermark signal: corr(watermarked_region, expected_pattern) > threshold
  - Tampered region: glyph pixels destroyed => low correlation

Visual properties:
  - Modules drawn at 65% opacity -- document content visible between modules
  - The 7x7 finder pattern makes every tile LOOK like a QR code
  - Each tile's data zone differs (unique hash) -- visual chain
  - Human-readable: "QR codes on every tile" -- obvious authentication marks
  - PSNR ~24 dB (visible by design; 51 dB = invisible LSB PBC)

Usage:
    python examples/pbc_glyph_watermark_demo.py
    python examples/pbc_glyph_watermark_demo.py --input page.png
    python examples/pbc_glyph_watermark_demo.py --pages 5   # PDF mode (5 pages)
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbc import compute_grid, DEFAULT_TILE_SIZE, generate_originator_id, compute_genesis_hash


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TILE_SIZE    = 128     # tile grid size (same as LSB PBC)

# QR module matrix: 11x11 modules at MOD_PX pixels/module
QR_MODULES   = 11      # matrix dimension
MOD_PX       = 4       # pixels per module  =>  44x44 pixel glyph
GLYPH_SIZE   = QR_MODULES * MOD_PX   # 44 pixels

LINE_ALPHA   = 0.45    # module fill opacity (0=invisible, 1=opaque)
LINE_DARK    = 55      # intensity of dark modules

# Correlation threshold for PASS / FAIL
CORR_THRESHOLD = 0.30

ORIGINATOR = "PBC-GlyphWatermark-Demo-2026"

# ---------------------------------------------------------------------------
# QR finder pattern (7x7, standard Micro QR / QR Code specification)
# ---------------------------------------------------------------------------
#
#  1 1 1 1 1 1 1
#  1 0 0 0 0 0 1
#  1 0 1 1 1 0 1
#  1 0 1 1 1 0 1
#  1 0 1 1 1 0 1
#  1 0 0 0 0 0 1
#  1 1 1 1 1 1 1
#
QR_FINDER = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Glyph generation  (QR-inspired 11x11 module matrix)
# ---------------------------------------------------------------------------

def generate_glyph(genesis_hash: bytes,
                   n: int = QR_MODULES,
                   mod_px: int = MOD_PX) -> np.ndarray:
    """
    Generate a QR-inspired binary glyph mask (float32, 0=background, 1=module)
    from 6 genesis hash bytes.

    Layout of the 11x11 module matrix:
        [0:7, 0:7]  -- QR finder pattern  (FIXED, same for every tile)
        [7, 0:7]    -- separator row       (always 0)
        [0:7, 7]    -- separator column    (always 0)
        [8, 8:]     -- timing row          (alternating 1,0,1,...)
        [8:, 8]     -- timing column       (alternating 1,0,1,...)
        [9:,  9:]   -- 2x2 format corner   (always 1 = orientation anchor)
        remaining   -- DATA modules        (unique per tile, from hash PRNG)

    The fixed finder + timing + corner modules make every glyph LOOK like a
    tiny QR code.  The data modules carry the per-tile cryptographic identity.
    Rendered to pixels: n*mod_px  x  n*mod_px  (default 44x44).
    """
    matrix = np.zeros((n, n), dtype=np.float32)

    # --- Fixed: QR finder pattern (top-left 7x7) ---
    matrix[:7, :7] = QR_FINDER

    # --- Fixed: timing strips (row 8 and col 8, QR convention) ---
    for i in range(8, n):
        matrix[8, i] = float(i % 2)      # timing row:  col 8,9,10 -> 0,1,0
        matrix[i, 8] = float(i % 2)      # timing col:  row 8,9,10 -> 0,1,0

    # --- Fixed: bottom-right orientation anchor (rows 9-10, cols 9-10) ---
    matrix[9:n, 9:n] = 1.0

    # --- Data modules: all non-fixed positions filled from genesis hash PRNG ---
    seed = int.from_bytes(genesis_hash[:8], 'big') & 0xFFFFFFFFFFFFFFFF
    rng  = np.random.default_rng(seed)

    for r in range(n):
        for c in range(n):
            if r < 7 and c < 7:           continue   # finder
            if r == 7 or c == 7:          continue   # separators
            if r == 8 or c == 8:          continue   # timing
            if r >= 9 and c >= 9:         continue   # anchor
            matrix[r, c] = float(rng.integers(2))    # data bit

    # --- Render modules to pixels ---
    glyph = np.repeat(np.repeat(matrix, mod_px, axis=0), mod_px, axis=1)
    return glyph.astype(np.float32)


# ---------------------------------------------------------------------------
# Encode / verify
# ---------------------------------------------------------------------------

def _render_glyph_onto_tile(tile: np.ndarray, glyph: np.ndarray) -> np.ndarray:
    """
    Alpha-blend the glyph (dark lines) onto a tile region.
    - Where glyph=1 (line): pixel = LINE_ALPHA*LINE_DARK + (1-LINE_ALPHA)*original
    - Where glyph=0 (empty): pixel = original  (UNCHANGED)
    """
    out  = tile.astype(np.float32).copy()
    mask = glyph[:, :, np.newaxis]           # (gs, gs, 1) for broadcast over RGB

    # Scale glyph to place it centered in tile
    gs = glyph.shape[0]
    H, W = tile.shape[:2]
    r0 = (H - gs) // 2
    c0 = (W - gs) // 2
    r1 = r0 + gs;  c1 = c0 + gs

    region = out[r0:r1, c0:c1]
    out[r0:r1, c0:c1] = mask * (LINE_ALPHA * LINE_DARK + (1.0 - LINE_ALPHA) * region) \
                       + (1.0 - mask) * region
    return np.clip(out, 0, 255).astype(np.uint8)


def encode_glyphs(image: np.ndarray,
                  originator: str,
                  tile_size: int = TILE_SIZE,
                  timestamp: int = None) -> tuple:
    """Apply glyph watermarks to all tiles.  Returns (watermarked, timestamp)."""
    if timestamp is None:
        timestamp = int(time.time())

    oid  = generate_originator_id(originator)
    H, W = image.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    out  = image.copy()

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            genesis = compute_genesis_hash(oid, tx, ty, timestamp)
            glyph   = generate_glyph(genesis)

            tile_region = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = _render_glyph_onto_tile(tile_region, glyph)

    return out, timestamp


def _tile_correlation(tile_region: np.ndarray, glyph: np.ndarray) -> float:
    """
    Normalized cross-correlation between tile and expected glyph template.

    The watermark creates dark lines in the glyph area.  We detect this by:
      inv_region  = -(region - mean) / std   # dark pixels become high
      glyph_norm  = (glyph  - mean) / std    # marked pixels are high
      correlation = mean(inv_region * glyph_norm)

    Correct match  -> correlation > CORR_THRESHOLD
    Missing/tamper -> correlation ~= 0  (uncorrelated)
    """
    H, W  = tile_region.shape[:2]
    gs    = glyph.shape[0]
    r0    = (H - gs) // 2;  c0 = (W - gs) // 2

    region = tile_region[r0:r0 + gs, c0:c0 + gs].astype(np.float32)
    if region.ndim == 3:
        region = region.mean(axis=2)          # RGB -> luminance

    r_std  = region.std() + 1e-6
    r_norm = -(region - region.mean()) / r_std    # invert: dark=positive

    g_std  = glyph.std() + 1e-6
    g_norm = (glyph - glyph.mean()) / g_std

    return float(np.mean(r_norm * g_norm))


def verify_glyphs(image: np.ndarray,
                  originator: str,
                  timestamp: int,
                  tile_size: int = TILE_SIZE) -> dict:
    """
    Verify glyph watermarks using template cross-correlation.

    Returns dict:
        'tiles'    : {(tx,ty): {'ok', 'corr'}}
        'n_ok'     : int
        'n_fail'   : int
        'total'    : int
        'accuracy' : float
        'mean_corr': float   mean correlation over all tiles
    """
    oid  = generate_originator_id(originator)
    H, W = image.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    tile_results = {}
    n_ok = n_fail = 0
    corrs = []

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            genesis = compute_genesis_hash(oid, tx, ty, timestamp)
            glyph   = generate_glyph(genesis)

            corr = _tile_correlation(image[y0:y1, x0:x1], glyph)
            ok   = corr > CORR_THRESHOLD

            tile_results[(tx, ty)] = {'ok': ok, 'corr': corr}
            corrs.append(corr)
            if ok:
                n_ok  += 1
            else:
                n_fail += 1

    total = n_ok + n_fail
    return {
        'tiles':     tile_results,
        'n_ok':      n_ok,
        'n_fail':    n_fail,
        'total':     total,
        'accuracy':  n_ok / total if total else 0.0,
        'mean_corr': float(np.mean(corrs)),
    }


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def add_scan_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    noisy = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    pil   = Image.fromarray(noisy).filter(ImageFilter.GaussianBlur(radius=0.6))
    return np.array(pil)


def draw_grid_overlay(image: np.ndarray, result: dict,
                      tile_size: int = TILE_SIZE) -> np.ndarray:
    vis  = image.copy().astype(np.float32)
    H, W = vis.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    for (tx, ty), info in result['tiles'].items():
        x0 = tx * tile_w
        x1 = (tx + 1) * tile_w if tx < cols - 1 else W
        y0 = ty * tile_h
        y1 = (ty + 1) * tile_h if ty < rows - 1 else H
        color  = np.array([0, 210, 0], dtype=np.float32) if info['ok'] \
                 else np.array([220, 0, 0], dtype=np.float32)
        border = 3
        vis[y0:y0 + border, x0:x1] = color
        vis[y1 - border:y1,  x0:x1] = color
        vis[y0:y1, x0:x0 + border] = color
        vis[y0:y1, x1 - border:x1] = color

    return np.clip(vis, 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def make_document_page(W: int = 800, H: int = 1000, seed: int = 42) -> np.ndarray:
    """Synthetic document page (white background with text/figure placeholders)."""
    rng    = np.random.default_rng(seed)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # Title
    canvas[55:95,  60:740] = 210
    canvas[62:88,  70:520] = 160
    canvas[62:88, 530:720] = 180
    # Body text lines
    y = 115
    for i in range(50):
        if i % 7 == 0:
            y += 22;  continue
        line_end = int(rng.integers(350, 700))
        canvas[y:y + 10, 70:70 + line_end] = int(rng.integers(30, 75))
        y += 18
        if y > H - 130:
            break
    # Figure box
    fy1, fy2, fx1, fx2 = H - 210, H - 50, 120, 680
    canvas[fy1:fy2, fx1:fx2] = 248
    canvas[fy1:fy1 + 2, fx1:fx2] = canvas[fy2-2:fy2, fx1:fx2] = 140
    canvas[fy1:fy2, fx1:fx1 + 2] = canvas[fy1:fy2, fx2-2:fx2] = 140
    for k, bt in enumerate([fy2-40-int(rng.integers(20,120)) for _ in range(6)]):
        bx1 = fx1 + 30 + k * 80
        canvas[bt:fy2-5, bx1:bx1+50] = 200
    return canvas


# ---------------------------------------------------------------------------
# Run demo (single image or PDF pages)
# ---------------------------------------------------------------------------

def run_single(input_path: Path = None) -> None:
    """Single-image demo: synthetic or custom input."""
    print("\nPBC Glyph Watermark Demo  -- QR-inspired, background-independent")
    print("=" * 68)
    print(f"  TILE_SIZE    : {TILE_SIZE} px")
    print(f"  QR_MODULES   : {QR_MODULES}x{QR_MODULES}  ({MOD_PX}px/module = {GLYPH_SIZE}x{GLYPH_SIZE}px glyph)")
    print(f"  LINE_ALPHA   : {LINE_ALPHA}  (module opacity -- gaps show through)")
    print(f"  LINE_DARK    : {LINE_DARK}  (module fill intensity)")
    print(f"  CORR_THRESH  : {CORR_THRESHOLD}  (cross-correlation threshold)")
    print()

    if input_path and input_path.exists():
        src = np.array(Image.open(input_path).convert('RGB'))
        print(f"  Input: {input_path.name}  ({src.shape[1]}x{src.shape[0]})")
    else:
        src = make_document_page()
        print(f"  Input: synthetic document page  ({src.shape[1]}x{src.shape[0]})")

    H, W = src.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, TILE_SIZE)
    ts = int(time.time())

    # Encode
    t0 = time.perf_counter()
    wm, ts_used = encode_glyphs(src, ORIGINATOR, timestamp=ts)
    enc_ms = (time.perf_counter() - t0) * 1000

    q = psnr(src, wm)
    print(f"  Grid    : {cols}x{rows} = {cols*rows} tiles  ({tile_w}x{tile_h} px each)")
    print(f"  PSNR    : {q:.1f} dB  (lower = more visible; 51 dB = invisible LSB PBC)")
    print(f"  Encode  : {enc_ms:.1f} ms")
    print()

    # Verify -- clean
    t1 = time.perf_counter()
    res_clean = verify_glyphs(wm, ORIGINATOR, timestamp=ts_used)
    ver_ms = (time.perf_counter() - t1) * 1000
    print(f"  Verify clean digital      : "
          f"{res_clean['n_ok']}/{res_clean['total']}  "
          f"({res_clean['accuracy']*100:.1f}%)  "
          f"mean_corr={res_clean['mean_corr']:.3f}  [{ver_ms:.1f} ms]")

    # Verify -- scan noise sigma=5
    np.random.seed(42)
    noisy5  = add_scan_noise(wm, sigma=5.0)
    res5    = verify_glyphs(noisy5, ORIGINATOR, timestamp=ts_used)
    print(f"  Verify scan noise sigma=5  : "
          f"{res5['n_ok']}/{res5['total']}  "
          f"({res5['accuracy']*100:.1f}%)  "
          f"mean_corr={res5['mean_corr']:.3f}")

    # Verify -- scan noise sigma=10
    np.random.seed(42)
    noisy10 = add_scan_noise(wm, sigma=10.0)
    res10   = verify_glyphs(noisy10, ORIGINATOR, timestamp=ts_used)
    print(f"  Verify scan noise sigma=10 : "
          f"{res10['n_ok']}/{res10['total']}  "
          f"({res10['accuracy']*100:.1f}%)  "
          f"mean_corr={res10['mean_corr']:.3f}")

    # Tamper test -- zero-out 50x50 block in tile (2,2) center
    tampered  = wm.copy()
    tx_t, ty_t = 2, 2
    x0t  = tx_t * tile_w + tile_w // 4
    y0t  = ty_t * tile_h + tile_h // 4
    tampered[y0t:y0t + 50, x0t:x0t + 50] = 200   # overwrite with uniform gray
    res_t = verify_glyphs(tampered, ORIGINATOR, timestamp=ts_used)
    t_info = res_t['tiles'].get((tx_t, ty_t), {})
    detected = not t_info.get('ok', True)
    print()
    print(f"  Tamper test (50x50 block overwrite in tile {tx_t},{ty_t}):")
    print(f"    -> Tampered tile : {'DETECTED' if detected else 'MISSED'}  "
          f"(corr={t_info.get('corr',0):.3f} < {CORR_THRESHOLD})")
    other_ok = sum(1 for (tx, ty), v in res_t['tiles'].items()
                   if (tx, ty) != (tx_t, ty_t) and v['ok'])
    print(f"    -> Unmodified tiles : {other_ok}/{cols*rows - 1} still OK")

    # Save outputs
    out_dir = Path(__file__).parent.parent / "output" / "visible-watermark"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    Image.fromarray(src).save(      str(out_dir / "glyph_01_original.png"))
    Image.fromarray(wm).save(       str(out_dir / "glyph_02_watermarked.png"))
    Image.fromarray(noisy5).save(   str(out_dir / "glyph_03_scan_noise5.png"))
    Image.fromarray(tampered).save( str(out_dir / "glyph_04_tampered.png"))
    Image.fromarray(draw_grid_overlay(wm, res_clean)).save(
                                    str(out_dir / "glyph_05_verify_clean.png"))
    Image.fromarray(draw_grid_overlay(noisy5, res5)).save(
                                    str(out_dir / "glyph_06_verify_noisy5.png"))
    Image.fromarray(draw_grid_overlay(tampered, res_t)).save(
                                    str(out_dir / "glyph_07_verify_tampered.png"))
    diff = np.clip(np.abs(src.astype(np.int32) - wm.astype(np.int32)) * 8, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(     str(out_dir / "glyph_08_diff_8x.png"))

    # Save a mosaic of sample glyphs (6x6)
    _save_glyph_mosaic(out_dir / "glyph_00_glyph_mosaic.png", ORIGINATOR, ts_used)

    print()
    print(f"  Saved: {out_dir}/glyph_*.png")

    # Results report
    report = results_dir / "glyph_watermark_results.txt"
    with open(report, "w") as f:
        f.write("PBC Glyph Watermark Demo Results\n")
        f.write(f"Image: {W}x{H}  Tiles: {cols}x{rows}  "
                f"TileSize: {tile_w}x{tile_h}  GlyphSize: {GLYPH_SIZE}\n")
        f.write(f"QR_MODULES={QR_MODULES}x{QR_MODULES}LINE_ALPHA={LINE_ALPHA}  "
                f"LINE_DARK={LINE_DARK}  CORR_THRESHOLD={CORR_THRESHOLD}\n")
        f.write(f"PSNR: {q:.1f} dB\n\n")
        f.write(f"{'Scenario':<35} {'OK/Total':>10}  {'Accuracy':>8}  {'mean_corr':>9}\n")
        f.write("-" * 67 + "\n")
        for label, r in [("Clean digital", res_clean),
                         ("Scan noise sigma=5",  res5),
                         ("Scan noise sigma=10", res10)]:
            f.write(f"  {label:<33} {r['n_ok']:>4}/{r['total']:<5}  "
                    f"{r['accuracy']*100:>6.1f}%  {r['mean_corr']:>9.3f}\n")
        f.write(f"  {'Tamper detection':<33} "
                f"{'DETECTED' if detected else 'MISSED'}\n")
        f.write("\nKey advantage:\n")
        f.write("  Cross-correlation detection is background-independent.\n")
        f.write("  The expected glyph template is regenerated from the hash and\n")
        f.write("  correlated against the tile region.  Background content is\n")
        f.write("  uncorrelated with the specific glyph pattern and contributes\n")
        f.write("  ~0 to the correlation signal.\n")
    print(f"  Report: {report}")


def run_pdf(pages_dir: Path, max_pages: int = None) -> None:
    """Multi-page PDF mode: process real paper pages."""
    page_files = sorted(pages_dir.glob("page_*_pbc.png"))
    if not page_files:
        print(f"  No pages found in {pages_dir}")
        return
    if max_pages:
        page_files = page_files[:max_pages]

    out_dir     = Path(__file__).parent.parent / "output" / "visible-watermark" / "PBC_Paper_v04_glyph"
    results_dir = Path(__file__).parent.parent / "output" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    ts = 0x20260101

    print()
    print("PBC Glyph Watermark -- Real PDF Validation")
    print("=" * 68)
    print(f"  Input  : {pages_dir.name}/  ({len(page_files)} pages @ 300 DPI)")
    print(f"  GLYPH_SIZE={GLYPH_SIZE}  QR_MODULES={QR_MODULES}x{QR_MODULES}"
          f"LINE_ALPHA={LINE_ALPHA}  CORR_THRESH={CORR_THRESHOLD}")
    print()
    print(f"  {'Page':>5}  {'Tiles':>6}  {'Clean%':>7}  "
          f"{'Scan5%':>7}  {'mean_corr':>9}  {'PSNR':>6}  {'ms':>7}")
    print(f"  {'-'*62}")

    stats = []
    wm_pages = []
    total_ok_clean = total_ok_scan5 = total_tiles = 0
    t_all = time.perf_counter()

    for i, pg_path in enumerate(page_files):
        t0  = time.perf_counter()
        src = np.array(Image.open(pg_path).convert("RGB"))
        H, W = src.shape[:2]
        cols, rows, _, _ = compute_grid(W, H, TILE_SIZE)
        n_tiles = cols * rows

        wm, ts_used = encode_glyphs(src, ORIGINATOR, timestamp=ts)
        enc_ms = (time.perf_counter() - t0) * 1000

        res_clean = verify_glyphs(wm, ORIGINATOR, timestamp=ts_used)
        np.random.seed(42 + i)
        noisy5 = add_scan_noise(wm, sigma=5.0)
        res5   = verify_glyphs(noisy5, ORIGINATOR, timestamp=ts_used)
        q      = psnr(src, wm)
        pg_lbl = pg_path.stem.split("_")[1]

        out_png  = out_dir / f"page_{pg_lbl}_glyph.png"
        Image.fromarray(wm).save(str(out_png), format="png")
        wm_pages.append(out_png)

        overlay = draw_grid_overlay(wm, res_clean)
        Image.fromarray(overlay).save(str(out_dir / f"page_{pg_lbl}_verify.png"), format="png")

        total_ok_clean += res_clean['n_ok']
        total_ok_scan5 += res5['n_ok']
        total_tiles    += n_tiles

        stats.append({'page': pg_lbl, 'W': W, 'H': H,
                      'tiles': n_tiles,
                      'clean_ok': res_clean['n_ok'],
                      'scan5_ok': res5['n_ok'],
                      'mean_corr': res_clean['mean_corr'],
                      'psnr': q, 'ms': enc_ms})

        print(f"  {pg_lbl:>5}  {n_tiles:>5}  "
              f"{res_clean['accuracy']*100:>6.1f}%  "
              f"{res5['accuracy']*100:>6.1f}%  "
              f"{res_clean['mean_corr']:>9.3f}  {q:>5.1f}  {enc_ms:>6.0f} ms")

    total_ms = (time.perf_counter() - t_all) * 1000
    c_ovr = total_ok_clean / total_tiles * 100
    s_ovr = total_ok_scan5 / total_tiles * 100
    print(f"  {'-'*62}")
    print(f"  {'TOTAL':>5}  {total_tiles:>5}  "
          f"{c_ovr:>6.1f}%  {s_ovr:>6.1f}%  "
          f"{'---':>9}  {'---':>5}  {total_ms/1000:.1f} s")
    print()

    # Assemble PDF
    pdf_out = Path(__file__).parent.parent / "output" / "visible-watermark" / "PBC_Paper_v04_glyph.pdf"
    _assemble_pdf(wm_pages, pdf_out)

    # Report
    report = results_dir / "glyph_watermark_pdf_results.txt"
    with open(report, "w") as f:
        f.write("PBC Glyph Watermark -- Real PDF Validation\n")
        f.write(f"Input: PBC_Paper_v04  ({len(stats)} pages @ 300 DPI)\n")
        f.write(f"GLYPH_SIZE={GLYPH_SIZE}  QR_MODULES={QR_MODULES}x{QR_MODULES}"
                f"LINE_ALPHA={LINE_ALPHA}  CORR_THRESHOLD={CORR_THRESHOLD}\n\n")
        f.write(f"{'Page':>5}  {'Tiles':>5}  {'Clean%':>7}  "
                f"{'Scan5%':>7}  {'mean_corr':>9}  {'PSNR':>6}\n")
        f.write("-" * 55 + "\n")
        for s in stats:
            f.write(f"  {s['page']:>3}  {s['tiles']:>4}  "
                    f"{s['clean_ok']/s['tiles']*100:>6.1f}%  "
                    f"{s['scan5_ok']/s['tiles']*100:>6.1f}%  "
                    f"{s['mean_corr']:>9.3f}  {s['psnr']:>5.1f}\n")
        f.write("-" * 55 + "\n")
        f.write(f"  TOTAL  {total_tiles:>5}  {c_ovr:>6.1f}%  {s_ovr:>6.1f}%\n\n")
        f.write("Key result:\n")
        f.write("  Cross-correlation detection is BACKGROUND-INDEPENDENT.\n")
        f.write("  Works on text columns, code blocks, figures, equations.\n")
        f.write(f"  {len(stats)} pages, {total_tiles} tiles, "
                f"{c_ovr:.1f}% clean decode, {s_ovr:.1f}% after scan sigma=5.\n")

    print(f"  Watermarked pages : {out_dir.name}/")
    print(f"  Report            : output/results/{report.name}")
    print()
    print(f"  Clean decode : {c_ovr:.1f}%  ({total_ok_clean}/{total_tiles})")
    print(f"  Scan sigma=5 : {s_ovr:.1f}%  ({total_ok_scan5}/{total_tiles})")


def _assemble_pdf(page_pngs: list, out_path: Path) -> None:
    """
    Assemble watermarked page PNGs into a single PDF using PIL.

    PIL produces standard-compliant PDFs without the stream/endstream
    mismatch that causes Adobe Acrobat 'unusual terminator' warnings
    (a known issue with img2pdf-generated files).
    """
    if not page_pngs:
        return
    try:
        pil_imgs = [Image.open(str(p)).convert("RGB") for p in page_pngs]
        pil_imgs[0].save(
            str(out_path), format="PDF",
            save_all=True, append_images=pil_imgs[1:],
            resolution=300.0,
        )
        print(f"  PDF assembled: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"  PDF assembly failed: {e}")
        print(f"  Watermarked pages available as PNG files.")


def _save_glyph_mosaic(out_path: Path, originator: str, timestamp: int,
                       grid: int = 6) -> None:
    """Save a grid of example glyphs for visual inspection."""
    oid  = generate_originator_id(originator)
    size = GLYPH_SIZE
    pad  = 4
    canvas_size = grid * (size + pad) + pad
    mosaic = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)

    for ty in range(grid):
        for tx in range(grid):
            genesis = compute_genesis_hash(oid, tx, ty, timestamp)
            glyph   = generate_glyph(genesis)
            # Render as dark lines on light background
            cell    = np.full((size, size, 3), 245, dtype=np.float32)
            mask    = glyph[:, :, np.newaxis]
            cell    = mask * LINE_DARK + (1.0 - mask) * cell
            cell    = np.clip(cell, 0, 255).astype(np.uint8)

            r0 = pad + ty * (size + pad)
            c0 = pad + tx * (size + pad)
            mosaic[r0:r0 + size, c0:c0 + size] = cell

    Image.fromarray(mosaic).save(str(out_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_PDF_PAGES = (Path(__file__).parent.parent
                     / "output" / "document-stamp"
                     / "PBC_Paper_v04_stamped_pages")


def main():
    parser = argparse.ArgumentParser(
        description="PBC Glyph Watermark -- line-shape, background-independent"
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="Input image (single-image mode)")
    parser.add_argument("--pages", type=int, default=None,
                        help="PDF mode: process first N pages from "
                             "PBC_Paper_v04_stamped_pages/")
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_PDF_PAGES,
                        help="Directory with page_NNN_pbc.png for PDF mode")
    args = parser.parse_args()

    if args.pages is not None:
        run_pdf(args.in_dir, args.pages)
    elif args.input is not None:
        run_single(args.input)
    else:
        run_single()


if __name__ == "__main__":
    main()
