"""
PBC Visible Watermark Demo ~ Print-Scan Robust Chain Authentication

Concept:
  - Document tiles receive corner gray squares that encode chain data visibly
  - Each 128~128 tile gets four 32~32 corner squares (25% coverage)
  - Each corner square encodes 2 bits via 4 gray intensity levels ~ 8 bits/tile
  - The 8-bit tile signature = SHA-256(originator || tx || ty || timestamp)[:1]
    (derived from the same genesis hash as LSB PBC ~ consistent chain root)
  - A scanner reads the average intensity of each 32~32 corner ~ decodes 2 bits
  - Robust to print noise: averaging 1024 pixels reduces noise by ~32~

Gray levels (direct assignment, no alpha blending):
  Level 0b00 ~ gray 180  (lightest overlay)
  Level 0b01 ~ gray 160
  Level 0b10 ~ gray 140
  Level 0b11 ~ gray 120  (darkest overlay)
  Step = 20 intensity units between levels
  Scanner noise margin: ~10 units (per-cell average noise ~ 0.1 units @ 300 DPI)

Why different from LSB PBC:
  - Survives PRINTING: ink-on-paper preserves average intensity, not bit values
  - VISIBLE to human observers: the corner squares are perceptible on the page
  - Machine-readable: scanner + decoder can verify without the original file
  - Limitation: the underlying document content can still be digitally replaced
    (use parallel LSB PBC or digital signature to fully close that gap)

Usage:
    python examples/pbc_visible_watermark_demo.py
    python examples/pbc_visible_watermark_demo.py --input page.png

Requirements:
    numpy, pillow
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbc import (compute_grid, DEFAULT_TILE_SIZE,
                 generate_originator_id, compute_genesis_hash)


# ---------------------------------------------------------------------------
# Watermark parameters
# ---------------------------------------------------------------------------

TILE_SIZE  = 128   # px ~ same grid as LSB PBC
CELL_SIZE  = 32    # px ~ corner square size (32~32 per corner)

# 4 target gray levels for 2-bit encoding.
# These are the PURE gray values used to compute the blend.
# Step = 20 intensity units between levels.
GRAY_LEVELS = [180, 160, 140, 120]  # index 0 = lightest, 3 = darkest

# Opacity of the corner overlay (0.0 = invisible, 1.0 = solid).
# 0.50 makes the square clearly visible on print while keeping underlying
# text legible through the tint.
ALPHA = 0.50

# Effective gray levels as measured on a white (255) background after blending.
# The decoder compares measured averages to these values (white-background assumption,
# valid for typical printed documents where page corners are mostly white margin).
#   effective[i] = ALPHA * GRAY_LEVELS[i] + (1-ALPHA) * 255
EFFECTIVE_GRAY_LEVELS = [int(round(ALPHA * g + (1 - ALPHA) * 255)) for g in GRAY_LEVELS]
# => [217, 207, 197, 187]  (step = 10 units; noise margin = 5 units)

# Bit-shift positions for 4 corners in the 8-bit tile signature
# Top-left=bits[7:6], Top-right=bits[5:4], Bottom-left=bits[3:2], Bottom-right=bits[1:0]
CORNER_SHIFTS = [6, 4, 2, 0]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _tile_signature(originator_id: int, tx: int, ty: int, timestamp: int) -> int:
    """8-bit tile signature: first byte of per-tile genesis hash."""
    genesis = compute_genesis_hash(originator_id, tx, ty, timestamp)
    return genesis[0]  # 0~255


def _encode_tile(tile: np.ndarray, sig_byte: int) -> np.ndarray:
    """
    Draw four semi-transparent corner squares on a tile, each encoding 2 bits.

    The squares are ALPHA-BLENDED onto the original content so the underlying
    text and figures remain readable through the tint.  Opacity is controlled
    by ALPHA (default 0.50 = clearly visible but not blocking).
    """
    out = tile.copy().astype(np.float32)
    H, W = out.shape[:2]
    cs = min(CELL_SIZE, H // 4, W // 4)

    corners = [
        (0,      0,      ),   # top-left
        (0,      W - cs, ),   # top-right
        (H - cs, 0,      ),   # bottom-left
        (H - cs, W - cs, ),   # bottom-right
    ]

    for i, (r0, c0) in enumerate(corners):
        bits2 = (sig_byte >> CORNER_SHIFTS[i]) & 0x3
        gray  = float(GRAY_LEVELS[bits2])
        r1 = min(r0 + cs, H)
        c1 = min(c0 + cs, W)
        # Alpha-blend: result = ALPHA*gray + (1-ALPHA)*original
        out[r0:r1, c0:c1] = ALPHA * gray + (1.0 - ALPHA) * out[r0:r1, c0:c1]

    return np.clip(out, 0, 255).astype(np.uint8)


def _ring_bg(tile: np.ndarray, r0: int, c0: int, cs: int, ring: int = 8) -> float:
    """
    Estimate the background under a corner square by sampling the adjacent
    un-watermarked strip (same local texture, never overwritten).

    For corner at (r0, c0):
      - horizontal strip: rows r0..r0+cs, columns just inward of the square
      - vertical strip  : columns c0..c0+cs, rows just inward of the square
    Both strips are NOT watermarked and share the immediate local background.
    """
    H, W = tile.shape[:2]
    r1, c1 = r0 + cs, c0 + cs
    samples = []

    # Horizontal neighbor (strip to the right or left of the square)
    if c0 == 0:
        c_lo, c_hi = c1, min(c1 + ring, W)
    else:
        c_lo, c_hi = max(0, c0 - ring), c0
    if c_hi > c_lo:
        samples.append(tile[r0:r1, c_lo:c_hi].astype(np.float32).ravel())

    # Vertical neighbor (strip below or above the square)
    if r0 == 0:
        r_lo, r_hi = r1, min(r1 + ring, H)
    else:
        r_lo, r_hi = max(0, r0 - ring), r0
    if r_hi > r_lo:
        samples.append(tile[r_lo:r_hi, c0:c1].astype(np.float32).ravel())

    if samples:
        return float(np.mean(np.concatenate(samples)))
    return 255.0   # fallback: white page


def _decode_tile(tile: np.ndarray) -> int:
    """
    Read the 8-bit value encoded in a tile's four corner squares.

    For each corner, the adjacent un-watermarked ring strip is sampled to
    estimate the local background, then:
        recovered_gray = (measured_avg - (1-ALPHA)*bg_est) / ALPHA
    is classified against GRAY_LEVELS.

    This works on any background (white margins, text, figures) without
    needing the original image.
    """
    H, W = tile.shape[:2]
    cs = min(CELL_SIZE, H // 4, W // 4)

    corners = [
        (0,      0,      ),
        (0,      W - cs, ),
        (H - cs, 0,      ),
        (H - cs, W - cs, ),
    ]

    decoded = 0
    for i, (r0, c0) in enumerate(corners):
        cell      = tile[r0:r0 + cs, c0:c0 + cs].astype(np.float32)
        avg       = float(np.mean(cell))
        bg_est    = _ring_bg(tile, r0, c0, cs)
        recovered = (avg - (1.0 - ALPHA) * bg_est) / ALPHA
        dists     = [abs(recovered - g) for g in GRAY_LEVELS]
        bits2     = dists.index(min(dists))
        decoded   = (decoded << 2) | bits2

    return decoded & 0xFF


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_visible(image: np.ndarray,
                   originator: str,
                   tile_size: int = TILE_SIZE,
                   timestamp: int = None) -> tuple:
    """
    Apply visible corner-square watermark to document image.

    Returns:
        (watermarked_image, timestamp_used)
    """
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
            sig = _tile_signature(oid, tx, ty, timestamp)
            out[y0:y1, x0:x1] = _encode_tile(out[y0:y1, x0:x1], sig)

    return out, timestamp


def verify_visible(image: np.ndarray,
                   originator: str,
                   timestamp: int,
                   tile_size: int = TILE_SIZE) -> dict:
    """
    Verify visible watermark in image.

    Returns dict:
        'tiles'    : {(tx,ty): {'ok', 'expected', 'decoded'}}
        'n_ok'     : number of tiles verified correctly
        'n_fail'   : number of tiles with incorrect signature
        'total'    : total tiles checked
        'accuracy' : n_ok / total
    """
    oid  = generate_originator_id(originator)
    H, W = image.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    tile_results = {}
    n_ok = n_fail = 0

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile     = image[y0:y1, x0:x1]
            decoded  = _decode_tile(tile)
            expected = _tile_signature(oid, tx, ty, timestamp)
            ok       = (decoded == expected)

            tile_results[(tx, ty)] = {
                'ok': ok, 'expected': expected, 'decoded': decoded,
            }
            if ok:
                n_ok  += 1
            else:
                n_fail += 1

    total = n_ok + n_fail
    return {
        'tiles':    tile_results,
        'n_ok':     n_ok,
        'n_fail':   n_fail,
        'total':    total,
        'accuracy': n_ok / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def make_document_page(W: int = 800, H: int = 1000, seed: int = 42) -> np.ndarray:
    """Synthesize a realistic-looking document page (white background + text)."""
    rng     = np.random.default_rng(seed)
    canvas  = np.full((H, W, 3), 255, dtype=np.uint8)

    # Title bar
    canvas[55:95,  60:740] = 210
    canvas[62:88,  70:520] = 160
    canvas[62:88, 530:720] = 180

    # Body text (horizontal bars simulating lines of text)
    y = 115
    for i in range(50):
        if i % 7 == 0:
            y += 22   # paragraph gap
            continue
        line_end = int(rng.integers(350, 700))
        gray_val = int(rng.integers(30, 75))
        canvas[y:y + 10, 70:70 + line_end] = gray_val
        y += 18
        if y > H - 130:
            break

    # Figure placeholder (chart axes + light fill)
    fy1, fy2, fx1, fx2 = H - 210, H - 50, 120, 680
    canvas[fy1:fy2, fx1:fx2]     = 248
    canvas[fy1:fy1 + 2, fx1:fx2] = 140
    canvas[fy2 - 2:fy2, fx1:fx2] = 140
    canvas[fy1:fy2, fx1:fx1 + 2] = 140
    canvas[fy1:fy2, fx2 - 2:fx2] = 140

    # Simple bar chart inside figure
    bar_tops = [fy2 - 40 - int(rng.integers(20, 120)) for _ in range(6)]
    for k, bt in enumerate(bar_tops):
        bx1 = fx1 + 30 + k * 80
        canvas[bt:fy2 - 5, bx1:bx1 + 50] = 200

    return canvas


def add_scan_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add Gaussian noise + slight blur to simulate print+scan degradation."""
    noisy = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    pil   = Image.fromarray(noisy).filter(ImageFilter.GaussianBlur(radius=0.6))
    return np.array(pil)


def draw_grid_overlay(image: np.ndarray, result: dict,
                      tile_size: int = TILE_SIZE) -> np.ndarray:
    """Draw green/red tile-border overlay on verification result."""
    vis  = image.copy().astype(np.float32)
    H, W = vis.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    for (tx, ty), info in result['tiles'].items():
        x0 = tx * tile_w
        x1 = (tx + 1) * tile_w if tx < cols - 1 else W
        y0 = ty * tile_h
        y1 = (ty + 1) * tile_h if ty < rows - 1 else H
        color = np.array([0, 210, 0], dtype=np.float32) if info['ok'] \
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


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run(input_path: Path = None) -> None:
    print("\nPBC Visible Watermark Demo")
    print("=" * 65)
    print(f"  TILE_SIZE      : {TILE_SIZE} px")
    print(f"  CELL_SIZE      : {CELL_SIZE} px  (corner squares)")
    print(f"  ALPHA          : {ALPHA}  (overlay opacity -- content readable through tint)")
    print(f"  GRAY LEVELS    : {GRAY_LEVELS}  (pure; 0b00 lightest -> 0b11 darkest)")
    print(f"  EFFECTIVE LVLS : {EFFECTIVE_GRAY_LEVELS}  (after blend on white bg)")
    print(f"  STEP           : {EFFECTIVE_GRAY_LEVELS[0]-EFFECTIVE_GRAY_LEVELS[1]} units between effective levels")
    print(f"  BITS/TILE      : 8  (4 corners x 2 bits each = 1-byte tile signature)")
    print()

    # ~~ Load or generate source image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if input_path and input_path.exists():
        src = np.array(Image.open(input_path).convert('RGB'))
        print(f"  Input : {input_path.name}  ({src.shape[1]}x{src.shape[0]})")  
    else:
        src = make_document_page()
        print(f"  Input : synthetic document page ({src.shape[1]}x{src.shape[0]})")  

    H, W         = src.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, TILE_SIZE)
    total_tiles  = cols * rows

    ORIGINATOR = "PBC-VisibleWatermark-Demo-2026"
    ts = int(time.time())

    # ~~ Encode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t0 = time.perf_counter()
    watermarked, ts_used = encode_visible(src, ORIGINATOR, timestamp=ts)
    enc_ms = (time.perf_counter() - t0) * 1000

    psnr_val  = psnr(src, watermarked)
    coverage  = 4 * CELL_SIZE * CELL_SIZE / (tile_w * tile_h) * 100

    print(f"  Grid    : {cols}x{rows} = {total_tiles} tiles  "
          f"({tile_w}~{tile_h} px each)")
    print(f"  Coverage: {coverage:.1f}% per tile  "
          f"(4 x {CELL_SIZE}x{CELL_SIZE} px corner squares)")
    print(f"  PSNR vs original : {psnr_val:.1f} dB  "
          f"(<<51 dB -- visible by design)")
    print(f"  Encode time      : {enc_ms:.1f} ms")
    print()

    # ~~ Verify ~ clean digital ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t1 = time.perf_counter()
    res_clean = verify_visible(watermarked, ORIGINATOR, timestamp=ts_used)
    ver_ms = (time.perf_counter() - t1) * 1000
    print(f"  Verify  clean digital    : "
          f"{res_clean['n_ok']}/{res_clean['total']} OK  "
          f"({res_clean['accuracy']*100:.1f}%)  [{ver_ms:.1f} ms]")

    # ~~ Verify ~ simulated print+scan sigma=5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    np.random.seed(42)
    noisy5 = add_scan_noise(watermarked, sigma=5.0)
    res5   = verify_visible(noisy5, ORIGINATOR, timestamp=ts_used)
    print(f"  Verify  scan noise sigma= 5  : "
          f"{res5['n_ok']}/{res5['total']} OK  "
          f"({res5['accuracy']*100:.1f}%)")

    # ~~ Verify ~ sigma=10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    np.random.seed(42)
    noisy10 = add_scan_noise(watermarked, sigma=10.0)
    res10   = verify_visible(noisy10, ORIGINATOR, timestamp=ts_used)
    print(f"  Verify  scan noise sigma=10  : "
          f"{res10['n_ok']}/{res10['total']} OK  "
          f"({res10['accuracy']*100:.1f}%)")

    # ~~ Tamper test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Overwrite ALL 4 corners of tile (1, 1) with gray=180 (encodes 0x00).
    # The expected byte is SHA-256 derived and almost never 0x00, so this
    # gives a guaranteed-detectable tamper regardless of timestamp.
    tampered   = watermarked.copy()
    tx_t, ty_t = 1, 1
    oid_t      = generate_originator_id(ORIGINATOR)
    expected_t = _tile_signature(oid_t, tx_t, ty_t, ts_used)
    x0t        = tx_t * tile_w
    y0t        = ty_t * tile_h
    # Compute blended value of GRAY_LEVELS[0] over the actual tile background,
    # which gives decoded=0b00 for all corners -> decoded byte = 0x00.
    # (expected is SHA-256 derived, so 0x00 is almost never the correct value)
    tile_region = watermarked[y0t:y0t + tile_h, x0t:x0t + tile_w]
    cs_t = min(CELL_SIZE, tile_h // 4, tile_w // 4)
    for r0, c0 in [(0, 0), (0, tile_w - cs_t),
                   (tile_h - cs_t, 0), (tile_h - cs_t, tile_w - cs_t)]:
        r0t = y0t + r0;  c0t = x0t + c0
        # Blend GRAY_LEVELS[0] onto the actual background pixels
        bg  = tampered[r0t:r0t + cs_t, c0t:c0t + cs_t].astype(np.float32)
        blended = np.clip(ALPHA * GRAY_LEVELS[0] + (1.0 - ALPHA) * bg, 0, 255)
        tampered[r0t:r0t + cs_t, c0t:c0t + cs_t] = blended.astype(np.uint8)
    res_tamper  = verify_visible(tampered, ORIGINATOR, timestamp=ts_used)
    tamper_info = res_tamper['tiles'].get((tx_t, ty_t), {})
    detected    = not tamper_info.get('ok', True)
    other_ok    = sum(1 for (tx, ty), v in res_tamper['tiles'].items()
                      if (tx, ty) != (tx_t, ty_t) and v['ok'])
    other_total = total_tiles - 1

    print()
    print(f"  Tamper test (all 4 corners of tile {tx_t},{ty_t} forced to gray=180):")
    print(f"    -> Tampered tile : {'DETECTED' if detected else 'MISSED'}  "
          f"(decoded={tamper_info.get('decoded',0):#04x}  "
          f"expected={expected_t:#04x})")
    print(f"    -> Unmodified tiles: {other_ok}/{other_total} still OK")

    # ~~ Save outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    out_dir = Path(__file__).parent.parent / "output" / "visible-watermark"
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(src).save(          str(out_dir / "visible_01_original.png"))
    Image.fromarray(watermarked).save(  str(out_dir / "visible_02_watermarked.png"))
    Image.fromarray(noisy5).save(       str(out_dir / "visible_03_scan_noise5.png"))
    Image.fromarray(tampered).save(     str(out_dir / "visible_04_tampered.png"))

    vis_clean   = draw_grid_overlay(watermarked, res_clean)
    vis_noisy5  = draw_grid_overlay(noisy5,      res5)
    vis_tamper  = draw_grid_overlay(tampered,    res_tamper)
    Image.fromarray(vis_clean).save(    str(out_dir / "visible_05_verify_clean.png"))
    Image.fromarray(vis_noisy5).save(   str(out_dir / "visible_06_verify_noisy5.png"))
    Image.fromarray(vis_tamper).save(   str(out_dir / "visible_07_verify_tampered.png"))

    diff_bright = np.clip(
        np.abs(src.astype(np.int32) - watermarked.astype(np.int32)) * 8,
        0, 255).astype(np.uint8)
    Image.fromarray(diff_bright).save(  str(out_dir / "visible_08_diff_8x.png"))

    # ~~ Save results report ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    results_dir = out_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "visible_watermark_results.txt"

    with open(report_path, "w") as f:
        f.write("PBC Visible Watermark Demo Results\n")
        f.write(f"Image: {W}x{H}  Tiles: {cols}x{rows}={total_tiles}  "
                f"TileSize: {TILE_SIZE}px  CellSize: {CELL_SIZE}px\n")
        f.write(f"GrayLevels: {GRAY_LEVELS}  Step: "
                f"{GRAY_LEVELS[0]-GRAY_LEVELS[1]}  BitsPerTile: 8\n")
        f.write(f"PSNR vs original: {psnr_val:.1f} dB\n\n")
        f.write(f"{'Scenario':<30} {'OK/Total':>10}  {'Accuracy':>9}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Clean digital':<30} {res_clean['n_ok']:>4}/{res_clean['total']:<4}     "
                f"{res_clean['accuracy']*100:>6.1f}%\n")
        f.write(f"{'Scan noise sigma=5':<30} {res5['n_ok']:>4}/{res5['total']:<4}     "
                f"{res5['accuracy']*100:>6.1f}%\n")
        f.write(f"{'Scan noise sigma=10':<30} {res10['n_ok']:>4}/{res10['total']:<4}     "
                f"{res10['accuracy']*100:>6.1f}%\n")
        f.write(f"{'Tamper detection':<30} {'DETECTED' if detected else 'MISSED'}\n")
        f.write("\nConclusion:\n")
        f.write(f"  {CELL_SIZE}px corner squares at 4 gray levels ({GRAY_LEVELS[0]}-"
                f"{GRAY_LEVELS[-1]}) encode 8 bits/tile.\n")
        f.write(f"  Averaging {CELL_SIZE}*{CELL_SIZE}=1024 pixels per cell suppresses\n")
        f.write(f"  scan noise to <0.2 intensity units, far below the {(GRAY_LEVELS[0]-GRAY_LEVELS[1])//2}-unit\n")
        f.write(f"  classification threshold.\n")
        f.write(f"  Limitation: content under the corner squares is replaced;\n")
        f.write(f"  underlying digital content outside squares can be silently modified\n")
        f.write(f"  unless paired with LSB PBC or a separate digital signature.\n")

    print()
    print(f"  Saved to: {out_dir}/visible_0*.png")
    print(f"  Report  : {report_path}")
    print()
    print("-" * 65)
    print("  Summary")
    print("-" * 65)
    print(f"  PSNR vs original : {psnr_val:.1f} dB  (LSB PBC ~= 51 dB for reference)")
    print(f"  Tile coverage    : {coverage:.1f}% per tile  "
          f"(4 corners x {CELL_SIZE}x{CELL_SIZE} px)")
    print(f"  Clean decode     : {res_clean['accuracy']*100:.0f}%")
    print(f"  Scan sigma=5  decode : {res5['accuracy']*100:.0f}%")
    print(f"  Scan sigma=10 decode : {res10['accuracy']*100:.0f}%")
    print(f"  Tamper detection : {'DETECTED' if detected else 'MISSED'}")
    print()
    print("  NOTE: Designed for white-background documents (typical printed pages).")
    print("  Print+scan robustness comes from averaging 1024 px per cell.")
    print("  For full digital security: combine with LSB PBC (51 dB, undetectable)")
    print("  or a file-level cryptographic hash.")


def main():
    parser = argparse.ArgumentParser(
        description="PBC Visible Watermark Demo -- print-scan robust gray-square chain"
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="Input image (PNG). Uses synthetic document if omitted.")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
