#!/usr/bin/env python3
"""
PBC JPEG Robustness — k=1, 2, 3 Comparison
============================================

Tests whether embedding at higher bit positions (k=2, k=3) survives JPEG
compression, as theoretically predicted by the bit-error-rate profile in
Table tab:jpeg_bits of the paper.

Expected outcome:
  k=1 (bit 0):  all ABSENT at all quality levels (42% BER at Q=100)
  k=2 (bit 1):  all ABSENT at all quality levels (19% BER at Q=100)
  k=3 (bit 2):  should pass at Q=100 (9.5% BER -> expected Hamming=4.6 < 6)
                may fail at Q<100

Trade-off:
  k=1: PSNR ~51.2 dB,  pixels_per_block = 86,  tiles = most
  k=2: PSNR ~45.2 dB,  pixels_per_block = 43,  tiles = same count
  k=3: PSNR ~36.0 dB,  pixels_per_block = 29,  tiles = same count

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import io
import math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc.encoder import encode
from pbc.decoder import verify, BlockStatus, TileStatus

IMG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
LEO_JPG   = os.path.join(IMG_DIR, 'leo.jpg')
TILE_SIZE = 128
ORIGINATOR = "jpeg_k_test"
TIMESTAMP  = 1_700_000_000   # fixed

QUALITY_LEVELS = [100, 97, 95, 90, 85, 80]
K_VALUES       = [1, 2, 3]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def jpeg_roundtrip(arr: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def count_tile_statuses(result):
    g = y = r = a = 0
    for tile in result.all_tiles:
        if   tile.status == TileStatus.GREEN:  g += 1
        elif tile.status == TileStatus.YELLOW: y += 1
        elif tile.status == TileStatus.RED:    r += 1
        else:                                   a += 1
    return g, y, r, a


def sep(title=""):
    bar = "=" * 72
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sep("PBC JPEG Robustness: k=1, 2, 3 Comparison")

    if not os.path.exists(LEO_JPG):
        print(f"  ERROR: leo.jpg not found at {LEO_JPG}")
        sys.exit(1)

    img = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = img.shape[:2]
    print(f"  Image: {W}x{H} px")

    results = {}  # (k, q) -> (green_tiles, total_tiles, psnr_encode, psnr_after_jpeg)

    sep("Results")
    hdr = f"  {'k':>2}  {'Q':>4}  {'PSNR_enc':>9}  {'PSNR_jpeg':>9}  {'GREEN':>6}  {'YELLOW':>7}  {'ABSENT':>7}  Verdict"
    print(hdr)
    print("  " + "-" * 74)

    for k in K_VALUES:
        pixels_per_block = (256 + 3*k - 1) // (3*k)
        # Encode with this k
        enc = encode(img, ORIGINATOR, timestamp=TIMESTAMP, tile_size=TILE_SIZE, k=k)
        ps_enc = psnr(img, enc)

        for q in QUALITY_LEVELS:
            after_jpeg = jpeg_roundtrip(enc, q)
            ps_jpeg = psnr(img, after_jpeg)

            result = verify(after_jpeg, tile_size=TILE_SIZE, k=k)
            total_tiles = result.cols * result.rows
            g, y, r, a = count_tile_statuses(result)

            verdict = "PASS" if g > 0 else "fail"
            results[(k, q)] = (g, total_tiles, ps_enc, ps_jpeg)

            print(f"  {k:>2}  {q:>4}  {ps_enc:>9.1f}  {ps_jpeg:>9.1f}  "
                  f"{g:>5}/{total_tiles}  {y:>6}/{total_tiles}  "
                  f"{a:>6}/{total_tiles}  {verdict}")
        print()

    # ------------------------------------------------------------------
    # Summary: which (k, Q) combinations produce at least one GREEN tile?
    # ------------------------------------------------------------------
    sep("Summary: JPEG Survival Matrix")
    print()
    print(f"  {'k \\ Q':>6}", end="")
    for q in QUALITY_LEVELS:
        print(f"  {q:>5}", end="")
    print()
    print("  " + "-" * (8 + len(QUALITY_LEVELS) * 7))
    for k in K_VALUES:
        print(f"  k={k:>1}   ", end="")
        for q in QUALITY_LEVELS:
            g, tot, _, _ = results[(k, q)]
            sym = "PASS" if g > 0 else "fail"
            print(f"  {sym:>5}", end="")
        print()

    # ------------------------------------------------------------------
    # PSNR trade-off table
    # ------------------------------------------------------------------
    sep("PSNR Trade-off (encode degradation, before JPEG)")
    print()
    print(f"  {'k':>2}  {'pixels/block':>13}  {'blocks/tile(128²)':>18}  {'PSNR (dB)':>10}")
    print("  " + "-" * 48)
    for k in K_VALUES:
        ppb = (256 + 3*k - 1) // (3*k)
        bpt = (128 * 128) // ppb
        ps_enc = results[(k, QUALITY_LEVELS[0])][2]
        print(f"  {k:>2}  {ppb:>13}  {bpt:>18}  {ps_enc:>10.1f}")

    # ------------------------------------------------------------------
    # LaTeX table rows for paper
    # ------------------------------------------------------------------
    sep("LaTeX rows for paper table")
    print()
    for k in K_VALUES:
        ppb = (256 + 3*k - 1) // (3*k)
        ps_enc = results[(k, QUALITY_LEVELS[0])][2]
        row_parts = []
        for q in QUALITY_LEVELS:
            g, tot, _, _ = results[(k, q)]
            sym = r"\checkmark" if g > 0 else r"$\times$"
            row_parts.append(sym)
        print(f"  $k={k}$ & {ppb} & {ps_enc:.1f} & "
              + " & ".join(row_parts) + r" \\")

    # ------------------------------------------------------------------
    # Write results file
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, 'results', 'jpeg_k3_results.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("PBC JPEG Robustness: k=1,2,3\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image: {W}x{H} px\n\n")
        f.write(f"{'k':>2}  {'Q':>4}  {'PSNR_enc':>9}  {'GREEN/total':>12}\n")
        f.write("-" * 32 + "\n")
        for k in K_VALUES:
            for q in QUALITY_LEVELS:
                g, tot, ps_enc, _ = results[(k, q)]
                f.write(f"{k:>2}  {q:>4}  {ps_enc:>9.1f}  {g:>5}/{tot}\n")
            f.write("\n")

    print(f"\n  Results saved to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
