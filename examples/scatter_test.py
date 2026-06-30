#!/usr/bin/env python3
"""
PBC-Scatter Density & Crop-Survivability Experiment

For each n_blocks in [190, 500, 1000, 2000, max]:
  1. Scatter-encode leo.jpg.
  2. Verify full image -> report blocks found, GREEN%, PSNR, timing.
  3. Apply 60%x80% non-aligned crop (same as crop_survivability.py).
  4. Verify cropped image WITHOUT crop offset (genesis detection only).
  5. Verify cropped image WITH crop offset (full chain following).
  6. Compare to grid-mode verification of the same crop.

Trade-off table is printed at the end to identify the sweet spot.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import math
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import compute_grid, DEFAULT_TILE_SIZE
from pbc.encoder import encode
from pbc.decoder import verify, BlockStatus
from pbc.scatter import scatter_encode, scatter_verify, max_scatter_blocks

IMG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG  = os.path.join(IMG_DIR, 'leo.jpg')
TILE_SIZE = DEFAULT_TILE_SIZE
SEED      = 42
ORIGINATOR = "ScatterTest"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255.0 ** 2 / mse)


def grid_crop_surviving(encoded: np.ndarray, crop: tuple) -> dict:
    """Run grid-mode verify on a crop, return status counts."""
    x0, y0, x1, y1 = crop
    cropped = encoded[y0:y1, x0:x1]
    result  = verify(cropped, strict=False, tile_size=TILE_SIZE)
    counts  = {s: 0 for s in BlockStatus}
    for tile in result.all_tiles:
        for br in tile.blocks:
            counts[br.status] += 1
    green   = counts[BlockStatus.GREEN]
    total   = sum(counts.values())
    return {'green': green, 'total': total,
            'pct': green / total * 100 if total else 0.0}


def main():
    print("PBC-Scatter Density & Crop-Survivability Experiment")
    print("=" * 75)

    if not os.path.exists(LEO_JPG):
        print(f"ERROR: leo.jpg not found at {LEO_JPG}")
        sys.exit(1)

    img = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = img.shape[:2]
    print(f"Image : {W}x{H} = {W*H:,} pixels  ({W*H/1e6:.2f} MP)")
    max_b = max_scatter_blocks(W, H)
    print(f"Max scatter blocks : {max_b:,}  (= {W*H} // {86})")
    print()

    # Crop parameters: 60% width, 80% height, centred, non-tile-aligned
    cw   = int(W * 0.6)
    ch   = int(H * 0.8)
    cx0  = (W - cw) // 2
    cy0  = (H - ch) // 2
    cx1  = cx0 + cw
    cy1  = cy0 + ch
    print(f"Crop  : ({cx0},{cy0}) -> ({cx1},{cy1})  = {cw}x{ch} px  (60%x80%, non-aligned)")
    print()

    # Grid-mode baseline for this crop (encode once, crop once)
    t0 = time.perf_counter()
    grid_enc = encode(img, ORIGINATOR, tile_size=TILE_SIZE)
    grid_enc_ms = (time.perf_counter() - t0) * 1000
    grid_crop_result = grid_crop_surviving(grid_enc, (cx0, cy0, cx1, cy1))

    print(f"Grid-mode baseline:")
    print(f"  Encode time : {grid_enc_ms:.0f} ms")
    print(f"  After crop  : {grid_crop_result['green']}/{grid_crop_result['total']} "
          f"blocks GREEN = {grid_crop_result['pct']:.1f}%")
    print()

    # Density sweep
    densities = [190, 500, 1000, 2000, max_b]
    # Remove duplicates and sort
    densities = sorted(set(min(d, max_b) for d in densities))

    print("=" * 75)
    header = (f"  {'n_blocks':>8}  {'density%':>9}  {'PSNR':>6}  "
              f"{'enc_ms':>7}  {'ver_ms':>7}  "
              f"{'full_green%':>11}  {'crop_green%':>11}  "
              f"{'crop+off%':>10}  {'grid_crop%':>10}")
    print(header)
    print("  " + "-" * 73)

    results_table = []

    for n in densities:
        density_pct = n / max_b * 100

        # Encode
        t0  = time.perf_counter()
        enc = scatter_encode(img, ORIGINATOR, n_blocks=n, seed=SEED)
        enc_ms = (time.perf_counter() - t0) * 1000

        ps = psnr(img, enc)

        # Verify full image
        t0  = time.perf_counter()
        res = scatter_verify(enc)
        ver_ms = (time.perf_counter() - t0) * 1000

        full_green = res.total_green
        full_total = res.total_blocks_found
        full_pct   = full_green / full_total * 100 if full_total else 0.0

        # Verify cropped image WITHOUT crop offset
        cropped = enc[cy0:cy1, cx0:cx1]
        res_crop_no_offset = scatter_verify(cropped)
        chains_no = res_crop_no_offset.chains
        # Count: how many genesis blocks found (each = 1 surviving provenance anchor)
        genesis_found = res_crop_no_offset.n_chains
        # GREEN blocks across all partial chains (genesis + however many pointers followed)
        crop_green_no = res_crop_no_offset.total_green
        crop_total_no = res_crop_no_offset.total_blocks_found
        crop_pct_no   = crop_green_no / n * 100 if n else 0.0

        # Verify cropped image WITH crop offset (full chain following)
        res_crop_off = scatter_verify(cropped, crop_offset=(cx0, cy0, W))
        crop_green_off = res_crop_off.total_green
        crop_pct_off   = crop_green_off / n * 100 if n else 0.0

        row = {
            'n': n,
            'density_pct': density_pct,
            'psnr': ps,
            'enc_ms': enc_ms,
            'ver_ms': ver_ms,
            'full_pct': full_pct,
            'crop_pct_no': crop_pct_no,
            'crop_pct_off': crop_pct_off,
            'genesis_found': genesis_found,
            'grid_crop_pct': grid_crop_result['pct'],
        }
        results_table.append(row)

        print(f"  {n:>8,}  {density_pct:>8.1f}%  {ps:>6.1f}  "
              f"{enc_ms:>7.0f}  {ver_ms:>7.0f}  "
              f"{full_pct:>10.1f}%  {crop_pct_no:>10.1f}%  "
              f"{crop_pct_off:>9.1f}%  {grid_crop_result['pct']:>9.1f}%")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("Column guide:")
    print("  n_blocks    : blocks embedded")
    print("  density%    : fraction of image capacity used")
    print("  PSNR        : imperceptibility (dB); lower = more visible change")
    print("  enc_ms      : scatter encoding time")
    print("  ver_ms      : scatter verification time (full sync scan)")
    print("  full_green% : fraction of blocks verified GREEN on full image")
    print("  crop_green% : fraction of ORIGINAL blocks surviving crop (no offset known)")
    print("  crop+off%   : same, with crop offset provided (full chain follows)")
    print("  grid_crop%  : grid-mode GREEN after same crop (baseline comparison)")
    print()

    print("=" * 75)
    print("Key findings:")
    print()

    r_max = results_table[-1]
    print(f"  Grid-mode after non-aligned crop : {r_max['grid_crop_pct']:.1f}% GREEN blocks")
    print()
    print("  Scatter-mode after same crop:")
    for r in results_table:
        note = ""
        if r['crop_pct_off'] > r['grid_crop_pct'] + 5:
            note = " <-- scatter beats grid"
        print(f"    n={r['n']:>6,}: no-offset={r['crop_pct_no']:.1f}%  "
              f"with-offset={r['crop_pct_off']:.1f}%  PSNR={r['psnr']:.1f} dB{note}")

    print()

    # Sweet spot: highest crop survival with PSNR > 44 dB
    candidates = [r for r in results_table if r['psnr'] > 44.0]
    if candidates:
        best = max(candidates, key=lambda r: r['crop_pct_off'])
        print(f"  Sweet spot (PSNR > 44 dB, max crop survival):")
        print(f"    n_blocks={best['n']:,}  "
              f"PSNR={best['psnr']:.1f} dB  "
              f"crop survival (with offset)={best['crop_pct_off']:.1f}%")
    print()
    print("  Note on 'no-offset' crop: even without knowing the crop parameters,")
    print("  scatter detects surviving genesis blocks — any block_index==0 whose")
    print("  genesis hash validates provides independent origin authentication.")
    print("  The chain following simply stops at the first pointer outside the crop.")


if __name__ == '__main__':
    sys.exit(main())
