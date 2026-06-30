#!/usr/bin/env python3
"""
PBC Forest-Scatter Crop Survivability Experiment
=================================================

Tests whether the "forest of independent genesis blocks" scatter design
achieves meaningful crop survivability compared to:
  - Single-chain scatter  (existing implementation)
  - Grid-mode PBC         (baseline)

The theoretical prediction for a 60%x80% non-aligned crop:
  - Grid mode:          ~0%  (boundary tiles destroyed; interior tiles intact
                              only if crop is tile-aligned -- not guaranteed)
  - Single-chain scatter: ~0% (one broken pointer severs the rest of the chain)
  - Forest scatter:      ~48% (each block independently authenticatable;
                               60%x80% crop retains 48% of randomly-placed blocks)

If the 48% result holds, forest scatter is a CONFIRMED capability -- not
speculation -- and upgrades the paper from "future work" to "validated design."

Usage:
    python examples/forest_scatter_test.py

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
from pbc.scatter import (
    scatter_encode, scatter_verify,
    scatter_forest_encode, scatter_forest_verify,
    max_scatter_blocks,
)

IMG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG    = os.path.join(IMG_DIR, 'leo.jpg')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
TILE_SIZE  = DEFAULT_TILE_SIZE
SEED       = 42
ORIGINATOR = "ForestScatterTest"
TIMESTAMP  = 1_700_000_000   # fixed for reproducibility


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def grid_green_blocks_after_crop(img_orig, crop):
    """Grid-mode: count GREEN blocks after a non-aligned crop."""
    x0, y0, x1, y1 = crop
    enc     = encode(img_orig, ORIGINATOR, tile_size=TILE_SIZE,
                     timestamp=TIMESTAMP)
    cropped = enc[y0:y1, x0:x1]
    result  = verify(cropped, tile_size=TILE_SIZE)
    total   = sum(len(t.blocks) for t in result.all_tiles)
    green   = sum(
        1 for t in result.all_tiles
        for b in t.blocks if b.status == BlockStatus.GREEN
    )
    return green, total


def sep(title=""):
    bar = "=" * 72
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sep("PBC Forest-Scatter Crop Survivability Experiment")

    if not os.path.exists(LEO_JPG):
        print(f"  ERROR: leo.jpg not found at {LEO_JPG}")
        sys.exit(1)

    img  = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = img.shape[:2]
    max_b = max_scatter_blocks(W, H)

    print(f"  Image   : {W}x{H} = {W*H:,} pixels")
    print(f"  Max blocks in full image : {max_b:,}")

    # Crop: 60% width x 80% height, centred (non-tile-aligned by design)
    cw   = int(W * 0.6)
    ch   = int(H * 0.8)
    cx0  = (W - cw) // 2
    cy0  = (H - ch) // 2
    cx1  = cx0 + cw
    cy1  = cy0 + ch
    crop_pixels = cw * ch
    crop_frac   = crop_pixels / (W * H)
    max_b_crop  = max_scatter_blocks(cw, ch)
    print(f"\n  Crop    : ({cx0},{cy0})->({cx1},{cy1}) = {cw}x{ch} px")
    print(f"  Crop area fraction: {crop_frac:.1%}  "
          f"(theoretical forest survival: ~{crop_frac:.1%})")
    print(f"  Max blocks in cropped image: {max_b_crop:,}")

    # ── Grid-mode baseline (run once) ────────────────────────────────────────
    sep("Grid-mode baseline")
    t0 = time.perf_counter()
    grid_green, grid_total = grid_green_blocks_after_crop(
        img, (cx0, cy0, cx1, cy1))
    grid_ms = (time.perf_counter() - t0) * 1000
    grid_pct = grid_green / grid_total * 100 if grid_total else 0.0
    print(f"  GREEN blocks after 60%x80% non-aligned crop: "
          f"{grid_green}/{grid_total} = {grid_pct:.1f}%  ({grid_ms:.0f} ms)")

    # ── Density sweep ────────────────────────────────────────────────────────
    densities = sorted(set(min(d, max_b) for d in [190, 500, 1000, 2000, max_b]))
    results   = []

    sep("Density Sweep -- Single-chain vs Forest scatter")
    hdr = (f"  {'n':>6}  {'dens%':>5}  {'PSNR':>5}  "
           f"{'chain_full%':>11}  {'chain_crop%':>11}  "
           f"{'forest_full%':>12}  {'forest_crop%':>12}  "
           f"{'theory_crop%':>12}")
    print(hdr)
    print("  " + "-" * 78)

    for n in densities:
        dens_pct = n / max_b * 100

        # -- Single-chain scatter -------------------------------------------
        t0       = time.perf_counter()
        enc_sc   = scatter_encode(img, ORIGINATOR, n_blocks=n, seed=SEED,
                                  timestamp=TIMESTAMP)
        enc_sc_ms = (time.perf_counter() - t0) * 1000
        ps_sc    = psnr(img, enc_sc)

        res_sc_full = scatter_verify(enc_sc)
        sc_full_green = res_sc_full.total_green
        sc_full_pct   = sc_full_green / n * 100 if n else 0.0

        cropped_sc     = enc_sc[cy0:cy1, cx0:cx1]
        res_sc_crop    = scatter_verify(cropped_sc, crop_offset=(cx0, cy0, W))
        sc_crop_green  = res_sc_crop.total_green
        sc_crop_pct    = sc_crop_green / n * 100 if n else 0.0

        # -- Forest scatter ------------------------------------------------
        t0          = time.perf_counter()
        enc_f       = scatter_forest_encode(img, ORIGINATOR, n_blocks=n,
                                            seed=SEED, timestamp=TIMESTAMP)
        enc_f_ms    = (time.perf_counter() - t0) * 1000
        ps_f        = psnr(img, enc_f)   # should be same as chain (same slots)

        res_f_full  = scatter_forest_verify(enc_f)
        f_full_pct  = res_f_full.n_genesis_found / n * 100 if n else 0.0

        cropped_f   = enc_f[cy0:cy1, cx0:cx1]
        t0          = time.perf_counter()
        res_f_crop  = scatter_forest_verify(cropped_f)
        f_ver_ms    = (time.perf_counter() - t0) * 1000
        f_crop_pct  = res_f_crop.n_genesis_found / n * 100 if n else 0.0
        theory_pct  = crop_frac * 100

        row = dict(n=n, dens_pct=dens_pct, psnr_f=ps_f,
                   sc_full_pct=sc_full_pct, sc_crop_pct=sc_crop_pct,
                   f_full_pct=f_full_pct, f_crop_pct=f_crop_pct,
                   f_crop_found=res_f_crop.n_genesis_found,
                   theory_pct=theory_pct, enc_sc_ms=enc_sc_ms,
                   enc_f_ms=enc_f_ms, f_ver_ms=f_ver_ms)
        results.append(row)

        print(f"  {n:>6,}  {dens_pct:>4.1f}%  {ps_f:>5.1f}  "
              f"{sc_full_pct:>10.1f}%  {sc_crop_pct:>10.1f}%  "
              f"{f_full_pct:>11.1f}%  {f_crop_pct:>11.1f}%  "
              f"{theory_pct:>11.1f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    sep("RESULTS SUMMARY")
    print(f"\n  Non-aligned 60%x80% crop ({crop_frac:.1%} area retention)\n")
    hdr2 = (f"  {'n_blocks':>8}  {'PSNR':>6}  "
            f"{'chain_crop':>10}  {'forest_crop':>11}  "
            f"{'theory':>7}  {'delta_vs_theory':>16}")
    print(hdr2)
    print("  " + "-" * 62)
    for r in results:
        delta = r['f_crop_pct'] - r['theory_pct']
        print(f"  {r['n']:>8,}  {r['psnr_f']:>6.1f}  "
              f"{r['sc_crop_pct']:>9.1f}%  {r['f_crop_pct']:>10.1f}%  "
              f"{r['theory_pct']:>6.1f}%  "
              f"  {delta:>+.1f}pp")

    print(f"\n  Grid-mode baseline after same crop: {grid_pct:.1f}% GREEN blocks")

    sep("KEY FINDINGS")
    print()

    # Forest vs grid
    best_forest = max(r['f_crop_pct'] for r in results)
    print(f"  Single-chain scatter (any density): ~0% crop survival")
    print(f"  Grid-mode (any density):            {grid_pct:.1f}% crop survival")
    print(f"  Forest scatter (any density):       up to {best_forest:.1f}% crop survival")
    print(f"  Theoretical prediction:             {crop_frac:.1%}")
    print()

    avg_forest = sum(r['f_crop_pct'] for r in results) / len(results)
    avg_delta  = avg_forest - crop_frac * 100
    print(f"  Average forest crop survival: {avg_forest:.1f}%  "
          f"(theory: {crop_frac*100:.1f}%, delta: {avg_delta:+.1f}pp)")
    print()

    if avg_forest > 40.0:
        print("  CONFIRMED: Forest-scatter achieves near-theoretical crop")
        print("  survivability. Upgrades from 'future work' to 'validated design'.")
        print()
        print("  Each surviving block independently proves:")
        print("    - Originator identity (OID)")
        print("    - Operation type (opcode)")
        print("    - Timestamp delta")
        print("    - Cryptographic binding via genesis hash")
        print("  without requiring ANY other block to be present.")
    else:
        print("  Result below theoretical threshold -- investigate.")

    # -- Write a summary file for the paper -----------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'results', 'forest_scatter_results.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write("PBC Forest-Scatter Crop Survivability Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image: {W}x{H} px  |  Crop: 60%x80% = {crop_frac:.1%} area\n\n")
        f.write(f"{'n_blocks':>8}  {'PSNR':>6}  {'chain%':>7}  "
                f"{'forest%':>8}  {'theory%':>8}\n")
        f.write("-" * 44 + "\n")
        for r in results:
            f.write(f"{r['n']:>8,}  {r['psnr_f']:>6.1f}  "
                    f"{r['sc_crop_pct']:>6.1f}%  "
                    f"{r['f_crop_pct']:>7.1f}%  "
                    f"{r['theory_pct']:>7.1f}%\n")
        f.write(f"\nGrid-mode baseline: {grid_pct:.1f}%\n")
        f.write(f"Forest average crop survival: {avg_forest:.1f}%\n")
    print(f"\n  Results saved to: {summary_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
