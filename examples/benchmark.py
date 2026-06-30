#!/usr/bin/env python3
"""
PBC Benchmarking Suite

Measures encoding / verification time and PSNR across six image sizes.
Generates CSV data and plots for Section 5 of the paper.

Usage:
    python examples/benchmark.py

Output files (in ./output/):
    benchmark_results.csv   — raw data (all runs)
    benchmark_summary.csv   — mean ± std per image size
    benchmark_timing.png    — encoding & verification time vs image size
    benchmark_psnr.png      — PSNR vs image size
    benchmark_scaling.png   — time/pixel vs image size (should be flat)

MIT License - Copyright (c) 2026 François Légaré
"""

import sys
import os
import time
import csv
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import compute_grid, DEFAULT_TILE_SIZE, BITS_PER_PIXEL, BLOCK_BITS
from pbc.encoder import encode
from pbc.decoder import verify

TILE_SIZE    = DEFAULT_TILE_SIZE   # 128
IMAGE_SIZES  = [256, 512, 1024, 2048, 4096]  # square side lengths
RUNS_PER_SIZE = 10                  # set to 30 for paper; 10 is faster for dev
ORIGINATOR   = "pbc-benchmark"


def synthetic_image(size: int) -> np.ndarray:
    """Create a synthetic RGB image with varied content (not uniform white)."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return img


def psnr(original: np.ndarray, encoded: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - encoded.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255.0 ** 2 / mse)


def run_benchmark(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    raw_rows = []

    print("PBC Benchmark")
    print("=" * 70)
    print(f"{'Size':>9}  {'Grid':>7}  {'Blocks':>7}  "
          f"{'Enc(ms)':>9}  {'Ver(ms)':>9}  {'PSNR':>7}")
    print("-" * 70)

    for size in IMAGE_SIZES:
        img = synthetic_image(size)
        cols, rows, tile_w, tile_h = compute_grid(size, size, TILE_SIZE)
        total_pixels = size * size
        num_blocks   = (total_pixels * BITS_PER_PIXEL) // BLOCK_BITS

        enc_times = []
        ver_times = []
        psnr_vals = []

        for run in range(RUNS_PER_SIZE):
            t0 = time.perf_counter()
            enc = encode(img, originator=ORIGINATOR, tile_size=TILE_SIZE)
            enc_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            verify(enc, tile_size=TILE_SIZE)
            ver_ms = (time.perf_counter() - t0) * 1000

            p = psnr(img, enc)

            enc_times.append(enc_ms)
            ver_times.append(ver_ms)
            psnr_vals.append(p)

            raw_rows.append({
                'size': size,
                'run': run,
                'cols': cols, 'rows': rows,
                'total_blocks': num_blocks,
                'encode_ms': enc_ms,
                'verify_ms': ver_ms,
                'psnr_db':   p,
            })

        enc_mean = np.mean(enc_times)
        enc_std  = np.std(enc_times)
        ver_mean = np.mean(ver_times)
        ver_std  = np.std(ver_times)
        psnr_mean = np.mean(psnr_vals)
        psnr_std  = np.std(psnr_vals)

        print(f"{size:>4}x{size:<4}  {cols}x{rows:>3}  {num_blocks:>7,}  "
              f"{enc_mean:>7.1f}±{enc_std:<4.1f}  "
              f"{ver_mean:>7.1f}±{ver_std:<4.1f}  "
              f"{psnr_mean:>5.1f}")

    print("=" * 70)
    print()

    # Write raw CSV
    results_dir = os.path.join(os.path.dirname(output_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    raw_path = os.path.join(results_dir, 'benchmark_results.csv')
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Raw data -> {raw_path}")

    # Write summary CSV
    summary_rows = []
    for size in IMAGE_SIZES:
        subset = [r for r in raw_rows if r['size'] == size]
        enc_times = [r['encode_ms'] for r in subset]
        ver_times = [r['verify_ms'] for r in subset]
        psnr_vals = [r['psnr_db']   for r in subset]
        r0 = subset[0]
        summary_rows.append({
            'size': size,
            'cols': r0['cols'],
            'rows': r0['rows'],
            'total_blocks': r0['total_blocks'],
            'encode_ms_mean': f"{np.mean(enc_times):.2f}",
            'encode_ms_std':  f"{np.std(enc_times):.2f}",
            'verify_ms_mean': f"{np.mean(ver_times):.2f}",
            'verify_ms_std':  f"{np.std(ver_times):.2f}",
            'psnr_mean': f"{np.mean(psnr_vals):.2f}",
            'psnr_std':  f"{np.std(psnr_vals):.3f}",
        })

    summary_path = os.path.join(results_dir, 'benchmark_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary  -> {summary_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sizes_px = [r['size'] ** 2 for r in summary_rows]
        labels   = [f"{r['size']}²" for r in summary_rows]
        enc_means = [float(r['encode_ms_mean']) for r in summary_rows]
        enc_stds  = [float(r['encode_ms_std'])  for r in summary_rows]
        ver_means = [float(r['verify_ms_mean']) for r in summary_rows]
        ver_stds  = [float(r['verify_ms_std'])  for r in summary_rows]
        psnr_means = [float(r['psnr_mean']) for r in summary_rows]
        psnr_stds  = [float(r['psnr_std'])  for r in summary_rows]

        x = range(len(IMAGE_SIZES))

        # --- Timing ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(x, enc_means, yerr=enc_stds, fmt='o-',
                    color='steelblue', capsize=4, label='Encode')
        ax.errorbar(x, ver_means, yerr=ver_stds, fmt='s--',
                    color='darkorange', capsize=4, label='Verify')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Image size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'PBC encoding & verification time (tile={TILE_SIZE}px, n={RUNS_PER_SIZE} runs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        timing_path = os.path.join(output_dir, 'benchmark_timing.png')
        fig.savefig(timing_path, dpi=150)
        plt.close(fig)
        print(f"Timing   -> {timing_path}")

        # PDF version for paper
        timing_pdf = os.path.join(output_dir, 'benchmark_timing.pdf')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(x, enc_means, yerr=enc_stds, fmt='o-',
                    color='steelblue', capsize=4, label='Encode')
        ax.errorbar(x, ver_means, yerr=ver_stds, fmt='s--',
                    color='darkorange', capsize=4, label='Verify')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Image size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'PBC encoding & verification time (tile={TILE_SIZE}px)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(timing_pdf)
        plt.close(fig)
        print(f"Timing   -> {timing_pdf} (PDF)")

        # --- PSNR ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(x, psnr_means, yerr=psnr_stds, fmt='D-',
                    color='green', capsize=4)
        ax.axhline(44.1, color='red', linestyle='--', alpha=0.7,
                   label='Theoretical lower bound 44.1 dB')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Image size')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR after PBC encoding (should be ~44 dB, size-independent)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        psnr_path = os.path.join(output_dir, 'benchmark_psnr.png')
        fig.savefig(psnr_path, dpi=150)
        plt.close(fig)
        print(f"PSNR     -> {psnr_path}")

        # --- Scaling: time/pixel ---
        enc_per_px = [enc_means[i] / sizes_px[i] * 1e6
                      for i in range(len(IMAGE_SIZES))]
        ver_per_px = [ver_means[i] / sizes_px[i] * 1e6
                      for i in range(len(IMAGE_SIZES))]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, enc_per_px, 'o-', color='steelblue', label='Encode')
        ax.plot(x, ver_per_px, 's--', color='darkorange', label='Verify')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Image size')
        ax.set_ylabel('Time per million pixels (ms)')
        ax.set_title('Per-pixel scaling (flat = O(n) complexity)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scaling_path = os.path.join(output_dir, 'benchmark_scaling.png')
        fig.savefig(scaling_path, dpi=150)
        plt.close(fig)
        print(f"Scaling  -> {scaling_path}")

    except ImportError:
        print("matplotlib not available — skipping plots")

    print()
    print("LaTeX table rows (paste into paper):")
    print()
    for r in summary_rows:
        size = r['size']
        cols = r['cols']
        rows_ = r['rows']
        blk  = int(r['total_blocks'])
        enc  = f"{float(r['encode_ms_mean']):.0f}"
        enc_s = f"{float(r['encode_ms_std']):.0f}"
        ver  = f"{float(r['verify_ms_mean']):.0f}"
        ver_s = f"{float(r['verify_ms_std']):.0f}"
        p    = f"{float(r['psnr_mean']):.1f}"
        p_s  = f"{float(r['psnr_std']):.2f}"
        print(f"${size}^2$ & ${cols}\\times{rows_}$ & {blk:,} & "
              f"${enc}\\pm{enc_s}$ & ${ver}\\pm{ver_s}$ & ${p}\\pm{p_s}$ \\\\")


if __name__ == '__main__':
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'benchmark')
    run_benchmark(output_dir)
