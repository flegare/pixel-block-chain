#!/usr/bin/env python3
"""
PBC Append Mode Overhead Benchmark

Measures the actual overhead of append mode (Edit Ledger) vs full re-encode
at various split fractions and image sizes.

Tests the paper's Section 7.5 claim:
  "The additional cost across all tiles in a 12 MP image is estimated at
   <5% overhead relative to full re-encode."

Outputs:
  - Measured overhead % per (image_size, split_fraction)
  - Comparison table suitable for paper

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, DEFAULT_TILE_SIZE
from pbc.encoder import encode, append_edit
from pbc.decoder import verify

TILE_SIZE = DEFAULT_TILE_SIZE   # 128
RUNS      = 5                   # repeats per configuration

# Image sizes (side length of square image)
IMAGE_SIZES = [512, 1024, 2048]

# Split fractions to test (fraction of chain already written before append)
SPLIT_FRACTIONS = [0.25, 0.50, 0.75]


def synthetic_image(size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def time_fn(fn, runs):
    """Run fn() <runs> times, return mean and std in ms."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    print("PBC Append Mode Overhead Benchmark")
    print("=" * 75)
    print(f"Tile size : {TILE_SIZE} px")
    print(f"Runs      : {RUNS} per configuration")
    print()

    all_results = []

    for size in IMAGE_SIZES:
        img = synthetic_image(size)
        print(f"Image {size}x{size}  ({size*size/1e6:.1f} MP)")
        print("-" * 65)

        # Time a full re-encode (baseline)
        enc_mean, enc_std = time_fn(
            lambda: encode(img, originator="Baseline", tile_size=TILE_SIZE),
            RUNS
        )

        # Pre-encode once so append_edit has something to work with
        encoded = encode(img, originator="Capture", opcode=OpCode.CAMERA_ISP,
                         tile_size=TILE_SIZE)

        print(f"  Full re-encode  : {enc_mean:7.1f} ± {enc_std:.1f} ms  (baseline)")

        for split in SPLIT_FRACTIONS:
            # append_edit uses split_fraction internally
            app_mean, app_std = time_fn(
                lambda s=split: append_edit(
                    encoded,
                    originator="Editor",
                    opcode=OpCode.EDIT_COLOR,
                    tile_size=TILE_SIZE,
                    split_fraction=s
                ),
                RUNS
            )
            overhead_pct = (app_mean - enc_mean) / enc_mean * 100
            print(f"  Append (split={split:.2f}): {app_mean:7.1f} ± {app_std:.1f} ms"
                  f"  overhead={overhead_pct:+.1f}%")
            all_results.append({
                'size': size,
                'split': split,
                'encode_ms': enc_mean,
                'append_ms': app_mean,
                'overhead_pct': overhead_pct,
            })
        print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 75)
    print("Summary — Overhead of append mode vs full re-encode")
    print()
    print(f"  {'Size':>8}  {'Split':>6}  {'Encode ms':>10}  {'Append ms':>10}  {'Overhead':>9}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*9}")
    for r in all_results:
        print(f"  {r['size']:>4}x{r['size']:<3}  {r['split']:.2f}    "
              f"{r['encode_ms']:>9.1f}   {r['append_ms']:>9.1f}   "
              f"{r['overhead_pct']:>+8.1f}%")

    print()
    overheads = [r['overhead_pct'] for r in all_results]
    print(f"  Mean overhead across all configurations : {np.mean(overheads):+.1f}%")
    print(f"  Max  overhead across all configurations : {max(overheads):+.1f}%")
    print()

    # ------------------------------------------------------------------
    # Verdict vs paper claim
    # ------------------------------------------------------------------
    claim_pct = 5.0
    worst = max(overheads)
    print("=" * 75)
    print(f"Paper claim (Section 7.5): overhead < {claim_pct:.0f}%")
    if worst < claim_pct:
        print(f"  CONFIRMED — worst measured overhead: {worst:.1f}%  < {claim_pct:.0f}%")
    else:
        print(f"  REFUTED   — worst measured overhead: {worst:.1f}%  > {claim_pct:.0f}%")
        print(f"  (Update paper text with measured value)")
    print()

    # ------------------------------------------------------------------
    # LaTeX table rows
    # ------------------------------------------------------------------
    print("LaTeX table rows:")
    print()
    for r in all_results:
        print(f"  ${r['size']}^2$ & {r['split']:.2f} & "
              f"{r['encode_ms']:.1f} & {r['append_ms']:.1f} & "
              f"{r['overhead_pct']:+.1f}\\% \\\\")

    # ------------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------------
    out_path = Path(__file__).parent.parent / "output" / "results" / "append_mode_benchmark_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("PBC Append Mode Overhead Benchmark Results\n")
        f.write(f"Tile size: {TILE_SIZE}  Runs: {RUNS}\n\n")
        f.write(f"  {'Size':>8}  {'Split':>6}  {'Encode ms':>10}  {'Append ms':>10}  {'Overhead':>9}\n")
        f.write(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*9}\n")
        for r in all_results:
            f.write(f"  {r['size']:>4}x{r['size']:<3}  {r['split']:.2f}    "
                    f"{r['encode_ms']:>9.1f}   {r['append_ms']:>9.1f}   "
                    f"{r['overhead_pct']:>+8.1f}%\n")
        f.write(f"\n  Mean overhead: {np.mean(overheads):+.1f}%\n")
        f.write(f"  Max  overhead: {max(overheads):+.1f}%\n\n")
        f.write(f"Paper claim: overhead < {claim_pct:.0f}%\n")
        if worst < claim_pct:
            f.write(f"  CONFIRMED — worst: {worst:.1f}% < {claim_pct:.0f}%\n")
        else:
            f.write(f"  REFUTED   — worst: {worst:.1f}% > {claim_pct:.0f}%\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
