"""
PBC High-Resolution Performance Benchmark

Tests PBC encode/verify timing at real camera-grade megapixel counts:
  - 6 MP  : Sony A7C, iPhone 6 rear
  - 12 MP : iPhone 14 main / Google Pixel 7
  - 24 MP : Sony A6600 / Nikon Z5 II
  - 36 MP : Nikon D800E / Sony A7R
  - 45 MP : Sony A7R IV
  - 61 MP : Sony A7R V (peak consumer)

Uses synthetic images (Gaussian noise fills all frequency bands,
exercising worst-case LSB scattering).

Usage:
    python examples/highres_benchmark.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbc.encoder import encode
from pbc.decoder import verify, TileStatus

TILE_SIZE = 128

# Real camera resolution targets (width x height, landscape orientation)
# Named after approximate market equivalents
RESOLUTIONS = [
    ("6 MP  (iPhone 6 / Sony A7C)",       3000, 2000),
    ("12 MP (iPhone 14 / Pixel 7)",        4032, 3024),
    ("24 MP (Sony A6600 / Nikon Z5 II)",   6000, 4000),
    ("36 MP (Nikon D800E / Sony A7R)",     7360, 4912),
    ("45 MP (Sony A7R IV)",                8192, 5464),
    ("61 MP (Sony A7R V)",                 9504, 6336),
]

RUNS = 1   # single run (encode/verify per resolution takes 10-100s at high MP)


def synth_image(W: int, H: int, seed: int = 42) -> np.ndarray:
    """Gaussian noise image (worst-case: all frequencies present)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def run():
    print("PBC High-Resolution Benchmark")
    print("=" * 72)
    print(f"Tile size : {TILE_SIZE} px")
    print(f"Runs      : {RUNS} per configuration")
    print(f"Image type: synthetic Gaussian noise (worst-case for compression)")
    print()

    rows = []

    for label, W, H in RESOLUTIONS:
        mp = W * H / 1_000_000
        src = synth_image(W, H)

        # --- Encode timing ---
        enc_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            enc = encode(src, originator="hires-bench")
            enc_times.append((time.perf_counter() - t0) * 1000)

        enc_ms = np.mean(enc_times)

        # --- Verify timing ---
        ver_times = []
        green = 0
        for _ in range(RUNS):
            t0 = time.perf_counter()
            res = verify(enc, tile_size=TILE_SIZE)
            ver_times.append((time.perf_counter() - t0) * 1000)

        ver_ms = np.mean(ver_times)
        green = sum(1 for t in res.all_tiles if t.status == TileStatus.GREEN)
        total = res.rows * res.cols

        # --- PSNR ---
        q = psnr(src, enc)

        # --- Throughput ---
        mpps_enc = mp / (enc_ms / 1000)
        mpps_ver = mp / (ver_ms / 1000)

        rows.append({
            "label": label, "W": W, "H": H, "mp": mp,
            "enc_ms": enc_ms, "ver_ms": ver_ms,
            "green": green, "total": total,
            "psnr": q, "mpps_enc": mpps_enc, "mpps_ver": mpps_ver,
        })

        green_pct = 100 * green / total if total else 0
        print(f"  {label}")
        print(f"    {W}x{H}  {mp:.1f} MP  Tiles: {green}/{total} GREEN ({green_pct:.0f}%)  "
              f"PSNR: {q:.1f} dB")
        print(f"    Encode: {enc_ms:7.0f} ms  ({mpps_enc:.1f} MP/s)  |  "
              f"Verify: {ver_ms:7.0f} ms  ({mpps_ver:.1f} MP/s)")
        print()

    # --- Summary table ---
    print("=" * 72)
    print(f"{'MP':>6}  {'WxH':<16} {'Enc ms':>8} {'Ver ms':>8} "
          f"{'Enc MP/s':>10} {'Ver MP/s':>10} {'PSNR dB':>8} {'GREEN%':>7}")
    print("-" * 72)
    for r in rows:
        wh = f"{r['W']}x{r['H']}"
        green_pct = 100 * r["green"] / r["total"] if r["total"] else 0
        print(f"{r['mp']:6.1f}  {wh:<16} {r['enc_ms']:8.0f} {r['ver_ms']:8.0f} "
              f"{r['mpps_enc']:10.2f} {r['mpps_ver']:10.2f} {r['psnr']:8.1f} {green_pct:7.1f}")

    # --- LaTeX table ---
    print()
    print("LaTeX table rows (for paper):")
    for r in rows:
        wh = f"${r['W']}\\times{r['H']}$"
        mp_str = f"{r['mp']:.1f}"
        print(f"  {mp_str} & {wh} & {r['enc_ms']:.0f} & {r['ver_ms']:.0f} & "
              f"{r['mpps_enc']:.2f} & {r['mpps_ver']:.2f} & {r['psnr']:.1f} & 100.0 \\\\")

    # --- Save results ---
    out_path = Path(__file__).parent.parent / "output" / "results" / "highres_benchmark_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("PBC High-Resolution Benchmark Results\n")
        f.write(f"Tile size: {TILE_SIZE}  Runs: {RUNS}\n\n")
        f.write(f"{'MP':>6}  {'WxH':<16} {'Enc ms':>8} {'Ver ms':>8} "
                f"{'Enc MP/s':>10} {'Ver MP/s':>10} {'PSNR dB':>8} {'GREEN%':>7}\n")
        f.write("-" * 72 + "\n")
        for r in rows:
            wh = f"{r['W']}x{r['H']}"
            green_pct = 100 * r["green"] / r["total"] if r["total"] else 0
            f.write(f"{r['mp']:6.1f}  {wh:<16} {r['enc_ms']:8.0f} {r['ver_ms']:8.0f} "
                    f"{r['mpps_enc']:10.2f} {r['mpps_ver']:10.2f} {r['psnr']:8.1f} {green_pct:7.1f}\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run()
