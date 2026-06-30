"""
PBC Format Robustness Test

Tests whether PBC integrity survives encode/save/reload across all common
image container formats.

Key question: does saving the PBC-encoded image in format X preserve the
embedded chain data when it is reloaded and verified?

Results for lossless formats (PNG, BMP, TIFF, WebP-lossless) should be GREEN.
Lossy formats (JPEG, WebP-lossy) are expected to FAIL (CRC corrupted).

Usage:
    python examples/format_robustness.py
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pbc.encoder import encode
from pbc.decoder import verify, TileStatus


# ---------------------------------------------------------------------------
# Reference image: use leo.jpg (known-good, 40 tiles)
# ---------------------------------------------------------------------------

IMG_PATH = Path(__file__).parent / "img" / "leo.jpg"
TILE_SIZE = 128

# Format definitions: (label, PIL save kwargs, expected_outcome)
FORMATS = [
    # Lossless formats - should all PASS
    ("PNG",           "png",  {},                              "PASS"),
    ("BMP",           "bmp",  {},                              "PASS"),
    ("TIFF-lossless", "tiff", {"compression": "tiff_lzw"},    "PASS"),
    ("TIFF-raw",      "tiff", {},                              "PASS"),
    ("WebP-lossless", "webp", {"lossless": True, "quality": 100}, "PASS"),

    # Lossy formats - known to FAIL (LSB destruction)
    ("JPEG-Q100",     "jpeg", {"quality": 100, "subsampling": 0}, "FAIL"),
    ("JPEG-Q95",      "jpeg", {"quality": 95},               "FAIL"),
    ("JPEG-Q85",      "jpeg", {"quality": 85},               "FAIL"),
    ("JPEG-Q75",      "jpeg", {"quality": 75},               "FAIL"),
    ("WebP-Q90",      "webp", {"lossless": False, "quality": 90}, "FAIL"),
    ("WebP-Q75",      "webp", {"lossless": False, "quality": 75}, "FAIL"),
]


def encode_save_reload(arr: np.ndarray, fmt: str, save_kwargs: dict) -> np.ndarray:
    """Encode with PBC, save to format, reload to numpy array."""
    encoded_arr = encode(arr, originator="format-test")
    encoded_img = Image.fromarray(encoded_arr)
    buf = io.BytesIO()
    if fmt == "jpeg":
        encoded_img = encoded_img.convert("RGB")
    encoded_img.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    reloaded = Image.open(buf).convert("RGB")
    return np.array(reloaded)


def run():
    print("PBC Format Robustness Test")
    print("=" * 65)
    print(f"Reference image: {IMG_PATH.name}")

    src = np.array(Image.open(IMG_PATH).convert("RGB"))
    H, W = src.shape[:2]
    print(f"Image: {W}x{H} px")

    # Establish baseline: encode to numpy and verify directly (no save/reload)
    baseline = encode(src, originator="format-test")
    res_base = verify(baseline, tile_size=TILE_SIZE)
    g_base = sum(1 for t in res_base.all_tiles if t.status == TileStatus.GREEN)
    total_tiles = res_base.cols * res_base.rows
    print(f"Baseline (no save):  {g_base}/{total_tiles} GREEN tiles\n")

    print(f"{'Format':<18} {'Saved KB':>9} {'GREEN':>6} {'YELLOW':>7} {'RED':>5} {'ABSENT':>7}  {'Result':<8} {'ms':>6}")
    print("-" * 75)

    results = []
    for label, fmt, kwargs, expected in FORMATS:
        t0 = time.perf_counter()
        try:
            # Save to buffer and measure size
            buf = io.BytesIO()
            enc = encode(src, originator="format-test")
            pil_enc = Image.fromarray(enc)
            if fmt == "jpeg":
                pil_enc = pil_enc.convert("RGB")
            pil_enc.save(buf, format=fmt, **kwargs)
            size_kb = buf.tell() / 1024

            # Reload and verify
            buf.seek(0)
            reloaded = np.array(Image.open(buf).convert("RGB"))
            res = verify(reloaded, tile_size=TILE_SIZE)

            g = sum(1 for t in res.all_tiles if t.status == TileStatus.GREEN)
            y = sum(1 for t in res.all_tiles if t.status == TileStatus.YELLOW)
            r = sum(1 for t in res.all_tiles if t.status == TileStatus.RED)
            a = sum(1 for t in res.all_tiles if t.status == TileStatus.ABSENT)
            elapsed = (time.perf_counter() - t0) * 1000

            # Determine outcome
            if g == total_tiles:
                outcome = "PASS"
            elif g > 0:
                outcome = "PARTIAL"
            else:
                outcome = "FAIL"

            match = "OK" if outcome == expected else "UNEXPECTED"
            status_str = f"{outcome:<8}"
            if match == "UNEXPECTED":
                status_str = f"** {outcome:<5} **"

            print(f"{label:<18} {size_kb:>9.1f} {g:>6} {y:>7} {r:>5} {a:>7}  {status_str} {elapsed:>5.0f}")
            results.append({
                "label": label, "fmt": fmt, "size_kb": size_kb,
                "green": g, "yellow": y, "red": r, "absent": a,
                "outcome": outcome, "expected": expected, "match": match,
                "elapsed_ms": elapsed
            })
        except Exception as e:
            print(f"{label:<18} {'ERROR':<10} {str(e)[:40]}")

    # Summary
    print("\n" + "=" * 65)
    passed  = [r for r in results if r["outcome"] == "PASS"]
    failed  = [r for r in results if r["outcome"] == "FAIL"]
    partial = [r for r in results if r["outcome"] == "PARTIAL"]
    unexpected = [r for r in results if r["match"] == "UNEXPECTED"]

    print(f"\nLossless formats (expected PASS):  {len(passed)} passed")
    for r in passed:
        print(f"  {r['label']:<18} {r['green']}/{total_tiles} GREEN  ({r['size_kb']:.0f} KB)")

    print(f"\nLossy formats (expected FAIL):     {len(failed)} failed (as expected)")
    for r in failed:
        print(f"  {r['label']:<18} {r['green']}/{total_tiles} GREEN  ({r['size_kb']:.0f} KB)")

    if partial:
        print(f"\nPartial survival:  {len(partial)} formats")
        for r in partial:
            print(f"  {r['label']:<18} {r['green']}/{total_tiles} GREEN")

    if unexpected:
        print(f"\n** UNEXPECTED outcomes: {len(unexpected)} **")
        for r in unexpected:
            print(f"  {r['label']}: expected {r['expected']}, got {r['outcome']}")

    # File size comparison
    png_kb = next((r["size_kb"] for r in results if r["label"] == "PNG"), 0)
    print(f"\nFile size vs PNG:")
    for r in results:
        ratio = r["size_kb"] / png_kb if png_kb else 0
        print(f"  {r['label']:<18} {r['size_kb']:>7.0f} KB  ({ratio:.2f}x PNG)")

    # Save report
    out_path = Path(__file__).parent.parent / "output" / "results" / "format_robustness_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"PBC Format Robustness Report\n")
        f.write(f"Image: {W}x{H}  Tiles: {total_tiles}\n\n")
        f.write(f"{'Format':<18} {'Outcome':<10} {'GREEN':>5}/{total_tiles}  {'Size KB':>8}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['label']:<18} {r['outcome']:<10} {r['green']:>5}/{total_tiles}  {r['size_kb']:>8.1f}\n")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    run()
