#!/usr/bin/env python3
"""
PBC JPEG Compression Robustness Test

Empirically measures PBC block survival rate after JPEG re-compression
at various quality levels using the leo.jpg test image.

Replaces the theoretical Table 6 in the paper with measured values.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, BITS_PER_CHANNEL
from pbc.encoder import encode
from pbc.decoder import verify, BlockStatus

IMG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG   = os.path.join(IMG_DIR, 'leo.jpg')
TILE_SIZE = 128

# Quality levels to test -- span the full range
QUALITY_LEVELS = [100, 95, 90, 85, 80, 75, 60, 50]


def count_blocks(result):
    """Return (total, green, yellow, red, absent) across all tiles."""
    green = yellow = red = absent = 0
    for tile in result.all_tiles:
        for br in tile.blocks:
            if   br.status == BlockStatus.GREEN:  green  += 1
            elif br.status == BlockStatus.YELLOW: yellow += 1
            elif br.status == BlockStatus.RED:    red    += 1
            else:                                 absent += 1
    total = green + yellow + red + absent
    return total, green, yellow, red, absent


def main():
    print("PBC JPEG Compression Robustness Test")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Load and encode
    # ------------------------------------------------------------------
    original = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = original.shape[:2]
    print(f"Image : {W} x {H} pixels  ({W*H:,} total)")
    print(f"Tile  : {TILE_SIZE} px  (k=1 LSB/channel)")
    print()

    encoded = encode(original,
                     originator="RobustnessTest",
                     opcode=OpCode.CAMERA_ISP,
                     tile_size=TILE_SIZE)

    # Baseline: lossless PNG round-trip (sanity check — must be 100%)
    buf = io.BytesIO()
    Image.fromarray(encoded).save(buf, format='PNG')
    buf.seek(0)
    png_reload = np.array(Image.open(buf).convert('RGB'))
    result_png = verify(png_reload, strict=False, tile_size=TILE_SIZE)
    total_base, g, y, r, a = count_blocks(result_png)
    print(f"Baseline PNG : {total_base} blocks  |  "
          f"GREEN {g} ({g/total_base*100:.1f}%)  "
          f"RED {r}  ABSENT {a}")
    print()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    col = "{:>8} | {:>6} | {:>7} | {:>7} | {:>6} | {:>7} | {:>9} | {}"
    print(col.format(
        "JPEG Q", "Blocks", "GREEN", "YELLOW", "RED", "ABSENT",
        "CRC OK %", "Block status assessment"))
    print("-" * 85)

    results = []

    for Q in QUALITY_LEVELS:
        # Compress and reload
        buf = io.BytesIO()
        Image.fromarray(encoded).save(buf, format='JPEG', quality=Q)
        buf.seek(0)
        reloaded = np.array(Image.open(buf).convert('RGB'))

        result = verify(reloaded, strict=False, tile_size=TILE_SIZE)
        total, g, y, r, a = count_blocks(result)

        # CRC OK = sync found AND CRC valid (GREEN + YELLOW)
        # Chain hash will almost always break after JPEG (expected), so
        # YELLOW is the normal outcome for CRC-surviving blocks.
        crc_ok_pct = (g + y) / total * 100 if total else 0.0

        if crc_ok_pct >= 99.0:
            assessment = "Fully intact"
        elif crc_ok_pct >= 90.0:
            assessment = "Mostly intact, chain broken"
        elif crc_ok_pct >= 50.0:
            assessment = "Partial survival"
        elif crc_ok_pct >= 10.0:
            assessment = "Mostly destroyed"
        else:
            assessment = "Unreliable"

        results.append((Q, total, g, y, r, a, crc_ok_pct, assessment))

        print(col.format(
            Q, total, g, y, r, a,
            f"{crc_ok_pct:.1f}%", assessment))

    # ------------------------------------------------------------------
    # Summary for paper table
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("LaTeX table rows (empirical -- replace theoretical Table 6):")
    print()
    for Q, total, g, y, r, a, crc_ok_pct, assessment in results:
        p_surv = crc_ok_pct / 100.0
        corrupt = (1 - p_surv) * 256  # expected corrupt bits per block
        print(f"  {Q:>3} & {p_surv:.3f} & {corrupt:.1f} & {assessment} \\\\")

    print()
    print("Notes:")
    print("  GREEN  = CRC valid + chain hash valid")
    print("  YELLOW = CRC valid, chain hash broken (expected: JPEG changed pixels)")
    print("  RED    = CRC invalid (data-field bit corruption)")
    print("  ABSENT = Sync frame destroyed (severe compression artefact)")
    print("  CRC OK% = (GREEN+YELLOW)/total  -- block payload survived compression")
    print()
    print("The chain hash always breaks after JPEG (pixel values change).")
    print("The meaningful robustness metric is CRC OK%: did the 240 non-hash")
    print("bits of the block survive compression with correct error detection?")


if __name__ == '__main__':
    sys.exit(main())
