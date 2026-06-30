#!/usr/bin/env python3
"""Regenerate paper figures from the public PBC implementation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pbc import SYNC_HAMMING_THRESHOLD, SYNC_PATTERN, PBCBlock, compute_genesis_hash  # noqa: E402
from pbc.scatter import (  # noqa: E402
    _extract_block_bytes,
    _generate_scatter_positions,
    scatter_forest_encode,
    scatter_forest_verify,
)


TIMESTAMP = 1_700_000_000


def recovered_positions(image: np.ndarray) -> list[int]:
    h, w = image.shape[:2]
    flat = image.reshape(-1, 3)
    bits_3d = (flat & 1).astype(np.uint8)
    bits_flat = bits_3d.reshape(-1)
    sync_bits = np.array(
        [(SYNC_PATTERN[i // 8] >> (7 - i % 8)) & 1 for i in range(48)],
        dtype=np.uint8,
    )
    n_pixel_pos = len(flat) - 16
    if n_pixel_pos <= 0:
        return []
    from numpy.lib.stride_tricks import as_strided

    windows = as_strided(
        bits_flat,
        shape=(n_pixel_pos, 48),
        strides=(bits_flat.strides[0] * 3, bits_flat.strides[0]),
    )
    hamming = (windows ^ sync_bits[np.newaxis, :]).sum(axis=1)
    sync_pixel_positions = np.where(hamming <= SYNC_HAMMING_THRESHOLD)[0]
    found = []
    for px in sync_pixel_positions:
        px = int(px)
        bdata = _extract_block_bytes(flat, px)
        if bdata is None:
            continue
        try:
            block = PBCBlock.from_bits(bdata)
        except Exception:
            continue
        if block.block_index != 0:
            continue
        expected = compute_genesis_hash(
            block.originator_id, block.tile_x, block.tile_y, block.timestamp_delta
        )
        if block.chain_hash == expected:
            found.append(px)
    return found


def regenerate_tamper_pair(output_dir: Path) -> None:
    subprocess.run([sys.executable, "examples/demo.py"], cwd=ROOT, check=True)
    demo_dir = ROOT / "output" / "demo"
    shutil.copy2(demo_dir / "04b_overlay_tampered.png", output_dir / "fig_tampered_overlay.png")
    shutil.copy2(demo_dir / "04e_tilemap_tampered.png", output_dir / "fig_tampered_tilemap.png")


def regenerate_forest_triptych(output_dir: Path) -> None:
    img = np.array(Image.open(ROOT / "examples" / "img" / "leo.jpg").convert("RGB"))
    h, w = img.shape[:2]
    seed = 42
    n_blocks = 1000
    enc = scatter_forest_encode(
        img, "ForestScatterTest", n_blocks=n_blocks, seed=seed, timestamp=TIMESTAMP
    )
    cw = int(w * 0.6)
    ch = int(h * 0.8)
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    cropped = enc[cy0:cy0 + ch, cx0:cx0 + cw]
    result = scatter_forest_verify(cropped)
    found = recovered_positions(cropped)

    full = Image.fromarray(img).convert("RGBA")
    over = Image.new("RGBA", full.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    for px in _generate_scatter_positions(w, h, n_blocks, seed):
        x = int(px % w)
        y = int(px // w)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(45, 156, 219, 165), outline=(255, 255, 255, 180))
    draw.rectangle((cx0, cy0, cx0 + cw - 1, cy0 + ch - 1), outline=(255, 230, 80, 255), width=6)
    Image.alpha_composite(full, over).convert("RGB").save(output_dir / "fig_forest_before_crop.png")
    Image.fromarray(cropped).save(output_dir / "fig_forest_after_crop.png")

    surv = Image.fromarray(cropped).convert("RGBA")
    over2 = Image.new("RGBA", surv.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(over2)
    for px in found:
        x = int(px % cw)
        y = int(px // cw)
        draw2.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(39, 174, 96, 210), outline=(255, 255, 255, 230))
    Image.alpha_composite(surv, over2).convert("RGB").save(output_dir / "fig_forest_survivors.png")
    print(f"Forest recovered {result.n_genesis_found}/{n_blocks} anchors ({result.survival_pct(n_blocks):.1f}%).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regenerate_tamper_pair(args.output_dir)
    regenerate_forest_triptych(args.output_dir)
    print(f"Wrote paper figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

