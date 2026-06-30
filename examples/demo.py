#!/usr/bin/env python3
"""
PBC Full Pipeline Demo

Demonstrates:
1. Encoding PBC into a real photograph (leo.jpg, k=1 LSB/channel)
2. Verifying an untouched PBC image  -> all tiles GREEN
3. Simulating tampering on tile (1,1) -> that tile ABSENT/RED, others intact
4. Simulating PBC-aware editing       -> affected tiles re-encoded (GREEN)
5. Verifying an image with no PBC     -> all tiles ABSENT
6. k=1 vs synthetic gradient: imperceptibility comparison

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import time
import math
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, generate_originator_id, compute_grid, BITS_PER_CHANNEL
from pbc.encoder import encode, encode_region
from pbc.decoder import verify, TileStatus
from pbc.visualizer import (generate_overlay, generate_heatmap,
                             generate_block_grid, generate_report_image,
                             render_tile_map)


TILE_SIZE    = 128   # Must match across encode / verify / tamper
IMG_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG      = os.path.join(IMG_DIR, 'leo.jpg')
DIFF_AMPLIFY = 60    # Amplification factor for difference visualisation


def create_test_image(width=600, height=400):
    """Create a colorful test image with gradients and patterns."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            img[y, x, 0] = int(255 * x / width)
            img[y, x, 1] = int(255 * y / height)
            img[y, x, 2] = int(255 * (1 - x / width))

    img[ 50:100,  50:150] = [200,  60,  60]   # Red box
    img[120:180, 200:350] = [ 60,  60, 200]   # Blue box
    img[200:260,  80:250] = [ 60, 180,  60]   # Green box

    return img


def psnr(original: np.ndarray, encoded: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - encoded.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255.0 ** 2 / mse)


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'output', 'demo')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  PIXEL BLOCK CHAIN (PBC) - Full Pipeline Demo")
    print(f"  Embedding: k={BITS_PER_CHANNEL} bit(s) per channel")
    print("=" * 60)
    print()

    # =========================================================================
    # 1. Load real photograph and encode
    # =========================================================================
    print(f"[1/6] Loading real photograph: {os.path.basename(LEO_JPG)}")
    original = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = original.shape[:2]
    print(f"      Image size: {W}x{H}")
    Image.fromarray(original).save(os.path.join(output_dir, '01_original.png'))
    print(f"      Saved: 01_original.png")

    cols, rows, tile_w, tile_h = compute_grid(W, H, TILE_SIZE)
    print(f"      Grid: {cols}x{rows} tiles, tile size ~{tile_w}x{tile_h} px")

    print("[2/6] Encoding PBC stream (grid architecture)...")
    t0 = time.time()
    encoded = encode(original, originator="DemoCamera-SN12345",
                     opcode=OpCode.CAMERA_ISP, tile_size=TILE_SIZE)
    t1 = time.time()

    p = psnr(original, encoded)
    max_diff = int(np.max(np.abs(original.astype(int) - encoded.astype(int))))
    print(f"      Encoded in {t1-t0:.3f}s")
    print(f"      PSNR: {p:.1f} dB   Max channel error: {max_diff}")

    Image.fromarray(encoded).save(os.path.join(output_dir, '02_encoded.png'))
    diff = np.clip(np.abs(original.astype(int) - encoded.astype(int)) * DIFF_AMPLIFY,
                   0, 255).astype(np.uint8)
    Image.fromarray(diff).save(os.path.join(output_dir, '02b_diff_amplified.png'))
    print(f"      Saved: 02_encoded.png, 02b_diff_amplified.png")
    print()

    # =========================================================================
    # 2. Verify untouched encoded image
    # =========================================================================
    print("[3/6] Verifying untouched PBC image...")
    t0 = time.time()
    result_clean = verify(encoded, strict=True, tile_size=TILE_SIZE)
    t1 = time.time()
    print(f"      Verified in {t1-t0:.3f}s")
    print(f"      GREEN tiles: {result_clean.green_count}/{len(result_clean.all_tiles)}")
    print(f"      Integrity: {result_clean.integrity_score:.1f}%")

    generate_report_image(encoded, result_clean).save(
        os.path.join(output_dir, '03_verify_clean.png'))
    Image.fromarray(render_tile_map(result_clean, cell_size=40)).save(
        os.path.join(output_dir, '03_tilemap_clean.png'))
    print(f"      Saved: 03_verify_clean.png, 03_tilemap_clean.png")
    print()

    # =========================================================================
    # 3. Simulate tampering -- target tile (1, 1)
    # =========================================================================
    print("[4/6] Simulating tampering on tile (1,1)...")
    tampered = encoded.copy()

    def tile_bounds(ttx, tty):
        """Return (x0, x1, y0, y1) matching encoder/decoder tile boundaries."""
        tx0 = ttx * tile_w;  tx1 = W if ttx == cols - 1 else tx0 + tile_w
        ty0 = tty * tile_h;  ty1 = H if tty == rows - 1 else ty0 + tile_h
        return tx0, tx1, ty0, ty1

    x0_t, x1_t, y0_t, y1_t = tile_bounds(1, 1)
    # Overwrite with uniform grey (destroys LSB patterns -> ABSENT)
    tampered[y0_t:y1_t, x0_t:x1_t] = 128

    t0 = time.time()
    result_tampered = verify(tampered, strict=True, tile_size=TILE_SIZE)
    t1 = time.time()

    print(f"      Verified in {t1-t0:.3f}s")
    print(f"      GREEN:  {result_tampered.green_count}")
    print(f"      YELLOW: {result_tampered.yellow_count}")
    print(f"      RED:    {result_tampered.red_count}")
    print(f"      ABSENT: {result_tampered.absent_count}")
    print(f"      Integrity: {result_tampered.integrity_score:.1f}%")

    generate_report_image(tampered, result_tampered).save(
        os.path.join(output_dir, '04_verify_tampered.png'))
    generate_overlay(tampered, result_tampered).save(
        os.path.join(output_dir, '04b_overlay_tampered.png'))
    generate_heatmap(result_tampered).save(
        os.path.join(output_dir, '04c_heatmap_tampered.png'))
    generate_block_grid(result_tampered).save(
        os.path.join(output_dir, '04d_grid_tampered.png'))
    Image.fromarray(render_tile_map(result_tampered, cell_size=40)).save(
        os.path.join(output_dir, '04e_tilemap_tampered.png'))
    print(f"      Saved: 04_verify_tampered.png, 04b-04e variants")
    print()

    # =========================================================================
    # 3b. Scattered tampering -- overwrite 4 tiles spread across the image
    # =========================================================================
    print("[4b]  Simulating scattered tampering on 4 tiles...")
    scattered = encoded.copy()

    # Pick 4 tiles in different quadrants
    tamper_tiles = [(0, 0), (cols - 1, 0), (0, rows - 1), (cols // 2, rows // 2)]
    for ttx, tty in tamper_tiles:
        tx0, tx1, ty0, ty1 = tile_bounds(ttx, tty)
        scattered[ty0:ty1, tx0:tx1] = 128   # uniform grey -> ABSENT

    result_scattered = verify(scattered, strict=True, tile_size=TILE_SIZE)
    print(f"      GREEN:  {result_scattered.green_count}")
    print(f"      ABSENT: {result_scattered.absent_count}")
    print(f"      Integrity: {result_scattered.integrity_score:.1f}%")

    generate_overlay(scattered, result_scattered).save(
        os.path.join(output_dir, '04f_overlay_scattered.png'))
    Image.fromarray(render_tile_map(result_scattered, cell_size=40)).save(
        os.path.join(output_dir, '04g_tilemap_scattered.png'))
    print(f"      Saved: 04f_overlay_scattered.png, 04g_tilemap_scattered.png")
    print()

    # =========================================================================
    # 4. Simulate PBC-aware re-encoding (colour correction on a region)
    # =========================================================================
    print("[5/6] Simulating PBC-aware edit (colour correction)...")
    edited = encoded.copy()
    # Brighten a region slightly
    ey0, ey1, ex0, ex1 = 50, 200, 50, 300
    region = edited[ey0:ey1, ex0:ex1].astype(int)
    region[:, :, 0] = np.clip(region[:, :, 0] + 15, 0, 255)
    region[:, :, 2] = np.clip(region[:, :, 2] - 10, 0, 255)
    edited[ey0:ey1, ex0:ex1] = region.astype(np.uint8)

    mask = np.zeros((H, W), dtype=bool)
    mask[ey0:ey1, ex0:ex1] = True
    edited_reencoded = encode_region(edited, mask,
                                     originator="PhotoEditor-v2.0",
                                     opcode=OpCode.EDIT_COLOR,
                                     tile_size=TILE_SIZE)

    t0 = time.time()
    result_edited = verify(edited_reencoded, strict=True, tile_size=TILE_SIZE)
    t1 = time.time()

    print(f"      Verified in {t1-t0:.3f}s")
    print(f"      GREEN:  {result_edited.green_count}")
    print(f"      YELLOW: {result_edited.yellow_count}")
    print(f"      RED:    {result_edited.red_count}")
    print(f"      ABSENT: {result_edited.absent_count}")
    print(f"      Integrity: {result_edited.integrity_score:.1f}%")

    generate_report_image(edited_reencoded, result_edited).save(
        os.path.join(output_dir, '05_verify_pbc_edit.png'))
    Image.fromarray(render_tile_map(result_edited, cell_size=40)).save(
        os.path.join(output_dir, '05_tilemap_pbc_edit.png'))
    print(f"      Saved: 05_verify_pbc_edit.png, 05_tilemap_pbc_edit.png")
    print()

    # =========================================================================
    # 5. Verify non-PBC image (all ABSENT)
    # =========================================================================
    print("[6/6] Verifying non-PBC image (original without encoding)...")
    result_none = verify(original, strict=True, tile_size=TILE_SIZE)
    print(f"      GREEN:  {result_none.green_count}")
    print(f"      ABSENT: {result_none.absent_count}")
    print(f"      Integrity: {result_none.integrity_score:.1f}%")

    generate_report_image(original, result_none).save(
        os.path.join(output_dir, '06_verify_no_pbc.png'))
    print(f"      Saved: 06_verify_no_pbc.png")
    print()

    # =========================================================================
    # 6. Imperceptibility comparison: synthetic gradient at k=1
    #    (same gradient as before, showing k=1 is cleaner)
    # =========================================================================
    print("[+] Synthetic gradient comparison (k=1 imperceptibility check)...")
    synth = create_test_image(600, 400)
    synth_enc = encode(synth, originator="DemoCamera-SN12345",
                       opcode=OpCode.CAMERA_ISP, tile_size=TILE_SIZE)
    p_synth  = psnr(synth, synth_enc)
    md_synth = int(np.max(np.abs(synth.astype(int) - synth_enc.astype(int))))
    print(f"      Synthetic PSNR: {p_synth:.1f} dB   Max channel error: {md_synth}")

    Image.fromarray(synth).save(os.path.join(output_dir, '07_synth_original.png'))
    Image.fromarray(synth_enc).save(os.path.join(output_dir, '07_synth_encoded.png'))
    diff_s = np.clip(np.abs(synth.astype(int) - synth_enc.astype(int)) * DIFF_AMPLIFY,
                     0, 255).astype(np.uint8)
    Image.fromarray(diff_s).save(os.path.join(output_dir, '07_synth_diff.png'))
    print(f"      Saved: 07_synth_original.png, 07_synth_encoded.png, 07_synth_diff.png")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("  DEMO COMPLETE -- output files in ./output/")
    print("=" * 60)
    print()
    print("  --- Real photo (leo.jpg) ---")
    print("  01_original.png            - Photograph before PBC encoding")
    print("  02_encoded.png             - PBC-encoded (visually identical)")
    print("  02b_diff_amplified.png     - Pixel changes amplified 60x")
    print("  03_verify_clean.png        - Verification: all tiles GREEN")
    print("  03_tilemap_clean.png       - Tile map: clean image")
    print("  04_verify_tampered.png     - Verification: tile (1,1) ABSENT")
    print("  04b_overlay_tampered.png   - Overlay view of tampering")
    print("  04c_heatmap_tampered.png   - Heatmap view")
    print("  04d_grid_tampered.png      - Block grid view")
    print("  04e_tilemap_tampered.png   - Tile map: tampered")
    print("  05_verify_pbc_edit.png     - Verification: PBC-aware edit")
    print("  05_tilemap_pbc_edit.png    - Tile map: edited")
    print("  06_verify_no_pbc.png       - Verification: no PBC (ABSENT)")
    print()
    print("  --- Synthetic gradient (imperceptibility comparison) ---")
    print("  07_synth_original.png      - Gradient before encoding")
    print("  07_synth_encoded.png       - Gradient after k=1 encoding")
    print("  07_synth_diff.png          - Difference amplified 60x")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
