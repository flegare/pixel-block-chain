#!/usr/bin/env python3
"""
AI Inpainting PBC Survival Test
================================
Tests how many PBC tiles survive when an image is edited by a tool that:

  Scenario A  -- Bit-perfect compositing  (what SD-inpainting / FLUX do):
                 new = original * (1-mask) + ai_generated * mask
                 The unmasked pixels are a pixel-exact copy.

  Scenario B  -- Soft-blend / feathered compositing  (some pipelines feather
                 the mask boundary):
                 a gaussian-blurred mask is used instead of hard 0/1.

  Scenario C  -- Full VAE round-trip  (img2img or pipelines without compositing):
                 the entire image goes through resize->encode->decode, introducing
                 small reconstruction errors everywhere (~+/-1–2 LSBs).

  Scenario D  -- JPEG re-save  (default output in many tools):
                 quality=85 JPEG -> every pixel's LSBs modified.

No GPU / diffusers required.  The "AI-generated" masked region is simulated by
a strong HSV color shift (eye color change) applied only inside the mask.

Usage:
    python examples/ai_inpainting_test.py
"""

import os
import sys
import time
import io
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbc.encoder import encode
from pbc.decoder import verify, TileStatus
from pbc import generate_originator_id, compute_grid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_PATH  = os.path.join(os.path.dirname(__file__), "img", "leo.jpg")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output", "ai-inpainting")
IDENTITY    = "Canon-EOS-R5-SN012345"

# Eye-region mask: approximate box as a fraction of the image
# (upper-centre area, roughly where eyes sit in a portrait)
EYE_MASK_FRACTIONS = dict(
    x0=0.35, y0=0.28,   # top-left corner (fraction of W, H)
    x1=0.65, y1=0.42,   # bottom-right corner
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_mask(W: int, H: int) -> np.ndarray:
    """Binary mask (uint8, 0/255) -- 255 = eye region to be 'edited'."""
    mask = np.zeros((H, W), dtype=np.uint8)
    x0 = int(EYE_MASK_FRACTIONS["x0"] * W)
    y0 = int(EYE_MASK_FRACTIONS["y0"] * H)
    x1 = int(EYE_MASK_FRACTIONS["x1"] * W)
    y1 = int(EYE_MASK_FRACTIONS["y1"] * H)
    mask[y0:y1, x0:x1] = 255
    return mask


def simulate_eye_color_change(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Strong color change inside the mask (simulates AI-generated eye pixels).
    Shifts the red channel strongly and inverts blue/green in the masked region.
    """
    result = arr.copy().astype(np.int32)
    m = mask > 127
    result[m, 0] = np.clip(result[m, 0].astype(np.int32) * 2 - 40,   0, 255)  # red up
    result[m, 1] = np.clip(result[m, 1].astype(np.int32) * 0.3 + 20,  0, 255)  # green down
    result[m, 2] = np.clip(result[m, 2].astype(np.int32) * 2 + 60,   0, 255)  # blue up
    return result.astype(np.uint8)


def vae_roundtrip(arr: np.ndarray, downsample: int = 8) -> np.ndarray:
    """
    Simulate VAE encode -> decode at 8x spatial compression.
    Equivalent to: resize down to 1/8, then back up with bilinear.
    Introduces +/-1-3 LSB reconstruction noise uniformly.
    """
    H, W = arr.shape[:2]
    img  = Image.fromarray(arr)
    small = img.resize((W // downsample, H // downsample), Image.BILINEAR)
    reconstructed = small.resize((W, H), Image.BILINEAR)
    return np.array(reconstructed)


def jpeg_roundtrip(arr: np.ndarray, quality: int = 85) -> np.ndarray:
    """Simulate JPEG save + reload at given quality."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def count_tiles(result, total_tiles) -> dict:
    counts = {s: 0 for s in TileStatus}
    for tile in result.all_tiles:
        counts[tile.status] += 1
    return counts


def tiles_in_mask(arr: np.ndarray, mask: np.ndarray) -> set:
    """Return set of (tx,ty) tile coordinates that overlap the eye mask."""
    H, W = arr.shape[:2]
    cols, rows, tw, th = compute_grid(W, H)
    affected = set()
    for ty in range(rows):
        for tx in range(cols):
            x0, y0 = tx * tw, ty * th
            x1 = x0 + tw if tx < cols - 1 else W
            y1 = y0 + th if ty < rows - 1 else H
            region_mask = mask[y0:y1, x0:x1]
            if region_mask.max() > 0:
                affected.add((tx, ty))
    return affected


def print_result(label: str, result, encoded_arr: np.ndarray, mask: np.ndarray,
                 elapsed_ms: float):
    total = len(result.all_tiles)
    counts = count_tiles(result, total)
    g = counts[TileStatus.GREEN]
    y = counts[TileStatus.YELLOW]
    r = counts[TileStatus.RED]
    a = counts[TileStatus.ABSENT]
    score = result.integrity_score
    masked_tiles = tiles_in_mask(encoded_arr, mask)

    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Tiles total:   {total}  ({int(encoded_arr.shape[1])}x{int(encoded_arr.shape[0])} image)")
    print(f"  GREEN:         {g:3d} / {total}  ({100*g/total:.1f}%)")
    print(f"  YELLOW:        {y:3d} / {total}  ({100*y/total:.1f}%)")
    print(f"  RED:           {r:3d} / {total}  ({100*r/total:.1f}%)")
    print(f"  ABSENT:        {a:3d} / {total}  ({100*a/total:.1f}%)")
    print(f"  Integrity:     {score:.1f}%")
    print(f"  Verify time:   {elapsed_ms:.1f} ms")
    print(f"  Mask covers:   {len(masked_tiles)} tile(s) -> {sorted(masked_tiles)}")

    # Show per-tile status for masked tiles
    masked_results = {(t.tx, t.ty): t.status for t in result.all_tiles
                      if (t.tx, t.ty) in masked_tiles}
    for coord, status in sorted(masked_results.items()):
        print(f"    tile {coord}: {status.name}")

    outside_green = sum(1 for t in result.all_tiles
                        if (t.tx, t.ty) not in masked_tiles
                        and t.status == TileStatus.GREEN)
    outside_total = total - len(masked_tiles)
    if outside_total > 0:
        print(f"  Outside-mask GREEN: {outside_green}/{outside_total} "
              f"({100*outside_green/outside_total:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PBC Survival Under AI Inpainting (Simulation)")
    print("=" * 60)

    # -- Load image ----------------------------------------------
    if not os.path.exists(IMAGE_PATH):
        print(f"\nImage not found: {IMAGE_PATH}")
        print("Using a synthetic 978x678 gradient instead.")
        arr_orig = np.zeros((678, 978, 3), dtype=np.uint8)
        for y in range(678):
            arr_orig[y, :, 0] = int(y / 678 * 200) + 30
            arr_orig[y, :, 1] = 120
            arr_orig[y, :, 2] = int((978 - y) / 978 * 180) + 40
    else:
        arr_orig = np.array(Image.open(IMAGE_PATH).convert("RGB"))
        print(f"\nImage: {IMAGE_PATH}  ({arr_orig.shape[1]}x{arr_orig.shape[0]})")

    H, W = arr_orig.shape[:2]
    oid  = generate_originator_id(IDENTITY)
    cols, rows, tw, th = compute_grid(W, H)
    total_tiles = cols * rows
    print(f"Grid:  {cols}x{rows} = {total_tiles} tiles  ({tw}x{th} px each)")
    print(f"OID:   0x{oid:08X}  ({IDENTITY})")

    # -- PBC-encode -----------------------------------------------
    print("\nEncoding with PBC…", end=" ", flush=True)
    t0 = time.perf_counter()
    encoded = encode(arr_orig, IDENTITY, opcode=0x0001, timestamp=12345)
    enc_ms  = (time.perf_counter() - t0) * 1000
    print(f"{enc_ms:.0f} ms")

    # Save encoded reference
    Image.fromarray(encoded.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, "ai_test_encoded.png"))

    # -- Build mask -----------------------------------------------
    mask = build_mask(W, H)
    eye_pixels = int(mask.sum() / 255)
    box_w = int((EYE_MASK_FRACTIONS['x1'] - EYE_MASK_FRACTIONS['x0']) * W)
    box_h = int((EYE_MASK_FRACTIONS['y1'] - EYE_MASK_FRACTIONS['y0']) * H)
    print(f"Eye mask: {eye_pixels:,} pixels  "
          f"({100*eye_pixels/(W*H):.1f}% of image)  "
          f"box ~{box_w}x{box_h} px")

    # -- Generate 'AI' eye pixels ---------------------------------
    ai_pixels = simulate_eye_color_change(encoded, mask)

    # ------------------------------------------------------------
    # SCENARIO A: Bit-perfect compositing
    #   unmasked pixels = exact copy from encoded
    #   Prediction: ~(total - mask_tiles) / total GREEN
    # ------------------------------------------------------------
    mask_f = (mask[:, :, np.newaxis] / 255.0)
    scenario_a = (encoded * (1 - mask_f) + ai_pixels * mask_f).astype(np.uint8)
    Image.fromarray(scenario_a).save(
        os.path.join(OUTPUT_DIR, "ai_test_scenario_a.png"))

    t0 = time.perf_counter()
    result_a = verify(scenario_a)
    ms_a = (time.perf_counter() - t0) * 1000
    print_result(
        "Scenario A -- Bit-perfect compositing (SD-inpaint / FLUX style)",
        result_a, encoded, mask, ms_a)

    # ------------------------------------------------------------
    # SCENARIO B: Feathered mask (Gaussian blur on mask boundary)
    #   Prediction: boundary tiles YELLOW or RED, others GREEN
    # ------------------------------------------------------------
    mask_img = Image.fromarray(mask)
    mask_blurred = np.array(mask_img.filter(ImageFilter.GaussianBlur(radius=12))) / 255.0
    mask_b = mask_blurred[:, :, np.newaxis]
    scenario_b = (encoded * (1 - mask_b) + ai_pixels * mask_b).astype(np.uint8)
    Image.fromarray(scenario_b).save(
        os.path.join(OUTPUT_DIR, "ai_test_scenario_b.png"))

    t0 = time.perf_counter()
    result_b = verify(scenario_b)
    ms_b = (time.perf_counter() - t0) * 1000
    print_result(
        "Scenario B -- Feathered mask (12px Gaussian blur on boundary)",
        result_b, encoded, mask, ms_b)

    # ------------------------------------------------------------
    # SCENARIO C: Full VAE round-trip (img2img, no compositing)
    #   Prediction: ~0% GREEN (LSB noise everywhere)
    # ------------------------------------------------------------
    scenario_c = vae_roundtrip(scenario_a, downsample=8)
    Image.fromarray(scenario_c).save(
        os.path.join(OUTPUT_DIR, "ai_test_scenario_c.png"))

    t0 = time.perf_counter()
    result_c = verify(scenario_c)
    ms_c = (time.perf_counter() - t0) * 1000
    print_result(
        "Scenario C -- Full VAE round-trip (img2img / no compositing)",
        result_c, encoded, mask, ms_c)

    # ------------------------------------------------------------
    # SCENARIO D: JPEG re-save (default output in many tools)
    #   Prediction: 0% GREEN
    # ------------------------------------------------------------
    scenario_d = jpeg_roundtrip(scenario_a, quality=85)

    t0 = time.perf_counter()
    result_d = verify(scenario_d)
    ms_d = (time.perf_counter() - t0) * 1000
    print_result(
        "Scenario D -- JPEG Q=85 re-save (most AI tool defaults)",
        result_d, encoded, mask, ms_d)

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<45}  {'GREEN':>6}  {'Score':>6}")
    print(f"  {'-'*57}")
    for label, result in [
        ("A: Bit-perfect compositing (SD-inpaint, PNG out)", result_a),
        ("B: Feathered mask boundary (PNG out)",             result_b),
        ("C: Full VAE round-trip / img2img (PNG out)",       result_c),
        ("D: JPEG Q=85 output (default of most tools)",      result_d),
    ]:
        g     = sum(1 for t in result.all_tiles if t.status == TileStatus.GREEN)
        total = len(result.all_tiles)
        print(f"  {label:<45}  {g:>3}/{total}  {result.integrity_score:>5.1f}%")

    print(f"\n  Total tiles:  {total_tiles}  |  Mask tiles: {len(tiles_in_mask(encoded, mask))}")

    masked_tiles_count = len(tiles_in_mask(encoded, mask))
    max_possible = total_tiles - masked_tiles_count
    print(f"  Max possible GREEN (outside mask): {max_possible}/{total_tiles} "
          f"= {100*max_possible/total_tiles:.1f}%")
    print()

    # -- Save mask visualisation ----------------------------------
    vis = Image.fromarray(encoded.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    x0i = int(EYE_MASK_FRACTIONS["x0"] * W)
    y0i = int(EYE_MASK_FRACTIONS["y0"] * H)
    x1i = int(EYE_MASK_FRACTIONS["x1"] * W)
    y1i = int(EYE_MASK_FRACTIONS["y1"] * H)
    d.rectangle([x0i, y0i, x1i, y1i], outline=(255, 80, 80, 220), width=3)
    d.rectangle([x0i, y0i, x1i, y1i], fill=(255, 80, 80, 40))
    vis = Image.alpha_composite(vis, overlay).convert("RGB")
    vis.save(os.path.join(OUTPUT_DIR, "ai_test_mask_vis.png"))
    print(f"  Output images saved to {OUTPUT_DIR}/ai_test_*.png")


if __name__ == "__main__":
    main()
