#!/usr/bin/env python3
"""
PBC Edit Ledger Demo
====================

Demonstrates the Edit Ledger feature introduced in PBC v04:
a self-contained, per-tile record of every authoring event from
RAW capture to final publication, embedded in the pixel data.

Workflow simulated:
  Step 1 -- Capture:         NikonZ9-SN2024 photographs the scene (Camera_ISP)
  Step 2 -- Color grade:     Alice (Lightroom) records Edit_Color on sky region
                             [chain split at 33%]
  Step 3 -- Subject retouch: Bob (Photoshop) records Edit_Retouch on portrait
                             [chain split at 66%]

Each tile's chain therefore contains up to three ledger entries depending on
whether it was touched by Alice, Bob, both, or neither.

NOTE on pixel changes and PBC:
  In a real PBC-aware editor, pixel modifications and chain appending happen
  atomically: the editor applies its changes then immediately re-embeds its
  PBC blocks into those modified pixels.  For this demo we separate the
  concerns -- pixel changes are shown as separate display images while the
  PBC chain is built in append mode on a clean copy -- which correctly
  demonstrates the ledger structure without pixel-corruption artefacts.

Output images:
  ledger_01_original.png         Source photograph (no PBC)
  ledger_02_captured.png         After camera PBC encoding (Camera_ISP)
  ledger_03_alice_region.png     Alice's edit region highlighted (yellow)
  ledger_04_bob_region.png       Bob's edit region highlighted (blue)
  ledger_05_after_alice.png      After Alice's append (Edit_Color blocks)
  ledger_06_after_bob.png        After Bob's append (Edit_Retouch blocks)
  ledger_07_overlay_final.png    Tile integrity overlay of final image
  ledger_08_tilemap_final.png    Tile status grid
  ledger_ledger_report.txt       Full Edit Ledger for sample tiles

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, generate_originator_id, compute_grid, BITS_PER_CHANNEL
from pbc.encoder import encode, append_edit
from pbc.decoder import verify, TileStatus, extract_edit_ledger
from pbc.visualizer import generate_overlay, render_tile_map

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TILE_SIZE = 128

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
LEO_JPG = os.path.join(IMG_DIR, 'leo.jpg')
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'edit-ledger')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'results')

# Simulated author identities
CAMERA_ID = "NikonZ9-SN2024"
ALICE_ID  = "Lightroom-Alice"
BOB_ID    = "Photoshop-Bob"

# Simulated timestamps: capture at T, edits 1h and 2h later
T_CAPTURE = 1_700_000_000
T_ALICE   = T_CAPTURE + 3600
T_BOB     = T_CAPTURE + 7200

# Chain split fractions:
#   Camera holds  blocks  0 .. 33% - 1
#   Alice  holds  blocks 33% .. 66% - 1
#   Bob    holds  blocks 66% .. end
SPLIT_ALICE = 1.0 / 3.0
SPLIT_BOB   = 2.0 / 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def highlight_region(image: np.ndarray, mask: np.ndarray,
                     color: tuple, alpha: float = 0.40) -> np.ndarray:
    """Blend a solid color over a region defined by a boolean mask."""
    out = image.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32)
    for ch in range(3):
        out[..., ch][mask] = out[..., ch][mask] * (1 - alpha) + c[ch] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def format_oid(oid: int) -> str:
    """Map originator ID back to a human-readable name when known."""
    known = {
        generate_originator_id(CAMERA_ID): CAMERA_ID,
        generate_originator_id(ALICE_ID):  ALICE_ID,
        generate_originator_id(BOB_ID):    BOB_ID,
    }
    return known.get(oid, f"0x{oid:08X}")


def sep(title=""):
    bar = "=" * 70
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sep("PBC Edit Ledger Demo")
    print(f"  Embedding depth : k={BITS_PER_CHANNEL} bit(s) per channel")
    print(f"  Tile size       : {TILE_SIZE}px")
    print(f"  Camera          : {CAMERA_ID}")
    print(f"  Editor 1 (Alice): {ALICE_ID}  -- color grade (sky + background)")
    print(f"  Editor 2 (Bob)  : {BOB_ID}  -- subject retouch (portrait centre)")

    # -------------------------------------------------------------------------
    # Step 0: Load source photograph
    # -------------------------------------------------------------------------
    sep("Step 0 -- Load source photograph")
    original = np.array(Image.open(LEO_JPG).convert('RGB'))
    H, W = original.shape[:2]
    print(f"  Source      : {os.path.basename(LEO_JPG)}  ({W}x{H} px)")

    cols, rows, tile_w, tile_h = compute_grid(W, H, TILE_SIZE)
    total_tiles    = cols * rows
    blocks_per_tile = (tile_w * tile_h) // 86   # approximate
    print(f"  Grid        : {cols}x{rows} = {total_tiles} tiles, "
          f"~{blocks_per_tile} blocks/tile")

    alice_portion = int(blocks_per_tile * SPLIT_ALICE)
    bob_portion   = int(blocks_per_tile * SPLIT_BOB) - alice_portion
    cam_portion   = alice_portion

    print(f"  Chain layout plan:")
    print(f"    Blocks   0..{cam_portion-1:>3}  ({cam_portion:>3} blk)  "
          f"-> Camera_ISP  / {CAMERA_ID}")
    print(f"    Blocks {cam_portion:>3}..{cam_portion+alice_portion-1:>3}  ({alice_portion:>3} blk)  "
          f"-> Edit_Color  / {ALICE_ID}  (in Alice's region)")
    print(f"    Blocks {cam_portion+alice_portion:>3}..{blocks_per_tile-1:>3}  "
          f"({blocks_per_tile - cam_portion - alice_portion:>3} blk)  "
          f"-> Edit_Retouch/ {BOB_ID}  (in overlap region)")

    Image.fromarray(original).save(os.path.join(OUT_DIR, 'ledger_01_original.png'))

    # -------------------------------------------------------------------------
    # Step 1: Camera capture -- encode entire image with Camera_ISP
    # -------------------------------------------------------------------------
    sep("Step 1 -- Camera capture")
    captured = encode(
        original,
        originator=CAMERA_ID,
        opcode=OpCode.CAMERA_ISP,
        timestamp=T_CAPTURE,
        tile_size=TILE_SIZE,
    )
    Image.fromarray(captured).save(os.path.join(OUT_DIR, 'ledger_02_captured.png'))
    print(f"  All {total_tiles} tiles encoded: opcode=CAMERA_ISP, ts=T+0s")
    print(f"  All {blocks_per_tile} blocks per tile carry CAMERA_ISP / {CAMERA_ID}")

    # -------------------------------------------------------------------------
    # Step 2: Alice -- color grade on sky / background (top 55% of image)
    #
    # In append mode: Alice writes Edit_Color blocks at positions [split..end]
    # in each tile that overlaps her region. The first 33% of each tile chain
    # (Camera's blocks) is left untouched.
    # -------------------------------------------------------------------------
    sep("Step 2 -- Color grade by Alice (Lightroom)")

    alice_y    = int(H * 0.55)
    alice_mask = np.zeros((H, W), dtype=bool)
    alice_mask[:alice_y, :] = True

    alice_tile_count = sum(
        1 for ty in range(rows) for tx in range(cols)
        if np.any(alice_mask[
            ty * tile_h : (ty + 1) * tile_h if ty < rows - 1 else H,
            tx * tile_w : (tx + 1) * tile_w if tx < cols - 1 else W
        ])
    )

    after_alice = append_edit(
        captured,
        originator=ALICE_ID,
        opcode=OpCode.EDIT_COLOR,
        timestamp=T_ALICE,
        tile_size=TILE_SIZE,
        region_mask=alice_mask,
        split_fraction=SPLIT_ALICE,
    )

    Image.fromarray(after_alice).save(os.path.join(OUT_DIR, 'ledger_05_after_alice.png'))

    # Save region highlight (yellow = warm light tones)
    alice_hl = highlight_region(original, alice_mask, color=(255, 230, 60))
    Image.fromarray(alice_hl).save(os.path.join(OUT_DIR, 'ledger_03_alice_region.png'))

    split_blk_alice = max(1, int(blocks_per_tile * SPLIT_ALICE))
    print(f"  Alice's region: top {alice_y}px ({alice_tile_count} tiles affected)")
    print(f"  Opcode        : EDIT_COLOR  |  ts=T+3600s")
    print(f"  Split at block: {split_blk_alice} ({int(SPLIT_ALICE*100)}% of chain)")
    print(f"  Chain in Alice's tiles:")
    print(f"    [0 .. {split_blk_alice-1}]    CAMERA_ISP  / {CAMERA_ID}  (untouched)")
    print(f"    [{split_blk_alice} .. {blocks_per_tile-1}]  EDIT_COLOR  / {ALICE_ID}  (appended)")

    # -------------------------------------------------------------------------
    # Step 3: Bob -- subject retouch on portrait centre
    # -------------------------------------------------------------------------
    sep("Step 3 -- Subject retouch by Bob (Photoshop)")

    bob_y0 = int(H * 0.30);  bob_y1 = int(H * 0.75)
    bob_x0 = int(W * 0.25);  bob_x1 = int(W * 0.75)
    bob_mask = np.zeros((H, W), dtype=bool)
    bob_mask[bob_y0:bob_y1, bob_x0:bob_x1] = True

    bob_tile_count = sum(
        1 for ty in range(rows) for tx in range(cols)
        if np.any(bob_mask[
            ty * tile_h : (ty + 1) * tile_h if ty < rows - 1 else H,
            tx * tile_w : (tx + 1) * tile_w if tx < cols - 1 else W
        ])
    )
    overlap_count = sum(
        1 for ty in range(rows) for tx in range(cols)
        if np.any(alice_mask[
            ty * tile_h : (ty + 1) * tile_h if ty < rows - 1 else H,
            tx * tile_w : (tx + 1) * tile_w if tx < cols - 1 else W
        ]) and np.any(bob_mask[
            ty * tile_h : (ty + 1) * tile_h if ty < rows - 1 else H,
            tx * tile_w : (tx + 1) * tile_w if tx < cols - 1 else W
        ])
    )

    final_image = append_edit(
        after_alice,
        originator=BOB_ID,
        opcode=OpCode.EDIT_RETOUCH,
        timestamp=T_BOB,
        tile_size=TILE_SIZE,
        region_mask=bob_mask,
        split_fraction=SPLIT_BOB,
    )

    Image.fromarray(final_image).save(os.path.join(OUT_DIR, 'ledger_06_after_bob.png'))

    # Save region highlight (blue = cool retouching)
    bob_hl = highlight_region(original, bob_mask, color=(80, 160, 255))
    Image.fromarray(bob_hl).save(os.path.join(OUT_DIR, 'ledger_04_bob_region.png'))

    split_blk_bob = max(1, int(blocks_per_tile * SPLIT_BOB))
    print(f"  Bob's region  : rows {bob_y0}-{bob_y1}px, cols {bob_x0}-{bob_x1}px "
          f"({bob_tile_count} tiles)")
    print(f"  Overlap w/ Alice: {overlap_count} tiles -- these carry ALL THREE authors")
    print(f"  Opcode        : EDIT_RETOUCH  |  ts=T+7200s")
    print(f"  Split at block: {split_blk_bob} ({int(SPLIT_BOB*100)}% of chain)")
    print(f"  Chain in overlap tiles:")
    print(f"    [0 .. {split_blk_alice-1}]    CAMERA_ISP  / {CAMERA_ID}")
    print(f"    [{split_blk_alice} .. {split_blk_bob-1}]  EDIT_COLOR  / {ALICE_ID}")
    print(f"    [{split_blk_bob} .. {blocks_per_tile-1}]  EDIT_RETOUCH/ {BOB_ID}")

    # -------------------------------------------------------------------------
    # Step 4: Verify final image
    # -------------------------------------------------------------------------
    sep("Step 4 -- Verify final image")

    result = verify(final_image, strict=False, tile_size=TILE_SIZE)
    print(f"  GREEN  (intact)    : {result.green_count:>3} / {total_tiles}  "
          f"({result.green_count/total_tiles*100:.0f}%)")
    print(f"  YELLOW (re-encoded): {result.yellow_count:>3} / {total_tiles}")
    print(f"  RED    (tampered)  : {result.red_count:>3} / {total_tiles}")
    print(f"  ABSENT (no PBC)    : {result.absent_count:>3} / {total_tiles}")
    print(f"  Integrity score    : {result.integrity_score:.1f}%")

    generate_overlay(final_image, result).save(
        os.path.join(OUT_DIR, 'ledger_07_overlay_final.png'))
    Image.fromarray(render_tile_map(result, cell_size=40)).save(
        os.path.join(OUT_DIR, 'ledger_08_tilemap_final.png'))

    if result.red_count > 0 or result.yellow_count > 0:
        print()
        print("  WARNING: unexpected non-GREEN tiles -- see ledger report for details.")

    # -------------------------------------------------------------------------
    # Step 5: Extract and print Edit Ledger for representative tiles
    # -------------------------------------------------------------------------
    sep("Step 5 -- Edit Ledger extraction")

    def tile_in(tx, ty, mask):
        x0 = tx * tile_w;  x1 = (tx + 1) * tile_w if tx < cols - 1 else W
        y0 = ty * tile_h;  y1 = (ty + 1) * tile_h if ty < rows - 1 else H
        return bool(np.any(mask[y0:y1, x0:x1]))

    # Find one representative tile for each combination
    samples = {}
    for ty in range(rows):
        for tx in range(cols):
            in_a = tile_in(tx, ty, alice_mask)
            in_b = tile_in(tx, ty, bob_mask)
            key = ("camera+alice+bob" if (in_a and in_b) else
                   "camera+alice"     if (in_a and not in_b) else
                   "camera+bob"       if (not in_a and in_b) else
                   "camera_only")
            if key not in samples:
                samples[key] = (tx, ty)

    report = [
        "PBC Edit Ledger Report",
        "=" * 70,
        f"Image : {W}x{H}  |  Grid: {cols}x{rows}  |  {total_tiles} tiles",
        f"Authors: {CAMERA_ID} / {ALICE_ID} / {BOB_ID}",
        "",
    ]

    for label, (tx, ty) in sorted(samples.items()):
        tile_result = result.tile_results[ty][tx]
        ledger = extract_edit_ledger(tile_result)
        status = tile_result.status.name

        header = f"Tile ({tx},{ty})  [{label}]  status={status}"
        dashes = "-" * 66
        print(f"\n  {header}")
        print(f"  {dashes}")
        report += [header, dashes]

        if not ledger:
            msg = "  [no ledger entries]"
            print(msg); report.append(msg)
        else:
            for i, entry in enumerate(ledger):
                author = format_oid(entry.originator_id)
                line = (f"  Entry {i+1}: {entry.opcode_name:<22}"
                        f" by {author:<28}"
                        f" blocks {entry.start_block:>4}-{entry.end_block:>4}"
                        f" ({entry.block_count:>4} blk)")
                print(line); report.append(line)

        report.append("")

    report_path = os.path.join(RESULTS_DIR, 'ledger_ledger_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n  Full ledger saved to: {os.path.basename(report_path)}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    sep("Demo complete -- output files in ./output/")
    print()
    print("  ledger_01_original.png        Source photograph (no PBC)")
    print("  ledger_02_captured.png        After camera PBC encoding (Camera_ISP)")
    print("  ledger_03_alice_region.png    Alice's edit region (yellow highlight)")
    print("  ledger_04_bob_region.png      Bob's edit region (blue highlight)")
    print("  ledger_05_after_alice.png     Image after Alice's chain append")
    print("  ledger_06_after_bob.png       Image after Bob's chain append (final)")
    print("  ledger_07_overlay_final.png   Tile integrity overlay of final image")
    print("  ledger_08_tilemap_final.png   Tile status grid")
    print("  ledger_ledger_report.txt      Full Edit Ledger for sample tiles")
    print()
    print("  Expected outcome:")
    print("  * All tiles GREEN -- the chain is valid end-to-end")
    print("  * camera_only tiles    : 1 ledger entry  (Camera_ISP throughout)")
    print("  * camera+alice tiles   : 2 ledger entries (Camera, then Alice)")
    print("  * camera+bob tiles     : 2 ledger entries (Camera, then Bob)")
    print("  * camera+alice+bob     : 3 ledger entries (Camera, Alice, Bob)")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
