#!/usr/bin/env python3
"""
PBC Ledger Condensing Demo
==========================

Demonstrates the practical difference between naive per-operation block
encoding and condensed Batch_Tonal / Batch_Structural encoding for a
realistic professional photography workflow.

WORKFLOW SIMULATED (matches Table tab:ledger_depth in the paper):

  Stage                           | Naive | Condensed | Tier
  --------------------------------|-------|-----------|-----
  Camera capture (Camera_ISP)     |     1 |         1 | genesis
  Lightroom: 80 tonal adjustments |    80 |         1 | 1 (Batch_Tonal)
  Lightroom: 3 crops/rotations    |     3 |         1 | 1 (Batch_Structural)
  Photoshop: 6 clone/heal ops     |     6 |         6 | 2 (not condensable)
  Photoshop: 20 dodge/burn passes |    20 |         1 | 1 (Batch_Tonal)
  Final export                    |     1 |         1 | single event
  --------------------------------|-------|-----------|-----
  TOTAL blocks written            |   111 |        11 |

KEY MEASUREMENTS:
  - Blocks written (budget impact)
  - Block budget used (% of tile capacity)
  - Condensing ratio
  - Remaining budget for future edits
  - Integrity verification: all tiles GREEN in both cases

Usage:
    python examples/ledger_condensing_demo.py

MIT License - Copyright (c) 2026 Francois Legare
"""

import os
import sys
import time
import struct

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc import OpCode, generate_originator_id, compute_grid
from pbc.encoder import encode_sequence
from pbc.decoder import verify, TileStatus, extract_edit_ledger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'img', 'leo.jpg')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'output', 'ledger-condensing')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'results')
TILE_SIZE  = 128
TIMESTAMP  = 1_700_000_000   # fixed for reproducibility

# Author identities
CAMERA_ID  = "NikonZ9-SN2024"
ALICE_ID   = "Lightroom-Alice"
BOB_ID     = "Photoshop-Bob"

# Extension field bitmasks for Batch blocks
#   high 16 bits = count of condensed operations
#   low  16 bits = opcode-family bitmask
TONAL_BITMASK      = 0x0001   # bit 0 = EDIT_COLOR family
STRUCTURAL_BITMASK = 0x0004   # bit 2 = EDIT_RESIZE family

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Workflow event lists
# ---------------------------------------------------------------------------

def build_naive_events():
    """111 explicit per-operation blocks -- the 'naive' recording model."""
    events = []
    # 1. Camera capture
    events.append((CAMERA_ID, OpCode.CAMERA_ISP, 1, 0))
    # 2. Lightroom: 80 tonal adjustments (Tier 1 -- each writes one block)
    for _ in range(80):
        events.append((ALICE_ID, OpCode.EDIT_COLOR, 1, 0))
    # 3. Lightroom: 3 crops/rotations (Tier 1)
    for _ in range(3):
        events.append((ALICE_ID, OpCode.EDIT_RESIZE, 1, 0))
    # 4. Photoshop: 6 clone/heal operations (Tier 2 -- not condensable)
    for _ in range(6):
        events.append((BOB_ID, OpCode.EDIT_RETOUCH, 1, 0))
    # 5. Photoshop: 20 dodge/burn passes (Tier 1)
    for _ in range(20):
        events.append((BOB_ID, OpCode.EDIT_COLOR, 1, 0))
    # 6. Final export
    events.append((BOB_ID, OpCode.EXPORT_COMPRESS, 1, 0))
    return events


def build_condensed_events():
    """11 blocks -- Tier-1 sessions condensed into single Batch blocks."""
    events = []
    # 1. Camera capture
    events.append((CAMERA_ID, OpCode.CAMERA_ISP, 1, 0))
    # 2. Lightroom 80 tonal ops -> one Batch_Tonal block (count=80, mask=TONAL)
    events.append((ALICE_ID, OpCode.BATCH_TONAL,
                   1, (80 << 16) | TONAL_BITMASK))
    # 3. Lightroom 3 crops -> one Batch_Structural block (count=3, mask=STRUCTURAL)
    events.append((ALICE_ID, OpCode.BATCH_STRUCTURAL,
                   1, (3 << 16) | STRUCTURAL_BITMASK))
    # 4. Photoshop 6 clone/heal (Tier 2 -- each block required)
    for _ in range(6):
        events.append((BOB_ID, OpCode.EDIT_RETOUCH, 1, 0))
    # 5. Photoshop 20 dodge/burn -> one Batch_Tonal block (count=20, mask=TONAL)
    events.append((BOB_ID, OpCode.BATCH_TONAL,
                   1, (20 << 16) | TONAL_BITMASK))
    # 6. Final export
    events.append((BOB_ID, OpCode.EXPORT_COMPRESS, 1, 0))
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blocks_per_tile(W, H):
    cols, rows, tw, th = compute_grid(W, H, TILE_SIZE)
    return (tw * th) // 86   # PIXELS_PER_BLOCK = 86


def opcode_name(op):
    try:
        return OpCode(op).name
    except ValueError:
        return f"0x{op:04X}"


def format_oid(oid):
    known = {
        generate_originator_id(CAMERA_ID): CAMERA_ID,
        generate_originator_id(ALICE_ID):  ALICE_ID,
        generate_originator_id(BOB_ID):    BOB_ID,
    }
    return known.get(oid, f"0x{oid:08X}")


def count_blocks_in_events(events):
    return sum(cnt for (_, _, cnt, _) in events)


def sep(title=""):
    bar = "=" * 70
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep("PBC Ledger Condensing Demo")

    # -- Load image ----------------------------------------------------------
    if not os.path.exists(IMAGE_PATH):
        print(f"  leo.jpg not found -- using synthetic 512x384 gradient.")
        arr = np.zeros((384, 512, 3), dtype=np.uint8)
        for y in range(384):
            arr[y, :, 0] = int(y / 384 * 200) + 30
            arr[y, :, 1] = 120
            arr[y, :, 2] = int((384 - y) / 384 * 180) + 40
    else:
        arr = np.array(Image.open(IMAGE_PATH).convert('RGB'))
        print(f"  Image: {IMAGE_PATH}")

    H, W = arr.shape[:2]
    cols, rows, tw, th = compute_grid(W, H, TILE_SIZE)
    total_tiles  = cols * rows
    bpt          = blocks_per_tile(W, H)   # blocks per tile capacity
    print(f"  Dimensions    : {W}x{H} px")
    print(f"  Grid          : {cols}x{rows} = {total_tiles} tiles")
    print(f"  Tile size     : {tw}x{th} px  (~{bpt} blocks/tile capacity)")

    naive_events     = build_naive_events()
    condensed_events = build_condensed_events()

    naive_blocks     = count_blocks_in_events(naive_events)
    condensed_blocks = count_blocks_in_events(condensed_events)

    # -- Print workflow table -----------------------------------------------
    sep("Workflow Scenario")
    row_fmt = "  {:<40} {:>7}  {:>11}  {}"
    print(row_fmt.format("Stage", "Naive", "Condensed", "Tier"))
    print("  " + "-" * 68)
    stages = [
        ("Camera capture (Camera_ISP)",      1,  1,  "genesis"),
        ("Lightroom: 80 tonal adjustments",  80, 1,  "1 -> Batch_Tonal (count=80)"),
        ("Lightroom: 3 crops/rotations",     3,  1,  "1 -> Batch_Structural (count=3)"),
        ("Photoshop: 6 clone/heal ops",      6,  6,  "2  (not condensable)"),
        ("Photoshop: 20 dodge/burn passes",  20, 1,  "1 -> Batch_Tonal (count=20)"),
        ("Final export",                     1,  1,  "single event"),
    ]
    for (label, n, c, tier) in stages:
        print(row_fmt.format(label, n, c, tier))
    print("  " + "-" * 68)
    print(row_fmt.format("TOTAL blocks written", naive_blocks, condensed_blocks, ""))

    # -- Encode both paths ---------------------------------------------------
    sep("Encoding")
    t0 = time.perf_counter()
    img_naive = encode_sequence(arr, naive_events, timestamp=TIMESTAMP,
                                tile_size=TILE_SIZE)
    ms_naive_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img_condensed = encode_sequence(arr, condensed_events, timestamp=TIMESTAMP,
                                    tile_size=TILE_SIZE)
    ms_condensed_enc = (time.perf_counter() - t0) * 1000

    print(f"  Naive encode:     {ms_naive_enc:.0f} ms")
    print(f"  Condensed encode: {ms_condensed_enc:.0f} ms")

    # Save outputs
    Image.fromarray(img_naive).save(
        os.path.join(OUTPUT_DIR, 'condensing_naive.png'))
    Image.fromarray(img_condensed).save(
        os.path.join(OUTPUT_DIR, 'condensing_condensed.png'))

    # -- Verify both ---------------------------------------------------------
    sep("Verification")

    t0 = time.perf_counter()
    result_naive = verify(img_naive, tile_size=TILE_SIZE)
    ms_naive_ver = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result_cond = verify(img_condensed, tile_size=TILE_SIZE)
    ms_cond_ver = (time.perf_counter() - t0) * 1000

    for label, result, ms_ver in [
        ("NAIVE",     result_naive, ms_naive_ver),
        ("CONDENSED", result_cond,  ms_cond_ver),
    ]:
        g = result.green_count
        y = result.yellow_count
        r = result.red_count
        a = result.absent_count
        t = total_tiles
        print(f"\n  [{label}]")
        print(f"    GREEN  : {g}/{t}  ({100*g/t:.1f}%)")
        print(f"    YELLOW : {y}/{t}")
        print(f"    RED    : {r}/{t}")
        print(f"    ABSENT : {a}/{t}")
        print(f"    Score  : {result.integrity_score:.1f}%  |  verify {ms_ver:.0f} ms")

    # -- Ledger extraction ---------------------------------------------------
    sep("Edit Ledger (sample tile -- condensed path)")

    # Pick tile (0,0) as representative
    sample_tx, sample_ty = 0, 0
    tile_result = result_cond.tile_results[sample_ty][sample_tx]
    ledger = extract_edit_ledger(tile_result)

    print(f"\n  Tile ({sample_tx},{sample_ty}) -- condensed chain")
    print(f"  {'Block':<10} {'Originator':<22} {'Opcode':<22} {'Details'}")
    print("  " + "-" * 70)
    for entry in ledger:
        ext_info = ""
        # Detect Batch blocks and decode Extension field
        if entry.opcode in (OpCode.BATCH_TONAL, OpCode.BATCH_STRUCTURAL):
            # Read the raw block to get extension value
            raw_block = tile_result.blocks[entry.start_block].block
            if raw_block is not None:
                condensed_count = (raw_block.extension >> 16) & 0xFFFF
                op_mask         = raw_block.extension & 0xFFFF
                ext_info = f"count={condensed_count}, opcode_mask=0x{op_mask:04X}"
        print(f"  {str(entry.start_block)+'--'+str(entry.end_block):<10}"
              f" {format_oid(entry.originator_id):<22}"
              f" {opcode_name(entry.opcode):<22}"
              f" {ext_info}")

    print(f"\n  Total ledger entries (condensed): {len(ledger)}")
    print(f"  Total valid blocks in tile:       "
          f"{sum(1 for b in tile_result.blocks if b.block is not None)}")

    # Naive ledger for comparison
    tile_naive = result_naive.tile_results[sample_ty][sample_tx]
    ledger_naive = extract_edit_ledger(tile_naive)
    print(f"\n  [{sample_tx},{sample_ty}] NAIVE -- ledger groups (consecutive same oid+opcode collapsed):")
    for entry in ledger_naive:
        print(f"    blk {entry.start_block:>3}-{entry.end_block:>3}"
              f"  {format_oid(entry.originator_id):<22}"
              f"  {opcode_name(entry.opcode)}")
    print(f"  Total ledger groups (naive): {len(ledger_naive)}")

    # -- Summary table -------------------------------------------------------
    sep("RESULTS SUMMARY")

    naive_pct     = 100.0 * naive_blocks     / bpt
    cond_pct      = 100.0 * condensed_blocks / bpt
    naive_remain  = bpt - naive_blocks
    cond_remain   = bpt - condensed_blocks
    ratio         = naive_blocks / condensed_blocks

    print(f"\n  {'Metric':<45}  {'Naive':>9}  {'Condensed':>9}")
    print(f"  {'-'*65}")
    print(f"  {'Blocks written per tile':<45}  {naive_blocks:>9}  {condensed_blocks:>9}")
    print(f"  {'Block budget used (%)':<45}  {naive_pct:>8.1f}%  {cond_pct:>8.1f}%")
    print(f"  {'Remaining block budget':<45}  {naive_remain:>9}  {cond_remain:>9}")
    print(f"  {'Tiles GREEN':<45}  {result_naive.green_count:>8}/{total_tiles}"
          f"  {result_cond.green_count:>8}/{total_tiles}")
    print(f"  {'Ledger groups (tile 0,0)':<45}  {len(ledger_naive):>9}  {len(ledger):>9}")
    print(f"\n  Condensing ratio: {naive_blocks}/{condensed_blocks} = {ratio:.1f}x block reduction")
    print(f"  After condensed workflow, {cond_remain}/{bpt} blocks ({100*cond_remain/bpt:.1f}%)"
          f" remain for future edits.")
    print(f"  A naive workflow exhausts capacity after ~{bpt/naive_blocks:.1f} edit sessions.")
    print(f"  A condensed workflow exhausts capacity after ~{bpt/condensed_blocks:.1f} edit sessions.")
    print()

    # Save results to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ledger_condensing_results.txt')
    with open(out_path, 'w') as f:
        f.write("PBC Ledger Condensing Demo Results\n\n")
        f.write(f"  {'Metric':<45}  {'Naive':>9}  {'Condensed':>9}\n")
        f.write(f"  {'-'*65}\n")
        f.write(f"  {'Blocks written per tile':<45}  {naive_blocks:>9}  {condensed_blocks:>9}\n")
        f.write(f"  {'Block budget used (%)':<45}  {naive_pct:>8.1f}%  {cond_pct:>8.1f}%\n")
        f.write(f"  {'Remaining block budget':<45}  {naive_remain:>9}  {cond_remain:>9}\n")
        f.write(f"  {'Tiles GREEN':<45}  {result_naive.green_count:>8}/{total_tiles}"
                f"  {result_cond.green_count:>8}/{total_tiles}\n")
        f.write(f"\n  Condensing ratio: {naive_blocks}/{condensed_blocks} = {ratio:.1f}x\n")
        f.write(f"  Naive exhausts capacity after ~{bpt/naive_blocks:.1f} sessions.\n")
        f.write(f"  Condensed exhausts capacity after ~{bpt/condensed_blocks:.1f} sessions.\n")
    print(f"  Results saved to: {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
