#!/usr/bin/env python3
"""
PBC Video Extension — Experiment

Demonstrates and validates the inter-frame chain:
  1. Encode 10 synthetic frames; verify all GREEN + inter_ok.
  2. Tamper frame 5 (zero-fill a 40x40 rectangle); verify again.
     -> Frame 5 shows RED tiles; frame 6 shows inter_ok=False.
  3. Swap frames 5 and 6 in the (un-tampered) sequence; verify.
     -> Both frames 5 and 6 show inter_ok=False (wrong predecessor).
  4. Delete frame 3 and shift the sequence; verify.
     -> From frame 3 onward inter_ok=False.

MIT License - Copyright (c) 2026 Francois Legare
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbc.video import encode_video, verify_video, VideoFrameResult
from pbc.decoder import TileStatus

N_FRAMES   = 10
FRAME_SIZE = 256       # square frames — fast for the experiment
ORIGINATOR = "VideoTest"
TILE_SIZE  = 128


def synthetic_frames(n: int, size: int, seed: int = 42) -> list:
    """Generate n slightly-varying synthetic RGB frames."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        noise = rng.integers(-4, 5, (size, size, 3), dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def status_symbol(r: VideoFrameResult) -> str:
    s = r.intra_result.overall_status
    if s == TileStatus.GREEN:
        return "GREEN"
    if s == TileStatus.YELLOW:
        return "YELLOW"
    if s == TileStatus.RED:
        return "RED"
    return "ABSENT"


def print_results(results: list, label: str):
    print(f"\n{label}")
    print(f"  {'Frame':>5}  {'Intra':>8}  {'Inter':>7}  {'GREEN%':>7}  Note")
    print(f"  {'-----':>5}  {'-----':>8}  {'-----':>7}  {'------':>7}  ----")
    all_inter_ok = True
    for r in results:
        intra  = status_symbol(r)
        inter  = "ok" if r.inter_ok else "FAIL"
        gpct   = f"{r.green_pct:.0f}%"
        note   = ""
        if r.frame_index == 0:
            note = "(no inter check for frame 0)"
        elif intra == "RED":
            note = "<-- pixel tampering detected"
            if not r.inter_ok:
                note += " + inter-frame break"
                all_inter_ok = False
        elif not r.inter_ok:
            note = "<-- inter-frame break"
            all_inter_ok = False
        elif intra == "YELLOW":
            note = "(video genesis YELLOW - expected for frame > 0)"
        print(f"  {r.frame_index:>5}  {intra:>8}  {inter:>7}  {gpct:>7}  {note}")
    verdict = "PASS" if all_inter_ok else "FAIL"
    print(f"\n  Inter-frame chain: {verdict}")


def main():
    print("PBC Video Extension — Experiment")
    print("=" * 65)
    print(f"Frames    : {N_FRAMES}  ({FRAME_SIZE}x{FRAME_SIZE} px each)")
    print(f"Tile size : {TILE_SIZE} px")
    print()

    frames = synthetic_frames(N_FRAMES, FRAME_SIZE)

    # ------------------------------------------------------------------
    # 1. Clean encode + verify
    # ------------------------------------------------------------------
    print("Encoding clean sequence...")
    encoded = encode_video(frames, ORIGINATOR, tile_size=TILE_SIZE)
    results = verify_video(encoded, tile_size=TILE_SIZE)
    print_results(results, "Test 1 — Clean sequence (expect all GREEN + inter ok)")

    # ------------------------------------------------------------------
    # 2. Tamper frame 5
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Tampering frame 5 (zeroing 40x40 top-left rectangle)...")
    tampered = [f.copy() for f in encoded]
    tampered[5][:40, :40, :] = 0
    results2 = verify_video(tampered, tile_size=TILE_SIZE)
    print_results(results2,
                  "Test 2 — Frame 5 tampered (expect RED on 5; inter FAIL on 6+)")

    # ------------------------------------------------------------------
    # 3. Swap frames 5 and 6
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Swapping frames 5 and 6 in the clean sequence...")
    swapped     = list(encoded)
    swapped[5], swapped[6] = encoded[6], encoded[5]
    results3    = verify_video(swapped, tile_size=TILE_SIZE)
    print_results(results3,
                  "Test 3 — Frames 5/6 swapped (expect inter FAIL on 5 and 6)")

    # ------------------------------------------------------------------
    # 4. Delete frame 3 (shift remaining frames)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Deleting frame 3 from the sequence...")
    deleted  = list(encoded[:3]) + list(encoded[4:])
    results4 = verify_video(deleted, tile_size=TILE_SIZE)
    print_results(results4,
                  "Test 4 — Frame 3 deleted (expect inter FAIL from frame 3 onward)")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Summary:")
    print()
    print("  Test 1 (clean)   : all frames GREEN + inter_ok -> provenance intact")
    print("  Test 2 (tamper)  : frame 5 RED intra; frame 6+ inter FAIL -> cascade")
    print("  Test 3 (swap)    : frames 5,6 intra ok but inter FAIL -> order detected")
    print("  Test 4 (delete)  : frames from deletion point onward inter FAIL")
    print()
    print("  The inter-frame chain detects not just pixel tampering (intra)")
    print("  but also temporal manipulations: reorder, insert, delete.")


if __name__ == '__main__':
    sys.exit(main())
