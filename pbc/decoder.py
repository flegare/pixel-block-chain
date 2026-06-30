"""
Pixel Block Chain (PBC) - Decoder and Verifier

Extracts PBC blocks from image pixels per tile and produces a grid-level
integrity map.

MIT License - Copyright (c) 2026 François Légaré
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from . import (
    PBCBlock, BLOCK_BITS, BITS_PER_CHANNEL, CHANNELS,
    BITS_PER_PIXEL, PIXELS_PER_BLOCK, LSB_MASK, SYNC_PATTERN,
    SYNC_BITS, SYNC_HAMMING_THRESHOLD, _crc16_ccitt,
    DEFAULT_TILE_SIZE, compute_grid, compute_genesis_hash
)
from .encoder import _extract_bits, _bits_to_bytes


# =============================================================================
# Block-level Verification States
# =============================================================================

class BlockStatus(IntEnum):
    """Verification state for each block within a tile."""
    GREEN  = 0  # CRC valid, chain hash valid — provenance intact
    YELLOW = 1  # CRC valid, chain hash broken — PBC-aware re-encoding
    RED    = 2  # CRC invalid — raw tampering or non-PBC modification
    ABSENT = 3  # No valid sync frame — AI-generated, pasted, or heavy compression


@dataclass
class BlockResult:
    """Verification result for a single block."""
    status:       BlockStatus
    block_index:  int
    pixel_start:  int   # offset within the *tile's* flat pixel array
    pixel_end:    int
    opcode:       int   = 0
    originator_id: int  = 0
    block:        Optional[PBCBlock] = None


# =============================================================================
# Tile-level Verification States
# =============================================================================

class TileStatus(IntEnum):
    """Aggregate verification state for one tile."""
    GREEN  = 0  # All blocks: CRC valid + chain valid
    YELLOW = 1  # At least one CRC-valid block with broken chain (re-encoded)
    RED    = 2  # At least one block with invalid CRC (tampered)
    ABSENT = 3  # No valid sync frames found


@dataclass
class TileResult:
    """Verification result for a single tile."""
    tx:            int
    ty:            int
    status:        TileStatus
    block_count:   int
    blocks:        List[BlockResult] = field(default_factory=list)
    originator_id: Optional[int]    = None


# =============================================================================
# Grid-level Result
# =============================================================================

@dataclass
class GridResult:
    """Full grid verification result."""
    width:          int
    height:         int
    cols:           int
    rows:           int
    tile_results:   List[List[TileResult]]  # [row][col]
    overall_status: TileStatus

    # ------------------------------------------------------------------
    # Convenience aggregates (mirrors VerificationResult interface)
    # ------------------------------------------------------------------
    @property
    def all_tiles(self) -> List[TileResult]:
        return [t for row in self.tile_results for t in row]

    @property
    def total_blocks(self) -> int:
        return sum(t.block_count for t in self.all_tiles)

    @property
    def green_count(self) -> int:
        return sum(1 for t in self.all_tiles if t.status == TileStatus.GREEN)

    @property
    def yellow_count(self) -> int:
        return sum(1 for t in self.all_tiles if t.status == TileStatus.YELLOW)

    @property
    def red_count(self) -> int:
        return sum(1 for t in self.all_tiles if t.status == TileStatus.RED)

    @property
    def absent_count(self) -> int:
        return sum(1 for t in self.all_tiles if t.status == TileStatus.ABSENT)

    @property
    def integrity_score(self) -> float:
        """Percentage of tiles that are GREEN."""
        total = len(self.all_tiles)
        if not total:
            return 0.0
        return self.green_count / total * 100.0

    def summary(self) -> str:
        """Human-readable verification summary."""
        total = len(self.all_tiles)
        lines = [
            f"PBC Verification Report",
            f"{'=' * 50}",
            f"Image: {self.width}x{self.height} ({self.width * self.height:,} pixels)",
            f"Grid:  {self.cols}x{self.rows} tiles  ({total} total)",
            f"Blocks: {self.total_blocks:,}",
            f"",
        ]
        if total:
            lines += [
                f"  GREEN  (intact):     {self.green_count:>5} ({self.green_count/total*100:.1f}%)",
                f"  YELLOW (re-encoded): {self.yellow_count:>5} ({self.yellow_count/total*100:.1f}%)",
                f"  RED    (tampered):   {self.red_count:>5} ({self.red_count/total*100:.1f}%)",
                f"  ABSENT (no PBC):     {self.absent_count:>5} ({self.absent_count/total*100:.1f}%)",
                f"",
                f"Integrity score: {self.integrity_score:.1f}%",
            ]

        score = self.integrity_score
        if score > 95:
            lines.append("Status: HIGH INTEGRITY — tile provenance chain intact.")
        elif score > 50:
            lines.append("Status: PARTIAL INTEGRITY — some tiles modified.")
        elif self.green_count > 0:
            lines.append("Status: LOW INTEGRITY — significant tampering detected.")
        else:
            lines.append("Status: NO PBC DATA — image has no provenance chain.")

        return '\n'.join(lines)


# =============================================================================
# Verifier
# =============================================================================

def verify(image: np.ndarray,
           strict: bool = False,
           tile_size: int = DEFAULT_TILE_SIZE,
           k: int = 1) -> GridResult:
    """
    Verify PBC integrity of an image using the grid architecture.

    Each tile is verified independently; tampering in one tile cannot
    cascade to neighbouring tiles.

    Args:
        image:     RGB image as numpy array (H, W, 3), dtype uint8.
        strict:    If True, use exact sync matching.  If False, allow
                   Hamming distance tolerance (default).
        tile_size: Target tile size used during encoding (default 128).
        k:         Bits per channel used during encoding (must match encoder).

    Returns:
        GridResult with per-tile and overall verification status.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

    H, W = image.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    pixels_per_block_k = (BLOCK_BITS + 3 * k - 1) // (3 * k)

    tile_results: List[List[TileResult]] = [
        [None] * cols for _ in range(rows)
    ]

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_pixels = image[y0:y1, x0:x1]
            tile_flat   = tile_pixels.reshape(-1, 3)
            tile_total  = tile_flat.shape[0]

            if tile_total < pixels_per_block_k:
                tile_results[ty][tx] = TileResult(
                    tx=tx, ty=ty,
                    status=TileStatus.ABSENT,
                    block_count=0
                )
                continue

            num_blocks = tile_total // pixels_per_block_k
            block_results, first_originator = _verify_tile(
                tile_flat, num_blocks, strict, tx, ty,
                pixels_per_block=pixels_per_block_k, k=k)

            tile_status = _aggregate_tile_status(block_results)
            tile_results[ty][tx] = TileResult(
                tx=tx, ty=ty,
                status=tile_status,
                block_count=len(block_results),
                blocks=block_results,
                originator_id=first_originator
            )

    # Overall status = worst status across all tiles
    status_priority = [TileStatus.RED, TileStatus.YELLOW,
                       TileStatus.ABSENT, TileStatus.GREEN]
    all_statuses = {t.status for row in tile_results for t in row}
    overall = TileStatus.GREEN
    for s in status_priority:
        if s in all_statuses:
            overall = s
            break

    return GridResult(
        width=W, height=H,
        cols=cols, rows=rows,
        tile_results=tile_results,
        overall_status=overall
    )


def _verify_tile(tile_flat: np.ndarray,
                 num_blocks: int,
                 strict: bool,
                 tx: int, ty: int,
                 pixels_per_block: int = PIXELS_PER_BLOCK,
                 k: int = 1) -> tuple:
    """
    Verify blocks within a single tile's flattened pixel array.

    Returns:
        (block_results, first_originator_id | None)
    """
    results: List[BlockResult] = []
    prev_block_bytes = None
    first_originator = None

    for block_idx in range(num_blocks):
        pixel_start = block_idx * pixels_per_block
        pixel_end   = min(pixel_start + pixels_per_block, tile_flat.shape[0])

        block_bits  = _extract_bits(tile_flat, pixel_start, BLOCK_BITS, k=k)
        block_bytes = _bits_to_bytes(block_bits)

        # Sync check
        if not _check_sync(block_bytes[:6], strict):
            results.append(BlockResult(
                status=BlockStatus.ABSENT,
                block_index=block_idx,
                pixel_start=pixel_start,
                pixel_end=pixel_end
            ))
            prev_block_bytes = None
            continue

        # Parse
        try:
            block = PBCBlock.from_bits(block_bytes)
        except Exception:
            results.append(BlockResult(
                status=BlockStatus.RED,
                block_index=block_idx,
                pixel_start=pixel_start,
                pixel_end=pixel_end
            ))
            prev_block_bytes = None
            continue

        # CRC check
        expected_crc = _crc16_ccitt(block_bytes[:24])
        if block.crc16 != expected_crc:
            results.append(BlockResult(
                status=BlockStatus.RED,
                block_index=block_idx,
                pixel_start=pixel_start,
                pixel_end=pixel_end,
                opcode=block.opcode,
                originator_id=block.originator_id,
                block=block
            ))
            prev_block_bytes = bytes(block_bytes)
            continue

        # Genesis hash check for block 0: detects PBC-aware re-encoding (YELLOW).
        # Use the image-derived tx/ty (ground truth) rather than block fields.
        if block_idx == 0:
            expected_genesis = compute_genesis_hash(
                block.originator_id,
                tx, ty, block.timestamp_delta)
            if block.chain_hash != expected_genesis:
                results.append(BlockResult(
                    status=BlockStatus.YELLOW,
                    block_index=block_idx,
                    pixel_start=pixel_start,
                    pixel_end=pixel_end,
                    opcode=block.opcode,
                    originator_id=block.originator_id,
                    block=block
                ))
                prev_block_bytes = bytes(block_bytes)
                continue

        # Chain hash check for blocks 1+
        if block_idx > 0:
            if prev_block_bytes is None:
                # Genesis block was absent/invalid — chain is unverifiable.
                # CRC is valid but provenance chain is broken → YELLOW.
                results.append(BlockResult(
                    status=BlockStatus.YELLOW,
                    block_index=block_idx,
                    pixel_start=pixel_start,
                    pixel_end=pixel_end,
                    opcode=block.opcode,
                    originator_id=block.originator_id,
                    block=block
                ))
                prev_block_bytes = bytes(block_bytes)
                continue
            expected_hash = block.compute_chain_hash(prev_block_bytes)
            if block.chain_hash != expected_hash:
                results.append(BlockResult(
                    status=BlockStatus.YELLOW,
                    block_index=block_idx,
                    pixel_start=pixel_start,
                    pixel_end=pixel_end,
                    opcode=block.opcode,
                    originator_id=block.originator_id,
                    block=block
                ))
                prev_block_bytes = bytes(block_bytes)
                continue

        # All checks passed
        if first_originator is None:
            first_originator = block.originator_id
        results.append(BlockResult(
            status=BlockStatus.GREEN,
            block_index=block_idx,
            pixel_start=pixel_start,
            pixel_end=pixel_end,
            opcode=block.opcode,
            originator_id=block.originator_id,
            block=block
        ))
        prev_block_bytes = bytes(block_bytes)

    return results, first_originator


def _aggregate_tile_status(block_results: List[BlockResult]) -> TileStatus:
    """Collapse per-block results into a single TileStatus."""
    if not block_results:
        return TileStatus.ABSENT

    statuses = {br.status for br in block_results}

    if BlockStatus.RED    in statuses:
        return TileStatus.RED
    if BlockStatus.YELLOW in statuses:
        return TileStatus.YELLOW
    if BlockStatus.ABSENT in statuses and BlockStatus.GREEN not in statuses:
        return TileStatus.ABSENT
    return TileStatus.GREEN


# =============================================================================
# Edit Ledger extraction
# =============================================================================

@dataclass
class LedgerEntry:
    """One contiguous run of identical (originator_id, opcode) in a tile chain."""
    originator_id:   int
    opcode:          int
    timestamp_delta: int
    start_block:     int
    end_block:       int

    @property
    def opcode_name(self) -> str:
        from . import OpCode
        try:
            return OpCode(self.opcode).name
        except ValueError:
            return f"0x{self.opcode:04X}"

    @property
    def block_count(self) -> int:
        return self.end_block - self.start_block + 1

    def __str__(self) -> str:
        return (f"blocks {self.start_block:>4}-{self.end_block:>4} "
                f"({self.block_count:>4} blk)  "
                f"oid=0x{self.originator_id:08X}  "
                f"op={self.opcode_name:<22}  "
                f"ts_delta={self.timestamp_delta}")


def extract_edit_ledger(tile_result: 'TileResult') -> List[LedgerEntry]:
    """
    Extract the Edit Ledger from a verified tile result.

    Traverses the tile's block chain in order and groups consecutive blocks
    that share the same (originator_id, opcode) pair into a single LedgerEntry.
    Each entry represents one authoring event: who applied which operation,
    across which block range, and at what timestamp.

    Only blocks whose `block` field is populated (i.e. CRC-valid blocks that
    were fully parsed) contribute to the ledger. ABSENT/RED blocks are skipped.

    Returns:
        Ordered list of LedgerEntry objects from Block 0 to the last block.
        An empty list is returned if no valid blocks are present.
    """
    entries: List[LedgerEntry] = []
    current_oid:  Optional[int] = None
    current_op:   Optional[int] = None
    current_ts:   int = 0
    start_idx:    int = 0

    for br in tile_result.blocks:
        if br.block is None:
            continue  # ABSENT or RED with no parsed data — skip

        oid = br.block.originator_id
        op  = br.block.opcode
        ts  = br.block.timestamp_delta
        idx = br.block.block_index

        if current_oid is None:
            # First valid block — open first entry
            current_oid = oid
            current_op  = op
            current_ts  = ts
            start_idx   = idx
        elif oid != current_oid or op != current_op:
            # Transition: close current entry and open a new one
            entries.append(LedgerEntry(
                originator_id=current_oid,
                opcode=current_op,
                timestamp_delta=current_ts,
                start_block=start_idx,
                end_block=idx - 1,
            ))
            current_oid = oid
            current_op  = op
            current_ts  = ts
            start_idx   = idx

    # Close the final entry
    if current_oid is not None and tile_result.blocks:
        last_idx = next(
            (br.block.block_index for br in reversed(tile_result.blocks)
             if br.block is not None),
            start_idx
        )
        entries.append(LedgerEntry(
            originator_id=current_oid,
            opcode=current_op,
            timestamp_delta=current_ts,
            start_block=start_idx,
            end_block=last_idx,
        ))

    return entries


# =============================================================================
# Block extraction helper (used by encoder's encode_region)
# =============================================================================

def _extract_block_at(flat_pixels: np.ndarray, pixel_offset: int) -> bytes:
    """Extract raw block bytes at a given pixel offset."""
    bits = _extract_bits(flat_pixels, pixel_offset, BLOCK_BITS)
    return _bits_to_bytes(bits)


def _check_sync(data: bytes, strict: bool = False) -> bool:
    """Check if bytes match the sync pattern within Hamming distance."""
    if len(data) < 6:
        return False  # truncated block — cannot match
    if strict:
        return data[:6] == SYNC_PATTERN

    distance = 0
    for i in range(6):
        xor = data[i] ^ SYNC_PATTERN[i]
        distance += bin(xor).count('1')

    return distance <= SYNC_HAMMING_THRESHOLD
