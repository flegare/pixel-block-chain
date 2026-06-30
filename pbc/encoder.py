"""
Pixel Block Chain (PBC) - Encoder

Embeds PBC blocks into image pixel data using LSB steganography.
Each tile in the adaptive grid receives its own independent chain.

MIT License - Copyright (c) 2026 François Légaré
"""

import time
import numpy as np
from typing import Optional

from . import (
    PBCBlock, OpCode, BLOCK_BITS, BITS_PER_CHANNEL, CHANNELS,
    BITS_PER_PIXEL, PIXELS_PER_BLOCK, LSB_MASK, CLEAR_MASK,
    SYNC_PATTERN, PBC_VERSION, TERMINAL_INDEX,
    DEFAULT_TILE_SIZE, compute_grid,
    compute_genesis_hash, generate_originator_id, _crc16_ccitt
)


def encode(image: np.ndarray,
           originator: str = "pbc-reference-encoder",
           opcode: int = OpCode.CAMERA_ISP,
           timestamp: Optional[int] = None,
           tile_size: int = DEFAULT_TILE_SIZE,
           k: int = 1) -> np.ndarray:
    """
    Encode PBC blocks into an image using the grid architecture.

    The image is partitioned into an adaptive grid of tiles.  Each tile
    receives its own independent block chain seeded by a genesis hash that
    includes the tile's (tx, ty) coordinates, guaranteeing cryptographic
    independence between tiles.

    Args:
        image:      RGB image as numpy array (H, W, 3), dtype uint8.
        originator: Identity string for originator ID generation.
        opcode:     Operation code for all blocks.
        timestamp:  Unix timestamp (defaults to current time).
        tile_size:  Target tile size in pixels (default 128).
        k:          Bits per channel to embed (1=LSB only, 2=2 LSBs, 3=3 LSBs).
                    k=1 gives ~51 dB PSNR; k=2 ~45 dB; k=3 ~36 dB.
                    k>=3 survives JPEG at Q=100 (bit2 error rate <9.5%).

    Returns:
        PBC-encoded image as numpy array (H, W, 3), dtype uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if k not in (1, 2, 3, 4):
        raise ValueError(f"k must be 1, 2, 3, or 4; got {k}")

    H, W = image.shape[:2]

    originator_id = generate_originator_id(originator)
    if timestamp is None:
        timestamp = int(time.time())
    ts_delta = timestamp % (2 ** 24)

    # k-dependent geometry
    pixels_per_block = (BLOCK_BITS + 3 * k - 1) // (3 * k)

    # Compute grid
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    encoded = image.copy()

    for ty in range(rows):
        for tx in range(cols):
            # Tile pixel bounds (edge tiles extend to image boundary)
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_pixels = encoded[y0:y1, x0:x1]     # view into encoded
            tile_flat   = tile_pixels.reshape(-1, 3) # (N, 3) flattened view
            tile_total  = tile_flat.shape[0]

            if tile_total < pixels_per_block:
                continue  # tile too small to fit even one block; skip

            num_blocks = tile_total // pixels_per_block

            # Per-tile genesis hash includes tile coordinates
            genesis_hash = compute_genesis_hash(
                originator_id, tx, ty, timestamp)

            prev_block_bytes = None
            pixel_offset     = 0

            for block_idx in range(num_blocks):
                block = PBCBlock()
                block.sync            = SYNC_PATTERN
                block.version         = PBC_VERSION
                block.originator_id   = originator_id
                block.opcode          = opcode
                block.block_index     = block_idx & 0xFFFF
                block.tile_x          = tx
                block.tile_y          = ty
                block.timestamp_delta = ts_delta
                block.extension       = 0

                if block_idx == 0:
                    block.chain_hash = genesis_hash
                else:
                    block.chain_hash = block.compute_chain_hash(prev_block_bytes)

                block.crc16 = block.compute_crc()

                block_bytes = block.to_bits()
                prev_block_bytes = bytes(block_bytes)

                bit_stream = _bytes_to_bits(block_bytes)
                _embed_bits(tile_flat, pixel_offset, bit_stream, k=k)
                pixel_offset += pixels_per_block

            # Write tile_flat back (numpy view already aliases encoded)
            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


def encode_region(image: np.ndarray,
                  region_mask: np.ndarray,
                  originator: str,
                  opcode: int,
                  timestamp: Optional[int] = None,
                  tile_size: int = DEFAULT_TILE_SIZE) -> np.ndarray:
    """
    Re-encode PBC blocks in tiles overlapping a modified region.

    A PBC-aware editor calls this after modifying pixels: only the tiles
    that overlap the mask are re-encoded (with the editor's originator ID
    and the supplied opcode).  Unaffected tiles keep their original chains.

    Args:
        image:       PBC-encoded RGB image (H, W, 3), uint8.
        region_mask: Boolean mask (H, W) — True where pixels were modified.
        originator:  Identity string for the editing software.
        opcode:      Operation code for the edit type.
        timestamp:   Unix timestamp.
        tile_size:   Target tile size (must match original encoding).

    Returns:
        Re-encoded image with updated chains in touched tiles.
    """
    if timestamp is None:
        timestamp = int(time.time())

    H, W = image.shape[:2]
    originator_id = generate_originator_id(originator)
    ts_delta = timestamp % (2 ** 24)

    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    encoded = image.copy()

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            # Check if this tile overlaps the modified region
            if not np.any(region_mask[y0:y1, x0:x1]):
                continue

            tile_pixels = encoded[y0:y1, x0:x1]
            tile_flat   = tile_pixels.reshape(-1, 3)
            tile_total  = tile_flat.shape[0]

            if tile_total < PIXELS_PER_BLOCK:
                continue

            num_blocks = tile_total // PIXELS_PER_BLOCK

            # Chain block 0 from the existing block 0 bytes so the decoder
            # sees a genesis mismatch and correctly flags this tile YELLOW
            # (PBC-aware re-encoding), instead of GREEN (untouched original).
            orig_bits    = _extract_bits(tile_flat, 0, BLOCK_BITS)
            orig_bytes   = _bits_to_bytes(orig_bits)
            genesis_hash = PBCBlock().compute_chain_hash(orig_bytes)

            prev_block_bytes = None
            pixel_offset     = 0

            for block_idx in range(num_blocks):
                block = PBCBlock()
                block.sync            = SYNC_PATTERN
                block.version         = PBC_VERSION
                block.originator_id   = originator_id
                block.opcode          = opcode
                block.block_index     = block_idx & 0xFFFF
                block.tile_x          = tx
                block.tile_y          = ty
                block.timestamp_delta = ts_delta
                block.extension       = 0

                if block_idx == 0:
                    block.chain_hash = genesis_hash
                else:
                    block.chain_hash = block.compute_chain_hash(prev_block_bytes)

                block.crc16  = block.compute_crc()
                block_bytes  = block.to_bits()
                prev_block_bytes = bytes(block_bytes)

                bit_stream = _bytes_to_bits(block_bytes)
                _embed_bits(tile_flat, pixel_offset, bit_stream)
                pixel_offset += PIXELS_PER_BLOCK

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


def append_edit(image: np.ndarray,
                originator: str,
                opcode: int,
                timestamp: Optional[int] = None,
                tile_size: int = DEFAULT_TILE_SIZE,
                region_mask: Optional[np.ndarray] = None,
                split_fraction: float = 0.5) -> np.ndarray:
    """
    Append edit blocks to existing tile chains (Edit Ledger / append mode).

    Overwrites blocks from split_fraction onward in each affected tile,
    continuing the chain from the block immediately before the split point.
    The first portion of the chain (blocks 0..split-1) is left untouched,
    preserving the prior history. The new blocks record the editor's
    originator ID and opcode.

    At decode time, the full chain is valid end-to-end. Reading the sequence
    of (originator_id, opcode) values across blocks reveals the Edit Ledger:
    each contiguous run of identical (oid, opcode) pairs is one ledger entry.

    Args:
        image:          PBC-encoded RGB image (H, W, 3), uint8.
        originator:     Identity string of the editing tool / person.
        opcode:         Operation code for this edit (from OpCode registry).
        timestamp:      Unix timestamp (defaults to current time).
        tile_size:      Must match the original encoding tile size.
        region_mask:    Boolean (H, W) mask — True for pixels affected by
                        this edit. If None, all tiles are updated.
        split_fraction: Fraction of each tile's blocks to preserve as prior
                        history. 0.5 means the first half keeps its existing
                        chain; the second half is overwritten with this edit.
                        Must be in (0.0, 1.0).

    Returns:
        Image with Edit Ledger appended to all affected tiles.
    """
    if not 0.0 < split_fraction < 1.0:
        raise ValueError(f"split_fraction must be in (0.0, 1.0), got {split_fraction}")
    if timestamp is None:
        timestamp = int(time.time())

    H, W = image.shape[:2]
    originator_id = generate_originator_id(originator)
    ts_delta = timestamp % (2 ** 24)

    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    encoded = image.copy()

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            if region_mask is not None and not np.any(region_mask[y0:y1, x0:x1]):
                continue

            tile_pixels = encoded[y0:y1, x0:x1]
            tile_flat   = tile_pixels.reshape(-1, 3)
            tile_total  = tile_flat.shape[0]

            num_blocks = tile_total // PIXELS_PER_BLOCK
            if num_blocks < 2:
                continue

            # The "pivot" block is the last block we keep intact.
            # New blocks are written starting at split_block.
            split_block = max(1, int(num_blocks * split_fraction))

            # Read the pivot block's bytes to chain from it.
            pivot_offset = (split_block - 1) * PIXELS_PER_BLOCK
            pivot_bits   = _extract_bits(tile_flat, pivot_offset, BLOCK_BITS)
            pivot_bytes  = bytes(_bits_to_bytes(pivot_bits))

            prev_block_bytes = pivot_bytes

            for block_idx in range(split_block, num_blocks):
                block = PBCBlock()
                block.sync            = SYNC_PATTERN
                block.version         = PBC_VERSION
                block.originator_id   = originator_id
                block.opcode          = opcode
                block.block_index     = block_idx & 0xFFFF
                block.tile_x          = tx
                block.tile_y          = ty
                block.timestamp_delta = ts_delta
                block.extension       = 0
                block.chain_hash      = block.compute_chain_hash(prev_block_bytes)
                block.crc16           = block.compute_crc()

                block_bytes      = block.to_bits()
                prev_block_bytes = bytes(block_bytes)

                bit_stream   = _bytes_to_bits(block_bytes)
                pixel_offset = block_idx * PIXELS_PER_BLOCK
                _embed_bits(tile_flat, pixel_offset, bit_stream)

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


# =============================================================================
# Bit manipulation helpers
# =============================================================================

def _bytes_to_bits(data: bytes) -> list:
    """Convert bytes to a list of individual bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _embed_bits(flat_pixels: np.ndarray, pixel_offset: int, bits: list,
                k: int = 1):
    """
    Embed a bit stream into the k least-significant bits of each channel.

    Each pixel contributes k bits per channel (3k bits total).
    k=1: embed in bit 0 only (LSB).  k=3: embed in bits 2-1-0.
    """
    lsb_mask_k   = (1 << k) - 1
    clear_mask_k = 0xFF ^ lsb_mask_k
    bit_idx    = 0
    total_bits = len(bits)
    px         = pixel_offset

    while bit_idx < total_bits and px < len(flat_pixels):
        for ch in range(CHANNELS):
            val = 0
            for b in range(k):
                val = (val << 1) | (bits[bit_idx] if bit_idx < total_bits else 0)
                bit_idx += 1
            flat_pixels[px, ch] = (int(flat_pixels[px, ch]) & clear_mask_k) | val
            if bit_idx >= total_bits:
                break
        px += 1


def _extract_bits(flat_pixels: np.ndarray, pixel_offset: int,
                  num_bits: int, k: int = 1) -> list:
    """
    Extract bits from the k least-significant bits of each channel.

    Reads k bits per channel (MSB first within each channel).
    k=1: read bit 0.  k=3: read bits 2-1-0 (MSB first).
    """
    lsb_mask_k = (1 << k) - 1
    bits = []
    px   = pixel_offset

    while len(bits) < num_bits and px < len(flat_pixels):
        for ch in range(CHANNELS):
            val = int(flat_pixels[px, ch]) & lsb_mask_k
            for b in range(k - 1, -1, -1):
                bits.append((val >> b) & 1)
            if len(bits) >= num_bits:
                break
        px += 1

    return bits[:num_bits]


def _bits_to_bytes(bits: list) -> bytes:
    """Convert a list of bits back to bytes."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
            else:
                byte = byte << 1
        result.append(byte)
    return bytes(result)


def encode_sequence(image: np.ndarray,
                    events: list,
                    timestamp: Optional[int] = None,
                    tile_size: int = DEFAULT_TILE_SIZE) -> np.ndarray:
    """
    Encode an explicit sequence of ledger events into all tile chains.

    Each element of ``events`` is a 4-tuple:
        (originator_str, opcode, block_count, extension)

    where:
        originator_str  -- identity string (same input as ``encode()``)
        opcode          -- operation code (OpCode registry, incl. Batch opcodes)
        block_count     -- consecutive blocks to write for this event;
                           1 for every condensed / Tier-2 entry,
                           N for a naïve per-operation entry
        extension       -- Extension field value:
                           0 for normal blocks;
                           (condensed_count << 16) | opcode_bitmask for Batch

    The genesis hash for block 0 of each tile uses the *first* event's
    originator ID plus tile (tx, ty) and timestamp, so block 0 verifies GREEN.
    Blocks at positions beyond the last written event retain original pixel
    values and report ABSENT (does not affect tile GREEN status).

    Primary use: demonstrating naïve vs. condensed ledger depth side-by-side.
    """
    if not events:
        return image.copy()
    if timestamp is None:
        timestamp = int(time.time())

    H, W = image.shape[:2]
    ts_delta = timestamp % (2 ** 24)
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)

    # Cache originator IDs
    oid_cache: dict = {}
    for (orig_str, _op, _cnt, _ext) in events:
        if orig_str not in oid_cache:
            oid_cache[orig_str] = generate_originator_id(orig_str)

    first_oid = oid_cache[events[0][0]]

    encoded = image.copy()

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_pixels = encoded[y0:y1, x0:x1]
            tile_flat   = tile_pixels.reshape(-1, 3)
            num_blocks  = tile_flat.shape[0] // PIXELS_PER_BLOCK

            if num_blocks < 1:
                continue

            genesis_hash     = compute_genesis_hash(first_oid, tx, ty, timestamp)
            prev_block_bytes = None
            global_idx       = 0

            for (orig_str, opcode, block_count, extension) in events:
                oid = oid_cache[orig_str]
                for _ in range(block_count):
                    if global_idx >= num_blocks:
                        break

                    block = PBCBlock()
                    block.sync            = SYNC_PATTERN
                    block.version         = PBC_VERSION
                    block.originator_id   = oid
                    block.opcode          = opcode
                    block.block_index     = global_idx & 0xFFFF
                    block.tile_x          = tx
                    block.tile_y          = ty
                    block.timestamp_delta = ts_delta
                    block.extension       = extension & 0xFFFFFFFF

                    if global_idx == 0:
                        block.chain_hash = genesis_hash
                    else:
                        block.chain_hash = block.compute_chain_hash(prev_block_bytes)

                    block.crc16      = block.compute_crc()
                    block_bytes      = block.to_bits()
                    prev_block_bytes = bytes(block_bytes)

                    bit_stream = _bytes_to_bits(block_bytes)
                    _embed_bits(tile_flat, global_idx * PIXELS_PER_BLOCK, bit_stream)
                    global_idx += 1

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded
