"""
Pixel Block Chain (PBC) - Reference (per-bit, per-block) implementation

The original pure-Python implementation of the bit helpers, CRC, grid encoder
and tile verifier, kept verbatim as the readable reference for the paper.  The
production paths in pbc/encoder.py, pbc/decoder.py and pbc/__init__.py are
vectorized with NumPy and are verified bit-exact against this module by
tests/test_reference_equivalence.py and tests/test_golden_bitexact.py.

Do not optimise this file: its value is that it is obviously correct.

MIT License - Copyright (c) 2026 François Légaré
"""

import time
import numpy as np
from typing import List, Optional

from . import (
    PBCBlock, OpCode, BLOCK_BITS, CHANNELS, PIXELS_PER_BLOCK,
    SYNC_PATTERN, PBC_VERSION, SYNC_HAMMING_THRESHOLD,
    DEFAULT_TILE_SIZE, compute_grid,
    compute_genesis_hash, generate_originator_id,
)


# =============================================================================
# CRC-16/CCITT (bitwise)
# =============================================================================

def _crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT (polynomial 0x1021)."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


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


# =============================================================================
# Reference grid encoder (per-block loop)
# =============================================================================

def encode_reference(image: np.ndarray,
                     originator: str = "pbc-reference-encoder",
                     opcode: int = OpCode.CAMERA_ISP,
                     timestamp: Optional[int] = None,
                     tile_size: int = DEFAULT_TILE_SIZE,
                     k: int = 1) -> np.ndarray:
    """Reference version of pbc.encoder.encode() (same arguments and output)."""
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

                block.crc16 = _crc16_ccitt(block.to_bits()[:24])

                block_bytes = block.to_bits()
                prev_block_bytes = bytes(block_bytes)

                bit_stream = _bytes_to_bits(block_bytes)
                _embed_bits(tile_flat, pixel_offset, bit_stream, k=k)
                pixel_offset += pixels_per_block

            # Write tile_flat back (numpy view already aliases encoded)
            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


# =============================================================================
# Reference tile verifier (per-block loop)
# =============================================================================

def verify_tile_reference(tile_flat: np.ndarray,
                          num_blocks: int,
                          strict: bool,
                          tx: int, ty: int,
                          pixels_per_block: int = PIXELS_PER_BLOCK,
                          k: int = 1) -> tuple:
    """Reference version of pbc.decoder._verify_tile()."""
    from .decoder import BlockResult, BlockStatus

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


def verify_reference(image: np.ndarray,
                     strict: bool = False,
                     tile_size: int = DEFAULT_TILE_SIZE,
                     k: int = 1):
    """Reference version of pbc.decoder.verify() (same arguments and GridResult)."""
    from .decoder import GridResult, TileResult, TileStatus, _aggregate_tile_status

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got shape {image.shape}")

    H, W = image.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    pixels_per_block_k = (BLOCK_BITS + 3 * k - 1) // (3 * k)

    tile_results = [[None] * cols for _ in range(rows)]

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_flat  = image[y0:y1, x0:x1].reshape(-1, 3)
            tile_total = tile_flat.shape[0]

            if tile_total < pixels_per_block_k:
                tile_results[ty][tx] = TileResult(
                    tx=tx, ty=ty, status=TileStatus.ABSENT, block_count=0)
                continue

            num_blocks = tile_total // pixels_per_block_k
            block_results, first_originator = verify_tile_reference(
                tile_flat, num_blocks, strict, tx, ty,
                pixels_per_block=pixels_per_block_k, k=k)

            tile_results[ty][tx] = TileResult(
                tx=tx, ty=ty,
                status=_aggregate_tile_status(block_results),
                block_count=len(block_results),
                blocks=block_results,
                originator_id=first_originator)

    status_priority = [TileStatus.RED, TileStatus.YELLOW,
                       TileStatus.ABSENT, TileStatus.GREEN]
    all_statuses = {t.status for row in tile_results for t in row}
    overall = TileStatus.GREEN
    for s in status_priority:
        if s in all_statuses:
            overall = s
            break

    return GridResult(width=W, height=H, cols=cols, rows=rows,
                      tile_results=tile_results, overall_status=overall)
