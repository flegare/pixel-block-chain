"""
Pixel Block Chain (PBC) - Encoder

Embeds PBC blocks into image pixel data using LSB steganography.
Each tile in the adaptive grid receives its own independent chain.

MIT License - Copyright (c) 2026 François Légaré
"""

import hashlib
import struct
import time
import numpy as np
from typing import Optional

from . import (
    PBCBlock, OpCode, BLOCK_BITS, BITS_PER_CHANNEL, CHANNELS,
    BITS_PER_PIXEL, PIXELS_PER_BLOCK, LSB_MASK, CLEAR_MASK,
    SYNC_PATTERN, PBC_VERSION, TERMINAL_INDEX,
    DEFAULT_TILE_SIZE, compute_grid,
    compute_genesis_hash, generate_originator_id, _crc16_ccitt, _CRC16_TABLE
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

            # Serialize the tile's whole chain, then embed it in one vectorized
            # write (bit-exact with the per-block loop in pbc/_reference.py).
            blocks = _serialize_chain(num_blocks, 0, originator_id, opcode, tx, ty,
                                      ts_delta, 0, genesis_hash)
            _write_blocks(tile_flat, blocks, 0, pixels_per_block, k)

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
            orig_bytes   = _read_blocks(tile_flat, 0, 1, PIXELS_PER_BLOCK)[0].tobytes()
            genesis_hash = PBCBlock().compute_chain_hash(orig_bytes)

            blocks = _serialize_chain(num_blocks, 0, originator_id, opcode, tx, ty,
                                      ts_delta, 0, genesis_hash)
            _write_blocks(tile_flat, blocks, 0, PIXELS_PER_BLOCK)

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
            pivot_bytes = _read_blocks(tile_flat, split_block - 1, 1, PIXELS_PER_BLOCK)[0].tobytes()

            blocks = _serialize_chain(num_blocks - split_block, split_block, originator_id,
                                      opcode, tx, ty, ts_delta, 0,
                                      PBCBlock().compute_chain_hash(pivot_bytes))
            _write_blocks(tile_flat, blocks, split_block, PIXELS_PER_BLOCK)

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


# =============================================================================
# Bit manipulation helpers
# =============================================================================

# Vectorized with NumPy.  Every function here is verified bit-exact against the
# per-bit reference implementation in pbc/_reference.py
# (tests/test_reference_equivalence.py, tests/test_golden_bitexact.py).

def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to an array of individual bits (uint8, MSB first)."""
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def _group_bits(bits: np.ndarray, k: int) -> np.ndarray:
    """Pack rows of bits k at a time, MSB first: (..., m*k) -> (..., m) uint8."""
    if k == 1:
        return bits
    groups = bits.reshape(bits.shape[:-1] + (-1, k))
    vals = np.zeros(groups.shape[:-1], dtype=np.uint8)
    for j in range(k):
        vals |= groups[..., j] << (k - 1 - j)
    return vals


def _flat_slots(flat_pixels: np.ndarray, start: int, count: int):
    """(read view or values, writer) for `count` channel slots from slot `start`."""
    if flat_pixels.flags.c_contiguous:
        seg = flat_pixels.reshape(-1)[start:start + count]
        def write(values):
            seg[:] = values
        return seg, write
    idx = np.arange(start, start + count)
    rows, cols = idx // CHANNELS, idx % CHANNELS
    def write(values):
        flat_pixels[rows, cols] = values
    return flat_pixels[rows, cols], write


def _embed_bits(flat_pixels: np.ndarray, pixel_offset: int, bits,
                k: int = 1):
    """
    Embed a bit stream into the k least-significant bits of each channel.

    Each pixel contributes k bits per channel (3k bits total).
    k=1: embed in bit 0 only (LSB).  k=3: embed in bits 2-1-0.
    The last channel slot is zero-padded; writing stops at the array end.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n_px = len(flat_pixels)
    if bits.size == 0 or pixel_offset >= n_px:
        return
    slots = min(-(-bits.size // k), (n_px - pixel_offset) * CHANNELS)
    padded = np.zeros(slots * k, dtype=np.uint8)
    used = min(bits.size, slots * k)
    padded[:used] = bits[:used]
    current, write = _flat_slots(flat_pixels, pixel_offset * CHANNELS, slots)
    write((current & (0xFF ^ ((1 << k) - 1))) | _group_bits(padded, k))


def _extract_bits(flat_pixels: np.ndarray, pixel_offset: int,
                  num_bits: int, k: int = 1) -> np.ndarray:
    """
    Extract bits from the k least-significant bits of each channel.

    Reads k bits per channel (MSB first within each channel).
    k=1: read bit 0.  k=3: read bits 2-1-0 (MSB first).
    Returns a uint8 bit array, shorter than num_bits if the array ends first.
    """
    n_px = len(flat_pixels)
    if num_bits <= 0 or pixel_offset >= n_px:
        return np.zeros(0, dtype=np.uint8)
    slots = min(-(-num_bits // k), (n_px - pixel_offset) * CHANNELS)
    vals = _flat_slots(flat_pixels, pixel_offset * CHANNELS, slots)[0] & ((1 << k) - 1)
    if k == 1:
        return vals[:num_bits]
    bits = np.unpackbits(vals[:, None], axis=1)[:, 8 - k:].reshape(-1)
    return bits[:num_bits]


def _bits_to_bytes(bits) -> bytes:
    """Convert a sequence of bits (MSB first) back to bytes, zero-padding the last."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


# =============================================================================
# Whole-tile block helpers (vectorized)
# =============================================================================

_SYNC_ROW = np.frombuffer(SYNC_PATTERN, dtype=np.uint8)
_CRC16_TABLE_NP = np.array(_CRC16_TABLE, dtype=np.uint32)


def _crc16_rows(rows: np.ndarray) -> np.ndarray:
    """CRC-16/CCITT of every row of a (n, L) uint8 array at once (uint32 values)."""
    crc = np.full(rows.shape[0], 0xFFFF, dtype=np.uint32)
    for j in range(rows.shape[1]):
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE_NP[(crc >> 8) ^ rows[:, j]]
    return crc


def _be_columns(values, nbytes: int, n: int, field_bytes: Optional[int] = None) -> np.ndarray:
    """Big-endian byte columns (n, nbytes) of an unsigned scalar or per-block sequence.

    Values must fit in `field_bytes` (default nbytes) bytes, as struct.pack requires;
    only the low `nbytes` bytes are kept (the 24-bit timestamp packs '>I'[1:]).
    """
    if np.ndim(values) == 0:                       # same value for every block
        v = int(values)
        if v < 0 or v >> (8 * (field_bytes or nbytes)):
            raise struct.error(f"value out of range for a {field_bytes or nbytes}-byte field")
        return np.frombuffer((v & ((1 << (8 * nbytes)) - 1)).to_bytes(nbytes, "big"),
                             dtype=np.uint8)
    v = np.asarray(values, dtype=np.int64)
    if np.any(v < 0) or np.any(v >> (8 * (field_bytes or nbytes))):
        raise struct.error(f"value out of range for a {field_bytes or nbytes}-byte field")
    shifts = np.arange((nbytes - 1) * 8, -1, -8)
    return ((np.broadcast_to(v, (n,))[:, None] >> shifts) & 0xFF).astype(np.uint8)


def _serialize_chain(n: int, first_index: int, originator_id, opcode, tile_x: int,
                     tile_y: int, timestamp_delta: int, extension,
                     chain_hash0: bytes) -> np.ndarray:
    """
    Serialize n consecutive blocks of one tile chain as (n, 32) uint8 rows.

    Block i gets block_index first_index + i; block 0 carries chain_hash0 and
    every later block the truncated SHA-256 of the previous block's 32 bytes.
    originator_id, opcode and extension may be scalars or per-block sequences.
    Field layout and CRC match PBCBlock.to_bits() / compute_crc() byte for byte.
    """
    rows = np.empty((n, 32), dtype=np.uint8)
    if n == 0:
        return rows
    rows[:, 0:6]   = _SYNC_ROW
    rows[:, 6]     = PBC_VERSION & 0xFF
    rows[:, 7:11]  = _be_columns(originator_id, 4, n)
    rows[:, 11:13] = _be_columns(opcode, 2, n)
    rows[:, 13:15] = _be_columns(np.arange(first_index, first_index + n) & 0xFFFF, 2, n)
    rows[:, 15]    = tile_x & 0xFF
    rows[:, 16]    = tile_y & 0xFF
    rows[:, 17:20] = _be_columns(timestamp_delta, 3, n, field_bytes=4)
    rows[:, 20:24] = _be_columns(extension, 4, n)
    crc = _crc16_rows(rows[:, :24])                # CRC excludes the chain hash
    rows[:, 24] = crc >> 8
    rows[:, 25] = crc & 0xFF

    # The hash chain is inherently sequential: block i hashes block i-1.
    buf = bytearray(rows.tobytes())
    chain_hash = bytes(chain_hash0[:6])
    sha256 = hashlib.sha256
    for o in range(0, n * 32, 32):
        buf[o + 26:o + 32] = chain_hash
        chain_hash = sha256(buf[o:o + 32]).digest()[:6]
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, 32)


def _write_blocks(tile_flat: np.ndarray, blocks: np.ndarray, first_block: int,
                  pixels_per_block: int, k: int = 1):
    """
    Embed (n, 32) block rows into a tile, block i at pixel
    (first_block + i) * pixels_per_block.  Same result as calling
    _embed_bits(tile_flat, offset, _bytes_to_bits(row), k) row by row; every
    block must fit inside the tile.
    """
    n = blocks.shape[0]
    if n == 0:
        return
    slots_used = -(-BLOCK_BITS // k)
    per_block = pixels_per_block * CHANNELS
    start, stop = first_block * per_block, (first_block + n) * per_block
    if not tile_flat.flags.c_contiguous or stop > tile_flat.size:
        for i in range(n):                          # general (slow) path, never hit by callers
            _embed_bits(tile_flat, (first_block + i) * pixels_per_block,
                        _bytes_to_bits(blocks[i].tobytes()), k=k)
        return
    bits = np.zeros((n, slots_used * k), dtype=np.uint8)
    bits[:, :BLOCK_BITS] = np.unpackbits(blocks, axis=1)
    region = tile_flat.reshape(-1)[start:stop].reshape(n, per_block)
    region[:, :slots_used] = (region[:, :slots_used] & (0xFF ^ ((1 << k) - 1))) \
        | _group_bits(bits, k)


def _read_blocks(tile_flat: np.ndarray, first_block: int, n: int,
                 pixels_per_block: int, k: int = 1) -> np.ndarray:
    """
    (n, 32) uint8 rows of the n blocks starting at block first_block.  Same bytes
    as _bits_to_bytes(_extract_bits(tile_flat, offset, BLOCK_BITS, k)) per block;
    every block must fit inside the tile.
    """
    slots_used = -(-BLOCK_BITS // k)
    per_block = pixels_per_block * CHANNELS
    flat = tile_flat.reshape(-1)
    region = flat[first_block * per_block:(first_block + n) * per_block]
    region = region.reshape(n, per_block)[:, :slots_used] & ((1 << k) - 1)
    if k > 1:
        region = np.unpackbits(region[:, :, None], axis=2)[:, :, 8 - k:] \
            .reshape(n, slots_used * k)[:, :BLOCK_BITS]
    return np.packbits(region, axis=1)


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

            genesis_hash = compute_genesis_hash(first_oid, tx, ty, timestamp)

            # Per-block fields of the event sequence, cut at the tile's capacity
            oids, opcodes, extensions = [], [], []
            for (orig_str, opcode, block_count, extension) in events:
                take = max(0, min(block_count, num_blocks - len(oids)))
                oids += [oid_cache[orig_str]] * take
                opcodes += [opcode] * take
                extensions += [extension & 0xFFFFFFFF] * take

            if oids:
                blocks = _serialize_chain(len(oids), 0, oids, opcodes, tx, ty,
                                          ts_delta, extensions, genesis_hash)
                _write_blocks(tile_flat, blocks, 0, PIXELS_PER_BLOCK)

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded
