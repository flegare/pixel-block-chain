"""
PBC-Scatter Extension

Blocks are placed at pseudo-random pixel positions rather than a fixed
tile grid.  The Extension field (32 bits) of each block stores the
absolute flat-pixel index of the NEXT block in the chain.  The terminal
block uses SCATTER_TERMINAL = 0xFFFFFFFF.

Key property:
  Because block positions are not constrained by a tile grid, any block
  that physically survives a crop (of any shape or alignment) can be
  found by a sync-frame scan of the cropped image.  When the crop
  parameters are known, the pointer chain can be fully followed by
  remapping Extension values from original-image coordinates to
  cropped-image coordinates.  When crop parameters are unknown, every
  surviving genesis block (block_index == 0) can still be independently
  authenticated via its genesis hash.

Block format is IDENTICAL to regular PBC — same 256-bit structure, same
SYNC_PATTERN, same CRC, same chain hash semantics.  Only the placement
strategy differs.

MIT License - Copyright (c) 2026 Francois Legare
"""

import hashlib
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from . import (
    PBCBlock, OpCode, PBC_VERSION,
    BLOCK_BITS, PIXELS_PER_BLOCK, SYNC_PATTERN, SYNC_HAMMING_THRESHOLD,
    compute_genesis_hash, generate_originator_id, _crc16_ccitt,
)
from .encoder import (
    _extract_bits, _bits_to_bytes, _bytes_to_bits, _embed_bits,
)
from .decoder import _check_sync

SCATTER_TERMINAL: int = 0xFFFFFFFF   # Extension value: end of chain


# =============================================================================
# Position generation
# =============================================================================

def _generate_scatter_positions(W: int, H: int,
                                  n_blocks: Optional[int],
                                  seed: int) -> np.ndarray:
    """
    Return n_blocks pixel indices in the flat (H*W,) pixel array for block
    placement, sampled without replacement from the set of non-overlapping
    86-pixel-aligned slots.

    Slots are: 0, 86, 172, ... — identical spacing to grid mode but shuffled.
    This guarantees no two blocks overlap, and the verifier can find them all
    by scanning every 86th pixel.

    Args:
        W, H:     Image dimensions.
        n_blocks: Number of blocks; None = fill to capacity.
        seed:     RNG seed for reproducibility.

    Returns:
        int64 array of pixel indices, shape (n_blocks,).
    """
    max_blocks = (W * H) // PIXELS_PER_BLOCK
    if n_blocks is None:
        n_blocks = max_blocks
    n_blocks = min(n_blocks, max_blocks)

    slots = np.arange(max_blocks, dtype=np.int64) * PIXELS_PER_BLOCK
    rng   = np.random.default_rng(seed)
    rng.shuffle(slots)
    return slots[:n_blocks]


def max_scatter_blocks(W: int, H: int) -> int:
    """Maximum number of scatter blocks that fit in a W×H image."""
    return (W * H) // PIXELS_PER_BLOCK


# =============================================================================
# Scatter encoder
# =============================================================================

def scatter_encode(image: np.ndarray,
                    originator: str,
                    n_blocks: Optional[int] = None,
                    seed: int = 0,
                    opcode: int = OpCode.CAMERA_ISP,
                    timestamp: Optional[int] = None) -> np.ndarray:
    """
    Encode an image with scatter block placement.

    All n_blocks form a single linked-list chain through pseudo-random
    pixel positions.  Chain hashes and CRC are identical to regular PBC.

    Args:
        image:      RGB (H, W, 3) uint8.
        originator: Identity string.
        n_blocks:   Blocks to embed; None = fill image to capacity.
        seed:       RNG seed controlling block positions (default 0).
        opcode:     Operation code for all blocks.
        timestamp:  Unix timestamp; uses current time if None.

    Returns:
        Scatter-encoded image (H, W, 3) uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    H, W          = image.shape[:2]
    originator_id = generate_originator_id(originator)
    if timestamp is None:
        timestamp = int(time.time())
    ts_delta = timestamp % (2 ** 24)

    positions = _generate_scatter_positions(W, H, n_blocks, seed)
    n         = len(positions)

    encoded = image.copy()
    flat    = encoded.reshape(-1, 3)          # view into encoded

    genesis_hash     = compute_genesis_hash(originator_id, 0, 0, timestamp)
    prev_block_bytes = None

    for block_idx in range(n):
        pixel_pos = int(positions[block_idx])
        next_pix  = int(positions[block_idx + 1]) if block_idx + 1 < n \
                    else SCATTER_TERMINAL

        block                 = PBCBlock()
        block.sync            = SYNC_PATTERN
        block.version         = PBC_VERSION
        block.originator_id   = originator_id
        block.opcode          = opcode
        block.block_index     = block_idx & 0xFFFF
        block.tile_x          = 0           # no tile grid in scatter mode
        block.tile_y          = 0
        block.timestamp_delta = ts_delta
        block.extension       = next_pix    # pointer to next block's pixel index

        if block_idx == 0:
            block.chain_hash = genesis_hash
        else:
            block.chain_hash = block.compute_chain_hash(prev_block_bytes)

        block.crc16      = block.compute_crc()
        block_bytes      = block.to_bits()
        prev_block_bytes = bytes(block_bytes)

        _embed_bits(flat, pixel_pos, _bytes_to_bits(block_bytes))

    return encoded


# =============================================================================
# Scatter verifier
# =============================================================================

@dataclass
class ScatterChainResult:
    """Result for one discovered chain."""
    genesis_pixel:   int
    originator_id:   int
    timestamp_delta: int
    n_blocks:        int
    n_green:         int    # CRC ok + chain hash correct
    n_yellow:        int    # CRC ok + chain hash mismatch
    n_broken:        int    # pointer lead to missing/invalid block
    chain_intact:    bool   # True iff all blocks GREEN and terminal reached

    @property
    def survival_pct(self) -> float:
        return self.n_green / self.n_blocks * 100 if self.n_blocks else 0.0


@dataclass
class ScatterResult:
    """Top-level result from scatter_verify()."""
    width:        int
    height:       int
    n_chains:     int
    chains:       List[ScatterChainResult] = field(default_factory=list)
    n_candidates: int   = 0   # positions that passed sync + CRC
    scan_ms:      float = 0.0

    @property
    def total_blocks_found(self) -> int:
        return sum(c.n_blocks for c in self.chains)

    @property
    def total_green(self) -> int:
        return sum(c.n_green for c in self.chains)


def _extract_block_bytes(flat: np.ndarray, pixel_pos: int) -> Optional[bytes]:
    """
    Extract and CRC-validate the 32-byte block starting at pixel_pos in flat.
    Returns bytes32 if CRC ok, else None.
    """
    if pixel_pos < 0 or pixel_pos + PIXELS_PER_BLOCK > len(flat):
        return None
    bits  = _extract_bits(flat, pixel_pos, BLOCK_BITS)
    bdata = bytes(_bits_to_bytes(bits))
    if not _check_sync(bdata[:6], strict=False):
        return None
    try:
        b = PBCBlock.from_bits(bdata)
    except Exception:
        return None
    if b.crc16 != _crc16_ccitt(bdata[:24]):
        return None
    return bdata


def scatter_verify(image: np.ndarray,
                    crop_offset: Optional[Tuple[int, int, int]] = None
                    ) -> ScatterResult:
    """
    Verify a scatter-encoded image.

    Procedure:
      1. Scan every pixel position using a vectorized Hamming-distance
         sync-frame check (O(N), ~0.033 expected false positives per image).
      2. CRC-validate every sync candidate.
      3. Identify genesis blocks (block_index == 0 with matching genesis hash).
      4. Follow each genesis block's pointer chain, verifying CRC and chain
         hashes along the way.

    Args:
        image:       RGB (H, W, 3) uint8 — may be a cropped image.
        crop_offset: (x0, y0, W_orig) — if verifying a cropped image and
                     the crop parameters are known, Extension field values
                     (absolute pixel indices in the original image) are
                     remapped to positions in this cropped image.
                     If None, Extension values are used as-is (valid for
                     un-cropped images; chain-following may fail for crops).

    Returns:
        ScatterResult with per-chain integrity detail.
    """
    import time as _time

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")

    H, W = image.shape[:2]
    flat = image.reshape(-1, 3)
    t0   = _time.perf_counter()

    # ── Sync scan: check every pixel position ────────────────────────────────
    # Extract one LSB per channel → 3 bits per pixel as separate uint8 bytes.
    bits_3d   = (flat & 1).astype(np.uint8)         # (H*W, 3)
    bits_flat = bits_3d.reshape(-1)                  # (H*W*3,) of 0s and 1s

    # Sync pattern as bit array
    sync_bits = np.array(
        [(SYNC_PATTERN[i // 8] >> (7 - i % 8)) & 1 for i in range(48)],
        dtype=np.uint8)

    # Strided view: window i starts at bit position i*3 (one window per pixel)
    n_pixel_pos = len(flat) - 16          # 48 bits = 16 pixels
    if n_pixel_pos <= 0:
        return ScatterResult(width=W, height=H, n_chains=0)

    from numpy.lib.stride_tricks import as_strided
    windows = as_strided(bits_flat,
                          shape   = (n_pixel_pos, 48),
                          strides = (bits_flat.strides[0] * 3,
                                     bits_flat.strides[0]))
    hamming = (windows ^ sync_bits[np.newaxis, :]).sum(axis=1)
    sync_pixel_positions = np.where(hamming <= SYNC_HAMMING_THRESHOLD)[0]

    # ── CRC validation ────────────────────────────────────────────────────────
    # valid_blocks: pixel_pos (in this image) → bytes32
    valid_blocks: Dict[int, bytes] = {}

    for px in sync_pixel_positions:
        px = int(px)
        bdata = _extract_block_bytes(flat, px)
        if bdata is not None:
            valid_blocks[px] = bdata

    n_candidates = len(valid_blocks)

    # ── Coordinate remapping helper ───────────────────────────────────────────
    def remap(orig_flat_idx: int) -> int:
        """Map absolute pixel index in original image to pixel index in crop."""
        if crop_offset is None:
            return orig_flat_idx
        x0_crop, y0_crop, W_orig = crop_offset
        orig_x = orig_flat_idx % W_orig
        orig_y = orig_flat_idx // W_orig
        crop_x = orig_x - x0_crop
        crop_y = orig_y - y0_crop
        if 0 <= crop_x < W and 0 <= crop_y < H:
            return crop_y * W + crop_x
        return -1  # outside crop

    # ── Find genesis blocks and follow chains ─────────────────────────────────
    chains: List[ScatterChainResult] = []
    visited_genesis: set = set()

    for px, bdata in valid_blocks.items():
        try:
            b0 = PBCBlock.from_bits(bdata)
        except Exception:
            continue

        if b0.block_index != 0:
            continue

        # Genesis hash check (scatter uses tile_x=0, tile_y=0)
        expected_genesis = compute_genesis_hash(
            b0.originator_id, 0, 0, b0.timestamp_delta)
        if b0.chain_hash != expected_genesis:
            continue

        if px in visited_genesis:
            continue
        visited_genesis.add(px)

        # Follow the pointer chain
        n_green = 1    # genesis block is GREEN
        n_yellow = n_broken = 0
        prev_bytes   = bdata
        current_b    = b0
        chain_intact = True

        while True:
            if current_b.extension == SCATTER_TERMINAL:
                break   # clean end of chain

            next_orig = current_b.extension
            if next_orig == 0 and current_b.block_index > 0:
                # Extension=0 on non-terminal block likely means un-encoded
                n_broken += 1
                chain_intact = False
                break

            next_px = remap(next_orig)

            if next_px < 0 or next_px + PIXELS_PER_BLOCK > len(flat):
                # Pointer out of bounds (typical after crop without offset)
                n_broken += 1
                chain_intact = False
                break

            next_bdata = valid_blocks.get(next_px)
            if next_bdata is None:
                n_broken += 1
                chain_intact = False
                break

            try:
                next_b = PBCBlock.from_bits(next_bdata)
            except Exception:
                n_broken += 1
                chain_intact = False
                break

            # Chain hash verification
            expected_chain = hashlib.sha256(prev_bytes).digest()[:6]
            if next_b.chain_hash == expected_chain:
                n_green += 1
            else:
                n_yellow += 1
                chain_intact = False

            prev_bytes = next_bdata
            current_b  = next_b

        n_total = n_green + n_yellow + n_broken
        chains.append(ScatterChainResult(
            genesis_pixel   = px,
            originator_id   = b0.originator_id,
            timestamp_delta = b0.timestamp_delta,
            n_blocks        = n_total,
            n_green         = n_green,
            n_yellow        = n_yellow,
            n_broken        = n_broken,
            chain_intact    = chain_intact,
        ))

    scan_ms = (_time.perf_counter() - t0) * 1000

    return ScatterResult(
        width        = W,
        height       = H,
        n_chains     = len(chains),
        chains       = chains,
        n_candidates = n_candidates,
        scan_ms      = scan_ms,
    )


# =============================================================================
# Forest scatter encoder  (each block is its own independent genesis)
# =============================================================================

def scatter_forest_encode(image: np.ndarray,
                           originator: str,
                           n_blocks: Optional[int] = None,
                           seed: int = 0,
                           opcode: int = OpCode.CAMERA_ISP,
                           timestamp: Optional[int] = None) -> np.ndarray:
    """
    Forest-scatter encoder: every block is an independent genesis block.

    Unlike the single-chain scatter encoder, there are NO pointers between
    blocks.  Each block is self-authenticating:

        tile_x        = forest_index & 0xFF
        tile_y        = (forest_index >> 8) & 0xFF
        block_index   = 0   (genesis for its own independent chain)
        extension     = SCATTER_TERMINAL  (no next block)
        chain_hash    = compute_genesis_hash(oid, tile_x, tile_y, timestamp)

    A verifier that finds any single surviving block can independently verify
    its origin without knowing anything about other blocks.  A non-aligned
    crop retaining fraction f of the image area retains approximately f of
    all forest blocks -- each independently verifiable.

    This is the key architectural difference from single-chain scatter: losing
    any block does NOT prevent verification of the remaining blocks.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    H, W          = image.shape[:2]
    originator_id = generate_originator_id(originator)
    if timestamp is None:
        timestamp = int(time.time())
    ts_delta = timestamp % (2 ** 24)

    positions = _generate_scatter_positions(W, H, n_blocks, seed)
    n         = len(positions)

    encoded = image.copy()
    flat    = encoded.reshape(-1, 3)

    for forest_idx in range(n):
        pixel_pos = int(positions[forest_idx])
        tile_x_f  = forest_idx & 0xFF
        tile_y_f  = (forest_idx >> 8) & 0xFF

        genesis_hash = compute_genesis_hash(originator_id,
                                            tile_x_f, tile_y_f, timestamp)

        block                 = PBCBlock()
        block.sync            = SYNC_PATTERN
        block.version         = PBC_VERSION
        block.originator_id   = originator_id
        block.opcode          = opcode
        block.block_index     = 0                 # every block is a genesis
        block.tile_x          = tile_x_f
        block.tile_y          = tile_y_f
        block.timestamp_delta = ts_delta
        block.extension       = SCATTER_TERMINAL  # no pointer chain
        block.chain_hash      = genesis_hash
        block.crc16           = block.compute_crc()

        block_bytes = block.to_bits()
        _embed_bits(flat, pixel_pos, _bytes_to_bits(block_bytes))

    return encoded


# =============================================================================
# Forest scatter verifier
# =============================================================================

@dataclass
class ForestResult:
    """Result from scatter_forest_verify()."""
    width:           int
    height:          int
    n_genesis_found: int    # independent blocks with valid genesis hash
    n_candidates:    int    # blocks passing CRC check
    scan_ms:         float

    def survival_pct(self, n_encoded: int) -> float:
        """Fraction of originally encoded blocks that survived (% of n_encoded)."""
        return self.n_genesis_found / n_encoded * 100 if n_encoded else 0.0


def scatter_forest_verify(image: np.ndarray) -> ForestResult:
    """
    Verify a forest-scatter-encoded image.

    Scans every pixel position for CRC-valid blocks.  For each block with
    block_index == 0, checks:
        compute_genesis_hash(oid, block.tile_x, block.tile_y, block.timestamp_delta)
        == block.chain_hash

    Every block that passes this check is an independently verified provenance
    anchor -- it requires no context from any other block to authenticate.

    Args:
        image: RGB (H, W, 3) uint8 -- may be a cropped image.

    Returns:
        ForestResult with count of independently verified genesis blocks.
    """
    import time as _time

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")

    H, W = image.shape[:2]
    flat = image.reshape(-1, 3)
    t0   = _time.perf_counter()

    # ── Full sync scan ────────────────────────────────────────────────────────
    bits_3d   = (flat & 1).astype(np.uint8)
    bits_flat = bits_3d.reshape(-1)

    sync_bits = np.array(
        [(SYNC_PATTERN[i // 8] >> (7 - i % 8)) & 1 for i in range(48)],
        dtype=np.uint8)

    n_pixel_pos = len(flat) - 16
    if n_pixel_pos <= 0:
        return ForestResult(width=W, height=H,
                            n_genesis_found=0, n_candidates=0, scan_ms=0.0)

    from numpy.lib.stride_tricks import as_strided
    windows = as_strided(bits_flat,
                          shape   = (n_pixel_pos, 48),
                          strides = (bits_flat.strides[0] * 3,
                                     bits_flat.strides[0]))
    hamming              = (windows ^ sync_bits[np.newaxis, :]).sum(axis=1)
    sync_pixel_positions = np.where(hamming <= SYNC_HAMMING_THRESHOLD)[0]

    # ── CRC check + forest genesis hash check ─────────────────────────────────
    n_candidates    = 0
    n_genesis_found = 0

    for px in sync_pixel_positions:
        px    = int(px)
        bdata = _extract_block_bytes(flat, px)
        if bdata is None:
            continue
        n_candidates += 1

        try:
            b = PBCBlock.from_bits(bdata)
        except Exception:
            continue

        if b.block_index != 0:
            continue   # forest blocks are always block_index=0

        expected = compute_genesis_hash(b.originator_id,
                                        b.tile_x, b.tile_y,
                                        b.timestamp_delta)
        if b.chain_hash == expected:
            n_genesis_found += 1

    scan_ms = (_time.perf_counter() - t0) * 1000

    return ForestResult(
        width           = W,
        height          = H,
        n_genesis_found = n_genesis_found,
        n_candidates    = n_candidates,
        scan_ms         = scan_ms,
    )
