"""
PBC Video Extension

Extends PBC to video sequences by adding a per-tile inter-frame chain:
  - Block 0 of each tile in frame N is anchored to the terminal block of
    the corresponding tile in frame N-1 via a video-genesis hash.
  - Block 0's Extension field stores the frame index.
  - Standard intra-tile chains are otherwise identical to image PBC.

Tamper-detection properties:
  - Modifying pixels in frame N corrupts its intra-tile chains (RED).
  - This also changes frame N's terminal block bytes, so frame N+1's
    inter-frame check fails (inter_ok = False).
  - Swapping or inserting frames breaks the inter-frame chain at those
    positions and all downstream frames.

MIT License - Copyright (c) 2026 Francois Legare
"""

import hashlib
import struct
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from . import (
    PBCBlock, OpCode, PBC_VERSION,
    BLOCK_BITS, PIXELS_PER_BLOCK, SYNC_PATTERN,
    DEFAULT_TILE_SIZE,
    compute_grid, compute_genesis_hash, generate_originator_id,
)
from .encoder import (
    encode, _extract_bits, _bits_to_bytes, _bytes_to_bits, _embed_bits,
)
from .decoder import verify, GridResult


# =============================================================================
# Inter-frame genesis hash
# =============================================================================

def compute_video_genesis_hash(originator_id: int,
                                tile_x: int,
                                tile_y: int,
                                timestamp: int,
                                frame_index: int,
                                prev_terminal_bytes: bytes) -> bytes:
    """
    Per-tile genesis hash for video frames (frame_index > 0).

    Mixes the previous frame's terminal block bytes into the hash,
    creating an inter-frame cryptographic anchor per tile.

    Args:
        originator_id:       32-bit originator identifier.
        tile_x, tile_y:      Tile grid coordinates.
        timestamp:           24-bit-truncated timestamp.
        frame_index:         0-based frame position in the sequence.
        prev_terminal_bytes: Full 32 bytes of the terminal block from the
                             corresponding tile in the preceding encoded frame.

    Returns:
        6-byte genesis hash (SHA-256 truncated to 48 bits).
    """
    data = struct.pack('>IBBII',
                       originator_id,
                       tile_x      & 0xFF,
                       tile_y      & 0xFF,
                       timestamp   & 0xFFFFFF,
                       frame_index & 0xFFFFFFFF)
    data += prev_terminal_bytes[:32]
    return hashlib.sha256(data).digest()[:6]


# =============================================================================
# Terminal block extraction
# =============================================================================

def _tile_terminal_bytes(encoded_frame: np.ndarray,
                          tile_size: int = DEFAULT_TILE_SIZE
                          ) -> Dict[Tuple[int, int], bytes]:
    """
    Extract the raw 32-byte representation of the last (terminal) block
    in each tile of an encoded frame.

    Returns:
        dict {(tx, ty): bytes32} for every tile with at least one block.
    """
    H, W = encoded_frame.shape[:2]
    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    terminal: Dict[Tuple[int, int], bytes] = {}

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_flat  = encoded_frame[y0:y1, x0:x1].reshape(-1, 3)
            num_blocks = tile_flat.shape[0] // PIXELS_PER_BLOCK
            if num_blocks == 0:
                continue

            last_offset = (num_blocks - 1) * PIXELS_PER_BLOCK
            bits        = _extract_bits(tile_flat, last_offset, BLOCK_BITS)
            terminal[(tx, ty)] = bytes(_bits_to_bytes(bits))

    return terminal


# =============================================================================
# Video frame encoder
# =============================================================================

def encode_video_frame(frame: np.ndarray,
                        originator: str,
                        frame_index: int,
                        prev_encoded_frame: Optional[np.ndarray] = None,
                        opcode: int = OpCode.CAMERA_ISP,
                        timestamp: Optional[int] = None,
                        tile_size: int = DEFAULT_TILE_SIZE) -> np.ndarray:
    """
    Encode a single video frame with an optional inter-frame chain anchor.

    For frame_index == 0 (or prev_encoded_frame is None):
        Behaviour is identical to encode().  Block 0's Extension stores 0.

    For frame_index > 0 with prev_encoded_frame supplied:
        Block 0 of each tile uses compute_video_genesis_hash() so the
        intra-tile chain is anchored to the previous frame's terminal block.
        Block 0's Extension field stores frame_index.

    Args:
        frame:              RGB image (H, W, 3) uint8.
        originator:         Identity string for originator ID generation.
        frame_index:        0-based position of this frame in the sequence.
        prev_encoded_frame: The preceding encoded frame (same shape), or None.
        opcode:             Operation code for all blocks.
        timestamp:          Unix timestamp; uses current time if None.
        tile_size:          Target tile size in pixels.

    Returns:
        PBC-encoded frame (H, W, 3) uint8.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame (H,W,3), got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError("Expected uint8 frame")

    H, W = frame.shape[:2]
    originator_id = generate_originator_id(originator)
    if timestamp is None:
        timestamp = int(time.time())
    ts_delta = timestamp % (2 ** 24)

    cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
    encoded = frame.copy()

    prev_terminal: Optional[Dict[Tuple[int, int], bytes]] = None
    if prev_encoded_frame is not None and frame_index > 0:
        prev_terminal = _tile_terminal_bytes(prev_encoded_frame, tile_size)

    for ty in range(rows):
        for tx in range(cols):
            x0 = tx * tile_w
            x1 = (tx + 1) * tile_w if tx < cols - 1 else W
            y0 = ty * tile_h
            y1 = (ty + 1) * tile_h if ty < rows - 1 else H

            tile_pixels = encoded[y0:y1, x0:x1]
            tile_flat   = tile_pixels.reshape(-1, 3)
            num_blocks  = tile_flat.shape[0] // PIXELS_PER_BLOCK
            if num_blocks == 0:
                continue

            # Genesis hash: video-aware if we have previous frame data
            if prev_terminal is not None and (tx, ty) in prev_terminal:
                genesis_hash = compute_video_genesis_hash(
                    originator_id, tx, ty, ts_delta,
                    frame_index, prev_terminal[(tx, ty)])
            else:
                genesis_hash = compute_genesis_hash(
                    originator_id, tx, ty, timestamp)

            prev_block_bytes = None
            pixel_offset     = 0

            for block_idx in range(num_blocks):
                block                 = PBCBlock()
                block.sync            = SYNC_PATTERN
                block.version         = PBC_VERSION
                block.originator_id   = originator_id
                block.opcode          = opcode
                block.block_index     = block_idx & 0xFFFF
                block.tile_x          = tx
                block.tile_y          = ty
                block.timestamp_delta = ts_delta
                # Store frame index in block 0's Extension field
                block.extension       = (frame_index & 0xFFFFFFFF) if block_idx == 0 else 0

                if block_idx == 0:
                    block.chain_hash = genesis_hash
                else:
                    block.chain_hash = block.compute_chain_hash(prev_block_bytes)

                block.crc16      = block.compute_crc()
                block_bytes      = block.to_bits()
                prev_block_bytes = bytes(block_bytes)

                _embed_bits(tile_flat, pixel_offset, _bytes_to_bits(block_bytes))
                pixel_offset += PIXELS_PER_BLOCK

            tile_pixels[:] = tile_flat.reshape(tile_pixels.shape)

    return encoded


# =============================================================================
# Video frame verifier
# =============================================================================

@dataclass
class VideoFrameResult:
    """Verification result for a single video frame."""
    frame_index:  int
    intra_result: GridResult
    inter_ok:     bool                         # True = all tiles pass inter-frame check
    inter_detail: Dict[Tuple[int, int], dict] = field(default_factory=dict)
    # inter_detail[(tx, ty)] = {'ok': bool, 'expected': bytes6, 'got': bytes6}
    # or {'ok': False, 'reason': str} on parse/lookup failure

    @property
    def intra_status(self) -> str:
        return self.intra_result.overall_status.name

    @property
    def green_pct(self) -> float:
        """Percentage of tiles that are GREEN (intact)."""
        return self.intra_result.integrity_score


def verify_video_frame(frame: np.ndarray,
                        frame_index: int,
                        prev_encoded_frame: Optional[np.ndarray] = None,
                        tile_size: int = DEFAULT_TILE_SIZE) -> VideoFrameResult:
    """
    Verify a single video frame.

    Intra-frame check: standard verify() — CRC and intra-tile chain.
      For frame_index > 0, block 0 will show YELLOW in the standard decoder
      (its chain_hash uses the video genesis formula, not the standard one).
      This is expected and is separately validated by the inter-frame check.

    Inter-frame check: if prev_encoded_frame is provided, verify that each
      tile's block 0 chain_hash matches compute_video_genesis_hash() derived
      from prev_encoded_frame's terminal blocks.  inter_ok = False when this
      check fails (indicating tampering, reordering, or insertion/deletion).

    Returns:
        VideoFrameResult with intra_result, inter_ok, and per-tile detail.
    """
    intra_result = verify(frame, strict=False, tile_size=tile_size)

    inter_ok     = True
    inter_detail: Dict[Tuple[int, int], dict] = {}

    if prev_encoded_frame is not None and frame_index > 0:
        H, W = frame.shape[:2]
        cols, rows, tile_w, tile_h = compute_grid(W, H, tile_size)
        prev_terminal = _tile_terminal_bytes(prev_encoded_frame, tile_size)

        for ty in range(rows):
            for tx in range(cols):
                x0 = tx * tile_w
                x1 = (tx + 1) * tile_w if tx < cols - 1 else W
                y0 = ty * tile_h
                y1 = (ty + 1) * tile_h if ty < rows - 1 else H

                tile_flat = frame[y0:y1, x0:x1].reshape(-1, 3)
                if tile_flat.shape[0] < PIXELS_PER_BLOCK:
                    continue

                bits  = _extract_bits(tile_flat, 0, BLOCK_BITS)
                bdata = bytes(_bits_to_bytes(bits))
                try:
                    b0 = PBCBlock.from_bits(bdata)
                except Exception:
                    inter_detail[(tx, ty)] = {'ok': False, 'reason': 'parse_error'}
                    inter_ok = False
                    continue

                if (tx, ty) not in prev_terminal:
                    inter_detail[(tx, ty)] = {'ok': False, 'reason': 'no_prev_terminal'}
                    inter_ok = False
                    continue

                expected = compute_video_genesis_hash(
                    b0.originator_id, tx, ty, b0.timestamp_delta,
                    frame_index, prev_terminal[(tx, ty)])
                got = b0.chain_hash
                ok  = (got == expected)

                inter_detail[(tx, ty)] = {
                    'ok':       ok,
                    'expected': expected,
                    'got':      got,
                }
                if not ok:
                    inter_ok = False

    return VideoFrameResult(
        frame_index  = frame_index,
        intra_result = intra_result,
        inter_ok     = inter_ok,
        inter_detail = inter_detail,
    )


# =============================================================================
# Sequence-level wrappers
# =============================================================================

def encode_video(frames: List[np.ndarray],
                  originator: str,
                  opcode: int = OpCode.CAMERA_ISP,
                  timestamp: Optional[int] = None,
                  tile_size: int = DEFAULT_TILE_SIZE) -> List[np.ndarray]:
    """
    Encode a sequence of frames with inter-frame chains.

    Each frame is encoded with encode_video_frame().  Frame 0 uses the
    standard genesis hash; frames 1+ anchor to the preceding frame.

    Returns:
        List of encoded frames (same length and shapes as input).
    """
    if timestamp is None:
        timestamp = int(time.time())
    encoded_frames: List[np.ndarray] = []
    prev_enc: Optional[np.ndarray]   = None

    for i, frame in enumerate(frames):
        enc = encode_video_frame(
            frame, originator, i,
            prev_encoded_frame = prev_enc,
            opcode             = opcode,
            timestamp          = timestamp + i,
            tile_size          = tile_size,
        )
        encoded_frames.append(enc)
        prev_enc = enc

    return encoded_frames


def verify_video(encoded_frames: List[np.ndarray],
                  tile_size: int = DEFAULT_TILE_SIZE) -> List[VideoFrameResult]:
    """
    Verify a sequence of encoded frames (intra + inter-frame checks).

    Returns:
        List of VideoFrameResult, one per frame.
    """
    results: List[VideoFrameResult] = []
    for i, frame in enumerate(encoded_frames):
        prev = encoded_frames[i - 1] if i > 0 else None
        results.append(verify_video_frame(frame, i, prev, tile_size))
    return results
