"""
Pixel Block Chain (PBC) - Core Protocol Constants and Data Structures

MIT License - Copyright (c) 2026 François Légaré
"""

import struct
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

# =============================================================================
# Protocol Constants
# =============================================================================

PBC_VERSION = 0x01

# Sync frame: 0xA5C396 repeated twice = 48 bits
SYNC_PATTERN = bytes([0xA5, 0xC3, 0x96, 0xA5, 0xC3, 0x96])
SYNC_BITS = 48

# Block field sizes in bits
FIELD_SYNC        = 48
FIELD_VERSION     = 8
FIELD_ORIGINATOR  = 32
FIELD_OPCODE      = 16
FIELD_BLOCK_INDEX = 16   # 16-bit index within a tile chain (was 32, split with Tile X/Y)
FIELD_TILE_X      = 8    # Tile column index (0-based)
FIELD_TILE_Y      = 8    # Tile row index (0-based)
FIELD_TIMESTAMP   = 24
FIELD_EXTENSION   = 32
FIELD_CRC         = 16
FIELD_CHAIN_HASH  = 48

BLOCK_BITS = (FIELD_SYNC + FIELD_VERSION + FIELD_ORIGINATOR + FIELD_OPCODE +
              FIELD_BLOCK_INDEX + FIELD_TILE_X + FIELD_TILE_Y +
              FIELD_TIMESTAMP + FIELD_EXTENSION +
              FIELD_CRC + FIELD_CHAIN_HASH)  # = 256

assert BLOCK_BITS == 256, f"Block size mismatch: {BLOCK_BITS}"

BITS_PER_CHANNEL = 1
CHANNELS = 3
BITS_PER_PIXEL = BITS_PER_CHANNEL * CHANNELS  # = 3
PIXELS_PER_BLOCK = (BLOCK_BITS + BITS_PER_PIXEL - 1) // BITS_PER_PIXEL  # = 86

# LSB mask for k=1
LSB_MASK   = (1 << BITS_PER_CHANNEL) - 1  # 0x01
CLEAR_MASK = 0xFF ^ LSB_MASK              # 0xFE

# Terminal block index (16-bit field)
TERMINAL_INDEX = 0xFFFF

# Sync detection Hamming distance threshold
SYNC_HAMMING_THRESHOLD = 6


# =============================================================================
# Grid / Tile Constants
# =============================================================================

DEFAULT_TILE_SIZE = 128  # Target tile size in pixels (width and height)
MIN_TILE_SIZE     = 64   # Minimum tile dimension


def compute_grid(width: int, height: int, target_tile: int = DEFAULT_TILE_SIZE):
    """
    Compute grid dimensions for an image.

    Returns:
        (cols, rows, tile_w, tile_h) where tile_w/tile_h are actual tile sizes
        adjusted so tiles evenly cover the image.  Edge tiles extend to the
        image boundary to absorb remainder pixels (no gaps).
    """
    cols = max(1, round(width  / target_tile))
    rows = max(1, round(height / target_tile))
    # Clamp to 8-bit coordinate fields
    cols = min(cols, 255)
    rows = min(rows, 255)
    tile_w = width  // cols
    tile_h = height // rows
    return cols, rows, tile_w, tile_h


# =============================================================================
# Operation Codes
# =============================================================================

class OpCode(IntEnum):
    """Standardized operation codes for PBC blocks."""
    CAMERA_RAW      = 0x0000
    CAMERA_ISP      = 0x0001
    EDIT_CROP       = 0x0010
    EDIT_COLOR      = 0x0011
    EDIT_RESIZE     = 0x0012
    EDIT_FILTER     = 0x0013
    EDIT_COMPOSITE  = 0x0014
    EDIT_RETOUCH    = 0x0020
    EDIT_AI_ENHANCE = 0x0030
    EDIT_AI_GENERATE= 0x0031
    EXPORT_COMPRESS  = 0x0040
    EXPORT_CONVERT   = 0x0041
    GRID_RESCALE     = 0x0050
    BATCH_TONAL      = 0x0060
    BATCH_STRUCTURAL = 0x0061
    CHAIN_REPAIR     = 0xFFFE
    UNKNOWN          = 0xFFFF


# =============================================================================
# Block Data Structure
# =============================================================================

@dataclass
class PBCBlock:
    """A single PBC block (256 bits / 32 bytes)."""
    sync:            bytes = field(default_factory=lambda: SYNC_PATTERN)
    version:         int   = PBC_VERSION
    originator_id:   int   = 0
    opcode:          int   = OpCode.CAMERA_ISP
    block_index:     int   = 0          # 16-bit position within tile chain
    tile_x:          int   = 0          # Tile column index (0-based)
    tile_y:          int   = 0          # Tile row index (0-based)
    timestamp_delta: int   = 0
    extension:       int   = 0
    crc16:           int   = 0
    chain_hash:      bytes = b'\x00' * 6  # 48 bits = 6 bytes

    def to_bits(self) -> bytearray:
        """Serialize block to a 256-bit (32-byte) bitstream."""
        data = bytearray()
        # Sync frame (48 bits = 6 bytes)
        data.extend(self.sync)
        # Version (8 bits = 1 byte)
        data.append(self.version & 0xFF)
        # Originator ID (32 bits = 4 bytes)
        data.extend(struct.pack('>I', self.originator_id))
        # Operation code (16 bits = 2 bytes)
        data.extend(struct.pack('>H', self.opcode))
        # Block index (16 bits = 2 bytes)
        data.extend(struct.pack('>H', self.block_index & 0xFFFF))
        # Tile X (8 bits = 1 byte)
        data.append(self.tile_x & 0xFF)
        # Tile Y (8 bits = 1 byte)
        data.append(self.tile_y & 0xFF)
        # Timestamp delta (24 bits = 3 bytes)
        data.extend(struct.pack('>I', self.timestamp_delta)[1:])
        # Extension (32 bits = 4 bytes)
        data.extend(struct.pack('>I', self.extension))
        # CRC-16 (16 bits = 2 bytes) — computed over all preceding fields
        data.extend(struct.pack('>H', self.crc16))
        # Chain hash (48 bits = 6 bytes)
        data.extend(self.chain_hash[:6])
        assert len(data) == 32, f"Block serialization error: {len(data)} bytes"
        return data

    @classmethod
    def from_bits(cls, data: bytes) -> 'PBCBlock':
        """Deserialize a 32-byte block."""
        if len(data) < 32:
            raise ValueError(f"Need 32 bytes, got {len(data)}")
        b = cls()
        b.sync            = bytes(data[0:6])
        b.version         = data[6]
        b.originator_id   = struct.unpack('>I', data[7:11])[0]
        b.opcode          = struct.unpack('>H', data[11:13])[0]
        b.block_index     = struct.unpack('>H', data[13:15])[0]
        b.tile_x          = data[15]
        b.tile_y          = data[16]
        b.timestamp_delta = struct.unpack('>I', b'\x00' + bytes(data[17:20]))[0]
        b.extension       = struct.unpack('>I', data[20:24])[0]
        b.crc16           = struct.unpack('>H', data[24:26])[0]
        b.chain_hash      = bytes(data[26:32])
        return b

    def compute_crc(self) -> int:
        """Compute CRC-16/CCITT over all fields before CRC (first 24 bytes)."""
        data = self.to_bits()[:24]
        return _crc16_ccitt(data)

    def compute_chain_hash(self, prev_block_bytes: bytes) -> bytes:
        """Compute truncated SHA-256 of previous block's content."""
        h = hashlib.sha256(prev_block_bytes).digest()
        return h[:6]  # Truncate to 48 bits


# =============================================================================
# CRC-16/CCITT
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
# Genesis Hash
# =============================================================================

def compute_genesis_hash(originator_id: int,
                         tile_x: int = 0, tile_y: int = 0,
                         timestamp: int = 0) -> bytes:
    """
    Compute the per-tile chain seed for Block 0.

    Tile coordinates are included so each tile in the grid has a
    cryptographically independent genesis hash.  Image dimensions are
    intentionally excluded so the hash remains valid after cropping.
    """
    data = struct.pack('>IBBI',
                       originator_id,
                       tile_x    & 0xFF,
                       tile_y    & 0xFF,
                       timestamp & 0xFFFFFF)
    h = hashlib.sha256(data).digest()
    return h[:6]  # 48-bit truncation


# =============================================================================
# Originator ID generation
# =============================================================================

def generate_originator_id(identity_string: str) -> int:
    """Generate a 32-bit originator ID from an identity string."""
    h = hashlib.sha256(identity_string.encode('utf-8')).digest()
    return struct.unpack('>I', h[:4])[0]
