"""
The vectorized production paths must match the per-bit reference implementation
in pbc/_reference.py exactly, on random and edge-case inputs.
"""

import random

import numpy as np

import pbc
from pbc import _reference as ref
from pbc import encoder as enc_mod
from pbc.encoder import encode
from pbc.decoder import verify


def test_crc_matches_reference():
    rng = np.random.default_rng(1)
    for n in list(range(0, 40)) + [24, 32, 256]:
        data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        assert pbc._crc16_ccitt(data) == ref._crc16_ccitt(data)
    rows = rng.integers(0, 256, (50, 24), dtype=np.uint8)
    assert enc_mod._crc16_rows(rows).tolist() == [ref._crc16_ccitt(r.tobytes()) for r in rows]


def test_bit_helpers_match_reference():
    rng = np.random.default_rng(2)
    r = random.Random(2)
    for _ in range(300):
        data = bytes(rng.integers(0, 256, r.randint(0, 70), dtype=np.uint8))
        assert enc_mod._bytes_to_bits(data).tolist() == ref._bytes_to_bits(data)

        bits = rng.integers(0, 2, r.randint(0, 300)).tolist()
        assert enc_mod._bits_to_bytes(bits) == ref._bits_to_bytes(bits)

        for k in (1, 2, 3, 4):
            npx = r.randint(0, 200)
            flat = rng.integers(0, 256, (npx, 3), dtype=np.uint8)
            off = r.randint(0, npx + 3)                        # includes offsets past the end
            nbits = r.choice([256, r.randint(0, 400)])        # includes partial blocks
            bits = rng.integers(0, 2, nbits).tolist()
            a, b = flat.copy(), flat.copy()
            enc_mod._embed_bits(a, off, bits, k=k)
            ref._embed_bits(b, off, bits, k=k)
            assert np.array_equal(a, b)
            assert enc_mod._extract_bits(a, off, nbits, k=k).tolist() == \
                ref._extract_bits(b, off, nbits, k=k)


def test_encode_and_verify_match_reference_on_random_images():
    rng = np.random.default_rng(3)
    r = random.Random(3)
    for _ in range(12):
        w, h = r.randint(1, 300), r.randint(1, 300)
        k = r.choice([1, 1, 2, 3, 4])
        ts = 1_700_000_000 + r.randint(0, 10 ** 6)
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

        fast = encode(img, originator="equiv", timestamp=ts, k=k)
        slow = ref.encode_reference(img, originator="equiv", timestamp=ts, k=k)
        assert np.array_equal(fast, slow), (w, h, k)

        tampered = fast.copy()
        for _ in range(3):
            y, x = r.randrange(h), r.randrange(w)
            if r.random() < 0.5:
                tampered[y:y + 20, x:x + 30] = rng.integers(0, 256, tampered[y:y + 20, x:x + 30].shape,
                                                            dtype=np.uint8)
            else:
                tampered[y, x, r.randrange(3)] ^= 1           # single-bit flips hit sync/CRC/hash paths
        for strict in (False, True):
            assert verify(tampered, strict=strict, k=k) == \
                ref.verify_reference(tampered, strict=strict, k=k), (w, h, k, strict)
