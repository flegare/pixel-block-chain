import numpy as np
import pytest

from pbc.encoder import encode
from pbc.decoder import TileStatus, verify
from pbc import compute_grid


TIMESTAMP = 1_700_000_000
TILE_SIZE = 128


def random_rgb(width=256, height=256, seed=123):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse else float("inf")


def test_encode_verify_round_trip_all_tiles_green():
    img = random_rgb()
    enc = encode(img, originator="pytest-camera", timestamp=TIMESTAMP)
    result = verify(enc, strict=True)

    assert result.green_count == len(result.all_tiles)
    assert result.red_count == 0
    assert result.absent_count == 0
    assert psnr(img, enc) > 50.0


def test_tamper_is_localized_without_cascade():
    img = random_rgb(width=384, height=384)
    enc = encode(img, originator="pytest-camera", timestamp=TIMESTAMP)

    cols, rows, tile_w, tile_h = compute_grid(384, 384, TILE_SIZE)
    assert (cols, rows) == (3, 3)

    tampered = enc.copy()
    tx, ty = 1, 1
    x0, x1 = tx * tile_w, (tx + 1) * tile_w
    y0, y1 = ty * tile_h, (ty + 1) * tile_h
    tampered[y0:y1, x0:x1] = 128

    result = verify(tampered, strict=True)
    changed = [t for t in result.all_tiles if t.status != TileStatus.GREEN]

    assert len(changed) == 1
    assert changed[0].tx == tx
    assert changed[0].ty == ty
    assert result.green_count == len(result.all_tiles) - 1


def test_non_multiple_dimensions_round_trip():
    img = random_rgb(width=333, height=257, seed=333)
    enc = encode(img, originator="pytest-camera", timestamp=TIMESTAMP)
    result = verify(enc, strict=True)

    assert result.width == 333
    assert result.height == 257
    assert result.green_count == len(result.all_tiles)


def test_small_image_below_block_capacity_is_absent():
    img = random_rgb(width=8, height=8, seed=16)
    enc = encode(img, originator="pytest-camera", timestamp=TIMESTAMP)
    result = verify(enc, strict=True)

    assert np.array_equal(enc, img)
    assert len(result.all_tiles) == 1
    assert result.absent_count == 1


def test_rejects_grayscale_and_rgba_inputs():
    gray = np.zeros((128, 128), dtype=np.uint8)
    rgba = np.zeros((128, 128, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        encode(gray, timestamp=TIMESTAMP)
    with pytest.raises(ValueError):
        verify(gray)
    with pytest.raises(ValueError):
        encode(rgba, timestamp=TIMESTAMP)
    with pytest.raises(ValueError):
        verify(rgba)


def test_rejects_non_uint8_input():
    img = np.zeros((128, 128, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        encode(img, timestamp=TIMESTAMP)
