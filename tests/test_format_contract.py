import io

import numpy as np
from PIL import Image

from pbc.encoder import encode
from pbc.decoder import verify


TIMESTAMP = 1_700_000_000


def random_rgb(width=256, height=256, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def save_reload(arr, fmt, **kwargs):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **kwargs)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def test_lossless_png_preserves_chain():
    img = random_rgb()
    enc = encode(img, originator="pytest-format", timestamp=TIMESTAMP)
    reloaded = save_reload(enc, "PNG")
    result = verify(reloaded, strict=True)

    assert result.green_count == len(result.all_tiles)


def test_jpeg_destroys_k1_chain():
    img = random_rgb()
    enc = encode(img, originator="pytest-format", timestamp=TIMESTAMP)
    reloaded = save_reload(enc, "JPEG", quality=95)
    result = verify(reloaded, strict=True)

    assert result.green_count == 0

