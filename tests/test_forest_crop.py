import numpy as np

from pbc.scatter import scatter_forest_encode, scatter_forest_verify


TIMESTAMP = 1_700_000_000
ORIGINATOR = "pytest-forest"


def random_rgb(width=360, height=240, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def crop_center(arr, width_frac, height_frac):
    h, w = arr.shape[:2]
    cw = int(w * width_frac)
    ch = int(h * height_frac)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def test_forest_full_image_recovers_all_seeded_anchors():
    img = random_rgb()
    n_blocks = 200
    enc = scatter_forest_encode(
        img, ORIGINATOR, n_blocks=n_blocks, seed=7, timestamp=TIMESTAMP
    )
    result = scatter_forest_verify(enc)

    assert result.n_genesis_found == n_blocks


def test_forest_crop_sweep_retains_independent_anchors():
    img = random_rgb()
    n_blocks = 500
    enc = scatter_forest_encode(
        img, ORIGINATOR, n_blocks=n_blocks, seed=42, timestamp=TIMESTAMP
    )

    cases = [
        (0.80, 0.80, 40.0),
        (0.60, 0.80, 25.0),
        (0.50, 0.50, 10.0),
    ]

    for width_frac, height_frac, min_survival in cases:
        cropped = crop_center(enc, width_frac, height_frac)
        result = scatter_forest_verify(cropped)
        survival = result.survival_pct(n_blocks)
        assert survival >= min_survival


def test_forest_random_seed_is_deterministic():
    img = random_rgb(seed=99)
    a = scatter_forest_encode(img, ORIGINATOR, n_blocks=100, seed=123, timestamp=TIMESTAMP)
    b = scatter_forest_encode(img, ORIGINATOR, n_blocks=100, seed=123, timestamp=TIMESTAMP)
    c = scatter_forest_encode(img, ORIGINATOR, n_blocks=100, seed=124, timestamp=TIMESTAMP)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
