#!/usr/bin/env python3
"""Deterministic validation runner for paper-critical PBC claims.

The runner emits machine-readable JSON plus a compact Markdown report.  It is
intended for reviewer reproduction, not as a replacement for the full benchmark
scripts under examples/.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pbc import OpCode, compute_grid  # noqa: E402
from pbc.encoder import encode  # noqa: E402
from pbc.decoder import TileStatus, verify  # noqa: E402
from pbc.scatter import scatter_forest_encode, scatter_forest_verify  # noqa: E402


TIMESTAMP = 1_700_000_000
TILE_SIZE = 128
OUT_DIR = ROOT / "results"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse) if mse else float("inf")


def make_synthetic_images() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(99)
    h, w = 384, 512
    y, x = np.mgrid[0:h, 0:w]
    gradient = np.stack(
        [
            (255 * x / w),
            (255 * y / h),
            (255 * (1 - x / w)),
        ],
        axis=2,
    ).astype(np.uint8)
    document = np.full((h, w, 3), 245, dtype=np.uint8)
    document[::24, 40:w - 40] = 30
    noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return [("synthetic_gradient", gradient), ("synthetic_document", document), ("synthetic_noise", noise)]


def load_validation_images() -> list[tuple[str, np.ndarray]]:
    images = []
    leo = ROOT / "examples" / "img" / "leo.jpg"
    if leo.exists():
        images.append(("leo", np.array(Image.open(leo).convert("RGB"))))
    images.extend(make_synthetic_images())
    return images


def validate_grid(images: list[tuple[str, np.ndarray]]) -> dict:
    rows = []
    for name, img in images:
        enc = encode(img, originator="paper-validation", timestamp=TIMESTAMP, tile_size=TILE_SIZE)
        clean = verify(enc, strict=True, tile_size=TILE_SIZE)

        h, w = img.shape[:2]
        cols, grid_rows, tile_w, tile_h = compute_grid(w, h, TILE_SIZE)
        tx, ty = min(1, cols - 1), min(1, grid_rows - 1)
        x0 = tx * tile_w
        x1 = (tx + 1) * tile_w if tx < cols - 1 else w
        y0 = ty * tile_h
        y1 = (ty + 1) * tile_h if ty < grid_rows - 1 else h
        tampered = enc.copy()
        tampered[y0:y1, x0:x1] = 128
        tamper = verify(tampered, strict=True, tile_size=TILE_SIZE)

        changed = [t for t in tamper.all_tiles if t.status != TileStatus.GREEN]
        rows.append(
            {
                "image": name,
                "width": w,
                "height": h,
                "tiles": len(clean.all_tiles),
                "psnr_k1_db": round(psnr(img, enc), 4),
                "clean_green_tiles": clean.green_count,
                "tampered_non_green_tiles": len(changed),
                "false_positive_tiles": max(0, len(changed) - 1),
            }
        )

    psnrs = np.array([r["psnr_k1_db"] for r in rows], dtype=np.float64)
    return {
        "images": rows,
        "mean_psnr_k1_db": round(float(psnrs.mean()), 4),
        "std_psnr_k1_db": round(float(psnrs.std(ddof=0)), 4),
        "all_clean_tiles_green": all(r["clean_green_tiles"] == r["tiles"] for r in rows),
        "all_tamper_runs_localized": all(r["tampered_non_green_tiles"] == 1 for r in rows),
        "total_false_positive_tiles": int(sum(r["false_positive_tiles"] for r in rows)),
    }


def validate_forest(img: np.ndarray) -> dict:
    n_blocks = 1000
    h, w = img.shape[:2]
    enc = scatter_forest_encode(
        img, "paper-validation-forest", n_blocks=n_blocks, seed=42, timestamp=TIMESTAMP
    )
    cw = int(w * 0.6)
    ch = int(h * 0.8)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = enc[y0:y0 + ch, x0:x0 + cw]
    result = scatter_forest_verify(crop)
    survival = result.survival_pct(n_blocks)
    return {
        "n_blocks": n_blocks,
        "crop_width_fraction": 0.6,
        "crop_height_fraction": 0.8,
        "anchors_recovered": result.n_genesis_found,
        "survival_pct": round(survival, 4),
        "within_paper_range_41_1_to_44_6": 41.1 <= survival <= 44.6,
    }


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    images = load_validation_images()
    grid = validate_grid(images)
    forest = validate_forest(images[0][1])

    report = {
        "schema": "pbc-paper-validation-v1",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "grid": grid,
        "forest": forest,
        "thresholds": {
            "min_mean_psnr_k1_db": 50.0,
            "forest_survival_range_pct": [41.1, 44.6],
            "expected_false_positive_tiles": 0,
        },
    }

    passed = (
        grid["mean_psnr_k1_db"] >= 50.0
        and grid["all_clean_tiles_green"]
        and grid["all_tamper_runs_localized"]
        and grid["total_false_positive_tiles"] == 0
        and forest["within_paper_range_41_1_to_44_6"]
    )
    report["passed"] = bool(passed)

    json_path = OUT_DIR / "paper_validation.json"
    md_path = OUT_DIR / "paper_validation.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# PBC Paper Validation",
                "",
                f"Passed: `{passed}`",
                "",
                f"Mean PSNR k=1: `{grid['mean_psnr_k1_db']}` dB",
                f"Std PSNR k=1: `{grid['std_psnr_k1_db']}` dB",
                f"Clean tiles all green: `{grid['all_clean_tiles_green']}`",
                f"Tamper runs localized: `{grid['all_tamper_runs_localized']}`",
                f"False positive tiles: `{grid['total_false_positive_tiles']}`",
                "",
                f"Forest anchors recovered: `{forest['anchors_recovered']}/{forest['n_blocks']}`",
                f"Forest survival: `{forest['survival_pct']}`%",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

