#!/usr/bin/env python3
"""Export the exact image manifest for the 99-image benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multi_image_eval_results.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def category_for(name: str) -> str:
    if name.startswith("coco_"):
        return "COCO"
    if name.startswith("flowers_"):
        return "Oxford Flowers 102"
    if name.startswith("pets_"):
        return "Oxford-IIIT Pets"
    if name == "leo.jpg":
        return "demo photograph"
    if name.startswith("synth_"):
        return "synthetic"
    return "unknown"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT).as_posix())
    except ValueError:
        return path.name


def expected_names(results_path: Path) -> tuple[list[str], list[str]]:
    real = []
    synthetic = []
    line_re = re.compile(r"^(?P<name>(?:coco_|flowers_|pets_)\S+|leo\.jpg|synth_\S+)\s+")
    for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = line_re.match(line)
        if not m:
            continue
        name = m.group("name")
        if name.startswith("synth_"):
            synthetic.append(name)
        else:
            real.append(name)
    return real, synthetic


def image_entry(path: Path, name: str) -> dict:
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
    return {
        "name": name,
        "manifest_path": f"benchmark/real/{name}",
        "source_group": category_for(name),
        "width": width,
        "height": height,
        "mode": mode,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the benchmark real images.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="multi_image_eval_results.txt used to define exact image names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "dataset_manifest.json",
    )
    args = parser.parse_args()

    real_names, synthetic_names = expected_names(args.results)
    missing = [name for name in real_names if not (args.source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark images: {missing}")

    real_images = [image_entry(args.source_dir / name, name) for name in real_names]
    counts = {}
    for entry in real_images:
        counts[entry["source_group"]] = counts.get(entry["source_group"], 0) + 1

    manifest = {
        "schema": "pbc-dataset-manifest-v2",
        "paper": "Pixel Block Chain: Spatial Tamper Localization and Crop-Resilient Provenance for Images in the Wild",
        "results_source": repo_relative(args.results),
        "real_image_source": "external benchmark image directory supplied via --source-dir",
        "summary": {
            "total_images": len(real_images) + len(synthetic_names),
            "real_images": len(real_images),
            "synthetic_images": len(synthetic_names),
            "real_image_counts": counts,
        },
        "real_images": real_images,
        "synthetic_images": [
            {
                "name": name,
                "source_group": "synthetic",
                "generator": "examples/multi_image_eval.py",
            }
            for name in synthetic_names
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output} with {len(real_images)} real images and "
        f"{len(synthetic_names)} synthetic entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
