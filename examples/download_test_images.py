"""
Download a diverse set of real photographic test images from HuggingFace
for use in the PBC multi-image evaluation.

Sources (all freely accessible, no auth required):
  - COCO (detection-datasets/coco): 80-category diverse scene photographs
  - Oxford Flowers: close-up macro botanical photography
  - Oxford IIIT Pets: animal portrait photography

Strategy:
  Collect images spanning distinct visual regimes:
    urban scenes, indoor, outdoor/nature, animals, food, macro/texture,
    portraits, sports, vehicles, document-style

Usage:
    python examples/download_test_images.py [--n 20] [--out examples/img]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# COCO category IDs to use for diversity sampling (1 image per group)
# 80 COCO categories mapped to visual diversity groups
COCO_GROUPS = {
    "animal_domestic":  [16, 17, 18],        # bird, cat, dog
    "animal_large":     [19, 20, 21, 22, 23, 24, 25],  # horse, sheep, cow, elephant, bear, zebra, giraffe
    "vehicle":          [2, 3, 4, 5, 6, 7, 8, 9],     # bicycle, car, motorcycle, plane, bus, train, truck, boat
    "outdoor_scene":    [10, 11, 13],         # traffic light, fire hydrant, stop sign
    "indoor_appliance": [72, 73, 74, 75, 76, 77, 78],  # tv, laptop, mouse, remote, keyboard, phone, microwave
    "furniture":        [57, 58, 59, 60, 61, 62],       # chair, couch, potted plant, bed, dining table, toilet
    "food":             [52, 53, 54, 55, 56],            # banana, apple, sandwich, orange, broccoli
    "food_cooked":      [58, 59, 60],                    # pizza, donut, cake (renumbered below)
    "sports":           [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    "person":           [1],
}
# Actual COCO food/pastry category IDs
COCO_GROUPS["food_cooked"] = [53, 54, 55, 56, 57, 58, 59, 60]  # all food


def pil_to_rgb_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def save_image(arr: np.ndarray, path: Path, label: str = ""):
    Image.fromarray(arr.astype(np.uint8)).save(str(path))
    h, w = arr.shape[:2]
    print(f"  saved  {path.name:<45} {w}x{h}  {label}")


# ---------------------------------------------------------------------------
# Source 1: COCO — diverse scenes
# ---------------------------------------------------------------------------

def download_coco(out_dir: Path, n: int, seed: int = 42) -> list:
    """Sample n category-diverse images from COCO validation set."""
    print("\n-- COCO (detection-datasets/coco) ----------------------------------")
    from datasets import load_dataset

    try:
        ds = load_dataset("detection-datasets/coco", split="val",
                          streaming=True)
    except Exception as e:
        print(f"  ERR loading COCO: {e}")
        return []

    # Collect a buffer, then sample by category diversity
    print("  Buffering COCO val examples (first 2000)...")
    buffer = []
    for row in ds:
        buffer.append(row)
        if len(buffer) >= 2000:
            break
    print(f"  Buffered {len(buffer)} examples")

    rng = np.random.default_rng(seed)

    # Build category-to-buffer-index mapping
    cat_to_indices = {}
    for buf_idx, row in enumerate(buffer):
        objs = row.get("objects", {})
        categories = objs.get("category", []) if objs else []
        for cat_id in categories:
            cat_to_indices.setdefault(cat_id, []).append(buf_idx)

    # Pick one image per diversity group
    selected_buf_indices = []
    selected_groups = []
    used_buf_indices = set()
    for group_name, cat_ids in COCO_GROUPS.items():
        candidates = []
        for cid in cat_ids:
            candidates.extend(cat_to_indices.get(cid, []))
        candidates = [c for c in candidates if c not in used_buf_indices]
        if not candidates:
            continue
        chosen = int(rng.choice(candidates))
        selected_buf_indices.append(chosen)
        selected_groups.append(group_name)
        used_buf_indices.add(chosen)
        if len(selected_buf_indices) >= n:
            break

    # Fill remaining with random if needed
    remaining_indices = [i for i in range(len(buffer)) if i not in used_buf_indices]
    rng.shuffle(remaining_indices)
    for i in remaining_indices:
        if len(selected_buf_indices) >= n:
            break
        selected_buf_indices.append(i)
        selected_groups.append("random")

    # Save images
    saved = []
    for save_i, (buf_idx, group) in enumerate(zip(selected_buf_indices, selected_groups)):
        row = buffer[buf_idx]
        try:
            pil_img = row["image"]
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.open(pil_img)
            arr = pil_to_rgb_numpy(pil_img)
            img_id = row.get("image_id", buf_idx)
            out_path = out_dir / f"coco_{save_i:02d}_{group}.png"
            if out_path.exists():
                print(f"  skip   {out_path.name} (already exists)")
            else:
                save_image(arr, out_path, f"(COCO id={img_id} group={group})")
            saved.append(out_path)
        except Exception as e:
            print(f"  WARN   buf_idx={buf_idx}: {e}")

    return saved


# ---------------------------------------------------------------------------
# Source 2: Oxford Flowers — macro / close-up photography
# ---------------------------------------------------------------------------

def download_flowers(out_dir: Path, n: int, seed: int = 42) -> list:
    """Sample n images from Oxford Flowers 102."""
    print("\n-- Oxford Flowers (nelorth/oxford-flowers) --------------------------")
    from datasets import load_dataset

    try:
        ds = load_dataset("nelorth/oxford-flowers", split="train", streaming=True)
    except Exception as e:
        print(f"  ERR loading flowers: {e}")
        return []

    rng = np.random.default_rng(seed)
    buffer = []
    for row in ds:
        buffer.append(row)
        if len(buffer) >= 500:
            break

    # Pick diverse flower classes
    class_to_idx = {}
    for idx, row in enumerate(buffer):
        lbl = row.get("label", 0)
        class_to_idx.setdefault(lbl, []).append(idx)

    classes = sorted(class_to_idx.keys())
    # Sample n images spread across different classes
    step = max(1, len(classes) // n)
    selected_classes = classes[::step][:n]

    saved = []
    for cls in selected_classes:
        buf_idx = int(rng.choice(class_to_idx[cls]))
        row = buffer[buf_idx]
        try:
            pil_img = row["image"]
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.open(pil_img)
            arr = pil_to_rgb_numpy(pil_img)
            out_path = out_dir / f"flowers_cls{cls:03d}.png"
            if out_path.exists():
                print(f"  skip   {out_path.name} (already exists)")
            else:
                save_image(arr, out_path, f"(Flowers class={cls})")
            saved.append(out_path)
        except Exception as e:
            print(f"  WARN   flowers cls={cls}: {e}")

    return saved


# ---------------------------------------------------------------------------
# Source 3: Oxford IIIT Pets — animal portrait photography
# ---------------------------------------------------------------------------

def download_pets(out_dir: Path, n: int, seed: int = 42) -> list:
    """Sample n images from Oxford IIIT Pets."""
    print("\n-- Oxford IIIT Pets (timm/oxford-iiit-pet) --------------------------")
    from datasets import load_dataset

    try:
        ds = load_dataset("timm/oxford-iiit-pet", split="train", streaming=True)
    except Exception as e:
        print(f"  ERR loading pets: {e}")
        return []

    rng = np.random.default_rng(seed)
    buffer = []
    for row in ds:
        buffer.append(row)
        if len(buffer) >= 300:
            break

    class_to_idx = {}
    for idx, row in enumerate(buffer):
        lbl = row.get("label", 0)
        class_to_idx.setdefault(lbl, []).append(idx)

    classes = sorted(class_to_idx.keys())
    step = max(1, len(classes) // n)
    selected_classes = classes[::step][:n]

    saved = []
    for cls in selected_classes:
        buf_idx = int(rng.choice(class_to_idx[cls]))
        row = buffer[buf_idx]
        try:
            pil_img = row["image"]
            if not isinstance(pil_img, Image.Image):
                pil_img = Image.open(pil_img)
            arr = pil_to_rgb_numpy(pil_img)
            out_path = out_dir / f"pets_cls{cls:02d}.png"
            if out_path.exists():
                print(f"  skip   {out_path.name} (already exists)")
            else:
                save_image(arr, out_path, f"(Pets class={cls})")
            saved.append(out_path)
        except Exception as e:
            print(f"  WARN   pets cls={cls}: {e}")

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_images(n: int, out_dir: Path, seed: int = 42) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split budget: COCO gets majority, supplement with flowers and pets
    n_coco    = max(1, int(n * 0.6))
    n_flowers = max(1, int(n * 0.2))
    n_pets    = max(1, n - n_coco - n_flowers)

    saved = []
    saved += download_coco(out_dir, n_coco, seed=seed)
    saved += download_flowers(out_dir, n_flowers, seed=seed)
    saved += download_pets(out_dir, n_pets, seed=seed)

    print(f"\n{'='*60}")
    print(f"  Total images downloaded: {len(saved)}")
    print(f"  Output directory:        {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download diverse test images from HuggingFace for PBC evaluation")
    parser.add_argument("--n",    type=int, default=20)
    parser.add_argument("--out",  type=str,
                        default=str(Path(__file__).parent / "img"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    print("PBC Test Image Downloader")
    print(f"  Target: {args.n} images")
    print(f"  Output: {out_dir}")
    print(f"  Seed:   {args.seed}")

    paths = download_images(args.n, out_dir, seed=args.seed)
    if not paths:
        print("\nERROR: No images downloaded.")
        sys.exit(1)

    print(f"\nDone. Run 'python examples/multi_image_eval.py' to evaluate.")
