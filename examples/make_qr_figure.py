"""
Generate the margin-QR illustration figure for the paper.

The input directory must contain rendered page images named
page_001_mqr.png, page_002_mqr.png, page_003_mqr.png, and page_004_mqr.png.

Example:
    python examples/make_qr_figure.py --src-dir output/margin-qr-pages
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "paper" / "figures" / "fig_margin_qr.png"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Directory containing page_001_mqr.png through page_004_mqr.png.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output image path (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args()


args = parse_args()
SRC_DIR = args.src_dir
OUT = args.out

missing = [SRC_DIR / f"page_{i:03d}_mqr.png" for i in [1, 2, 3, 4]
           if not (SRC_DIR / f"page_{i:03d}_mqr.png").exists()]
if missing:
    print("Missing required rendered page image(s):", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    sys.exit(2)

# Full page image (downscaled for display)
page1 = Image.open(SRC_DIR / "page_001_mqr.png").convert("RGB")
W, H = page1.size  # 2550 × 3301

# QR stamp is 220×220 px placed at bottom-right corner
# The stamp script places it at offset (W-220-10, H-220-10) approximately
# Let's extract a generous crop around the bottom-right
PAD = 40
QR_SIZE = 220
qr_x0 = W - QR_SIZE - PAD * 2
qr_y0 = H - QR_SIZE - PAD * 2
qr_x1 = W
qr_y1 = H

def get_qr_crop(page_img):
    arr = np.array(page_img)
    return arr[qr_y0:qr_y1, qr_x0:qr_x1]

pages = [Image.open(SRC_DIR / f"page_{i:03d}_mqr.png").convert("RGB")
         for i in [1, 2, 3, 4]]
qr_crops = [get_qr_crop(p) for p in pages]

# ─── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 5), facecolor="white")
gs = fig.add_gridspec(1, 2, width_ratios=[0.38, 0.62],
                      left=0.02, right=0.98, top=0.92, bottom=0.05,
                      wspace=0.06)

# Left: full page thumbnail
ax_page = fig.add_subplot(gs[0])
thumb = np.array(page1.resize((int(W * 0.12), int(H * 0.12)),
                               Image.LANCZOS))
ax_page.imshow(thumb, cmap=None)
ax_page.set_title("Page 1 — stamped", fontsize=8, pad=3)
ax_page.set_xticks([])
ax_page.set_yticks([])
for spine in ax_page.spines.values():
    spine.set_edgecolor("#aaaaaa")
    spine.set_linewidth(0.8)

# Draw a red rectangle around where the QR is on the thumbnail
th, tw = thumb.shape[:2]
rx0 = int(qr_x0 / W * tw) - 2
ry0 = int(qr_y0 / H * th) - 2
rw = int((qr_x1 - qr_x0) / W * tw) + 4
rh = int((qr_y1 - qr_y0) / H * th) + 4
rect = Rectangle((rx0, ry0), rw, rh,
                  linewidth=1.5, edgecolor="red", facecolor="none")
ax_page.add_patch(rect)

# Right: 2×2 grid of QR close-ups
gs_qr = gs[1].subgridspec(2, 2, hspace=0.12, wspace=0.08)
labels = ["Page 1 — Genesis", "Page 2 — chain[1]",
          "Page 3 — chain[2]", "Page 4 — chain[3]"]
for idx, (crop, label) in enumerate(zip(qr_crops, labels)):
    ax = fig.add_subplot(gs_qr[idx // 2, idx % 2])
    ax.imshow(crop, cmap=None, interpolation="nearest")
    ax.set_title(label, fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.6)

fig.suptitle(
    "Margin QR Document Stamp — bottom-right corner detail (pages 1–4)",
    fontsize=9, y=0.98, color="#222222"
)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
